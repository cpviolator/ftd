# Hopper Voltage Beamformer & GEMM Optimization Report

**Date**: March 2026
**Platform**: SM90 (Hopper GH200, 132 SMs, 989 TFLOPS FP8 real, 60 MB L2, 4 TB/s HBM3)
**Target**: DSA-2000 Chronoscope voltage beamformer
**Library**: `cutlass_interface` (FTD package, dsa-2000-monorepo)

---

## 1. Problem Statement

The DSA-2000 voltage beamformer computes power-detected beamformed output
from 1651 antennas (padded to 1664 for MMA alignment) and 4000 beams
across 1600 frequency channels. The core computation is a complex
matrix-vector outer product with fused power detection:

```
C_power[beam, time] = sum_pol |W[beam, ant] × V[ant, time]^T|²
```

where W is the beamformer weight matrix (constant per observation) and V
is the voltage data arriving as 4-bit complex sign-magnitude integers.

### DSA-2000 GEMM Dimensions

| Mode | M | N | K | Batch | Description |
|------|---|---|---|-------|-------------|
| Long integration | 4000 | 4000 | 1664 | 1–32 | n_time samples, few channels |
| Short integration (unfused) | 8 | 4000 | 1664 | 1600 | 8 time samples × all channels |
| Short (ch_fuse=8) | 64 | 4000 | 1664 | 200 | 8 channels fused along M |
| Short (ch_fuse=8, 8 payloads) | 512 | 4000 | 1664 | 200 | + payload batching |

The short-integration mode (M=8) is the critical path: 8 time samples per
GEMM call gives <1% tensor core utilization with standard tile sizes.
This report documents the work done to make both the beamformer pipeline
and the underlying GEMM optimal for Hopper.

---

## 2. Architecture: DSA-110 vs DSA-2000

The DSA-2000 beamformer is a ground-up redesign, not an evolution of
DSA-110. Every architectural decision addresses a specific scaling
bottleneck.

### 2.1 Why the DSA-110 Approach Fails at Scale

The DSA-110 beamformer (`dsaX_bfCorr.cu`) used cuBLAS FP16 GEMMs with
a 4M complex decomposition (4 real GEMMs per complex multiply), FP16
accumulation with a `0.035×` input scaling factor to prevent overflow,
and separate power detection + transpose kernels. It processed 96
antennas split into two 48-antenna arms with independent weight matrices.

At DSA-2000 scale (1664 antennas, 4000 beams):

| Issue | DSA-110 | Impact at DSA-2000 scale |
|-------|---------|--------------------------|
| Accumulation precision | FP16 (65504 max, 0.035× scaling) | K=1664: max unscaled = 1664×49 = 81,536 → overflows FP16 |
| GEMM decomposition | 4 cuBLAS calls × 2 arms = 8 launches | 8 kernel launches per block, each with cuBLAS overhead |
| Operand size | FP16 weights: 48×256×2B = 24 KB/arm | FP16 weights: 1664×4000×2B = 12.7 MB → exceeds L2 |
| Intermediate buffers | Re/Im GEMM output → power kernel | 4000×M×4B×2 per pol per batch = gigabytes of scratch |
| Antenna split | 16 per-packet `cudaMemcpy` calls | No natural split for single-dish array |
| Tile selection | cuBLAS internal (no user control) | Cannot optimize for M=8 regime |

### 2.2 DSA-2000 Design Decisions

Each change maps to a specific bottleneck:

**FP8 E4M3 with FP32 accumulation** — FP8 halves operand bandwidth vs
FP16. The INT4 sign-magnitude input [-7,+7] maps exactly to FP8 E4M3
(lossless via 16-entry LUT). FP32 accumulation eliminates overflow risk
entirely — no scaling factor needed, full dynamic range preserved.

**Conjugate permutation trick** — Eliminates the 4M decomposition. A
single byte permutation (`__byte_perm(v ^ 0x00800080, 0, 0x2301)`)
transforms `[re, im]` → `[im, -re]`, so one `mma.sync.aligned.m16n8k32`
instruction simultaneously produces both `Re(A×B^H)` and `Im(A×B^H)` in
the accumulator fragments. This halves the required MMA count and
eliminates all intermediate Re/Im buffers.

**Fused power detection** — `|Re|² + |Im|²` is computed in registers
during the GEMM store phase. The power result is the only global memory
write, halving output bandwidth vs a separate power kernel.

**Prepare/execute pattern** — Weight matrix B is pre-converted to FP8
once (`prepare_b()`), then reused across all beamform calls. Amortizes
conversion cost across the entire observation.

**Channel fusion** — When beamformer weights are identical across
consecutive frequency channels, stacks `ch_fuse` channels along M:
`M_fused = n_time × ch_fuse`. Reduces GEMM batch count while increasing
M, dramatically improving tensor core utilization.

**Payload batching** — Stacks multiple time payloads along M:
`M_total = n_time × ch_fuse × n_payloads`. Combined with channel fusion,
grows M from 8 to 512.

**Wider output tiles** — Auto-selected tile sizes (32×32 up to 64×128)
increase arithmetic intensity from 64 to 171 FLOPs/byte.

### 2.3 Quantitative Comparison

| Metric | DSA-110 | DSA-2000 | Improvement |
|--------|---------|----------|-------------|
| Kernel launches per block | 10+ | 3 | 3× fewer |
| GEMM calls (complex multiply) | 4/arm × 2 arms = 8 | 1/pol × 2 pols = 2 | 4× fewer |
| Operand precision | FP16 (2 B/element) | FP8 (1 B/element) | 2× bandwidth |
| Accumulation | FP16 (overflow risk, 0.035× scale) | FP32 (exact, no scaling) | Lossless |
| Intermediate Re/Im buffers | Yes (2× GEMM output size) | No (fused power) | 2× memory saved |
| Input conversion | 4-bit → FP16 (lossy scale) | 4-bit → FP8 (exact LUT) | Lossless |
| Weight memory | 144 MB (2 arms × FP16) | 12.7 MB (FP8) | 11× smaller |
| Arithmetic intensity | ~64 FLOPs/byte (cuBLAS) | 171 FLOPs/byte (64×128 tile) | 2.7× higher |

---

## 3. Direct PTX Kernel Design

The direct GEMM and HERK kernels use hand-written PTX MMA instructions
rather than the CUTLASS 3.x `wgmma`-based mainloop. This was a
deliberate architectural choice driven by the C7510 wgmma serialization
issue on Hopper.

### 3.1 C7510 Wgmma Serialization (Hopper-Specific)

NVIDIA advisory C7510 describes a performance issue on SM90 where
`wgmma.mma_async` instructions are serialized at function call
boundaries. CUTLASS 3.x kernels use deeply nested template hierarchies,
and `ptxas` conservatively inserts `wgmma.wait_group 0` barriers at
every template instantiation boundary.

**Measured impact on GH200**: The 4M CUTLASS path achieves ~2800–3200
complex TFLOPS (70–80% of the 3956 TFLOPS theoretical). C7510 accounts
for ~15–20% of the gap (remainder: FP16→FP8 cast, interleave/
deinterleave, launch overhead).

**Mitigation attempts**:
- Device LTO (`-dlto`): Blocked on SM90 by nvcc/nvlink bug (strips `a`
  suffix from `sm_90a` → `sm_90`, ptxas rejects wgmma instructions)
- `--maxrregcount=255`: Rejected by ptxas for large CUTLASS kernels (<5%
  impact anyway, auto-skipped on Hopper)
- CUDA 12.9+ compiler: Partial improvement to `ptxas` inter-procedural
  analysis

**Solution**: The direct PTX kernel uses `mma.sync.aligned.m16n8k32`
(synchronous MMA), which has no asynchronous pipeline and no
serialization vulnerability. It is structurally immune to C7510.

### 3.2 Kernel Architecture

```
Input:  FP8 interleaved complex [N × 2K per batch] (pre-cast once)
Output: FP32 power [M × N per batch] (fused |Re|² + |Im|²)
MMA:    mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32
Load:   cp.async.cg.shared.global (bypasses registers, 128-bit vectorized)
```

**Memory pipeline**: 3-buffer cp.async pipeline (READ_AHEAD=2). Prologue
issues 2 loads; main loop does wait→prefetch→compute→sync. FP8 data
loaded via `cp.async.cg` (bypasses registers entirely vs old
cast-in-loop approach).

**Compute**: 4 MMA sub-iterations per K_CHUNK (=64), each MMA consuming
32 FP8 elements. Unrolled warp-level MMA with compile-time constant
fragment indexing to prevent register spilling.

**Store**: Two modes:
- **ScratchMode=true** (L2-resident): Coalesced writes to N×N scratch
  buffer, then pack kernel extracts triangle/power
- **ScratchMode=false**: SMEM-staged scattered writes to packed output

### 3.3 Tile Configurations

| Tile | AI (FLOPs/byte) | SMEM (3-buf) | Use Case |
|------|:---:|:---:|---|
| 32×32 | 64 | 24 KB | Small problems, highest occupancy |
| 32×128 | 102 | 60 KB | Wide-N (many beams) |
| 64×64 | 128 | 48 KB | Balanced |
| 64×128 | 171 | 72 KB | Large M (fused channels + payloads) |

Auto-selection priority: 64×128 > 64×64 > 32×128 > 32×32, constrained
by problem dimensions.

For the production VoltBF workload (M=512, N=4000, K=1664), the 64×128
tile yields 2.7× higher arithmetic intensity than the original 32×32.

### 3.4 Persistent Variant

When K ≤ K_CHUNK (=64), a persistent kernel variant launches
`sm_count × 4` blocks that loop over work items, eliminating block
scheduling overhead. Uses single-buffer direct `uint4` loads instead of
the multi-buffer cp.async pipeline.

**Critical K-gating**: At K > K_CHUNK, the persistent kernel must restart
the cp.async pipeline for each work item → 1.5–2× slower than the
non-persistent kernel (which launches one block per work item and runs
the pipeline once). Auto-dispatch enforces this gate.

---

## 4. GEMM Optimization: Direct vs 4M Dispatch

### 4.1 GemmMode Auto-Selection

The PIMPL API (`CutlassComplexGemm`) selects between two GEMM paths:

| Path | Mechanism | Strengths | Weaknesses |
|------|-----------|-----------|------------|
| **Direct PTX** | Single `mma.sync` kernel, conjugate permutation | Low launch overhead, fused power, immune to C7510 | Lower peak throughput than wgmma pipeline |
| **4M CUTLASS** | 4 real sub-GEMMs via CUTLASS 3.x wgmma mainloop | Higher peak at large problems | 4× launches, intermediate buffers, C7510 penalty |

**Auto-dispatch heuristic** (fallback when no autotuner cache):
```
Direct when: batch ≤ 2 OR M×N ≤ 4096
4M otherwise (large problems benefit from wgmma pipeline throughput)
```

**Autotuner**: `GemmStrategyCache` sweeps all valid `GemmConfig` values
plus direct kernel candidates (non-persistent + persistent when K ≤ 64).
Each candidate runs 2 warmup + 5 timed iterations. Results cached per
(M, N, K, batch, precision) in
`cutlass_gemm_strategy_cache_{build_fingerprint}.txt`.

### 4.2 Power GEMM Variants

Three entry points for the voltage beamformer, each optimizing for a
different input format:

1. **`gemm_prepared_power(A_re, A_im, C_power, ...)`** — FP16 planar input.
   Direct kernel casts to FP8 internally.

2. **`gemm_prepared_power_fp8(A_fp8_interleaved, C_power, ...)`** — Pre-cast
   FP8 interleaved input (used by the fused transpose kernel path). Skips
   internal FP16→FP8 conversion entirely.

3. **`gemm_prepared_power_int4(A_int4, C_power, ...)`** — Packed INT4 QC
   input. Internal conversion: INT4 → FP8 interleaved via LUT, using a
   persistent pre-allocated buffer (grown on demand, no per-call malloc).

**Stokes I accumulation** via beta parameter:
- Pol 0: `beta = 0` (overwrite: `C = |W × V_pol0^T|²`)
- Pol 1: `beta = 1` (accumulate: `C += |W × V_pol1^T|²`)

### 4.3 GemmConfig Runtime Selection

All FP8 tile/cluster/stage configurations are always compiled (no
rebuild-to-retune). `GemmConfig` enum provides ~17 variants:

| Config | Tile | Cluster | Notes |
|--------|------|---------|-------|
| T128×64_C1×1_S3 | 128×64 | 1×1 | SM120 default (3-stage, 82 KB SMEM) |
| T128×128_C1×1 | 128×128 | 1×1 | SM100 default |
| T128×256_C1×1 | 128×256 | 1×1 | Wide-N |
| T128×128_C1×2 | 128×128 | 1×2 | 2-SM cluster |
| T128×128_C2×2 | 128×128 | 2×2 | 4-SM cluster |
| SmallM | 64×128 | 1×1 | SM100 only, for M ≤ 64 |

`select_gemm_config(M)` auto-selects SmallM when M ≤ 64 (SM100 only;
SM120 SfAtom TMA requires M_tile ≥ 128).

---

## 5. Voltage Pipeline Implementation

### 5.1 Three-Stage Pipeline

```
Stage 1: QC transpose + pol-split → direct GEMM (pol0, β=0) → direct GEMM (pol1, β=1) → time integrate
```

**Stage 1 — QC Transpose + Pol-Split**: A single fused kernel
(`qc_fused_transpose_fp8_kernel`) performs transpose, polarization split,
channel fusion, payload batching, and INT4→FP8 conversion in one memory
pass. Uses a 16-entry device-constant LUT for INT4 sign-magnitude to
FP8 E4M3 conversion (lossless).

**Stage 2 — Direct GEMM with Fused Power**: Two launches (one per
polarization). The first writes `C_power = |Re|² + |Im|²`; the second
accumulates `C_power += |Re|² + |Im|²` via beta=1.

**Stage 3 — Time Integration + Quantization**: Sums power over
`n_time_power_sum` consecutive time samples and quantizes to uint8.

### 5.2 Channel Fusion Performance Impact

On GB10 (SM121, 48 SMs):

| Configuration | M | Batch | TFLOPS | Speedup vs Unfused |
|--------------|---|-------|--------|-------------------|
| Unfused (M=8) | 8 | 200 | 3.6 | 1.0× |
| ch_fuse=2 | 16 | 800 | 7.0 | 1.9× |
| ch_fuse=4 | 32 | 400 | 12.6 | 3.5× |
| ch_fuse=8 | 64 | 200 | 20.6 | 5.7× |
| ch_fuse=8, 2 payloads | 128 | 200 | 30.6 | 8.5× |
| ch_fuse=8, 4 payloads | 256 | 200 | 40.7 | 11.3× |
| ch_fuse=8, 8 payloads | 512 | 200 | 47.1 | 13.1× |
| ch_fuse=8, 16 payloads | 1024 | 200 | 50.6 | 14.1× |

Channel fusion is the single largest performance win. Growing M from 8 to
64 (ch_fuse=8) turns a bandwidth-starved problem into one that can
meaningfully utilize tensor cores. Payload batching further amortizes
kernel launch overhead by processing multiple time payloads in a single
GEMM.

### 5.3 Stokes I Pipeline Asymmetry

A consistent pattern in the benchmark data: `gemm_pol1` takes ~5% longer
than `gemm_pol0`. This is because pol1 uses `beta=1` (read-modify-write
to the power buffer) while pol0 uses `beta=0` (write-only). The
additional global memory read for beta accumulation adds bandwidth
pressure.

| Benchmark | gemm_pol0 (ms) | gemm_pol1 (ms) | Overhead |
|-----------|---------------|---------------|----------|
| voltbf_1ch | 4.48 | 9.19 | 2.05× |
| voltbf_4ch | 13.98 | 15.01 | 1.07× |
| voltbf_8ch | 27.02 | 29.89 | 1.11× |
| voltbf_16ch | 54.56 | 59.24 | 1.09× |
| voltbf_32ch | 113.53 | 123.55 | 1.09× |

The 1ch case shows a 2× overhead because the problem is small enough
that the read-modify-write dominates; at larger batch counts the GEMM
compute dominates and the overhead drops to ~9%.

---

## 6. HERK Optimization for Hopper

The visibility beamformer (visbf) uses Hermitian Rank-K Update (HERK)
to compute cross-correlations. The direct HERK kernel shares the same
PTX MMA architecture as the direct GEMM but writes to packed triangular
output format.

### 6.1 Direct HERK Kernel

**Conjugate permutation trick** (same as GEMM): Single MMA produces both
Re(C) and Im(C) simultaneously. Output written to packed lower triangle
format (`N×(N+1)/2` complex elements per batch), achieving 8× memory
compression vs full N×N output.

**cp.async pipeline**: 3-buffer pipeline with READ_AHEAD=2. Pre-cast
FP16→FP8 once, then kernel loads FP8 via `cp.async.cg.shared.global`.

**Bug fix (Feb 2026)**: `cp_async_wait<READ_AHEAD-1>()` (= `wait<1>`)
was a no-op when ≤1 cp.async group was pending. For K ≤ K_CHUNK or the
last chunk, the kernel read uninitialized shared memory. Fixed: use
`wait<0>` for the last chunk.

### 6.2 Tiled Kernel Variants

| Tile | AI | SMEM | Occupancy | Throughput (N=3328, K=64, batch=32, GB10) |
|------|:---:|:---:|:---------:|:---:|
| N32 (32×32) | 64 | 24 KB | 4 blocks/SM | 94 TFLOPS |
| N64 (64×64) | 128 | 48 KB | 2 blocks/SM | 121 TFLOPS (+29%) |

N64 uses multi-pass loading (2×32-row cp.async passes per buffer) and
strided warp mapping. Despite halved occupancy, the doubled arithmetic
intensity yields a net 29% throughput improvement.

### 6.3 Persistent HERK

For K ≤ 64 with many batch elements: launches `sm_count × 4` persistent
blocks with single-buffer direct loads. Eliminates block scheduling
overhead that dominates at small K with millions of triangle blocks.

### 6.4 HerkMode Auto-Dispatch

```
Direct when: K ≥ K_CHUNK (64) AND K ≤ threshold
  threshold = N/4 (single HERK or batch fits in L2)
  threshold = N/2 (batched HERK with active batch tiling)
Baseline (4M) otherwise.
```

### 6.5 Batch-Tiled Scratch (Latest Optimization)

The direct HERK kernel supports two output modes:
- **ScratchMode=true**: Coalesced writes to L2-resident N×N scratch, then
  `pack_scratch_to_triangle` extracts the lower triangle
- **ScratchMode=false**: SMEM-staged scattered writes to packed triangle

Previously, scratch was enabled only when the **entire** batch's scratch
fit in L2 (`scratch_bytes_total ≤ L2`). For N=1664 on GH200 (60 MB L2),
scratch per batch = 10.56 MB, so batch > 5 disabled scratch entirely.

**Fix**: Batch-tiled scratch processes `scratch_batch_tile` batches at a
time (`tile = L2 / scratch_bytes_per_batch`), enabling scratch for any N
where a single batch fits. Also adds L2 persistence on the FP8 precast
buffer to keep input data resident across tiles.

This change applies to both FP16 and FP32 output paths on both SM90 and
SM100/SM120.

### 6.6 Warp-Specialized Pipeline (Experimental)

An mbarrier-based producer-consumer pipeline was prototyped:
- Warp 0 = producer (cp.async loads)
- Warps 1–3 = consumers (MMA compute)

Eliminates `__syncthreads()` from the K-loop. Currently ~5× slower than
the sync kernel because the single-warp producer must issue 4× the
number of load passes (32 threads vs 128). Requires TMA bulk loads or
multi-warp load cooperation for parity. Filed as experimental.

---

## 7. Strategy Autotuning

### 7.1 HERK Strategy Sweep

Sweeps all combinations for each (N, K, batch, precision):

**Direct candidates** (16): 4 DirectHerkConfig × 2 HerkTileSize × 2 HerkPipelineMode

**Baseline candidates** (6 on SM90): All valid GemmConfig values

Cache file: `cutlass_herk_strategy_cache_{build_fingerprint}.txt`

### 7.2 GEMM Strategy Sweep

Sweeps GemmConfig + direct PTX kernel candidates:

**4M candidates**: All valid GemmConfig for architecture
**Direct candidates**: Non-persistent (all K) + persistent (K ≤ 64 only)

Each candidate: 2 warmup + 5 timed iterations. FLOPS formula:
`8 × M × N × K × batch`.

Cache file: `cutlass_gemm_strategy_cache_{build_fingerprint}.txt`

### 7.3 Kernel-Level Autotuning

`tune_cache.hpp` provides automatic blockDim/gridDim tuning for overhead
kernels (cast, pack, deinterleave). Sweep: blockDim ∈ {32,64,128,256,
512,1024} × gridDim capped at {sm_count, 2×, 4×, 8×, full}. Three
macros: `TUNED_LAUNCH_1D`, `TUNED_LAUNCH_ROW`, `TUNED_LAUNCH_2D`.

Cache file: `cutlass_kernel_cache_{build_fingerprint}.txt`

---

## 8. Performance Results

### 8.1 Latest Benchmark Data (GB10 Spark, SM121)

**Voltage beamformer (long integration, N=4096, K=1664)**:

| Channels | Batch | Total (ms) | TFLOPS | Throughput (Hz) |
|----------|-------|-----------|--------|----------------|
| 1 | 1 | 17.30 | 31.9 | 57.8 |
| 2 | 2 | 34.18 | 32.0 | 29.3 |
| 4 | 4 | 44.47 | 60.2 | 22.5 |
| 8 | 8 | 91.47 | 61.3 | 10.9 |
| 16 | 16 | 207.74 | 61.3 | 4.8 |
| 32 | 32 | 627.83 | 58.9 | 1.6 |

Peak: 61.3 TFLOPS at 8–16 channels (26% of GB10 theoretical FP8).

**Voltage beamformer (short integration, M=8, ch_fuse sweep)**:

| Config | M_eff | Total (ms) | TFLOPS | Throughput (Hz) |
|--------|-------|-----------|--------|----------------|
| Unfused | 8 | 50.71 | 3.6 | 19.7 |
| ch_fuse=2 | 16 | 202.36 | 7.0 | 4.9 |
| ch_fuse=4 | 32 | 114.53 | 12.6 | 8.7 |
| ch_fuse=8 | 64 | 71.57 | 20.6 | 14.0 |
| ch_fuse=8, 16 payloads | 1024 | 559.81 | 50.6 | 28.6 |

**Visibility beamformer (HERK, N=1664, K=128)**:

| Channels | Batch | HERK (ms) | ms/batch | % of Pipeline |
|----------|-------|----------|----------|--------------|
| 1 | 4 | 0.19 | 0.047 | 1.9% |
| 4 | 16 | 0.68 | 0.042 | 1.7% |
| 8 | 32 | 1.40 | 0.044 | 1.7% |
| 16 | 64 | 2.58 | 0.040 | 1.6% |
| 32 | 128 | 5.31 | 0.041 | 1.6% |

Near-perfect linear scaling: ~0.042 ms/batch regardless of batch count,
confirming batch-tiled scratch is working correctly.

**Dedispersion (CUTLASS FP8 vs cuBLAS FP32)**:

| Batch | CUTLASS (ms) | cuBLAS (ms) | Speedup |
|-------|-------------|------------|---------|
| 16 | 37.66 | 83.39 | 2.21× |
| 32 | 45.40 | 96.60 | 2.13× |
| 64 | 62.86 | 107.45 | 1.71× |
| 128 | 95.42 | 167.85 | 1.76× |
| 256 | 159.13 | 325.36 | 2.04× |

### 8.2 Hopper (GH200) Performance Context

From C7510 analysis on GH200 (SM90, 132 SMs):
- Peak theoretical FP8: 989 TFLOPS real, 3956 TFLOPS complex (4M)
- Observed 4M peak: ~2800–3200 complex TFLOPS (70–80% of theoretical)
- C7510 penalty: ~15–20% of the throughput gap
- Direct PTX kernel: immune to C7510, optimal for VoltBF workloads

### 8.3 HERK vs Tensor Core Correlator (GB10)

| K | CUTLASS Direct (ms) | TCC (ms) | Speedup | TFLOPS |
|---|---|---|---|---|
| 16 | — | — | 5.93× | 391.5 |
| 64 | — | — | 1.61× | 316.8 |
| 128 | — | — | 1.08× | 167.1 |
| 512 | — | — | 1.07× | 43.4 |

Peak: 395.7 effective TFLOPS at N=4096, batch=32. CUTLASS beats TCC
1.4–5.3× across all tested sizes, with 8× memory advantage (packed
triangle vs full N×N). Scales to 2500 batches in 52 GB while TCC OOMs
at ~355 batches.

---

## 9. Pipeline Stage Breakdown

For the long-integration voltage beamformer, the pipeline bottleneck
depends on channel count:

| Stage | 1 ch (%) | 4 ch (%) | 16 ch (%) | 32 ch (%) |
|-------|---------|---------|----------|----------|
| QC transpose | 2.5 | 5.8 | 9.8 | 10.7 |
| GEMM pol0 | 25.9 | 31.4 | 26.3 | 18.1 |
| GEMM pol1 | 53.1 | 33.7 | 28.5 | 19.7 |
| Time integrate | 7.2 | 11.7 | 16.1 | 10.9 |
| Corner turn | 11.4 | 17.4 | 19.4 | 40.6 |

At 1–2 channels, GEMM (especially pol1 with beta accumulation) dominates.
At 32 channels, corner turn (output data rearrangement) becomes the
bottleneck at 40.6%.

For the visibility beamformer, HERK is only 1.4–2.7% of the pipeline.
The bottlenecks are FFT (41–44%) and beam imaging (36–40%).

---

## 10. Summary of Optimizations

### Kernel-Level
1. **Direct PTX MMA** — Immune to C7510 wgmma serialization on Hopper
2. **Conjugate permutation trick** — 2× MMA efficiency for complex GEMM
3. **Fused power detection** — Eliminates intermediate Re/Im buffers
4. **Wider tiles** (up to 64×128) — 2.7× higher arithmetic intensity
5. **Persistent kernel** (K ≤ 64) — Eliminates block scheduling overhead
6. **3-buffer cp.async pipeline** — Overlaps I/O and compute
7. **Tiled HERK N64** — 29% throughput improvement over N32

### Pipeline-Level
8. **Channel fusion** — 5.7× speedup (M=8 → M=64)
9. **Payload batching** — Additional 2.5× (M=64 → M=512)
10. **Fused QC transpose kernel** — Single-pass INT4→FP8 + transpose + fusion
11. **Prepare/execute pattern** — Amortizes weight conversion
12. **Batch-tiled scratch** — Enables L2-resident scratch for any batch count

### System-Level
13. **Runtime-selectable GemmConfig** — No rebuild to retune
14. **Strategy autotuner** — Sweeps direct vs 4M + all tile configs
15. **Kernel-level autotuner** — Optimal blockDim/gridDim for overhead kernels
16. **CUDA graph capture** — Zero launch overhead for repeated calls

---

## 11. Files Reference

| File | Purpose |
|------|---------|
| `voltage_pipeline.cu` | VoltBF pipeline: Config, beamform(), channel fusion, payload batching |
| `cutlass_gemm_api.h` | PIMPL header: power GEMM API, GemmMode, PowerMode enums |
| `cutlass_gemm_api.cu` | PIMPL impl: dispatch, INT4 conversion, power reduction |
| `shared/gemm_kernel_common.hpp` | Direct GEMM kernel (unified SM90/SM100) |
| `shared/herk_kernel_common.hpp` | Direct HERK kernel (unified SM90/SM100) |
| `shared/config_common.hpp` | GemmConfig, GemmMode, HerkMode, PersistentMode enums |
| `strategy_cache.hpp` | HERK + GEMM strategy autotuning |
| `tune_cache.hpp` | Kernel-level autotuning |
| `gemm_complex_fp8/api_impl.hpp` | SM90 dispatch: HERK/GEMM auto-selection, batch tiling |
| `bench_gemm_production.cu` | Production GEMM benchmark (VoltBF + Dedisp workloads) |
| `reports/c7510_wgmma_serialization_report.md` | C7510 analysis and direct kernel advantage |
| `reports/dsa2000_vs_dsa110_voltage_beamformer.md` | DSA-110 → DSA-2000 architecture comparison |
| `reports/warp_specialized_herk_report.md` | Tiled HERK + warp-specialized pipeline |
