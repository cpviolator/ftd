# Tiled HERK Kernel & Warp-Specialized Pipeline — Technical Report

**Date**: March 2026
**Platform**: GB10 Spark (SM121), CUDA 13.0, CUTLASS 3.x
**Scope**: Phases 0–2 of the Bandwidth-Optimized Direct HERK Kernel

## Executive Summary

Three phases of the direct HERK kernel were implemented:

1. **Phase 0 — Unification**: SM90 and SM100 HERK kernel code unified into
   `shared/herk_kernel_common.hpp`, eliminating ~2500 lines of duplication.
2. **Phase 1 — 64×64 Tiled Kernel**: `HerkTileConfig<64>` doubles arithmetic
   intensity (128 vs 64 FLOPs/byte) at the cost of halving occupancy (2 vs 4
   blocks/SM on SM120). Net effect: **29% throughput improvement** at
   N=3328, K=64, batch=32 (121 vs 94 TFLOPS).
3. **Phase 2 — Warp-Specialized Pipeline**: mbarrier-based producer-consumer
   pipeline that eliminates `__syncthreads()` from the K-loop. Currently
   **~5× slower** than the sync kernel due to 4× load pass overhead in the
   single-warp producer. Filed as experimental; requires TMA bulk loads for
   parity.

---

## Phase 0: Unification into shared/herk_kernel_common.hpp

### Motivation

The SM90 (`gemm_complex_fp8/herk_kernel.hpp`) and SM100
(`gemm_complex_sm100/herk_kernel.hpp`) HERK kernels shared ~90% identical
code: all templated kernels, PTX helpers, dispatch/launch functions, and
`DirectHerkTraits`. Maintaining two copies was error-prone (the cp.async
drain bug was independently fixed in both files).

### Approach

Following the same pattern as the already-unified `shared/gemm_kernel_common.hpp`:

- **`shared/herk_kernel_common.hpp`** (2857 lines): all shared code,
  textually included (no `#pragma once`) within each arch's namespace.
- **`gemm_complex_fp8/herk_kernel.hpp`** (12 lines): thin `#include`.
- **`gemm_complex_sm100/herk_kernel.hpp`** (~403 lines): thin `#include` +
  SM100-only FP16-input cast-in-loop legacy kernel (`herk_direct_fp8_kernel`).

### Shared content

| Category | Functions/Types |
|----------|----------------|
| Traits | `DirectHerkTraits<K_CHUNK, NR_BUFS>`, 4 named aliases |
| PTX helpers | `conj_perm_fp8()`, `mma_fp8_m16n8k32()`, `cp_async_cg_16()`, `cp_async_commit()`, `cp_async_wait<N>()` |
| Tile loaders | `load_tile_cp_async_t<Traits>`, `load_tile_direct_fp8_t<Traits>` |
| Compute | `compute_mma_block<Traits>`, `compute_mma_herk_tiled_t<Traits, Tile>` |
| Store | `store_triangle_block<OutputType>`, `store_triangle_herk_tiled_t<Tile, OutputType>` |
| Store (scratch) | `store_scratch_block<OutputType>`, `store_scratch_herk_tiled_t<Tile, OutputType>` |
| Kernels | `herk_precast_tiled_kernel_t`, `herk_persistent_tiled_kernel_t`, `herk_warp_specialized_kernel_t` |
| Launchers | `launch_precast_tiled_t`, `launch_persistent_tiled_t`, `launch_ws_tiled_t` |
| Dispatch | `launch_herk_direct_dispatch()`, `launch_herk_persistent_dispatch()`, `launch_herk_ws_dispatch()` |

### Verification

Bit-exact regression against pre-unification binaries confirmed via
`crosscheck=gpu` (batched vs per-batch, expects identical output).

---

## Phase 1: 64×64 Tiled HERK Kernel

### Design

The kernel is templated on tile size via `HerkTileConfig<BLOCK_N>`:

```cpp
template <int BLOCK_N_>
struct HerkTileConfig {
    static constexpr int BLOCK_N = BLOCK_N_;
    static constexpr int THREADS = 128;
    static constexpr int NUM_WARPS = 4;
    static constexpr int FRAGS_Y = BLOCK_N / 8;       // 4 (N32), 8 (N64)
    static constexpr int FRAGS_X = BLOCK_N / 8;       // 4 (N32), 8 (N64)
    static constexpr int LOAD_PASSES = BLOCK_N / 32;   // 1 (N32), 2 (N64)
    static constexpr int STORE_PHASES = BLOCK_N / 32;  // 1 (N32), 2 (N64)
    static constexpr int FYS_PER_WARP = FRAGS_Y / NUM_WARPS; // 1 (N32), 2 (N64)
};
using HerkTile32 = HerkTileConfig<32>;  // AI=64,  24 KB, OCC=4
using HerkTile64 = HerkTileConfig<64>;  // AI=128, 48 KB, OCC=2
```

### Key N64 kernel changes vs N32

#### 1. Multi-pass loading

A 64-row tile requires 2 × 32-row cp.async passes per buffer. The same
`load_tile_cp_async_t<Traits>` function is reused with a row offset:

```cpp
for (int lp = 0; lp < Tile::LOAD_PASSES; lp++) {
    load_tile_cp_async_t<Traits>(A, smem_A + lp * 32 * K_CHUNK,
                                  firstRow + lp * 32, kt, K, N, tid);
    load_tile_cp_async_t<Traits>(B, smem_B + lp * 32 * K_CHUNK,
                                  firstCol + lp * 32, kt, K, N, tid);
}
```

Same pattern as `GemmTile64x64` in `gemm_kernel_common.hpp`.

#### 2. Strided warp mapping

Each warp handles 2 fragment rows (vs 1 for N32) via strided indexing:

```cpp
// fy = warp_id + fy_local * NUM_WARPS
// Accumulator: acc[FYS_PER_WARP][FRAGS_X][4] = acc[2][8][4] = 64 floats/warp
for (int fy_local = 0; fy_local < Tile::FYS_PER_WARP; fy_local++) {
    int fy = warp_id + fy_local * NUM_WARPS;
    ...
}
```

This keeps all accumulator indexing compile-time constant, preventing nvcc
from spilling 64 FP32 registers to local memory.

#### 3. Phased store

Two phases of 32 columns each bound SMEM usage to 64×64×4 = 16 KB per phase:

```cpp
for (int phase = 0; phase < Tile::STORE_PHASES; phase++) {
    // Reduction to SMEM (warp-local fragments → coalesced layout)
    // Triangle-masked global store with global_col = phase * 32 + local_col
}
```

#### 4. SMEM budget

| Tile | Pipeline SMEM (K64_B3) | Store SMEM | Total | SM120 OCC | SM90 OCC |
|------|------------------------|------------|-------|-----------|----------|
| 32×32 | 24 KB | 8 KB | 32 KB | 4 | 9 |
| 64×64 | 48 KB | 16 KB | 64 KB | 2 | 4 |

SM120 has 99 KB per SM. N64's 48 KB pipeline SMEM comfortably fits with
OCC=2 (store phase reuses pipeline buffer space — not additive).

### Arithmetic intensity analysis

| Tile | FLOPs/tile (K=64) | Bytes loaded/tile | AI (FLOPs/byte) |
|------|-------------------|-------------------|-----------------|
| 32×32 | 32×32×64×8 = 524K | 2×32×64 = 4 KB × 2 = 8 KB | 64 |
| 64×64 | 64×64×64×8 = 2.1M | 2×64×64 = 8 KB × 2 = 16 KB | 128 |

The 2× AI improvement means the N64 tile is compute-bound at lower K values
and benefits more from the tensor core peak (187.5 TFLOPS FP8 on GB10).

### Performance (GB10, N=3328, K=64, batch=32)

| Tile | Throughput | vs N32 |
|------|-----------|--------|
| HerkTile32 (Sync) | 94 TFLOPS | baseline |
| HerkTile64 (Sync) | 121 TFLOPS | **+29%** |

The N64 tile wins despite halving occupancy because the 2× higher AI keeps
tensor cores fed at this K value. The crossover point where N32's higher
occupancy wins has not been characterized but is expected at very large K
where both tiles are fully compute-bound.

### Dispatch and API

- **Enum**: `HerkTileSize { N32, N64 }` in `shared/config_common.hpp`
- **API**: `set_herk_tile_size(HerkTileSize)` on both engine classes
- **Auto-select**: N32 by default. Autotuner sweeps both for direct candidates.
- **CLI**: `herk_tile=32` or `herk_tile=64`
- **Helpers**: `herk_tile_name()`, `all_herk_tile_sizes()`

### Strategy cache integration

`StrategyEntry` and `TuneCandidate` extended with `herk_tile` field. The
autotuner sweeps `DirectHerkConfig × HerkTileSize × HerkPipelineMode × 6
strategy combos` for direct candidates. N64 is skipped when N < 64.

Cache format extended from 19 to 21 columns (herk_tile, herk_tile_name,
herk_pipeline, pipeline_name). Backward-compatible: old formats load
correctly with defaults (N32, Sync).

---

## Phase 2: Warp-Specialized Pipeline (Experimental)

### Motivation

The sync kernel's K-loop uses `__syncthreads()` for pipeline synchronization:
all 128 threads load, sync, compute, sync, repeat. The hypothesis was that
separating load and compute into producer/consumer warps with fine-grained
mbarrier synchronization would improve throughput by overlapping memory and
compute operations.

### Design

```
herk_warp_specialized_kernel_t<Traits, Tile, OutputType, ScratchMode>
```

- **Warp 0** = producer: issues cp.async loads for both A and B tiles
- **Warps 1–3** = consumers: execute MMA operations
- **Synchronization**: mbarrier-based full/empty barriers (no `__syncthreads()`)

#### mbarrier helpers

New file `shared/mbarrier_helpers.hpp` (~89 lines) provides:

```cpp
__device__ void mbarrier_init(uint64_t* bar, int expected_count);
__device__ void mbarrier_arrive(uint64_t* bar);
__device__ void mbarrier_wait_parity(uint64_t* bar, int phase_bit);
__device__ void mbarrier_arrive_expect_tx(uint64_t* bar, int tx_count);
__device__ void cp_async_bulk_global_to_shared(void* smem, const void* gmem, int bytes, uint64_t* bar);
```

Raw PTX: `mbarrier.init.shared.b64`, `mbarrier.arrive.shared.b64`,
`mbarrier.try_wait.parity.shared.b64`. SM90+ only (`__CUDA_ARCH__ >= 900`),
no-op stubs for pre-SM90. No CUTLASS dependency.

#### Pipeline protocol

```
SMEM barriers:
  full_barriers[NR_BUFS]  — producer signals data is ready
  empty_barriers[NR_BUFS] — consumers signal buffer is reusable

Init:
  full_barriers[i]  expects 1 arrival  (producer lane 0)
  empty_barriers[i] expects 3 arrivals (one lane 0 per consumer warp)

Producer loop:                          Consumer loop:
  if (buf >= NR_BUFS)                     mbarrier_wait_parity(
    mbarrier_wait_parity(                   &full_barriers[buf_idx],
      &empty_barriers[buf_idx],             phase)
      phase)                              compute_mma_block(buf_idx)
  for lp in LOAD_PASSES:                  __syncwarp()
    for tp in 0..3:  // 4 passes          if (lane == 0)
      load_tile_cp_async(                     mbarrier_arrive(
        lane + tp * 32)                         &empty_barriers[buf_idx])
  cp_async_commit()                       buf_idx = (buf_idx+1) % NR_BUFS
  cp_async_wait<0>()                      phase ^= (buf_idx == 0)
  __syncwarp()
  if (lane == 0)
    mbarrier_arrive(
      &full_barriers[buf_idx])
  buf_idx = (buf_idx+1) % NR_BUFS
  phase ^= (buf_idx == 0)
```

Single `__syncthreads()` at init only. Store phase: all 4 warps participate
after a final barrier sync.

#### Consumer fragment distribution

3 consumer warps use cyclic distribution across fragment rows:

```cpp
// Consumer warp c handles fy = c, c+3, c+6, ...
for (int fy = consumer_warp; fy < FRAGS_Y; fy += NUM_CONSUMERS) {
    int acc_idx = fy / NUM_CONSUMERS;
    ...
}
```

Accumulator sizing: `WS_FYS_PER_CONSUMER = ceil(FRAGS_Y / 3)`
- N32: ceil(4/3) = 2 fragments/consumer
- N64: ceil(8/3) = 3 fragments/consumer

### Bugs found and fixed

#### Bug 1: empty_barrier expected_count mismatch

**Symptom**: Kernel produced garbage output sporadically.

**Root cause**: `mbarrier_init(&empty_barriers[i], 1)` — only 1 arrival
expected, but 3 consumer warps each arrive. After the first consumer
signals, the barrier completes, and the producer reuses the buffer while
the other 2 consumers are still reading.

**Fix**: `mbarrier_init(&empty_barriers[i], NUM_WARPS - 1)` (= 3).

#### Bug 2: All producer threads arriving at full_barrier

**Symptom**: Phase tracking scrambled, barriers firing out of sequence.

**Root cause**: All 32 threads in warp 0 executing
`mbarrier_arrive(&full_barriers[buf_idx])`, producing 32 arrivals against
expected=1. This caused the mbarrier to wrap through multiple phases.

**Fix**: Guard with `if (lane == 0) mbarrier_arrive(...)`.

### Throughput analysis

#### Producer vs consumer instruction balance

| Role | Instr/K_CHUNK (N32) | Instr/K_CHUNK (N64) | Ratio |
|------|---------------------|---------------------|-------|
| Producer (1 warp) | ~96 (4 passes × 24) | ~192 (8 passes × 24) | — |
| Consumer (per warp) | ~80 (16 MMAs) | ~160 (64 MMAs) | — |
| Producer/Consumer | 1.2× | 1.2× | balanced |

Despite the instruction count appearing balanced, the producer's `cp_async_wait<0>()`
serializes all pending loads before signaling the barrier. The 4 load passes
(covering 128 virtual thread IDs with 32 real threads) each issue the same
number of cp.async instructions that the sync kernel distributes across 128
threads in a single pass.

#### Measured performance (GB10, N=3328, K=64, batch=32)

| Kernel | Throughput | vs Sync N32 |
|--------|-----------|-------------|
| Sync + HerkTile32 | 94 TFLOPS | baseline |
| Sync + HerkTile64 | 121 TFLOPS | +29% |
| WarpSpec + HerkTile32 | ~18 TFLOPS | **-81%** |
| WarpSpec + HerkTile64 | ~18 TFLOPS | **-81%** |

#### Root cause of 5× slowdown

The fundamental issue is **4× cp.async instruction overhead in the producer**:

1. The sync kernel distributes 128 threads across 32 rows × 4 16-byte loads
   per row = 128 cp.async instructions total (1 per thread).
2. The WS kernel has only 32 producer threads covering the same 128 loads,
   requiring 4 serial passes per buffer fill.
3. Combined with `cp_async_wait<0>()` serializing all loads before signaling,
   the producer completes one buffer fill in ~4× the time the sync kernel
   takes — and the consumers stall waiting for each buffer.

The producer/consumer ratio looks balanced on paper (1.2× producer) but the
cp.async issue count per warp is 4× higher than the sync kernel's per-warp
load count, and the consumer warps are idle during all 4 passes.

### Error recovery in autotuner

The WS kernel may crash (CUDA errors from mbarrier protocol issues or
out-of-bounds access during development). The strategy autotuner was
extended with error recovery:

```cpp
// After each timed run:
cudaError_t err = cudaGetLastError();
if (err != cudaSuccess) {
    // Clear sticky error, mark candidate as failed, continue sweep
}
```

This prevents a single failing candidate from aborting the entire autotune.

### Future optimization path

The WS kernel requires **TMA bulk loads** to achieve parity with sync:

1. **TMA `cp.async.bulk.tensor`** (SM90+): Single instruction loads an entire
   tile (e.g., 32×64 = 2 KB) via the TMA engine. The producer warp would
   issue 2 TMA loads per buffer (A + B tiles) instead of 128 cp.async
   instructions.

2. **Multi-warp producer**: Alternative approach using 2 producer warps (64
   threads) to cover the 128-thread load pattern in 2 passes instead of 4.
   Reduces consumer count to 2 but halves the producer overhead.

3. The `cp_async_bulk_global_to_shared()` wrapper in `mbarrier_helpers.hpp`
   is already implemented but not yet used in the kernel — it awaits TMA
   descriptor setup integration.

---

## Dispatch Architecture

### Full dispatch chain (tiled + pipeline selection)

```
api_impl.hpp: should_use_direct decision
  │
  ├─ HerkPipelineMode::Sync
  │   ├─ PersistentMode check (K ≤ K_CHUNK=64)
  │   │   ├─ Yes → launch_herk_persistent_dispatch(config, tile, ...)
  │   │   │        → launch_persistent_tiled_dispatch<OutputType>(config, tile)
  │   │   │          → switch(tile): launch_persistent_tiled_t<Traits, Tile32/64>
  │   │   │            → herk_persistent_tiled_kernel_t<Traits, Tile><<<>>>
  │   │   └─ No  → launch_herk_direct_dispatch(config, tile, ...)
  │   │            → launch_herk_direct_tiled_dispatch<OutputType>(config, tile)
  │   │              → switch(tile): launch_precast_tiled_t<Traits, Tile32/64>
  │   │                → herk_precast_tiled_kernel_t<Traits, Tile><<<>>>
  │   │
  └─ HerkPipelineMode::WarpSpecialized
      └─ launch_herk_ws_dispatch(config, tile, ...)
           → launch_ws_tile_dispatch<Traits, OutputType>(tile)
             → switch(tile): launch_ws_tiled_t<Traits, Tile32/64>
               → herk_warp_specialized_kernel_t<Traits, Tile><<<>>>
```

8 dispatch sites per arch × 2 arches = 16 total dispatch sites updated in
`api_impl.hpp` (SM90 + SM100).

### Strategy autotuner candidate space

For each (N, K, batch, precision) key, the autotuner sweeps:

| Path | Candidates |
|------|-----------|
| Direct | 4 DirectHerkConfig × 2 HerkTileSize × 2 HerkPipelineMode × 6 strategy combos = 96 |
| Baseline | valid GemmConfig count × 6 strategy combos (pipeline=Sync only) |

Direct candidates additionally sweep persistent (on/off when K ≤ 64).
N64 is skipped when N < 64.

---

## Design Insights and Lessons Learned

### 1. Arithmetic intensity matters more than occupancy for small tiles

The N64 tile's 2× AI improvement outweighs its 2× occupancy reduction at
K=64. The reason: at OCC=4 (N32), the pipeline is already memory-bandwidth
limited — halving occupancy but doubling data reuse per tile nets a win.
The crossover will occur at large K where both tiles are compute-bound and
N32's higher occupancy provides better latency hiding.

### 2. Warp specialization needs hardware load acceleration

The fundamental limitation of the WS approach with cp.async is that
`cp.async.cg.shared.global` is a per-thread instruction — there's no way
to make one thread load another thread's data. A 32-thread producer warp
physically cannot match 128 threads issuing loads in parallel.

TMA (`cp.async.bulk.tensor`) changes this calculus: a single thread can
initiate a bulk transfer of an entire tile (kilobytes) through the TMA
engine. This would make the producer's per-buffer cost O(1) instructions
instead of O(tile_size/32).

### 3. mbarrier protocol debugging is non-trivial

The two bugs found (expected_count mismatch and all-threads-arrive) are
representative of mbarrier programming pitfalls:

- **Expected count must exactly match the number of arriving agents**.
  Over-arrival wraps the phase counter unpredictably.
- **Only one thread per agent should arrive** at an mbarrier. The PTX
  semantics count arrivals, not arriving warps.
- `compute-sanitizer --tool racecheck` can detect some of these issues but
  not all (phase scrambling manifests as non-deterministic output, not a
  clean race condition).

### 4. Register pressure from accumulator sizing

The N64 tiled kernel uses `acc[2][8][4]` = 64 FP32 registers per warp.
The WS kernel's cyclic distribution gives `acc[3][8][4]` = 96 FP32 registers
per consumer warp (N64 case). At 96 registers for accumulators alone, nvcc
may spill — monitor with `--ptxas-options=-v` for register counts above
~200 per thread.

The `#pragma unroll` + `if (fy != target) continue` pattern (rather than
direct indexing into a variable-sized accumulator) keeps nvcc's register
allocator from pessimizing: it sees compile-time-constant array indices and
avoids spilling.

### 5. Textual include pattern enables cross-arch unification

The textual include pattern (no `#pragma once`, included within a namespace)
works well for sharing kernel code across SM90/SM100/SM120:

- Templates are instantiated within each arch's namespace
- The PTX (`mma.sync.aligned.m16n8k32`) is portable across SM80+
- Arch-specific code (SM100 cast-in-loop kernel) stays in the arch wrapper
- One bug fix propagates to both arches automatically

---

## Files Modified (All Phases)

| Phase | File | Change |
|-------|------|--------|
| 0 | `shared/herk_kernel_common.hpp` | Created: unified HERK kernels (2857 lines) |
| 0 | `gemm_complex_fp8/herk_kernel.hpp` | Slimmed to thin include (12 lines) |
| 0 | `gemm_complex_sm100/herk_kernel.hpp` | Slimmed to thin include + SM100 legacy (~403 lines) |
| 1 | `shared/herk_kernel_common.hpp` | Added `HerkTileConfig`, tiled kernel templates, dispatch |
| 1 | `shared/config_common.hpp` | Added `HerkTileSize` enum, `herk_tile_name()`, `all_herk_tile_sizes()` |
| 1 | `gemm_complex_fp8.hpp` | Added `herk_tile_size_` member + setter |
| 1 | `gemm_complex_sm100.hpp` | Added `herk_tile_size_` member + setter |
| 1 | `gemm_complex_fp8/api_impl.hpp` | 8 tile dispatch sites |
| 1 | `gemm_complex_sm100/api_impl.hpp` | 8 tile dispatch sites |
| 1 | `strategy_cache.hpp` | `herk_tile` field in StrategyEntry/TuneCandidate, sweep, save/load |
| 2 | `shared/mbarrier_helpers.hpp` | Created: mbarrier PTX wrappers (89 lines) |
| 2 | `shared/config_common.hpp` | Added `HerkPipelineMode` enum, `herk_pipeline_name()` |
| 2 | `shared/herk_kernel_common.hpp` | Added `herk_warp_specialized_kernel_t`, WS dispatch |
| 2 | `gemm_complex_fp8.hpp` | Added `herk_pipeline_mode_` member + setter |
| 2 | `gemm_complex_sm100.hpp` | Added `herk_pipeline_mode_` member + setter |
| 2 | `gemm_complex_fp8/api_impl.hpp` | 4 WarpSpecialized dispatch sites |
| 2 | `gemm_complex_sm100/api_impl.hpp` | 4 WarpSpecialized dispatch sites |
| 2 | `strategy_cache.hpp` | `herk_pipeline` field, pipeline sweep, error recovery |
| 2 | `cutlass_gemm_api.cu` | `InternalHerkTileSize` + `InternalHerkPipelineMode` aliases |
| 2 | `example_usage.cu` | Updated template params for autotune |
| 2 | `example_usage_sm100.cu` | Updated template params for autotune |

---

## Reproduction

```bash
# Build
cd build && cmake .. -DCUTLASS_DIR=/path/to/cutlass -DCOMPLEX_FP8_ARCH=120a && make -j

# Phase 0: Bit-exact regression
ctest

# Phase 1: 64×64 tile correctness
./example_complex_sm100 N=128 K=64 batch=4 direct=1 herk_tile=64 crosscheck=cpu
./example_complex_sm100 N=512 K=256 batch=16 direct=1 herk_tile=64 crosscheck=gpu

# Phase 2: Warp-specialized correctness
./example_complex_sm100 N=512 K=128 batch=32 direct=1 herk_pipe=ws crosscheck=cpu

# Phase 2: Race condition check
compute-sanitizer --tool racecheck ./example_complex_sm100 N=128 K=64 batch=4 herk_pipe=ws

# Full autotune sweep (includes both tiles × both pipelines)
./example_complex_sm100 N=3328 K=128 batch=128 direct=1 tune=true

# HERK regression benchmark
bash herk_vs_tcc.sh
```

---

## Status and Next Steps

| Phase | Status | Value |
|-------|--------|-------|
| 0 — Unification | Complete | Maintenance: single file for all direct HERK code |
| 1 — 64×64 Tile | Complete, production-ready | +29% throughput at representative sizes |
| 2 — Warp-Specialized | Experimental (5× slower) | Foundation for TMA-based pipeline |

**Next optimization targets**:
1. TMA bulk loads in WS kernel (requires TMA descriptor management)
2. Multi-warp producer (2 load warps, 2 compute warps) as intermediate step
3. Characterize N32 vs N64 crossover across K and N dimensions
4. Profile persistent + N64 combination at K ≤ 64
