# DSA-2000 vs DSA-110 Voltage Beamformer: Architecture Comparison

## 1. Introduction

This report compares the voltage beamformer GPU implementations for the
DSA-110 and DSA-2000 radio telescope arrays. The DSA-110 beamformer
(`dsaX_bfCorr.cu` in dsa110-xengine) was the production system for 110
antennas and 256 beams. The DSA-2000 beamformer (`voltage_pipeline.cu` in
FTD/cutlass_interface) is a ground-up redesign targeting 1651 antennas
(padded to 1664 for MMA alignment) and 4000+ beams on next-generation
GPU hardware (Hopper/Blackwell).

The DSA-2000 design was driven by three fundamental constraints that make
the DSA-110 approach untenable at scale:

1. **17x more antennas, 8x more beams** -- the GEMM problem size grows
   quadratically in antenna count, demanding higher arithmetic intensity.
2. **Memory bandwidth wall** -- at FP16 precision, the working set for
   1664 antennas exceeds what cuBLAS can efficiently stream, while FP8
   halves the operand footprint.
3. **Kernel launch overhead** -- the DSA-110's 4-GEMM decomposition with
   separate power detection requires 6+ kernel launches per arm per
   polarisation; at DSA-2000 batch counts this becomes a bottleneck.

## 2. Array Parameters

| Parameter | DSA-110 | DSA-2000 |
|-----------|---------|----------|
| Total antennas | 96 (2 arms x 48) | 1651 (padded to 1664) |
| Beams | 512 (2 x 256 per arm) | 4000 |
| Channels per packet | 384 (48 groups x 8) | 1600 (200 groups x 8) |
| Time samples per block | 2048 (NPACKETS_PER_BLOCK) | 8 (configurable) |
| Polarisations | 2 | 2 |
| Input precision | 4-bit complex (two's complement) | 4-bit complex (sign-magnitude) |
| GEMM precision | FP16 (cuBLAS) | FP8 E4M3 (CUTLASS) |
| Accumulation | FP16 (with 0.035x scaling) | FP32 (native) |
| Output | uint8 power | uint8 or FP32 filterbank |

## 3. DSA-110 Architecture (`dsaX_bfCorr.cu`)

### 3.1 Pipeline Stages

The DSA-110 beamformer processes data through 5 stages per arm:

```
H2D copy -> antenna split -> transpose+fluff -> 4x cuBLAS GEMM -> power+transpose
```

**Stage 1: Data copy and antenna split**

The full input array `[NPACKETS_PER_BLOCK x NANTS x NCHAN_PER_PACKET x 4]`
is copied to GPU. A per-packet `cudaMemcpy` loop splits the 96 antennas
into two arms of 48 (EW and NS), issuing `NPACKETS_PER_BLOCK` (16)
individual D2D copies:

```c
for (i1=0; i1<NPACKETS_PER_BLOCK; i1++)
    cudaMemcpy(d->d_input + i1*(NANTS/2)*...,
               d->d_big_input + i1*NANTS*... + iArm*(NANTS/2)*...,
               (NANTS/2)*NCHAN_PER_PACKET*4, cudaMemcpyDeviceToDevice);
```

**Stage 2: Transpose and fluff**

Two kernels reorder the 4-bit packed data into FP16 real and imaginary
planar arrays:

- `transpose_input_bf` -- reorders `[packets, ants, chan, 2t, 2pol, 4bit]`
  to frequency-major layout using shared-memory tiled transpose
- `fluff_input_bf` -- unpacks 4-bit offset-binary nibbles to FP16,
  multiplied by `0.035` to prevent FP16 overflow during accumulation

The `0.035` scaling factor is applied during fluff to keep intermediate
values in FP16 dynamic range. This is necessary because
`cublasHgemmStridedBatched` accumulates in FP16, and without scaling,
the products of 4-bit integers (up to 7x7=49 per element, summed over
K=3072) would overflow FP16's max value of 65504. Note: `halfFac=4` is
used only by the correlator path (which splits K into 4 sub-accumulations);
the beamformer relies solely on the 0.035 input scaling.

**Stage 3: 4M Complex GEMM Decomposition**

Complex matrix multiplication `C = A * B` where `C = Cr + i*Ci`,
`A = Ar + i*Ai`, `B = Br + i*Bi` is decomposed into 4 real GEMMs:

```
Cr = Ar*Br - Ai*Bi    (two GEMMs: ac with beta=0, then -bd with beta=1)
Ci = Ai*Br + Ar*Bi    (two GEMMs: bc with beta=0, then ad with beta=1)
```

Each call uses `cublasHgemmStridedBatched` with:
- `m = NPACKETS_PER_BLOCK/4 = 512` (time samples, groups of 4 packets)
- `n = NBEAMS/2 = 256` (beams per arm)
- `k = 4*(NANTS/2)*8*2*2 = 3072` (flattened: 4 time-groups x 48 antennas x 8 channels x 2 times x 2 pols)
- `batchCount = NCHAN_PER_PACKET/8 = 48` (frequency groups)
- FP16 alpha=1 or alpha=-1 (for the `Ai*Bi` subtraction)
- FP16 beta=0 (first of pair) or beta=1 (accumulate second)

The K dimension of 3072 packs together `4 time-groups x 48 antennas x
8 channels x 2 times x 2 pols` into a single contiguous vector. This
maximises the K dimension to improve arithmetic intensity but couples
time, channel, and polarisation dimensions.

**Stage 4: Power detection + transpose**

`transpose_scale_bf` kernel computes `|Cr|^2 + |Ci|^2` element-wise and
transposes the output to beam-major order. Output is cast to `unsigned char`
(uint8). No explicit scaling is applied in this kernel -- the `0.035` factor
baked into the fluff stage effectively controls the output magnitude.

### 3.2 Per-Arm Weights

The weight matrices are computed on-GPU by `populate_weights_matrix()`,
which calculates geometric delay phases from antenna positions and
calibration solutions. Critically, the two arms have **separate weight
matrices** because:

- EW arm: `theta = sep * (127 - beam) * PI/10800`
- NS arm: `theta = sep * (127 - beam) * PI/10800 - (PI/180) * dec`

The declination offset means the NS arm's geometric delays differ from
the EW arm's. The weight computation applies both the geometric phase
rotation and per-antenna calibration solutions (normalised to unit
magnitude) as a complex multiply:

```c
twr = cos(afac * antpos[ant]);
twi = sin(afac * antpos[ant]);
wr[idx] = __float2half(twr * calib_re - twi * calib_im);
wi[idx] = __float2half(twi * calib_re + twr * calib_im);
```

Each arm's weights are stored as separate real and imaginary FP16 arrays
indexed by `iArm * offset`:

```c
i2 = iArm * 4*(NANTS/2)*8*2*2 * (NBEAMS/2) * (NCHAN_PER_PACKET/8);
// ... weights_r + i2, weights_i + i2
```

### 3.3 Limitations at DSA-2000 Scale

| Issue | Impact at 1664 antennas, 4000 beams |
|-------|-------------------------------------|
| FP16 accumulation | K scales with antennas: products up to 49, sum over K -- overflows FP16 (65504) even with aggressive scaling at large K |
| 4x GEMM overhead | 4 cuBLAS launches + synchronisation per arm |
| Per-packet memcpy loop | NPACKETS_PER_BLOCK individual `cudaMemcpy` calls for antenna split -- O(packets) launch overhead |
| No kernel fusion | Separate transpose, fluff, GEMM, power kernels -- 6+ launches per arm |
| FP16 operand size | Weight matrix: `4000 x 1664 x 2 x sizeof(half)` = 26 MB per pol -- exceeds L2 |
| cuBLAS tile selection | No control over internal tile shape -- cannot optimise for small-M case |
| Two-arm sequential | Array split forces 2x serial GEMM calls; DSA-2000's single dish array has no natural split |

## 4. DSA-2000 Architecture (`voltage_pipeline.cu`)

### 4.1 Design Principles

The DSA-2000 beamformer addresses every limitation above through:

1. **FP8 E4M3 operands with FP32 accumulation** -- halves memory bandwidth
   vs FP16 while eliminating overflow risk entirely
2. **Direct GEMM kernel** -- a single custom MMA kernel replaces the 4x cuBLAS
   decomposition, using the conjugate permutation trick to compute both
   Re(C) and Im(C) from one set of MMA instructions
3. **Fused power detection** -- `|Re|^2 + |Im|^2` is computed in the GEMM
   store phase, eliminating a separate power kernel and its associated
   global memory round-trip
4. **Prepare/execute pattern** -- weight matrices (B operand) are pre-converted
   to FP8 once and reused across all beamform calls, amortising conversion cost
5. **Channel fusion + payload batching** -- multiple channels and time payloads
   are fused along the M dimension, increasing the effective M from 8 to 512+
   and dramatically improving tensor core utilisation
6. **Wider output tiles** -- auto-selected tile sizes (32x32 up to 64x128)
   increase arithmetic intensity from 64 to 171 FLOPs/byte

### 4.2 Pipeline Stages (Direct FP8 Path)

```
QC transpose+polsplit -> direct GEMM (pol0, beta=0) -> direct GEMM (pol1, beta=1) -> time integrate+quantise
```

**Stage 1: QC INT4 transpose + polarisation split**

A single kernel (`qc_transpose_polsplit_kernel`) transposes the input from
`[channel, pol, antenna, time]` to `[channel, time, antenna]` layout and
splits polarisations. The data stays in INT4 sign-magnitude format (1 byte
per complex element) -- no FP16 conversion, no scaling factor.

For the fused path (channel fusion + payload batching),
`qc_fused_transpose_fp8_kernel` performs transpose + pol-split + channel
fusion + payload batching + INT4-to-FP8 conversion in a **single kernel**
using a 16-entry device-constant LUT:

```c
__device__ __constant__ uint8_t int4_to_fp8_lut[16] = {
    0x00, 0x38, 0x40, 0x44, 0x48, 0x4A, 0x4C, 0x4E,  // +0..+7
    0x00, 0xB8, 0xC0, 0xC4, 0xC8, 0xCA, 0xCC, 0xCE   // -0..-7
};
```

FP8 E4M3 can exactly represent all integers in [-7, +7], so the conversion
is lossless with no scaling factor needed.

**Stage 2: Direct GEMM with fused power**

The direct GEMM kernel (`gemm_direct_fp8_kernel`) uses direct PTX MMA
instructions:

```
mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32
```

The **conjugate permutation trick** eliminates the 4M decomposition entirely.
For complex input `A = [re, im]` and `B = [re, im]`, a single byte permutation:

```c
__byte_perm(v ^ 0x00800080, 0, 0x2301)  // [re,im] -> [im,-re]
```

transforms B such that loading `B` and `conj_perm(B)` as the two MMA
operand halves produces both `Re(A*B^H)` and `Im(A*B^H)` in the accumulator
fragments. One MMA instruction does the work of two in the 4M approach.

Power detection (`|Re|^2 + |Im|^2`) is fused into the store phase:

```c
float power = alpha * (re * re + im * im);
if (beta != 0.0f) power += beta * C_power[idx];
C_power[idx] = power;
```

Stokes I (sum over polarisations) uses beta accumulation:
- Pol 0: `beta = 0` (overwrite)
- Pol 1: `beta = 1` (accumulate)

This requires only 2 kernel launches total (one per polarisation), vs
DSA-110's 8 (4 GEMMs x 2 arms) plus 2 power kernels.

**Stage 3: Time integration + quantisation**

`time_integrate_quantize_kernel` sums power over `n_time_power_sum`
consecutive time samples and quantises to uint8. This is the same
conceptual step as DSA-110 but operates on pre-computed Stokes I power
rather than needing to compute `|Re|^2 + |Im|^2`.

### 4.3 Channel Fusion and Payload Batching

The most impactful optimisation for the small-M regime (M=8 time samples
per integration) is **M-axis batching**:

**Channel fusion** (`ch_fuse = 1/2/4/8`): When beamformer weights are
identical across consecutive frequency channels (true when channels share
the same calibration solution), multiple channels can be stacked along M:

```
M_fused = n_time x ch_fuse     (e.g., 8 x 8 = 64)
batch   = n_channels / ch_fuse  (e.g., 1600 / 8 = 200)
```

This reduces GEMM batch count by `ch_fuse` while increasing M by the same
factor, dramatically improving tensor core utilisation.

**Payload batching** (`max_payloads = 1..16`): Multiple consecutive time
payloads are stacked along M:

```
M_total = n_time x ch_fuse x n_payloads  (e.g., 8 x 8 x 8 = 512)
```

The `beamform_batched()` API accepts an array of QC input pointers and
processes all payloads in a single GEMM call.

Performance impact on GB10 (SM121):
- M=8 unfused: 1.4 TFLOPS (0.3% TC utilisation)
- ch_fuse=8: 10.7 TFLOPS (7.6x speedup)
- ch_fuse=8 x 8 payloads: 46.9 TFLOPS (33x speedup)

### 4.4 Wider Output Tiles

The direct GEMM kernel's output tile size determines arithmetic intensity
(compute-to-memory ratio). The recent wider-tile optimisation parameterised
the tile dimensions via `GemmTileConfig<BLOCK_M, BLOCK_N>`:

| Tile | AI (FLOPs/byte) | SMEM (3-buf) | Use case |
|------|:---:|:---:|---|
| 32x32 (original) | 64 | 24 KB | Small problems, backward compat |
| 32x128 | 102 | 60 KB | Wide-N (many beams) |
| 64x64 | 128 | 48 KB | Balanced |
| 64x128 | 171 | 72 KB | Large M (fused channels + payloads) |

Auto-selection picks the largest tile that fits the problem dimensions:
- M >= 64 and N >= 128: 64x128
- M >= 64 and N >= 64: 64x64
- N >= 128: 32x128
- Otherwise: 32x32

For the production VoltBF workload (M=512, N=4000, K=1664), the 64x128 tile
yields 2.7x higher arithmetic intensity than the original 32x32, shifting
the kernel closer to the compute-bound regime.

### 4.5 Prepare/Execute Pattern

Weight conversion is amortised across beamform calls:

```cpp
// Once (on weight update):
api->prepare_b(weights_re, weights_im, N_beams, N_ant, batch, FP8, stream);

// Per payload (hot path):
api->gemm_prepared_power_fp8(volt_fp8, power, M, N, K, batch, alpha, beta, stream);
```

`prepare_b()` converts FP16 weights to FP8 E4M3 interleaved format and
pre-arranges the conjugate-permuted copy needed by the direct kernel.
This conversion runs once when weights change (typically once per
observation pointing) and is not on the critical path.

## 5. Detailed Comparison

### 5.1 Complex Multiplication Strategy

| Aspect | DSA-110 | DSA-2000 |
|--------|---------|----------|
| Method | 4M decomposition (4 real GEMMs) | Conjugate permutation trick (1 MMA kernel) |
| GEMM calls | 4 per arm per pol | 1 per pol (all antennas) |
| Intermediate storage | Full N x N Re and Im matrices (FP16) | None (power computed in registers) |
| Power detection | Separate kernel (transpose_scale_bf) | Fused in GEMM store phase |
| Total kernel launches | 10+ per block (split+transpose+fluff+4xGEMM+power per arm) | 3 (transpose+2xGEMM_power) |

The conjugate permutation trick works because for FP8 interleaved complex
pairs `[re, im]`, the byte permutation `[im, -re]` (achieved by XOR with
`0x0080` then byte swap) produces a vector whose dot product with `[re, im]`
gives the imaginary part of the complex product, while the original dot
product gives the real part. Both are computed simultaneously in the MMA
accumulator fragments, indexed by the receiver fragment's real/imaginary
position.

### 5.2 Numerical Precision

| Aspect | DSA-110 | DSA-2000 |
|--------|---------|----------|
| Input | 4-bit two's complement [-8,+7] | 4-bit sign-magnitude [-7,+7] |
| Conversion | 4-bit -> FP16 (with 0.035x scale) | 4-bit -> FP8 E4M3 (LUT, lossless) |
| MMA precision | FP16 x FP16 -> FP16 | FP8 x FP8 -> FP32 |
| Overflow risk | High (mitigated by 0.035x scaling) | None (FP32 accumulator: 3.4e38 max) |
| Dynamic range | ~3.3 decimal digits (FP16) | ~7.2 decimal digits (FP32 accum) |

The DSA-110's 0.035 scaling is a workaround for FP16 accumulation
overflow. With K=3072 and input values up to 7, worst-case unscaled
accumulator values reach `3072 x 7 x 7 x 0.035^2 = 37` per element
(scaling applied to both A and B), well within FP16 range but leaving
limited headroom. The DSA-2000's FP32 accumulation eliminates this
concern entirely -- no scaling factor is needed, and the full dynamic
range of the input is preserved through to the power output.

### 5.3 Memory Efficiency

| Buffer | DSA-110 (96 ant, 512 beam, 2 arms) | DSA-2000 (1664 ant, 4000 beam) |
|--------|---|---|
| Input (per block) | 2048 x 96 x 384 x 4 = 301 MB | 1664 x 1600 x 2 = 5.1 MB (INT4) |
| Weights | 2 arms x 48 x 3072 x 256 x 2B = 144 MB | 1664 x 4000 x 2B = 12.7 MB (FP8) |
| Fluffed voltage | 2 x 48 x 3072 x 512 x 2B = 288 MB | N/A (no fluff, direct FP8) |
| GEMM output | 2 x 48 x 512 x 256 x 2B = 24 MB | N/A (power in 1 buffer) |
| Power output | 48 x 512 x 512 x 1B = 12.6 MB | 200 x 512 x 4000 x 4B = 1.5 GB (fused) |

The DSA-2000 avoids the FP16 intermediate buffers entirely in the direct
path. Voltage data stays in INT4 format (1 byte/complex) until the GEMM
kernel itself converts to FP8, halving the voltage buffer size vs FP16.
The power output buffer is large but is the final result -- no intermediate
Re/Im GEMM outputs are materialised.

### 5.4 Performance Scaling

DSA-110 GEMM parameters (per arm): M=512, N=256, K=3072, batch=48
- Arithmetic intensity: moderate (large M and K from packing time/chan/pol)
- cuBLAS overhead: 4 GEMM launches x 2 arms = 8 GEMM kernel launches
- FLOPs: `8 x 512 x 256 x 3072 x 48 = 154 GFLOPs` per block (both arms)
- K includes time, channel, and pol dimensions folded together

DSA-2000 GEMM parameters (with ch_fuse=8, 8 payloads):
M=512, N=4000, K=1664, batch=200
- Arithmetic intensity: 171 FLOPs/byte (64x128 tile)
- Kernel launches: 2 (one per polarisation)
- FLOPs: `8 x 512 x 4000 x 1664 x 200 = 5.4 TFLOPs` per block

The DSA-2000's effective compute is ~35x larger per beamform call,
driven by 16x more beams and 17x more antennas. The kernel launch count
dropped from 10+ to 3, and the compute density (FLOPs per launch)
increased by >1000x. The DSA-110's trick of folding time/channel/pol
into K (K=3072) gave it reasonable arithmetic intensity for 48 antennas,
but this approach does not generalise -- DSA-2000's K=1664 is purely
the antenna dimension, and the M-axis batching (channel fusion + payload
batching) provides the same arithmetic intensity benefit without coupling
unrelated dimensions.

## 6. Why Each Change Was Made

### 6.1 cuBLAS -> CUTLASS Direct Kernel

**Problem**: cuBLAS FP16 GEMM has no FP32 accumulation mode for FP8 inputs
on Hopper/Blackwell. Even with the FP16 4M approach, cuBLAS cannot expose
the conjugate permutation trick that halves the required MMA count.

**Solution**: A custom PTX MMA kernel (`mma.sync.aligned.m16n8k32`) that:
- Uses FP8 E4M3 operands (half the bandwidth of FP16)
- Accumulates in FP32 (no overflow risk, no scaling needed)
- Applies the conjugate permutation trick (1 MMA = both Re and Im)
- Fuses power detection into the store phase (no intermediate buffer)

### 6.2 4M Decomposition -> Conjugate Permutation

**Problem**: The 4M method requires 4 independent GEMM calls with
intermediate Re/Im output buffers. For DSA-2000's N=4000, these
intermediates are `4000 x M x sizeof(float) x 2` per polarisation
per batch element -- gigabytes of scratch memory, plus 4x the
launch overhead.

**Solution**: The conjugate permutation trick (`__byte_perm(v ^ 0x00800080, 0, 0x2301)`)
rearranges FP8 complex pairs so that a single `m16n8k32` MMA produces
both real and imaginary parts of the complex product in different
accumulator fragments. This eliminates all intermediate buffers and
reduces GEMM launches from 4 to 1.

### 6.3 FP16 -> FP8 Operands

**Problem**: DSA-2000's K=1664 weight matrices at FP16 would be 26 MB per
polarisation. With 200 batch elements, the total B operand is 6.4 GB --
far too large for L2 caching, making the GEMM memory-bandwidth-bound
even with large tiles.

**Solution**: FP8 E4M3 halves the operand size. Since the input data
originates as 4-bit integers in [-7, +7], FP8 E4M3 represents them
exactly (no quantisation error). The prepare/execute pattern converts
weights to FP8 once, and the fused transpose kernel converts voltages
to FP8 with a LUT -- no intermediate FP16 representation.

### 6.4 Separate Power Kernel -> Fused Store

**Problem**: DSA-110's `transpose_scale_bf` reads the full Re and Im GEMM
output from global memory, computes `Re^2 + Im^2`, and writes power back.
This doubles the memory traffic for the output phase.

**Solution**: The direct GEMM kernel computes `Re^2 + Im^2` directly from
the MMA accumulator registers in shared memory, never writing Re/Im to
global memory. The power result is the only global memory write, halving
output bandwidth.

### 6.5 Per-Packet Memcpy -> Fused Transpose Kernel

**Problem**: DSA-110 splits antennas into two arms via 16 individual
`cudaMemcpy` calls per block, each copying a non-contiguous slice.
This serialises the split and adds 16x launch overhead.

**Solution**: DSA-2000 has a single fused kernel that performs transpose,
polarisation split, channel fusion, payload batching, and INT4-to-FP8
conversion in one pass. No antenna arm split is needed because all 1664
antennas are processed in a single GEMM.

### 6.6 Input Scaling -> FP32 Accumulation

**Problem**: DSA-110 pre-scales input by 0.035 during the "fluff" stage
to prevent FP16 accumulator overflow. This scale factor introduces
quantisation error (FP16 has only 3.3 decimal digits of precision), and
the optimal value depends on K -- if K changes (more antennas or different
folding), the scale must be retuned. The correlator uses a different
strategy (`halfFac=4` to split K into sub-accumulations), highlighting
the fragility of the FP16 approach.

**Solution**: FP32 accumulation has 7+ orders of magnitude headroom
beyond worst-case accumulator values. No pre-scaling needed. The INT4
values are converted exactly to FP8 and accumulated exactly in FP32.

### 6.7 Channel Fusion + Payload Batching

**Problem**: DSA-110 achieved a large GEMM M dimension (M=512) by folding
2048 time packets into groups. DSA-2000's short-integration mode has only
M=8 time samples per GEMM call. With m16n8k32 MMA (16 rows per MMA),
M=8 wastes 50% of each MMA tile on padding, giving <1% tensor core
utilisation.

**Solution**: Channel fusion stacks `ch_fuse` consecutive channels along
M (when weights are shared), and payload batching stacks multiple time
payloads. With ch_fuse=8 and 8 payloads, M grows from 8 to 512, filling
MMA tiles efficiently. Unlike DSA-110's approach of folding channel and
polarisation into K, channel fusion preserves the independence of the
frequency and antenna dimensions. This is the single largest performance
win (33x on GB10).

### 6.8 Wider Output Tiles

**Problem**: The initial 32x32 output tile gives an arithmetic intensity
of 64 FLOPs/byte, well below the GPU's compute-to-bandwidth ridge point
(~570 FLOPs/byte on Blackwell). The kernel remains memory-bandwidth-bound
even with large M.

**Solution**: Parameterised tile sizes (up to 64x128) increase arithmetic
intensity to 171 FLOPs/byte by reusing A-tile data across more B-column
MMA fragments. The auto-selection logic picks the largest tile fitting the
problem dimensions.

## 7. Summary

The DSA-2000 voltage beamformer is a fundamentally different architecture
from DSA-110, driven by the 20x increase in antenna count and 16x increase
in beam count. Every design decision -- FP8 precision, conjugate permutation,
fused power detection, channel fusion, wider tiles -- addresses a specific
scaling bottleneck that would make the DSA-110 approach infeasible at
DSA-2000 scale.

| Metric | DSA-110 | DSA-2000 | Improvement |
|--------|---------|----------|-------------|
| Kernel launches per block | 10+ | 3 | 3x fewer |
| GEMM calls for complex multiply | 4 per arm x 2 arms | 1 per pol x 2 pols | 4x fewer |
| Operand precision | FP16 (2 bytes) | FP8 (1 byte) | 2x bandwidth |
| Accumulation precision | FP16 (overflow risk) | FP32 (exact) | No scaling needed |
| Intermediate Re/Im buffers | Yes (2x GEMM output) | No (fused power) | 2x memory saved |
| GEMM M dimension | 512 (folded time) | 512 (ch_fuse + payloads) | Comparable |
| GEMM K dimension | 3072 (folded ant+ch+t+pol) | 1664 (pure antenna, padded) | Cleaner separation |
| Arithmetic intensity | moderate (cuBLAS-selected) | 171 (custom tile) | Controlled |
| Weight preparation | Per-block GPU compute | Once (prepare/execute) | Amortised |
| CUDA graph support | No | Yes | Zero launch overhead |
| Input format | 4-bit -> FP16 (lossy scale) | 4-bit -> FP8 (exact LUT) | Lossless |
