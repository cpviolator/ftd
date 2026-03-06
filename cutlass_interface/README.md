# Complex FP8 GEMM — CUTLASS Interface

Complex-number GEMM using FP8 tensor cores via NVIDIA CUTLASS. Decomposes complex multiplication into 4 real FP8 sub-GEMMs (Gauss/4M algorithm). Targets Hopper (SM90), Blackwell datacenter (SM100), and Blackwell consumer (SM120).

## Prerequisites

- **CUDA Toolkit** 12.9+ (Hopper) or 13.0+ (Blackwell)
- **CUTLASS** 3.5+ (Hopper) or 4.0+ (Blackwell) — source tree required at build time
- **CMake** 3.24+ (required for `CUDA_ARCHITECTURES "90a"` support)
- **GPU**: Hopper (H100/H200/GH200), Blackwell datacenter (GB200), or Blackwell consumer (GB10/RTX 50-series)

## Quick Start

```bash
mkdir build && cd build

# Build all (default — FP8 + FP6 + FP4, all tile configs):
cmake .. -DCUTLASS_DIR=/path/to/cutlass -DCOMPLEX_FP8_ARCH=120a

# Blackwell GB200 datacenter (SM100):
cmake .. -DCUTLASS_DIR=/path/to/cutlass -DCOMPLEX_FP8_ARCH=100a

# Hopper (SM90, default arch):
cmake .. -DCUTLASS_DIR=/path/to/cutlass

make -j$(nproc)
```

`BUILD_ALL=ON` (default) enables all precision paths (FP8 + FP6 + FP4) and all FP8 tile configs (runtime-selectable via `GemmConfig`). For faster builds:

```bash
# Minimal build (FP8 only, ~15 min faster):
cmake .. -DCUTLASS_DIR=/path/to/cutlass -DCOMPLEX_FP8_ARCH=120a \
    -DCOMPLEX_FP8_BUILD_ALL=OFF -DCOMPLEX_SM100_ENABLE_FP6=OFF -DCOMPLEX_SM100_ENABLE_FP4=OFF
```

## CMake Options

### Core Options

| Option | Default | Description |
|--------|---------|-------------|
| `CUTLASS_DIR` | *(required)* | Path to CUTLASS source tree (visible in cmake-gui file browser) |
| `COMPLEX_FP8_ARCH` | `90a` | Target arch: `90a` (Hopper), `100a` (Blackwell DC), `120a` (Blackwell consumer) |
| `COMPLEX_FP8_BUILD_ALL` | `ON` | Build all precision and feature variants. When ON, force-enables FP6, FP4, and all arch features. Set to OFF for manual control (faster builds) |
| `COMPLEX_SM100_ENABLE_FP6` | `ON` (Blackwell) | Enable FP6 E3M2 block-scaled GEMM. Adds ~30 min build time |
| `COMPLEX_SM100_ENABLE_FP4` | `ON` (Blackwell) | Enable FP4 E2M1 block-scaled GEMM. Adds ~15 min build time |
| `COMPLEX_FP8_SM100_STAGES` | `0` (auto) | Pipeline stages. SM120 auto-defaults to 3 |
| *(removed)* | | `COMPLEX_SM100_MULTI_CONFIG` removed — all FP8 tile/cluster configs are now always compiled and selectable at runtime via `GemmConfig` / `TileConfig` |

### Advanced — Tile Tuning (toggle "show advanced" in cmake-gui)

All FP8 tile/cluster configurations are always compiled and runtime-selectable via `GemmConfig`. The per-tile CMake options below only affect block-scaled (FP6/FP4) tile defaults and are marked `ADVANCED` — they don't appear in the standard `ccmake` view.

#### Hopper (SM90)

| Option | Default | Description |
|--------|---------|-------------|
| `COMPLEX_FP8_TILE_M` | `128` | Tile M dimension (64, 128, 256) |
| `COMPLEX_FP8_TILE_N` | `256` | Tile N dimension (64, 128, 256) |
| `COMPLEX_FP8_TILE_K` | `128` | Tile K dimension (must be >=128 for FP8 wgmma) |
| `COMPLEX_FP8_CLUSTER_M` | `1` | Cluster M. >1 enables TMA multicast (1, 2, 4) |
| `COMPLEX_FP8_CLUSTER_N` | `1` | Cluster N. >1 enables TMA multicast (1, 2, 4) |
| `COMPLEX_FP8_USE_PINGPONG` | `OFF` | PingpongFP8FastAccum schedule. Best at 8192+ |
| `COMPLEX_FP8_USE_FAST_ACCUM` | `ON` | FastAccum (skip FP32 promotion). Disable for training accuracy |

#### Blackwell (SM100/SM120)

| Option | Default | Description |
|--------|---------|-------------|
| `COMPLEX_FP8_SM100_MMA_M` | `128` | Default MMA tile M. FP8 tiles all compiled; affects block-scaled defaults only |
| `COMPLEX_FP8_SM100_MMA_N` | `128` | Default MMA tile N. FP8 tiles all compiled; affects block-scaled defaults only |
| `COMPLEX_FP8_SM100_MMA_K` | `128` | MMA tile K |
| `COMPLEX_FP8_SM100_CLUSTER_M` | `1` | Cluster M. FP8 cluster variants all compiled; affects block-scaled defaults only |
| `COMPLEX_FP8_SM100_CLUSTER_N` | `1` | Cluster N. FP8 cluster variants all compiled; affects block-scaled defaults only |

### Advanced — Other (toggle "show advanced" in cmake-gui)

| Option | Default | Description |
|--------|---------|-------------|
| `COMPLEX_SM100_FP6_MMA_K` | `128` | MMA K for FP6 kernels (128, 256) |
| `COMPLEX_SM100_FP4_MMA_K` | `256` | MMA K for FP4 kernels (128, 256, 512) |
| `COMPLEX_SM100_BLKSCALED_MMA_M` | same as FP8 | Block-scaled tile M. Reduce on SM120 if SMEM overflows |
| `COMPLEX_SM100_BLKSCALED_MMA_N` | same as FP8 | Block-scaled tile N. Reduce on SM120 if SMEM overflows |
| `COMPLEX_FP8_HERK_FULL_MATRIX` | `OFF` | Debug: 4 sub-GEMMs full matrix. OFF = production: 3 sub-GEMMs, 25% fewer FLOPs |
| `COMPLEX_FP8_EXTRA_VECTORIZATION` | `ON` | Extra 128-bit vectorized memory operations |
| `COMPLEX_FP8_DEVICE_LTO` | `OFF` | Device link-time optimization. **Blackwell only** — auto-disabled on SM90 due to nvcc/nvlink bug (strips `a` suffix during device link). Also unsupported on aarch64 |
| `COMPLEX_FP8_MAX_REGISTERS` | `0` | `--maxrregcount` limit (0 = compiler default, 255 = max). Auto-skipped on SM90 (ptxas rejects for large CUTLASS kernels) |

Primary options are visible in `cmake-gui` or `ccmake`. Per-tile tuning options are in the "Advanced" view (`ccmake -LA` or cmake-gui "Advanced" checkbox).

## Build Targets

| Target | Binary | Description |
|--------|--------|-------------|
| `cutlass_gemm_api` | `libcutlass_gemm_api.a` | Static library — the main deliverable |
| `example_complex_sm100` | `example_complex_sm100` | SM100/SM120 example, correctness test, and benchmark |
| `example_complex_fp8` | `example_complex_fp8` | SM90 example, correctness test, and benchmark |
| `benchmark_configs_sm100` | `benchmark_configs_sm100` | SM100/SM120 multi-precision scaling sweep |
| `benchmark_configs` | `benchmark_configs` | SM90 configuration parameter sweep |
| `sweep_triangle_sm100` | `sweep_triangle_sm100` | Triangle decomposition strategy sweep |
| `test_herk_int4` | `test_herk_int4` | INT4 HERK end-to-end correctness test |
| `test_gemm_api` | `test_gemm_api` | Unified GEMM/HERK API end-to-end correctness test (all precisions + prepare/execute) |
| `test_tune_api` | `test_tune_api` | Strategy autotuning API test (tune=true sweep + cache + correctness) |
| `test_fp6_batched` | `test_fp6_batched` | FP6/FP4 diagnostic test (built when FP6 enabled) |
| `bench_gemm_production` | `bench_gemm_production` | Production GEMM benchmark (VoltBF + Dedisp workloads, PIMPL API, supports autotuning) |

## Binary Usage

### example_complex_sm100 (Blackwell)

Primary example and benchmark for SM100/SM120. Tests correctness against a CPU FP64 reference, then benchmarks performance.

```bash
# Square GEMM (M=N=K=4096)
./example_complex_sm100 4096

# Rectangular with batch
./example_complex_sm100 1024 2048 512 32

# FP6 precision
./example_complex_sm100 4096 fp6e3m2

# Benchmark only (skip correctness test)
./example_complex_sm100 4096 mode=bench

# Test only
./example_complex_sm100 4096 mode=test

# Direct HERK kernel (faster at small K)
./example_complex_sm100 4096 4096 256 32 direct=1

# Triangle decomposition with CUDA graph
./example_complex_sm100 4096 slabs=8 graph=1

# Autotune overhead kernels (writes cutlass_kernel_cache_{build_fingerprint}.txt)
./example_complex_sm100 4096 tune=2

# Batched HERK cross-check (validate per-batch independence)
./example_complex_sm100 4096 4096 512 32 crosscheck=cpu
```

**CLI Options:**

| Key | Values | Default | Description |
|-----|--------|---------|-------------|
| `mode` | `test`, `bench`, `both` | `both` | Run correctness test, benchmark, or both |
| `direct` | `auto`, `0`, `1` | `auto` | Direct HERK: auto selects by K threshold, 1=force, 0=off |
| `slabs` | `2`-`32` | `0` (auto) | Triangle: number of horizontal slabs |
| `min_slab` | integer | `0` (auto) | Triangle: minimum slab height in rows |
| `graduated` | `0`, `1` | `0` | Triangle: use sqrt-spaced slab boundaries |
| `graph` | `0`, `1` | `0` | Triangle: CUDA graph capture/replay (SM120) |
| `herk_graph` | `0`, `1` | `0` | Baseline HERK: CUDA graph capture/replay |
| `tile` | `0`, `1` | `1` | Batch tiling for L2 scratch reuse |
| `persistent` | `auto`, `0`, `1` | `auto` | Persistent direct HERK: auto selects when K ≤ 64, 1=force, 0=off |
| `tune` | `true`, `false`, `0`-`3` | `0` | `true`=full strategy autotune (sweeps all runtime options), `0`-`3`=kernel-level tune verbosity |
| `gemm_mode` | `auto`, `direct`, `4m` | `auto` | GEMM dispatch: auto selects via autotuner/heuristic, direct=PTX kernel, 4m=CUTLASS sub-GEMMs |
| `crosscheck` | `cpu`, `gpu`, `off` | `off` | Batched HERK independence validation |
| `verbose` | `0`, `1` | `0` | Print slab decomposition details |

**Positional arguments:** `[M] [N] [K] [batch] [precision]`
- Precision: `fp8` (default), `fp6e3m2`, `fp6e2m3`, `fp4`
- Single value sets M=N=K (e.g., `./example_complex_sm100 4096`)

### example_complex_fp8 (Hopper)

Same interface as `example_complex_sm100` but for SM90. FP8 only (no FP6/FP4).

```bash
./example_complex_fp8 4096
./example_complex_fp8 4096 4096 512 32 mode=bench direct=auto
```

Supports the same CLI options except `graph` (SM120 only) and precision selection (FP8 only).

### benchmark_configs_sm100 (Blackwell)

Multi-precision scaling benchmark across problem sizes. Reports TFLOPS and GB/s for each operation type.

```bash
# Default sizes (256, 512, 1024, 2048, 4096, 8192)
./benchmark_configs_sm100

# Custom sizes
./benchmark_configs_sm100 512 1024 4096

# Batched benchmarks
./benchmark_configs_sm100 1024 4096 batch=32

# Single precision only
./benchmark_configs_sm100 4096 precision=fp6e3m2

# With triangle optimization
./benchmark_configs_sm100 4096 slabs=8 graph=1
```

**CLI Options:**

| Key | Values | Default | Description |
|-----|--------|---------|-------------|
| `batch` | integer | `1` | Batch count (>1 adds batched benchmark columns) |
| `precision` | `fp8`, `fp6e3m2`, `fp6e2m3`, `fp4` | all | Filter to single precision |
| `slabs` | `2`-`32` | `0` (auto) | Triangle slab count |
| `min_slab` | integer | `0` (auto) | Minimum slab height |
| `graduated` | `0`, `1` | `0` | Sqrt-spaced slab boundaries |
| `graph` | `0`, `1` | `0` | CUDA graph capture/replay |
| `verbose` | `0`, `1` | `0` | Slab decomposition details |

**Positional arguments:** One or more square problem sizes (N=M=K).

**Output columns (unbatched):** Standard GEMM, Hermitian, Gram, HERK packed, HERK triangle, Triangle+Graph.
**Output columns (batched, when batch>1):** Batched GEMM, Batched HERK, Batched Triangle, Batched Triangle+Graph.

### benchmark_configs (Hopper)

SM90 kernel configuration sweep. Tests different CUTLASS kernel schedules, tile shapes, cluster shapes, swizzle patterns, and raster orders.

```bash
# Default (4096x4096x4096)
./benchmark_configs

# Custom size
./benchmark_configs 8192

# Rectangular
./benchmark_configs 4096 4096 2048
```

**Positional arguments:** `[M] [N] [K]` — single value sets M=N=K.

### sweep_triangle_sm100

Benchmarks different triangle decomposition strategies for batched HERK. Compares auto-adaptive, uniform, and graduated slab spacing across slab counts.

```bash
# Default (N=4096, K=4096, batch=128)
./sweep_triangle_sm100

# Custom dimensions
./sweep_triangle_sm100 8192 1024 64

# Include CUDA graph comparison
./sweep_triangle_sm100 4096 graph=1
```

**Positional arguments:** `[N] [K] [batch]`
- `4096` sets N=K=4096, batch=128
- `4096 128` sets N=K=4096, batch=128
- `4096 1024 64` sets N=4096, K=1024, batch=64

**CLI Options:**

| Key | Values | Default | Description |
|-----|--------|---------|-------------|
| `graph` | `0`, `1` | `0` | Include CUDA graph sweep in comparison |

**Output:** Sweeps slab counts (2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 24, 32) and reports FLOPs saved, speedup vs baseline, and TFLOPS for each configuration.

### test_gemm_api

End-to-end correctness test for the unified PIMPL API. Tests all precision combinations (FP8/FP6/FP4 GEMM, FP8 prepare/execute with FP6/FP4, HERK FP32 output) against CPU FP64 reference.

```bash
# Default (M=N=K=128, batch=2)
./test_gemm_api

# Custom dimensions
./test_gemm_api 256 512 4
```

**Positional arguments:** `[N] [K] [batch]`

**Output:** 8 tests covering all GEMM compute precisions, prepare/execute paths, and HERK FP32 output.

### test_herk_int4

End-to-end correctness test for the PIMPL HERK API with INT4 sign-magnitude input. Validates against a CPU FP64 reference.

```bash
# Default (N=64, K=128, batch=2)
./test_herk_int4

# Custom dimensions
./test_herk_int4 256 512 4
```

**Positional arguments:** `[N] [K] [batch]`

**Output:** Max relative error per batch element (expected: <0.04% at K=1024).

### test_tune_api

End-to-end test for the strategy autotuning API (`tune=true`). Verifies that:
1. First call triggers a benchmark sweep across all viable strategy combinations
2. Second call hits the persistent cache (fast)
3. `tune=false` backward compatibility (uses Direct fallback when FP8 baseline exceeds SMEM)
4. Cache file is written and contains valid entries

```bash
# Default (N=128, K=128, batch=2)
./test_tune_api

# Custom dimensions
./test_tune_api 256 128 4
```

**Positional arguments:** `[N] [K] [batch]`

### test_fp6_batched

Diagnostic test for FP6/FP4 real GEMM dispatch. Requires FP6 to be enabled at build time.

```bash
# Default (M=N=K=128)
./test_fp6_batched

# Custom dimensions
./test_fp6_batched 256 512 1024
```

**Positional arguments:** `[M] [N] [K]`

### bench_gemm_production

Production GEMM benchmark for beamformer (voltage BF) and dedisp workloads. Uses the PIMPL API only (no CUTLASS headers needed). Tests full `gemm()` (with B conversion), `gemm_prepared()` (pre-converted B), `gemm_prepared_direct()`, and `gemm_prepared_fused()` paths. Supports `gemm_mode=auto|direct|4m` to control GEMM dispatch (direct PTX kernel vs 4 CUTLASS sub-GEMMs).

```bash
# Run all suites with defaults
./bench_gemm_production

# Voltage beamformer only, with autotuning
./bench_gemm_production voltbf tune=true

# All suites, custom timing, with fused and direct paths
./bench_gemm_production all runs=20 warmup=3 bestof=5 fused=1 direct=1

# Dedisp only, with tile config override
./bench_gemm_production dedisp tile=gemm

# With GEMM autotuning and verbose output
./bench_gemm_production tune=true tune_verb=2

# Force direct PTX kernel for all GEMM operations
./bench_gemm_production voltbf gemm_mode=direct
```

| Key | Values | Default | Description |
|-----|--------|---------|-------------|
| (positional) | `all`, `voltbf`, `dedisp` | `all` | Suite selection |
| `runs` | integer | 10 | Timed runs per measurement |
| `warmup` | integer | 5 | Warmup iterations |
| `bestof` | integer | 3 | Repeat measurement N times, take best |
| `fused` | `0`, `1` | `0` | Also benchmark batch-fused M path |
| `direct` | `0`, `1` | `0` | Also benchmark direct kernel path |
| `tile` | `auto`, `herk`, `gemm`, `wide`, `small` | `auto` | TileConfig override |
| `tune` | `true` | `false` | Run GEMM autotuning before each problem |
| `tune_verb` | `0`-`3` | `1` | GEMM strategy tune verbosity |
| `gemm_mode` | `auto`, `direct`, `4m` | `auto` | GEMM dispatch: auto=autotuner/heuristic, direct=PTX kernel, 4m=CUTLASS sub-GEMMs |

| `ch_fuse` | `1`-`8` | `1` | Channel fusion factor for fused VoltBF path |
| `n_batch` | `1`-`16` | `1` | Payload batch count for fused VoltBF path |

When `ch_fuse > 1` or `n_batch > 1`, bench_pipeline runs the fused path: `qc_fused_transpose_fp8_kernel` → `gemm_prepared_power_fp8()` × 2 → time integration. Reports per-stage timing and TFLOPS.

**Problem shapes:**
- Voltage BF: M=128/256, N=4000, K=1664, batch=128-2500 (B uniform across batches)
- Voltage BF short-integration: M=8, N=4000, K=1664, batch=1600 (unfused baseline)
- Voltage BF fused: M=64/512, N=4000, K=1664, batch=200 (ch_fuse=8, n_payloads=1/8)
- Dedisp TCFDD: M=32-256, N=512-4096, K=512-2048, batch=257/2049 (B varies per batch)

## Using the Library

External consumers include only `cutlass_gemm_api.h` and link against `libcutlass_gemm_api.a`. The PIMPL pattern isolates all CUTLASS template machinery from consumer code.

### Precision and Format Enums

The unified API uses runtime enum parameters instead of separate methods per precision:

```cpp
#include "cutlass_gemm_api.h"
using cutlass_gemm_api::InputPrecision;    // FP16, INT4
using cutlass_gemm_api::ComputePrecision;  // FP8, FP6, FP4
using cutlass_gemm_api::OutputPrecision;   // FP16, FP32
using cutlass_gemm_api::OutputFormat;      // PackedTriangle
using cutlass_gemm_api::HerkMode;          // Auto, ForceDirect, ForceBaseline
using cutlass_gemm_api::GemmMode;          // Auto, ForceDirect, Force4M
```

| Enum | Values | Description |
|------|--------|-------------|
| `InputPrecision` | `FP16`, `INT4` | Input data format. INT4 = DSA-2000 QC sign-magnitude (1 byte/complex) |
| `ComputePrecision` | `FP8`, `FP6`, `FP4` | Tensor core compute precision (internal conversion) |
| `OutputPrecision` | `FP16`, `FP32` | Output element type. FP32 avoids FP16 overflow at large K |
| `OutputFormat` | `PackedTriangle` | HERK output layout. PackedTriangle = N*(N+1)/2 elements |
| `HerkMode` | `Auto`, `ForceDirect`, `ForceBaseline` | HERK dispatch: Auto selects by K, ForceDirect/ForceBaseline override |
| `GemmMode` | `Auto`, `ForceDirect`, `Force4M` | GEMM dispatch: Auto checks autotuner cache then heuristic, ForceDirect uses direct PTX kernel, Force4M uses 4 CUTLASS sub-GEMMs |

### GEMM

```cpp
cutlass_gemm_api::CutlassComplexGemm api;

// FP8 complex GEMM (FP16 output — default)
api.gemm(d_Ar, d_Ai, d_Br, d_Bi, d_Cr, d_Ci, M, N, K, batch);

// FP8 complex GEMM (FP32 output)
api.gemm(d_Ar, d_Ai, d_Br, d_Bi, d_Cr, d_Ci, M, N, K, batch,
         ComputePrecision::FP8, OutputPrecision::FP32);

// FP6 complex GEMM (FP32 output, Blackwell only)
api.gemm(d_Ar, d_Ai, d_Br, d_Bi, d_Cr, d_Ci, M, N, K, batch,
         ComputePrecision::FP6, OutputPrecision::FP32);

// FP4 complex GEMM (FP32 output, Blackwell only)
api.gemm(d_Ar, d_Ai, d_Br, d_Bi, d_Cr, d_Ci, M, N, K, batch,
         ComputePrecision::FP4, OutputPrecision::FP32);

// Override GEMM dispatch: direct PTX kernel vs 4 CUTLASS sub-GEMMs
api.set_gemm_mode(GemmMode::Auto);        // autotuner cache first, then heuristic (default)
api.set_gemm_mode(GemmMode::ForceDirect); // always use direct PTX kernel (immune to C7510)
api.set_gemm_mode(GemmMode::Force4M);     // always use 4 CUTLASS sub-GEMMs
```

### HERK

```cpp
cutlass_gemm_api::CutlassComplexGemm api;

// FP16 input → FP16 packed triangle output (default)
api.herk(d_A, d_C, N, K, batch);

// FP16 input → FP32 packed triangle output (native, no FP16 intermediate)
api.herk(d_A, d_C, N, K, batch,
         InputPrecision::FP16, ComputePrecision::FP8, OutputPrecision::FP32);

// INT4 input → FP16 packed triangle output
api.herk(d_A_int4, d_C, N, K, batch,
         InputPrecision::INT4, ComputePrecision::FP8, OutputPrecision::FP16);

// INT4 input → FP32 packed triangle output
api.herk(d_A_int4, d_C_fp32, N, K, batch,
         InputPrecision::INT4, ComputePrecision::FP8, OutputPrecision::FP32);

// Override direct/baseline kernel selection (default: Auto)
api.set_herk_mode(HerkMode::ForceDirect);   // always use direct PTX kernel
api.set_herk_mode(HerkMode::ForceBaseline); // always use CUTLASS baseline
api.set_herk_mode(HerkMode::Auto);          // K-adaptive (default)

// Override FP8 tile/cluster configuration (default: Auto)
api.set_tile_config(TileConfig::Auto);        // arch-dependent default; autotuner overrides if tune=true
api.set_tile_config(TileConfig::HerkOptimal); // SM120: 128x64 3stg, SM100: 128x128 3stg, SM90: 128x256 Coop
api.set_tile_config(TileConfig::GemmOptimal); // 128x128 auto-stages (GEMM throughput optimal)
api.set_tile_config(TileConfig::WideN);       // 128x256 (SM100/SM90 only)
api.set_tile_config(TileConfig::Cluster1x2);  // 1x2 cluster TMA multicast (SM100 only)
api.set_tile_config(TileConfig::Cluster2x2);  // 2x2 cluster (SM100 only)
api.set_tile_config(TileConfig::SmallTile);   // 128x64 (SM120/SM100) or 64x128 (SM90)
api.set_tile_config(TileConfig::Pingpong);    // 128x256 Pingpong schedule (SM90 only)
```

All FP8 tile/cluster/stage configurations are always compiled into the binary and selectable at runtime. The strategy autotuner (`tune=true`) sweeps all valid configs for the target architecture and caches the optimal choice per problem size. No rebuild required to change tile shapes.

### Prepare/Execute (Amortized B Conversion)

```cpp
cutlass_gemm_api::CutlassComplexGemm api;

// One-time: convert B to internal format (FP8, FP6, or FP4)
api.prepare_b(d_Br, d_Bi, N, K, batch, ComputePrecision::FP8, stream);

// Per-call: only converts A, reuses pre-converted B
for (int i = 0; i < num_iterations; i++) {
    api.gemm_prepared(d_Ar[i], d_Ai[i], d_Cr, d_Ci, M, N, K, batch,
                      OutputPrecision::FP32, alpha, beta, stream);
}
```

#### Power Detection (Fused |Re|^2 + |Im|^2)

For beamformer workloads that only need power (Stokes I), the power GEMM variants
fuse the complex GEMM with power detection in a single kernel, eliminating 4 FP32
intermediate buffers and the separate power computation pass:

```cpp
cutlass_gemm_api::CutlassComplexGemm api;
api.prepare_b(d_Br, d_Bi, N, K, batch, ComputePrecision::FP8, stream);

// FP16 input → fused power: C = |A * B^T|^2
api.gemm_prepared_power(d_Ar, d_Ai, d_power, M, N, K, batch,
                        1.0f, 0.0f, stream);

// Pre-cast FP8 interleaved input → fused power (skip internal FP16→FP8 cast)
api.gemm_prepared_power_fp8(d_A_fp8, d_power, M, N, K, batch,
                            1.0f, 0.0f, stream);

// INT4 QC input → fused power (INT4→FP8 conversion + power in one call)
// Uses a pre-allocated internal FP8 buffer (grown on demand, no per-call malloc)
api.gemm_prepared_power_int4(d_A_int4, d_power, M, N, K, batch,
                             1.0f, 0.0f, stream);
```

The `beta` parameter enables Stokes I accumulation across polarisations:
```cpp
// pol0: C = |W × V_pol0^T|^2
api.gemm_prepared_power_fp8(d_fp8_pol0, d_power, M, N, K, batch, 1.0f, 0.0f, stream);
// pol1: C += |W × V_pol1^T|^2
api.gemm_prepared_power_fp8(d_fp8_pol1, d_power, M, N, K, batch, 1.0f, 1.0f, stream);
```

### Supported Combinations

| Input | Compute | Output | GEMM | HERK | Power | Notes |
|-------|---------|--------|------|------|-------|-------|
| FP16 | FP8 | FP16 | Yes | Yes | — | Default, lowest latency |
| FP16 | FP8 | FP32 | Yes | Yes | Yes | Native FP32 — no FP16 intermediate |
| FP16 | FP6 | FP32 | Yes | — | — | Blackwell only, lossless for INT4 input |
| FP16 | FP4 | FP32 | Yes | — | — | Blackwell only, **lossy** (~20% error for INT4 input) |
| FP8 | FP8 | FP32 | — | — | Yes | Pre-cast FP8 interleaved, skips FP16→FP8 cast |
| INT4 | FP8 | FP16 | — | Yes | — | DSA-2000 QC format, exact |
| INT4 | FP8 | FP32 | — | Yes | Yes | Native FP32, exact for integers ≤7 |

### CMake Integration

```cmake
add_subdirectory(cutlass_interface)
target_link_libraries(my_app PRIVATE cutlass_gemm_api)
```

Or link the pre-built static library:

```cmake
target_link_libraries(my_app PRIVATE /path/to/libcutlass_gemm_api.a)
target_include_directories(my_app PRIVATE /path/to/cutlass_interface)
```

### Strategy Autotuning

Pass `tune=true` to `herk()` (PIMPL API) or use `tune=true` CLI flag (example binaries) to automatically benchmark all internal strategy combinations and cache the fastest for the given problem size. The sweep covers ALL runtime options: HerkMode, HerkStrategy, CUDA graph, PersistentMode, herk_graph, and **GemmConfig** (FP8 tile/cluster/schedule). Batch tiling is always ON (not swept). Total candidates: ~24 (SM120), ~54 (SM100), ~42 (SM90).

**PIMPL API:**

```cpp
using cutlass_gemm_api::InputPrecision;
using cutlass_gemm_api::ComputePrecision;
using cutlass_gemm_api::OutputPrecision;
using cutlass_gemm_api::OutputFormat;

cutlass_gemm_api::CutlassComplexGemm api;

// Optional: override cache file (default: cutlass_strategy_cache_{build_fingerprint}.txt)
api.set_tune_cache_path("/data/my_strategy_cache.txt");

// First call: runs ~5-second benchmark sweep, picks optimal strategy, caches to disk
api.herk(d_A, d_C, N, K, batch,
         InputPrecision::FP16, ComputePrecision::FP8,
         OutputPrecision::FP16, OutputFormat::PackedTriangle,
         1.0f, 0.0f, stream, /*tune=*/true);

// Subsequent calls: instant cache hit, applies ALL optimal settings automatically
api.herk(d_A, d_C, N, K, batch,
         InputPrecision::FP16, ComputePrecision::FP8,
         OutputPrecision::FP16, OutputFormat::PackedTriangle,
         1.0f, 0.0f, stream, /*tune=*/true);
```

**Example binaries:**

```bash
# Full autotune: sweeps all runtime options, overrides CLI flags
./example_complex_sm100 3328 3328 128 32 tune=true

# Kernel-level tuning only (does NOT sweep strategies)
./example_complex_sm100 3328 tune=2
```

The cache is persistent across restarts. Each build configuration gets its own file (e.g. `cutlass_strategy_cache_sm120_m128x128x128_stg3_fp6k64_fp4k128.txt`). GPU name mismatch invalidates the cache.

When `tune=false` (default), the PIMPL API automatically falls back to the Direct HERK kernel if the FP8 baseline GEMM exceeds device SMEM (e.g. stg3 on SM120). This ensures correct behavior without requiring the caller to know about SMEM constraints.

The sweep tests up to 12 strategy combinations (candidates that cannot run on the current hardware are automatically skipped):

**Direct mode candidates** (6):

| # | Kernel | Decomposition | CUDA Graph | Persistent |
|---|--------|---------------|------------|------------|
| 1 | Direct | Baseline | off | off |
| 2 | Direct | Baseline | off | on |
| 3 | Direct | TriangleAware | off | off |
| 4 | Direct | TriangleAware | off | on |
| 5 | Direct | TriangleAware | on | off |
| 6 | Direct | TriangleAware | on | on |

**Baseline mode candidates** (6, skipped when FP8 baseline exceeds SMEM):

| # | Kernel | Decomposition | CUDA Graph | HERK Graph |
|---|--------|---------------|------------|------------|
| 7 | Baseline | Baseline | off | off |
| 8 | Baseline | TriangleAware | off | off |
| 9 | Baseline | TriangleAware | on | off |
| 10 | Baseline | Baseline | off | on |
| 11 | Baseline | TriangleAware | off | on |
| 12 | Baseline | TriangleAware | on | on |

### GEMM Strategy Autotuning

Pass `tune=true` to `gemm()` or `gemm_prepared()` (PIMPL API) or use `tune=true` CLI flag (`bench_gemm_production`) to benchmark all GEMM dispatch strategies and cache the fastest for the given problem size. The sweep includes:

- **4M candidates**: All valid `GemmConfig` tile/cluster/schedule configs (3 on SM120, 8 on SM100, 6 on SM90)
- **Direct kernel candidate**: Non-persistent direct PTX kernel (always included)
- **Persistent direct candidate**: Persistent variant (only when K ≤ 64)

Each candidate runs 2 warmup + 5 timed iterations. The winner's `use_direct`, `direct_tile`, and `use_persistent` flags are cached to disk alongside the `GemmConfig`. Cache format: `GEMM M N K batch prec config time tflops bw direct tile persistent`.

```cpp
cutlass_gemm_api::CutlassComplexGemm api;

// GEMM autotuning: sweeps 4M configs + direct PTX kernel candidates
api.gemm_prepared(d_Ar, d_Ai, d_Cr, d_Ci, M, N, K, batch,
                  OutputPrecision::FP32, 1.0f, 0.0f, stream, /*tune=*/true);

// Control GEMM dispatch mode (independent of autotuning)
api.set_gemm_mode(GemmMode::Auto);        // autotuner cache first, then heuristic (default)
api.set_gemm_mode(GemmMode::ForceDirect); // always use direct PTX kernel
api.set_gemm_mode(GemmMode::Force4M);     // always use 4 CUTLASS sub-GEMMs
```

The direct PTX kernel uses `mma.sync.aligned.m16n8k32` (synchronous MMA) and is immune to the C7510 `wgmma` serialization issue on Hopper. See `reports/c7510_wgmma_serialization_report.md` for details.

### Kernel-Level Tuning

Separate from strategy autotuning, the library provides kernel-level tuning that optimizes blockDim/gridDim for overhead kernels (cast, pack, deinterleave, MXFP preprocessing). This benefits both the GEMM and HERK paths.

```cpp
cutlass_gemm_api::CutlassComplexGemm api;

// Set verbosity: 0=silent, 1=show cached, 2=tune+summary, 3=tune+detail
api.set_kernel_tune_verbosity(2);

// Optional: set cache file path (default: auto-generated per build)
api.set_kernel_tune_cache_path("/data/my_kernel_cache.txt");

// First GEMM/HERK call triggers the sweep (~30-60 seconds)
api.gemm_prepared(A_re, A_im, C_re, C_im, M, N, K, batch,
                  cutlass_gemm_api::OutputPrecision::FP32, 1.0f, 0.0f, stream);

// Subsequent calls use cached params (zero overhead)
```

The cache is persistent and GPU-name-validated. Each build configuration gets its own cache file (e.g. `cutlass_kernel_cache_sm120_m128x64x128_fp6k128_fp4k256_bs128x128_stg3_vec.txt`). Legacy `cutlass_tunecache.txt` files (with per-row build fingerprints) are loaded as a fallback and filtered to matching rows.

**Build fingerprint structure:** `{arch}_{tile}[_fp6k{K}][_fp4k{K}][_bs{M}x{N}][_stg{N}][_vec][_lto][_reg{N}]`

Examples:
- SM120: `sm120_m128x64x128_fp6k128_fp4k256_bs128x128_stg3_vec`
- SM90: `sm90_t128x256x128_c1x1_vec`

**Tuning macros** (for library developers adding new kernels):

| Macro | Pattern | Use case |
|-------|---------|----------|
| `TUNED_LAUNCH_1D` | 1D grid-stride | cast, deinterleave, stack kernels |
| `TUNED_LAUNCH_ROW` | row-per-block | pack_triangle, antisymmetrize_pack |
| `TUNED_LAUNCH_2D` | 2D element-wise | enforce_hermitian, antisymmetrize_to_triangle |

Sweep parameters: blockDim in {32, 64, 128, 256, 512, 1024} x gridDim capped at {sm_count, 2x, 4x, 8x, full}. Timing: 3 warmup + 10 timed launches. CUDA graph capture is auto-detected (skips tuning during capture).

**Not tunable:** 2D transposed kernels with `__shared__` memory tiles (blockDim is coupled to tile array sizes).

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `CUTLASS_TUNECACHE_PATH` | current working directory | Directory for all tune cache files (kernel cache, strategy cache). If set, must point to an existing directory — the program exits with an error if the directory does not exist. |

Both the kernel-level cache (`cutlass_kernel_cache_*.txt`) and the strategy cache (`cutlass_strategy_cache_*.txt`) respect this variable. This is useful for:

- **Shared cache directories** across multiple build configurations
- **Persistent storage** when binaries are launched from varying working directories
- **Centralized tuning data** in cluster/container environments

```bash
# Direct all cache I/O to a shared directory
export CUTLASS_TUNECACHE_PATH=/data/cutlass_cache

# First run: tune and save
./example_complex_sm100 3328 3328 128 32 tune=true

# Subsequent runs from any directory: loads cached results automatically
cd /somewhere/else
./example_complex_sm100 3328 3328 128 32
```

If unset, cache files are read/written in the current working directory (default behavior). The `set_tune_cache_path()` and `set_kernel_tune_cache_path()` PIMPL API methods override this variable for their respective caches.

## Performance Optimization Guide

### 1. Choose the Right Precision

Lower precision = higher throughput but potentially lower accuracy. The accuracy
impact depends critically on the **input data distribution** — see
[Precision Accuracy Analysis](#precision-accuracy-analysis) for measured error data.

| Precision | Throughput | Accuracy (INT4 input) | Accuracy (FP16 input) | When to Use |
|-----------|-----------|----------------------|----------------------|-------------|
| FP8 E4M3 | Baseline | **Exact** (0 error) | ~0.1% relative | Default for most workloads |
| FP6 E3M2 | ~25% less bandwidth | **Exact** (0 error) | ~0.1% relative | Bandwidth-bound workloads (large K) |
| FP4 E2M1 | ~2x throughput | **~20% peak error** | ~50% peak error | **Not recommended** — see warning below |
| INT4 | Same as FP8 | Exact (via FP8) | N/A | Low-bit integer data (e.g., DSA-2000 QC) |

> **Warning — FP4 E2M1 accuracy:** FP4 cannot exactly represent the integers 5 and 7
> (quantized to 4 and 8 respectively). For INT4 sign-magnitude input [-7,7], this
> corrupts 2 of 15 non-zero magnitudes, producing **~20% worst-case error** relative
> to peak output values at all K. This error is structural (does not improve with
> larger K) and makes FP4 unsuitable for applications requiring faithful correlation
> of integer-quantized data. Use FP8 or FP6 E3M2 instead — both are lossless for
> integers in [-7,7].

### 2. Tile Shape Selection (Runtime)

All FP8 tile/cluster/stage configurations are compiled into a single binary and selectable at runtime via `GemmConfig` (internal) or `TileConfig` (PIMPL API). No rebuild required to change tile shapes. The strategy autotuner (`tune=true`) benchmarks all valid configs per architecture.

**SM120 (3 configs):** `128x64 3-stage` (HERK default, ~82 KB), `128x64 auto-stage`, `128x128 auto-stage` (GEMM optimal)

**SM100 (8+ configs):** `128x128 3-stage` (default), `128x64`, `128x256`, plus cluster variants (`1x2`, `2x2`) and 2SM cooperative (`256x128`, `256x256`)

**SM90 (6 configs):** `128x256 Coop` (default), `128x128 Coop`, Pingpong variants, `128x256 1x2 cluster`, `64x128 Coop`

**Block-scaled (FP6/FP4):** Tile shape still controlled by cmake (`COMPLEX_SM100_BLKSCALED_MMA_M/N`). SmallM (64×128) auto-selected when M ≤ 64 on SM100.

Use `benchmark_configs_sm100` to sweep all configs on your hardware, or `tune=true` for automatic selection.

### 3. Triangle Decomposition (HERK)

For Hermitian rank-K update (HERK), triangle-aware decomposition avoids computing the redundant upper triangle, saving up to 50% of FLOPs.

```bash
# Auto-adaptive (recommended)
./example_complex_sm100 4096 4096 512 32

# Manual slab count (tune for your problem)
./example_complex_sm100 4096 slabs=8

# Use sweep to find optimal slab count
./sweep_triangle_sm100 4096 1024 128
```

**Guidelines:**
- Auto-adaptive works well in most cases
- For large N with many batches: more slabs (8-16) can improve occupancy
- For small batch counts: fewer slabs (4-6) reduce launch overhead
- SM120: enable `graph=1` for repeated HERK calls (caches CUDA graphs)
- SM100: grouped GEMM path launches all slabs in a single kernel

### 4. Direct HERK Kernel

The direct HERK kernel combines FP8 MMA + triangle packing into a single kernel, avoiding intermediate N x N scratch buffers. Best at small K. Each batch element is computed independently — individual results are always preserved.

```bash
# Auto-select based on K (recommended)
./example_complex_sm100 4096 4096 256 32 direct=auto

# Force direct (always use direct kernel)
./example_complex_sm100 4096 4096 256 32 direct=1
```

**Crossover points (approximate):**
- Single HERK: direct wins when K <= N/4
- Batched HERK with tiling: direct wins when K <= N/2
- At K > N, baseline CUTLASS is always faster

### 5. Batch Tiling

For batched HERK, the library automatically tiles the batch dimension to keep intermediate buffers in L2 cache.

```bash
# Enabled by default
./example_complex_sm100 4096 4096 512 128 tile=1

# Disable to compare
./example_complex_sm100 4096 4096 512 128 tile=0
```

Scratch per batch element = 4 * N^2 bytes. With 24 MB L2 (GB10): N=1024 fits 6 batches, N=4096 fits 1 batch.

### 6. Kernel Autotuning

Overhead kernels (cast, pack, deinterleave, etc.) can be autotuned for optimal thread block dimensions.

```bash
# Tune and save results (writes cutlass_kernel_cache_{build_fingerprint}.txt)
./example_complex_sm100 4096 tune=2

# Full detail output
./example_complex_sm100 4096 tune=3

# Use cached results (no re-tuning)
./example_complex_sm100 4096 tune=1

# Full strategy autotune (sweeps ALL runtime options: mode, strategy, persistent, graph, etc.)
./example_complex_sm100 3328 3328 128 32 tune=true
```

The kernel tune cache (`cutlass_kernel_cache_{build_fingerprint}.txt`) uses one file per build configuration (mirroring the strategy cache naming). Re-run `tune=2` after changing CMake options.

The strategy cache (`cutlass_strategy_cache_{build_fingerprint}.txt`) uses one file per build configuration. `tune=true` sweeps ~12 combinations of HerkMode × HerkStrategy × CUDA graph × PersistentMode × herk_graph and caches the optimal configuration. Subsequent runs hit the cache instantly.

### 7. CUDA Graph Caching

For repeated HERK calls with the same dimensions (common in iterative algorithms), CUDA graph capture eliminates host-side launch overhead.

```bash
# Triangle graph (SM120)
./example_complex_sm100 4096 slabs=8 graph=1

# Baseline HERK graph
./example_complex_sm100 4096 4096 512 32 herk_graph=1
```

The graph cache holds 4 entries (LRU). Graphs are invalidated when dimensions or pointers change.

### 8. Prepare/Execute Pattern

When B is constant across many GEMM calls (e.g., correlator matrices, beamformer weights), pre-convert B once and reuse:

```cpp
cutlass_gemm_api::CutlassComplexGemm api;

// One-time: convert B to FP8 (and MXFP on Blackwell)
api.prepare_b(d_Br, d_Bi, N, K, batch,
              cutlass_gemm_api::ComputePrecision::FP8, stream);

// Per-call: only converts A
for (int i = 0; i < num_iterations; i++) {
    api.gemm_prepared(d_Ar[i], d_Ai[i], d_Cr, d_Ci, M, N, K, batch,
                      cutlass_gemm_api::OutputPrecision::FP32, 1.0f, 0.0f, stream);
}
```

#### Power GEMM variants (Fused |Re|^2 + |Im|^2)

For voltage beamformer workloads, the power variants eliminate separate Re/Im output and power computation:

```cpp
// FP16 → power
api.gemm_prepared_power(d_Ar, d_Ai, d_power, M, N, K, batch, 1.0f, 0.0f, stream);

// Pre-cast FP8 → power (skip internal FP16→FP8 cast)
api.gemm_prepared_power_fp8(d_fp8_interleaved, d_power, M, N, K, batch, 1.0f, 0.0f, stream);

// INT4 QC → power (uses pre-allocated internal FP8 buffer, no per-call malloc)
api.gemm_prepared_power_int4(d_qc_int4, d_power, M, N, K, batch, 1.0f, 0.0f, stream);
```

The `gemm_prepared_power_int4()` internally maintains a pre-allocated FP8 buffer that grows on demand (`cudaMalloc` once, reused across calls). This eliminates the ~20-40µs overhead from per-call `cudaMallocAsync`/`cudaFreeAsync`.

#### Channel fusion + payload batching (short-integration optimisation)

For short time integrations (n_time=8), M is very small and the GEMM is deeply memory-bound. Two levels of M-axis batching exploit constant B weights:

1. **Channel fusion** (`ch_fuse`): Fuse `ch_fuse` consecutive channels along M → `M_fused = n_time × ch_fuse`
2. **Payload batching** (`n_payloads`): Concatenate payloads along M → `M_total = n_time × ch_fuse × n_payloads`

This is implemented in `voltage_pipeline.cu` via `beamform_batched()`:

```cpp
ggp::VoltagePipeline::Config cfg;
cfg.n_time = 8;
cfg.n_channels = 1600;
cfg.ch_fuse = 8;         // Fuse 8 channels along M → M=64, batch=200
cfg.max_payloads = 8;    // Buffer for up to 8 payloads → M=512

ggp::VoltagePipeline pipe(cfg);
pipe.set_weights(d_weights, stream);

// Process 8 payloads in one GEMM call (M=512, batch=200)
pipe.beamform_batched(d_output, d_qc_inputs, 8, scale, stream);
```

A fused kernel (`qc_fused_transpose_fp8_kernel`) performs QC INT4 transpose + pol-split + channel fusion + payload batching + INT4→FP8 conversion in a single memory pass, then calls `gemm_prepared_power_fp8()` directly with the pre-cast FP8 data.

**GB10 performance (n_ant=1664, n_beam=4000, n_ch=1600):**

| Configuration | M | Batch | TFLOPS | TC Util | Speedup |
|--------------|--:|------:|-------:|--------:|--------:|
| Unfused (n_time=8) | 8 | 1600 | 1.4 | 0.6% | 1x |
| ch_fuse=8 | 64 | 200 | 10.7 | 4.5% | 7.6x |
| ch_fuse=8 × 8 payloads | 512 | 200 | 46.9 | 19.7% | 33x |

### 9. Codegen Flags (Advanced)

These compile-time flags affect the generated PTX/SASS code quality:

```bash
# Enable device LTO (helps Hopper wgmma serialization)
cmake .. -DCOMPLEX_FP8_DEVICE_LTO=ON

# Limit register usage (improves occupancy, may hurt ILP)
cmake .. -DCOMPLEX_FP8_MAX_REGISTERS=255

# Extra vectorization (wider loads/stores for overhead kernels)
cmake .. -DCOMPLEX_FP8_EXTRA_VECTORIZATION=ON   # ON by default
```

Use `build_sweep_sm120.sh` or `build_sweep_sm90.sh` to systematically test different flag combinations, then `find_best_flags.sh` to benchmark across all builds:

```bash
# Build all tier 1 configs (configure + build + ctest)
bash build_sweep_sm120.sh --tier 1

# Find the best config for a given problem
bash find_best_flags.sh --N 3328 --K 128 --batch 32
```

### 10. Build Time vs Runtime Tradeoffs

| Configuration | Build Time | Runtime Benefit |
|--------------|-----------|-----------------|
| FP8 only | ~15 min | Baseline |
| +FP4 | ~30 min | 2x throughput (lossy) |
| +FP6 | ~60 min | 25% less bandwidth |
| +Small-M | ~90 min | 2x efficiency for M<=64 (auto on SM100 with FP6/FP4) |
| +Device LTO | ~2x build time | 5-15% speedup (Hopper) |

All FP8 tile/cluster configs are now always compiled (no `MULTI_CONFIG` build flag needed). SmallM is auto-enabled on SM100 when FP6 or FP4 is enabled. For production deployments, build with all precisions you need and run `tune=true` once on the target hardware to find the optimal config.

### 11. Build Sweep Scripts

Three scripts automate compile-time configuration sweeping and benchmarking:

#### `build_sweep_sm120.sh` — SM120/SM100 Build Sweep

Cycles through meaningful compile-time combinations, doing cmake configure + build + ctest for each. Does NOT benchmark — use `find_best_flags.sh` for that.

```bash
bash build_sweep_sm120.sh                    # tier 1 (optimal guesses + baseline)
bash build_sweep_sm120.sh --tier 2           # + codegen flags
bash build_sweep_sm120.sh --tier all         # + block-scaled + MMA_N variations
bash build_sweep_sm120.sh --dry-run          # preview configs without building
bash build_sweep_sm120.sh --clean            # remove sweep_builds/ and sweep_results/
```

**Tier 1** (3 configs): `optimal_herk` (STAGES=3), `optimal_gemm` (STAGES=3 + TILE_N=128), `baseline`

**Tier 2** (+4 configs): codegen flags on top of stg3 — LTO, reg255, novec, LTO+reg255

**Tier all** (+4 configs): block-scaled MMA_N=64, FP8 MMA_N=64, GEMM+LTO, novec+LTO (11 total)

Output: build directories in `sweep_builds/`, logs in `sweep_results/`. Each config is validated with ctest. Status tracking: `CONFIGURE_FAIL` / `BUILD_FAIL` / `TEST_FAIL` / `OK`.

#### `build_sweep_sm90.sh` — SM90 Build Sweep

Same structure as SM120 but with Hopper-specific options (tile shapes, cluster, PingPong schedule). Device LTO and MAX_REGISTERS are excluded (broken on SM90).

```bash
bash build_sweep_sm90.sh --tier 1            # optimal + tile shapes (4 configs)
bash build_sweep_sm90.sh --tier all          # + cross-product combinations (16 total)
bash build_sweep_sm90.sh --dry-run
```

**Tier 1** (4 configs): `optimal` (defaults: 128x256x128), `t256x128`, `t128x128`, `t128x256_novec`

**Tier 2** (+5 configs): PingPong, NoFastAccum, NoGroupedGemm, Cluster 2x1/1x2

**Tier all** (+7 configs): cross-product tile×schedule combinations (16 total)

Output: `sweep_builds_sm90/` and `sweep_results_sm90/`.

#### `find_best_flags.sh` — Benchmark Across Builds

Runs a single problem across all pre-built configurations to find the best compile-time config. Two-phase per build: (1) `tune=true` to autotune strategy + kernel params (timing discarded), then (2) `tune=0 mode=bench` for clean measurement.

```bash
# Default problem (N=1664, K=16, batch=256) across SM120 builds
bash find_best_flags.sh

# Custom problem
bash find_best_flags.sh --N 4096 --K 64 --batch 128

# SM90 builds
bash find_best_flags.sh --build-dir sweep_builds_sm90/

# List builds without running
bash find_best_flags.sh --dry-run
```

Auto-detects both `example_complex_sm100` (SM100/SM120) and `example_complex_fp8` (SM90) binaries in each build subdirectory.

**Output:** Per-build line with time, ms/item, TFLOPS, and GB/s. Ranked summary table. Best config highlighted with reproduce command. CSV saved to `sweep_results/`.

### Production Deployment Summary

> **IMPORTANT: HERK vs GEMM builds require different configurations on SM120 (GB10).**
> The cutlass_interface library must be compiled differently depending on the workload:
> - **HERK workloads** (XEngine correlation): use the default 128×64 FP8 tile (3-stage pipeline)
> - **GEMM workloads** (dedispersion, beamforming): use `-DCOMPLEX_SM100_FP8_TILE_N=128` for the wider 128×128 tile (2-stage auto-carveout)
>
> This is because SM120 has only 99 KB SMEM. The 128×128 tile with 3-stage pipeline
> needs 110 KB and cannot launch. The library auto-selects 128×64 for HERK (where
> the direct HERK kernel bypasses CUTLASS tiles anyway), but GEMM-heavy consumers
> benefit from the wider tile even at 2 stages.

**HERK-optimized build (XEngine / FTD):**

```bash
cmake .. -DCUTLASS_DIR=/path/to/cutlass \
         -DCOMPLEX_FP8_ARCH=120a \
         -DCOMPLEX_SM100_ENABLE_FP6=ON \
         -DCOMPLEX_SM100_ENABLE_FP4=ON \
         -DCOMPLEX_FP8_SM100_STAGES=3
```

Auto-selects 128×64 FP8 tile (86 KB SharedStorage, 3-stage). The direct HERK kernel
(24 KB SMEM) dominates HERK performance and does not use CUTLASS tiles, so the FP8
tile choice has no impact on HERK throughput.

**GEMM-optimized build (dedisp_tcfdd / beamformer):**

```bash
cmake .. -DCUTLASS_DIR=/path/to/cutlass \
         -DCOMPLEX_FP8_ARCH=120a \
         -DCOMPLEX_SM100_ENABLE_FP6=ON \
         -DCOMPLEX_SM100_ENABLE_FP4=ON \
         -DCOMPLEX_FP8_SM100_STAGES=3 \
         -DCOMPLEX_SM100_FP8_TILE_N=128
```

Forces 128×128 FP8 tile (78 KB SharedStorage, 2-stage auto-carveout). 5-28% faster
for large square/rectangular GEMMs (N≥2048) compared to the 128×64 tile.

**GB10 HERK benchmark (Feb 2026) — apples-to-apples, both compute packed triangle:**

| Problem | HERK-opt (128×64) | GEMM-opt (128×128) | TCC fp8 |
|---------|------------------:|-------------------:|--------:|
| N=3328 K=16 b=128 | 19.1 ms | 19.4 ms | 115.7 ms |
| N=3328 K=256 b=128 | 48.4 ms | 48.2 ms | 117.1 ms |
| FP8 baseline (d=0) N=3328 K=256 b=128 | 139.7 ms | 141.6 ms | — |

**GB10 GEMM tile comparison (Feb 2026) — CUTLASS full GEMM only:**

| Problem | HERK-opt (128×64) | GEMM-opt (128×128) |
|---------|------------------:|-------------------:|
| N=1024 K=1024 b=32 | 6.6 ms | 6.5 ms |
| N=2048 K=2048 b=32 | 73.1 ms | **69.6 ms** |
| N=4096 K=4096 b=32 | 171.6 ms | **165.5 ms** |

> **TCC is not shown in the GEMM table.** TCC is a correlation engine (HERK) — it
> computes only the Hermitian triangle (~N²/2 elements), not a full N×N matrix
> multiplication. CUTLASS's standalone GEMM computes the full N×N output (N² elements).
> Comparing TCC times against full GEMM times is misleading because TCC is doing
> fundamentally less work. For Hermitian workloads, use the HERK table above.

Key findings:
- **HERK: identical performance** — direct kernel (24 KB) bypasses CUTLASS tiles
- **HERK: CUTLASS 1.4–6x faster than TCC** across all tested K values
- **GEMM: 128×128 tile is 5-28% faster** at N≥2048, identical at N≤1024

**Optimal runtime parameters:**

| Parameter | Value | Notes |
|-----------|-------|-------|
| `direct` | `auto` (default) | Direct HERK at small K, baseline at large K |
| `tune` | `2` (first run) | Autotune overhead kernels, then use cached results |
| `graph` | `1` | CUDA graph for repeated HERK calls |
| `tile` | `1` (default) | Batch tiling to keep scratch in L2 |
| `persistent` | `auto` (default) | Persistent kernel when K ≤ 64 |

**First-run workflow:**

```bash
# 1. Tune overhead kernels (writes cutlass_kernel_cache_{build_fingerprint}.txt)
./example_complex_sm100 3328 3328 256 128 tune=2

# 2. Full strategy autotune (writes cutlass_strategy_cache_*.txt)
./example_complex_sm100 3328 3328 256 128 tune=true

# 3. Subsequent runs use cached results automatically
./example_complex_sm100 3328 3328 256 128 graph=1
```

## File Map

```
cutlass_interface/
├── cutlass_gemm_api.h              # Public API header (include this)
├── cutlass_gemm_api.cu             # PIMPL implementation
├── gemm_complex_fp8.hpp            # SM90 (Hopper) umbrella header
├── gemm_complex_fp8/               # SM90 sub-headers
│   ├── config.hpp
│   ├── cast_kernels.hpp
│   ├── pack_kernels.hpp
│   ├── type_chains.hpp
│   ├── buffers.hpp
│   ├── herk_kernel.hpp
│   ├── api_impl.hpp
│   ├── herk_impl.hpp
│   ├── dispatch_impl.hpp
│   └── triangle_impl.hpp
├── gemm_complex_sm100.hpp          # SM100/SM120 (Blackwell) umbrella header
├── gemm_complex_sm100/             # SM100/SM120 sub-headers
│   ├── config.hpp
│   ├── cast_kernels.hpp
│   ├── pack_kernels.hpp
│   ├── int4_kernels.hpp
│   ├── fp6_kernels.hpp
│   ├── fp4_kernels.hpp
│   ├── mxfp_kernels.hpp
│   ├── herk_kernel.hpp
│   ├── buffers.hpp
│   ├── api_impl.hpp
│   ├── herk_impl.hpp
│   ├── dispatch_impl.hpp
│   └── triangle_impl.hpp
├── gemm_sm100_type_chains.hpp      # CUTLASS type chain definitions
├── gemm_blockscaled_dispatch.h     # Block-scaled forward declarations
├── gemm_blockscaled_dispatch.cu    # Block-scaled kernel instantiations
├── system_info.hpp                 # GPU info and roofline helpers
├── tune_cache.hpp                  # Kernel autotuning cache (CUTLASS_TUNECACHE_PATH support)
├── strategy_cache.hpp              # Strategy-level tune cache + run_autotune() template
├── CMakeLists.txt                  # Build configuration
├── example_usage.cu                # SM90 example/benchmark
├── example_usage_sm100.cu          # SM100/SM120 example/benchmark
├── benchmark_configs.cu            # SM90 parameter sweep
├── benchmark_configs_sm100.cu      # SM100/SM120 parameter sweep
├── sweep_triangle_sm100.cu         # Triangle decomposition sweep
├── test_fp6_batched.cu             # FP6/FP4 diagnostic test
├── test_gemm_api.cu                # Unified GEMM/HERK API test (all precisions)
├── test_herk_int4.cu               # INT4 HERK test
├── test_tune_api.cu                # Strategy autotuning API test
├── build_sweep_sm120.sh            # SM120/SM100 compile-time config sweep (build + ctest)
├── build_sweep_sm90.sh             # SM90 compile-time config sweep (build + ctest)
├── find_best_flags.sh              # Benchmark across builds to find best config
└── herk_vs_tcc.sh                  # HERK vs TCC benchmark sweep script
```

## HERK Performance Study: CUTLASS vs TCC (SM120 / GB10 Spark)

A comprehensive benchmark study compared this CUTLASS HERK implementation against the Tensor Core Correlator (TCC) across two workload classes on a GB10 Spark (SM121, 48 SMs). Full report: `reports/herk_vs_tcc_benchmark_report.txt`. Raw data: `herk_results/herk_vs_tcc_20260224_154412.csv`.

> **Note (Feb 2026):** The original benchmark was run before fixing a cp.async pipeline
> drain bug in the direct HERK kernel (`herk_kernel.hpp`). At K <= 64 (K_CHUNK), the
> kernel read uninitialized shared memory, producing incorrect results.
> **The stg3 small-K results below are therefore invalid** and need re-benchmarking.
> The direct kernel correctness bug is now fixed — `crosscheck=gpu` passes bit-exact at
> all K values. Additionally, FP8 type chains now use `StageCountAutoCarveout` on SM120
> (auto-selects stages that fit in 99 KB SMEM), so FP8 baseline GEMM works on SM120.
> Re-run the benchmark with `bash herk_vs_tcc.sh` to obtain corrected numbers.

### Results Summary (pre-fix, see note above)

**Square problems (N=K, batch=32):**

| Problem | Best Config | CUTLASS (ms) | TCC (ms) | Speedup |
|---------|-------------|-------------|----------|---------|
| 256x256 | stg3_reg255 | 0.09 | 0.195 | **2.17x** |
| 512x512 | stg3 | 0.50 | 0.783 | **1.57x** |
| 1024x1024 | stg3 | 2.07 | 3.539 | **1.71x** |
| 2048x2048 | stg3_reg255 | 16.79 | 23.677 | **1.41x** |
| 4096x4096 | stg3_novec | 44.46 | 234.920 | **5.28x** |

**Small-K problems (N=3328, batch=128) — DSA-2000 correlator workload:**

| K | Best Config | CUTLASS (ms) | TCC (ms) | Speedup |
|---|-------------|-------------|----------|---------|
| 16 | stg3 | 23.28 | 115.924 | **4.98x** |
| 32 | stg3 | 23.20 | 116.137 | **5.01x** |
| 64 | stg3_reg255 | 23.84 | 115.704 | **4.85x** |
| 128 | stg3 | 26.45 | 116.034 | **4.39x** |
| 256 | stg3_reg255 | 30.19 | 116.576 | **3.86x** |
| 512 | stg3 | 36.96 | 121.628 | **3.29x** |

### Key Findings

1. **`COMPLEX_FP8_SM100_STAGES=3` is set by default on SM120.** SM120 uses a 128×64 FP8 tile (vs 128×128 on SM100), so 3-stage SharedStorage fits in 99 KB SMEM (~82 KB). `StageCount<3>` is used directly — no auto carveout fallback needed. Block-scaled chains (FP6/FP4) use separate tile shapes and auto carveout.

2. **Direct HERK kernel** is correct after the cp.async drain fix. Verified bit-exact per-batch via `crosscheck=gpu` at K=32, 64, 128, 256. The direct path is 1.5-3.4x faster than baseline at small K.

3. **`direct=auto` works correctly** — no manual override needed. The auto mode selects direct HERK at small K and baseline at large K.

4. **Secondary compile flags (reg255, novec) have <5% impact** within the stg3 family. Pipeline stages dominate.

5. **Peak: 395.7 effective TFLOPS** at N=4096 (stg3_novec, Tri+Graph, batch=32) — valid for large K where baseline path works.

### Recommended Build for SM120

```bash
cmake .. -DCUTLASS_DIR=/path/to/cutlass \
         -DCOMPLEX_FP8_ARCH=120a \
         -DCOMPLEX_SM100_ENABLE_FP6=ON \
         -DCOMPLEX_SM100_ENABLE_FP4=ON \
         -DCOMPLEX_FP8_SM100_STAGES=3
```

### Maximal Batch Comparison (N=3328, K=16, post-fix)

After fixing the cp.async drain bug, a focused comparison was run at the DSA-2000 correlator problem size (N=3328, K=16) scaling batch count to each system's practical memory limit on GB10 (120 GB unified memory, budget capped at 60 GB for stability).

CUTLASS uses packed lower-triangle output (~21 MB/batch), while TCC uses full N×N complex float32 (~169 MB/batch) — an **8x memory advantage** for CUTLASS.

| System | Batch | Time (ms) | TFLOPS | Memory (GB) | ms/item | Speedup |
|--------|------:|----------:|-------:|------------:|--------:|--------:|
| CUTLASS direct+Tri+Graph | 128 | 23.84 | 5.7 | 2.7 | 0.186 | **4.9x** |
| TCC fp8 | 128 | 116.06 | 1.56 | 21.6 | 0.907 | — |
| CUTLASS | 300 | 55.67 | 5.7 | 6.3 | 0.186 | **4.9x** |
| TCC fp8 | 300 | 270.60 | 1.57 | 50.7 | 0.902 | — |
| CUTLASS | 2500 | 462.57 | 5.7 | 52.5 | 0.185 | — |
| TCC | 2500 | — | — | 422 | OOM | OOM |

Key observations:
- **CUTLASS is 3.7–4.9x faster** at every batch size tested
- **CUTLASS scales to 2500+ batches** in the same 60 GB that TCC needs for 300
- Both systems are memory-bandwidth bound at K=16 (IO=48 FLOPs/byte vs ridge=435), but CUTLASS achieves **120 GB/s** vs TCC's **25 GB/s**
- Per-item throughput is constant at **0.186 ms/item** for CUTLASS — GPU is fully saturated at batch=1 (5460 triangle blocks >> 192 max concurrent blocks on 48 SMs)

### All-Precision Comparison (N=3328, batch=128)

Complete throughput comparison across all precisions. FP6 requires K >= 128, FP4 requires K >= 256. INT4 uses the PIMPL API (`herk(..., InputPrecision::INT4)`). All CUTLASS results use Tri+Graph mode. "—" = precision unavailable at that K.

**Time (ms) and effective TFLOPS:**

| K | CUTLASS FP8 direct | CUTLASS INT4 | CUTLASS FP6 | CUTLASS FP4 | TCC FP8 | TCC FP4 |
|--:|------------------:|-------------:|------------:|------------:|--------:|--------:|
| 16 | 23.84 / 5.7T | 24.05 / 7.5T | — | — | 116.1 / 1.6T | — |
| 128 | 29.77 / 36.6T | 31.97 / 45.4T | 95.00 / 11.5T | — | 116.0 / 12.5T | — |
| 256 | 48.05 / 45.3T | 50.13 / 57.9T | 94.91 / 22.9T | 88.39 / 24.6T | 116.5 / 24.9T | 116.2 / 25.0T |
| 512 | 81.04 / 53.7T | 88.19 / 65.8T | 132.65 / 32.8T | 118.38 / 36.8T | 121.7 / 47.7T | 117.4 / 49.5T |

**Speedup vs TCC FP8:**

| K | FP8 direct | INT4 | FP6 | FP4 | TCC FP4 |
|--:|----------:|-----:|----:|----:|--------:|
| 16 | **4.87x** | **4.83x** | — | — | — |
| 128 | **3.90x** | **3.63x** | **1.22x** | — | — |
| 256 | **2.42x** | **2.32x** | **1.23x** | **1.32x** | 1.00x |
| 512 | **1.50x** | **1.38x** | 0.92x | 1.03x | 1.04x |

**Maximal batch scaling (K=128):**

| System | Batch | Time (ms) | TFLOPS | Memory (GB) | ms/item |
|--------|------:|----------:|-------:|------------:|--------:|
| CUTLASS FP8 direct | 128 | 29.77 | 36.6 | 2.7 | 0.233 |
| CUTLASS INT4 | 128 | 31.97 | 45.4 | 2.7 | 0.250 |
| CUTLASS FP6 Tri+Graph | 128 | 95.00 | 11.5 | 2.9 | 0.742 |
| TCC FP8 | 128 | 116.04 | 12.5 | 21.6 | 0.907 |
| CUTLASS FP6 | 300 | 225.72 | 11.3 | 6.9 | 0.752 |
| TCC FP8 | 300 | 272.79 | 12.5 | 50.7 | 0.909 |
| CUTLASS FP6 | 2500 | 1843.07 | 11.5 | 57.5 | 0.737 |
| TCC | 2500 | — | — | 422 | OOM |

Key observations:
- **FP8 direct and INT4 are the fastest paths** (1.4–4.9x vs TCC), sharing the same direct kernel engine. INT4 adds only ~7% overhead from INT4→FP16 conversion
- **FP4 beats TCC FP8 at K=256** (1.32x) but is comparable at K=512. FP4 has 2x theoretical throughput but MXFP overhead is higher than FP6. **Caveat:** FP4 introduces ~20% worst-case error for INT4 input — see [Precision Accuracy Analysis](#precision-accuracy-analysis)
- **FP6 is 1.2x faster than TCC** at K=128/256 but slightly slower at K=512 — MXFP scale factor overhead erodes the bandwidth advantage at compute-rich K
- **TCC FP4 vs TCC FP8**: essentially identical speed (FP4 halves bandwidth but TCC is memory-bound on output)
- All CUTLASS paths retain the **8x memory advantage** from packed triangle output

### SMEM Pre-check and Test Skipping

`GemmComplexSm100::fp8_baseline_available()` checks at runtime whether the FP8 GemmKernel SharedStorage fits in device SMEM. With the auto carveout fix, FP8 kernels on SM120 now fit (~78 KB vs 99 KB limit), so this check passes. The benchmark uses this to gate FP8-dependent tests — if a future tile/stage configuration causes overflow, tests print `SKIPPED` instead of noisy `[CUTLASS] initialize FAILED` diagnostics.

### Reproducing the Study

```bash
# Full sweep (builds configs, runs all benchmarks, generates CSV)
bash herk_vs_tcc.sh

# Quick re-run with existing binaries
bash herk_vs_tcc.sh --skip-build

# See all options
bash herk_vs_tcc.sh --help
```

## Precision Accuracy Analysis

Quantitative accuracy measurements for each compute precision against an FP64
CPU reference HERK. All tests use N=64, batch=8, with INT4 sign-magnitude input
[-7,7] (DSA-2000 native format). Errors are reported as absolute values;
"% of peak" is the worst-case error divided by the largest reference output element.

### FP4 E2M1 Quantization Table

FP4 E2M1 represents values with 1 sign + 2 exponent + 1 mantissa bits.
For INT4 input magnitudes 0–7:

| Input | FP4 output | Error | Status |
|------:|-----------:|------:|--------|
| 0 | 0 | 0 | Exact |
| 1 | 1 | 0 | Exact |
| 2 | 2 | 0 | Exact |
| 3 | 3 | 0 | Exact |
| 4 | 4 | 0 | Exact |
| **5** | **4** | **1** | **Lossy** |
| 6 | 6 | 0 | Exact |
| **7** | **8** | **1** | **Lossy** |

### Error vs K (INT4 input [-7,7], N=64, batch=8)

| K | FP8 max_abs | FP6 max_abs | FP4 max_abs | FP4 mean_abs | FP4 % of peak |
|----:|------------:|------------:|------------:|-------------:|--------------:|
| 16 | 0 | 0 | 212 | 28 | 23.8% |
| 32 | 0 | 0 | 394 | 43 | 23.0% |
| 64 | 0 | 0 | 673 | 64 | 23.1% |
| 128 | 0 | 0 | 1,163 | 99 | 20.6% |
| 256 | 0 | 0 | 2,301 | 157 | 21.3% |
| 512 | 0 | 0 | 4,081 | 252 | 19.8% |
| 1024 | 0 | 0 | 8,230 | 424 | 19.9% |
| 2048 | 0 | 0 | 15,226 | 724 | 19.1% |
| 4096 | 0 | 0 | 30,040 | 1,280 | 18.9% |

### Key Findings

1. **FP8 E4M3 and FP6 E3M2 are exact for INT4 input.** Both formats can represent
   every integer in [-7,7] without rounding, so HERK results computed with FP8 or
   FP6 match the FP64 reference to machine precision (FP32 accumulation).

2. **FP4 E2M1 introduces ~20% worst-case error at all K values.** The error comes
   from quantizing 5→4 and 7→8 (2 of 15 non-zero magnitudes). This is a structural
   error that scales linearly with K in absolute terms but remains constant as a
   percentage of peak output (~19–24%).

3. **FP4 mean error is much smaller (~1–3% of peak)** because most output elements
   involve a mix of exact and inexact input values. However, the worst-case error
   is too large for scientific applications requiring faithful correlation.

4. **FP16 output overflows at K > 668** for INT4 input, since the worst-case
   diagonal element is K × 98 (all inputs equal to 7). Use FP32 output for K > 512.

### Recommendation

For DSA-2000 and similar radio astronomy correlators with INT4-quantized input:

- **Use FP8 E4M3** (default) — exact, fastest via direct HERK kernel
- **FP6 E3M2 is a safe alternative** — also exact for INT4, 25% less bandwidth
- **Do not use FP4 E2M1** for production correlation — ~20% worst-case error is
  unacceptable for visibility data that feeds calibration and imaging pipelines
- FP4 may be acceptable for non-scientific workloads (ML inference, approximate
  nearest-neighbor) where ~20% error is tolerable

To reproduce: `python3 analyze_precision.py --n 64 --int4 --batch 8`

## Supported Precisions

| Precision | Input | Output | INT4 Accuracy | FP16 Accuracy | Blackwell | Hopper |
|-----------|-------|--------|---------------|---------------|-----------|--------|
| FP8 E4M3 | FP16 complex | FP16/FP32 | **Exact** | ~0.1% relative | SM100/SM120 | SM90 |
| FP6 E3M2 | FP16 complex | FP16/FP32 | **Exact** | ~0.1% relative | SM100/SM120 | — |
| FP4 E2M1 | FP16 complex | FP16/FP32 | **~20% peak error** | ~50% peak error | SM100/SM120 | — |
| INT4 | INT4 sign-mag | FP16/FP32 | Exact (via FP8) | N/A | SM100/SM120 | SM90 |
| INT8 | INT8 | FP32 | Exact | N/A | SM100 only | — |

> **INT4 accuracy column** measures worst-case error vs FP64 reference for INT4
> sign-magnitude input [-7,7]. FP8 and FP6 are lossless for this input range.
> FP4 quantizes 5→4 and 7→8, introducing structural error that does not diminish
> with K. See [Precision Accuracy Analysis](#precision-accuracy-analysis) for details.

## Known Issues

1. **C7510 wgmma serialization (Hopper only):** nvcc inserts unnecessary `wgmma.fence` barriers at function boundaries, causing 15-30% throughput loss vs cuBLAS. Mitigations: CUDA 12.9+ (improved inlining), `--maxrregcount=255`, device LTO (`-dlto`), CUTLASS 4.x kernel restructuring. Enable verbose ptxas to diagnose: `cmake .. -DCOMPLEX_FP8_VERBOSE_PTXAS=ON`.

2. **Device LTO broken on SM90 (Hopper):** nvcc/nvlink bug strips the `a` suffix from arch-accelerated targets (`sm_90a` → `sm_90`), making ptxas reject wgmma/setmaxnreg instructions. CMake auto-disables `COMPLEX_FP8_DEVICE_LTO` on Hopper with a warning. Also unsupported on aarch64. Blackwell (SM100/SM120) is not affected.

3. **SM90 `--maxrregcount=255`:** ptxas rejects this value for large CUTLASS kernels ("Too big maxrregcount value specified, will be ignored"). CMake auto-skips on Hopper. <5% impact per benchmark study.

4. **SM120 grouped GEMM:** SM120 `mma_builder` has `static_assert(!IsPtrArrayKernel)`. Grouped GEMM and `GroupedKernelTypeChain` are SM100-only (gated behind `#ifndef COMPLEX_FP8_SM100_TARGET_SM120`). SM120 uses per-slab loop with optional CUDA graph instead.

5. **SM120 MMA_N=64 does not compile:** TMA SfAtom constraint with block-scaled types requires M_tile >= 128. Configs with `COMPLEX_FP8_SM100_MMA_N=64` fail at compile time on SM120.

6. **Build times:** Heavy template instantiation. Each kernel config adds ~15 min. FP6+FP4 adds ~45 min. Multi-config ON adds ~3h (SM100 only).
