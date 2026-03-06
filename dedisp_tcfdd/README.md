# DedispersionSim

**High-Performance Dedispersion Simulation & Benchmarking Library**

A C++/CUDA library for simulating and benchmarking radio astronomy dedispersion algorithms, optimized for detecting pulsars and Fast Radio Bursts (FRBs).

---

## Table of Contents

- [Overview](#overview)
  - [Key Features](#key-features)
  - [Target Hardware](#target-hardware)
- [Architecture](#architecture)
  - [File Structure](#file-structure)
  - [Core Data Structures](#core-data-structures)
  - [Configuration Parameters](#configuration-parameters)
- [Dedispersion Algorithms](#dedispersion-algorithms)
  - [Brute Force Dedispersion](#brute-force-dedispersion)
  - [Fast Dispersion Measure Transform (FDMT)](#fast-dispersion-measure-transform-fdmt)
  - [Fourier Domain Dedispersion (FDD)](#fourier-domain-dedispersion-fdd)
  - [Astro-Accelerate Tiled](#astro-accelerate-tiled)
  - [Algorithm Comparison](#algorithm-comparison)
- [GPU Implementation](#gpu-implementation)
  - [CUDA Architecture](#cuda-architecture)
  - [GPU Kernels](#gpu-kernels)
  - [Memory Layout](#memory-layout)
  - [Precision Modes](#precision-modes)
  - [CUTLASS FDD Modes (Blackwell)](#cutlass-fdd-modes-blackwell)
- [API Reference](#api-reference)
  - [DedispApplication Class](#dedispplication-class)
  - [Core Algorithm Methods](#core-algorithm-methods)
  - [Batched Processing Methods](#batched-processing-methods)
  - [Signal Simulation](#signal-simulation)
  - [Candidate Detection](#candidate-detection)
- [Command Line Interface](#command-line-interface)
  - [Basic Usage](#basic-usage)
  - [Algorithm Selection](#algorithm-selection)
  - [Common Options](#common-options)
  - [GPU-Specific Options](#gpu-specific-options)
  - [Example Commands](#example-commands)
- [Performance](#performance)
  - [Computational Complexity](#computational-complexity)
  - [Memory Bandwidth](#memory-bandwidth)
  - [Scaling Predictions](#scaling-predictions)
  - [Arithmetic Intensity](#arithmetic-intensity)
- [Building](#building)
  - [Prerequisites](#prerequisites)
  - [Build Steps](#build-steps)
  - [Build Targets](#build-targets)
  - [CUTLASS Support](#cutlass-support-optional)
- [References](#references)

---

## Overview

DedispersionSim is a high-performance simulation and benchmarking tool designed for radio astronomy data processing. Its primary function is to simulate the detection and removal of dispersion caused by the Interstellar Medium (ISM) on pulsed radio signals from pulsars and Fast Radio Bursts (FRBs).

The library implements multiple dedispersion algorithms optimized for different hardware architectures, ranging from traditional CPU-based methods to modern GPU-accelerated approaches leveraging NVIDIA CUDA, cuFFT, and cuBLAS.

### Key Features

- **Four distinct dedispersion algorithms** (Brute Force, FDMT, FDD, Astro-Accel)
- **GPU-accelerated implementations** using CUDA 12+
- **Batched and pipelined processing** for high-throughput scenarios
- **Single and double precision** floating-point arithmetic support
- **Signal simulation** with configurable noise, pulse injection, and scattering
- **Performance estimation** and benchmarking utilities
- **Matched filtering** for pulse candidate detection

### Target Hardware

The GPU implementations are optimized for NVIDIA Hopper (H100) and Blackwell architectures with compute capability 9.0+. The library leverages Tensor Cores for accelerated GEMM operations in the FDD algorithm. On Blackwell GPUs (SM100/SM120), optional CUTLASS integration provides FP8, FP6, and FP4 tensor core GEMM backends with FP32 accumulation and output.

---

## Architecture

### File Structure

| File | Description |
|------|-------------|
| `dedisp.cpp` | Main CPU implementation with all algorithms |
| `gpu_dedisp.cu` | GPU-accelerated dedispersion base implementation |
| `gpu_fdd.cu` | GPU Fourier Domain Dedispersion with cuBLAS |
| `gpu_fdd_improved.cu` | Optimized GPU FDD with cuRAND and FP16 support |
| `gpu_fdd_improved_19T.cu` | High-throughput FDD targeting 19 TFLOPS |
| `gpu_fdd_improved_cuda13x.cu` | FDD optimized for CUDA 13.x features |
| `fdmt_perfect.cpp` | CPU-optimized FDMT reference implementation |
| `CMakeLists.txt` | CMake build configuration |

### Core Data Structures

The library uses a templated design supporting both `float` (single) and `double` precision:

```cpp
template<typename Real>
class DedispApplication {
  using Complex = std::complex<Real>;
  using CMatrix = std::vector<std::vector<Complex>>;
  using RMatrix = std::vector<std::vector<Real>>;
  using RBatchMatrix = std::vector<RMatrix>;
  // ...
};
```

### Configuration Parameters

The `SimParams` structure contains all simulation configuration:

| Parameter | Type | Description |
|-----------|------|-------------|
| `num_freq_channels` | `size_t` | Number of frequency channels (power of 2) |
| `num_time_samples` | `size_t` | Number of time samples (power of 2) |
| `num_dm_trials` | `int` | Number of DM trials to search |
| `f_min` / `f_max` | `Real` | Frequency range in MHz |
| `max_dm_search` | `Real` | Maximum DM to search (pc/cm³) |
| `batch_size` | `int` | Items per pipeline batch |
| `num_pipelines` | `int` | Number of parallel pipelines |

---

## Dedispersion Algorithms

### Brute Force Dedispersion

The simplest form of incoherent dedispersion. For each DM trial and frequency channel, the time series is shifted by the corresponding dispersive delay and summed.

**Complexity:** O(N_f × N_DM × N_t)

**Reference:** Taylor, J. H. (1974). "A method for measuring dispersion in pulsar signals." *Astronomy and Astrophysics Supplement Series*, 15, 367-376.

### Fast Dispersion Measure Transform (FDMT)

A recursive dynamic programming approach that exploits the quadratic nature of dispersion. It recursively combines frequency-time sub-bands using sparse transformation matrices in the Fourier domain.

**Complexity:** O(N_t × (N_f × log₂N_f + N_DM))

The implementation uses Sparse Matrix-Matrix Multiplication (SpMM) for the core computation. Each transformation matrix has only 2 non-zero elements per row.

**Reference:** Zackay, B., & Ofek, E. O. (2017). "An accurate and efficient algorithm for detection of radio bursts with an unknown dispersion measure."

### Fourier Domain Dedispersion (FDD)

Performs dedispersion entirely in the Fourier domain by applying phasor corrections. The frequency-time data is FFT-transformed, multiplied by pre-computed phasor matrices, then inverse-transformed.

**Complexity:** O(N_DM × N_f × N_t)

The GPU implementation recasts the core operation as batched GEMM, which maps efficiently to Tensor Cores. Despite having more FLOPs than FDMT, the regular memory access pattern achieves higher hardware utilization.

**Reference:** Bassa, C. G., et al. "Fourier-domain dedispersion." [arXiv:2110.03482](https://arxiv.org/abs/2110.03482)

#### Reversibility

The FDD pipeline (FFT → GEMM → IFFT) is **not generally reversible**. The FFT and IFFT steps are unitary and perfectly invertible, but the core GEMM at each frequency bin k is a matrix multiply `C = A @ B` where A is `[Batch × Nf]` and B is the `[Nf × Ndm]` phasor matrix. Reversibility depends on the relationship between `Ndm` and `Nf`:

- **Ndm < Nf** (the default: 256 DMs, 512 channels): **Not reversible.** The GEMM projects Nf frequency channels down to Ndm DM trials, discarding `Nf - Ndm` dimensions of information. No inverse exists.
- **Ndm >= Nf**: Mathematically reversible via pseudo-inverse `A = C @ pinv(B)`, since the phasor matrix (Vandermonde-like with distinct complex exponential entries) has full column rank.

Even when Ndm >= Nf, practical recovery is limited by the phasor matrix condition number (which grows with DM range and frequency span) and by quantization noise in reduced-precision modes (FP8/FP16).

### Astro-Accelerate Tiled

A cache-optimized brute-force approach using tiling over time and DM dimensions. Inspired by the AstroAccelerate GPU pipeline, it improves data locality for better cache utilization on modern CPUs.

**Reference:** Novotný, J., et al. (2023). "AstroAccelerate—A GPU-based processing pipeline for pulsar and fast transient searches."

### Algorithm Comparison

| Algorithm | Complexity | GPU Optimization | Best For |
|-----------|------------|------------------|----------|
| Brute Force | O(N_f·N_DM·N_t) | Poor | Reference/validation |
| FDMT (SpMM) | O(N_t·N_f·log N_f) | Moderate | Fewer FLOPs needed |
| FDD (GEMM) | O(N_DM·N_f·N_t) | Excellent | GPU throughput |
| Astro-Accel | O(N_f·N_DM·N_t) | Good (tiled) | CPU cache efficiency |

---

## GPU Implementation

### CUDA Architecture

The GPU implementations target NVIDIA H100/Blackwell GPUs with compute capability 9.0a. Key optimizations include:

- **Batched cuFFT** for parallel FFT/IFFT operations
- **cuBLAS strided batched GEMM** for phasor application
- **cuRAND** for GPU-side noise generation
- **FP16 (half precision)** support via cuBLAS for 2x throughput
- **Block Floating Point (BFP)** quantization for memory efficiency

### GPU Kernels

Key CUDA kernels in the FDD pipeline:

| Kernel | Function |
|--------|----------|
| `kernel_copy_pad` | Copies real input to complex buffer with zero-padding |
| `kernel_transpose_f_k` | Batched transpose [b][f][k] → [b][k][f] |
| `kernel_inject_bursts` | GPU-side pulse injection with dispersive delay |
| `kernel_find_block_maxes` | BFP reduction - find max magnitude per block |
| `kernel_find_global_max` | BFP reduction - find global max from block maxes |

### Memory Layout

Data is organized for coalesced memory access:

- **Input intensity:** `[batch][freq][time]` - contiguous time samples
- **FFT output:** `[batch×freq][time_padded]` - for batched cuFFT
- **Phasors:** `[time][dm][freq]` - enables batched GEMM
- **Output:** `[batch][dm][time]` - dedispersed time series per DM

### Precision Modes

The `gpu_fdd_improved` implementation supports multiple precision modes:

| Mode | Description | Throughput |
|------|-------------|------------|
| FP64 (double) | Highest accuracy | Baseline |
| FP32 (float) | Good balance | ~2x FP64 |
| FP16 (half) | Tensor Core accelerated | ~4x FP64 |

FP16 mode achieves approximately 2x speedup on H100 by leveraging Tensor Core HGEMM operations.

### CUTLASS FDD Modes (Blackwell)

The `gpu_fdd_improved_cuda13x` binary adds three CUTLASS-based FDD modes targeting Blackwell (SM100/SM120) tensor cores. All three use the 4-split complex GEMM algorithm (4 real sub-GEMMs) with FP32 output and FP32 accumulation. The input is always FP16 — the library handles quantization internally.

| `--fdd-mode` | Internal Precision | Tensor Core Op | Output | Accuracy |
|---------------|-------------------|----------------|--------|----------|
| `cutlass` | FP8 E4M3 | Standard MMA | FP32 | Highest (baseline) |
| `cutlass_fp6` | FP6 E3M2 | MXFP Block-Scaled | FP32 | Equivalent to FP8 |
| `cutlass_fp4` | FP4 E2M1 | MXFP Block-Scaled | FP32 | Slightly lossy |

**Why FP32 output is mandatory:** The FDD GEMM signal is approximately 0.03% of the DC bias. For typical problem sizes (K=512), the raw GEMM output for DC frequency bins can reach ~5×10⁶, while the pulsed signal is ~1500. FP16 (max 65504, ULP ~32 at that range) cannot resolve this difference. FP32 output preserves the signal.

#### MXFP Block-Scaled Quantization

The FP6 and FP4 modes use NVIDIA's Microscaling (MXFP) format with per-32-element scale factors (UE8M0 exponent-only scaling). Each group of 32 contiguous elements shares a single scale factor, and individual elements are stored in reduced-precision sub-byte format.

**FP6 E3M2** (3-bit exponent, 2-bit mantissa): Exactly represents all integers in [-7, +7] with range ±28. This covers the full dynamic range of FP8 E4M3 quantized FDD data, so FP6 is expected to match FP8 accuracy.

**FP4 E2M1** (2-bit exponent, 1-bit mantissa): Only 4 representable magnitudes per sign: {0, 0.5, 1.0, 1.5} × 2^e. Cannot exactly represent the integers 5 and 7 (5→4, 7→8). This introduces rounding error that may reduce match rates for weak signals.

#### Precision vs Signal Strength

The FDD pipeline detects injected pulses by matched filtering against the dedispersed GEMM output. The "match rate" counts how many of 128 injected pulses are recovered (correct DM ±5 pc/cm³ and time ±0.05s). Results depend on signal amplitude relative to noise (Nf=512, Nt=1024, Ndm=256, seed=42):

**Low noise (noise_stddev=5, amplitude=10) — all precisions equivalent:**

| Mode | Matches |
|------|---------|
| `cublas` (FP32) | 125/128 |
| `cublas_lt_fp16` | 125/128 |
| `cublas_lt_fp8` | 125/128 |
| `cutlass` (FP8) | 125/128 |
| `cutlass_fp6` | 125/128 |
| `cutlass_fp4` | 127/128 |

**Higher noise (noise_stddev=10, amplitude=5) — precision differences emerge:**

| Mode | Matches |
|------|---------|
| `cublas` (FP32) | 95/128 |
| `cublas_lt_fp16` | 95/128 |
| `cublas_lt_fp8` | 92/128 |
| `cutlass` (FP8) | 93/128 |
| `cutlass_fp6` | 86/128 |
| `cutlass_fp4` | 79/128 |

At challenging SNR (amplitude/noise_stddev = 0.5), precision loss in FP4 and FP6 reduces match rates by ~15-40% vs FP32. At moderate SNR (amplitude/noise_stddev ≥ 2), all precisions converge to the same match rate — the remaining misses are DM grid resolution limits, not precision artifacts.

**Recommendation:** FP6 is the best tradeoff — it matches FP8 representational fidelity and offers ~25% bandwidth reduction. Use FP4 only when throughput is the priority and some accuracy loss is acceptable. For weak-signal science cases, use `cutlass` (FP8).

---

## API Reference

### DedispApplication Class

The main templated class encapsulating all functionality:

```cpp
template<typename Real>
class DedispApplication {
public:
  DedispApplication(const SimParams& params);
  void run();
  bool validate_parameters() const;
};
```

### Core Algorithm Methods

**fft_fdmt()** - Fast DM Transform:
```cpp
RMatrix fft_fdmt(const RMatrix& intensity_matrix,
                 Real f_min, Real f_max, Real max_dm,
                 int num_dm_trials, Real time_resolution);
```

**brute_force_dedispersion()** - Traditional approach:
```cpp
RMatrix brute_force_dedispersion(const RMatrix& intensity_matrix,
                                  Real f_min, Real f_max, Real max_dm,
                                  int num_dm_trials, Real time_resolution);
```

**fdd()** - Fourier Domain Dedispersion:
```cpp
RMatrix fdd(const RMatrix& intensity_matrix,
            Real f_min, Real f_max, Real max_dm,
            int num_dm_trials, Real time_resolution);
```

### Batched Processing Methods

**setup_fdd_precomputation()** - One-time phasor table creation:
```cpp
void setup_fdd_precomputation();
```
Pre-computes the shared phasor matrices for all DM trials and time samples. Called once before processing batches.

**process_fdd_sub_batch()** - Process a batch of spectra:
```cpp
RBatchMatrix process_fdd_sub_batch(const RBatchMatrix& sub_batch);
```
Processes a batch of frequency-time matrices through the FDD pipeline, returning dedispersed DM-time matrices.

### Signal Simulation

**inject_pulse()** - Add dispersed pulse to data:
```cpp
void inject_pulse(RMatrix& intensity_matrix,
                  const PulsarParams& pulsar,
                  Real pulse_start_time, Real time_resolution);
```

**generate_pulse_profile()** - Create pulse shape:
```cpp
std::vector<Real> generate_pulse_profile(
    const PulsarParams& pulsar,
    Real freq_GHz, Real time_resolution);
```

### Candidate Detection

**find_pulse_candidates()** - Search for pulses:
```cpp
std::vector<PulseCandidate> find_pulse_candidates(
    const RMatrix& dedispersed_data,
    const SimParams& params,
    Real time_resolution, int num_candidates);
```
Performs matched filtering with configurable boxcar/Gaussian kernels and returns candidates sorted by SNR.

---

## Command Line Interface

### Basic Usage

```bash
./dedisp --algorithm <name> --precision <type> [options...]
```

### Algorithm Selection

| `--algorithm` | Description |
|---------------|-------------|
| `fdmt` | Fast DM Transform (single instance) |
| `brute` | Brute force dedispersion |
| `fdd` | Fourier Domain Dedispersion |
| `astro-accel` | Cache-optimized tiled approach |
| `fdd-gemm-batched` | Batched FDD with GEMM (GPU) |
| `fdmt-spmm-batched` | Batched FDMT with SpMM |
| `test` | Run all algorithms and compare |
| `compare-fdd` | Serial vs. batched FDD timing |
| `compare-fdmt` | Serial vs. batched FDMT timing |

### Common Options

| Option | Default | Description |
|--------|---------|-------------|
| `--precision <single\|double>` | `double` | Floating-point precision |
| `--num-freq-channels <int>` | `512` | Frequency channels |
| `--num-time-samples <int>` | `2048` | Time samples |
| `--num-dm-trials <int>` | `256` | DM trials |
| `--max-dm-search <float>` | `100.0` | Max DM (pc/cm³) |
| `--batch-size <int>` | `128` | Batch size (must be multiple of 32 for FP16/FP8/CUTLASS) |
| `--num-pipelines <int>` | `1` | Parallel pipelines |
| `--dry-run <yes\|no>` | `no` | Estimate only |
| `--noise-mean <float>` | `0.0` | Noise mean (Gaussian noise added to filterbank) |
| `--noise-stddev <float>` | `5.0` | Noise standard deviation |
| `--min-amplitude <float>` | `10.0` | Min injected pulse amplitude |
| `--max-amplitude <float>` | `10.0` | Max injected pulse amplitude |
| `--print-plots <yes\|no>` | `no` | ASCII waterfall plots |

### GPU-Specific Options

| Option | Default | Description |
|--------|---------|-------------|
| `--fdd-mode <mode>` | `cublas` | GEMM backend (see below) |
| `--cublas-precision <fp16\|fp32>` | `fp32` | cuBLAS precision (cublas modes only) |
| `--seed <int>` | random | RNG seed for reproducible noise/pulses |

**Available `--fdd-mode` values:**

| Mode | Backend | Hardware | Description |
|------|---------|----------|-------------|
| `kernel` | Custom CUDA kernel | Any | Hand-written GEMM kernel |
| `cublas` | cuBLAS | Any | Standard cuBLAS GEMM |
| `cublas_lt_fp16` | cuBLASLt | Hopper+ | FP16 Tensor Core GEMM |
| `cublas_lt_fp8` | cuBLASLt | Hopper+ | FP8 Tensor Core GEMM |
| `cutlass` | CUTLASS FP8 | Blackwell | FP8 E4M3, FP32 output |
| `cutlass_fp6` | CUTLASS FP6 | Blackwell | FP6 E3M2 MXFP, FP32 output |
| `cutlass_fp4` | CUTLASS FP4 | Blackwell | FP4 E2M1 MXFP, FP32 output |

The CUTLASS modes require Blackwell GPUs (SM100/SM120) and linking against `libcutlass_gemm_api.a`. See [CUTLASS Support](#cutlass-support-optional) for build instructions.

### Example Commands

**Fast FDMT test:**
```bash
./dedisp --algorithm fdmt --precision single \
         --num-freq-channels 256 --num-time-samples 1024 --print-plots no
```

**High-throughput GPU FDD:**
```bash
./gpu_fdd_improved --algorithm fdd-gemm-batched --precision single \
                   --batch-size 16 --num-pipelines 4 --cublas-precision fp16
```

**Performance estimation (dry run):**
```bash
./dedisp --dry-run yes --algorithm test
```

**Batched FDMT with SpMM:**
```bash
./dedisp --algorithm fdmt-spmm-batched --batch-size 16 --num-pipelines 2 \
         --max-dm-search 50.0 --print-plots no
```

**Algorithm comparison:**
```bash
./dedisp --algorithm compare-fdd --batch-size 10 --num-pipelines 5
```

**cuBLAS FP32 (reference):**
```bash
./gpu_fdd_cuda13x --precision single --batch-size 128 --fdd-mode cublas \
    --noise-stddev 5 --noise-mean 0 --min-amplitude 10 --max-amplitude 10 --seed 42
# Expected: ~125/128 OK
```

**cuBLASLt FP16 (Tensor Core):**
```bash
./gpu_fdd_cuda13x --precision single --batch-size 128 --fdd-mode cublas_lt_fp16 \
    --noise-stddev 5 --noise-mean 0 --min-amplitude 10 --max-amplitude 10 --seed 42
# Expected: ~125/128 OK
```

**cuBLASLt FP8 (Tensor Core):**
```bash
./gpu_fdd_cuda13x --precision single --batch-size 128 --fdd-mode cublas_lt_fp8 \
    --noise-stddev 5 --noise-mean 0 --min-amplitude 10 --max-amplitude 10 --seed 42
# Expected: ~125/128 OK
```

**CUTLASS FP8 (Blackwell baseline):**
```bash
./gpu_fdd_cuda13x --precision single --batch-size 128 --fdd-mode cutlass \
    --noise-stddev 5 --noise-mean 0 --min-amplitude 10 --max-amplitude 10 --seed 42
# Expected: ~125/128 OK
```

**CUTLASS FP6 (MXFP block-scaled, equivalent accuracy):**
```bash
./gpu_fdd_cuda13x --precision single --batch-size 128 --fdd-mode cutlass_fp6 \
    --noise-stddev 5 --noise-mean 0 --min-amplitude 10 --max-amplitude 10 --seed 42
# Expected: ~125/128 OK
```

**CUTLASS FP4 (lossy, highest throughput):**
```bash
./gpu_fdd_cuda13x --precision single --batch-size 128 --fdd-mode cutlass_fp4 \
    --noise-stddev 5 --noise-mean 0 --min-amplitude 10 --max-amplitude 10 --seed 42
# Expected: ~125-127/128 OK
```

---

## Performance

### Computational Complexity

The library reports FLOP counts and arithmetic intensity for each algorithm:

- **FFT/IFFT FLOPs:** 5 × N × log₂(N) per transform
- **GEMM FLOPs:** 8 × N_DM × N_f × N_t (complex multiply-add)
- **SpMM FLOPs:** 2 × nnz × N (sparse multiply-add)

### Memory Bandwidth

Key memory requirements per batch item:

- **Input:** N_f × N_t × sizeof(Real) bytes
- **FFT buffer:** N_f × N_t_padded × 2 × sizeof(Real) bytes (complex)
- **Phasor table:** N_t × N_DM × N_f × 2 × sizeof(Real) bytes
- **Output:** N_DM × N_t × sizeof(Real) bytes

### Scaling Predictions

Both FDMT and FDD scale as O(N_DM × N_f × N_t) for the core compute phase. However, FDD achieves higher hardware utilization due to:

- Linear, coalesced memory access patterns
- Direct mapping to Tensor Core GEMM operations
- Pre-computed phasor tables (one-time cost)

For large problem sizes on Hopper/Blackwell GPUs, FDD typically achieves **2-5x higher throughput** than SpMM-based FDMT despite having more theoretical FLOPs.

### Arithmetic Intensity

Arithmetic intensity (FLOPs/Byte) determines whether an algorithm is compute-bound or memory-bound:

| Algorithm | Typical AI | Bound By |
|-----------|------------|----------|
| Brute Force | < 1 | Memory |
| FDMT (SpMM) | 1-5 | Memory/Compute |
| FDD (GEMM) | 5-50+ | Compute |

---

## Building

### Prerequisites

- C++17 compatible compiler (GCC 9+, Clang 10+)
- CMake 3.10 or higher
- OpenMP support
- CUDA Toolkit 12.0+ (for GPU builds)
- cuFFT, cuBLAS, cuRAND libraries

### Build Steps

```bash
mkdir build && cd build
cmake ..
make -j$(nproc)
```

### Build Targets

| Target | Description |
|--------|-------------|
| `dedisp` | CPU-only implementation (all algorithms) |
| `fdmt_perfect` | Optimized CPU FDMT reference |
| `gpu_dedisp` | Basic GPU dedispersion |
| `gpu_fdd` | GPU FDD with cuBLAS |
| `gpu_fdd_improved` | Optimized GPU FDD |
| `gpu_fdd_improved_19T` | High-throughput GPU FDD |
| `gpu_fdd_improved_cuda13x` | CUDA 13.x optimized build |

### CUTLASS Support (Optional)

CUTLASS support enables FP8, FP6, and FP4 tensor core GEMM backends for the FDD pipeline on Blackwell GPUs. The CUTLASS library must be built separately, then linked into dedisp.

> **IMPORTANT: Dedispersion uses standalone GEMM (not HERK) and requires a GEMM-optimized
> build of the CUTLASS library on GB10 (SM120).** The CUTLASS library must be compiled
> differently for GEMM workloads vs HERK workloads because SM120 has only 99 KB shared
> memory per block. The GEMM-optimized build uses a 128×128 FP8 tile with 2-stage
> auto-carveout (78 KB), which is 5-28% faster for large square/rectangular GEMMs (N≥2048)
> compared to the default HERK-optimized 128×64 tile. See `cutlass_interface/README.md`
> § "Production Deployment Summary" for full benchmark data and rationale.

**Step 1: Build the CUTLASS GEMM library (GEMM-optimized for dedispersion)**

```bash
cd /path/to/cutlass_interface
mkdir build && cd build
cmake .. \
    -DCUTLASS_DIR=/path/to/cutlass \
    -DCOMPLEX_FP8_ARCH=120a \
    -DCOMPLEX_SM100_ENABLE_FP6=ON \
    -DCOMPLEX_SM100_ENABLE_FP4=ON \
    -DCOMPLEX_FP8_SM100_STAGES=3 \
    -DCOMPLEX_SM100_FP8_TILE_N=128
make cutlass_gemm_api -j
```

This produces `libcutlass_gemm_api.a` — a self-contained static library with no CUTLASS header dependencies for consumers. The `-DCOMPLEX_SM100_FP8_TILE_N=128` flag selects the wider 128×128 FP8 tile optimized for standalone GEMM workloads.

> **Note:** If building the CUTLASS library for XEngine/FTD (HERK workloads), omit
> `-DCOMPLEX_SM100_FP8_TILE_N=128` — the default 128×64 tile is correct for HERK.
> HERK performance is identical with either tile because the direct HERK kernel (24 KB
> SMEM) bypasses CUTLASS tiles entirely.

**Step 2: Build dedisp with CUTLASS**

```bash
cd /path/to/dedisp/build
cmake .. \
    -DGPU_ARCH=121 \
    -DUSE_CUTLASS_GEMM=ON \
    -DCUTLASS_GEMM_LIB=/path/to/cutlass_interface/build/libcutlass_gemm_api.a \
    -DCUTLASS_GEMM_INCLUDE=/path/to/cutlass_interface/build
make gpu_fdd_improved_cuda13x -j
```

**Notes:**
- `GPU_ARCH=121` targets SM120 (consumer Blackwell, e.g. GB10 Spark). Use `100a` for datacenter Blackwell (B200/GB200).
- FP6 and FP4 modes require the CUTLASS library to be built with `COMPLEX_SM100_ENABLE_FP6=ON` and `COMPLEX_SM100_ENABLE_FP4=ON` respectively. Without these flags, only `cutlass` (FP8) mode is available.
- Build times are long (~15 min per kernel instantiation) due to deep CUTLASS template expansion. The FP6/FP4 kernels add additional compilation time.
- The GEMM-optimized CUTLASS library (`COMPLEX_SM100_FP8_TILE_N=128`) should **not** be shared with XEngine/FTD builds. Each consumer should link its own `libcutlass_gemm_api.a` built with the appropriate tile configuration.

#### Kernel Autotuning

The CUTLASS backend includes an automatic kernel-level tuning system that optimizes the launch parameters (blockDim, gridDim) of overhead kernels — cast, pack, deinterleave, and MXFP preprocessing. These kernels run before the GEMM itself and can account for 10-30% of total wall time at small M.

Enable tuning via `DedispConfig`:

```cpp
dedisp_api::DedispConfig config;
config.compute_mode = dedisp_api::ComputeMode::CUTLASS_FP8;
config.kernel_tune_verbosity = 2;  // tune + summary (one-line per kernel)
config.kernel_tune_cache_path = "/path/to/my_kernel_cache.txt";
// ... other config fields ...
```

| `kernel_tune_verbosity` | Behavior |
|-------------------------|----------|
| 0 (default) | Silent — no tuning, no output |
| 1 | Show cached/default launch params per kernel |
| 2 | Tune kernels + one-line summary per kernel |
| 3 | Tune kernels + full per-config timing detail |

Levels >= 2 trigger a tuning sweep (~30-60 seconds) on the first pipeline call. Results are cached to `kernel_tune_cache_path` and reused across sessions and restarts. The cache is GPU-name- and build-config-validated — if the hardware or build changes, tuning re-runs automatically.

**Recommendation:** Run with `kernel_tune_verbosity=2` once after building to populate the cache. Subsequent runs with `kernel_tune_verbosity=0` (or 1 for diagnostics) use the cached results with zero overhead.

#### Caveats

- **Blackwell only:** The CUTLASS modes use SM100/SM120-specific MMA instructions (FP8 standard MMA and MXFP block-scaled MMA). They will not work on Hopper or earlier GPUs.
- **FP6/FP4 microblock shadowing:** MXFP block-scaled quantization groups 32 elements under a shared scale factor. When strong DC components dominate a block, weak signal components lose precision. This reduces match rates for low-amplitude injected pulses. See [Precision vs Signal Strength](#precision-vs-signal-strength) for measured impact.
- **FP4 is lossy:** FP4 E2M1 cannot exactly represent the integers 5 and 7 (rounded to 4 and 8). For the FDD pipeline where input values are small integers from FFT output, this introduces systematic rounding error.
- **Large problem sizes:** The MXFP preprocessing kernels require grid dimensions that scale with `N × batch_count`. For very large configurations (e.g., Nf=2048, Ndm=2048, 4 pipelines → batch_count=513), this can exceed 10⁶ thread blocks. A 3D grid workaround handles this automatically, but memory requirements also scale accordingly.
- **Pre-scaling is preserved:** The FP6/FP4 modes reuse the same FP16→FP8-range pre-scaling as the FP8 mode (scale_A = 448/max_A). The MXFP preprocessing further quantizes with per-block scale factors, and the epilogue alpha undoes the pre-scaling. This is harmless — MXFP SFs adapt to the data range automatically.

---

## References

1. Taylor, J. H. (1974). "A method for measuring dispersion in pulsar signals." *Astronomy and Astrophysics Supplement Series*, 15, 367-376.

2. Zackay, B., & Ofek, E. O. (2017). "An accurate and efficient algorithm for detection of radio bursts with an unknown dispersion measure, for single dish telescopes and interferometers." *The Astrophysical Journal*, 835(1), 11. [DOI](https://doi.org/10.3847/1538-4357/835/1/11)

3. Bassa, C. G., Romein, J. W., Veenboer, B., van der Vlugt, S., & Wijnholds, S. J. "Fourier-domain dedispersion." [arXiv:2110.03482](https://arxiv.org/abs/2110.03482)

4. Novotný, J., et al. (2023). "AstroAccelerate—A GPU-based processing pipeline for pulsar and fast transient searches: Development and performance." [arXiv:2311.05341](https://arxiv.org/abs/2311.05341)

---

## License

See LICENSE file for details.