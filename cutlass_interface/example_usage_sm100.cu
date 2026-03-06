/*
 * example_usage_sm100.cu
 *
 * Complex FP8 GEMM benchmark on NVIDIA Blackwell (SM100/SM120).
 *
 * Three-phase execution:
 *   TEST PHASE       — validates HERK output against Gram reference,
 *                      checks Hermitian properties (real diagonal,
 *                      conjugate symmetry, lower-triangle agreement)
 *   MULTI-PRECISION  — validates FP6/FP4 paths vs FP8 reference
 *   PRODUCTION PHASE — clean benchmarks, no validation overhead
 *
 * Usage:
 *   ./example_complex_sm100 [M] [N] [K] [batch] [precision] [key=value ...]
 *   ./example_complex_sm100                           # defaults to 4096³, FP8
 *   ./example_complex_sm100 8192 8192 8192
 *   ./example_complex_sm100 4096 4096 4096 1 fp6e3m2  # FP6 E3M2 compute
 *   ./example_complex_sm100 4096 4096 4096 1 fp4      # FP4 E2M1 compute
 *   ./example_complex_sm100 4096 mode=bench           # benchmarks only
 *   ./example_complex_sm100 4096 mode=test            # tests only
 *
 * Precision options: fp8 (default), fp6e3m2, fp6e2m3, fp4
 * Mode options: test, bench, both (default)
 * FP6/FP4 require compilation with -DCOMPLEX_SM100_ENABLE_FP6=ON / -DCOMPLEX_SM100_ENABLE_FP4=ON
 *
 * Compile (GB200 target):
 *   cmake .. -DCUTLASS_DIR=/path/to/cutlass -DCOMPLEX_FP8_ARCH=100a
 *   make -j$(nproc)
 *
 * Compile (GB10 Spark development):
 *   cmake .. -DCUTLASS_DIR=/path/to/cutlass -DCOMPLEX_FP8_ARCH=120a
 *   make -j$(nproc)
 */

#include "gemm_complex_sm100.hpp"
#include "strategy_cache.hpp"
#include "system_info.hpp"
#include <iostream>
#include <iomanip>
#include <vector>
#include <random>
#include <chrono>
#include <cstdlib>
#include <cmath>
#include <algorithm>
#include <cstring>

using namespace gemm_complex_sm100;


// ========================================================================================
// Run Mode
// ========================================================================================

enum class BenchMode { Both, Test, Bench };

BenchMode parse_mode(const char* str) {
    if (strcmp(str, "test") == 0)  return BenchMode::Test;
    if (strcmp(str, "bench") == 0) return BenchMode::Bench;
    if (strcmp(str, "both") == 0)  return BenchMode::Both;
    std::cerr << "Error: unknown mode '" << str << "'\n"
              << "  Valid options: test, bench, both\n";
    std::exit(1);
}


// ========================================================================================
// Triangle Config Parsing
// ========================================================================================

bool has_help_flag(int argc, char** argv) {
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0)
            return true;
    }
    return false;
}

void print_help(const char* prog) {
    printf("Usage: %s [M] [N] [K] [batch] [precision] [key=value ...]\n", prog);
    printf("\n");
    printf("Complex FP8 GEMM benchmark on NVIDIA Blackwell (SM100/SM120).\n");
    printf("Runs test phase (correctness validation) then production phase (benchmarks).\n");
    printf("\n");
    printf("Positional arguments:\n");
    printf("  M N K        Matrix dimensions (default: 4096 4096 4096)\n");
    printf("  batch        Batch count (default: 1)\n");
    printf("  precision    Compute precision: fp8, fp6e3m2, fp6e2m3, fp4 (default: fp8)\n");
    printf("\n");
    printf("Shorthand:\n");
    printf("  %s 4096              # M=N=K=4096\n", prog);
    printf("  %s 4096 fp6e3m2      # M=N=K=4096, FP6 E3M2\n", prog);
    printf("  %s 4096 4096 4096 4  # batched\n", prog);
    printf("\n");
    printf("General keys:\n");
    printf("  mode=<test|bench|both>  Run mode (default: both)\n");
    printf("\n");
    printf("Triangle config keys (key=value, for HERK TriangleAware):\n");
    printf("  slabs=<2-32>    Number of horizontal slabs (0=auto, default: 0)\n");
    printf("  min_slab=<int>  Minimum slab height in rows (0=auto, default: 0)\n");
    printf("  graduated=<0|1> Use sqrt-spaced slab boundaries (default: 0)\n");
    printf("  verbose=<0|1>   Print slab decomposition to stderr (default: 0)\n");
    printf("  graph=<0|1>     SM120: capture slab launches as CUDA graph for replay (default: 0)\n");
    printf("  herk_graph=<0|1> Capture baseline HERK as CUDA graph for replay (default: 0)\n");
    printf("  direct=<auto|0|1> Direct HERK mode: auto=K-adaptive, 1=force, 0=off (default: auto)\n");
    printf("  persistent=<auto|0|1> Persistent direct HERK: auto=work-adaptive, 1=force, 0=off (default: auto)\n");
    printf("  tile=<0|1>       Enable batch tiling for L2 scratch reuse (default: 1)\n");
    printf("  tune=<true|false|0|1|2|3>  true=full strategy autotune (default), false=off, 0-3=kernel-level tune verbosity\n");
    printf("  strategy_verbosity=<0|1|2|3>  Strategy autotune output verbosity (default: 1, or 2 when tune=true)\n");
    printf("  gemm_mode=<auto|direct|4m>  GEMM dispatch: auto=autotuner, direct=PTX kernel, 4m=CUTLASS sub-GEMMs (default: auto)\n");
    printf("  crosscheck=<cpu|gpu|off>  Batched HERK independence cross-check (default: off)\n");
    printf("                   cpu: FP64 reference per batch element\n");
    printf("                   gpu: per-batch direct launch vs batched launch (bit-exact)\n");
    printf("\n");
    printf("Examples:\n");
    printf("  %s 4096                           # 4096^3, FP8, default triangle\n", prog);
    printf("  %s 4096 4096 4096 128 graph=1     # batched with CUDA graph\n", prog);
    printf("  %s 4096 slabs=8 verbose=1         # explicit 8 slabs, verbose\n", prog);
    printf("  %s 4096 4096 4096 1 fp4           # FP4 E2M1 compute\n", prog);
    printf("  %s 4096 mode=bench                # benchmarks only, skip tests\n", prog);
    printf("  %s 4096 mode=test                 # tests only, skip benchmarks\n", prog);
}

TriangleConfig parse_triangle_config(int argc, char** argv) {
    TriangleConfig tc;
    for (int i = 1; i < argc; ++i) {
        std::string arg(argv[i]);
        auto eq = arg.find('=');
        if (eq == std::string::npos) continue;
        std::string key = arg.substr(0, eq);
        std::string val = arg.substr(eq + 1);
        if (key == "slabs")      tc.target_slabs = std::atoi(val.c_str());
        else if (key == "min_slab")   tc.min_slab_height = std::atoi(val.c_str());
        else if (key == "graduated")  tc.graduated = (val == "1" || val == "true");
        else if (key == "verbose")    tc.verbose = (val == "1" || val == "true");
        else if (key == "graph")      tc.use_cuda_graph = (val == "1" || val == "true");
    }
    return tc;
}

bool parse_herk_graph(int argc, char** argv) {
    for (int i = 1; i < argc; ++i) {
        std::string arg(argv[i]);
        if (arg.substr(0, 11) == "herk_graph=")
            return arg.substr(11) == "1" || arg.substr(11) == "true";
    }
    return false;
}

HerkMode parse_herk_mode(int argc, char** argv) {
    for (int i = 1; i < argc; ++i) {
        std::string arg(argv[i]);
        if (arg.substr(0, 7) == "direct=") {
            std::string val = arg.substr(7);
            if (val == "1" || val == "true")  return HerkMode::ForceDirect;
            if (val == "0" || val == "false") return HerkMode::ForceBaseline;
            return HerkMode::Auto;  // "auto" or any other value
        }
        // Backward compat: accept fused= as alias for direct=
        if (arg.substr(0, 6) == "fused=") {
            std::string val = arg.substr(6);
            if (val == "1" || val == "true")  return HerkMode::ForceDirect;
            if (val == "0" || val == "false") return HerkMode::ForceBaseline;
            return HerkMode::Auto;
        }
    }
    return HerkMode::Auto;
}

PersistentMode parse_persistent_mode(int argc, char** argv) {
    for (int i = 1; i < argc; ++i) {
        std::string arg(argv[i]);
        if (arg.substr(0, 11) == "persistent=") {
            std::string val = arg.substr(11);
            if (val == "1" || val == "true")  return PersistentMode::ForceOn;
            if (val == "0" || val == "false") return PersistentMode::ForceOff;
            return PersistentMode::Auto;  // "auto" or any other value
        }
    }
    return PersistentMode::Auto;
}

bool parse_batch_tiling(int argc, char** argv) {
    for (int i = 1; i < argc; ++i) {
        std::string arg(argv[i]);
        if (arg.substr(0, 5) == "tile=")
            return arg.substr(5) == "1" || arg.substr(5) == "true";
    }
    return true;  // default ON
}

int parse_tune(int argc, char** argv) {
    // Returns: -1 = full strategy autotune (default), 0-3 = kernel-level tune verbosity
    // tune=true: full autotune, tune=false: off, tune=0-3: kernel verbosity
    for (int i = 1; i < argc; ++i) {
        std::string arg(argv[i]);
        if (arg.substr(0, 5) == "tune=") {
            std::string val = arg.substr(5);
            if (val == "true")  return -1;
            if (val == "false") return 0;
            return std::atoi(val.c_str());
        }
    }
    return -1;  // Default: full strategy autotune (opt-out via tune=false)
}

int parse_strategy_verbosity(int argc, char** argv) {
    for (int i = 1; i < argc; ++i) {
        std::string arg(argv[i]);
        if (arg.find("strategy_verbosity=") == 0) {
            return std::atoi(arg.c_str() + 19);
        }
    }
    return -1;  // not set (use default)
}

GemmMode parse_gemm_mode(int argc, char** argv) {
    for (int i = 1; i < argc; ++i) {
        std::string arg(argv[i]);
        if (arg.substr(0, 10) == "gemm_mode=") {
            std::string val = arg.substr(10);
            if (val == "direct")  return GemmMode::ForceDirect;
            if (val == "4m")      return GemmMode::Force4M;
            return GemmMode::Auto;  // "auto" or any other value
        }
    }
    return GemmMode::Auto;
}

enum class CrossCheck { Off, CPU, GPU };

CrossCheck parse_crosscheck(int argc, char** argv) {
    for (int i = 1; i < argc; ++i) {
        std::string arg(argv[i]);
        if (arg.substr(0, 11) == "crosscheck=") {
            std::string val = arg.substr(11);
            if (val == "cpu")                  return CrossCheck::CPU;
            if (val == "gpu")                  return CrossCheck::GPU;
            if (val == "off" || val == "0")     return CrossCheck::Off;
            std::cerr << "Error: unknown crosscheck mode '" << val << "'\n"
                      << "  Valid options: cpu, gpu, off\n";
            std::exit(1);
        }
    }
    return CrossCheck::Off;
}

BenchMode parse_bench_mode(int argc, char** argv) {
    for (int i = 1; i < argc; ++i) {
        std::string arg(argv[i]);
        if (arg.substr(0, 5) == "mode=") {
            return parse_mode(arg.substr(5).c_str());
        }
    }
    return BenchMode::Both;
}


// ========================================================================================
// Helpers
// ========================================================================================

void fill_random_fp16(std::vector<__half>& buf, float scale = 1.0f) {
    static std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-scale, scale);
    for (auto& v : buf) v = __float2half(dist(rng));
}

// ========================================================================================
// CPU FP64 HERK reference — single batch element
// ========================================================================================
//
// Computes C = alpha * A * A^H for one batch element using FP64 accumulation.
// Input:  h_A is [N, 2*K] interleaved complex FP16 (row-major, each row = [re0,im0,re1,im1,...])
// Output: h_C is [N*(N+1)] interleaved packed lower triangle FP16 ([Re,Im] per element)
//
// This is an independent reference that shares zero code with the GPU kernel.

void cpu_herk_single(
    const __half* h_A,      // [N, 2*K] interleaved complex FP16
    __half* h_C,            // [N*(N+1)] interleaved packed lower triangle output
    int N, int K,
    float alpha, float beta,
    const __half* h_C_old = nullptr)
{
    const int64_t row_stride = 2 * K;  // FP16 elements per row of A

    // OMP parallelizes the row loop; schedule(dynamic) balances the triangular workload
    // (row 0 has 1 col, row N-1 has N cols)
    #pragma omp parallel for schedule(dynamic)
    for (int row = 0; row < N; row++) {
        for (int col = 0; col <= row; col++) {
            double sum_re = 0.0, sum_im = 0.0;

            for (int k = 0; k < K; k++) {
                // A[row, k] complex
                double a_re = static_cast<double>(__half2float(h_A[row * row_stride + 2*k]));
                double a_im = static_cast<double>(__half2float(h_A[row * row_stride + 2*k + 1]));
                // A[col, k] complex, conjugated
                double b_re = static_cast<double>(__half2float(h_A[col * row_stride + 2*k]));
                double b_im = static_cast<double>(__half2float(h_A[col * row_stride + 2*k + 1]));

                // a * conj(b) = (a_re*b_re + a_im*b_im) + i*(a_im*b_re - a_re*b_im)
                sum_re += a_re * b_re + a_im * b_im;
                sum_im += a_im * b_re - a_re * b_im;
            }

            if (row == col) sum_im = 0.0;  // Hermitian diagonal is real

            int64_t tri_idx = static_cast<int64_t>(row) * (row + 1) / 2 + col;
            double val_re = alpha * sum_re;
            double val_im = alpha * sum_im;

            if (beta != 0.0f && h_C_old) {
                val_re += beta * static_cast<double>(__half2float(h_C_old[tri_idx * 2]));
                val_im += beta * static_cast<double>(__half2float(h_C_old[tri_idx * 2 + 1]));
            }

            h_C[tri_idx * 2]     = __float2half(static_cast<float>(val_re));
            h_C[tri_idx * 2 + 1] = __float2half(static_cast<float>(val_im));
        }
    }
}

// Adaptive warmup/iteration counts based on problem size
void adaptive_bench_counts(int N, int& warmup, int& iters) {
    if (N <= 512)       { warmup = 5; iters = 20; }
    else if (N <= 2048) { warmup = 3; iters = 10; }
    else                { warmup = 2; iters = 5;  }
}

template <typename Fn>
double benchmark_ms(Fn&& fn, int warmup, int iters) {
    for (int i = 0; i < warmup; ++i) fn();
    cudaDeviceSynchronize();
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iters; ++i) fn();
    cudaDeviceSynchronize();
    auto t1 = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::milli>(t1 - t0).count() / iters;
}

// benchmark_ms variant that tracks kernel launch status.
// fn must return cutlass::Status. If the first warmup call fails,
// returns {-1.0, status} immediately without running the full benchmark.
template <typename Fn>
std::pair<double, cutlass::Status> benchmark_ms_checked(Fn&& fn, int warmup, int iters) {
    // Pre-flight: single call to check if the kernel can launch
    auto preflight_status = fn();
    cudaDeviceSynchronize();
    if (preflight_status != cutlass::Status::kSuccess) {
        return {-1.0, preflight_status};
    }
    // Remaining warmup
    for (int i = 1; i < warmup; ++i) fn();
    cudaDeviceSynchronize();
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iters; ++i) fn();
    cudaDeviceSynchronize();
    auto t1 = std::chrono::high_resolution_clock::now();
    return {std::chrono::duration<double, std::milli>(t1 - t0).count() / iters,
            cutlass::Status::kSuccess};
}

void print_bench(const char* label, double ms, double tflops,
                 double baseline_ms = 0.0, int slabs = 0,
                 double ext_bytes = 0.0) {
    // Compute bandwidth metrics when external bytes are provided
    double gbs = 0, ai = 0;
    if (ext_bytes > 0) {
        gbs = ext_bytes / (ms * 1e-3) / 1e9;
        double flops = tflops * 1e12 * ms * 1e-3;  // recover total FLOPs
        ai = flops / ext_bytes;
    }

    if (baseline_ms > 0.0) {
        double speedup = baseline_ms / ms;
        if (ext_bytes > 0) {
            if (slabs > 0) {
                printf("  %-18s %8.2f ms  %6.1f TFLOPS  %7.1f GB/s  IO=%-.0f  (%.2fx vs BL, %d slabs)\n",
                       label, ms, tflops, gbs, ai, speedup, slabs);
            } else {
                printf("  %-18s %8.2f ms  %6.1f TFLOPS  %7.1f GB/s  IO=%-.0f  (%.2fx vs BL)\n",
                       label, ms, tflops, gbs, ai, speedup);
            }
        } else {
            if (slabs > 0) {
                printf("  %-18s %8.2f ms  %6.1f TFLOPS  (%.2fx vs BL, %d slabs)\n",
                       label, ms, tflops, speedup, slabs);
            } else {
                printf("  %-18s %8.2f ms  %6.1f TFLOPS  (%.2fx vs BL)\n",
                       label, ms, tflops, speedup);
            }
        }
    } else {
        if (ext_bytes > 0) {
            printf("  %-18s %8.2f ms  %6.1f TFLOPS  %7.1f GB/s  IO=%-.0f\n",
                   label, ms, tflops, gbs, ai);
        } else {
            printf("  %-18s %8.2f ms  %6.1f TFLOPS\n", label, ms, tflops);
        }
    }
}

void print_result(const char* label, int M, int N, int K, double ms,
                  int num_sub_gemms = 4, int batch = 1,
                  double baseline_ms = 0.0, int slabs = 0,
                  double ext_bytes = 0.0) {
    double flops = (double)num_sub_gemms * 2.0 * M * N * K * batch;
    double tflops = flops / (ms * 1e-3) / 1e12;
    print_bench(label, ms, tflops, baseline_ms, slabs, ext_bytes);
}

// ---- Roofline analysis collection ----

struct RooflineEntry {
    std::string label;
    double tflops;
    double tc_util;       // % of peak for this precision
    double bw_gbs;        // achieved bandwidth GB/s
    double io_intensity;  // FLOPs/byte
    double peak_tflops;   // peak for this precision (for ridge)
};

static double peak_tflops_for_precision(ComputePrecision p,
                                        const cutlass_complex::SystemInfo& si) {
    if (p == ComputePrecision::FP6_E3M2 || p == ComputePrecision::FP6_E2M3)
        return si.peak_fp6_tflops;
    if (p == ComputePrecision::FP4_E2M1)
        return si.peak_fp4_tflops;
    return si.peak_fp8_tflops;
}

static RooflineEntry make_roofline(const char* label, double ms,
                                   int num_sub_gemms, int M, int N, int K,
                                   int batch, double ext_bytes,
                                   double peak_tflops) {
    RooflineEntry e;
    e.label = label;
    double flops = (double)num_sub_gemms * 2.0 * M * N * K * batch;
    e.tflops = flops / (ms * 1e-3) / 1e12;
    e.bw_gbs = ext_bytes / (ms * 1e-3) / 1e9;
    e.io_intensity = flops / ext_bytes;
    e.tc_util = (peak_tflops > 0) ? 100.0 * e.tflops / peak_tflops : 0;
    e.peak_tflops = peak_tflops;
    return e;
}

static void print_roofline_table(const std::vector<RooflineEntry>& entries,
                                 const cutlass_complex::SystemInfo& si) {
    if (entries.empty()) return;
    double ridge_fp8 = cutlass_complex::ridge_point(si.peak_fp8_tflops, si.memory_bw_gbs);
    printf("\n--- Roofline Analysis ---\n");
    printf("  Peak FP8:      %6.1f TFLOPS\n", si.peak_fp8_tflops);
    if (si.peak_fp6_tflops > 0 && si.peak_fp6_tflops != si.peak_fp8_tflops)
        printf("  Peak FP6:      %6.1f TFLOPS\n", si.peak_fp6_tflops);
    if (si.peak_fp4_tflops > 0 && si.peak_fp4_tflops != si.peak_fp8_tflops)
        printf("  Peak FP4:      %6.1f TFLOPS\n", si.peak_fp4_tflops);
    printf("  Memory BW:     %6.1f GB/s (theoretical)\n", si.memory_bw_gbs);
    printf("  Ridge point:   %5.0f FLOPs/byte (FP8)\n", ridge_fp8);
    for (const auto& e : entries) {
        double ridge = cutlass_complex::ridge_point(e.peak_tflops, si.memory_bw_gbs);
        const char* bound = (e.io_intensity >= ridge) ? "compute-bound" : "memory-bound";
        printf("  %-22s %6.1f TFLOPS  %5.1f%% TC util  %6.1f GB/s  %5.0f FLOPs/B  (%s)\n",
               e.label.c_str(), e.tflops, e.tc_util, e.bw_gbs, e.io_intensity, bound);
    }
}

// Pack lower triangle from full N×N matrix (used by Gram benchmarks).
__global__ void pack_lower_triangle_kernel(
    const __half* __restrict__ full,
    __half* __restrict__ packed,
    int N)
{
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total = (int64_t)N * (N + 1) / 2;
    if (idx >= total) return;

    int row = (int)((-1.0 + sqrt(1.0 + 8.0 * (double)idx)) / 2.0);
    while ((int64_t)(row + 1) * (row + 2) / 2 <= idx) row++;
    while ((int64_t)row * (row + 1) / 2 > idx)         row--;
    int col = (int)(idx - (int64_t)row * (row + 1) / 2);

    packed[idx] = full[(int64_t)row * N + col];
}

void pack_lower_triangle(const __half* d_full, __half* d_packed, int N,
                          cudaStream_t stream = 0) {
    int64_t total = (int64_t)N * (N + 1) / 2;
    int threads = 256;
    int blocks  = (int)((total + threads - 1) / threads);
    pack_lower_triangle_kernel<<<blocks, threads, 0, stream>>>(d_full, d_packed, N);
}


// Tensor core ops per cycle per SM (dense, non-sparse):
//   FP8:  2048 ops/cycle/SM
//   FP6:  2048 ops/cycle/SM  (same as FP8 — same MMA tile dimensions)
//   FP4:  4096 ops/cycle/SM  (2x FP8 — 2x K dimension in MMA)
// Source: mmapeak on DGX Spark (GB10): FP8=213.7, FP6=213.7, FP4=427.3 TFLOPS
struct PeakTflops {
    double fp8, fp6, fp4;
};

PeakTflops compute_peak_tflops(const cudaDeviceProp& prop) {
    int clock_khz = 0;
    cudaDeviceGetAttribute(&clock_khz, cudaDevAttrClockRate, 0);
    double base = (double)prop.multiProcessorCount * (clock_khz / 1.0e6) * 1e-3;
    return { base * 2048.0,   // FP8
             base * 2048.0,   // FP6 (same as FP8)
             base * 4096.0 }; // FP4 (2x FP8)
}

const char* precision_name(ComputePrecision p) {
    switch (p) {
        case ComputePrecision::FP8_E4M3:  return "FP8 E4M3";
        case ComputePrecision::FP6_E3M2:  return "FP6 E3M2";
        case ComputePrecision::FP6_E2M3:  return "FP6 E2M3";
        case ComputePrecision::FP4_E2M1:  return "FP4 E2M1";
        default:                          return "Unknown";
    }
}

const char* precision_short(ComputePrecision p) {
    switch (p) {
        case ComputePrecision::FP8_E4M3:  return "FP8";
        case ComputePrecision::FP6_E3M2:  return "FP6 E3M2";
        case ComputePrecision::FP6_E2M3:  return "FP6 E2M3";
        case ComputePrecision::FP4_E2M1:  return "FP4";
        default:                          return "?";
    }
}

/// Minimum K for block-scaled precisions (MXFP TMA needs complete K-groups).
int minimum_k_for_precision(ComputePrecision p) {
    switch (p) {
#ifdef COMPLEX_SM100_ENABLE_FP6
    case ComputePrecision::FP6_E3M2: return COMPLEX_SM100_FP6_MMA_K;
    case ComputePrecision::FP6_E2M3: return COMPLEX_SM100_FP6_MMA_K;
#endif
#ifdef COMPLEX_SM100_ENABLE_FP4
    case ComputePrecision::FP4_E2M1: return COMPLEX_SM100_FP4_MMA_K;
#endif
    default: return 1;  // FP8/INT8 handle short K via mainloop predication
    }
}

ComputePrecision parse_precision(const char* str) {
    if (strcmp(str, "fp8") == 0 || strcmp(str, "FP8") == 0)
        return ComputePrecision::FP8_E4M3;
    if (strcmp(str, "fp6e3m2") == 0 || strcmp(str, "FP6E3M2") == 0 ||
        strcmp(str, "fp6_e3m2") == 0 || strcmp(str, "FP6_E3M2") == 0)
        return ComputePrecision::FP6_E3M2;
    if (strcmp(str, "fp6e2m3") == 0 || strcmp(str, "FP6E2M3") == 0 ||
        strcmp(str, "fp6_e2m3") == 0 || strcmp(str, "FP6_E2M3") == 0)
        return ComputePrecision::FP6_E2M3;
    if (strcmp(str, "fp4") == 0 || strcmp(str, "FP4") == 0 ||
        strcmp(str, "fp4e2m1") == 0 || strcmp(str, "FP4E2M1") == 0 ||
        strcmp(str, "fp4_e2m1") == 0 || strcmp(str, "FP4_E2M1") == 0)
        return ComputePrecision::FP4_E2M1;

    std::cerr << "Error: unknown precision '" << str << "'\n"
              << "  Valid options: fp8, fp6e3m2, fp6e2m3, fp4\n"
              << "Run with --help for usage information.\n";
    std::exit(1);
}

void validate_precision_compiled(ComputePrecision p) {
    if (p == ComputePrecision::FP8_E4M3) return;  // always available

    if (p == ComputePrecision::FP6_E3M2 || p == ComputePrecision::FP6_E2M3) {
#ifndef COMPLEX_SM100_ENABLE_FP6
        std::cerr << "Error: " << precision_name(p) << " requested but not compiled.\n"
                  << "  Rebuild with: cmake .. -DCOMPLEX_SM100_ENABLE_FP6=ON\n";
        std::exit(1);
#endif
    }
    if (p == ComputePrecision::FP4_E2M1) {
#ifndef COMPLEX_SM100_ENABLE_FP4
        std::cerr << "Error: " << precision_name(p) << " requested but not compiled.\n"
                  << "  Rebuild with: cmake .. -DCOMPLEX_SM100_ENABLE_FP4=ON\n";
        std::exit(1);
#endif
    }
}

void parse_args(int argc, char** argv, int& M, int& N, int& K, int& batch,
                ComputePrecision& precision) {
    M = 4096; N = 4096; K = 4096; batch = 1;
    precision = ComputePrecision::FP8_E4M3;

    // Collect positional args (skip key=value triangle config args)
    std::vector<char*> pos;
    for (int i = 1; i < argc; ++i) {
        if (strchr(argv[i], '=') == nullptr) pos.push_back(argv[i]);
    }
    int npos = static_cast<int>(pos.size());

    if (npos >= 5) {
        M = std::atoi(pos[0]); N = std::atoi(pos[1]);
        K = std::atoi(pos[2]); batch = std::atoi(pos[3]);
        precision = parse_precision(pos[4]);
    } else if (npos >= 4) {
        M = std::atoi(pos[0]); N = std::atoi(pos[1]);
        K = std::atoi(pos[2]); batch = std::atoi(pos[3]);
    } else if (npos >= 3) {
        M = std::atoi(pos[0]); N = std::atoi(pos[1]); K = std::atoi(pos[2]);
    } else if (npos == 2) {
        // Could be "M precision" — check if pos[1] is a precision string
        if (std::atoi(pos[1]) == 0 && strcmp(pos[1], "0") != 0) {
            M = N = K = std::atoi(pos[0]);
            precision = parse_precision(pos[1]);
        } else {
            M = std::atoi(pos[0]); N = std::atoi(pos[1]);
        }
    } else if (npos == 1) {
        M = N = K = std::atoi(pos[0]);
    }
    if (M <= 0 || N <= 0 || K <= 0 || batch <= 0) {
        std::cerr << "Error: M, N, K, batch must be positive.\n";
        std::cerr << "Run with --help for usage information.\n";
        std::exit(1);
    }
    if (M % 128 != 0 || N % 128 != 0 || K % 128 != 0) {
        std::cerr << "Warning: M=" << M << " N=" << N << " K=" << K
                  << " — not multiples of 128, may reduce performance.\n";
    }
    validate_precision_compiled(precision);
}


// ========================================================================================
// Triangle Slab Count (replicates adaptive algorithm for display)
// ========================================================================================

int compute_triangle_slabs(int N, int sm_count, const TriangleConfig& tri) {
    constexpr int kTileM = kFP8TileM;
    constexpr int kTileN = kFP8TileN;

    int target_slabs;
    if (tri.target_slabs > 0) {
        target_slabs = tri.target_slabs;
    } else {
        target_slabs = 2;
        for (int T = 32; T >= 2; --T) {
            int S = ((N + T - 1) / T);
            S = ((S + kTileM - 1) / kTileM) * kTileM;
            int tiles_m = (S + kTileM - 1) / kTileM;
            int tiles_n = (S + kTileN - 1) / kTileN;
            int64_t tiles_slab0 = static_cast<int64_t>(tiles_m) * tiles_n;
            if (tiles_slab0 >= sm_count) {
                target_slabs = T;
                break;
            }
        }
    }

    int min_slab_height;
    if (tri.min_slab_height > 0) {
        min_slab_height = tri.min_slab_height;
    } else {
        double ratio = static_cast<double>(kTileM) / kTileN;
        int tiles_needed = static_cast<int>(std::ceil(
            std::sqrt(static_cast<double>(sm_count) / ratio)));
        min_slab_height = kTileM * std::max(tiles_needed, 1);
    }

    int slab_height = (N + target_slabs - 1) / target_slabs;
    slab_height = ((slab_height + kTileM - 1) / kTileM) * kTileM;
    slab_height = std::max(slab_height, min_slab_height);
    int num_slabs = (N + slab_height - 1) / slab_height;

    if (tri.graduated) {
        // Graduated may merge duplicate boundaries — compute exact count
        std::vector<int> boundaries(num_slabs + 1);
        boundaries[0] = 0;
        for (int i = 1; i <= num_slabs; ++i) {
            double frac = std::sqrt(static_cast<double>(i) / num_slabs);
            int b = static_cast<int>(std::round(frac * N));
            b = ((b + kTileM - 1) / kTileM) * kTileM;
            b = std::min(b, N);
            boundaries[i] = b;
        }
        boundaries[num_slabs] = N;
        // Dedup
        int unique = 1;
        for (int i = 1; i <= num_slabs; ++i) {
            if (boundaries[i] > boundaries[unique - 1]) {
                boundaries[unique++] = boundaries[i];
            }
        }
        num_slabs = unique - 1;
    }

    return num_slabs;
}


// ========================================================================================
// Hermiticity Validation
// ========================================================================================
//
// Validates HERK output against Gram reference (4 full sub-GEMMs) and checks
// Hermitian properties of the authoritative (lower) triangle:
//
//   1. Re(C) agreement — lower triangle of HERK matches Gram reference
//   2. Im(C) agreement — lower triangle of HERK matches Gram reference
//   3. Real diagonal    — Im(C)[i,i] == 0 for all i
//   4. Conjugate symmetry — Re(C)[i,j] == Re(C)[j,i] and Im(C)[i,j] == -Im(C)[j,i]
//      (only testable for Baseline, which writes Re(C) as a full matrix)
//
// Returns validation results struct.

struct HerkValidation {
    bool pass;
    float max_re_err;       // max |HERK_re - Gram_re| in lower triangle
    float max_im_err;       // max |HERK_im - Gram_im| in lower triangle
    float max_diag_im;      // max |Im(C)[i,i]|
    float max_symm_re_err;  // max |Re(C)[i,j] - Re(C)[j,i]| (Baseline only)
    bool re_symmetry_tested;   // true only for Baseline (full Re matrix)
};

HerkValidation validate_herk(
    const __half* herk_Cr, const __half* herk_Ci,
    const __half* ref_Cr,  const __half* ref_Ci,
    int N, bool test_re_symmetry,
    float tol = 0.5f)       // FP8 E4M3 → FP16 tolerance
{
    HerkValidation v{};
    v.re_symmetry_tested = test_re_symmetry;

    // Check lower triangle (including diagonal) against Gram reference
    float local_max_re_err = 0, local_max_im_err = 0, local_max_diag_im = 0;
    #pragma omp parallel for schedule(dynamic) reduction(max:local_max_re_err, local_max_im_err, local_max_diag_im)
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j <= i; ++j) {
            int64_t ij = static_cast<int64_t>(i) * N + j;

            float h_re = __half2float(herk_Cr[ij]);
            float h_im = __half2float(herk_Ci[ij]);
            float r_re = __half2float(ref_Cr[ij]);
            float r_im = __half2float(ref_Ci[ij]);

            local_max_re_err = std::max(local_max_re_err, std::fabs(h_re - r_re));
            local_max_im_err = std::max(local_max_im_err, std::fabs(h_im - r_im));

            if (i == j) {
                local_max_diag_im = std::max(local_max_diag_im, std::fabs(h_im));
            }
        }
    }
    v.max_re_err = local_max_re_err;
    v.max_im_err = local_max_im_err;
    v.max_diag_im = local_max_diag_im;

    // Re(C) conjugate symmetry: Re(C)[i,j] == Re(C)[j,i]
    // Only valid for Baseline which writes Re(C) as a full N×N matrix.
    // Note: Im(C) anti-symmetry is NOT tested across triangles because
    // the anti-symmetrize kernel only writes the authoritative (lower)
    // triangle — the upper triangle contains stale data, not valid Im values.
    if (test_re_symmetry) {
        float local_max_symm = 0;
        #pragma omp parallel for schedule(dynamic) reduction(max:local_max_symm)
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < i; ++j) {
                int64_t ij = static_cast<int64_t>(i) * N + j;
                int64_t ji = static_cast<int64_t>(j) * N + i;

                float re_ij = __half2float(herk_Cr[ij]);
                float re_ji = __half2float(herk_Cr[ji]);

                local_max_symm = std::max(local_max_symm, std::fabs(re_ij - re_ji));
            }
        }
        v.max_symm_re_err = local_max_symm;
    }

    v.pass = (v.max_re_err < tol) && (v.max_im_err < tol) && (v.max_diag_im < tol);
    if (test_re_symmetry) {
        v.pass = v.pass && (v.max_symm_re_err < tol);
    }
    return v;
}

void print_validation(const char* label, const HerkValidation& v) {
    std::cout << "  " << label << ": " << (v.pass ? "PASS" : "FAIL") << "\n"
              << std::scientific << std::setprecision(2)
              << "    Re(C) lower-tri vs Gram ref:  max err = " << v.max_re_err << "\n"
              << "    Im(C) lower-tri vs Gram ref:  max err = " << v.max_im_err << "\n"
              << "    Diagonal Im(C)==0:            max |Im| = " << v.max_diag_im << "\n";
    if (v.re_symmetry_tested) {
        std::cout << "    Re symmetry  C[i,j]==C[j,i]:  max err = " << v.max_symm_re_err << "\n";
    } else {
        std::cout << "    Re symmetry: skipped (triangle-only Re output)\n";
    }
    std::cout << std::fixed;
}


// ========================================================================================
// Main
// ========================================================================================

int main(int argc, char** argv) {
    if (has_help_flag(argc, argv)) {
        auto sysinfo = cutlass_complex::query_system_info();
        cutlass_complex::print_system_info(sysinfo);
        cutlass_complex::print_build_config();
        printf("\n");
        print_help(argv[0]);
        return 0;
    }

    int M, N, K, batch;
    ComputePrecision precision;
    parse_args(argc, argv, M, N, K, batch, precision);
    TriangleConfig tri = parse_triangle_config(argc, argv);
    BenchMode mode = parse_bench_mode(argc, argv);
    bool herk_graph = parse_herk_graph(argc, argv);
    HerkMode herk_mode = parse_herk_mode(argc, argv);
    PersistentMode persistent_mode = parse_persistent_mode(argc, argv);
    bool batch_tiling = parse_batch_tiling(argc, argv);
    int tune_level = parse_tune(argc, argv);
    int strat_verb = parse_strategy_verbosity(argc, argv);
    GemmMode gemm_mode = parse_gemm_mode(argc, argv);
    CrossCheck crosscheck = parse_crosscheck(argc, argv);
    if (tune_level >= 0) {
        tune_cache::TuneCache::instance().set_verbosity(tune_level);
    }
    if (strat_verb >= 0) {
        strategy_cache::StrategyCache::instance().set_verbosity(strat_verb);
        strategy_cache::GemmStrategyCache::instance().set_verbosity(strat_verb);
    }
    tune_cache::TuneCache::instance().load();
    bool is_fp8 = (precision == ComputePrecision::FP8_E4M3);
    bool run_tests = (mode == BenchMode::Both || mode == BenchMode::Test);
    bool run_bench = (mode == BenchMode::Both || mode == BenchMode::Bench);

    // Adaptive warmup/iteration counts
    int warmup, iters;
    adaptive_bench_counts(N, warmup, iters);

    // ---- System info ----
    auto sysinfo = cutlass_complex::query_system_info();
    int sm_count = sysinfo.sm_count;
    std::vector<RooflineEntry> roofline;  // collected during benchmarks

    std::cout << "Complex GEMM Benchmark — CUTLASS 4.x, Blackwell SM100/SM120\n";
    cutlass_complex::print_system_info(sysinfo);
    cutlass_complex::print_build_config();

    std::cout << "Problem: " << M << "x" << N << "x" << K;
    if (batch > 1) std::cout << " batch=" << batch;
#ifdef COMPLEX_SM100_FP8_TILE_N
    std::cout << "  Tile: " << COMPLEX_FP8_SM100_MMA_M << "x"
              << COMPLEX_SM100_FP8_TILE_N << "x" << COMPLEX_FP8_SM100_MMA_K
#elif defined(COMPLEX_FP8_SM100_TARGET_SM120)
    std::cout << "  Tile: 128x64x" << COMPLEX_FP8_SM100_MMA_K
#else
    std::cout << "  Tile: " << COMPLEX_FP8_SM100_MMA_M << "x"
              << COMPLEX_FP8_SM100_MMA_N << "x" << COMPLEX_FP8_SM100_MMA_K
#endif
              << "  Cluster: " << COMPLEX_FP8_SM100_CLUSTER_M << "x"
              << COMPLEX_FP8_SM100_CLUSTER_N << "x1"
#if defined(COMPLEX_FP8_SM100_USE_2SM) && COMPLEX_FP8_SM100_USE_2SM
              << "  2SM"
#else
              << "  1SM"
#endif
              << "\nPrecision: " << precision_name(precision) << " compute, FP16 I/O, FP32 accumulate\n";
    if (mode != BenchMode::Both)
        std::cout << "Mode: " << (mode == BenchMode::Test ? "test" : "bench") << "\n";
    if (run_bench)
        std::cout << "Benchmark: " << warmup << " warmup + " << iters << " iterations\n";
    if (tri.target_slabs > 0 || tri.min_slab_height > 0 || tri.graduated || tri.verbose)
        std::cout << "Triangle: slabs=" << tri.target_slabs << " min_slab=" << tri.min_slab_height
                  << " graduated=" << tri.graduated << " verbose=" << tri.verbose << "\n";
    if (herk_graph)
        std::cout << "HERK graph: ON (baseline HERK captured as CUDA graph)\n";
    if (herk_mode == HerkMode::Auto)
        std::cout << "Direct HERK: AUTO (direct when K <= N/4, baseline otherwise)\n";
    else if (herk_mode == HerkMode::ForceDirect)
        std::cout << "Direct HERK: ON (single-kernel TCC-inspired direct HERK)\n";
    if (crosscheck != CrossCheck::Off)
        std::cout << "Cross-check: " << (crosscheck == CrossCheck::CPU ? "CPU (FP64 reference)" : "GPU (per-batch independence)") << "\n";
    std::cout << "\n";

    // ---- Allocate host data (interleaved complex: [Re0,Im0,Re1,Im1,...]) ----
    int64_t size_A = static_cast<int64_t>(M) * K;
    int64_t size_B = static_cast<int64_t>(K) * N;
    int64_t size_C = static_cast<int64_t>(M) * N;
    int64_t herk_C_size = static_cast<int64_t>(M) * M;

    std::vector<__half> h_A(2 * size_A), h_B(2 * size_B);
    std::vector<__half> h_C(2 * size_C, __float2half(0.0f));

    fill_random_fp16(h_A, 0.5f);
    fill_random_fp16(h_B, 0.5f);

    // ---- Allocate device memory (interleaved) ----
    __half *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, 2 * size_A * sizeof(__half));
    cudaMalloc(&d_B, 2 * size_B * sizeof(__half));
    cudaMalloc(&d_C, 2 * size_C * sizeof(__half));

    cudaMemcpy(d_A, h_A.data(), 2 * size_A * sizeof(__half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B.data(), 2 * size_B * sizeof(__half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, h_C.data(), 2 * size_C * sizeof(__half), cudaMemcpyHostToDevice);

    // Planar views for Gram benchmarks and test validation
    __half *d_Ar, *d_Ai, *d_Br, *d_Bi, *d_Cr, *d_Ci;
    cudaMalloc(&d_Ar, size_A * sizeof(__half));
    cudaMalloc(&d_Ai, size_A * sizeof(__half));
    cudaMalloc(&d_Br, size_B * sizeof(__half));
    cudaMalloc(&d_Bi, size_B * sizeof(__half));
    cudaMalloc(&d_Cr, size_C * sizeof(__half));
    cudaMalloc(&d_Ci, size_C * sizeof(__half));

    GemmComplexSm100 gemm;

    // ---- Strategy autotune or manual configuration ----
    if (tune_level == -1) {
        // Full autotune: sweep all runtime options, override CLI flags
        // Default strategy verbosity to 2 for full sweep output, unless user overrode
        if (strat_verb < 0)
            strategy_cache::StrategyCache::instance().set_verbosity(2);

        // Map compute precision to cache key (0=FP8, 1=FP6, 2=FP4)
        int prec_key = 0;
        if (precision == ComputePrecision::FP6_E3M2 || precision == ComputePrecision::FP6_E2M3)
            prec_key = 1;
        else if (precision == ComputePrecision::FP4_E2M1)
            prec_key = 2;

        // Build viable config list for GemmConfig sweep
        auto viable = all_baseline_configs();
        std::vector<int> viable_ints;
        for (auto c : viable) viable_ints.push_back(static_cast<int>(c));

        auto entry = strategy_cache::run_autotune<
            GemmComplexSm100, HerkMode, PersistentMode, DirectHerkConfig, HerkTileSize,
            HerkPipelineMode,
            HerkOp, FillMode, HerkStrategy, CutlassParams, TriangleConfig>(
                gemm, N, K, batch, prec_key, nullptr, viable_ints);

        // Apply ALL cached settings (overrides CLI flags)
        strategy_cache::apply_cached_settings<GemmComplexSm100, HerkMode, PersistentMode,
                                              GemmConfig, DirectHerkConfig, HerkTileSize,
                                              HerkPipelineMode>(
            gemm, entry);
        tri.use_cuda_graph = entry.use_cuda_graph;
        herk_mode = entry.herk_mode == 0 ? HerkMode::ForceDirect : HerkMode::ForceBaseline;
        persistent_mode = entry.persistent_mode == 1 ? PersistentMode::ForceOn : PersistentMode::ForceOff;
        herk_graph = entry.herk_graph;
        batch_tiling = entry.batch_tiling;

        printf("[Autotune] Applied: %s+%s+persistent=%d+graph=%d+herk_graph=%d+tile=%d+gemm_config=%d(%s)+direct_config=%d(%s) (%.1f TFLOPS)\n",
               entry.herk_mode == 0 ? "Direct" : "Baseline",
               entry.herk_strategy == 1 ? "TriangleAware" : "Baseline",
               entry.persistent_mode, (int)entry.use_cuda_graph,
               (int)entry.herk_graph, (int)entry.batch_tiling,
               entry.gemm_config, config_name(static_cast<GemmConfig>(entry.gemm_config)),
               entry.direct_config, direct_herk_config_name(static_cast<DirectHerkConfig>(entry.direct_config)),
               entry.tflops);
    } else {
        // Manual configuration from CLI flags
        if (herk_graph) gemm.set_herk_graph(true);
        gemm.set_herk_mode(herk_mode);
        gemm.set_persistent_mode(persistent_mode);
        gemm.set_batch_tiling(batch_tiling);
    }

    // Pre-check: does FP8 baseline GEMM fit in device SMEM?
    // On builds like stg3+SM120 the kernel needs 112 KB but device has 99 KB.
    bool fp8_baseline_ok = GemmComplexSm100::fp8_baseline_available();
    if (!fp8_baseline_ok) {
        std::cout << "Note: FP8 baseline GEMM kernel exceeds device SMEM — "
                     "FP8-dependent tests will be SKIPPED.\n\n";
    }

    // Deinterleave A, B to planar for Gram / test-phase calls
    GemmComplexSm100::deinterleave_complex(d_A, d_Ar, d_Ai, size_A, nullptr);
    GemmComplexSm100::deinterleave_complex(d_B, d_Br, d_Bi, size_B, nullptr);
    cudaMemset(d_Cr, 0, size_C * sizeof(__half));
    cudaMemset(d_Ci, 0, size_C * sizeof(__half));


    // ====================================================================
    // TEST PHASE — correctness validation
    // ====================================================================
    if (run_tests) {
    std::cout << "╔══════════════════════════════════════════════════════╗\n"
              << "║                    TEST PHASE                       ║\n"
              << "╚══════════════════════════════════════════════════════╝\n\n";

    std::cout << "Standard GEMM (C=A*B):\n";
    if (is_fp8 && !fp8_baseline_ok) {
        std::cout << "  SKIPPED (FP8 kernel exceeds device SMEM)\n\n";
    } else {
        CutlassParams test_params;
        test_params.precision = precision;
        cudaMemset(d_C, 0, 2 * size_C * sizeof(__half));
        auto status = gemm.GEMM(d_A, d_B, d_C, M, N, K, 1.0f, 0.0f, ComplexMode::Standard, test_params);
        std::cout << "  Launch: " << (status == cutlass::Status::kSuccess ? "PASS" : "SKIPPED") << "\n\n";
    }

    std::cout << "Hermitian GEMM (C=A*B^H):\n";
    if (is_fp8 && !fp8_baseline_ok) {
        std::cout << "  SKIPPED (FP8 kernel exceeds device SMEM)\n\n";
    } else {
        CutlassParams test_params;
        test_params.precision = precision;
        cudaMemset(d_C, 0, 2 * size_C * sizeof(__half));
        auto status = gemm.GEMM(d_A, d_B, d_C, M, N, K, 1.0f, 0.0f, ComplexMode::Hermitian, test_params);
        std::cout << "  Launch: " << (status == cutlass::Status::kSuccess ? "PASS" : "SKIPPED") << "\n\n";
    }

    __half *d_RefCr = nullptr, *d_RefCi = nullptr;
    std::vector<__half> h_RefCr, h_RefCi;
    bool have_gram_ref = false;
    float herk_tol = is_fp8 ? 0.5f : 2.0f;

    if (is_fp8 && !fp8_baseline_ok) {
        std::cout << "Gram reference (C=A*A^H, 4 sub-GEMMs):\n"
                  << "  SKIPPED (FP8 kernel exceeds device SMEM)\n\n"
                  << "HERK Baseline (3 sub-GEMMs + anti-symmetrize):\n"
                  << "  SKIPPED (FP8 kernel exceeds device SMEM)\n\n"
                  << "HERK Triangle (block-row decomposition):\n"
                  << "  SKIPPED (FP8 kernel exceeds device SMEM)\n\n";
    } else {
        std::cout << "Gram reference (C=A*A^H, 4 sub-GEMMs):\n";
        cudaMalloc(&d_RefCr, herk_C_size * sizeof(__half));
        cudaMalloc(&d_RefCi, herk_C_size * sizeof(__half));
        cudaMemset(d_RefCr, 0, herk_C_size * sizeof(__half));
        cudaMemset(d_RefCi, 0, herk_C_size * sizeof(__half));
        {
            auto status = gemm.run_gram_planar(d_Ar, d_Ai, d_RefCr, d_RefCi,
                                                M, K, 1.0f, 0.0f, GramMode::AAH, precision);
            std::cout << "  Launch: " << (status == cutlass::Status::kSuccess ? "PASS" : "SKIPPED") << "\n";
            std::cout << "  (used as ground truth for HERK validation)\n\n";
            have_gram_ref = (status == cutlass::Status::kSuccess);
        }

        // Copy reference to host
        if (have_gram_ref) {
            h_RefCr.resize(herk_C_size);
            h_RefCi.resize(herk_C_size);
            cudaMemcpy(h_RefCr.data(), d_RefCr, herk_C_size * sizeof(__half), cudaMemcpyDeviceToHost);
            cudaMemcpy(h_RefCi.data(), d_RefCi, herk_C_size * sizeof(__half), cudaMemcpyDeviceToHost);
        }

        std::cout << "HERK Baseline (3 sub-GEMMs + anti-symmetrize):\n";
        {
            __half *d_HCr, *d_HCi;
            cudaMalloc(&d_HCr, herk_C_size * sizeof(__half));
            cudaMalloc(&d_HCi, herk_C_size * sizeof(__half));
            cudaMemset(d_HCr, 0, herk_C_size * sizeof(__half));
            cudaMemset(d_HCi, 0, herk_C_size * sizeof(__half));

            auto status = gemm.herk_planar(d_Ar, d_Ai, d_HCr, d_HCi,
                                            M, K, 1.0f, 0.0f,
                                            HerkOp::NoTrans, FillMode::Lower,
                                            HerkStrategy::Baseline, precision);
            std::cout << "  Launch: " << (status == cutlass::Status::kSuccess ? "PASS" : "SKIPPED") << "\n";

            if (status == cutlass::Status::kSuccess && have_gram_ref) {
                std::vector<__half> h_HCr(herk_C_size), h_HCi(herk_C_size);
                cudaMemcpy(h_HCr.data(), d_HCr, herk_C_size * sizeof(__half), cudaMemcpyDeviceToHost);
                cudaMemcpy(h_HCi.data(), d_HCi, herk_C_size * sizeof(__half), cudaMemcpyDeviceToHost);

                // Baseline writes full Re(C) — can test conjugate symmetry
                auto v = validate_herk(h_HCr.data(), h_HCi.data(),
                                       h_RefCr.data(), h_RefCi.data(),
                                       M, /*test_symmetry=*/true, herk_tol);
                print_validation("Hermiticity", v);
            }

            cudaFree(d_HCr);
            cudaFree(d_HCi);
        }
        std::cout << "\n";

        std::cout << "HERK Triangle (block-row decomposition):\n";
        {
            __half *d_HCr, *d_HCi;
            cudaMalloc(&d_HCr, herk_C_size * sizeof(__half));
            cudaMalloc(&d_HCi, herk_C_size * sizeof(__half));
            cudaMemset(d_HCr, 0, herk_C_size * sizeof(__half));
            cudaMemset(d_HCi, 0, herk_C_size * sizeof(__half));

            auto status = gemm.herk_planar(d_Ar, d_Ai, d_HCr, d_HCi,
                                            M, K, 1.0f, 0.0f,
                                            HerkOp::NoTrans, FillMode::Lower,
                                            HerkStrategy::TriangleAware, precision,
                                            nullptr, GemmConfig::Default, tri);
            std::cout << "  Launch: " << (status == cutlass::Status::kSuccess ? "PASS" : "SKIPPED") << "\n";

            if (status == cutlass::Status::kSuccess && have_gram_ref) {
                std::vector<__half> h_HCr(herk_C_size), h_HCi(herk_C_size);
                cudaMemcpy(h_HCr.data(), d_HCr, herk_C_size * sizeof(__half), cudaMemcpyDeviceToHost);
                cudaMemcpy(h_HCi.data(), d_HCi, herk_C_size * sizeof(__half), cudaMemcpyDeviceToHost);

                // Triangle-aware only writes lower triangle of Re(C) — skip symmetry
                auto v = validate_herk(h_HCr.data(), h_HCi.data(),
                                       h_RefCr.data(), h_RefCi.data(),
                                       M, /*test_symmetry=*/false, herk_tol);
                print_validation("Hermiticity", v);
            }

            cudaFree(d_HCr);
            cudaFree(d_HCi);
        }

        if (d_RefCr) cudaFree(d_RefCr);
        if (d_RefCi) cudaFree(d_RefCi);
    }

    // ---- Batched HERK test ----
    std::cout << "Batched HERK (FP8, TriangleAware vs Baseline):\n";
    {
        const int batch_count = 4;
        int64_t per_A = static_cast<int64_t>(M) * K;
        int64_t packed_elems = static_cast<int64_t>(M) * (M + 1) / 2;

        // Allocate batched interleaved A: batch_count copies of d_A
        __half* d_A_batched;
        cudaMalloc(&d_A_batched, 2 * per_A * batch_count * sizeof(__half));
        for (int b = 0; b < batch_count; ++b) {
            cudaMemcpy(d_A_batched + b * 2 * per_A, d_A,
                       2 * per_A * sizeof(__half), cudaMemcpyDeviceToDevice);
        }

        // Allocate batched packed C outputs
        __half *d_C_baseline, *d_C_triangle;
        cudaMalloc(&d_C_baseline, 2 * packed_elems * batch_count * sizeof(__half));
        cudaMalloc(&d_C_triangle, 2 * packed_elems * batch_count * sizeof(__half));
        cudaMemset(d_C_baseline, 0, 2 * packed_elems * batch_count * sizeof(__half));
        cudaMemset(d_C_triangle, 0, 2 * packed_elems * batch_count * sizeof(__half));

        // Baseline
        CutlassParams bp;
        bp.herk_strategy = HerkStrategy::Baseline;
        auto status_base = gemm.HERK_batched(
            d_A_batched, d_C_baseline, M, K, batch_count,
            1.0f, 0.0f, HerkOp::NoTrans, FillMode::Lower, bp);
        std::cout << "  Baseline:  "
                  << (status_base == cutlass::Status::kSuccess ? "PASS" : "SKIPPED") << "\n";

        // TriangleAware
        CutlassParams tp;
        tp.herk_strategy = HerkStrategy::TriangleAware;
        tp.triangle_config = tri;
        auto status_tri = gemm.HERK_batched(
            d_A_batched, d_C_triangle, M, K, batch_count,
            1.0f, 0.0f, HerkOp::NoTrans, FillMode::Lower, tp);
        std::cout << "  Triangle:  "
                  << (status_tri == cutlass::Status::kSuccess ? "PASS" : "SKIPPED") << "\n";

        // Validate: compare TriangleAware output against Baseline output
        if (status_base == cutlass::Status::kSuccess && status_tri == cutlass::Status::kSuccess) {
            int64_t total_packed = packed_elems * batch_count;
            std::vector<__half> h_base(2 * total_packed), h_tri(2 * total_packed);
            cudaMemcpy(h_base.data(), d_C_baseline, 2 * total_packed * sizeof(__half),
                       cudaMemcpyDeviceToHost);
            cudaMemcpy(h_tri.data(), d_C_triangle, 2 * total_packed * sizeof(__half),
                       cudaMemcpyDeviceToHost);

            float max_err = 0.0f;
            for (int64_t i = 0; i < 2 * total_packed; ++i) {
                float diff = fabsf(__half2float(h_tri[i]) - __half2float(h_base[i]));
                max_err = fmaxf(max_err, diff);
            }
            bool match = (max_err < herk_tol);
            std::cout << "  Baseline vs Triangle match: " << (match ? "PASS" : "FAIL")
                      << "  (max err = " << max_err << ", tol = " << herk_tol << ")\n";
        }

        cudaFree(d_A_batched);
        cudaFree(d_C_baseline);
        cudaFree(d_C_triangle);
    }

    // ---- Direct HERK vs CUTLASS baseline test ----
    std::cout << "Direct HERK vs CUTLASS baseline (FP8 NoTrans):\n";
    if (is_fp8 && !fp8_baseline_ok) {
        std::cout << "  SKIPPED (FP8 baseline GEMM exceeds device SMEM — cannot compare)\n";
    } else {
        const int batch_count = 4;
        int64_t per_A = static_cast<int64_t>(M) * K;
        int64_t packed_elems = static_cast<int64_t>(M) * (M + 1) / 2;

        __half* d_A_batched;
        cudaMalloc(&d_A_batched, 2 * per_A * batch_count * sizeof(__half));
        for (int b = 0; b < batch_count; ++b) {
            cudaMemcpy(d_A_batched + b * 2 * per_A, d_A,
                       2 * per_A * sizeof(__half), cudaMemcpyDeviceToDevice);
        }

        __half *d_C_cutlass, *d_C_direct;
        cudaMalloc(&d_C_cutlass, 2 * packed_elems * batch_count * sizeof(__half));
        cudaMalloc(&d_C_direct,  2 * packed_elems * batch_count * sizeof(__half));
        cudaMemset(d_C_cutlass, 0, 2 * packed_elems * batch_count * sizeof(__half));
        cudaMemset(d_C_direct,  0, 2 * packed_elems * batch_count * sizeof(__half));

        // Run CUTLASS baseline path (direct disabled)
        HerkMode saved_mode = gemm.herk_mode();
        gemm.set_herk_mode(HerkMode::ForceBaseline);
        CutlassParams bp;
        bp.herk_strategy = HerkStrategy::Baseline;
        auto status_cutlass = gemm.HERK_batched(
            d_A_batched, d_C_cutlass, M, K, batch_count,
            1.0f, 0.0f, HerkOp::NoTrans, FillMode::Lower, bp);
        std::cout << "  Baseline:  "
                  << (status_cutlass == cutlass::Status::kSuccess ? "PASS" : "SKIPPED") << "\n";

        // Run direct path
        gemm.set_herk_mode(HerkMode::ForceDirect);
        auto status_direct = gemm.HERK_batched(
            d_A_batched, d_C_direct, M, K, batch_count,
            1.0f, 0.0f, HerkOp::NoTrans, FillMode::Lower, bp);
        gemm.set_herk_mode(saved_mode);
        std::cout << "  Direct:    "
                  << (status_direct == cutlass::Status::kSuccess ? "PASS" : "SKIPPED") << "\n";

        // Compare
        if (status_cutlass == cutlass::Status::kSuccess && status_direct == cutlass::Status::kSuccess) {
            int64_t total_packed = packed_elems * batch_count;
            std::vector<__half> h_cutlass(2 * total_packed), h_direct(2 * total_packed);
            cudaMemcpy(h_cutlass.data(), d_C_cutlass, 2 * total_packed * sizeof(__half),
                       cudaMemcpyDeviceToHost);
            cudaMemcpy(h_direct.data(), d_C_direct, 2 * total_packed * sizeof(__half),
                       cudaMemcpyDeviceToHost);

            float max_err = 0.0f, max_rel = 0.0f;
            int64_t nonzero_cutlass = 0, nonzero_direct = 0;
            for (int64_t i = 0; i < 2 * total_packed; ++i) {
                float vc = __half2float(h_cutlass[i]);
                float vd = __half2float(h_direct[i]);
                if (vc != 0.0f) nonzero_cutlass++;
                if (vd != 0.0f) nonzero_direct++;
                float diff = fabsf(vd - vc);
                if (diff > max_err) {
                    max_err = diff;
                }
                float denom = fmaxf(fabsf(vc), 1e-6f);
                max_rel = fmaxf(max_rel, diff / denom);
            }
            // FP8 accumulation error grows with K: tolerance = K * eps_fp8 * scale
            float direct_tol = fmaxf(1.0f, K * 0.01f);  // generous tolerance
            bool match = (max_err < direct_tol);
            printf("  Match:     %s  (max abs err = %.4e, max rel err = %.4e, tol = %.2f)\n",
                   match ? "PASS" : "FAIL", max_err, max_rel, direct_tol);
            printf("  Nonzero: Baseline=%ld Direct=%ld (total=%ld)\n",
                   (long)nonzero_cutlass, (long)nonzero_direct, (long)(2*total_packed));
            // Print first 5 mismatch examples
            int printed = 0;
            for (int64_t i = 0; i < 2 * total_packed && printed < 5; ++i) {
                float vc = __half2float(h_cutlass[i]);
                float vd = __half2float(h_direct[i]);
                float diff = fabsf(vd - vc);
                if (diff > 0.001f) {
                    printf("    [%ld] Baseline=%.4f Direct=%.4f diff=%.4e\n",
                           (long)i, vc, vd, diff);
                    printed++;
                }
            }
        }

        cudaFree(d_A_batched);
        cudaFree(d_C_cutlass);
        cudaFree(d_C_direct);
    } // end else (fp8_baseline_ok)

    // ---- Batched HERK independence cross-check ----
    if (crosscheck != CrossCheck::Off) {
        const char* xc_mode = (crosscheck == CrossCheck::CPU) ? "CPU (FP64 reference)" : "GPU (per-batch independence)";
        printf("\nDirect HERK Batched Independence Cross-Check [%s]:\n", xc_mode);

        const int xc_batch = 4;
        int64_t per_A = static_cast<int64_t>(M) * K;
        int64_t packed_elems = static_cast<int64_t>(M) * (M + 1) / 2;
        int64_t packed_batch_stride = 2 * packed_elems;  // Re+Im interleaved
        int64_t A_batch_stride = 2 * per_A;              // Re+Im interleaved

        // Generate DIFFERENT random data per batch element (separate seed from main test)
        std::vector<__half> h_A_xc(A_batch_stride * xc_batch);
        {
            std::mt19937 rng(12345);
            std::uniform_real_distribution<float> dist(-0.5f, 0.5f);
            for (auto& v : h_A_xc) v = __float2half(dist(rng));
        }

        // Verify data is actually different per batch (sanity check)
        {
            float checksum0 = 0, checksum1 = 0;
            for (int64_t i = 0; i < std::min(A_batch_stride, (int64_t)1000); i++) {
                checksum0 += __half2float(h_A_xc[i]);
                checksum1 += __half2float(h_A_xc[A_batch_stride + i]);
            }
            printf("  Data checksums: batch0=%.4f batch1=%.4f (must differ)\n", checksum0, checksum1);
        }

        // Upload to device
        __half* d_A_xc;
        cudaMalloc(&d_A_xc, A_batch_stride * xc_batch * sizeof(__half));
        cudaMemcpy(d_A_xc, h_A_xc.data(), A_batch_stride * xc_batch * sizeof(__half),
                   cudaMemcpyHostToDevice);

        // Run direct kernel on full batch
        __half* d_C_xc;
        cudaMalloc(&d_C_xc, packed_batch_stride * xc_batch * sizeof(__half));
        cudaMemset(d_C_xc, 0, packed_batch_stride * xc_batch * sizeof(__half));

        HerkMode saved_mode = gemm.herk_mode();
        gemm.set_herk_mode(HerkMode::ForceDirect);
        CutlassParams xc_params;
        xc_params.herk_strategy = HerkStrategy::Baseline;
        auto xc_status = gemm.HERK_batched(
            d_A_xc, d_C_xc, M, K, xc_batch,
            1.0f, 0.0f, HerkOp::NoTrans, FillMode::Lower, xc_params);
        cudaDeviceSynchronize();

        if (xc_status != cutlass::Status::kSuccess) {
            printf("  Direct HERK batched launch SKIPPED (SMEM overflow)\n");
        } else {
            // Copy batched result to host
            std::vector<__half> h_C_batched(packed_batch_stride * xc_batch);
            cudaMemcpy(h_C_batched.data(), d_C_xc,
                       packed_batch_stride * xc_batch * sizeof(__half),
                       cudaMemcpyDeviceToHost);

            if (crosscheck == CrossCheck::CPU) {
                // ---- CPU FP64 reference per batch element ----
                printf("  Computing CPU FP64 reference (%d batches, N=%d, K=%d)...\n",
                       xc_batch, M, K);
                auto t0 = std::chrono::high_resolution_clock::now();

                // Compute CPU reference and compare per batch (parallelized)
                std::vector<bool> batch_pass(xc_batch, true);
                std::vector<float> batch_max_abs(xc_batch, 0.0f);
                std::vector<float> batch_max_rel(xc_batch, 0.0f);
                std::vector<int64_t> batch_worst_idx(xc_batch, -1);

                #pragma omp parallel for schedule(dynamic)
                for (int b = 0; b < xc_batch; b++) {
                    std::vector<__half> h_C_ref(packed_batch_stride);
                    cpu_herk_single(
                        h_A_xc.data() + b * A_batch_stride,
                        h_C_ref.data(),
                        M, K, 1.0f, 0.0f);

                    // Compare GPU batch b vs CPU reference
                    float max_abs = 0.0f, max_rel = 0.0f;
                    int64_t worst_idx = -1;
                    for (int64_t i = 0; i < packed_batch_stride; i++) {
                        float vg = __half2float(h_C_batched[b * packed_batch_stride + i]);
                        float vr = __half2float(h_C_ref[i]);
                        float diff = fabsf(vg - vr);
                        if (diff > max_abs) { max_abs = diff; worst_idx = i; }
                        max_rel = fmaxf(max_rel, diff / fmaxf(fabsf(vr), 1e-6f));
                    }
                    float tol = fmaxf(1.0f, K * 0.02f);
                    batch_pass[b] = (max_abs < tol);
                    batch_max_abs[b] = max_abs;
                    batch_max_rel[b] = max_rel;
                    batch_worst_idx[b] = worst_idx;
                }

                // Print results sequentially (preserve output order)
                bool all_pass = true;
                float tol = fmaxf(1.0f, K * 0.02f);
                for (int b = 0; b < xc_batch; b++) {
                    if (!batch_pass[b]) all_pass = false;
                    printf("  Batch %d: %s  max_abs=%.4e  max_rel=%.4e  (tol=%.2f)\n",
                           b, batch_pass[b] ? "PASS" : "FAIL", batch_max_abs[b], batch_max_rel[b], tol);
                    if (!batch_pass[b] && batch_worst_idx[b] >= 0) {
                        float vg = __half2float(h_C_batched[b * packed_batch_stride + batch_worst_idx[b]]);
                        // Recompute CPU ref for the worst element (unavoidable without storing full ref)
                        std::vector<__half> h_C_ref_tmp(packed_batch_stride);
                        cpu_herk_single(h_A_xc.data() + b * A_batch_stride,
                                        h_C_ref_tmp.data(), M, K, 1.0f, 0.0f);
                        float vr = __half2float(h_C_ref_tmp[batch_worst_idx[b]]);
                        printf("    worst @[%ld]: GPU=%.6f CPU=%.6f diff=%.4e\n",
                               (long)batch_worst_idx[b], vg, vr, fabsf(vg - vr));
                    }
                }

                auto t1 = std::chrono::high_resolution_clock::now();
                double cpu_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
                printf("  CPU reference time: %.1f ms\n", cpu_ms);
                printf("  Overall: %s\n", all_pass ? "PASS" : "FAIL");

            } else {
                // ---- GPU per-batch independence check ----
                // Run direct kernel individually per batch element, compare against batched result.
                // Uses CUDA event timing (autotuner-style) to measure batched vs individual performance.
                __half* d_C_individual;
                cudaMalloc(&d_C_individual, packed_batch_stride * xc_batch * sizeof(__half));

                cudaEvent_t ev_start, ev_stop;
                cudaEventCreate(&ev_start);
                cudaEventCreate(&ev_stop);

                // Time the batched launch (warmup + timed)
                cudaMemset(d_C_xc, 0, packed_batch_stride * xc_batch * sizeof(__half));
                gemm.HERK_batched(d_A_xc, d_C_xc, M, K, xc_batch,
                    1.0f, 0.0f, HerkOp::NoTrans, FillMode::Lower, xc_params);  // warmup
                cudaDeviceSynchronize();
                cudaEventRecord(ev_start);
                gemm.HERK_batched(d_A_xc, d_C_xc, M, K, xc_batch,
                    1.0f, 0.0f, HerkOp::NoTrans, FillMode::Lower, xc_params);
                cudaEventRecord(ev_stop);
                cudaEventSynchronize(ev_stop);
                float batched_ms = 0;
                cudaEventElapsedTime(&batched_ms, ev_start, ev_stop);

                // Re-copy the batched result (it was overwritten by warmup+timed runs)
                cudaMemcpy(h_C_batched.data(), d_C_xc,
                           packed_batch_stride * xc_batch * sizeof(__half),
                           cudaMemcpyDeviceToHost);

                // Time the individual per-batch launches (warmup + timed)
                bool all_launched = true;
                for (int b = 0; b < xc_batch; b++) {
                    cudaMemset(d_C_individual + b * packed_batch_stride, 0,
                               packed_batch_stride * sizeof(__half));
                    auto s = gemm.HERK_batched(
                        d_A_xc + b * A_batch_stride,
                        d_C_individual + b * packed_batch_stride,
                        M, K, 1,
                        1.0f, 0.0f, HerkOp::NoTrans, FillMode::Lower, xc_params);
                    if (s != cutlass::Status::kSuccess) {
                        printf("  Batch %d individual launch SKIPPED (SMEM overflow)\n", b);
                        all_launched = false;
                    }
                }
                cudaDeviceSynchronize();  // warmup

                // Timed run
                for (int b = 0; b < xc_batch; b++) {
                    cudaMemset(d_C_individual + b * packed_batch_stride, 0,
                               packed_batch_stride * sizeof(__half));
                }
                cudaEventRecord(ev_start);
                for (int b = 0; b < xc_batch; b++) {
                    gemm.HERK_batched(
                        d_A_xc + b * A_batch_stride,
                        d_C_individual + b * packed_batch_stride,
                        M, K, 1,
                        1.0f, 0.0f, HerkOp::NoTrans, FillMode::Lower, xc_params);
                }
                cudaEventRecord(ev_stop);
                cudaEventSynchronize(ev_stop);
                float individual_ms = 0;
                cudaEventElapsedTime(&individual_ms, ev_start, ev_stop);

                printf("  Timing: batched=%.3f ms  individual=%d x %.3f ms = %.3f ms  (%.2fx)\n",
                       batched_ms, xc_batch, individual_ms / xc_batch,
                       individual_ms, individual_ms / fmaxf(batched_ms, 0.001f));

                if (all_launched) {
                    std::vector<__half> h_C_individual(packed_batch_stride * xc_batch);
                    cudaMemcpy(h_C_individual.data(), d_C_individual,
                               packed_batch_stride * xc_batch * sizeof(__half),
                               cudaMemcpyDeviceToHost);

                    bool all_pass = true;
                    for (int b = 0; b < xc_batch; b++) {
                        int64_t mismatches = 0;
                        float max_diff = 0.0f;
                        int64_t first_mismatch = -1;
                        for (int64_t i = 0; i < packed_batch_stride; i++) {
                            float vb = __half2float(h_C_batched[b * packed_batch_stride + i]);
                            float vi = __half2float(h_C_individual[b * packed_batch_stride + i]);
                            float diff = fabsf(vb - vi);
                            max_diff = fmaxf(max_diff, diff);
                            if (vb != vi) {
                                mismatches++;
                                if (first_mismatch < 0) first_mismatch = i;
                            }
                        }
                        bool pass = (mismatches == 0);
                        if (!pass) all_pass = false;
                        printf("  Batch %d: %s  mismatches=%ld/%ld  max_diff=%.4e\n",
                               b, pass ? "PASS" : "FAIL",
                               (long)mismatches, (long)packed_batch_stride, max_diff);
                        if (!pass && first_mismatch >= 0) {
                            float vb = __half2float(h_C_batched[b * packed_batch_stride + first_mismatch]);
                            float vi = __half2float(h_C_individual[b * packed_batch_stride + first_mismatch]);
                            printf("    first mismatch @[%ld]: batched=%.6f individual=%.6f\n",
                                   (long)first_mismatch, vb, vi);
                        }
                    }
                    printf("  Overall: %s (bit-exact per-batch)\n", all_pass ? "PASS" : "FAIL");
                }

                cudaEventDestroy(ev_start);
                cudaEventDestroy(ev_stop);
                cudaFree(d_C_individual);
            }
        }

        gemm.set_herk_mode(saved_mode);
        cudaFree(d_A_xc);
        cudaFree(d_C_xc);
    }

    std::cout << "\n";

    } // end run_tests


    // ====================================================================
    // MULTI-PRECISION TEST — FP6/FP4 correctness (compile-time gated)
    // ====================================================================
    //
    // These tests validate the narrow-precision paths via run_planar(),
    // herk_planar() (Baseline + Triangle), and packed HERK() (Baseline vs Triangle).
    // Uses 1024³ problem size for faster validation. Expected accuracy:
    //   FP8 > FP6_E3M2 > FP6_E2M3 > FP4 (due to decreasing mantissa bits)
    //
#if defined(COMPLEX_SM100_ENABLE_FP6) || defined(COMPLEX_SM100_ENABLE_FP4)
    if (run_tests) {
        std::cout << "╔══════════════════════════════════════════════════════╗\n"
                  << "║              MULTI-PRECISION TEST PHASE              ║\n"
                  << "╚══════════════════════════════════════════════════════╝\n\n";

        // Use smaller problem size for narrow precision tests
        const int test_M = std::min(M, 1024);
        const int test_N = std::min(N, 1024);
        const int test_K = std::min(K, 1024);

        int64_t test_sA = static_cast<int64_t>(test_M) * test_K;
        int64_t test_sB = static_cast<int64_t>(test_K) * test_N;
        int64_t test_sC = static_cast<int64_t>(test_M) * test_N;

        // Allocate planar test buffers
        __half *t_Ar, *t_Ai, *t_Br, *t_Bi;
        __half *t_Cr_ref, *t_Ci_ref;  // FP8 reference
        __half *t_Cr, *t_Ci;          // narrow precision result
        cudaMalloc(&t_Ar, test_sA * sizeof(__half));
        cudaMalloc(&t_Ai, test_sA * sizeof(__half));
        cudaMalloc(&t_Br, test_sB * sizeof(__half));
        cudaMalloc(&t_Bi, test_sB * sizeof(__half));
        cudaMalloc(&t_Cr_ref, test_sC * sizeof(__half));
        cudaMalloc(&t_Ci_ref, test_sC * sizeof(__half));
        cudaMalloc(&t_Cr, test_sC * sizeof(__half));
        cudaMalloc(&t_Ci, test_sC * sizeof(__half));

        // Fill with random data (small scale to stay within narrow ranges)
        std::vector<__half> h_tAr(test_sA), h_tAi(test_sA);
        std::vector<__half> h_tBr(test_sB), h_tBi(test_sB);
        fill_random_fp16(h_tAr, 0.3f);
        fill_random_fp16(h_tAi, 0.3f);
        fill_random_fp16(h_tBr, 0.3f);
        fill_random_fp16(h_tBi, 0.3f);

        cudaMemcpy(t_Ar, h_tAr.data(), test_sA * sizeof(__half), cudaMemcpyHostToDevice);
        cudaMemcpy(t_Ai, h_tAi.data(), test_sA * sizeof(__half), cudaMemcpyHostToDevice);
        cudaMemcpy(t_Br, h_tBr.data(), test_sB * sizeof(__half), cudaMemcpyHostToDevice);
        cudaMemcpy(t_Bi, h_tBi.data(), test_sB * sizeof(__half), cudaMemcpyHostToDevice);

        // FP8 reference
        cudaMemset(t_Cr_ref, 0, test_sC * sizeof(__half));
        cudaMemset(t_Ci_ref, 0, test_sC * sizeof(__half));
        gemm.run_planar(t_Ar, t_Ai, t_Br, t_Bi, t_Cr_ref, t_Ci_ref,
                        test_M, test_N, test_K, 1.0f, 0.0f, ComplexMode::Standard);

        std::vector<__half> h_ref_re(test_sC), h_ref_im(test_sC);
        cudaMemcpy(h_ref_re.data(), t_Cr_ref, test_sC * sizeof(__half), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_ref_im.data(), t_Ci_ref, test_sC * sizeof(__half), cudaMemcpyDeviceToHost);

        // Helper lambda to compute max absolute error vs FP8 reference
        auto compute_max_error = [&](const __half* d_out_re, const __half* d_out_im)
            -> std::pair<float, float> {
            std::vector<__half> h_re(test_sC), h_im(test_sC);
            cudaMemcpy(h_re.data(), d_out_re, test_sC * sizeof(__half), cudaMemcpyDeviceToHost);
            cudaMemcpy(h_im.data(), d_out_im, test_sC * sizeof(__half), cudaMemcpyDeviceToHost);
            float max_re = 0.0f, max_im = 0.0f;
            for (int64_t i = 0; i < test_sC; ++i) {
                float err_re = fabsf(__half2float(h_re[i]) - __half2float(h_ref_re[i]));
                float err_im = fabsf(__half2float(h_im[i]) - __half2float(h_ref_im[i]));
                max_re = fmaxf(max_re, err_re);
                max_im = fmaxf(max_im, err_im);
            }
            return {max_re, max_im};
        };

        // Tolerance per precision — Baseline vs Triangle can differ slightly
        // due to sub-GEMM decomposition changes (accumulation order).
        auto tolerance_for = [](ComputePrecision prec) -> float {
            switch (prec) {
                case ComputePrecision::FP8_E4M3:  return 0.5f;
                case ComputePrecision::FP6_E3M2:  return 1.5f;
                case ComputePrecision::FP6_E2M3:  return 1.5f;
                case ComputePrecision::FP4_E2M1:  return 2.0f;
                default:                          return 1.5f;
            }
        };

        // Helper lambda to test a precision on a given API path
        auto test_precision = [&](const char* name, ComputePrecision prec) {
            std::cout << "  --- " << name << " ---\n";

            // 1. Standard GEMM via run_planar() [F5 paired cast]
            {
                cudaMemset(t_Cr, 0, test_sC * sizeof(__half));
                cudaMemset(t_Ci, 0, test_sC * sizeof(__half));
                auto status = gemm.run_planar(t_Ar, t_Ai, t_Br, t_Bi, t_Cr, t_Ci,
                                               test_M, test_N, test_K, 1.0f, 0.0f,
                                               ComplexMode::Standard, prec);
                auto [err_re, err_im] = compute_max_error(t_Cr, t_Ci);
                std::cout << "    run_planar(Standard): "
                          << (status == cutlass::Status::kSuccess ? "PASS" : "SKIPPED")
                          << "  err Re=" << std::scientific << err_re
                          << " Im=" << err_im << "\n";
            }

            // 2. Hermitian GEMM via run_planar() [F5 transposed paired cast]
            {
                cudaMemset(t_Cr, 0, test_sC * sizeof(__half));
                cudaMemset(t_Ci, 0, test_sC * sizeof(__half));
                auto status = gemm.run_planar(t_Ar, t_Ai, t_Br, t_Bi, t_Cr, t_Ci,
                                               test_M, test_N, test_K, 1.0f, 0.0f,
                                               ComplexMode::Hermitian, prec);
                std::cout << "    run_planar(Hermitian): "
                          << (status == cutlass::Status::kSuccess ? "PASS" : "SKIPPED") << "\n";
            }

            // 3. Gram AAH via run_gram_planar() [F2 dual cast]
            {
                int gM = test_M, gK = test_K;
                int64_t gC = static_cast<int64_t>(gM) * gM;
                __half *gCr, *gCi;
                cudaMalloc(&gCr, gC * sizeof(__half));
                cudaMalloc(&gCi, gC * sizeof(__half));
                cudaMemset(gCr, 0, gC * sizeof(__half));
                cudaMemset(gCi, 0, gC * sizeof(__half));
                auto status = gemm.run_gram_planar(t_Ar, t_Ai, gCr, gCi,
                                                    gM, gK, 1.0f, 0.0f,
                                                    GramMode::AAH, prec);
                std::cout << "    run_gram_planar(AAH): "
                          << (status == cutlass::Status::kSuccess ? "PASS" : "SKIPPED") << "\n";
                cudaFree(gCr); cudaFree(gCi);
            }

            // 4. BLAS3 GEMM [F1 deinterleave+cast]
            {
                // Create interleaved test data
                int64_t iA = test_sA * 2, iB = test_sB * 2, iC = test_sC * 2;
                __half *d_iA, *d_iB, *d_iC;
                cudaMalloc(&d_iA, iA * sizeof(__half));
                cudaMalloc(&d_iB, iB * sizeof(__half));
                cudaMalloc(&d_iC, iC * sizeof(__half));

                // Interleave A and B from planar
                GemmComplexSm100::interleave_complex(t_Ar, t_Ai, d_iA, test_sA, nullptr);
                GemmComplexSm100::interleave_complex(t_Br, t_Bi, d_iB, test_sB, nullptr);
                cudaMemset(d_iC, 0, iC * sizeof(__half));

                CutlassParams p;
                p.precision = prec;
                auto status = gemm.GEMM(d_iA, d_iB, d_iC,
                                        test_M, test_N, test_K,
                                        1.0f, 0.0f, ComplexMode::Standard, p);
                std::cout << "    GEMM(Standard,BLAS3): "
                          << (status == cutlass::Status::kSuccess ? "PASS" : "SKIPPED") << "\n";
                cudaFree(d_iA); cudaFree(d_iB); cudaFree(d_iC);
            }

            // 5. HERK Baseline via herk_planar() [F2 dual cast]
            {
                int hN = test_M, hK = test_K;
                int64_t hC = static_cast<int64_t>(hN) * hN;
                __half *hCr, *hCi;
                cudaMalloc(&hCr, hC * sizeof(__half));
                cudaMalloc(&hCi, hC * sizeof(__half));
                cudaMemset(hCr, 0, hC * sizeof(__half));
                cudaMemset(hCi, 0, hC * sizeof(__half));
                auto status = gemm.herk_planar(t_Ar, t_Ai, hCr, hCi,
                                                hN, hK, 1.0f, 0.0f,
                                                HerkOp::NoTrans, FillMode::Lower,
                                                HerkStrategy::Baseline, prec);
                std::cout << "    herk_planar(Baseline): "
                          << (status == cutlass::Status::kSuccess ? "PASS" : "SKIPPED") << "\n";
                cudaFree(hCr); cudaFree(hCi);
            }

            // 6. HERK Triangle via herk_planar() — validate against Baseline
            {
                int hN = test_M, hK = test_K;
                int64_t hC = static_cast<int64_t>(hN) * hN;
                __half *hCr_bl, *hCi_bl, *hCr_tr, *hCi_tr;
                cudaMalloc(&hCr_bl, hC * sizeof(__half));
                cudaMalloc(&hCi_bl, hC * sizeof(__half));
                cudaMalloc(&hCr_tr, hC * sizeof(__half));
                cudaMalloc(&hCi_tr, hC * sizeof(__half));
                cudaMemset(hCr_bl, 0, hC * sizeof(__half));
                cudaMemset(hCi_bl, 0, hC * sizeof(__half));
                cudaMemset(hCr_tr, 0, hC * sizeof(__half));
                cudaMemset(hCi_tr, 0, hC * sizeof(__half));

                auto s_bl = gemm.herk_planar(t_Ar, t_Ai, hCr_bl, hCi_bl,
                                              hN, hK, 1.0f, 0.0f,
                                              HerkOp::NoTrans, FillMode::Lower,
                                              HerkStrategy::Baseline, prec);
                auto s_tr = gemm.herk_planar(t_Ar, t_Ai, hCr_tr, hCi_tr,
                                              hN, hK, 1.0f, 0.0f,
                                              HerkOp::NoTrans, FillMode::Lower,
                                              HerkStrategy::TriangleAware, prec);

                if (s_bl == cutlass::Status::kSuccess && s_tr == cutlass::Status::kSuccess) {
                    // Compare lower triangle
                    std::vector<__half> h_bl_r(hC), h_bl_i(hC), h_tr_r(hC), h_tr_i(hC);
                    cudaMemcpy(h_bl_r.data(), hCr_bl, hC * sizeof(__half), cudaMemcpyDeviceToHost);
                    cudaMemcpy(h_bl_i.data(), hCi_bl, hC * sizeof(__half), cudaMemcpyDeviceToHost);
                    cudaMemcpy(h_tr_r.data(), hCr_tr, hC * sizeof(__half), cudaMemcpyDeviceToHost);
                    cudaMemcpy(h_tr_i.data(), hCi_tr, hC * sizeof(__half), cudaMemcpyDeviceToHost);

                    float max_err = 0.0f;
                    for (int i = 0; i < hN; ++i) {
                        for (int j = 0; j <= i; ++j) {
                            int64_t ij = static_cast<int64_t>(i) * hN + j;
                            float dr = fabsf(__half2float(h_tr_r[ij]) - __half2float(h_bl_r[ij]));
                            float di = fabsf(__half2float(h_tr_i[ij]) - __half2float(h_bl_i[ij]));
                            max_err = fmaxf(max_err, fmaxf(dr, di));
                        }
                    }
                    float tol = tolerance_for(prec);
                    std::cout << "    herk_planar(Triangle): "
                              << (max_err < tol ? "PASS" : "FAIL")
                              << "  (vs Baseline, max err = " << max_err << ")\n";
                } else {
                    std::cout << "    herk_planar(Triangle): "
                              << (s_tr == cutlass::Status::kSuccess ? "PASS" : "SKIPPED") << "\n";
                }

                cudaFree(hCr_bl); cudaFree(hCi_bl);
                cudaFree(hCr_tr); cudaFree(hCi_tr);
            }

            // 7. HERK packed via HERK() BLAS3 — Baseline vs Triangle
            {
                int hN = test_M, hK = test_K;
                int64_t packed_elems = static_cast<int64_t>(hN) * (hN + 1) / 2;
                __half *hC_bl, *hC_tr;
                cudaMalloc(&hC_bl, 2 * packed_elems * sizeof(__half));
                cudaMalloc(&hC_tr, 2 * packed_elems * sizeof(__half));
                cudaMemset(hC_bl, 0, 2 * packed_elems * sizeof(__half));
                cudaMemset(hC_tr, 0, 2 * packed_elems * sizeof(__half));

                // Create interleaved A from planar
                __half *d_iA;
                cudaMalloc(&d_iA, 2 * test_sA * sizeof(__half));
                GemmComplexSm100::interleave_complex(t_Ar, t_Ai, d_iA, test_sA, nullptr);

                CutlassParams bp_bl, bp_tr;
                bp_bl.precision = prec;
                bp_bl.herk_strategy = HerkStrategy::Baseline;
                bp_tr.precision = prec;
                bp_tr.herk_strategy = HerkStrategy::TriangleAware;

                auto s_bl = gemm.HERK(d_iA, hC_bl, hN, hK, 1.0f, 0.0f,
                                       HerkOp::NoTrans, FillMode::Lower, bp_bl);
                auto s_tr = gemm.HERK(d_iA, hC_tr, hN, hK, 1.0f, 0.0f,
                                       HerkOp::NoTrans, FillMode::Lower, bp_tr);

                if (s_bl == cutlass::Status::kSuccess && s_tr == cutlass::Status::kSuccess) {
                    std::vector<__half> h_pbl(2 * packed_elems), h_ptr(2 * packed_elems);
                    cudaMemcpy(h_pbl.data(), hC_bl, 2 * packed_elems * sizeof(__half),
                               cudaMemcpyDeviceToHost);
                    cudaMemcpy(h_ptr.data(), hC_tr, 2 * packed_elems * sizeof(__half),
                               cudaMemcpyDeviceToHost);

                    float max_err = 0.0f;
                    for (int64_t i = 0; i < 2 * packed_elems; ++i) {
                        float diff = fabsf(__half2float(h_ptr[i]) - __half2float(h_pbl[i]));
                        max_err = fmaxf(max_err, diff);
                    }
                    float tol = tolerance_for(prec);
                    std::cout << "    HERK packed(BL vs Tri): "
                              << (max_err < tol ? "PASS" : "FAIL")
                              << "  (max err = " << max_err << ")\n";
                } else {
                    std::cout << "    HERK packed(Baseline): "
                              << (s_bl == cutlass::Status::kSuccess ? "PASS" : "SKIPPED")
                              << "  HERK packed(Triangle): "
                              << (s_tr == cutlass::Status::kSuccess ? "PASS" : "SKIPPED") << "\n";
                }

                cudaFree(d_iA);
                cudaFree(hC_bl); cudaFree(hC_tr);
            }

            std::cout << "\n";
        };

        std::cout << "Multi-precision test: " << test_M << "x" << test_N << "x" << test_K
                  << " (all API paths vs FP8 reference)\n\n";

#ifdef COMPLEX_SM100_ENABLE_FP6
        if (test_K >= COMPLEX_SM100_FP6_MMA_K) {
            test_precision("FP6 E3M2 (range +-28, ~0.5 precision)", ComputePrecision::FP6_E3M2);
            test_precision("FP6 E2M3 (range +-7.5, ~0.125 precision)", ComputePrecision::FP6_E2M3);
        } else {
            printf("  --- FP6: SKIPPED (K=%d < tile_K=%d) ---\n",
                   test_K, COMPLEX_SM100_FP6_MMA_K);
        }
#endif  // COMPLEX_SM100_ENABLE_FP6

#ifdef COMPLEX_SM100_ENABLE_FP4
        if (test_K >= COMPLEX_SM100_FP4_MMA_K) {
            test_precision("FP4 E2M1 (range +-6, ~1.0 precision)", ComputePrecision::FP4_E2M1);
        } else {
            printf("  --- FP4: SKIPPED (K=%d < tile_K=%d) ---\n",
                   test_K, COMPLEX_SM100_FP4_MMA_K);
        }
#endif  // COMPLEX_SM100_ENABLE_FP4

        cudaFree(t_Ar); cudaFree(t_Ai);
        cudaFree(t_Br); cudaFree(t_Bi);
        cudaFree(t_Cr_ref); cudaFree(t_Ci_ref);
        cudaFree(t_Cr); cudaFree(t_Ci);
    }
#endif  // COMPLEX_SM100_ENABLE_FP6 || COMPLEX_SM100_ENABLE_FP4


    // ====================================================================
    // PRODUCTION PHASE — clean benchmarks, no validation
    // ====================================================================
    if (run_bench) {
    std::cout << "\n"
              << "╔══════════════════════════════════════════════════════╗\n"
              << "║                 PRODUCTION PHASE                    ║\n"
              << "╚══════════════════════════════════════════════════════╝\n\n";

    // Benchmark lambda — runs all operations for a given precision
    // Uses baseline FLOPs (herk_sub_gemms) as denominator for all HERK variants
    // so that TFLOPS directly reflects throughput improvement.
    auto run_all_benchmarks = [&](ComputePrecision p) {
        const char* pname = precision_name(p);
        const char* pshort = precision_short(p);
        double peak = peak_tflops_for_precision(p, sysinfo);
        int herk_sub_gemms = 3;
#if defined(COMPLEX_FP8_HERK_FULL_MATRIX) && COMPLEX_FP8_HERK_FULL_MATRIX
        herk_sub_gemms = 4;
#endif

        printf("\n--- %s ---\n", pname);
        printf("  %-18s %11s  %13s  %9s  %s\n", "Operation", "Time", "TFLOPS", "GB/s", "IO");

        // Standard GEMM: C = A*B (4 sub-GEMMs)
        {
            CutlassParams bp;
            bp.precision = p;
            auto [ms, status] = benchmark_ms_checked([&]() {
                return gemm.GEMM(d_A, d_B, d_C, M, N, K, 1.0f, 0.0f, ComplexMode::Standard, bp);
            }, warmup, iters);
            if (status == cutlass::Status::kSuccess) {
                double eb = cutlass_complex::gemm_external_bytes(M, N, K);
                print_result("Standard", M, N, K, ms, 4, 1, 0.0, 0, eb);
                roofline.push_back(make_roofline(
                    (std::string(pshort) + " Standard").c_str(),
                    ms, 4, M, N, K, 1, eb, peak));
            } else {
                printf("  %-18s  SKIPPED (launch status=%d)\n", "Standard", (int)status);
            }
        }

        // Hermitian GEMM: C = A*B^H (4 sub-GEMMs)
        {
            CutlassParams bp;
            bp.precision = p;
            auto [ms, status] = benchmark_ms_checked([&]() {
                return gemm.GEMM(d_A, d_B, d_C, M, N, K, 1.0f, 0.0f, ComplexMode::Hermitian, bp);
            }, warmup, iters);
            if (status == cutlass::Status::kSuccess) {
                double eb = cutlass_complex::gemm_external_bytes(M, N, K);
                print_result("Hermitian", M, N, K, ms, 4, 1, 0.0, 0, eb);
                roofline.push_back(make_roofline(
                    (std::string(pshort) + " Hermitian").c_str(),
                    ms, 4, M, N, K, 1, eb, peak));
            } else {
                printf("  %-18s  SKIPPED (launch status=%d)\n", "Hermitian", (int)status);
            }
        }

        // Gram A*A^H + pack (4 sub-GEMMs)
        {
            __half *gCr, *gCi, *gPr, *gPi;
            int64_t packed_size = (int64_t)M * (M + 1) / 2;
            cudaMalloc(&gCr, herk_C_size * sizeof(__half));
            cudaMalloc(&gCi, herk_C_size * sizeof(__half));
            cudaMalloc(&gPr, packed_size * sizeof(__half));
            cudaMalloc(&gPi, packed_size * sizeof(__half));

            auto [ms, status] = benchmark_ms_checked([&]() {
                auto s = gemm.run_gram_planar(d_Ar, d_Ai, gCr, gCi,
                                     M, K, 1.0f, 0.0f, GramMode::AAH, p);
                if (s == cutlass::Status::kSuccess) {
                    pack_lower_triangle(gCr, gPr, M);
                    pack_lower_triangle(gCi, gPi, M);
                }
                return s;
            }, warmup, iters);
            if (status == cutlass::Status::kSuccess) {
                double eb = cutlass_complex::herk_external_bytes(M, K);
                print_result("Gram A*A^H", M, M, K, ms, 4, 1, 0.0, 0, eb);
                roofline.push_back(make_roofline(
                    (std::string(pshort) + " Gram").c_str(),
                    ms, 4, M, M, K, 1, eb, peak));
            } else {
                printf("  %-18s  SKIPPED (launch status=%d)\n", "Gram A*A^H", (int)status);
            }

            cudaFree(gCr); cudaFree(gCi);
            cudaFree(gPr); cudaFree(gPi);
        }

        // HERK Baseline (3 sub-GEMMs, packed output)
        double herk_baseline_ms = -1.0;
        {
            int64_t packed_size = (int64_t)M * (M + 1) / 2;
            __half *hC;
            cudaMalloc(&hC, 2 * packed_size * sizeof(__half));

            CutlassParams bp;
            bp.precision = p;
            double herk_eb = cutlass_complex::herk_external_bytes(M, K);
            auto [ms, status] = benchmark_ms_checked([&]() {
                return gemm.HERK(d_A, hC, M, K, 1.0f, 0.0f,
                          HerkOp::NoTrans, FillMode::Lower, bp);
            }, warmup, iters);
            if (status == cutlass::Status::kSuccess) {
                herk_baseline_ms = ms;
                print_result("HERK Baseline", M, M, K, herk_baseline_ms, herk_sub_gemms,
                             1, 0.0, 0, herk_eb);
                roofline.push_back(make_roofline(
                    (std::string(pshort) + " HERK BL").c_str(),
                    ms, herk_sub_gemms, M, M, K, 1, herk_eb, peak));
            } else {
                printf("  %-18s  SKIPPED (launch status=%d)\n", "HERK Baseline", (int)status);
            }

            cudaFree(hC);
        }

        // HERK Triangle (packed output, baseline FLOPs for comparable TFLOPS)
        int slabs = compute_triangle_slabs(M, sm_count, tri);
        {
            int64_t packed_size = (int64_t)M * (M + 1) / 2;
            __half *hC;
            cudaMalloc(&hC, 2 * packed_size * sizeof(__half));

            CutlassParams bp;
            bp.herk_strategy = HerkStrategy::TriangleAware;
            bp.precision = p;
            bp.triangle_config = tri;
            double herk_eb = cutlass_complex::herk_external_bytes(M, K);
            auto [ms, status] = benchmark_ms_checked([&]() {
                return gemm.HERK(d_A, hC, M, K, 1.0f, 0.0f,
                          HerkOp::NoTrans, FillMode::Lower, bp);
            }, warmup, iters);
            if (status == cutlass::Status::kSuccess) {
                // Use baseline FLOPs so TFLOPS reflects speedup directly
                print_result("HERK Triangle", M, M, K, ms, herk_sub_gemms, 1,
                             herk_baseline_ms > 0 ? herk_baseline_ms : 0.0, slabs, herk_eb);
                roofline.push_back(make_roofline(
                    (std::string(pshort) + " HERK Tri").c_str(),
                    ms, herk_sub_gemms, M, M, K, 1, herk_eb, peak));
            } else {
                printf("  %-18s  SKIPPED (launch status=%d)\n", "HERK Triangle", (int)status);
            }

            cudaFree(hC);
        }

        // HERK Tri+Graph (SM120 CUDA graph path)
#ifdef COMPLEX_FP8_SM100_TARGET_SM120
        {
            int64_t packed_size = (int64_t)M * (M + 1) / 2;
            __half *hC;
            cudaMalloc(&hC, 2 * packed_size * sizeof(__half));

            CutlassParams bp;
            bp.herk_strategy = HerkStrategy::TriangleAware;
            bp.precision = p;
            bp.triangle_config = tri;
            bp.triangle_config.use_cuda_graph = true;
            double herk_eb = cutlass_complex::herk_external_bytes(M, K);
            auto [ms, status] = benchmark_ms_checked([&]() {
                return gemm.HERK(d_A, hC, M, K, 1.0f, 0.0f,
                          HerkOp::NoTrans, FillMode::Lower, bp);
            }, warmup, iters);
            if (status == cutlass::Status::kSuccess) {
                print_result("HERK Tri+Graph", M, M, K, ms, herk_sub_gemms, 1,
                             herk_baseline_ms > 0 ? herk_baseline_ms : 0.0, slabs, herk_eb);
                roofline.push_back(make_roofline(
                    (std::string(pshort) + " HERK Tri+Grph").c_str(),
                    ms, herk_sub_gemms, M, M, K, 1, herk_eb, peak));
            } else {
                printf("  %-18s  SKIPPED (launch status=%d)\n", "HERK Tri+Graph", (int)status);
            }

            cudaFree(hC);
        }
#endif
    };

    // Run benchmarks for primary precision
    if (K >= minimum_k_for_precision(precision)) {
        run_all_benchmarks(precision);
    } else {
        printf("\n--- %s: SKIPPED (K=%d < tile_K=%d) ---\n",
               precision_name(precision), K, minimum_k_for_precision(precision));
    }

    // Run benchmarks for additional compiled-in precisions
#ifdef COMPLEX_SM100_ENABLE_FP6
    if (K >= COMPLEX_SM100_FP6_MMA_K) {
        if (precision != ComputePrecision::FP6_E3M2)
            run_all_benchmarks(ComputePrecision::FP6_E3M2);
        if (precision != ComputePrecision::FP6_E2M3)
            run_all_benchmarks(ComputePrecision::FP6_E2M3);
    } else {
        printf("\n--- FP6: SKIPPED (K=%d < tile_K=%d) ---\n",
               K, COMPLEX_SM100_FP6_MMA_K);
    }
#endif
#ifdef COMPLEX_SM100_ENABLE_FP4
    if (K >= COMPLEX_SM100_FP4_MMA_K) {
        if (precision != ComputePrecision::FP4_E2M1)
            run_all_benchmarks(ComputePrecision::FP4_E2M1);
    } else {
        printf("\n--- FP4: SKIPPED (K=%d < tile_K=%d) ---\n",
               K, COMPLEX_SM100_FP4_MMA_K);
    }
#endif


    // ---- Batched HERK Benchmark (all compiled precisions) ----
    {
        const int batch_count = (batch > 1) ? batch : 4;
        int64_t packed_size = static_cast<int64_t>(M) * (M + 1) / 2;
        int64_t per_A = static_cast<int64_t>(M) * K;

        __half* d_A_batched;
        cudaMalloc(&d_A_batched, 2 * per_A * batch_count * sizeof(__half));
        for (int b = 0; b < batch_count; ++b) {
            cudaMemcpy(d_A_batched + b * 2 * per_A, d_A,
                       2 * per_A * sizeof(__half), cudaMemcpyDeviceToDevice);
        }

        __half* hC;
        cudaMalloc(&hC, 2 * packed_size * batch_count * sizeof(__half));

        auto run_batched_bench = [&](ComputePrecision p) {
            double gflops_base = 3.0 * batch_count * 2.0 * M * M * K / 1e9;
            double batched_eb = cutlass_complex::herk_external_bytes(M, K, batch_count);
            int slabs = compute_triangle_slabs(M, sm_count, tri);

            // Select peak TFLOPS for this precision
            double peak_tflops = sysinfo.peak_fp8_tflops;
            if (p == ComputePrecision::FP6_E3M2 || p == ComputePrecision::FP6_E2M3)
                peak_tflops = sysinfo.peak_fp6_tflops;
            else if (p == ComputePrecision::FP4_E2M1)
                peak_tflops = sysinfo.peak_fp4_tflops;

            // Utilization sub-line printer
            // Shows TC%, external I/O (user buffers), and total DRAM estimate
            // (includes deinterleave, FP8/MXFP cast, scratch C, pack, interleave).
            double internal_bytes = cutlass_complex::herk_internal_bytes(M, K, batch_count);
            double external_bytes = batched_eb;
            auto print_util = [&](double ms) {
                double time_s = ms * 1e-3;
                double achieved = gflops_base / ms;  // TFLOPS
                double tc_pct = (peak_tflops > 0) ? achieved / peak_tflops * 100.0 : 0;
                double ext_gbs = external_bytes / time_s / 1e9;
                double total_gbs = (external_bytes + internal_bytes) / time_s / 1e9;
                double ext_pct = (sysinfo.memory_bw_gbs > 0) ? ext_gbs / sysinfo.memory_bw_gbs * 100 : 0;
                double total_pct = (sysinfo.memory_bw_gbs > 0) ? total_gbs / sysinfo.memory_bw_gbs * 100 : 0;
                printf("    Util: %4.1f%% TC | ext %5.1f GB/s (%4.1f%%) | total %5.1f GB/s (%4.1f%%)\n",
                       tc_pct, ext_gbs, ext_pct, total_gbs, total_pct);
            };

            // Tile status in header
            int tile_display = batch_tiling
                ? cutlass_complex::compute_herk_batch_tile(M, batch_count, sysinfo.l2_cache_bytes)
                : 0;
            if (tile_display > 0)
                printf("\n--- Batched HERK (batch=%d, %s, tile=%d) ---\n", batch_count, precision_short(p), tile_display);
            else
                printf("\n--- Batched HERK (batch=%d, %s, tile=off) ---\n", batch_count, precision_short(p));

            // Baseline
            double bl_ms = -1.0;
            {
                CutlassParams bp;
                bp.herk_strategy = HerkStrategy::Baseline;
                bp.precision = p;
                auto [ms, status] = benchmark_ms_checked([&]() {
                    return gemm.HERK_batched(d_A_batched, hC, M, K, batch_count,
                                      1.0f, 0.0f, HerkOp::NoTrans, FillMode::Lower, bp);
                }, warmup, iters);
                if (status == cutlass::Status::kSuccess) {
                    bl_ms = ms;
                    double tflops = gflops_base / bl_ms;
                    double gbs = batched_eb / (bl_ms * 1e-3) / 1e9;
                    double io = (gflops_base * 1e9) / batched_eb;
                    double per_item = bl_ms / batch_count;
                    printf("  Batched Baseline:     %7.2f ms  (%6.1f TFLOPS, %5.1f GB/s, IO=%.0f, %.3f ms/item)\n",
                           bl_ms, tflops, gbs, io, per_item);
                    print_util(bl_ms);
                    roofline.push_back(make_roofline(
                        (std::string(precision_short(p)) + " Bat BL").c_str(),
                        bl_ms, 3, M, M, K, batch_count, batched_eb, peak_tflops));
                } else {
                    printf("  Batched Baseline:      SKIPPED (launch status=%d)\n", (int)status);
                }
            }

            // TriangleAware
            {
                CutlassParams tp;
                tp.herk_strategy = HerkStrategy::TriangleAware;
                tp.precision = p;
                tp.triangle_config = tri;
                auto [ms, status] = benchmark_ms_checked([&]() {
                    return gemm.HERK_batched(d_A_batched, hC, M, K, batch_count,
                                      1.0f, 0.0f, HerkOp::NoTrans, FillMode::Lower, tp);
                }, warmup, iters);
                if (status == cutlass::Status::kSuccess) {
                    double speedup = (bl_ms > 0) ? bl_ms / ms : 0;
                    double tflops = gflops_base / ms;
                    double gbs = batched_eb / (ms * 1e-3) / 1e9;
                    double io = (gflops_base * 1e9) / batched_eb;
                    double per_item = ms / batch_count;
                    if (bl_ms > 0) {
                        printf("  Batched Triangle:     %7.2f ms  (%6.1f TFLOPS, %5.1f GB/s, IO=%.0f, %.3f ms/item, %.2fx)\n",
                               ms, tflops, gbs, io, per_item, speedup);
                    } else {
                        printf("  Batched Triangle:     %7.2f ms  (%6.1f TFLOPS, %5.1f GB/s, IO=%.0f, %.3f ms/item)\n",
                               ms, tflops, gbs, io, per_item);
                    }
                    print_util(ms);
                    roofline.push_back(make_roofline(
                        (std::string(precision_short(p)) + " Bat Tri").c_str(),
                        ms, 3, M, M, K, batch_count, batched_eb, peak_tflops));
                } else {
                    printf("  Batched Triangle:      SKIPPED (launch status=%d)\n", (int)status);
                }
            }

            // Tri+Graph (SM120 only)
#ifdef COMPLEX_FP8_SM100_TARGET_SM120
            {
                CutlassParams tp;
                tp.herk_strategy = HerkStrategy::TriangleAware;
                tp.precision = p;
                tp.triangle_config = tri;
                tp.triangle_config.use_cuda_graph = true;
                auto [ms, status] = benchmark_ms_checked([&]() {
                    return gemm.HERK_batched(d_A_batched, hC, M, K, batch_count,
                                      1.0f, 0.0f, HerkOp::NoTrans, FillMode::Lower, tp);
                }, warmup, iters);
                if (status == cutlass::Status::kSuccess) {
                    double speedup = (bl_ms > 0) ? bl_ms / ms : 0;
                    double tflops = gflops_base / ms;
                    double gbs = batched_eb / (ms * 1e-3) / 1e9;
                    double io = (gflops_base * 1e9) / batched_eb;
                    double per_item = ms / batch_count;
                    if (bl_ms > 0) {
                        printf("  Batched Tri+Graph:    %7.2f ms  (%6.1f TFLOPS, %5.1f GB/s, IO=%.0f, %.3f ms/item, %.2fx)\n",
                               ms, tflops, gbs, io, per_item, speedup);
                    } else {
                        printf("  Batched Tri+Graph:    %7.2f ms  (%6.1f TFLOPS, %5.1f GB/s, IO=%.0f, %.3f ms/item)\n",
                               ms, tflops, gbs, io, per_item);
                    }
                    print_util(ms);
                    roofline.push_back(make_roofline(
                        (std::string(precision_short(p)) + " Bat Tri+Grph").c_str(),
                        ms, 3, M, M, K, batch_count, batched_eb, peak_tflops));
                } else {
                    printf("  Batched Tri+Graph:     SKIPPED (launch status=%d)\n", (int)status);
                }
            }
#endif
        };

        // Run for primary precision
        if (K >= minimum_k_for_precision(precision))
            run_batched_bench(precision);

        // Additional precisions
#ifdef COMPLEX_SM100_ENABLE_FP6
        if (K >= COMPLEX_SM100_FP6_MMA_K) {
            if (precision != ComputePrecision::FP6_E3M2)
                run_batched_bench(ComputePrecision::FP6_E3M2);
            if (precision != ComputePrecision::FP6_E2M3)
                run_batched_bench(ComputePrecision::FP6_E2M3);
        }
#endif
#ifdef COMPLEX_SM100_ENABLE_FP4
        if (K >= COMPLEX_SM100_FP4_MMA_K) {
            if (precision != ComputePrecision::FP4_E2M1)
                run_batched_bench(ComputePrecision::FP4_E2M1);
        }
#endif

        cudaFree(d_A_batched);
        cudaFree(hC);
    }

    // ---- Production GEMM Benchmarks ----
    // Tests batched planar complex GEMM at beamformer and dedisp shapes.
    // Uses FP8 compute, FP32 output via the planar batched API.
    if (is_fp8 && fp8_baseline_ok) {
        printf("\n--- Production GEMM Benchmarks ---\n");
        printf("  Complex GEMM via 4M decomposition, FP8 compute, FP32 output\n");
        printf("  %-18s %11s  %13s  %9s  %7s\n", "Problem", "Time", "TFLOPS", "GB/s", "ms/item");

        struct GemmProblem {
            const char* name;
            int M, N, K, batch;
        };

        // Voltage BF: M=n_time, N=n_beam, K=n_ant, batch=n_ch
        // Dedisp: M=Nt_complex, N=n_dm, K=n_ch, batch=n_beam
        std::vector<GemmProblem> gemm_problems = {
            // Standard VoltBF problems (n_time=4000, long integration)
            {"VoltBF 4000x4000x1664 b1",   4000, 4000, 1664,   1},
            {"VoltBF 4000x4000x1664 b4",   4000, 4000, 1664,   4},
            {"VoltBF 4000x4000x1664 b16",  4000, 4000, 1664,  16},
            {"VoltBF 4000x4000x1664 b32",  4000, 4000, 1664,  32},
            // Short-integration VoltBF (n_time=8, unfused baseline)
            {"VoltBF-short 8x4000x1664 b1600",   8, 4000, 1664, 1600},
            // Short-integration with ch_fuse=8 (M=64, batch=200)
            {"VoltBF-fused 64x4000x1664 b200",  64, 4000, 1664,  200},
            // Short-integration ch_fuse=8 x 8 payloads (M=512, batch=200)
            {"VoltBF-batch 512x4000x1664 b200", 512, 4000, 1664,  200},
            // Dedisp problems
            {"Dedisp 257x512x128 b128",     257,  512,  128, 128},
            {"Dedisp 257x2048x512 b128",    257, 2048,  512, 128},
            {"Dedisp 257x4096x1024 b128",   257, 4096, 1024, 128},
        };

        // Add CLI problem if it doesn't duplicate an existing one
        if (batch > 0) {
            bool found = false;
            for (auto& p : gemm_problems)
                if (p.M == M && p.N == N && p.K == K && p.batch == batch) { found = true; break; }
            if (!found) {
                char cli_name[64];
                snprintf(cli_name, sizeof(cli_name), "CLI %dx%dx%d b%d", M, N, K, batch);
                gemm_problems.push_back({strdup(cli_name), M, N, K, batch});
            }
        }

        int n_gemm_problems = (int)gemm_problems.size();

        // Set GEMM strategy verbosity for autotune
        if (tune_level == -1 && strat_verb < 0)
            strategy_cache::GemmStrategyCache::instance().set_verbosity(2);

        for (int gi = 0; gi < n_gemm_problems; ++gi) {
            auto& gp = gemm_problems[gi];

            // Check memory
            int64_t need_A = static_cast<int64_t>(gp.M) * gp.K * gp.batch;
            int64_t need_B = static_cast<int64_t>(gp.N) * gp.K * gp.batch;
            int64_t need_C = static_cast<int64_t>(gp.M) * gp.N * gp.batch;
            int64_t total_bytes = (need_A + need_B) * 2 * sizeof(__half) +
                                  need_C * 2 * sizeof(float) +
                                  (need_A + need_B) * 2;  // FP8 scratch estimate
            size_t free_mem, total_mem;
            cudaMemGetInfo(&free_mem, &total_mem);
            if ((double)total_bytes > 0.90 * free_mem) {
                printf("  %-40s  SKIPPED (need %.1f GB, have %.1f GB free)\n",
                       gp.name, total_bytes / 1e9, free_mem / 1e9);
                continue;
            }

            __half *gAr, *gAi, *gBr, *gBi;
            float *gCr, *gCi;
            cudaMalloc(&gAr, need_A * sizeof(__half));
            cudaMalloc(&gAi, need_A * sizeof(__half));
            cudaMalloc(&gBr, need_B * sizeof(__half));
            cudaMalloc(&gBi, need_B * sizeof(__half));
            cudaMalloc(&gCr, need_C * sizeof(float));
            cudaMalloc(&gCi, need_C * sizeof(float));
            cudaMemset(gAr, 0, need_A * sizeof(__half));
            cudaMemset(gAi, 0, need_A * sizeof(__half));
            cudaMemset(gBr, 0, need_B * sizeof(__half));
            cudaMemset(gBi, 0, need_B * sizeof(__half));

            // Run GEMM autotuning if tune=true
            if (tune_level == -1) {
                auto all_cfgs = all_baseline_configs();
                std::vector<int> cfg_ints;
                for (auto c : all_cfgs) cfg_ints.push_back(static_cast<int>(c));

                auto gentry = strategy_cache::run_gemm_autotune<
                    GemmComplexSm100, GemmConfig, ComplexMode, ComputePrecision>(
                        gemm, gp.M, gp.N, gp.K, gp.batch, 0 /*FP8*/, nullptr, cfg_ints);

                strategy_cache::apply_gemm_cached_settings<GemmComplexSm100, GemmConfig>(
                    gemm, gentry);

                printf("  [%s] Autotuned: cfg=%d(%s) %.1f TFLOPS\n",
                       gp.name, gentry.gemm_config,
                       config_name(static_cast<GemmConfig>(gentry.gemm_config)),
                       gentry.tflops);
            }

            // Benchmark
            auto [ms, status] = benchmark_ms_checked([&]() {
                return gemm.run_planar_batched_fp32out(
                    gAr, gAi, gBr, gBi, gCr, gCi,
                    gp.M, gp.N, gp.K, gp.batch,
                    1.0f, 0.0f, ComplexMode::Standard, nullptr);
            }, warmup, iters);

            if (status == cutlass::Status::kSuccess) {
                double flops = 8.0 * gp.M * gp.N * gp.K * gp.batch;
                double tflops = flops / (ms * 1e-3) / 1e12;
                double ext_bytes = (double)gp.batch * (2.0 * gp.M * gp.K * 2 +
                                                        2.0 * gp.N * gp.K * 2 +
                                                        2.0 * gp.M * gp.N * 4);
                double gbs = ext_bytes / (ms * 1e-3) / 1e9;
                double per_item = ms / gp.batch;
                printf("  %-40s %8.2f ms  %6.1f TFLOPS  %7.1f GB/s  %.3f ms/item\n",
                       gp.name, ms, tflops, gbs, per_item);

                roofline.push_back(make_roofline(
                    gp.name, ms, 4, gp.M, gp.N, gp.K, gp.batch,
                    ext_bytes, sysinfo.peak_fp8_tflops));
            } else {
                printf("  %-40s  SKIPPED (status=%d)\n", gp.name, (int)status);
            }

            cudaFree(gAr); cudaFree(gAi);
            cudaFree(gBr); cudaFree(gBi);
            cudaFree(gCr); cudaFree(gCi);
        }
    }

    } // end run_bench

    // ====================================================================
    // Summary
    // ====================================================================
    printf("\n--- Summary ---\n");
    printf("Theoretical dense peak (non-sparse, %d SMs x %d MHz):\n",
           sysinfo.sm_count, sysinfo.clock_mhz);
    printf("  FP8:  ~%6.1f TFLOPS  (2048 ops/cycle/SM)\n", sysinfo.peak_fp8_tflops);
    printf("  FP6:  ~%6.1f TFLOPS  (2048 ops/cycle/SM, same as FP8)\n", sysinfo.peak_fp6_tflops);
    printf("  FP4:  ~%6.1f TFLOPS  (4096 ops/cycle/SM, 2x FP8)\n", sysinfo.peak_fp4_tflops);
    printf("GEMM: 4 sub-GEMMs (4M algorithm), HERK Baseline: 3, HERK Triangle: 2 tri + 1 full\n");
    printf("All HERK/Gram output packed lower triangle, N*(N+1)/2 complex elements\n");
    printf("TFLOPS = baseline FLOPs / wall-clock time (all HERK variants use baseline FLOPs)\n");

    // ---- Roofline ----
    print_roofline_table(roofline, sysinfo);

    // ---- Small-K Analysis ----
    if (K < M && run_bench) {
        int bc = (batch > 1) ? batch : 4;
        double ext = cutlass_complex::herk_external_bytes(M, K, bc);
        double intl = cutlass_complex::herk_internal_bytes(M, K, bc);
        double flops_total = 3.0 * bc * 2.0 * M * M * K;
        double ext_io = flops_total / ext;
        double int_io = flops_total / intl;
        double ridge = cutlass_complex::ridge_point(sysinfo.peak_fp8_tflops, sysinfo.memory_bw_gbs);

        double input_gb = 4.0 * bc * M * K / 1e9;
        double output_gb = 2.0 * bc * M * ((int64_t)M + 1) / 1e9;
        double scratch_gb = 2.0 * bc * (double)M * M * sizeof(__half) / 1e9;

        printf("\n--- Small-K Analysis (K=%d, N=%d, batch=%d) ---\n", K, M, bc);
        printf("External I/O:  %5.1f GB  (read %.1f GB input + write %.1f GB packed output)\n",
               ext / 1e9, input_gb, output_gb);
        printf("Internal I/O: ~%.0f GB  (scratch N^2 x batch dominates: 2 x %.1f GB)\n",
               intl / 1e9, scratch_gb);
        printf("Output/input ratio: %d:1 (N/K) -- output-dominated traffic\n", M / K);
        printf("\n");
        printf("External IO: %.0f FLOPs/byte (%s ridge %.0f -> %s)\n",
               ext_io, ext_io >= ridge ? ">=" : "<", ridge,
               ext_io >= ridge ? "appears compute-bound" : "memory-bound");
        printf("Internal IO: %.0f FLOPs/byte (%s ridge %.0f -> %s)\n",
               int_io, int_io >= ridge ? ">=" : "<", ridge,
               int_io >= ridge ? "compute-bound" : "actually memory-bound");
        printf("\nBottleneck: 2x N^2 x batch scratch buffers (%.1f GB each) >> input (%.1f GB)\n",
               scratch_gb, input_gb);
        printf("Each batch produces N^2=%lldM element scratch but only reads N*K=%lldK elements.\n",
               (long long)((int64_t)M * M / 1000000),
               (long long)((int64_t)M * K / 1000));
        int bt = cutlass_complex::compute_herk_batch_tile(M, bc, sysinfo.l2_cache_bytes);
        if (bt > 0 && bt < bc) {
            double tiled_scratch_gb = 2.0 * bt * (double)M * M * sizeof(__half) / 1e9;
            double tiled_intl = cutlass_complex::herk_internal_bytes(M, K, bt);
            double tiled_int_io = (3.0 * bt * 2.0 * M * M * K) / tiled_intl;
            printf("\nBatch tiling: active (batch_tile=%d, %d iterations)\n", bt, (bc + bt - 1) / bt);
            printf("  Tiled scratch: %.1f GB (vs %.1f GB full) -- fits in %d MB L2\n",
                   tiled_scratch_gb, scratch_gb, sysinfo.l2_cache_bytes / (1024*1024));
            printf("  Tiled internal IO: %.0f FLOPs/byte (%s ridge %.0f)\n",
                   tiled_int_io, tiled_int_io >= ridge ? ">=" : "<", ridge);
        } else {
            printf("\nBatch tiling: not needed (scratch fits in L2)\n");
        }
        printf("Already active: stacked-K, 2-stream overlap, CUDA graph\n");
    }

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    cudaFree(d_Ar); cudaFree(d_Ai);
    cudaFree(d_Br); cudaFree(d_Bi);
    cudaFree(d_Cr); cudaFree(d_Ci);

    return 0;
}
