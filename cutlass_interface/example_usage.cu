/*
 * example_usage.cu
 *
 * Complex FP8 GEMM benchmark on NVIDIA Hopper (SM90).
 *
 * Two-phase execution:
 *   TEST PHASE       — validates HERK output against Gram reference,
 *                      checks Hermitian properties (real diagonal,
 *                      conjugate symmetry, lower-triangle agreement)
 *   PRODUCTION PHASE — clean benchmarks, no validation overhead
 *
 * Usage:
 *   ./example_complex_fp8 [M] [N] [K] [batch] [key=value ...]
 *   ./example_complex_fp8                        # defaults to 4096³
 *   ./example_complex_fp8 8192 8192 8192
 *   ./example_complex_fp8 512 512 512 64         # 64 batched 512³
 *   ./example_complex_fp8 4096 mode=bench        # benchmarks only
 *   ./example_complex_fp8 4096 mode=test         # tests only
 *
 * Mode options: test, bench, both (default)
 *
 * Compile:
 *   cmake .. -DCUTLASS_DIR=/path/to/cutlass -DCMAKE_CUDA_ARCHITECTURES=90a
 *   make -j$(nproc)
 */

#include "gemm_complex_fp8.hpp"
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
#include <string>
#ifdef _OPENMP
#include <omp.h>
#endif

using namespace gemm_complex_fp8;


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
    printf("Usage: %s [M] [N] [K] [batch] [key=value ...]\n", prog);
    printf("\n");
    printf("Complex FP8 GEMM benchmark on NVIDIA Hopper (SM90).\n");
    printf("Runs test phase (correctness validation) then production phase (benchmarks).\n");
    printf("\n");
    printf("Positional arguments:\n");
    printf("  M N K        Matrix dimensions (default: 4096 4096 4096)\n");
    printf("  batch        Batch count (default: 1)\n");
    printf("\n");
    printf("Shorthand:\n");
    printf("  %s 4096              # M=N=K=4096\n", prog);
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
    printf("  herk_graph=<0|1> Capture baseline HERK as CUDA graph for replay (default: 0)\n");
    printf("  direct=<auto|0|1> Direct HERK mode: auto=K-adaptive, 1=force, 0=off (default: auto)\n");
    printf("  persistent=<auto|0|1> Persistent direct HERK: auto=work-adaptive, 1=force, 0=off (default: auto)\n");
    printf("  tile=<0|1>       Enable batch tiling for L2 scratch reuse (default: 1)\n");
    printf("  tune=<true|false|0|1|2|3>  true=full strategy autotune (default), false=off, 0-3=kernel-level tune verbosity\n");
    printf("  strategy_verbosity=<0|1|2|3>  Strategy autotune output verbosity (default: 1, or 2 when tune=true)\n");
    printf("\n");
    printf("Examples:\n");
    printf("  %s 4096                       # 4096^3, default triangle\n", prog);
    printf("  %s 8192 8192 8192             # large square\n", prog);
    printf("  %s 4096 slabs=8 verbose=1     # explicit 8 slabs, verbose\n", prog);
    printf("  %s 4096 4096 4096 64          # 64 batched 4096^3\n", prog);
    printf("  %s 4096 mode=bench            # benchmarks only, skip tests\n", prog);
    printf("  %s 4096 mode=test             # tests only, skip benchmarks\n", prog);
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

BenchMode parse_bench_mode(int argc, char** argv) {
    for (int i = 1; i < argc; ++i) {
        std::string arg(argv[i]);
        if (arg.substr(0, 5) == "mode=") {
            return parse_mode(arg.substr(5).c_str());
        }
    }
    return BenchMode::Both;
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


GemmMode parse_gemm_mode(int argc, char** argv) {
    for (int i = 1; i < argc; ++i) {
        std::string arg(argv[i]);
        if (arg.substr(0, 10) == "gemm_mode=") {
            std::string val = arg.substr(10);
            if (val == "direct")  return GemmMode::ForceDirect;
            if (val == "4m")      return GemmMode::Force4M;
            return GemmMode::Auto;
        }
    }
    return GemmMode::Auto;
}

// ========================================================================================
// Helpers
// ========================================================================================

void fill_random_fp16(std::vector<__half>& buf, float scale = 1.0f) {
    static std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-scale, scale);
    for (auto& v : buf) v = __float2half(dist(rng));
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

void print_bench(const char* label, double ms, double tflops,
                 double baseline_ms = 0.0, int slabs = 0,
                 double ext_bytes = 0.0) {
    double gbs = 0, ai = 0;
    if (ext_bytes > 0) {
        gbs = ext_bytes / (ms * 1e-3) / 1e9;
        double flops = tflops * 1e12 * ms * 1e-3;
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
    double tc_util;       // % of peak FP8
    double bw_gbs;        // achieved bandwidth GB/s
    double io_intensity;  // FLOPs/byte
};

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
    return e;
}

static void print_roofline_table(const std::vector<RooflineEntry>& entries,
                                 const cutlass_complex::SystemInfo& si) {
    if (entries.empty()) return;
    double ridge = cutlass_complex::ridge_point(si.peak_fp8_tflops, si.memory_bw_gbs);
    printf("\n--- Roofline Analysis ---\n");
    printf("  Peak FP8:      %6.1f TFLOPS\n", si.peak_fp8_tflops);
    printf("  Memory BW:     %6.1f GB/s (theoretical)\n", si.memory_bw_gbs);
    printf("  Ridge point:   %5.0f FLOPs/byte\n", ridge);
    for (const auto& e : entries) {
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


/// Compute theoretical FP8 dense peak for Hopper (SM90).
/// H100 4th-gen tensor cores: 8192 FP8 ops/cycle/SM (dense, no sparsity).
///   SXM5: 132 SMs x 1830 MHz x 8192 = 1,979 TFLOPS
///   PCIe: 114 SMs x 1620 MHz x 8192 = 1,513 TFLOPS
double compute_peak_tflops(const cudaDeviceProp& prop) {
    int clock_khz = 0;
    cudaDeviceGetAttribute(&clock_khz, cudaDevAttrClockRate, 0);
    double base = (double)prop.multiProcessorCount * (clock_khz / 1.0e6) * 1e-3;
    return base * 8192.0;
}

void parse_args(int argc, char** argv, int& M, int& N, int& K, int& batch) {
    M = 4096; N = 4096; K = 4096; batch = 1;
    // Collect positional args (skip key=value config args)
    std::vector<char*> pos;
    for (int i = 1; i < argc; ++i) {
        if (strchr(argv[i], '=') == nullptr) pos.push_back(argv[i]);
    }
    int npos = static_cast<int>(pos.size());
    if (npos >= 4) {
        M = std::atoi(pos[0]); N = std::atoi(pos[1]);
        K = std::atoi(pos[2]); batch = std::atoi(pos[3]);
    } else if (npos >= 3) {
        M = std::atoi(pos[0]); N = std::atoi(pos[1]); K = std::atoi(pos[2]);
    } else if (npos == 2) {
        M = std::atoi(pos[0]); N = std::atoi(pos[1]);
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
                  << " -- not multiples of 128, may reduce performance.\n";
    }
}


// ========================================================================================
// Triangle Slab Count (replicates adaptive algorithm for display)
// ========================================================================================

int compute_triangle_slabs(int N, int sm_count, const TriangleConfig& tri) {
    constexpr int kTileM = COMPLEX_FP8_TILE_M;
    constexpr int kTileN = COMPLEX_FP8_TILE_N;

    int target_slabs;
    if (tri.target_slabs > 0) {
        target_slabs = tri.target_slabs;
    } else {
        target_slabs = 2;
        for (int T = 32; T >= 2; --T) {
            int S = ((N + T - 1) / T);
            S = ((S + kTileM - 1) / kTileM) * kTileM;
#if defined(CUTLASS_ARCH_MMA_MODIFIABLE_TMA_SM90_SUPPORTED) && \
    !defined(COMPLEX_FP8_DISABLE_GROUPED_GEMM)
            // Grouped path: check TOTAL tiles across all slabs (matches library logic)
            int64_t total_tiles = 0;
            int pos = 0;
            for (int slab = 0; slab < T && pos < N; ++slab) {
                int row_end = std::min(pos + S, N);
                int M_block = row_end - pos;
                int N_block = row_end;
                int tm = (M_block + kTileM - 1) / kTileM;
                int tn = (N_block + kTileN - 1) / kTileN;
                total_tiles += static_cast<int64_t>(tm) * tn;
                pos = row_end;
            }
            if (total_tiles >= 4 * sm_count) {
                target_slabs = T;
                break;
            }
#else
            // Per-slab path: check slab 0 tiles only
            int tiles_m = (S + kTileM - 1) / kTileM;
            int tiles_n = (S + kTileN - 1) / kTileN;
            int64_t tiles_slab0 = static_cast<int64_t>(tiles_m) * tiles_n;
            if (tiles_slab0 >= sm_count) {
                target_slabs = T;
                break;
            }
#endif
        }
    }

    int min_slab_height;
    if (tri.min_slab_height > 0) {
        min_slab_height = tri.min_slab_height;
    } else {
#if defined(CUTLASS_ARCH_MMA_MODIFIABLE_TMA_SM90_SUPPORTED) && \
    !defined(COMPLEX_FP8_DISABLE_GROUPED_GEMM)
        // Grouped path: no per-slab occupancy constraint (matches library logic)
        min_slab_height = kTileM;
#else
        // Per-slab path: each slab must fill the GPU independently
        double ratio = static_cast<double>(kTileM) / kTileN;
        int tiles_needed = static_cast<int>(std::ceil(
            std::sqrt(static_cast<double>(sm_count) / ratio)));
        min_slab_height = kTileM * std::max(tiles_needed, 1);
#endif
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
//   4. Conjugate symmetry — Re(C)[i,j] == Re(C)[j,i]
//      (only testable for Baseline, which writes Re(C) as a full matrix)
//
// Returns validation results struct.

struct HerkValidation {
    bool pass;
    float max_re_err;       // max |HERK_re - Gram_re| in lower triangle
    float max_im_err;       // max |HERK_im - Gram_im| in lower triangle
    float max_diag_im;      // max |Im(C)[i,i]|
    float max_symm_re_err;  // max |Re(C)[i,j] - Re(C)[j,i]| (Baseline only)
    bool re_symmetry_tested;
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
    parse_args(argc, argv, M, N, K, batch);
    TriangleConfig tri = parse_triangle_config(argc, argv);
    BenchMode mode = parse_bench_mode(argc, argv);
    bool herk_graph = parse_herk_graph(argc, argv);
    HerkMode herk_mode = parse_herk_mode(argc, argv);
    PersistentMode persistent_mode = parse_persistent_mode(argc, argv);
    bool batch_tiling = parse_batch_tiling(argc, argv);
    int tune_level = parse_tune(argc, argv);
    int strat_verb = parse_strategy_verbosity(argc, argv);
    GemmMode gemm_mode = parse_gemm_mode(argc, argv);
    (void)gemm_mode;  // Available for future GEMM dispatch integration
    if (tune_level >= 0) {
        tune_cache::TuneCache::instance().set_verbosity(tune_level);
    }
    if (strat_verb >= 0) {
        strategy_cache::StrategyCache::instance().set_verbosity(strat_verb);
        strategy_cache::GemmStrategyCache::instance().set_verbosity(strat_verb);
    }
    tune_cache::TuneCache::instance().load();
    bool run_tests = (mode == BenchMode::Both || mode == BenchMode::Test);
    bool run_bench = (mode == BenchMode::Both || mode == BenchMode::Bench);

    // Adaptive warmup/iteration counts
    int warmup, iters;
    adaptive_bench_counts(N, warmup, iters);

    // ---- System info ----
    auto sysinfo = cutlass_complex::query_system_info();
    int sm_count = sysinfo.sm_count;
    std::vector<RooflineEntry> roofline;  // collected during benchmarks

    std::cout << "Complex GEMM Benchmark — CUTLASS 3.x, Hopper SM90\n";
    cutlass_complex::print_system_info(sysinfo);
    cutlass_complex::print_build_config();

    // ---- Schedule name ----
    const char* schedule_name =
#if defined(COMPLEX_FP8_USE_PINGPONG) && COMPLEX_FP8_USE_PINGPONG
        "Pingpong";
#elif defined(COMPLEX_FP8_USE_FAST_ACCUM) && !COMPLEX_FP8_USE_FAST_ACCUM
        "Cooperative";
#else
        "Cooperative (FastAccum)";
#endif

    std::cout << "Problem: " << M << "x" << N << "x" << K;
    if (batch > 1) std::cout << " batch=" << batch;
    std::cout << "  Tile: " << COMPLEX_FP8_TILE_M << "x"
              << COMPLEX_FP8_TILE_N << "x" << COMPLEX_FP8_TILE_K
              << "  Cluster: " << COMPLEX_FP8_CLUSTER_M << "x"
              << COMPLEX_FP8_CLUSTER_N << "x1"
              << "  " << schedule_name
              << "\nPrecision: FP8 E4M3 compute, FP16/FP32 I/O, FP32 accumulate\n";
    if (mode != BenchMode::Both)
        std::cout << "Mode: " << (mode == BenchMode::Test ? "test" : "bench") << "\n";
    if (run_bench)
        std::cout << "Benchmark: " << warmup << " warmup + " << iters << " iterations\n";
    if (tri.target_slabs > 0 || tri.min_slab_height > 0 || tri.graduated || tri.verbose)
        std::cout << "Triangle: slabs=" << tri.target_slabs << " min_slab=" << tri.min_slab_height
                  << " graduated=" << tri.graduated << " verbose=" << tri.verbose << "\n";
    if (herk_graph)
        std::cout << "HERK graph: ON (baseline HERK captured as CUDA graph)\n";
    if (herk_mode == HerkMode::ForceDirect)
        std::cout << "Direct HERK: ON (forced single-kernel direct path)\n";
    else if (herk_mode == HerkMode::ForceBaseline)
        std::cout << "Direct HERK: OFF (forced baseline multi-launch path)\n";
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

    GemmComplexFP8 gemm;

    // ---- Strategy autotune or manual configuration ----
    if (tune_level == -1) {
        // Full autotune: sweep all runtime options, override CLI flags
        // Default strategy verbosity to 2 for full sweep output, unless user overrode
        if (strat_verb < 0)
            strategy_cache::StrategyCache::instance().set_verbosity(2);

        // Build viable config list for GemmConfig sweep
        auto viable = all_baseline_configs();
        std::vector<int> viable_ints;
        for (auto c : viable) viable_ints.push_back(static_cast<int>(c));

        auto entry = strategy_cache::run_autotune<
            GemmComplexFP8, HerkMode, PersistentMode, DirectHerkConfig, HerkTileSize,
            HerkPipelineMode,
            HerkOp, FillMode, HerkStrategy, CutlassParams, TriangleConfig>(
                gemm, N, K, batch, 0 /* FP8 only on SM90 */, nullptr, viable_ints);

        // Apply ALL cached settings (overrides CLI flags)
        strategy_cache::apply_cached_settings<GemmComplexFP8, HerkMode, PersistentMode,
                                              GemmConfig, DirectHerkConfig, HerkTileSize,
                                              HerkPipelineMode>(
            gemm, entry);
        tri.use_cuda_graph = entry.use_cuda_graph;
        herk_mode = entry.herk_mode == 0 ? HerkMode::ForceDirect : HerkMode::ForceBaseline;
        persistent_mode = entry.persistent_mode == 1 ? PersistentMode::ForceOn : PersistentMode::ForceOff;
        herk_graph = entry.herk_graph;
        batch_tiling = entry.batch_tiling;

        printf("[Autotune] Applied: %s+%s+persistent=%d+graph=%d+herk_graph=%d+tile=%d+direct_config=%d(%s) (%.1f TFLOPS)\n",
               entry.herk_mode == 0 ? "Direct" : "Baseline",
               entry.herk_strategy == 1 ? "TriangleAware" : "Baseline",
               entry.persistent_mode, (int)entry.use_cuda_graph,
               (int)entry.herk_graph, (int)entry.batch_tiling,
               entry.direct_config, direct_herk_config_name(static_cast<DirectHerkConfig>(entry.direct_config)),
               entry.tflops);
    } else {
        // Manual configuration from CLI flags
        if (herk_graph) gemm.set_herk_graph(true);
        gemm.set_herk_mode(herk_mode);
        gemm.set_persistent_mode(persistent_mode);
        gemm.set_batch_tiling(batch_tiling);
    }

    // Deinterleave A, B to planar for Gram / test-phase calls
    GemmComplexFP8::deinterleave_complex(d_A, d_Ar, d_Ai, size_A, nullptr);
    GemmComplexFP8::deinterleave_complex(d_B, d_Br, d_Bi, size_B, nullptr);
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
    {
        cudaMemset(d_C, 0, 2 * size_C * sizeof(__half));
        auto status = gemm.GEMM(d_A, d_B, d_C, M, N, K);
        std::cout << "  Launch: " << (status == cutlass::Status::kSuccess ? "PASS" : "FAIL") << "\n\n";
    }

    std::cout << "Hermitian GEMM (C=A*B^H):\n";
    {
        cudaMemset(d_C, 0, 2 * size_C * sizeof(__half));
        auto status = gemm.GEMM(d_A, d_B, d_C, M, N, K, 1.0f, 0.0f, ComplexMode::Hermitian);
        std::cout << "  Launch: " << (status == cutlass::Status::kSuccess ? "PASS" : "FAIL") << "\n\n";
    }

    std::cout << "Gram reference (C=A*A^H, 4 sub-GEMMs):\n";
    __half *d_RefCr, *d_RefCi;
    cudaMalloc(&d_RefCr, herk_C_size * sizeof(__half));
    cudaMalloc(&d_RefCi, herk_C_size * sizeof(__half));
    cudaMemset(d_RefCr, 0, herk_C_size * sizeof(__half));
    cudaMemset(d_RefCi, 0, herk_C_size * sizeof(__half));
    {
        auto status = gemm.run_gram_planar(d_Ar, d_Ai, d_RefCr, d_RefCi,
                                            M, K, 1.0f, 0.0f, GramMode::AAH);
        std::cout << "  Launch: " << (status == cutlass::Status::kSuccess ? "PASS" : "FAIL") << "\n";
        std::cout << "  (used as ground truth for HERK validation)\n\n";
    }

    // Copy reference to host
    std::vector<__half> h_RefCr(herk_C_size), h_RefCi(herk_C_size);
    cudaMemcpy(h_RefCr.data(), d_RefCr, herk_C_size * sizeof(__half), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_RefCi.data(), d_RefCi, herk_C_size * sizeof(__half), cudaMemcpyDeviceToHost);

    float herk_tol = 0.5f;

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
                                        HerkStrategy::Baseline);
        std::cout << "  Launch: " << (status == cutlass::Status::kSuccess ? "PASS" : "FAIL") << "\n";

        if (status == cutlass::Status::kSuccess) {
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
                                        HerkStrategy::TriangleAware, nullptr, tri);
        std::cout << "  Launch: " << (status == cutlass::Status::kSuccess ? "PASS" : "FAIL") << "\n";

        if (status == cutlass::Status::kSuccess) {
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

    cudaFree(d_RefCr);
    cudaFree(d_RefCi);

    // ---- Batched HERK test ----
    std::cout << "Batched HERK (FP8, TriangleAware vs Baseline):\n";
    {
        const int batch_count = std::max(batch, 4);
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
                  << (status_base == cutlass::Status::kSuccess ? "PASS" : "FAIL") << "\n";

        // TriangleAware
        CutlassParams tp;
        tp.herk_strategy = HerkStrategy::TriangleAware;
        tp.triangle_config = tri;
        auto status_tri = gemm.HERK_batched(
            d_A_batched, d_C_triangle, M, K, batch_count,
            1.0f, 0.0f, HerkOp::NoTrans, FillMode::Lower, tp);
        std::cout << "  Triangle:  "
                  << (status_tri == cutlass::Status::kSuccess ? "PASS" : "FAIL") << "\n";

        // Validate: compare TriangleAware output against Baseline output
        if (status_base == cutlass::Status::kSuccess && status_tri == cutlass::Status::kSuccess) {
            int64_t total_packed = packed_elems * batch_count;
            std::vector<__half> h_base(2 * total_packed), h_tri(2 * total_packed);
            cudaMemcpy(h_base.data(), d_C_baseline, 2 * total_packed * sizeof(__half),
                       cudaMemcpyDeviceToHost);
            cudaMemcpy(h_tri.data(), d_C_triangle, 2 * total_packed * sizeof(__half),
                       cudaMemcpyDeviceToHost);

            float max_err = 0.0f;
            #pragma omp parallel for reduction(max:max_err)
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
    std::cout << "\n";

    } // end run_tests


    // ====================================================================
    // PRODUCTION PHASE — clean benchmarks, no validation
    // ====================================================================
    if (run_bench) {
    std::cout << "\n"
              << "╔══════════════════════════════════════════════════════╗\n"
              << "║                 PRODUCTION PHASE                    ║\n"
              << "╚══════════════════════════════════════════════════════╝\n";

    int herk_sub_gemms = 3;
#if defined(COMPLEX_FP8_HERK_FULL_MATRIX) && COMPLEX_FP8_HERK_FULL_MATRIX
    herk_sub_gemms = 4;
#endif

    printf("\n--- FP8 E4M3 ---\n");
    printf("  %-18s %11s  %13s  %9s  %s\n", "Operation", "Time", "TFLOPS", "GB/s", "IO");

    // Standard GEMM: C = A*B (4 sub-GEMMs)
    {
        auto ms = benchmark_ms([&]() {
            gemm.GEMM(d_A, d_B, d_C, M, N, K);
        }, warmup, iters);
        double eb = cutlass_complex::gemm_external_bytes(M, N, K);
        print_result("Standard", M, N, K, ms, 4, 1, 0.0, 0, eb);
        roofline.push_back(make_roofline("Standard", ms, 4, M, N, K, 1, eb,
                                          sysinfo.peak_fp8_tflops));
    }

    // Hermitian GEMM: C = A*B^H (4 sub-GEMMs)
    {
        auto ms = benchmark_ms([&]() {
            gemm.GEMM(d_A, d_B, d_C, M, N, K, 1.0f, 0.0f, ComplexMode::Hermitian);
        }, warmup, iters);
        double eb = cutlass_complex::gemm_external_bytes(M, N, K);
        print_result("Hermitian", M, N, K, ms, 4, 1, 0.0, 0, eb);
        roofline.push_back(make_roofline("Hermitian", ms, 4, M, N, K, 1, eb,
                                          sysinfo.peak_fp8_tflops));
    }

    // Gram A*A^H + pack (4 sub-GEMMs)
    {
        __half *gCr, *gCi, *gPr, *gPi;
        int64_t packed_size = (int64_t)M * (M + 1) / 2;
        cudaMalloc(&gCr, herk_C_size * sizeof(__half));
        cudaMalloc(&gCi, herk_C_size * sizeof(__half));
        cudaMalloc(&gPr, packed_size * sizeof(__half));
        cudaMalloc(&gPi, packed_size * sizeof(__half));

        auto ms = benchmark_ms([&]() {
            gemm.run_gram_planar(d_Ar, d_Ai, gCr, gCi,
                                 M, K, 1.0f, 0.0f, GramMode::AAH);
            pack_lower_triangle(gCr, gPr, M);
            pack_lower_triangle(gCi, gPi, M);
        }, warmup, iters);
        double eb = cutlass_complex::herk_external_bytes(M, K);
        print_result("Gram A*A^H", M, M, K, ms, 4, 1, 0.0, 0, eb);
        roofline.push_back(make_roofline("Gram", ms, 4, M, M, K, 1, eb,
                                          sysinfo.peak_fp8_tflops));

        cudaFree(gCr); cudaFree(gCi);
        cudaFree(gPr); cudaFree(gPi);
    }

    // HERK Baseline (3 sub-GEMMs, packed output)
    double herk_baseline_ms;
    {
        int64_t packed_size = (int64_t)M * (M + 1) / 2;
        __half *hC;
        cudaMalloc(&hC, 2 * packed_size * sizeof(__half));

        double herk_eb = cutlass_complex::herk_external_bytes(M, K);
        herk_baseline_ms = benchmark_ms([&]() {
            gemm.HERK(d_A, hC, M, K);
        }, warmup, iters);
        print_result("HERK Baseline", M, M, K, herk_baseline_ms, herk_sub_gemms,
                     1, 0.0, 0, herk_eb);
        roofline.push_back(make_roofline("HERK BL", herk_baseline_ms,
                                          herk_sub_gemms, M, M, K, 1, herk_eb,
                                          sysinfo.peak_fp8_tflops));

        cudaFree(hC);
    }

    // HERK Triangle (packed output, baseline FLOPs for comparable TFLOPS)
    int slabs = compute_triangle_slabs(M, sm_count, tri);
    {
        int64_t packed_size = (int64_t)M * (M + 1) / 2;
        __half *hC;
        cudaMalloc(&hC, 2 * packed_size * sizeof(__half));

        CutlassParams tri_params;
        tri_params.herk_strategy = HerkStrategy::TriangleAware;
        tri_params.triangle_config = tri;
        double herk_eb = cutlass_complex::herk_external_bytes(M, K);
        auto ms = benchmark_ms([&]() {
            gemm.HERK(d_A, hC, M, K, 1.0f, 0.0f,
                      HerkOp::NoTrans, FillMode::Lower, tri_params);
        }, warmup, iters);
        // Use baseline FLOPs so TFLOPS reflects speedup directly
        print_result("HERK Triangle", M, M, K, ms, herk_sub_gemms, 1,
                     herk_baseline_ms, slabs, herk_eb);
        roofline.push_back(make_roofline("HERK Tri", ms, herk_sub_gemms, M, M, K,
                                          1, herk_eb, sysinfo.peak_fp8_tflops));

        cudaFree(hC);
    }


    // ---- Batched HERK Benchmark (FP8) ----
    if (batch > 1) {
        const int batch_count = batch;
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

        double gflops_base = 3.0 * batch_count * 2.0 * M * M * K / 1e9;
        double batched_eb = cutlass_complex::herk_external_bytes(M, K, batch_count);

        // Utilization sub-line printer
        // Shows TC%, external I/O (user buffers), and total DRAM estimate
        // (includes deinterleave, FP8 cast, scratch C, pack, interleave).
        double internal_bytes = cutlass_complex::herk_internal_bytes(M, K, batch_count);
        double external_bytes = batched_eb;
        auto print_util = [&](double ms) {
            double time_s = ms * 1e-3;
            double achieved = gflops_base / ms;  // TFLOPS
            double tc_pct = (sysinfo.peak_fp8_tflops > 0) ? achieved / sysinfo.peak_fp8_tflops * 100.0 : 0;
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
            printf("\n--- Batched HERK (batch=%d, FP8, tile=%d) ---\n", batch_count, tile_display);
        else
            printf("\n--- Batched HERK (batch=%d, FP8, tile=off) ---\n", batch_count);

        // Baseline
        double bl_ms;
        {
            CutlassParams bp;
            bp.herk_strategy = HerkStrategy::Baseline;
            bl_ms = benchmark_ms([&]() {
                gemm.HERK_batched(d_A_batched, hC, M, K, batch_count,
                                  1.0f, 0.0f, HerkOp::NoTrans, FillMode::Lower, bp);
            }, warmup, iters);
            double tflops = gflops_base / bl_ms;
            double gbs = batched_eb / (bl_ms * 1e-3) / 1e9;
            double io = (gflops_base * 1e9) / batched_eb;
            double per_item = bl_ms / batch_count;
            printf("  Batched Baseline:     %7.2f ms  (%6.1f TFLOPS, %5.1f GB/s, IO=%.0f, %.3f ms/item)\n",
                   bl_ms, tflops, gbs, io, per_item);
            print_util(bl_ms);
            roofline.push_back(make_roofline("Bat BL", bl_ms, 3, M, M, K,
                                              batch_count, batched_eb,
                                              sysinfo.peak_fp8_tflops));
        }

        // TriangleAware
        {
            CutlassParams tp;
            tp.herk_strategy = HerkStrategy::TriangleAware;
            tp.triangle_config = tri;
            auto ms = benchmark_ms([&]() {
                gemm.HERK_batched(d_A_batched, hC, M, K, batch_count,
                                  1.0f, 0.0f, HerkOp::NoTrans, FillMode::Lower, tp);
            }, warmup, iters);
            double speedup = bl_ms / ms;
            double tflops = gflops_base / ms;
            double gbs = batched_eb / (ms * 1e-3) / 1e9;
            double io = (gflops_base * 1e9) / batched_eb;
            double per_item = ms / batch_count;
            printf("  Batched Triangle:     %7.2f ms  (%6.1f TFLOPS, %5.1f GB/s, IO=%.0f, %.3f ms/item, %.2fx)\n",
                   ms, tflops, gbs, io, per_item, speedup);
            print_util(ms);
            roofline.push_back(make_roofline("Bat Tri", ms, 3, M, M, K,
                                              batch_count, batched_eb,
                                              sysinfo.peak_fp8_tflops));
        }

        cudaFree(d_A_batched);
        cudaFree(hC);
    }

    // ---- Production GEMM Benchmarks ----
    // Tests batched planar complex GEMM at beamformer and dedisp shapes.
    // Uses FP8 compute, FP32 output via the planar batched API.
    {
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
            {"VoltBF 4000x4000x1664 b1",   4000, 4000, 1664,   1},
            {"VoltBF 4000x4000x1664 b4",   4000, 4000, 1664,   4},
            {"VoltBF 4000x4000x1664 b16",  4000, 4000, 1664,  16},
            {"VoltBF 4000x4000x1664 b32",  4000, 4000, 1664,  32},
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
                    GemmComplexFP8, GemmConfig, ComplexMode>(
                        gemm, gp.M, gp.N, gp.K, gp.batch, 0 /*FP8*/, nullptr, cfg_ints);

                strategy_cache::apply_gemm_cached_settings<GemmComplexFP8, GemmConfig>(
                    gemm, gentry);

                printf("  [%s] Autotuned: cfg=%d(%s) %.1f TFLOPS\n",
                       gp.name, gentry.gemm_config,
                       config_name(static_cast<GemmConfig>(gentry.gemm_config)),
                       gentry.tflops);
            }

            // Benchmark
            auto ms = benchmark_ms([&]() {
                gemm.run_planar_batched_fp32out(
                    gAr, gAi, gBr, gBi, gCr, gCi,
                    gp.M, gp.N, gp.K, gp.batch,
                    1.0f, 0.0f, ComplexMode::Standard, nullptr);
            }, warmup, iters);

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
    printf("  FP8:  ~%6.1f TFLOPS  (8192 ops/cycle/SM)\n", sysinfo.peak_fp8_tflops);
    printf("GEMM: 4 sub-GEMMs (4M algorithm), HERK Baseline: 3, HERK Triangle: 2 tri + 1 full\n");
    printf("All HERK/Gram output packed lower triangle, N*(N+1)/2 complex elements\n");
    printf("TFLOPS = baseline FLOPs / wall-clock time (all HERK variants use baseline FLOPs)\n");

    // ---- Roofline ----
    print_roofline_table(roofline, sysinfo);

    // ---- Small-K Analysis ----
    if (K < M && run_bench && batch > 1) {
        int bc = batch;
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
