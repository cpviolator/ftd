/*
 * benchmark_configs_sm100.cu
 *
 * Multi-size, multi-precision, multi-config scaling benchmark for complex FP8
 * GEMM on NVIDIA Blackwell (SM100/SM120).
 *
 * Sweeps square problem sizes (256..8192) across all compiled-in precisions
 * (FP8, FP6 E3M2/E2M3, FP4 E2M1) and GemmConfigs (Default, Tile_128x256,
 * Cluster_1x2, Cluster_2x2) using the GemmComplexSm100 library API.
 *
 * Operations benchmarked per (config, precision, size):
 *   - Standard GEMM:  C = A * B            (4 sub-GEMMs)
 *   - Hermitian GEMM: C = A * B^H          (4 sub-GEMMs)
 *   - Gram A*A^H:     C = A * A^H + pack   (4 sub-GEMMs)
 *   - HERK Baseline:  packed output         (3 sub-GEMMs)
 *   - HERK Triangle:  triangle-aware        (2 tri + 1 full, all precisions)
 *   - Tri+Graph:      triangle + CUDA graph (SM120 optimization, all precisions)
 *
 * When batch>1, additional batched benchmarks:
 *   - Bat.GEMM:       batched standard GEMM
 *   - Bat.HERK:       batched HERK baseline
 *   - Bat.Tri:        batched HERK triangle-aware
 *   - Bat.Tri+G:      batched HERK triangle + CUDA graph
 *
 * Compile:
 *   cmake . && make -j
 *
 * All FP8 tile/cluster configs are always compiled (runtime-selectable).
 * The benchmark sweeps all configs valid for the current architecture.
 */

#include "gemm_complex_sm100.hpp"
#include "system_info.hpp"
#include <iostream>
#include <iomanip>
#include <vector>
#include <random>
#include <chrono>
#include <cmath>
#include <cstring>
#include <algorithm>
#include <string>

using namespace gemm_complex_sm100;


// ========================================================================================
// Helpers
// ========================================================================================

void fill_random_fp16(__half* buf, int64_t count, float scale = 0.5f) {
    static std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-scale, scale);
    for (int64_t i = 0; i < count; ++i) buf[i] = __float2half(dist(rng));
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

const char* precision_name(ComputePrecision p) {
    switch (p) {
        case ComputePrecision::FP8_E4M3:  return "FP8 E4M3";
#ifdef COMPLEX_SM100_ENABLE_FP6
        case ComputePrecision::FP6_E3M2:  return "FP6 E3M2";
        case ComputePrecision::FP6_E2M3:  return "FP6 E2M3";
#endif
#ifdef COMPLEX_SM100_ENABLE_FP4
        case ComputePrecision::FP4_E2M1:  return "FP4 E2M1";
#endif
        default:                          return "Unknown";
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

/// Returns true if the given GemmConfig is valid for the given precision.
/// Tile_128x256 is FP8-only (block-scaled kernels use a different tile).
bool is_valid_combo(GemmConfig cfg, ComputePrecision prec) {
    // Wide-N and cluster/2SM tiles are FP8-only (block-scaled uses separate tile configs)
    if (prec != ComputePrecision::FP8_E4M3) {
        switch (cfg) {
        case GemmConfig::FP8_T128x256_C1x1_SAuto:
        case GemmConfig::FP8_T128x128_C1x2:
        case GemmConfig::FP8_T128x128_C2x2:
        case GemmConfig::FP8_T128x256_C1x2:
        case GemmConfig::FP8_T256x128_C2x1_2SM:
        case GemmConfig::FP8_T256x256_C2x2_2SM:
            return false;
        default: break;
        }
    }
    return is_config_valid_for_arch(cfg);
}

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


// ========================================================================================
// Argument Parsing
// ========================================================================================

TriangleConfig parse_triangle_config(int argc, char** argv) {
    TriangleConfig tc;
    for (int i = 1; i < argc; ++i) {
        std::string arg(argv[i]);
        auto eq = arg.find('=');
        if (eq == std::string::npos) continue;
        std::string key = arg.substr(0, eq);
        std::string val = arg.substr(eq + 1);
        if (key == "slabs")          tc.target_slabs = std::atoi(val.c_str());
        else if (key == "min_slab")  tc.min_slab_height = std::atoi(val.c_str());
        else if (key == "graduated") tc.graduated = (val == "1" || val == "true");
        else if (key == "verbose")   tc.verbose = (val == "1" || val == "true");
        else if (key == "graph")     tc.use_cuda_graph = (val == "1" || val == "true");
    }
    return tc;
}

void parse_args(int argc, char** argv,
                std::vector<int>& sizes, int& batch,
                ComputePrecision& precision_filter, bool& filter_precision) {
    sizes.clear();
    batch = 1;
    filter_precision = false;
    precision_filter = ComputePrecision::FP8_E4M3;

    for (int i = 1; i < argc; ++i) {
        std::string arg(argv[i]);
        if (arg == "--help" || arg == "-h") continue;

        auto eq = arg.find('=');
        if (eq != std::string::npos) {
            std::string key = arg.substr(0, eq);
            std::string val = arg.substr(eq + 1);
            if (key == "batch") {
                batch = std::atoi(val.c_str());
                if (batch < 1) batch = 1;
            } else if (key == "precision") {
                filter_precision = true;
                if (val == "fp8")           precision_filter = ComputePrecision::FP8_E4M3;
#ifdef COMPLEX_SM100_ENABLE_FP6
                else if (val == "fp6e3m2")  precision_filter = ComputePrecision::FP6_E3M2;
                else if (val == "fp6e2m3")  precision_filter = ComputePrecision::FP6_E2M3;
#endif
#ifdef COMPLEX_SM100_ENABLE_FP4
                else if (val == "fp4")      precision_filter = ComputePrecision::FP4_E2M1;
#endif
                else {
                    fprintf(stderr, "Warning: unknown precision '%s', using fp8\n", val.c_str());
                    precision_filter = ComputePrecision::FP8_E4M3;
                }
            }
            // Other key=value pairs handled by parse_triangle_config
            continue;
        }

        // Try as integer (positional size)
        char* end;
        long v = strtol(arg.c_str(), &end, 10);
        if (*end == '\0' && v > 0) {
            sizes.push_back((int)v);
        }
    }

    // Default sizes if none specified
    if (sizes.empty()) {
        sizes = {256, 512, 1024, 2048, 4096, 8192};
    }
}


// ========================================================================================
// Main
// ========================================================================================

void print_help(const char* prog) {
    printf("Usage: %s [sizes...] [key=value ...] [--help]\n", prog);
    printf("\n");
    printf("Multi-size, multi-precision scaling benchmark for complex FP8 GEMM\n");
    printf("on NVIDIA Blackwell (SM100/SM120).\n");
    printf("\n");
    printf("Positional arguments:\n");
    printf("  sizes        One or more square problem sizes (default: 256 512 1024 2048 4096 8192)\n");
    printf("\n");
    printf("Key=value options:\n");
    printf("  batch=N          Batch count for batched benchmarks (default: 1, unbatched only)\n");
    printf("  precision=P      Filter to one precision: fp8, fp6e3m2, fp6e2m3, fp4 (default: all)\n");
    printf("  slabs=<2-32>     Triangle: number of horizontal slabs (0=auto)\n");
    printf("  min_slab=<int>   Triangle: minimum slab height in rows (0=auto)\n");
    printf("  graduated=<0|1>  Triangle: sqrt-spaced slab boundaries (default: 0)\n");
    printf("  verbose=<0|1>    Triangle: print slab decomposition to stderr (default: 0)\n");
    printf("  graph=<0|1>      Triangle: CUDA graph capture/replay (default: 0)\n");
    printf("\n");
    printf("Unbatched operations (always run, 6 columns):\n");
    printf("  Standard     C = A * B              (4 sub-GEMMs)\n");
    printf("  Hermitian    C = A * B^H            (4 sub-GEMMs)\n");
    printf("  Gram A*AH    C = A * A^H + pack     (4 sub-GEMMs)\n");
    printf("  HERK BL      packed output           (3 sub-GEMMs FP8, 4 FP6/FP4)\n");
    printf("  HERK Tri     triangle-aware          (2 tri + 1 full)\n");
    printf("  Tri+Graph    triangle + CUDA graph   (SM120 optimization)\n");
    printf("\n");
    printf("Batched operations (when batch>1, 4 additional columns):\n");
    printf("  Bat.GEMM     batched standard GEMM   (4 sub-GEMMs x batch)\n");
    printf("  Bat.HERK     batched HERK baseline    (3 or 4 sub-GEMMs x batch)\n");
    printf("  Bat.Tri      batched triangle-aware   (baseline FLOPs reference)\n");
    printf("  Bat.Tri+G    batched tri + CUDA graph (baseline FLOPs reference)\n");
    printf("\n");
    printf("Examples:\n");
    printf("  %s                                  # default sweep, unbatched\n", prog);
    printf("  %s 1024                             # single size, unbatched\n", prog);
    printf("  %s 1024 batch=4                     # single size, batched\n", prog);
    printf("  %s 1024 2048 batch=4 precision=fp8  # custom sizes, batched, FP8 only\n", prog);
    printf("  %s 4096 batch=128 verbose=1         # verbose triangle info\n", prog);
    printf("  %s 4096 batch=128 slabs=8 graph=1   # explicit slabs, CUDA graph\n", prog);
    printf("\n");
    printf("Compile options:\n");
    printf("  -DCOMPLEX_SM100_ENABLE_FP6=ON    Enable FP6 E3M2/E2M3 benchmarks\n");
    printf("  -DCOMPLEX_SM100_ENABLE_FP4=ON    Enable FP4 E2M1 benchmarks\n");
    printf("  All FP8 tile/cluster configs are always compiled (runtime-selectable)\n");
}

int main(int argc, char** argv) {
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            auto sysinfo = cutlass_complex::query_system_info();
            cutlass_complex::print_system_info(sysinfo);
            cutlass_complex::print_build_config();
            printf("\n");
            print_help(argv[0]);
            return 0;
        }
    }

    // ---- Parse arguments ----
    std::vector<int> sizes;
    int batch = 1;
    ComputePrecision precision_filter;
    bool filter_precision = false;
    parse_args(argc, argv, sizes, batch, precision_filter, filter_precision);
    TriangleConfig tri_cfg = parse_triangle_config(argc, argv);

    // ---- Device info ----
    auto sysinfo = cutlass_complex::query_system_info();
    int sm_count = sysinfo.sm_count;

    std::cout << "Complex FP8 GEMM Scaling Benchmark — CUTLASS 4.x, Blackwell SM100/SM120\n";
    cutlass_complex::print_system_info(sysinfo);
    cutlass_complex::print_build_config();

#ifdef COMPLEX_SM100_FP8_TILE_N
    std::cout << "Default tile: " << COMPLEX_FP8_SM100_MMA_M << "x"
              << COMPLEX_SM100_FP8_TILE_N << "x" << COMPLEX_FP8_SM100_MMA_K
#elif defined(COMPLEX_FP8_SM100_TARGET_SM120)
    std::cout << "Default tile: 128x64x" << COMPLEX_FP8_SM100_MMA_K
#else
    std::cout << "Default tile: " << COMPLEX_FP8_SM100_MMA_M << "x"
              << COMPLEX_FP8_SM100_MMA_N << "x" << COMPLEX_FP8_SM100_MMA_K
#endif
              << "  Cluster: " << COMPLEX_FP8_SM100_CLUSTER_M << "x"
              << COMPLEX_FP8_SM100_CLUSTER_N << "x1"
#if defined(COMPLEX_FP8_SM100_USE_2SM) && COMPLEX_FP8_SM100_USE_2SM
              << "  2SM"
#else
              << "  1SM"
#endif
              << "\n";

    std::cout << "FP8 configs: all compiled (runtime-selectable)\n";
    if (batch > 1) {
        std::cout << "Batch count: " << batch << "\n";
    }
    std::cout << "\n";

    // ---- Problem sizes ----
    std::sort(sizes.begin(), sizes.end());
    const int max_N = sizes.back();

    // ---- GemmConfigs to benchmark (all valid configs for current arch) ----
    auto configs = all_baseline_configs();

    // ---- Precisions to benchmark ----
    std::vector<ComputePrecision> precisions;
    if (filter_precision) {
        precisions.push_back(precision_filter);
    } else {
        precisions.push_back(ComputePrecision::FP8_E4M3);
#ifdef COMPLEX_SM100_ENABLE_FP6
        precisions.push_back(ComputePrecision::FP6_E3M2);
        precisions.push_back(ComputePrecision::FP6_E2M3);
#endif
#ifdef COMPLEX_SM100_ENABLE_FP4
        precisions.push_back(ComputePrecision::FP4_E2M1);
#endif
    }

    // ---- Allocate once for max size, reuse for smaller ----
    int64_t max_A = (int64_t)max_N * max_N;
    int64_t max_C = (int64_t)max_N * max_N;
    int64_t max_packed = (int64_t)max_N * (max_N + 1) / 2;

    __half *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, 2 * max_A * sizeof(__half));
    cudaMalloc(&d_B, 2 * max_A * sizeof(__half));
    cudaMalloc(&d_C, 2 * max_C * sizeof(__half));

    __half *d_Ar, *d_Ai;
    cudaMalloc(&d_Ar, max_A * sizeof(__half));
    cudaMalloc(&d_Ai, max_A * sizeof(__half));

    __half *d_GCr, *d_GCi, *d_GPr, *d_GPi;
    cudaMalloc(&d_GCr, max_C * sizeof(__half));
    cudaMalloc(&d_GCi, max_C * sizeof(__half));
    cudaMalloc(&d_GPr, max_packed * sizeof(__half));
    cudaMalloc(&d_GPi, max_packed * sizeof(__half));

    __half *d_HC;
    cudaMalloc(&d_HC, 2 * max_packed * sizeof(__half));

    {
        std::vector<__half> h_buf(2 * max_A);
        fill_random_fp16(h_buf.data(), 2 * max_A, 0.5f);
        cudaMemcpy(d_A, h_buf.data(), 2 * max_A * sizeof(__half), cudaMemcpyHostToDevice);
        fill_random_fp16(h_buf.data(), 2 * max_A, 0.5f);
        cudaMemcpy(d_B, h_buf.data(), 2 * max_A * sizeof(__half), cudaMemcpyHostToDevice);
    }
    cudaMemset(d_C, 0, 2 * max_C * sizeof(__half));

    // ---- Batched buffers (allocated only when batch > 1) ----
    __half *d_A_bat = nullptr, *d_B_bat = nullptr;
    __half *d_C_bat = nullptr, *d_HC_bat = nullptr;
    if (batch > 1) {
        cudaMalloc(&d_A_bat, 2 * max_A * batch * sizeof(__half));
        cudaMalloc(&d_B_bat, 2 * max_A * batch * sizeof(__half));
        cudaMalloc(&d_C_bat, 2 * max_C * batch * sizeof(__half));
        cudaMalloc(&d_HC_bat, 2 * max_packed * batch * sizeof(__half));

        // Fill by replicating single-batch data
        for (int b = 0; b < batch; ++b) {
            cudaMemcpy(d_A_bat + b * 2 * max_A, d_A,
                       2 * max_A * sizeof(__half), cudaMemcpyDeviceToDevice);
            cudaMemcpy(d_B_bat + b * 2 * max_A, d_B,
                       2 * max_A * sizeof(__half), cudaMemcpyDeviceToDevice);
        }
        cudaMemset(d_C_bat, 0, 2 * max_C * batch * sizeof(__half));
    }

    GemmComplexSm100::deinterleave_complex(d_A, d_Ar, d_Ai, max_A, nullptr);
    cudaDeviceSynchronize();

    GemmComplexSm100 gemm;

    // ---- Column headers ----
    const int num_unbatched = 6;
    const int num_batched = 4;
    const int num_ops = (batch > 1) ? (num_unbatched + num_batched) : num_unbatched;
    const char* col_headers[] = {
        "Standard", "Hermitian", "Gram A*AH", "HERK BL", "HERK Tri", "Tri+Graph",
        "Bat.GEMM", "Bat.HERK", "Bat.Tri", "Bat.Tri+G"
    };

    // ---- Run benchmarks: config x precision x size ----
    constexpr int MAX_OPS = 10;  // 6 unbatched + 4 batched
    for (auto cfg : configs) {
        for (auto prec : precisions) {
            if (!is_valid_combo(cfg, prec)) continue;

            printf("\n--- Config: %s | %s ---\n", config_name(cfg), precision_name(prec));
            printf("%8s", "N");
            for (int c = 0; c < num_ops; ++c)
                printf("  %10s", col_headers[c]);
            printf("\n");

            double last_ms[MAX_OPS] = {};
            int last_N = 0;

            for (int N : sizes) {
	      int M = N, K = N;

                // Skip sizes where K < minimum tile_K for this precision
                int min_k = minimum_k_for_precision(prec);
                if (K < min_k) {
                    printf("%8d", N);
                    for (int c = 0; c < num_ops; ++c)
                        printf("  %10s", "K<tile");
                    printf("    (K=%d < tile_K=%d)\n", K, min_k);
                    continue;
                }

                int64_t size_C = (int64_t)M * N;
                int64_t packed_elems = (int64_t)N * (N + 1) / 2;

                int warmup, iters;
                if (N <= 512)       { warmup = 5; iters = 20; }
                else if (N <= 2048) { warmup = 3; iters = 10; }
                else                { warmup = 2; iters = 5;  }

                CutlassParams params;
                params.precision = prec;
                params.config = cfg;
                params.triangle_config = tri_cfg;

                double ms[MAX_OPS];
                for (int c = 0; c < MAX_OPS; ++c) ms[c] = -1.0;

                // 1. Standard GEMM
                {
                    cudaMemset(d_C, 0, 2 * size_C * sizeof(__half));
                    ms[0] = benchmark_ms([&]() {
                        gemm.GEMM(d_A, d_B, d_C, M, N, K, 1.0f, 0.0f,
                                  ComplexMode::Standard, params);
                    }, warmup, iters);
                }

                // 2. Hermitian GEMM
                {
                    cudaMemset(d_C, 0, 2 * size_C * sizeof(__half));
                    ms[1] = benchmark_ms([&]() {
                        gemm.GEMM(d_A, d_B, d_C, M, N, K, 1.0f, 0.0f,
                                  ComplexMode::Hermitian, params);
                    }, warmup, iters);
                }

                // 3. Gram A*A^H + pack
                {
                    cudaMemset(d_GCr, 0, size_C * sizeof(__half));
                    cudaMemset(d_GCi, 0, size_C * sizeof(__half));
                    ms[2] = benchmark_ms([&]() {
                        gemm.run_gram_planar(d_Ar, d_Ai, d_GCr, d_GCi,
                                             M, K, 1.0f, 0.0f, GramMode::AAH, prec,
                                             nullptr, cfg);
                        pack_lower_triangle(d_GCr, d_GPr, N);
                        pack_lower_triangle(d_GCi, d_GPi, N);
                    }, warmup, iters);
                }

                // 4. HERK Baseline
                {
                    ms[3] = benchmark_ms([&]() {
                        gemm.HERK(d_A, d_HC, N, K, 1.0f, 0.0f,
                                  HerkOp::NoTrans, FillMode::Lower, params);
                    }, warmup, iters);
                }

                // 5. HERK Triangle
                {
                    CutlassParams tri_params;
                    tri_params.precision = prec;
                    tri_params.config = cfg;
                    tri_params.herk_strategy = HerkStrategy::TriangleAware;
                    tri_params.triangle_config = tri_cfg;
                    ms[4] = benchmark_ms([&]() {
                        gemm.HERK(d_A, d_HC, N, K, 1.0f, 0.0f,
                                  HerkOp::NoTrans, FillMode::Lower, tri_params);
                    }, warmup, iters);
                }

                // 6. HERK Triangle + CUDA Graph (SM120 optimization)
                {
                    CutlassParams graph_params;
                    graph_params.precision = prec;
                    graph_params.config = cfg;
                    graph_params.herk_strategy = HerkStrategy::TriangleAware;
                    graph_params.triangle_config = tri_cfg;
                    graph_params.triangle_config.use_cuda_graph = true;
                    ms[5] = benchmark_ms([&]() {
                        gemm.HERK(d_A, d_HC, N, K, 1.0f, 0.0f,
                                  HerkOp::NoTrans, FillMode::Lower, graph_params);
                    }, warmup, iters);
                }

                // 7-10. Batched benchmarks (when batch > 1)
                if (batch > 1) {
                    // 7. Batched Standard GEMM
                    {
                        cudaMemset(d_C_bat, 0, 2 * size_C * batch * sizeof(__half));
                        ms[6] = benchmark_ms([&]() {
                            gemm.GEMM_batched(d_A_bat, d_B_bat, d_C_bat, M, N, K, batch,
                                              1.0f, 0.0f, ComplexMode::Standard, params);
                        }, warmup, iters);
                    }

                    // 8. Batched HERK Baseline
                    {
                        cudaMemset(d_HC_bat, 0, 2 * packed_elems * batch * sizeof(__half));
                        ms[7] = benchmark_ms([&]() {
                            gemm.HERK_batched(d_A_bat, d_HC_bat, N, K, batch,
                                              1.0f, 0.0f, HerkOp::NoTrans, FillMode::Lower, params);
                        }, warmup, iters);
                    }

                    // 9. Batched HERK Triangle
                    {
                        CutlassParams tri_params;
                        tri_params.precision = prec;
                        tri_params.config = cfg;
                        tri_params.herk_strategy = HerkStrategy::TriangleAware;
                        tri_params.triangle_config = tri_cfg;
                        cudaMemset(d_HC_bat, 0, 2 * packed_elems * batch * sizeof(__half));
                        ms[8] = benchmark_ms([&]() {
                            gemm.HERK_batched(d_A_bat, d_HC_bat, N, K, batch,
                                              1.0f, 0.0f, HerkOp::NoTrans, FillMode::Lower, tri_params);
                        }, warmup, iters);
                    }

                    // 10. Batched HERK Triangle + CUDA Graph
                    {
                        CutlassParams graph_params;
                        graph_params.precision = prec;
                        graph_params.config = cfg;
                        graph_params.herk_strategy = HerkStrategy::TriangleAware;
                        graph_params.triangle_config = tri_cfg;
                        graph_params.triangle_config.use_cuda_graph = true;
                        cudaMemset(d_HC_bat, 0, 2 * packed_elems * batch * sizeof(__half));
                        ms[9] = benchmark_ms([&]() {
                            gemm.HERK_batched(d_A_bat, d_HC_bat, N, K, batch,
                                              1.0f, 0.0f, HerkOp::NoTrans, FillMode::Lower, graph_params);
                        }, warmup, iters);
                    }
                }

                printf("%8d", N);
                for (int c = 0; c < num_ops; ++c) {
                    if (ms[c] < 0)
                        printf("  %10s", "N/A");
                    else
                        printf("  %10.2f", ms[c]);
                }
                printf("\n");

                for (int c = 0; c < num_ops; ++c) last_ms[c] = ms[c];
                last_N = N;
            }

            // TFLOPS row for the largest completed size
            if (last_N > 0) {
                int M = last_N, N_val = last_N, K = last_N;
                printf("%8s", "TFLOPS");
                for (int c = 0; c < num_ops; ++c) {
                    if (last_ms[c] < 0) {
                        printf("  %10s", "N/A");
                        continue;
                    }
                    double flops;
                    int sub_gemms;
                    if (c <= 2) {
                        // Standard, Hermitian, Gram: 4 sub-GEMMs
                        sub_gemms = 4;
                        flops = (double)sub_gemms * 2.0 * M * N_val * K;
                    } else if (c == 3) {
                        // HERK BL: 3 sub-GEMMs (FP8), 4 (FP6/FP4)
                        sub_gemms = (prec == ComputePrecision::FP8_E4M3) ? 3 : 4;
                        flops = (double)sub_gemms * 2.0 * M * N_val * K;
                    } else if (c <= 5) {
                        // HERK Tri and Tri+Graph: use baseline FLOPs as reference
                        int baseline_sub = (prec == ComputePrecision::FP8_E4M3) ? 3 : 4;
                        flops = (double)baseline_sub * 2.0 * M * N_val * K;
                    } else if (c == 6) {
                        // Batched GEMM: 4 sub-GEMMs x batch
                        sub_gemms = 4;
                        flops = (double)sub_gemms * 2.0 * M * N_val * K * batch;
                    } else {
                        // Batched HERK / Tri / Tri+G: baseline sub-GEMMs x batch
                        int baseline_sub = (prec == ComputePrecision::FP8_E4M3) ? 3 : 4;
                        flops = (double)baseline_sub * 2.0 * M * N_val * K * batch;
                    }
                    double tflops = (last_ms[c] > 0) ? (flops / (last_ms[c] * 1e-3) / 1e12) : 0.0;
                    printf("  %10.1f", tflops);
                }
                printf("\n");
            }
        }
    }

    // ---- Summary ----
    printf("\n--- Summary ---\n");
    printf("Theoretical dense peak (non-sparse, %d SMs x %d MHz):\n", sm_count, sysinfo.clock_mhz);
    printf("  FP8:  ~%6.1f TFLOPS  (2048 ops/cycle/SM)\n", sysinfo.peak_fp8_tflops);
    printf("  FP6:  ~%6.1f TFLOPS  (2048 ops/cycle/SM, same as FP8)\n", sysinfo.peak_fp6_tflops);
    printf("  FP4:  ~%6.1f TFLOPS  (4096 ops/cycle/SM, 2x FP8)\n", sysinfo.peak_fp4_tflops);
    printf("Configs tested: %zu\n", configs.size());
    printf("GEMM/Gram: 4 sub-GEMMs (4M algorithm)\n");
    printf("HERK BL: 3 sub-GEMMs (FP8), 4 sub-GEMMs (FP6/FP4)\n");
    printf("HERK Tri: 2 triangular + 1 full (all precisions)\n");
    printf("Tri+Graph: HERK Tri with CUDA graph capture/replay (all precisions, SM120 optimization)\n");
    printf("All HERK output: packed lower triangle, N*(N+1)/2 complex elements\n");
    printf("Timing: warmup/iters adaptive (5/20 for N<=512, 3/10 for N<=2048, 2/5 for N>2048)\n");
    if (batch > 1) {
        printf("Batch count: %d\n", batch);
        printf("Bat.GEMM: batched standard GEMM (4 sub-GEMMs x batch, loop)\n");
        printf("Bat.HERK: batched HERK baseline (3 or 4 sub-GEMMs x batch)\n");
        printf("Bat.Tri:  batched HERK triangle-aware (baseline FLOPs reference)\n");
        printf("Bat.Tri+G: batched HERK triangle + CUDA graph (baseline FLOPs reference)\n");
    }

    // ---- Cleanup ----
    cudaFree(d_A);  cudaFree(d_B);  cudaFree(d_C);
    cudaFree(d_Ar); cudaFree(d_Ai);
    cudaFree(d_GCr); cudaFree(d_GCi);
    cudaFree(d_GPr); cudaFree(d_GPi);
    cudaFree(d_HC);
    if (d_A_bat)  cudaFree(d_A_bat);
    if (d_B_bat)  cudaFree(d_B_bat);
    if (d_C_bat)  cudaFree(d_C_bat);
    if (d_HC_bat) cudaFree(d_HC_bat);

    return 0;
}
