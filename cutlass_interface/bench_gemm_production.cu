/** @file bench_gemm_production.cu
 * @brief Production GEMM benchmark for beamformer and dedisp workloads.
 *
 * Benchmarks complex planar batched GEMM at production problem sizes via the
 * PIMPL API. Tests both the full gemm() path (with B conversion) and the
 * prepare/execute path (gemm_prepared, gemm_prepared_direct,
 * gemm_prepared_fused) to measure per-call overhead vs sustained throughput.
 *
 * Links against libcutlass_gemm_api.a only (no CUTLASS headers needed).
 *
 * Problem shapes:
 *   Voltage BF:    M=128,  N=4000, K=1664, batch=128-2500
 *   Dedisp TCFDD:  M=32-256, N=256-4096, K=256-2048, batch=varies
 *
 * Usage:
 *   bench_gemm_production [suite]  [key=value ...]
 *
 *   suite: voltbf, dedisp, all (default: all)
 *
 *   Options:
 *     runs=N       Number of timed runs per measurement (default: 10)
 *     warmup=N     Warmup iterations (default: 5)
 *     bestof=N     Repeat full measurement N times, take best (default: 3)
 *     fused=1      Also benchmark batch-fused M path (default: 0)
 *     direct=1     Also benchmark direct kernel path (default: 0)
 *     tile=auto|herk|gemm|widen  TileConfig override (default: auto)
 *     tune=true     Run GEMM autotuning before each problem (default: false)
 *     tune_verb=N   GEMM strategy tune verbosity 0-3 (default: 1)
 */

#include "cutlass_gemm_api.h"
#include "system_info.hpp"

using cutlass_gemm_api::CutlassComplexGemm;
using cutlass_gemm_api::ComputePrecision;
using cutlass_gemm_api::OutputPrecision;
using cutlass_gemm_api::TileConfig;
using cutlass_gemm_api::GemmMode;

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <vector>
#include <string>
#include <cfloat>
#include <algorithm>
#include <random>
#ifdef _OPENMP
#include <omp.h>
#endif

#define CHECK_CUDA(x) do { \
    auto err = (x); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(err)); \
        exit(1); \
    } \
} while(0)

// ========================================================================
// Problem shape definitions
// ========================================================================

struct ProblemShape {
    const char* name;
    int M, N, K, batch;
};

// Voltage beamformer: B (weights) is uniform across batches (prepare/execute path)
// M = n_time (time samples), N = n_beam, K = n_ant, batch = n_ch * n_pol * n_time_inner
static const ProblemShape voltbf_problems[] = {
    // M=64 (n_time=64): short integration
    {"VoltBF_64x4000x1664_b128",     64, 4000, 1664, 128},
    {"VoltBF_64x4000x1664_b256",     64, 4000, 1664, 256},
    {"VoltBF_64x4000x1664_b512",     64, 4000, 1664, 512},
    {"VoltBF_64x4000x1664_b1024",    64, 4000, 1664, 1024},
    {"VoltBF_64x4000x1664_b2500",    64, 4000, 1664, 2500},
    // M=128 (n_time=128): baseline integration
    {"VoltBF_128x4000x1664_b128",   128, 4000, 1664, 128},
    {"VoltBF_128x4000x1664_b256",   128, 4000, 1664, 256},
    {"VoltBF_128x4000x1664_b512",   128, 4000, 1664, 512},
    {"VoltBF_128x4000x1664_b1024",  128, 4000, 1664, 1024},
    {"VoltBF_128x4000x1664_b2500",  128, 4000, 1664, 2500},
    // M=256 (n_time=256): long integration
    {"VoltBF_256x4000x1664_b128",   256, 4000, 1664, 128},
    {"VoltBF_256x4000x1664_b256",   256, 4000, 1664, 256},
    {"VoltBF_256x4000x1664_b512",   256, 4000, 1664, 512},
    {"VoltBF_256x4000x1664_b1024",  256, 4000, 1664, 1024},
    {"VoltBF_256x4000x1664_b2500",  256, 4000, 1664, 2500},
};

// Dedisp TCFDD: B (phasors) varies per batch, prepare/execute still useful
static const ProblemShape dedisp_problems[] = {
    {"Dedisp_32x512x512_b257",      32,  512,  512,  257},
    {"Dedisp_32x1024x512_b257",     32,  1024, 512,  257},
    {"Dedisp_32x2048x512_b257",     32,  2048, 512,  257},
    {"Dedisp_32x4096x512_b257",     32,  4096, 512,  257},
    {"Dedisp_64x1024x1024_b257",    64,  1024, 1024, 257},
    {"Dedisp_128x2048x1024_b257",   128, 2048, 1024, 257},
    {"Dedisp_256x4096x2048_b257",   256, 4096, 2048, 257},
    {"Dedisp_32x512x512_b2049",     32,  512,  512,  2049},
    {"Dedisp_64x1024x512_b2049",    64,  1024, 512,  2049},
};

// ========================================================================
// Benchmark helpers
// ========================================================================

/// Compute complex GEMM FLOPs: 8*M*N*K per batch element (4M decomposition)
static double gemm_flops(int M, int N, int K, int batch) {
    return 8.0 * M * N * K * batch;
}

/// External I/O bytes: read A(2MK) + B(2NK) + write C(2MN), all FP16
static double gemm_io_bytes(int M, int N, int K, int batch) {
    return 4.0 * batch * ((int64_t)M * K + (int64_t)N * K + (int64_t)M * N);
}

/// Arithmetic intensity (FLOP/byte)
static double gemm_ai(int M, int N, int K) {
    // Per batch element: 8MNK / (4(MK + NK + MN))
    double flop = 8.0 * M * N * K;
    double bytes = 4.0 * ((int64_t)M * K + (int64_t)N * K + (int64_t)M * N);
    return flop / bytes;
}

/// Memory required for a batch of planar complex GEMM
static int64_t gemm_memory_bytes(int M, int N, int K, int batch) {
    int64_t sA = (int64_t)M * K * batch * 2 * sizeof(__half);  // A_re + A_im
    int64_t sB = (int64_t)N * K * batch * 2 * sizeof(__half);  // B_re + B_im
    int64_t sC = (int64_t)M * N * batch * 2 * sizeof(float);   // C_re + C_im (FP32 output)
    return sA + sB + sC;
}

struct BenchResult {
    float best_ms;
    double tflops;
    double pct_peak;
    double ms_per_item;
    double gb_s;
};

static BenchResult bench_gemm(
    CutlassComplexGemm& api,
    const __half* d_A_re, const __half* d_A_im,
    const __half* d_B_re, const __half* d_B_im,
    float* d_C_re, float* d_C_im,
    int M, int N, int K, int batch,
    double peak_tflops,
    int warmup, int runs, int bestof,
    const char* label,
    bool tune_first = true)
{
    // Run autotuning on the first call if requested (tune=true on first warmup)
    bool did_tune = false;
    // Warmup
    for (int i = 0; i < warmup; ++i) {
        bool do_tune = (tune_first && !did_tune);
        int ret = api.gemm(d_A_re, d_A_im, d_B_re, d_B_im,
                           d_C_re, d_C_im, M, N, K, batch,
                           ComputePrecision::FP8, OutputPrecision::FP32,
                           1.0f, 0.0f, nullptr, do_tune);
        if (do_tune) did_tune = true;
        if (ret != 0) {
            fprintf(stderr, "  [%s] GEMM FAILED on warmup (ret=%d)\n", label, ret);
            return {-1, 0, 0, 0, 0};
        }
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    double flops = gemm_flops(M, N, K, batch);
    double io = gemm_io_bytes(M, N, K, batch);

    float best_ms = FLT_MAX;
    for (int trial = 0; trial < bestof; ++trial) {
        cudaEvent_t start, stop;
        CHECK_CUDA(cudaEventCreate(&start));
        CHECK_CUDA(cudaEventCreate(&stop));

        CHECK_CUDA(cudaEventRecord(start));
        for (int i = 0; i < runs; ++i) {
            api.gemm(d_A_re, d_A_im, d_B_re, d_B_im,
                     d_C_re, d_C_im, M, N, K, batch,
                     ComputePrecision::FP8, OutputPrecision::FP32);
        }
        CHECK_CUDA(cudaEventRecord(stop));
        CHECK_CUDA(cudaEventSynchronize(stop));

        float elapsed_ms;
        CHECK_CUDA(cudaEventElapsedTime(&elapsed_ms, start, stop));
        best_ms = std::min(best_ms, elapsed_ms / runs);

        CHECK_CUDA(cudaEventDestroy(start));
        CHECK_CUDA(cudaEventDestroy(stop));
    }

    BenchResult r;
    r.best_ms = best_ms;
    r.tflops = flops / (best_ms * 1e9);
    r.pct_peak = (peak_tflops > 0) ? (100.0 * r.tflops / peak_tflops) : 0;
    r.ms_per_item = best_ms / batch;
    r.gb_s = io / (best_ms * 1e6);
    return r;
}

static BenchResult bench_gemm_prepared(
    CutlassComplexGemm& api,
    const __half* d_A_re, const __half* d_A_im,
    float* d_C_re, float* d_C_im,
    int M, int N, int K, int batch,
    double peak_tflops,
    int warmup, int runs, int bestof,
    const char* label,
    bool tune_first = true)
{
    // Warmup
    bool did_tune = false;
    for (int i = 0; i < warmup; ++i) {
        bool do_tune = (tune_first && !did_tune);
        int ret = api.gemm_prepared(d_A_re, d_A_im,
                                    d_C_re, d_C_im, M, N, K, batch,
                                    OutputPrecision::FP32,
                                    1.0f, 0.0f, nullptr, do_tune);
        if (do_tune) did_tune = true;
        if (ret != 0) {
            fprintf(stderr, "  [%s] gemm_prepared FAILED on warmup (ret=%d)\n", label, ret);
            return {-1, 0, 0, 0, 0};
        }
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    double flops = gemm_flops(M, N, K, batch);
    double io = gemm_io_bytes(M, N, K, batch);

    float best_ms = FLT_MAX;
    for (int trial = 0; trial < bestof; ++trial) {
        cudaEvent_t start, stop;
        CHECK_CUDA(cudaEventCreate(&start));
        CHECK_CUDA(cudaEventCreate(&stop));

        CHECK_CUDA(cudaEventRecord(start));
        for (int i = 0; i < runs; ++i) {
            api.gemm_prepared(d_A_re, d_A_im,
                              d_C_re, d_C_im, M, N, K, batch,
                              OutputPrecision::FP32);
        }
        CHECK_CUDA(cudaEventRecord(stop));
        CHECK_CUDA(cudaEventSynchronize(stop));

        float elapsed_ms;
        CHECK_CUDA(cudaEventElapsedTime(&elapsed_ms, start, stop));
        best_ms = std::min(best_ms, elapsed_ms / runs);

        CHECK_CUDA(cudaEventDestroy(start));
        CHECK_CUDA(cudaEventDestroy(stop));
    }

    BenchResult r;
    r.best_ms = best_ms;
    r.tflops = flops / (best_ms * 1e9);
    r.pct_peak = (peak_tflops > 0) ? (100.0 * r.tflops / peak_tflops) : 0;
    r.ms_per_item = best_ms / batch;
    r.gb_s = io / (best_ms * 1e6);
    return r;
}

static BenchResult bench_gemm_prepared_direct(
    CutlassComplexGemm& api,
    const __half* d_A_re, const __half* d_A_im,
    float* d_C_re, float* d_C_im,
    int M, int N, int K, int batch,
    double peak_tflops,
    int warmup, int runs, int bestof,
    const char* label)
{
    // Warmup
    for (int i = 0; i < warmup; ++i) {
        int ret = api.gemm_prepared_direct(d_A_re, d_A_im,
                                           d_C_re, d_C_im, M, N, K, batch);
        if (ret != 0) {
            fprintf(stderr, "  [%s] gemm_prepared_direct FAILED on warmup (ret=%d)\n", label, ret);
            return {-1, 0, 0, 0, 0};
        }
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    double flops = gemm_flops(M, N, K, batch);
    double io = gemm_io_bytes(M, N, K, batch);

    float best_ms = FLT_MAX;
    for (int trial = 0; trial < bestof; ++trial) {
        cudaEvent_t start, stop;
        CHECK_CUDA(cudaEventCreate(&start));
        CHECK_CUDA(cudaEventCreate(&stop));

        CHECK_CUDA(cudaEventRecord(start));
        for (int i = 0; i < runs; ++i) {
            api.gemm_prepared_direct(d_A_re, d_A_im,
                                     d_C_re, d_C_im, M, N, K, batch);
        }
        CHECK_CUDA(cudaEventRecord(stop));
        CHECK_CUDA(cudaEventSynchronize(stop));

        float elapsed_ms;
        CHECK_CUDA(cudaEventElapsedTime(&elapsed_ms, start, stop));
        best_ms = std::min(best_ms, elapsed_ms / runs);

        CHECK_CUDA(cudaEventDestroy(start));
        CHECK_CUDA(cudaEventDestroy(stop));
    }

    BenchResult r;
    r.best_ms = best_ms;
    r.tflops = flops / (best_ms * 1e9);
    r.pct_peak = (peak_tflops > 0) ? (100.0 * r.tflops / peak_tflops) : 0;
    r.ms_per_item = best_ms / batch;
    r.gb_s = io / (best_ms * 1e6);
    return r;
}

static BenchResult bench_gemm_prepared_fused(
    CutlassComplexGemm& api,
    const __half* d_A_re, const __half* d_A_im,
    float* d_C_re, float* d_C_im,
    int M, int N, int K, int batch,
    int fuse_factor,
    double peak_tflops,
    int warmup, int runs, int bestof,
    const char* label)
{
    // Warmup
    for (int i = 0; i < warmup; ++i) {
        int ret = api.gemm_prepared_fused(d_A_re, d_A_im,
                                          d_C_re, d_C_im, M, N, K, batch,
                                          fuse_factor, OutputPrecision::FP32);
        if (ret != 0) {
            fprintf(stderr, "  [%s] gemm_prepared_fused FAILED on warmup (ret=%d)\n", label, ret);
            return {-1, 0, 0, 0, 0};
        }
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    double flops = gemm_flops(M, N, K, batch);
    double io = gemm_io_bytes(M, N, K, batch);

    float best_ms = FLT_MAX;
    for (int trial = 0; trial < bestof; ++trial) {
        cudaEvent_t start, stop;
        CHECK_CUDA(cudaEventCreate(&start));
        CHECK_CUDA(cudaEventCreate(&stop));

        CHECK_CUDA(cudaEventRecord(start));
        for (int i = 0; i < runs; ++i) {
            api.gemm_prepared_fused(d_A_re, d_A_im,
                                    d_C_re, d_C_im, M, N, K, batch,
                                    fuse_factor, OutputPrecision::FP32);
        }
        CHECK_CUDA(cudaEventRecord(stop));
        CHECK_CUDA(cudaEventSynchronize(stop));

        float elapsed_ms;
        CHECK_CUDA(cudaEventElapsedTime(&elapsed_ms, start, stop));
        best_ms = std::min(best_ms, elapsed_ms / runs);

        CHECK_CUDA(cudaEventDestroy(start));
        CHECK_CUDA(cudaEventDestroy(stop));
    }

    BenchResult r;
    r.best_ms = best_ms;
    r.tflops = flops / (best_ms * 1e9);
    r.pct_peak = (peak_tflops > 0) ? (100.0 * r.tflops / peak_tflops) : 0;
    r.ms_per_item = best_ms / batch;
    r.gb_s = io / (best_ms * 1e6);
    return r;
}

// ========================================================================
// Suite runners
// ========================================================================

static void print_result(const char* problem_name, const char* method,
                         const BenchResult& r) {
    if (r.best_ms < 0) {
        printf("  %-42s %-20s  FAILED\n", problem_name, method);
        return;
    }
    printf("  %-42s %-20s %8.2f ms  %6.1f TFLOPS  %5.1f%% TC  %8.3f ms/item  %7.1f GB/s\n",
           problem_name, method, r.best_ms, r.tflops, r.pct_peak, r.ms_per_item, r.gb_s);
}

static void run_suite(
    const char* suite_name,
    const ProblemShape* problems, int num_problems,
    bool b_uniform,   // true if B is the same across batch elements
    double peak_tflops,
    int warmup, int runs, int bestof,
    bool bench_fused, bool bench_direct,
    TileConfig tile_config,
    GemmMode gemm_mode = GemmMode::Auto,
    bool tune = true)
{
    printf("\n");
    printf("======================================================================\n");
    printf("  %s Suite  (%d problems, B %s across batches)\n",
           suite_name, num_problems, b_uniform ? "uniform" : "varies");
    printf("======================================================================\n\n");

    // Table header
    printf("  %-42s %-20s %8s  %6s        %5s     %8s     %7s\n",
           "Problem", "Method", "Time", "TFLOPS", "TC%", "ms/item", "BW");
    printf("  %-42s %-20s %8s  %6s        %5s     %8s     %7s\n",
           std::string(42, '-').c_str(), std::string(20, '-').c_str(),
           "--------", "------", "-----", "--------", "-------");

    CutlassComplexGemm api;
    if (tile_config != TileConfig::Auto) {
        api.set_tile_config(tile_config);
    }
    if (gemm_mode != GemmMode::Auto) {
        api.set_gemm_mode(gemm_mode);
    }

    for (int p = 0; p < num_problems; ++p) {
        const auto& prob = problems[p];

        // Check memory
        int64_t total_mem_needed = gemm_memory_bytes(prob.M, prob.N, prob.K, prob.batch);
        // Internal FP8 buffers: ~2x for A conversion, 2x for B conversion, 2x for workspace
        total_mem_needed += (int64_t)prob.M * prob.K * prob.batch * 2;  // FP8 A
        total_mem_needed += (int64_t)prob.N * prob.K * prob.batch * 4;  // FP8 B + neg interleaved

        size_t free_mem, total_mem;
        CHECK_CUDA(cudaMemGetInfo(&free_mem, &total_mem));

        if ((double)total_mem_needed > 0.90 * free_mem) {
            printf("  %-42s  SKIPPED (need %.1f GB, have %.1f GB free)\n",
                   prob.name, total_mem_needed / 1e9, free_mem / 1e9);
            continue;
        }

        double ai = gemm_ai(prob.M, prob.N, prob.K);
        printf("  --- %s  M=%d N=%d K=%d batch=%d  AI=%.0f FLOP/byte ---\n",
               prob.name, prob.M, prob.N, prob.K, prob.batch, ai);

        // Allocate data
        int64_t sA = (int64_t)prob.M * prob.K * prob.batch;
        int64_t sB = (int64_t)prob.N * prob.K * prob.batch;
        int64_t sC = (int64_t)prob.M * prob.N * prob.batch;

        __half *d_A_re, *d_A_im, *d_B_re, *d_B_im;
        float *d_C_re, *d_C_im;

        CHECK_CUDA(cudaMalloc(&d_A_re, sA * sizeof(__half)));
        CHECK_CUDA(cudaMalloc(&d_A_im, sA * sizeof(__half)));
        CHECK_CUDA(cudaMalloc(&d_B_re, sB * sizeof(__half)));
        CHECK_CUDA(cudaMalloc(&d_B_im, sB * sizeof(__half)));
        CHECK_CUDA(cudaMalloc(&d_C_re, sC * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_C_im, sC * sizeof(float)));

        // Fill with random FP16 data
        {
            std::vector<__half> h_A(sA), h_B(sB);
            std::mt19937 rng(42);
            std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
            for (int64_t i = 0; i < sA; ++i) h_A[i] = __float2half(dist(rng));
            for (int64_t i = 0; i < sB; ++i) h_B[i] = __float2half(dist(rng));
            CHECK_CUDA(cudaMemcpy(d_A_re, h_A.data(), sA * sizeof(__half), cudaMemcpyHostToDevice));
            CHECK_CUDA(cudaMemcpy(d_A_im, h_A.data(), sA * sizeof(__half), cudaMemcpyHostToDevice));
            CHECK_CUDA(cudaMemcpy(d_B_re, h_B.data(), sB * sizeof(__half), cudaMemcpyHostToDevice));
            CHECK_CUDA(cudaMemcpy(d_B_im, h_B.data(), sB * sizeof(__half), cudaMemcpyHostToDevice));
        }
        CHECK_CUDA(cudaMemset(d_C_re, 0, sC * sizeof(float)));
        CHECK_CUDA(cudaMemset(d_C_im, 0, sC * sizeof(float)));

        // 1. Full GEMM path (includes B conversion each call)
        {
            auto r = bench_gemm(api, d_A_re, d_A_im, d_B_re, d_B_im,
                                d_C_re, d_C_im,
                                prob.M, prob.N, prob.K, prob.batch,
                                peak_tflops, warmup, runs, bestof, "gemm", tune);
            print_result(prob.name, "gemm()", r);
        }

        // 2. Prepare/execute path (B pre-converted)
        if (b_uniform) {
            // For uniform B: prepare with batch_count=1 and use uniform B
            // For non-uniform B: prepare with full batch count
            // Prepare with full batch for apples-to-apples comparison
            api.prepare_b(d_B_re, d_B_im, prob.N, prob.K, prob.batch,
                          ComputePrecision::FP8);

            {
                auto r = bench_gemm_prepared(api, d_A_re, d_A_im,
                                             d_C_re, d_C_im,
                                             prob.M, prob.N, prob.K, prob.batch,
                                             peak_tflops, warmup, runs, bestof,
                                             "prepared", tune);
                print_result(prob.name, "gemm_prepared()", r);
            }

            // 3. Direct kernel path
            if (bench_direct) {
                auto r = bench_gemm_prepared_direct(api, d_A_re, d_A_im,
                                                    d_C_re, d_C_im,
                                                    prob.M, prob.N, prob.K, prob.batch,
                                                    peak_tflops, warmup, runs, bestof,
                                                    "direct");
                print_result(prob.name, "gemm_prepared_direct()", r);
            }

            // 4. Batch-fused M path (fuse_factor=0 means auto)
            if (bench_fused) {
                auto r = bench_gemm_prepared_fused(api, d_A_re, d_A_im,
                                                   d_C_re, d_C_im,
                                                   prob.M, prob.N, prob.K, prob.batch,
                                                   0,  // auto fuse_factor
                                                   peak_tflops, warmup, runs, bestof,
                                                   "fused");
                int ff = (prob.M > 0) ? std::max(1, 256 / prob.M) : 1;
                char label[64];
                snprintf(label, sizeof(label), "fused(ff=%d)", ff);
                print_result(prob.name, label, r);
            }
        } else {
            // B varies per batch -- prepare with full batch
            api.prepare_b(d_B_re, d_B_im, prob.N, prob.K, prob.batch,
                          ComputePrecision::FP8);

            auto r = bench_gemm_prepared(api, d_A_re, d_A_im,
                                         d_C_re, d_C_im,
                                         prob.M, prob.N, prob.K, prob.batch,
                                         peak_tflops, warmup, runs, bestof,
                                         "prepared", tune);
            print_result(prob.name, "gemm_prepared()", r);
        }

        // Cleanup
        CHECK_CUDA(cudaFree(d_A_re));
        CHECK_CUDA(cudaFree(d_A_im));
        CHECK_CUDA(cudaFree(d_B_re));
        CHECK_CUDA(cudaFree(d_B_im));
        CHECK_CUDA(cudaFree(d_C_re));
        CHECK_CUDA(cudaFree(d_C_im));
        printf("\n");
    }
}

// ========================================================================
// Main
// ========================================================================

int main(int argc, char* argv[]) {
    enum Suite { ALL, VOLTBF, DEDISP };
    Suite suite = ALL;
    int runs = 10;
    int warmup = 5;
    int bestof = 3;
    bool bench_fused = false;
    bool bench_direct = false;
    bool tune = true;
    int tune_verb = -1;  // -1 = not set (use default)
    TileConfig tile_config = TileConfig::Auto;
    GemmMode gemm_mode = GemmMode::Auto;

    for (int i = 1; i < argc; ++i) {
        std::string arg(argv[i]);

        // Suite names (positional)
        if (arg == "voltbf" || arg == "volt" || arg == "bf")
            { suite = VOLTBF; continue; }
        if (arg == "dedisp" || arg == "tcfdd")
            { suite = DEDISP; continue; }
        if (arg == "all")
            { suite = ALL; continue; }
        if (arg == "--help" || arg == "-h") {
            printf("Usage: %s [suite] [key=value ...]\n", argv[0]);
            printf("  suite: all (default), voltbf, dedisp\n");
            printf("  runs=N warmup=N bestof=N fused=0|1 direct=0|1\n");
            printf("  tile=auto|herk|gemm|wide\n");
            printf("  gemm_mode=auto|direct|4m  GEMM dispatch mode (default: auto)\n");
            printf("  tune=false    Disable GEMM autotuning (default: on)\n");
            printf("  tune_verb=N   GEMM strategy tune verbosity 0-3 (default: 1)\n");
            return 0;
        }

        // Key=value options
        auto eq = arg.find('=');
        if (eq != std::string::npos) {
            std::string key = arg.substr(0, eq);
            std::string val = arg.substr(eq + 1);
            if (key == "runs")      runs = atoi(val.c_str());
            else if (key == "warmup")  warmup = atoi(val.c_str());
            else if (key == "bestof")  bestof = atoi(val.c_str());
            else if (key == "fused")   bench_fused = (val == "1" || val == "true");
            else if (key == "direct")  bench_direct = (val == "1" || val == "true");
            else if (key == "tune")    tune = !(val == "0" || val == "false");
            else if (key == "tune_verb") tune_verb = atoi(val.c_str());
            else if (key == "tile") {
                if (val == "herk")     tile_config = TileConfig::HerkOptimal;
                else if (val == "gemm")  tile_config = TileConfig::GemmOptimal;
                else if (val == "wide")  tile_config = TileConfig::WideN;
                else if (val == "small") tile_config = TileConfig::SmallTile;
                else                     tile_config = TileConfig::Auto;
            }
            else if (key == "gemm_mode") {
                if (val == "direct")     gemm_mode = GemmMode::ForceDirect;
                else if (val == "4m")    gemm_mode = GemmMode::Force4M;
                else                     gemm_mode = GemmMode::Auto;
            }
        }
    }

    // Print system info
    CutlassComplexGemm::print_build_info();
    auto sysinfo = cutlass_complex::query_system_info();
    double peak_tflops = sysinfo.peak_fp8_tflops;

    // Set GEMM tune verbosity if requested
    if (tune_verb >= 0) {
        CutlassComplexGemm api_tmp;
        api_tmp.set_gemm_tune_verbosity(tune_verb);
    } else if (tune) {
        // Default to verbosity 2 (sweep detail) when tune=true
        CutlassComplexGemm api_tmp;
        api_tmp.set_gemm_tune_verbosity(2);
    }

    printf("\n=== Production GEMM Benchmark ===\n");
    printf("  Timing: %d warmup, %d runs, best of %d\n", warmup, runs, bestof);
    printf("  Peak FP8: %.1f TFLOPS (%s)\n", peak_tflops, sysinfo.gpu_name.c_str());
    if (tile_config != TileConfig::Auto)
        printf("  TileConfig: override active\n");
    if (tune) printf("  GEMM autotuning: enabled\n");
    if (bench_fused) printf("  Batch-fused M: enabled\n");
    if (bench_direct) printf("  Direct kernel: enabled\n");

    // Run suites
    if (suite == ALL || suite == VOLTBF) {
        int n = sizeof(voltbf_problems) / sizeof(voltbf_problems[0]);
        run_suite("Voltage Beamformer", voltbf_problems, n,
                  /*b_uniform=*/true, peak_tflops, warmup, runs, bestof,
                  bench_fused, bench_direct, tile_config, gemm_mode, tune);
    }

    if (suite == ALL || suite == DEDISP) {
        int n = sizeof(dedisp_problems) / sizeof(dedisp_problems[0]);
        run_suite("Dedisp TCFDD", dedisp_problems, n,
                  /*b_uniform=*/false, peak_tflops, warmup, runs, bestof,
                  bench_fused, bench_direct, tile_config, gemm_mode, tune);
    }

    // Summary table
    printf("\n=== Summary ===\n");
    printf("  Complex GEMM via FP8 4M decomposition, FP32 output\n");
    printf("  Prepare/execute path: B pre-converted once, only A converted per call\n");
    printf("  Direct kernel: single-launch conjugate-permutation PTX kernel\n");
    if (bench_fused) {
        printf("  Batch-fused M: fuses consecutive batch elements along M dimension\n");
        printf("    Increases M_fused for better wave quantization (fewer wasted tiles)\n");
    }
    printf("Done.\n");

    return 0;
}
