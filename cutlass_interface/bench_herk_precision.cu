/** @file bench_herk_precision.cu
 * @brief Multi-precision HERK benchmark via PIMPL API.
 *
 * Benchmarks batched HERK at configurable input/compute/output precisions.
 * Supports FP16 and INT4 input, FP8/FP6/FP4 compute, FP16/FP32 output.
 * Reports wall-clock time, effective TFLOPS, and achieved bandwidth.
 *
 * Links against libcutlass_gemm_api.a only (no CUTLASS headers needed).
 *
 * Usage:
 *   bench_herk_precision N K batch [options]
 *   Options:
 *     input=fp16|int4         Input precision (default: fp16)
 *     compute=fp8|fp6|fp4     Compute precision (default: fp8)
 *     output=fp16|fp32        Output precision (default: fp16)
 *     direct=auto|1|0         HERK dispatch mode (default: auto)
 *     runs=N                  Number of timed runs (default: 10)
 *     warmup=N                Warmup iterations (default: 5)
 *     bestof=N                Repeat full measurement N times, take best (default: 3)
 */

#include "cutlass_gemm_api.h"
#include "system_info.hpp"
using cutlass_gemm_api::InputPrecision;
using cutlass_gemm_api::ComputePrecision;
using cutlass_gemm_api::OutputPrecision;
using cutlass_gemm_api::OutputFormat;
using cutlass_gemm_api::HerkMode;

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

/// Encode a complex value as INT4 sign-magnitude (1 byte per element).
/// High nibble = Re, low nibble = Im. Bit 3 = sign, bits 2:0 = magnitude.
static uint8_t encode_int4(int re, int im) {
    uint8_t re_nibble = (re < 0) ? (0x8 | (-re & 0x7)) : (re & 0x7);
    uint8_t im_nibble = (im < 0) ? (0x8 | (-im & 0x7)) : (im & 0x7);
    return (re_nibble << 4) | im_nibble;
}

static const char* input_name(InputPrecision p) {
    return p == InputPrecision::INT4 ? "INT4" : "FP16";
}
static const char* compute_name(ComputePrecision p) {
    switch(p) {
        case ComputePrecision::FP8: return "FP8";
        case ComputePrecision::FP6: return "FP6";
        case ComputePrecision::FP4: return "FP4";
    }
    return "?";
}
static const char* output_name(OutputPrecision p) {
    return p == OutputPrecision::FP32 ? "FP32" : "FP16";
}

int main(int argc, char* argv[]) {
    if (argc < 4) {
        printf("Usage: %s N K batch [input=fp16|int4] [compute=fp8|fp6|fp4] "
               "[output=fp16|fp32] [direct=auto|1|0] [runs=10] [warmup=5] [bestof=3]\n",
               argv[0]);
        return 1;
    }

    int N = atoi(argv[1]);
    int K = atoi(argv[2]);
    int batch = atoi(argv[3]);

    InputPrecision input_prec = InputPrecision::FP16;
    ComputePrecision compute_prec = ComputePrecision::FP8;
    OutputPrecision output_prec = OutputPrecision::FP16;
    int runs = 10;
    int warmup = 5;
    int bestof = 3;
    int direct_mode = -1;  // -1 = auto, 0 = force baseline, 1 = force direct

    for (int i = 4; i < argc; ++i) {
        std::string arg(argv[i]);
        if (arg.find("input=") == 0) {
            std::string v = arg.substr(6);
            if (v == "int4" || v == "INT4") input_prec = InputPrecision::INT4;
            else input_prec = InputPrecision::FP16;
        } else if (arg.find("compute=") == 0) {
            std::string v = arg.substr(8);
            if (v == "fp6" || v == "FP6") compute_prec = ComputePrecision::FP6;
            else if (v == "fp4" || v == "FP4") compute_prec = ComputePrecision::FP4;
            else compute_prec = ComputePrecision::FP8;
        } else if (arg.find("output=") == 0) {
            std::string v = arg.substr(7);
            if (v == "fp32" || v == "FP32") output_prec = OutputPrecision::FP32;
            else output_prec = OutputPrecision::FP16;
        } else if (arg.find("runs=") == 0) {
            runs = atoi(arg.substr(5).c_str());
        } else if (arg.find("warmup=") == 0) {
            warmup = atoi(arg.substr(7).c_str());
        } else if (arg.find("bestof=") == 0) {
            bestof = atoi(arg.substr(7).c_str());
        } else if (arg.find("direct=") == 0) {
            std::string v = arg.substr(7);
            if (v == "1" || v == "force") direct_mode = 1;
            else if (v == "0" || v == "off") direct_mode = 0;
            else direct_mode = -1;
        }
    }

    cutlass_gemm_api::CutlassComplexGemm::print_build_info();

    // Query GPU peak performance for efficiency calculation
    auto sysinfo = cutlass_complex::query_system_info();
    double peak_tflops = sysinfo.peak_fp8_tflops;

    int64_t per_A = (int64_t)N * K;
    int64_t packed = (int64_t)N * (N + 1) / 2;

    // Memory calculation
    int64_t input_bytes, output_elem_size;
    if (input_prec == InputPrecision::INT4) {
        input_bytes = per_A * batch;  // 1 byte per complex element
    } else {
        input_bytes = per_A * batch * 2 * sizeof(__half);  // 2 halves per complex
    }
    output_elem_size = (output_prec == OutputPrecision::FP32) ? sizeof(float) : sizeof(__half);
    int64_t output_bytes = packed * batch * 2 * output_elem_size;

    // Internal buffers estimate
    int64_t internal_fp16 = (input_prec == InputPrecision::INT4) ? per_A * batch * 4 : 0;
    int64_t internal_fp8 = per_A * batch * 2;  // precast buffer
    int64_t total_est = input_bytes + output_bytes + internal_fp16 + internal_fp8;

    size_t free_mem, total_mem;
    cudaMemGetInfo(&free_mem, &total_mem);

    printf("\n=== HERK Benchmark ===\n");
    printf("  N=%d  K=%d  batch=%d\n", N, K, batch);
    printf("  Pipeline: %s -> %s compute -> %s output\n",
           input_name(input_prec), compute_name(compute_prec), output_name(output_prec));
    printf("  Timing: %d warmup, %d runs, best of %d\n", warmup, runs, bestof);

    double flops = 8.0 * N * N * K * batch;
    double tflops_total = flops / 1e12;
    printf("\n  Memory:\n");
    printf("    Input A:       %9.2f MB  (%s)\n", input_bytes / 1e6, input_name(input_prec));
    printf("    Output C:      %9.2f MB  (%s packed triangle)\n", output_bytes / 1e6, output_name(output_prec));
    if (internal_fp16 > 0)
        printf("    Internal FP16: %9.2f MB  (INT4->FP16 conversion)\n", internal_fp16 / 1e6);
    printf("    Internal FP8:  %9.2f MB  (precast buffer)\n", internal_fp8 / 1e6);
    printf("    Estimated:     %9.2f MB  (%.1f%% of %.1f GB free)\n",
           total_est / 1e6, 100.0 * total_est / free_mem, free_mem / 1e9);
    printf("    GPU: %.1f GB free / %.1f GB total\n", free_mem / 1e9, total_mem / 1e9);
    printf("\n  Compute:\n");
    printf("    FLOPs: %.3f TFLOP  (8*N^2*K*batch)\n", tflops_total);
    printf("    AI:    %.0f FLOPs/byte (input only)\n", flops / input_bytes);

    if ((double)total_est > 0.85 * free_mem) {
        printf("\n  WARNING: estimated memory use > 85%% of free -- OOM risk!\n");
    }

    // Allocate input
    void* d_A = nullptr;
    if (input_prec == InputPrecision::INT4) {
        std::vector<uint8_t> h_int4(per_A * batch);
        #pragma omp parallel
        {
            #ifdef _OPENMP
            std::mt19937 rng(42 + omp_get_thread_num());
            #else
            std::mt19937 rng(42);
            #endif
            std::uniform_int_distribution<int> dist(-7, 7);
            #pragma omp for
            for (int64_t i = 0; i < per_A * batch; ++i) {
                int re = dist(rng);
                int im = dist(rng);
                h_int4[i] = encode_int4(re, im);
            }
        }
        CHECK_CUDA(cudaMalloc(&d_A, per_A * batch));
        CHECK_CUDA(cudaMemcpy(d_A, h_int4.data(), per_A * batch, cudaMemcpyHostToDevice));
    } else {
        std::vector<__half> h_fp16(per_A * batch * 2);
        #pragma omp parallel
        {
            #ifdef _OPENMP
            std::mt19937 rng(42 + omp_get_thread_num());
            #else
            std::mt19937 rng(42);
            #endif
            std::uniform_int_distribution<int> dist(-100, 100);
            #pragma omp for
            for (int64_t i = 0; i < per_A * batch * 2; ++i) {
                float v = (float)dist(rng) / 100.0f;
                h_fp16[i] = __float2half(v);
            }
        }
        CHECK_CUDA(cudaMalloc(&d_A, per_A * batch * 2 * sizeof(__half)));
        CHECK_CUDA(cudaMemcpy(d_A, h_fp16.data(), per_A * batch * 2 * sizeof(__half),
                              cudaMemcpyHostToDevice));
    }

    // Allocate output
    void* d_C = nullptr;
    CHECK_CUDA(cudaMalloc(&d_C, packed * batch * 2 * output_elem_size));

    // Create API instance
    cutlass_gemm_api::CutlassComplexGemm api;

    // Set direct mode if specified
    if (direct_mode == 1) {
        api.set_herk_mode(HerkMode::ForceDirect);
        printf("  Direct kernel: FORCED ON\n");
    } else if (direct_mode == 0) {
        api.set_herk_mode(HerkMode::ForceBaseline);
        printf("  Direct kernel: FORCED OFF\n");
    } else {
        printf("  Direct kernel: AUTO\n");
    }

    // Warmup
    printf("\n  Warming up (%d iterations)...\n", warmup);
    for (int i = 0; i < warmup; ++i) {
        CHECK_CUDA(cudaMemset(d_C, 0, packed * batch * 2 * output_elem_size));
        int ret = api.herk(d_A, d_C, N, K, batch, input_prec, compute_prec, output_prec);
        if (ret != 0) {
            printf("  HERK FAILED on warmup iter %d (ret=%d)\n", i, ret);
            CHECK_CUDA(cudaFree(d_A));
            CHECK_CUDA(cudaFree(d_C));
            return 1;
        }
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    printf("  Warmup complete.\n");

    // Timed runs: best of N measurements
    float best_ms = FLT_MAX;
    for (int trial = 0; trial < bestof; ++trial) {
        cudaEvent_t start, stop;
        CHECK_CUDA(cudaEventCreate(&start));
        CHECK_CUDA(cudaEventCreate(&stop));

        CHECK_CUDA(cudaEventRecord(start));
        for (int i = 0; i < runs; ++i) {
            api.herk(d_A, d_C, N, K, batch, input_prec, compute_prec, output_prec);
        }
        CHECK_CUDA(cudaEventRecord(stop));
        CHECK_CUDA(cudaEventSynchronize(stop));

        float elapsed_ms;
        CHECK_CUDA(cudaEventElapsedTime(&elapsed_ms, start, stop));
        float ms_per = elapsed_ms / runs;

        double tflops = flops / (ms_per * 1e9);
        printf("  Trial %d/%d: %.2f ms  %.1f TFLOPS  (%.3f ms/item)\n",
               trial + 1, bestof, ms_per, tflops, ms_per / batch);

        best_ms = std::min(best_ms, ms_per);

        CHECK_CUDA(cudaEventDestroy(start));
        CHECK_CUDA(cudaEventDestroy(stop));
    }

    // Compute final metrics
    double best_tflops = flops / (best_ms * 1e9);
    double io_bytes = (double)(input_bytes + output_bytes);
    double gb_s = io_bytes / (best_ms * 1e6);

    printf("\n=== Result ===\n");
    printf("  Pipeline: %s -> %s -> %s\n", input_name(input_prec), compute_name(compute_prec), output_name(output_prec));
    printf("  N=%d  K=%d  batch=%d\n", N, K, batch);
    printf("  Best: %.2f ms  %.1f TFLOPS  %.1f GB/s  (%.3f ms/item)\n",
           best_ms, best_tflops, gb_s, best_ms / batch);
    if (peak_tflops > 0) {
        printf("  Peak efficiency: %.1f%%  (vs %.1f TFLOPS %s FP8 peak)\n",
               100.0 * best_tflops / peak_tflops, peak_tflops, sysinfo.gpu_name.c_str());
    }

    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_C));

    printf("Done.\n");
    return 0;
}
