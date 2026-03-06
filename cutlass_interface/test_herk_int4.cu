// test_herk_int4.cu — End-to-end test and benchmark of the PIMPL HERK API
//
// Tests: unified herk() with INT4 and FP16 input against CPU reference.
// Benchmarks: INT4->FP16 and INT4->FP32 HERK with bandwidth/roofline reporting.
// Supports key=value CLI args consistent with example_usage_sm100.cu.
//
#include "cutlass_gemm_api.h"
using cutlass_gemm_api::InputPrecision;
using cutlass_gemm_api::ComputePrecision;
using cutlass_gemm_api::OutputPrecision;
using cutlass_gemm_api::OutputFormat;
using cutlass_gemm_api::HerkMode;
#include "system_info.hpp"
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cmath>
#include <cstring>
#include <vector>
#include <complex>
#include <string>
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

// ---- INT4 sign-magnitude encoding ----

// Encode a complex value as INT4 sign-magnitude byte:
//   high nibble = Re (bit 3=sign, bits 2:0=magnitude)
//   low nibble  = Im (same encoding)
static uint8_t encode_int4(int re, int im) {
    uint8_t re_nibble = (re < 0) ? (0x8 | (-re & 0x7)) : (re & 0x7);
    uint8_t im_nibble = (im < 0) ? (0x8 | (-im & 0x7)) : (im & 0x7);
    return (re_nibble << 4) | im_nibble;
}

// ---- CPU reference ----

// CPU reference HERK: C = A * A^H, packed lower triangle
// A is [N x K] complex, C is [N*(N+1)/2] complex
static void cpu_herk(
    const std::vector<std::complex<double>>& A,
    std::vector<std::complex<double>>& C,
    int N, int K)
{
    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j <= i; ++j) {
            std::complex<double> sum(0, 0);
            for (int k = 0; k < K; ++k) {
                sum += A[i * K + k] * std::conj(A[j * K + k]);
            }
            int64_t idx = static_cast<int64_t>(i) * (i + 1) / 2 + j;
            if (i == j)
                C[idx] = std::complex<double>(sum.real(), 0.0);
            else
                C[idx] = sum;
        }
    }
}

// ---- Bandwidth helpers ----

// External I/O bytes for INT4 HERK (1 byte per complex input element)
static double herk_external_bytes_int4(int64_t N, int64_t K, int batch, bool fp32_out) {
    double input = (double)batch * N * K;                                      // INT4: 1 byte per complex
    double output = (double)batch * N * ((int64_t)N + 1) * (fp32_out ? 4 : 2);  // packed triangle (Re+Im)
    return input + output;
}

// ---- Benchmark result printer ----

struct BenchResult {
    double ms;
    double tflops;
    double bw_gbs;
    double io_intensity;  // FLOPs/byte
    double tc_util;       // % of peak FP8
};

static BenchResult compute_bench_result(
    float elapsed_ms, int N, int K, int batch,
    bool fp32_out, const cutlass_complex::SystemInfo& sysinfo)
{
    BenchResult r;
    r.ms = elapsed_ms;
    double flops = 8.0 * N * N * K * batch;
    r.tflops = flops / (r.ms * 1e9);
    double ext_bytes = herk_external_bytes_int4(N, K, batch, fp32_out);
    r.bw_gbs = ext_bytes / (r.ms * 1e-3) / 1e9;
    r.io_intensity = flops / ext_bytes;
    r.tc_util = (sysinfo.peak_fp8_tflops > 0) ? 100.0 * r.tflops / sysinfo.peak_fp8_tflops : 0;
    return r;
}

static void print_bench_result(const char* label, const BenchResult& r, int N, int K, int batch) {
    printf("  %-16s N=%-5d K=%-5d batch=%-4d  %7.2f ms  %6.1f TFLOPS  %5.1f%% TC  %6.1f GB/s  %5.0f FLOPs/B  (%.3f ms/item)\n",
           label, N, K, batch, r.ms, r.tflops, r.tc_util, r.bw_gbs, r.io_intensity, r.ms / batch);
}

// ---- CLI parsing ----

enum class RunMode { Test, Bench, Both };

static RunMode parse_mode(int argc, char** argv) {
    for (int i = 1; i < argc; ++i) {
        std::string arg(argv[i]);
        if (arg.substr(0, 5) == "mode=") {
            std::string val = arg.substr(5);
            if (val == "test") return RunMode::Test;
            if (val == "bench") return RunMode::Bench;
            return RunMode::Both;
        }
    }
    return RunMode::Both;
}

// Returns: -1 = full strategy autotune, 0-3 = kernel-level tune verbosity, -2 = not set
static int parse_tune(int argc, char** argv) {
    for (int i = 1; i < argc; ++i) {
        std::string arg(argv[i]);
        if (arg.substr(0, 5) == "tune=") {
            std::string val = arg.substr(5);
            if (val == "true")  return -1;   // strategy autotune
            if (val == "false") return -2;    // not set
            return std::atoi(val.c_str());    // kernel-level verbosity
        }
    }
    return -2;  // not set
}

static HerkMode parse_herk_mode(int argc, char** argv) {
    for (int i = 1; i < argc; ++i) {
        std::string arg(argv[i]);
        if (arg.substr(0, 7) == "direct=") {
            std::string val = arg.substr(7);
            if (val == "1" || val == "true")  return HerkMode::ForceDirect;
            if (val == "0" || val == "false") return HerkMode::ForceBaseline;
            return HerkMode::Auto;
        }
        // Backward compat
        if (arg.substr(0, 6) == "fused=") {
            std::string val = arg.substr(6);
            if (val == "1" || val == "true")  return HerkMode::ForceDirect;
            if (val == "0" || val == "false") return HerkMode::ForceBaseline;
            return HerkMode::Auto;
        }
    }
    return HerkMode::Auto;
}

static int parse_strategy_verbosity(int argc, char** argv) {
    for (int i = 1; i < argc; ++i) {
        std::string arg(argv[i]);
        if (arg.find("strategy_verbosity=") == 0) {
            return std::atoi(arg.c_str() + 19);
        }
    }
    return -1;  // not set (use default)
}

static void print_memory_info(int N, int K, int batch) {
    int64_t per_A = (int64_t)N * K;
    int64_t packed = (int64_t)N * (N + 1) / 2;

    int64_t input_bytes = per_A * batch;                       // INT4 input
    int64_t fp16_bytes = per_A * batch * 4;                    // INT4->FP16 interleaved
    int64_t fp8_bytes = per_A * batch * 2;                     // FP16->FP8 precast
    int64_t out_fp16_bytes = packed * batch * 2 * 2;           // FP16 packed triangle output
    int64_t out_fp32_bytes = packed * batch * 2 * 4;           // FP32 packed triangle output
    int64_t total = input_bytes + fp16_bytes + fp8_bytes + out_fp32_bytes;

    size_t free_mem, total_mem;
    cudaMemGetInfo(&free_mem, &total_mem);

    printf("Memory breakdown:\n");
    printf("  A_int4 input:     %7.2f MB\n", input_bytes / 1e6);
    printf("  A_fp16 internal:  %7.2f MB\n", fp16_bytes / 1e6);
    printf("  A_fp8 precast:    %7.2f MB\n", fp8_bytes / 1e6);
    printf("  C_fp16 output:    %7.2f MB\n", out_fp16_bytes / 1e6);
    printf("  C_fp32 output:    %7.2f MB\n", out_fp32_bytes / 1e6);
    printf("  Estimated total:  %7.2f MB  (%.1f%% of %.1f GB free)\n",
           total / 1e6, 100.0 * total / free_mem, free_mem / 1e9);
    printf("  GPU memory:       %.1f GB free / %.1f GB total\n",
           free_mem / 1e9, total_mem / 1e9);

    double flops = 8.0 * N * N * K * batch;
    printf("  Compute:          %.1f TFLOP  (8*N^2*K*batch)\n", flops / 1e12);
}

static void print_help(const char* prog) {
    printf("Usage: %s [N] [K] [batch] [key=value ...]\n", prog);
    printf("\n");
    printf("End-to-end test and benchmark of the PIMPL HERK API with INT4\n");
    printf("sign-magnitude complex input (DSA-2000 QC format).\n");
    printf("\n");
    printf("Positional arguments:\n");
    printf("  N            Matrix dimension (default: 64)\n");
    printf("  K            Inner dimension (default: 128)\n");
    printf("  batch        Batch count (default: 2)\n");
    printf("\n");
    printf("Key=value arguments:\n");
    printf("  mode=<test|bench|both>       Run mode (default: both)\n");
    printf("  tune=<true|false|0-3>        true=strategy autotune, 0-3=kernel tune verbosity (default: false)\n");
    printf("  direct=<auto|0|1>            Direct HERK: auto=K-adaptive, 1=force, 0=off (default: auto)\n");
    printf("  strategy_verbosity=<0-3>     Strategy autotune output verbosity (default: 1)\n");
    printf("\n");
    printf("Modes:\n");
    printf("  test         Correctness tests only (INT4->FP16, INT4->FP32, FP16 input) vs CPU reference\n");
    printf("  bench        Benchmarks only (skip CPU reference), with bandwidth/roofline reporting\n");
    printf("  both         Correctness tests, then benchmarks if N >= 128 (default)\n");
    printf("\n");
    printf("INT4 format: 1 byte per complex element, high nibble = Re, low nibble = Im,\n");
    printf("             sign-magnitude encoding, range [-7, +7].\n");
    printf("\n");
    printf("Examples:\n");
    printf("  %s 64 128 2                          # correctness test (small)\n", prog);
    printf("  %s 1664 1664 16 tune=true            # autotune + correctness + bench\n", prog);
    printf("  %s 3328 4096 16 mode=bench           # benchmark only, no CPU reference\n", prog);
    printf("  %s 1664 128 128 tune=true direct=auto mode=bench\n", prog);
}

int main(int argc, char* argv[]) {
    // ---- Help flag ----
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

    // ---- Parse positional args (skip key=value) ----
    std::vector<char*> pos;
    for (int i = 1; i < argc; ++i) {
        if (strchr(argv[i], '=') == nullptr &&
            strcmp(argv[i], "--help") != 0 && strcmp(argv[i], "-h") != 0) {
            pos.push_back(argv[i]);
        }
    }

    int N = 64, K = 128, batch = 2;
    if (pos.size() > 0) N = atoi(pos[0]);
    if (pos.size() > 1) K = atoi(pos[1]);
    if (pos.size() > 2) batch = atoi(pos[2]);

    // ---- Parse key=value args ----
    RunMode run_mode = parse_mode(argc, argv);
    int tune_level = parse_tune(argc, argv);
    HerkMode herk_mode = parse_herk_mode(argc, argv);
    int strat_verb = parse_strategy_verbosity(argc, argv);

    bool do_strategy_tune = (tune_level == -1);
    int kernel_tune_level = (tune_level >= 0) ? tune_level : 0;
    // When strategy tuning, default strategy verbosity to 2 if not explicitly set
    if (do_strategy_tune && strat_verb < 0) strat_verb = 2;

    bool do_test = (run_mode == RunMode::Test || run_mode == RunMode::Both);
    bool do_bench = (run_mode == RunMode::Bench || run_mode == RunMode::Both);

    // ---- System info ----
    auto sysinfo = cutlass_complex::query_system_info();
    cutlass_complex::print_system_info(sysinfo);
    cutlass_complex::print_build_config();

    if (do_bench && !do_test)
        printf("\nHERK INT4 Benchmark: N=%d K=%d batch=%d\n", N, K, batch);
    else
        printf("\nTesting HERK INT4 API: N=%d K=%d batch=%d\n", N, K, batch);

    // Print parsed settings
    printf("Settings: mode=%s  tune=%s  direct=%s",
           (run_mode == RunMode::Test) ? "test" : (run_mode == RunMode::Bench) ? "bench" : "both",
           do_strategy_tune ? "true" : (tune_level >= 0) ? std::to_string(tune_level).c_str() : "false",
           (herk_mode == HerkMode::ForceDirect) ? "1" : (herk_mode == HerkMode::ForceBaseline) ? "0" : "auto");
    if (strat_verb >= 0) printf("  strategy_verbosity=%d", strat_verb);
    printf("\n");

    int64_t per_A = (int64_t)N * K;
    int64_t packed = (int64_t)N * (N + 1) / 2;

    print_memory_info(N, K, batch);

    // ---- Create single API instance ----
    cutlass_gemm_api::CutlassComplexGemm api;
    api.set_herk_mode(herk_mode);
    if (kernel_tune_level > 0) api.set_kernel_tune_verbosity(kernel_tune_level);
    if (strat_verb >= 0) api.set_strategy_tune_verbosity(strat_verb);

    int failures = 0;

    // ---- Generate random INT4 data ----
    std::vector<uint8_t> h_int4(per_A * batch);
    std::vector<std::complex<double>> h_A_cpu;
    std::vector<std::complex<double>> h_C_ref;

    std::mt19937 rng(42);
    std::uniform_int_distribution<int> dist(-7, 7);
    if (do_test) {
        h_A_cpu.resize(per_A * batch);
        h_C_ref.resize(packed * batch, {0, 0});
        for (int64_t i = 0; i < per_A * batch; ++i) {
            int re = dist(rng);
            int im = dist(rng);
            h_int4[i] = encode_int4(re, im);
            h_A_cpu[i] = std::complex<double>(re, im);
        }

        printf("\nComputing CPU reference (N=%d K=%d batch=%d)...\n", N, K, batch);
        #pragma omp parallel for schedule(dynamic)
        for (int b = 0; b < batch; ++b) {
            std::vector<std::complex<double>> A_batch(h_A_cpu.begin() + b * per_A,
                                                       h_A_cpu.begin() + (b + 1) * per_A);
            std::vector<std::complex<double>> C_batch(packed, {0, 0});
            cpu_herk(A_batch, C_batch, N, K);
            for (int64_t i = 0; i < packed; ++i)
                h_C_ref[b * packed + i] = C_batch[i];
        }
    } else {
        for (int64_t i = 0; i < per_A * batch; ++i) {
            int re = dist(rng);
            int im = dist(rng);
            h_int4[i] = encode_int4(re, im);
        }
    }

    // Upload INT4 data to GPU
    uint8_t* d_int4;
    CHECK_CUDA(cudaMalloc(&d_int4, per_A * batch));
    CHECK_CUDA(cudaMemcpy(d_int4, h_int4.data(), per_A * batch, cudaMemcpyHostToDevice));

    // ---- Correctness tests ----
    if (do_test) {
    // ---- Test 1: herk (INT4 input, FP16 output) ----
    printf("\n--- Test 1: herk (INT4 input, FP16 output) ---\n");
    {
        __half* d_C;
        CHECK_CUDA(cudaMalloc(&d_C, packed * batch * 2 * sizeof(__half)));
        CHECK_CUDA(cudaMemset(d_C, 0, packed * batch * 2 * sizeof(__half)));

        int ret = api.herk(d_int4, d_C, N, K, batch,
                           InputPrecision::INT4, ComputePrecision::FP8,
                           OutputPrecision::FP16,
                           OutputFormat::PackedTriangle,
                           1.0f, 0.0f, nullptr, do_strategy_tune);
        CHECK_CUDA(cudaDeviceSynchronize());

        if (ret != 0) {
            printf("  FAIL: herk_batched_int4 returned %d\n", ret);
            failures++;
        } else {
            std::vector<__half> h_C(packed * batch * 2);
            CHECK_CUDA(cudaMemcpy(h_C.data(), d_C, packed * batch * 2 * sizeof(__half),
                                  cudaMemcpyDeviceToHost));

            double max_err = 0, max_ref = 0;
            #pragma omp parallel for reduction(max:max_err, max_ref)
            for (int64_t i = 0; i < packed * batch; ++i) {
                double re = __half2float(h_C[i * 2]);
                double im = __half2float(h_C[i * 2 + 1]);
                double re_ref = h_C_ref[i].real();
                double im_ref = h_C_ref[i].imag();
                max_err = std::max(max_err, std::abs(re - re_ref));
                max_err = std::max(max_err, std::abs(im - im_ref));
                max_ref = std::max(max_ref, std::abs(re_ref));
                max_ref = std::max(max_ref, std::abs(im_ref));
            }

            double rel_err = (max_ref > 0) ? max_err / max_ref : max_err;
            double tol = std::max(1.0, K * 0.02);
            bool pass = max_err < tol;
            printf("  max_ref=%.1f  max_err=%.2f  rel=%.4f  tol=%.1f  %s\n",
                   max_ref, max_err, rel_err, tol, pass ? "PASS" : "FAIL");
            if (!pass) {
                failures++;
                for (int64_t i = 0; i < std::min((int64_t)5, packed); ++i) {
                    double re = __half2float(h_C[i * 2]);
                    double im = __half2float(h_C[i * 2 + 1]);
                    printf("    [%ld] got=(%.1f,%.1f) ref=(%.1f,%.1f)\n", i,
                           re, im, h_C_ref[i].real(), h_C_ref[i].imag());
                }
            }
        }
        CHECK_CUDA(cudaFree(d_C));
    }

    // ---- Test 2: herk (INT4 input, FP32 output) ----
    printf("\n--- Test 2: herk (INT4 input, FP32 output) ---\n");
    {
        float* d_C;
        CHECK_CUDA(cudaMalloc(&d_C, packed * batch * 2 * sizeof(float)));
        CHECK_CUDA(cudaMemset(d_C, 0, packed * batch * 2 * sizeof(float)));

        int ret = api.herk(d_int4, d_C, N, K, batch,
                           InputPrecision::INT4, ComputePrecision::FP8,
                           OutputPrecision::FP32,
                           OutputFormat::PackedTriangle,
                           1.0f, 0.0f, nullptr, do_strategy_tune);
        CHECK_CUDA(cudaDeviceSynchronize());

        if (ret != 0) {
            printf("  FAIL: herk_batched_int4_fp32 returned %d\n", ret);
            failures++;
        } else {
            std::vector<float> h_C(packed * batch * 2);
            CHECK_CUDA(cudaMemcpy(h_C.data(), d_C, packed * batch * 2 * sizeof(float),
                                  cudaMemcpyDeviceToHost));

            double max_err = 0, max_ref = 0;
            #pragma omp parallel for reduction(max:max_err, max_ref)
            for (int64_t i = 0; i < packed * batch; ++i) {
                double re = h_C[i * 2];
                double im = h_C[i * 2 + 1];
                double re_ref = h_C_ref[i].real();
                double im_ref = h_C_ref[i].imag();
                max_err = std::max(max_err, std::abs(re - re_ref));
                max_err = std::max(max_err, std::abs(im - im_ref));
                max_ref = std::max(max_ref, std::abs(re_ref));
                max_ref = std::max(max_ref, std::abs(im_ref));
            }

            double rel_err = (max_ref > 0) ? max_err / max_ref : max_err;
            double tol = std::max(1.0, K * 0.02);
            bool pass = max_err < tol;
            printf("  max_ref=%.1f  max_err=%.2f  rel=%.4f  tol=%.1f  %s\n",
                   max_ref, max_err, rel_err, tol, pass ? "PASS" : "FAIL");
            if (!pass) {
                failures++;
                for (int64_t i = 0; i < std::min((int64_t)5, packed); ++i) {
                    printf("    [%ld] got=(%.1f,%.1f) ref=(%.1f,%.1f)\n", i,
                           h_C[i * 2], h_C[i * 2 + 1],
                           h_C_ref[i].real(), h_C_ref[i].imag());
                }
            }
        }
        CHECK_CUDA(cudaFree(d_C));
    }

    // ---- Test 3: herk (FP16 interleaved input, FP16 output) ----
    printf("\n--- Test 3: herk (FP16 input, FP16 output) ---\n");
    {
        std::vector<__half> h_A_fp16(per_A * batch * 2);
        #pragma omp parallel for
        for (int64_t i = 0; i < per_A * batch; ++i) {
            h_A_fp16[i * 2]     = __float2half(static_cast<float>(h_A_cpu[i].real()));
            h_A_fp16[i * 2 + 1] = __float2half(static_cast<float>(h_A_cpu[i].imag()));
        }

        __half* d_A;
        CHECK_CUDA(cudaMalloc(&d_A, per_A * batch * 2 * sizeof(__half)));
        CHECK_CUDA(cudaMemcpy(d_A, h_A_fp16.data(), per_A * batch * 2 * sizeof(__half),
                              cudaMemcpyHostToDevice));

        __half* d_C;
        CHECK_CUDA(cudaMalloc(&d_C, packed * batch * 2 * sizeof(__half)));
        CHECK_CUDA(cudaMemset(d_C, 0, packed * batch * 2 * sizeof(__half)));

        int ret = api.herk(d_A, d_C, N, K, batch,
                           InputPrecision::FP16, ComputePrecision::FP8,
                           OutputPrecision::FP16,
                           OutputFormat::PackedTriangle,
                           1.0f, 0.0f, nullptr, do_strategy_tune);
        CHECK_CUDA(cudaDeviceSynchronize());

        if (ret != 0) {
            printf("  FAIL: herk_batched returned %d\n", ret);
            failures++;
        } else {
            std::vector<__half> h_C(packed * batch * 2);
            CHECK_CUDA(cudaMemcpy(h_C.data(), d_C, packed * batch * 2 * sizeof(__half),
                                  cudaMemcpyDeviceToHost));

            double max_err = 0, max_ref = 0;
            #pragma omp parallel for reduction(max:max_err, max_ref)
            for (int64_t i = 0; i < packed * batch; ++i) {
                double re = __half2float(h_C[i * 2]);
                double im = __half2float(h_C[i * 2 + 1]);
                double re_ref = h_C_ref[i].real();
                double im_ref = h_C_ref[i].imag();
                max_err = std::max(max_err, std::abs(re - re_ref));
                max_err = std::max(max_err, std::abs(im - im_ref));
                max_ref = std::max(max_ref, std::abs(re_ref));
                max_ref = std::max(max_ref, std::abs(im_ref));
            }

            double rel_err = (max_ref > 0) ? max_err / max_ref : max_err;
            double tol = std::max(1.0, K * 0.02);
            bool pass = max_err < tol;
            printf("  max_ref=%.1f  max_err=%.2f  rel=%.4f  tol=%.1f  %s\n",
                   max_ref, max_err, rel_err, tol, pass ? "PASS" : "FAIL");
            if (!pass) failures++;
        }

        CHECK_CUDA(cudaFree(d_A));
        CHECK_CUDA(cudaFree(d_C));
    }
    } // end do_test

    // ---- Benchmarks ----
    if (do_bench && N >= 128) {
        // Adaptive warmup/iteration counts
        int warmup = (N <= 512) ? 5 : (N <= 2048) ? 3 : 2;
        int iters  = (N <= 512) ? 20 : (N <= 2048) ? 10 : 5;

        BenchResult r_fp16, r_fp32;

        // ---- Benchmark: INT4 -> FP16 output ----
        printf("\n--- Benchmark: herk (INT4, FP16 output) ---\n");
        printf("  warmup=%d  iters=%d\n", warmup, iters);
        {
            __half* d_C_bench;
            CHECK_CUDA(cudaMalloc(&d_C_bench, packed * batch * 2 * sizeof(__half)));

            for (int i = 0; i < warmup; ++i) {
                CHECK_CUDA(cudaMemset(d_C_bench, 0, packed * batch * 2 * sizeof(__half)));
                api.herk(d_int4, d_C_bench, N, K, batch,
                         InputPrecision::INT4, ComputePrecision::FP8,
                         OutputPrecision::FP16,
                         OutputFormat::PackedTriangle,
                         1.0f, 0.0f, nullptr, do_strategy_tune);
            }
            CHECK_CUDA(cudaDeviceSynchronize());

            cudaEvent_t start, stop;
            CHECK_CUDA(cudaEventCreate(&start));
            CHECK_CUDA(cudaEventCreate(&stop));

            CHECK_CUDA(cudaEventRecord(start));
            for (int i = 0; i < iters; ++i) {
                api.herk(d_int4, d_C_bench, N, K, batch,
                         InputPrecision::INT4, ComputePrecision::FP8,
                         OutputPrecision::FP16);
            }
            CHECK_CUDA(cudaEventRecord(stop));
            CHECK_CUDA(cudaEventSynchronize(stop));

            float elapsed_ms;
            CHECK_CUDA(cudaEventElapsedTime(&elapsed_ms, start, stop));
            r_fp16 = compute_bench_result(elapsed_ms / iters, N, K, batch, false, sysinfo);
            print_bench_result("INT4->FP16:", r_fp16, N, K, batch);

            CHECK_CUDA(cudaEventDestroy(start));
            CHECK_CUDA(cudaEventDestroy(stop));
            CHECK_CUDA(cudaFree(d_C_bench));
        }

        // ---- Benchmark: INT4 -> FP32 output ----
        printf("\n--- Benchmark: herk (INT4, FP32 output) ---\n");
        {
            float* d_C_fp32;
            CHECK_CUDA(cudaMalloc(&d_C_fp32, packed * batch * 2 * sizeof(float)));

            for (int i = 0; i < warmup; ++i) {
                CHECK_CUDA(cudaMemset(d_C_fp32, 0, packed * batch * 2 * sizeof(float)));
                api.herk(d_int4, d_C_fp32, N, K, batch,
                         InputPrecision::INT4, ComputePrecision::FP8,
                         OutputPrecision::FP32,
                         OutputFormat::PackedTriangle,
                         1.0f, 0.0f, nullptr, do_strategy_tune);
            }
            CHECK_CUDA(cudaDeviceSynchronize());

            cudaEvent_t start2, stop2;
            CHECK_CUDA(cudaEventCreate(&start2));
            CHECK_CUDA(cudaEventCreate(&stop2));

            CHECK_CUDA(cudaEventRecord(start2));
            for (int i = 0; i < iters; ++i) {
                api.herk(d_int4, d_C_fp32, N, K, batch,
                         InputPrecision::INT4, ComputePrecision::FP8,
                         OutputPrecision::FP32);
            }
            CHECK_CUDA(cudaEventRecord(stop2));
            CHECK_CUDA(cudaEventSynchronize(stop2));

            float elapsed_fp32;
            CHECK_CUDA(cudaEventElapsedTime(&elapsed_fp32, start2, stop2));
            r_fp32 = compute_bench_result(elapsed_fp32 / iters, N, K, batch, true, sysinfo);
            print_bench_result("INT4->FP32:", r_fp32, N, K, batch);
            printf("  FP32 vs FP16 overhead: %.1f%%\n",
                   100.0 * (r_fp32.ms - r_fp16.ms) / r_fp16.ms);

            CHECK_CUDA(cudaEventDestroy(start2));
            CHECK_CUDA(cudaEventDestroy(stop2));
            CHECK_CUDA(cudaFree(d_C_fp32));
        }

        // ---- Roofline Summary ----
        double ridge = cutlass_complex::ridge_point(sysinfo.peak_fp8_tflops, sysinfo.memory_bw_gbs);
        printf("\n--- Roofline Analysis ---\n");
        printf("  Peak FP8:      %6.1f TFLOPS\n", sysinfo.peak_fp8_tflops);
        printf("  Memory BW:     %6.1f GB/s (theoretical)\n", sysinfo.memory_bw_gbs);
        printf("  Ridge point:   %5.0f FLOPs/byte\n", ridge);
        printf("  INT4->FP16:    %6.1f TFLOPS  %5.1f%% TC util  %6.1f GB/s  %5.0f FLOPs/B  (%s)\n",
               r_fp16.tflops, r_fp16.tc_util, r_fp16.bw_gbs, r_fp16.io_intensity,
               (r_fp16.io_intensity >= ridge) ? "compute-bound" : "memory-bound");
        printf("  INT4->FP32:    %6.1f TFLOPS  %5.1f%% TC util  %6.1f GB/s  %5.0f FLOPs/B  (%s)\n",
               r_fp32.tflops, r_fp32.tc_util, r_fp32.bw_gbs, r_fp32.io_intensity,
               (r_fp32.io_intensity >= ridge) ? "compute-bound" : "memory-bound");
    } else if (do_bench && N < 128) {
        printf("\nSkipping benchmarks (N=%d < 128)\n", N);
    }

    CHECK_CUDA(cudaFree(d_int4));
    printf("\nDone. %s\n", failures > 0 ? "FAILURES detected!" : "All tests passed.");
    return failures > 0 ? 1 : 0;
}
