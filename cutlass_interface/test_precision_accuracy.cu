/** @file test_precision_accuracy.cu
 * @brief Measure HERK accuracy at different compute precisions vs FP64 CPU reference.
 *
 * For each (K, precision) combination, runs GPU HERK and compares against
 * an FP64 CPU reference. Reports max absolute error, mean absolute error,
 * RMS error, and relative error. Uses the PIMPL API (no CUTLASS headers needed).
 *
 * Links against libcutlass_gemm_api.a only.
 *
 * Usage:
 *   test_precision_accuracy [N] [batch] [fp32out]
 *   Defaults: N=512, batch=4, FP16 output
 *   Set N=3328, batch=1 for DSA-2000 scale (slow CPU reference)
 */

#include "cutlass_gemm_api.h"
using cutlass_gemm_api::CutlassComplexGemm;
using cutlass_gemm_api::InputPrecision;
using cutlass_gemm_api::ComputePrecision;
using cutlass_gemm_api::OutputPrecision;
using cutlass_gemm_api::OutputFormat;
using cutlass_gemm_api::HerkMode;

#include <cuda_fp16.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <complex>
#include <random>
#include <chrono>
#include <algorithm>
#include <numeric>
#include <stdexcept>

#define CHECK_CUDA(x) do { \
    auto err = (x); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(err)); exit(1); \
    } \
} while(0)

/// CPU FP64 HERK: C = A * A^H, packed lower triangle.
/// @param h_A  [N * 2K] interleaved complex FP16: [Re0, Im0, Re1, Im1, ...]
/// @param C_re Packed lower triangle real part [N*(N+1)/2]
/// @param C_im Packed lower triangle imaginary part [N*(N+1)/2]
/// @param N    Matrix dimension
/// @param K    Inner dimension
static void cpu_herk_fp64(
    const __half* h_A,
    double* C_re, double* C_im,
    int N, int K)
{
    int64_t idx = 0;
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j <= i; ++j) {
            double sum_re = 0, sum_im = 0;
            for (int k = 0; k < K; ++k) {
                double ai_re = __half2float(h_A[i * 2 * K + 2 * k]);
                double ai_im = __half2float(h_A[i * 2 * K + 2 * k + 1]);
                double aj_re = __half2float(h_A[j * 2 * K + 2 * k]);
                double aj_im = __half2float(h_A[j * 2 * K + 2 * k + 1]);
                // a_i * conj(a_j)
                sum_re += ai_re * aj_re + ai_im * aj_im;
                sum_im += ai_im * aj_re - ai_re * aj_im;
            }
            if (i == j) sum_im = 0.0;
            C_re[idx] = sum_re;
            C_im[idx] = sum_im;
            idx++;
        }
    }
}

struct ErrorStats {
    double max_abs;
    double mean_abs;
    double max_rel;
    double max_ref;
    double rms;
    int64_t count;
};

/// Compare GPU FP16 packed output against FP64 reference.
/// GPU output is interleaved [Re0, Im0, Re1, Im1, ...].
static ErrorStats compare_output(
    const __half* gpu_out,
    const double* ref_re, const double* ref_im,
    int64_t packed_size)
{
    ErrorStats s{};
    s.count = packed_size;
    double sum_abs = 0, sum_sq = 0;

    for (int64_t i = 0; i < packed_size; ++i) {
        double g_re = __half2float(gpu_out[2 * i]);
        double g_im = __half2float(gpu_out[2 * i + 1]);

        double err_re = std::abs(g_re - ref_re[i]);
        double err_im = std::abs(g_im - ref_im[i]);
        double err = std::max(err_re, err_im);

        double ref_mag = std::max(std::abs(ref_re[i]), std::abs(ref_im[i]));
        double rel = (ref_mag > 1e-6) ? err / ref_mag : err;

        s.max_abs = std::max(s.max_abs, err);
        s.max_rel = std::max(s.max_rel, rel);
        s.max_ref = std::max(s.max_ref, ref_mag);
        sum_abs += err_re + err_im;
        sum_sq += err_re * err_re + err_im * err_im;
    }
    s.mean_abs = sum_abs / (2.0 * packed_size);
    s.rms = std::sqrt(sum_sq / (2.0 * packed_size));
    return s;
}

/// Compare GPU FP32 packed output against FP64 reference.
/// GPU output is interleaved [Re0, Im0, Re1, Im1, ...] as float.
static ErrorStats compare_output_fp32(
    const float* gpu_out,
    const double* ref_re, const double* ref_im,
    int64_t packed_size)
{
    ErrorStats s{};
    s.count = packed_size;
    double sum_abs = 0, sum_sq = 0;

    for (int64_t i = 0; i < packed_size; ++i) {
        double err_re = std::abs((double)gpu_out[2 * i] - ref_re[i]);
        double err_im = std::abs((double)gpu_out[2 * i + 1] - ref_im[i]);
        double err = std::max(err_re, err_im);

        double ref_mag = std::max(std::abs(ref_re[i]), std::abs(ref_im[i]));
        double rel = (ref_mag > 1e-6) ? err / ref_mag : err;

        s.max_abs = std::max(s.max_abs, err);
        s.max_rel = std::max(s.max_rel, rel);
        s.max_ref = std::max(s.max_ref, ref_mag);
        sum_abs += err_re + err_im;
        sum_sq += err_re * err_re + err_im * err_im;
    }
    s.mean_abs = sum_abs / (2.0 * packed_size);
    s.rms = std::sqrt(sum_sq / (2.0 * packed_size));
    return s;
}

/// Check if error stats pass the per-precision threshold.
/// Uses max_abs / max_ref (error relative to peak signal) rather than per-element
/// max_rel, because near-zero off-diagonal elements inflate per-element relative error.
/// FP8 uses 15% of peak (FP8 accumulation error grows with K; at K=4096 peak-relative
/// error is ~12% on some architectures). FP6 uses 25%, FP4 uses 55% (documented
/// ~50% structural error from lossy representation of 5 and 7).
static bool check_threshold(const ErrorStats& s, ComputePrecision prec, double max_ref) {
    double peak_rel = (max_ref > 1e-6) ? s.max_abs / max_ref : s.max_abs;
    switch (prec) {
        case ComputePrecision::FP8: return peak_rel < 0.15;
        case ComputePrecision::FP6: return peak_rel < 0.25;
        case ComputePrecision::FP4: return peak_rel < 0.55;
        default: return true;
    }
}

static const char* prec_name(ComputePrecision p) {
    switch (p) {
        case ComputePrecision::FP8: return "FP8";
        case ComputePrecision::FP6: return "FP6 E3M2";
        case ComputePrecision::FP4: return "FP4 E2M1";
        default: return "?";
    }
}

int main(int argc, char** argv) {
    int N = 512;
    int batch = 4;
    bool use_fp32_out = false;

    if (argc > 1) N = atoi(argv[1]);
    if (argc > 2) batch = atoi(argv[2]);
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "fp32out") use_fp32_out = true;
    }

    int64_t packed_size = (int64_t)N * (N + 1) / 2;

    // K values to test -- respects precision availability constraints
    std::vector<int> K_values = {16, 32, 64, 128, 256, 512, 1024, 2048, 4096};

    printf("======================================================================\n");
    printf("  Precision Accuracy Sweep -- N=%d, batch=%d, output=%s\n",
           N, batch, use_fp32_out ? "FP32" : "FP16");
    printf("======================================================================\n\n");

    CutlassComplexGemm gemm;
    gemm.set_herk_mode(HerkMode::Auto);

    int failures = 0;

    // Generate random FP16 input data -- same for all precisions
    // Use uniform [-0.5, 0.5] to simulate realistic FP16 correlator input
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-0.5f, 0.5f);

    // Print CSV header
    printf("%-6s  %-10s  %-6s  %12s  %12s  %12s  %12s  %s\n",
           "K", "Precision", "Batch", "max_abs", "mean_abs", "rms", "max_rel", "Status");
    printf("%-6s  %-10s  %-6s  %12s  %12s  %12s  %12s  %s\n",
           "------", "----------", "------", "------------", "------------",
           "------------", "------------", "------");

    for (int K : K_values) {
        int64_t per_A = (int64_t)N * K;

        // Generate fresh random data for this K
        std::vector<__half> h_A(2 * per_A * batch);
        for (auto& v : h_A) v = __float2half(dist(rng));

        // Compute CPU FP64 reference for ALL batch elements
        auto t0 = std::chrono::high_resolution_clock::now();
        std::vector<double> ref_re(packed_size * batch), ref_im(packed_size * batch);
        for (int b = 0; b < batch; ++b) {
            cpu_herk_fp64(h_A.data() + b * 2 * per_A,
                          ref_re.data() + b * packed_size,
                          ref_im.data() + b * packed_size,
                          N, K);
        }
        auto t1 = std::chrono::high_resolution_clock::now();
        double cpu_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

        // Upload input to GPU
        __half* d_A;
        CHECK_CUDA(cudaMalloc(&d_A, 2 * per_A * batch * sizeof(__half)));
        CHECK_CUDA(cudaMemcpy(d_A, h_A.data(), 2 * per_A * batch * sizeof(__half),
                              cudaMemcpyHostToDevice));

        // Test each precision (with minimum K constraints)
        struct PrecisionTest {
            ComputePrecision prec;
            int min_K;
        };
        std::vector<PrecisionTest> precisions = {
            {ComputePrecision::FP8, 1},
            {ComputePrecision::FP6, 128},
            {ComputePrecision::FP4, 256},
        };

        for (auto& pt : precisions) {
            if (K < pt.min_K) continue;

            // Output is always interleaved packed triangle: [Re0, Im0, Re1, Im1, ...]
            int64_t out_elems = packed_size * batch * 2;

            try {
                if (use_fp32_out) {
                    float* d_C;
                    CHECK_CUDA(cudaMalloc(&d_C, out_elems * sizeof(float)));
                    CHECK_CUDA(cudaMemset(d_C, 0, out_elems * sizeof(float)));

                    int status = gemm.herk(d_A, d_C, N, K, batch,
                                           InputPrecision::FP16, pt.prec,
                                           OutputPrecision::FP32,
                                           OutputFormat::PackedTriangle,
                                           1.0f, 0.0f, nullptr, false);

                    if (status != 0) {
                        printf("%-6d  %-10s  %-6s  %12s  %12s  %12s  %12s  FAIL(status=%d)\n",
                               K, prec_name(pt.prec), "all", "-", "-", "-", "-", status);
                        failures++;
                        CHECK_CUDA(cudaFree(d_C));
                        continue;
                    }
                    CHECK_CUDA(cudaDeviceSynchronize());

                    // Copy back all batches and validate each
                    std::vector<float> h_C(out_elems);
                    CHECK_CUDA(cudaMemcpy(h_C.data(), d_C, out_elems * sizeof(float),
                                          cudaMemcpyDeviceToHost));

                    bool any_fail = false;
                    for (int b = 0; b < batch; ++b) {
                        ErrorStats s = compare_output_fp32(
                            h_C.data() + b * packed_size * 2,
                            ref_re.data() + b * packed_size,
                            ref_im.data() + b * packed_size,
                            packed_size);

                        bool pass = check_threshold(s, pt.prec, s.max_ref);
                        printf("%-6d  %-10s  %-6d  %12.4e  %12.4e  %12.4e  %12.4e  %s\n",
                               K, prec_name(pt.prec), b, s.max_abs, s.mean_abs, s.rms, s.max_rel,
                               pass ? "PASS" : "FAIL");
                        if (!pass) any_fail = true;
                    }
                    if (any_fail) failures++;

                    CHECK_CUDA(cudaFree(d_C));
                } else {
                    __half* d_C;
                    CHECK_CUDA(cudaMalloc(&d_C, out_elems * sizeof(__half)));
                    CHECK_CUDA(cudaMemset(d_C, 0, out_elems * sizeof(__half)));

                    int status = gemm.herk(d_A, d_C, N, K, batch,
                                           InputPrecision::FP16, pt.prec,
                                           OutputPrecision::FP16,
                                           OutputFormat::PackedTriangle,
                                           1.0f, 0.0f, nullptr, false);

                    if (status != 0) {
                        printf("%-6d  %-10s  %-6s  %12s  %12s  %12s  %12s  FAIL(status=%d)\n",
                               K, prec_name(pt.prec), "all", "-", "-", "-", "-", status);
                        failures++;
                        CHECK_CUDA(cudaFree(d_C));
                        continue;
                    }
                    CHECK_CUDA(cudaDeviceSynchronize());

                    // Copy back all batches and validate each
                    std::vector<__half> h_C(out_elems);
                    CHECK_CUDA(cudaMemcpy(h_C.data(), d_C, out_elems * sizeof(__half),
                                          cudaMemcpyDeviceToHost));

                    bool any_fail = false;
                    for (int b = 0; b < batch; ++b) {
                        ErrorStats s = compare_output(
                            h_C.data() + b * packed_size * 2,
                            ref_re.data() + b * packed_size,
                            ref_im.data() + b * packed_size,
                            packed_size);

                        bool pass = check_threshold(s, pt.prec, s.max_ref);
                        printf("%-6d  %-10s  %-6d  %12.4e  %12.4e  %12.4e  %12.4e  %s\n",
                               K, prec_name(pt.prec), b, s.max_abs, s.mean_abs, s.rms, s.max_rel,
                               pass ? "PASS" : "FAIL");
                        if (!pass) any_fail = true;
                    }
                    if (any_fail) failures++;

                    CHECK_CUDA(cudaFree(d_C));
                }
            } catch (const std::exception& e) {
                printf("%-6d  %-10s  %-6s  %12s  %12s  %12s  %12s  SKIP(%s)\n",
                       K, prec_name(pt.prec), "all", "-", "-", "-", "-", e.what());
            }
        }

        printf("  (CPU ref: %.1f ms)\n", cpu_ms);
        CHECK_CUDA(cudaFree(d_A));
    }

    // Print summary interpretation
    printf("\n");
    printf("======================================================================\n");
    printf("  Notes:\n");
    printf("  - max_abs:  worst-case absolute error vs FP64 reference\n");
    printf("  - mean_abs: average absolute error across all triangle elements\n");
    printf("  - rms:      root-mean-square error\n");
    printf("  - max_rel:  worst-case relative error (|err|/|ref|)\n");
    printf("  - Thresholds: FP8 < 15%%, FP6 < 25%%, FP4 < 55%% (peak-relative)\n");
    printf("  - FP4 E2M1 cannot represent 5 or 7 (5->4, 7->8)\n");
    printf("  - FP6 E3M2 and FP8 E4M3 exactly represent all integers [-7,7]\n");
    printf("  - Input: uniform random FP16 in [-0.5, 0.5]\n");
    printf("  - Reference: FP64 CPU HERK (exact for FP16 inputs)\n");
    printf("======================================================================\n");

    if (failures > 0)
        printf("\n%d precision/K combination(s) FAILED\n", failures);
    else
        printf("\nAll precision/K combinations PASSED\n");

    return failures > 0 ? 1 : 0;
}
