/**
 * @file test_gemm_api.cu
 * @brief Correctness tests for the CUTLASS complex GEMM PIMPL API.
 *
 * Tests: unified gemm() with FP8/FP6/FP4 compute and FP16/FP32 output,
 *        prepare_b + gemm_prepared, unified herk() with FP32 output.
 *
 * Only includes cutlass_gemm_api.h -- proves PIMPL isolation.
 *
 * Usage: test_gemm_api [test_name]
 *   test_name: gemm_fp16, gemm_fp32, gemm_fp6, gemm_fp4,
 *              prepare_execute, prepare_fp6, prepare_fp4,
 *              herk_fp32, herk_beta, herk_single_batch,
 *              gemm_non_aligned, gemm_fp32_output, herk_large_k,
 *              gemm_alpha_beta, herk_direct, herk_baseline,
 *              herk_direct_vs_baseline, herk_fp6, herk_fp4,
 *              herk_fp4_lossy, herk_k_not_aligned,
 *              gemm_k_not_aligned, gemm_continuous,
 *              herk_continuous, gemm_int4_fp32,
 *              gemm_prepared_int4, gemm_power_int4,
 *              power_4m_vs_direct, gemm_direct_int4,
 *              all (default)
 */

#include "cutlass_gemm_api.h"
using cutlass_gemm_api::InputPrecision;
using cutlass_gemm_api::ComputePrecision;
using cutlass_gemm_api::OutputPrecision;
using cutlass_gemm_api::OutputFormat;
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <complex>
#include <cstring>
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

// CPU reference: C = alpha * A * B + beta * C (complex, planar)
static void cpu_gemm(
    const std::vector<std::complex<double>>& A,
    const std::vector<std::complex<double>>& B,
    std::vector<std::complex<double>>& C,
    int M, int N, int K, float alpha, float beta)
{
    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            std::complex<double> sum(0, 0);
            for (int k = 0; k < K; ++k) {
                // B is [N x K] row-major (TN layout)
                sum += A[i * K + k] * B[j * K + k];
            }
            C[i * N + j] = std::complex<double>(alpha) * sum + std::complex<double>(beta) * C[i * N + j];
        }
    }
}

// CPU reference HERK: C = A * A^H, packed lower triangle
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

// Helper: generate random FP16 data and double-precision reference
static void generate_data(std::vector<__half>& h_re, std::vector<__half>& h_im,
                          std::vector<std::complex<double>>& ref,
                          int64_t elems, unsigned seed = 42) {
    h_re.resize(elems);
    h_im.resize(elems);
    ref.resize(elems);
    srand(seed);
    for (int64_t i = 0; i < elems; ++i) {
        float re = (float)(rand() % 15 - 7);
        float im = (float)(rand() % 15 - 7);
        h_re[i] = __float2half(re);
        h_im[i] = __float2half(im);
        ref[i] = std::complex<double>(re, im);
    }
}

// Compare FP16 interleaved output against reference
static bool compare_fp16(const __half* h_C_re, const __half* h_C_im,
                         const std::vector<std::complex<double>>& ref,
                         int64_t elems, double tol_abs) {
    double max_err = 0, max_ref = 0;
    #pragma omp parallel for reduction(max:max_err, max_ref)
    for (int64_t i = 0; i < elems; ++i) {
        double re = __half2float(h_C_re[i]);
        double im = __half2float(h_C_im[i]);
        max_err = std::max(max_err, std::abs(re - ref[i].real()));
        max_err = std::max(max_err, std::abs(im - ref[i].imag()));
        max_ref = std::max(max_ref, std::abs(ref[i].real()));
        max_ref = std::max(max_ref, std::abs(ref[i].imag()));
    }
    double rel = (max_ref > 0) ? max_err / max_ref : max_err;
    printf("  max_ref=%.1f  max_err=%.2f  rel=%.6f  tol=%.1f\n",
           max_ref, max_err, rel, tol_abs);
    return max_err < tol_abs;
}

// Compare FP32 planar output against reference
static bool compare_fp32(const float* h_C_re, const float* h_C_im,
                         const std::vector<std::complex<double>>& ref,
                         int64_t elems, double tol_abs) {
    double max_err = 0, max_ref = 0;
    #pragma omp parallel for reduction(max:max_err, max_ref)
    for (int64_t i = 0; i < elems; ++i) {
        max_err = std::max(max_err, std::abs((double)h_C_re[i] - ref[i].real()));
        max_err = std::max(max_err, std::abs((double)h_C_im[i] - ref[i].imag()));
        max_ref = std::max(max_ref, std::abs(ref[i].real()));
        max_ref = std::max(max_ref, std::abs(ref[i].imag()));
    }
    double rel = (max_ref > 0) ? max_err / max_ref : max_err;
    printf("  max_ref=%.1f  max_err=%.2f  rel=%.6f  tol=%.1f\n",
           max_ref, max_err, rel, tol_abs);
    return max_err < tol_abs;
}

// Helper: generate continuous (non-integer) FP16 data and double-precision reference
static void generate_data_continuous(std::vector<__half>& h_re, std::vector<__half>& h_im,
                                     std::vector<std::complex<double>>& ref,
                                     int64_t elems, unsigned seed = 42) {
    h_re.resize(elems);
    h_im.resize(elems);
    ref.resize(elems);
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-0.5f, 0.5f);
    for (int64_t i = 0; i < elems; ++i) {
        float re = dist(rng);
        float im = dist(rng);
        h_re[i] = __float2half(re);
        h_im[i] = __float2half(im);
        // Use the actual FP16 value for reference (accounts for FP16 rounding)
        ref[i] = std::complex<double>(__half2float(h_re[i]), __half2float(h_im[i]));
    }
}

// Helper: generate continuous interleaved HERK data + double reference
static void generate_herk_data_continuous(std::vector<__half>& h_A,
                                          std::vector<std::complex<double>>& refA,
                                          int64_t per_A, int batch, unsigned seed = 42) {
    h_A.resize(per_A * batch * 2);
    refA.resize(per_A * batch);
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-0.5f, 0.5f);
    for (int64_t i = 0; i < per_A * batch; ++i) {
        float re = dist(rng);
        float im = dist(rng);
        h_A[i * 2]     = __float2half(re);
        h_A[i * 2 + 1] = __float2half(im);
        refA[i] = std::complex<double>(__half2float(h_A[i * 2]),
                                       __half2float(h_A[i * 2 + 1]));
    }
}

// Test 1: gemm_planar_batched (FP16 output)
static int test_gemm_fp16() {
    printf("=== Test: gemm (FP8, FP16 out) ===\n");
    const int M = 64, N = 64, K = 128, batch = 2;
    int64_t per_A = (int64_t)M * K, per_B = (int64_t)N * K, per_C = (int64_t)M * N;

    std::vector<__half> hA_re, hA_im, hB_re, hB_im;
    std::vector<std::complex<double>> refA, refB;
    generate_data(hA_re, hA_im, refA, per_A * batch, 42);
    generate_data(hB_re, hB_im, refB, per_B * batch, 123);

    // CPU reference
    std::vector<std::complex<double>> refC(per_C * batch, {0, 0});
    for (int b = 0; b < batch; ++b) {
        std::vector<std::complex<double>> A(refA.begin() + b * per_A, refA.begin() + (b + 1) * per_A);
        std::vector<std::complex<double>> B(refB.begin() + b * per_B, refB.begin() + (b + 1) * per_B);
        std::vector<std::complex<double>> C(per_C, {0, 0});
        cpu_gemm(A, B, C, M, N, K, 1.0f, 0.0f);
        for (int64_t i = 0; i < per_C; ++i) refC[b * per_C + i] = C[i];
    }

    // Upload
    __half *dA_re, *dA_im, *dB_re, *dB_im, *dC_re, *dC_im;
    CHECK_CUDA(cudaMalloc(&dA_re, per_A * batch * sizeof(__half)));
    CHECK_CUDA(cudaMalloc(&dA_im, per_A * batch * sizeof(__half)));
    CHECK_CUDA(cudaMalloc(&dB_re, per_B * batch * sizeof(__half)));
    CHECK_CUDA(cudaMalloc(&dB_im, per_B * batch * sizeof(__half)));
    CHECK_CUDA(cudaMalloc(&dC_re, per_C * batch * sizeof(__half)));
    CHECK_CUDA(cudaMalloc(&dC_im, per_C * batch * sizeof(__half)));
    CHECK_CUDA(cudaMemcpy(dA_re, hA_re.data(), per_A * batch * sizeof(__half), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dA_im, hA_im.data(), per_A * batch * sizeof(__half), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dB_re, hB_re.data(), per_B * batch * sizeof(__half), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dB_im, hB_im.data(), per_B * batch * sizeof(__half), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemset(dC_re, 0, per_C * batch * sizeof(__half)));
    CHECK_CUDA(cudaMemset(dC_im, 0, per_C * batch * sizeof(__half)));

    cutlass_gemm_api::CutlassComplexGemm api;
    int ret = api.gemm(dA_re, dA_im, dB_re, dB_im, dC_re, dC_im,
                        M, N, K, batch,
                        ComputePrecision::FP8, OutputPrecision::FP16);
    CHECK_CUDA(cudaDeviceSynchronize());

    if (ret != 0) { printf("  FAIL: gemm (FP8, FP16) returned %d\n", ret); return 1; }

    std::vector<__half> hC_re(per_C * batch), hC_im(per_C * batch);
    CHECK_CUDA(cudaMemcpy(hC_re.data(), dC_re, per_C * batch * sizeof(__half), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(hC_im.data(), dC_im, per_C * batch * sizeof(__half), cudaMemcpyDeviceToHost));

    double tol = std::max(1.0, K * 0.02);
    bool pass = compare_fp16(hC_re.data(), hC_im.data(), refC, per_C * batch, tol);

    cudaFree(dA_re); cudaFree(dA_im); cudaFree(dB_re); cudaFree(dB_im);
    cudaFree(dC_re); cudaFree(dC_im);

    printf("  %s\n\n", pass ? "PASS" : "FAIL");
    return pass ? 0 : 1;
}

// Test 2: gemm_planar_batched_fp32 (FP32 output)
static int test_gemm_fp32() {
    printf("=== Test: gemm (FP8, FP32 out) ===\n");
    const int M = 64, N = 64, K = 128, batch = 2;
    int64_t per_A = (int64_t)M * K, per_B = (int64_t)N * K, per_C = (int64_t)M * N;

    std::vector<__half> hA_re, hA_im, hB_re, hB_im;
    std::vector<std::complex<double>> refA, refB;
    generate_data(hA_re, hA_im, refA, per_A * batch, 42);
    generate_data(hB_re, hB_im, refB, per_B * batch, 123);

    std::vector<std::complex<double>> refC(per_C * batch, {0, 0});
    for (int b = 0; b < batch; ++b) {
        std::vector<std::complex<double>> A(refA.begin() + b * per_A, refA.begin() + (b + 1) * per_A);
        std::vector<std::complex<double>> B(refB.begin() + b * per_B, refB.begin() + (b + 1) * per_B);
        std::vector<std::complex<double>> C(per_C, {0, 0});
        cpu_gemm(A, B, C, M, N, K, 1.0f, 0.0f);
        for (int64_t i = 0; i < per_C; ++i) refC[b * per_C + i] = C[i];
    }

    __half *dA_re, *dA_im, *dB_re, *dB_im;
    float *dC_re, *dC_im;
    CHECK_CUDA(cudaMalloc(&dA_re, per_A * batch * sizeof(__half)));
    CHECK_CUDA(cudaMalloc(&dA_im, per_A * batch * sizeof(__half)));
    CHECK_CUDA(cudaMalloc(&dB_re, per_B * batch * sizeof(__half)));
    CHECK_CUDA(cudaMalloc(&dB_im, per_B * batch * sizeof(__half)));
    CHECK_CUDA(cudaMalloc(&dC_re, per_C * batch * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&dC_im, per_C * batch * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(dA_re, hA_re.data(), per_A * batch * sizeof(__half), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dA_im, hA_im.data(), per_A * batch * sizeof(__half), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dB_re, hB_re.data(), per_B * batch * sizeof(__half), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dB_im, hB_im.data(), per_B * batch * sizeof(__half), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemset(dC_re, 0, per_C * batch * sizeof(float)));
    CHECK_CUDA(cudaMemset(dC_im, 0, per_C * batch * sizeof(float)));

    cutlass_gemm_api::CutlassComplexGemm api;
    int ret = api.gemm(dA_re, dA_im, dB_re, dB_im, dC_re, dC_im,
                        M, N, K, batch,
                        ComputePrecision::FP8, OutputPrecision::FP32);
    CHECK_CUDA(cudaDeviceSynchronize());

    if (ret != 0) { printf("  FAIL: gemm (FP8, FP32) returned %d\n", ret); return 1; }

    std::vector<float> hC_re(per_C * batch), hC_im(per_C * batch);
    CHECK_CUDA(cudaMemcpy(hC_re.data(), dC_re, per_C * batch * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(hC_im.data(), dC_im, per_C * batch * sizeof(float), cudaMemcpyDeviceToHost));

    double tol = std::max(1.0, K * 0.02);
    bool pass = compare_fp32(hC_re.data(), hC_im.data(), refC, per_C * batch, tol);

    cudaFree(dA_re); cudaFree(dA_im); cudaFree(dB_re); cudaFree(dB_im);
    cudaFree(dC_re); cudaFree(dC_im);

    printf("  %s\n\n", pass ? "PASS" : "FAIL");
    return pass ? 0 : 1;
}

// Test 3: prepare_b + gemm_prepared_fp32 (prepare/execute pattern)
static int test_prepare_execute() {
    printf("=== Test: prepare_b + gemm_prepared ===\n");
    const int M = 64, N = 64, K = 128, batch = 2;
    int64_t per_A = (int64_t)M * K, per_B = (int64_t)N * K, per_C = (int64_t)M * N;

    std::vector<__half> hA_re, hA_im, hB_re, hB_im;
    std::vector<std::complex<double>> refA, refB;
    generate_data(hA_re, hA_im, refA, per_A * batch, 42);
    generate_data(hB_re, hB_im, refB, per_B * batch, 123);

    std::vector<std::complex<double>> refC(per_C * batch, {0, 0});
    for (int b = 0; b < batch; ++b) {
        std::vector<std::complex<double>> A(refA.begin() + b * per_A, refA.begin() + (b + 1) * per_A);
        std::vector<std::complex<double>> B(refB.begin() + b * per_B, refB.begin() + (b + 1) * per_B);
        std::vector<std::complex<double>> C(per_C, {0, 0});
        cpu_gemm(A, B, C, M, N, K, 1.0f, 0.0f);
        for (int64_t i = 0; i < per_C; ++i) refC[b * per_C + i] = C[i];
    }

    __half *dA_re, *dA_im, *dB_re, *dB_im;
    float *dC_re, *dC_im;
    CHECK_CUDA(cudaMalloc(&dA_re, per_A * batch * sizeof(__half)));
    CHECK_CUDA(cudaMalloc(&dA_im, per_A * batch * sizeof(__half)));
    CHECK_CUDA(cudaMalloc(&dB_re, per_B * batch * sizeof(__half)));
    CHECK_CUDA(cudaMalloc(&dB_im, per_B * batch * sizeof(__half)));
    CHECK_CUDA(cudaMalloc(&dC_re, per_C * batch * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&dC_im, per_C * batch * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(dA_re, hA_re.data(), per_A * batch * sizeof(__half), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dA_im, hA_im.data(), per_A * batch * sizeof(__half), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dB_re, hB_re.data(), per_B * batch * sizeof(__half), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dB_im, hB_im.data(), per_B * batch * sizeof(__half), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemset(dC_re, 0, per_C * batch * sizeof(float)));
    CHECK_CUDA(cudaMemset(dC_im, 0, per_C * batch * sizeof(float)));

    cutlass_gemm_api::CutlassComplexGemm api;

    // Step 1: Prepare B (FP8 compute)
    printf("  prepare_b(FP8)...\n");
    api.prepare_b(dB_re, dB_im, N, K, batch, ComputePrecision::FP8);
    CHECK_CUDA(cudaDeviceSynchronize());

    // Step 2: Execute with prepared B
    printf("  gemm_prepared()...\n");
    int ret = api.gemm_prepared(dA_re, dA_im, dC_re, dC_im, M, N, K, batch);
    CHECK_CUDA(cudaDeviceSynchronize());

    if (ret != 0) { printf("  FAIL: gemm_prepared returned %d\n", ret); return 1; }

    std::vector<float> hC_re(per_C * batch), hC_im(per_C * batch);
    CHECK_CUDA(cudaMemcpy(hC_re.data(), dC_re, per_C * batch * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(hC_im.data(), dC_im, per_C * batch * sizeof(float), cudaMemcpyDeviceToHost));

    double tol = std::max(1.0, K * 0.02);
    bool pass = compare_fp32(hC_re.data(), hC_im.data(), refC, per_C * batch, tol);

    cudaFree(dA_re); cudaFree(dA_im); cudaFree(dB_re); cudaFree(dB_im);
    cudaFree(dC_re); cudaFree(dC_im);

    printf("  %s\n\n", pass ? "PASS" : "FAIL");
    return pass ? 0 : 1;
}

// Test 4: gemm_planar_batched_fp6_fp32 (FP6 complex GEMM → FP32)
static int test_gemm_fp6() {
    printf("=== Test: gemm (FP6, FP32 out) ===\n");
#if defined(COMPLEX_FP8_TARGET_SM90)
    printf("  SKIPPED (FP6 not available on SM90)\n\n");
    return -1;
#elif !defined(COMPLEX_SM100_ENABLE_FP6)
    printf("  SKIPPED (FP6 not enabled in this build)\n\n");
    return -1;
#endif
    const int M = 128, N = 128, K = 128, batch = 2;
    int64_t per_A = (int64_t)M * K, per_B = (int64_t)N * K, per_C = (int64_t)M * N;

    std::vector<__half> hA_re, hA_im, hB_re, hB_im;
    std::vector<std::complex<double>> refA, refB;
    generate_data(hA_re, hA_im, refA, per_A * batch, 42);
    generate_data(hB_re, hB_im, refB, per_B * batch, 123);

    std::vector<std::complex<double>> refC(per_C * batch, {0, 0});
    for (int b = 0; b < batch; ++b) {
        std::vector<std::complex<double>> A(refA.begin() + b * per_A, refA.begin() + (b + 1) * per_A);
        std::vector<std::complex<double>> B(refB.begin() + b * per_B, refB.begin() + (b + 1) * per_B);
        std::vector<std::complex<double>> C(per_C, {0, 0});
        cpu_gemm(A, B, C, M, N, K, 1.0f, 0.0f);
        for (int64_t i = 0; i < per_C; ++i) refC[b * per_C + i] = C[i];
    }

    __half *dA_re, *dA_im, *dB_re, *dB_im;
    float *dC_re, *dC_im;
    CHECK_CUDA(cudaMalloc(&dA_re, per_A * batch * sizeof(__half)));
    CHECK_CUDA(cudaMalloc(&dA_im, per_A * batch * sizeof(__half)));
    CHECK_CUDA(cudaMalloc(&dB_re, per_B * batch * sizeof(__half)));
    CHECK_CUDA(cudaMalloc(&dB_im, per_B * batch * sizeof(__half)));
    CHECK_CUDA(cudaMalloc(&dC_re, per_C * batch * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&dC_im, per_C * batch * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(dA_re, hA_re.data(), per_A * batch * sizeof(__half), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dA_im, hA_im.data(), per_A * batch * sizeof(__half), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dB_re, hB_re.data(), per_B * batch * sizeof(__half), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dB_im, hB_im.data(), per_B * batch * sizeof(__half), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemset(dC_re, 0, per_C * batch * sizeof(float)));
    CHECK_CUDA(cudaMemset(dC_im, 0, per_C * batch * sizeof(float)));

    cutlass_gemm_api::CutlassComplexGemm api;
    int ret = api.gemm(dA_re, dA_im, dB_re, dB_im, dC_re, dC_im,
                        M, N, K, batch,
                        ComputePrecision::FP6, OutputPrecision::FP32);
    CHECK_CUDA(cudaDeviceSynchronize());

    if (ret != 0) { printf("  FAIL: gemm (FP6, FP32) returned %d\n", ret); return 1; }

    std::vector<float> hC_re(per_C * batch), hC_im(per_C * batch);
    CHECK_CUDA(cudaMemcpy(hC_re.data(), dC_re, per_C * batch * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(hC_im.data(), dC_im, per_C * batch * sizeof(float), cudaMemcpyDeviceToHost));

    // FP6 E3M2 is exact for integers [-7,7], so tolerance same as FP8
    double tol = std::max(1.0, K * 0.02);
    bool pass = compare_fp32(hC_re.data(), hC_im.data(), refC, per_C * batch, tol);

    cudaFree(dA_re); cudaFree(dA_im); cudaFree(dB_re); cudaFree(dB_im);
    cudaFree(dC_re); cudaFree(dC_im);

    printf("  %s\n\n", pass ? "PASS" : "FAIL");
    return pass ? 0 : 1;
}

// Test 5: gemm_planar_batched_fp4_fp32 (FP4 complex GEMM → FP32, lossy)
static int test_gemm_fp4() {
    printf("=== Test: gemm (FP4, FP32 out) ===\n");
#if defined(COMPLEX_FP8_TARGET_SM90)
    printf("  SKIPPED (FP4 not available on SM90)\n\n");
    return -1;
#elif !defined(COMPLEX_SM100_ENABLE_FP4)
    printf("  SKIPPED (FP4 not enabled in this build)\n\n");
    return -1;
#endif
    // Use small integer values to limit FP4 quantization loss
    // FP4 E2M1 can exactly represent: 0, 0.5, 1, 1.5, 2, 3, 4, 6 (and negatives)
    // Integers 5→4, 7→8 are lossy, so we use values in [-4, 4]
    const int M = 128, N = 128, K = 128, batch = 2;
    int64_t per_A = (int64_t)M * K, per_B = (int64_t)N * K, per_C = (int64_t)M * N;

    std::vector<__half> hA_re(per_A * batch), hA_im(per_A * batch);
    std::vector<__half> hB_re(per_B * batch), hB_im(per_B * batch);
    std::vector<std::complex<double>> refA(per_A * batch), refB(per_B * batch);

    srand(42);
    for (int64_t i = 0; i < per_A * batch; ++i) {
        float re = (float)(rand() % 9 - 4);  // [-4, 4] exact in FP4
        float im = (float)(rand() % 9 - 4);
        hA_re[i] = __float2half(re); hA_im[i] = __float2half(im);
        refA[i] = std::complex<double>(re, im);
    }
    srand(123);
    for (int64_t i = 0; i < per_B * batch; ++i) {
        float re = (float)(rand() % 9 - 4);
        float im = (float)(rand() % 9 - 4);
        hB_re[i] = __float2half(re); hB_im[i] = __float2half(im);
        refB[i] = std::complex<double>(re, im);
    }

    std::vector<std::complex<double>> refC(per_C * batch, {0, 0});
    for (int b = 0; b < batch; ++b) {
        std::vector<std::complex<double>> A(refA.begin() + b * per_A, refA.begin() + (b + 1) * per_A);
        std::vector<std::complex<double>> B(refB.begin() + b * per_B, refB.begin() + (b + 1) * per_B);
        std::vector<std::complex<double>> C(per_C, {0, 0});
        cpu_gemm(A, B, C, M, N, K, 1.0f, 0.0f);
        for (int64_t i = 0; i < per_C; ++i) refC[b * per_C + i] = C[i];
    }

    __half *dA_re, *dA_im, *dB_re, *dB_im;
    float *dC_re, *dC_im;
    CHECK_CUDA(cudaMalloc(&dA_re, per_A * batch * sizeof(__half)));
    CHECK_CUDA(cudaMalloc(&dA_im, per_A * batch * sizeof(__half)));
    CHECK_CUDA(cudaMalloc(&dB_re, per_B * batch * sizeof(__half)));
    CHECK_CUDA(cudaMalloc(&dB_im, per_B * batch * sizeof(__half)));
    CHECK_CUDA(cudaMalloc(&dC_re, per_C * batch * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&dC_im, per_C * batch * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(dA_re, hA_re.data(), per_A * batch * sizeof(__half), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dA_im, hA_im.data(), per_A * batch * sizeof(__half), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dB_re, hB_re.data(), per_B * batch * sizeof(__half), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dB_im, hB_im.data(), per_B * batch * sizeof(__half), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemset(dC_re, 0, per_C * batch * sizeof(float)));
    CHECK_CUDA(cudaMemset(dC_im, 0, per_C * batch * sizeof(float)));

    cutlass_gemm_api::CutlassComplexGemm api;
    int ret = api.gemm(dA_re, dA_im, dB_re, dB_im, dC_re, dC_im,
                        M, N, K, batch,
                        ComputePrecision::FP4, OutputPrecision::FP32);
    CHECK_CUDA(cudaDeviceSynchronize());

    if (ret != 0) { printf("  FAIL: gemm (FP4, FP32) returned %d\n", ret); return 1; }

    std::vector<float> hC_re(per_C * batch), hC_im(per_C * batch);
    CHECK_CUDA(cudaMemcpy(hC_re.data(), dC_re, per_C * batch * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(hC_im.data(), dC_im, per_C * batch * sizeof(float), cudaMemcpyDeviceToHost));

    // FP4 with [-4,4] inputs is exact, so tolerance same as FP8
    double tol = std::max(1.0, K * 0.02);
    bool pass = compare_fp32(hC_re.data(), hC_im.data(), refC, per_C * batch, tol);

    cudaFree(dA_re); cudaFree(dA_im); cudaFree(dB_re); cudaFree(dB_im);
    cudaFree(dC_re); cudaFree(dC_im);

    printf("  %s\n\n", pass ? "PASS" : "FAIL");
    return pass ? 0 : 1;
}

// Test 6: prepare_b at FP6 precision + gemm_prepared_fp32
static int test_prepare_fp6() {
    printf("=== Test: prepare_b(FP6) + gemm_prepared ===\n");
#if defined(COMPLEX_FP8_TARGET_SM90)
    printf("  SKIPPED (FP6 not available on SM90)\n\n");
    return -1;
#elif !defined(COMPLEX_SM100_ENABLE_FP6)
    printf("  SKIPPED (FP6 not enabled in this build)\n\n");
    return -1;
#endif
    const int M = 128, N = 128, K = 128, batch = 2;
    int64_t per_A = (int64_t)M * K, per_B = (int64_t)N * K, per_C = (int64_t)M * N;

    std::vector<__half> hA_re, hA_im, hB_re, hB_im;
    std::vector<std::complex<double>> refA, refB;
    generate_data(hA_re, hA_im, refA, per_A * batch, 42);
    generate_data(hB_re, hB_im, refB, per_B * batch, 123);

    std::vector<std::complex<double>> refC(per_C * batch, {0, 0});
    for (int b = 0; b < batch; ++b) {
        std::vector<std::complex<double>> A(refA.begin() + b * per_A, refA.begin() + (b + 1) * per_A);
        std::vector<std::complex<double>> B(refB.begin() + b * per_B, refB.begin() + (b + 1) * per_B);
        std::vector<std::complex<double>> C(per_C, {0, 0});
        cpu_gemm(A, B, C, M, N, K, 1.0f, 0.0f);
        for (int64_t i = 0; i < per_C; ++i) refC[b * per_C + i] = C[i];
    }

    __half *dA_re, *dA_im, *dB_re, *dB_im;
    float *dC_re, *dC_im;
    CHECK_CUDA(cudaMalloc(&dA_re, per_A * batch * sizeof(__half)));
    CHECK_CUDA(cudaMalloc(&dA_im, per_A * batch * sizeof(__half)));
    CHECK_CUDA(cudaMalloc(&dB_re, per_B * batch * sizeof(__half)));
    CHECK_CUDA(cudaMalloc(&dB_im, per_B * batch * sizeof(__half)));
    CHECK_CUDA(cudaMalloc(&dC_re, per_C * batch * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&dC_im, per_C * batch * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(dA_re, hA_re.data(), per_A * batch * sizeof(__half), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dA_im, hA_im.data(), per_A * batch * sizeof(__half), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dB_re, hB_re.data(), per_B * batch * sizeof(__half), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dB_im, hB_im.data(), per_B * batch * sizeof(__half), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemset(dC_re, 0, per_C * batch * sizeof(float)));
    CHECK_CUDA(cudaMemset(dC_im, 0, per_C * batch * sizeof(float)));

    cutlass_gemm_api::CutlassComplexGemm api;
    printf("  prepare_b(FP6)...\n");
    api.prepare_b(dB_re, dB_im, N, K, batch, ComputePrecision::FP6);
    CHECK_CUDA(cudaDeviceSynchronize());

    printf("  gemm_prepared()...\n");
    int ret = api.gemm_prepared(dA_re, dA_im, dC_re, dC_im, M, N, K, batch);
    CHECK_CUDA(cudaDeviceSynchronize());

    if (ret != 0) { printf("  FAIL: gemm_prepared (FP6) returned %d\n", ret); return 1; }

    std::vector<float> hC_re(per_C * batch), hC_im(per_C * batch);
    CHECK_CUDA(cudaMemcpy(hC_re.data(), dC_re, per_C * batch * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(hC_im.data(), dC_im, per_C * batch * sizeof(float), cudaMemcpyDeviceToHost));

    double tol = std::max(1.0, K * 0.02);
    bool pass = compare_fp32(hC_re.data(), hC_im.data(), refC, per_C * batch, tol);

    cudaFree(dA_re); cudaFree(dA_im); cudaFree(dB_re); cudaFree(dB_im);
    cudaFree(dC_re); cudaFree(dC_im);

    printf("  %s\n\n", pass ? "PASS" : "FAIL");
    return pass ? 0 : 1;
}

// Test 7: prepare_b at FP4 precision + gemm_prepared_fp32
static int test_prepare_fp4() {
    printf("=== Test: prepare_b(FP4) + gemm_prepared ===\n");
#if defined(COMPLEX_FP8_TARGET_SM90)
    printf("  SKIPPED (FP4 not available on SM90)\n\n");
    return -1;
#elif !defined(COMPLEX_SM100_ENABLE_FP4)
    printf("  SKIPPED (FP4 not enabled in this build)\n\n");
    return -1;
#endif
    const int M = 128, N = 128, K = 128, batch = 2;
    int64_t per_A = (int64_t)M * K, per_B = (int64_t)N * K, per_C = (int64_t)M * N;

    // Use [-4,4] for FP4-exact values
    std::vector<__half> hA_re(per_A * batch), hA_im(per_A * batch);
    std::vector<__half> hB_re(per_B * batch), hB_im(per_B * batch);
    std::vector<std::complex<double>> refA(per_A * batch), refB(per_B * batch);

    srand(42);
    for (int64_t i = 0; i < per_A * batch; ++i) {
        float re = (float)(rand() % 9 - 4);
        float im = (float)(rand() % 9 - 4);
        hA_re[i] = __float2half(re); hA_im[i] = __float2half(im);
        refA[i] = std::complex<double>(re, im);
    }
    srand(123);
    for (int64_t i = 0; i < per_B * batch; ++i) {
        float re = (float)(rand() % 9 - 4);
        float im = (float)(rand() % 9 - 4);
        hB_re[i] = __float2half(re); hB_im[i] = __float2half(im);
        refB[i] = std::complex<double>(re, im);
    }

    std::vector<std::complex<double>> refC(per_C * batch, {0, 0});
    for (int b = 0; b < batch; ++b) {
        std::vector<std::complex<double>> A(refA.begin() + b * per_A, refA.begin() + (b + 1) * per_A);
        std::vector<std::complex<double>> B(refB.begin() + b * per_B, refB.begin() + (b + 1) * per_B);
        std::vector<std::complex<double>> C(per_C, {0, 0});
        cpu_gemm(A, B, C, M, N, K, 1.0f, 0.0f);
        for (int64_t i = 0; i < per_C; ++i) refC[b * per_C + i] = C[i];
    }

    __half *dA_re, *dA_im, *dB_re, *dB_im;
    float *dC_re, *dC_im;
    CHECK_CUDA(cudaMalloc(&dA_re, per_A * batch * sizeof(__half)));
    CHECK_CUDA(cudaMalloc(&dA_im, per_A * batch * sizeof(__half)));
    CHECK_CUDA(cudaMalloc(&dB_re, per_B * batch * sizeof(__half)));
    CHECK_CUDA(cudaMalloc(&dB_im, per_B * batch * sizeof(__half)));
    CHECK_CUDA(cudaMalloc(&dC_re, per_C * batch * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&dC_im, per_C * batch * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(dA_re, hA_re.data(), per_A * batch * sizeof(__half), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dA_im, hA_im.data(), per_A * batch * sizeof(__half), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dB_re, hB_re.data(), per_B * batch * sizeof(__half), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dB_im, hB_im.data(), per_B * batch * sizeof(__half), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemset(dC_re, 0, per_C * batch * sizeof(float)));
    CHECK_CUDA(cudaMemset(dC_im, 0, per_C * batch * sizeof(float)));

    cutlass_gemm_api::CutlassComplexGemm api;
    printf("  prepare_b(FP4)...\n");
    api.prepare_b(dB_re, dB_im, N, K, batch, ComputePrecision::FP4);
    CHECK_CUDA(cudaDeviceSynchronize());

    printf("  gemm_prepared()...\n");
    int ret = api.gemm_prepared(dA_re, dA_im, dC_re, dC_im, M, N, K, batch);
    CHECK_CUDA(cudaDeviceSynchronize());

    if (ret != 0) { printf("  FAIL: gemm_prepared (FP4) returned %d\n", ret); return 1; }

    std::vector<float> hC_re(per_C * batch), hC_im(per_C * batch);
    CHECK_CUDA(cudaMemcpy(hC_re.data(), dC_re, per_C * batch * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(hC_im.data(), dC_im, per_C * batch * sizeof(float), cudaMemcpyDeviceToHost));

    double tol = std::max(1.0, K * 0.02);
    bool pass = compare_fp32(hC_re.data(), hC_im.data(), refC, per_C * batch, tol);

    cudaFree(dA_re); cudaFree(dA_im); cudaFree(dB_re); cudaFree(dB_im);
    cudaFree(dC_re); cudaFree(dC_im);

    printf("  %s\n\n", pass ? "PASS" : "FAIL");
    return pass ? 0 : 1;
}

// Test 8: herk_batched_fp32 (FP16 input → FP32 output)
static int test_herk_fp32() {
    printf("=== Test: herk (FP16 in, FP32 out) ===\n");
    const int N = 64, K = 128, batch = 2;
    int64_t per_A = (int64_t)N * K;
    int64_t packed = (int64_t)N * (N + 1) / 2;

    // Generate interleaved FP16 complex data [batch x N x K x 2]
    std::vector<__half> h_A(per_A * batch * 2);
    std::vector<std::complex<double>> refA(per_A * batch);
    srand(42);
    for (int64_t i = 0; i < per_A * batch; ++i) {
        float re = (float)(rand() % 15 - 7);
        float im = (float)(rand() % 15 - 7);
        h_A[i * 2]     = __float2half(re);
        h_A[i * 2 + 1] = __float2half(im);
        refA[i] = std::complex<double>(re, im);
    }

    // CPU reference
    std::vector<std::complex<double>> refC(packed * batch, {0, 0});
    for (int b = 0; b < batch; ++b) {
        std::vector<std::complex<double>> A(refA.begin() + b * per_A,
                                             refA.begin() + (b + 1) * per_A);
        std::vector<std::complex<double>> C(packed, {0, 0});
        cpu_herk(A, C, N, K);
        for (int64_t i = 0; i < packed; ++i) refC[b * packed + i] = C[i];
    }

    __half* d_A;
    float* d_C;
    CHECK_CUDA(cudaMalloc(&d_A, per_A * batch * 2 * sizeof(__half)));
    CHECK_CUDA(cudaMalloc(&d_C, packed * batch * 2 * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(d_A, h_A.data(), per_A * batch * 2 * sizeof(__half), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemset(d_C, 0, packed * batch * 2 * sizeof(float)));

    cutlass_gemm_api::CutlassComplexGemm api;
    int ret = api.herk(d_A, d_C, N, K, batch,
                       InputPrecision::FP16, ComputePrecision::FP8,
                       OutputPrecision::FP32);
    CHECK_CUDA(cudaDeviceSynchronize());

    if (ret != 0) { printf("  FAIL: herk (FP16, FP32) returned %d\n", ret); return 1; }

    std::vector<float> h_C(packed * batch * 2);
    CHECK_CUDA(cudaMemcpy(h_C.data(), d_C, packed * batch * 2 * sizeof(float), cudaMemcpyDeviceToHost));

    // Compare interleaved FP32 output vs reference
    double max_err = 0, max_ref = 0;
    #pragma omp parallel for reduction(max:max_err, max_ref)
    for (int64_t i = 0; i < packed * batch; ++i) {
        double re = h_C[i * 2], im = h_C[i * 2 + 1];
        max_err = std::max(max_err, std::abs(re - refC[i].real()));
        max_err = std::max(max_err, std::abs(im - refC[i].imag()));
        max_ref = std::max(max_ref, std::abs(refC[i].real()));
        max_ref = std::max(max_ref, std::abs(refC[i].imag()));
    }
    double rel = (max_ref > 0) ? max_err / max_ref : max_err;
    double tol = std::max(1.0, K * 0.02);
    bool pass = max_err < tol;
    printf("  max_ref=%.1f  max_err=%.2f  rel=%.6f  tol=%.1f\n", max_ref, max_err, rel, tol);

    cudaFree(d_A); cudaFree(d_C);
    printf("  %s\n\n", pass ? "PASS" : "FAIL");
    return pass ? 0 : 1;
}

// CPU reference HERK with alpha/beta: C = alpha * A * A^H + beta * C, packed lower triangle
static void cpu_herk_ab(
    const std::vector<std::complex<double>>& A,
    std::vector<std::complex<double>>& C,
    int N, int K, double alpha, double beta)
{
    std::vector<std::complex<double>> AAH(N * (N + 1) / 2);
    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j <= i; ++j) {
            std::complex<double> sum(0, 0);
            for (int k = 0; k < K; ++k) {
                sum += A[i * K + k] * std::conj(A[j * K + k]);
            }
            int64_t idx = static_cast<int64_t>(i) * (i + 1) / 2 + j;
            if (i == j)
                AAH[idx] = std::complex<double>(sum.real(), 0.0);
            else
                AAH[idx] = sum;
        }
    }
    int64_t packed = N * (N + 1) / 2;
    for (int64_t i = 0; i < packed; ++i) {
        C[i] = std::complex<double>(alpha) * AAH[i] + std::complex<double>(beta) * C[i];
    }
}

// Test: HERK with beta accumulation
static int test_herk_beta() {
    printf("=== Test: herk (beta accumulation) ===\n");
    const int N = 64, K = 128, batch = 2;
    int64_t per_A = (int64_t)N * K;
    int64_t packed = (int64_t)N * (N + 1) / 2;

    // Generate interleaved FP16 complex data
    std::vector<__half> h_A(per_A * batch * 2);
    std::vector<std::complex<double>> refA(per_A * batch);
    srand(42);
    for (int64_t i = 0; i < per_A * batch; ++i) {
        float re = (float)(rand() % 15 - 7);
        float im = (float)(rand() % 15 - 7);
        h_A[i * 2]     = __float2half(re);
        h_A[i * 2 + 1] = __float2half(im);
        refA[i] = std::complex<double>(re, im);
    }

    __half* d_A;
    __half* d_C;
    CHECK_CUDA(cudaMalloc(&d_A, per_A * batch * 2 * sizeof(__half)));
    CHECK_CUDA(cudaMalloc(&d_C, packed * batch * 2 * sizeof(__half)));
    CHECK_CUDA(cudaMemcpy(d_A, h_A.data(), per_A * batch * 2 * sizeof(__half), cudaMemcpyHostToDevice));

    cutlass_gemm_api::CutlassComplexGemm api;

    // Step 1: C = 1.0 * A * A^H + 0.0 * C  (fill C)
    CHECK_CUDA(cudaMemset(d_C, 0, packed * batch * 2 * sizeof(__half)));
    int ret = api.herk(d_A, d_C, N, K, batch);
    CHECK_CUDA(cudaDeviceSynchronize());
    if (ret != 0) { printf("  FAIL: first herk returned %d\n", ret); return 1; }

    // Step 2: C = 0.5 * A * A^H + 1.0 * C
    ret = api.herk(d_A, d_C, N, K, batch,
                   InputPrecision::FP16, ComputePrecision::FP8,
                   OutputPrecision::FP16, OutputFormat::PackedTriangle,
                   0.5f, 1.0f);
    CHECK_CUDA(cudaDeviceSynchronize());
    if (ret != 0) { printf("  FAIL: second herk returned %d\n", ret); return 1; }

    // CPU reference: C = 0.5 * A*A^H + 1.0 * (1.0 * A*A^H) = 1.5 * A*A^H
    std::vector<std::complex<double>> refC(packed * batch, {0, 0});
    for (int b = 0; b < batch; ++b) {
        std::vector<std::complex<double>> A_b(refA.begin() + b * per_A,
                                               refA.begin() + (b + 1) * per_A);
        std::vector<std::complex<double>> C_b(packed, {0, 0});
        cpu_herk_ab(A_b, C_b, N, K, 1.0, 0.0);       // C_b = A*A^H
        cpu_herk_ab(A_b, C_b, N, K, 0.5, 1.0);        // C_b = 0.5*A*A^H + C_b = 1.5*A*A^H
        for (int64_t i = 0; i < packed; ++i) refC[b * packed + i] = C_b[i];
    }

    // Compare
    std::vector<__half> h_C(packed * batch * 2);
    CHECK_CUDA(cudaMemcpy(h_C.data(), d_C, packed * batch * 2 * sizeof(__half), cudaMemcpyDeviceToHost));

    double max_err = 0, max_ref = 0;
    #pragma omp parallel for reduction(max:max_err, max_ref)
    for (int64_t i = 0; i < packed * batch; ++i) {
        double re = __half2float(h_C[i * 2]);
        double im = __half2float(h_C[i * 2 + 1]);
        max_err = std::max(max_err, std::abs(re - refC[i].real()));
        max_err = std::max(max_err, std::abs(im - refC[i].imag()));
        max_ref = std::max(max_ref, std::abs(refC[i].real()));
        max_ref = std::max(max_ref, std::abs(refC[i].imag()));
    }
    double rel = (max_ref > 0) ? max_err / max_ref : max_err;
    double tol = std::max(2.0, K * 0.04);  // tighter for accumulated error
    bool pass = max_err < tol;
    printf("  max_ref=%.1f  max_err=%.2f  rel=%.6f  tol=%.1f\n", max_ref, max_err, rel, tol);

    cudaFree(d_A); cudaFree(d_C);
    printf("  %s\n\n", pass ? "PASS" : "FAIL");
    return pass ? 0 : 1;
}

// Test: HERK with batch=1
static int test_herk_single_batch() {
    printf("=== Test: herk (batch=1) ===\n");
    const int N = 64, K = 128, batch = 1;
    int64_t per_A = (int64_t)N * K;
    int64_t packed = (int64_t)N * (N + 1) / 2;

    std::vector<__half> h_A(per_A * 2);
    std::vector<std::complex<double>> refA(per_A);
    srand(42);
    for (int64_t i = 0; i < per_A; ++i) {
        float re = (float)(rand() % 15 - 7);
        float im = (float)(rand() % 15 - 7);
        h_A[i * 2]     = __float2half(re);
        h_A[i * 2 + 1] = __float2half(im);
        refA[i] = std::complex<double>(re, im);
    }

    std::vector<std::complex<double>> refC(packed, {0, 0});
    cpu_herk(refA, refC, N, K);

    __half* d_A;
    __half* d_C;
    CHECK_CUDA(cudaMalloc(&d_A, per_A * 2 * sizeof(__half)));
    CHECK_CUDA(cudaMalloc(&d_C, packed * 2 * sizeof(__half)));
    CHECK_CUDA(cudaMemcpy(d_A, h_A.data(), per_A * 2 * sizeof(__half), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemset(d_C, 0, packed * 2 * sizeof(__half)));

    cutlass_gemm_api::CutlassComplexGemm api;
    int ret = api.herk(d_A, d_C, N, K, batch);
    CHECK_CUDA(cudaDeviceSynchronize());

    if (ret != 0) { printf("  FAIL: herk returned %d\n", ret); return 1; }

    std::vector<__half> h_C(packed * 2);
    CHECK_CUDA(cudaMemcpy(h_C.data(), d_C, packed * 2 * sizeof(__half), cudaMemcpyDeviceToHost));

    double max_err = 0, max_ref = 0;
    #pragma omp parallel for reduction(max:max_err, max_ref)
    for (int64_t i = 0; i < packed; ++i) {
        double re = __half2float(h_C[i * 2]);
        double im = __half2float(h_C[i * 2 + 1]);
        max_err = std::max(max_err, std::abs(re - refC[i].real()));
        max_err = std::max(max_err, std::abs(im - refC[i].imag()));
        max_ref = std::max(max_ref, std::abs(refC[i].real()));
        max_ref = std::max(max_ref, std::abs(refC[i].imag()));
    }
    double rel = (max_ref > 0) ? max_err / max_ref : max_err;
    double tol = std::max(1.0, K * 0.02);
    bool pass = max_err < tol;
    printf("  max_ref=%.1f  max_err=%.2f  rel=%.6f  tol=%.1f\n", max_ref, max_err, rel, tol);

    cudaFree(d_A); cudaFree(d_C);
    printf("  %s\n\n", pass ? "PASS" : "FAIL");
    return pass ? 0 : 1;
}

// Test: GEMM with non-power-of-2 dimensions
static int test_gemm_non_aligned() {
    printf("=== Test: gemm (non-power-of-2 M=48, N=80) ===\n");
    const int M = 48, N = 80, K = 128, batch = 1;
    int64_t per_A = (int64_t)M * K, per_B = (int64_t)N * K, per_C = (int64_t)M * N;

    std::vector<__half> hA_re, hA_im, hB_re, hB_im;
    std::vector<std::complex<double>> refA, refB;
    generate_data(hA_re, hA_im, refA, per_A * batch, 42);
    generate_data(hB_re, hB_im, refB, per_B * batch, 123);

    std::vector<std::complex<double>> refC(per_C * batch, {0, 0});
    {
        std::vector<std::complex<double>> C(per_C, {0, 0});
        cpu_gemm(refA, refB, C, M, N, K, 1.0f, 0.0f);
        for (int64_t i = 0; i < per_C; ++i) refC[i] = C[i];
    }

    __half *dA_re, *dA_im, *dB_re, *dB_im, *dC_re, *dC_im;
    CHECK_CUDA(cudaMalloc(&dA_re, per_A * batch * sizeof(__half)));
    CHECK_CUDA(cudaMalloc(&dA_im, per_A * batch * sizeof(__half)));
    CHECK_CUDA(cudaMalloc(&dB_re, per_B * batch * sizeof(__half)));
    CHECK_CUDA(cudaMalloc(&dB_im, per_B * batch * sizeof(__half)));
    CHECK_CUDA(cudaMalloc(&dC_re, per_C * batch * sizeof(__half)));
    CHECK_CUDA(cudaMalloc(&dC_im, per_C * batch * sizeof(__half)));
    CHECK_CUDA(cudaMemcpy(dA_re, hA_re.data(), per_A * batch * sizeof(__half), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dA_im, hA_im.data(), per_A * batch * sizeof(__half), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dB_re, hB_re.data(), per_B * batch * sizeof(__half), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dB_im, hB_im.data(), per_B * batch * sizeof(__half), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemset(dC_re, 0, per_C * batch * sizeof(__half)));
    CHECK_CUDA(cudaMemset(dC_im, 0, per_C * batch * sizeof(__half)));

    cutlass_gemm_api::CutlassComplexGemm api;
    int ret = api.gemm(dA_re, dA_im, dB_re, dB_im, dC_re, dC_im,
                        M, N, K, batch,
                        ComputePrecision::FP8, OutputPrecision::FP16);
    CHECK_CUDA(cudaDeviceSynchronize());

    if (ret != 0) { printf("  FAIL: gemm returned %d\n", ret); return 1; }

    std::vector<__half> hC_re(per_C * batch), hC_im(per_C * batch);
    CHECK_CUDA(cudaMemcpy(hC_re.data(), dC_re, per_C * batch * sizeof(__half), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(hC_im.data(), dC_im, per_C * batch * sizeof(__half), cudaMemcpyDeviceToHost));

    double tol = std::max(1.0, K * 0.02);
    bool pass = compare_fp16(hC_re.data(), hC_im.data(), refC, per_C * batch, tol);

    cudaFree(dA_re); cudaFree(dA_im); cudaFree(dB_re); cudaFree(dB_im);
    cudaFree(dC_re); cudaFree(dC_im);

    printf("  %s\n\n", pass ? "PASS" : "FAIL");
    return pass ? 0 : 1;
}

// Test: GEMM with FP32 output via PIMPL
static int test_gemm_fp32_output() {
    printf("=== Test: gemm (FP8, FP32 out, batch=2) ===\n");
    const int M = 64, N = 64, K = 128, batch = 2;
    int64_t per_A = (int64_t)M * K, per_B = (int64_t)N * K, per_C = (int64_t)M * N;

    std::vector<__half> hA_re, hA_im, hB_re, hB_im;
    std::vector<std::complex<double>> refA, refB;
    generate_data(hA_re, hA_im, refA, per_A * batch, 55);
    generate_data(hB_re, hB_im, refB, per_B * batch, 77);

    std::vector<std::complex<double>> refC(per_C * batch, {0, 0});
    for (int b = 0; b < batch; ++b) {
        std::vector<std::complex<double>> A(refA.begin() + b * per_A, refA.begin() + (b + 1) * per_A);
        std::vector<std::complex<double>> B(refB.begin() + b * per_B, refB.begin() + (b + 1) * per_B);
        std::vector<std::complex<double>> C(per_C, {0, 0});
        cpu_gemm(A, B, C, M, N, K, 1.0f, 0.0f);
        for (int64_t i = 0; i < per_C; ++i) refC[b * per_C + i] = C[i];
    }

    __half *dA_re, *dA_im, *dB_re, *dB_im;
    float *dC_re, *dC_im;
    CHECK_CUDA(cudaMalloc(&dA_re, per_A * batch * sizeof(__half)));
    CHECK_CUDA(cudaMalloc(&dA_im, per_A * batch * sizeof(__half)));
    CHECK_CUDA(cudaMalloc(&dB_re, per_B * batch * sizeof(__half)));
    CHECK_CUDA(cudaMalloc(&dB_im, per_B * batch * sizeof(__half)));
    CHECK_CUDA(cudaMalloc(&dC_re, per_C * batch * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&dC_im, per_C * batch * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(dA_re, hA_re.data(), per_A * batch * sizeof(__half), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dA_im, hA_im.data(), per_A * batch * sizeof(__half), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dB_re, hB_re.data(), per_B * batch * sizeof(__half), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dB_im, hB_im.data(), per_B * batch * sizeof(__half), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemset(dC_re, 0, per_C * batch * sizeof(float)));
    CHECK_CUDA(cudaMemset(dC_im, 0, per_C * batch * sizeof(float)));

    cutlass_gemm_api::CutlassComplexGemm api;
    int ret = api.gemm(dA_re, dA_im, dB_re, dB_im, dC_re, dC_im,
                        M, N, K, batch,
                        ComputePrecision::FP8, OutputPrecision::FP32);
    CHECK_CUDA(cudaDeviceSynchronize());

    if (ret != 0) { printf("  FAIL: gemm (FP8, FP32) returned %d\n", ret); return 1; }

    std::vector<float> hC_re(per_C * batch), hC_im(per_C * batch);
    CHECK_CUDA(cudaMemcpy(hC_re.data(), dC_re, per_C * batch * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(hC_im.data(), dC_im, per_C * batch * sizeof(float), cudaMemcpyDeviceToHost));

    double tol = std::max(1.0, K * 0.02);
    bool pass = compare_fp32(hC_re.data(), hC_im.data(), refC, per_C * batch, tol);

    cudaFree(dA_re); cudaFree(dA_im); cudaFree(dB_re); cudaFree(dB_im);
    cudaFree(dC_re); cudaFree(dC_im);

    printf("  %s\n\n", pass ? "PASS" : "FAIL");
    return pass ? 0 : 1;
}

// Test: HERK with large K (> K_CHUNK=64)
static int test_herk_large_k() {
    printf("=== Test: herk (large K=512) ===\n");
    const int N = 64, K = 512, batch = 2;
    int64_t per_A = (int64_t)N * K;
    int64_t packed = (int64_t)N * (N + 1) / 2;

    std::vector<__half> h_A(per_A * batch * 2);
    std::vector<std::complex<double>> refA(per_A * batch);
    srand(42);
    for (int64_t i = 0; i < per_A * batch; ++i) {
        float re = (float)(rand() % 15 - 7);
        float im = (float)(rand() % 15 - 7);
        h_A[i * 2]     = __float2half(re);
        h_A[i * 2 + 1] = __float2half(im);
        refA[i] = std::complex<double>(re, im);
    }

    std::vector<std::complex<double>> refC(packed * batch, {0, 0});
    for (int b = 0; b < batch; ++b) {
        std::vector<std::complex<double>> A(refA.begin() + b * per_A,
                                             refA.begin() + (b + 1) * per_A);
        std::vector<std::complex<double>> C(packed, {0, 0});
        cpu_herk(A, C, N, K);
        for (int64_t i = 0; i < packed; ++i) refC[b * packed + i] = C[i];
    }

    __half* d_A;
    __half* d_C;
    CHECK_CUDA(cudaMalloc(&d_A, per_A * batch * 2 * sizeof(__half)));
    CHECK_CUDA(cudaMalloc(&d_C, packed * batch * 2 * sizeof(__half)));
    CHECK_CUDA(cudaMemcpy(d_A, h_A.data(), per_A * batch * 2 * sizeof(__half), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemset(d_C, 0, packed * batch * 2 * sizeof(__half)));

    cutlass_gemm_api::CutlassComplexGemm api;
    int ret = api.herk(d_A, d_C, N, K, batch);
    CHECK_CUDA(cudaDeviceSynchronize());

    if (ret != 0) { printf("  FAIL: herk returned %d\n", ret); return 1; }

    std::vector<__half> h_C(packed * batch * 2);
    CHECK_CUDA(cudaMemcpy(h_C.data(), d_C, packed * batch * 2 * sizeof(__half), cudaMemcpyDeviceToHost));

    double max_err = 0, max_ref = 0;
    #pragma omp parallel for reduction(max:max_err, max_ref)
    for (int64_t i = 0; i < packed * batch; ++i) {
        double re = __half2float(h_C[i * 2]);
        double im = __half2float(h_C[i * 2 + 1]);
        max_err = std::max(max_err, std::abs(re - refC[i].real()));
        max_err = std::max(max_err, std::abs(im - refC[i].imag()));
        max_ref = std::max(max_ref, std::abs(refC[i].real()));
        max_ref = std::max(max_ref, std::abs(refC[i].imag()));
    }
    double rel = (max_ref > 0) ? max_err / max_ref : max_err;
    // Use peak-relative tolerance: 1% of max reference magnitude.
    // Absolute tolerance (K*0.02) doesn't account for input magnitude —
    // integers [-7,7] produce outputs ~196x larger than [-0.5,0.5] inputs.
    double rel_tol = 0.01;
    bool pass = (max_ref > 0) ? (rel < rel_tol) : (max_err < 1.0);
    printf("  max_ref=%.1f  max_err=%.2f  rel=%.6f  tol=%.1f%%\n", max_ref, max_err, rel, rel_tol * 100);

    cudaFree(d_A); cudaFree(d_C);
    printf("  %s\n\n", pass ? "PASS" : "FAIL");
    return pass ? 0 : 1;
}

// Test: GEMM with non-trivial alpha/beta
static int test_gemm_alpha_beta() {
    printf("=== Test: gemm (alpha/beta, FP8, FP32 out) ===\n");
    const int M = 64, N = 64, K = 128, batch = 2;
    int64_t per_A = (int64_t)M * K, per_B = (int64_t)N * K, per_C = (int64_t)M * N;

    std::vector<__half> hA_re, hA_im, hB_re, hB_im;
    std::vector<std::complex<double>> refA, refB;
    generate_data(hA_re, hA_im, refA, per_A * batch, 42);
    generate_data(hB_re, hB_im, refB, per_B * batch, 123);

    __half *dA_re, *dA_im, *dB_re, *dB_im;
    float *dC_re, *dC_im;
    CHECK_CUDA(cudaMalloc(&dA_re, per_A * batch * sizeof(__half)));
    CHECK_CUDA(cudaMalloc(&dA_im, per_A * batch * sizeof(__half)));
    CHECK_CUDA(cudaMalloc(&dB_re, per_B * batch * sizeof(__half)));
    CHECK_CUDA(cudaMalloc(&dB_im, per_B * batch * sizeof(__half)));
    CHECK_CUDA(cudaMalloc(&dC_re, per_C * batch * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&dC_im, per_C * batch * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(dA_re, hA_re.data(), per_A * batch * sizeof(__half), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dA_im, hA_im.data(), per_A * batch * sizeof(__half), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dB_re, hB_re.data(), per_B * batch * sizeof(__half), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dB_im, hB_im.data(), per_B * batch * sizeof(__half), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemset(dC_re, 0, per_C * batch * sizeof(float)));
    CHECK_CUDA(cudaMemset(dC_im, 0, per_C * batch * sizeof(float)));

    cutlass_gemm_api::CutlassComplexGemm api;

    // Step 1: C = 1.0 * A * B + 0.0 * C
    int ret = api.gemm(dA_re, dA_im, dB_re, dB_im, dC_re, dC_im,
                       M, N, K, batch,
                       ComputePrecision::FP8, OutputPrecision::FP32,
                       1.0f, 0.0f);
    CHECK_CUDA(cudaDeviceSynchronize());
    if (ret != 0) { printf("  FAIL: gemm step 1 returned %d\n", ret); return 1; }

    // Step 2: C = 0.5 * A * B + 1.0 * C  =>  C = 1.5 * A * B
    ret = api.gemm(dA_re, dA_im, dB_re, dB_im, dC_re, dC_im,
                   M, N, K, batch,
                   ComputePrecision::FP8, OutputPrecision::FP32,
                   0.5f, 1.0f);
    CHECK_CUDA(cudaDeviceSynchronize());
    if (ret != 0) { printf("  FAIL: gemm step 2 returned %d\n", ret); return 1; }

    // CPU reference: C = 1.5 * A * B
    std::vector<std::complex<double>> refC(per_C * batch, {0, 0});
    for (int b = 0; b < batch; ++b) {
        std::vector<std::complex<double>> A(refA.begin() + b * per_A, refA.begin() + (b + 1) * per_A);
        std::vector<std::complex<double>> B(refB.begin() + b * per_B, refB.begin() + (b + 1) * per_B);
        std::vector<std::complex<double>> C(per_C, {0, 0});
        cpu_gemm(A, B, C, M, N, K, 1.5f, 0.0f);
        for (int64_t i = 0; i < per_C; ++i) refC[b * per_C + i] = C[i];
    }

    std::vector<float> hC_re(per_C * batch), hC_im(per_C * batch);
    CHECK_CUDA(cudaMemcpy(hC_re.data(), dC_re, per_C * batch * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(hC_im.data(), dC_im, per_C * batch * sizeof(float), cudaMemcpyDeviceToHost));

    double tol = std::max(2.0, K * 0.04);
    bool pass = compare_fp32(hC_re.data(), hC_im.data(), refC, per_C * batch, tol);

    cudaFree(dA_re); cudaFree(dA_im); cudaFree(dB_re); cudaFree(dB_im);
    cudaFree(dC_re); cudaFree(dC_im);
    printf("  %s\n\n", pass ? "PASS" : "FAIL");
    return pass ? 0 : 1;
}

// Test: Direct HERK kernel correctness
static int test_herk_direct() {
    printf("=== Test: herk (direct kernel, K=32) ===\n");
    const int N = 64, K = 32, batch = 4;
    int64_t per_A = (int64_t)N * K;
    int64_t packed = (int64_t)N * (N + 1) / 2;

    std::vector<__half> h_A(per_A * batch * 2);
    std::vector<std::complex<double>> refA(per_A * batch);
    srand(42);
    for (int64_t i = 0; i < per_A * batch; ++i) {
        float re = (float)(rand() % 15 - 7);
        float im = (float)(rand() % 15 - 7);
        h_A[i * 2]     = __float2half(re);
        h_A[i * 2 + 1] = __float2half(im);
        refA[i] = std::complex<double>(re, im);
    }

    std::vector<std::complex<double>> refC(packed * batch, {0, 0});
    for (int b = 0; b < batch; ++b) {
        std::vector<std::complex<double>> A(refA.begin() + b * per_A,
                                             refA.begin() + (b + 1) * per_A);
        std::vector<std::complex<double>> C(packed, {0, 0});
        cpu_herk(A, C, N, K);
        for (int64_t i = 0; i < packed; ++i) refC[b * packed + i] = C[i];
    }

    __half* d_A;
    __half* d_C;
    CHECK_CUDA(cudaMalloc(&d_A, per_A * batch * 2 * sizeof(__half)));
    CHECK_CUDA(cudaMalloc(&d_C, packed * batch * 2 * sizeof(__half)));
    CHECK_CUDA(cudaMemcpy(d_A, h_A.data(), per_A * batch * 2 * sizeof(__half), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemset(d_C, 0, packed * batch * 2 * sizeof(__half)));

    cutlass_gemm_api::CutlassComplexGemm api;
    api.set_herk_mode(cutlass_gemm_api::HerkMode::ForceDirect);
    int ret = api.herk(d_A, d_C, N, K, batch);
    CHECK_CUDA(cudaDeviceSynchronize());

    if (ret != 0) { printf("  FAIL: herk (direct) returned %d\n", ret); cudaFree(d_A); cudaFree(d_C); return 1; }

    std::vector<__half> h_C(packed * batch * 2);
    CHECK_CUDA(cudaMemcpy(h_C.data(), d_C, packed * batch * 2 * sizeof(__half), cudaMemcpyDeviceToHost));

    double max_err = 0, max_ref = 0;
    #pragma omp parallel for reduction(max:max_err, max_ref)
    for (int64_t i = 0; i < packed * batch; ++i) {
        double re = __half2float(h_C[i * 2]);
        double im = __half2float(h_C[i * 2 + 1]);
        max_err = std::max(max_err, std::abs(re - refC[i].real()));
        max_err = std::max(max_err, std::abs(im - refC[i].imag()));
        max_ref = std::max(max_ref, std::abs(refC[i].real()));
        max_ref = std::max(max_ref, std::abs(refC[i].imag()));
    }
    double rel = (max_ref > 0) ? max_err / max_ref : max_err;
    double tol = std::max(1.0, K * 0.02);
    bool pass = max_err < tol;
    printf("  max_ref=%.1f  max_err=%.2f  rel=%.6f  tol=%.1f\n", max_ref, max_err, rel, tol);

    cudaFree(d_A); cudaFree(d_C);
    printf("  %s\n\n", pass ? "PASS" : "FAIL");
    return pass ? 0 : 1;
}

// Test: Baseline HERK correctness
static int test_herk_baseline() {
    printf("=== Test: herk (baseline, K=128) ===\n");
    const int N = 64, K = 128, batch = 4;
    int64_t per_A = (int64_t)N * K;
    int64_t packed = (int64_t)N * (N + 1) / 2;

    std::vector<__half> h_A(per_A * batch * 2);
    std::vector<std::complex<double>> refA(per_A * batch);
    srand(42);
    for (int64_t i = 0; i < per_A * batch; ++i) {
        float re = (float)(rand() % 15 - 7);
        float im = (float)(rand() % 15 - 7);
        h_A[i * 2]     = __float2half(re);
        h_A[i * 2 + 1] = __float2half(im);
        refA[i] = std::complex<double>(re, im);
    }

    std::vector<std::complex<double>> refC(packed * batch, {0, 0});
    for (int b = 0; b < batch; ++b) {
        std::vector<std::complex<double>> A(refA.begin() + b * per_A,
                                             refA.begin() + (b + 1) * per_A);
        std::vector<std::complex<double>> C(packed, {0, 0});
        cpu_herk(A, C, N, K);
        for (int64_t i = 0; i < packed; ++i) refC[b * packed + i] = C[i];
    }

    __half* d_A;
    __half* d_C;
    CHECK_CUDA(cudaMalloc(&d_A, per_A * batch * 2 * sizeof(__half)));
    CHECK_CUDA(cudaMalloc(&d_C, packed * batch * 2 * sizeof(__half)));
    CHECK_CUDA(cudaMemcpy(d_A, h_A.data(), per_A * batch * 2 * sizeof(__half), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemset(d_C, 0, packed * batch * 2 * sizeof(__half)));

    cutlass_gemm_api::CutlassComplexGemm api;
    api.set_herk_mode(cutlass_gemm_api::HerkMode::ForceBaseline);
    int ret = api.herk(d_A, d_C, N, K, batch);
    CHECK_CUDA(cudaDeviceSynchronize());

    if (ret != 0) { printf("  FAIL: herk (baseline) returned %d\n", ret); cudaFree(d_A); cudaFree(d_C); return 1; }

    std::vector<__half> h_C(packed * batch * 2);
    CHECK_CUDA(cudaMemcpy(h_C.data(), d_C, packed * batch * 2 * sizeof(__half), cudaMemcpyDeviceToHost));

    double max_err = 0, max_ref = 0;
    #pragma omp parallel for reduction(max:max_err, max_ref)
    for (int64_t i = 0; i < packed * batch; ++i) {
        double re = __half2float(h_C[i * 2]);
        double im = __half2float(h_C[i * 2 + 1]);
        max_err = std::max(max_err, std::abs(re - refC[i].real()));
        max_err = std::max(max_err, std::abs(im - refC[i].imag()));
        max_ref = std::max(max_ref, std::abs(refC[i].real()));
        max_ref = std::max(max_ref, std::abs(refC[i].imag()));
    }
    double rel = (max_ref > 0) ? max_err / max_ref : max_err;
    double tol = std::max(1.0, K * 0.02);
    bool pass = max_err < tol;
    printf("  max_ref=%.1f  max_err=%.2f  rel=%.6f  tol=%.1f\n", max_ref, max_err, rel, tol);

    cudaFree(d_A); cudaFree(d_C);
    printf("  %s\n\n", pass ? "PASS" : "FAIL");
    return pass ? 0 : 1;
}

// Test: Direct vs Baseline self-consistency
static int test_herk_direct_vs_baseline() {
    printf("=== Test: herk (direct vs baseline) ===\n");
    const int N = 64, K = 64, batch = 4;
    int64_t per_A = (int64_t)N * K;
    int64_t packed = (int64_t)N * (N + 1) / 2;

    std::vector<__half> h_A(per_A * batch * 2);
    srand(42);
    for (int64_t i = 0; i < per_A * batch; ++i) {
        float re = (float)(rand() % 15 - 7);
        float im = (float)(rand() % 15 - 7);
        h_A[i * 2]     = __float2half(re);
        h_A[i * 2 + 1] = __float2half(im);
    }

    __half* d_A;
    __half *d_C_direct, *d_C_baseline;
    CHECK_CUDA(cudaMalloc(&d_A, per_A * batch * 2 * sizeof(__half)));
    CHECK_CUDA(cudaMalloc(&d_C_direct, packed * batch * 2 * sizeof(__half)));
    CHECK_CUDA(cudaMalloc(&d_C_baseline, packed * batch * 2 * sizeof(__half)));
    CHECK_CUDA(cudaMemcpy(d_A, h_A.data(), per_A * batch * 2 * sizeof(__half), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemset(d_C_direct, 0, packed * batch * 2 * sizeof(__half)));
    CHECK_CUDA(cudaMemset(d_C_baseline, 0, packed * batch * 2 * sizeof(__half)));

    // Run direct
    {
        cutlass_gemm_api::CutlassComplexGemm api;
        api.set_herk_mode(cutlass_gemm_api::HerkMode::ForceDirect);
        int ret = api.herk(d_A, d_C_direct, N, K, batch);
        CHECK_CUDA(cudaDeviceSynchronize());
        if (ret != 0) { printf("  FAIL: herk (direct) returned %d\n", ret);
            cudaFree(d_A); cudaFree(d_C_direct); cudaFree(d_C_baseline); return 1; }
    }

    // Run baseline
    {
        cutlass_gemm_api::CutlassComplexGemm api;
        api.set_herk_mode(cutlass_gemm_api::HerkMode::ForceBaseline);
        int ret = api.herk(d_A, d_C_baseline, N, K, batch);
        CHECK_CUDA(cudaDeviceSynchronize());
        if (ret != 0) { printf("  FAIL: herk (baseline) returned %d\n", ret);
            cudaFree(d_A); cudaFree(d_C_direct); cudaFree(d_C_baseline); return 1; }
    }

    // Compare direct vs baseline
    std::vector<__half> h_direct(packed * batch * 2), h_baseline(packed * batch * 2);
    CHECK_CUDA(cudaMemcpy(h_direct.data(), d_C_direct, packed * batch * 2 * sizeof(__half), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_baseline.data(), d_C_baseline, packed * batch * 2 * sizeof(__half), cudaMemcpyDeviceToHost));

    double max_err = 0, max_ref = 0;
    #pragma omp parallel for reduction(max:max_err, max_ref)
    for (int64_t i = 0; i < packed * batch * 2; ++i) {
        double d = __half2float(h_direct[i]);
        double b = __half2float(h_baseline[i]);
        max_err = std::max(max_err, std::abs(d - b));
        max_ref = std::max(max_ref, std::abs(b));
    }
    double rel = (max_ref > 0) ? max_err / max_ref : max_err;
    double tol = std::max(1.0, K * 0.02);
    bool pass = max_err < tol;
    printf("  max_ref=%.1f  max_err=%.2f  rel=%.6f  tol=%.1f\n", max_ref, max_err, rel, tol);

    cudaFree(d_A); cudaFree(d_C_direct); cudaFree(d_C_baseline);
    printf("  %s\n\n", pass ? "PASS" : "FAIL");
    return pass ? 0 : 1;
}

// Test: FP6 HERK correctness (integer input, exact in E3M2)
static int test_herk_fp6() {
#if defined(COMPLEX_FP8_TARGET_SM90) || !defined(COMPLEX_SM100_ENABLE_FP6)
    printf("=== Test: herk_fp6 === SKIPPED (FP6 not available)\n\n");
    return -1;
#else
    printf("=== Test: herk (FP6, FP16 out) ===\n");
    const int N = 128, K = 128, batch = 2;
    int64_t per_A = (int64_t)N * K;
    int64_t packed = (int64_t)N * (N + 1) / 2;

    std::vector<__half> h_A(per_A * batch * 2);
    std::vector<std::complex<double>> refA(per_A * batch);
    srand(42);
    for (int64_t i = 0; i < per_A * batch; ++i) {
        float re = (float)(rand() % 15 - 7);
        float im = (float)(rand() % 15 - 7);
        h_A[i * 2]     = __float2half(re);
        h_A[i * 2 + 1] = __float2half(im);
        refA[i] = std::complex<double>(re, im);
    }

    std::vector<std::complex<double>> refC(packed * batch, {0, 0});
    for (int b = 0; b < batch; ++b) {
        std::vector<std::complex<double>> A(refA.begin() + b * per_A,
                                             refA.begin() + (b + 1) * per_A);
        std::vector<std::complex<double>> C(packed, {0, 0});
        cpu_herk(A, C, N, K);
        for (int64_t i = 0; i < packed; ++i) refC[b * packed + i] = C[i];
    }

    __half* d_A;
    __half* d_C;
    CHECK_CUDA(cudaMalloc(&d_A, per_A * batch * 2 * sizeof(__half)));
    CHECK_CUDA(cudaMalloc(&d_C, packed * batch * 2 * sizeof(__half)));
    CHECK_CUDA(cudaMemcpy(d_A, h_A.data(), per_A * batch * 2 * sizeof(__half), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemset(d_C, 0, packed * batch * 2 * sizeof(__half)));

    cutlass_gemm_api::CutlassComplexGemm api;
    int ret = api.herk(d_A, d_C, N, K, batch,
                       InputPrecision::FP16, ComputePrecision::FP6,
                       OutputPrecision::FP16);
    CHECK_CUDA(cudaDeviceSynchronize());

    if (ret != 0) { printf("  FAIL: herk (FP6) returned %d\n", ret); cudaFree(d_A); cudaFree(d_C); return 1; }

    std::vector<__half> h_C(packed * batch * 2);
    CHECK_CUDA(cudaMemcpy(h_C.data(), d_C, packed * batch * 2 * sizeof(__half), cudaMemcpyDeviceToHost));

    double max_err = 0, max_ref = 0;
    #pragma omp parallel for reduction(max:max_err, max_ref)
    for (int64_t i = 0; i < packed * batch; ++i) {
        double re = __half2float(h_C[i * 2]);
        double im = __half2float(h_C[i * 2 + 1]);
        max_err = std::max(max_err, std::abs(re - refC[i].real()));
        max_err = std::max(max_err, std::abs(im - refC[i].imag()));
        max_ref = std::max(max_ref, std::abs(refC[i].real()));
        max_ref = std::max(max_ref, std::abs(refC[i].imag()));
    }
    double rel = (max_ref > 0) ? max_err / max_ref : max_err;
    // FP6 E3M2 has 2 mantissa bits (vs FP8's 3), slightly more accumulation error
    double tol = std::max(1.0, K * 0.04);
    bool pass = max_err < tol;
    printf("  max_ref=%.1f  max_err=%.2f  rel=%.6f  tol=%.1f\n", max_ref, max_err, rel, tol);

    cudaFree(d_A); cudaFree(d_C);
    printf("  %s\n\n", pass ? "PASS" : "FAIL");
    return pass ? 0 : 1;
#endif
}

// Test: FP4 HERK correctness (exact integer subset [-4,4])
static int test_herk_fp4() {
#if defined(COMPLEX_FP8_TARGET_SM90) || !defined(COMPLEX_SM100_ENABLE_FP4)
    printf("=== Test: herk_fp4 === SKIPPED (FP4 not available)\n\n");
    return -1;
#else
    printf("=== Test: herk (FP4, FP16 out, [-4,4]) ===\n");
    const int N = 128, K = 256, batch = 2;
    int64_t per_A = (int64_t)N * K;
    int64_t packed = (int64_t)N * (N + 1) / 2;

    std::vector<__half> h_A(per_A * batch * 2);
    std::vector<std::complex<double>> refA(per_A * batch);
    srand(42);
    for (int64_t i = 0; i < per_A * batch; ++i) {
        float re = (float)(rand() % 9 - 4);
        float im = (float)(rand() % 9 - 4);
        h_A[i * 2]     = __float2half(re);
        h_A[i * 2 + 1] = __float2half(im);
        refA[i] = std::complex<double>(re, im);
    }

    std::vector<std::complex<double>> refC(packed * batch, {0, 0});
    for (int b = 0; b < batch; ++b) {
        std::vector<std::complex<double>> A(refA.begin() + b * per_A,
                                             refA.begin() + (b + 1) * per_A);
        std::vector<std::complex<double>> C(packed, {0, 0});
        cpu_herk(A, C, N, K);
        for (int64_t i = 0; i < packed; ++i) refC[b * packed + i] = C[i];
    }

    __half* d_A;
    __half* d_C;
    CHECK_CUDA(cudaMalloc(&d_A, per_A * batch * 2 * sizeof(__half)));
    CHECK_CUDA(cudaMalloc(&d_C, packed * batch * 2 * sizeof(__half)));
    CHECK_CUDA(cudaMemcpy(d_A, h_A.data(), per_A * batch * 2 * sizeof(__half), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemset(d_C, 0, packed * batch * 2 * sizeof(__half)));

    cutlass_gemm_api::CutlassComplexGemm api;
    int ret = api.herk(d_A, d_C, N, K, batch,
                       InputPrecision::FP16, ComputePrecision::FP4,
                       OutputPrecision::FP16);
    CHECK_CUDA(cudaDeviceSynchronize());

    if (ret != 0) { printf("  FAIL: herk (FP4) returned %d\n", ret); cudaFree(d_A); cudaFree(d_C); return 1; }

    std::vector<__half> h_C(packed * batch * 2);
    CHECK_CUDA(cudaMemcpy(h_C.data(), d_C, packed * batch * 2 * sizeof(__half), cudaMemcpyDeviceToHost));

    double max_err = 0, max_ref = 0;
    #pragma omp parallel for reduction(max:max_err, max_ref)
    for (int64_t i = 0; i < packed * batch; ++i) {
        double re = __half2float(h_C[i * 2]);
        double im = __half2float(h_C[i * 2 + 1]);
        max_err = std::max(max_err, std::abs(re - refC[i].real()));
        max_err = std::max(max_err, std::abs(im - refC[i].imag()));
        max_ref = std::max(max_ref, std::abs(refC[i].real()));
        max_ref = std::max(max_ref, std::abs(refC[i].imag()));
    }
    double rel = (max_ref > 0) ? max_err / max_ref : max_err;
    double tol = std::max(1.0, K * 0.02);
    bool pass = max_err < tol;
    printf("  max_ref=%.1f  max_err=%.2f  rel=%.6f  tol=%.1f\n", max_ref, max_err, rel, tol);

    cudaFree(d_A); cudaFree(d_C);
    printf("  %s\n\n", pass ? "PASS" : "FAIL");
    return pass ? 0 : 1;
#endif
}

// Test: FP4 HERK structural error gate (lossy [-7,7] input)
static int test_herk_fp4_lossy() {
#if defined(COMPLEX_FP8_TARGET_SM90) || !defined(COMPLEX_SM100_ENABLE_FP4)
    printf("=== Test: herk_fp4_lossy === SKIPPED (FP4 not available)\n\n");
    return -1;
#else
    printf("=== Test: herk (FP4, FP16 out, [-7,7] lossy) ===\n");
    const int N = 128, K = 256, batch = 2;
    int64_t per_A = (int64_t)N * K;
    int64_t packed = (int64_t)N * (N + 1) / 2;

    std::vector<__half> h_A(per_A * batch * 2);
    std::vector<std::complex<double>> refA(per_A * batch);
    srand(42);
    for (int64_t i = 0; i < per_A * batch; ++i) {
        float re = (float)(rand() % 15 - 7);
        float im = (float)(rand() % 15 - 7);
        h_A[i * 2]     = __float2half(re);
        h_A[i * 2 + 1] = __float2half(im);
        refA[i] = std::complex<double>(re, im);
    }

    std::vector<std::complex<double>> refC(packed * batch, {0, 0});
    for (int b = 0; b < batch; ++b) {
        std::vector<std::complex<double>> A(refA.begin() + b * per_A,
                                             refA.begin() + (b + 1) * per_A);
        std::vector<std::complex<double>> C(packed, {0, 0});
        cpu_herk(A, C, N, K);
        for (int64_t i = 0; i < packed; ++i) refC[b * packed + i] = C[i];
    }

    __half* d_A;
    __half* d_C;
    CHECK_CUDA(cudaMalloc(&d_A, per_A * batch * 2 * sizeof(__half)));
    CHECK_CUDA(cudaMalloc(&d_C, packed * batch * 2 * sizeof(__half)));
    CHECK_CUDA(cudaMemcpy(d_A, h_A.data(), per_A * batch * 2 * sizeof(__half), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemset(d_C, 0, packed * batch * 2 * sizeof(__half)));

    cutlass_gemm_api::CutlassComplexGemm api;
    int ret = api.herk(d_A, d_C, N, K, batch,
                       InputPrecision::FP16, ComputePrecision::FP4,
                       OutputPrecision::FP16);
    CHECK_CUDA(cudaDeviceSynchronize());

    if (ret != 0) { printf("  FAIL: herk (FP4) returned %d\n", ret); cudaFree(d_A); cudaFree(d_C); return 1; }

    std::vector<__half> h_C(packed * batch * 2);
    CHECK_CUDA(cudaMemcpy(h_C.data(), d_C, packed * batch * 2 * sizeof(__half), cudaMemcpyDeviceToHost));

    double max_err = 0, max_ref = 0;
    #pragma omp parallel for reduction(max:max_err, max_ref)
    for (int64_t i = 0; i < packed * batch; ++i) {
        double re = __half2float(h_C[i * 2]);
        double im = __half2float(h_C[i * 2 + 1]);
        max_err = std::max(max_err, std::abs(re - refC[i].real()));
        max_err = std::max(max_err, std::abs(im - refC[i].imag()));
        max_ref = std::max(max_ref, std::abs(refC[i].real()));
        max_ref = std::max(max_ref, std::abs(refC[i].imag()));
    }
    double rel = (max_ref > 0) ? max_err / max_ref : max_err;
    // FP4 E2M1 can't represent 5 or 7 (5->4, 7->8), expect ~15-20% structural error.
    // Use 25% of peak reference as tolerance (quantization error is proportional to signal).
    bool pass = rel < 0.25;
    printf("  max_ref=%.1f  max_err=%.2f  rel=%.6f  rel_limit=0.25\n", max_ref, max_err, rel);

    cudaFree(d_A); cudaFree(d_C);
    printf("  %s\n\n", pass ? "PASS" : "FAIL");
    return pass ? 0 : 1;
#endif
}

// Test: HERK with K not aligned to K_CHUNK (64)
static int test_herk_k_not_aligned() {
    printf("=== Test: herk (K=96, not aligned to K_CHUNK) ===\n");
    const int N = 64, K = 96, batch = 2;
    int64_t per_A = (int64_t)N * K;
    int64_t packed = (int64_t)N * (N + 1) / 2;

    std::vector<__half> h_A(per_A * batch * 2);
    std::vector<std::complex<double>> refA(per_A * batch);
    srand(42);
    for (int64_t i = 0; i < per_A * batch; ++i) {
        float re = (float)(rand() % 15 - 7);
        float im = (float)(rand() % 15 - 7);
        h_A[i * 2]     = __float2half(re);
        h_A[i * 2 + 1] = __float2half(im);
        refA[i] = std::complex<double>(re, im);
    }

    std::vector<std::complex<double>> refC(packed * batch, {0, 0});
    for (int b = 0; b < batch; ++b) {
        std::vector<std::complex<double>> A(refA.begin() + b * per_A,
                                             refA.begin() + (b + 1) * per_A);
        std::vector<std::complex<double>> C(packed, {0, 0});
        cpu_herk(A, C, N, K);
        for (int64_t i = 0; i < packed; ++i) refC[b * packed + i] = C[i];
    }

    __half* d_A;
    __half* d_C;
    CHECK_CUDA(cudaMalloc(&d_A, per_A * batch * 2 * sizeof(__half)));
    CHECK_CUDA(cudaMalloc(&d_C, packed * batch * 2 * sizeof(__half)));
    CHECK_CUDA(cudaMemcpy(d_A, h_A.data(), per_A * batch * 2 * sizeof(__half), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemset(d_C, 0, packed * batch * 2 * sizeof(__half)));

    cutlass_gemm_api::CutlassComplexGemm api;
    int ret = api.herk(d_A, d_C, N, K, batch);
    CHECK_CUDA(cudaDeviceSynchronize());

    if (ret != 0) { printf("  FAIL: herk (K=96) returned %d\n", ret); cudaFree(d_A); cudaFree(d_C); return 1; }

    std::vector<__half> h_C(packed * batch * 2);
    CHECK_CUDA(cudaMemcpy(h_C.data(), d_C, packed * batch * 2 * sizeof(__half), cudaMemcpyDeviceToHost));

    double max_err = 0, max_ref = 0;
    #pragma omp parallel for reduction(max:max_err, max_ref)
    for (int64_t i = 0; i < packed * batch; ++i) {
        double re = __half2float(h_C[i * 2]);
        double im = __half2float(h_C[i * 2 + 1]);
        max_err = std::max(max_err, std::abs(re - refC[i].real()));
        max_err = std::max(max_err, std::abs(im - refC[i].imag()));
        max_ref = std::max(max_ref, std::abs(refC[i].real()));
        max_ref = std::max(max_ref, std::abs(refC[i].imag()));
    }
    double rel = (max_ref > 0) ? max_err / max_ref : max_err;
    double tol = std::max(1.0, K * 0.02);
    bool pass = max_err < tol;
    printf("  max_ref=%.1f  max_err=%.2f  rel=%.6f  tol=%.1f\n", max_ref, max_err, rel, tol);

    cudaFree(d_A); cudaFree(d_C);
    printf("  %s\n\n", pass ? "PASS" : "FAIL");
    return pass ? 0 : 1;
}

// Test: GEMM with K not aligned to MMA_K
static int test_gemm_k_not_aligned() {
    printf("=== Test: gemm (K=96, not aligned to MMA_K) ===\n");
    const int M = 64, N = 64, K = 96, batch = 1;
    int64_t per_A = (int64_t)M * K, per_B = (int64_t)N * K, per_C = (int64_t)M * N;

    std::vector<__half> hA_re, hA_im, hB_re, hB_im;
    std::vector<std::complex<double>> refA, refB;
    generate_data(hA_re, hA_im, refA, per_A * batch, 42);
    generate_data(hB_re, hB_im, refB, per_B * batch, 123);

    std::vector<std::complex<double>> refC(per_C * batch, {0, 0});
    cpu_gemm(refA, refB, refC, M, N, K, 1.0f, 0.0f);

    __half *dA_re, *dA_im, *dB_re, *dB_im, *dC_re, *dC_im;
    CHECK_CUDA(cudaMalloc(&dA_re, per_A * batch * sizeof(__half)));
    CHECK_CUDA(cudaMalloc(&dA_im, per_A * batch * sizeof(__half)));
    CHECK_CUDA(cudaMalloc(&dB_re, per_B * batch * sizeof(__half)));
    CHECK_CUDA(cudaMalloc(&dB_im, per_B * batch * sizeof(__half)));
    CHECK_CUDA(cudaMalloc(&dC_re, per_C * batch * sizeof(__half)));
    CHECK_CUDA(cudaMalloc(&dC_im, per_C * batch * sizeof(__half)));
    CHECK_CUDA(cudaMemcpy(dA_re, hA_re.data(), per_A * batch * sizeof(__half), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dA_im, hA_im.data(), per_A * batch * sizeof(__half), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dB_re, hB_re.data(), per_B * batch * sizeof(__half), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dB_im, hB_im.data(), per_B * batch * sizeof(__half), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemset(dC_re, 0, per_C * batch * sizeof(__half)));
    CHECK_CUDA(cudaMemset(dC_im, 0, per_C * batch * sizeof(__half)));

    cutlass_gemm_api::CutlassComplexGemm api;
    int ret = api.gemm(dA_re, dA_im, dB_re, dB_im, dC_re, dC_im,
                        M, N, K, batch,
                        ComputePrecision::FP8, OutputPrecision::FP16);
    CHECK_CUDA(cudaDeviceSynchronize());

    if (ret != 0) { printf("  FAIL: gemm (K=96) returned %d\n", ret);
        cudaFree(dA_re); cudaFree(dA_im); cudaFree(dB_re); cudaFree(dB_im);
        cudaFree(dC_re); cudaFree(dC_im); return 1; }

    std::vector<__half> hC_re(per_C * batch), hC_im(per_C * batch);
    CHECK_CUDA(cudaMemcpy(hC_re.data(), dC_re, per_C * batch * sizeof(__half), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(hC_im.data(), dC_im, per_C * batch * sizeof(__half), cudaMemcpyDeviceToHost));

    double tol = std::max(1.0, K * 0.02);
    bool pass = compare_fp16(hC_re.data(), hC_im.data(), refC, per_C * batch, tol);

    cudaFree(dA_re); cudaFree(dA_im); cudaFree(dB_re); cudaFree(dB_im);
    cudaFree(dC_re); cudaFree(dC_im);
    printf("  %s\n\n", pass ? "PASS" : "FAIL");
    return pass ? 0 : 1;
}

// Test: GEMM with continuous (non-integer) FP16 data
static int test_gemm_continuous() {
    printf("=== Test: gemm (continuous data, FP8, FP32 out) ===\n");
    const int M = 64, N = 64, K = 128, batch = 2;
    int64_t per_A = (int64_t)M * K, per_B = (int64_t)N * K, per_C = (int64_t)M * N;

    std::vector<__half> hA_re, hA_im, hB_re, hB_im;
    std::vector<std::complex<double>> refA, refB;
    generate_data_continuous(hA_re, hA_im, refA, per_A * batch, 42);
    generate_data_continuous(hB_re, hB_im, refB, per_B * batch, 123);

    std::vector<std::complex<double>> refC(per_C * batch, {0, 0});
    for (int b = 0; b < batch; ++b) {
        std::vector<std::complex<double>> A(refA.begin() + b * per_A, refA.begin() + (b + 1) * per_A);
        std::vector<std::complex<double>> B(refB.begin() + b * per_B, refB.begin() + (b + 1) * per_B);
        std::vector<std::complex<double>> C(per_C, {0, 0});
        cpu_gemm(A, B, C, M, N, K, 1.0f, 0.0f);
        for (int64_t i = 0; i < per_C; ++i) refC[b * per_C + i] = C[i];
    }

    __half *dA_re, *dA_im, *dB_re, *dB_im;
    float *dC_re, *dC_im;
    CHECK_CUDA(cudaMalloc(&dA_re, per_A * batch * sizeof(__half)));
    CHECK_CUDA(cudaMalloc(&dA_im, per_A * batch * sizeof(__half)));
    CHECK_CUDA(cudaMalloc(&dB_re, per_B * batch * sizeof(__half)));
    CHECK_CUDA(cudaMalloc(&dB_im, per_B * batch * sizeof(__half)));
    CHECK_CUDA(cudaMalloc(&dC_re, per_C * batch * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&dC_im, per_C * batch * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(dA_re, hA_re.data(), per_A * batch * sizeof(__half), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dA_im, hA_im.data(), per_A * batch * sizeof(__half), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dB_re, hB_re.data(), per_B * batch * sizeof(__half), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dB_im, hB_im.data(), per_B * batch * sizeof(__half), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemset(dC_re, 0, per_C * batch * sizeof(float)));
    CHECK_CUDA(cudaMemset(dC_im, 0, per_C * batch * sizeof(float)));

    cutlass_gemm_api::CutlassComplexGemm api;
    int ret = api.gemm(dA_re, dA_im, dB_re, dB_im, dC_re, dC_im,
                        M, N, K, batch,
                        ComputePrecision::FP8, OutputPrecision::FP32);
    CHECK_CUDA(cudaDeviceSynchronize());

    if (ret != 0) { printf("  FAIL: gemm (continuous) returned %d\n", ret);
        cudaFree(dA_re); cudaFree(dA_im); cudaFree(dB_re); cudaFree(dB_im);
        cudaFree(dC_re); cudaFree(dC_im); return 1; }

    std::vector<float> hC_re(per_C * batch), hC_im(per_C * batch);
    CHECK_CUDA(cudaMemcpy(hC_re.data(), dC_re, per_C * batch * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(hC_im.data(), dC_im, per_C * batch * sizeof(float), cudaMemcpyDeviceToHost));

    double max_err = 0, max_ref = 0;
    #pragma omp parallel for reduction(max:max_err, max_ref)
    for (int64_t i = 0; i < per_C * batch; ++i) {
        max_err = std::max(max_err, std::abs((double)hC_re[i] - refC[i].real()));
        max_err = std::max(max_err, std::abs((double)hC_im[i] - refC[i].imag()));
        max_ref = std::max(max_ref, std::abs(refC[i].real()));
        max_ref = std::max(max_ref, std::abs(refC[i].imag()));
    }
    double rel = (max_ref > 0) ? max_err / max_ref : max_err;
    double tol_abs = std::max(1.0, K * 0.05);
    bool pass = max_err < tol_abs && rel < 0.05;
    printf("  max_ref=%.4f  max_err=%.6f  rel=%.6f  tol_abs=%.1f  rel_limit=0.05\n",
           max_ref, max_err, rel, tol_abs);

    cudaFree(dA_re); cudaFree(dA_im); cudaFree(dB_re); cudaFree(dB_im);
    cudaFree(dC_re); cudaFree(dC_im);
    printf("  %s\n\n", pass ? "PASS" : "FAIL");
    return pass ? 0 : 1;
}

// Test: HERK with continuous (non-integer) FP16 data
static int test_herk_continuous() {
    printf("=== Test: herk (continuous data, FP8, FP32 out) ===\n");
    const int N = 128, K = 128, batch = 2;
    int64_t per_A = (int64_t)N * K;
    int64_t packed = (int64_t)N * (N + 1) / 2;

    std::vector<__half> h_A;
    std::vector<std::complex<double>> refA;
    generate_herk_data_continuous(h_A, refA, per_A, batch, 42);

    std::vector<std::complex<double>> refC(packed * batch, {0, 0});
    for (int b = 0; b < batch; ++b) {
        std::vector<std::complex<double>> A(refA.begin() + b * per_A,
                                             refA.begin() + (b + 1) * per_A);
        std::vector<std::complex<double>> C(packed, {0, 0});
        cpu_herk(A, C, N, K);
        for (int64_t i = 0; i < packed; ++i) refC[b * packed + i] = C[i];
    }

    __half* d_A;
    float* d_C;
    CHECK_CUDA(cudaMalloc(&d_A, per_A * batch * 2 * sizeof(__half)));
    CHECK_CUDA(cudaMalloc(&d_C, packed * batch * 2 * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(d_A, h_A.data(), per_A * batch * 2 * sizeof(__half), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemset(d_C, 0, packed * batch * 2 * sizeof(float)));

    cutlass_gemm_api::CutlassComplexGemm api;
    int ret = api.herk(d_A, d_C, N, K, batch,
                       InputPrecision::FP16, ComputePrecision::FP8,
                       OutputPrecision::FP32);
    CHECK_CUDA(cudaDeviceSynchronize());

    if (ret != 0) { printf("  FAIL: herk (continuous) returned %d\n", ret); cudaFree(d_A); cudaFree(d_C); return 1; }

    std::vector<float> h_C(packed * batch * 2);
    CHECK_CUDA(cudaMemcpy(h_C.data(), d_C, packed * batch * 2 * sizeof(float), cudaMemcpyDeviceToHost));

    double max_err = 0, max_ref = 0;
    #pragma omp parallel for reduction(max:max_err, max_ref)
    for (int64_t i = 0; i < packed * batch; ++i) {
        double re = h_C[i * 2], im = h_C[i * 2 + 1];
        max_err = std::max(max_err, std::abs(re - refC[i].real()));
        max_err = std::max(max_err, std::abs(im - refC[i].imag()));
        max_ref = std::max(max_ref, std::abs(refC[i].real()));
        max_ref = std::max(max_ref, std::abs(refC[i].imag()));
    }
    double rel = (max_ref > 0) ? max_err / max_ref : max_err;
    double tol_abs = std::max(1.0, K * 0.05);
    bool pass = max_err < tol_abs && rel < 0.05;
    printf("  max_ref=%.4f  max_err=%.6f  rel=%.6f  tol_abs=%.1f  rel_limit=0.05\n",
           max_ref, max_err, rel, tol_abs);

    cudaFree(d_A); cudaFree(d_C);
    printf("  %s\n\n", pass ? "PASS" : "FAIL");
    return pass ? 0 : 1;
}

// Helper: generate INT4 sign-magnitude QC data and double-precision reference
static void generate_int4_data(std::vector<uint8_t>& h_qc,
    std::vector<std::complex<double>>& ref, int64_t n, unsigned seed = 42) {
    h_qc.resize(n);
    ref.resize(n);
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> dist(0, 255);
    for (int64_t i = 0; i < n; ++i) {
        uint8_t byte = static_cast<uint8_t>(dist(rng));
        h_qc[i] = byte;
        // Decode sign-magnitude nibbles
        int re_nib = (byte >> 4) & 0xF;
        int im_nib = byte & 0xF;
        auto decode_nib = [](int nib) -> double {
            int sign = (nib >> 3) & 1;
            int mag = nib & 7;
            return sign ? -(double)mag : (double)mag;
        };
        ref[i] = std::complex<double>(decode_nib(re_nib), decode_nib(im_nib));
    }
}

// Test: gemm() with INT4 input, FP8 compute, FP32 output
static int test_gemm_int4_fp32() {
    printf("=== Test: gemm (INT4 input, FP8, FP32 out) ===\n");
    const int M = 64, N = 64, K = 128, batch = 2;
    int64_t per_A = (int64_t)M * K, per_B = (int64_t)N * K, per_C = (int64_t)M * N;

    // INT4 A data
    std::vector<uint8_t> hA_qc;
    std::vector<std::complex<double>> refA;
    generate_int4_data(hA_qc, refA, per_A * batch, 42);

    // FP16 B data
    std::vector<__half> hB_re, hB_im;
    std::vector<std::complex<double>> refB;
    generate_data(hB_re, hB_im, refB, per_B * batch, 123);

    // CPU reference
    std::vector<std::complex<double>> refC(per_C * batch, {0, 0});
    for (int b = 0; b < batch; ++b) {
        std::vector<std::complex<double>> A(refA.begin() + b * per_A, refA.begin() + (b + 1) * per_A);
        std::vector<std::complex<double>> B(refB.begin() + b * per_B, refB.begin() + (b + 1) * per_B);
        std::vector<std::complex<double>> C(per_C, {0, 0});
        cpu_gemm(A, B, C, M, N, K, 1.0f, 0.0f);
        for (int64_t i = 0; i < per_C; ++i) refC[b * per_C + i] = C[i];
    }

    // Upload: A as INT4 bytes, B as FP16 planar
    uint8_t* dA_qc;
    __half *dB_re, *dB_im;
    float *dC_re, *dC_im;
    CHECK_CUDA(cudaMalloc(&dA_qc, per_A * batch));
    CHECK_CUDA(cudaMalloc(&dB_re, per_B * batch * sizeof(__half)));
    CHECK_CUDA(cudaMalloc(&dB_im, per_B * batch * sizeof(__half)));
    CHECK_CUDA(cudaMalloc(&dC_re, per_C * batch * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&dC_im, per_C * batch * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(dA_qc, hA_qc.data(), per_A * batch, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dB_re, hB_re.data(), per_B * batch * sizeof(__half), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dB_im, hB_im.data(), per_B * batch * sizeof(__half), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemset(dC_re, 0, per_C * batch * sizeof(float)));
    CHECK_CUDA(cudaMemset(dC_im, 0, per_C * batch * sizeof(float)));

    cutlass_gemm_api::CutlassComplexGemm api;
    // A_re is reinterpreted as const uint8_t* when input==INT4
    int ret = api.gemm(reinterpret_cast<const __half*>(dA_qc), nullptr,
                        dB_re, dB_im, dC_re, dC_im,
                        M, N, K, batch,
                        ComputePrecision::FP8, OutputPrecision::FP32,
                        1.0f, 0.0f, nullptr, false, InputPrecision::INT4);
    CHECK_CUDA(cudaDeviceSynchronize());

    if (ret != 0) { printf("  FAIL: gemm (INT4, FP8, FP32) returned %d\n", ret); return 1; }

    std::vector<float> hC_re(per_C * batch), hC_im(per_C * batch);
    CHECK_CUDA(cudaMemcpy(hC_re.data(), dC_re, per_C * batch * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(hC_im.data(), dC_im, per_C * batch * sizeof(float), cudaMemcpyDeviceToHost));

    double tol = std::max(1.0, K * 0.02);
    bool pass = compare_fp32(hC_re.data(), hC_im.data(), refC, per_C * batch, tol);

    cudaFree(dA_qc); cudaFree(dB_re); cudaFree(dB_im);
    cudaFree(dC_re); cudaFree(dC_im);

    printf("  %s\n\n", pass ? "PASS" : "FAIL");
    return pass ? 0 : 1;
}

// Test: prepare_b(FP16 B) + gemm_prepared(INT4 A)
static int test_gemm_prepared_int4() {
    printf("=== Test: prepare_b + gemm_prepared (INT4 A) ===\n");
    const int M = 64, N = 64, K = 128, batch = 2;
    int64_t per_A = (int64_t)M * K, per_B = (int64_t)N * K, per_C = (int64_t)M * N;

    std::vector<uint8_t> hA_qc;
    std::vector<std::complex<double>> refA;
    generate_int4_data(hA_qc, refA, per_A * batch, 42);

    std::vector<__half> hB_re, hB_im;
    std::vector<std::complex<double>> refB;
    generate_data(hB_re, hB_im, refB, per_B * batch, 123);

    std::vector<std::complex<double>> refC(per_C * batch, {0, 0});
    for (int b = 0; b < batch; ++b) {
        std::vector<std::complex<double>> A(refA.begin() + b * per_A, refA.begin() + (b + 1) * per_A);
        std::vector<std::complex<double>> B(refB.begin() + b * per_B, refB.begin() + (b + 1) * per_B);
        std::vector<std::complex<double>> C(per_C, {0, 0});
        cpu_gemm(A, B, C, M, N, K, 1.0f, 0.0f);
        for (int64_t i = 0; i < per_C; ++i) refC[b * per_C + i] = C[i];
    }

    uint8_t* dA_qc;
    __half *dB_re, *dB_im;
    float *dC_re, *dC_im;
    CHECK_CUDA(cudaMalloc(&dA_qc, per_A * batch));
    CHECK_CUDA(cudaMalloc(&dB_re, per_B * batch * sizeof(__half)));
    CHECK_CUDA(cudaMalloc(&dB_im, per_B * batch * sizeof(__half)));
    CHECK_CUDA(cudaMalloc(&dC_re, per_C * batch * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&dC_im, per_C * batch * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(dA_qc, hA_qc.data(), per_A * batch, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dB_re, hB_re.data(), per_B * batch * sizeof(__half), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dB_im, hB_im.data(), per_B * batch * sizeof(__half), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemset(dC_re, 0, per_C * batch * sizeof(float)));
    CHECK_CUDA(cudaMemset(dC_im, 0, per_C * batch * sizeof(float)));

    cutlass_gemm_api::CutlassComplexGemm api;
    printf("  prepare_b(FP8)...\n");
    api.prepare_b(dB_re, dB_im, N, K, batch, ComputePrecision::FP8);
    CHECK_CUDA(cudaDeviceSynchronize());

    printf("  gemm_prepared(INT4 A)...\n");
    int ret = api.gemm_prepared(reinterpret_cast<const __half*>(dA_qc), nullptr,
                                 dC_re, dC_im, M, N, K, batch,
                                 OutputPrecision::FP32,
                                 1.0f, 0.0f, nullptr, false, InputPrecision::INT4);
    CHECK_CUDA(cudaDeviceSynchronize());

    if (ret != 0) { printf("  FAIL: gemm_prepared (INT4) returned %d\n", ret); return 1; }

    std::vector<float> hC_re(per_C * batch), hC_im(per_C * batch);
    CHECK_CUDA(cudaMemcpy(hC_re.data(), dC_re, per_C * batch * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(hC_im.data(), dC_im, per_C * batch * sizeof(float), cudaMemcpyDeviceToHost));

    double tol = std::max(1.0, K * 0.02);
    bool pass = compare_fp32(hC_re.data(), hC_im.data(), refC, per_C * batch, tol);

    cudaFree(dA_qc); cudaFree(dB_re); cudaFree(dB_im);
    cudaFree(dC_re); cudaFree(dC_im);

    printf("  %s\n\n", pass ? "PASS" : "FAIL");
    return pass ? 0 : 1;
}

// Test: prepare_b + gemm_prepared_power_int4 (fused power detection)
static int test_gemm_power_int4() {
    printf("=== Test: prepare_b + gemm_prepared_power_int4 ===\n");
    const int M = 64, N = 64, K = 128, batch = 2;
    int64_t per_A = (int64_t)M * K, per_B = (int64_t)N * K, per_C = (int64_t)M * N;

    std::vector<uint8_t> hA_qc;
    std::vector<std::complex<double>> refA;
    generate_int4_data(hA_qc, refA, per_A * batch, 42);

    std::vector<__half> hB_re, hB_im;
    std::vector<std::complex<double>> refB;
    generate_data(hB_re, hB_im, refB, per_B * batch, 123);

    // CPU reference: power = |Re(A*B^T)|^2 + |Im(A*B^T)|^2
    std::vector<std::complex<double>> refC(per_C * batch, {0, 0});
    for (int b = 0; b < batch; ++b) {
        std::vector<std::complex<double>> A(refA.begin() + b * per_A, refA.begin() + (b + 1) * per_A);
        std::vector<std::complex<double>> B(refB.begin() + b * per_B, refB.begin() + (b + 1) * per_B);
        std::vector<std::complex<double>> C(per_C, {0, 0});
        cpu_gemm(A, B, C, M, N, K, 1.0f, 0.0f);
        for (int64_t i = 0; i < per_C; ++i) refC[b * per_C + i] = C[i];
    }
    std::vector<double> ref_power(per_C * batch);
    for (int64_t i = 0; i < per_C * batch; ++i) {
        ref_power[i] = refC[i].real() * refC[i].real() + refC[i].imag() * refC[i].imag();
    }

    uint8_t* dA_qc;
    __half *dB_re, *dB_im;
    float* dC_power;
    CHECK_CUDA(cudaMalloc(&dA_qc, per_A * batch));
    CHECK_CUDA(cudaMalloc(&dB_re, per_B * batch * sizeof(__half)));
    CHECK_CUDA(cudaMalloc(&dB_im, per_B * batch * sizeof(__half)));
    CHECK_CUDA(cudaMalloc(&dC_power, per_C * batch * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(dA_qc, hA_qc.data(), per_A * batch, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dB_re, hB_re.data(), per_B * batch * sizeof(__half), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dB_im, hB_im.data(), per_B * batch * sizeof(__half), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemset(dC_power, 0, per_C * batch * sizeof(float)));

    cutlass_gemm_api::CutlassComplexGemm api;
    printf("  prepare_b(FP8)...\n");
    api.prepare_b(dB_re, dB_im, N, K, batch, ComputePrecision::FP8);
    CHECK_CUDA(cudaDeviceSynchronize());

    printf("  gemm_prepared_power_int4()...\n");
    int ret = api.gemm_prepared_power_int4(dA_qc, dC_power, M, N, K, batch);
    CHECK_CUDA(cudaDeviceSynchronize());

    if (ret != 0) { printf("  FAIL: gemm_prepared_power_int4 returned %d\n", ret); return 1; }

    std::vector<float> hC_power(per_C * batch);
    CHECK_CUDA(cudaMemcpy(hC_power.data(), dC_power, per_C * batch * sizeof(float), cudaMemcpyDeviceToHost));

    // Compare power output
    double max_err = 0, max_ref = 0;
    #pragma omp parallel for reduction(max:max_err, max_ref)
    for (int64_t i = 0; i < per_C * batch; ++i) {
        max_err = std::max(max_err, std::abs((double)hC_power[i] - ref_power[i]));
        max_ref = std::max(max_ref, std::abs(ref_power[i]));
    }
    double rel = (max_ref > 0) ? max_err / max_ref : max_err;
    // Power has squared error, so use wider tolerance
    double tol = std::max(1.0, K * K * 0.001);
    bool pass = max_err < tol;
    printf("  max_ref=%.1f  max_err=%.2f  rel=%.6f  tol=%.1f\n", max_ref, max_err, rel, tol);

    cudaFree(dA_qc); cudaFree(dB_re); cudaFree(dB_im); cudaFree(dC_power);

    printf("  %s\n\n", pass ? "PASS" : "FAIL");
    return pass ? 0 : 1;
}

// Test: 4M Power vs Direct Power cross-check (PowerMode::Force4M vs ForceDirect)
static int test_power_4m_vs_direct() {
    printf("=== Test: power 4M vs Direct cross-check ===\n");
    const int M = 64, N = 256, K = 128, batch = 4;
    int64_t per_A = (int64_t)M * K, per_B = (int64_t)N * K, per_C = (int64_t)M * N;

    std::vector<__half> hA_re, hA_im, hB_re, hB_im;
    std::vector<std::complex<double>> refA, refB;
    generate_data(hA_re, hA_im, refA, per_A * batch, 42);
    generate_data(hB_re, hB_im, refB, per_B * batch, 123);

    __half *dA_re, *dA_im, *dB_re, *dB_im;
    float *dC_direct, *dC_4m;
    CHECK_CUDA(cudaMalloc(&dA_re, per_A * batch * sizeof(__half)));
    CHECK_CUDA(cudaMalloc(&dA_im, per_A * batch * sizeof(__half)));
    CHECK_CUDA(cudaMalloc(&dB_re, per_B * batch * sizeof(__half)));
    CHECK_CUDA(cudaMalloc(&dB_im, per_B * batch * sizeof(__half)));
    CHECK_CUDA(cudaMalloc(&dC_direct, per_C * batch * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&dC_4m, per_C * batch * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(dA_re, hA_re.data(), per_A * batch * sizeof(__half), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dA_im, hA_im.data(), per_A * batch * sizeof(__half), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dB_re, hB_re.data(), per_B * batch * sizeof(__half), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dB_im, hB_im.data(), per_B * batch * sizeof(__half), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemset(dC_direct, 0, per_C * batch * sizeof(float)));
    CHECK_CUDA(cudaMemset(dC_4m, 0, per_C * batch * sizeof(float)));

    // Direct power path
    cutlass_gemm_api::CutlassComplexGemm api_direct;
    api_direct.set_power_mode(cutlass_gemm_api::PowerMode::ForceDirect);
    api_direct.set_auto_tune(false);
    api_direct.prepare_b(dB_re, dB_im, N, K, batch, ComputePrecision::FP8);
    CHECK_CUDA(cudaDeviceSynchronize());

    printf("  Running ForceDirect power...\n");
    int ret1 = api_direct.gemm_prepared_power(dA_re, dA_im, dC_direct, M, N, K, batch);
    CHECK_CUDA(cudaDeviceSynchronize());
    if (ret1 != 0) { printf("  FAIL: ForceDirect returned %d\n", ret1); return 1; }

    // 4M power path
    cutlass_gemm_api::CutlassComplexGemm api_4m;
    api_4m.set_power_mode(cutlass_gemm_api::PowerMode::Force4M);
    api_4m.set_auto_tune(false);
    api_4m.prepare_b(dB_re, dB_im, N, K, batch, ComputePrecision::FP8);
    CHECK_CUDA(cudaDeviceSynchronize());

    printf("  Running Force4M power...\n");
    int ret2 = api_4m.gemm_prepared_power(dA_re, dA_im, dC_4m, M, N, K, batch);
    CHECK_CUDA(cudaDeviceSynchronize());
    if (ret2 != 0) { printf("  FAIL: Force4M returned %d\n", ret2); return 1; }

    // Compare: both paths should produce very similar results
    std::vector<float> hC_direct(per_C * batch), hC_4m(per_C * batch);
    CHECK_CUDA(cudaMemcpy(hC_direct.data(), dC_direct, per_C * batch * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(hC_4m.data(), dC_4m, per_C * batch * sizeof(float), cudaMemcpyDeviceToHost));

    double max_err = 0, max_ref = 0;
    #pragma omp parallel for reduction(max:max_err, max_ref)
    for (int64_t i = 0; i < per_C * batch; ++i) {
        max_err = std::max(max_err, std::abs((double)hC_4m[i] - (double)hC_direct[i]));
        max_ref = std::max(max_ref, std::abs((double)hC_direct[i]));
    }
    double rel = (max_ref > 0) ? max_err / max_ref : max_err;
    // Both paths use FP8 internally, so expect small numerical differences
    // from different accumulation order (4M uses 4 sub-GEMMs, direct uses fused)
    double tol = std::max(2.0, K * 0.05);
    bool pass = max_err < tol;
    printf("  max_ref=%.1f  max_err=%.2f  rel=%.6f  tol=%.1f\n", max_ref, max_err, rel, tol);

    cudaFree(dA_re); cudaFree(dA_im); cudaFree(dB_re); cudaFree(dB_im);
    cudaFree(dC_direct); cudaFree(dC_4m);

    printf("  %s\n\n", pass ? "PASS" : "FAIL");
    return pass ? 0 : 1;
}

// Test: prepare_b + gemm_prepared_direct_int4 (direct kernel, Re/Im output)
static int test_gemm_direct_int4() {
    printf("=== Test: prepare_b + gemm_prepared_direct_int4 ===\n");
    const int M = 64, N = 64, K = 128, batch = 2;
    int64_t per_A = (int64_t)M * K, per_B = (int64_t)N * K, per_C = (int64_t)M * N;

    std::vector<uint8_t> hA_qc;
    std::vector<std::complex<double>> refA;
    generate_int4_data(hA_qc, refA, per_A * batch, 42);

    std::vector<__half> hB_re, hB_im;
    std::vector<std::complex<double>> refB;
    generate_data(hB_re, hB_im, refB, per_B * batch, 123);

    std::vector<std::complex<double>> refC(per_C * batch, {0, 0});
    for (int b = 0; b < batch; ++b) {
        std::vector<std::complex<double>> A(refA.begin() + b * per_A, refA.begin() + (b + 1) * per_A);
        std::vector<std::complex<double>> B(refB.begin() + b * per_B, refB.begin() + (b + 1) * per_B);
        std::vector<std::complex<double>> C(per_C, {0, 0});
        cpu_gemm(A, B, C, M, N, K, 1.0f, 0.0f);
        for (int64_t i = 0; i < per_C; ++i) refC[b * per_C + i] = C[i];
    }

    uint8_t* dA_qc;
    __half *dB_re, *dB_im;
    float *dC_re, *dC_im;
    CHECK_CUDA(cudaMalloc(&dA_qc, per_A * batch));
    CHECK_CUDA(cudaMalloc(&dB_re, per_B * batch * sizeof(__half)));
    CHECK_CUDA(cudaMalloc(&dB_im, per_B * batch * sizeof(__half)));
    CHECK_CUDA(cudaMalloc(&dC_re, per_C * batch * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&dC_im, per_C * batch * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(dA_qc, hA_qc.data(), per_A * batch, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dB_re, hB_re.data(), per_B * batch * sizeof(__half), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dB_im, hB_im.data(), per_B * batch * sizeof(__half), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemset(dC_re, 0, per_C * batch * sizeof(float)));
    CHECK_CUDA(cudaMemset(dC_im, 0, per_C * batch * sizeof(float)));

    cutlass_gemm_api::CutlassComplexGemm api;
    printf("  prepare_b(FP8)...\n");
    api.prepare_b(dB_re, dB_im, N, K, batch, ComputePrecision::FP8);
    CHECK_CUDA(cudaDeviceSynchronize());

    printf("  gemm_prepared_direct_int4()...\n");
    int ret = api.gemm_prepared_direct_int4(dA_qc, dC_re, dC_im, M, N, K, batch);
    CHECK_CUDA(cudaDeviceSynchronize());

    if (ret != 0) { printf("  FAIL: gemm_prepared_direct_int4 returned %d\n", ret); return 1; }

    std::vector<float> hC_re(per_C * batch), hC_im(per_C * batch);
    CHECK_CUDA(cudaMemcpy(hC_re.data(), dC_re, per_C * batch * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(hC_im.data(), dC_im, per_C * batch * sizeof(float), cudaMemcpyDeviceToHost));

    double tol = std::max(1.0, K * 0.02);
    bool pass = compare_fp32(hC_re.data(), hC_im.data(), refC, per_C * batch, tol);

    cudaFree(dA_qc); cudaFree(dB_re); cudaFree(dB_im);
    cudaFree(dC_re); cudaFree(dC_im);

    printf("  %s\n\n", pass ? "PASS" : "FAIL");
    return pass ? 0 : 1;
}

static void print_help(const char* prog) {
    printf("Usage: %s [test_name]\n", prog);
    printf("\n");
    printf("Correctness tests for the CUTLASS complex GEMM PIMPL API.\n");
    printf("Only includes cutlass_gemm_api.h — proves PIMPL header isolation.\n");
    printf("\n");
    printf("Positional arguments:\n");
    printf("  test_name    Test to run (default: all)\n");
    printf("\n");
    printf("Available tests:\n");
    printf("  gemm_fp16          GEMM planar batched, FP16 output\n");
    printf("  gemm_fp32          GEMM planar batched, FP32 output\n");
    printf("  gemm_fp6           GEMM planar batched FP6, FP32 output\n");
    printf("  gemm_fp4           GEMM planar batched FP4, FP32 output\n");
    printf("  prepare_execute    Prepare B + execute GEMM, FP32 output\n");
    printf("  prepare_fp6        Prepare B (FP6) + execute GEMM, FP32 output\n");
    printf("  prepare_fp4        Prepare B (FP4) + execute GEMM, FP32 output\n");
    printf("  herk_fp32          HERK batched, FP32 output\n");
    printf("  herk_beta          HERK beta accumulation (alpha=0.5, beta=1.0)\n");
    printf("  herk_single_batch  HERK with batch=1 edge case\n");
    printf("  gemm_non_aligned   GEMM with non-aligned dims (M=33, N=77)\n");
    printf("  gemm_fp32_output   GEMM FP32 output (different seeds)\n");
    printf("  herk_large_k       HERK with large K=512 (>K_CHUNK)\n");
    printf("  gemm_alpha_beta    GEMM with alpha/beta accumulation\n");
    printf("  herk_direct        HERK direct PTX kernel correctness\n");
    printf("  herk_baseline      HERK baseline CUTLASS path correctness\n");
    printf("  herk_direct_vs_baseline  Direct vs baseline self-consistency\n");
    printf("  herk_fp6           HERK FP6 compute, FP32 output\n");
    printf("  herk_fp4           HERK FP4 compute, exact [-4,4] input\n");
    printf("  herk_fp4_lossy     HERK FP4 structural error gate [-7,7]\n");
    printf("  herk_k_not_aligned HERK with K=96 (not aligned to K_CHUNK)\n");
    printf("  gemm_k_not_aligned GEMM with K=96 (not aligned to MMA_K)\n");
    printf("  gemm_continuous    GEMM with continuous FP16 data\n");
    printf("  herk_continuous    HERK with continuous FP16 data\n");
    printf("  gemm_int4_fp32     GEMM with INT4 A input, FP8, FP32 output\n");
    printf("  gemm_prepared_int4 Prepare B + GEMM with INT4 A input\n");
    printf("  gemm_power_int4    Prepare B + power detection with INT4 A\n");
    printf("  power_4m_vs_direct 4M Power vs Direct Power cross-check\n");
    printf("  gemm_direct_int4   Prepare B + direct GEMM with INT4 A\n");
    printf("  all                Run all tests (default)\n");
    printf("\n");
    printf("Examples:\n");
    printf("  %s                    # run all tests\n", prog);
    printf("  %s gemm_fp32          # run FP32 GEMM test only\n", prog);
    printf("  %s herk_fp32          # run HERK FP32 test only\n", prog);
}

int main(int argc, char* argv[]) {
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            cutlass_gemm_api::CutlassComplexGemm::print_build_info();
            printf("\n");
            print_help(argv[0]);
            return 0;
        }
    }

    cutlass_gemm_api::CutlassComplexGemm::print_build_info();
    const char* test = (argc > 1) ? argv[1] : "all";
    int failures = 0;

    // Return 0 for skipped tests (-1) when run individually
    if (strcmp(test, "gemm_fp16") == 0)         return test_gemm_fp16();
    if (strcmp(test, "gemm_fp32") == 0)          return test_gemm_fp32();
    if (strcmp(test, "gemm_fp6") == 0)           return std::max(0, test_gemm_fp6());
    if (strcmp(test, "gemm_fp4") == 0)           return std::max(0, test_gemm_fp4());
    if (strcmp(test, "prepare_execute") == 0)     return test_prepare_execute();
    if (strcmp(test, "prepare_fp6") == 0)         return std::max(0, test_prepare_fp6());
    if (strcmp(test, "prepare_fp4") == 0)         return std::max(0, test_prepare_fp4());
    if (strcmp(test, "herk_fp32") == 0)           return test_herk_fp32();
    if (strcmp(test, "herk_beta") == 0)           return test_herk_beta();
    if (strcmp(test, "herk_single_batch") == 0)   return test_herk_single_batch();
    if (strcmp(test, "gemm_non_aligned") == 0)    return test_gemm_non_aligned();
    if (strcmp(test, "gemm_fp32_output") == 0)    return test_gemm_fp32_output();
    if (strcmp(test, "herk_large_k") == 0)        return test_herk_large_k();
    if (strcmp(test, "gemm_alpha_beta") == 0)     return test_gemm_alpha_beta();
    if (strcmp(test, "herk_direct") == 0)         return test_herk_direct();
    if (strcmp(test, "herk_baseline") == 0)       return test_herk_baseline();
    if (strcmp(test, "herk_direct_vs_baseline") == 0) return test_herk_direct_vs_baseline();
    if (strcmp(test, "herk_fp6") == 0)            return std::max(0, test_herk_fp6());
    if (strcmp(test, "herk_fp4") == 0)            return std::max(0, test_herk_fp4());
    if (strcmp(test, "herk_fp4_lossy") == 0)      return std::max(0, test_herk_fp4_lossy());
    if (strcmp(test, "herk_k_not_aligned") == 0)  return test_herk_k_not_aligned();
    if (strcmp(test, "gemm_k_not_aligned") == 0)  return test_gemm_k_not_aligned();
    if (strcmp(test, "gemm_continuous") == 0)      return test_gemm_continuous();
    if (strcmp(test, "herk_continuous") == 0)      return test_herk_continuous();
    if (strcmp(test, "gemm_int4_fp32") == 0)       return test_gemm_int4_fp32();
    if (strcmp(test, "gemm_prepared_int4") == 0)   return test_gemm_prepared_int4();
    if (strcmp(test, "gemm_power_int4") == 0)      return test_gemm_power_int4();
    if (strcmp(test, "power_4m_vs_direct") == 0)   return test_power_4m_vs_direct();
    if (strcmp(test, "gemm_direct_int4") == 0)     return test_gemm_direct_int4();

    if (strcmp(test, "all") == 0) {
        int total = 0, skipped = 0;
        auto run = [&](int (*fn)()) {
            int ret = fn();
            if (ret == -1) { skipped++; }
            else { total++; failures += ret; }
        };
        run(test_gemm_fp16);
        run(test_gemm_fp32);
        run(test_gemm_fp6);
        run(test_gemm_fp4);
        run(test_prepare_execute);
        run(test_prepare_fp6);
        run(test_prepare_fp4);
        run(test_herk_fp32);
        run(test_herk_beta);
        run(test_herk_single_batch);
        run(test_gemm_non_aligned);
        run(test_gemm_fp32_output);
        run(test_herk_large_k);
        run(test_gemm_alpha_beta);
        run(test_herk_direct);
        run(test_herk_baseline);
        run(test_herk_direct_vs_baseline);
        run(test_herk_fp6);
        run(test_herk_fp4);
        run(test_herk_fp4_lossy);
        run(test_herk_k_not_aligned);
        run(test_gemm_k_not_aligned);
        run(test_gemm_continuous);
        run(test_herk_continuous);
        run(test_gemm_int4_fp32);
        run(test_gemm_prepared_int4);
        run(test_gemm_power_int4);
        run(test_power_4m_vs_direct);
        run(test_gemm_direct_int4);
        printf("=== %d/%d tests passed", total - failures, total);
        if (skipped > 0) printf(", %d skipped", skipped);
        printf(" ===\n");
        return failures;
    }

    fprintf(stderr, "Unknown test: %s\n", test);
    fprintf(stderr, "Available: gemm_fp16, gemm_fp32, gemm_fp6, gemm_fp4, "
                    "prepare_execute, prepare_fp6, prepare_fp4, herk_fp32, "
                    "herk_beta, herk_single_batch, gemm_non_aligned, gemm_fp32_output, "
                    "herk_large_k, gemm_alpha_beta, herk_direct, herk_baseline, "
                    "herk_direct_vs_baseline, herk_fp6, herk_fp4, herk_fp4_lossy, "
                    "herk_k_not_aligned, gemm_k_not_aligned, gemm_continuous, "
                    "herk_continuous, gemm_int4_fp32, gemm_prepared_int4, "
                    "gemm_power_int4, power_4m_vs_direct, gemm_direct_int4, all\n");
    return 1;
}
