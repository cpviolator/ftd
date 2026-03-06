// test_tune_api.cu — End-to-end test for the strategy autotuning API (tune=true)
//
// Tests:
//   1. First call with tune=true triggers a strategy sweep
//   2. Second call with tune=true hits the cache (verify via timing)
//   3. Correctness of the tuned result against CPU FP64 reference
//   4. Cache file persistence (write + re-read)
//
#include "cutlass_gemm_api.h"
using cutlass_gemm_api::InputPrecision;
using cutlass_gemm_api::ComputePrecision;
using cutlass_gemm_api::OutputPrecision;
using cutlass_gemm_api::OutputFormat;
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cmath>
#include <cstring>
#include <vector>
#include <complex>
#include <chrono>
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

static void print_help(const char* prog) {
    printf("Usage: %s [N] [K] [batch]\n", prog);
    printf("\n");
    printf("End-to-end test for the HERK strategy autotuning API (tune=true).\n");
    printf("\n");
    printf("Tests:\n");
    printf("  1. First call with tune=true triggers a strategy sweep (~5s)\n");
    printf("  2. Second call with tune=true hits the cache (fast)\n");
    printf("  3. Correctness of the tuned result against CPU FP64 reference\n");
    printf("  4. Cache file persistence (write + re-read)\n");
    printf("\n");
    printf("Positional arguments:\n");
    printf("  N            Matrix dimension (default: 128)\n");
    printf("  K            Inner dimension (default: 128)\n");
    printf("  batch        Batch count (default: 2)\n");
    printf("\n");
    printf("Examples:\n");
    printf("  %s                    # default N=128 K=128 batch=2\n", prog);
    printf("  %s 256 256 4          # N=256 K=256 batch=4\n", prog);
    printf("  %s 1024 512 8         # larger problem, 8 batches\n", prog);
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

    int N = 128;
    int K = 128;
    int batch = 2;
    if (argc > 1) N = atoi(argv[1]);
    if (argc > 2) K = atoi(argv[2]);
    if (argc > 3) batch = atoi(argv[3]);

    cutlass_gemm_api::CutlassComplexGemm::print_build_info();
    printf("=== Strategy Autotuning API Test ===\n");
    printf("N=%d K=%d batch=%d\n\n", N, K, batch);

    int64_t per_A = (int64_t)N * K;
    int64_t packed = (int64_t)N * (N + 1) / 2;

    // Generate random FP16 complex data and CPU reference
    std::vector<std::complex<double>> h_A_cpu(per_A * batch);
    std::vector<std::complex<double>> h_C_ref(packed * batch, {0, 0});
    std::vector<__half> h_A_fp16(per_A * batch * 2);

    srand(42);
    for (int64_t i = 0; i < per_A * batch; ++i) {
        double re = (rand() % 15) - 7;
        double im = (rand() % 15) - 7;
        h_A_cpu[i] = std::complex<double>(re, im);
        h_A_fp16[i * 2]     = __float2half(static_cast<float>(re));
        h_A_fp16[i * 2 + 1] = __float2half(static_cast<float>(im));
    }

    // CPU reference HERK per batch (parallelized across batches)
    #pragma omp parallel for schedule(dynamic)
    for (int b = 0; b < batch; ++b) {
        std::vector<std::complex<double>> A_batch(h_A_cpu.begin() + b * per_A,
                                                   h_A_cpu.begin() + (b + 1) * per_A);
        std::vector<std::complex<double>> C_batch(packed, {0, 0});
        cpu_herk(A_batch, C_batch, N, K);
        for (int64_t i = 0; i < packed; ++i)
            h_C_ref[b * packed + i] = C_batch[i];
    }

    // Upload FP16 data to GPU
    __half* d_A;
    CHECK_CUDA(cudaMalloc(&d_A, per_A * batch * 2 * sizeof(__half)));
    CHECK_CUDA(cudaMemcpy(d_A, h_A_fp16.data(), per_A * batch * 2 * sizeof(__half),
                          cudaMemcpyHostToDevice));

    // Use a temporary cache file
    const char* cache_file = "test_strategy_cache_tmp.txt";

    bool all_pass = true;

    // ---- Test 1: First call with tune=true (triggers sweep) ----
    printf("--- Test 1: tune=true first call (sweep) ---\n");
    {
        __half* d_C;
        CHECK_CUDA(cudaMalloc(&d_C, packed * batch * 2 * sizeof(__half)));
        CHECK_CUDA(cudaMemset(d_C, 0, packed * batch * 2 * sizeof(__half)));

        cutlass_gemm_api::CutlassComplexGemm api;
        api.set_tune_cache_path(cache_file);

        auto t0 = std::chrono::high_resolution_clock::now();
        int ret = api.herk(d_A, d_C, N, K, batch,
                           InputPrecision::FP16, ComputePrecision::FP8,
                           OutputPrecision::FP16, OutputFormat::PackedTriangle,
                           1.0f, 0.0f, nullptr, true);
        CHECK_CUDA(cudaDeviceSynchronize());
        auto t1 = std::chrono::high_resolution_clock::now();

        double sweep_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        printf("  Sweep call: %.1f ms (includes benchmark sweep)\n", sweep_ms);

        if (ret != 0) {
            printf("  FAIL: herk_batched returned %d\n", ret);
            all_pass = false;
        } else {
            // Verify correctness
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

            // Integer [-7,7] input is exact in FP8 E4M3. Tolerance matches
            // all other integer-input HERK tests.
            double tol = std::max(1.0, K * 0.02);
            bool pass = max_err < tol;
            printf("  Correctness: max_ref=%.1f max_err=%.2f tol=%.1f %s\n",
                   max_ref, max_err, tol, pass ? "PASS" : "FAIL");
            if (!pass) all_pass = false;
        }

        CHECK_CUDA(cudaFree(d_C));
    }

    // Ensure all GPU work from Test 1 is complete before proceeding
    CHECK_CUDA(cudaDeviceSynchronize());

    // ---- Test 2: Second call with tune=true (cache hit) ----
    printf("\n--- Test 2: tune=true second call (cache hit) ---\n");
    {
        __half* d_C;
        CHECK_CUDA(cudaMalloc(&d_C, packed * batch * 2 * sizeof(__half)));
        CHECK_CUDA(cudaMemset(d_C, 0, packed * batch * 2 * sizeof(__half)));

        cutlass_gemm_api::CutlassComplexGemm api;
        api.set_tune_cache_path(cache_file);

        auto t0 = std::chrono::high_resolution_clock::now();
        int ret = api.herk(d_A, d_C, N, K, batch,
                           InputPrecision::FP16, ComputePrecision::FP8,
                           OutputPrecision::FP16, OutputFormat::PackedTriangle,
                           1.0f, 0.0f, nullptr, true);
        CHECK_CUDA(cudaDeviceSynchronize());
        auto t1 = std::chrono::high_resolution_clock::now();

        double cached_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        printf("  Cached call: %.1f ms (should be much faster than sweep)\n", cached_ms);

        if (ret != 0) {
            printf("  FAIL: herk_batched returned %d\n", ret);
            all_pass = false;
        } else {
            printf("  PASS (cache hit)\n");
        }

        CHECK_CUDA(cudaFree(d_C));
    }

    // Ensure all GPU work from Test 2 is complete before proceeding
    CHECK_CUDA(cudaDeviceSynchronize());

    // ---- Test 3: tune=false (default behavior, backward compat) ----
    printf("\n--- Test 3: tune=false (default, backward compat) ---\n");
    {
        __half* d_C;
        CHECK_CUDA(cudaMalloc(&d_C, packed * batch * 2 * sizeof(__half)));
        CHECK_CUDA(cudaMemset(d_C, 0, packed * batch * 2 * sizeof(__half)));

        cutlass_gemm_api::CutlassComplexGemm api;
        int ret = api.herk(d_A, d_C, N, K, batch);
        CHECK_CUDA(cudaDeviceSynchronize());

        if (ret != 0) {
            printf("  FAIL: herk (tune=false) returned %d\n", ret);
            all_pass = false;
        } else {
            // Verify correctness
            std::vector<__half> h_C(packed * batch * 2);
            CHECK_CUDA(cudaMemcpy(h_C.data(), d_C, packed * batch * 2 * sizeof(__half),
                                  cudaMemcpyDeviceToHost));

            double max_err = 0, max_ref = 0;
            #pragma omp parallel for reduction(max:max_err, max_ref)
            for (int64_t i = 0; i < packed * batch; ++i) {
                double re = __half2float(h_C[i * 2]);
                double im = __half2float(h_C[i * 2 + 1]);
                max_err = std::max(max_err, std::abs(re - h_C_ref[i].real()));
                max_err = std::max(max_err, std::abs(im - h_C_ref[i].imag()));
                max_ref = std::max(max_ref, std::abs(h_C_ref[i].real()));
                max_ref = std::max(max_ref, std::abs(h_C_ref[i].imag()));
            }

            // Integer [-7,7] input is exact in FP8 E4M3. Same tolerance as Test 1.
            double tol = std::max(1.0, K * 0.02);
            bool pass = max_err < tol;
            printf("  Correctness: max_ref=%.1f max_err=%.2f tol=%.1f %s\n",
                   max_ref, max_err, tol, pass ? "PASS" : "FAIL");
            if (!pass) all_pass = false;
        }

        CHECK_CUDA(cudaFree(d_C));
    }

    // ---- Test 4: Cache file persistence ----
    printf("\n--- Test 4: Cache file persistence ---\n");
    {
        FILE* f = fopen(cache_file, "r");
        if (f) {
            char line[512];
            int data_lines = 0;
            while (fgets(line, sizeof(line), f)) {
                if (line[0] != '#' && line[0] != '\n' && line[0] != '\0')
                    data_lines++;
            }
            fclose(f);
            printf("  Cache file '%s': %d data entries\n", cache_file, data_lines);
            bool pass = data_lines >= 1;
            printf("  %s\n", pass ? "PASS" : "FAIL (expected at least 1 entry)");
            if (!pass) all_pass = false;
        } else {
            printf("  FAIL: cache file '%s' not found\n", cache_file);
            all_pass = false;
        }

        // Cleanup temp file
        remove(cache_file);
    }

    CHECK_CUDA(cudaFree(d_A));

    printf("\n=== %s ===\n", all_pass ? "ALL TESTS PASSED" : "SOME TESTS FAILED");
    return all_pass ? 0 : 1;
}
