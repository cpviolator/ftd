/** @file test_imaging.cu
 *  @brief End-to-end tests for the imaging pipeline.
 *
 *  Tests all 4 combinations of VisPrecision x FftPrecision:
 *    FP16 vis + FP16 FFT, FP16 vis + FP32 FFT,
 *    FP32 vis + FP16 FFT, FP32 vis + FP32 FFT.
 *
 *  This is a standalone single-TU binary.
 */

#include "kernels/imaging_kernels.cuh"
#include "imaging_pipeline.cu"

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cufft.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <algorithm>
#include <cstring>
#include <string>

// ================================================================
// Test utilities
// ================================================================

#define CUDA_CHECK_TEST(call)                                                \
    do {                                                                      \
        cudaError_t err = (call);                                             \
        if (err != cudaSuccess) {                                             \
            fprintf(stderr, "  CUDA error at %s:%d: %s\n",                  \
                    __FILE__, __LINE__, cudaGetErrorString(err));             \
            return false;                                                      \
        }                                                                     \
    } while (0)

static int tests_passed = 0;
static int tests_failed = 0;

#define RUN_TEST(name)                                                       \
    do {                                                                      \
        printf("  %-55s", #name "...");                                      \
        fflush(stdout);                                                       \
        if (name()) { printf("[PASS]\n"); tests_passed++; }                  \
        else        { printf("[FAIL]\n"); tests_failed++; }                  \
    } while (0)

static int tri_idx(int row, int col) { return row * (row + 1) / 2 + col; }

// ================================================================
// Test 1: Gridding scatter (FP16 vis -> FP16 grid)
// ================================================================
static bool test_gridding_scatter_fp16() {
    const int N = 4, n_bl = N * (N + 1) / 2, Ng = 32;
    const double freq = 1.0e9;
    const float cs = 1.0f;
    const double lambda = 299792458.0 / freq;

    std::vector<float> bl_h(n_bl * 2, 0.0f);
    std::vector<double> freq_h(1, freq);
    const int bl = tri_idx(1, 0);
    bl_h[bl * 2]     = static_cast<float>(2.0 * lambda);
    bl_h[bl * 2 + 1] = static_cast<float>(3.0 * lambda);

    std::vector<__half2> vis_h(n_bl);
    for (auto& v : vis_h) v = __halves2half2(__float2half(0.0f), __float2half(0.0f));
    vis_h[bl] = __halves2half2(__float2half(1.0f), __float2half(0.5f));

    float* bl_d; double* freq_d; __half2* vis_d; __half2* grid_d;
    CUDA_CHECK_TEST(cudaMalloc(&bl_d, n_bl * 2 * sizeof(float)));
    CUDA_CHECK_TEST(cudaMalloc(&freq_d, sizeof(double)));
    CUDA_CHECK_TEST(cudaMalloc(&vis_d, n_bl * sizeof(__half2)));
    CUDA_CHECK_TEST(cudaMalloc(&grid_d, Ng * Ng * sizeof(__half2)));
    CUDA_CHECK_TEST(cudaMemcpy(bl_d, bl_h.data(), n_bl * 2 * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK_TEST(cudaMemcpy(freq_d, freq_h.data(), sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK_TEST(cudaMemcpy(vis_d, vis_h.data(), n_bl * sizeof(__half2), cudaMemcpyHostToDevice));
    CUDA_CHECK_TEST(cudaMemset(grid_d, 0, Ng * Ng * sizeof(__half2)));

    imaging_kernels::launch_scatter_fp16(vis_d, bl_d, freq_d, grid_d, n_bl, Ng, 1, 0, cs);
    CUDA_CHECK_TEST(cudaDeviceSynchronize());

    std::vector<__half2> g(Ng * Ng);
    CUDA_CHECK_TEST(cudaMemcpy(g.data(), grid_d, Ng * Ng * sizeof(__half2), cudaMemcpyDeviceToHost));

    int iu = static_cast<int>(roundf(2.0f + Ng * 0.5f));
    int iv = static_cast<int>(roundf(3.0f + Ng * 0.5f));
    float re = __half2float(__low2half(g[iv * Ng + iu]));
    float im = __half2float(__high2half(g[iv * Ng + iu]));
    bool ok = (fabsf(re - 1.0f) < 0.01f) && (fabsf(im - 0.5f) < 0.01f);

    int iu_c = static_cast<int>(roundf(-2.0f + Ng * 0.5f));
    int iv_c = static_cast<int>(roundf(-3.0f + Ng * 0.5f));
    float re_c = __half2float(__low2half(g[iv_c * Ng + iu_c]));
    float im_c = __half2float(__high2half(g[iv_c * Ng + iu_c]));
    ok = ok && (fabsf(re_c - 1.0f) < 0.01f) && (fabsf(im_c + 0.5f) < 0.01f);

    if (!ok) fprintf(stderr, "  grid(%d,%d)=(%.3f,%.3f) conj(%d,%d)=(%.3f,%.3f)\n",
                     iu, iv, re, im, iu_c, iv_c, re_c, im_c);

    cudaFree(bl_d); cudaFree(freq_d); cudaFree(vis_d); cudaFree(grid_d);
    return ok;
}

// ================================================================
// Test 2: Gridding scatter (FP32 vis -> FP32 grid)
// ================================================================
static bool test_gridding_scatter_fp32_grid() {
    const int N = 4, n_bl = N * (N + 1) / 2, Ng = 32;
    const double freq = 1.0e9;
    const float cs = 1.0f;
    const double lambda = 299792458.0 / freq;

    std::vector<float> bl_h(n_bl * 2, 0.0f);
    std::vector<double> freq_h(1, freq);
    const int bl = tri_idx(1, 0);
    bl_h[bl * 2]     = static_cast<float>(2.0 * lambda);
    bl_h[bl * 2 + 1] = static_cast<float>(3.0 * lambda);

    // FP32 vis: [n_bl * 2] float (Re, Im interleaved)
    std::vector<float> vis_h(n_bl * 2, 0.0f);
    vis_h[bl * 2]     = 1.0f;
    vis_h[bl * 2 + 1] = 0.5f;

    float* bl_d; double* freq_d; float* vis_d; float* grid_d;
    CUDA_CHECK_TEST(cudaMalloc(&bl_d, n_bl * 2 * sizeof(float)));
    CUDA_CHECK_TEST(cudaMalloc(&freq_d, sizeof(double)));
    CUDA_CHECK_TEST(cudaMalloc(&vis_d, n_bl * 2 * sizeof(float)));
    CUDA_CHECK_TEST(cudaMalloc(&grid_d, Ng * Ng * 2 * sizeof(float)));
    CUDA_CHECK_TEST(cudaMemcpy(bl_d, bl_h.data(), n_bl * 2 * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK_TEST(cudaMemcpy(freq_d, freq_h.data(), sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK_TEST(cudaMemcpy(vis_d, vis_h.data(), n_bl * 2 * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK_TEST(cudaMemset(grid_d, 0, Ng * Ng * 2 * sizeof(float)));

    imaging_kernels::launch_scatter_fp32_fp32grid(vis_d, bl_d, freq_d, grid_d, n_bl, Ng, 1, 0, cs);
    CUDA_CHECK_TEST(cudaDeviceSynchronize());

    std::vector<float> g(Ng * Ng * 2);
    CUDA_CHECK_TEST(cudaMemcpy(g.data(), grid_d, Ng * Ng * 2 * sizeof(float), cudaMemcpyDeviceToHost));

    int iu = static_cast<int>(roundf(2.0f + Ng * 0.5f));
    int iv = static_cast<int>(roundf(3.0f + Ng * 0.5f));
    int idx = (iv * Ng + iu) * 2;
    bool ok = (fabsf(g[idx] - 1.0f) < 1e-5f) && (fabsf(g[idx + 1] - 0.5f) < 1e-5f);

    // Conjugate
    int iu_c = static_cast<int>(roundf(-2.0f + Ng * 0.5f));
    int iv_c = static_cast<int>(roundf(-3.0f + Ng * 0.5f));
    int idx_c = (iv_c * Ng + iu_c) * 2;
    ok = ok && (fabsf(g[idx_c] - 1.0f) < 1e-5f) && (fabsf(g[idx_c + 1] + 0.5f) < 1e-5f);

    if (!ok) fprintf(stderr, "  grid(%d,%d)=(%.5f,%.5f) conj(%d,%d)=(%.5f,%.5f)\n",
                     iu, iv, g[idx], g[idx+1], iu_c, iv_c, g[idx_c], g[idx_c+1]);

    cudaFree(bl_d); cudaFree(freq_d); cudaFree(vis_d); cudaFree(grid_d);
    return ok;
}

// ================================================================
// Test 3: Conjugate symmetry
// ================================================================
static bool test_conjugate_symmetry() {
    const int N = 3, n_bl = N * (N + 1) / 2, Ng = 32;
    const double freq = 1.0e9;
    const float cs = 1.0f;
    const double lambda = 299792458.0 / freq;

    std::vector<float> bl_h(n_bl * 2, 0.0f);
    std::vector<double> freq_h(1, freq);
    std::vector<__half2> vis_h(n_bl);
    for (auto& v : vis_h) v = __halves2half2(__float2half(0.0f), __float2half(0.0f));

    // Auto at u=5
    bl_h[tri_idx(1, 1) * 2] = static_cast<float>(5.0 * lambda);
    vis_h[tri_idx(1, 1)] = __halves2half2(__float2half(2.0f), __float2half(1.0f));

    // Cross at u=3
    bl_h[tri_idx(2, 0) * 2] = static_cast<float>(3.0 * lambda);
    vis_h[tri_idx(2, 0)] = __halves2half2(__float2half(4.0f), __float2half(2.0f));

    float* bl_d; double* freq_d; __half2* vis_d; __half2* grid_d;
    CUDA_CHECK_TEST(cudaMalloc(&bl_d, n_bl * 2 * sizeof(float)));
    CUDA_CHECK_TEST(cudaMalloc(&freq_d, sizeof(double)));
    CUDA_CHECK_TEST(cudaMalloc(&vis_d, n_bl * sizeof(__half2)));
    CUDA_CHECK_TEST(cudaMalloc(&grid_d, Ng * Ng * sizeof(__half2)));
    CUDA_CHECK_TEST(cudaMemcpy(bl_d, bl_h.data(), n_bl * 2 * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK_TEST(cudaMemcpy(freq_d, freq_h.data(), sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK_TEST(cudaMemcpy(vis_d, vis_h.data(), n_bl * sizeof(__half2), cudaMemcpyHostToDevice));
    CUDA_CHECK_TEST(cudaMemset(grid_d, 0, Ng * Ng * sizeof(__half2)));

    imaging_kernels::launch_scatter_fp16(vis_d, bl_d, freq_d, grid_d, n_bl, Ng, 1, 0, cs);
    CUDA_CHECK_TEST(cudaDeviceSynchronize());

    std::vector<__half2> g(Ng * Ng);
    CUDA_CHECK_TEST(cudaMemcpy(g.data(), grid_d, Ng * Ng * sizeof(__half2), cudaMemcpyDeviceToHost));

    bool ok = true;
    int hN = Ng / 2;
    // Auto: at u=5 only (not -5)
    float re_a = __half2float(__low2half(g[hN * Ng + hN + 5]));
    float re_an = __half2float(__low2half(g[hN * Ng + hN - 5]));
    if (fabsf(re_a - 2.0f) > 0.01f || fabsf(re_an) > 0.01f) ok = false;

    // Cross: both u=3 and u=-3
    float re_c = __half2float(__low2half(g[hN * Ng + hN + 3]));
    float im_c = __half2float(__high2half(g[hN * Ng + hN + 3]));
    float re_cn = __half2float(__low2half(g[hN * Ng + hN - 3]));
    float im_cn = __half2float(__high2half(g[hN * Ng + hN - 3]));
    if (fabsf(re_c - 4.0f) > 0.01f || fabsf(im_c - 2.0f) > 0.01f) ok = false;
    if (fabsf(re_cn - 4.0f) > 0.01f || fabsf(im_cn + 2.0f) > 0.01f) ok = false;

    cudaFree(bl_d); cudaFree(freq_d); cudaFree(vis_d); cudaFree(grid_d);
    return ok;
}

// ================================================================
// Test 4: FP16 vs FP32 vis precision (both into FP16 grid)
// ================================================================
static bool test_fp16_vs_fp32_precision() {
    const int N = 8, n_bl = N * (N + 1) / 2, Ng = 64, Nf = 2;
    const double freq0 = 1.0e9;
    const float cs = 0.5f;
    const double lambda = 299792458.0 / freq0;

    srand(42);
    std::vector<float> bl_h(n_bl * 2);
    for (auto& v : bl_h) v = (rand() / (float)RAND_MAX - 0.5f) * 10.0f * lambda;
    std::vector<double> freq_h(Nf);
    for (int i = 0; i < Nf; i++) freq_h[i] = freq0 + i * 1.0e6;

    std::vector<__half2> vis16(static_cast<size_t>(Nf) * n_bl);
    std::vector<float> vis32(static_cast<size_t>(Nf) * n_bl * 2);
    for (int i = 0; i < Nf * n_bl; i++) {
        float re = (rand() / (float)RAND_MAX - 0.5f) * 2.0f;
        float im = (rand() / (float)RAND_MAX - 0.5f) * 2.0f;
        vis16[i] = __halves2half2(__float2half(re), __float2half(im));
        vis32[i * 2] = re; vis32[i * 2 + 1] = im;
    }

    const int64_t ge = static_cast<int64_t>(Nf) * Ng * Ng;
    float* bl_d; double* freq_d; __half2* v16_d; float* v32_d;
    __half2* g16_d; __half2* g32_d;
    CUDA_CHECK_TEST(cudaMalloc(&bl_d, n_bl * 2 * sizeof(float)));
    CUDA_CHECK_TEST(cudaMalloc(&freq_d, Nf * sizeof(double)));
    CUDA_CHECK_TEST(cudaMalloc(&v16_d, Nf * n_bl * sizeof(__half2)));
    CUDA_CHECK_TEST(cudaMalloc(&v32_d, Nf * n_bl * 2 * sizeof(float)));
    CUDA_CHECK_TEST(cudaMalloc(&g16_d, ge * sizeof(__half2)));
    CUDA_CHECK_TEST(cudaMalloc(&g32_d, ge * sizeof(__half2)));
    CUDA_CHECK_TEST(cudaMemcpy(bl_d, bl_h.data(), n_bl * 2 * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK_TEST(cudaMemcpy(freq_d, freq_h.data(), Nf * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK_TEST(cudaMemcpy(v16_d, vis16.data(), Nf * n_bl * sizeof(__half2), cudaMemcpyHostToDevice));
    CUDA_CHECK_TEST(cudaMemcpy(v32_d, vis32.data(), Nf * n_bl * 2 * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK_TEST(cudaMemset(g16_d, 0, ge * sizeof(__half2)));
    CUDA_CHECK_TEST(cudaMemset(g32_d, 0, ge * sizeof(__half2)));

    imaging_kernels::launch_scatter_fp16(v16_d, bl_d, freq_d, g16_d, n_bl, Ng, Nf, 0, cs);
    imaging_kernels::launch_scatter_fp32in(v32_d, bl_d, freq_d, g32_d, n_bl, Ng, Nf, 0, cs);
    CUDA_CHECK_TEST(cudaDeviceSynchronize());

    std::vector<__half2> h16(ge), h32(ge);
    CUDA_CHECK_TEST(cudaMemcpy(h16.data(), g16_d, ge * sizeof(__half2), cudaMemcpyDeviceToHost));
    CUDA_CHECK_TEST(cudaMemcpy(h32.data(), g32_d, ge * sizeof(__half2), cudaMemcpyDeviceToHost));

    double max_err = 0.0; int nz = 0;
    for (int64_t i = 0; i < ge; i++) {
        float r1 = __half2float(__low2half(h16[i])), m1 = __half2float(__high2half(h16[i]));
        float r2 = __half2float(__low2half(h32[i])), m2 = __half2float(__high2half(h32[i]));
        float mag = sqrtf(r2 * r2 + m2 * m2);
        if (mag > 0.01f) {
            nz++;
            double e = sqrtf((r1-r2)*(r1-r2) + (m1-m2)*(m1-m2)) / mag;
            if (e > max_err) max_err = e;
        }
    }
    bool ok = (max_err < 0.01) && (nz > 0);
    if (!ok) fprintf(stderr, "  max_rel=%.4f%% nz=%d\n", max_err * 100, nz);

    cudaFree(bl_d); cudaFree(freq_d); cudaFree(v16_d); cudaFree(v32_d);
    cudaFree(g16_d); cudaFree(g32_d);
    return ok;
}

// ================================================================
// Test 5: FFT round-trip (FP32 path)
// ================================================================
static bool test_fft_roundtrip_fp32() {
    const int Ng = 32;
    std::vector<__half2> g(Ng * Ng);
    for (auto& v : g) v = __halves2half2(__float2half(0.0f), __float2half(0.0f));
    g[(Ng / 2) * Ng + (Ng / 2)] = __halves2half2(__float2half(1.0f), __float2half(0.0f));

    __half2* fp16_d; cufftComplex* fp32_d;
    CUDA_CHECK_TEST(cudaMalloc(&fp16_d, Ng * Ng * sizeof(__half2)));
    CUDA_CHECK_TEST(cudaMalloc(&fp32_d, Ng * Ng * sizeof(cufftComplex)));
    CUDA_CHECK_TEST(cudaMemcpy(fp16_d, g.data(), Ng * Ng * sizeof(__half2), cudaMemcpyHostToDevice));

    imaging_kernels::launch_cast_fp16_to_fp32(fp16_d, fp32_d, Ng * Ng);
    CUDA_CHECK_TEST(cudaDeviceSynchronize());

    cufftHandle plan;
    int nf[2] = {Ng, Ng};
    cufftPlanMany(&plan, 2, nf, nullptr, 1, 0, nullptr, 1, 0, CUFFT_C2C, 1);
    cufftExecC2C(plan, fp32_d, fp32_d, CUFFT_INVERSE);
    CUDA_CHECK_TEST(cudaDeviceSynchronize());

    std::vector<cufftComplex> img(Ng * Ng);
    CUDA_CHECK_TEST(cudaMemcpy(img.data(), fp32_d, Ng * Ng * sizeof(cufftComplex), cudaMemcpyDeviceToHost));

    double max_err = 0.0;
    for (int i = 0; i < Ng * Ng; i++) {
        double err = fabs(img[i].x * img[i].x + img[i].y * img[i].y - 1.0);
        if (err > max_err) max_err = err;
    }

    bool ok = (max_err < 0.01);
    if (!ok) fprintf(stderr, "  max|mag^2-1|=%.6f\n", max_err);

    cufftDestroy(plan); cudaFree(fp16_d); cudaFree(fp32_d);
    return ok;
}

// ================================================================
// Test 6: FFT round-trip (FP16 path via cufftXt)
// ================================================================
static bool test_fft_roundtrip_fp16() {
    const int Ng = 32;  // power of 2 required
    std::vector<__half2> g(Ng * Ng);
    for (auto& v : g) v = __halves2half2(__float2half(0.0f), __float2half(0.0f));
    // DC point at center
    g[(Ng / 2) * Ng + (Ng / 2)] = __halves2half2(__float2half(1.0f), __float2half(0.0f));

    __half2* grid_d;
    CUDA_CHECK_TEST(cudaMalloc(&grid_d, Ng * Ng * sizeof(__half2)));
    CUDA_CHECK_TEST(cudaMemcpy(grid_d, g.data(), Ng * Ng * sizeof(__half2), cudaMemcpyHostToDevice));

    // Create FP16 cuFFT plan
    cufftHandle plan;
    cufftCreate(&plan);
    long long nf[2] = {Ng, Ng};
    size_t ws = 0;
    cufftResult err = cufftXtMakePlanMany(plan, 2, nf,
                                           nullptr, 1, 0, CUDA_C_16F,
                                           nullptr, 1, 0, CUDA_C_16F,
                                           1, &ws, CUDA_C_16F);
    if (err != CUFFT_SUCCESS) {
        fprintf(stderr, "  cufftXtMakePlanMany(FP16) failed: %d\n", (int)err);
        cudaFree(grid_d);
        return false;
    }

    cufftXtExec(plan, grid_d, grid_d, CUFFT_INVERSE);
    CUDA_CHECK_TEST(cudaDeviceSynchronize());

    std::vector<__half2> img(Ng * Ng);
    CUDA_CHECK_TEST(cudaMemcpy(img.data(), grid_d, Ng * Ng * sizeof(__half2), cudaMemcpyDeviceToHost));

    // Magnitude should be 1.0 everywhere (point at center -> phase rotation only)
    double max_err = 0.0;
    for (int i = 0; i < Ng * Ng; i++) {
        float re = __half2float(__low2half(img[i]));
        float im = __half2float(__high2half(img[i]));
        double err2 = fabs(re * re + im * im - 1.0);
        if (err2 > max_err) max_err = err2;
    }

    bool ok = (max_err < 0.05);  // FP16 has ~0.1% precision
    if (!ok) fprintf(stderr, "  max|mag^2-1|=%.6f\n", max_err);

    cufftDestroy(plan); cudaFree(grid_d);
    return ok;
}

// ================================================================
// Test 7: Full pipeline API -- FP16 vis + FP32 FFT (original path)
// ================================================================
static bool test_pipeline_fp16vis_fp32fft() {
    const int N = 4, n_bl = N * (N + 1) / 2, Nf = 4, Ng = 32, n_beam = 2;
    const double freq0 = 1.0e9;
    const float cs = 1.0f;
    const double lambda = 299792458.0 / freq0;

    std::vector<float> bl_h(n_bl * 2, 0.0f);
    bl_h[tri_idx(1, 0) * 2] = static_cast<float>(2.0 * lambda);
    bl_h[tri_idx(1, 0) * 2 + 1] = static_cast<float>(3.0 * lambda);
    std::vector<double> freq_h(Nf);
    for (int i = 0; i < Nf; i++) freq_h[i] = freq0 + i * 1.0e6;
    std::vector<__half2> vis_h(static_cast<size_t>(Nf) * n_bl);
    for (auto& v : vis_h) v = __halves2half2(__float2half(1.0f), __float2half(0.0f));
    std::vector<int> beam_h = {Ng / 2, Ng / 2, Ng / 2 + 1, Ng / 2 + 1};

    float* bl_d; double* freq_d; __half2* vis_d; int* beam_d; float* bout_d;
    CUDA_CHECK_TEST(cudaMalloc(&bl_d, n_bl * 2 * sizeof(float)));
    CUDA_CHECK_TEST(cudaMalloc(&freq_d, Nf * sizeof(double)));
    CUDA_CHECK_TEST(cudaMalloc(&vis_d, Nf * n_bl * sizeof(__half2)));
    CUDA_CHECK_TEST(cudaMalloc(&beam_d, n_beam * 2 * sizeof(int)));
    CUDA_CHECK_TEST(cudaMalloc(&bout_d, Nf * n_beam * sizeof(float)));
    CUDA_CHECK_TEST(cudaMemcpy(bl_d, bl_h.data(), n_bl * 2 * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK_TEST(cudaMemcpy(freq_d, freq_h.data(), Nf * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK_TEST(cudaMemcpy(vis_d, vis_h.data(), Nf * n_bl * sizeof(__half2), cudaMemcpyHostToDevice));
    CUDA_CHECK_TEST(cudaMemcpy(beam_d, beam_h.data(), n_beam * 2 * sizeof(int), cudaMemcpyHostToDevice));

    imaging_pipeline::ImagingPipeline pipe;
    if (pipe.configure(N, Nf, Ng, n_beam, imaging_pipeline::FftPrecision::FP32) != 0) return false;
    pipe.set_baseline_uv(bl_d, n_bl);
    pipe.set_frequencies(freq_d);
    pipe.set_cell_size(cs);
    pipe.set_beam_pixels(beam_d, n_beam);

    if (pipe.grid_and_image(vis_d, bout_d, imaging_pipeline::VisPrecision::FP16) != 0) return false;
    CUDA_CHECK_TEST(cudaDeviceSynchronize());

    std::vector<float> bout(Nf * n_beam);
    CUDA_CHECK_TEST(cudaMemcpy(bout.data(), bout_d, Nf * n_beam * sizeof(float), cudaMemcpyDeviceToHost));

    bool ok = true;
    for (int i = 0; i < Nf * n_beam; i++)
        if (bout[i] < -0.001f || std::isnan(bout[i])) ok = false;
    if (*std::max_element(bout.begin(), bout.end()) < 1e-6f) ok = false;

    pipe.destroy();
    cudaFree(bl_d); cudaFree(freq_d); cudaFree(vis_d); cudaFree(beam_d); cudaFree(bout_d);
    return ok;
}

// ================================================================
// Test 8: Full pipeline API -- FP16 vis + FP16 FFT
// ================================================================
static bool test_pipeline_fp16vis_fp16fft() {
    const int N = 4, n_bl = N * (N + 1) / 2, Nf = 4, Ng = 32, n_beam = 2;
    const double freq0 = 1.0e9;
    const float cs = 1.0f;
    const double lambda = 299792458.0 / freq0;

    std::vector<float> bl_h(n_bl * 2, 0.0f);
    bl_h[tri_idx(1, 0) * 2] = static_cast<float>(2.0 * lambda);
    bl_h[tri_idx(1, 0) * 2 + 1] = static_cast<float>(3.0 * lambda);
    std::vector<double> freq_h(Nf);
    for (int i = 0; i < Nf; i++) freq_h[i] = freq0 + i * 1.0e6;
    std::vector<__half2> vis_h(static_cast<size_t>(Nf) * n_bl);
    for (auto& v : vis_h) v = __halves2half2(__float2half(1.0f), __float2half(0.0f));
    std::vector<int> beam_h = {Ng / 2, Ng / 2, Ng / 2 + 1, Ng / 2 + 1};

    float* bl_d; double* freq_d; __half2* vis_d; int* beam_d; float* bout_d;
    CUDA_CHECK_TEST(cudaMalloc(&bl_d, n_bl * 2 * sizeof(float)));
    CUDA_CHECK_TEST(cudaMalloc(&freq_d, Nf * sizeof(double)));
    CUDA_CHECK_TEST(cudaMalloc(&vis_d, Nf * n_bl * sizeof(__half2)));
    CUDA_CHECK_TEST(cudaMalloc(&beam_d, n_beam * 2 * sizeof(int)));
    CUDA_CHECK_TEST(cudaMalloc(&bout_d, Nf * n_beam * sizeof(float)));
    CUDA_CHECK_TEST(cudaMemcpy(bl_d, bl_h.data(), n_bl * 2 * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK_TEST(cudaMemcpy(freq_d, freq_h.data(), Nf * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK_TEST(cudaMemcpy(vis_d, vis_h.data(), Nf * n_bl * sizeof(__half2), cudaMemcpyHostToDevice));
    CUDA_CHECK_TEST(cudaMemcpy(beam_d, beam_h.data(), n_beam * 2 * sizeof(int), cudaMemcpyHostToDevice));

    imaging_pipeline::ImagingPipeline pipe;
    if (pipe.configure(N, Nf, Ng, n_beam, imaging_pipeline::FftPrecision::FP16) != 0) return false;
    pipe.set_baseline_uv(bl_d, n_bl);
    pipe.set_frequencies(freq_d);
    pipe.set_cell_size(cs);
    pipe.set_beam_pixels(beam_d, n_beam);

    if (pipe.grid_and_image(vis_d, bout_d, imaging_pipeline::VisPrecision::FP16) != 0) return false;
    CUDA_CHECK_TEST(cudaDeviceSynchronize());

    std::vector<float> bout(Nf * n_beam);
    CUDA_CHECK_TEST(cudaMemcpy(bout.data(), bout_d, Nf * n_beam * sizeof(float), cudaMemcpyDeviceToHost));

    bool ok = true;
    for (int i = 0; i < Nf * n_beam; i++)
        if (bout[i] < -0.001f || std::isnan(bout[i])) ok = false;
    if (*std::max_element(bout.begin(), bout.end()) < 1e-6f) ok = false;

    pipe.destroy();
    cudaFree(bl_d); cudaFree(freq_d); cudaFree(vis_d); cudaFree(beam_d); cudaFree(bout_d);
    return ok;
}

// ================================================================
// Test 9: Full pipeline API -- FP32 vis + FP32 FFT (direct, no widen)
// ================================================================
static bool test_pipeline_fp32vis_fp32fft() {
    const int N = 4, n_bl = N * (N + 1) / 2, Nf = 4, Ng = 32, n_beam = 2;
    const double freq0 = 1.0e9;
    const float cs = 1.0f;
    const double lambda = 299792458.0 / freq0;

    std::vector<float> bl_h(n_bl * 2, 0.0f);
    bl_h[tri_idx(1, 0) * 2] = static_cast<float>(2.0 * lambda);
    bl_h[tri_idx(1, 0) * 2 + 1] = static_cast<float>(3.0 * lambda);
    std::vector<double> freq_h(Nf);
    for (int i = 0; i < Nf; i++) freq_h[i] = freq0 + i * 1.0e6;
    // FP32 vis
    std::vector<float> vis_h(static_cast<size_t>(Nf) * n_bl * 2);
    for (size_t i = 0; i < vis_h.size(); i += 2) { vis_h[i] = 1.0f; vis_h[i + 1] = 0.0f; }
    std::vector<int> beam_h = {Ng / 2, Ng / 2, Ng / 2 + 1, Ng / 2 + 1};

    float* bl_d; double* freq_d; float* vis_d; int* beam_d; float* bout_d;
    CUDA_CHECK_TEST(cudaMalloc(&bl_d, n_bl * 2 * sizeof(float)));
    CUDA_CHECK_TEST(cudaMalloc(&freq_d, Nf * sizeof(double)));
    CUDA_CHECK_TEST(cudaMalloc(&vis_d, Nf * n_bl * 2 * sizeof(float)));
    CUDA_CHECK_TEST(cudaMalloc(&beam_d, n_beam * 2 * sizeof(int)));
    CUDA_CHECK_TEST(cudaMalloc(&bout_d, Nf * n_beam * sizeof(float)));
    CUDA_CHECK_TEST(cudaMemcpy(bl_d, bl_h.data(), n_bl * 2 * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK_TEST(cudaMemcpy(freq_d, freq_h.data(), Nf * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK_TEST(cudaMemcpy(vis_d, vis_h.data(), Nf * n_bl * 2 * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK_TEST(cudaMemcpy(beam_d, beam_h.data(), n_beam * 2 * sizeof(int), cudaMemcpyHostToDevice));

    imaging_pipeline::ImagingPipeline pipe;
    if (pipe.configure(N, Nf, Ng, n_beam, imaging_pipeline::FftPrecision::FP32) != 0) return false;
    pipe.set_baseline_uv(bl_d, n_bl);
    pipe.set_frequencies(freq_d);
    pipe.set_cell_size(cs);
    pipe.set_beam_pixels(beam_d, n_beam);

    if (pipe.grid_and_image(vis_d, bout_d, imaging_pipeline::VisPrecision::FP32) != 0) return false;
    CUDA_CHECK_TEST(cudaDeviceSynchronize());

    std::vector<float> bout(Nf * n_beam);
    CUDA_CHECK_TEST(cudaMemcpy(bout.data(), bout_d, Nf * n_beam * sizeof(float), cudaMemcpyDeviceToHost));

    bool ok = true;
    for (int i = 0; i < Nf * n_beam; i++)
        if (bout[i] < -0.001f || std::isnan(bout[i])) ok = false;
    if (*std::max_element(bout.begin(), bout.end()) < 1e-6f) ok = false;

    pipe.destroy();
    cudaFree(bl_d); cudaFree(freq_d); cudaFree(vis_d); cudaFree(beam_d); cudaFree(bout_d);
    return ok;
}

// ================================================================
// Test 10: Full pipeline API -- FP32 vis + FP16 FFT
// ================================================================
static bool test_pipeline_fp32vis_fp16fft() {
    const int N = 4, n_bl = N * (N + 1) / 2, Nf = 4, Ng = 32, n_beam = 2;
    const double freq0 = 1.0e9;
    const float cs = 1.0f;
    const double lambda = 299792458.0 / freq0;

    std::vector<float> bl_h(n_bl * 2, 0.0f);
    bl_h[tri_idx(1, 0) * 2] = static_cast<float>(2.0 * lambda);
    bl_h[tri_idx(1, 0) * 2 + 1] = static_cast<float>(3.0 * lambda);
    std::vector<double> freq_h(Nf);
    for (int i = 0; i < Nf; i++) freq_h[i] = freq0 + i * 1.0e6;
    std::vector<float> vis_h(static_cast<size_t>(Nf) * n_bl * 2);
    for (size_t i = 0; i < vis_h.size(); i += 2) { vis_h[i] = 1.0f; vis_h[i + 1] = 0.0f; }
    std::vector<int> beam_h = {Ng / 2, Ng / 2, Ng / 2 + 1, Ng / 2 + 1};

    float* bl_d; double* freq_d; float* vis_d; int* beam_d; float* bout_d;
    CUDA_CHECK_TEST(cudaMalloc(&bl_d, n_bl * 2 * sizeof(float)));
    CUDA_CHECK_TEST(cudaMalloc(&freq_d, Nf * sizeof(double)));
    CUDA_CHECK_TEST(cudaMalloc(&vis_d, Nf * n_bl * 2 * sizeof(float)));
    CUDA_CHECK_TEST(cudaMalloc(&beam_d, n_beam * 2 * sizeof(int)));
    CUDA_CHECK_TEST(cudaMalloc(&bout_d, Nf * n_beam * sizeof(float)));
    CUDA_CHECK_TEST(cudaMemcpy(bl_d, bl_h.data(), n_bl * 2 * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK_TEST(cudaMemcpy(freq_d, freq_h.data(), Nf * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK_TEST(cudaMemcpy(vis_d, vis_h.data(), Nf * n_bl * 2 * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK_TEST(cudaMemcpy(beam_d, beam_h.data(), n_beam * 2 * sizeof(int), cudaMemcpyHostToDevice));

    imaging_pipeline::ImagingPipeline pipe;
    if (pipe.configure(N, Nf, Ng, n_beam, imaging_pipeline::FftPrecision::FP16) != 0) return false;
    pipe.set_baseline_uv(bl_d, n_bl);
    pipe.set_frequencies(freq_d);
    pipe.set_cell_size(cs);
    pipe.set_beam_pixels(beam_d, n_beam);

    if (pipe.grid_and_image(vis_d, bout_d, imaging_pipeline::VisPrecision::FP32) != 0) return false;
    CUDA_CHECK_TEST(cudaDeviceSynchronize());

    std::vector<float> bout(Nf * n_beam);
    CUDA_CHECK_TEST(cudaMemcpy(bout.data(), bout_d, Nf * n_beam * sizeof(float), cudaMemcpyDeviceToHost));

    bool ok = true;
    for (int i = 0; i < Nf * n_beam; i++)
        if (bout[i] < -0.001f || std::isnan(bout[i])) ok = false;
    if (*std::max_element(bout.begin(), bout.end()) < 1e-6f) ok = false;

    pipe.destroy();
    cudaFree(bl_d); cudaFree(freq_d); cudaFree(vis_d); cudaFree(beam_d); cudaFree(bout_d);
    return ok;
}

// ================================================================
// Test 11: Beam extraction from FP16 image
// ================================================================
static bool test_beam_extraction_fp16() {
    const int Ng = 16, n_beam = 3;
    const float norm = 1.0f / (Ng * Ng);

    std::vector<__half2> img_h(Ng * Ng);
    for (auto& z : img_h) z = __halves2half2(__float2half(0.0f), __float2half(0.0f));
    img_h[4 * Ng + 5] = __halves2half2(__float2half(3.0f), __float2half(4.0f));  // |z|^2=25
    img_h[7 * Ng + 8] = __halves2half2(__float2half(1.0f), __float2half(0.0f));  // |z|^2=1
    img_h[0 * Ng + 0] = __halves2half2(__float2half(0.0f), __float2half(2.0f));  // |z|^2=4

    std::vector<int> pix = {4, 5, 7, 8, 0, 0};

    __half2* img_d; int* pix_d; float* out_d;
    CUDA_CHECK_TEST(cudaMalloc(&img_d, Ng * Ng * sizeof(__half2)));
    CUDA_CHECK_TEST(cudaMalloc(&pix_d, n_beam * 2 * sizeof(int)));
    CUDA_CHECK_TEST(cudaMalloc(&out_d, n_beam * sizeof(float)));
    CUDA_CHECK_TEST(cudaMemcpy(img_d, img_h.data(), Ng * Ng * sizeof(__half2), cudaMemcpyHostToDevice));
    CUDA_CHECK_TEST(cudaMemcpy(pix_d, pix.data(), n_beam * 2 * sizeof(int), cudaMemcpyHostToDevice));

    imaging_kernels::launch_extract_beam_fp16(img_d, pix_d, out_d, Ng, n_beam, norm, n_beam);
    CUDA_CHECK_TEST(cudaDeviceSynchronize());

    std::vector<float> out(n_beam);
    CUDA_CHECK_TEST(cudaMemcpy(out.data(), out_d, n_beam * sizeof(float), cudaMemcpyDeviceToHost));

    bool ok = true;
    float exp[] = {25.0f * norm, 1.0f * norm, 4.0f * norm};
    for (int i = 0; i < n_beam; i++)
        if (fabsf(out[i] - exp[i]) > 1e-3f) { ok = false;
            fprintf(stderr, "  beam[%d]=%.6f exp=%.6f\n", i, out[i], exp[i]); }

    cudaFree(img_d); cudaFree(pix_d); cudaFree(out_d);
    return ok;
}

// ================================================================
// Test 12: Beam extraction from FP32 image
// ================================================================
static bool test_beam_extraction_fp32() {
    const int Ng = 16, n_beam = 3;
    const float norm = 1.0f / (Ng * Ng);

    std::vector<cufftComplex> img_h(Ng * Ng);
    for (auto& z : img_h) { z.x = 0.0f; z.y = 0.0f; }
    img_h[4 * Ng + 5] = {3.0f, 4.0f};
    img_h[7 * Ng + 8] = {1.0f, 0.0f};
    img_h[0 * Ng + 0] = {0.0f, 2.0f};

    std::vector<int> pix = {4, 5, 7, 8, 0, 0};

    cufftComplex* img_d; int* pix_d; float* out_d;
    CUDA_CHECK_TEST(cudaMalloc(&img_d, Ng * Ng * sizeof(cufftComplex)));
    CUDA_CHECK_TEST(cudaMalloc(&pix_d, n_beam * 2 * sizeof(int)));
    CUDA_CHECK_TEST(cudaMalloc(&out_d, n_beam * sizeof(float)));
    CUDA_CHECK_TEST(cudaMemcpy(img_d, img_h.data(), Ng * Ng * sizeof(cufftComplex), cudaMemcpyHostToDevice));
    CUDA_CHECK_TEST(cudaMemcpy(pix_d, pix.data(), n_beam * 2 * sizeof(int), cudaMemcpyHostToDevice));

    imaging_kernels::launch_extract_beam(img_d, pix_d, out_d, Ng, n_beam, norm, n_beam);
    CUDA_CHECK_TEST(cudaDeviceSynchronize());

    std::vector<float> out(n_beam);
    CUDA_CHECK_TEST(cudaMemcpy(out.data(), out_d, n_beam * sizeof(float), cudaMemcpyDeviceToHost));

    bool ok = true;
    float exp[] = {25.0f * norm, 1.0f * norm, 4.0f * norm};
    for (int i = 0; i < n_beam; i++)
        if (fabsf(out[i] - exp[i]) > 1e-5f) ok = false;

    cudaFree(img_d); cudaFree(pix_d); cudaFree(out_d);
    return ok;
}

// ================================================================
// Test registry for --test selection
// ================================================================
struct TestEntry {
    const char* name;
    bool (*func)();
    const char* group;
};

static TestEntry all_tests[] = {
    {"gridding_scatter_fp16",      test_gridding_scatter_fp16,      "gridding"},
    {"gridding_scatter_fp32_grid", test_gridding_scatter_fp32_grid, "gridding"},
    {"conjugate_symmetry",         test_conjugate_symmetry,         "gridding"},
    {"fp16_vs_fp32_precision",     test_fp16_vs_fp32_precision,     "gridding"},
    {"fft_roundtrip_fp32",         test_fft_roundtrip_fp32,         "fft"},
    {"fft_roundtrip_fp16",         test_fft_roundtrip_fp16,         "fft"},
    {"beam_extraction_fp16",       test_beam_extraction_fp16,       "fft"},
    {"beam_extraction_fp32",       test_beam_extraction_fp32,       "fft"},
    {"pipeline_fp16vis_fp32fft",   test_pipeline_fp16vis_fp32fft,   "pipeline"},
    {"pipeline_fp16vis_fp16fft",   test_pipeline_fp16vis_fp16fft,   "pipeline"},
    {"pipeline_fp32vis_fp32fft",   test_pipeline_fp32vis_fp32fft,   "pipeline"},
    {"pipeline_fp32vis_fp16fft",   test_pipeline_fp32vis_fp16fft,   "pipeline"},
};
static const int num_tests = sizeof(all_tests) / sizeof(all_tests[0]);

// ================================================================
// Main — supports: ./test_imaging [--test <name>] [--group <group>] [--list]
// ================================================================
int main(int argc, char** argv) {
    const char* filter_test = nullptr;
    const char* filter_group = nullptr;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--test") == 0 && i + 1 < argc) {
            filter_test = argv[++i];
        } else if (strcmp(argv[i], "--group") == 0 && i + 1 < argc) {
            filter_group = argv[++i];
        } else if (strcmp(argv[i], "--list") == 0) {
            for (int j = 0; j < num_tests; j++)
                printf("%s [%s]\n", all_tests[j].name, all_tests[j].group);
            return 0;
        }
    }

    printf("=== Imaging Pipeline Tests ===\n\n");

    for (int i = 0; i < num_tests; i++) {
        if (filter_test && strcmp(filter_test, all_tests[i].name) != 0) continue;
        if (filter_group && strcmp(filter_group, all_tests[i].group) != 0) continue;

        printf("  %-55s", (std::string(all_tests[i].name) + "...").c_str());
        fflush(stdout);
        if (all_tests[i].func()) { printf("[PASS]\n"); tests_passed++; }
        else                     { printf("[FAIL]\n"); tests_failed++; }
    }

    printf("\n%d/%d tests passed\n", tests_passed, tests_passed + tests_failed);
    return tests_failed > 0 ? 1 : 0;
}
