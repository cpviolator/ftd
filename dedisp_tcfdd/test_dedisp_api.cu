/**
 * @file test_dedisp_api.cu
 * @brief End-to-end test for the dedisp PIMPL API.
 *
 * Only includes dedisp_api.h + <cuda_runtime.h>, proving PIMPL isolation.
 *
 * Usage:
 *   test_dedisp_api [mode] [test]
 *
 *   mode: cublas (default), cublas_lt_fp16, cublas_lt_fp8
 *   test: inject_search (default), dedisperse_only, high_dm, multi_batch
 */

#include "dedisp_api.h"
#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>
#include <cstring>

#define TEST_CHECK(call) do {                                            \
    cudaError_t err = (call);                                           \
    if (err != cudaSuccess) {                                           \
        fprintf(stderr, "CUDA Error at %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(err));                               \
        return 1;                                                       \
    }                                                                   \
} while(0)

static dedisp_api::ComputeMode parse_mode(const char* s) {
    if (!s) return dedisp_api::ComputeMode::CuBLAS_FP32;
    if (strcmp(s, "cublas_lt_fp16") == 0) return dedisp_api::ComputeMode::CuBLASLt_FP16;
    if (strcmp(s, "cublas_lt_fp8") == 0)  return dedisp_api::ComputeMode::CuBLASLt_FP8;
#ifdef HAS_CUTLASS_GEMM
    if (strcmp(s, "cutlass_fp8") == 0)    return dedisp_api::ComputeMode::CUTLASS_FP8;
    if (strcmp(s, "cutlass_fp6") == 0)    return dedisp_api::ComputeMode::CUTLASS_FP6;
    if (strcmp(s, "cutlass_fp4") == 0)    return dedisp_api::ComputeMode::CUTLASS_FP4;
#endif
    return dedisp_api::ComputeMode::CuBLAS_FP32;
}

// Test 1: Inject + dedisperse_and_search (combined API)
static int test_inject_search(dedisp_api::ComputeMode mode) {
    printf("=== Test: inject_search (mode=%d) ===\n\n", (int)mode);

    const int Nf = 64, Nt = 256, Ndm = 64, batch_size = 1;

    dedisp_api::DedispConfig config;
    config.Nf = Nf; config.Nt = Nt; config.Ndm = Ndm;
    config.f_min_MHz = 1200.0f; config.f_max_MHz = 1600.0f;
    config.max_dm = 100.0f; config.total_obs_time_s = 0.1f;
    config.compute_mode = mode; config.max_batch_size = batch_size;

    dedisp_api::DedispPipeline pipeline(config);
    int ret = pipeline.initialize();
    if (ret != 0) { printf("FAIL: initialize() returned %d\n", ret); return 1; }
    printf("  mode=%s, Nt_padded=%d\n", pipeline.get_compute_mode_string(), pipeline.get_nt_padded());

    size_t fb_bytes = (size_t)batch_size * Nf * Nt * sizeof(float);
    float* d_fb = nullptr;
    TEST_CHECK(cudaMalloc(&d_fb, fb_bytes));
    TEST_CHECK(cudaMemset(d_fb, 0, fb_bytes));

    dedisp_api::InjectionParams inj;
    inj.dm = 50.0f; inj.amplitude = 100.0f;
    inj.pulse_start_time_s = 0.05f; inj.width_s = 0.0f; inj.scattering_s = 0.0f;

    ret = pipeline.inject_signal(d_fb, &inj, 1, batch_size);
    if (ret != 0) { printf("FAIL: inject_signal() returned %d\n", ret); cudaFree(d_fb); return 1; }

    int widths[] = {1, 2, 4, 8};
    dedisp_api::SearchConfig search_cfg;
    search_cfg.widths = widths; search_cfg.num_widths = 4;
    search_cfg.noise_mean = 0.0f; search_cfg.noise_stddev = 1.0f;
    search_cfg.max_candidates = 10;

    dedisp_api::Candidate cands[10];
    int num_found = 0;
    ret = pipeline.dedisperse_and_search(d_fb, cands, &num_found, search_cfg, batch_size);
    if (ret != 0) { printf("FAIL: dedisperse_and_search() returned %d\n", ret); cudaFree(d_fb); return 1; }

    cudaFree(d_fb);

    bool pass = false;
    if (num_found > 0) {
        float dm_err = std::fabs(cands[0].dm - inj.dm);
        float t_err  = std::fabs(cands[0].time_s - inj.pulse_start_time_s);
        printf("  Best: DM=%.2f (err=%.2f), time=%.4fs (err=%.4f), SNR=%.1f\n",
               cands[0].dm, dm_err, cands[0].time_s, t_err, cands[0].snr);
        pass = (dm_err < 10.0f && t_err < 0.02f && cands[0].snr > 0.0f);
    }
    printf("  %s\n\n", pass ? "PASS" : "FAIL");
    return pass ? 0 : 1;
}

// Test 2: Dedisperse-only (split API: dedisperse then search separately)
static int test_dedisperse_only(dedisp_api::ComputeMode mode) {
    printf("=== Test: dedisperse_only (mode=%d) ===\n\n", (int)mode);

    const int Nf = 64, Nt = 256, Ndm = 64, batch_size = 1;

    dedisp_api::DedispConfig config;
    config.Nf = Nf; config.Nt = Nt; config.Ndm = Ndm;
    config.f_min_MHz = 1200.0f; config.f_max_MHz = 1600.0f;
    config.max_dm = 100.0f; config.total_obs_time_s = 0.1f;
    config.compute_mode = mode; config.max_batch_size = batch_size;

    dedisp_api::DedispPipeline pipeline(config);
    int ret = pipeline.initialize();
    if (ret != 0) { printf("FAIL: initialize() returned %d\n", ret); return 1; }

    int Nt_padded = pipeline.get_nt_padded();
    printf("  mode=%s, Nt_padded=%d\n", pipeline.get_compute_mode_string(), Nt_padded);

    // Allocate input filterbank and inject signal
    size_t fb_bytes = (size_t)batch_size * Nf * Nt * sizeof(float);
    float* d_fb = nullptr;
    TEST_CHECK(cudaMalloc(&d_fb, fb_bytes));
    TEST_CHECK(cudaMemset(d_fb, 0, fb_bytes));

    dedisp_api::InjectionParams inj;
    inj.dm = 50.0f; inj.amplitude = 100.0f;
    inj.pulse_start_time_s = 0.05f; inj.width_s = 0.0f; inj.scattering_s = 0.0f;
    ret = pipeline.inject_signal(d_fb, &inj, 1, batch_size);
    if (ret != 0) { printf("FAIL: inject_signal() returned %d\n", ret); cudaFree(d_fb); return 1; }

    // Allocate output dedispersed array
    size_t dd_bytes = (size_t)batch_size * Ndm * Nt * sizeof(float);
    float* d_dd = nullptr;
    TEST_CHECK(cudaMalloc(&d_dd, dd_bytes));
    TEST_CHECK(cudaMemset(d_dd, 0, dd_bytes));

    // Step 1: Dedisperse only
    ret = pipeline.dedisperse(d_fb, d_dd, batch_size);
    if (ret != 0) { printf("FAIL: dedisperse() returned %d\n", ret); cudaFree(d_fb); cudaFree(d_dd); return 1; }
    printf("  dedisperse() OK\n");

    // Step 2: Search on dedispersed data
    int widths[] = {1, 2, 4, 8};
    dedisp_api::SearchConfig search_cfg;
    search_cfg.widths = widths; search_cfg.num_widths = 4;
    search_cfg.noise_mean = 0.0f; search_cfg.noise_stddev = 1.0f;
    search_cfg.max_candidates = 10;

    dedisp_api::Candidate cands[10];
    int num_found = 0;
    ret = pipeline.search(d_dd, cands, &num_found, search_cfg, batch_size);
    if (ret != 0) { printf("FAIL: search() returned %d\n", ret); cudaFree(d_fb); cudaFree(d_dd); return 1; }
    printf("  search() OK — found %d candidates\n", num_found);

    cudaFree(d_fb);
    cudaFree(d_dd);

    bool pass = false;
    if (num_found > 0) {
        float dm_err = std::fabs(cands[0].dm - inj.dm);
        float t_err  = std::fabs(cands[0].time_s - inj.pulse_start_time_s);
        printf("  Best: DM=%.2f (err=%.2f), time=%.4fs (err=%.4f), SNR=%.1f\n",
               cands[0].dm, dm_err, cands[0].time_s, t_err, cands[0].snr);
        pass = (dm_err < 10.0f && t_err < 0.02f && cands[0].snr > 0.0f);
    }
    printf("  %s\n\n", pass ? "PASS" : "FAIL");
    return pass ? 0 : 1;
}

// Test 3: High DM near boundary
static int test_high_dm(dedisp_api::ComputeMode mode) {
    printf("=== Test: high_dm (mode=%d) ===\n\n", (int)mode);

    const int Nf = 128, Nt = 512, Ndm = 128, batch_size = 1;

    dedisp_api::DedispConfig config;
    config.Nf = Nf; config.Nt = Nt; config.Ndm = Ndm;
    config.f_min_MHz = 1200.0f; config.f_max_MHz = 1600.0f;
    config.max_dm = 200.0f; config.total_obs_time_s = 0.2f;
    config.compute_mode = mode; config.max_batch_size = batch_size;

    dedisp_api::DedispPipeline pipeline(config);
    int ret = pipeline.initialize();
    if (ret != 0) { printf("FAIL: initialize() returned %d\n", ret); return 1; }

    size_t fb_bytes = (size_t)batch_size * Nf * Nt * sizeof(float);
    float* d_fb = nullptr;
    TEST_CHECK(cudaMalloc(&d_fb, fb_bytes));
    TEST_CHECK(cudaMemset(d_fb, 0, fb_bytes));

    // Inject at DM=150 (near max_dm/2, high dispersion)
    dedisp_api::InjectionParams inj;
    inj.dm = 150.0f; inj.amplitude = 100.0f;
    inj.pulse_start_time_s = 0.1f; inj.width_s = 0.0f; inj.scattering_s = 0.0f;
    ret = pipeline.inject_signal(d_fb, &inj, 1, batch_size);
    if (ret != 0) { printf("FAIL: inject_signal() returned %d\n", ret); cudaFree(d_fb); return 1; }

    int widths[] = {1, 2, 4, 8};
    dedisp_api::SearchConfig search_cfg;
    search_cfg.widths = widths; search_cfg.num_widths = 4;
    search_cfg.noise_mean = 0.0f; search_cfg.noise_stddev = 1.0f;
    search_cfg.max_candidates = 10;

    dedisp_api::Candidate cands[10];
    int num_found = 0;
    ret = pipeline.dedisperse_and_search(d_fb, cands, &num_found, search_cfg, batch_size);
    if (ret != 0) { printf("FAIL: dedisperse_and_search() returned %d\n", ret); cudaFree(d_fb); return 1; }

    cudaFree(d_fb);

    bool pass = false;
    if (num_found > 0) {
        float dm_err = std::fabs(cands[0].dm - inj.dm);
        float t_err  = std::fabs(cands[0].time_s - inj.pulse_start_time_s);
        printf("  Best: DM=%.2f (err=%.2f), time=%.4fs (err=%.4f), SNR=%.1f\n",
               cands[0].dm, dm_err, cands[0].time_s, t_err, cands[0].snr);
        // Higher DM = wider tolerance on DM accuracy
        pass = (dm_err < 20.0f && t_err < 0.03f && cands[0].snr > 0.0f);
    }
    printf("  %s\n\n", pass ? "PASS" : "FAIL");
    return pass ? 0 : 1;
}

// Test 4: Multi-batch processing
static int test_multi_batch(dedisp_api::ComputeMode mode) {
    printf("=== Test: multi_batch (mode=%d) ===\n\n", (int)mode);

    const int Nf = 64, Nt = 256, Ndm = 64, batch_size = 4;

    dedisp_api::DedispConfig config;
    config.Nf = Nf; config.Nt = Nt; config.Ndm = Ndm;
    config.f_min_MHz = 1200.0f; config.f_max_MHz = 1600.0f;
    config.max_dm = 100.0f; config.total_obs_time_s = 0.1f;
    config.compute_mode = mode; config.max_batch_size = batch_size;

    dedisp_api::DedispPipeline pipeline(config);
    int ret = pipeline.initialize();
    if (ret != 0) { printf("FAIL: initialize() returned %d\n", ret); return 1; }

    size_t fb_bytes = (size_t)batch_size * Nf * Nt * sizeof(float);
    float* d_fb = nullptr;
    TEST_CHECK(cudaMalloc(&d_fb, fb_bytes));
    TEST_CHECK(cudaMemset(d_fb, 0, fb_bytes));

    // Inject same signal into first batch element
    dedisp_api::InjectionParams inj;
    inj.dm = 50.0f; inj.amplitude = 100.0f;
    inj.pulse_start_time_s = 0.05f; inj.width_s = 0.0f; inj.scattering_s = 0.0f;
    ret = pipeline.inject_signal(d_fb, &inj, 1, batch_size);
    if (ret != 0) { printf("FAIL: inject_signal() returned %d\n", ret); cudaFree(d_fb); return 1; }

    int widths[] = {1, 2, 4, 8};
    dedisp_api::SearchConfig search_cfg;
    search_cfg.widths = widths; search_cfg.num_widths = 4;
    search_cfg.noise_mean = 0.0f; search_cfg.noise_stddev = 1.0f;
    search_cfg.max_candidates = 10;

    dedisp_api::Candidate cands[10];
    int num_found = 0;
    ret = pipeline.dedisperse_and_search(d_fb, cands, &num_found, search_cfg, batch_size);
    if (ret != 0) { printf("FAIL: dedisperse_and_search() returned %d\n", ret); cudaFree(d_fb); return 1; }

    cudaFree(d_fb);

    bool pass = false;
    if (num_found > 0) {
        float dm_err = std::fabs(cands[0].dm - inj.dm);
        float t_err  = std::fabs(cands[0].time_s - inj.pulse_start_time_s);
        printf("  batch_size=%d, found %d candidates\n", batch_size, num_found);
        printf("  Best: DM=%.2f (err=%.2f), time=%.4fs (err=%.4f), SNR=%.1f\n",
               cands[0].dm, dm_err, cands[0].time_s, t_err, cands[0].snr);
        pass = (dm_err < 10.0f && t_err < 0.02f && cands[0].snr > 0.0f);
    }
    printf("  %s\n\n", pass ? "PASS" : "FAIL");
    return pass ? 0 : 1;
}

int main(int argc, char* argv[]) {
    const char* mode_str = (argc > 1) ? argv[1] : "cublas";
    const char* test_str = (argc > 2) ? argv[2] : "inject_search";

    dedisp_api::ComputeMode mode = parse_mode(mode_str);
    printf("Mode: %s, Test: %s\n\n", mode_str, test_str);

    if (strcmp(test_str, "inject_search") == 0)    return test_inject_search(mode);
    if (strcmp(test_str, "dedisperse_only") == 0)   return test_dedisperse_only(mode);
    if (strcmp(test_str, "high_dm") == 0)            return test_high_dm(mode);
    if (strcmp(test_str, "multi_batch") == 0)        return test_multi_batch(mode);

    fprintf(stderr, "Unknown test: %s\n", test_str);
    fprintf(stderr, "Available: inject_search, dedisperse_only, high_dm, multi_batch\n");
    return 1;
}
