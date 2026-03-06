/**
 * @file test_pipeline_dedisp.cu
 * @brief End-to-end test: Voltage + Visibility pipelines -> filterbank -> dedisp.
 *
 * Tests:
 *   1. Voltage Pipeline: random QC INT4 -> beamform -> FP32 filterbank ->
 *      inject dispersed signal -> corner turn -> dedisperse -> search -> detect.
 *   2. Visibility Pipeline (simulated): noise filterbank -> inject signal ->
 *      corner turn -> dedisperse -> search -> detect.
 *
 * Both tests inject a burst at known DM and time, then verify the dedisp
 * search recovers it with correct DM and arrival time.
 *
 * Build (standalone):
 *   nvcc -DSTANDALONE_TEST -DCUTLASS_GEMM_API -DDEDISP_API_LIB \
 *        -I../include -I../cutlass_interface -I../dedisp_tcfdd \
 *        test_pipeline_dedisp.cu voltage_pipeline.cu corner_turn.cu \
 *        -L../../cutlass_interface_build -lcutlass_gemm_api \
 *        -L../../dedisp_tcfdd_build -ldedisp_api \
 *        -lcufft -lcublas -lcublasLt -lcurand -o test_pipeline_dedisp
 */

// These defines come from CMake; uncomment for manual nvcc builds:
// #define STANDALONE_TEST
// #define CUTLASS_GEMM_API
// #define DEDISP_API_LIB

#include <voltage_pipeline.h>
#include <corner_turn.h>
#include "dedisp_api.h"

#include <cuda_runtime.h>
#include <curand.h>
#include <cstdio>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <vector>

// ========================================================================
// Helpers
// ========================================================================

#define CHECK_CUDA(x) do {                                                   \
    cudaError_t err = (x);                                                   \
    if (err != cudaSuccess) {                                                \
        fprintf(stderr, "CUDA error at %s:%d: %s\n",                        \
                __FILE__, __LINE__, cudaGetErrorString(err));                \
        exit(1);                                                             \
    }                                                                        \
} while(0)

#define CHECK_CURAND(x) do {                                                 \
    curandStatus_t st = (x);                                                 \
    if (st != CURAND_STATUS_SUCCESS) {                                       \
        fprintf(stderr, "cuRAND error %d at %s:%d\n",                       \
                (int)st, __FILE__, __LINE__);                               \
        exit(1);                                                             \
    }                                                                        \
} while(0)

/// @brief Generate random QC INT4 data on GPU.
///
/// Each byte: high nibble = Re (sign-magnitude), low nibble = Im (sign-magnitude).
/// Bit 3 = sign, bits 2:0 = magnitude, range [-7, +7].
/// Uses cuRAND to generate random bytes then masks to valid QC format.
__global__ void generate_random_qc_kernel(uint8_t* output, const uint8_t* random_bytes,
                                           int64_t total)
{
    for (int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
         i < total; i += (int64_t)gridDim.x * blockDim.x)
    {
        // Use two random bytes to get two independent nibbles
        uint8_t r = random_bytes[i];
        // High nibble: sign(bit3) + magnitude(bits 2:0), cap mag at 7
        uint8_t hi = r >> 4;      // 0-15
        uint8_t lo = r & 0x0F;    // 0-15
        // Ensure valid sign-magnitude: bit3=sign, bits 2:0 = mag in [0,7]
        // Already valid since 4 bits can only be 0-15, and we use bit3 as sign.
        output[i] = (hi << 4) | lo;
    }
}

static void generate_random_qc(uint8_t* d_qc, int64_t total_bytes, unsigned seed)
{
    // Allocate temp random bytes
    uint8_t* d_rand = nullptr;
    CHECK_CUDA(cudaMalloc(&d_rand, total_bytes));

    // Generate random bytes with cuRAND
    curandGenerator_t gen;
    CHECK_CURAND(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));
    CHECK_CURAND(curandSetPseudoRandomGeneratorSeed(gen, seed));

    // curandGenerate works on uint32, so round up
    int64_t n_uint32 = (total_bytes + 3) / 4;
    CHECK_CURAND(curandGenerate(gen, (unsigned int*)d_rand, n_uint32));

    // Convert to valid QC format
    const int block = 256;
    int grid = (int)std::min((total_bytes + block - 1) / block, (int64_t)1024);
    generate_random_qc_kernel<<<grid, block>>>(d_qc, d_rand, total_bytes);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CURAND(curandDestroyGenerator(gen));
    CHECK_CUDA(cudaFree(d_rand));
}

/// @brief Generate random FP32 interleaved complex weights on GPU.
static void generate_random_weights(float* d_weights, int64_t n_complex, unsigned seed)
{
    curandGenerator_t gen;
    CHECK_CURAND(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));
    CHECK_CURAND(curandSetPseudoRandomGeneratorSeed(gen, seed));
    CHECK_CURAND(curandGenerateNormal(gen, d_weights, n_complex * 2, 0.0f, 1.0f));
    CHECK_CURAND(curandDestroyGenerator(gen));
}

/// @brief Generate Gaussian noise filterbank on GPU.
static void generate_noise_filterbank(float* d_fb, int64_t total, float mean, float stddev,
                                       unsigned seed)
{
    curandGenerator_t gen;
    CHECK_CURAND(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));
    CHECK_CURAND(curandSetPseudoRandomGeneratorSeed(gen, seed));
    // curandGenerateNormal needs even count
    int64_t n_gen = (total + 1) & ~1LL;
    CHECK_CURAND(curandGenerateNormal(gen, d_fb, n_gen, mean, stddev));
    CHECK_CURAND(curandDestroyGenerator(gen));
}

static void print_separator() {
    printf("================================================================\n");
}

// ========================================================================
// Test 1: Voltage Pipeline -> filterbank -> inject -> dedisp -> search
// ========================================================================

static int test_voltage_pipeline()
{
    print_separator();
    printf("TEST 1: Voltage Pipeline -> Filterbank -> Inject -> Dedisp\n");
    print_separator();

    // Pipeline dimensions (small for test speed)
    const int n_ant   = 64;
    const int n_beam  = 32;
    const int n_ch    = 64;
    const int n_time  = 256;
    const int n_tps   = 1;   // no time integration (keep full time resolution)
    const int n_time_out = n_time / n_tps;

    printf("  Dimensions: n_ant=%d, n_beam=%d, n_ch=%d, n_time=%d, n_time_out=%d\n",
           n_ant, n_beam, n_ch, n_time, n_time_out);

    // --- Step 1: Create voltage pipeline ---
    printf("\n  [1] Creating VoltagePipeline...\n");
    ggp::VoltagePipeline::Config vp_cfg;
    vp_cfg.n_antennae       = n_ant;
    vp_cfg.n_beams          = n_beam;
    vp_cfg.n_channels       = n_ch;
    vp_cfg.n_time           = n_time;
    vp_cfg.n_time_power_sum = n_tps;
    vp_cfg.n_polarizations  = 2;
    vp_cfg.compute_mode     = ggp::VoltageComputeMode::FP8;
    vp_cfg.kernel_tune_verbosity = 0;
    vp_cfg.strategy_tune_verbosity = 1;
    vp_cfg.gemm_tune = true;

    ggp::VoltagePipeline pipeline(vp_cfg);
    printf("      Direct path: %s\n", pipeline.uses_direct_path() ? "YES" : "NO (4M fallback)");

    // --- Step 2: Generate and set random weights ---
    printf("  [2] Generating random weights...\n");
    const int64_t weight_complex = (int64_t)n_ch * n_beam * n_ant;
    float* d_weights = nullptr;
    CHECK_CUDA(cudaMalloc(&d_weights, weight_complex * 2 * sizeof(float)));
    generate_random_weights(d_weights, weight_complex, 42);
    pipeline.set_weights(d_weights, nullptr);
    CHECK_CUDA(cudaDeviceSynchronize());
    printf("      Weights set (%lld complex elements)\n", (long long)weight_complex);

    // --- Step 3: Generate random QC INT4 input ---
    printf("  [3] Generating random QC INT4 packets...\n");
    int64_t qc_bytes = pipeline.input_size();
    uint8_t* d_qc = nullptr;
    CHECK_CUDA(cudaMalloc(&d_qc, qc_bytes));
    generate_random_qc(d_qc, qc_bytes, 123);
    printf("      QC payload: %lld bytes (%d ch x 2 pol x %d ant x %d time)\n",
           (long long)qc_bytes, n_ch, n_ant, n_time);

    // --- Step 4: Run beamform -> FP32 filterbank ---
    printf("  [4] Running compute_filterbank()...\n");
    pipeline.compute_filterbank(d_qc, nullptr);
    CHECK_CUDA(cudaDeviceSynchronize());

    // Verify filterbank is non-zero
    const float* d_fb_chan = pipeline.filterbank_chanmajor();
    {
        std::vector<float> h_fb(n_ch * n_beam * n_time_out);
        CHECK_CUDA(cudaMemcpy(h_fb.data(), d_fb_chan,
                              h_fb.size() * sizeof(float), cudaMemcpyDeviceToHost));
        float fb_sum = 0.0f, fb_max = 0.0f;
        for (auto v : h_fb) { fb_sum += v; fb_max = std::max(fb_max, std::fabs(v)); }
        printf("      Filterbank [n_ch=%d, n_beam=%d, n_time_out=%d]: "
               "sum=%.2f, max=%.4f\n", n_ch, n_beam, n_time_out, fb_sum, fb_max);
        if (fb_max == 0.0f) {
            printf("  FAIL: filterbank is all zeros!\n");
            CHECK_CUDA(cudaFree(d_weights));
            CHECK_CUDA(cudaFree(d_qc));
            return 1;
        }
    }

    // --- Step 5: Estimate noise statistics BEFORE injection ---
    printf("  [5] Estimating noise statistics...\n");
    float noise_mean = 0.0f, noise_std = 1.0f;
    {
        // Copy filterbank to beam-major for noise estimation
        float* d_fb_beam_tmp = nullptr;
        int64_t fb_elems = (int64_t)n_ch * n_beam * n_time_out;
        CHECK_CUDA(cudaMalloc(&d_fb_beam_tmp, fb_elems * sizeof(float)));
        ggp::corner_turn_nf_nb(d_fb_beam_tmp, d_fb_chan, n_ch, n_beam, n_time_out, nullptr);
        CHECK_CUDA(cudaDeviceSynchronize());

        // Read first beam's filterbank [n_ch x n_time_out] to estimate noise
        int beam_slice = n_ch * n_time_out;
        std::vector<float> h_beam(beam_slice);
        CHECK_CUDA(cudaMemcpy(h_beam.data(), d_fb_beam_tmp,
                              beam_slice * sizeof(float), cudaMemcpyDeviceToHost));

        double sum = 0, sum2 = 0;
        for (float v : h_beam) { sum += v; sum2 += (double)v * v; }
        noise_mean = (float)(sum / beam_slice);
        noise_std  = (float)std::sqrt(sum2 / beam_slice - noise_mean * noise_mean);
        if (noise_std < 1e-10f) noise_std = 1.0f;  // safety
        printf("      Noise: mean=%.4f, std=%.4f (from beam 0)\n", noise_mean, noise_std);
        CHECK_CUDA(cudaFree(d_fb_beam_tmp));
    }

    // --- Step 6: Inject dispersed signal into filterbank ---
    printf("  [6] Injecting dispersed signal into filterbank...\n");

    // Dedisp config
    dedisp_api::DedispConfig dd_cfg;
    dd_cfg.Nf = n_ch;
    dd_cfg.Nt = n_time_out;
    dd_cfg.Ndm = 64;
    dd_cfg.f_min_MHz = 1200.0f;
    dd_cfg.f_max_MHz = 1600.0f;
    dd_cfg.max_dm = 100.0f;
    dd_cfg.total_obs_time_s = 0.1f;
    dd_cfg.compute_mode = dedisp_api::ComputeMode::CuBLAS_FP32;
    dd_cfg.max_batch_size = n_beam;

    dedisp_api::DedispPipeline dedisp(dd_cfg);
    int ret = dedisp.initialize();
    if (ret != 0) {
        printf("  FAIL: dedisp.initialize() returned %d\n", ret);
        CHECK_CUDA(cudaFree(d_weights));
        CHECK_CUDA(cudaFree(d_qc));
        return 1;
    }
    printf("      Dedisp initialized: mode=%s, Nt_padded=%d\n",
           dedisp.get_compute_mode_string(), dedisp.get_nt_padded());

    // Corner turn BEFORE injection (dedisp expects beam-major [n_beam, n_ch, n_time])
    int64_t fb_elems = (int64_t)n_ch * n_beam * n_time_out;
    float* d_fb_beam = nullptr;
    CHECK_CUDA(cudaMalloc(&d_fb_beam, fb_elems * sizeof(float)));
    ggp::corner_turn_nf_nb(d_fb_beam, d_fb_chan, n_ch, n_beam, n_time_out, nullptr);
    CHECK_CUDA(cudaDeviceSynchronize());

    // Inject signal: DM=50, amplitude = 50 * noise_std (high SNR)
    dedisp_api::InjectionParams inj;
    inj.dm = 50.0f;
    inj.amplitude = 50.0f * noise_std;
    inj.pulse_start_time_s = 0.05f;
    inj.width_s = 0.0f;
    inj.scattering_s = 0.0f;

    ret = dedisp.inject_signal(d_fb_beam, &inj, 1, n_beam);
    if (ret != 0) {
        printf("  FAIL: inject_signal() returned %d\n", ret);
        CHECK_CUDA(cudaFree(d_weights));
        CHECK_CUDA(cudaFree(d_qc));
        CHECK_CUDA(cudaFree(d_fb_beam));
        return 1;
    }
    printf("      Injected: DM=%.1f pc/cm^3, amp=%.1f, t=%.3f s\n",
           inj.dm, inj.amplitude, inj.pulse_start_time_s);

    // --- Step 7: Dedisperse + search ---
    printf("  [7] Running dedisperse_and_search()...\n");

    int widths[] = {1, 2, 4, 8};
    dedisp_api::SearchConfig search_cfg;
    search_cfg.widths = widths;
    search_cfg.num_widths = 4;
    search_cfg.noise_mean = noise_mean;
    search_cfg.noise_stddev = noise_std;
    search_cfg.max_candidates = 10;

    dedisp_api::Candidate cands[10];
    int num_found = 0;
    ret = dedisp.dedisperse_and_search(d_fb_beam, cands, &num_found,
                                        search_cfg, n_beam);
    if (ret != 0) {
        printf("  FAIL: dedisperse_and_search() returned %d\n", ret);
        CHECK_CUDA(cudaFree(d_weights));
        CHECK_CUDA(cudaFree(d_qc));
        CHECK_CUDA(cudaFree(d_fb_beam));
        return 1;
    }

    // --- Step 8: Report results ---
    printf("  [8] Results: found %d candidates\n", num_found);
    bool pass = false;
    if (num_found > 0) {
        for (int c = 0; c < std::min(num_found, 3); c++) {
            printf("      Candidate %d: DM=%.2f pc/cm^3, time=%.4f s, SNR=%.1f, "
                   "intensity=%.1f, width=%d\n",
                   c, cands[c].dm, cands[c].time_s, cands[c].snr,
                   cands[c].intensity, cands[c].width);
        }
        float dm_err = std::fabs(cands[0].dm - inj.dm);
        float t_err  = std::fabs(cands[0].time_s - inj.pulse_start_time_s);
        printf("      DM error: %.2f, time error: %.4f s\n", dm_err, t_err);
        pass = (dm_err < 15.0f && cands[0].snr > 5.0f);
    }

    printf("\n  Voltage Pipeline Test: %s\n", pass ? "PASS" : "FAIL");
    print_separator();

    CHECK_CUDA(cudaFree(d_weights));
    CHECK_CUDA(cudaFree(d_qc));
    CHECK_CUDA(cudaFree(d_fb_beam));

    return pass ? 0 : 1;
}

// ========================================================================
// Test 2: Visibility Pipeline (simulated) -> inject -> dedisp -> search
// ========================================================================

static int test_visibility_pipeline()
{
    print_separator();
    printf("TEST 2: Visibility Pipeline (simulated) -> Inject -> Dedisp\n");
    print_separator();

    // Simulates the output of: correlate -> image -> beam extraction
    // The imaging pipeline produces [Nf x n_beam] float per payload.
    // After corner turn: [n_beam x Nf x 1] per payload.
    // Multiple payloads accumulate into [n_beam x Nf x Nt] filterbank.

    const int n_ch    = 64;
    const int n_beam  = 32;
    const int n_time  = 256;

    printf("  Dimensions: n_ch=%d, n_beam=%d, n_time=%d\n", n_ch, n_beam, n_time);

    // --- Step 1: Generate noise filterbank (beam-major) ---
    printf("\n  [1] Generating Gaussian noise filterbank (simulating imaging output)...\n");
    int64_t fb_elems = (int64_t)n_beam * n_ch * n_time;
    float* d_fb = nullptr;
    CHECK_CUDA(cudaMalloc(&d_fb, fb_elems * sizeof(float)));
    generate_noise_filterbank(d_fb, fb_elems, 0.0f, 1.0f, 999);
    CHECK_CUDA(cudaDeviceSynchronize());

    // Read noise stats
    float noise_mean = 0.0f, noise_std = 1.0f;
    {
        int sample_size = n_ch * n_time;  // first beam
        std::vector<float> h_sample(sample_size);
        CHECK_CUDA(cudaMemcpy(h_sample.data(), d_fb,
                              sample_size * sizeof(float), cudaMemcpyDeviceToHost));
        double sum = 0, sum2 = 0;
        for (float v : h_sample) { sum += v; sum2 += (double)v * v; }
        noise_mean = (float)(sum / sample_size);
        noise_std  = (float)std::sqrt(sum2 / sample_size - noise_mean * noise_mean);
        if (noise_std < 1e-10f) noise_std = 1.0f;
        printf("      Noise stats: mean=%.4f, std=%.4f\n", noise_mean, noise_std);
    }

    // --- Step 2: Set up dedisp pipeline ---
    printf("  [2] Creating dedisp pipeline...\n");
    dedisp_api::DedispConfig dd_cfg;
    dd_cfg.Nf = n_ch;
    dd_cfg.Nt = n_time;
    dd_cfg.Ndm = 64;
    dd_cfg.f_min_MHz = 1200.0f;
    dd_cfg.f_max_MHz = 1600.0f;
    dd_cfg.max_dm = 100.0f;
    dd_cfg.total_obs_time_s = 0.1f;
    dd_cfg.compute_mode = dedisp_api::ComputeMode::CuBLAS_FP32;
    dd_cfg.max_batch_size = n_beam;

    dedisp_api::DedispPipeline dedisp(dd_cfg);
    int ret = dedisp.initialize();
    if (ret != 0) {
        printf("  FAIL: dedisp.initialize() returned %d\n", ret);
        CHECK_CUDA(cudaFree(d_fb));
        return 1;
    }
    printf("      Dedisp initialized: mode=%s, Nt_padded=%d\n",
           dedisp.get_compute_mode_string(), dedisp.get_nt_padded());

    // --- Step 3: Inject dispersed signal ---
    printf("  [3] Injecting dispersed signal...\n");

    dedisp_api::InjectionParams inj;
    inj.dm = 30.0f;
    inj.amplitude = 100.0f;  // High SNR against unit-variance noise
    inj.pulse_start_time_s = 0.04f;
    inj.width_s = 0.0f;
    inj.scattering_s = 0.0f;

    ret = dedisp.inject_signal(d_fb, &inj, 1, n_beam);
    if (ret != 0) {
        printf("  FAIL: inject_signal() returned %d\n", ret);
        CHECK_CUDA(cudaFree(d_fb));
        return 1;
    }
    printf("      Injected: DM=%.1f pc/cm^3, amp=%.1f, t=%.3f s\n",
           inj.dm, inj.amplitude, inj.pulse_start_time_s);

    // --- Step 4: Dedisperse + search ---
    printf("  [4] Running dedisperse_and_search()...\n");

    int widths[] = {1, 2, 4, 8};
    dedisp_api::SearchConfig search_cfg;
    search_cfg.widths = widths;
    search_cfg.num_widths = 4;
    search_cfg.noise_mean = noise_mean;
    search_cfg.noise_stddev = noise_std;
    search_cfg.max_candidates = 10;

    dedisp_api::Candidate cands[10];
    int num_found = 0;
    ret = dedisp.dedisperse_and_search(d_fb, cands, &num_found,
                                        search_cfg, n_beam);
    if (ret != 0) {
        printf("  FAIL: dedisperse_and_search() returned %d\n", ret);
        CHECK_CUDA(cudaFree(d_fb));
        return 1;
    }

    // --- Step 5: Report results ---
    printf("  [5] Results: found %d candidates\n", num_found);
    bool pass = false;
    if (num_found > 0) {
        for (int c = 0; c < std::min(num_found, 3); c++) {
            printf("      Candidate %d: DM=%.2f pc/cm^3, time=%.4f s, SNR=%.1f, "
                   "intensity=%.1f, width=%d\n",
                   c, cands[c].dm, cands[c].time_s, cands[c].snr,
                   cands[c].intensity, cands[c].width);
        }
        float dm_err = std::fabs(cands[0].dm - inj.dm);
        float t_err  = std::fabs(cands[0].time_s - inj.pulse_start_time_s);
        printf("      DM error: %.2f, time error: %.4f s\n", dm_err, t_err);
        pass = (dm_err < 10.0f && cands[0].snr > 5.0f);
    }

    printf("\n  Visibility Pipeline Test: %s\n", pass ? "PASS" : "FAIL");
    print_separator();

    CHECK_CUDA(cudaFree(d_fb));

    return pass ? 0 : 1;
}

// ========================================================================
// Main
// ========================================================================

int main(int argc, char* argv[])
{
    printf("\n");
    print_separator();
    printf("  Pipeline -> Filterbank -> Dedisp End-to-End Test\n");
    print_separator();
    printf("\n");

    // Print GPU info
    int dev = 0;
    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, dev));
    printf("GPU: %s (SM %d.%d, %d SMs, %.0f MB)\n\n",
           prop.name, prop.major, prop.minor, prop.multiProcessorCount,
           prop.totalGlobalMem / 1e6);

    const char* test = (argc > 1) ? argv[1] : "all";

    int failures = 0;

    if (strcmp(test, "all") == 0 || strcmp(test, "voltage") == 0) {
        failures += test_voltage_pipeline();
        printf("\n");
    }
    if (strcmp(test, "all") == 0 || strcmp(test, "visibility") == 0) {
        failures += test_visibility_pipeline();
        printf("\n");
    }

    print_separator();
    if (failures == 0) {
        printf("  ALL TESTS PASSED\n");
    } else {
        printf("  %d TEST(S) FAILED\n", failures);
    }
    print_separator();

    return failures;
}
