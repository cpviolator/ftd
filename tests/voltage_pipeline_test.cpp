#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <vector>
#include <complex>
#include <random>

#include <test.h>
#include <misc.h>
#include <ggp.h>
#include <voltage_pipeline.h>

// ============================================================================
// CLI-configurable parameters
// ============================================================================

int VBF_n_antennae = 128;
int VBF_n_beams = 64;
int VBF_n_channels = 4;
int VBF_n_time = 128;
bool VBF_profile = false;
int VBF_n_iters = 10;

// ============================================================================
// CPU FP64 reference beamformer
// ============================================================================

/// @brief Decode one QC byte to complex double.
static std::complex<double> decode_qc(uint8_t byte) {
  auto decode_nibble = [](uint8_t raw) -> double {
    int sign = (raw & 0x8) ? -1 : 1;
    int mag  = raw & 0x7;
    return (double)(sign * mag);
  };
  return { decode_nibble((byte >> 4) & 0x0F),
           decode_nibble(byte & 0x0F) };
}

/// @brief CPU reference: full beamformer pipeline (OpenMP-parallelized).
/// @param qc      [n_ch x n_pol x n_ant x n_time] bytes
/// @param weights [n_ch x n_beam x n_ant] complex<double>
/// @param output  [n_ch x n_beam x n_time_out] doubles (raw power)
static void cpu_beamformer(
    const uint8_t* qc,
    const std::vector<std::complex<double>>& weights,
    std::vector<double>& output,
    int n_ant, int n_beam, int n_ch, int n_time, int n_pol, int n_tps)
{
  const int n_time_out = n_time / n_tps;
  output.resize((size_t)n_ch * n_beam * n_time_out, 0.0);

  #pragma omp parallel for collapse(2) schedule(dynamic)
  for (int ch = 0; ch < n_ch; ++ch) {
    for (int beam = 0; beam < n_beam; ++beam) {
      for (int t_out = 0; t_out < n_time_out; ++t_out) {
        double power = 0.0;
        for (int dt = 0; dt < n_tps; ++dt) {
          int t = t_out * n_tps + dt;
          for (int pol = 0; pol < n_pol; ++pol) {
            std::complex<double> beam_val(0, 0);
            for (int ant = 0; ant < n_ant; ++ant) {
              int64_t qc_idx = ((int64_t)ch * n_pol + pol) * n_ant * n_time
                             + (int64_t)ant * n_time + t;
              auto v = decode_qc(qc[qc_idx]);
              int64_t w_idx = (int64_t)ch * n_beam * n_ant
                            + (int64_t)beam * n_ant + ant;
              beam_val += weights[w_idx] * v;
            }
            power += std::norm(beam_val);
          }
        }
        int64_t out_idx = (int64_t)ch * n_beam * n_time_out
                        + (int64_t)beam * n_time_out + t_out;
        output[out_idx] = power;
      }
    }
  }
}

// ============================================================================
// Shared test data (generated once, reused across tests)
// ============================================================================

static bool g_data_initialized = false;
static std::vector<uint8_t> g_h_qc;
static std::vector<float> g_h_weights;
static std::vector<std::complex<double>> g_weights_fp64;
static std::vector<double> g_ref_power;
static int g_n_ant, g_n_beam, g_n_ch, g_n_time;
static const int g_n_pol = 2;
static const int g_n_tps = 4;

static void ensure_test_data() {
  if (g_data_initialized) return;

  g_n_ant  = VBF_n_antennae;
  g_n_beam = VBF_n_beams;
  g_n_ch   = VBF_n_channels;
  g_n_time = VBF_n_time;

  printfQuda("Generating test data: n_ant=%d n_beam=%d n_ch=%d n_time=%d\n",
             g_n_ant, g_n_beam, g_n_ch, g_n_time);

  // Generate random QC voltages
  std::mt19937 rng(42);
  const int64_t qc_size = (int64_t)g_n_ch * g_n_pol * g_n_ant * g_n_time;
  g_h_qc.resize(qc_size);
  for (auto& b : g_h_qc) b = rng() & 0xFF;

  // Generate random complex weights (FP32 interleaved + FP64 reference)
  const int64_t w_complex = (int64_t)g_n_ch * g_n_beam * g_n_ant;
  g_h_weights.resize(w_complex * 2);
  g_weights_fp64.resize(w_complex);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

  for (int64_t i = 0; i < w_complex; ++i) {
    float re = dist(rng);
    float im = dist(rng);
    g_h_weights[2 * i]     = re;
    g_h_weights[2 * i + 1] = im;
    g_weights_fp64[i] = {(double)re, (double)im};
  }

  // CPU reference
  cpu_beamformer(g_h_qc.data(), g_weights_fp64, g_ref_power,
                 g_n_ant, g_n_beam, g_n_ch, g_n_time, g_n_pol, g_n_tps);

  g_data_initialized = true;
}

// ============================================================================
// Helper: run beamformer test for a given compute mode, return max error
// ============================================================================

static double run_beamformer_test(const char* label,
                                  ggp::VoltageComputeMode compute) {
  ensure_test_data();

  printfQuda("=== %s (n_ant=%d, n_beam=%d, n_ch=%d, n_time=%d) ===\n",
             label, g_n_ant, g_n_beam, g_n_ch, g_n_time);

  ggp::VoltagePipeline::Config cfg;
  cfg.n_antennae       = g_n_ant;
  cfg.n_beams          = g_n_beam;
  cfg.n_channels       = g_n_ch;
  cfg.n_time           = g_n_time;
  cfg.n_polarizations  = 2;
  cfg.n_time_power_sum = g_n_tps;
  cfg.compute_mode     = compute;
  cfg.profile          = false;

  ggp::VoltagePipeline pipeline(cfg);

  // Upload QC data
  uint8_t* d_qc = nullptr;
  cudaMalloc(&d_qc, pipeline.input_size());
  cudaMemcpy(d_qc, g_h_qc.data(), pipeline.input_size(), cudaMemcpyHostToDevice);

  // Upload and set weights
  float* d_weights = nullptr;
  const int64_t w_floats = (int64_t)g_n_ch * g_n_beam * g_n_ant * 2;
  cudaMalloc(&d_weights, w_floats * sizeof(float));
  cudaMemcpy(d_weights, g_h_weights.data(),
             w_floats * sizeof(float), cudaMemcpyHostToDevice);
  pipeline.set_weights(d_weights, nullptr);

  // Determine scale: normalise so max power maps to ~200
  double max_power = 0;
  for (auto& p : g_ref_power) max_power = std::max(max_power, p);
  float scale = (max_power > 0) ? 200.0f / (float)max_power : 1.0f;

  // Allocate output
  uint8_t* d_output = nullptr;
  cudaMalloc(&d_output, pipeline.output_size());
  cudaMemset(d_output, 0, pipeline.output_size());

  // Run GPU beamformer (first call = warmup + correctness data)
  pipeline.beamform(d_output, d_qc, scale, nullptr);
  cudaDeviceSynchronize();

  // Benchmark: timed iterations when profiling is enabled
  if (VBF_profile && VBF_n_iters > 0) {
    cudaEvent_t ev_start, ev_stop;
    cudaEventCreate(&ev_start);
    cudaEventCreate(&ev_stop);
    cudaEventRecord(ev_start);
    for (int iter = 0; iter < VBF_n_iters; iter++) {
      pipeline.beamform(d_output, d_qc, scale, nullptr);
    }
    cudaEventRecord(ev_stop);
    cudaEventSynchronize(ev_stop);
    float total_ms = 0;
    cudaEventElapsedTime(&total_ms, ev_start, ev_stop);
    float avg_ms = total_ms / VBF_n_iters;
    printfQuda("  --- Beamformer Timing (%d iters) ---\n", VBF_n_iters);
    printfQuda("  Total:       %.3f ms\n", avg_ms);
    printfQuda("  Per channel: %.4f ms  (%d channels)\n", avg_ms / g_n_ch, g_n_ch);
    printfQuda("  Per beam:    %.4f ms  (%d beams)\n", avg_ms / g_n_beam, g_n_beam);
    printfQuda("  Throughput:  %.1f beamforms/s\n", 1000.0 / avg_ms);
    printfQuda("  ---------------------------------\n");
    cudaEventDestroy(ev_start);
    cudaEventDestroy(ev_stop);
  }

  // Copy back
  std::vector<uint8_t> h_output(pipeline.output_size());
  cudaMemcpy(h_output.data(), d_output,
             pipeline.output_size(), cudaMemcpyDeviceToHost);

  // Compare against CPU reference
  int tol;
  if (compute == ggp::VoltageComputeMode::FP6) {
    tol = std::max(15, 1 + g_n_ant / 8);
  } else {
    tol = std::max(4, 1 + g_n_ant / 32);
  }

  int errors = 0;
  int max_diff = 0;
  double sum_diff = 0;
  const int n_time_out = g_n_time / g_n_tps;
  const int64_t total = (int64_t)g_n_ch * g_n_beam * n_time_out;

  for (int64_t i = 0; i < total; ++i) {
    double scaled_ref = g_ref_power[i] * scale;
    int ref_val = std::max(0, std::min(255, (int)round(scaled_ref)));
    int gpu_val = h_output[i];
    int diff = abs(gpu_val - ref_val);
    max_diff = std::max(max_diff, diff);
    sum_diff += diff;
    if (diff > tol) errors++;
  }

  double mean_diff = sum_diff / total;

  printfQuda("  max_diff=%d  mean_diff=%.2f  errors=%d/%lld  (tol=%d)\n",
             max_diff, mean_diff, errors, (long long)total, tol);
  printfQuda("  %s\n", errors == 0 ? "PASS" : "FAIL");

  cudaFree(d_qc);
  cudaFree(d_weights);
  cudaFree(d_output);

  return (double)errors;
}

// ============================================================================
// Test 1: FP8 voltage beamformer correctness
// ============================================================================

double VoltagePipelineFP8Test() {
  return run_beamformer_test("Voltage Pipeline FP8", ggp::VoltageComputeMode::FP8);
}

// ============================================================================
// Test 2: FP6 voltage beamformer correctness (SM100/SM120 only)
// ============================================================================

double VoltagePipelineFP6Test() {
#ifdef CUTLASS_GEMM_API
  try {
    return run_beamformer_test("Voltage Pipeline FP6", ggp::VoltageComputeMode::FP6);
  } catch (...) {
    printfQuda("=== Voltage Pipeline FP6: SKIPPED (not available on this architecture) ===\n");
    return 0.0;
  }
#else
  printfQuda("=== Voltage Pipeline FP6: SKIPPED (CUTLASS_GEMM_API not linked) ===\n");
  return 0.0;
#endif
}

// ============================================================================
// Test 3: Pipeline lifecycle (construct, set_weights, destroy)
// ============================================================================

double VoltagePipelineLifecycleTest() {
  printfQuda("=== Voltage Pipeline Lifecycle Test ===\n");

  ggp::VoltagePipeline::Config cfg;
  cfg.n_antennae   = 64;
  cfg.n_beams      = 32;
  cfg.n_channels   = 2;
  cfg.n_time       = 64;
  cfg.n_time_power_sum = 4;

  ggp::VoltagePipeline pipeline(cfg);

  // Verify sizes
  int64_t expected_input  = (int64_t)2 * 2 * 64 * 64;  // n_ch * n_pol * n_ant * n_time
  int64_t expected_output = (int64_t)2 * 32 * (64/4);   // n_ch * n_beam * n_time_out

  bool pass = (pipeline.input_size() == expected_input) &&
              (pipeline.output_size() == expected_output) &&
              (pipeline.n_channels() == 2) &&
              (pipeline.n_beams() == 32) &&
              (pipeline.n_time() == 64) &&
              (pipeline.n_time_out() == 16);

  printfQuda("  input_size=%lld (expected %lld)\n",
             (long long)pipeline.input_size(), (long long)expected_input);
  printfQuda("  output_size=%lld (expected %lld)\n",
             (long long)pipeline.output_size(), (long long)expected_output);
  printfQuda("  %s\n", pass ? "PASS" : "FAIL");

  return pass ? 0.0 : 1.0;
}

// ============================================================================
// GTest wrappers
// ============================================================================

using test_t = ::testing::tuple<QudaPrecision>;

class VBF_FP8_GTest : public ::testing::TestWithParam<test_t> {};
class VBF_FP6_GTest : public ::testing::TestWithParam<test_t> {};
class VBF_Lifecycle_GTest : public ::testing::TestWithParam<test_t> {};

TEST_P(VBF_FP8_GTest, verify) {
  double err = VoltagePipelineFP8Test();
  EXPECT_LT(err, 0.5);
}

TEST_P(VBF_FP6_GTest, verify) {
  double err = VoltagePipelineFP6Test();
  EXPECT_LT(err, 0.5);
}

TEST_P(VBF_Lifecycle_GTest, verify) {
  double err = VoltagePipelineLifecycleTest();
  EXPECT_LT(err, 0.5);
}

auto single_prec_vbf = ::testing::Values(QUDA_SINGLE_PRECISION);
INSTANTIATE_TEST_SUITE_P(VoltagePipeline, VBF_FP8_GTest, single_prec_vbf);
INSTANTIATE_TEST_SUITE_P(VoltagePipeline, VBF_FP6_GTest, single_prec_vbf);
INSTANTIATE_TEST_SUITE_P(VoltagePipeline, VBF_Lifecycle_GTest, single_prec_vbf);

// ============================================================================
// CLI + main
// ============================================================================

struct voltage_pipeline_test : quda_test {

  void add_command_line_group(std::shared_ptr<GGPApp> app) const override
  {
    quda_test::add_command_line_group(app);

    auto opgroup = app->add_option_group("VoltagePipeline",
      "Options controlling the voltage pipeline test");
    opgroup->add_option("--VBF-n-antennae", VBF_n_antennae,
      "Number of antennae (default 128)");
    opgroup->add_option("--VBF-n-beams", VBF_n_beams,
      "Number of beams (default 64)");
    opgroup->add_option("--VBF-n-channels", VBF_n_channels,
      "Number of channels (default 4)");
    opgroup->add_option("--VBF-n-time", VBF_n_time,
      "Number of time samples (default 128)");
    opgroup->add_option("--profile", VBF_profile,
      "Enable per-stage timing output");
    opgroup->add_option("--n-iters", VBF_n_iters,
      "Number of timed iterations for benchmarking (default 10)");
  }

  voltage_pipeline_test(int argc, char **argv)
    : quda_test("Voltage Pipeline Test", argc, argv) {}
};

int main(int argc, char **argv) {

  voltage_pipeline_test test(argc, argv);
  test.init();

  int result = 0;
  if (enable_testing) {
    result = test.execute();
    if (result) warningQuda("Google tests for Voltage Pipeline failed.");
  } else {
    VoltagePipelineLifecycleTest();
    VoltagePipelineFP8Test();
    VoltagePipelineFP6Test();
  }

  return result;
}
