#include <iostream>
#include <vector>
#include <complex>
#include <cmath>
#include <numeric>
#include <algorithm>
#include <random>
#include <ctime>
#include <string>
#include <iomanip> // For std::setprecision
#include <map>
#include <sstream>
#include <chrono>
#include <type_traits>
#include <memory>
#include <omp.h>

#include <cuda_runtime.h>
#include <cuda.h>
#include <cufft.h>
#include <cublas_v2.h>
#include <curand.h> // --- ADDED for GPU-side data generation ---

#include <cublasLt.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>

// --- ADDED: Thrust Headers for Reduction ---
#include <thrust/device_ptr.h>
#include <thrust/transform_reduce.h>
#include <thrust/functional.h>
#include <thrust/execution_policy.h>
#include <thrust/system/cuda/execution_policy.h>

#ifdef HAS_CUTLASS_GEMM
#include "cutlass_gemm_api.h"
#endif

// ═══════════════════════════════════════════════════════════════
// Namespace-scope textual includes: macros, helpers, CUDA kernels
// ═══════════════════════════════════════════════════════════════
#include "gpu_fdd_improved/config.hpp"
#include "gpu_fdd_improved/host_helpers.hpp"
#include "gpu_fdd_improved/kernels_fdd.hpp"
#include "gpu_fdd_improved/kernels_data_prep.hpp"
#include "gpu_fdd_improved/kernels_candidate.hpp"


// -- END CUDA KERNELS --

// ═══════════════════════════════════════════════════════════════
// FddGpuPipeline — GPU pipeline state manager
// ═══════════════════════════════════════════════════════════════
#include "gpu_fdd_improved/pipeline_class.hpp"

// ═══════════════════════════════════════════════════════════════
// DedispApplication — High-level orchestration
// ═══════════════════════════════════════════════════════════════

/**
 * @brief Main application class for running dedispersion simulations.
 */
template <typename Real>
class DedispApplication {
 public:
  using Complex = std::complex<Real>;
  using RMatrix = std::vector<std::vector<Real>>;
  using RBatchMatrix = std::vector<RMatrix>;

  using ComplexTypeGpu =
      typename std::conditional<std::is_same<Real, float>::value, cufftComplex,
                                cufftDoubleComplex>::type;


  struct SimParams {
    bool dry_run;
    std::string algorithm_type;
    std::string fdd_compute_mode;
    std::string filter_type;
    std::string signal_type;
    size_t num_freq_channels;
    size_t num_time_samples;
    int num_dm_trials;
    Real f_min;
    Real f_max;
    Real min_dm_search;
    Real max_dm_search;
    Real total_obs_time;
    int num_candidates_to_find;
    Real noise_mean;
    Real noise_stddev;
    Real min_amplitude;
    Real max_amplitude;
    bool print_plots;
    std::vector<int> pulse_widths;
    unsigned int seed;
    int batch_size;
    int num_pipelines;
    Real pulse_width_s;
    Real scattering_time_s;
  };

  struct PulseCandidate {
    Real intensity = 0;
    Real dm = 0;
    Real time = 0;
    Real snr = 0;
    size_t dm_bin = 0;
    size_t time_bin = 0;
    int found_with_width = 0;

    bool operator<(const PulseCandidate& other) const { return snr < other.snr; }
  };

  struct PerfStats {
    long long flops;
    long long fft_flops = 0;
    long long precomp_flops = 0;
    long long core_flops = 0;
    long long total_mem_bytes;
    double arithmetic_intensity;
  };

  struct VerificationResult {
    int batch_idx;
    Real inj_dm;
    Real inj_time;
    Real inj_amp;
    Real det_dm;
    Real det_time;
    Real det_snr;
    Real dm_error;
    Real time_error;
    bool matched;
  };

  DedispApplication(const SimParams& params)
    : params_(params),
      dev_shared_phasors_by_time_(),
      dev_full_intensity_batch(),
      dev_dedispersed_full_batch(),
      d_pulsar_params_(),
      d_time_delays_(),
      d_f_k_values_(),
      d_search_results_(),
      gpu_pipeline_(nullptr),
      rand_gen_(NULL),
      compute_stream_(NULL)
  {
    CUDA_CHECK(cudaStreamCreate(&compute_stream_));
    CURAND_CHECK(curandCreateGenerator(&rand_gen_, CURAND_RNG_PSEUDO_PHILOX4_32_10));

    unsigned int seed = (params_.seed == 0) ? std::time(0) : params_.seed;
    CURAND_CHECK(curandSetPseudoRandomGeneratorSeed(rand_gen_, (unsigned long long)seed));
  }

  ~DedispApplication() {
    if (gpu_pipeline_) {
        gpu_pipeline_.reset();
    }

    CURAND_CHECK(curandDestroyGenerator(rand_gen_));
    CUDA_CHECK(cudaStreamDestroy(compute_stream_));
  }

  bool validate_parameters() const;

  // Class-body textual include for application method implementations
  #include "gpu_fdd_improved/application_impl.hpp"

public:
  // Reversibility verification test
  #include "gpu_fdd_improved/reversibility_test.hpp"

  // (private methods defined in application_impl.hpp via private: label)

private:
  const Real PI = 3.14159265358979323846;
  SimParams params_;

  size_t shared_nt_padded_;

  DeviceBuffer dev_shared_phasors_by_time_;
  DeviceBuffer d_time_delays_;
  DeviceBuffer d_f_k_values_;

  DeviceBuffer dev_full_intensity_batch;
  DeviceBuffer dev_dedispersed_full_batch;
  DeviceBuffer d_pulsar_params_;
  DeviceBuffer d_search_results_;

  std::unique_ptr<FddGpuPipeline<Real>> gpu_pipeline_;
  curandGenerator_t rand_gen_;
  cudaStream_t compute_stream_;
};

// Out-of-class definition: validate_parameters
template <typename Real>
bool DedispApplication<Real>::validate_parameters() const {
  auto check_positive = [&](const auto& value, const std::string& name) {
    if (value <= 0) {
      std::cerr << "Error: --" << name << " must be a positive value."
                << std::endl;
      return false;
    }
    return true;
  };

  if (!check_positive(params_.num_freq_channels, "num-freq-channels"))
    return false;
  if (!check_positive(params_.num_time_samples, "num-time-samples"))
    return false;
  if (!check_positive(params_.num_dm_trials, "num-dm-trials")) return false;
  if (!check_positive(params_.f_min, "f-min-MHz")) return false;
  if (!check_positive(params_.f_max, "f-max-MHz")) return false;
  if (!check_positive(params_.total_obs_time, "total-obs-time-s")) return false;
  if (!check_positive(params_.batch_size, "batch-size")) return false;
  if (!check_positive(params_.num_pipelines, "num-pipelines")) return false;

  if (params_.noise_stddev < 0) {
    std::cerr << "Error: --noise-stddev cannot be negative." << std::endl;
    return false;
  }
  if (params_.f_min >= params_.f_max) {
    std::cerr << "Error: --f-min-MHz must be strictly less than --f-max-MHz."
              << std::endl;
    return false;
  }
  if (params_.min_amplitude > params_.max_amplitude) {
    std::cerr
        << "Error: --min-amplitude cannot be greater than --max-amplitude."
        << std::endl;
    return false;
  }

  return true;
}

void print_usage(const char* prog_name) {
  std::cerr
      << "Usage: " << prog_name
      << " [--help] | [--dry-run <yes|no>] --algorithm <...> --precision "
         "<single|double> [options...]\n\n"
      << "Execution Modes:\n"
      << "  --dry-run <yes|no>           - If yes, estimate performance "
         "without running the main algorithm (Default: no).\n\n"
      << "Algorithms:\n"
      << "  fdd-gemm-batched      - High-performance, batched FDD using a "
         "GPU pipeline (default).\n"
      << "Precision:\n"
      << "  single                - Use 32-bit float precision.\n"
      << "  double                - Use 64-bit double precision.\n\n"
      << "  --fdd-mode <mode>                  - Compute backend and precision:\n"
      << "                                       'kernel'         (Custom FP32, default)\n"
      << "                                       'cublas'         (Standard FP32)\n"
      << "                                       'cublas_lt_fp16' (Tensor Core FP16)\n"
      << "                                       'cublas_lt_fp8'  (Tensor Core FP8 - Requires Ada/Hopper)\n"
      << "                                       'cutlass'        (CUTLASS FP8 - Requires Hopper/Blackwell + libcutlass_gemm_api)\n"
      << "                                       'cutlass_fp6'    (CUTLASS FP6 E3M2 - MXFP block-scaled, Blackwell only)\n"
      << "                                       'cutlass_fp4'    (CUTLASS FP4 E2M1 - MXFP block-scaled, Blackwell only, lossy, 2x throughput)\n\n"
      << "Options:\n"
      << "  --num-pipelines <int>              - Number of parallel pipeline "
         "objects to create (Default: 1).\n"
      << "  --batch-size <int>                 - The number of items to "
         "process in a batch (Default: 1).\n"
      << "  --num-freq-channels <int>          - Number of frequency channels "
         "(Default: 512).\n"
      << "  --num-time-samples <int>           - Number of time samples "
         "(Default: 2048).\n"
      << "  --num-dm-trials <int>              - Number of dispersion measure "
         "trials to search (Default: 256).\n"
      << "  --f-min-MHz <float>                - Minimum frequency of the "
         "band in MHz (Default: 1200.0).\n"
      << "  --f-max-MHz <float>                - Maximum frequency of the "
         "band in MHz (Default: 1600.0).\n"
      << "  --max-dm-search <float>            - Maximum DM to search in "
         "pc/cm^3 (Default: 100.0).\n"
      << "  --total-obs-time-s <float>         - Total observation time in "
         "seconds (Default: 0.1).\n"
      << "  --num-candidates-to-find <int>     - Number of top pulse "
         "candidates to find (Default: 10).\n"
      << "  --pulse-widths <\"1,2,4,8\">         - Comma-separated list of "
         "pulse widths to search (Default: 1,2,4,8).\n"
      << "  --filter-type <boxcar|gaussian|matched>  - Type of convolution "
         "filter for matched filtering (Default: boxcar).\n"
      << "  --noise-mean <float>               - Mean of the background "
         "Gaussian noise (Default: 0.0).\n"
      << "  --noise-stddev <float>             - Standard deviation of the "
         "noise (Default: 1.5). Set to 0 for no noise.\n"
      << "  --min-amplitude <float>            - Minimum amplitude for "
         "random bursts (burst mode only).\n"
      << "  --max-amplitude <float>            - Maximum amplitude for "
         "random bursts (burst mode only).\n"
      << "  --print-plots <yes|no>             - Enable or disable ASCII "
         "waterfall plots (Default:no).\n"
      << "  --seed <int>                       - Seed for random number "
         "generator. (Default: 1234, use 0 for time-based seed).\n"
      << "  --test-reversibility <yes|no>      - Run FDD reversibility "
         "verification test (forward->inverse round-trip). (Default: no).\n\n"
      << "Examples:\n\n"
      << "  # cuBLAS FP32 (Reference)\n"
      << "  " << prog_name << " --precision single --batch-size 128 --fdd-mode cublas \\\n"
      << "      --noise-stddev 5 --noise-mean 0 --min-amplitude 10 --max-amplitude 10 --seed 42\n\n"
      << "  # cuBLASLt FP16 (Tensor Core)\n"
      << "  " << prog_name << " --precision single --batch-size 128 --fdd-mode cublas_lt_fp16 \\\n"
      << "      --noise-stddev 5 --noise-mean 0 --min-amplitude 10 --max-amplitude 10 --seed 42\n\n"
      << "  # cuBLASLt FP8 (Tensor Core)\n"
      << "  " << prog_name << " --precision single --batch-size 128 --fdd-mode cublas_lt_fp8 \\\n"
      << "      --noise-stddev 5 --noise-mean 0 --min-amplitude 10 --max-amplitude 10 --seed 42\n\n"
      << "  # CUTLASS FP8 (Blackwell, requires libcutlass_gemm_api)\n"
      << "  " << prog_name << " --precision single --batch-size 128 --fdd-mode cutlass \\\n"
      << "      --noise-stddev 5 --noise-mean 0 --min-amplitude 10 --max-amplitude 10 --seed 42\n\n"
      << "  # CUTLASS FP6 (MXFP Block-Scaled, equivalent accuracy)\n"
      << "  " << prog_name << " --precision single --batch-size 128 --fdd-mode cutlass_fp6 \\\n"
      << "      --noise-stddev 5 --noise-mean 0 --min-amplitude 10 --max-amplitude 10 --seed 42\n\n"
      << "  # CUTLASS FP4 (Lossy, highest throughput)\n"
      << "  " << prog_name << " --precision single --batch-size 128 --fdd-mode cutlass_fp4 \\\n"
      << "      --noise-stddev 5 --noise-mean 0 --min-amplitude 10 --max-amplitude 10 --seed 42\n"
      << std::endl;
}

int main(int argc, char* argv[]) {
  if (argc > 1 && std::string(argv[1]) == "--help") {
    print_usage(argv[0]);
    return 0;
  }

  std::map<std::string, std::string> args;
  for (int i = 1; i < argc; i += 2) {
    if (i + 1 < argc) {
      args[argv[i]] = argv[i + 1];
    } else {
      std::cerr << "Error: Flag " << argv[i] << " is missing a value.\n";
      print_usage(argv[0]);
      return 1;
    }
  }

  typename DedispApplication<double>::SimParams params_d;

  params_d.dry_run = false;
  params_d.algorithm_type = "fdd-gemm-batched";
  params_d.fdd_compute_mode = "kernel";
  params_d.filter_type = "boxcar";
  params_d.signal_type = "burst";
  params_d.num_freq_channels = 512;
  params_d.num_time_samples = 1024;
  params_d.num_dm_trials = 256;
  params_d.f_min = 1200.0;
  params_d.f_max = 1600.0;
  params_d.min_dm_search = 0.0;
  params_d.max_dm_search = 100.0;
  params_d.total_obs_time = 0.25;
  params_d.num_candidates_to_find = 10;
  params_d.noise_mean = 0.0;
  params_d.noise_stddev = 5.0;
  params_d.min_amplitude = 10.0;
  params_d.max_amplitude = 10.0;
  params_d.print_plots = false;
  params_d.pulse_widths = {1, 2, 4, 8};
  params_d.seed = 1234;
  params_d.pulse_width_s = 0.001;
  params_d.scattering_time_s = 0.001;
  params_d.batch_size = 128;
  params_d.num_pipelines = 1;

  try {
    if (args.count("--dry-run"))
      params_d.dry_run = (args.at("--dry-run") == "yes");
    if (args.count("--algorithm"))
      params_d.algorithm_type = args.at("--algorithm");
    if (args.count("--fdd-mode"))
      params_d.fdd_compute_mode = args.at("--fdd-mode");
    if (args.count("--filter-type"))
      params_d.filter_type = args.at("--filter-type");
    if (args.count("--signal-type"))
      params_d.signal_type = args.at("--signal-type");
    if (args.count("--pulse-width-s"))
      params_d.pulse_width_s = std::stod(args.at("--pulse-width-s"));
    if (args.count("--scattering-time-s"))
      params_d.scattering_time_s = std::stod(args.at("--scattering-time-s"));
    if (args.count("--num-freq-channels"))
      params_d.num_freq_channels = std::stoi(args.at("--num-freq-channels"));
    if (args.count("--num-time-samples"))
      params_d.num_time_samples = std::stoi(args.at("--num-time-samples"));
    if (args.count("--num-dm-trials"))
      params_d.num_dm_trials = std::stoi(args.at("--num-dm-trials"));
    if (args.count("--f-min-MHz"))
      params_d.f_min = std::stod(args.at("--f-min-MHz"));
    if (args.count("--f-max-MHz"))
      params_d.f_max = std::stod(args.at("--f-max-MHz"));
    if (args.count("--min-dm-search"))
      params_d.min_dm_search = std::stod(args.at("--min-dm-search"));
    if (args.count("--max-dm-search"))
      params_d.max_dm_search = std::stod(args.at("--max-dm-search"));
    if (args.count("--total-obs-time-s"))
      params_d.total_obs_time = std::stod(args.at("--total-obs-time-s"));
    if (args.count("--num-candidates-to-find"))
      params_d.num_candidates_to_find =
          std::stoi(args.at("--num-candidates-to-find"));
    if (args.count("--noise-mean"))
      params_d.noise_mean = std::stod(args.at("--noise-mean"));
    if (args.count("--noise-stddev"))
      params_d.noise_stddev = std::stod(args.at("--noise-stddev"));
    if (args.count("--min-amplitude"))
      params_d.min_amplitude = std::stod(args.at("--min-amplitude"));
    if (args.count("--max-amplitude"))
      params_d.max_amplitude = std::stod(args.at("--max-amplitude"));
    if (args.count("--print-plots"))
      params_d.print_plots = (args.at("--print-plots") == "yes");
    if (args.count("--seed")) params_d.seed = std::stoi(args.at("--seed"));
    if (args.count("--batch-size"))
      params_d.batch_size = std::stoi(args.at("--batch-size"));
    if (args.count("--num-pipelines"))
      params_d.num_pipelines = std::stoi(args.at("--num-pipelines"));
    if (args.count("--pulse-widths")) {
      params_d.pulse_widths.clear();
      std::stringstream ss(args.at("--pulse-widths"));
      std::string width_str;
      while (std::getline(ss, width_str, ',')) {
        params_d.pulse_widths.push_back(std::stoi(width_str));
      }
    }
  } catch (const std::exception& e) {
    std::cerr << "Error parsing arguments: " << e.what() << "\n";
    print_usage(argv[0]);
    return 1;
  }

  // --- Reversibility test: early dispatch before normal pipeline ---
  if (args.count("--test-reversibility") && args.at("--test-reversibility") == "yes") {
    std::cout << "\nRunning FDD reversibility verification test...\n";
    // Force cublas mode (FP32) for mathematical verification
    typename DedispApplication<float>::SimParams params_f;
    params_f.dry_run = false;
    params_f.algorithm_type = "fdd-gemm-batched";
    params_f.fdd_compute_mode = "cublas";
    params_f.filter_type = "boxcar";
    params_f.signal_type = "burst";
    params_f.num_freq_channels = params_d.num_freq_channels;
    params_f.num_time_samples = params_d.num_time_samples;
    params_f.num_dm_trials = params_d.num_dm_trials;
    params_f.f_min = params_d.f_min;
    params_f.f_max = params_d.f_max;
    params_f.min_dm_search = params_d.min_dm_search;
    params_f.max_dm_search = params_d.max_dm_search;
    params_f.total_obs_time = params_d.total_obs_time;
    params_f.num_candidates_to_find = 1;
    params_f.noise_mean = 0.0f;
    params_f.noise_stddev = 0.0f;
    params_f.min_amplitude = 1.0f;
    params_f.max_amplitude = 1.0f;
    params_f.print_plots = false;
    params_f.pulse_widths = {1};
    params_f.seed = 1234;
    params_f.pulse_width_s = 0.001f;
    params_f.scattering_time_s = 0.001f;
    params_f.batch_size = 1;
    params_f.num_pipelines = 1;
    DedispApplication<float> app(params_f);
    app.run_reversibility_test();
    return 0;
  }

  {
    DedispApplication<double> validator(params_d);
    if (!validator.validate_parameters()) {
      return 1;
    }
  }

  std::string valid_modes[] = {"kernel", "cublas", "cublas_lt_fp16", "cublas_lt_fp8", "cutlass", "cutlass_fp6", "cutlass_fp4"};
  bool mode_found = false;
  for(const auto& m : valid_modes) {
      if(params_d.fdd_compute_mode == m) mode_found = true;
  }
  if(!mode_found) {
      std::cerr << "Error: Invalid --fdd-mode '" << params_d.fdd_compute_mode << "'.\n";
      print_usage(argv[0]);
      return 1;
  }

  if (args.count("--dry-run") && args.at("--dry-run") != "yes" &&
      args.at("--dry-run") != "no") {
    std::cerr << "Error: Invalid value '" << args.at("--dry-run")
              << "' for flag '--dry-run'. Allowed values are: yes, no\n";
    print_usage(argv[0]);
    return 1;
  }
  if (params_d.algorithm_type != "fdd-gemm-batched") {
    std::cerr << "Error: Invalid value '" << params_d.algorithm_type
	      << "' for flag '--algorithm'. Only 'fdd-gemm-batched' is supported.\n";
    print_usage(argv[0]);
    return 1;
  }
  if (params_d.filter_type != "boxcar" &&
      params_d.filter_type != "gaussian" &&
      params_d.filter_type != "matched") {
    std::cerr << "Error: Invalid value '" << params_d.filter_type
              << "' for flag '--filter-type'. Allowed values are: boxcar, "
                 "gaussian, matched\n";
    print_usage(argv[0]);
    return 1;
  }
  if (params_d.signal_type != "burst" && params_d.signal_type != "pulsar") {
    std::cerr << "Error: Invalid value '" << params_d.signal_type
              << "' for flag '--signal-type'. Allowed values are: burst, "
                 "pulsar\n";
    print_usage(argv[0]);
    return 1;
  }

  std::cout << "\n--- Simulation Parameters ---" << std::endl;
  std::cout << "Dry Run: " << (params_d.dry_run ? "Yes" : "No") << std::endl;
  std::cout << "Algorithm: " << params_d.algorithm_type << std::endl;
  if (args.count("--precision"))
    std::cout << "Precision: " << args.at("--precision") << std::endl;
  else
    std::cout << "Precision: double (default)\n";
  std::cout << "Filter Type: " << params_d.filter_type << std::endl;
  std::cout << "Signal Type: " << params_d.signal_type << std::endl;
  std::cout << "Num Freq Channels: " << params_d.num_freq_channels
            << std::endl;
  std::cout << "Num Time Samples: " << params_d.num_time_samples << std::endl;
  std::cout << "Num DM Trials: " << params_d.num_dm_trials << std::endl;
  std::cout << "Min Frequency (MHz): " << params_d.f_min << std::endl;
  std::cout << "Max Frequency (MHz): " << params_d.f_max << std::endl;
  std::cout << "Min DM Search: " << params_d.min_dm_search << std::endl;
  std::cout << "Max DM Search: " << params_d.max_dm_search << std::endl;
  std::cout << "Total Obs Time (s): " << params_d.total_obs_time << std::endl;
  std::cout << "Num Candidates to Find: " << params_d.num_candidates_to_find
            << std::endl;
  std::cout << "Batch Size: " << params_d.batch_size << std::endl;
  std::cout << "Num Pipelines: " << params_d.num_pipelines << std::endl;
  std::cout << "Noise Mean: " << params_d.noise_mean << std::endl;
  std::cout << "Noise Stddev: " << params_d.noise_stddev << std::endl;
  std::cout << "Min Amplitude: " << params_d.min_amplitude << std::endl;
  std::cout << "Max Amplitude: " << params_d.max_amplitude << std::endl;
  std::cout << "Print Plots: " << (params_d.print_plots ? "Yes" : "No")
            << std::endl;
  std::cout << "Pulse Widths to Search: ";
  for (size_t i = 0; i < params_d.pulse_widths.size(); ++i) {
    std::cout << params_d.pulse_widths[i]
              << (i == params_d.pulse_widths.size() - 1 ? "" : ",");
  }
  std::cout << "\nSeed: " << params_d.seed
            << (params_d.seed == 0 ? " (time-based)" : "") << std::endl;
  std::cout << "---------------------------\n" << std::endl;

  std::string precision_arg =
      args.count("--precision") ? args.at("--precision") : "double";

  if (params_d.fdd_compute_mode == "cublas_lt_fp16" ||
      params_d.fdd_compute_mode == "cublas_lt_fp8" ||
      params_d.fdd_compute_mode == "cutlass" ||
      params_d.fdd_compute_mode == "cutlass_fp6" ||
      params_d.fdd_compute_mode == "cutlass_fp4") {

    if (params_d.batch_size % 32 != 0) {
      std::cerr << "Error: For low-precision modes (FP16/FP8/CUTLASS), --batch-size must be a multiple of 32.\n"
		<< "       Current: " << params_d.batch_size << "\n";
      return 1;
    }

    if (precision_arg == "double") {
      std::cerr << "Error: FDD mode '" << params_d.fdd_compute_mode
		<< "' requires '--precision single'.\n";
      return 1;
    }

#ifndef HAS_CUTLASS_GEMM
    if (params_d.fdd_compute_mode == "cutlass" || params_d.fdd_compute_mode == "cutlass_fp6"
        || params_d.fdd_compute_mode == "cutlass_fp4") {
      std::cerr << "Error: CUTLASS mode requires building with -DUSE_CUTLASS_GEMM=ON.\n";
      return 1;
    }
#endif
  }

  if (precision_arg == "single") {
    std::cout << "Running with SINGLE precision (float)." << std::endl;
    typename DedispApplication<float>::SimParams params_f;
    params_f.dry_run = params_d.dry_run;
    params_f.algorithm_type = params_d.algorithm_type;
    params_f.filter_type = params_d.filter_type;
    params_f.fdd_compute_mode = params_d.fdd_compute_mode;
    params_f.signal_type = params_d.signal_type;
    params_f.num_freq_channels = params_d.num_freq_channels;
    params_f.num_time_samples = params_d.num_time_samples;
    params_f.num_dm_trials = params_d.num_dm_trials;
    params_f.f_min = params_d.f_min;
    params_f.f_max = params_d.f_max;
    params_f.min_dm_search = params_d.min_dm_search;
    params_f.max_dm_search = params_d.max_dm_search;
    params_f.total_obs_time = params_d.total_obs_time;
    params_f.num_candidates_to_find = params_d.num_candidates_to_find;
    params_f.noise_mean = params_d.noise_mean;
    params_f.noise_stddev = params_d.noise_stddev;
    params_f.min_amplitude = params_d.min_amplitude;
    params_f.max_amplitude = params_d.max_amplitude;
    params_f.print_plots = params_d.print_plots;
    params_f.pulse_widths = params_d.pulse_widths;
    params_f.seed = params_d.seed;
    params_f.pulse_width_s = params_d.pulse_width_s;
    params_f.scattering_time_s = params_d.scattering_time_s;
    params_f.batch_size = params_d.batch_size;
    params_f.num_pipelines = params_d.num_pipelines;
    DedispApplication<float> app(params_f);
    app.run();
  } else if (precision_arg == "double") {
    std::cout << "Running with DOUBLE precision (double)." << std::endl;
    DedispApplication<double> app(params_d);
    app.run();
  } else {
    std::cerr << "Error: Invalid precision '" << precision_arg
              << "'. Please use 'single' or 'double'.\n";
    print_usage(argv[0]);
    return 1;
  }

  return 0;
}
