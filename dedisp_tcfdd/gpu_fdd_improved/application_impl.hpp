// --- application_impl.hpp ---
// DedispApplication method implementations.
// Textual include inside the DedispApplication class body — no #pragma once.

  void run() {
    if (params_.dry_run) {
      this->run_dry_run();
    } else {
      this->run_batched_algorithm();
    }
  }

private:

  void setup_fdd_precomputation() {
    const Real time_resolution =
        params_.total_obs_time / params_.num_time_samples;
    const Real d_freq_MHz =
        (params_.f_max - params_.f_min) / params_.num_freq_channels;

    // --- 1. Calculate Nt_padded (Host) ---
    const Real f_min_GHz = params_.f_min / 1000.0;
    const Real f_max_GHz = params_.f_max / 1000.0;
    const Real max_delay_s = DISPERSION_CONSTANT * params_.max_dm_search * (1.0/(f_min_GHz*f_min_GHz) - 1.0/(f_max_GHz*f_max_GHz));

    const int max_delay_bins = std::ceil(max_delay_s / time_resolution);

    size_t required_size = params_.num_time_samples + max_delay_bins;

    shared_nt_padded_ = 1;
    while (shared_nt_padded_ < required_size) shared_nt_padded_ *= 2;

    if (max_delay_s > params_.total_obs_time) {
      std::cerr << "\n*** WARNING: Max Dispersion Delay (" << max_delay_s
		<< "s) exceeds Observation Time (" << params_.total_obs_time
		<< "s). ***\n"
		<< "*** High DM signals will be truncated (missing tail) "
	           "resulting in poor SNR! ***\n"
		<< "*** Increase --total-obs-time-s to fix this. ***\n" << std::endl;
    }

    std::cout << "Configuring FDD Padding:" << std::endl;
    std::cout << "  Max Dispersion Delay: " << max_delay_s << " s ("
	      << max_delay_bins << " bins)" << std::endl;
    std::cout << "  Padding Nt from " << params_.num_time_samples
	      << " to " << shared_nt_padded_ << " (Next Power of 2)" << std::endl;

    // NEW: R2C Optimization - Calculate Nt_complex
    size_t shared_nt_complex = shared_nt_padded_ / 2 + 1;

    // --- 2 & 3. Calculate Tables Directly on GPU ---
    size_t delays_size_bytes = (size_t)params_.num_dm_trials * params_.num_freq_channels * sizeof(Real);
    d_time_delays_.allocate(delays_size_bytes);

    size_t fk_size_bytes = shared_nt_complex * sizeof(Real);
    d_f_k_values_.allocate(fk_size_bytes);

    // Launch Kernel
    size_t total_work = std::max((size_t)params_.num_dm_trials * params_.num_freq_channels, (size_t)shared_nt_complex);
    dim3 block_setup(256);
    dim3 grid_setup((total_work + 255) / 256);

    kernel_generate_precomp<Real><<<grid_setup, block_setup, 0, compute_stream_>>>(
        d_time_delays_.get<Real>(),
        d_f_k_values_.get<Real>(),
        params_.num_dm_trials, params_.num_freq_channels, shared_nt_complex,
        params_.f_min, params_.f_max, params_.min_dm_search, params_.max_dm_search,
        time_resolution, DISPERSION_CONSTANT
    );
    CUDA_CHECK(cudaGetLastError());

    // --- 5. Allocate Phasor Table (REDUCED SIZE 50%) ---
    size_t phasor_size_bytes = (size_t)shared_nt_complex * params_.num_dm_trials *
                              params_.num_freq_channels * sizeof(ComplexTypeGpu);

    dev_shared_phasors_by_time_.allocateManaged(phasor_size_bytes);

    // --- CUDA 13.0.2 COMPLIANT BLOCK ---
    int device_id = 0;
    CUDA_CHECK(cudaGetDevice(&device_id));

    cudaMemLocation location;
    location.type = cudaMemLocationTypeDevice;
    location.id = device_id;

    CUDA_CHECK(cudaMemAdvise(dev_shared_phasors_by_time_.get(),
			     phasor_size_bytes,
			     cudaMemAdviseSetPreferredLocation,
			     location));

    // --- 6. Generate Phasors (REDUCED GRID) ---
    size_t total_phasors = (size_t)params_.num_freq_channels * (size_t)params_.num_dm_trials * (size_t)shared_nt_complex;

    unsigned int grid_z = shared_nt_complex;
    if (grid_z > 65535) {
       std::cerr << "Error: Nt_complex exceeds Grid Z limit. Pipeline update required." << std::endl;
       exit(1);
    }

    dim3 block(16, 16);
    dim3 grid((params_.num_freq_channels + block.x - 1) / block.x,
              (params_.num_dm_trials + block.y - 1) / block.y,
              grid_z);

    bool use_conjugate = (params_.fdd_compute_mode == "cublas" ||
                          params_.fdd_compute_mode == "cublas_lt_fp16" ||
                          params_.fdd_compute_mode == "cublas_lt_fp8" ||
                          params_.fdd_compute_mode == "cutlass" ||
                          params_.fdd_compute_mode == "cutlass_fp6" ||
                          params_.fdd_compute_mode == "cutlass_fp4");

    kernel_generate_phasors<Real, ComplexTypeGpu><<<grid, block>>>(
        dev_shared_phasors_by_time_.get<ComplexTypeGpu>(),
        d_f_k_values_.get<Real>(),
        d_time_delays_.get<Real>(),
        params_.num_freq_channels,
        params_.num_dm_trials,
        shared_nt_complex,
        use_conjugate);

    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaGetLastError());

    // --- 7. Initialize Pipeline ---
    int total_items = params_.batch_size * params_.num_pipelines;
    gpu_pipeline_ = std::make_unique<FddGpuPipeline<Real>>(total_items, params_.num_freq_channels, params_.num_time_samples,
							   params_.num_dm_trials, shared_nt_padded_, params_.fdd_compute_mode);

    // [PATCH] Pre-load phasors into pipeline persistent memory and free app copy
    if (params_.fdd_compute_mode == "cublas_lt_fp16" || params_.fdd_compute_mode == "cublas_lt_fp8"
        || params_.fdd_compute_mode == "cutlass" || params_.fdd_compute_mode == "cutlass_fp6"
        || params_.fdd_compute_mode == "cutlass_fp4") {
        gpu_pipeline_->prepare_phasors(dev_shared_phasors_by_time_.get<ComplexTypeGpu>(), 0);
        dev_shared_phasors_by_time_.free();
    }


  }

  // --- END PIPELINE CLASSES SECTION ---
  void print_perf_stats(const std::string& func_name, const PerfStats& stats) {
    const double PETA = 1e15;
    const double TERA = 1e12;
    const double GIGA = 1e9;
    const double MEGA = 1e6;
    const double KILO = 1e3;

    auto format_flops = [&](long long f) {
      if (f > PETA)
        std::cout << std::fixed << std::setprecision(2) << f / PETA << " PFLOPS";
      else if (f > TERA)
        std::cout << std::fixed << std::setprecision(2) << f / TERA << " TFLOPS";
      else if (f > GIGA)
        std::cout << std::fixed << std::setprecision(2) << f / GIGA << " GFLOPS";
      else if (f > MEGA)
        std::cout << std::fixed << std::setprecision(2) << f / MEGA << " MFLOPS";
      else if (f > KILO)
        std::cout << std::fixed << std::setprecision(2) << f / KILO << " kFLOPS";
      else
        std::cout << f << " FLOPS";
    };

    std::cout << "-- Perf [" << func_name << "] --\n"
              << "   Total FLOPS: ";
    format_flops(stats.flops);
    std::cout << "\n";

    if (stats.fft_flops > 0 && stats.core_flops > 0) {
      std::cout << "     - FFT/IFFT:    ";
      format_flops(stats.fft_flops);
      std::cout << "\n";
      std::cout << "     - Precomputation: ";
      format_flops(stats.precomp_flops);
      std::cout << "\n";
      std::cout << "     - Core Compute: ";
      format_flops(stats.core_flops);
      std::cout << "\n";
    }

    auto format_mem = [&](long long bytes) {
      if (bytes > GIGA)
        std::cout << std::fixed << std::setprecision(2) << bytes / GIGA << " GB";
      else if (bytes > MEGA)
        std::cout << std::fixed << std::setprecision(2) << bytes / MEGA << " MB";
      else if (bytes > KILO)
        std::cout << std::fixed << std::setprecision(2) << bytes / KILO << " KB";
      else
        std::cout << bytes << " B";
    };

    std::cout << "   Total Memory: ";
    format_mem(stats.total_mem_bytes);
    std::cout << "\n";
    std::cout << "   Arithmetic Intensity: " << std::fixed
              << std::setprecision(4) << stats.arithmetic_intensity
              << " FLOPS/Byte\n";
  }

  PerfStats estimate_fdd_batched_perf() const {
    PerfStats stats;
    long double total_items =
      static_cast<long double>(params_.batch_size) * params_.num_pipelines;
    long double Nf = params_.num_freq_channels;
    long double Nt = params_.num_time_samples;
    long double Ndm = params_.num_dm_trials;

    size_t Nt_padded = 1;
    while (Nt_padded < Nt) Nt_padded *= 2;

    long double flops_per_transform = 5.0L * Nt_padded * log2(Nt_padded);
    stats.fft_flops =
        static_cast<long long>(total_items * (Nf + Ndm) * flops_per_transform);

    long long time_delay_flops = Ndm * Nf * 8;
    long long fk_value_flops = Nt_padded * 2;
    long long polar_flops = Nt_padded * Ndm * Nf * 23;
    stats.precomp_flops = time_delay_flops + fk_value_flops + polar_flops;

    stats.core_flops =
        static_cast<long long>(8.0L * total_items * Ndm * Nt_padded * Nf);

    stats.flops = stats.fft_flops + stats.precomp_flops + stats.core_flops;

    long long mem_in_host = total_items * sizeof(PulsarParams<Real>);
    long long mem_out = total_items * Ndm * Nt * sizeof(Real);
    long long mem_state = 1 * (Ndm * Nf * Nt_padded * sizeof(Complex));
    long long mem_dev_in = total_items * Nf * Nt * sizeof(Real);
    long long mem_dev_params = mem_in_host;
    long long mem_transient_fft = total_items * Nf * Nt_padded * sizeof(Complex);
    long long mem_transient_fft_T = total_items * Nf * Nt_padded * sizeof(Complex);
    long long mem_transient_ifft =
        total_items * Ndm * Nt_padded * sizeof(Complex);

    stats.total_mem_bytes = mem_in_host + mem_out + mem_state + mem_dev_in +
                            mem_dev_params + mem_transient_fft +
                            mem_transient_fft_T + mem_transient_ifft;

    if(params_.fdd_compute_mode == "cublas") {
        stats.total_mem_bytes += Nt_padded * sizeof(void*) * 3;
        stats.total_mem_bytes += Nt_padded * Ndm * total_items * sizeof(Complex);
    }

    stats.arithmetic_intensity =
        (stats.total_mem_bytes > 0)
            ? (double)stats.flops / stats.total_mem_bytes
            : 0.0;

    return stats;
  }

  PerfStats estimate_find_candidates_perf() const {
    PerfStats stats;
    long long total_flops = 0;
    for (const auto width : params_.pulse_widths) {
      total_flops += 2 * params_.num_time_samples * params_.num_dm_trials;
      total_flops += params_.num_time_samples * params_.num_dm_trials;
      total_flops += 3 * params_.num_time_samples;
    }
    stats.flops = total_flops * params_.num_candidates_to_find;
    long long mem_in =
        params_.num_dm_trials * params_.num_time_samples * sizeof(Real);
    stats.total_mem_bytes = mem_in;
    stats.arithmetic_intensity =
        (stats.total_mem_bytes > 0)
            ? (double)stats.flops / stats.total_mem_bytes
            : 0.0;
    return stats;
  }

  RMatrix convolve_boxcar(const RMatrix& data, int width) {
    const size_t num_dm_trials = data.size();
    const size_t num_time_samples = data[0].size();
    RMatrix convolved_data(num_dm_trials,
                           std::vector<Real>(num_time_samples, 0.0));

    Real current_sum;
#pragma omp parallel for private(current_sum)
    for (size_t dm_idx = 0; dm_idx < num_dm_trials; ++dm_idx) {
      current_sum = 0.0;
      for (size_t t_idx = 0; t_idx < num_time_samples; ++t_idx) {
        current_sum += data[dm_idx][t_idx];
        if (t_idx >= width) {
          current_sum -= data[dm_idx][t_idx - width];
        }
        if (t_idx >= width - 1) {
          convolved_data[dm_idx][t_idx - width + 1] = current_sum;
        }
      }
    }
    return convolved_data;
  }

  RMatrix convolve_gaussian(const RMatrix& data, int width) {
    const size_t num_dm_trials = data.size();
    const size_t num_time_samples = data[0].size();
    RMatrix convolved_data(num_dm_trials,
                           std::vector<Real>(num_time_samples, 0.0));

    std::vector<Real> kernel(width);
    Real sigma = (Real)width / 4.0;
    Real sum = 0.0;
    for (int i = 0; i < width; ++i) {
      Real x = i - (width - 1.0) / 2.0;
      kernel[i] = std::exp(-0.5 * (x * x) / (sigma * sigma));
      sum += kernel[i];
    }
    for (int i = 0; i < width; ++i) {
      kernel[i] /= sum;
    }

    Real conv_sum = 0.0;
#pragma omp parallel for collapse(2) private(conv_sum)
    for (size_t dm_idx = 0; dm_idx < num_dm_trials; ++dm_idx) {
      for (size_t t_idx = 0; t_idx <= num_time_samples - width; ++t_idx) {
        conv_sum = 0.0;
        for (int k = 0; k < width; ++k) {
          conv_sum += data[dm_idx][t_idx + k] * kernel[k];
        }
        convolved_data[dm_idx][t_idx] = conv_sum;
      }
    }
    return convolved_data;
  }

  std::vector<PulseCandidate> find_pulse_candidates(
    const RMatrix& dedispersed_data, const SimParams& params,
    Real time_resolution, int num_candidates_to_find) {

    const size_t num_dm_trials = dedispersed_data.size();
    const size_t num_time_samples = dedispersed_data[0].size();

    std::vector<PulseCandidate> candidates;
    RMatrix search_data = dedispersed_data;

    for (int i = 0; i < num_candidates_to_find; ++i) {
      PulseCandidate best_candidate_this_iter;

      for (const auto width : params.pulse_widths) {
        if (width > num_time_samples) continue;

        RMatrix convolved_data;
        if (params.filter_type == "gaussian") {
          convolved_data = convolve_gaussian(search_data, width);
        } else {
          convolved_data = convolve_boxcar(search_data, width);
        }

        Real max_intensity = -1e9;
        size_t max_t_idx = 0, max_dm_idx = 0;
        for (size_t dm_idx = 0; dm_idx < num_dm_trials; ++dm_idx) {
          for (size_t t_idx = 0; t_idx < num_time_samples - width + 1;
               ++t_idx) {
            if (convolved_data[dm_idx][t_idx] > max_intensity) {
              max_intensity = convolved_data[dm_idx][t_idx];
              max_t_idx = t_idx;
              max_dm_idx = dm_idx;
            }
          }
        }

        if (max_intensity < -1e8) continue;

        std::vector<Real> off_pulse_series;
        off_pulse_series.reserve(num_time_samples);
        for (size_t t = 0; t < num_time_samples; ++t) {
          if (t < max_t_idx || t >= max_t_idx + width) {
            off_pulse_series.push_back(dedispersed_data[max_dm_idx][t]);
          }
        }

        auto stats = calculate_robust_stats(off_pulse_series);
        Real mean = stats.first;
        Real std_dev = stats.second;
        Real snr =
            (std_dev > 1e-9)
                ? (max_intensity - (width * mean)) / (std::sqrt((Real)width) * std_dev)
                : 0.0;

        if (snr > best_candidate_this_iter.snr) {
          best_candidate_this_iter.intensity = max_intensity;
          best_candidate_this_iter.dm_bin = max_dm_idx;
          best_candidate_this_iter.time_bin = max_t_idx;
          best_candidate_this_iter.dm =
              (num_dm_trials > 1)
                  ? params.min_dm_search + (static_cast<Real>(max_dm_idx) / (num_dm_trials - 1)) *
                        (params.max_dm_search - params.min_dm_search)
                  : params.min_dm_search;
          best_candidate_this_iter.time = max_t_idx * time_resolution;
          best_candidate_this_iter.snr = snr;
          best_candidate_this_iter.found_with_width = width;
        }
      }

      if (best_candidate_this_iter.snr <= 0) break;
      candidates.push_back(best_candidate_this_iter);

      Real dm_step = (params.max_dm_search - params.min_dm_search) / (params.num_dm_trials - 1);
      const Real physical_dm_mask_width = 2.0;
      int mask_radius_dm =
          static_cast<int>(ceil(physical_dm_mask_width / dm_step));
      mask_radius_dm = std::max(2, mask_radius_dm);

      int mask_radius_time = std::max(10, best_candidate_this_iter.found_with_width);

      for (int dm_offset = -mask_radius_dm; dm_offset <= mask_radius_dm;
           ++dm_offset) {
        for (int t_offset = -mask_radius_time; t_offset <= mask_radius_time;
             ++t_offset) {
          long long current_dm_bin =
              best_candidate_this_iter.dm_bin + dm_offset;
          long long current_time_bin =
              best_candidate_this_iter.time_bin + t_offset;

          if (current_dm_bin >= 0 && current_dm_bin < num_dm_trials &&
              current_time_bin >= 0 && current_time_bin < num_time_samples) {
            search_data[current_dm_bin][current_time_bin] = -1e9;
          }
        }
      }
    }
    return candidates;
  }

  void print_ascii_waterfall(const RMatrix& intensity_matrix,
                             const PulseCandidate& candidate,
                             const SimParams& params, Real time_resolution) {
    const int plot_height = 120;
    const int plot_width = 160;

    std::vector<std::string> plot(plot_height, std::string(plot_width, ' '));

    Real t_ref_s = candidate.time;
    Real f_ref_GHz = params.f_min / 1000.0;
    Real dispersion_delay_constant = DISPERSION_CONSTANT * candidate.dm;
    Real freq_step_MHz =
        (params.f_max - params.f_min) / params.num_freq_channels;

    for (int plot_y = 0; plot_y < plot_height; ++plot_y) {
      int original_freq_bin = static_cast<int>(
          ((double)plot_y / (plot_height - 1)) * (params.num_freq_channels - 1));
      Real f_current_MHz = params.f_min + original_freq_bin * freq_step_MHz;
      Real f_current_GHz = f_current_MHz / 1000.0;
      Real inv_f_sq = 1.0 / (f_current_GHz * f_current_GHz);
      Real inv_f_ref_sq = 1.0 / (f_ref_GHz * f_ref_GHz);
      Real time_at_f_current =
          t_ref_s + dispersion_delay_constant * (inv_f_sq - inv_f_ref_sq);
      int original_time_bin =
          static_cast<int>(round(time_at_f_current / time_resolution));
      original_time_bin =
          (original_time_bin % params.num_time_samples +
           params.num_time_samples) %
          params.num_time_samples;
      int plot_x = static_cast<int>(
          ((double)original_time_bin / (params.num_time_samples - 1)) *
          (plot_width - 1));

      if (plot_x >= 0 && plot_x < plot_width) {
        plot[plot_y][plot_x] = '#';
      }
    }

    std::cout << "\n--- ASCII Waterfall Plot (Scaled to " << plot_width << "x"
              << plot_height << ") ---" << std::endl;
    std::cout << "Candidate SNR: " << candidate.snr << ", DM: " << candidate.dm
              << ", Time (at f_min): " << candidate.time
              << "s, Width: " << candidate.found_with_width << " bins"
              << std::endl;
    std::cout
        << "Freq (MHz) vs. Time (s). '#' indicates the detected dispersive "
           "sweep."
        << std::endl;

    for (int i = plot_height - 1; i >= 0; --i) {
      if (i == 0 || i == plot_height - 1 || i % (plot_height / 16) == 0) {
        int original_freq_bin = static_cast<int>(
            ((double)i / (plot_height - 1)) * (params.num_freq_channels - 1));
        Real current_freq = params.f_min + original_freq_bin * freq_step_MHz;
        std::cout << std::fixed << std::setprecision(1) << std::setw(7)
                  << current_freq << " |";
        std::cout << plot[i] << "|" << std::endl;
      }
    }
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "Time -> from " << 0.0 << "s to " << params.total_obs_time
              << "s" << std::endl;
  }

  void print_verification_table(const std::vector<VerificationResult>& results) {
    std::cout << "\n=======================================================================================================\n";
    std::cout << "                                  VERIFICATION SUMMARY                                                 \n";
    std::cout << "=======================================================================================================\n";
    std::cout << " Batch | Inj DM  | Det DM  | DM Err  | Inj Time | Det Time | Time Err | Inj Amp | Det SNR | Status \n";
    std::cout << "-------|---------|---------|---------|----------|----------|----------|---------|---------|--------\n";

    std::vector<VerificationResult> sorted_res = results;
    std::sort(sorted_res.begin(), sorted_res.end(), [](const VerificationResult& a, const VerificationResult& b) {
      return a.inj_dm < b.inj_dm;
    });

    int match_count = 0;
    for (const auto& r : sorted_res) {
      std::cout << std::setw(6) << r.batch_idx << " | "
		<< std::fixed << std::setprecision(3)
		<< std::setw(7) << r.inj_dm << " | "
		<< std::setw(7) << r.det_dm << " | "
		<< (std::abs(r.dm_error) > 1.0 ? "\033[1;31m" : "")
		<< std::setw(7) << r.dm_error << "\033[0m | "
		<< std::setw(8) << r.inj_time << " | "
		<< std::setw(8) << r.det_time << " | "
		<< (std::abs(r.time_error) > 0.01 ? "\033[1;31m" : "")
		<< std::setw(8) << r.time_error << "\033[0m | "
		<< std::setw(7) << r.inj_amp << " | "
		<< std::setw(7) << r.det_snr << " | "
		<< (r.matched ? "\033[1;32m OK \033[0m" : "\033[1;31m MISS \033[0m")
		<< std::endl;
      if(r.matched) match_count++;
    }
    std::cout << "-------------------------------------------------------------------------------------------------------\n";
    std::cout << " Total Matches: " << match_count << " / " << results.size() << "\n";
    std::cout << "=======================================================================================================\n\n";
  }

void run_batched_algorithm() {
    const Real time_resolution =
        params_.total_obs_time / params_.num_time_samples;
    const int total_items = params_.batch_size * params_.num_pipelines;
    const size_t Nf = params_.num_freq_channels;
    const size_t Nt = params_.num_time_samples;

    // --- CPU-side data generation ---
    auto start_data_setup = std::chrono::high_resolution_clock::now();

    std::default_random_engine generator(
        (params_.seed == 0) ? std::time(0) : params_.seed);
    std::uniform_real_distribution<Real> dm_dist(params_.min_dm_search, params_.max_dm_search);

    std::uniform_real_distribution<Real> time_dist(
        params_.total_obs_time * 0.1, params_.total_obs_time * 0.8);

    std::uniform_real_distribution<Real> amp_dist(params_.min_amplitude,
                                                  params_.max_amplitude);

    std::vector<PulsarParams<Real>> h_pulsar_params(total_items);
    for (int i = 0; i < total_items; ++i) {
      Real pulse_dm = dm_dist(generator);
      Real pulse_time = time_dist(generator);
      Real pulse_amp = amp_dist(generator);

      if (params_.signal_type == "pulsar") {
        h_pulsar_params[i] = {pulse_dm, params_.pulse_width_s,
                              params_.scattering_time_s, pulse_amp, pulse_time};
      } else {
        h_pulsar_params[i] = {pulse_dm, 0, 0, pulse_amp, pulse_time};
      }
    }

    // Allocate and copy small params buffer to GPU
    size_t params_size_bytes = total_items * sizeof(PulsarParams<Real>);
    d_pulsar_params_.allocate(params_size_bytes);
    CUDA_CHECK(cudaMemcpy(d_pulsar_params_.get(), h_pulsar_params.data(),
                          params_size_bytes, cudaMemcpyHostToDevice));

    std::cout << "\nRunning " << params_.algorithm_type << " with "
              << params_.num_pipelines
              << " pipeline thread(s), each processing a batch of "
              << params_.batch_size << " (Total: " << total_items << ")"
              << std::endl;

    if (params_.algorithm_type == "fdd-gemm-batched") {
      std::string mode = params_.fdd_compute_mode;
      std::cout << "FDD Compute Configuration:" << std::endl;

      if (mode == "kernel") {
	std::cout << "  Backend:    Custom CUDA Kernel" << std::endl;
	std::cout << "  Precision:  FP32 (Single)" << std::endl;
      } else if (mode == "cublas") {
	std::cout << "  Backend:    cuBLAS Standard" << std::endl;
	std::cout << "  Precision:  FP32 (Single)" << std::endl;
      } else if (mode == "cublas_lt_fp16") {
	std::cout << "  Backend:    cuBLASLt (Tensor Cores)" << std::endl;
	std::cout << "  Precision:  FP16 (Half) - 4-Split Complex" << std::endl;
      } else if (mode == "cublas_lt_fp8") {
	std::cout << "  Backend:    cuBLASLt (Tensor Cores)" << std::endl;
	std::cout << "  Precision:  FP8 (E4M3) - 4-Split Complex" << std::endl;
	std::cout << "  Note:       Using fused quantization kernels." << std::endl;

	int dev = 0;
	CUDA_CHECK(cudaGetDevice(&dev));
	cudaDeviceProp prop;
	CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));

	if (params_.num_freq_channels % 32 != 0) {
	  std::cerr << "\n[FATAL ERROR] FP8/Hopper modes require 32-byte alignment (channels % 32 == 0) "
		    << "to utilize CUDA 13 vector load optimizations." << std::endl;
	  exit(EXIT_FAILURE);
	}

	if (prop.major < 9) {
	  std::cerr << "\n[FATAL ERROR] FP8 Mode requires NVIDIA Hopper Architecture (SM 9.0+)." << std::endl;
	  std::cerr << "              Detected GPU: " << prop.name << " (SM " << prop.major << "." << prop.minor << ")" << std::endl;
	  std::cerr << "              Action: Switch to --fdd-mode cublas_lt_fp16 or run on H100/H200 hardware." << std::endl;
	  exit(EXIT_FAILURE);
	} else {
	  std::cout << "\n[FOUND] FP8 Mode Capable (SM " << prop.major << "." << prop.minor << ")" << std::endl;
	}

      } else if (mode == "cutlass") {
	std::cout << "  Backend:    CUTLASS FP8 (Tensor Cores)" << std::endl;
	std::cout << "  Precision:  FP8 (E4M3) in, FP32 accum, FP32 out - 4-Split Complex" << std::endl;
      } else if (mode == "cutlass_fp6") {
	std::cout << "  Backend:    CUTLASS FP6 E3M2 (Block-Scaled Tensor Cores)" << std::endl;
	std::cout << "  Precision:  FP6 (MXFP E3M2) in, FP32 accum, FP32 out - 4-Split Complex" << std::endl;
      } else if (mode == "cutlass_fp4") {
	std::cout << "  Backend:    CUTLASS FP4 E2M1 (Block-Scaled Tensor Cores)" << std::endl;
	std::cout << "  Precision:  FP4 (MXFP E2M1) in, FP32 accum, FP32 out - 4-Split Complex" << std::endl;
	std::cout << "  Note:       FP4 is lossy (5->4, 7->8). Use for speed testing." << std::endl;
      } else {
	std::cerr << "  Warning: Unknown mode '" << mode << "'" << std::endl;
      }
    }

    // Allocate GPU buffers
    size_t input_size_bytes = (size_t)total_items * Nf * Nt * sizeof(Real);
    size_t output_size_bytes = (size_t)total_items * params_.num_dm_trials * Nt * sizeof(Real);

    std::cout << "[Memory] Allocating Input Buffer: " << (input_size_bytes/1e9) << " GB" << std::endl;
    check_memory_availability(input_size_bytes);
    dev_full_intensity_batch.allocate(input_size_bytes);

    std::cout << "[Memory] Allocating Output Buffer: " << (output_size_bytes/1e9) << " GB" << std::endl;
    check_memory_availability(output_size_bytes);
    dev_dedispersed_full_batch.allocate(output_size_bytes);

    // 1. Generate Noise
    if (params_.noise_stddev > 1e-9) {
      CURAND_CHECK(curandSetStream(rand_gen_, compute_stream_));

      size_t total_samples = (size_t)total_items * Nf * Nt;

      const size_t CHUNK_SIZE = 32 * 1024 * 1024;

      for (size_t offset = 0; offset < total_samples; offset += CHUNK_SIZE) {
          size_t current_chunk = std::min(CHUNK_SIZE, total_samples - offset);

          if (current_chunk % 4 != 0) {
              current_chunk -= (current_chunk % 4);
          }

          if (current_chunk == 0) continue;

          if constexpr (std::is_same_v<Real, float>) {
            CURAND_CHECK(curandGenerateNormal(
                rand_gen_,
                dev_full_intensity_batch.get<float>() + offset,
                current_chunk,
                params_.noise_mean, params_.noise_stddev));
          } else {
             if (current_chunk % 2 != 0) current_chunk -= 1;

             CURAND_CHECK(curandGenerateNormalDouble(
                rand_gen_,
                dev_full_intensity_batch.get<double>() + offset,
                current_chunk,
                params_.noise_mean, params_.noise_stddev));
          }
      }
    } else {
      CUDA_CHECK(cudaMemsetAsync(dev_full_intensity_batch.get(), 0,
                                 input_size_bytes, compute_stream_));
    }

    // 2. Inject Signals
    if (params_.signal_type == "burst") {
      const Real f_ref_GHz = params_.f_max / 1000.0;
      dim3 grid_inject((total_items * Nf + 255) / 256);
      dim3 block_inject(256);
      kernel_inject_bursts<Real><<<grid_inject, block_inject>>>(
          dev_full_intensity_batch.get<Real>(),
          d_pulsar_params_.get<PulsarParams<Real>>(),
          total_items, Nf, Nt, params_.f_min, params_.f_max,
          time_resolution, f_ref_GHz);
      CUDA_CHECK(cudaGetLastError());
    } else {
      fprintf(stderr, "Warning: GPU-side signal injection for 'pulsar' mode not impl.\n");
    }

    // 3. DSA-100 Digitization Simulation: Quantize to 8-bit [0, 255]
    {
        dim3 grid_q((total_items * Nf * Nt + 255) / 256);
        dim3 block_q(256);
        kernel_quantize_u8<Real><<<grid_q, block_q, 0, compute_stream_>>>(
            dev_full_intensity_batch.get<Real>(), total_items * Nf * Nt);
        CUDA_CHECK(cudaGetLastError());
    }


    auto end_data_setup = std::chrono::high_resolution_clock::now();
    auto data_setup_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                         end_data_setup - start_data_setup)
                         .count();
    std::cout << "GPU Data Generation (Params + Noise + Inject) took "
              << data_setup_ms << " ms." << std::endl;

    // Only allocate if strictly needed for debug plotting (slow path)
    RBatchMatrix full_intensity_batch;
    if (params_.print_plots) {
       std::cout << "[INFO] Allocating Host buffers for plotting..." << std::endl;
       full_intensity_batch.resize(total_items, RMatrix(Nf, std::vector<Real>(Nt)));
    }

    std::vector<VerificationResult> summary_metrics;

    if (params_.algorithm_type == "fdd-gemm-batched") {
      auto start_setup = std::chrono::high_resolution_clock::now();
      print_perf_stats("fdd-gemm-batched", this->estimate_fdd_batched_perf());
      setup_fdd_precomputation();
      auto end_setup = std::chrono::high_resolution_clock::now();

      auto start_exec = std::chrono::high_resolution_clock::now();

      gpu_pipeline_->execute(dev_full_intensity_batch.get<Real>(),
                             dev_shared_phasors_by_time_.get<ComplexTypeGpu>(),
                             dev_dedispersed_full_batch.get<Real>(),
                             total_items, compute_stream_);

      auto end_exec = std::chrono::high_resolution_clock::now();
      auto setup_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                          end_setup - start_setup)
                          .count();
      auto exec_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                         end_exec - start_exec)
                         .count();
      std::cout << "Pre-computation took " << setup_ms
                << " ms. Total Execution took " << exec_ms
                << " ms. Per item took " << (double)exec_ms / total_items << " ms."
                << std::endl;

      // --- GPU SEARCH LOOP ---
      size_t search_res_count = total_items * params_.num_dm_trials;
      d_search_results_.allocate(search_res_count * sizeof(DeviceCandidate));

      CUDA_CHECK(cudaMemset(d_search_results_.get(), 0, search_res_count * sizeof(DeviceCandidate)));

      dim3 block_rst(256);
      dim3 grid_rst((search_res_count + 255) / 256);
      kernel_reset_candidates<<<grid_rst, block_rst, 0, compute_stream_>>>(
          d_search_results_.get<DeviceCandidate>(), search_res_count);

      dim3 search_block(256);
      dim3 search_grid(total_items, params_.num_dm_trials);

      for (int width : params_.pulse_widths) {
          if (width > Nt) continue;

          kernel_find_best_candidate<Real><<<search_grid, search_block, 0, compute_stream_>>>(
              dev_dedispersed_full_batch.get<Real>(),
              d_search_results_.get<DeviceCandidate>(),
              params_.num_dm_trials, Nt,
              width,
              (float)params_.noise_mean, (float)params_.noise_stddev
          );
      }
      CUDA_CHECK(cudaGetLastError());

      // Copy Results Back
      std::vector<char> raw_host_buffer(search_res_count * sizeof(DeviceCandidate));

      CUDA_CHECK(cudaMemcpy(raw_host_buffer.data(), d_search_results_.get(),
                            search_res_count * sizeof(DeviceCandidate),
                            cudaMemcpyDeviceToHost));

      const DeviceCandidate* host_candidates = reinterpret_cast<DeviceCandidate*>(raw_host_buffer.data());

      // Final CPU Reduction (Batch Level)
      for(int i=0; i<total_items; ++i) {
          int best_dm_idx = -1;
          float best_snr = -1.0f;
          float best_time_val = 0.0f;

          for(int dm=0; dm < params_.num_dm_trials; ++dm) {
              int idx = i * params_.num_dm_trials + dm;
              if (host_candidates[idx].max_snr > best_snr) {
                  best_snr = host_candidates[idx].max_snr;
                  best_dm_idx = dm;
                  best_time_val = host_candidates[idx].time_idx * time_resolution;
              }
          }

          VerificationResult res;
          res.batch_idx = i;
          const PulsarParams<Real>& inj = h_pulsar_params[i];
          res.inj_dm = inj.dm;
          res.inj_time = inj.pulse_start_time;
          res.inj_amp = inj.amplitude;

          if (best_dm_idx == -1) {
              res.det_dm = -1; res.matched = false;
          } else {
              res.det_dm = (params_.num_dm_trials > 1)
                  ? params_.min_dm_search + (static_cast<Real>(best_dm_idx) / (params_.num_dm_trials - 1)) * (params_.max_dm_search - params_.min_dm_search)
                  : params_.min_dm_search;
              res.det_time = best_time_val;
              res.det_snr = best_snr;
              res.dm_error = res.det_dm - inj.dm;
              res.time_error = res.det_time - inj.pulse_start_time;

              bool dm_ok = std::abs(res.dm_error) < 5.0;
              bool time_ok = std::abs(res.time_error) < 0.05;
              res.matched = (dm_ok && time_ok && res.det_snr > 5.0);
          }
          summary_metrics.push_back(res);
      }
    }

    CUDA_CHECK(cudaStreamSynchronize(compute_stream_));

    // --- PRINT FINAL SUMMARY ---
    print_verification_table(summary_metrics);
}

  void run_dry_run() {
    std::cout << "\n--- Dry Run: Performance Estimation Only ---\n";
    std::cout << "Selected Algorithm: " << params_.algorithm_type << "\n\n";

    if (params_.algorithm_type == "fdd-gemm-batched") {
      print_perf_stats("fdd-gemm-batched", this->estimate_fdd_batched_perf());
    } else {
      std::cerr << "Error: Unknown algorithm type '" << params_.algorithm_type
		<< "' in dry run."
		<< "\n";
    }

    print_perf_stats("find_pulse_candidates",
		     this->estimate_find_candidates_perf());
  }

  std::pair<Real, Real> calculate_robust_stats(
      const std::vector<Real>& series, int iterations = 3,
      Real sigma_threshold = 3.0) const {
    std::vector<Real> clipped_series = series;
    if (clipped_series.empty()) {
      return {0.0, 0.0};
    }

    Real mean = 0.0;
    Real std_dev = 0.0;

    for (int i = 0; i < iterations; ++i) {
      if (clipped_series.empty()) {
        break;
      }
      Real sum = std::accumulate(clipped_series.begin(), clipped_series.end(),
                                 (Real)0.0);
      mean = sum / clipped_series.size();
      Real sq_sum =
          std::inner_product(clipped_series.begin(), clipped_series.end(),
                             clipped_series.begin(), (Real)0.0);
      std_dev = std::sqrt(sq_sum / clipped_series.size() - mean * mean);

      if (std_dev < 1e-9) {
        break;
      }

      std::vector<Real> next_clipped_series;
      Real lower_bound = mean - sigma_threshold * std_dev;
      Real upper_bound = mean + sigma_threshold * std_dev;

      for (const auto& val : clipped_series) {
        if (val >= lower_bound && val <= upper_bound) {
          next_clipped_series.push_back(val);
        }
      }
      clipped_series = next_clipped_series;
    }

    return {mean, std_dev};
  }
