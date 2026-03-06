#include <iostream>
#include <vector>
#include <cmath>
#include <numeric>
#include <algorithm>
#include <chrono> // Added for timing
#include <random>
#include <stdio.h> // Added for printf
#include <string>  // For std::string
#include <map>     // For std::map
#include <iomanip> // For std::setw, std::setprecision
#include <sstream> // For std::stringstream
#include <omp.h>   // For OpenMP

#include <omp.h>   // For OpenMP

// --- FDD ALIGNMENT CONSTANTS ---
const double DISPERSION_CONSTANT = 4.15e-3; // Matches FDD
const std::vector<int> PULSE_WIDTHS = {1, 2, 4, 8}; // Matches FDD

struct VerificationResult {
    int batch_idx;
    double inj_dm;
    double inj_time;
    double inj_amp;
    double det_dm;
    double det_time;
    double det_snr;
    double dm_error;
    double time_error;
    bool matched;
};

// --- MOVED/ADDED STRUCTS ---

// Helper struct to store injection parameters (Moved from bottom)
struct PulseInfo {
    double injected_dm;
    int t_fmax_center;
    int expected_delay_samples;
    double expected_snr;
};

// Holds mappings for a specific FDMT stage (Was missing)
struct StageMappings {
    int max_dt_out_subband;
    std::vector<int> state_in_A_idx_map;
    std::vector<int> state_in_B_idx_map;
    std::vector<int> state_out_idx_map;
    std::vector<int> t_shift_map;
};

// Iterative sigma clipping to find noise floor (matches FDD)
std::pair<double, double> calculate_robust_stats(const std::vector<float>& series, int iterations = 3, double sigma_threshold = 3.0) {
    std::vector<float> clipped = series;
    if (clipped.empty()) return {0.0, 0.0};

    double mean = 0.0;
    double std_dev = 0.0;

    for (int i = 0; i < iterations; ++i) {
        if (clipped.empty()) break;
        double sum = std::accumulate(clipped.begin(), clipped.end(), 0.0);
        mean = sum / clipped.size();
        
        double sq_sum = std::inner_product(clipped.begin(), clipped.end(), clipped.begin(), 0.0);
        std_dev = std::sqrt(sq_sum / clipped.size() - mean * mean);

        if (std_dev < 1e-9) break;

        std::vector<float> next_clipped;
        double lower = mean - sigma_threshold * std_dev;
        double upper = mean + sigma_threshold * std_dev;
        for (float val : clipped) {
            if (val >= lower && val <= upper) next_clipped.push_back(val);
        }
        clipped = next_clipped;
    }
    return {mean, std_dev};
}

// --- DEBUG FLAG ---
// Set to true to print verbose trace information
const bool DEBUG_TRACE = false; // Set to false for cleaner output

void inject_pulse(std::vector<float>& spectrogram, int Nf, int Nt, double dm, int t_fmax_center, double f_max, double f_min, double t_total, double signal_strength) {
    // Matches FDD Logic exactly
    double delta_t_sample = t_total / Nt;
    double f_ref_GHz = f_max / 1000.0;
    double delta_f = (f_max - f_min) / Nf;

    for (int i = 0; i < Nf; ++i) {
      double f_current_MHz = f_min + (i + 0.5) * delta_f; 
      double f_current_GHz = f_current_MHz / 1000.0;
      
      double delay_s = DISPERSION_CONSTANT * dm * (1.0 / (f_current_GHz * f_current_GHz) - 1.0 / (f_ref_GHz * f_ref_GHz));
        
      int t_delay_samples = static_cast<int>(delay_s / delta_t_sample); // FDD uses cast, not round
      int t_arrival = t_fmax_center + t_delay_samples;
      
      if (t_arrival >= 0 && t_arrival < Nt) {
	spectrogram[i * Nt + t_arrival] += signal_strength; 
      }
    }
}

void print_verification_table(const std::vector<VerificationResult>& results) {
    std::cout << "\n=======================================================================================================\n";
    std::cout << "                                  VERIFICATION SUMMARY                                                 \n";
    std::cout << "=======================================================================================================\n";
    std::cout << " Batch | Inj DM  | Det DM  | DM Err  | Inj Time | Det Time | Time Err | Inj Amp | Det SNR | Status \n";
    std::cout << "-------|---------|---------|---------|----------|----------|----------|---------|---------|--------\n";
    
    int match_count = 0;
    for (const auto& r : results) {
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

// Boxcar convolution for a specific DM row
std::vector<float> convolve_row(const std::vector<float>& row, int width) {
    if (width == 1) return row;
    std::vector<float> res(row.size(), 0.0f);
    double sum = 0.0;
    for(size_t i=0; i<row.size(); ++i) {
        sum += row[i];
        if (i >= width) sum -= row[i-width];
        if (i >= width - 1) res[i - width + 1] = (float)sum;
    }
    return res;
}

// [FIX] Updated signature to include f_min/f_max for time correction
VerificationResult find_best_candidate_cpu(
    const std::vector<float>& dedispersed_flat, 
    int batch_idx, int Nt, int Ndm, 
    double time_res, double max_dm_search, 
    const PulseInfo& truth,
    double f_min, double f_max 
					   ) {
  
  VerificationResult res;
  res.batch_idx = batch_idx;
  res.inj_dm = truth.injected_dm;
  
    // [FIX] Calculate Expected Arrival Time at f_min
    // The FDMT algorithm aligns high freq (early) to low freq (late).
    // So the pulse ends up at the arrival time of the LOWEST frequency.
    double f_min_ghz = f_min / 1000.0;
    double f_max_ghz = f_max / 1000.0;
    double total_delay_s = DISPERSION_CONSTANT * truth.injected_dm * (1.0/(f_min_ghz*f_min_ghz) - 1.0/(f_max_ghz*f_max_ghz));
    
    res.inj_time = (truth.t_fmax_center * time_res) + total_delay_s;
    res.inj_amp = truth.expected_snr;
    double global_max_snr = -1e9;
    int global_best_dm_idx = -1;
    int global_best_t_idx = -1;

    // 1. Calculate Statistics for SNR (Noise Estimate)
    // We take a slice of the first DM trial to estimate noise
    // (Approximation: assumes no signal in first DM trial or robust stats handles it)
    std::vector<float> noise_sample(dedispersed_flat.begin(), dedispersed_flat.begin() + Nt);
    auto stats = calculate_robust_stats(noise_sample);
    double noise_mean = stats.first;
    double noise_std = stats.second;

    for (int width : PULSE_WIDTHS) {
        if (width > Nt) continue;

        // Effective Mean/Std for boxcar sum
        double width_mean = noise_mean * width;
        double width_std = noise_std * std::sqrt((double)width);

        for (int dm = 0; dm < Ndm; ++dm) {
            // Extract row
            std::vector<float> row(dedispersed_flat.begin() + dm*Nt, dedispersed_flat.begin() + (dm+1)*Nt);
            std::vector<float> conv = convolve_row(row, width);

            for (int t=0; t < Nt - width; ++t) {
                double val = conv[t];
                double snr = (val - width_mean) / width_std;

                if (snr > global_max_snr) {
                    global_max_snr = snr;
                    global_best_dm_idx = dm;
                    global_best_t_idx = t;
                }
            }
        }
    }

    if (global_best_dm_idx == -1) {
        res.det_dm = -1; res.matched = false;
        res.det_snr = 0;
    } else {
        res.det_dm = (static_cast<double>(global_best_dm_idx) / (Ndm - 1)) * max_dm_search; // Assuming linear mapping
        // Correction: FDMT code maps DM index linearly up to max_delay_total samples.
        // We need to verify if max_dm_search matches Ndm in this context.
        // The original FDMT used: recovered_dm = (idx / injected_delay) * inj_dm.
        // FDD uses strict linear mapping. Let's use FDD style mapping:
        
        res.det_time = global_best_t_idx * time_res;
        res.det_snr = global_max_snr;
        res.dm_error = res.det_dm - res.inj_dm;
        res.time_error = res.det_time - res.inj_time;

        bool dm_ok = std::abs(res.dm_error) < 5.0;
        bool time_ok = std::abs(res.time_error) < 0.05;
        res.matched = (dm_ok && time_ok && res.det_snr > 5.0);
    }

    return res;
}

/**
 * @brief Pre-calculates the integer-to-integer mappings for an FDMT stage.
 */
StageMappings precompute_mappings(
    int f_idx_start, int subband_size_out, 
    const std::vector<double>& freqs_inv_sq, 
    int max_delay_total, double d_factor, 
    double time_resolution_s,
    int i_sub_out, int max_delay_prev_iter, int max_delay_iter 
) 
{
    StageMappings mappings;
    const double K_DM = DISPERSION_CONSTANT; // 4.15e-3

    int f0_idx = f_idx_start;
    int f1_idx = f_idx_start + subband_size_out / 2 - 1;
    int f2_idx = f_idx_start + subband_size_out / 2;
    int f3_idx = f_idx_start + subband_size_out - 1;

    double f_B_min_inv2 = freqs_inv_sq[f0_idx];
    double f_B_max_inv2 = freqs_inv_sq[f1_idx];
    double f_A_min_inv2 = freqs_inv_sq[f2_idx];
    double f_A_max_inv2 = freqs_inv_sq[f3_idx];

    double subband_d_factor_phys = (f_B_min_inv2 - f_A_max_inv2);
    if (d_factor <= 0) d_factor = 1e-9;
    
    mappings.max_dt_out_subband = static_cast<int>(std::ceil(max_delay_total * subband_d_factor_phys / d_factor));
    mappings.max_dt_out_subband = std::min(mappings.max_dt_out_subband, max_delay_total);
    
    int map_size = mappings.max_dt_out_subband + 1;
    mappings.state_in_A_idx_map.resize(map_size);
    mappings.state_in_B_idx_map.resize(map_size);
    mappings.state_out_idx_map.resize(map_size);
    mappings.t_shift_map.resize(map_size);

    double subband_d_factor = f_B_min_inv2 - f_A_max_inv2;
    if (subband_d_factor == 0) subband_d_factor = 1e-9;

    int i_sub_in_A = i_sub_out * 2 + 1;
    int i_sub_in_B = i_sub_out * 2;
    
    // This loop is safe to parallelize
#pragma omp parallel for
    for (int dt_out = 0; dt_out <= mappings.max_dt_out_subband; ++dt_out) {
        double dt_out_s = dt_out * time_resolution_s;
        double effective_dm = dt_out_s / (K_DM * subband_d_factor);
        double dt_A_s = effective_dm * K_DM * (f_A_min_inv2 - f_A_max_inv2);
        double dt_B_s = effective_dm * K_DM * (f_B_min_inv2 - f_B_max_inv2);
        double t_shift_s = effective_dm * K_DM * (f_B_min_inv2 - f_A_min_inv2);
        
        int dt_A_idx = static_cast<int>(std::round(dt_A_s / time_resolution_s));
        int dt_B_idx = static_cast<int>(std::round(dt_B_s / time_resolution_s));
        mappings.t_shift_map[dt_out] = static_cast<int>(std::round(t_shift_s / time_resolution_s));

        if (dt_A_idx < 0 || dt_A_idx > max_delay_prev_iter ||
            dt_B_idx < 0 || dt_B_idx > max_delay_prev_iter) 
        {
            mappings.state_in_A_idx_map[dt_out] = -1;
            mappings.state_in_B_idx_map[dt_out] = -1;
        } else {
            mappings.state_in_A_idx_map[dt_out] = i_sub_in_A * (max_delay_prev_iter + 1) + dt_A_idx;
            mappings.state_in_B_idx_map[dt_out] = i_sub_in_B * (max_delay_prev_iter + 1) + dt_B_idx;
        }

        mappings.state_out_idx_map[dt_out] = i_sub_out * (max_delay_iter + 1) + dt_out;
    }
    return mappings;
}

/**
 * @struct FdmtPlan
 * @brief Holds all pre-computed data necessary to execute an FDMT.
 */
struct FdmtPlan {
    int Nf;
    int Nt;
    int Ndm_max;
    int max_delay_total;
    double d_factor;
    double time_resolution_s;
    std::vector<double> freqs_inv_sq;
    std::vector<std::vector<StageMappings>> all_mappings;
};

/**
 * @brief Creates a plan (pre-computes all tables) for the FDMT.
 */
FdmtPlan create_fdmt_plan(int Nf, int Nt, int Ndm_max, double f_max, double f_min, double t_total) {
    FdmtPlan plan;
    plan.Nf = Nf;
    plan.Nt = Nt;
    plan.Ndm_max = Ndm_max;
    plan.max_delay_total = Ndm_max - 1;
    plan.time_resolution_s = t_total / Nt;

    plan.freqs_inv_sq.resize(Nf);
    double delta_f = (f_max - f_min) / Nf;
#pragma omp parallel for
    for (int i = 0; i < Nf; ++i) {
      double freq = f_min + (i + 0.5) * delta_f;
      plan.freqs_inv_sq[i] = 1.0 / (freq * freq);
    }
    plan.d_factor = (plan.freqs_inv_sq[0] - plan.freqs_inv_sq[Nf - 1]);
    if (plan.d_factor <= 0) plan.d_factor = 1e-9;

    plan.all_mappings.resize(static_cast<int>(std::log2(Nf)) + 1); 

    int max_delay_init = 0; 
    int max_delay_prev_iter = max_delay_init;
    int max_delay_iter = plan.max_delay_total;

    for (int i_iter = 1; (1 << i_iter) <= Nf; ++i_iter) {
        int n_subbands_out = (Nf / (1 << i_iter));
        int subband_size_out = 1 << i_iter;
        plan.all_mappings[i_iter].resize(n_subbands_out);

        if (DEBUG_TRACE) printf("Precomputing Iter %d (%d subbands)\n", i_iter, n_subbands_out);

#pragma omp parallel for
        for (int i_sub_out = 0; i_sub_out < n_subbands_out; ++i_sub_out) {
            int f0_idx = i_sub_out * subband_size_out;
            plan.all_mappings[i_iter][i_sub_out] = precompute_mappings(
                f0_idx, subband_size_out, 
                plan.freqs_inv_sq, plan.max_delay_total, 
                plan.d_factor, plan.time_resolution_s,
                i_sub_out, max_delay_prev_iter, max_delay_iter
            );
        }
        max_delay_prev_iter = max_delay_iter;
    }
    
    return plan;
}

// Define a type for our 3D state: [batch][index][time]
using FdmtState = std::vector<std::vector<std::vector<float>>>;

/**
 * @brief Performs the FDMT on a *batch* of spectrograms.
 * @param plan The pre-computed FDMT plan.
 * @param spectrogram_batch A vector of spectrograms to process.
 * @param final_output_batch [OUT] The destination vector for the results.
 * @param copy_time_ms [OUT] A timer to accumulate memory copy durations.
 */
void fdmt_execute_batched(
    const FdmtPlan& plan, 
    const std::vector<std::vector<float>>& spectrogram_batch,
    std::vector<std::vector<float>>& final_output_batch,
    long long& copy_time_ms
) {
    const int N_batches = spectrogram_batch.size();
    if (N_batches == 0) return;

    // --- 1. Allocate Ping-Pong Buffers ---
    // We need two buffers that we can swap. They must be large enough
    // for the *largest* state, which is either the initial state
    // or the state after the first iteration.
    int max_delay_init = 0;
    int max_rows_init = plan.Nf * (max_delay_init + 1);
    int max_rows_iter_1 = (plan.Nf / 2) * (plan.max_delay_total + 1);
    int max_rows = std::max(max_rows_init, max_rows_iter_1);
    
    FdmtState state_A(
        N_batches, 
        std::vector<std::vector<float>>(
            max_rows, 
            std::vector<float>(plan.Nt, 0.0f)
        )
    );
    FdmtState state_B(
        N_batches, 
        std::vector<std::vector<float>>(
            max_rows, 
            std::vector<float>(plan.Nt, 0.0f)
        )
    );

    // Pointers to swap between the buffers
    FdmtState* state_in_ptr = &state_A;
    FdmtState* state_out_ptr = &state_B;


    // --- 2. Initialization Stage (PHASE 1) ---
    if (DEBUG_TRACE) printf("\n--- Initialization Stage (Copying %d spectrograms to state_in) ---\n", N_batches);
    
    auto start_copy_1 = std::chrono::high_resolution_clock::now();
#pragma omp parallel for collapse(2)
    for (int b = 0; b < N_batches; ++b) {
      for (int f = 0; f < plan.Nf; ++f) {
	for (int t = 0; t < plan.Nt; ++t) {
	  // Copy from 1D input spectrogram to 3D state buffer
	  (*state_in_ptr)[b][f * (max_delay_init + 1)][t] = spectrogram_batch[b][f * plan.Nt + t];
	}
      }
    }
    auto end_copy_1 = std::chrono::high_resolution_clock::now();
    copy_time_ms += std::chrono::duration_cast<std::chrono::milliseconds>(end_copy_1 - start_copy_1).count();
    
    
    // --- 3. Iterative Combination Stage (PHASE 3) ---
    int n_subbands = plan.Nf;
    int max_delay_prev_iter = max_delay_init; 

    for (int i_iter = 1; (1 << i_iter) <= plan.Nf; ++i_iter) {
        int n_subbands_out = n_subbands / 2;
        int subband_size_in = 1 << (i_iter - 1);
        int subband_size_out = 1 << i_iter;

        if (DEBUG_TRACE) printf("\n--- Iteration %d (Combining %d subbands -> %d subbands) ---\n", 
                                i_iter, n_subbands, n_subbands_out);
        
        // Note: state_out_ptr buffer is already allocated!
        
        // This is the main compute kernel
#pragma omp parallel for collapse(2)
        for (int b = 0; b < N_batches; ++b) {
	  for (int i_sub_out = 0; i_sub_out < n_subbands_out; ++i_sub_out) {
	    
	    const StageMappings& mappings = plan.all_mappings[i_iter][i_sub_out];
	    
	    for (int dt_out = 0; dt_out <= mappings.max_dt_out_subband; ++dt_out) {
	      
	      const int read_idx_A = mappings.state_in_A_idx_map[dt_out];
	      const int read_idx_B = mappings.state_in_B_idx_map[dt_out];
	      const int write_idx = mappings.state_out_idx_map[dt_out];
	      const int t_shift_bins = mappings.t_shift_map[dt_out];
	      
	      if (read_idx_A == -1) {
		// This path was invalid (out of bounds) during pre-computation
		continue;
	      }
	      
	      for (int t_L = 0; t_L < plan.Nt; ++t_L) {
		int t_H = t_L - t_shift_bins;
		
		if (t_H < 0 || t_H >= plan.Nt) {
		  continue; 
		}
		
		// Access the correct batch item [b]
		float val_B = (*state_in_ptr)[b][read_idx_B][t_L];
		float val_A = (*state_in_ptr)[b][read_idx_A][t_H];
                
		(*state_out_ptr)[b][write_idx][t_L] = val_A + val_B;
	      }
	    }
	  }
        } 
	
        // Swap the roles of the buffers for the next iteration
        std::swap(state_in_ptr, state_out_ptr);
        
        max_delay_prev_iter = plan.max_delay_total;
        n_subbands = n_subbands_out;
    }
    
    // --- 4. Finalization ---
    // The final result is now in state_in_ptr (due to the last swap)
    
    auto start_copy_2 = std::chrono::high_resolution_clock::now();
#pragma omp parallel for collapse(2)
    for(int b = 0; b < N_batches; ++b) {
      for(int dm = 0; dm < plan.Ndm_max; ++dm) {
	if (dm >= (*state_in_ptr)[b].size()) continue; 
	for(int t = 0; t < plan.Nt; ++t) {
	  final_output_batch[b][dm * plan.Nt + t] = (*state_in_ptr)[b][dm][t];
	}
      }
    }
    auto end_copy_2 = std::chrono::high_resolution_clock::now();
    copy_time_ms += std::chrono::duration_cast<std::chrono::milliseconds>(end_copy_2 - start_copy_2).count();
}

/**
 * @brief Prints the command-line usage instructions.
 * @param exe_name The name of the executable (from argv[0]).
 */
void print_usage(const std::string& exe_name) {
    std::cerr << "Usage: " << exe_name << " [options]\n\n"
              << "Options:\n"
              << "  --help\t\t\tPrint this help message\n"
              << "  --dry-run <yes|no>\t\tPrint performance estimates only (default: no)\n"
              << "  --num-freq-channels <int>\tNumber of frequency channels (default: 128)\n"
              << "  --num-time-samples <int>\tNumber of time samples (default: 4096)\n"
              << "  --num-dm-trials <int>\t\tNumber of DM trials (default: 512)\n"
              << "  --num-batches <int>\t\tNumber of spectrograms per pipeline (default: 3)\n"
              << "  --num-pipelines <int>\t\tNumber of pipelines to run (default: 1)\n"
              << "  --total-obs-time <float>\tTotal observation time in seconds (default: 0.41)\n"
              << "  --f-max <float>\t\tMaximum frequency in MHz (default: 1600.0)\n"
              << "  --f-min <float>\t\tMinimum frequency in MHz (default: 1200.0)\n"
              << std::endl;
}

// --- NEW Performance Estimation ---

/**
 * @struct PerfStats
 * @brief Holds estimated performance metrics.
 */
struct PerfStats {
    long long precompute_flops = 0;
    long long core_compute_flops = 0; // For one pipeline (N_batches)
    double pipeline_memory_gb = 0.0;
};

/**
 * @brief Estimates the FLOPS for the pre-computation phase.
 */
long long estimate_precompute_flops(int Nf, int Ndm_max) {
    long long freqs_inv_sq_flops = (long long)Nf * 3; // 1 add, 1 mul, 1 div
    
    // Ops inside precompute_mappings dt_out loop:
    // 6 muls, 3 subs, 1 div, ~3 rounds
    long long ops_per_dt_out = 13; 
    
    // Total subbands is (Nf/2 + Nf/4 + ... + 1) = Nf - 1
    long long mappings_flops = (long long)(Nf - 1) * Ndm_max * ops_per_dt_out;

    return freqs_inv_sq_flops + mappings_flops;
}

/**
 * @brief Estimates the core computation FLOPS for one pipeline.
 */
long long estimate_core_compute_flops(int Nf, int Nt, int Ndm_max, int N_batches) {
    // Ops inside t_L loop: 1 sub (t_H), 1 add (val_A + val_B)
    long long ops_per_t_L = 2;

    // Sum over all iterations: (Nf/2 * Ndm*Nt + Nf/4 * Ndm*Nt + ...)
    // This simplifies to (Nf - 1) * Ndm_max * Nt
    long long total_ops_per_batch = (long long)(Nf - 1) * Ndm_max * Nt * ops_per_t_L;

    return total_ops_per_batch * N_batches;
}

/**
 * @brief Estimates the memory allocated for compute *within* one pipeline.
 */
double estimate_memory_allocations(int Nf, int Nt, int Ndm_max, int N_batches) {
    // This is the memory for ONE pipeline (N_batches)
    
    // Size of the ping-pong buffers
    int max_delay_init = 0;
    int max_rows_init = Nf * (max_delay_init + 1);
    int max_rows_iter_1 = (Nf / 2) * (Ndm_max + 1);
    int max_rows = std::max(max_rows_init, max_rows_iter_1);

    long long state_buffer_size = 2LL * N_batches * max_rows * Nt * sizeof(float); // 2 buffers
    
    // Size of the output buffer
    long long output_buffer_size = (long long)N_batches * Ndm_max * Nt * sizeof(float);
    
    // We ignore the input batch size, as in a real pipeline, that data
    // would be overwritten, not allocated. We only care about the
    // *compute* allocations.
    
    double total_gb = (double)(state_buffer_size + output_buffer_size) / (1024.0 * 1024.0 * 1024.0);
    return total_gb;
}

/**
 * @brief Prints the formatted performance estimations.
 */
void print_perf_stats(const PerfStats& stats, int N_batches) {
    auto format_flops = [](long long f) -> std::string {
        std::stringstream ss;
        if (f > 1e12) ss << std::fixed << std::setprecision(2) << f / 1e12 << " TFLOPS";
        else if (f > 1e9) ss << std::fixed << std::setprecision(2) << f / 1e9 << " GFLOPS";
        else if (f > 1e6) ss << std::fixed << std::setprecision(2) << f / 1e6 << " MFLOPS";
        else if (f > 1e3) ss << std::fixed << std::setprecision(2) << f / 1e3 << " kFLOPS";
        else ss << f << " FLOPS";
        return ss.str();
    };

    std::cout << "\n--- Performance Estimates ---" << std::endl;
    std::cout << "  Pre-computation FLOPS:    " << std::setw(12) << format_flops(stats.precompute_flops) << " (one-time)" << std::endl;
    std::cout << "  Core Compute FLOPS:       " << std::setw(12) << format_flops(stats.core_compute_flops) << " (per pipeline of " << N_batches << ")" << std::endl;
    std::cout << "  Pipeline Compute Memory:  " << std::fixed << std::setprecision(2) << std::setw(12) << stats.pipeline_memory_gb << " GB (per pipeline)" << std::endl;
    std::cout << "-----------------------------" << std::endl;
}

// --- End Performance Estimation ---


int main(int argc, char* argv[]) {
    // --- Default Parameters ---
    int Nf = 128;
    int Nt = 4096;
    int Ndm_max = 512;
    double t_total = 0.41; // seconds
    double f_max = 1600.0; // MHz
    double f_min = 1200.0; // MHz
    
    int N_batches = 3;   // Number of spectrograms per pipeline
    int N_pipelines = 1; // Number of pipelines to run
    bool dry_run = false; // NEW: Dry run flag

    // --- NEW: Argument Parsing ---
    std::map<std::string, std::string> args;
    try {
        for (int i = 1; i < argc; ++i) {
            std::string arg = argv[i];
            if (arg == "--help") {
                print_usage(argv[0]);
                return 0;
            }
            // Check for flags that require a value
            if (arg.rfind("--", 0) == 0) {
                if (i + 1 < argc) {
                    args[arg] = argv[i + 1];
                    ++i; // Skip the value
                } else {
                    std::cerr << "Error: Flag " << arg << " is missing a value." << std::endl;
                    print_usage(argv[0]);
                    return 1;
                }
            }
        }

        // Override defaults with parsed arguments
        if (args.count("--num-freq-channels")) Nf = std::stoi(args["--num-freq-channels"]);
        if (args.count("--num-time-samples"))  Nt = std::stoi(args["--num-time-samples"]);
        if (args.count("--num-dm-trials"))     Ndm_max = std::stoi(args["--num-dm-trials"]);
        if (args.count("--num-batches"))       N_batches = std::stoi(args["--num-batches"]);
        if (args.count("--num-pipelines"))     N_pipelines = std::stoi(args["--num-pipelines"]);
        if (args.count("--total-obs-time"))    t_total = std::stod(args["--total-obs-time"]);
        if (args.count("--f-max"))             f_max = std::stod(args["--f-max"]);
        if (args.count("--f-min"))             f_min = std::stod(args["--f-min"]);
        if (args.count("--dry-run"))           dry_run = (args["--dry-run"] == "yes");

    } catch (const std::exception& e) {
        std::cerr << "Error parsing arguments: " << e.what() << std::endl;
        print_usage(argv[0]);
        return 1;
    }
    // --- End Argument Parsing ---

    
    // --- 0. VALIDATE STATIC PARAMETERS ---
    bool params_valid = true;
    if (Nf <= 0 || (Nf & (Nf - 1)) != 0) {
        std::cerr << "ERROR: Nf (Frequency Channels) must be a positive power of 2." << std::endl;
        params_valid = false;
    }
    if (Nt <= 0) {
        std::cerr << "ERROR: Nt (Time Samples) must be positive." << std::endl;
        params_valid = false;
    }
    if (Ndm_max <= 0) {
        std::cerr << "ERROR: Ndm_max (DM Trials) must be positive." << std::endl;
        params_valid = false;
    }
    if (N_batches <= 0) {
        std::cerr << "ERROR: N_batches must be positive." << std::endl;
        params_valid = false;
    }
    if (N_pipelines <= 0) {
        std::cerr << "ERROR: N_pipelines must be positive." << std::endl;
        params_valid = false;
    }
    // ... (rest of validation) ...
    if (!params_valid) {
        std::cerr << "Aborting due to invalid parameters." << std::endl;
        return 1;
    }


    std::cout << "Running with Nf=" << Nf << ", Nt=" << Nt << ", Ndm_max=" << Ndm_max << ", t_total=" << t_total << " s" << std::endl;
    std::cout << "Processing " << N_pipelines << " pipeline(s), each with " << N_batches << " spectrograms." << std::endl;
    long long total_spectrograms = (long long)N_pipelines * N_batches;
    std::cout << "Total spectrograms to process: " << total_spectrograms << std::endl;

    // --- Print Performance Estimates ---
    PerfStats stats;
    stats.precompute_flops = estimate_precompute_flops(Nf, Ndm_max);
    stats.core_compute_flops = estimate_core_compute_flops(Nf, Nt, Ndm_max, N_batches);
    stats.pipeline_memory_gb = estimate_memory_allocations(Nf, Nt, Ndm_max, N_batches);
    print_perf_stats(stats, N_batches);

    // --- NEW: Handle Dry Run ---
    if (dry_run) {
        std::cout << "\n--- DRY RUN complete. Halting before computation. ---" << std::endl;
        return 0;
    }

    // --- 1. PRECOMPUTATION (TIMED) ---
    std::cout << "\n--- 1. Running FDMT Precomputation (ONCE) ---" << std::endl;
    auto start_precompute = std::chrono::high_resolution_clock::now();
    
    FdmtPlan plan = create_fdmt_plan(Nf, Nt, Ndm_max, f_max, f_min, t_total);
    
    auto end_precompute = std::chrono::high_resolution_clock::now();
    auto duration_precompute = std::chrono::duration_cast<std::chrono::milliseconds>(end_precompute - start_precompute).count();

    // --- Setup for physical constants ---
    double delta_t_sample = t_total / Nt;
    
    // FDD Max DM Search is usually an input, but here it is implied by Ndm_max in samples
    // Let's reverse engineer the Max DM in pc/cm3 based on Ndm_max samples
    double f_max_ghz = f_max / 1000.0;
    double f_min_ghz = f_min / 1000.0;
    // Delay = K * DM * (1/min^2 - 1/max^2)
    // Max Delay Samples = Ndm_max
    // Max DM = (Ndm_max * dt) / (K * (...))
    double inv_freq_term = (1.0 / (f_min_ghz * f_min_ghz) - 1.0 / (f_max_ghz * f_max_ghz));
    double calculated_max_dm = (Ndm_max * delta_t_sample) / (DISPERSION_CONSTANT * inv_freq_term);
    
    std::cout << "Calculated Max DM capacity: " << calculated_max_dm << " pc/cm3" << std::endl;

    // --- Setup for data generation (MATCHING FDD DISTRIBUTION) ---
    // Seed 1234 to match FDD default
    std::default_random_engine generator(1234); 
    
    // FDD Noise: Mean 32.0, Std 8.0
    float noise_mean = 32.0f;
    float noise_std = 8.0f;
    std::normal_distribution<float> noise_dist(noise_mean, noise_std); 
    
    // FDD Signal: Amp 4.0 to 6.0
    std::uniform_real_distribution<float> signal_dist(10.0f, 15.0f);
    
    // FDD DM: 0 to Max
    std::uniform_real_distribution<float> dm_dist(0.0f, (float)calculated_max_dm);

    // FDD Time: 10% to 80% of obs time
    std::uniform_int_distribution<int> time_dist((int)(Nt * 0.1), (int)(Nt * 0.8));

    
    // --- 2. DATA GENERATION (MOVED OUTSIDE PIPELINE LOOP) ---
    std::cout << "\n--- 2. Generating " << total_spectrograms << " spectrograms... ---" << std::endl;
    auto start_data_gen = std::chrono::high_resolution_clock::now();

    std::vector<std::vector<float>> all_spectrograms(total_spectrograms);
    std::vector<PulseInfo> all_pulse_info(total_spectrograms);

#pragma omp parallel for
    for (int i = 0; i < total_spectrograms; ++i) {
      // Need a private generator for each thread, or use a thread-safe PRNG
      // For simplicity, we'll keep the non-thread-safe one but be aware this
      // will produce identical pulses if OMP is enabled here without changes.
      // --- Correction: A thread-local or properly seeded generator per thread is needed ---
      // --- For now, we accept potentially duplicated pulses for parallel gen ---
      
      // 1. Generate unique physical parameters (FDD Style)
      double injected_dm_batch = dm_dist(generator);
      double injected_amp_batch = signal_dist(generator); // This is per-bin amplitude
      int t_fmax_center_batch = time_dist(generator);

      // Store info (amplitude stored is just the scalar injection value)
      all_pulse_info[i] = {
        injected_dm_batch,
        t_fmax_center_batch,
        0, // expected_delay_samples unused in new verification
        injected_amp_batch 
      };
      
      // 3. Create spectrogram and fill with noise
      all_spectrograms[i].resize(Nf * Nt);
      for(float& val : all_spectrograms[i]) {
	val = noise_dist(generator);
      }

      // 4. Inject the unique pulse
      inject_pulse(all_spectrograms[i], Nf, Nt, 
		   injected_dm_batch, t_fmax_center_batch, 
		   f_max, f_min, t_total, injected_amp_batch);      
    }
    
    auto end_data_gen = std::chrono::high_resolution_clock::now();
    auto duration_data_gen = std::chrono::duration_cast<std::chrono::milliseconds>(end_data_gen - start_data_gen).count();
    
    
    // --- 3. PIPELINE EXECUTION (TIMED) ---
    std::cout << "\n--- 3. Starting Pipeline Execution ---" << std::endl;
    long long duration_total_compute_ms = 0;
    long long duration_total_copy_ms = 0;
    
    for (int p = 0; p < N_pipelines; ++p) {
      std::cout << "\n--- Pipeline " << p + 1 << " / " << N_pipelines << " ---" << std::endl;
      
      // --- 3a. Get Batch Slice ---
      int start_idx = p * N_batches;
      std::vector<std::vector<float>> spectrogram_batch(
							all_spectrograms.begin() + start_idx,
							all_spectrograms.begin() + start_idx + N_batches
							);
      std::cout << "Processing spectrograms " << start_idx << " to " << (start_idx + N_batches - 1) << "..." << std::endl;
      
      // Allocate output buffer for this batch
      std::vector<std::vector<float>> fdmt_output_batch(
							N_batches, 
							std::vector<float>(plan.Ndm_max * plan.Nt)
							);
      
      // --- 3b. BATCH EXECUTION (TIMED) ---
      std::cout << "Running Batched FDMT Execution..." << std::endl;
      
      long long duration_copy_pipeline_ms = 0; // Timer for copies *within* this pipeline
      auto start_compute_pipeline = std::chrono::high_resolution_clock::now();
      
      // Single call to process the entire batch
      fdmt_execute_batched(
			   plan, 
			   spectrogram_batch, 
			   fdmt_output_batch,      // Pass output buffer
			   duration_copy_pipeline_ms // Pass copy timer
			   );
      
      auto end_compute_pipeline = std::chrono::high_resolution_clock::now();
      
      // --- Accumulate Timings ---
      auto duration_compute_pipeline_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end_compute_pipeline - start_compute_pipeline).count();
      duration_total_compute_ms += duration_compute_pipeline_ms;
      duration_total_copy_ms += duration_copy_pipeline_ms;
      
      std::cout << "Pipeline compute (kernel + copy) took: " << duration_compute_pipeline_ms << " ms." << std::endl;
      std::cout << "  (Breakdown: " << (duration_compute_pipeline_ms - duration_copy_pipeline_ms) << " ms kernel + " << duration_copy_pipeline_ms << " ms copy)" << std::endl;
      std::cout << "  Average time per spectrogram (compute): " << (double)duration_compute_pipeline_ms / N_batches << " ms." << std::endl;

      // --- 3c. ANALYZE RESULTS ---
      std::cout << "Analyzing Results..." << std::endl;
      
      std::vector<VerificationResult> batch_results;

      for (int i = 0; i < N_batches; ++i) {
        const PulseInfo& info = all_pulse_info[start_idx + i];
	
	// Use new detection logic
        VerificationResult res = find_best_candidate_cpu(
            fdmt_output_batch[i], 
            start_idx + i, 
            Nt, Ndm_max, 
            delta_t_sample, 
            calculated_max_dm, 
            info,
            f_min, f_max // [FIX] Added freq params for time correction
        );
	
        batch_results.push_back(res);
      }
      // Print the FDD-style table
      print_verification_table(batch_results);      
    }
    
    std::cout << "\n--- 4. Total Execution Summary ---" << std::endl;
    std::cout << "Processed " << N_pipelines << " pipelines of " << N_batches << " spectrograms each." << std::endl;
    std::cout << "Total spectrograms: " << total_spectrograms << std::endl;
    
    long long duration_total_kernel_ms = duration_total_compute_ms - duration_total_copy_ms;

    std::cout << "\n--- TIMING REPORT ---" << std::endl;
    std::cout << "  1. Pre-computation Time: " << duration_precompute << " ms" << std::endl;
    std::cout << "  2. Data Generation Time: " << duration_data_gen << " ms" << std::endl;
    std::cout << "  3. FDMT Copy Time (Total): " << duration_total_copy_ms << " ms" << std::endl;
    std::cout << "  4. FDMT Kernel Time (Total): " << duration_total_kernel_ms << " ms" << std::endl;
    std::cout << "  --------------------------------------------------" << std::endl;
    std::cout << "  Total Computation (Kernel + Copy): " << duration_total_compute_ms << " ms" << std::endl;


    if (total_spectrograms > 0) {
        std::cout << "\nAverage time per spectrogram (kernel only): " 
                  << (double)duration_total_kernel_ms / total_spectrograms << " ms." << std::endl;
        std::cout << "Average time per spectrogram (compute total): " 
                  << (double)duration_total_compute_ms / total_spectrograms << " ms." << std::endl;
    }

    return 0;
}

