/**
 * @file dedisp_api.cu
 * @brief PIMPL implementation for the FDD dedispersion pipeline API.
 *
 * This TU includes the same textual sub-headers as the umbrella
 * (config, host_helpers, kernels_fdd, kernels_data_prep, kernels_candidate)
 * plus the extracted pipeline class. Does NOT include application_impl.hpp,
 * reversibility_test.hpp, or main().
 *
 * @see dedisp_api.h for the public API.
 */

#include "dedisp_api.h"

#include <iostream>
#include <vector>
#include <complex>
#include <cmath>
#include <numeric>
#include <algorithm>
#include <string>
#include <iomanip>
#include <memory>
#include <type_traits>

#include <cuda_runtime.h>
#include <cuda.h>
#include <cufft.h>
#include <cublas_v2.h>
#include <curand.h>

#include <cublasLt.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>

// Thrust for dynamic scaling reduction
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

// ═══════════════════════════════════════════════════════════════
// FddGpuPipeline class (extracted into textual include)
// ═══════════════════════════════════════════════════════════════
#include "gpu_fdd_improved/pipeline_class.hpp"


namespace dedisp_api {

// ═══════════════════════════════════════════════════════════════
// Helper: map ComputeMode enum to internal string
// ═══════════════════════════════════════════════════════════════
static const char* compute_mode_to_string(ComputeMode mode) {
    switch (mode) {
        case ComputeMode::CuBLAS_FP32:    return "cublas";
        case ComputeMode::CuBLASLt_FP16:  return "cublas_lt_fp16";
        case ComputeMode::CuBLASLt_FP8:   return "cublas_lt_fp8";
        case ComputeMode::CUTLASS_FP8:     return "cutlass";
        case ComputeMode::CUTLASS_FP6:     return "cutlass_fp6";
        case ComputeMode::CUTLASS_FP4:     return "cutlass_fp4";
        default:                           return "cublas";
    }
}

// ═══════════════════════════════════════════════════════════════
// Impl struct — all internal state hidden from consumers
// ═══════════════════════════════════════════════════════════════
struct DedispPipeline::Impl {
    DedispConfig config;
    std::string compute_mode_str;

    int nt_padded = 0;
    int nt_complex = 0;

    std::unique_ptr<FddGpuPipeline<float>> pipeline;

    // Phasor table (managed memory, kept alive for cublas/kernel modes)
    DeviceBuffer phasor_table;

    // Precomputation tables
    DeviceBuffer d_time_delays;
    DeviceBuffer d_f_k_values;

    // Search results (DeviceCandidate array on GPU)
    DeviceBuffer d_search_results;

    // Internal dedispersed buffer for combined dedisperse+search
    DeviceBuffer d_dedispersed_internal;

    // Injection params upload buffer
    DeviceBuffer d_injection_params;

    cudaStream_t internal_stream = nullptr;
    bool initialized = false;

    Impl(const DedispConfig& cfg)
        : config(cfg),
          compute_mode_str(compute_mode_to_string(cfg.compute_mode))
    {
        CUDA_CHECK(cudaStreamCreate(&internal_stream));
    }

    ~Impl() {
        pipeline.reset();
        if (internal_stream) {
            cudaStreamDestroy(internal_stream);
            internal_stream = nullptr;
        }
    }
};

// ═══════════════════════════════════════════════════════════════
// DedispPipeline public methods
// ═══════════════════════════════════════════════════════════════

/// @copydoc DedispPipeline::DedispPipeline
DedispPipeline::DedispPipeline(const DedispConfig& config)
    : impl_(new Impl(config))
{}

/// @copydoc DedispPipeline::~DedispPipeline
DedispPipeline::~DedispPipeline() {
    delete impl_;
}

/// @copydoc DedispPipeline::initialize
int DedispPipeline::initialize(cudaStream_t stream) {
    if (impl_->initialized) return 0;

    cudaStream_t s = stream ? stream : impl_->internal_stream;
    const auto& cfg = impl_->config;

    // --- 1. Calculate Nt_padded ---
    const float time_resolution = cfg.total_obs_time_s / cfg.Nt;
    const float f_min_GHz = cfg.f_min_MHz / 1000.0f;
    const float f_max_GHz = cfg.f_max_MHz / 1000.0f;
    const float max_delay_s = (float)DISPERSION_CONSTANT * cfg.max_dm *
        (1.0f / (f_min_GHz * f_min_GHz) - 1.0f / (f_max_GHz * f_max_GHz));
    const int max_delay_bins = (int)std::ceil(max_delay_s / time_resolution);

    size_t required_size = (size_t)cfg.Nt + max_delay_bins;
    impl_->nt_padded = 1;
    while ((size_t)impl_->nt_padded < required_size) impl_->nt_padded *= 2;
    impl_->nt_complex = impl_->nt_padded / 2 + 1;

    std::cout << "[DedispAPI] Nt=" << cfg.Nt << " -> Nt_padded=" << impl_->nt_padded
              << " (Nt_complex=" << impl_->nt_complex << ")" << std::endl;

    // --- 2. Generate precomputation tables on GPU ---
    size_t delays_bytes = (size_t)cfg.Ndm * cfg.Nf * sizeof(float);
    impl_->d_time_delays.allocate(delays_bytes);

    size_t fk_bytes = (size_t)impl_->nt_complex * sizeof(float);
    impl_->d_f_k_values.allocate(fk_bytes);

    size_t total_work = std::max((size_t)cfg.Ndm * cfg.Nf, (size_t)impl_->nt_complex);
    dim3 block_setup(256);
    dim3 grid_setup((total_work + 255) / 256);

    kernel_generate_precomp<float><<<grid_setup, block_setup, 0, s>>>(
        impl_->d_time_delays.get<float>(),
        impl_->d_f_k_values.get<float>(),
        cfg.Ndm, cfg.Nf, impl_->nt_complex,
        cfg.f_min_MHz, cfg.f_max_MHz, cfg.min_dm, cfg.max_dm,
        time_resolution, (float)DISPERSION_CONSTANT
    );
    CUDA_CHECK(cudaGetLastError());

    // --- 3. Determine tiling and generate phasor table ---
    const std::string& mode = impl_->compute_mode_str;
    bool use_conjugate = (mode != "kernel");
    bool cutlass_mode = (mode == "cutlass" || mode == "cutlass_fp6" || mode == "cutlass_fp4");

    // Compute Nt_tile for CUTLASS modes: auto-size from free GPU memory
    int nt_tile = 0;  // 0 = no tiling
#ifdef HAS_CUTLASS_GEMM
    if (cutlass_mode) {
        // Query free memory BEFORE pool allocation to estimate what the pipeline
        // constructor will leave us. Pools + B tile must fit.
        size_t free_byte = 0, total_byte = 0;
        cudaMemGetInfo(&free_byte, &total_byte);

        // Estimate pool sizes (mirrors pipeline_class.hpp constructor logic)
        size_t sz_real_io = (size_t)cfg.max_batch_size * std::max(cfg.Nf, cfg.Ndm) *
                            (impl_->nt_padded) * sizeof(float);
        size_t sz_comp_io = (size_t)cfg.max_batch_size * std::max(cfg.Nf, cfg.Ndm) *
                            (impl_->nt_complex) * sizeof(cufftComplex);
        size_t sz_planar_A = (size_t)cfg.max_batch_size * cfg.Nf * impl_->nt_complex * sizeof(__half);
        size_t sz_planar_C = (size_t)cfg.max_batch_size * cfg.Ndm * impl_->nt_complex * sizeof(float);
        size_t pool1_est = std::max({sz_real_io, sz_comp_io, sz_planar_A * 2});
        size_t pool2_est = std::max({sz_real_io, sz_comp_io, sz_planar_C * 2});
        pool2_est = std::max(pool2_est, (size_t)cfg.max_batch_size * cfg.Ndm *
                             impl_->nt_complex * sizeof(cufftComplex));

        // Full B size = Nf * Ndm * Nt_complex * sizeof(__half) * 2 (re + im)
        size_t full_B_bytes = (size_t)cfg.Nf * cfg.Ndm * impl_->nt_complex * sizeof(__half) * 2;

        // Memory available for B tile after pools + 4 GB overhead
        size_t overhead = (size_t)4 * 1024 * 1024 * 1024ULL;
        size_t pools_total = pool1_est + pool2_est;

        if (free_byte > pools_total + overhead) {
            size_t avail_for_B = free_byte - pools_total - overhead;
            // Total per-tile memory includes FP16 B tile + CUTLASS internal FP8 conversion buffers.
            // CUTLASS gemm() internally allocates: FP8 A (M×K×batch), FP8 B (N×K×batch),
            // FP16 scratch (M×N×batch). Total ≈ 2.5× the FP16 B tile for typical dedisp shapes.
            size_t bytes_per_batch_fp16 = (size_t)cfg.Nf * cfg.Ndm * sizeof(__half) * 2;
            size_t bytes_per_batch_workspace = (size_t)cfg.Nf * cfg.Ndm * 2 +  // FP8 B (re+im)
                (size_t)cfg.max_batch_size * cfg.Nf * 2 +                       // FP8 A (re+im)
                (size_t)cfg.max_batch_size * cfg.Ndm * sizeof(__half) * 2;      // FP16 scratch
            size_t bytes_per_batch_total = bytes_per_batch_fp16 + bytes_per_batch_workspace;
            int max_nt_tile = (bytes_per_batch_total > 0) ? (int)(avail_for_B / bytes_per_batch_total) : impl_->nt_complex;
            max_nt_tile = std::max(1, std::min(max_nt_tile, impl_->nt_complex));

            // Only tile if full B + workspace doesn't fit (with comfortable margin)
            size_t full_total_bytes = (size_t)impl_->nt_complex * (bytes_per_batch_fp16 + bytes_per_batch_workspace) / 2;
            // ^ Rough estimate: full B FP16 + workspace for prepare_b path
            if (full_B_bytes > avail_for_B * 90 / 100) {
                nt_tile = max_nt_tile;
                int num_tiles = (impl_->nt_complex + nt_tile - 1) / nt_tile;
                size_t tile_B_bytes = bytes_per_batch_fp16 * nt_tile;
                printf("[DedispAPI] Time-domain tiling: Nt_tile=%d (%d tiles), "
                       "B tile=%.1f GB (full=%.1f GB, workspace/tile=%.1f GB, avail=%.1f GB)\n",
                       nt_tile, num_tiles, tile_B_bytes / 1e9,
                       full_B_bytes / 1e9, (bytes_per_batch_workspace * nt_tile) / 1e9,
                       avail_for_B / 1e9);
            } else {
                printf("[DedispAPI] Full B fits in memory (%.1f GB), no tiling needed\n",
                       full_B_bytes / 1e9);
            }
        } else {
            // Very tight memory — use smallest viable tile with full workspace accounting
            size_t bytes_per_batch_fp16 = (size_t)cfg.Nf * cfg.Ndm * sizeof(__half) * 2;
            size_t bytes_per_batch_workspace = (size_t)cfg.Nf * cfg.Ndm * 2 +
                (size_t)cfg.max_batch_size * cfg.Nf * 2 +
                (size_t)cfg.max_batch_size * cfg.Ndm * sizeof(__half) * 2;
            size_t bytes_per_batch_total = bytes_per_batch_fp16 + bytes_per_batch_workspace;
            nt_tile = std::max(1, (int)((free_byte > pools_total + (size_t)1024*1024*1024ULL)
                ? (free_byte - pools_total - (size_t)1024*1024*1024ULL) / bytes_per_batch_total : 1));
            nt_tile = std::min(nt_tile, impl_->nt_complex);
            printf("[DedispAPI] Tight memory: Nt_tile=%d\n", nt_tile);
        }
    }
#endif

    // Generate full phasor table only for non-tiled modes
    if (!cutlass_mode || nt_tile == 0 || nt_tile >= impl_->nt_complex) {
        // Non-tiled: generate full phasor table as before
        size_t phasor_bytes = (size_t)impl_->nt_complex * cfg.Ndm * cfg.Nf * sizeof(cufftComplex);
        impl_->phasor_table.allocateManaged(phasor_bytes);

        int device_id = 0;
        CUDA_CHECK(cudaGetDevice(&device_id));
        cudaMemLocation location;
        location.type = cudaMemLocationTypeDevice;
        location.id = device_id;
        CUDA_CHECK(cudaMemAdvise(impl_->phasor_table.get(), phasor_bytes,
                                 cudaMemAdviseSetPreferredLocation, location));

        unsigned int grid_z = impl_->nt_complex;
        dim3 block_phasor(16, 16);
        dim3 grid_phasor((cfg.Nf + 15) / 16, (cfg.Ndm + 15) / 16, grid_z);

        kernel_generate_phasors<float, cufftComplex><<<grid_phasor, block_phasor, 0, s>>>(
            impl_->phasor_table.get<cufftComplex>(),
            impl_->d_f_k_values.get<float>(),
            impl_->d_time_delays.get<float>(),
            cfg.Nf, cfg.Ndm, impl_->nt_complex,
            use_conjugate);

        CUDA_CHECK(cudaStreamSynchronize(s));
        CUDA_CHECK(cudaGetLastError());

        // Reset nt_tile to 0 (no tiling) for non-CUTLASS or fits-in-memory case
        if (cutlass_mode) nt_tile = 0;
    }

    // --- 4. Create pipeline ---
    impl_->pipeline = std::make_unique<FddGpuPipeline<float>>(
        cfg.max_batch_size, cfg.Nf, cfg.Nt, cfg.Ndm,
        impl_->nt_padded, impl_->compute_mode_str,
        cfg.kernel_tune_verbosity, cfg.kernel_tune_cache_path,
        nt_tile,
        cfg.strategy_tune_verbosity, cfg.strategy_tune_cache_path,
        cfg.gemm_tune);

    // --- 5. Pre-load phasors or set precomp tables for tiled mode ---
    if (impl_->pipeline->is_tiled()) {
        // Tiled CUTLASS: pass precomp table pointers for on-the-fly generation
        impl_->pipeline->set_precomp_tables(
            impl_->d_time_delays.get<float>(),
            impl_->d_f_k_values.get<float>(),
            use_conjugate);
    } else if (mode == "cublas_lt_fp16" || mode == "cublas_lt_fp8" ||
               mode == "cutlass" || mode == "cutlass_fp6" || mode == "cutlass_fp4") {
        impl_->pipeline->prepare_phasors(impl_->phasor_table.get<cufftComplex>(), s);
        impl_->phasor_table.free();
    }

    impl_->initialized = true;
    std::cout << "[DedispAPI] Pipeline initialized (mode=" << mode << ")" << std::endl;
    return 0;
}

/// @copydoc DedispPipeline::dedisperse
int DedispPipeline::dedisperse(const float* input, float* output,
                               int batch_size, cudaStream_t stream) {
    if (!impl_->initialized) {
        fprintf(stderr, "[DedispAPI] Error: pipeline not initialized\n");
        return -1;
    }
    if (batch_size > impl_->config.max_batch_size) {
        fprintf(stderr, "[DedispAPI] Error: batch_size %d > max_batch_size %d\n",
                batch_size, impl_->config.max_batch_size);
        return -1;
    }

    cudaStream_t s = stream ? stream : impl_->internal_stream;

    impl_->pipeline->execute(
        input,
        impl_->phasor_table.get<cufftComplex>(),
        output,
        batch_size, s);

    return 0;
}

/// @copydoc DedispPipeline::search
int DedispPipeline::search(const float* dedispersed, Candidate* candidates,
                           int* num_found, const SearchConfig& search_cfg,
                           int batch_size, cudaStream_t stream) {
    if (!impl_->initialized) {
        fprintf(stderr, "[DedispAPI] Error: pipeline not initialized\n");
        return -1;
    }

    cudaStream_t s = stream ? stream : impl_->internal_stream;
    const auto& cfg = impl_->config;

    // Allocate GPU search results: [batch_size x Ndm]
    size_t search_res_count = (size_t)batch_size * cfg.Ndm;
    impl_->d_search_results.allocate(search_res_count * sizeof(DeviceCandidate));

    // Reset candidates
    dim3 block_rst(256);
    dim3 grid_rst((search_res_count + 255) / 256);
    kernel_reset_candidates<<<grid_rst, block_rst, 0, s>>>(
        impl_->d_search_results.get<DeviceCandidate>(), search_res_count);

    // Run boxcar search for each width
    dim3 search_block(256);
    dim3 search_grid(batch_size, cfg.Ndm);

    for (int w = 0; w < search_cfg.num_widths; ++w) {
        int width = search_cfg.widths[w];
        if (width > cfg.Nt) continue;

        kernel_find_best_candidate<float><<<search_grid, search_block, 0, s>>>(
            dedispersed,
            impl_->d_search_results.get<DeviceCandidate>(),
            cfg.Ndm, cfg.Nt,
            width,
            search_cfg.noise_mean, search_cfg.noise_stddev);
    }
    CUDA_CHECK(cudaGetLastError());

    // Copy results to host
    std::vector<DeviceCandidate> host_results(search_res_count);
    CUDA_CHECK(cudaMemcpyAsync(host_results.data(), impl_->d_search_results.get(),
                               search_res_count * sizeof(DeviceCandidate),
                               cudaMemcpyDeviceToHost, s));
    CUDA_CHECK(cudaStreamSynchronize(s));

    // CPU reduction: find best candidate per batch element
    const float time_resolution = cfg.total_obs_time_s / cfg.Nt;
    int found = 0;

    for (int b = 0; b < batch_size && found < search_cfg.max_candidates; ++b) {
        float best_snr = -1e9f;
        int best_dm_idx = -1;
        int best_time_idx = -1;
        float best_intensity = 0.0f;
        int best_width = 0;

        for (int dm = 0; dm < cfg.Ndm; ++dm) {
            int idx = b * cfg.Ndm + dm;
            if (host_results[idx].max_snr > best_snr) {
                best_snr = host_results[idx].max_snr;
                best_dm_idx = host_results[idx].dm_idx;
                best_time_idx = host_results[idx].time_idx;
                best_intensity = host_results[idx].max_intensity;
                best_width = host_results[idx].width;
            }
        }

        if (best_dm_idx >= 0 && best_snr > 0.0f) {
            Candidate c;
            c.dm = (cfg.Ndm > 1)
                ? cfg.min_dm + ((float)best_dm_idx / (cfg.Ndm - 1)) * (cfg.max_dm - cfg.min_dm)
                : cfg.min_dm;
            c.time_s = best_time_idx * time_resolution;
            c.snr = best_snr;
            c.intensity = best_intensity;
            c.dm_idx = best_dm_idx;
            c.time_idx = best_time_idx;
            c.width = best_width;
            candidates[found++] = c;
        }
    }

    *num_found = found;
    return 0;
}

/// @copydoc DedispPipeline::dedisperse_and_search
int DedispPipeline::dedisperse_and_search(const float* input, Candidate* candidates,
                                          int* num_found, const SearchConfig& search_cfg,
                                          int batch_size, cudaStream_t stream) {
    if (!impl_->initialized) {
        fprintf(stderr, "[DedispAPI] Error: pipeline not initialized\n");
        return -1;
    }

    const auto& cfg = impl_->config;

    // Allocate internal dedispersed buffer
    size_t output_bytes = (size_t)batch_size * cfg.Ndm * cfg.Nt * sizeof(float);
    impl_->d_dedispersed_internal.allocate(output_bytes);

    // Dedisperse
    int ret = dedisperse(input, impl_->d_dedispersed_internal.get<float>(),
                         batch_size, stream);
    if (ret != 0) return ret;

    // Search
    return search(impl_->d_dedispersed_internal.get<float>(), candidates,
                  num_found, search_cfg, batch_size, stream);
}

/// @copydoc DedispPipeline::inject_signal
int DedispPipeline::inject_signal(float* buffer, const InjectionParams* params,
                                  int num_signals, int batch_size,
                                  cudaStream_t stream) {
    if (!impl_->initialized) {
        fprintf(stderr, "[DedispAPI] Error: pipeline not initialized\n");
        return -1;
    }

    cudaStream_t s = stream ? stream : impl_->internal_stream;
    const auto& cfg = impl_->config;

    // Convert InjectionParams -> PulsarParams<float> and upload
    std::vector<PulsarParams<float>> h_params(num_signals);
    for (int i = 0; i < num_signals; ++i) {
        h_params[i].dm = params[i].dm;
        h_params[i].amplitude = params[i].amplitude;
        h_params[i].pulse_start_time = params[i].pulse_start_time_s;
        h_params[i].width_s = params[i].width_s;
        h_params[i].scattering_s = params[i].scattering_s;
    }

    size_t params_bytes = num_signals * sizeof(PulsarParams<float>);
    impl_->d_injection_params.allocate(params_bytes);
    CUDA_CHECK(cudaMemcpyAsync(impl_->d_injection_params.get(), h_params.data(),
                               params_bytes, cudaMemcpyHostToDevice, s));

    const float f_ref_GHz = cfg.f_max_MHz / 1000.0f;
    const float time_resolution = cfg.total_obs_time_s / cfg.Nt;

    dim3 grid_inject(((size_t)batch_size * cfg.Nf + 255) / 256);
    dim3 block_inject(256);

    kernel_inject_bursts<float><<<grid_inject, block_inject, 0, s>>>(
        buffer,
        impl_->d_injection_params.get<PulsarParams<float>>(),
        batch_size, cfg.Nf, cfg.Nt,
        cfg.f_min_MHz, cfg.f_max_MHz,
        time_resolution, f_ref_GHz);

    CUDA_CHECK(cudaGetLastError());
    return 0;
}

/// @copydoc DedispPipeline::get_nt_padded
int DedispPipeline::get_nt_padded() const {
    return impl_->nt_padded;
}

/// @copydoc DedispPipeline::get_compute_mode_string
const char* DedispPipeline::get_compute_mode_string() const {
    return impl_->compute_mode_str.c_str();
}

/// @copydoc DedispPipeline::set_verbose
void DedispPipeline::set_verbose(bool v) {
    if (impl_->pipeline) impl_->pipeline->set_verbose(v);
}

} // namespace dedisp_api
