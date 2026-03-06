/**
 * @file dedisp_api.h
 * @brief Lightweight PIMPL API for FDD dedispersion pipeline.
 *
 * This header has **no internal CUDA/cuFFT/cuBLAS/CUTLASS dependencies** beyond
 * @c <cuda_runtime.h>. Consumers include only this header and link against
 * @c libdedisp_api.a, which contains the heavy internal pipeline compiled in
 * @c dedisp_api.cu.
 *
 * Three use cases:
 *  1. **Dedisperse only** -- pass filterbank [batch, Nf, Nt], get [batch, Ndm, Nt]
 *  2. **Dedisperse + search** -- pass filterbank, get pulse candidates
 *  3. **Inject + dedisperse + search** -- inject synthetic signal, dedisperse, search
 *
 * @see dedisp_api.cu for the PIMPL implementation.
 */
#pragma once

#include <cuda_runtime.h>
#include <cstddef>
#include <cstdint>

/**
 * @namespace dedisp_api
 * @brief Public namespace for the FDD dedispersion pipeline API.
 */
namespace dedisp_api {

/// @brief Compute backend and precision for the FDD GEMM stage.
enum class ComputeMode {
    CuBLAS_FP32,      ///< Standard cuBLAS FP32 batched GEMM.
    CuBLASLt_FP16,    ///< cuBLASLt tensor core FP16 (4-split complex).
    CuBLASLt_FP8,     ///< cuBLASLt tensor core FP8 E4M3 (4-split complex).
    CUTLASS_FP8,       ///< CUTLASS FP8 E4M3 (requires Hopper/Blackwell + libcutlass_gemm_api).
    CUTLASS_FP6,       ///< CUTLASS FP6 E3M2 MXFP block-scaled (Blackwell only).
    CUTLASS_FP4        ///< CUTLASS FP4 E2M1 MXFP block-scaled (Blackwell only, lossy, 2x throughput).
};

/// @brief Configuration for the dedispersion pipeline.
struct DedispConfig {
    int Nf;                   ///< Number of frequency channels.
    int Nt;                   ///< Number of time samples (unpadded).
    int Ndm;                  ///< Number of DM trials (partitions the [min_dm, max_dm] range).
    float f_min_MHz;          ///< Minimum frequency of the band in MHz.
    float f_max_MHz;          ///< Maximum frequency of the band in MHz.
    float min_dm = 0.0f;      ///< Minimum DM to search in pc/cm^3 (default 0).
    float max_dm;             ///< Maximum DM to search in pc/cm^3.
    float total_obs_time_s;   ///< Total observation time in seconds.
    ComputeMode compute_mode; ///< Compute backend selection.
    int max_batch_size;       ///< Maximum batch size for pipeline allocation.

    /// @brief Kernel-level tuning verbosity (CUTLASS modes only).
    ///
    /// Controls automatic blockDim/gridDim tuning for overhead kernels
    /// (cast, pack, deinterleave, MXFP preprocessing).
    ///   - 0: Silent (default, no tuning).
    ///   - 1: Show cached/default launch params per kernel.
    ///   - 2: Tune kernels + one-line summary per kernel.
    ///   - 3: Tune kernels + full per-config timing detail.
    ///
    /// Levels >= 2 trigger a ~30-60s sweep on the first call; results are
    /// cached to @p kernel_tune_cache_path and reused across sessions.
    int kernel_tune_verbosity = 0;

    /// @brief File path for the kernel-level tune cache (CUTLASS modes only).
    ///
    /// Default (nullptr): "cutlass_kernel_cache_{build_fingerprint}.txt" in the
    /// current directory.
    /// Set to a custom path to share/isolate the cache across deployments.
    const char* kernel_tune_cache_path = nullptr;

    /// @brief Strategy-level tuning verbosity (CUTLASS modes only).
    ///
    /// Controls HERK and GEMM strategy autotuning output.
    ///   - 0: Silent.
    ///   - 1: Summary — cache hit/miss + winner (default).
    ///   - 2: Sweep detail — per-candidate timing during autotune.
    ///   - 3: Full diagnostics — sweep + skip reasons.
    int strategy_tune_verbosity = 1;

    /// @brief File path for the strategy/GEMM tune cache (CUTLASS modes only).
    ///
    /// Default (nullptr): "cutlass_strategy_cache_{build_fingerprint}.txt".
    /// HERK and GEMM entries share the same file.
    const char* strategy_tune_cache_path = nullptr;

    /// @brief Enable GEMM strategy autotuning (CUTLASS modes only).
    ///
    /// When true, the first gemm()/gemm_prepared() call per problem size
    /// triggers a GemmConfig sweep to find the optimal tile/cluster config.
    /// Results are cached to @p strategy_tune_cache_path.
    bool gemm_tune = true;
};

/// @brief A detected pulse candidate.
struct Candidate {
    float dm;           ///< Detected DM in pc/cm^3.
    float time_s;       ///< Detected arrival time in seconds.
    float snr;          ///< Signal-to-noise ratio.
    float intensity;    ///< Raw intensity (boxcar sum).
    int dm_idx;         ///< DM trial index.
    int time_idx;       ///< Time sample index.
    int width;          ///< Matched filter width (samples).
};

/// @brief Parameters for synthetic signal injection.
struct InjectionParams {
    float dm;                  ///< Dispersion measure in pc/cm^3.
    float amplitude;           ///< Signal amplitude.
    float pulse_start_time_s;  ///< Pulse arrival time at f_max in seconds.
    float width_s;             ///< Pulse intrinsic width in seconds (0 for burst).
    float scattering_s;        ///< Scattering timescale in seconds (0 for burst).
};

/// @brief Configuration for the candidate search stage.
struct SearchConfig {
    const int* widths;     ///< Array of boxcar widths to search (samples).
    int num_widths;        ///< Number of elements in @p widths.
    float noise_mean;      ///< Expected noise mean (for SNR calculation).
    float noise_stddev;    ///< Expected noise standard deviation.
    int max_candidates;    ///< Maximum number of candidates to return.
};

/**
 * @class DedispPipeline
 * @brief PIMPL-wrapped FDD dedispersion pipeline.
 *
 * Manages FFT plans, GEMM backend, phasor tables, and scratch memory.
 * All device pointers passed to methods must reside in GPU memory.
 * The class manages internal CUDA streams and scratch buffers; it is
 * **not** thread-safe (use one instance per thread or synchronize externally).
 *
 * @note Non-copyable, non-movable. Use a pointer or unique_ptr for ownership.
 */
class DedispPipeline {
public:
    /// @brief Construct a new DedispPipeline instance.
    ///
    /// Stores configuration but does not allocate GPU resources.
    /// Call initialize() before any pipeline operation.
    ///
    /// @param config Pipeline configuration.
    explicit DedispPipeline(const DedispConfig& config);

    /// @brief Destroy the instance and free all internal GPU resources.
    ~DedispPipeline();

    /// @brief Initialize the pipeline: compute phasors, allocate FFT plans and buffers.
    ///
    /// Must be called exactly once before dedisperse(), search(), or inject_signal().
    ///
    /// @param stream CUDA stream for initialization kernels (nullptr = default stream).
    /// @return 0 on success, non-zero on failure.
    int initialize(cudaStream_t stream = nullptr);

    /// @brief Run FDD dedispersion on a filterbank batch.
    ///
    /// @param input      Device pointer to input filterbank [batch_size x Nf x Nt], row-major float.
    /// @param output     Device pointer to output dedispersed [batch_size x Ndm x Nt], row-major float.
    /// @param batch_size Number of batch elements to process (must be <= max_batch_size).
    /// @param stream     CUDA stream (nullptr = default stream).
    /// @return 0 on success, non-zero on failure.
    int dedisperse(const float* input, float* output,
                   int batch_size, cudaStream_t stream = nullptr);

    /// @brief Run candidate search on already-dedispersed data.
    ///
    /// @param dedispersed  Device pointer to dedispersed data [batch_size x Ndm x Nt].
    /// @param candidates   Host pointer to output candidate array.
    /// @param num_found    Host pointer; on return, number of candidates found.
    /// @param search_cfg   Search configuration (widths, noise stats, etc.).
    /// @param batch_size   Number of batch elements.
    /// @param stream       CUDA stream (nullptr = default stream).
    /// @return 0 on success, non-zero on failure.
    int search(const float* dedispersed, Candidate* candidates,
               int* num_found, const SearchConfig& search_cfg,
               int batch_size, cudaStream_t stream = nullptr);

    /// @brief Combined dedisperse + search in a single call.
    ///
    /// Internally allocates a scratch buffer for the dedispersed output.
    ///
    /// @param input       Device pointer to input filterbank [batch_size x Nf x Nt].
    /// @param candidates  Host pointer to output candidate array.
    /// @param num_found   Host pointer; on return, number of candidates found.
    /// @param search_cfg  Search configuration.
    /// @param batch_size  Number of batch elements.
    /// @param stream      CUDA stream (nullptr = default stream).
    /// @return 0 on success, non-zero on failure.
    int dedisperse_and_search(const float* input, Candidate* candidates,
                              int* num_found, const SearchConfig& search_cfg,
                              int batch_size, cudaStream_t stream = nullptr);

    /// @brief Inject synthetic dispersed burst signals into a filterbank buffer.
    ///
    /// The buffer should already contain noise (or zeros). Signals are added
    /// atomically so multiple calls accumulate.
    ///
    /// @param buffer      Device pointer to filterbank [batch_size x Nf x Nt] (in-place).
    /// @param params      Host pointer to injection parameter array [num_signals].
    /// @param num_signals Number of signals to inject (one per batch element).
    /// @param batch_size  Total batch size of the buffer.
    /// @param stream      CUDA stream (nullptr = default stream).
    /// @return 0 on success, non-zero on failure.
    int inject_signal(float* buffer, const InjectionParams* params,
                      int num_signals, int batch_size,
                      cudaStream_t stream = nullptr);

    /// @brief Get the padded time dimension (next power of 2 >= Nt + max_delay).
    /// @return Nt_padded value.
    int get_nt_padded() const;

    /// @brief Get the internal compute mode as a human-readable string.
    /// @return Null-terminated string (e.g. "cublas", "cutlass_fp6").
    const char* get_compute_mode_string() const;

    /// @brief Enable/disable per-call performance metric printing.
    /// @param v true to print (default), false to suppress.
    void set_verbose(bool v);

private:
    struct Impl;
    Impl* impl_;

    // Non-copyable, non-movable
    DedispPipeline(const DedispPipeline&) = delete;
    DedispPipeline& operator=(const DedispPipeline&) = delete;
    DedispPipeline(DedispPipeline&&) = delete;
    DedispPipeline& operator=(DedispPipeline&&) = delete;
};

} // namespace dedisp_api
