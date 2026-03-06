#pragma once

#include <cstdint>
#include <ggp.h>
#include <imaging_pipeline.h>

namespace quda { class TimeProfile; }

namespace ggp {

  // Forward declarations
  class XEngine;

  /**
   * @brief End-to-end visibility beamformer pipeline.
   *
   * Orchestrates the following stages:
   *   1. XEngine correlation (raw voltages -> packed triangle visibilities)
   *   2. Imaging: pillbox gridding -> 2D IFFT -> beam pixel extraction
   *      (requires configure_imaging() to be called first)
   *   3. Corner turn: channel-major [Nf, Nb, Nt] -> beam-major [Nb, Nf, Nt]
   *   4. [Optional] Dedispersion via dedisp_api (FDD pipeline)
   */
  class VisibilityPipeline {

  private:
    // Configuration
    uint64_t n_channels_;
    uint64_t n_beams_;
    uint64_t n_time_;
    uint64_t n_antennae_;
    uint64_t n_time_inner_;
    uint64_t n_polarizations_;
    QudaBLASEngine engine_;

    // Sub-stages (cuBLAS path uses XEngine)
    XEngine *xengine_ = nullptr;
    XEngineParam xe_param_;

    // CUTLASS path: device buffers for direct QC -> packed triangle
    void *d_qc_input_ = nullptr;      // Device copy of raw QC input
    void *d_tri_output_ = nullptr;     // Device packed triangle output (FP16 or FP32)
    uint64_t in_payload_size_ = 0;     // QC input bytes
    uint64_t tri_output_size_ = 0;     // Triangle output bytes
    bool herk_output_fp16_ = false;    // True = FP16 output (2B/elem), false = FP32 (4B/elem)

    // Pipeline streams and events (CUTLASS path)
    cudaStream_t stream_h2d_ = nullptr;       // H2D copies
    cudaStream_t stream_compute_ = nullptr;   // HERK + imaging + corner turn
    cudaStream_t stream_d2h_ = nullptr;       // D2H copies
    cudaEvent_t  ev_h2d_done_ = nullptr;      // signals H2D completion
    cudaEvent_t  ev_compute_done_ = nullptr;  // signals compute completion

    // Double-buffered device memory (Level 2 pipeline)
    void  *d_qc_input_db_[2] = {};
    void  *d_tri_output_db_[2] = {};
    cudaEvent_t ev_compute_done_db_[2] = {};

    // Pinned host staging buffers (for truly async DMA)
    void *h_staging_in_ = nullptr;
    void *h_staging_out_ = nullptr;

    // Imaging pipeline
    imaging_pipeline::ImagingPipeline *imager_ = nullptr;
    float *d_beam_output_ = nullptr;     ///< [Nf * n_beam] float (imaging output)
    int Ng_ = 0;
    int n_beam_extract_ = 0;
    imaging_pipeline::FftPrecision fft_prec_ = imaging_pipeline::FftPrecision::FP32;
    bool imaging_configured_ = false;

    // Device buffers for filterbank stages
    float *d_filterbank_chanmajor_ = nullptr; // [Nf, Nb, Nt] after imaging
    float *d_filterbank_beammajor_ = nullptr; // [Nb, Nf, Nt] after corner turn

    // dedisp_api (conditional on DEDISP_API_LIB)
#ifdef DEDISP_API_LIB
    void *dedisp_pipeline_ = nullptr;
#endif

    quda::TimeProfile *profile_tp_ = nullptr;

    bool mem_init_ = false;
    bool streams_init_ = false;
    void init_memory();
    void destroy_memory();
    void init_streams();
    void destroy_streams();

    /// H2D + HERK on stream_compute_, no D2H. Used by compute() to avoid wasted copy.
    void correlate_device_only_(void *raw_input);

  public:

    struct Config {
      // XEngine params
      uint64_t n_antennae = 64;
      uint64_t n_channels = 4;
      uint64_t n_time_per_payload = 1024;
      uint64_t n_time_inner = 2;
      uint64_t n_polarizations = 2;
      QudaPacketFormat packet_format = QUDA_PACKET_FORMAT_DSA2K;
      QudaPrecision compute_prec = QUDA_SINGLE_PRECISION;
      QudaBLASEngine engine = QUDA_BLAS_ENGINE_CUBLAS;

      // Pipeline params
      uint64_t n_beams = 256;   // Output beams (from imaging stage)

      // Dedispersion params (used only if dedisp_api linked)
      int n_dm_trials = 256;
      float f_min_MHz = 700.0f;
      float f_max_MHz = 1500.0f;
      float max_dm = 1000.0f;
      float total_obs_time_s = 1.0f;

      // Imaging params
      int Ng = 4096;                          ///< UV grid size (Ng x Ng)
      float cell_size_rad = 1e-4f;            ///< UV cell size in radians
      int n_beam_pixels = 256;                ///< Number of beam pixels to extract from image
      imaging_pipeline::FftPrecision fft_prec = imaging_pipeline::FftPrecision::FP32;

      // HERK output precision
      bool herk_output_fp16 = false;           ///< True = FP16 packed triangle output (2x memory savings)

      // CUTLASS tuning params
      int kernel_tune_verbosity = 0;           ///< Kernel-level tuning verbosity (0=off, 2=tune+summary)
      int strategy_tune_verbosity = 1;         ///< Strategy-level tuning verbosity (0=silent, 1=summary, 2=sweep)
      const char* kernel_tune_cache_path = nullptr;   ///< Kernel tune cache file (nullptr = default)
      const char* strategy_tune_cache_path = nullptr;  ///< Strategy tune cache file (nullptr = default)
      bool herk_tune = true;                   ///< Enable HERK and GEMM strategy autotuning
      bool profile = false;                    ///< Enable per-stage timing output
    };

    VisibilityPipeline(const Config &config);
    ~VisibilityPipeline();

    /**
     * @brief Run XEngine correlation on raw voltage data.
     *
     * @param vis_output Device pointer to visibility output (packed triangle)
     * @param raw_input  Host/device pointer to raw voltage input
     */
    void correlate(void *vis_output, void *raw_input);

    /**
     * @brief Transpose filterbank from channel-major to beam-major layout.
     *
     * @param d_out Output [n_beams, n_channels, n_time] float, device
     * @param d_in  Input  [n_channels, n_beams, n_time] float, device
     */
    void corner_turn(float *d_out, const float *d_in);

    /**
     * @brief Run FDD dedispersion on beam-major filterbank data.
     *
     * Requires DEDISP_API_LIB to be defined (linked with libdedisp_api.a).
     *
     * @param d_output     Device pointer to output [batch_size, Ndm, Nt]
     * @param d_input      Device pointer to input  [batch_size, Nf, Nt]
     * @param batch_size   Number of beams to process in this call
     * @return 0 on success, non-zero on failure. Returns -1 if dedisp_api not linked.
     */
    int dedisperse(float *d_output, const float *d_input, int batch_size);

    /**
     * @brief Process multiple payloads with stream-pipelined overlap.
     *
     * Uses 3 CUDA streams (H2D, compute, D2H) and double-buffered device
     * memory to overlap H2D/compute/D2H across successive payloads.
     * HERKs execute sequentially on stream_compute_ (CUTLASS singleton
     * constraint), while H2D and D2H overlap on separate streams.
     *
     * @param vis_outputs  Array of n_payloads host pointers for packed triangle output
     * @param raw_inputs   Array of n_payloads host pointers for raw QC input
     * @param n_payloads   Number of payloads to process
     */
    void correlate_pipelined(void **vis_outputs, void *const *raw_inputs, int n_payloads);

    /**
     * @brief Run the full pipeline: correlate + imaging + corner turn.
     *
     * Runs correlation, then (if configure_imaging() has been called)
     * grids visibilities, runs 2D IFFT, extracts beam pixels, and
     * corner-turns into beam-major layout. Dedispersion is invoked
     * separately via dedisperse() after accumulating enough time samples.
     *
     * @param output Output pointer for correlation results
     * @param input  Input pointer for raw voltage data
     */
    void compute(void *output, void *input);

    /**
     * @brief Configure imaging stage with baseline UV coords and frequencies.
     *
     * Must be called before compute() can run the imaging stage.
     * Allocates imaging pipeline resources and beam output buffer.
     *
     * @param baseline_uv_m  Device pointer to [n_baselines x 2] float UV coords in metres
     * @param n_baselines    Number of baselines (must match N*(N+1)/2)
     * @param freq_hz        Device pointer to [Nf] double frequencies in Hz
     * @param beam_pixels    Device pointer to [n_beam x 2] int pixel coords (row, col)
     * @param n_beam         Number of beam pixels to extract
     * @param cell_size_rad  UV cell size in radians
     * @param stream         CUDA stream
     */
    void configure_imaging(const float* baseline_uv_m, int n_baselines,
                           const double* freq_hz,
                           const int* beam_pixels, int n_beam,
                           float cell_size_rad,
                           cudaStream_t stream = nullptr);

    /// @brief Enable/disable per-stage profiling via TimeProfile.
    void set_profile(bool enabled);

    /// @brief Print and reset accumulated profile data.
    void print_profile();

    /// @brief Get pointer to beam-major filterbank (after compute()).
    const float* filterbank_beammajor() const { return d_filterbank_beammajor_; }

    /// @brief Get pointer to channel-major filterbank (after compute()).
    const float* filterbank_chanmajor() const { return d_filterbank_chanmajor_; }

    // Accessors
    uint64_t n_channels() const { return n_channels_; }
    uint64_t n_beams() const { return n_beams_; }
    uint64_t n_time() const { return n_time_; }
  };

} // namespace ggp
