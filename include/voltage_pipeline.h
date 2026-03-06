#pragma once

#include <cstdint>
#include <cuda_runtime.h>

#ifdef DEDISP_API_LIB
// Forward-declare only; full header included in .cu
namespace dedisp_api { class DedispPipeline; }
#endif

namespace quda { class TimeProfile; }

namespace ggp {

/// @brief Compute precision for the voltage beamformer GEMM.
enum class VoltageComputeMode {
    FP8 = 0,  ///< FP8 E4M3 -- all architectures (SM90/SM100/SM120)
    FP6 = 1   ///< FP6 E3M2 -- SM100/SM120 only, lossless for QC INT4
};

/**
 * @brief End-to-end voltage beamformer pipeline for DSA-2000.
 *
 * Two compute strategies are available:
 *
 * FP8 direct path (default for VoltageComputeMode::FP8):
 *   1. QC INT4 transpose + pol-split (no decode, keeps INT4 byte format)
 *   2. gemm_prepared_power_int4() — PIMPL API converts INT4 -> FP8
 *      interleaved internally, then runs direct complex GEMM with fused
 *      power detection via conjugate permutation trick.
 *      Two launches per payload with beta-accumulation:
 *        pol0: C = |W x V_pol0^T|^2        (beta = 0)
 *        pol1: C += |W x V_pol1^T|^2       (beta = 1)
 *      Computes Stokes I = |pol0|^2 + |pol1|^2 directly.
 *   3. Time integration + uint8 quantisation
 *
 * 4M sub-GEMM fallback path (for FP6/FP4 or when direct is unavailable):
 *   1. QC INT4 decode -> FP16 planar (transpose + pol-split)
 *   2. Batched complex GEMM per polarisation (4M decomposition)
 *   3. Power sum |Re|^2 + |Im|^2, time integration, uint8 quantisation
 *
 * Uses the CUTLASS PIMPL API (cutlass_gemm_api.h) with the prepare_b /
 * gemm_prepared pattern: weights are pre-converted to FP8/FP6 once,
 * voltages are converted per-payload.
 *
 * The CUTLASS dependency is optional -- gated behind CUTLASS_GEMM_API
 * at compile time. Without it, the constructor errors out.
 */
class VoltagePipeline {

private:
    // Configuration
    uint64_t n_antennae_;
    uint64_t n_beams_;
    uint64_t n_channels_;
    uint64_t n_time_;
    uint64_t n_polarizations_;
    uint64_t n_time_power_sum_;
    uint64_t n_time_out_;
    VoltageComputeMode compute_mode_;
    bool use_direct_ = false;  ///< True when using direct GEMM + fused power detect path.
    uint64_t ch_fuse_;                    ///< Channel fusion factor (from Config)
    uint64_t max_payloads_;               ///< Max payloads for batched path (from Config)

    // Device buffers (void* in header -- cast in .cu)
    void *d_weights_re_ = nullptr;    ///< [n_ch x n_beam x n_ant] FP16
    void *d_weights_im_ = nullptr;

    // --- Direct GEMM path (INT4 -> FP8 via PIMPL) ---
    void *d_volt_int4_[2] = {};       ///< [n_ch x n_time x n_ant] INT4 QC bytes, per pol
    void *d_power_ = nullptr;         ///< [n_ch x n_time x n_beam] FP32 power accumulator

    // --- Fused transpose + FP8 path (ch_fuse > 1 or multi-payload) ---
    void *d_volt_fp8_[2] = {};        ///< [batch_fused x M_total x 2*n_ant] FP8 interleaved, per pol
    int64_t d_volt_fp8_cap_ = 0;      ///< Current FP8 buffer capacity in bytes (per pol)

    // --- 4M sub-GEMM fallback path (FP6/FP4) ---
    void *d_volt_re_[2] = {};         ///< [n_ch x n_time x n_ant] FP16, per pol
    void *d_volt_im_[2] = {};
    void *d_beam_re_[2] = {};         ///< [n_ch x n_time x n_beam] FP32, per pol
    void *d_beam_im_[2] = {};

    // --- FP32 filterbank buffers ---
    float *d_filterbank_chanmajor_ = nullptr;  ///< [n_ch x n_beam x n_time_out] FP32
    float *d_filterbank_beammajor_ = nullptr;  ///< [n_beam x n_ch x n_time_out] FP32

    void *gemm_impl_ = nullptr;       ///< Opaque CutlassComplexGemm*
    bool weights_prepared_ = false;
    bool mem_init_ = false;
    bool gemm_tune_ = false;          ///< Whether to pass tune=true to GEMM calls

    // --- dedisp_api (conditional) ---
#ifdef DEDISP_API_LIB
    void *dedisp_pipeline_ = nullptr;
#endif

    // --- CUDA graph cache (direct path only) ---
    bool graph_enabled_ = false;
    void *graph_exec_ = nullptr;      ///< Opaque cudaGraphExec_t
    const uint8_t *graph_last_input_ = nullptr;
    uint8_t *graph_last_output_ = nullptr;
    float graph_last_scale_ = 0.0f;

    quda::TimeProfile *profile_tp_ = nullptr;

    void init_memory();
    void destroy_memory();

    /// @brief Execute beamform body (called directly or during graph capture).
    void beamform_body(uint8_t *d_output, const uint8_t *d_qc_input,
                       float scale, cudaStream_t stream);

    /// @brief Execute fused beamform body for multi-payload batching.
    void beamform_fused_body(uint8_t *d_output,
                             const uint8_t * const *d_qc_inputs,
                             int n_payloads,
                             float scale, cudaStream_t stream);

public:

    /// @brief Configuration for the voltage beamformer pipeline.
    struct Config {
        uint64_t n_antennae = 64;          ///< Number of antennas (GEMM K dimension)
        uint64_t n_beams = 256;            ///< Number of output beams (GEMM N dimension)
        uint64_t n_channels = 4;           ///< Number of frequency channels (batch dimension)
        uint64_t n_time = 1024;            ///< Number of time samples per payload (GEMM M dimension)
        uint64_t n_polarizations = 2;      ///< Number of polarisations (always 2 for DSA-2000)
        uint64_t n_time_power_sum = 4;     ///< Time integration factor for power sum
        VoltageComputeMode compute_mode = VoltageComputeMode::FP8;
        int kernel_tune_verbosity = 0;     ///< Kernel-level tuning verbosity (0-3)
        int strategy_tune_verbosity = 1;   ///< Strategy-level tuning verbosity (0-3)
        bool gemm_tune = true;             ///< Enable GEMM strategy autotuning
        const char* kernel_tune_cache_path = nullptr;  ///< Kernel tune cache file
        const char* strategy_tune_cache_path = nullptr; ///< Strategy/GEMM tune cache file

        // Dedispersion params (used only if dedisp_api linked)
        int n_dm_trials = 256;             ///< Number of DM trials for dedispersion
        float f_min_MHz = 700.0f;          ///< Minimum frequency in MHz
        float f_max_MHz = 1500.0f;         ///< Maximum frequency in MHz
        float max_dm = 1000.0f;            ///< Maximum DM in pc/cm^3
        float total_obs_time_s = 1.0f;     ///< Total observation time in seconds
        bool profile = false;              ///< Enable per-stage timing output

        // Short-integration M-fusion parameters
        uint64_t ch_fuse = 1;             ///< Channel fusion factor (1/2/4/8). Fuses ch_fuse
                                           ///< consecutive channels along M dimension so that
                                           ///< M_fused = n_time * ch_fuse. Requires n_channels
                                           ///< divisible by ch_fuse. B weights must be identical
                                           ///< across fused channels.
        uint64_t max_payloads = 1;         ///< Max payloads for beamform_batched() (sizes FP8
                                           ///< scratch buffer). Set > 1 to enable multi-payload
                                           ///< batching along M: M_total = n_time * ch_fuse *
                                           ///< n_payloads. Range: 1-16.
    };

    /// @brief Construct and allocate internal buffers.
    /// @param config  Pipeline configuration.
    /// @pre config.n_time must be divisible by config.n_time_power_sum.
    /// @pre config.n_polarizations must be 2.
    explicit VoltagePipeline(const Config &config);
    ~VoltagePipeline();

    /// @brief Load complex beamforming weights and pre-prepare for CUTLASS.
    ///
    /// Converts FP32 interleaved complex weights to FP16 planar format, then
    /// calls prepare_b() to pre-convert to FP8/FP6 for reuse.
    ///
    /// @param d_weights  Device pointer to FP32 interleaved complex weights.
    ///                   Layout: [n_channels x n_beams x n_antennae x 2] floats.
    /// @param stream     Optional CUDA stream.
    void set_weights(const float *d_weights, cudaStream_t stream = nullptr);

    /// @brief Process one payload: QC voltages -> uint8 beam powers.
    ///
    /// For FP8 compute mode, uses the direct GEMM path with fused power
    /// detection. For FP6, uses the 4M sub-GEMM fallback.
    ///
    /// When CUDA graph capture is enabled (set_graph_enabled(true)), the
    /// first call captures the kernel sequence into a graph. Subsequent
    /// calls with the same pointers and scale replay the graph, eliminating
    /// host-side launch overhead.
    ///
    /// @param d_output    Device pointer to uint8 output.
    ///                    Layout: [n_channels x n_beams x n_time_out] uint8.
    /// @param d_qc_input  Device pointer to QC INT4 input voltages.
    ///                    Layout: [n_channels x n_pol x n_antennae x n_time] bytes.
    /// @param scale       Quantisation scale factor applied before clamping to [0,255].
    /// @param stream      Optional CUDA stream.
    /// @pre set_weights() must have been called.
    void beamform(uint8_t *d_output, const uint8_t *d_qc_input,
                  float scale = 1.0f, cudaStream_t stream = nullptr);

    /// @brief Process multiple payloads in a single GEMM call.
    ///
    /// Fuses ch_fuse consecutive channels along M and concatenates n_payloads
    /// payloads along M, so the GEMM sees M_total = n_time * ch_fuse * n_payloads.
    /// Requires all payloads to use the same weights (set_weights() not called
    /// between payloads).
    ///
    /// Uses a fused kernel that performs QC INT4 transpose + pol-split +
    /// channel fusion + payload batching + INT4->FP8 conversion in a single
    /// pass, then calls gemm_prepared_power_fp8() directly (skipping the
    /// PIMPL's internal INT4->FP8 re-conversion).
    ///
    /// @param d_output      Device pointer to uint8 output.
    ///                      Layout: [n_payloads x n_channels x n_beams x n_time_out] uint8.
    /// @param d_qc_inputs   Device pointer to array of n_payloads device pointers,
    ///                      each pointing to a QC payload of input_size() bytes.
    /// @param n_payloads    Number of payloads to batch (1 to max_payloads from Config).
    /// @param scale         Quantisation scale factor applied before clamping to [0,255].
    /// @param stream        Optional CUDA stream.
    /// @pre set_weights() must have been called.
    /// @pre n_payloads <= max_payloads from Config.
    /// @pre ch_fuse > 1 or n_payloads > 1 (otherwise use beamform() directly).
    void beamform_batched(uint8_t *d_output, const uint8_t * const *d_qc_inputs,
                          int n_payloads, float scale = 1.0f,
                          cudaStream_t stream = nullptr);

    /// @brief Get the channel fusion factor.
    uint64_t ch_fuse() const { return ch_fuse_; }

    /// @brief Get the output size in bytes for one payload.
    int64_t output_size() const;

    /// @brief Get the input QC payload size in bytes.
    int64_t input_size() const;

    /// @brief Enable/disable CUDA graph capture for repeated beamform() calls.
    ///
    /// When enabled, the first beamform() call captures the kernel sequence
    /// into a CUDA graph. Subsequent calls with the same pointers replay
    /// the graph, eliminating host-side launch overhead.
    /// If pointers or scale change, the graph is re-captured automatically.
    /// Only effective for the direct FP8 path.
    ///
    /// @param enabled  true to enable graph capture.
    void set_graph_enabled(bool enabled) { graph_enabled_ = enabled; }

    /// @brief Check whether CUDA graph capture is enabled.
    bool graph_enabled() const { return graph_enabled_; }

    /// @brief Check whether the direct FP8 GEMM path is active.
    bool uses_direct_path() const { return use_direct_; }

    /// @brief Enable/disable per-stage profiling via TimeProfile.
    void set_profile(bool enabled);

    /// @brief Print and reset accumulated profile data.
    void print_profile();

    /// @brief Beamform to FP32 filterbank (no uint8 quantisation).
    ///
    /// Runs the same beamform stages as beamform() but outputs FP32
    /// power values to the internal d_filterbank_chanmajor_ buffer
    /// in [n_ch, n_beam, n_time_out] layout.
    ///
    /// @param d_qc_input  Device pointer to QC INT4 input voltages.
    /// @param stream      Optional CUDA stream.
    /// @pre set_weights() must have been called.
    void compute_filterbank(const uint8_t *d_qc_input,
                            cudaStream_t stream = nullptr);

    /// @brief Full pipeline: beamform -> FP32 filterbank -> corner turn -> dedisp.
    ///
    /// 1. compute_filterbank() -> d_filterbank_chanmajor_
    /// 2. corner_turn()        -> d_filterbank_beammajor_
    /// 3. dedisperse() (if d_dedisp_out != nullptr and dedisp_api linked)
    /// 4. quantize FP32 -> uint8 output
    ///
    /// @param d_output      uint8 filterbank output [n_ch x n_beam x n_time_out]
    /// @param d_qc_input    QC INT4 input voltages
    /// @param d_dedisp_out  Optional dedisp output [n_beam x n_dm x n_time_out], nullptr to skip
    /// @param scale         Quantisation scale for uint8 output
    /// @param stream        Optional CUDA stream.
    /// @pre set_weights() must have been called.
    void compute(uint8_t *d_output, const uint8_t *d_qc_input,
                 float *d_dedisp_out = nullptr,
                 float scale = 1.0f,
                 cudaStream_t stream = nullptr);

    /// @brief Corner turn: [n_ch, n_beam, n_time_out] -> [n_beam, n_ch, n_time_out].
    /// @param d_out  Output device buffer.
    /// @param d_in   Input device buffer.
    /// @param stream Optional CUDA stream.
    void corner_turn(float *d_out, const float *d_in,
                     cudaStream_t stream = nullptr);

    /// @brief Dedisperse beam-major filterbank via dedisp_api.
    /// @param d_output    Output [batch_size x Ndm x n_time_out]
    /// @param d_input     Input [batch_size x n_ch x n_time_out]
    /// @param batch_size  Number of beams to process
    /// @param stream      Optional CUDA stream
    /// @return 0 on success, -1 if dedisp_api not linked
    int dedisperse(float *d_output, const float *d_input, int batch_size,
                   cudaStream_t stream = nullptr);

    // Accessors
    uint64_t n_channels() const { return n_channels_; }
    uint64_t n_beams() const { return n_beams_; }
    uint64_t n_time() const { return n_time_; }
    uint64_t n_time_out() const { return n_time_out_; }
    const float* filterbank_chanmajor() const { return d_filterbank_chanmajor_; }
    const float* filterbank_beammajor() const { return d_filterbank_beammajor_; }

    // Non-copyable, non-movable
    VoltagePipeline(const VoltagePipeline&) = delete;
    VoltagePipeline& operator=(const VoltagePipeline&) = delete;
};

} // namespace ggp
