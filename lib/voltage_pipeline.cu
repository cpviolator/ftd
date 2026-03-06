/**
 * @file voltage_pipeline.cu
 * @brief DSA-2000 voltage beamformer pipeline implementation.
 *
 * Contains custom CUDA kernels and CUTLASS GEMM orchestration for two paths:
 *
 * Direct FP8 path (default for FP8 compute):
 *   1. qc_transpose_polsplit — QC INT4 transpose + pol-split (no decode,
 *      keeps INT4 byte format; half the memory of FP8 interleaved)
 *   2. gemm_prepared_power_int4 — PIMPL API converts INT4→FP8 internally,
 *      then runs direct GEMM with fused power detection
 *      (conjugate permutation trick, two launches for Stokes I via beta)
 *   3. time_integrate_quantize — time integration + uint8 quantisation
 *
 * 4M sub-GEMM fallback path (FP6/FP4):
 *   1. qc_to_fp16_planar_polsplit — QC INT4 decode + transpose + pol-split
 *   2. gemm_prepared × 2 — 4M sub-GEMM per polarisation
 *   3. power_sum_quantize — |A|^2 + |B|^2, time integration, uint8 output
 *
 * The GEMM uses the prepare_b / gemm_prepared pattern from cutlass_gemm_api.h
 * so that weights are pre-converted to FP8/FP6 once and reused.
 *
 * CUDA graph support: for the direct path, the first beamform() call can be
 * captured into a CUDA graph and replayed on subsequent calls with the same
 * parameters, eliminating host-side launch overhead.
 *
 * CUTLASS dependency is gated behind CUTLASS_GEMM_API.
 * For standalone testing (without full GGP build), define STANDALONE_TEST.
 */

#include <voltage_pipeline.h>
#include <corner_turn.h>
#include <cstdio>
#include <cstdint>
#include <algorithm>
#include <stdexcept>
#include <cuda_fp16.h>

#ifdef DEDISP_API_LIB
#include <dedisp_api.h>
#endif

// ========================================================================
// Error/log macros: FTD conventions or standalone fallback
// ========================================================================

#ifdef STANDALONE_TEST
  // Standalone: no QUDA/GGP dependencies
  #define VP_ERROR(fmt, ...) do { \
      fprintf(stderr, "VoltagePipeline ERROR: " fmt "\n", ##__VA_ARGS__); \
      throw std::runtime_error("VoltagePipeline error"); \
  } while(0)
  #define VP_LOG(fmt, ...) fprintf(stderr, "VoltagePipeline: " fmt "\n", ##__VA_ARGS__)
  // Profile macros are no-ops in standalone mode (no quda::TimeProfile)
  #define VP_PROFILE_START(stage) ((void)0)
  #define VP_PROFILE_STOP(stage) ((void)0)
  #define VP_PROFILE_SYNC_STOP_START(stop_stage, start_stage) ((void)0)
  #define VP_PROFILE_SYNC_STOP(stage) ((void)0)
#else
  // FTD build: use QUDA error/log
  #include <util_ggp.h>
  #include <timer.h>
  #define VP_ERROR(fmt, ...) errorQuda(fmt, ##__VA_ARGS__)
  #define VP_LOG(fmt, ...) logQuda(QUDA_VERBOSE, fmt, ##__VA_ARGS__)
  // Profile macros wrapping quda::TimeProfile
  #define VP_PROFILE_START(stage) \
      if (profile_tp_) profile_tp_->TPSTART(quda::stage)
  #define VP_PROFILE_STOP(stage) \
      if (profile_tp_) profile_tp_->TPSTOP(quda::stage)
  #define VP_PROFILE_SYNC_STOP_START(stop_stage, start_stage) \
      if (profile_tp_) { \
          VP_CHECK_CUDA(cudaDeviceSynchronize()); \
          profile_tp_->TPSTOP(quda::stop_stage); \
          profile_tp_->TPSTART(quda::start_stage); \
      }
  #define VP_PROFILE_SYNC_STOP(stage) \
      if (profile_tp_) { \
          VP_CHECK_CUDA(cudaDeviceSynchronize()); \
          profile_tp_->TPSTOP(quda::stage); \
      }
#endif

#define VP_CHECK_CUDA(x) do { \
    cudaError_t err = (x); \
    if (err != cudaSuccess) { \
        VP_ERROR("CUDA error at %s:%d: %s", __FILE__, __LINE__, cudaGetErrorString(err)); \
    } \
} while(0)

// ========================================================================
// CUDA Kernels (architecture-independent, no CUTLASS)
// ========================================================================

namespace {

// --------------------------------------------------------------------
// Kernel: QC INT4 -> FP16 planar with polarisation split and transpose
// (used by 4M sub-GEMM fallback path)
// --------------------------------------------------------------------

/// @brief Decode QC sign-magnitude bytes to FP16, split polarisations,
///        and transpose from [ant x time] to [time x ant] per batch.
///
/// Input layout:  [n_ch x n_pol x n_ant x n_time] bytes
/// Output layout: [n_ch x n_time x n_ant] __half (separate Re/Im, per pol)
__global__ void qc_to_fp16_planar_polsplit_kernel(
    const uint8_t* __restrict__ qc_data,
    __half* __restrict__ volt_re_0,
    __half* __restrict__ volt_im_0,
    __half* __restrict__ volt_re_1,
    __half* __restrict__ volt_im_1,
    int n_ant, int n_time, int n_ch)
{
    const int64_t total = (int64_t)n_ch * n_time * n_ant;
    for (int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
         i < total;
         i += (int64_t)gridDim.x * blockDim.x)
    {
        const int ant  = (int)(i % n_ant);
        const int time = (int)((i / n_ant) % n_time);
        const int chan = (int)(i / ((int64_t)n_ant * n_time));

        const int64_t batch_offset = (int64_t)chan * 2 * n_ant * n_time;
        const int64_t elem_offset  = (int64_t)ant * n_time + time;
        const int64_t pol_stride   = (int64_t)n_ant * n_time;

        const uint8_t byte_0 = qc_data[batch_offset + elem_offset];
        const uint8_t byte_1 = qc_data[batch_offset + pol_stride + elem_offset];

        auto decode = [](uint8_t raw) -> float {
            int sign = (raw & 0x8) ? -1 : 1;
            int mag  = raw & 0x7;
            return (float)(sign * mag);
        };

        const float re_0 = decode((byte_0 >> 4) & 0x0F);
        const float im_0 = decode(byte_0 & 0x0F);
        const float re_1 = decode((byte_1 >> 4) & 0x0F);
        const float im_1 = decode(byte_1 & 0x0F);

        volt_re_0[i] = __float2half(re_0);
        volt_im_0[i] = __float2half(im_0);
        volt_re_1[i] = __float2half(re_1);
        volt_im_1[i] = __float2half(im_1);
    }
}

// --------------------------------------------------------------------
// LUT: INT4 sign-magnitude nibble -> FP8 E4M3 byte
// 16 entries: nibble 0-7 -> +0..+7, nibble 8-15 -> -0..-7
// Same mapping as cast_int4_to_fp8_interleaved in cutlass_gemm_api.cu.
// --------------------------------------------------------------------

__device__ __constant__ uint8_t int4_to_fp8_lut[16] = {
    0x00, 0x38, 0x40, 0x44, 0x48, 0x4A, 0x4C, 0x4E,  // 0..7
    0x00, 0xB8, 0xC0, 0xC4, 0xC8, 0xCA, 0xCC, 0xCE   // -0..-7
};

// --------------------------------------------------------------------
// Kernel: Fused QC transpose + pol-split + ch-fuse + payload-batch + INT4->FP8
// (eliminates separate transpose and cast passes for short-integration)
// --------------------------------------------------------------------

/// @brief Fused QC transpose + pol-split + channel-fuse + payload-batch + INT4->FP8.
///
/// Input:  n_payloads contiguous payloads, each [n_ch x 2 x n_ant x n_time] bytes (QC)
/// Output: [n_ch/ch_fuse x M_total x 2*n_ant] FP8 E4M3 interleaved, per pol
///         where M_total = n_time * ch_fuse * n_payloads
///
/// M-axis ordering (fastest to slowest within each output batch element):
///   time (0..n_time-1), then ch_in_group (0..ch_fuse-1), then payload (0..n_payloads-1)
///
/// Each thread processes one (output_batch, m_idx, ant) element, reading from the
/// appropriate payload/channel/pol/ant/time location and writing interleaved FP8
/// [Re_fp8, Im_fp8] to the output.
__global__ void qc_fused_transpose_fp8_kernel(
    const uint8_t* __restrict__ qc_data,
    uint8_t* __restrict__ fp8_pol0,
    uint8_t* __restrict__ fp8_pol1,
    int n_ant, int n_time, int n_ch,
    int ch_fuse, int n_payloads,
    int64_t payload_stride)
{
    const int batch_fused = n_ch / ch_fuse;
    const int M_total = n_time * ch_fuse * n_payloads;
    const int64_t total = (int64_t)batch_fused * M_total * n_ant;

    for (int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
         i < total;
         i += (int64_t)gridDim.x * blockDim.x)
    {
        // Decompose flat index -> (batch_idx, m_idx, ant)
        const int ant       = (int)(i % n_ant);
        const int m_idx     = (int)((i / n_ant) % M_total);
        const int batch_idx = (int)(i / ((int64_t)n_ant * M_total));

        // Decompose m_idx -> (time, ch_in_group, payload)
        const int time        = m_idx % n_time;
        const int ch_in_group = (m_idx / n_time) % ch_fuse;
        const int payload     = m_idx / (n_time * ch_fuse);

        // Source channel index
        const int chan = batch_idx * ch_fuse + ch_in_group;

        // QC input: payload_offset + [chan, pol, ant, time]
        const int64_t payload_offset = (int64_t)payload * payload_stride;
        const int64_t ch_offset = (int64_t)chan * 2 * n_ant * n_time;
        const int64_t elem_offset = (int64_t)ant * n_time + time;
        const int64_t pol_stride = (int64_t)n_ant * n_time;

        const uint8_t byte_0 = qc_data[payload_offset + ch_offset + elem_offset];
        const uint8_t byte_1 = qc_data[payload_offset + ch_offset + pol_stride + elem_offset];

        // Decode INT4 sign-magnitude nibbles -> FP8 E4M3 via LUT
        const uint8_t re_fp8_0 = int4_to_fp8_lut[(byte_0 >> 4) & 0x0F];
        const uint8_t im_fp8_0 = int4_to_fp8_lut[byte_0 & 0x0F];
        const uint8_t re_fp8_1 = int4_to_fp8_lut[(byte_1 >> 4) & 0x0F];
        const uint8_t im_fp8_1 = int4_to_fp8_lut[byte_1 & 0x0F];

        // Output: [batch_fused x M_total x 2*n_ant] FP8 interleaved (Re, Im pairs)
        const int64_t out_idx = ((int64_t)batch_idx * M_total + m_idx) * (2 * n_ant)
                              + 2 * ant;

        fp8_pol0[out_idx]     = re_fp8_0;
        fp8_pol0[out_idx + 1] = im_fp8_0;
        fp8_pol1[out_idx]     = re_fp8_1;
        fp8_pol1[out_idx + 1] = im_fp8_1;
    }
}

// --------------------------------------------------------------------
// Kernel: QC INT4 transpose + pol-split (used by direct GEMM path)
// The INT4→FP8 conversion is now handled by the PIMPL API internally.
// --------------------------------------------------------------------

/// @brief Kernel: QC INT4 transpose + pol-split (no decode).
///
/// Transposes QC data from [ch, pol, ant, time] to [ch, time, ant] layout
/// and splits polarisations, keeping INT4 byte format (1 byte = 1 complex).
/// The PIMPL API's gemm_prepared_power_int4() handles INT4→FP8 conversion
/// internally.
///
/// Input layout:  [n_ch x 2 x n_ant x n_time] bytes (QC format)
/// Output layout: [n_ch x n_time x n_ant] bytes (QC format), per pol
__global__ void qc_transpose_polsplit_kernel(
    const uint8_t* __restrict__ qc_data,
    uint8_t* __restrict__ int4_pol0,
    uint8_t* __restrict__ int4_pol1,
    int n_ant, int n_time, int n_ch)
{
    const int64_t total = (int64_t)n_ch * n_time * n_ant;
    for (int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
         i < total;
         i += (int64_t)gridDim.x * blockDim.x)
    {
        const int ant  = (int)(i % n_ant);
        const int time = (int)((i / n_ant) % n_time);
        const int chan = (int)(i / ((int64_t)n_ant * n_time));

        // QC input: [chan, pol, ant, time]
        const int64_t ch_offset = (int64_t)chan * 2 * n_ant * n_time;
        const int64_t elem_offset = (int64_t)ant * n_time + time;
        const int64_t pol_stride = (int64_t)n_ant * n_time;

        // Just copy bytes — no decode, keep INT4 format
        int4_pol0[i] = qc_data[ch_offset + elem_offset];
        int4_pol1[i] = qc_data[ch_offset + pol_stride + elem_offset];
    }
}

// --------------------------------------------------------------------
// Kernel: Time integration + quantise (for direct GEMM power output)
// --------------------------------------------------------------------

/// @brief Integrate pre-computed power over time bins and quantise to uint8.
///
/// Used with the direct GEMM path where |Re|^2 + |Im|^2 is already fused
/// into the GEMM store phase, and both polarisations are accumulated via
/// beta into a single power buffer.
///
/// Input:  d_power [n_ch x n_time x n_beam] FP32 (Stokes I power per sample)
/// Output: [n_ch x n_beam x n_time_out] uint8
__global__ void time_integrate_quantize_kernel(
    const float* __restrict__ power,
    uint8_t* __restrict__ output,
    int n_beam, int n_time, int n_ch,
    int n_time_power_sum,
    float scale)
{
    const int n_time_out = n_time / n_time_power_sum;
    const int64_t total = (int64_t)n_ch * n_beam * n_time_out;

    for (int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
         i < total;
         i += (int64_t)gridDim.x * blockDim.x)
    {
        // Output index: [chan, beam, t_out]
        const int t_out = (int)(i % n_time_out);
        const int beam  = (int)((i / n_time_out) % n_beam);
        const int chan  = (int)(i / ((int64_t)n_time_out * n_beam));

        // Sum power over n_time_power_sum consecutive time samples
        float acc = 0.0f;
        for (int dt = 0; dt < n_time_power_sum; ++dt) {
            const int t = t_out * n_time_power_sum + dt;
            // Power layout: [chan x time x beam]
            const int64_t idx = (int64_t)chan * n_time * n_beam
                              + (int64_t)t * n_beam + beam;
            acc += power[idx];
        }

        // Scale and quantise to [0, 255]
        const float scaled = acc * scale;
        const int val = __float2int_rn(scaled);
        output[i] = static_cast<uint8_t>(min(max(val, 0), 255));
    }
}

// --------------------------------------------------------------------
// Kernel: Power sum + time integration + quantise (4M fallback path)
// --------------------------------------------------------------------

/// @brief Compute beam power = |A|^2 + |B|^2, integrate over time, quantise to uint8.
///
/// GEMM output layout: [n_ch x n_time x n_beam] FP32 planar (per pol).
/// Output layout:       [n_ch x n_beam x n_time_out] uint8
__global__ void power_sum_quantize_kernel(
    const float* __restrict__ beam_re_0,
    const float* __restrict__ beam_im_0,
    const float* __restrict__ beam_re_1,
    const float* __restrict__ beam_im_1,
    uint8_t* __restrict__ output,
    int n_beam, int n_time, int n_ch,
    int n_time_power_sum,
    float scale)
{
    const int n_time_out = n_time / n_time_power_sum;
    const int64_t total = (int64_t)n_ch * n_beam * n_time_out;

    for (int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
         i < total;
         i += (int64_t)gridDim.x * blockDim.x)
    {
        const int t_out = (int)(i % n_time_out);
        const int beam  = (int)((i / n_time_out) % n_beam);
        const int chan  = (int)(i / ((int64_t)n_time_out * n_beam));

        float power = 0.0f;
        for (int dt = 0; dt < n_time_power_sum; ++dt) {
            const int t = t_out * n_time_power_sum + dt;
            const int64_t idx = (int64_t)chan * n_time * n_beam
                              + (int64_t)t * n_beam + beam;

            const float ra = beam_re_0[idx];
            const float ia = beam_im_0[idx];
            const float rb = beam_re_1[idx];
            const float ib = beam_im_1[idx];

            power += ra * ra + ia * ia + rb * rb + ib * ib;
        }

        const float scaled = power * scale;
        const int val = __float2int_rn(scaled);
        output[i] = static_cast<uint8_t>(min(max(val, 0), 255));
    }
}

// --------------------------------------------------------------------
// Kernel: Time integration -> FP32 output (for direct GEMM path, no quantisation)
// --------------------------------------------------------------------

/// @brief Integrate pre-computed power over time bins, output FP32 filterbank.
///
/// Same accumulation as time_integrate_quantize_kernel but outputs float.
/// Used by compute_filterbank() for the direct FP8 path.
///
/// Input:  d_power [n_ch x n_time x n_beam] FP32 (Stokes I power per sample)
/// Output: d_filterbank [n_ch x n_beam x n_time_out] FP32
__global__ void time_integrate_fp32_kernel(
    const float* __restrict__ power,
    float* __restrict__ output,
    int n_beam, int n_time, int n_ch,
    int n_time_power_sum)
{
    const int n_time_out = n_time / n_time_power_sum;
    const int64_t total = (int64_t)n_ch * n_beam * n_time_out;

    for (int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
         i < total;
         i += (int64_t)gridDim.x * blockDim.x)
    {
        const int t_out = (int)(i % n_time_out);
        const int beam  = (int)((i / n_time_out) % n_beam);
        const int chan  = (int)(i / ((int64_t)n_time_out * n_beam));

        float acc = 0.0f;
        for (int dt = 0; dt < n_time_power_sum; ++dt) {
            const int t = t_out * n_time_power_sum + dt;
            const int64_t idx = (int64_t)chan * n_time * n_beam
                              + (int64_t)t * n_beam + beam;
            acc += power[idx];
        }

        output[i] = acc;
    }
}

// --------------------------------------------------------------------
// Kernel: Power sum + time integration -> FP32 output (4M fallback, no quantisation)
// --------------------------------------------------------------------

/// @brief Compute beam power = |A|^2 + |B|^2, integrate over time, output FP32.
///
/// Same as power_sum_quantize_kernel but outputs float without scale/clamp.
/// Used by compute_filterbank() for the 4M sub-GEMM fallback path.
///
/// GEMM output layout: [n_ch x n_time x n_beam] FP32 planar (per pol).
/// Output layout:       [n_ch x n_beam x n_time_out] FP32
__global__ void power_sum_fp32_kernel(
    const float* __restrict__ beam_re_0,
    const float* __restrict__ beam_im_0,
    const float* __restrict__ beam_re_1,
    const float* __restrict__ beam_im_1,
    float* __restrict__ output,
    int n_beam, int n_time, int n_ch,
    int n_time_power_sum)
{
    const int n_time_out = n_time / n_time_power_sum;
    const int64_t total = (int64_t)n_ch * n_beam * n_time_out;

    for (int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
         i < total;
         i += (int64_t)gridDim.x * blockDim.x)
    {
        const int t_out = (int)(i % n_time_out);
        const int beam  = (int)((i / n_time_out) % n_beam);
        const int chan  = (int)(i / ((int64_t)n_time_out * n_beam));

        float power = 0.0f;
        for (int dt = 0; dt < n_time_power_sum; ++dt) {
            const int t = t_out * n_time_power_sum + dt;
            const int64_t idx = (int64_t)chan * n_time * n_beam
                              + (int64_t)t * n_beam + beam;

            const float ra = beam_re_0[idx];
            const float ia = beam_im_0[idx];
            const float rb = beam_re_1[idx];
            const float ib = beam_im_1[idx];

            power += ra * ra + ia * ia + rb * rb + ib * ib;
        }

        output[i] = power;
    }
}

// --------------------------------------------------------------------
// Kernel: Quantise FP32 -> uint8
// --------------------------------------------------------------------

/// @brief Scale FP32 values and clamp to [0,255] uint8.
///
/// Used to produce char-sized filterbank output from FP32 filterbank data.
__global__ void quantize_fp32_to_uint8_kernel(
    const float* __restrict__ input,
    uint8_t* __restrict__ output,
    float scale,
    int64_t total)
{
    for (int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
         i < total; i += (int64_t)gridDim.x * blockDim.x)
    {
        float val = input[i] * scale;
        int ival = __float2int_rn(val);
        output[i] = (uint8_t)min(max(ival, 0), 255);
    }
}

// --------------------------------------------------------------------
// Kernel: Deinterleave FP32 complex -> FP16 planar (for weight conversion)
// --------------------------------------------------------------------

/// @brief Device kernel: deinterleave FP32 [Re,Im,...] -> FP16 planar Re[], Im[].
__global__ void deinterleave_fp32_to_fp16_kernel(
    const float* __restrict__ input,
    __half* __restrict__ out_re,
    __half* __restrict__ out_im,
    int64_t n_complex)
{
    for (int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
         i < n_complex;
         i += (int64_t)gridDim.x * blockDim.x)
    {
        out_re[i] = __float2half(input[2 * i]);
        out_im[i] = __float2half(input[2 * i + 1]);
    }
}

} // anonymous namespace

// ========================================================================
// CUTLASS path
// ========================================================================

#ifdef CUTLASS_GEMM_API

#include "cutlass_gemm_api.h"

namespace ggp {

VoltagePipeline::VoltagePipeline(const Config &config) {
    n_antennae_       = config.n_antennae;
    n_beams_          = config.n_beams;
    n_channels_       = config.n_channels;
    n_time_           = config.n_time;
    n_polarizations_  = config.n_polarizations;
    n_time_power_sum_ = config.n_time_power_sum;
    compute_mode_     = config.compute_mode;
    ch_fuse_          = config.ch_fuse;
    max_payloads_     = config.max_payloads;

    if (n_polarizations_ != 2)
        VP_ERROR("VoltagePipeline requires n_polarizations == 2");
    if (n_time_ % n_time_power_sum_ != 0)
        VP_ERROR("n_time (%llu) must be divisible by n_time_power_sum (%llu)",
                 (unsigned long long)n_time_, (unsigned long long)n_time_power_sum_);
    if (n_antennae_ < 1 || n_beams_ < 1 || n_channels_ < 1 || n_time_ < 1)
        VP_ERROR("All dimensions must be >= 1");
    if (ch_fuse_ < 1 || (ch_fuse_ & (ch_fuse_ - 1)) != 0)
        VP_ERROR("ch_fuse must be a power of 2, got %llu", (unsigned long long)ch_fuse_);
    if (n_channels_ % ch_fuse_ != 0)
        VP_ERROR("n_channels (%llu) must be divisible by ch_fuse (%llu)",
                 (unsigned long long)n_channels_, (unsigned long long)ch_fuse_);
    if (max_payloads_ < 1 || max_payloads_ > 16)
        VP_ERROR("max_payloads must be 1-16, got %llu", (unsigned long long)max_payloads_);

    n_time_out_ = n_time_ / n_time_power_sum_;

    // Direct GEMM path available for FP8 compute only
    use_direct_ = (compute_mode_ == VoltageComputeMode::FP8);

    // Create CUTLASS GEMM API instance (owns its own internal state)
    auto *api = new cutlass_gemm_api::CutlassComplexGemm();
    if (config.kernel_tune_verbosity > 0)
        api->set_kernel_tune_verbosity(config.kernel_tune_verbosity);
    if (config.kernel_tune_cache_path)
        api->set_kernel_tune_cache_path(config.kernel_tune_cache_path);
    api->set_strategy_tune_verbosity(config.strategy_tune_verbosity);
    api->set_gemm_tune_verbosity(config.strategy_tune_verbosity);
    if (config.strategy_tune_cache_path) {
        api->set_tune_cache_path(config.strategy_tune_cache_path);
        api->set_gemm_tune_cache_path(config.strategy_tune_cache_path);
    }
    gemm_tune_ = config.gemm_tune;
    gemm_impl_ = static_cast<void*>(api);

    if (config.profile) set_profile(true);

    init_memory();

#ifdef DEDISP_API_LIB
    // Create dedispersion pipeline
    dedisp_api::DedispConfig dedisp_config;
    dedisp_config.Nf = static_cast<int>(n_channels_);
    dedisp_config.Nt = static_cast<int>(n_time_out_);
    dedisp_config.Ndm = config.n_dm_trials;
    dedisp_config.f_min_MHz = config.f_min_MHz;
    dedisp_config.f_max_MHz = config.f_max_MHz;
    dedisp_config.max_dm = config.max_dm;
    dedisp_config.total_obs_time_s = config.total_obs_time_s;
    dedisp_config.compute_mode = dedisp_api::ComputeMode::CuBLAS_FP32;
    dedisp_config.max_batch_size = static_cast<int>(n_beams_);
    dedisp_config.kernel_tune_verbosity = config.kernel_tune_verbosity;
    dedisp_config.kernel_tune_cache_path = config.kernel_tune_cache_path;

    auto *dedisp_pipe = new dedisp_api::DedispPipeline(dedisp_config);
    dedisp_pipe->initialize();
    dedisp_pipeline_ = static_cast<void*>(dedisp_pipe);
#endif
}

VoltagePipeline::~VoltagePipeline() {
#ifndef STANDALONE_TEST
    if (profile_tp_) {
        delete profile_tp_;
        profile_tp_ = nullptr;
    }
#endif
    destroy_memory();
    if (gemm_impl_) {
        delete static_cast<cutlass_gemm_api::CutlassComplexGemm*>(gemm_impl_);
        gemm_impl_ = nullptr;
    }
#ifdef DEDISP_API_LIB
    if (dedisp_pipeline_) {
        delete static_cast<dedisp_api::DedispPipeline*>(dedisp_pipeline_);
        dedisp_pipeline_ = nullptr;
    }
#endif
}

void VoltagePipeline::set_profile(bool enabled) {
#ifndef STANDALONE_TEST
    if (enabled && !profile_tp_) {
        graph_enabled_ = false;  // disable graph capture for accurate profiling
        profile_tp_ = new quda::TimeProfile("VoltagePipeline", false);
    } else if (!enabled && profile_tp_) {
        delete profile_tp_;
        profile_tp_ = nullptr;
    }
#else
    (void)enabled;
#endif
}

void VoltagePipeline::print_profile() {
#ifndef STANDALONE_TEST
    if (profile_tp_) {
        profile_tp_->Print();
        profile_tp_->TPRESET();
    }
#endif
}

void VoltagePipeline::init_memory() {
    if (mem_init_) return;

    const int64_t weight_elems = (int64_t)n_channels_ * n_beams_ * n_antennae_;
    const int64_t volt_elems   = (int64_t)n_channels_ * n_time_ * n_antennae_;
    const int64_t beam_elems   = (int64_t)n_channels_ * n_time_ * n_beams_;

    // FP32 filterbank buffers
    const int64_t fb_elems = (int64_t)n_channels_ * n_beams_ * n_time_out_;
    VP_CHECK_CUDA(cudaMalloc(&d_filterbank_chanmajor_, fb_elems * sizeof(float)));
    VP_CHECK_CUDA(cudaMalloc(&d_filterbank_beammajor_, fb_elems * sizeof(float)));

    // Weight buffers (FP16 planar) -- always needed for set_weights()
    __half *wre, *wim;
    VP_CHECK_CUDA(cudaMalloc(&wre, weight_elems * sizeof(__half)));
    VP_CHECK_CUDA(cudaMalloc(&wim, weight_elems * sizeof(__half)));
    d_weights_re_ = wre;
    d_weights_im_ = wim;

    if (use_direct_) {
        // Direct path: INT4 QC voltage buffers (1 byte per complex element)
        const int64_t int4_volt_bytes = volt_elems;  // packed INT4 QC format
        for (int p = 0; p < 2; ++p) {
            uint8_t *buf;
            VP_CHECK_CUDA(cudaMalloc(&buf, int4_volt_bytes));
            d_volt_int4_[p] = buf;
        }

        // When using fused path (ch_fuse > 1 or multi-payload), need FP8 + larger power buf
        const bool use_fused = (ch_fuse_ > 1 || max_payloads_ > 1);
        if (use_fused) {
            const int64_t batch_fused = (int64_t)n_channels_ / ch_fuse_;
            const int64_t M_total = (int64_t)n_time_ * ch_fuse_ * max_payloads_;
            // FP8 interleaved: [batch_fused x M_total x 2*n_ant] bytes per pol
            const int64_t fp8_bytes = batch_fused * M_total * 2 * (int64_t)n_antennae_;
            for (int p = 0; p < 2; ++p) {
                uint8_t *buf;
                VP_CHECK_CUDA(cudaMalloc(&buf, fp8_bytes));
                d_volt_fp8_[p] = buf;
            }
            d_volt_fp8_cap_ = fp8_bytes;

            // Power buffer sized for M_total (larger than single-payload)
            const int64_t power_elems = batch_fused * M_total * (int64_t)n_beams_;
            float *power;
            VP_CHECK_CUDA(cudaMalloc(&power, power_elems * sizeof(float)));
            d_power_ = power;
        } else {
            // Single power buffer for fused |Re|^2 + |Im|^2 output
            float *power;
            VP_CHECK_CUDA(cudaMalloc(&power, beam_elems * sizeof(float)));
            d_power_ = power;
        }
    } else {
        // 4M fallback: FP16 planar voltage scratch + separate Re/Im output
        for (int p = 0; p < 2; ++p) {
            __half *vre, *vim;
            float *bre, *bim;
            VP_CHECK_CUDA(cudaMalloc(&vre, volt_elems * sizeof(__half)));
            VP_CHECK_CUDA(cudaMalloc(&vim, volt_elems * sizeof(__half)));
            VP_CHECK_CUDA(cudaMalloc(&bre, beam_elems * sizeof(float)));
            VP_CHECK_CUDA(cudaMalloc(&bim, beam_elems * sizeof(float)));
            d_volt_re_[p] = vre;
            d_volt_im_[p] = vim;
            d_beam_re_[p] = bre;
            d_beam_im_[p] = bim;
        }
    }

    mem_init_ = true;
}

void VoltagePipeline::destroy_memory() {
    if (!mem_init_) return;

    auto safe_free = [](void *&p) {
        if (p) { cudaFree(p); p = nullptr; }
    };

    if (d_filterbank_chanmajor_) { cudaFree(d_filterbank_chanmajor_); d_filterbank_chanmajor_ = nullptr; }
    if (d_filterbank_beammajor_) { cudaFree(d_filterbank_beammajor_); d_filterbank_beammajor_ = nullptr; }

    safe_free(d_weights_re_);
    safe_free(d_weights_im_);
    for (int p = 0; p < 2; ++p) {
        safe_free(d_volt_int4_[p]);
        safe_free(d_volt_fp8_[p]);
        safe_free(d_volt_re_[p]);
        safe_free(d_volt_im_[p]);
        safe_free(d_beam_re_[p]);
        safe_free(d_beam_im_[p]);
    }
    d_volt_fp8_cap_ = 0;
    safe_free(d_power_);

    if (graph_exec_) {
        cudaGraphExecDestroy(static_cast<cudaGraphExec_t>(graph_exec_));
        graph_exec_ = nullptr;
    }

    mem_init_ = false;
}

int64_t VoltagePipeline::output_size() const {
    return (int64_t)n_channels_ * n_beams_ * n_time_out_;
}

int64_t VoltagePipeline::input_size() const {
    return (int64_t)n_channels_ * n_polarizations_
         * n_antennae_ * n_time_;
}

void VoltagePipeline::set_weights(const float *d_weights, cudaStream_t stream) {
    auto *api = static_cast<cutlass_gemm_api::CutlassComplexGemm*>(gemm_impl_);

    const int64_t n_complex = (int64_t)n_channels_ * n_beams_ * n_antennae_;

    // Deinterleave FP32 complex -> FP16 planar
    const int block = 256;
    const int64_t grid = std::min((n_complex + block - 1) / block,
                                  static_cast<int64_t>(1024));
    deinterleave_fp32_to_fp16_kernel<<<grid, block, 0, stream>>>(
        d_weights,
        static_cast<__half*>(d_weights_re_),
        static_cast<__half*>(d_weights_im_),
        n_complex);
    VP_CHECK_CUDA(cudaGetLastError());

    // Map VoltageComputeMode -> cutlass_gemm_api::ComputePrecision
    cutlass_gemm_api::ComputePrecision compute;
    switch (compute_mode_) {
        case VoltageComputeMode::FP6:
            compute = cutlass_gemm_api::ComputePrecision::FP6;
            break;
        default:
            compute = cutlass_gemm_api::ComputePrecision::FP8;
            break;
    }

    // Pre-prepare weights via generic BLAS API.
    // With ch_fuse > 1, B is identical across ch_fuse consecutive channels,
    // so we prepare batch = n_ch / ch_fuse unique weight sets.
    const int batch_prepared = static_cast<int>(n_channels_ / ch_fuse_);
    api->prepare_b(
        static_cast<const __half*>(d_weights_re_),
        static_cast<const __half*>(d_weights_im_),
        static_cast<int>(n_beams_),       // N
        static_cast<int>(n_antennae_),    // K
        batch_prepared,                   // batch (reduced by ch_fuse)
        compute,
        stream);

    weights_prepared_ = true;

    // Invalidate CUDA graph cache (weight data changed)
    if (graph_exec_) {
        cudaGraphExecDestroy(static_cast<cudaGraphExec_t>(graph_exec_));
        graph_exec_ = nullptr;
    }
}

void VoltagePipeline::beamform(
    uint8_t *d_output, const uint8_t *d_qc_input,
    float scale, cudaStream_t stream)
{
    if (!weights_prepared_)
        VP_ERROR("set_weights() must be called before beamform()");

    // CUDA graph capture/replay for the direct path
    if (graph_enabled_ && use_direct_) {
        bool cache_valid = graph_exec_ != nullptr
                        && graph_last_input_ == d_qc_input
                        && graph_last_output_ == d_output
                        && graph_last_scale_ == scale;

        if (cache_valid) {
            // Replay cached graph (eliminates host-side launch overhead)
            VP_CHECK_CUDA(cudaGraphLaunch(
                static_cast<cudaGraphExec_t>(graph_exec_), stream));
            return;
        }

        // Capture new graph
        if (graph_exec_) {
            cudaGraphExecDestroy(static_cast<cudaGraphExec_t>(graph_exec_));
            graph_exec_ = nullptr;
        }

        VP_CHECK_CUDA(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
#ifndef STANDALONE_TEST
        auto *saved_tp = profile_tp_;
        profile_tp_ = nullptr;  // disable profiling during graph capture
#endif
        beamform_body(d_output, d_qc_input, scale, stream);
#ifndef STANDALONE_TEST
        profile_tp_ = saved_tp;
#endif
        cudaGraph_t graph;
        VP_CHECK_CUDA(cudaStreamEndCapture(stream, &graph));
        cudaGraphExec_t exec;
        VP_CHECK_CUDA(cudaGraphInstantiate(&exec, graph, 0));
        cudaGraphDestroy(graph);
        graph_exec_ = exec;

        graph_last_input_ = d_qc_input;
        graph_last_output_ = d_output;
        graph_last_scale_ = scale;

        // Launch the newly created graph
        VP_CHECK_CUDA(cudaGraphLaunch(exec, stream));
        return;
    }

    beamform_body(d_output, d_qc_input, scale, stream);
}

void VoltagePipeline::beamform_body(
    uint8_t *d_output, const uint8_t *d_qc_input,
    float scale, cudaStream_t stream)
{
    auto *api = static_cast<cutlass_gemm_api::CutlassComplexGemm*>(gemm_impl_);

    const int n_ant  = static_cast<int>(n_antennae_);
    const int n_time = static_cast<int>(n_time_);
    const int n_ch   = static_cast<int>(n_channels_);
    const int n_beam = static_cast<int>(n_beams_);
    const int n_tps  = static_cast<int>(n_time_power_sum_);

    if (use_direct_) {
        // ================================================================
        // Direct FP8 path: fused QC->FP8 + direct GEMM + power detect
        // ================================================================

        VP_PROFILE_START(QUDA_PROFILE_PREAMBLE);

        // ---- Stage 1: QC INT4 transpose + pol-split (no decode) ----
        {
            const int64_t total = (int64_t)n_ch * n_time * n_ant;
            const int block = 256;
            const int grid = (int)std::min((total + block - 1) / block,
                                            static_cast<int64_t>(1024));
            qc_transpose_polsplit_kernel<<<grid, block, 0, stream>>>(
                d_qc_input,
                static_cast<uint8_t*>(d_volt_int4_[0]),
                static_cast<uint8_t*>(d_volt_int4_[1]),
                n_ant, n_time, n_ch);
            VP_CHECK_CUDA(cudaGetLastError());
        }

        VP_PROFILE_SYNC_STOP_START(QUDA_PROFILE_PREAMBLE, QUDA_PROFILE_COMPUTE);

        // ---- Stage 2: Fused power detection via direct GEMM (INT4 input) ----
        // The PIMPL API converts INT4 -> FP8 interleaved internally.
        // Two launches with beta accumulation:
        //   pol0: C_power  = |W x V_pol0^T|^2       (beta = 0)
        //   pol1: C_power += |W x V_pol1^T|^2       (beta = 1)
        // Computes Stokes I = |pol0|^2 + |pol1|^2 in a single buffer.
        for (int p = 0; p < 2; ++p) {
            int ret = api->gemm_prepared_power_int4(
                d_volt_int4_[p],
                static_cast<float*>(d_power_),
                n_time,         // M
                n_beam,         // N
                n_ant,          // K
                n_ch,           // batch
                1.0f,
                (p == 0) ? 0.0f : 1.0f,  // beta: overwrite for pol0, accumulate for pol1
                stream);

            if (ret != 0) {
                VP_ERROR("Direct GEMM power (INT4) failed for pol %d with status %d", p, ret);
            }
        }

        VP_PROFILE_SYNC_STOP_START(QUDA_PROFILE_COMPUTE, QUDA_PROFILE_EPILOGUE);

        // ---- Stage 3: Time integration + quantise ----
        {
            const int n_time_out = n_time / n_tps;
            const int64_t total = (int64_t)n_ch * n_beam * n_time_out;
            const int block = 256;
            const int grid = (int)std::min((total + block - 1) / block,
                                            static_cast<int64_t>(1024));
            time_integrate_quantize_kernel<<<grid, block, 0, stream>>>(
                static_cast<const float*>(d_power_), d_output,
                n_beam, n_time, n_ch,
                n_tps, scale);
            VP_CHECK_CUDA(cudaGetLastError());
        }

        VP_PROFILE_SYNC_STOP(QUDA_PROFILE_EPILOGUE);
#ifndef STANDALONE_TEST
        if (profile_tp_) print_profile();
#endif
    } else {
        // ================================================================
        // 4M sub-GEMM fallback path (FP6/FP4)
        // ================================================================

        VP_PROFILE_START(QUDA_PROFILE_PREAMBLE);

        // ---- Stage 1: QC INT4 -> FP16 planar (with transpose + pol-split) ----
        {
            const int64_t total = (int64_t)n_ch * n_time * n_ant;
            const int block = 256;
            const int grid = (int)std::min((total + block - 1) / block,
                                            static_cast<int64_t>(1024));
            qc_to_fp16_planar_polsplit_kernel<<<grid, block, 0, stream>>>(
                d_qc_input,
                static_cast<__half*>(d_volt_re_[0]),
                static_cast<__half*>(d_volt_im_[0]),
                static_cast<__half*>(d_volt_re_[1]),
                static_cast<__half*>(d_volt_im_[1]),
                n_ant, n_time, n_ch);
            VP_CHECK_CUDA(cudaGetLastError());
        }

        VP_PROFILE_SYNC_STOP_START(QUDA_PROFILE_PREAMBLE, QUDA_PROFILE_COMPUTE);

        // ---- Stage 2: Batched complex GEMM (per polarisation) ----
        for (int p = 0; p < 2; ++p) {
            int ret = api->gemm_prepared(
                static_cast<const __half*>(d_volt_re_[p]),
                static_cast<const __half*>(d_volt_im_[p]),
                d_beam_re_[p], d_beam_im_[p],
                n_time,         // M
                n_beam,         // N
                n_ant,          // K
                n_ch,           // batch
                cutlass_gemm_api::OutputPrecision::FP32,
                1.0f, 0.0f,
                stream,
                /*tune=*/gemm_tune_);

            if (ret != 0) {
                VP_ERROR("GEMM failed for polarisation %d with status %d", p, ret);
            }
        }

        VP_PROFILE_SYNC_STOP_START(QUDA_PROFILE_COMPUTE, QUDA_PROFILE_EPILOGUE);

        // ---- Stage 3: Power sum + quantise -> uint8 ----
        {
            const int n_time_out = n_time / n_tps;
            const int64_t total = (int64_t)n_ch * n_beam * n_time_out;
            const int block = 256;
            const int grid = (int)std::min((total + block - 1) / block,
                                            static_cast<int64_t>(1024));
            power_sum_quantize_kernel<<<grid, block, 0, stream>>>(
                static_cast<const float*>(d_beam_re_[0]),
                static_cast<const float*>(d_beam_im_[0]),
                static_cast<const float*>(d_beam_re_[1]),
                static_cast<const float*>(d_beam_im_[1]),
                d_output,
                n_beam, n_time, n_ch,
                n_tps, scale);
            VP_CHECK_CUDA(cudaGetLastError());
        }

        VP_PROFILE_SYNC_STOP(QUDA_PROFILE_EPILOGUE);
#ifndef STANDALONE_TEST
        if (profile_tp_) print_profile();
#endif
    }
}

void VoltagePipeline::beamform_batched(
    uint8_t *d_output, const uint8_t * const *d_qc_inputs,
    int n_payloads, float scale, cudaStream_t stream)
{
    if (!weights_prepared_)
        VP_ERROR("set_weights() must be called before beamform_batched()");
    if (!use_direct_)
        VP_ERROR("beamform_batched() requires FP8 compute mode (direct path)");
    if (n_payloads < 1 || n_payloads > (int)max_payloads_)
        VP_ERROR("n_payloads (%d) must be 1-%llu", n_payloads, (unsigned long long)max_payloads_);
    if (ch_fuse_ == 1 && n_payloads == 1)
        VP_ERROR("beamform_batched() requires ch_fuse > 1 or n_payloads > 1");

    beamform_fused_body(d_output, d_qc_inputs, n_payloads, scale, stream);
}

void VoltagePipeline::beamform_fused_body(
    uint8_t *d_output, const uint8_t * const *d_qc_inputs,
    int n_payloads, float scale, cudaStream_t stream)
{
    auto *api = static_cast<cutlass_gemm_api::CutlassComplexGemm*>(gemm_impl_);

    const int n_ant  = static_cast<int>(n_antennae_);
    const int n_time = static_cast<int>(n_time_);
    const int n_ch   = static_cast<int>(n_channels_);
    const int n_beam = static_cast<int>(n_beams_);
    const int n_tps  = static_cast<int>(n_time_power_sum_);
    const int cf     = static_cast<int>(ch_fuse_);

    const int batch_fused = n_ch / cf;
    const int M_total     = n_time * cf * n_payloads;

    VP_PROFILE_START(QUDA_PROFILE_PREAMBLE);

    // ---- Stage 1: Copy QC payloads to contiguous device buffer ----
    // The fused kernel reads all payloads from a single contiguous buffer.
    // d_qc_inputs is a host array of device pointers; we need contiguous data.
    const int64_t payload_size = (int64_t)n_ch * 2 * n_ant * n_time;
    const int64_t total_qc_bytes = payload_size * n_payloads;

    // Use d_volt_int4_[0] as temporary contiguous buffer (big enough for 1 payload).
    // For multiple payloads, we need a larger buffer. We'll use cudaMallocAsync
    // only if the existing int4 buffer isn't big enough.
    uint8_t* d_qc_contiguous = nullptr;
    bool qc_needs_free = false;
    if (n_payloads == 1) {
        // Single payload — just point to it directly
        d_qc_contiguous = const_cast<uint8_t*>(d_qc_inputs[0]);
    } else {
        // Multiple payloads — copy to contiguous buffer
        VP_CHECK_CUDA(cudaMallocAsync(&d_qc_contiguous, total_qc_bytes, stream));
        qc_needs_free = true;
        for (int p = 0; p < n_payloads; ++p) {
            VP_CHECK_CUDA(cudaMemcpyAsync(
                d_qc_contiguous + p * payload_size,
                d_qc_inputs[p],
                payload_size,
                cudaMemcpyDeviceToDevice, stream));
        }
    }

    // ---- Stage 2: Fused QC transpose + pol-split + ch-fuse + FP8 conversion ----
    {
        const int64_t total = (int64_t)batch_fused * M_total * n_ant;
        const int block = 256;
        const int grid = (int)std::min((total + block - 1) / block,
                                        static_cast<int64_t>(1024));
        qc_fused_transpose_fp8_kernel<<<grid, block, 0, stream>>>(
            d_qc_contiguous,
            static_cast<uint8_t*>(d_volt_fp8_[0]),
            static_cast<uint8_t*>(d_volt_fp8_[1]),
            n_ant, n_time, n_ch,
            cf, n_payloads,
            payload_size);
        VP_CHECK_CUDA(cudaGetLastError());
    }

    if (qc_needs_free) {
        cudaFreeAsync(d_qc_contiguous, stream);
    }

    VP_PROFILE_SYNC_STOP_START(QUDA_PROFILE_PREAMBLE, QUDA_PROFILE_COMPUTE);

    // ---- Stage 3: Direct GEMM with fused power (pre-cast FP8 input) ----
    // Two launches for Stokes I: pol0 (beta=0), pol1 (beta=1)
    for (int p = 0; p < 2; ++p) {
        int ret = api->gemm_prepared_power_fp8(
            d_volt_fp8_[p],
            static_cast<float*>(d_power_),
            M_total,        // M (fused: n_time * ch_fuse * n_payloads)
            n_beam,         // N
            n_ant,          // K
            batch_fused,    // batch (reduced by ch_fuse)
            1.0f,
            (p == 0) ? 0.0f : 1.0f,
            stream);

        if (ret != 0) {
            VP_ERROR("Direct GEMM power (FP8 fused) failed for pol %d with status %d", p, ret);
        }
    }

    VP_PROFILE_SYNC_STOP_START(QUDA_PROFILE_COMPUTE, QUDA_PROFILE_EPILOGUE);

    // ---- Stage 4: Time integration + quantise ----
    // Power layout: [batch_fused x M_total x n_beam]
    // Output layout: [n_payloads x n_ch x n_beam x n_time_out]
    // Since M_total = n_time * ch_fuse * n_payloads, and batch = n_ch / ch_fuse,
    // the power buffer is effectively [n_ch/ch_fuse x (n_time*ch_fuse*n_payloads) x n_beam]
    // which maps to [n_payloads x n_ch x n_time x n_beam] when unrolled.
    {
        const int n_time_out = n_time / n_tps;
        const int64_t total = (int64_t)batch_fused * (M_total / n_tps) * n_beam;
        const int block = 256;
        const int grid = (int)std::min((total + block - 1) / block,
                                        static_cast<int64_t>(1024));
        // Reuse time_integrate_quantize_kernel with adjusted dimensions:
        // Treat the entire fused dimension as n_ch_eff and M_total as n_time_eff
        time_integrate_quantize_kernel<<<grid, block, 0, stream>>>(
            static_cast<const float*>(d_power_), d_output,
            n_beam, M_total, batch_fused,
            n_tps, scale);
        VP_CHECK_CUDA(cudaGetLastError());
    }

    VP_PROFILE_SYNC_STOP(QUDA_PROFILE_EPILOGUE);
#ifndef STANDALONE_TEST
    if (profile_tp_) print_profile();
#endif
}

void VoltagePipeline::compute_filterbank(const uint8_t *d_qc_input,
                                          cudaStream_t stream)
{
    if (!weights_prepared_)
        VP_ERROR("set_weights() must be called before compute_filterbank()");

    auto *api = static_cast<cutlass_gemm_api::CutlassComplexGemm*>(gemm_impl_);

    const int n_ant  = static_cast<int>(n_antennae_);
    const int n_time = static_cast<int>(n_time_);
    const int n_ch   = static_cast<int>(n_channels_);
    const int n_beam = static_cast<int>(n_beams_);
    const int n_tps  = static_cast<int>(n_time_power_sum_);

    if (use_direct_) {
        // ---- Stage 1: QC INT4 transpose + pol-split (no decode) ----
        {
            const int64_t total = (int64_t)n_ch * n_time * n_ant;
            const int block = 256;
            const int grid = (int)std::min((total + block - 1) / block,
                                            static_cast<int64_t>(1024));
            qc_transpose_polsplit_kernel<<<grid, block, 0, stream>>>(
                d_qc_input,
                static_cast<uint8_t*>(d_volt_int4_[0]),
                static_cast<uint8_t*>(d_volt_int4_[1]),
                n_ant, n_time, n_ch);
            VP_CHECK_CUDA(cudaGetLastError());
        }

        // ---- Stage 2: Fused power detection via direct GEMM (INT4 input) ----
        for (int p = 0; p < 2; ++p) {
            int ret = api->gemm_prepared_power_int4(
                d_volt_int4_[p],
                static_cast<float*>(d_power_),
                n_time, n_beam, n_ant, n_ch,
                1.0f, (p == 0) ? 0.0f : 1.0f,
                stream);
            if (ret != 0)
                VP_ERROR("Direct GEMM power (INT4) failed for pol %d with status %d", p, ret);
        }

        // ---- Stage 3: Time integration -> FP32 filterbank ----
        {
            const int n_time_out = n_time / n_tps;
            const int64_t total = (int64_t)n_ch * n_beam * n_time_out;
            const int block = 256;
            const int grid = (int)std::min((total + block - 1) / block,
                                            static_cast<int64_t>(1024));
            time_integrate_fp32_kernel<<<grid, block, 0, stream>>>(
                static_cast<const float*>(d_power_),
                d_filterbank_chanmajor_,
                n_beam, n_time, n_ch,
                n_tps);
            VP_CHECK_CUDA(cudaGetLastError());
        }
    } else {
        // ---- Stage 1: QC INT4 -> FP16 planar ----
        {
            const int64_t total = (int64_t)n_ch * n_time * n_ant;
            const int block = 256;
            const int grid = (int)std::min((total + block - 1) / block,
                                            static_cast<int64_t>(1024));
            qc_to_fp16_planar_polsplit_kernel<<<grid, block, 0, stream>>>(
                d_qc_input,
                static_cast<__half*>(d_volt_re_[0]),
                static_cast<__half*>(d_volt_im_[0]),
                static_cast<__half*>(d_volt_re_[1]),
                static_cast<__half*>(d_volt_im_[1]),
                n_ant, n_time, n_ch);
            VP_CHECK_CUDA(cudaGetLastError());
        }

        // ---- Stage 2: Batched complex GEMM (per polarisation) ----
        for (int p = 0; p < 2; ++p) {
            int ret = api->gemm_prepared(
                static_cast<const __half*>(d_volt_re_[p]),
                static_cast<const __half*>(d_volt_im_[p]),
                d_beam_re_[p], d_beam_im_[p],
                n_time, n_beam, n_ant, n_ch,
                cutlass_gemm_api::OutputPrecision::FP32,
                1.0f, 0.0f, stream,
                /*tune=*/gemm_tune_);
            if (ret != 0)
                VP_ERROR("GEMM failed for polarisation %d with status %d", p, ret);
        }

        // ---- Stage 3: Power sum -> FP32 filterbank ----
        {
            const int n_time_out = n_time / n_tps;
            const int64_t total = (int64_t)n_ch * n_beam * n_time_out;
            const int block = 256;
            const int grid = (int)std::min((total + block - 1) / block,
                                            static_cast<int64_t>(1024));
            power_sum_fp32_kernel<<<grid, block, 0, stream>>>(
                static_cast<const float*>(d_beam_re_[0]),
                static_cast<const float*>(d_beam_im_[0]),
                static_cast<const float*>(d_beam_re_[1]),
                static_cast<const float*>(d_beam_im_[1]),
                d_filterbank_chanmajor_,
                n_beam, n_time, n_ch,
                n_tps);
            VP_CHECK_CUDA(cudaGetLastError());
        }
    }
}

void VoltagePipeline::compute(uint8_t *d_output, const uint8_t *d_qc_input,
                               float *d_dedisp_out, float scale,
                               cudaStream_t stream)
{
    // Step 1: Beamform -> FP32 filterbank [n_ch, n_beam, n_time_out]
    VP_PROFILE_START(QUDA_PROFILE_COMPUTE);
    compute_filterbank(d_qc_input, stream);
    VP_PROFILE_SYNC_STOP(QUDA_PROFILE_COMPUTE);

    // Step 2: Corner turn -> [n_beam, n_ch, n_time_out]
    VP_PROFILE_START(QUDA_PROFILE_PREAMBLE);
    corner_turn(d_filterbank_beammajor_, d_filterbank_chanmajor_, stream);
    VP_PROFILE_SYNC_STOP(QUDA_PROFILE_PREAMBLE);

    // Step 3: Dedispersion (if requested and available)
    if (d_dedisp_out) {
        dedisperse(d_dedisp_out, d_filterbank_beammajor_,
                   static_cast<int>(n_beams_), stream);
    }

    // Step 4: Quantise FP32 -> uint8
    VP_PROFILE_START(QUDA_PROFILE_EPILOGUE);
    {
        const int64_t total = (int64_t)n_channels_ * n_beams_ * n_time_out_;
        const int block = 256;
        const int grid = (int)std::min((total + block - 1) / block,
                                        static_cast<int64_t>(1024));
        quantize_fp32_to_uint8_kernel<<<grid, block, 0, stream>>>(
            d_filterbank_chanmajor_, d_output, scale, total);
        VP_CHECK_CUDA(cudaGetLastError());
    }
    VP_PROFILE_SYNC_STOP(QUDA_PROFILE_EPILOGUE);
#ifndef STANDALONE_TEST
    if (profile_tp_) print_profile();
#endif
}

void VoltagePipeline::corner_turn(float *d_out, const float *d_in,
                                   cudaStream_t stream)
{
    corner_turn_nf_nb(d_out, d_in,
                      static_cast<int>(n_channels_),
                      static_cast<int>(n_beams_),
                      static_cast<int>(n_time_out_),
                      stream);
}

int VoltagePipeline::dedisperse(float *d_output, const float *d_input,
                                 int batch_size, cudaStream_t stream)
{
#ifdef DEDISP_API_LIB
    if (!dedisp_pipeline_) return -1;
    auto *pipeline = static_cast<dedisp_api::DedispPipeline*>(dedisp_pipeline_);
    return pipeline->dedisperse(d_input, d_output, batch_size, stream);
#else
    (void)d_output; (void)d_input; (void)batch_size; (void)stream;
    return -1;
#endif
}

} // namespace ggp

// ========================================================================
// Non-CUTLASS fallback
// ========================================================================

#else // !CUTLASS_GEMM_API

namespace ggp {

VoltagePipeline::VoltagePipeline(const Config &) {
    VP_ERROR("VoltagePipeline requires CUTLASS_GEMM_API to be compiled");
}
VoltagePipeline::~VoltagePipeline() {}
void VoltagePipeline::init_memory() {}
void VoltagePipeline::destroy_memory() {}
int64_t VoltagePipeline::output_size() const { return 0; }
int64_t VoltagePipeline::input_size() const { return 0; }
void VoltagePipeline::set_weights(const float *, cudaStream_t) {
    VP_ERROR("VoltagePipeline requires CUTLASS_GEMM_API");
}
void VoltagePipeline::beamform(uint8_t *, const uint8_t *, float, cudaStream_t) {
    VP_ERROR("VoltagePipeline requires CUTLASS_GEMM_API");
}
void VoltagePipeline::beamform_body(uint8_t *, const uint8_t *, float, cudaStream_t) {
    VP_ERROR("VoltagePipeline requires CUTLASS_GEMM_API");
}
void VoltagePipeline::beamform_batched(uint8_t *, const uint8_t * const *, int, float, cudaStream_t) {
    VP_ERROR("VoltagePipeline requires CUTLASS_GEMM_API");
}
void VoltagePipeline::beamform_fused_body(uint8_t *, const uint8_t * const *, int, float, cudaStream_t) {
    VP_ERROR("VoltagePipeline requires CUTLASS_GEMM_API");
}
void VoltagePipeline::compute_filterbank(const uint8_t *, cudaStream_t) {
    VP_ERROR("VoltagePipeline requires CUTLASS_GEMM_API");
}
void VoltagePipeline::compute(uint8_t *, const uint8_t *, float *, float, cudaStream_t) {
    VP_ERROR("VoltagePipeline requires CUTLASS_GEMM_API");
}
void VoltagePipeline::corner_turn(float *, const float *, cudaStream_t) {
    VP_ERROR("VoltagePipeline requires CUTLASS_GEMM_API");
}
int VoltagePipeline::dedisperse(float *, const float *, int, cudaStream_t) {
    return -1;
}
void VoltagePipeline::set_profile(bool) {}
void VoltagePipeline::print_profile() {}

} // namespace ggp

#endif
