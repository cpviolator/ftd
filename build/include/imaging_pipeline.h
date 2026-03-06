/** @file imaging_pipeline.h
 *  @brief Clean C++ API for UV-plane gridding and 2D FFT imaging.
 *
 *  PIMPL pattern -- no CUTLASS or QUDA dependencies.
 *  Requires only cuda_fp16.h and cuda_runtime.h.
 *  Consumers link libimaging_pipeline.a and include this header.
 */

#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace imaging_pipeline {

/// @brief Visibility data precision coming from HERK/correlator output.
enum class VisPrecision {
    FP16,  ///< __half2 packed complex (Re, Im) per element
    FP32   ///< float pairs (Re, Im) interleaved per element
};

/// @brief FFT computation precision.
///
/// Controls the internal grid format and cuFFT plan type:
///  - FP16: Grid accumulates as __half2, cuFFT via cufftXtMakePlanMany(CUDA_C_16F).
///          Requires Ng to be a power of 2. No FP32 scratch buffer needed —
///          ~3x more frequency channels per tile than FP32.
///  - FP32: Grid accumulates as cufftComplex (or widened from FP16 grid),
///          cuFFT via cufftExecC2C. Works with any Ng.
///
/// Memory per channel at Ng=4096:
///  - FP16: 67 MB  (__half2 only)
///  - FP32: 134 MB (cufftComplex only, or 201 MB if widening from FP16 grid)
enum class FftPrecision {
    FP16,  ///< Native FP16 C2C via cufftXt (power-of-2 Ng only)
    FP32   ///< Standard FP32 C2C via cufftExecC2C
};

/// @brief Pillbox gridding + 2D IFFT imaging pipeline.
///
/// Grids packed-triangle visibilities onto a UV plane, applies optional
/// taper, runs batched 2D IFFT, and extracts beam pixel intensities.
/// Frequency channels are tiled to fit in GPU memory.
///
/// Supports all 4 combinations of VisPrecision x FftPrecision:
///
/// | VisPrecision | FftPrecision | Grid format  | Pipeline                              |
/// |--------------|--------------|--------------|---------------------------------------|
/// | FP16         | FP16         | __half2      | scatter->taper->cufftXt->beam(fp16)   |
/// | FP16         | FP32         | __half2->FP32| scatter->taper->widen->cuFFT->beam    |
/// | FP32         | FP16         | __half2      | scatter(cvt)->taper->cufftXt->beam    |
/// | FP32         | FP32         | cufftComplex | scatter(direct)->taper->cuFFT->beam   |
///
/// @note All device pointers must reside in GPU memory.
/// @note Thread-safety: one instance per CUDA stream.
class ImagingPipeline {
public:
    ImagingPipeline();
    ~ImagingPipeline();

    // Non-copyable, movable
    ImagingPipeline(const ImagingPipeline&) = delete;
    ImagingPipeline& operator=(const ImagingPipeline&) = delete;
    ImagingPipeline(ImagingPipeline&&) noexcept;
    ImagingPipeline& operator=(ImagingPipeline&&) noexcept;

    /// @brief Configure grid dimensions, FFT precision, and allocate buffers.
    ///
    /// Must be called before any processing. Queries GPU memory
    /// to determine frequency tiling.
    ///
    /// @param N          Number of antennas (n_baselines = N*(N+1)/2)
    /// @param Nf         Total number of frequency channels
    /// @param Ng         UV grid size (Ng x Ng per channel), default 4096
    /// @param n_beam     Number of beam pixels to extract (0 = skip extraction)
    /// @param fft_prec   FFT precision (FP16 or FP32, default FP32)
    /// @param stream     CUDA stream for allocations
    /// @return 0 on success, non-zero on error
    /// @pre For FP16 FFT, Ng must be a power of 2.
    int configure(int N, int Nf, int Ng = 4096, int n_beam = 0,
                  FftPrecision fft_prec = FftPrecision::FP32,
                  cudaStream_t stream = nullptr);

    /// @brief Set baseline UV coordinates in metres.
    /// @param baseline_uv_m Device pointer to [n_baselines x 2] float.
    /// @param n_baselines   Must match N*(N+1)/2.
    /// @return 0 on success
    int set_baseline_uv(const float* baseline_uv_m, int n_baselines);

    /// @brief Set frequency array.
    /// @param freq_hz Device pointer to [Nf] double frequencies in Hz.
    /// @return 0 on success
    int set_frequencies(const double* freq_hz);

    /// @brief Set UV cell size.
    /// @param cell_size_rad Cell size in radians.
    void set_cell_size(float cell_size_rad);

    /// @brief Set optional UV taper weights.
    /// @param taper Device pointer to [Ng x Ng] float. nullptr to disable.
    void set_taper(const float* taper);

    /// @brief Set beam pixel coordinates for extraction.
    /// @param pixel_coords Device pointer to [n_beam x 2] int (row, col).
    /// @param n_beam       Number of beam pixels.
    /// @return 0 on success
    int set_beam_pixels(const int* pixel_coords, int n_beam);

    /// @brief Process a single frequency tile.
    ///
    /// @param vis_tile   Device pointer to visibility data.
    ///                   FP16: [Nf_this * n_baselines] __half2.
    ///                   FP32: [Nf_this * n_baselines * 2] float.
    /// @param beam_tile  Output [Nf_this * n_beam] float. nullptr if n_beam==0.
    /// @param f_start    First channel index (offset into freq_hz).
    /// @param Nf_this    Number of channels in this tile.
    /// @param prec       Visibility precision from correlator.
    /// @param stream     CUDA stream.
    /// @return 0 on success
    int process_tile(const void* vis_tile, float* beam_tile,
                     int f_start, int Nf_this,
                     VisPrecision prec, cudaStream_t stream = nullptr);

    /// @brief Full pipeline: grid all Nf channels with auto frequency tiling.
    ///
    /// @param vis_tri     Device pointer to all visibilities (packed triangle).
    /// @param beam_output Device pointer to [Nf * n_beam] float. nullptr ok.
    /// @param prec        Visibility precision.
    /// @param stream      CUDA stream.
    /// @return 0 on success
    int grid_and_image(const void* vis_tri, float* beam_output,
                       VisPrecision prec = VisPrecision::FP16,
                       cudaStream_t stream = nullptr);

    /// @brief Copy FP16 UV grid to host (only valid when fft_prec==FP16
    ///        or when fft_prec==FP32 with FP16 vis input).
    /// @param out       Host pointer to [channels * Ng * Ng] __half2.
    /// @param tile_idx  Frequency tile index (0 = most recent).
    /// @return 0 on success, -1 if FP16 grid not available
    int get_uv_grid(__half2* out, int tile_idx = 0);

    /// @brief Copy FP32 image/grid plane to host (post-IFFT for FP32 path).
    /// @param out       Host pointer to [channels * Ng * Ng * 2] float.
    /// @param tile_idx  Frequency tile index (0 = most recent).
    /// @return 0 on success, -1 if FP32 buffer not available
    int get_image_plane(float* out, int tile_idx = 0);

    /// @brief Copy FP16 image plane to host (post-IFFT for FP16 FFT path).
    /// @param out       Host pointer to [channels * Ng * Ng] __half2.
    /// @param tile_idx  Frequency tile index (0 = most recent).
    /// @return 0 on success, -1 if FP16 FFT not configured
    int get_image_plane_fp16(__half2* out, int tile_idx = 0);

    /// @brief Query the configured FFT precision.
    FftPrecision get_fft_precision() const;

    /// @brief Query the computed frequency tile size.
    /// @return Channels per tile (0 if not configured).
    int get_nf_tile() const;

    /// @brief Query the number of frequency tiles.
    int get_num_tiles() const;

    // ---- Per-sub-stage timing (for benchmarking) ----

    /// @brief Per-sub-stage timing breakdown from grid_and_image / process_tile.
    struct StageTimes {
        float scatter_ms  = 0;  ///< Pillbox grid scatter
        float taper_ms    = 0;  ///< UV taper application
        float widen_ms    = 0;  ///< FP16→FP32 grid widening (FP32 FFT + FP16 vis only)
        float fft_ms      = 0;  ///< cuFFT execution (IFFT)
        float beam_ms     = 0;  ///< Beam pixel extraction
        float total_ms    = 0;  ///< Sum of above
    };

    /// @brief Enable/disable per-sub-stage CUDA event timing.
    /// @param enabled When true, CUDA events are recorded between sub-stages.
    void set_timing_enabled(bool enabled);

    /// @brief Get accumulated sub-stage times from the last grid_and_image() call.
    /// @return StageTimes with per-sub-stage milliseconds (accumulated across tiles).
    /// @pre set_timing_enabled(true) must have been called before grid_and_image().
    StageTimes get_stage_times() const;

    /// @brief Release all GPU resources.
    void destroy();

private:
    struct Impl;
    Impl* impl_;
};

} // namespace imaging_pipeline
