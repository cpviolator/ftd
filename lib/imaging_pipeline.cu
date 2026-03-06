/** @file imaging_pipeline.cu
 *  @brief Implementation of the ImagingPipeline PIMPL class.
 *
 *  Buffer management, cuFFT plan creation (FP16 via cufftXt and FP32),
 *  frequency tiling, and kernel launch orchestration.
 */

#include "imaging_pipeline.h"
#include "kernels/imaging_kernels.cuh"

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cufft.h>
#include <cufftXt.h>
#include <library_types.h>
#include <cstdio>
#include <cstdlib>
#include <algorithm>

namespace imaging_pipeline {

// ================================================================
// Error checking macros
// ================================================================

#define CUDA_CHECK(call)                                                     \
    do {                                                                      \
        cudaError_t err = (call);                                             \
        if (err != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",                     \
                    __FILE__, __LINE__, cudaGetErrorString(err));             \
            return -1;                                                        \
        }                                                                     \
    } while (0)

#define CUFFT_CHECK(call)                                                    \
    do {                                                                      \
        cufftResult err = (call);                                             \
        if (err != CUFFT_SUCCESS) {                                           \
            fprintf(stderr, "cuFFT error at %s:%d: %d\n",                    \
                    __FILE__, __LINE__, static_cast<int>(err));               \
            return -1;                                                        \
        }                                                                     \
    } while (0)

/// Check if n is a power of 2.
static bool is_power_of_2(int n) {
    return n > 0 && (n & (n - 1)) == 0;
}

// ================================================================
// PIMPL Implementation
// ================================================================

struct ImagingPipeline::Impl {
    // Configuration
    int N = 0;
    int Nf = 0;
    int Ng = 4096;
    int n_baselines = 0;
    int n_beam = 0;
    float cell_size_rad = 0.0f;
    int Nf_tile = 0;
    int n_tiles = 0;
    int last_tile_channels = 0;
    FftPrecision fft_precision = FftPrecision::FP32;
    bool configured = false;

    // ---- Buffers ----
    // FP16 grid: used when fft_precision==FP16, or when fft_precision==FP32
    //            with FP16/FP32 vis input (old widen path)
    __half2*       uv_grid_fp16 = nullptr;     // [Nf_tile * Ng * Ng]

    // FP32 grid/scratch: used when fft_precision==FP32
    cufftComplex*  uv_grid_fp32 = nullptr;     // [Nf_tile * Ng * Ng]

    // Borrowed pointers
    const float*   baseline_uv_m_d = nullptr;
    const double*  freq_hz_d = nullptr;
    const float*   taper_d = nullptr;
    const int*     beam_pixels_d = nullptr;

    // ---- cuFFT plans ----
    // FP32 plans (cufftPlanMany)
    cufftHandle fft_plan_fp32_full = 0;
    cufftHandle fft_plan_fp32_last = 0;
    bool fp32_full_valid = false;
    bool fp32_last_valid = false;

    // FP16 plans (cufftXtMakePlanMany)
    cufftHandle fft_plan_fp16_full = 0;
    cufftHandle fft_plan_fp16_last = 0;
    bool fp16_full_valid = false;
    bool fp16_last_valid = false;

    // ---- Per-sub-stage timing ----
    bool timing_enabled = false;
    ImagingPipeline::StageTimes stage_times{};

    // Record a CUDA event on the stream (only when timing is enabled).
    static void record_event(cudaEvent_t& ev, cudaStream_t stream) {
        cudaEventRecord(ev, stream);
    }

    // Measure elapsed time between two events (synchronizes on stop event).
    static float elapsed_ms(cudaEvent_t start, cudaEvent_t stop) {
        cudaEventSynchronize(stop);
        float ms = 0;
        cudaEventElapsedTime(&ms, start, stop);
        return ms;
    }

    void free_buffers() {
        if (uv_grid_fp16) { cudaFree(uv_grid_fp16); uv_grid_fp16 = nullptr; }
        if (uv_grid_fp32) { cudaFree(uv_grid_fp32); uv_grid_fp32 = nullptr; }
    }

    void free_plans() {
        if (fp32_full_valid) { cufftDestroy(fft_plan_fp32_full); fp32_full_valid = false; }
        if (fp32_last_valid) { cufftDestroy(fft_plan_fp32_last); fp32_last_valid = false; }
        if (fp16_full_valid) { cufftDestroy(fft_plan_fp16_full); fp16_full_valid = false; }
        if (fp16_last_valid) { cufftDestroy(fft_plan_fp16_last); fp16_last_valid = false; }
    }

    ~Impl() {
        free_buffers();
        free_plans();
    }

    // Static tile-processing helpers (must be inside Impl to access private type)
    static int process_tile_fp16_fft(Impl* impl,
                                      const void* vis_tile, float* beam_tile,
                                      int f_start, int Nf_this,
                                      VisPrecision prec, cudaStream_t stream);
    static int process_tile_fp32_fft(Impl* impl,
                                      const void* vis_tile, float* beam_tile,
                                      int f_start, int Nf_this,
                                      VisPrecision prec, cudaStream_t stream);
};

// ================================================================
// Constructor / Destructor / Move
// ================================================================

ImagingPipeline::ImagingPipeline() : impl_(new Impl()) {}

ImagingPipeline::~ImagingPipeline() { delete impl_; }

ImagingPipeline::ImagingPipeline(ImagingPipeline&& o) noexcept : impl_(o.impl_) {
    o.impl_ = nullptr;
}

ImagingPipeline& ImagingPipeline::operator=(ImagingPipeline&& o) noexcept {
    if (this != &o) { delete impl_; impl_ = o.impl_; o.impl_ = nullptr; }
    return *this;
}

// ================================================================
// Helper: create FP16 cuFFT plan via cufftXt
// ================================================================

static int create_fp16_plan(cufftHandle& plan, int Ng, int batch) {
    cufftResult err = cufftCreate(&plan);
    if (err != CUFFT_SUCCESS) {
        fprintf(stderr, "cufftCreate failed: %d\n", static_cast<int>(err));
        return -1;
    }

    long long n_fft[2] = {Ng, Ng};
    size_t workSize = 0;

    err = cufftXtMakePlanMany(plan, 2, n_fft,
                               nullptr, 1, 0, CUDA_C_16F,
                               nullptr, 1, 0, CUDA_C_16F,
                               static_cast<long long>(batch),
                               &workSize, CUDA_C_16F);
    if (err != CUFFT_SUCCESS) {
        fprintf(stderr, "cufftXtMakePlanMany(CUDA_C_16F) failed: %d\n",
                static_cast<int>(err));
        cufftDestroy(plan);
        return -1;
    }
    return 0;
}

// ================================================================
// configure()
// ================================================================

int ImagingPipeline::configure(int N, int Nf, int Ng, int n_beam,
                                FftPrecision fft_prec,
                                cudaStream_t stream) {
    impl_->free_buffers();
    impl_->free_plans();

    // Validate FP16 FFT constraint
    if (fft_prec == FftPrecision::FP16 && !is_power_of_2(Ng)) {
        fprintf(stderr, "configure: FP16 FFT requires Ng to be a power of 2 "
                        "(got Ng=%d)\n", Ng);
        return -1;
    }

    impl_->N = N;
    impl_->Nf = Nf;
    impl_->Ng = Ng;
    impl_->n_baselines = N * (N + 1) / 2;
    impl_->n_beam = n_beam;
    impl_->fft_precision = fft_prec;

    // Query available GPU memory
    size_t free_mem = 0, total_mem = 0;
    CUDA_CHECK(cudaMemGetInfo(&free_mem, &total_mem));
    const size_t usable_mem = static_cast<size_t>(free_mem * 0.8);

    const int64_t grid_plane = static_cast<int64_t>(Ng) * Ng;

    // Per-channel memory depends on FFT precision
    size_t per_ch_bytes;
    if (fft_prec == FftPrecision::FP16) {
        // FP16 path: only __half2 grid, no FP32 scratch
        per_ch_bytes = grid_plane * sizeof(__half2);
    } else {
        // FP32 path: cufftComplex grid (+ optional FP16 grid for widen path)
        // Allocate both to support all VisPrecision inputs
        per_ch_bytes = grid_plane * sizeof(__half2) +
                       grid_plane * sizeof(cufftComplex);
    }
    if (per_ch_bytes == 0) return -1;

    impl_->Nf_tile = static_cast<int>(usable_mem / per_ch_bytes);
    if (impl_->Nf_tile < 1) impl_->Nf_tile = 1;
    if (impl_->Nf_tile > Nf) impl_->Nf_tile = Nf;
    impl_->n_tiles = (Nf + impl_->Nf_tile - 1) / impl_->Nf_tile;
    impl_->last_tile_channels = Nf - (impl_->n_tiles - 1) * impl_->Nf_tile;

    // Allocate buffers
    if (fft_prec == FftPrecision::FP16) {
        // FP16 path: only __half2 grid
        const size_t bytes = static_cast<size_t>(impl_->Nf_tile) * grid_plane * sizeof(__half2);
        CUDA_CHECK(cudaMalloc(&impl_->uv_grid_fp16, bytes));
    } else {
        // FP32 path: both buffers (FP16 for scatter, FP32 for FFT)
        const size_t fp16_bytes = static_cast<size_t>(impl_->Nf_tile) * grid_plane * sizeof(__half2);
        const size_t fp32_bytes = static_cast<size_t>(impl_->Nf_tile) * grid_plane * sizeof(cufftComplex);
        CUDA_CHECK(cudaMalloc(&impl_->uv_grid_fp16, fp16_bytes));
        CUDA_CHECK(cudaMalloc(&impl_->uv_grid_fp32, fp32_bytes));
    }

    // Create cuFFT plans
    if (fft_prec == FftPrecision::FP16) {
        // FP16 plans via cufftXt
        if (create_fp16_plan(impl_->fft_plan_fp16_full, Ng, impl_->Nf_tile) != 0)
            return -1;
        impl_->fp16_full_valid = true;

        if (impl_->last_tile_channels < impl_->Nf_tile && impl_->last_tile_channels > 0) {
            if (create_fp16_plan(impl_->fft_plan_fp16_last, Ng, impl_->last_tile_channels) != 0)
                return -1;
            impl_->fp16_last_valid = true;
        }
    } else {
        // FP32 plans via cufftPlanMany
        int rank = 2;
        int n_fft[2] = {Ng, Ng};
        CUFFT_CHECK(cufftPlanMany(&impl_->fft_plan_fp32_full, rank, n_fft,
                                   nullptr, 1, 0, nullptr, 1, 0,
                                   CUFFT_C2C, impl_->Nf_tile));
        impl_->fp32_full_valid = true;

        if (impl_->last_tile_channels < impl_->Nf_tile && impl_->last_tile_channels > 0) {
            CUFFT_CHECK(cufftPlanMany(&impl_->fft_plan_fp32_last, rank, n_fft,
                                       nullptr, 1, 0, nullptr, 1, 0,
                                       CUFFT_C2C, impl_->last_tile_channels));
            impl_->fp32_last_valid = true;
        }
    }

    impl_->configured = true;
    return 0;
}

// ================================================================
// Setters
// ================================================================

int ImagingPipeline::set_baseline_uv(const float* baseline_uv_m, int n_baselines) {
    if (n_baselines != impl_->n_baselines) {
        fprintf(stderr, "set_baseline_uv: expected %d baselines, got %d\n",
                impl_->n_baselines, n_baselines);
        return -1;
    }
    impl_->baseline_uv_m_d = baseline_uv_m;
    return 0;
}

int ImagingPipeline::set_frequencies(const double* freq_hz) {
    impl_->freq_hz_d = freq_hz;
    return 0;
}

void ImagingPipeline::set_cell_size(float cell_size_rad) {
    impl_->cell_size_rad = cell_size_rad;
}

void ImagingPipeline::set_taper(const float* taper) {
    impl_->taper_d = taper;
}

int ImagingPipeline::set_beam_pixels(const int* pixel_coords, int n_beam) {
    impl_->beam_pixels_d = pixel_coords;
    impl_->n_beam = n_beam;
    return 0;
}

// ================================================================
// process_tile() — FP16 FFT path
// ================================================================

int ImagingPipeline::Impl::process_tile_fp16_fft(Impl* impl,
                                  const void* vis_tile, float* beam_tile,
                                  int f_start, int Nf_this,
                                  VisPrecision prec, cudaStream_t stream)
{
    const int Ng = impl->Ng;
    const int n_baselines = impl->n_baselines;
    const int64_t grid_plane = static_cast<int64_t>(Ng) * Ng;
    const int block = 256;
    const bool timing = impl->timing_enabled;

    cudaEvent_t ev0, ev1, ev2, ev3, ev4;
    if (timing) {
        cudaEventCreate(&ev0); cudaEventCreate(&ev1);
        cudaEventCreate(&ev2); cudaEventCreate(&ev3);
        cudaEventCreate(&ev4);
        record_event(ev0, stream);
    }

    // Step 1: Zero FP16 grid + scatter visibilities
    const size_t grid_bytes = static_cast<size_t>(Nf_this) * grid_plane * sizeof(__half2);
    CUDA_CHECK(cudaMemsetAsync(impl->uv_grid_fp16, 0, grid_bytes, stream));

    const int64_t total_vis = static_cast<int64_t>(Nf_this) * n_baselines;
    const int grid = static_cast<int>(std::min(
        (total_vis + block - 1) / block, static_cast<int64_t>(65535)));

    if (prec == VisPrecision::FP16) {
        imaging_kernels::pillbox_grid_scatter_fp16_kernel<<<grid, block, 0, stream>>>(
            static_cast<const __half2*>(vis_tile),
            impl->baseline_uv_m_d, impl->freq_hz_d, impl->uv_grid_fp16,
            n_baselines, Ng, Nf_this, f_start, impl->cell_size_rad, total_vis);
    } else {
        imaging_kernels::pillbox_grid_scatter_fp32in_kernel<<<grid, block, 0, stream>>>(
            static_cast<const float*>(vis_tile),
            impl->baseline_uv_m_d, impl->freq_hz_d, impl->uv_grid_fp16,
            n_baselines, Ng, Nf_this, f_start, impl->cell_size_rad, total_vis);
    }
    if (timing) record_event(ev1, stream);

    // Step 2: UV taper (FP16, optional)
    if (impl->taper_d) {
        const int64_t total_grid = static_cast<int64_t>(Nf_this) * grid_plane;
        const int tg = static_cast<int>(std::min(
            (total_grid + block - 1) / block, static_cast<int64_t>(65535)));
        imaging_kernels::apply_uv_taper_fp16_kernel<<<tg, block, 0, stream>>>(
            impl->uv_grid_fp16, impl->taper_d, Ng, total_grid);
    }
    if (timing) record_event(ev2, stream);

    // Step 3: FP16 IFFT in-place via cufftXt
    {
        bool is_last = (Nf_this < impl->Nf_tile);
        cufftHandle plan = (is_last && impl->fp16_last_valid)
                           ? impl->fft_plan_fp16_last
                           : impl->fft_plan_fp16_full;
        CUFFT_CHECK(cufftSetStream(plan, stream));
        CUFFT_CHECK(cufftXtExec(plan, impl->uv_grid_fp16,
                                 impl->uv_grid_fp16, CUFFT_INVERSE));
    }
    if (timing) record_event(ev3, stream);

    // Step 4: Beam extraction from FP16 image
    if (beam_tile && impl->n_beam > 0 && impl->beam_pixels_d) {
        const int64_t total_beam = static_cast<int64_t>(Nf_this) * impl->n_beam;
        const int bg = static_cast<int>(std::min(
            (total_beam + block - 1) / block, static_cast<int64_t>(65535)));
        const float norm = 1.0f / (static_cast<float>(Ng) * Ng);
        imaging_kernels::extract_beam_intensity_fp16_kernel<<<bg, block, 0, stream>>>(
            impl->uv_grid_fp16, impl->beam_pixels_d, beam_tile,
            Ng, impl->n_beam, norm, total_beam);
    }
    if (timing) {
        record_event(ev4, stream);
        impl->stage_times.scatter_ms += elapsed_ms(ev0, ev1);
        impl->stage_times.taper_ms   += elapsed_ms(ev1, ev2);
        impl->stage_times.fft_ms     += elapsed_ms(ev2, ev3);
        impl->stage_times.beam_ms    += elapsed_ms(ev3, ev4);
        cudaEventDestroy(ev0); cudaEventDestroy(ev1);
        cudaEventDestroy(ev2); cudaEventDestroy(ev3);
        cudaEventDestroy(ev4);
    }

    return 0;
}

// ================================================================
// process_tile() — FP32 FFT path
// ================================================================

int ImagingPipeline::Impl::process_tile_fp32_fft(Impl* impl,
                                  const void* vis_tile, float* beam_tile,
                                  int f_start, int Nf_this,
                                  VisPrecision prec, cudaStream_t stream)
{
    const int Ng = impl->Ng;
    const int n_baselines = impl->n_baselines;
    const int64_t grid_plane = static_cast<int64_t>(Ng) * Ng;
    const int block = 256;
    const int64_t total_vis = static_cast<int64_t>(Nf_this) * n_baselines;
    const int vis_grid = static_cast<int>(std::min(
        (total_vis + block - 1) / block, static_cast<int64_t>(65535)));
    const bool timing = impl->timing_enabled;

    cudaEvent_t ev0, ev1, ev2, ev3, ev4, ev5;
    if (timing) {
        cudaEventCreate(&ev0); cudaEventCreate(&ev1);
        cudaEventCreate(&ev2); cudaEventCreate(&ev3);
        cudaEventCreate(&ev4); cudaEventCreate(&ev5);
        record_event(ev0, stream);
    }

    if (prec == VisPrecision::FP32) {
        // ---- Direct FP32 path: scatter into FP32 grid, FFT in-place ----
        const size_t fp32_bytes = static_cast<size_t>(Nf_this) * grid_plane * sizeof(cufftComplex);
        CUDA_CHECK(cudaMemsetAsync(impl->uv_grid_fp32, 0, fp32_bytes, stream));

        imaging_kernels::pillbox_grid_scatter_fp32_fp32grid_kernel<<<vis_grid, block, 0, stream>>>(
            static_cast<const float*>(vis_tile),
            impl->baseline_uv_m_d, impl->freq_hz_d,
            reinterpret_cast<float*>(impl->uv_grid_fp32),
            n_baselines, Ng, Nf_this, f_start, impl->cell_size_rad, total_vis);
        if (timing) record_event(ev1, stream);

        if (impl->taper_d) {
            const int64_t total_grid = static_cast<int64_t>(Nf_this) * grid_plane;
            const int tg = static_cast<int>(std::min(
                (total_grid + block - 1) / block, static_cast<int64_t>(65535)));
            imaging_kernels::apply_uv_taper_fp32_kernel<<<tg, block, 0, stream>>>(
                reinterpret_cast<float*>(impl->uv_grid_fp32),
                impl->taper_d, Ng, total_grid);
        }
        if (timing) record_event(ev2, stream);
        // No widen step in direct FP32 path
        if (timing) record_event(ev3, stream);
    } else {
        // ---- Widen path: scatter into FP16 grid, widen, FFT FP32 ----
        const size_t fp16_bytes = static_cast<size_t>(Nf_this) * grid_plane * sizeof(__half2);
        CUDA_CHECK(cudaMemsetAsync(impl->uv_grid_fp16, 0, fp16_bytes, stream));

        imaging_kernels::pillbox_grid_scatter_fp16_kernel<<<vis_grid, block, 0, stream>>>(
            static_cast<const __half2*>(vis_tile),
            impl->baseline_uv_m_d, impl->freq_hz_d, impl->uv_grid_fp16,
            n_baselines, Ng, Nf_this, f_start, impl->cell_size_rad, total_vis);
        if (timing) record_event(ev1, stream);

        if (impl->taper_d) {
            const int64_t total_grid = static_cast<int64_t>(Nf_this) * grid_plane;
            const int tg = static_cast<int>(std::min(
                (total_grid + block - 1) / block, static_cast<int64_t>(65535)));
            imaging_kernels::apply_uv_taper_fp16_kernel<<<tg, block, 0, stream>>>(
                impl->uv_grid_fp16, impl->taper_d, Ng, total_grid);
        }
        if (timing) record_event(ev2, stream);

        // Widen FP16 -> FP32
        {
            const int64_t total_grid = static_cast<int64_t>(Nf_this) * grid_plane;
            const int cg = static_cast<int>(std::min(
                (total_grid + block - 1) / block, static_cast<int64_t>(65535)));
            imaging_kernels::cast_fp16_grid_to_fp32_kernel<<<cg, block, 0, stream>>>(
                impl->uv_grid_fp16, impl->uv_grid_fp32, total_grid);
        }
        if (timing) record_event(ev3, stream);
    }

    // FP32 IFFT in-place
    {
        bool is_last = (Nf_this < impl->Nf_tile);
        cufftHandle plan = (is_last && impl->fp32_last_valid)
                           ? impl->fft_plan_fp32_last
                           : impl->fft_plan_fp32_full;
        CUFFT_CHECK(cufftSetStream(plan, stream));
        CUFFT_CHECK(cufftExecC2C(plan, impl->uv_grid_fp32,
                                  impl->uv_grid_fp32, CUFFT_INVERSE));
    }
    if (timing) record_event(ev4, stream);

    // Beam extraction from FP32 image
    if (beam_tile && impl->n_beam > 0 && impl->beam_pixels_d) {
        const int64_t total_beam = static_cast<int64_t>(Nf_this) * impl->n_beam;
        const int bg = static_cast<int>(std::min(
            (total_beam + block - 1) / block, static_cast<int64_t>(65535)));
        const float norm = 1.0f / (static_cast<float>(Ng) * Ng);
        imaging_kernels::extract_beam_intensity_kernel<<<bg, block, 0, stream>>>(
            impl->uv_grid_fp32, impl->beam_pixels_d, beam_tile,
            Ng, impl->n_beam, norm, total_beam);
    }
    if (timing) {
        record_event(ev5, stream);
        impl->stage_times.scatter_ms += elapsed_ms(ev0, ev1);
        impl->stage_times.taper_ms   += elapsed_ms(ev1, ev2);
        impl->stage_times.widen_ms   += elapsed_ms(ev2, ev3);
        impl->stage_times.fft_ms     += elapsed_ms(ev3, ev4);
        impl->stage_times.beam_ms    += elapsed_ms(ev4, ev5);
        cudaEventDestroy(ev0); cudaEventDestroy(ev1);
        cudaEventDestroy(ev2); cudaEventDestroy(ev3);
        cudaEventDestroy(ev4); cudaEventDestroy(ev5);
    }

    return 0;
}

// ================================================================
// process_tile() — dispatch
// ================================================================

int ImagingPipeline::process_tile(const void* vis_tile, float* beam_tile,
                                   int f_start, int Nf_this,
                                   VisPrecision prec, cudaStream_t stream) {
    if (!impl_->configured) {
        fprintf(stderr, "process_tile: pipeline not configured\n");
        return -1;
    }

    if (impl_->fft_precision == FftPrecision::FP16)
        return Impl::process_tile_fp16_fft(impl_, vis_tile, beam_tile,
                                            f_start, Nf_this, prec, stream);
    else
        return Impl::process_tile_fp32_fft(impl_, vis_tile, beam_tile,
                                            f_start, Nf_this, prec, stream);
}

// ================================================================
// grid_and_image()
// ================================================================

int ImagingPipeline::grid_and_image(const void* vis_tri, float* beam_output,
                                     VisPrecision prec, cudaStream_t stream) {
    if (!impl_->configured) {
        fprintf(stderr, "grid_and_image: pipeline not configured\n");
        return -1;
    }

    // Reset accumulated timing for this call
    if (impl_->timing_enabled)
        impl_->stage_times = StageTimes{};

    const int n_baselines = impl_->n_baselines;

    for (int tile = 0; tile < impl_->n_tiles; tile++) {
        const int f_start = tile * impl_->Nf_tile;
        const int Nf_this = std::min(impl_->Nf_tile, impl_->Nf - f_start);

        const void* vis_tile;
        if (prec == VisPrecision::FP16) {
            vis_tile = static_cast<const __half2*>(vis_tri) +
                       static_cast<int64_t>(f_start) * n_baselines;
        } else {
            vis_tile = static_cast<const float*>(vis_tri) +
                       static_cast<int64_t>(f_start) * n_baselines * 2;
        }

        float* beam_tile = nullptr;
        if (beam_output && impl_->n_beam > 0) {
            beam_tile = beam_output + static_cast<int64_t>(f_start) * impl_->n_beam;
        }

        int err = process_tile(vis_tile, beam_tile, f_start, Nf_this, prec, stream);
        if (err != 0) return err;
    }

    // Compute total_ms as sum of sub-stages
    if (impl_->timing_enabled) {
        auto& st = impl_->stage_times;
        st.total_ms = st.scatter_ms + st.taper_ms + st.widen_ms
                    + st.fft_ms + st.beam_ms;
    }

    return 0;
}

// ================================================================
// Diagnostic accessors
// ================================================================

int ImagingPipeline::get_uv_grid(__half2* out, int tile_idx) {
    if (!impl_->configured || !impl_->uv_grid_fp16) return -1;
    const int64_t grid_plane = static_cast<int64_t>(impl_->Ng) * impl_->Ng;
    const int Nf_this = (tile_idx == impl_->n_tiles - 1)
                        ? impl_->last_tile_channels : impl_->Nf_tile;
    const size_t bytes = static_cast<size_t>(Nf_this) * grid_plane * sizeof(__half2);
    CUDA_CHECK(cudaMemcpy(out, impl_->uv_grid_fp16, bytes, cudaMemcpyDeviceToHost));
    return 0;
}

int ImagingPipeline::get_image_plane(float* out, int tile_idx) {
    if (!impl_->configured || !impl_->uv_grid_fp32) return -1;
    const int64_t grid_plane = static_cast<int64_t>(impl_->Ng) * impl_->Ng;
    const int Nf_this = (tile_idx == impl_->n_tiles - 1)
                        ? impl_->last_tile_channels : impl_->Nf_tile;
    const size_t bytes = static_cast<size_t>(Nf_this) * grid_plane * sizeof(cufftComplex);
    CUDA_CHECK(cudaMemcpy(out, impl_->uv_grid_fp32, bytes, cudaMemcpyDeviceToHost));
    return 0;
}

int ImagingPipeline::get_image_plane_fp16(__half2* out, int tile_idx) {
    if (!impl_->configured || !impl_->uv_grid_fp16 ||
        impl_->fft_precision != FftPrecision::FP16) return -1;
    const int64_t grid_plane = static_cast<int64_t>(impl_->Ng) * impl_->Ng;
    const int Nf_this = (tile_idx == impl_->n_tiles - 1)
                        ? impl_->last_tile_channels : impl_->Nf_tile;
    const size_t bytes = static_cast<size_t>(Nf_this) * grid_plane * sizeof(__half2);
    CUDA_CHECK(cudaMemcpy(out, impl_->uv_grid_fp16, bytes, cudaMemcpyDeviceToHost));
    return 0;
}

FftPrecision ImagingPipeline::get_fft_precision() const {
    return impl_->fft_precision;
}

int ImagingPipeline::get_nf_tile() const {
    return impl_->configured ? impl_->Nf_tile : 0;
}

int ImagingPipeline::get_num_tiles() const {
    return impl_->configured ? impl_->n_tiles : 0;
}

void ImagingPipeline::set_timing_enabled(bool enabled) {
    impl_->timing_enabled = enabled;
}

ImagingPipeline::StageTimes ImagingPipeline::get_stage_times() const {
    return impl_->stage_times;
}

void ImagingPipeline::destroy() {
    if (impl_) {
        impl_->free_buffers();
        impl_->free_plans();
        impl_->configured = false;
    }
}

} // namespace imaging_pipeline
