/** @file imaging_kernels.cuh
 *  @brief Standalone CUDA kernels for UV-plane gridding and imaging.
 *
 *  Pillbox scatter (FP16/FP32 input -> FP16/FP32 grid), UV taper,
 *  FP16->FP32 cast for cuFFT, and beam intensity extraction (FP16/FP32).
 *  No QUDA or CUTLASS dependencies -- only cuda_fp16.h + cuda_runtime.h.
 *
 *  NOTE: This header is intended for single-TU inclusion. If included by
 *  multiple .cu files that are linked together, you will get multiple-
 *  definition errors. In that case, include only imaging_pipeline.h
 *  and use the launcher wrappers instead.
 */

#ifndef IMAGING_KERNELS_CUH_
#define IMAGING_KERNELS_CUH_

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cufft.h>
#include <cstdint>

namespace imaging_kernels {

static constexpr double C_LIGHT = 299792458.0;

// ================================================================
//  Helper: decode packed-triangle index k -> (row, col)
// ================================================================
__device__ __forceinline__ void tri_decode(int k, int& row, int& col) {
    row = static_cast<int>((sqrtf(8.0f * k + 1.0f) - 1.0f) * 0.5f);
    col = k - row * (row + 1) / 2;
}

// ================================================================
//  Helper: UV cell from baseline + frequency
// ================================================================
__device__ __forceinline__ void uv_cell(
    float u_m, float v_m, double freq, float cell_size_rad,
    float half_ng, float& u_lam, float& v_lam, int& iu, int& iv)
{
    const double lambda = C_LIGHT / freq;
    u_lam = static_cast<float>(u_m / lambda);
    v_lam = static_cast<float>(v_m / lambda);
    iu = static_cast<int>(roundf(u_lam / cell_size_rad + half_ng));
    iv = static_cast<int>(roundf(v_lam / cell_size_rad + half_ng));
}

// ================================================================
// Kernel 1: Pillbox scatter -- FP16 vis -> FP16 grid
// ================================================================
__global__ void pillbox_grid_scatter_fp16_kernel(
    const __half2* __restrict__ vis_tri,
    const float*   __restrict__ baseline_uv_m,
    const double*  __restrict__ freq_hz,
    __half2*       __restrict__ uv_grid,
    int n_baselines, int Ng, int Nf_tile,
    int freq_offset, float cell_size_rad,
    int64_t total_elements)
{
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
    const int64_t grid_plane = static_cast<int64_t>(Ng) * Ng;
    const float half_ng = 0.5f * Ng;

    for (int64_t i = idx; i < total_elements; i += stride) {
        const int f_local = static_cast<int>(i / n_baselines);
        const int k = static_cast<int>(i % n_baselines);

        int row, col;
        tri_decode(k, row, col);
        const bool is_auto = (row == col);

        const __half2 vis = vis_tri[f_local * static_cast<int64_t>(n_baselines) + k];
        const float vis_re = __half2float(__low2half(vis));
        const float vis_im = __half2float(__high2half(vis));

        float u_lam, v_lam;
        int iu, iv;
        uv_cell(baseline_uv_m[k * 2], baseline_uv_m[k * 2 + 1],
                freq_hz[freq_offset + f_local], cell_size_rad, half_ng,
                u_lam, v_lam, iu, iv);

        if (iu >= 0 && iu < Ng && iv >= 0 && iv < Ng) {
            const int64_t grid_idx = f_local * grid_plane +
                                     static_cast<int64_t>(iv) * Ng + iu;
            atomicAdd(&uv_grid[grid_idx], vis);
        }

        if (!is_auto) {
            const int iu_c = static_cast<int>(roundf(-u_lam / cell_size_rad + half_ng));
            const int iv_c = static_cast<int>(roundf(-v_lam / cell_size_rad + half_ng));
            if (iu_c >= 0 && iu_c < Ng && iv_c >= 0 && iv_c < Ng) {
                const int64_t gi = f_local * grid_plane +
                                   static_cast<int64_t>(iv_c) * Ng + iu_c;
                atomicAdd(&uv_grid[gi],
                          __halves2half2(__float2half(vis_re), __float2half(-vis_im)));
            }
        }
    }
}

// ================================================================
// Kernel 2: Pillbox scatter -- FP32 vis -> FP16 grid
// ================================================================
__global__ void pillbox_grid_scatter_fp32in_kernel(
    const float*   __restrict__ vis_tri,
    const float*   __restrict__ baseline_uv_m,
    const double*  __restrict__ freq_hz,
    __half2*       __restrict__ uv_grid,
    int n_baselines, int Ng, int Nf_tile,
    int freq_offset, float cell_size_rad,
    int64_t total_elements)
{
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
    const int64_t grid_plane = static_cast<int64_t>(Ng) * Ng;
    const float half_ng = 0.5f * Ng;

    for (int64_t i = idx; i < total_elements; i += stride) {
        const int f_local = static_cast<int>(i / n_baselines);
        const int k = static_cast<int>(i % n_baselines);

        int row, col;
        tri_decode(k, row, col);
        const bool is_auto = (row == col);

        const int64_t vis_idx = (static_cast<int64_t>(f_local) * n_baselines + k) * 2;
        const float vis_re = vis_tri[vis_idx];
        const float vis_im = vis_tri[vis_idx + 1];
        const __half2 vis_h2 = __halves2half2(__float2half(vis_re), __float2half(vis_im));

        float u_lam, v_lam;
        int iu, iv;
        uv_cell(baseline_uv_m[k * 2], baseline_uv_m[k * 2 + 1],
                freq_hz[freq_offset + f_local], cell_size_rad, half_ng,
                u_lam, v_lam, iu, iv);

        if (iu >= 0 && iu < Ng && iv >= 0 && iv < Ng) {
            const int64_t gi = f_local * grid_plane +
                               static_cast<int64_t>(iv) * Ng + iu;
            atomicAdd(&uv_grid[gi], vis_h2);
        }

        if (!is_auto) {
            const int iu_c = static_cast<int>(roundf(-u_lam / cell_size_rad + half_ng));
            const int iv_c = static_cast<int>(roundf(-v_lam / cell_size_rad + half_ng));
            if (iu_c >= 0 && iu_c < Ng && iv_c >= 0 && iv_c < Ng) {
                const int64_t gi = f_local * grid_plane +
                                   static_cast<int64_t>(iv_c) * Ng + iu_c;
                atomicAdd(&uv_grid[gi],
                          __halves2half2(__float2half(vis_re), __float2half(-vis_im)));
            }
        }
    }
}

// ================================================================
// Kernel 3: Pillbox scatter -- FP16 vis -> FP32 grid
// ================================================================
/// For FP32 FFT path with FP16 correlator output.
/// Reads __half2 vis, converts to float, atomicAdd into cufftComplex grid.
__global__ void pillbox_grid_scatter_fp16in_fp32grid_kernel(
    const __half2* __restrict__ vis_tri,
    const float*   __restrict__ baseline_uv_m,
    const double*  __restrict__ freq_hz,
    float*         __restrict__ uv_grid,   // interleaved Re,Im float pairs
    int n_baselines, int Ng, int Nf_tile,
    int freq_offset, float cell_size_rad,
    int64_t total_elements)
{
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
    const int64_t grid_plane = static_cast<int64_t>(Ng) * Ng;
    const float half_ng = 0.5f * Ng;

    for (int64_t i = idx; i < total_elements; i += stride) {
        const int f_local = static_cast<int>(i / n_baselines);
        const int k = static_cast<int>(i % n_baselines);

        int row, col;
        tri_decode(k, row, col);
        const bool is_auto = (row == col);

        const __half2 vis = vis_tri[f_local * static_cast<int64_t>(n_baselines) + k];
        const float vis_re = __half2float(__low2half(vis));
        const float vis_im = __half2float(__high2half(vis));

        float u_lam, v_lam;
        int iu, iv;
        uv_cell(baseline_uv_m[k * 2], baseline_uv_m[k * 2 + 1],
                freq_hz[freq_offset + f_local], cell_size_rad, half_ng,
                u_lam, v_lam, iu, iv);

        if (iu >= 0 && iu < Ng && iv >= 0 && iv < Ng) {
            const int64_t gi = (f_local * grid_plane +
                                static_cast<int64_t>(iv) * Ng + iu) * 2;
            atomicAdd(&uv_grid[gi],     vis_re);
            atomicAdd(&uv_grid[gi + 1], vis_im);
        }

        if (!is_auto) {
            const int iu_c = static_cast<int>(roundf(-u_lam / cell_size_rad + half_ng));
            const int iv_c = static_cast<int>(roundf(-v_lam / cell_size_rad + half_ng));
            if (iu_c >= 0 && iu_c < Ng && iv_c >= 0 && iv_c < Ng) {
                const int64_t gi = (f_local * grid_plane +
                                    static_cast<int64_t>(iv_c) * Ng + iu_c) * 2;
                atomicAdd(&uv_grid[gi],      vis_re);
                atomicAdd(&uv_grid[gi + 1], -vis_im);
            }
        }
    }
}

// ================================================================
// Kernel 4: Pillbox scatter -- FP32 vis -> FP32 grid
// ================================================================
/// Direct FP32 path: no precision loss from FP16 intermediate.
/// Reads float pairs (Re, Im), atomicAdd into cufftComplex grid.
__global__ void pillbox_grid_scatter_fp32_fp32grid_kernel(
    const float*   __restrict__ vis_tri,
    const float*   __restrict__ baseline_uv_m,
    const double*  __restrict__ freq_hz,
    float*         __restrict__ uv_grid,   // interleaved Re,Im float pairs
    int n_baselines, int Ng, int Nf_tile,
    int freq_offset, float cell_size_rad,
    int64_t total_elements)
{
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
    const int64_t grid_plane = static_cast<int64_t>(Ng) * Ng;
    const float half_ng = 0.5f * Ng;

    for (int64_t i = idx; i < total_elements; i += stride) {
        const int f_local = static_cast<int>(i / n_baselines);
        const int k = static_cast<int>(i % n_baselines);

        int row, col;
        tri_decode(k, row, col);
        const bool is_auto = (row == col);

        const int64_t vis_idx = (static_cast<int64_t>(f_local) * n_baselines + k) * 2;
        const float vis_re = vis_tri[vis_idx];
        const float vis_im = vis_tri[vis_idx + 1];

        float u_lam, v_lam;
        int iu, iv;
        uv_cell(baseline_uv_m[k * 2], baseline_uv_m[k * 2 + 1],
                freq_hz[freq_offset + f_local], cell_size_rad, half_ng,
                u_lam, v_lam, iu, iv);

        if (iu >= 0 && iu < Ng && iv >= 0 && iv < Ng) {
            const int64_t gi = (f_local * grid_plane +
                                static_cast<int64_t>(iv) * Ng + iu) * 2;
            atomicAdd(&uv_grid[gi],     vis_re);
            atomicAdd(&uv_grid[gi + 1], vis_im);
        }

        if (!is_auto) {
            const int iu_c = static_cast<int>(roundf(-u_lam / cell_size_rad + half_ng));
            const int iv_c = static_cast<int>(roundf(-v_lam / cell_size_rad + half_ng));
            if (iu_c >= 0 && iu_c < Ng && iv_c >= 0 && iv_c < Ng) {
                const int64_t gi = (f_local * grid_plane +
                                    static_cast<int64_t>(iv_c) * Ng + iu_c) * 2;
                atomicAdd(&uv_grid[gi],      vis_re);
                atomicAdd(&uv_grid[gi + 1], -vis_im);
            }
        }
    }
}

// ================================================================
// Kernel 5: UV taper -- FP16 grid (in-place)
// ================================================================
__global__ void apply_uv_taper_fp16_kernel(
    __half2*       __restrict__ uv_grid,
    const float*   __restrict__ taper,
    int Ng, int64_t total_elements)
{
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
    const int64_t grid_plane = static_cast<int64_t>(Ng) * Ng;

    for (int64_t i = idx; i < total_elements; i += stride) {
        const float w = taper[i % grid_plane];
        const __half w_h = __float2half(w);
        __half2 val = uv_grid[i];
        uv_grid[i] = __halves2half2(__hmul(__low2half(val), w_h),
                                     __hmul(__high2half(val), w_h));
    }
}

// ================================================================
// Kernel 6: UV taper -- FP32 grid (in-place)
// ================================================================
__global__ void apply_uv_taper_fp32_kernel(
    float*         __restrict__ uv_grid,  // interleaved Re, Im
    const float*   __restrict__ taper,
    int Ng, int64_t total_elements)       // total_elements = Nf_tile * Ng * Ng
{
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
    const int64_t grid_plane = static_cast<int64_t>(Ng) * Ng;

    for (int64_t i = idx; i < total_elements; i += stride) {
        const float w = taper[i % grid_plane];
        uv_grid[i * 2]     *= w;
        uv_grid[i * 2 + 1] *= w;
    }
}

// ================================================================
// Kernel 7: Cast FP16 grid -> FP32 for cuFFT
// ================================================================
__global__ void cast_fp16_grid_to_fp32_kernel(
    const __half2*   __restrict__ grid_fp16,
    cufftComplex*    __restrict__ grid_fp32,
    int64_t total_elements)
{
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;

    for (int64_t i = idx; i < total_elements; i += stride) {
        const __half2 val = grid_fp16[i];
        grid_fp32[i].x = __half2float(__low2half(val));
        grid_fp32[i].y = __half2float(__high2half(val));
    }
}

// ================================================================
// Kernel 8: Beam extraction -- FP32 image (cufftComplex)
// ================================================================
__global__ void extract_beam_intensity_kernel(
    const cufftComplex* __restrict__ image,
    const int*          __restrict__ beam_pixels,
    float*              __restrict__ beam_output,
    int Ng, int n_beam, float norm,
    int64_t total_elements)
{
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
    const int64_t grid_plane = static_cast<int64_t>(Ng) * Ng;

    for (int64_t i = idx; i < total_elements; i += stride) {
        const int f_local = static_cast<int>(i / n_beam);
        const int b = static_cast<int>(i % n_beam);
        const int pix_row = beam_pixels[b * 2];
        const int pix_col = beam_pixels[b * 2 + 1];

        if (pix_row >= 0 && pix_row < Ng && pix_col >= 0 && pix_col < Ng) {
            const int64_t img_idx = f_local * grid_plane +
                                    static_cast<int64_t>(pix_row) * Ng + pix_col;
            const cufftComplex z = image[img_idx];
            beam_output[i] = (z.x * z.x + z.y * z.y) * norm;
        } else {
            beam_output[i] = 0.0f;
        }
    }
}

// ================================================================
// Kernel 9: Beam extraction -- FP16 image (__half2, post-cufftXt)
// ================================================================
/// For FP16 FFT path where IFFT output is __half2 directly.
__global__ void extract_beam_intensity_fp16_kernel(
    const __half2*  __restrict__ image,
    const int*      __restrict__ beam_pixels,
    float*          __restrict__ beam_output,
    int Ng, int n_beam, float norm,
    int64_t total_elements)
{
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
    const int64_t grid_plane = static_cast<int64_t>(Ng) * Ng;

    for (int64_t i = idx; i < total_elements; i += stride) {
        const int f_local = static_cast<int>(i / n_beam);
        const int b = static_cast<int>(i % n_beam);
        const int pix_row = beam_pixels[b * 2];
        const int pix_col = beam_pixels[b * 2 + 1];

        if (pix_row >= 0 && pix_row < Ng && pix_col >= 0 && pix_col < Ng) {
            const int64_t img_idx = f_local * grid_plane +
                                    static_cast<int64_t>(pix_row) * Ng + pix_col;
            const __half2 z = image[img_idx];
            const float re = __half2float(__low2half(z));
            const float im = __half2float(__high2half(z));
            beam_output[i] = (re * re + im * im) * norm;
        } else {
            beam_output[i] = 0.0f;
        }
    }
}

// ================================================================
// Host launcher wrappers
// ================================================================

inline void launch_scatter_fp16(
    const __half2* vis, const float* bl, const double* freq, __half2* grid,
    int n_bl, int Ng, int Nf, int f_off, float cs,
    cudaStream_t s = nullptr)
{
    const int64_t total = static_cast<int64_t>(Nf) * n_bl;
    const int block = 256;
    const int g = static_cast<int>((total + block - 1) / block);
    pillbox_grid_scatter_fp16_kernel<<<g, block, 0, s>>>(
        vis, bl, freq, grid, n_bl, Ng, Nf, f_off, cs, total);
}

inline void launch_scatter_fp32in(
    const float* vis, const float* bl, const double* freq, __half2* grid,
    int n_bl, int Ng, int Nf, int f_off, float cs,
    cudaStream_t s = nullptr)
{
    const int64_t total = static_cast<int64_t>(Nf) * n_bl;
    const int block = 256;
    const int g = static_cast<int>((total + block - 1) / block);
    pillbox_grid_scatter_fp32in_kernel<<<g, block, 0, s>>>(
        vis, bl, freq, grid, n_bl, Ng, Nf, f_off, cs, total);
}

inline void launch_scatter_fp16in_fp32grid(
    const __half2* vis, const float* bl, const double* freq, float* grid,
    int n_bl, int Ng, int Nf, int f_off, float cs,
    cudaStream_t s = nullptr)
{
    const int64_t total = static_cast<int64_t>(Nf) * n_bl;
    const int block = 256;
    const int g = static_cast<int>((total + block - 1) / block);
    pillbox_grid_scatter_fp16in_fp32grid_kernel<<<g, block, 0, s>>>(
        vis, bl, freq, grid, n_bl, Ng, Nf, f_off, cs, total);
}

inline void launch_scatter_fp32_fp32grid(
    const float* vis, const float* bl, const double* freq, float* grid,
    int n_bl, int Ng, int Nf, int f_off, float cs,
    cudaStream_t s = nullptr)
{
    const int64_t total = static_cast<int64_t>(Nf) * n_bl;
    const int block = 256;
    const int g = static_cast<int>((total + block - 1) / block);
    pillbox_grid_scatter_fp32_fp32grid_kernel<<<g, block, 0, s>>>(
        vis, bl, freq, grid, n_bl, Ng, Nf, f_off, cs, total);
}

inline void launch_cast_fp16_to_fp32(
    const __half2* in, cufftComplex* out, int64_t n,
    cudaStream_t s = nullptr)
{
    const int block = 256;
    const int g = static_cast<int>((n + block - 1) / block);
    cast_fp16_grid_to_fp32_kernel<<<g, block, 0, s>>>(in, out, n);
}

inline void launch_extract_beam(
    const cufftComplex* img, const int* pix, float* out,
    int Ng, int n_beam, float norm, int64_t n,
    cudaStream_t s = nullptr)
{
    const int block = 256;
    const int g = static_cast<int>((n + block - 1) / block);
    extract_beam_intensity_kernel<<<g, block, 0, s>>>(
        img, pix, out, Ng, n_beam, norm, n);
}

inline void launch_extract_beam_fp16(
    const __half2* img, const int* pix, float* out,
    int Ng, int n_beam, float norm, int64_t n,
    cudaStream_t s = nullptr)
{
    const int block = 256;
    const int g = static_cast<int>((n + block - 1) / block);
    extract_beam_intensity_fp16_kernel<<<g, block, 0, s>>>(
        img, pix, out, Ng, n_beam, norm, n);
}

} // namespace imaging_kernels

#endif // IMAGING_KERNELS_CUH_
