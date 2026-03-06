// ========================================================================================
// SM90 Pack/Triangle Kernels — Shared kernels + SM90-specific FP32 variants
// ========================================================================================

#include "../shared/pack_kernels_common.hpp"

// ========================================================================================
// FP32 Output Variants — Pack Triangle + Fused Anti-Symmetrize+Pack
// ========================================================================================
//
// FP32 variants of the pack kernels for native FP32 HERK output.
// Reads FP32 planar scratch buffers (from run_subgemms_fp8_fp32out) and writes
// directly to FP32 packed output, eliminating the FP16 intermediate and the
// widen_fp16_to_fp32 conversion.

/// FP32 variant of pack_triangle_kernel.
/// Reads float scratch and writes float packed output.
__global__ void pack_triangle_fp32_kernel(
    const float* __restrict__ full,
    const float* __restrict__ packed_old,
    float* __restrict__ packed_out,
    int N,
    int64_t batch_stride_full,
    int64_t batch_stride_packed,
    float beta,
    int fill_mode)
{
    const int major = blockIdx.x;
    if (major >= N) return;
    const int batch = blockIdx.y;

    const float* src = full       + batch * batch_stride_full;
    float*       dst = packed_out + batch * batch_stride_packed;
    const int64_t packed_start = static_cast<int64_t>(major) * (major + 1) / 2;
    const int num_minor = major + 1;

    if (fill_mode == 1) {
        const int64_t row_off = static_cast<int64_t>(major) * N;
        if (packed_old) {
            const float* old = packed_old + batch * batch_stride_packed;
            for (int c = threadIdx.x; c < num_minor; c += blockDim.x) {
                float val = src[row_off + c] + beta * old[packed_start + c];
                dst[packed_start + c] = val;
            }
        } else {
            for (int c = threadIdx.x; c < num_minor; c += blockDim.x) {
                dst[packed_start + c] = src[row_off + c];
            }
        }
    } else {
        if (packed_old) {
            const float* old = packed_old + batch * batch_stride_packed;
            for (int r = threadIdx.x; r < num_minor; r += blockDim.x) {
                float val = src[static_cast<int64_t>(r) * N + major]
                          + beta * old[packed_start + r];
                dst[packed_start + r] = val;
            }
        } else {
            for (int r = threadIdx.x; r < num_minor; r += blockDim.x) {
                dst[packed_start + r] = src[static_cast<int64_t>(r) * N + major];
            }
        }
    }
}

/// Launch FP32 triangle packing: full[N×N] float → packed[N*(N+1)/2] float
inline void pack_triangle_fp32(
    const float* full,
    float* packed,
    int N,
    FillMode fill,
    cudaStream_t stream = nullptr,
    int batch_count = 1,
    const float* packed_old = nullptr,
    float beta = 0.0f)
{
    int64_t total = static_cast<int64_t>(N) * (N + 1) / 2;
    int64_t batch_stride_full   = static_cast<int64_t>(N) * N;
    int64_t batch_stride_packed = total;
    int mode_int = (fill == FillMode::Upper) ? 0 : 1;
    TUNED_LAUNCH_ROW(pack_triangle_fp32_kernel, "pack_triangle_fp32",
        N, batch_count,
        (batch_stride_full + total) * 4 * batch_count, stream,
        full, packed_old, packed, N, batch_stride_full, batch_stride_packed, beta, mode_int);
    CUDA_CHECK(cudaGetLastError());
}

/// FP32 variant of antisymmetrize_pack_kernel.
/// Reads float temp (full N×N) and writes float packed output.
__global__ void antisymmetrize_pack_fp32_kernel(
    const float* __restrict__ temp,
    const float* __restrict__ C_imag_old,
    float* __restrict__ C_imag_packed,
    int N,
    int64_t batch_stride_full,
    int64_t batch_stride_packed,
    float alpha, float beta,
    int fill_mode)
{
    const int major = blockIdx.x;
    if (major >= N) return;
    const int batch = blockIdx.y;

    const float* t   = temp         + batch * batch_stride_full;
    const float* old = C_imag_old   + batch * batch_stride_packed;
    float*       out = C_imag_packed + batch * batch_stride_packed;
    const int64_t packed_start = static_cast<int64_t>(major) * (major + 1) / 2;
    const int num_minor = major + 1;

    if (fill_mode == 1) {
        const int row = major;
        const int64_t row_off = static_cast<int64_t>(row) * N;
        for (int col = threadIdx.x; col < num_minor; col += blockDim.x) {
            float result;
            if (row == col) {
                result = 0.0f;
            } else {
                float tij = t[row_off + col];
                float tji = t[static_cast<int64_t>(col) * N + row];
                result = alpha * (tij - tji) + beta * old[packed_start + col];
            }
            out[packed_start + col] = result;
        }
    } else {
        const int col = major;
        for (int row = threadIdx.x; row < num_minor; row += blockDim.x) {
            float result;
            if (row == col) {
                result = 0.0f;
            } else {
                float tij = t[static_cast<int64_t>(row) * N + col];
                float tji = t[static_cast<int64_t>(col) * N + row];
                result = alpha * (tij - tji) + beta * old[packed_start + row];
            }
            out[packed_start + row] = result;
        }
    }
}

/// Launch FP32 fused anti-symmetrize + pack: float temp[N×N] → float packed[N*(N+1)/2]
inline void antisymmetrize_pack_fp32(
    const float* temp,
    const float* C_imag_old,
    float* C_imag_packed,
    int N,
    float alpha, float beta,
    FillMode fill,
    cudaStream_t stream = nullptr,
    int batch_count = 1)
{
    int64_t total = static_cast<int64_t>(N) * (N + 1) / 2;
    int64_t batch_stride_full   = static_cast<int64_t>(N) * N;
    int64_t batch_stride_packed = total;
    int mode_int = (fill == FillMode::Upper) ? 0 : 1;
    TUNED_LAUNCH_ROW(antisymmetrize_pack_fp32_kernel, "antisymmetrize_pack_fp32",
        N, batch_count,
        (batch_stride_full + 2 * total) * 4 * batch_count, stream,
        temp, C_imag_old, C_imag_packed, N,
        batch_stride_full, batch_stride_packed,
        alpha, beta, mode_int);
    CUDA_CHECK(cudaGetLastError());
}

/// FP32 variant of pack_antisymmetrize_triangle_kernel.
/// Reads FP32 planar scratch buffers and writes FP32 packed Re + packed Im output.
__global__ void pack_antisymmetrize_triangle_fp32_kernel(
    const float* __restrict__ scratch_Re,
    const float* __restrict__ temp_Im,
    const float* __restrict__ old_Re,
    const float* __restrict__ old_Im,
    float* __restrict__ out_Re,
    float* __restrict__ out_Im,
    int N,
    int64_t batch_stride_full,
    int64_t batch_stride_packed,
    float alpha, float beta,
    int fill_mode)
{
    const int major = blockIdx.x;
    if (major >= N) return;
    const int batch = blockIdx.y;

    const float* re_src = scratch_Re + batch * batch_stride_full;
    const float* im_src = temp_Im    + batch * batch_stride_full;
    float*       re_dst = out_Re     + batch * batch_stride_packed;
    float*       im_dst = out_Im     + batch * batch_stride_packed;
    const int64_t packed_start = static_cast<int64_t>(major) * (major + 1) / 2;
    const int num_minor = major + 1;

    if (fill_mode == 1) {
        const int64_t row_off = static_cast<int64_t>(major) * N;
        if (old_Re) {
            const float* re_old = old_Re + batch * batch_stride_packed;
            const float* im_old = old_Im + batch * batch_stride_packed;
            for (int c = threadIdx.x; c < num_minor; c += blockDim.x) {
                float re = re_src[row_off + c] + beta * re_old[packed_start + c];
                re_dst[packed_start + c] = re;

                float im;
                if (major == c) {
                    im = 0.0f;
                } else {
                    float tij = im_src[row_off + c];
                    float tji = im_src[static_cast<int64_t>(c) * N + major];
                    im = alpha * (tij - tji) + beta * im_old[packed_start + c];
                }
                im_dst[packed_start + c] = im;
            }
        } else {
            for (int c = threadIdx.x; c < num_minor; c += blockDim.x) {
                re_dst[packed_start + c] = re_src[row_off + c];

                float im;
                if (major == c) {
                    im = 0.0f;
                } else {
                    float tij = im_src[row_off + c];
                    float tji = im_src[static_cast<int64_t>(c) * N + major];
                    im = alpha * (tij - tji);
                }
                im_dst[packed_start + c] = im;
            }
        }
    } else {
        if (old_Re) {
            const float* re_old = old_Re + batch * batch_stride_packed;
            const float* im_old = old_Im + batch * batch_stride_packed;
            for (int r = threadIdx.x; r < num_minor; r += blockDim.x) {
                float re = re_src[static_cast<int64_t>(r) * N + major]
                         + beta * re_old[packed_start + r];
                re_dst[packed_start + r] = re;

                float im;
                if (r == major) {
                    im = 0.0f;
                } else {
                    float tij = im_src[static_cast<int64_t>(r) * N + major];
                    float tji = im_src[static_cast<int64_t>(major) * N + r];
                    im = alpha * (tij - tji) + beta * im_old[packed_start + r];
                }
                im_dst[packed_start + r] = im;
            }
        } else {
            for (int r = threadIdx.x; r < num_minor; r += blockDim.x) {
                re_dst[packed_start + r] = re_src[static_cast<int64_t>(r) * N + major];

                float im;
                if (r == major) {
                    im = 0.0f;
                } else {
                    float tij = im_src[static_cast<int64_t>(r) * N + major];
                    float tji = im_src[static_cast<int64_t>(major) * N + r];
                    im = alpha * (tij - tji);
                }
                im_dst[packed_start + r] = im;
            }
        }
    }
}

/// Launch FP32 fused pack + antisymmetrize: float scratch_Re[N×N] + float temp_Im[N×N]
/// → float packed Re[N*(N+1)/2] + float packed Im[N*(N+1)/2]
inline void pack_antisymmetrize_triangle_fp32(
    const float* scratch_Re,
    const float* temp_Im,
    float* out_Re,
    float* out_Im,
    int N,
    float alpha, float beta,
    FillMode fill,
    cudaStream_t stream = nullptr,
    int batch_count = 1,
    const float* old_Re = nullptr,
    const float* old_Im = nullptr)
{
    int64_t batch_stride_full   = static_cast<int64_t>(N) * N;
    int64_t batch_stride_packed = static_cast<int64_t>(N) * (N + 1) / 2;
    int mode_int = (fill == FillMode::Upper) ? 0 : 1;
    TUNED_LAUNCH_ROW(pack_antisymmetrize_triangle_fp32_kernel, "pack_antisymmetrize_triangle_fp32",
        N, batch_count,
        (2 * batch_stride_full + 4 * batch_stride_packed) * 4 * batch_count, stream,
        scratch_Re, temp_Im, old_Re, old_Im, out_Re, out_Im,
        N, batch_stride_full, batch_stride_packed,
        alpha, beta, mode_int);
    CUDA_CHECK(cudaGetLastError());
}

/// FP32 interleave kernel: planar float Re[n] + Im[n] → interleaved float [Re0,Im0,Re1,Im1,...]
__global__ void interleave_fp32_kernel(
    const float* __restrict__ real_in,
    const float* __restrict__ imag_in,
    float* __restrict__ interleaved_out,
    int64_t num_complex_elements)
{
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < num_complex_elements) {
        interleaved_out[2 * idx]     = real_in[idx];
        interleaved_out[2 * idx + 1] = imag_in[idx];
    }
}

/// Launch FP32 interleave: planar float → interleaved float
inline void interleave_complex_fp32(
    const float* real_in, const float* imag_in,
    float* interleaved_out,
    int64_t num_complex_elements,
    cudaStream_t stream = nullptr)
{
    constexpr int kBlock = 256;
    int grid = static_cast<int>((num_complex_elements + kBlock - 1) / kBlock);
    interleave_fp32_kernel<<<grid, kBlock, 0, stream>>>(
        real_in, imag_in, interleaved_out, num_complex_elements);
    CUDA_CHECK(cudaGetLastError());
}
