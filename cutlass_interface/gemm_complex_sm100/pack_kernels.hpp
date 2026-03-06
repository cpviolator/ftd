// ========================================================================================
// SM100/SM120 Pack/Triangle Kernels — Shared kernels + SM100-specific extensions
// ========================================================================================

#include "../shared/pack_kernels_common.hpp"

// ========================================================================================
// SM100-Specific: Fused pack + antisymmetrize + interleave
// ========================================================================================
//
// Fused kernel that reads N×N planar Re and Im scratch buffers and writes directly
// to interleaved packed triangle output [Re0,Im0,Re1,Im1,...]. Eliminates the
// intermediate planar packed Re/Im buffers and the separate interleave pass.

__global__ void pack_antisymmetrize_interleave_kernel_sm100(
    const __half* __restrict__ scratch_Re,    // [batch × N × N] full Re result
    const __half* __restrict__ temp_Im,       // [batch × N × N] full Im result
    __half* __restrict__ C_interleaved,       // Output: [batch × packed × 2] interleaved
    const __half* __restrict__ C_old_interleaved, // Previous C (interleaved packed, for beta != 0)
    int N,
    int64_t batch_stride_full,                // N × N
    int64_t batch_stride_packed_interleaved,  // N*(N+1)/2 * 2 (interleaved pairs)
    float alpha, float beta,
    int fill_mode)
{
    const int major = blockIdx.x;
    if (major >= N) return;
    const int batch = blockIdx.y;

    const __half* re_src = scratch_Re + batch * batch_stride_full;
    const __half* im_src = temp_Im    + batch * batch_stride_full;
    __half*       out    = C_interleaved + batch * batch_stride_packed_interleaved;
    const int64_t packed_start = static_cast<int64_t>(major) * (major + 1) / 2;
    const int num_minor = major + 1;

    if (fill_mode == 1) {
        // Lower: row=major, cols [0..major]
        const int64_t row_off = static_cast<int64_t>(major) * N;
        if (C_old_interleaved) {
            const __half* old_c = C_old_interleaved + batch * batch_stride_packed_interleaved;
            for (int c = threadIdx.x; c < num_minor; c += blockDim.x) {
                int64_t out_idx = (packed_start + c) * 2;
                float re = __half2float(re_src[row_off + c])
                         + beta * __half2float(old_c[out_idx]);
                out[out_idx] = __float2half(re);

                float im;
                if (major == c) {
                    im = 0.0f;
                } else {
                    float tij = __half2float(im_src[row_off + c]);
                    float tji = __half2float(im_src[static_cast<int64_t>(c) * N + major]);
                    im = alpha * (tij - tji) + beta * __half2float(old_c[out_idx + 1]);
                }
                out[out_idx + 1] = __float2half(im);
            }
        } else {
            for (int c = threadIdx.x; c < num_minor; c += blockDim.x) {
                int64_t out_idx = (packed_start + c) * 2;
                out[out_idx] = re_src[row_off + c];

                float im;
                if (major == c) {
                    im = 0.0f;
                } else {
                    float tij = __half2float(im_src[row_off + c]);
                    float tji = __half2float(im_src[static_cast<int64_t>(c) * N + major]);
                    im = alpha * (tij - tji);
                }
                out[out_idx + 1] = __float2half(im);
            }
        }
    } else {
        // Upper: col=major, rows [0..major]
        if (C_old_interleaved) {
            const __half* old_c = C_old_interleaved + batch * batch_stride_packed_interleaved;
            for (int r = threadIdx.x; r < num_minor; r += blockDim.x) {
                int64_t out_idx = (packed_start + r) * 2;
                float re = __half2float(re_src[static_cast<int64_t>(r) * N + major])
                         + beta * __half2float(old_c[out_idx]);
                out[out_idx] = __float2half(re);

                float im;
                if (r == major) {
                    im = 0.0f;
                } else {
                    float tij = __half2float(im_src[static_cast<int64_t>(r) * N + major]);
                    float tji = __half2float(im_src[static_cast<int64_t>(major) * N + r]);
                    im = alpha * (tij - tji) + beta * __half2float(old_c[out_idx + 1]);
                }
                out[out_idx + 1] = __float2half(im);
            }
        } else {
            for (int r = threadIdx.x; r < num_minor; r += blockDim.x) {
                int64_t out_idx = (packed_start + r) * 2;
                out[out_idx] = re_src[static_cast<int64_t>(r) * N + major];

                float im;
                if (r == major) {
                    im = 0.0f;
                } else {
                    float tij = __half2float(im_src[static_cast<int64_t>(r) * N + major]);
                    float tji = __half2float(im_src[static_cast<int64_t>(major) * N + r]);
                    im = alpha * (tij - tji);
                }
                out[out_idx + 1] = __float2half(im);
            }
        }
    }
}

/// Launch fused pack + antisymmetrize + interleave: scratch_Re[N×N] + temp_Im[N×N]
/// → interleaved packed output [Re0,Im0,Re1,Im1,...].
/// Eliminates Cr_packed/Ci_packed intermediate buffers and the separate interleave pass.
inline void pack_antisymmetrize_interleave(
    const __half* scratch_Re,
    const __half* temp_Im,
    __half* C_interleaved,          // Output: packed interleaved [Re,Im,Re,Im,...]
    int N,
    float alpha, float beta,
    FillMode fill,
    cudaStream_t stream = nullptr,
    int batch_count = 1,
    const __half* C_old_interleaved = nullptr)   // For beta != 0: old C in interleaved packed format
{
    int64_t batch_stride_full = static_cast<int64_t>(N) * N;
    int64_t batch_stride_packed_interleaved = static_cast<int64_t>(N) * (N + 1);  // N*(N+1)/2 * 2
    int mode_int = (fill == FillMode::Upper) ? 0 : 1;
    // read 2×N² full + interleaved packed old + write interleaved packed
    int64_t total_bytes = (2 * batch_stride_full + 2 * batch_stride_packed_interleaved) * 2 * batch_count;
    TUNED_LAUNCH_ROW(pack_antisymmetrize_interleave_kernel_sm100, "pack_antisymmetrize_interleave",
        N, batch_count, total_bytes, stream,
        scratch_Re, temp_Im, C_interleaved,
        C_old_interleaved,
        N, batch_stride_full, batch_stride_packed_interleaved,
        alpha, beta, mode_int);
    CUDA_CHECK(cudaGetLastError());
}

// ========================================================================================
// FP32 Output Variants — Pack + Antisymmetrize + Interleave for FP32 scratch → FP32 output
// ========================================================================================

/// FP32 variant of pack_antisymmetrize_interleave_kernel_sm100.
/// Reads FP32 planar scratch buffers (from run_subgemms_fp8_fp32out) and writes
/// directly to interleaved packed FP32 output, eliminating the FP16 intermediate
/// and the widen_fp16_to_fp32 conversion.
__global__ void pack_antisymmetrize_interleave_fp32_kernel_sm100(
    const float* __restrict__ scratch_Re,      // [batch × N × N] full Re result (FP32)
    const float* __restrict__ temp_Im,         // [batch × N × N] full Im result (FP32)
    float* __restrict__ C_interleaved,         // Output: [batch × packed × 2] interleaved FP32
    const float* __restrict__ C_old_interleaved, // Previous C (interleaved packed FP32, for beta != 0)
    int N,
    int64_t batch_stride_full,
    int64_t batch_stride_packed_interleaved,
    float alpha, float beta,
    int fill_mode)
{
    const int major = blockIdx.x;
    if (major >= N) return;
    const int batch = blockIdx.y;

    const float* re_src = scratch_Re + batch * batch_stride_full;
    const float* im_src = temp_Im    + batch * batch_stride_full;
    float*       out    = C_interleaved + batch * batch_stride_packed_interleaved;
    const int64_t packed_start = static_cast<int64_t>(major) * (major + 1) / 2;
    const int num_minor = major + 1;

    if (fill_mode == 1) {
        // Lower: row=major, cols [0..major]
        const int64_t row_off = static_cast<int64_t>(major) * N;
        if (C_old_interleaved) {
            const float* old_c = C_old_interleaved + batch * batch_stride_packed_interleaved;
            for (int c = threadIdx.x; c < num_minor; c += blockDim.x) {
                int64_t out_idx = (packed_start + c) * 2;
                float re = re_src[row_off + c] + beta * old_c[out_idx];
                out[out_idx] = re;

                float im;
                if (major == c) {
                    im = 0.0f;
                } else {
                    float tij = im_src[row_off + c];
                    float tji = im_src[static_cast<int64_t>(c) * N + major];
                    im = alpha * (tij - tji) + beta * old_c[out_idx + 1];
                }
                out[out_idx + 1] = im;
            }
        } else {
            for (int c = threadIdx.x; c < num_minor; c += blockDim.x) {
                int64_t out_idx = (packed_start + c) * 2;
                out[out_idx] = re_src[row_off + c];

                float im;
                if (major == c) {
                    im = 0.0f;
                } else {
                    float tij = im_src[row_off + c];
                    float tji = im_src[static_cast<int64_t>(c) * N + major];
                    im = alpha * (tij - tji);
                }
                out[out_idx + 1] = im;
            }
        }
    } else {
        // Upper: col=major, rows [0..major]
        if (C_old_interleaved) {
            const float* old_c = C_old_interleaved + batch * batch_stride_packed_interleaved;
            for (int r = threadIdx.x; r < num_minor; r += blockDim.x) {
                int64_t out_idx = (packed_start + r) * 2;
                float re = re_src[static_cast<int64_t>(r) * N + major]
                         + beta * old_c[out_idx];
                out[out_idx] = re;

                float im;
                if (r == major) {
                    im = 0.0f;
                } else {
                    float tij = im_src[static_cast<int64_t>(r) * N + major];
                    float tji = im_src[static_cast<int64_t>(major) * N + r];
                    im = alpha * (tij - tji) + beta * old_c[out_idx + 1];
                }
                out[out_idx + 1] = im;
            }
        } else {
            for (int r = threadIdx.x; r < num_minor; r += blockDim.x) {
                int64_t out_idx = (packed_start + r) * 2;
                out[out_idx] = re_src[static_cast<int64_t>(r) * N + major];

                float im;
                if (r == major) {
                    im = 0.0f;
                } else {
                    float tij = im_src[static_cast<int64_t>(r) * N + major];
                    float tji = im_src[static_cast<int64_t>(major) * N + r];
                    im = alpha * (tij - tji);
                }
                out[out_idx + 1] = im;
            }
        }
    }
}

/// Launch fused pack + antisymmetrize + interleave for FP32 scratch → FP32 output.
inline void pack_antisymmetrize_interleave_fp32(
    const float* scratch_Re,
    const float* temp_Im,
    float* C_interleaved,
    int N,
    float alpha, float beta,
    FillMode fill,
    cudaStream_t stream = nullptr,
    int batch_count = 1,
    const float* C_old_interleaved = nullptr)
{
    int64_t batch_stride_full = static_cast<int64_t>(N) * N;
    int64_t batch_stride_packed_interleaved = static_cast<int64_t>(N) * (N + 1);  // N*(N+1)/2 * 2
    int mode_int = (fill == FillMode::Upper) ? 0 : 1;
    // read 2×N² full (4B each) + interleaved packed old + write interleaved packed
    int64_t total_bytes = (2 * batch_stride_full + 2 * batch_stride_packed_interleaved) * 4 * batch_count;
    TUNED_LAUNCH_ROW(pack_antisymmetrize_interleave_fp32_kernel_sm100, "pack_antisymmetrize_interleave_fp32",
        N, batch_count, total_bytes, stream,
        scratch_Re, temp_Im, C_interleaved,
        C_old_interleaved,
        N, batch_stride_full, batch_stride_packed_interleaved,
        alpha, beta, mode_int);
    CUDA_CHECK(cudaGetLastError());
}
