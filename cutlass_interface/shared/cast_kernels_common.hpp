// ========================================================================================
// Shared Cast/Transpose Kernels — Common between SM90 and SM100/SM120
// ========================================================================================
//
// All 28 CUDA kernels and their wrapper functions for FP16↔FP8 conversion,
// transpose, deinterleave, dual/quad output, paired cast, stacked-K, and
// FP8 native deinterleave.
//
// This header is #include'd from within each architecture's cast_kernels.hpp.

#pragma once

// ========================================================================================
// FP16 → FP8 Conversion Kernel (Preprocessing)
// ========================================================================================

/// Vectorized FP16 → FP8 E4M3 conversion kernel
/// Processes 8 elements per thread (128-bit load of 8×FP16, 64-bit store of 8×FP8)
__global__ void cast_fp16_to_fp8_e4m3_kernel(
    const __half* __restrict__ input,
    __nv_fp8_e4m3*  __restrict__ output,
    int64_t num_elements)
{
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;

    for (int64_t i = idx * 8; i < num_elements; i += stride * 8) {
        if (i + 7 < num_elements) {
            const float4* in_vec = reinterpret_cast<const float4*>(input + i);
            float4 data = *in_vec;
            const __half* halfs = reinterpret_cast<const __half*>(&data);

            __nv_fp8_e4m3 out[8];
            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                out[j] = __nv_fp8_e4m3(__float2half_rn(
                    fminf(fmaxf(__half2float(halfs[j]), -448.0f), 448.0f)
                ));
            }

            uint2* out_vec = reinterpret_cast<uint2*>(output + i);
            *out_vec = *reinterpret_cast<const uint2*>(out);
        } else {
            for (int64_t j = i; j < num_elements && j < i + 8; ++j) {
                float val = __half2float(input[j]);
                val = fminf(fmaxf(val, -448.0f), 448.0f);
                output[j] = __nv_fp8_e4m3(val);
            }
        }
    }
}

/// Launch helper for FP16 → FP8 conversion
inline void cast_fp16_to_fp8_e4m3(
    const __half* input,
    __nv_fp8_e4m3* output,
    int64_t num_elements,
    cudaStream_t stream = nullptr)
{
    int64_t total_bytes = num_elements * 3;  // read FP16 (2B) + write FP8 (1B)
    TUNED_LAUNCH_1D(cast_fp16_to_fp8_e4m3_kernel, "cast_fp16_to_fp8",
        num_elements, 8, total_bytes, stream,
        input, output, num_elements);
    CUDA_CHECK(cudaGetLastError());
}


// ========================================================================================
// FP8 Transpose Kernel
// ========================================================================================

constexpr int kTransposeTile = 32;

__global__ void transpose_fp8_kernel(
    const cutlass::float_e4m3_t* __restrict__ input,
    cutlass::float_e4m3_t* __restrict__ output,
    int rows, int cols)
{
    __shared__ uint8_t tile[kTransposeTile][kTransposeTile + 1];

    const int bx = blockIdx.x * kTransposeTile;
    const int by = blockIdx.y * kTransposeTile;

    {
        int row = by + threadIdx.x;
        int col = bx + threadIdx.y;
        if (row < rows && col < cols) {
            tile[threadIdx.x][threadIdx.y] =
                reinterpret_cast<const uint8_t*>(input)[row + static_cast<int64_t>(col) * rows];
        }
    }

    __syncthreads();

    {
        int out_row = bx + threadIdx.x;
        int out_col = by + threadIdx.y;
        if (out_row < cols && out_col < rows) {
            reinterpret_cast<uint8_t*>(output)[out_row + static_cast<int64_t>(out_col) * cols] =
                tile[threadIdx.y][threadIdx.x];
        }
    }
}

inline void transpose_fp8(
    const cutlass::float_e4m3_t* input,
    cutlass::float_e4m3_t* output,
    int rows, int cols,
    cudaStream_t stream = nullptr)
{
    dim3 block(kTransposeTile, kTransposeTile);
    dim3 grid((cols + kTransposeTile - 1) / kTransposeTile,
              (rows + kTransposeTile - 1) / kTransposeTile);
    transpose_fp8_kernel<<<grid, block, 0, stream>>>(input, output, rows, cols);
    CUDA_CHECK(cudaGetLastError());
}


// ========================================================================================
// Fused FP16→FP8 Cast + Transpose Kernel
// ========================================================================================

__global__ void cast_fp16_to_fp8_transposed_kernel(
    const __half* __restrict__ input,
    uint8_t* __restrict__ output,
    int rows, int cols,
    int64_t batch_stride_in,
    int64_t batch_stride_out)
{
    __shared__ uint8_t tile[kTransposeTile][kTransposeTile + 1];

    const int bx = blockIdx.x * kTransposeTile;
    const int by = blockIdx.y * kTransposeTile;
    const int batch = blockIdx.z;

    const __half* in_b  = input  + batch * batch_stride_in;
    uint8_t*      out_b = output + batch * batch_stride_out;

    {
        int row = by + threadIdx.x;
        int col = bx + threadIdx.y;
        if (row < rows && col < cols) {
            float val = __half2float(in_b[row + static_cast<int64_t>(col) * rows]);
            val = fminf(fmaxf(val, -448.0f), 448.0f);
            __nv_fp8_e4m3 fp8_val(__float2half_rn(val));
            tile[threadIdx.x][threadIdx.y] = reinterpret_cast<const uint8_t&>(fp8_val);
        }
    }

    __syncthreads();

    {
        int out_row = bx + threadIdx.x;
        int out_col = by + threadIdx.y;
        if (out_row < cols && out_col < rows) {
            out_b[out_row + static_cast<int64_t>(out_col) * cols] =
                tile[threadIdx.y][threadIdx.x];
        }
    }
}

inline void cast_fp16_to_fp8_e4m3_transposed(
    const __half* input,
    cutlass::float_e4m3_t* output,
    int rows, int cols,
    cudaStream_t stream = nullptr,
    int batch_count = 1)
{
    dim3 block(kTransposeTile, kTransposeTile);
    dim3 grid((cols + kTransposeTile - 1) / kTransposeTile,
              (rows + kTransposeTile - 1) / kTransposeTile,
              batch_count);
    int64_t stride = static_cast<int64_t>(rows) * cols;
    cast_fp16_to_fp8_transposed_kernel<<<grid, block, 0, stream>>>(
        input, reinterpret_cast<uint8_t*>(output), rows, cols, stride, stride);
    CUDA_CHECK(cudaGetLastError());
}


// ========================================================================================
// F1: Fused Deinterleave + Cast (interleaved FP16 → 2 × planar FP8)
// ========================================================================================

__global__ void deinterleave_cast_fp16_to_fp8_kernel(
    const __half* __restrict__ interleaved,
    __nv_fp8_e4m3* __restrict__ out_real,
    __nv_fp8_e4m3* __restrict__ out_imag,
    int64_t num_complex_elements)
{
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;

    for (int64_t i = idx * 4; i < num_complex_elements; i += stride * 4) {
        if (i + 3 < num_complex_elements) {
            const float4* in_vec = reinterpret_cast<const float4*>(interleaved + 2 * i);
            float4 data = *in_vec;
            const __half* halfs = reinterpret_cast<const __half*>(&data);

            __nv_fp8_e4m3 re[4], im[4];
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                float r = __half2float(halfs[2 * j]);
                float m = __half2float(halfs[2 * j + 1]);
                r = fminf(fmaxf(r, -448.0f), 448.0f);
                m = fminf(fmaxf(m, -448.0f), 448.0f);
                re[j] = __nv_fp8_e4m3(__float2half_rn(r));
                im[j] = __nv_fp8_e4m3(__float2half_rn(m));
            }

            *reinterpret_cast<uint32_t*>(out_real + i) =
                *reinterpret_cast<const uint32_t*>(re);
            *reinterpret_cast<uint32_t*>(out_imag + i) =
                *reinterpret_cast<const uint32_t*>(im);
        } else {
            for (int64_t j = i; j < num_complex_elements && j < i + 4; ++j) {
                float r = __half2float(interleaved[2 * j]);
                float m = __half2float(interleaved[2 * j + 1]);
                r = fminf(fmaxf(r, -448.0f), 448.0f);
                m = fminf(fmaxf(m, -448.0f), 448.0f);
                out_real[j] = __nv_fp8_e4m3(r);
                out_imag[j] = __nv_fp8_e4m3(m);
            }
        }
    }
}

inline void deinterleave_cast_fp16_to_fp8(
    const __half* interleaved,
    __nv_fp8_e4m3* out_real, __nv_fp8_e4m3* out_imag,
    int64_t num_complex_elements,
    cudaStream_t stream = nullptr)
{
    int64_t total_bytes = num_complex_elements * 6;
    TUNED_LAUNCH_1D(deinterleave_cast_fp16_to_fp8_kernel, "deinterleave_cast_fp16_to_fp8",
        num_complex_elements, 4, total_bytes, stream,
        interleaved, out_real, out_imag, num_complex_elements);
    CUDA_CHECK(cudaGetLastError());
}


// ========================================================================================
// F1: Fused Deinterleave + Cast + Transpose
// ========================================================================================

__global__ void deinterleave_cast_fp16_to_fp8_transposed_kernel(
    const __half* __restrict__ input,
    uint8_t* __restrict__ out_real,
    uint8_t* __restrict__ out_imag,
    int rows, int cols,
    int64_t batch_stride_in,
    int64_t batch_stride_out)
{
    __shared__ uint8_t tile_re[kTransposeTile][kTransposeTile + 1];
    __shared__ uint8_t tile_im[kTransposeTile][kTransposeTile + 1];

    const int bx = blockIdx.x * kTransposeTile;
    const int by = blockIdx.y * kTransposeTile;
    const int batch = blockIdx.z;

    const __half* in_b  = input    + batch * batch_stride_in * 2;
    uint8_t*      re_b  = out_real + batch * batch_stride_out;
    uint8_t*      im_b  = out_imag + batch * batch_stride_out;

    {
        int row = by + threadIdx.x;
        int col = bx + threadIdx.y;
        if (row < rows && col < cols) {
            int64_t base = 2 * (row + static_cast<int64_t>(col) * rows);
            float re_val = __half2float(in_b[base]);
            float im_val = __half2float(in_b[base + 1]);
            re_val = fminf(fmaxf(re_val, -448.0f), 448.0f);
            im_val = fminf(fmaxf(im_val, -448.0f), 448.0f);
            __nv_fp8_e4m3 fp8_re(__float2half_rn(re_val));
            __nv_fp8_e4m3 fp8_im(__float2half_rn(im_val));
            tile_re[threadIdx.x][threadIdx.y] = reinterpret_cast<const uint8_t&>(fp8_re);
            tile_im[threadIdx.x][threadIdx.y] = reinterpret_cast<const uint8_t&>(fp8_im);
        }
    }

    __syncthreads();

    {
        int out_row = bx + threadIdx.x;
        int out_col = by + threadIdx.y;
        if (out_row < cols && out_col < rows) {
            int64_t out_idx = out_row + static_cast<int64_t>(out_col) * cols;
            re_b[out_idx] = tile_re[threadIdx.y][threadIdx.x];
            im_b[out_idx] = tile_im[threadIdx.y][threadIdx.x];
        }
    }
}

inline void deinterleave_cast_fp16_to_fp8_transposed(
    const __half* input,
    cutlass::float_e4m3_t* out_real, cutlass::float_e4m3_t* out_imag,
    int rows, int cols,
    cudaStream_t stream = nullptr,
    int batch_count = 1)
{
    dim3 block(kTransposeTile, kTransposeTile);
    dim3 grid((cols + kTransposeTile - 1) / kTransposeTile,
              (rows + kTransposeTile - 1) / kTransposeTile,
              batch_count);
    int64_t stride = static_cast<int64_t>(rows) * cols;
    deinterleave_cast_fp16_to_fp8_transposed_kernel<<<grid, block, 0, stream>>>(
        input, reinterpret_cast<uint8_t*>(out_real),
        reinterpret_cast<uint8_t*>(out_imag),
        rows, cols, stride, stride);
    CUDA_CHECK(cudaGetLastError());
}


// ========================================================================================
// F2: Fused Cast + Duplicate (planar FP16 → 2× planar FP8)
// ========================================================================================

__global__ void cast_fp16_to_fp8_e4m3_dual_kernel(
    const __half* __restrict__ input,
    __nv_fp8_e4m3* __restrict__ output1,
    __nv_fp8_e4m3* __restrict__ output2,
    int64_t num_elements)
{
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;

    for (int64_t i = idx * 8; i < num_elements; i += stride * 8) {
        if (i + 7 < num_elements) {
            const float4* in_vec = reinterpret_cast<const float4*>(input + i);
            float4 data = *in_vec;
            const __half* halfs = reinterpret_cast<const __half*>(&data);

            __nv_fp8_e4m3 out[8];
            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                out[j] = __nv_fp8_e4m3(__float2half_rn(
                    fminf(fmaxf(__half2float(halfs[j]), -448.0f), 448.0f)
                ));
            }

            uint2 packed = *reinterpret_cast<const uint2*>(out);
            *reinterpret_cast<uint2*>(output1 + i) = packed;
            *reinterpret_cast<uint2*>(output2 + i) = packed;
        } else {
            for (int64_t j = i; j < num_elements && j < i + 8; ++j) {
                float val = __half2float(input[j]);
                val = fminf(fmaxf(val, -448.0f), 448.0f);
                __nv_fp8_e4m3 fp8_val(val);
                output1[j] = fp8_val;
                output2[j] = fp8_val;
            }
        }
    }
}

inline void cast_fp16_to_fp8_e4m3_dual(
    const __half* input,
    __nv_fp8_e4m3* output1, __nv_fp8_e4m3* output2,
    int64_t num_elements,
    cudaStream_t stream = nullptr)
{
    int64_t total_bytes = num_elements * 4;
    TUNED_LAUNCH_1D(cast_fp16_to_fp8_e4m3_dual_kernel, "cast_fp16_to_fp8_dual",
        num_elements, 8, total_bytes, stream,
        input, output1, output2, num_elements);
    CUDA_CHECK(cudaGetLastError());
}

/// Vectorized FP16 → 4× FP8 E4M3 quad-output kernel (linear layout)
__global__ void cast_fp16_to_fp8_e4m3_quad_kernel(
    const __half* __restrict__ input1,
    const __half* __restrict__ input2,
    __nv_fp8_e4m3* __restrict__ out1_a,
    __nv_fp8_e4m3* __restrict__ out1_b,
    __nv_fp8_e4m3* __restrict__ out2_a,
    __nv_fp8_e4m3* __restrict__ out2_b,
    int64_t num_elements)
{
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;

    for (int64_t i = idx * 8; i < num_elements; i += stride * 8) {
        if (i + 7 < num_elements) {
            float4 data1 = *reinterpret_cast<const float4*>(input1 + i);
            const __half* halfs1 = reinterpret_cast<const __half*>(&data1);
            __nv_fp8_e4m3 fp8_1[8];
            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                fp8_1[j] = __nv_fp8_e4m3(__float2half_rn(
                    fminf(fmaxf(__half2float(halfs1[j]), -448.0f), 448.0f)
                ));
            }
            uint2 packed1 = *reinterpret_cast<const uint2*>(fp8_1);
            *reinterpret_cast<uint2*>(out1_a + i) = packed1;
            *reinterpret_cast<uint2*>(out1_b + i) = packed1;

            float4 data2 = *reinterpret_cast<const float4*>(input2 + i);
            const __half* halfs2 = reinterpret_cast<const __half*>(&data2);
            __nv_fp8_e4m3 fp8_2[8];
            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                fp8_2[j] = __nv_fp8_e4m3(__float2half_rn(
                    fminf(fmaxf(__half2float(halfs2[j]), -448.0f), 448.0f)
                ));
            }
            uint2 packed2 = *reinterpret_cast<const uint2*>(fp8_2);
            *reinterpret_cast<uint2*>(out2_a + i) = packed2;
            *reinterpret_cast<uint2*>(out2_b + i) = packed2;
        } else {
            for (int64_t j = i; j < num_elements && j < i + 8; ++j) {
                float val1 = __half2float(input1[j]);
                val1 = fminf(fmaxf(val1, -448.0f), 448.0f);
                __nv_fp8_e4m3 fp8_val1(val1);
                out1_a[j] = fp8_val1;
                out1_b[j] = fp8_val1;

                float val2 = __half2float(input2[j]);
                val2 = fminf(fmaxf(val2, -448.0f), 448.0f);
                __nv_fp8_e4m3 fp8_val2(val2);
                out2_a[j] = fp8_val2;
                out2_b[j] = fp8_val2;
            }
        }
    }
}

inline void cast_fp16_to_fp8_e4m3_quad(
    const __half* input1, const __half* input2,
    __nv_fp8_e4m3* out1_a, __nv_fp8_e4m3* out1_b,
    __nv_fp8_e4m3* out2_a, __nv_fp8_e4m3* out2_b,
    int64_t num_elements,
    cudaStream_t stream = nullptr)
{
    int64_t total_bytes = num_elements * 8;
    TUNED_LAUNCH_1D(cast_fp16_to_fp8_e4m3_quad_kernel, "cast_fp16_to_fp8_quad",
        num_elements, 8, total_bytes, stream,
        input1, input2, out1_a, out1_b, out2_a, out2_b, num_elements);
    CUDA_CHECK(cudaGetLastError());
}


// ========================================================================================
// Strategy 5D: Stacked FP16 → FP8 cast + separate outputs
// ========================================================================================

__global__ void cast_fp16_to_fp8_stacked_and_separate_kernel(
    const __half* __restrict__ input_real,
    const __half* __restrict__ input_imag,
    __nv_fp8_e4m3* __restrict__ stacked_out,
    __nv_fp8_e4m3* __restrict__ xi_out,
    __nv_fp8_e4m3* __restrict__ xr_out,
    int M, int K)
{
    const int64_t total = static_cast<int64_t>(M) * K;
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
    const int64_t K2 = static_cast<int64_t>(K) * 2;

    for (int64_t i = idx * 8; i < total; i += stride * 8) {
        if (i + 7 < total) {
            float4 data_r = *reinterpret_cast<const float4*>(input_real + i);
            float4 data_i = *reinterpret_cast<const float4*>(input_imag + i);
            const __half* halfs_r = reinterpret_cast<const __half*>(&data_r);
            const __half* halfs_i = reinterpret_cast<const __half*>(&data_i);

            __nv_fp8_e4m3 fp8_r[8], fp8_i[8];
            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                fp8_r[j] = __nv_fp8_e4m3(__float2half_rn(
                    fminf(fmaxf(__half2float(halfs_r[j]), -448.0f), 448.0f)
                ));
                fp8_i[j] = __nv_fp8_e4m3(__float2half_rn(
                    fminf(fmaxf(__half2float(halfs_i[j]), -448.0f), 448.0f)
                ));
            }
            uint2 packed_r = *reinterpret_cast<const uint2*>(fp8_r);
            uint2 packed_i = *reinterpret_cast<const uint2*>(fp8_i);

            *reinterpret_cast<uint2*>(xr_out + i) = packed_r;
            *reinterpret_cast<uint2*>(xi_out + i) = packed_i;

            int row = static_cast<int>(i / K);
            int col = static_cast<int>(i % K);
            int64_t s_base = static_cast<int64_t>(row) * K2 + col;
            if (col + 8 <= K) {
                *reinterpret_cast<uint2*>(stacked_out + s_base)     = packed_r;
                *reinterpret_cast<uint2*>(stacked_out + s_base + K) = packed_i;
            } else {
                for (int j = 0; j < 8; ++j) {
                    int r = static_cast<int>((i + j) / K);
                    int c = static_cast<int>((i + j) % K);
                    int64_t so = static_cast<int64_t>(r) * K2 + c;
                    stacked_out[so]     = fp8_r[j];
                    stacked_out[so + K] = fp8_i[j];
                }
            }
        } else {
            for (int64_t j = i; j < total && j < i + 8; ++j) {
                float vr = __half2float(input_real[j]);
                float vi = __half2float(input_imag[j]);
                vr = fminf(fmaxf(vr, -448.0f), 448.0f);
                vi = fminf(fmaxf(vi, -448.0f), 448.0f);
                __nv_fp8_e4m3 fr(vr), fi(vi);
                xr_out[j] = fr;
                xi_out[j] = fi;
                int row = static_cast<int>(j / K);
                int col = static_cast<int>(j % K);
                stacked_out[static_cast<int64_t>(row) * K2 + col]     = fr;
                stacked_out[static_cast<int64_t>(row) * K2 + K + col] = fi;
            }
        }
    }
}

inline void cast_fp16_to_fp8_stacked_and_separate(
    const __half* input_real, const __half* input_imag,
    __nv_fp8_e4m3* stacked_out, __nv_fp8_e4m3* xi_out, __nv_fp8_e4m3* xr_out,
    int M, int K, cudaStream_t stream = nullptr)
{
    int64_t total = static_cast<int64_t>(M) * K;
    int64_t total_bytes = total * 8;
    TUNED_LAUNCH_1D(cast_fp16_to_fp8_stacked_and_separate_kernel, "cast_fp16_to_fp8_stacked",
        total, 8, total_bytes, stream,
        input_real, input_imag, stacked_out, xi_out, xr_out, M, K);
    CUDA_CHECK(cudaGetLastError());
}


// ========================================================================================
// Fused FP16 → 2× FP8 dual-output + transpose
// ========================================================================================

__global__ void cast_fp16_to_fp8_transposed_dual_kernel(
    const __half* __restrict__ input,
    uint8_t* __restrict__ output1,
    uint8_t* __restrict__ output2,
    int rows, int cols,
    int64_t batch_stride_in,
    int64_t batch_stride_out)
{
    __shared__ uint8_t tile[kTransposeTile][kTransposeTile + 1];

    const int bx = blockIdx.x * kTransposeTile;
    const int by = blockIdx.y * kTransposeTile;
    const int batch = blockIdx.z;

    const __half* in_b   = input   + batch * batch_stride_in;
    uint8_t*      out1_b = output1 + batch * batch_stride_out;
    uint8_t*      out2_b = output2 + batch * batch_stride_out;

    {
        int row = by + threadIdx.x;
        int col = bx + threadIdx.y;
        if (row < rows && col < cols) {
            float val = __half2float(in_b[row + static_cast<int64_t>(col) * rows]);
            val = fminf(fmaxf(val, -448.0f), 448.0f);
            __nv_fp8_e4m3 fp8_val(__float2half_rn(val));
            tile[threadIdx.x][threadIdx.y] = reinterpret_cast<const uint8_t&>(fp8_val);
        }
    }

    __syncthreads();

    {
        int out_row = bx + threadIdx.x;
        int out_col = by + threadIdx.y;
        if (out_row < cols && out_col < rows) {
            int64_t out_idx = out_row + static_cast<int64_t>(out_col) * cols;
            uint8_t val = tile[threadIdx.y][threadIdx.x];
            out1_b[out_idx] = val;
            out2_b[out_idx] = val;
        }
    }
}

inline void cast_fp16_to_fp8_e4m3_transposed_dual(
    const __half* input,
    cutlass::float_e4m3_t* output1, cutlass::float_e4m3_t* output2,
    int rows, int cols,
    cudaStream_t stream = nullptr,
    int batch_count = 1)
{
    dim3 block(kTransposeTile, kTransposeTile);
    dim3 grid((cols + kTransposeTile - 1) / kTransposeTile,
              (rows + kTransposeTile - 1) / kTransposeTile,
              batch_count);
    int64_t stride = static_cast<int64_t>(rows) * cols;
    cast_fp16_to_fp8_transposed_dual_kernel<<<grid, block, 0, stream>>>(
        input, reinterpret_cast<uint8_t*>(output1),
        reinterpret_cast<uint8_t*>(output2),
        rows, cols, stride, stride);
    CUDA_CHECK(cudaGetLastError());
}

/// Fused FP16 → 4× FP8 quad-output + transpose
__global__ void cast_fp16_to_fp8_transposed_quad_kernel(
    const __half* __restrict__ input1,
    const __half* __restrict__ input2,
    uint8_t* __restrict__ out1_a,
    uint8_t* __restrict__ out1_b,
    uint8_t* __restrict__ out2_a,
    uint8_t* __restrict__ out2_b,
    int rows, int cols,
    int64_t batch_stride_in,
    int64_t batch_stride_out)
{
    __shared__ uint8_t tile1[kTransposeTile][kTransposeTile + 1];
    __shared__ uint8_t tile2[kTransposeTile][kTransposeTile + 1];

    const int bx = blockIdx.x * kTransposeTile;
    const int by = blockIdx.y * kTransposeTile;
    const int batch = blockIdx.z;

    const __half* in1_b  = input1 + batch * batch_stride_in;
    const __half* in2_b  = input2 + batch * batch_stride_in;
    uint8_t*      o1a_b  = out1_a + batch * batch_stride_out;
    uint8_t*      o1b_b  = out1_b + batch * batch_stride_out;
    uint8_t*      o2a_b  = out2_a + batch * batch_stride_out;
    uint8_t*      o2b_b  = out2_b + batch * batch_stride_out;

    {
        int row = by + threadIdx.x;
        int col = bx + threadIdx.y;
        if (row < rows && col < cols) {
            int64_t in_idx = row + static_cast<int64_t>(col) * rows;
            float val1 = __half2float(in1_b[in_idx]);
            val1 = fminf(fmaxf(val1, -448.0f), 448.0f);
            __nv_fp8_e4m3 fp8_1(__float2half_rn(val1));
            tile1[threadIdx.x][threadIdx.y] = reinterpret_cast<const uint8_t&>(fp8_1);

            float val2 = __half2float(in2_b[in_idx]);
            val2 = fminf(fmaxf(val2, -448.0f), 448.0f);
            __nv_fp8_e4m3 fp8_2(__float2half_rn(val2));
            tile2[threadIdx.x][threadIdx.y] = reinterpret_cast<const uint8_t&>(fp8_2);
        }
    }

    __syncthreads();

    {
        int out_row = bx + threadIdx.x;
        int out_col = by + threadIdx.y;
        if (out_row < cols && out_col < rows) {
            int64_t out_idx = out_row + static_cast<int64_t>(out_col) * cols;
            uint8_t v1 = tile1[threadIdx.y][threadIdx.x];
            o1a_b[out_idx] = v1;
            o1b_b[out_idx] = v1;
            uint8_t v2 = tile2[threadIdx.y][threadIdx.x];
            o2a_b[out_idx] = v2;
            o2b_b[out_idx] = v2;
        }
    }
}

inline void cast_fp16_to_fp8_e4m3_transposed_quad(
    const __half* input1, const __half* input2,
    cutlass::float_e4m3_t* out1_a, cutlass::float_e4m3_t* out1_b,
    cutlass::float_e4m3_t* out2_a, cutlass::float_e4m3_t* out2_b,
    int rows, int cols,
    cudaStream_t stream = nullptr,
    int batch_count = 1)
{
    dim3 block(kTransposeTile, kTransposeTile);
    dim3 grid((cols + kTransposeTile - 1) / kTransposeTile,
              (rows + kTransposeTile - 1) / kTransposeTile,
              batch_count);
    int64_t stride = static_cast<int64_t>(rows) * cols;
    cast_fp16_to_fp8_transposed_quad_kernel<<<grid, block, 0, stream>>>(
        input1, input2,
        reinterpret_cast<uint8_t*>(out1_a), reinterpret_cast<uint8_t*>(out1_b),
        reinterpret_cast<uint8_t*>(out2_a), reinterpret_cast<uint8_t*>(out2_b),
        rows, cols, stride, stride);
    CUDA_CHECK(cudaGetLastError());
}


// ========================================================================================
// F2: Fused Deinterleave + Cast + Duplicate (interleaved FP16 → 4× planar FP8)
// ========================================================================================

__global__ void deinterleave_cast_fp16_to_fp8_dual_kernel(
    const __half* __restrict__ interleaved,
    __nv_fp8_e4m3* __restrict__ out_real_A,
    __nv_fp8_e4m3* __restrict__ out_imag_A,
    __nv_fp8_e4m3* __restrict__ out_real_B,
    __nv_fp8_e4m3* __restrict__ out_imag_B,
    int64_t num_complex_elements)
{
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;

    for (int64_t i = idx * 4; i < num_complex_elements; i += stride * 4) {
        if (i + 3 < num_complex_elements) {
            const float4* in_vec = reinterpret_cast<const float4*>(interleaved + 2 * i);
            float4 data = *in_vec;
            const __half* halfs = reinterpret_cast<const __half*>(&data);

            __nv_fp8_e4m3 re[4], im[4];
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                float r = __half2float(halfs[2 * j]);
                float m = __half2float(halfs[2 * j + 1]);
                r = fminf(fmaxf(r, -448.0f), 448.0f);
                m = fminf(fmaxf(m, -448.0f), 448.0f);
                re[j] = __nv_fp8_e4m3(__float2half_rn(r));
                im[j] = __nv_fp8_e4m3(__float2half_rn(m));
            }

            uint32_t packed_re = *reinterpret_cast<const uint32_t*>(re);
            uint32_t packed_im = *reinterpret_cast<const uint32_t*>(im);
            *reinterpret_cast<uint32_t*>(out_real_A + i) = packed_re;
            *reinterpret_cast<uint32_t*>(out_imag_A + i) = packed_im;
            *reinterpret_cast<uint32_t*>(out_real_B + i) = packed_re;
            *reinterpret_cast<uint32_t*>(out_imag_B + i) = packed_im;
        } else {
            for (int64_t j = i; j < num_complex_elements && j < i + 4; ++j) {
                float r = __half2float(interleaved[2 * j]);
                float m = __half2float(interleaved[2 * j + 1]);
                r = fminf(fmaxf(r, -448.0f), 448.0f);
                m = fminf(fmaxf(m, -448.0f), 448.0f);
                __nv_fp8_e4m3 fp8_re(r);
                __nv_fp8_e4m3 fp8_im(m);
                out_real_A[j] = fp8_re;  out_real_B[j] = fp8_re;
                out_imag_A[j] = fp8_im;  out_imag_B[j] = fp8_im;
            }
        }
    }
}

inline void deinterleave_cast_fp16_to_fp8_dual(
    const __half* interleaved,
    __nv_fp8_e4m3* out_real_A, __nv_fp8_e4m3* out_imag_A,
    __nv_fp8_e4m3* out_real_B, __nv_fp8_e4m3* out_imag_B,
    int64_t num_complex_elements,
    cudaStream_t stream = nullptr)
{
    int64_t total_bytes = num_complex_elements * 8;
    TUNED_LAUNCH_1D(deinterleave_cast_fp16_to_fp8_dual_kernel, "deinterleave_cast_fp16_to_fp8_dual",
        num_complex_elements, 4, total_bytes, stream,
        interleaved, out_real_A, out_imag_A, out_real_B, out_imag_B, num_complex_elements);
    CUDA_CHECK(cudaGetLastError());
}


/// Fused deinterleave + cast + transpose + dual-output kernel
__global__ void deinterleave_cast_fp16_to_fp8_transposed_dual_kernel(
    const __half* __restrict__ input,
    uint8_t* __restrict__ out_real_A,
    uint8_t* __restrict__ out_imag_A,
    uint8_t* __restrict__ out_real_B,
    uint8_t* __restrict__ out_imag_B,
    int rows, int cols,
    int64_t batch_stride_in,
    int64_t batch_stride_out)
{
    __shared__ uint8_t tile_re[kTransposeTile][kTransposeTile + 1];
    __shared__ uint8_t tile_im[kTransposeTile][kTransposeTile + 1];

    const int bx = blockIdx.x * kTransposeTile;
    const int by = blockIdx.y * kTransposeTile;
    const int batch = blockIdx.z;

    const __half* in_b    = input      + batch * batch_stride_in * 2;
    uint8_t*      reA_b   = out_real_A + batch * batch_stride_out;
    uint8_t*      imA_b   = out_imag_A + batch * batch_stride_out;
    uint8_t*      reB_b   = out_real_B + batch * batch_stride_out;
    uint8_t*      imB_b   = out_imag_B + batch * batch_stride_out;

    {
        int row = by + threadIdx.x;
        int col = bx + threadIdx.y;
        if (row < rows && col < cols) {
            int64_t base = 2 * (row + static_cast<int64_t>(col) * rows);
            float re_val = __half2float(in_b[base]);
            float im_val = __half2float(in_b[base + 1]);
            re_val = fminf(fmaxf(re_val, -448.0f), 448.0f);
            im_val = fminf(fmaxf(im_val, -448.0f), 448.0f);
            __nv_fp8_e4m3 fp8_re(__float2half_rn(re_val));
            __nv_fp8_e4m3 fp8_im(__float2half_rn(im_val));
            tile_re[threadIdx.x][threadIdx.y] = reinterpret_cast<const uint8_t&>(fp8_re);
            tile_im[threadIdx.x][threadIdx.y] = reinterpret_cast<const uint8_t&>(fp8_im);
        }
    }

    __syncthreads();

    {
        int out_row = bx + threadIdx.x;
        int out_col = by + threadIdx.y;
        if (out_row < cols && out_col < rows) {
            int64_t out_idx = out_row + static_cast<int64_t>(out_col) * cols;
            uint8_t re_val = tile_re[threadIdx.y][threadIdx.x];
            uint8_t im_val = tile_im[threadIdx.y][threadIdx.x];
            reA_b[out_idx] = re_val;  reB_b[out_idx] = re_val;
            imA_b[out_idx] = im_val;  imB_b[out_idx] = im_val;
        }
    }
}

inline void deinterleave_cast_fp16_to_fp8_transposed_dual(
    const __half* input,
    cutlass::float_e4m3_t* out_real_A, cutlass::float_e4m3_t* out_imag_A,
    cutlass::float_e4m3_t* out_real_B, cutlass::float_e4m3_t* out_imag_B,
    int rows, int cols,
    cudaStream_t stream = nullptr,
    int batch_count = 1)
{
    dim3 block(kTransposeTile, kTransposeTile);
    dim3 grid((cols + kTransposeTile - 1) / kTransposeTile,
              (rows + kTransposeTile - 1) / kTransposeTile,
              batch_count);
    int64_t stride = static_cast<int64_t>(rows) * cols;
    deinterleave_cast_fp16_to_fp8_transposed_dual_kernel<<<grid, block, 0, stream>>>(
        input, reinterpret_cast<uint8_t*>(out_real_A),
        reinterpret_cast<uint8_t*>(out_imag_A),
        reinterpret_cast<uint8_t*>(out_real_B),
        reinterpret_cast<uint8_t*>(out_imag_B),
        rows, cols, stride, stride);
    CUDA_CHECK(cudaGetLastError());
}


// ========================================================================================
// F5: Fused Paired Cast (2 independent inputs → 2 independent outputs)
// ========================================================================================

__global__ void cast_fp16_to_fp8_e4m3_paired_kernel(
    const __half* __restrict__ input1,
    const __half* __restrict__ input2,
    __nv_fp8_e4m3* __restrict__ output1,
    __nv_fp8_e4m3* __restrict__ output2,
    int64_t num_elements)
{
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;

    for (int64_t i = idx * 8; i < num_elements; i += stride * 8) {
        if (i + 7 < num_elements) {
            const float4* in_vec1 = reinterpret_cast<const float4*>(input1 + i);
            float4 data1 = *in_vec1;
            const __half* halfs1 = reinterpret_cast<const __half*>(&data1);

            const float4* in_vec2 = reinterpret_cast<const float4*>(input2 + i);
            float4 data2 = *in_vec2;
            const __half* halfs2 = reinterpret_cast<const __half*>(&data2);

            __nv_fp8_e4m3 out1[8], out2[8];
            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                out1[j] = __nv_fp8_e4m3(__float2half_rn(
                    fminf(fmaxf(__half2float(halfs1[j]), -448.0f), 448.0f)
                ));
                out2[j] = __nv_fp8_e4m3(__float2half_rn(
                    fminf(fmaxf(__half2float(halfs2[j]), -448.0f), 448.0f)
                ));
            }

            *reinterpret_cast<uint2*>(output1 + i) = *reinterpret_cast<const uint2*>(out1);
            *reinterpret_cast<uint2*>(output2 + i) = *reinterpret_cast<const uint2*>(out2);
        } else {
            for (int64_t j = i; j < num_elements && j < i + 8; ++j) {
                float val1 = __half2float(input1[j]);
                val1 = fminf(fmaxf(val1, -448.0f), 448.0f);
                output1[j] = __nv_fp8_e4m3(val1);

                float val2 = __half2float(input2[j]);
                val2 = fminf(fmaxf(val2, -448.0f), 448.0f);
                output2[j] = __nv_fp8_e4m3(val2);
            }
        }
    }
}

inline void cast_fp16_to_fp8_e4m3_paired(
    const __half* input1, const __half* input2,
    __nv_fp8_e4m3* output1, __nv_fp8_e4m3* output2,
    int64_t num_elements,
    cudaStream_t stream = nullptr)
{
    int64_t total_bytes = num_elements * 6;
    TUNED_LAUNCH_1D(cast_fp16_to_fp8_e4m3_paired_kernel, "cast_fp16_to_fp8_paired",
        num_elements, 8, total_bytes, stream,
        input1, input2, output1, output2, num_elements);
    CUDA_CHECK(cudaGetLastError());
}


// ========================================================================================
// Strategy 5D: Stacked-K GEMM Preprocessing Kernels
// ========================================================================================

__global__ void stack_fp8_kernel(
    const uint8_t* __restrict__ src1,
    const uint8_t* __restrict__ src2,
    uint8_t* __restrict__ dst,
    int M, int K)
{
    const int64_t total = static_cast<int64_t>(M) * K;
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
    const int64_t K2 = static_cast<int64_t>(K) * 2;

    for (int64_t i = idx * 16; i < total; i += stride * 16) {
        if (i + 15 < total) {
            uint4 d1 = *reinterpret_cast<const uint4*>(src1 + i);
            uint4 d2 = *reinterpret_cast<const uint4*>(src2 + i);
            int row = static_cast<int>(i / K);
            int col = static_cast<int>(i % K);
            int64_t base = static_cast<int64_t>(row) * K2 + col;
            if (col + 16 <= K) {
                *reinterpret_cast<uint4*>(dst + base)     = d1;
                *reinterpret_cast<uint4*>(dst + base + K) = d2;
            } else {
                const uint8_t* b1 = reinterpret_cast<const uint8_t*>(&d1);
                const uint8_t* b2 = reinterpret_cast<const uint8_t*>(&d2);
                for (int j = 0; j < 16; ++j) {
                    int64_t pos = i + j;
                    int r = static_cast<int>(pos / K);
                    int c = static_cast<int>(pos % K);
                    int64_t out = static_cast<int64_t>(r) * K2 + c;
                    dst[out]     = b1[j];
                    dst[out + K] = b2[j];
                }
            }
        } else {
            for (int64_t j = i; j < total && j < i + 16; ++j) {
                int r = static_cast<int>(j / K);
                int c = static_cast<int>(j % K);
                int64_t out = static_cast<int64_t>(r) * K2 + c;
                dst[out]     = src1[j];
                dst[out + K] = src2[j];
            }
        }
    }
}

__global__ void negate_and_stack_fp8_kernel(
    const uint8_t* __restrict__ src1,
    const uint8_t* __restrict__ src2,
    uint8_t* __restrict__ dst,
    int M, int K)
{
    const int64_t total = static_cast<int64_t>(M) * K;
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
    const int64_t K2 = static_cast<int64_t>(K) * 2;

    for (int64_t i = idx * 16; i < total; i += stride * 16) {
        if (i + 15 < total) {
            uint4 d1 = *reinterpret_cast<const uint4*>(src1 + i);
            uint4 d2 = *reinterpret_cast<const uint4*>(src2 + i);
            uint32_t* neg = reinterpret_cast<uint32_t*>(&d2);
            neg[0] ^= 0x80808080u;
            neg[1] ^= 0x80808080u;
            neg[2] ^= 0x80808080u;
            neg[3] ^= 0x80808080u;
            int row = static_cast<int>(i / K);
            int col = static_cast<int>(i % K);
            int64_t base = static_cast<int64_t>(row) * K2 + col;
            if (col + 16 <= K) {
                *reinterpret_cast<uint4*>(dst + base)     = d1;
                *reinterpret_cast<uint4*>(dst + base + K) = d2;
            } else {
                const uint8_t* b1 = reinterpret_cast<const uint8_t*>(&d1);
                const uint8_t* b2 = reinterpret_cast<const uint8_t*>(&d2);
                for (int j = 0; j < 16; ++j) {
                    int64_t pos = i + j;
                    int r = static_cast<int>(pos / K);
                    int c = static_cast<int>(pos % K);
                    int64_t out = static_cast<int64_t>(r) * K2 + c;
                    dst[out]     = b1[j];
                    dst[out + K] = b2[j];
                }
            }
        } else {
            for (int64_t j = i; j < total && j < i + 16; ++j) {
                int r = static_cast<int>(j / K);
                int c = static_cast<int>(j % K);
                int64_t out = static_cast<int64_t>(r) * K2 + c;
                dst[out]     = src1[j];
                dst[out + K] = src2[j] ^ 0x80;
            }
        }
    }
}

inline void stack_fp8(
    const cutlass::float_e4m3_t* src1,
    const cutlass::float_e4m3_t* src2,
    cutlass::float_e4m3_t* dst,
    int M, int K, cudaStream_t stream = nullptr)
{
    int64_t total = static_cast<int64_t>(M) * K;
    int64_t total_bytes = total * 4;
    TUNED_LAUNCH_1D(stack_fp8_kernel, "stack_fp8",
        total, 16, total_bytes, stream,
        reinterpret_cast<const uint8_t*>(src1),
        reinterpret_cast<const uint8_t*>(src2),
        reinterpret_cast<uint8_t*>(dst), M, K);
    CUDA_CHECK(cudaGetLastError());
}

inline void negate_and_stack_fp8(
    const cutlass::float_e4m3_t* src1,
    const cutlass::float_e4m3_t* src2_negated,
    cutlass::float_e4m3_t* dst,
    int M, int K, cudaStream_t stream = nullptr)
{
    int64_t total = static_cast<int64_t>(M) * K;
    int64_t total_bytes = total * 4;
    TUNED_LAUNCH_1D(negate_and_stack_fp8_kernel, "negate_and_stack_fp8",
        total, 16, total_bytes, stream,
        reinterpret_cast<const uint8_t*>(src1),
        reinterpret_cast<const uint8_t*>(src2_negated),
        reinterpret_cast<uint8_t*>(dst), M, K);
    CUDA_CHECK(cudaGetLastError());
}


// ========================================================================================
// Strategy 3A: FP8 Native Input Deinterleave (byte-shuffle, no cast)
// ========================================================================================

__global__ void deinterleave_fp8_kernel(
    const uint8_t* __restrict__ interleaved,
    uint8_t* __restrict__ out_real,
    uint8_t* __restrict__ out_imag,
    int64_t num_complex_elements)
{
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;

    for (int64_t i = idx * 8; i < num_complex_elements; i += stride * 8) {
        if (i + 7 < num_complex_elements) {
            uint4 data = *reinterpret_cast<const uint4*>(interleaved + 2 * i);
            const uint8_t* bytes = reinterpret_cast<const uint8_t*>(&data);

            uint8_t re[8], im[8];
            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                re[j] = bytes[2 * j];
                im[j] = bytes[2 * j + 1];
            }

            *reinterpret_cast<uint2*>(out_real + i) = *reinterpret_cast<const uint2*>(re);
            *reinterpret_cast<uint2*>(out_imag + i) = *reinterpret_cast<const uint2*>(im);
        } else {
            for (int64_t j = i; j < num_complex_elements && j < i + 8; ++j) {
                out_real[j] = interleaved[2 * j];
                out_imag[j] = interleaved[2 * j + 1];
            }
        }
    }
}

inline void deinterleave_fp8(
    const __nv_fp8_e4m3* interleaved,
    __nv_fp8_e4m3* out_real, __nv_fp8_e4m3* out_imag,
    int64_t num_complex_elements,
    cudaStream_t stream = nullptr)
{
    int64_t total_bytes = num_complex_elements * 4;
    TUNED_LAUNCH_1D(deinterleave_fp8_kernel, "deinterleave_fp8",
        num_complex_elements, 8, total_bytes, stream,
        reinterpret_cast<const uint8_t*>(interleaved),
        reinterpret_cast<uint8_t*>(out_real),
        reinterpret_cast<uint8_t*>(out_imag),
        num_complex_elements);
    CUDA_CHECK(cudaGetLastError());
}


/// Fused paired FP16 → 2× FP8 E4M3 kernel with transpose (32×32 SMEM tiling)
__global__ void cast_fp16_to_fp8_transposed_paired_kernel(
    const __half* __restrict__ input1,
    const __half* __restrict__ input2,
    uint8_t* __restrict__ output1,
    uint8_t* __restrict__ output2,
    int rows, int cols,
    int64_t batch_stride_in,
    int64_t batch_stride_out)
{
    __shared__ uint8_t tile1[kTransposeTile][kTransposeTile + 1];
    __shared__ uint8_t tile2[kTransposeTile][kTransposeTile + 1];

    const int bx = blockIdx.x * kTransposeTile;
    const int by = blockIdx.y * kTransposeTile;
    const int batch = blockIdx.z;

    const __half* in1_b  = input1  + batch * batch_stride_in;
    const __half* in2_b  = input2  + batch * batch_stride_in;
    uint8_t*      out1_b = output1 + batch * batch_stride_out;
    uint8_t*      out2_b = output2 + batch * batch_stride_out;

    {
        int row = by + threadIdx.x;
        int col = bx + threadIdx.y;
        if (row < rows && col < cols) {
            int64_t in_idx = row + static_cast<int64_t>(col) * rows;

            float val1 = __half2float(in1_b[in_idx]);
            val1 = fminf(fmaxf(val1, -448.0f), 448.0f);
            __nv_fp8_e4m3 fp8_val1(__float2half_rn(val1));
            tile1[threadIdx.x][threadIdx.y] = reinterpret_cast<const uint8_t&>(fp8_val1);

            float val2 = __half2float(in2_b[in_idx]);
            val2 = fminf(fmaxf(val2, -448.0f), 448.0f);
            __nv_fp8_e4m3 fp8_val2(__float2half_rn(val2));
            tile2[threadIdx.x][threadIdx.y] = reinterpret_cast<const uint8_t&>(fp8_val2);
        }
    }

    __syncthreads();

    {
        int out_row = bx + threadIdx.x;
        int out_col = by + threadIdx.y;
        if (out_row < cols && out_col < rows) {
            int64_t out_idx = out_row + static_cast<int64_t>(out_col) * cols;
            out1_b[out_idx] = tile1[threadIdx.y][threadIdx.x];
            out2_b[out_idx] = tile2[threadIdx.y][threadIdx.x];
        }
    }
}

inline void cast_fp16_to_fp8_e4m3_transposed_paired(
    const __half* input1, const __half* input2,
    cutlass::float_e4m3_t* output1, cutlass::float_e4m3_t* output2,
    int rows, int cols,
    cudaStream_t stream = nullptr,
    int batch_count = 1)
{
    dim3 block(kTransposeTile, kTransposeTile);
    dim3 grid((cols + kTransposeTile - 1) / kTransposeTile,
              (rows + kTransposeTile - 1) / kTransposeTile,
              batch_count);
    int64_t stride = static_cast<int64_t>(rows) * cols;
    cast_fp16_to_fp8_transposed_paired_kernel<<<grid, block, 0, stream>>>(
        input1, input2, reinterpret_cast<uint8_t*>(output1),
        reinterpret_cast<uint8_t*>(output2),
        rows, cols, stride, stride);
    CUDA_CHECK(cudaGetLastError());
}


// ========================================================================================
// Planar FP16 → Interleaved FP8 with Optional Im Negation (Direct GEMM)
// ========================================================================================
//
// Converts separate Re/Im FP16 buffers to interleaved FP8 E4M3 format:
//   Input:  re[num_elements], im[num_elements] (planar FP16)
//   Output: out[2*num_elements] (interleaved FP8: [re0, im0, re1, im1, ...])
//
// When negate_im=true: output odd bytes have sign bit flipped (XOR 0x80),
// producing [re0, -im0, re1, -im1, ...] for the direct GEMM B_neg buffer.
//

template <bool NegateIm>
__global__ void cast_fp16_planar_to_fp8_interleaved_kernel(
    const __half* __restrict__ re_in,
    const __half* __restrict__ im_in,
    uint8_t* __restrict__ out,
    int64_t num_elements)
{
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;

    // Process 4 complex elements (8 FP8 values) per thread for vectorized stores
    for (int64_t i = idx * 4; i < num_elements; i += stride * 4) {
        if (i + 3 < num_elements) {
            // Vectorized: load 4 Re + 4 Im FP16 values
            const uint2* re_vec = reinterpret_cast<const uint2*>(re_in + i);
            const uint2* im_vec = reinterpret_cast<const uint2*>(im_in + i);
            uint2 re_data = *re_vec;
            uint2 im_data = *im_vec;

            const __half* re_h = reinterpret_cast<const __half*>(&re_data);
            const __half* im_h = reinterpret_cast<const __half*>(&im_data);

            uint8_t fp8_out[8];
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                float re_val = __half2float(re_h[j]);
                float im_val = __half2float(im_h[j]);
                re_val = fminf(fmaxf(re_val, -448.0f), 448.0f);
                im_val = fminf(fmaxf(im_val, -448.0f), 448.0f);

                __nv_fp8_e4m3 re_fp8 = __nv_fp8_e4m3(re_val);
                __nv_fp8_e4m3 im_fp8 = __nv_fp8_e4m3(im_val);

                uint8_t re_byte = *reinterpret_cast<uint8_t*>(&re_fp8);
                uint8_t im_byte = *reinterpret_cast<uint8_t*>(&im_fp8);

                if constexpr (NegateIm) {
                    im_byte ^= 0x80u;  // Flip FP8 sign bit
                }

                fp8_out[2 * j]     = re_byte;
                fp8_out[2 * j + 1] = im_byte;
            }

            // 64-bit store (8 FP8 values)
            *reinterpret_cast<uint2*>(out + 2 * i) = *reinterpret_cast<const uint2*>(fp8_out);
        } else {
            // Scalar tail
            for (int64_t j = i; j < num_elements && j < i + 4; ++j) {
                float re_val = __half2float(re_in[j]);
                float im_val = __half2float(im_in[j]);
                re_val = fminf(fmaxf(re_val, -448.0f), 448.0f);
                im_val = fminf(fmaxf(im_val, -448.0f), 448.0f);

                __nv_fp8_e4m3 re_fp8 = __nv_fp8_e4m3(re_val);
                __nv_fp8_e4m3 im_fp8 = __nv_fp8_e4m3(im_val);

                uint8_t re_byte = *reinterpret_cast<uint8_t*>(&re_fp8);
                uint8_t im_byte = *reinterpret_cast<uint8_t*>(&im_fp8);

                if constexpr (NegateIm) {
                    im_byte ^= 0x80u;
                }

                out[2 * j]     = re_byte;
                out[2 * j + 1] = im_byte;
            }
        }
    }
}

/// Cast planar FP16 to interleaved FP8 (no negation)
inline void cast_fp16_planar_to_fp8_interleaved(
    const __half* re_in, const __half* im_in,
    __nv_fp8_e4m3* out,
    int64_t num_elements,
    cudaStream_t stream = nullptr)
{
    // Total bytes: read 2*num FP16 (4*num B) + write 2*num FP8 (2*num B)
    int64_t total_bytes = num_elements * 6;
    TUNED_LAUNCH_1D(cast_fp16_planar_to_fp8_interleaved_kernel<false>,
        "cast_planar_to_interleaved",
        num_elements, 4, total_bytes, stream,
        re_in, im_in, reinterpret_cast<uint8_t*>(out), num_elements);
    CUDA_CHECK(cudaGetLastError());
}

/// Cast planar FP16 to interleaved FP8 with imaginary negation (for B_neg)
inline void cast_fp16_planar_to_fp8_interleaved_negate_im(
    const __half* re_in, const __half* im_in,
    __nv_fp8_e4m3* out,
    int64_t num_elements,
    cudaStream_t stream = nullptr)
{
    int64_t total_bytes = num_elements * 6;
    TUNED_LAUNCH_1D(cast_fp16_planar_to_fp8_interleaved_kernel<true>,
        "cast_planar_to_interleaved_neg",
        num_elements, 4, total_bytes, stream,
        re_in, im_in, reinterpret_cast<uint8_t*>(out), num_elements);
    CUDA_CHECK(cudaGetLastError());
}

// ========================================================================================
// QC INT4 → FP8 E4M3 Interleaved Cast + Transpose
// ========================================================================================
//
// Fuses QC sign-magnitude decode + FP8 E4M3 cast + transpose into a single kernel.
// Eliminates FP16 intermediate planar buffers for the voltage beamformer direct GEMM path.
//
// QC byte format: high nibble = Re (sign-magnitude), low nibble = Im.
// Sign-magnitude: bit 3 = sign (1 = negative), bits 2:0 = magnitude [0..7].
// FP8 E4M3 can exactly represent all integers in [-7, 7].
//
// Input layout:  [n_ch × n_ant × n_time] bytes (one QC byte per complex element)
//                For the VBF: input is per-pol, after pol-split from [n_ch × n_pol × n_ant × n_time]
//
// Output layout: [n_ch × n_time × n_ant × 2] bytes (interleaved FP8: Re, Im pairs)
//                This is the layout expected by the direct GEMM kernel: [batch × M × 2*K]
//

/// @brief QC sign-magnitude nibble → FP8 E4M3 lookup table.
///
/// Maps 4-bit sign-magnitude encoded integers to their FP8 E4M3 (e4m3fn) bit patterns.
/// Sign-magnitude: bit 3 = sign, bits 2:0 = magnitude → value = ±(magnitude).
/// FP8 E4M3: Sign(1) | Exp(4) | Man(3), bias = 7.
///
/// Index:  0     1     2     3     4     5     6     7     8-15 (mirrored with sign)
/// Value:  0.0   1.0   2.0   3.0   4.0   5.0   6.0   7.0  -0...-7
/// FP8:    0x00  0x38  0x40  0x44  0x48  0x4A  0x4C  0x4E  0x80 0xB8 0xC0 0xC4 0xC8 0xCA 0xCC 0xCE
__device__ __constant__ uint8_t qc_to_fp8_lut[16] = {
    0x00, 0x38, 0x40, 0x44, 0x48, 0x4A, 0x4C, 0x4E,  // +0 to +7
    0x80, 0xB8, 0xC0, 0xC4, 0xC8, 0xCA, 0xCC, 0xCE   // -0 to -7
};

/// @brief Decode one QC byte to interleaved FP8 (2 bytes: Re FP8, Im FP8).
__device__ __forceinline__ uint16_t qc_byte_to_fp8_pair(uint8_t qc) {
    uint8_t re_nib = (qc >> 4) & 0x0F;
    uint8_t im_nib = qc & 0x0F;
    uint8_t re_fp8 = qc_to_fp8_lut[re_nib];
    uint8_t im_fp8 = qc_to_fp8_lut[im_nib];
    return static_cast<uint16_t>(re_fp8) | (static_cast<uint16_t>(im_fp8) << 8);
}

/// @brief Kernel: QC INT4 → FP8 E4M3 interleaved with transpose.
///
/// Each thread processes 4 QC bytes → 8 FP8 bytes (4 complex elements).
///
/// @param qc_data   Input: [n_ant × n_time] QC bytes (per batch/pol)
/// @param fp8_out   Output: [n_time × n_ant × 2] FP8 bytes (per batch/pol)
/// @param n_ant     K dimension (antenna)
/// @param n_time    M dimension (time)
/// @param n_ch      Batch count (channels)
__global__ void qc_to_fp8_interleaved_transpose_kernel(
    const uint8_t* __restrict__ qc_data,
    uint8_t* __restrict__ fp8_out,
    int n_ant, int n_time, int n_ch)
{
    const int64_t total = (int64_t)n_ch * n_time * n_ant;
    for (int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
         i < total;
         i += (int64_t)gridDim.x * blockDim.x)
    {
        // Output index: [chan, time, ant] — transposed from input [chan, ant, time]
        const int ant  = (int)(i % n_ant);
        const int time = (int)((i / n_ant) % n_time);
        const int chan = (int)(i / ((int64_t)n_ant * n_time));

        // Input index: [chan, ant, time]
        const int64_t in_idx = (int64_t)chan * n_ant * n_time
                             + (int64_t)ant * n_time + time;
        const uint8_t qc = qc_data[in_idx];

        // Output: [chan, time, ant] → interleaved FP8 [Re, Im]
        // fp8_out[chan * n_time * n_ant * 2 + time * n_ant * 2 + ant * 2]
        const int64_t out_idx = (int64_t)chan * n_time * n_ant * 2
                              + (int64_t)time * n_ant * 2
                              + (int64_t)ant * 2;

        uint16_t fp8_pair = qc_byte_to_fp8_pair(qc);
        *reinterpret_cast<uint16_t*>(fp8_out + out_idx) = fp8_pair;
    }
}

/// @brief Launch QC → FP8 interleaved transpose kernel for both polarisations.
///
/// Handles the full VBF QC decode for one polarisation:
/// Input:  [n_ch × n_ant × n_time] QC bytes (after pol extraction)
/// Output: [n_ch × n_time × 2*n_ant] FP8 bytes (interleaved, ready for direct GEMM)
///
/// @param qc_pol    QC data for one polarisation: [n_ch × n_ant × n_time]
/// @param fp8_out   Output buffer: [n_ch × n_time × 2*n_ant] FP8 interleaved
/// @param n_ant     Antenna count (K dimension)
/// @param n_time    Time sample count (M dimension)
/// @param n_ch      Channel count (batch dimension)
/// @param stream    CUDA stream
inline void qc_to_fp8_interleaved_transpose(
    const uint8_t* qc_pol,
    uint8_t* fp8_out,
    int n_ant, int n_time, int n_ch,
    cudaStream_t stream = nullptr)
{
    const int64_t total = (int64_t)n_ch * n_time * n_ant;
    const int64_t total_bytes = total * 3;  // read 1 QC byte + write 2 FP8 bytes per element
    TUNED_LAUNCH_1D(qc_to_fp8_interleaved_transpose_kernel,
        "qc_to_fp8_interleaved",
        total, 1, total_bytes, stream,
        qc_pol, fp8_out, n_ant, n_time, n_ch);
    CUDA_CHECK(cudaGetLastError());
}

/// @brief Kernel: QC INT4 → FP8 interleaved with transpose + pol-split (single launch).
///
/// Processes both polarisations from the full QC input in one kernel launch.
///
/// Input:  [n_ch × 2 × n_ant × n_time] QC bytes
/// Output: fp8_pol0, fp8_pol1: each [n_ch × n_time × 2*n_ant] FP8 interleaved
__global__ void qc_to_fp8_interleaved_polsplit_kernel(
    const uint8_t* __restrict__ qc_data,
    uint8_t* __restrict__ fp8_pol0,
    uint8_t* __restrict__ fp8_pol1,
    int n_ant, int n_time, int n_ch)
{
    const int64_t total = (int64_t)n_ch * n_time * n_ant;
    for (int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
         i < total;
         i += (int64_t)gridDim.x * blockDim.x)
    {
        // Output index decomposition: [chan, time, ant]
        const int ant  = (int)(i % n_ant);
        const int time = (int)((i / n_ant) % n_time);
        const int chan = (int)(i / ((int64_t)n_ant * n_time));

        // QC input: [chan, pol, ant, time]
        const int64_t ch_offset = (int64_t)chan * 2 * n_ant * n_time;
        const int64_t elem_offset = (int64_t)ant * n_time + time;
        const int64_t pol_stride = (int64_t)n_ant * n_time;

        const uint8_t qc_0 = qc_data[ch_offset + elem_offset];
        const uint8_t qc_1 = qc_data[ch_offset + pol_stride + elem_offset];

        // FP8 interleaved output: [chan, time, ant*2] (Re, Im pairs)
        const int64_t out_idx = (int64_t)chan * n_time * n_ant * 2
                              + (int64_t)time * n_ant * 2
                              + (int64_t)ant * 2;

        uint16_t fp8_pair_0 = qc_byte_to_fp8_pair(qc_0);
        uint16_t fp8_pair_1 = qc_byte_to_fp8_pair(qc_1);

        *reinterpret_cast<uint16_t*>(fp8_pol0 + out_idx) = fp8_pair_0;
        *reinterpret_cast<uint16_t*>(fp8_pol1 + out_idx) = fp8_pair_1;
    }
}

/// @brief Launch full QC → FP8 interleaved + transpose + pol-split.
///
/// Single-launch kernel that decodes QC INT4 to FP8 E4M3 interleaved format
/// for both polarisations, with transpose from [ant × time] to [time × ant].
/// Output is directly consumable by the direct GEMM kernel.
///
/// @param qc_data   Full QC input: [n_ch × 2 × n_ant × n_time] bytes
/// @param fp8_pol0  FP8 output for pol 0: [n_ch × n_time × 2*n_ant] bytes
/// @param fp8_pol1  FP8 output for pol 1: [n_ch × n_time × 2*n_ant] bytes
/// @param n_ant     Antenna count (K dimension)
/// @param n_time    Time sample count (M dimension)
/// @param n_ch      Channel count (batch dimension)
/// @param stream    CUDA stream
inline void qc_to_fp8_interleaved_polsplit(
    const uint8_t* qc_data,
    uint8_t* fp8_pol0, uint8_t* fp8_pol1,
    int n_ant, int n_time, int n_ch,
    cudaStream_t stream = nullptr)
{
    const int64_t total = (int64_t)n_ch * n_time * n_ant;
    // Read: 2 QC bytes (both pols), write: 4 FP8 bytes (2 per pol) = 6 bytes per element
    const int64_t total_bytes = total * 6;
    TUNED_LAUNCH_1D(qc_to_fp8_interleaved_polsplit_kernel,
        "qc_to_fp8_polsplit",
        total, 1, total_bytes, stream,
        qc_data, fp8_pol0, fp8_pol1, n_ant, n_time, n_ch);
    CUDA_CHECK(cudaGetLastError());
}
