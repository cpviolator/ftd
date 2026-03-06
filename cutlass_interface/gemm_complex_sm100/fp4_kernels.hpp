#ifdef COMPLEX_SM100_ENABLE_FP4

// ----- FP4 E2M1 conversion helpers -----
// E2M1: sign(1) + exponent(2) + mantissa(1) = 4 bits
// Bias = 1, range = ±6 (2^(3-1) * (1 + 0.5) = 4 * 1.5 = 6)

__device__ __forceinline__ uint8_t fp16_to_fp4_e2m1(float val) {
    val = fminf(fmaxf(val, -6.0f), 6.0f);
    uint8_t sign = (val < 0.0f) ? 1 : 0;
    float abs_val = fabsf(val);

    if (abs_val < 0.25f) {
        return sign << 3;
    }

    int exp_unbiased = 0;
    float frac = frexpf(abs_val, &exp_unbiased);
    frac *= 2.0f;
    exp_unbiased -= 1;

    // Bias = 1, valid biased exponents: 1..3
    int exp_biased = exp_unbiased + 1;
    exp_biased = max(1, min(3, exp_biased));

    // 1-bit mantissa
    int mantissa = (frac - 1.0f >= 0.5f) ? 1 : 0;

    return (sign << 3) | ((exp_biased & 0x3) << 1) | (mantissa & 0x1);
}


/// FP16 → FP4 E2M1 cast kernel with nibble packing (linear layout)
/// Packs 2 values per byte: [v0(4) | v1(4)], v0 in high nibble
__global__ void cast_fp16_to_fp4_e2m1_kernel_sm100(
    const __half* __restrict__ input,
    uint8_t* __restrict__ output,
    int64_t num_elements)
{
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;

    // Process 8 elements per iteration → 4 output bytes (vectorizable)
    for (int64_t i = idx * 8; i < num_elements; i += stride * 8) {
        if (i + 7 < num_elements) {
            // Vectorized: load 8 × FP16 = 128 bits
            const float4* in_vec = reinterpret_cast<const float4*>(input + i);
            float4 data = *in_vec;
            const __half* halfs = reinterpret_cast<const __half*>(&data);

            uint8_t packed[4];
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                uint8_t hi = fp16_to_fp4_e2m1(__half2float(halfs[2 * j]));
                uint8_t lo = fp16_to_fp4_e2m1(__half2float(halfs[2 * j + 1]));
                packed[j] = hi | (lo << 4);
            }

            // Store 4 packed bytes = 32 bits
            *reinterpret_cast<uint32_t*>(output + i / 2) =
                *reinterpret_cast<const uint32_t*>(packed);
        } else {
            // Scalar tail: pack pairs
            for (int64_t j = i; j < num_elements; j += 2) {
                uint8_t hi = fp16_to_fp4_e2m1(__half2float(input[j]));
                uint8_t lo = (j + 1 < num_elements)
                    ? fp16_to_fp4_e2m1(__half2float(input[j + 1]))
                    : 0;
                output[j / 2] = hi | (lo << 4);
            }
        }
    }
}

/// Launch helper: FP16 → FP4 E2M1 cast (linear layout)
inline void cast_fp16_to_fp4_e2m1_sm100(
    const __half* input,
    void* output,
    int64_t num_elements,
    cudaStream_t stream = nullptr)
{
    int64_t total_bytes = num_elements * 3;  // read FP16 (2B) + write FP4 (0.5B) ≈ 3B
    TUNED_LAUNCH_1D(cast_fp16_to_fp4_e2m1_kernel_sm100, "cast_fp16_to_fp4_e2m1",
        num_elements, 8, total_bytes, stream,
        input, reinterpret_cast<uint8_t*>(output), num_elements);
    CUDA_CHECK(cudaGetLastError());
}


/// FP16 → FP4 E2M1 cast kernel with transpose (32×32 SMEM tiling, nibble-packed output)
__global__ void cast_fp16_to_fp4_e2m1_transposed_kernel_sm100(
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
            tile[threadIdx.x][threadIdx.y] = fp16_to_fp4_e2m1(val);
        }
    }

    __syncthreads();

    {
        int out_row = bx + threadIdx.x;
        int out_col = by + threadIdx.y;
        if (out_row < cols && out_col < rows) {
            int64_t linear_idx = out_row + static_cast<int64_t>(out_col) * cols;
            uint8_t v = tile[threadIdx.y][threadIdx.x];

            // CuTe LSB-first: even → low nibble, odd → high nibble
            int64_t byte_idx = linear_idx / 2;
            if (linear_idx % 2 == 0) {
                atomicOr(reinterpret_cast<unsigned int*>(out_b + (byte_idx & ~3)),
                         static_cast<unsigned int>(v & 0xF) << ((byte_idx & 3) * 8));
            } else {
                atomicOr(reinterpret_cast<unsigned int*>(out_b + (byte_idx & ~3)),
                         static_cast<unsigned int>(v << 4) << ((byte_idx & 3) * 8));
            }
        }
    }
}

/// Launch helper: FP16 → FP4 E2M1 transposed cast
inline void cast_fp16_to_fp4_e2m1_transposed_sm100(
    const __half* input,
    void* output,
    int rows, int cols,
    cudaStream_t stream = nullptr,
    int batch_count = 1)
{
    dim3 block(kTransposeTile, kTransposeTile);
    dim3 grid((cols + kTransposeTile - 1) / kTransposeTile,
              (rows + kTransposeTile - 1) / kTransposeTile,
              batch_count);
    int64_t stride_in = static_cast<int64_t>(rows) * cols;
    int64_t stride_out = (static_cast<int64_t>(cols) * rows + 1) / 2;  // nibble-packed

    CUDA_CHECK(cudaMemsetAsync(output, 0, stride_out * batch_count, stream));

    cast_fp16_to_fp4_e2m1_transposed_kernel_sm100<<<grid, block, 0, stream>>>(
        input, reinterpret_cast<uint8_t*>(output), rows, cols, stride_in, stride_out);
    CUDA_CHECK(cudaGetLastError());
}


// ========================================================================================
// F1: Fused Deinterleave + Cast for FP4 (interleaved FP16 → 2 × planar FP4)
// ========================================================================================

/// F1 linear: interleaved complex FP16 → 2 planar FP4 E2M1 buffers (nibble-packed)
__global__ void deinterleave_cast_fp16_to_fp4_e2m1_kernel_sm100(
    const __half* __restrict__ interleaved,
    uint8_t* __restrict__ out_real,
    uint8_t* __restrict__ out_imag,
    int64_t num_complex_elements)
{
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;

    // Process 4 complex elements → 4 Re + 4 Im → 2 bytes each (nibble-packed)
    for (int64_t i = idx * 4; i < num_complex_elements; i += stride * 4) {
        if (i + 3 < num_complex_elements) {
            const float4* in_vec = reinterpret_cast<const float4*>(interleaved + 2 * i);
            float4 data = *in_vec;
            const __half* halfs = reinterpret_cast<const __half*>(&data);

            uint8_t re_packed[2], im_packed[2];
            #pragma unroll
            for (int j = 0; j < 2; ++j) {
                uint8_t re_hi = fp16_to_fp4_e2m1(__half2float(halfs[4 * j]));
                uint8_t re_lo = fp16_to_fp4_e2m1(__half2float(halfs[4 * j + 2]));
                uint8_t im_hi = fp16_to_fp4_e2m1(__half2float(halfs[4 * j + 1]));
                uint8_t im_lo = fp16_to_fp4_e2m1(__half2float(halfs[4 * j + 3]));
                re_packed[j] = re_hi | (re_lo << 4);
                im_packed[j] = im_hi | (im_lo << 4);
            }

            int64_t out_byte = i / 2;
            *reinterpret_cast<uint16_t*>(out_real + out_byte) =
                *reinterpret_cast<const uint16_t*>(re_packed);
            *reinterpret_cast<uint16_t*>(out_imag + out_byte) =
                *reinterpret_cast<const uint16_t*>(im_packed);
        } else {
            for (int64_t j = i; j < num_complex_elements; j += 2) {
                uint8_t re_hi = fp16_to_fp4_e2m1(__half2float(interleaved[2 * j]));
                uint8_t im_hi = fp16_to_fp4_e2m1(__half2float(interleaved[2 * j + 1]));
                uint8_t re_lo = 0, im_lo = 0;
                if (j + 1 < num_complex_elements) {
                    re_lo = fp16_to_fp4_e2m1(__half2float(interleaved[2 * (j + 1)]));
                    im_lo = fp16_to_fp4_e2m1(__half2float(interleaved[2 * (j + 1) + 1]));
                }
                out_real[j / 2] = re_hi | (re_lo << 4);
                out_imag[j / 2] = im_hi | (im_lo << 4);
            }
        }
    }
}

inline void deinterleave_cast_fp16_to_fp4_e2m1_sm100(
    const __half* interleaved,
    void* out_real, void* out_imag,
    int64_t num_complex_elements,
    cudaStream_t stream = nullptr)
{
    // read interleaved FP16 (4B/complex) + write 2× FP4 (2×0.5B/complex) = 5B/complex
    int64_t total_bytes = num_complex_elements * 5;
    TUNED_LAUNCH_1D(deinterleave_cast_fp16_to_fp4_e2m1_kernel_sm100, "deinterleave_cast_fp16_to_fp4_e2m1",
        num_complex_elements, 4, total_bytes, stream,
        interleaved, reinterpret_cast<uint8_t*>(out_real),
        reinterpret_cast<uint8_t*>(out_imag), num_complex_elements);
    CUDA_CHECK(cudaGetLastError());
}

/// F1 transposed: interleaved complex FP16 ColMajor → 2 transposed planar FP4 E2M1
__global__ void deinterleave_cast_fp16_to_fp4_e2m1_transposed_kernel_sm100(
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
            tile_re[threadIdx.x][threadIdx.y] = fp16_to_fp4_e2m1(re_val);
            tile_im[threadIdx.x][threadIdx.y] = fp16_to_fp4_e2m1(im_val);
        }
    }

    __syncthreads();

    {
        int out_row = bx + threadIdx.x;
        int out_col = by + threadIdx.y;
        if (out_row < cols && out_col < rows) {
            int64_t linear_idx = out_row + static_cast<int64_t>(out_col) * cols;
            uint8_t v_re = tile_re[threadIdx.y][threadIdx.x];
            uint8_t v_im = tile_im[threadIdx.y][threadIdx.x];

            // CuTe LSB-first: even → low nibble, odd → high nibble
            int64_t byte_idx = linear_idx / 2;
            if (linear_idx % 2 == 0) {
                atomicOr(reinterpret_cast<unsigned int*>(re_b + (byte_idx & ~3)),
                         static_cast<unsigned int>(v_re & 0xF) << ((byte_idx & 3) * 8));
                atomicOr(reinterpret_cast<unsigned int*>(im_b + (byte_idx & ~3)),
                         static_cast<unsigned int>(v_im & 0xF) << ((byte_idx & 3) * 8));
            } else {
                atomicOr(reinterpret_cast<unsigned int*>(re_b + (byte_idx & ~3)),
                         static_cast<unsigned int>(v_re << 4) << ((byte_idx & 3) * 8));
                atomicOr(reinterpret_cast<unsigned int*>(im_b + (byte_idx & ~3)),
                         static_cast<unsigned int>(v_im << 4) << ((byte_idx & 3) * 8));
            }
        }
    }
}

inline void deinterleave_cast_fp16_to_fp4_e2m1_transposed_sm100(
    const __half* input,
    void* out_real, void* out_imag,
    int rows, int cols,
    cudaStream_t stream = nullptr,
    int batch_count = 1)
{
    dim3 block(kTransposeTile, kTransposeTile);
    dim3 grid((cols + kTransposeTile - 1) / kTransposeTile,
              (rows + kTransposeTile - 1) / kTransposeTile,
              batch_count);
    int64_t stride_in = static_cast<int64_t>(rows) * cols;
    int64_t stride_out = (static_cast<int64_t>(cols) * rows + 1) / 2;

    CUDA_CHECK(cudaMemsetAsync(out_real, 0, stride_out * batch_count, stream));
    CUDA_CHECK(cudaMemsetAsync(out_imag, 0, stride_out * batch_count, stream));

    deinterleave_cast_fp16_to_fp4_e2m1_transposed_kernel_sm100<<<grid, block, 0, stream>>>(
        input, reinterpret_cast<uint8_t*>(out_real),
        reinterpret_cast<uint8_t*>(out_imag),
        rows, cols, stride_in, stride_out);
    CUDA_CHECK(cudaGetLastError());
}


// ========================================================================================
// F2: Fused Cast + Duplicate for FP4 (planar FP16 → 2× planar FP4)
// ========================================================================================

/// F2 linear: FP16 → 2× FP4 E2M1 dual-output (nibble-packed)
__global__ void cast_fp16_to_fp4_e2m1_dual_kernel_sm100(
    const __half* __restrict__ input,
    uint8_t* __restrict__ output1,
    uint8_t* __restrict__ output2,
    int64_t num_elements)
{
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;

    for (int64_t i = idx * 8; i < num_elements; i += stride * 8) {
        if (i + 7 < num_elements) {
            const float4* in_vec = reinterpret_cast<const float4*>(input + i);
            float4 data = *in_vec;
            const __half* halfs = reinterpret_cast<const __half*>(&data);

            uint8_t packed[4];
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                uint8_t hi = fp16_to_fp4_e2m1(__half2float(halfs[2 * j]));
                uint8_t lo = fp16_to_fp4_e2m1(__half2float(halfs[2 * j + 1]));
                packed[j] = hi | (lo << 4);
            }

            uint32_t packed32 = *reinterpret_cast<const uint32_t*>(packed);
            *reinterpret_cast<uint32_t*>(output1 + i / 2) = packed32;
            *reinterpret_cast<uint32_t*>(output2 + i / 2) = packed32;
        } else {
            for (int64_t j = i; j < num_elements; j += 2) {
                uint8_t hi = fp16_to_fp4_e2m1(__half2float(input[j]));
                uint8_t lo = (j + 1 < num_elements)
                    ? fp16_to_fp4_e2m1(__half2float(input[j + 1]))
                    : 0;
                uint8_t packed = hi | (lo << 4);
                output1[j / 2] = packed;
                output2[j / 2] = packed;
            }
        }
    }
}

inline void cast_fp16_to_fp4_e2m1_dual_sm100(
    const __half* input,
    void* output1, void* output2,
    int64_t num_elements,
    cudaStream_t stream = nullptr)
{
    // read FP16 (2B) + write 2× FP4 (2×0.5B) = 3B/element
    int64_t total_bytes = num_elements * 3;
    TUNED_LAUNCH_1D(cast_fp16_to_fp4_e2m1_dual_kernel_sm100, "cast_fp16_to_fp4_e2m1_dual",
        num_elements, 8, total_bytes, stream,
        input, reinterpret_cast<uint8_t*>(output1),
        reinterpret_cast<uint8_t*>(output2), num_elements);
    CUDA_CHECK(cudaGetLastError());
}

/// F2 transposed: FP16 ColMajor → 2× transposed FP4 E2M1 (32×32 SMEM tiling)
__global__ void cast_fp16_to_fp4_e2m1_transposed_dual_kernel_sm100(
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

    const __half* in_b  = input   + batch * batch_stride_in;
    uint8_t*      o1_b  = output1 + batch * batch_stride_out;
    uint8_t*      o2_b  = output2 + batch * batch_stride_out;

    {
        int row = by + threadIdx.x;
        int col = bx + threadIdx.y;
        if (row < rows && col < cols) {
            float val = __half2float(in_b[row + static_cast<int64_t>(col) * rows]);
            tile[threadIdx.x][threadIdx.y] = fp16_to_fp4_e2m1(val);
        }
    }

    __syncthreads();

    {
        int out_row = bx + threadIdx.x;
        int out_col = by + threadIdx.y;
        if (out_row < cols && out_col < rows) {
            int64_t linear_idx = out_row + static_cast<int64_t>(out_col) * cols;
            uint8_t v = tile[threadIdx.y][threadIdx.x];

            // CuTe LSB-first: even → low nibble, odd → high nibble
            int64_t byte_idx = linear_idx / 2;
            if (linear_idx % 2 == 0) {
                unsigned int contrib = static_cast<unsigned int>(v & 0xF) << ((byte_idx & 3) * 8);
                atomicOr(reinterpret_cast<unsigned int*>(o1_b + (byte_idx & ~3)), contrib);
                atomicOr(reinterpret_cast<unsigned int*>(o2_b + (byte_idx & ~3)), contrib);
            } else {
                unsigned int contrib = static_cast<unsigned int>(v << 4) << ((byte_idx & 3) * 8);
                atomicOr(reinterpret_cast<unsigned int*>(o1_b + (byte_idx & ~3)), contrib);
                atomicOr(reinterpret_cast<unsigned int*>(o2_b + (byte_idx & ~3)), contrib);
            }
        }
    }
}

inline void cast_fp16_to_fp4_e2m1_transposed_dual_sm100(
    const __half* input,
    void* output1, void* output2,
    int rows, int cols,
    cudaStream_t stream = nullptr,
    int batch_count = 1)
{
    dim3 block(kTransposeTile, kTransposeTile);
    dim3 grid((cols + kTransposeTile - 1) / kTransposeTile,
              (rows + kTransposeTile - 1) / kTransposeTile,
              batch_count);
    int64_t stride_in = static_cast<int64_t>(rows) * cols;
    int64_t stride_out = (static_cast<int64_t>(cols) * rows + 1) / 2;

    CUDA_CHECK(cudaMemsetAsync(output1, 0, stride_out * batch_count, stream));
    CUDA_CHECK(cudaMemsetAsync(output2, 0, stride_out * batch_count, stream));

    cast_fp16_to_fp4_e2m1_transposed_dual_kernel_sm100<<<grid, block, 0, stream>>>(
        input, reinterpret_cast<uint8_t*>(output1),
        reinterpret_cast<uint8_t*>(output2),
        rows, cols, stride_in, stride_out);
    CUDA_CHECK(cudaGetLastError());
}


// ========================================================================================
// F5: Fused Paired Cast for FP4 (2 independent inputs → 2 independent FP4 outputs)
// ========================================================================================

/// F5 linear: 2× FP16 → 2× FP4 E2M1 paired cast (nibble-packed)
__global__ void cast_fp16_to_fp4_e2m1_paired_kernel_sm100(
    const __half* __restrict__ input1,
    const __half* __restrict__ input2,
    uint8_t* __restrict__ output1,
    uint8_t* __restrict__ output2,
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

            uint8_t packed1[4], packed2[4];
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                // CuTe LSB-first: even element in low nibble, odd in high nibble
                uint8_t even1 = fp16_to_fp4_e2m1(__half2float(halfs1[2 * j]));
                uint8_t odd1  = fp16_to_fp4_e2m1(__half2float(halfs1[2 * j + 1]));
                packed1[j] = even1 | (odd1 << 4);
                uint8_t even2 = fp16_to_fp4_e2m1(__half2float(halfs2[2 * j]));
                uint8_t odd2  = fp16_to_fp4_e2m1(__half2float(halfs2[2 * j + 1]));
                packed2[j] = even2 | (odd2 << 4);
            }

            *reinterpret_cast<uint32_t*>(output1 + i / 2) =
                *reinterpret_cast<const uint32_t*>(packed1);
            *reinterpret_cast<uint32_t*>(output2 + i / 2) =
                *reinterpret_cast<const uint32_t*>(packed2);
        } else {
            // CuTe LSB-first: even element in low nibble, odd in high nibble
            for (int64_t j = i; j < num_elements; j += 2) {
                uint8_t even1 = fp16_to_fp4_e2m1(__half2float(input1[j]));
                uint8_t odd1 = (j + 1 < num_elements)
                    ? fp16_to_fp4_e2m1(__half2float(input1[j + 1])) : 0;
                output1[j / 2] = even1 | (odd1 << 4);

                uint8_t even2 = fp16_to_fp4_e2m1(__half2float(input2[j]));
                uint8_t odd2 = (j + 1 < num_elements)
                    ? fp16_to_fp4_e2m1(__half2float(input2[j + 1])) : 0;
                output2[j / 2] = even2 | (odd2 << 4);
            }
        }
    }
}

inline void cast_fp16_to_fp4_e2m1_paired_sm100(
    const __half* input1, const __half* input2,
    void* output1, void* output2,
    int64_t num_elements,
    cudaStream_t stream = nullptr)
{
    // read 2× FP16 (2×2B) + write 2× FP4 (2×0.5B) = 5B/element
    int64_t total_bytes = num_elements * 5;
    TUNED_LAUNCH_1D(cast_fp16_to_fp4_e2m1_paired_kernel_sm100, "cast_fp16_to_fp4_e2m1_paired",
        num_elements, 8, total_bytes, stream,
        input1, input2, reinterpret_cast<uint8_t*>(output1),
        reinterpret_cast<uint8_t*>(output2), num_elements);
    CUDA_CHECK(cudaGetLastError());
}

/// F5 transposed: 2× FP16 → 2× transposed FP4 E2M1 paired cast (32×32 SMEM)
__global__ void cast_fp16_to_fp4_e2m1_transposed_paired_kernel_sm100(
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

    const __half* in1_b = input1  + batch * batch_stride_in;
    const __half* in2_b = input2  + batch * batch_stride_in;
    uint8_t*      o1_b  = output1 + batch * batch_stride_out;
    uint8_t*      o2_b  = output2 + batch * batch_stride_out;

    {
        int row = by + threadIdx.x;
        int col = bx + threadIdx.y;
        if (row < rows && col < cols) {
            int64_t idx = row + static_cast<int64_t>(col) * rows;
            tile1[threadIdx.x][threadIdx.y] = fp16_to_fp4_e2m1(__half2float(in1_b[idx]));
            tile2[threadIdx.x][threadIdx.y] = fp16_to_fp4_e2m1(__half2float(in2_b[idx]));
        }
    }

    __syncthreads();

    {
        int out_row = bx + threadIdx.x;
        int out_col = by + threadIdx.y;
        if (out_row < cols && out_col < rows) {
            int64_t linear_idx = out_row + static_cast<int64_t>(out_col) * cols;
            uint8_t v1 = tile1[threadIdx.y][threadIdx.x];
            uint8_t v2 = tile2[threadIdx.y][threadIdx.x];

            // CuTe LSB-first: even elements in low nibble, odd in high nibble
            int64_t byte_idx = linear_idx / 2;
            if (linear_idx % 2 == 0) {
                // Even element → low nibble
                atomicOr(reinterpret_cast<unsigned int*>(o1_b + (byte_idx & ~3)),
                         static_cast<unsigned int>(v1 & 0xF) << ((byte_idx & 3) * 8));
                atomicOr(reinterpret_cast<unsigned int*>(o2_b + (byte_idx & ~3)),
                         static_cast<unsigned int>(v2 & 0xF) << ((byte_idx & 3) * 8));
            } else {
                // Odd element → high nibble
                atomicOr(reinterpret_cast<unsigned int*>(o1_b + (byte_idx & ~3)),
                         static_cast<unsigned int>(v1 << 4) << ((byte_idx & 3) * 8));
                atomicOr(reinterpret_cast<unsigned int*>(o2_b + (byte_idx & ~3)),
                         static_cast<unsigned int>(v2 << 4) << ((byte_idx & 3) * 8));
            }
        }
    }
}

inline void cast_fp16_to_fp4_e2m1_transposed_paired_sm100(
    const __half* input1, const __half* input2,
    void* output1, void* output2,
    int rows, int cols,
    cudaStream_t stream = nullptr,
    int batch_count = 1)
{
    dim3 block(kTransposeTile, kTransposeTile);
    dim3 grid((cols + kTransposeTile - 1) / kTransposeTile,
              (rows + kTransposeTile - 1) / kTransposeTile,
              batch_count);
    int64_t stride_in = static_cast<int64_t>(rows) * cols;
    int64_t stride_out = (static_cast<int64_t>(cols) * rows + 1) / 2;

    CUDA_CHECK(cudaMemsetAsync(output1, 0, stride_out * batch_count, stream));
    CUDA_CHECK(cudaMemsetAsync(output2, 0, stride_out * batch_count, stream));

    cast_fp16_to_fp4_e2m1_transposed_paired_kernel_sm100<<<grid, block, 0, stream>>>(
        input1, input2, reinterpret_cast<uint8_t*>(output1),
        reinterpret_cast<uint8_t*>(output2),
        rows, cols, stride_in, stride_out);
    CUDA_CHECK(cudaGetLastError());
}


#endif  // COMPLEX_SM100_ENABLE_FP4
