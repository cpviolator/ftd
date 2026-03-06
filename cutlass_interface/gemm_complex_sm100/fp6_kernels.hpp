// ========================================================================================
// FP6/FP4 Cast Kernels — Sub-Byte Precision (compile-time gated)
// ========================================================================================
//
// These kernels convert FP16 to narrow floating-point formats with sub-byte packing.
// Each format has different range and precision trade-offs:
//
//   FP6 E3M2: 3-bit exponent, 2-bit mantissa. Range ±28,   precision ~0.5
//   FP6 E2M3: 2-bit exponent, 3-bit mantissa. Range ±7.5,  precision ~0.125
//   FP4 E2M1: 2-bit exponent, 1-bit mantissa. Range ±6,    precision ~1.0
//
// Packing formats:
//   FP6: 4 values → 3 bytes (24 bits). Packed LSB-first: byte[0]=v0|(v1<<6), byte[1]=(v1>>2)|(v2<<4), byte[2]=(v2>>4)|(v3<<2)
//   FP4: 2 values → 1 byte (8 bits).  Packed LSB-first: even in low nibble, odd in high nibble
//
// All kernels read FP16, clamp to representable range, quantize, and pack.
// Output buffers must be pre-allocated with the correct sub-byte size:
//   FP6: ceil(num_elements * 6 / 8) bytes = ceil(num_elements * 3 / 4) bytes
//   FP4: ceil(num_elements / 2) bytes

#ifdef COMPLEX_SM100_ENABLE_FP6

// ----- FP6 E3M2 conversion helpers -----
// E3M2: sign(1) + exponent(3) + mantissa(2) = 6 bits
// Bias = 3, range = ±28 (2^(7-3) * (1 + 0.75) = 16 * 1.75 = 28)

__device__ __forceinline__ uint8_t fp16_to_fp6_e3m2(float val) {
    val = fminf(fmaxf(val, -28.0f), 28.0f);
    uint8_t sign = (val < 0.0f) ? 1 : 0;
    float abs_val = fabsf(val);

    if (abs_val < 0.25f) {
        // Subnormal or zero: return signed zero
        return sign << 5;
    }

    // Normal: find exponent and mantissa
    int exp_unbiased = 0;
    float frac = frexpf(abs_val, &exp_unbiased);
    // frexp returns [0.5, 1.0), we need [1.0, 2.0)
    frac *= 2.0f;
    exp_unbiased -= 1;

    // Bias = 3, valid biased exponents: 1..7 (0 reserved for subnormals)
    int exp_biased = exp_unbiased + 3;
    exp_biased = max(1, min(7, exp_biased));

    // 2-bit mantissa (implicit leading 1): encode fractional part
    float mantissa_frac = frac - 1.0f;  // [0, 1)
    int mantissa = __float2int_rn(mantissa_frac * 4.0f);  // 2 bits → 0..3
    mantissa = min(3, max(0, mantissa));

    return (sign << 5) | ((exp_biased & 0x7) << 2) | (mantissa & 0x3);
}

// ----- FP6 E2M3 conversion helpers -----
// E2M3: sign(1) + exponent(2) + mantissa(3) = 6 bits
// Bias = 1, range = ±7.5 (2^(3-1) * (1 + 0.875) = 4 * 1.875 = 7.5)

__device__ __forceinline__ uint8_t fp16_to_fp6_e2m3(float val) {
    val = fminf(fmaxf(val, -7.5f), 7.5f);
    uint8_t sign = (val < 0.0f) ? 1 : 0;
    float abs_val = fabsf(val);

    if (abs_val < 0.0625f) {
        return sign << 5;
    }

    int exp_unbiased = 0;
    float frac = frexpf(abs_val, &exp_unbiased);
    frac *= 2.0f;
    exp_unbiased -= 1;

    // Bias = 1, valid biased exponents: 1..3 (0 reserved for subnormals)
    int exp_biased = exp_unbiased + 1;
    exp_biased = max(1, min(3, exp_biased));

    // 3-bit mantissa
    float mantissa_frac = frac - 1.0f;
    int mantissa = __float2int_rn(mantissa_frac * 8.0f);  // 3 bits → 0..7
    mantissa = min(7, max(0, mantissa));

    return (sign << 5) | ((exp_biased & 0x3) << 3) | (mantissa & 0x7);
}


/// FP16 → FP6 E3M2 cast kernel with 6-bit packing (linear layout)
/// Processes 4 elements per pack group: 4 × 6 bits = 24 bits = 3 bytes
__global__ void cast_fp16_to_fp6_e3m2_kernel_sm100(
    const __half* __restrict__ input,
    uint8_t* __restrict__ output,
    int64_t num_elements)
{
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;

    // Process 4 elements per iteration → 3 output bytes
    for (int64_t i = idx * 4; i < num_elements; i += stride * 4) {
        uint8_t v[4];
        int count = min(static_cast<int64_t>(4), num_elements - i);
        for (int j = 0; j < count; ++j) {
            v[j] = fp16_to_fp6_e3m2(__half2float(input[i + j]));
        }
        for (int j = count; j < 4; ++j) v[j] = 0;

        // Pack 4 × 6-bit values into 3 bytes (MSB-first):
        //   byte0 = v0[5:0] | v1[5:4]
        //   byte1 = v1[3:0] | v2[5:2]
        //   byte2 = v2[1:0] | v3[5:0]
        int64_t out_base = (i / 4) * 3;
        if (out_base + 2 < (num_elements * 3 + 3) / 4) {
            output[out_base + 0] = v[0] | (v[1] << 6);
            output[out_base + 1] = (v[1] >> 2) | (v[2] << 4);
            output[out_base + 2] = (v[2] >> 4) | (v[3] << 2);
        }
    }
}

/// Launch helper: FP16 → FP6 E3M2 cast (linear layout)
inline void cast_fp16_to_fp6_e3m2_sm100(
    const __half* input,
    void* output,
    int64_t num_elements,
    cudaStream_t stream = nullptr)
{
    int64_t total_bytes = num_elements * 3;  // read FP16 (2B) + write FP6 (0.75B) ≈ 3B
    TUNED_LAUNCH_1D(cast_fp16_to_fp6_e3m2_kernel_sm100, "cast_fp16_to_fp6_e3m2",
        num_elements, 4, total_bytes, stream,
        input, reinterpret_cast<uint8_t*>(output), num_elements);
    CUDA_CHECK(cudaGetLastError());
}


/// FP16 → FP6 E2M3 cast kernel with 6-bit packing (linear layout)
__global__ void cast_fp16_to_fp6_e2m3_kernel_sm100(
    const __half* __restrict__ input,
    uint8_t* __restrict__ output,
    int64_t num_elements)
{
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;

    for (int64_t i = idx * 4; i < num_elements; i += stride * 4) {
        uint8_t v[4];
        int count = min(static_cast<int64_t>(4), num_elements - i);
        for (int j = 0; j < count; ++j) {
            v[j] = fp16_to_fp6_e2m3(__half2float(input[i + j]));
        }
        for (int j = count; j < 4; ++j) v[j] = 0;

        int64_t out_base = (i / 4) * 3;
        if (out_base + 2 < (num_elements * 3 + 3) / 4) {
            output[out_base + 0] = v[0] | (v[1] << 6);
            output[out_base + 1] = (v[1] >> 2) | (v[2] << 4);
            output[out_base + 2] = (v[2] >> 4) | (v[3] << 2);
        }
    }
}

/// Launch helper: FP16 → FP6 E2M3 cast (linear layout)
inline void cast_fp16_to_fp6_e2m3_sm100(
    const __half* input,
    void* output,
    int64_t num_elements,
    cudaStream_t stream = nullptr)
{
    int64_t total_bytes = num_elements * 3;  // read FP16 (2B) + write FP6 (0.75B) ≈ 3B
    TUNED_LAUNCH_1D(cast_fp16_to_fp6_e2m3_kernel_sm100, "cast_fp16_to_fp6_e2m3",
        num_elements, 4, total_bytes, stream,
        input, reinterpret_cast<uint8_t*>(output), num_elements);
    CUDA_CHECK(cudaGetLastError());
}


/// FP16 → FP6 E3M2 cast kernel with transpose (32×32 SMEM tiling)
/// Input: FP16 ColMajor [rows × cols]. Output: packed FP6 ColMajor [cols × rows] (transposed).
__global__ void cast_fp16_to_fp6_e3m2_transposed_kernel_sm100(
    const __half* __restrict__ input,
    uint8_t* __restrict__ output,
    int rows, int cols,
    int64_t batch_stride_in,
    int64_t batch_stride_out)
{
    // SMEM: store 6-bit values as full bytes for transpose, pack on output
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
            tile[threadIdx.x][threadIdx.y] = fp16_to_fp6_e3m2(val);
        }
    }

    __syncthreads();

    {
        // Transposed write with 6-bit packing (4 values → 3 bytes)
        int out_row = bx + threadIdx.x;
        int out_col = by + threadIdx.y;
        if (out_row < cols && out_col < rows) {
            int64_t linear_idx = out_row + static_cast<int64_t>(out_col) * cols;
            // Pack groups of 4: idx within group = linear_idx % 4
            int64_t group = linear_idx / 4;
            int pos = static_cast<int>(linear_idx % 4);
            uint8_t v = tile[threadIdx.y][threadIdx.x];

            // CuTe LSB-first: element at index i occupies bits [i*6, i*6+5].
            // Each element contributes v << (pos * 6) to the 24-bit group.
            // Use 32-bit atomicOr at 4-byte aligned address to avoid misalignment.
            int64_t base = group * 3;
            int64_t aligned_base = base & ~int64_t(3);
            int bit_shift = static_cast<int>((base & 3) * 8) + pos * 6;
            unsigned int val = static_cast<unsigned int>(v & 0x3F) << bit_shift;
            atomicOr(reinterpret_cast<unsigned int*>(out_b + aligned_base), val);
            // If the 6-bit value crosses a 32-bit word boundary, write the overflow
            if (bit_shift + 6 > 32) {
                unsigned int overflow = static_cast<unsigned int>(v & 0x3F) >> (32 - bit_shift);
                atomicOr(reinterpret_cast<unsigned int*>(out_b + aligned_base + 4), overflow);
            }
        }
    }
}

/// Launch helper: FP16 → FP6 E3M2 transposed cast
inline void cast_fp16_to_fp6_e3m2_transposed_sm100(
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
    int64_t stride_out = (static_cast<int64_t>(cols) * rows * 3 + 3) / 4;  // packed FP6 bytes

    // Zero output before atomicOr
    CUDA_CHECK(cudaMemsetAsync(output, 0, stride_out * batch_count, stream));

    cast_fp16_to_fp6_e3m2_transposed_kernel_sm100<<<grid, block, 0, stream>>>(
        input, reinterpret_cast<uint8_t*>(output), rows, cols, stride_in, stride_out);
    CUDA_CHECK(cudaGetLastError());
}


/// FP16 → FP6 E2M3 cast kernel with transpose (32×32 SMEM tiling)
__global__ void cast_fp16_to_fp6_e2m3_transposed_kernel_sm100(
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
            tile[threadIdx.x][threadIdx.y] = fp16_to_fp6_e2m3(val);
        }
    }

    __syncthreads();

    {
        int out_row = bx + threadIdx.x;
        int out_col = by + threadIdx.y;
        if (out_row < cols && out_col < rows) {
            int64_t linear_idx = out_row + static_cast<int64_t>(out_col) * cols;
            int64_t group = linear_idx / 4;
            int pos = static_cast<int>(linear_idx % 4);
            uint8_t v = tile[threadIdx.y][threadIdx.x];

            // CuTe LSB-first: element at index i occupies bits [i*6, i*6+5].
            int64_t base = group * 3;
            int64_t aligned_base = base & ~int64_t(3);
            int bit_shift = static_cast<int>((base & 3) * 8) + pos * 6;
            unsigned int val = static_cast<unsigned int>(v & 0x3F) << bit_shift;
            atomicOr(reinterpret_cast<unsigned int*>(out_b + aligned_base), val);
            if (bit_shift + 6 > 32) {
                unsigned int overflow = static_cast<unsigned int>(v & 0x3F) >> (32 - bit_shift);
                atomicOr(reinterpret_cast<unsigned int*>(out_b + aligned_base + 4), overflow);
            }
        }
    }
}

/// Launch helper: FP16 → FP6 E2M3 transposed cast
inline void cast_fp16_to_fp6_e2m3_transposed_sm100(
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
    int64_t stride_out = (static_cast<int64_t>(cols) * rows * 3 + 3) / 4;

    CUDA_CHECK(cudaMemsetAsync(output, 0, stride_out * batch_count, stream));

    cast_fp16_to_fp6_e2m3_transposed_kernel_sm100<<<grid, block, 0, stream>>>(
        input, reinterpret_cast<uint8_t*>(output), rows, cols, stride_in, stride_out);
    CUDA_CHECK(cudaGetLastError());
}


// ========================================================================================
// F1: Fused Deinterleave + Cast for FP6 (interleaved FP16 → 2 × planar FP6)
// ========================================================================================

/// F1 linear: interleaved complex FP16 → 2 planar FP6 E3M2 buffers
__global__ void deinterleave_cast_fp16_to_fp6_e3m2_kernel_sm100(
    const __half* __restrict__ interleaved,
    uint8_t* __restrict__ out_real,
    uint8_t* __restrict__ out_imag,
    int64_t num_complex_elements)
{
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;

    // Process 4 complex elements → 4 Re + 4 Im → 3 bytes each
    for (int64_t i = idx * 4; i < num_complex_elements; i += stride * 4) {
        uint8_t re[4], im[4];
        int count = min(static_cast<int64_t>(4), num_complex_elements - i);

        if (i + 3 < num_complex_elements) {
            // Vectorized: load 8 FP16 = 128 bits
            const float4* in_vec = reinterpret_cast<const float4*>(interleaved + 2 * i);
            float4 data = *in_vec;
            const __half* halfs = reinterpret_cast<const __half*>(&data);
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                re[j] = fp16_to_fp6_e3m2(__half2float(halfs[2 * j]));
                im[j] = fp16_to_fp6_e3m2(__half2float(halfs[2 * j + 1]));
            }
        } else {
            for (int j = 0; j < count; ++j) {
                re[j] = fp16_to_fp6_e3m2(__half2float(interleaved[2 * (i + j)]));
                im[j] = fp16_to_fp6_e3m2(__half2float(interleaved[2 * (i + j) + 1]));
            }
            for (int j = count; j < 4; ++j) { re[j] = 0; im[j] = 0; }
        }

        // Pack 4 × 6-bit → 3 bytes
        int64_t out_base = (i / 4) * 3;
        if (out_base + 2 < (num_complex_elements * 3 + 3) / 4) {
            out_real[out_base + 0] = re[0] | (re[1] << 6);
            out_real[out_base + 1] = (re[1] >> 2) | (re[2] << 4);
            out_real[out_base + 2] = (re[2] >> 4) | (re[3] << 2);
            out_imag[out_base + 0] = im[0] | (im[1] << 6);
            out_imag[out_base + 1] = (im[1] >> 2) | (im[2] << 4);
            out_imag[out_base + 2] = (im[2] >> 4) | (im[3] << 2);
        }
    }
}

inline void deinterleave_cast_fp16_to_fp6_e3m2_sm100(
    const __half* interleaved,
    void* out_real, void* out_imag,
    int64_t num_complex_elements,
    cudaStream_t stream = nullptr)
{
    // read interleaved FP16 (4B/complex) + write 2× FP6 (2×0.75B/complex) ≈ 6B/complex
    int64_t total_bytes = num_complex_elements * 6;
    TUNED_LAUNCH_1D(deinterleave_cast_fp16_to_fp6_e3m2_kernel_sm100, "deinterleave_cast_fp16_to_fp6_e3m2",
        num_complex_elements, 4, total_bytes, stream,
        interleaved, reinterpret_cast<uint8_t*>(out_real),
        reinterpret_cast<uint8_t*>(out_imag), num_complex_elements);
    CUDA_CHECK(cudaGetLastError());
}

/// F1 linear: interleaved complex FP16 → 2 planar FP6 E2M3 buffers
__global__ void deinterleave_cast_fp16_to_fp6_e2m3_kernel_sm100(
    const __half* __restrict__ interleaved,
    uint8_t* __restrict__ out_real,
    uint8_t* __restrict__ out_imag,
    int64_t num_complex_elements)
{
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;

    for (int64_t i = idx * 4; i < num_complex_elements; i += stride * 4) {
        uint8_t re[4], im[4];
        int count = min(static_cast<int64_t>(4), num_complex_elements - i);

        if (i + 3 < num_complex_elements) {
            const float4* in_vec = reinterpret_cast<const float4*>(interleaved + 2 * i);
            float4 data = *in_vec;
            const __half* halfs = reinterpret_cast<const __half*>(&data);
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                re[j] = fp16_to_fp6_e2m3(__half2float(halfs[2 * j]));
                im[j] = fp16_to_fp6_e2m3(__half2float(halfs[2 * j + 1]));
            }
        } else {
            for (int j = 0; j < count; ++j) {
                re[j] = fp16_to_fp6_e2m3(__half2float(interleaved[2 * (i + j)]));
                im[j] = fp16_to_fp6_e2m3(__half2float(interleaved[2 * (i + j) + 1]));
            }
            for (int j = count; j < 4; ++j) { re[j] = 0; im[j] = 0; }
        }

        int64_t out_base = (i / 4) * 3;
        if (out_base + 2 < (num_complex_elements * 3 + 3) / 4) {
            out_real[out_base + 0] = re[0] | (re[1] << 6);
            out_real[out_base + 1] = (re[1] >> 2) | (re[2] << 4);
            out_real[out_base + 2] = (re[2] >> 4) | (re[3] << 2);
            out_imag[out_base + 0] = im[0] | (im[1] << 6);
            out_imag[out_base + 1] = (im[1] >> 2) | (im[2] << 4);
            out_imag[out_base + 2] = (im[2] >> 4) | (im[3] << 2);
        }
    }
}

inline void deinterleave_cast_fp16_to_fp6_e2m3_sm100(
    const __half* interleaved,
    void* out_real, void* out_imag,
    int64_t num_complex_elements,
    cudaStream_t stream = nullptr)
{
    // read interleaved FP16 (4B/complex) + write 2× FP6 (2×0.75B/complex) ≈ 6B/complex
    int64_t total_bytes = num_complex_elements * 6;
    TUNED_LAUNCH_1D(deinterleave_cast_fp16_to_fp6_e2m3_kernel_sm100, "deinterleave_cast_fp16_to_fp6_e2m3",
        num_complex_elements, 4, total_bytes, stream,
        interleaved, reinterpret_cast<uint8_t*>(out_real),
        reinterpret_cast<uint8_t*>(out_imag), num_complex_elements);
    CUDA_CHECK(cudaGetLastError());
}

/// F1 transposed: interleaved complex FP16 ColMajor → 2 transposed planar FP6 E3M2
__global__ void deinterleave_cast_fp16_to_fp6_e3m2_transposed_kernel_sm100(
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
            tile_re[threadIdx.x][threadIdx.y] = fp16_to_fp6_e3m2(re_val);
            tile_im[threadIdx.x][threadIdx.y] = fp16_to_fp6_e3m2(im_val);
        }
    }

    __syncthreads();

    {
        int out_row = bx + threadIdx.x;
        int out_col = by + threadIdx.y;
        if (out_row < cols && out_col < rows) {
            int64_t linear_idx = out_row + static_cast<int64_t>(out_col) * cols;
            int64_t group = linear_idx / 4;
            int pos = static_cast<int>(linear_idx % 4);
            uint8_t v_re = tile_re[threadIdx.y][threadIdx.x];
            uint8_t v_im = tile_im[threadIdx.y][threadIdx.x];

            // CuTe LSB-first: element at index i occupies bits [i*6, i*6+5].
            // Use 32-bit atomicOr at 4-byte aligned address to avoid misalignment.
            int64_t base = group * 3;
            int64_t aligned_base = base & ~int64_t(3);
            int bit_shift = static_cast<int>((base & 3) * 8) + pos * 6;

            unsigned int val_re = static_cast<unsigned int>(v_re & 0x3F) << bit_shift;
            unsigned int val_im = static_cast<unsigned int>(v_im & 0x3F) << bit_shift;
            atomicOr(reinterpret_cast<unsigned int*>(re_b + aligned_base), val_re);
            atomicOr(reinterpret_cast<unsigned int*>(im_b + aligned_base), val_im);
            // If the 6-bit value crosses a 32-bit word boundary, write the overflow
            if (bit_shift + 6 > 32) {
                unsigned int overflow_re = static_cast<unsigned int>(v_re & 0x3F) >> (32 - bit_shift);
                unsigned int overflow_im = static_cast<unsigned int>(v_im & 0x3F) >> (32 - bit_shift);
                atomicOr(reinterpret_cast<unsigned int*>(re_b + aligned_base + 4), overflow_re);
                atomicOr(reinterpret_cast<unsigned int*>(im_b + aligned_base + 4), overflow_im);
            }
        }
    }
}

inline void deinterleave_cast_fp16_to_fp6_e3m2_transposed_sm100(
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
    int64_t stride_out = (static_cast<int64_t>(cols) * rows * 3 + 3) / 4;

    CUDA_CHECK(cudaMemsetAsync(out_real, 0, stride_out * batch_count, stream));
    CUDA_CHECK(cudaMemsetAsync(out_imag, 0, stride_out * batch_count, stream));

    deinterleave_cast_fp16_to_fp6_e3m2_transposed_kernel_sm100<<<grid, block, 0, stream>>>(
        input, reinterpret_cast<uint8_t*>(out_real),
        reinterpret_cast<uint8_t*>(out_imag),
        rows, cols, stride_in, stride_out);
    CUDA_CHECK(cudaGetLastError());
}

/// F1 transposed: interleaved complex FP16 ColMajor → 2 transposed planar FP6 E2M3
__global__ void deinterleave_cast_fp16_to_fp6_e2m3_transposed_kernel_sm100(
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
            tile_re[threadIdx.x][threadIdx.y] = fp16_to_fp6_e2m3(re_val);
            tile_im[threadIdx.x][threadIdx.y] = fp16_to_fp6_e2m3(im_val);
        }
    }

    __syncthreads();

    {
        int out_row = bx + threadIdx.x;
        int out_col = by + threadIdx.y;
        if (out_row < cols && out_col < rows) {
            int64_t linear_idx = out_row + static_cast<int64_t>(out_col) * cols;
            int64_t group = linear_idx / 4;
            int pos = static_cast<int>(linear_idx % 4);
            uint8_t v_re = tile_re[threadIdx.y][threadIdx.x];
            uint8_t v_im = tile_im[threadIdx.y][threadIdx.x];

            // CuTe LSB-first: element at index i occupies bits [i*6, i*6+5].
            // Use 32-bit atomicOr at 4-byte aligned address to avoid misalignment.
            int64_t base = group * 3;
            int64_t aligned_base = base & ~int64_t(3);
            int bit_shift = static_cast<int>((base & 3) * 8) + pos * 6;

            unsigned int val_re = static_cast<unsigned int>(v_re & 0x3F) << bit_shift;
            unsigned int val_im = static_cast<unsigned int>(v_im & 0x3F) << bit_shift;
            atomicOr(reinterpret_cast<unsigned int*>(re_b + aligned_base), val_re);
            atomicOr(reinterpret_cast<unsigned int*>(im_b + aligned_base), val_im);
            // If the 6-bit value crosses a 32-bit word boundary, write the overflow
            if (bit_shift + 6 > 32) {
                unsigned int overflow_re = static_cast<unsigned int>(v_re & 0x3F) >> (32 - bit_shift);
                unsigned int overflow_im = static_cast<unsigned int>(v_im & 0x3F) >> (32 - bit_shift);
                atomicOr(reinterpret_cast<unsigned int*>(re_b + aligned_base + 4), overflow_re);
                atomicOr(reinterpret_cast<unsigned int*>(im_b + aligned_base + 4), overflow_im);
            }
        }
    }
}

inline void deinterleave_cast_fp16_to_fp6_e2m3_transposed_sm100(
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
    int64_t stride_out = (static_cast<int64_t>(cols) * rows * 3 + 3) / 4;

    CUDA_CHECK(cudaMemsetAsync(out_real, 0, stride_out * batch_count, stream));
    CUDA_CHECK(cudaMemsetAsync(out_imag, 0, stride_out * batch_count, stream));

    deinterleave_cast_fp16_to_fp6_e2m3_transposed_kernel_sm100<<<grid, block, 0, stream>>>(
        input, reinterpret_cast<uint8_t*>(out_real),
        reinterpret_cast<uint8_t*>(out_imag),
        rows, cols, stride_in, stride_out);
    CUDA_CHECK(cudaGetLastError());
}


// ========================================================================================
// F2: Fused Cast + Duplicate for FP6 (planar FP16 → 2× planar FP6)
// ========================================================================================

/// F2 linear: FP16 → 2× FP6 E3M2 dual-output (self-product paths)
__global__ void cast_fp16_to_fp6_e3m2_dual_kernel_sm100(
    const __half* __restrict__ input,
    uint8_t* __restrict__ output1,
    uint8_t* __restrict__ output2,
    int64_t num_elements)
{
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;

    for (int64_t i = idx * 4; i < num_elements; i += stride * 4) {
        uint8_t v[4];
        int count = min(static_cast<int64_t>(4), num_elements - i);
        for (int j = 0; j < count; ++j) {
            v[j] = fp16_to_fp6_e3m2(__half2float(input[i + j]));
        }
        for (int j = count; j < 4; ++j) v[j] = 0;

        int64_t out_base = (i / 4) * 3;
        if (out_base + 2 < (num_elements * 3 + 3) / 4) {
            uint8_t b0 = v[0] | (v[1] << 6);
            uint8_t b1 = (v[1] >> 2) | (v[2] << 4);
            uint8_t b2 = (v[2] >> 4) | (v[3] << 2);
            output1[out_base + 0] = b0; output2[out_base + 0] = b0;
            output1[out_base + 1] = b1; output2[out_base + 1] = b1;
            output1[out_base + 2] = b2; output2[out_base + 2] = b2;
        }
    }
}

inline void cast_fp16_to_fp6_e3m2_dual_sm100(
    const __half* input,
    void* output1, void* output2,
    int64_t num_elements,
    cudaStream_t stream = nullptr)
{
    // read FP16 (2B) + write 2× FP6 (2×0.75B) ≈ 4B/element
    int64_t total_bytes = num_elements * 4;
    TUNED_LAUNCH_1D(cast_fp16_to_fp6_e3m2_dual_kernel_sm100, "cast_fp16_to_fp6_e3m2_dual",
        num_elements, 4, total_bytes, stream,
        input, reinterpret_cast<uint8_t*>(output1),
        reinterpret_cast<uint8_t*>(output2), num_elements);
    CUDA_CHECK(cudaGetLastError());
}

/// F2 linear: FP16 → 2× FP6 E2M3 dual-output
__global__ void cast_fp16_to_fp6_e2m3_dual_kernel_sm100(
    const __half* __restrict__ input,
    uint8_t* __restrict__ output1,
    uint8_t* __restrict__ output2,
    int64_t num_elements)
{
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;

    for (int64_t i = idx * 4; i < num_elements; i += stride * 4) {
        uint8_t v[4];
        int count = min(static_cast<int64_t>(4), num_elements - i);
        for (int j = 0; j < count; ++j) {
            v[j] = fp16_to_fp6_e2m3(__half2float(input[i + j]));
        }
        for (int j = count; j < 4; ++j) v[j] = 0;

        int64_t out_base = (i / 4) * 3;
        if (out_base + 2 < (num_elements * 3 + 3) / 4) {
            uint8_t b0 = v[0] | (v[1] << 6);
            uint8_t b1 = (v[1] >> 2) | (v[2] << 4);
            uint8_t b2 = (v[2] >> 4) | (v[3] << 2);
            output1[out_base + 0] = b0; output2[out_base + 0] = b0;
            output1[out_base + 1] = b1; output2[out_base + 1] = b1;
            output1[out_base + 2] = b2; output2[out_base + 2] = b2;
        }
    }
}

inline void cast_fp16_to_fp6_e2m3_dual_sm100(
    const __half* input,
    void* output1, void* output2,
    int64_t num_elements,
    cudaStream_t stream = nullptr)
{
    // read FP16 (2B) + write 2× FP6 (2×0.75B) ≈ 4B/element
    int64_t total_bytes = num_elements * 4;
    TUNED_LAUNCH_1D(cast_fp16_to_fp6_e2m3_dual_kernel_sm100, "cast_fp16_to_fp6_e2m3_dual",
        num_elements, 4, total_bytes, stream,
        input, reinterpret_cast<uint8_t*>(output1),
        reinterpret_cast<uint8_t*>(output2), num_elements);
    CUDA_CHECK(cudaGetLastError());
}

/// F2 transposed: FP16 ColMajor → 2× transposed FP6 E3M2 (32×32 SMEM tiling)
__global__ void cast_fp16_to_fp6_e3m2_transposed_dual_kernel_sm100(
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
            tile[threadIdx.x][threadIdx.y] = fp16_to_fp6_e3m2(val);
        }
    }

    __syncthreads();

    {
        int out_row = bx + threadIdx.x;
        int out_col = by + threadIdx.y;
        if (out_row < cols && out_col < rows) {
            int64_t linear_idx = out_row + static_cast<int64_t>(out_col) * cols;
            int64_t group = linear_idx / 4;
            int pos = static_cast<int>(linear_idx % 4);
            uint8_t v = tile[threadIdx.y][threadIdx.x];

            // CuTe LSB-first: element at index i occupies bits [i*6, i*6+5].
            // Use 32-bit atomicOr at 4-byte aligned address.
            int64_t base = group * 3;
            int64_t aligned_base = base & ~int64_t(3);
            int bit_shift = static_cast<int>((base & 3) * 8) + pos * 6;

            unsigned int val = static_cast<unsigned int>(v & 0x3F) << bit_shift;
            atomicOr(reinterpret_cast<unsigned int*>(o1_b + aligned_base), val);
            atomicOr(reinterpret_cast<unsigned int*>(o2_b + aligned_base), val);
            if (bit_shift + 6 > 32) {
                unsigned int overflow = static_cast<unsigned int>(v & 0x3F) >> (32 - bit_shift);
                atomicOr(reinterpret_cast<unsigned int*>(o1_b + aligned_base + 4), overflow);
                atomicOr(reinterpret_cast<unsigned int*>(o2_b + aligned_base + 4), overflow);
            }
        }
    }
}

inline void cast_fp16_to_fp6_e3m2_transposed_dual_sm100(
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
    int64_t stride_out = (static_cast<int64_t>(cols) * rows * 3 + 3) / 4;

    CUDA_CHECK(cudaMemsetAsync(output1, 0, stride_out * batch_count, stream));
    CUDA_CHECK(cudaMemsetAsync(output2, 0, stride_out * batch_count, stream));

    cast_fp16_to_fp6_e3m2_transposed_dual_kernel_sm100<<<grid, block, 0, stream>>>(
        input, reinterpret_cast<uint8_t*>(output1),
        reinterpret_cast<uint8_t*>(output2),
        rows, cols, stride_in, stride_out);
    CUDA_CHECK(cudaGetLastError());
}

/// F2 transposed: FP16 ColMajor → 2× transposed FP6 E2M3
__global__ void cast_fp16_to_fp6_e2m3_transposed_dual_kernel_sm100(
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
            tile[threadIdx.x][threadIdx.y] = fp16_to_fp6_e2m3(val);
        }
    }

    __syncthreads();

    {
        int out_row = bx + threadIdx.x;
        int out_col = by + threadIdx.y;
        if (out_row < cols && out_col < rows) {
            int64_t linear_idx = out_row + static_cast<int64_t>(out_col) * cols;
            int64_t group = linear_idx / 4;
            int pos = static_cast<int>(linear_idx % 4);
            uint8_t v = tile[threadIdx.y][threadIdx.x];

            // CuTe LSB-first: element at index i occupies bits [i*6, i*6+5].
            // Use 32-bit atomicOr at 4-byte aligned address.
            int64_t base = group * 3;
            int64_t aligned_base = base & ~int64_t(3);
            int bit_shift = static_cast<int>((base & 3) * 8) + pos * 6;

            unsigned int val = static_cast<unsigned int>(v & 0x3F) << bit_shift;
            atomicOr(reinterpret_cast<unsigned int*>(o1_b + aligned_base), val);
            atomicOr(reinterpret_cast<unsigned int*>(o2_b + aligned_base), val);
            if (bit_shift + 6 > 32) {
                unsigned int overflow = static_cast<unsigned int>(v & 0x3F) >> (32 - bit_shift);
                atomicOr(reinterpret_cast<unsigned int*>(o1_b + aligned_base + 4), overflow);
                atomicOr(reinterpret_cast<unsigned int*>(o2_b + aligned_base + 4), overflow);
            }
        }
    }
}

inline void cast_fp16_to_fp6_e2m3_transposed_dual_sm100(
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
    int64_t stride_out = (static_cast<int64_t>(cols) * rows * 3 + 3) / 4;

    CUDA_CHECK(cudaMemsetAsync(output1, 0, stride_out * batch_count, stream));
    CUDA_CHECK(cudaMemsetAsync(output2, 0, stride_out * batch_count, stream));

    cast_fp16_to_fp6_e2m3_transposed_dual_kernel_sm100<<<grid, block, 0, stream>>>(
        input, reinterpret_cast<uint8_t*>(output1),
        reinterpret_cast<uint8_t*>(output2),
        rows, cols, stride_in, stride_out);
    CUDA_CHECK(cudaGetLastError());
}


// ========================================================================================
// F5: Fused Paired Cast for FP6 (2 independent inputs → 2 independent FP6 outputs)
// ========================================================================================

/// F5 linear: 2× FP16 → 2× FP6 E3M2 paired cast
__global__ void cast_fp16_to_fp6_e3m2_paired_kernel_sm100(
    const __half* __restrict__ input1,
    const __half* __restrict__ input2,
    uint8_t* __restrict__ output1,
    uint8_t* __restrict__ output2,
    int64_t num_elements)
{
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;

    for (int64_t i = idx * 4; i < num_elements; i += stride * 4) {
        uint8_t v1[4], v2[4];
        int count = min(static_cast<int64_t>(4), num_elements - i);
        for (int j = 0; j < count; ++j) {
            v1[j] = fp16_to_fp6_e3m2(__half2float(input1[i + j]));
            v2[j] = fp16_to_fp6_e3m2(__half2float(input2[i + j]));
        }
        for (int j = count; j < 4; ++j) { v1[j] = 0; v2[j] = 0; }

        int64_t out_base = (i / 4) * 3;
        if (out_base + 2 < (num_elements * 3 + 3) / 4) {
            output1[out_base + 0] = v1[0] | (v1[1] << 6);
            output1[out_base + 1] = (v1[1] >> 2) | (v1[2] << 4);
            output1[out_base + 2] = (v1[2] >> 4) | (v1[3] << 2);
            output2[out_base + 0] = v2[0] | (v2[1] << 6);
            output2[out_base + 1] = (v2[1] >> 2) | (v2[2] << 4);
            output2[out_base + 2] = (v2[2] >> 4) | (v2[3] << 2);
        }
    }
}

inline void cast_fp16_to_fp6_e3m2_paired_sm100(
    const __half* input1, const __half* input2,
    void* output1, void* output2,
    int64_t num_elements,
    cudaStream_t stream = nullptr)
{
    // read 2× FP16 (2×2B) + write 2× FP6 (2×0.75B) ≈ 6B/element
    int64_t total_bytes = num_elements * 6;
    TUNED_LAUNCH_1D(cast_fp16_to_fp6_e3m2_paired_kernel_sm100, "cast_fp16_to_fp6_e3m2_paired",
        num_elements, 4, total_bytes, stream,
        input1, input2, reinterpret_cast<uint8_t*>(output1),
        reinterpret_cast<uint8_t*>(output2), num_elements);
    CUDA_CHECK(cudaGetLastError());
}

/// F5 linear: 2× FP16 → 2× FP6 E2M3 paired cast
__global__ void cast_fp16_to_fp6_e2m3_paired_kernel_sm100(
    const __half* __restrict__ input1,
    const __half* __restrict__ input2,
    uint8_t* __restrict__ output1,
    uint8_t* __restrict__ output2,
    int64_t num_elements)
{
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;

    for (int64_t i = idx * 4; i < num_elements; i += stride * 4) {
        uint8_t v1[4], v2[4];
        int count = min(static_cast<int64_t>(4), num_elements - i);
        for (int j = 0; j < count; ++j) {
            v1[j] = fp16_to_fp6_e2m3(__half2float(input1[i + j]));
            v2[j] = fp16_to_fp6_e2m3(__half2float(input2[i + j]));
        }
        for (int j = count; j < 4; ++j) { v1[j] = 0; v2[j] = 0; }

        int64_t out_base = (i / 4) * 3;
        if (out_base + 2 < (num_elements * 3 + 3) / 4) {
            output1[out_base + 0] = v1[0] | (v1[1] << 6);
            output1[out_base + 1] = (v1[1] >> 2) | (v1[2] << 4);
            output1[out_base + 2] = (v1[2] >> 4) | (v1[3] << 2);
            output2[out_base + 0] = v2[0] | (v2[1] << 6);
            output2[out_base + 1] = (v2[1] >> 2) | (v2[2] << 4);
            output2[out_base + 2] = (v2[2] >> 4) | (v2[3] << 2);
        }
    }
}

inline void cast_fp16_to_fp6_e2m3_paired_sm100(
    const __half* input1, const __half* input2,
    void* output1, void* output2,
    int64_t num_elements,
    cudaStream_t stream = nullptr)
{
    // read 2× FP16 (2×2B) + write 2× FP6 (2×0.75B) ≈ 6B/element
    int64_t total_bytes = num_elements * 6;
    TUNED_LAUNCH_1D(cast_fp16_to_fp6_e2m3_paired_kernel_sm100, "cast_fp16_to_fp6_e2m3_paired",
        num_elements, 4, total_bytes, stream,
        input1, input2, reinterpret_cast<uint8_t*>(output1),
        reinterpret_cast<uint8_t*>(output2), num_elements);
    CUDA_CHECK(cudaGetLastError());
}

/// F5 transposed: 2× FP16 → 2× transposed FP6 E3M2 paired cast (32×32 SMEM)
__global__ void cast_fp16_to_fp6_e3m2_transposed_paired_kernel_sm100(
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
            tile1[threadIdx.x][threadIdx.y] = fp16_to_fp6_e3m2(__half2float(in1_b[idx]));
            tile2[threadIdx.x][threadIdx.y] = fp16_to_fp6_e3m2(__half2float(in2_b[idx]));
        }
    }

    __syncthreads();

    {
        int out_row = bx + threadIdx.x;
        int out_col = by + threadIdx.y;
        if (out_row < cols && out_col < rows) {
            int64_t linear_idx = out_row + static_cast<int64_t>(out_col) * cols;
            int64_t group = linear_idx / 4;
            int pos = static_cast<int>(linear_idx % 4);
            uint8_t v1 = tile1[threadIdx.y][threadIdx.x];
            uint8_t v2 = tile2[threadIdx.y][threadIdx.x];

            // CuTe LSB-first: element at index i occupies bits [i*6, i*6+5].
            int64_t base = group * 3;
            int64_t aligned_base = base & ~int64_t(3);
            int bit_shift = static_cast<int>((base & 3) * 8) + pos * 6;
            {
                unsigned int val1 = static_cast<unsigned int>(v1 & 0x3F) << bit_shift;
                unsigned int val2 = static_cast<unsigned int>(v2 & 0x3F) << bit_shift;
                atomicOr(reinterpret_cast<unsigned int*>(o1_b + aligned_base), val1);
                atomicOr(reinterpret_cast<unsigned int*>(o2_b + aligned_base), val2);
                if (bit_shift + 6 > 32) {
                    unsigned int ov1 = static_cast<unsigned int>(v1 & 0x3F) >> (32 - bit_shift);
                    unsigned int ov2 = static_cast<unsigned int>(v2 & 0x3F) >> (32 - bit_shift);
                    atomicOr(reinterpret_cast<unsigned int*>(o1_b + aligned_base + 4), ov1);
                    atomicOr(reinterpret_cast<unsigned int*>(o2_b + aligned_base + 4), ov2);
                }
            }
        }
    }
}

inline void cast_fp16_to_fp6_e3m2_transposed_paired_sm100(
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
    int64_t stride_out = (static_cast<int64_t>(cols) * rows * 3 + 3) / 4;

    CUDA_CHECK(cudaMemsetAsync(output1, 0, stride_out * batch_count, stream));
    CUDA_CHECK(cudaMemsetAsync(output2, 0, stride_out * batch_count, stream));

    cast_fp16_to_fp6_e3m2_transposed_paired_kernel_sm100<<<grid, block, 0, stream>>>(
        input1, input2, reinterpret_cast<uint8_t*>(output1),
        reinterpret_cast<uint8_t*>(output2),
        rows, cols, stride_in, stride_out);
    CUDA_CHECK(cudaGetLastError());
}

/// F5 transposed: 2× FP16 → 2× transposed FP6 E2M3 paired cast (32×32 SMEM)
__global__ void cast_fp16_to_fp6_e2m3_transposed_paired_kernel_sm100(
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
            tile1[threadIdx.x][threadIdx.y] = fp16_to_fp6_e2m3(__half2float(in1_b[idx]));
            tile2[threadIdx.x][threadIdx.y] = fp16_to_fp6_e2m3(__half2float(in2_b[idx]));
        }
    }

    __syncthreads();

    {
        int out_row = bx + threadIdx.x;
        int out_col = by + threadIdx.y;
        if (out_row < cols && out_col < rows) {
            int64_t linear_idx = out_row + static_cast<int64_t>(out_col) * cols;
            int64_t group = linear_idx / 4;
            int pos = static_cast<int>(linear_idx % 4);
            uint8_t v1 = tile1[threadIdx.y][threadIdx.x];
            uint8_t v2 = tile2[threadIdx.y][threadIdx.x];

            // CuTe LSB-first: element at index i occupies bits [i*6, i*6+5].
            int64_t base = group * 3;
            int64_t aligned_base = base & ~int64_t(3);
            int bit_shift = static_cast<int>((base & 3) * 8) + pos * 6;
            {
                unsigned int val1 = static_cast<unsigned int>(v1 & 0x3F) << bit_shift;
                unsigned int val2 = static_cast<unsigned int>(v2 & 0x3F) << bit_shift;
                atomicOr(reinterpret_cast<unsigned int*>(o1_b + aligned_base), val1);
                atomicOr(reinterpret_cast<unsigned int*>(o2_b + aligned_base), val2);
                if (bit_shift + 6 > 32) {
                    unsigned int ov1 = static_cast<unsigned int>(v1 & 0x3F) >> (32 - bit_shift);
                    unsigned int ov2 = static_cast<unsigned int>(v2 & 0x3F) >> (32 - bit_shift);
                    atomicOr(reinterpret_cast<unsigned int*>(o1_b + aligned_base + 4), ov1);
                    atomicOr(reinterpret_cast<unsigned int*>(o2_b + aligned_base + 4), ov2);
                }
            }
        }
    }
}

inline void cast_fp16_to_fp6_e2m3_transposed_paired_sm100(
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
    int64_t stride_out = (static_cast<int64_t>(cols) * rows * 3 + 3) / 4;

    CUDA_CHECK(cudaMemsetAsync(output1, 0, stride_out * batch_count, stream));
    CUDA_CHECK(cudaMemsetAsync(output2, 0, stride_out * batch_count, stream));

    cast_fp16_to_fp6_e2m3_transposed_paired_kernel_sm100<<<grid, block, 0, stream>>>(
        input1, input2, reinterpret_cast<uint8_t*>(output1),
        reinterpret_cast<uint8_t*>(output2),
        rows, cols, stride_in, stride_out);
    CUDA_CHECK(cudaGetLastError());
}


#endif  // COMPLEX_SM100_ENABLE_FP6
