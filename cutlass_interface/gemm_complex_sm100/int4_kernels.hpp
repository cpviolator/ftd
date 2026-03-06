// ========================================================================================
// INT4 Deinterleave Kernels — Sign-Magnitude INT4 Complex Input
// ========================================================================================
//
// Input format: each byte = 1 complex element
//   High nibble (bits 7:4) = Re in sign-magnitude (bit 7=sign, bits 6:4=magnitude)
//   Low nibble  (bits 3:0) = Im in sign-magnitude (bit 3=sign, bits 2:0=magnitude)
//   Range: [-7, +7] per component
//
// Four target-specific deinterleave kernels:
//   (A) INT4 → FP8 E4M3   (exact for [-7,7], LUT-based)
//   (B) INT4 → INT8        (exact, sign-magnitude → two's complement, LUT-based)
//   (C) INT4 → FP16        (exact, front-end for MXFP FP6/FP4 pipeline, LUT-based)
//   (D) INT4 → FP16 interleaved (for HERK pipeline, LUT-based)
//
// All kernels use constant-memory LUTs to eliminate per-element float arithmetic,
// and process 16 elements per thread per iteration with 128-bit vectorized loads/stores.
//

// ========================================================================================
// Constant-memory LUTs
// ========================================================================================

/// @brief 16-entry nibble → FP8 E4M3 bit-pattern LUT.
///
/// Maps a 4-bit sign-magnitude nibble (bit 3 = sign, bits 2:0 = magnitude)
/// to the uint8_t bit pattern of the corresponding FP8 E4M3 value.
/// All integers in [-7, +7] are exactly representable in FP8 E4M3.
__device__ __constant__ uint8_t int4_to_fp8_nib_lut[16] = {
    0x00, 0x38, 0x40, 0x44, 0x48, 0x4A, 0x4C, 0x4E,  // +0 to +7
    0x80, 0xB8, 0xC0, 0xC4, 0xC8, 0xCA, 0xCC, 0xCE   // -0 to -7
};

/// @brief 16-entry nibble → INT8 two's complement LUT.
///
/// Maps a 4-bit sign-magnitude nibble to the int8_t two's complement value.
__device__ __constant__ int8_t int4_to_int8_nib_lut[16] = {
    0, 1, 2, 3, 4, 5, 6, 7,         // +0 to +7
    0, -1, -2, -3, -4, -5, -6, -7   // -0 to -7
};

/// @brief 16-entry nibble → FP16 bit-pattern LUT.
///
/// Maps a 4-bit sign-magnitude nibble to the uint16_t bit pattern of
/// the corresponding __half value.  Eliminates int->float->half conversion.
__device__ __constant__ uint16_t int4_to_fp16_nib_lut[16] = {
    0x0000, 0x3C00, 0x4000, 0x4200, 0x4400, 0x4500, 0x4600, 0x4700,  // +0..+7
    0x8000, 0xBC00, 0xC000, 0xC200, 0xC400, 0xC500, 0xC600, 0xC700   // -0..-7
};

/// @brief Legacy decode function (kept for INT8 scalar tail).
__device__ __forceinline__ int8_t int4_sm_to_int8(uint8_t nibble) {
    return int4_to_int8_nib_lut[nibble & 0xF];
}

// ----- (A) INT4 → FP8 E4M3 Deinterleave (LUT-based, 128-bit vectorized) -----

__global__ void deinterleave_int4_to_fp8_kernel_sm100(
    const uint8_t* __restrict__ input,         // packed INT4 complex [Re|Im per byte]
    __nv_fp8_e4m3* __restrict__ out_real,       // FP8 real (contiguous)
    __nv_fp8_e4m3* __restrict__ out_imag,       // FP8 imag (contiguous)
    int64_t num_complex_elements)
{
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;

    uint8_t* re_out = reinterpret_cast<uint8_t*>(out_real);
    uint8_t* im_out = reinterpret_cast<uint8_t*>(out_imag);

    // Process 16 complex elements per iteration: 16 bytes in, 16 bytes Re out, 16 bytes Im out
    for (int64_t i = idx * 16; i < num_complex_elements; i += stride * 16) {
        if (i + 15 < num_complex_elements) {
            // 128-bit vectorized load: 16 QC bytes
            uint4 packed = *reinterpret_cast<const uint4*>(input + i);
            const uint8_t* bytes = reinterpret_cast<const uint8_t*>(&packed);

            uint8_t re[16], im[16];
            #pragma unroll
            for (int j = 0; j < 16; ++j) {
                re[j] = int4_to_fp8_nib_lut[(bytes[j] >> 4) & 0xF];
                im[j] = int4_to_fp8_nib_lut[bytes[j] & 0xF];
            }

            // 128-bit vectorized stores
            *reinterpret_cast<uint4*>(re_out + i) = *reinterpret_cast<const uint4*>(re);
            *reinterpret_cast<uint4*>(im_out + i) = *reinterpret_cast<const uint4*>(im);
        } else {
            for (int64_t j = i; j < num_complex_elements && j < i + 16; ++j) {
                re_out[j] = int4_to_fp8_nib_lut[(input[j] >> 4) & 0xF];
                im_out[j] = int4_to_fp8_nib_lut[input[j] & 0xF];
            }
        }
    }
}

/// Launch helper: INT4 complex → planar FP8
inline void deinterleave_int4_to_fp8(
    const uint8_t* input,
    __nv_fp8_e4m3* out_real, __nv_fp8_e4m3* out_imag,
    int64_t num_complex_elements,
    cudaStream_t stream = nullptr)
{
    int64_t total_bytes = num_complex_elements * 3;  // read 1B (packed INT4) + write 2B (2x FP8)
    TUNED_LAUNCH_1D(deinterleave_int4_to_fp8_kernel_sm100, "deinterleave_int4_to_fp8",
        num_complex_elements, 8, total_bytes, stream,
        input, out_real, out_imag, num_complex_elements);
    CUDA_CHECK(cudaGetLastError());
}


// ----- (B) INT4 → INT8 Deinterleave (LUT-based, 128-bit vectorized) -----

__global__ void deinterleave_int4_to_int8_kernel_sm100(
    const uint8_t* __restrict__ input,         // packed INT4 complex [Re|Im per byte]
    int8_t* __restrict__ out_real,              // INT8 real (contiguous)
    int8_t* __restrict__ out_imag,              // INT8 imag (contiguous)
    int64_t num_complex_elements)
{
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;

    // Process 16 complex elements per iteration
    for (int64_t i = idx * 16; i < num_complex_elements; i += stride * 16) {
        if (i + 15 < num_complex_elements) {
            uint4 packed = *reinterpret_cast<const uint4*>(input + i);
            const uint8_t* bytes = reinterpret_cast<const uint8_t*>(&packed);

            int8_t re[16], im[16];
            #pragma unroll
            for (int j = 0; j < 16; ++j) {
                re[j] = int4_to_int8_nib_lut[(bytes[j] >> 4) & 0xF];
                im[j] = int4_to_int8_nib_lut[bytes[j] & 0xF];
            }

            *reinterpret_cast<uint4*>(out_real + i) = *reinterpret_cast<const uint4*>(re);
            *reinterpret_cast<uint4*>(out_imag + i) = *reinterpret_cast<const uint4*>(im);
        } else {
            for (int64_t j = i; j < num_complex_elements && j < i + 16; ++j) {
                out_real[j] = int4_to_int8_nib_lut[(input[j] >> 4) & 0xF];
                out_imag[j] = int4_to_int8_nib_lut[input[j] & 0xF];
            }
        }
    }
}

/// Launch helper: INT4 complex → planar INT8
inline void deinterleave_int4_to_int8(
    const uint8_t* input,
    int8_t* out_real, int8_t* out_imag,
    int64_t num_complex_elements,
    cudaStream_t stream = nullptr)
{
    int64_t total_bytes = num_complex_elements * 3;  // read 1B (packed INT4) + write 2B (2x INT8)
    TUNED_LAUNCH_1D(deinterleave_int4_to_int8_kernel_sm100, "deinterleave_int4_to_int8",
        num_complex_elements, 8, total_bytes, stream,
        input, out_real, out_imag, num_complex_elements);
    CUDA_CHECK(cudaGetLastError());
}


// ----- (C) INT4 → FP16 Deinterleave (LUT-based, 128-bit vectorized) -----

__global__ void deinterleave_int4_to_fp16_kernel_sm100(
    const uint8_t* __restrict__ input,         // packed INT4 complex [Re|Im per byte]
    __half* __restrict__ out_real,              // FP16 real (contiguous)
    __half* __restrict__ out_imag,              // FP16 imag (contiguous)
    int64_t num_complex_elements)
{
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;

    // Process 8 complex elements per iteration: 8 bytes in, 16 bytes Re out, 16 bytes Im out
    for (int64_t i = idx * 8; i < num_complex_elements; i += stride * 8) {
        if (i + 7 < num_complex_elements) {
            uint2 packed = *reinterpret_cast<const uint2*>(input + i);
            const uint8_t* bytes = reinterpret_cast<const uint8_t*>(&packed);

            uint16_t re[8], im[8];
            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                re[j] = int4_to_fp16_nib_lut[(bytes[j] >> 4) & 0xF];
                im[j] = int4_to_fp16_nib_lut[bytes[j] & 0xF];
            }

            // 128-bit vectorized stores: 8 x FP16 = 16 bytes = uint4
            *reinterpret_cast<uint4*>(out_real + i) = *reinterpret_cast<const uint4*>(re);
            *reinterpret_cast<uint4*>(out_imag + i) = *reinterpret_cast<const uint4*>(im);
        } else {
            for (int64_t j = i; j < num_complex_elements && j < i + 8; ++j) {
                uint16_t re_bits = int4_to_fp16_nib_lut[(input[j] >> 4) & 0xF];
                uint16_t im_bits = int4_to_fp16_nib_lut[input[j] & 0xF];
                out_real[j] = *reinterpret_cast<const __half*>(&re_bits);
                out_imag[j] = *reinterpret_cast<const __half*>(&im_bits);
            }
        }
    }
}

/// Launch helper: INT4 complex → planar FP16
inline void deinterleave_int4_to_fp16(
    const uint8_t* input,
    __half* out_real, __half* out_imag,
    int64_t num_complex_elements,
    cudaStream_t stream = nullptr)
{
    int64_t total_bytes = num_complex_elements * 5;  // read 1B (packed INT4) + write 4B (2x FP16)
    TUNED_LAUNCH_1D(deinterleave_int4_to_fp16_kernel_sm100, "deinterleave_int4_to_fp16",
        num_complex_elements, 4, total_bytes, stream,
        input, out_real, out_imag, num_complex_elements);
    CUDA_CHECK(cudaGetLastError());
}


// ----- (D) INT4 → Interleaved FP16 (for HERK pipeline, LUT-based) -----
//
// Converts packed INT4 sign-magnitude complex to interleaved FP16 complex:
//   Input:  1 byte per complex element (high nibble=Re, low nibble=Im)
//   Output: 2 x __half per complex element [Re, Im, Re, Im, ...]
//
// This is the front-end for HERK_batched_int4: INT4 → interleaved FP16 → HERK.

__global__ void cast_int4_to_fp16_interleaved_kernel_sm100(
    const uint8_t* __restrict__ input,     // packed INT4 complex [num_elements bytes]
    __half* __restrict__ output,           // interleaved FP16 [num_elements x 2 halves]
    int64_t num_complex_elements)
{
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;

    // Process 8 complex elements per iteration:
    //   Read: 8 bytes (uint2)
    //   Write: 16 x __half = 32 bytes (2 x uint4)
    for (int64_t i = idx * 8; i < num_complex_elements; i += stride * 8) {
        if (i + 7 < num_complex_elements) {
            uint2 packed = *reinterpret_cast<const uint2*>(input + i);
            const uint8_t* bytes = reinterpret_cast<const uint8_t*>(&packed);

            // Each byte -> 2 FP16 values (Re, Im) packed as uint32_t
            uint16_t out_h[16];
            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                out_h[j * 2]     = int4_to_fp16_nib_lut[(bytes[j] >> 4) & 0xF];
                out_h[j * 2 + 1] = int4_to_fp16_nib_lut[bytes[j] & 0xF];
            }

            // Store 16 x FP16 = 32 bytes = 2 x uint4
            *reinterpret_cast<uint4*>(output + i * 2)      = *reinterpret_cast<const uint4*>(&out_h[0]);
            *reinterpret_cast<uint4*>(output + i * 2 + 8)  = *reinterpret_cast<const uint4*>(&out_h[8]);
        } else {
            for (int64_t j = i; j < num_complex_elements && j < i + 8; ++j) {
                uint16_t re_bits = int4_to_fp16_nib_lut[(input[j] >> 4) & 0xF];
                uint16_t im_bits = int4_to_fp16_nib_lut[input[j] & 0xF];
                output[j * 2]     = *reinterpret_cast<const __half*>(&re_bits);
                output[j * 2 + 1] = *reinterpret_cast<const __half*>(&im_bits);
            }
        }
    }
}

/// Launch helper: INT4 complex → interleaved FP16
inline void cast_int4_to_fp16_interleaved(
    const uint8_t* input,
    __half* output,
    int64_t num_complex_elements,
    cudaStream_t stream = nullptr)
{
    int64_t total_bytes = num_complex_elements * 5;  // read 1B + write 4B
    TUNED_LAUNCH_1D(cast_int4_to_fp16_interleaved_kernel_sm100, "cast_int4_to_fp16_interleaved",
        num_complex_elements, 4, total_bytes, stream,
        input, output, num_complex_elements);
    CUDA_CHECK(cudaGetLastError());
}

