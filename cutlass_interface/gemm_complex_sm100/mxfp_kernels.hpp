// ========================================================================================
// Precision-aware cast dispatch helper
// ========================================================================================
//
// Dispatches FP16 → narrow cast based on ComputePrecision enum.
// Handles the correct clamp range and packing format for each precision.

inline void cast_fp16_to_narrow_sm100(
    const __half* input,
    void* output,
    int64_t num_elements,
    ComputePrecision precision,
    cudaStream_t stream = nullptr)
{
    switch (precision) {
    case ComputePrecision::FP8_E4M3:
        cast_fp16_to_fp8_e4m3(
            input,
            reinterpret_cast<__nv_fp8_e4m3*>(output),
            num_elements, stream);
        break;
#ifdef COMPLEX_SM100_ENABLE_FP6
    case ComputePrecision::FP6_E3M2:
        cast_fp16_to_fp6_e3m2_sm100(input, output, num_elements, stream);
        break;
    case ComputePrecision::FP6_E2M3:
        cast_fp16_to_fp6_e2m3_sm100(input, output, num_elements, stream);
        break;
#endif
#ifdef COMPLEX_SM100_ENABLE_FP4
    case ComputePrecision::FP4_E2M1:
        cast_fp16_to_fp4_e2m1_sm100(input, output, num_elements, stream);
        break;
#endif
    default:
        throw std::runtime_error(
            "Unsupported compute precision. Enable the required precision at compile time: "
            "-DCOMPLEX_SM100_ENABLE_FP6=1 or -DCOMPLEX_SM100_ENABLE_FP4=1");
    }
}

/// Returns the number of bytes needed to store `num_elements` in the given precision.
inline int64_t bytes_for_elements(int64_t num_elements, ComputePrecision precision) {
    switch (precision) {
    case ComputePrecision::FP8_E4M3:
        return num_elements;  // 1 byte/element
#ifdef COMPLEX_SM100_ENABLE_FP6
    case ComputePrecision::FP6_E3M2:
    case ComputePrecision::FP6_E2M3:
        return (num_elements * 3 + 3) / 4;  // 0.75 bytes/element, rounded up
#endif
#ifdef COMPLEX_SM100_ENABLE_FP4
    case ComputePrecision::FP4_E2M1:
        return (num_elements + 1) / 2;  // 0.5 bytes/element, rounded up
#endif
    default:
        return num_elements;
    }
}


// ========================================================================================
// MXFP Scale Factor Computation Kernel
// ========================================================================================
//
// MXFP (Microscaling) block-scaled types require per-block scale factors.
// Each block of 32 consecutive K-elements shares one float_ue8m0_t scale factor.
// The scale factor is: 2^ceil(log2(max_abs(block) / narrow_type_max)).
//
// The hardware MMA multiplies: result += (data × scale_A) × (data × scale_B)
// So data is pre-divided by scale, and scale is stored separately.
//
// Scale factors must be written in the SfAtom interleaved layout:
//   For (m, k_block) where k_block = k/32:
//     tile_m = m / 128,  tile_k = k_block / 4
//     local_m = m % 128, local_k_block = k_block % 4
//     m_inner = local_m % 32, m_outer = local_m / 32
//     offset = (tile_m * num_k_tiles + tile_k) * 512
//            + m_inner * 16 + m_outer * 4 + local_k_block
//

#if defined(COMPLEX_SM100_ENABLE_FP6) || defined(COMPLEX_SM100_ENABLE_FP4)

/// Runtime-selectable narrow format for fused MXFP preprocessing.
enum class NarrowFormat { FP6_E3M2, FP6_E2M3, FP4_E2M1 };

/// Convert a float value to the specified narrow format (6-bit or 4-bit).
/// Returns the narrow representation in the low bits of a uint8_t.
template <NarrowFormat Format>
__device__ __forceinline__ uint8_t convert_to_narrow(float val) {
    if constexpr (Format == NarrowFormat::FP6_E3M2) return fp16_to_fp6_e3m2(val);
    else if constexpr (Format == NarrowFormat::FP6_E2M3) return fp16_to_fp6_e2m3(val);
    else if constexpr (Format == NarrowFormat::FP4_E2M1) return fp16_to_fp4_e2m1(val);
}

/// Compute the SfAtom physical offset for a given (m, k_block) coordinate.
/// k_block = k / 32 (which 32-element K block).
/// num_k_tiles = ceil(K / 128) = ceil(total_k_blocks / 4).
__device__ __forceinline__
int64_t sf_atom_offset(int m, int k_block, int num_k_tiles) {
    int tile_m = m / 128;
    int tile_k = k_block / 4;
    int local_m = m % 128;
    int local_k_block = k_block % 4;
    int m_inner = local_m % 32;
    int m_outer = local_m / 32;
    return static_cast<int64_t>(tile_m * num_k_tiles + tile_k) * 512
         + m_inner * 16 + m_outer * 4 + local_k_block;
}

/// Compute MXFP scale factors from FP16 data and write in SfAtom layout.
/// One thread per (m, k_block) pair. Reads 32 consecutive FP16 values,
/// computes max abs, converts to float_ue8m0_t power-of-2 scale.
///
/// Also writes the quantized (scaled) narrow-type data to the output buffer.
/// For FP6: 4 values → 3 bytes MSB-first packing.
/// For FP4: 2 values → 1 byte nibble packing.
///
/// Template parameter NarrowBits: 4 for FP4, 6 for FP6.
/// max_abs_val: max representable value (448 for FP8, 28 for FP6 E3M2, 7.5 for FP6 E2M3, 6 for FP4)
template <int NarrowBits, int BlockSize = 32>
__global__ void compute_mxfp_scale_factors_kernel_sm100(
    const __half* __restrict__ input,   // FP16 data, M × K RowMajor
    uint8_t* __restrict__ sf_output,    // scale factors in SfAtom layout
    int M, int K,
    int num_k_tiles,                    // ceil(K / 128)
    float narrow_max)                   // max representable value of narrow type
{
    const int m = blockIdx.y * blockDim.y + threadIdx.y;
    const int k_block = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_k_blocks = (K + BlockSize - 1) / BlockSize;

    if (m >= M || k_block >= total_k_blocks) return;

    // Read 32 consecutive K elements and find max abs
    const int64_t base = static_cast<int64_t>(m) * K + k_block * BlockSize;
    const int valid = min(BlockSize, K - k_block * BlockSize);

    float max_abs = 0.0f;
    for (int i = 0; i < valid; ++i) {
        float val = __half2float(input[base + i]);
        float abs_val = fabsf(val);
        max_abs = fmaxf(max_abs, abs_val);
    }

    // Compute power-of-2 scale: smallest power of 2 >= (max_abs / narrow_max)
    // float_ue8m0_t represents 2^(biased_exp - 127), biased_exp in [0..254], 255 = NaN
    // Scale = 2^ceil(log2(max_abs / narrow_max))
    // If max_abs == 0, scale = 2^(-127) (minimum positive)
    uint8_t biased_exp;
    if (max_abs == 0.0f) {
        biased_exp = 0;  // 2^(-127), effectively zero scaling
    } else {
        float ratio = max_abs / narrow_max;
        // ceil(log2(ratio)) → find smallest power-of-2 >= ratio
        // Use IEEE 754 float exponent extraction
        int exp_bits;
        frexpf(ratio, &exp_bits);  // ratio = mantissa * 2^exp_bits, mantissa in [0.5, 1)
        // If ratio is an exact power of 2, frexp gives exp_bits such that 2^(exp_bits-1) == ratio
        // Otherwise we need the next power of 2
        float p2 = ldexpf(1.0f, exp_bits - 1);  // 2^(exp_bits-1)
        if (p2 < ratio) exp_bits++;  // round up
        int unbiased = exp_bits - 1;  // actual exponent
        int biased = unbiased + 127;
        biased_exp = static_cast<uint8_t>(max(0, min(254, biased)));
    }

    // Write scale factor to SfAtom layout
    int64_t sf_idx = sf_atom_offset(m, k_block, num_k_tiles);
    sf_output[sf_idx] = biased_exp;
}

/// Launch scale factor computation for a matrix of dimensions M × K.
inline void compute_mxfp_scale_factors_sm100(
    const __half* input,
    void* sf_output,
    int M, int K,
    ComputePrecision precision,
    cudaStream_t stream = nullptr)
{
    const int total_k_blocks = (K + 31) / 32;
    const int num_k_tiles = (K + 127) / 128;

    float narrow_max;
    switch (precision) {
#ifdef COMPLEX_SM100_ENABLE_FP6
    case ComputePrecision::FP6_E3M2: narrow_max = 28.0f; break;
    case ComputePrecision::FP6_E2M3: narrow_max = 7.5f; break;
#endif
#ifdef COMPLEX_SM100_ENABLE_FP4
    case ComputePrecision::FP4_E2M1: narrow_max = 6.0f; break;
#endif
    default:
        throw std::runtime_error("compute_mxfp_scale_factors: unsupported precision");
    }

    // 2D grid: x = k_blocks, y = M rows
    dim3 block(32, 8);  // 256 threads
    dim3 grid((total_k_blocks + block.x - 1) / block.x,
              (M + block.y - 1) / block.y);

    compute_mxfp_scale_factors_kernel_sm100<32>
        <<<grid, block, 0, stream>>>(
            input, static_cast<uint8_t*>(sf_output),
            M, K, num_k_tiles, narrow_max);
}

/// Apply MXFP block scaling to FP16 data in-place or to a separate output buffer.
/// Divides each FP16 element by its block's scale factor (read from SfAtom layout).
/// The resulting FP16 values are in the range [-narrow_max, narrow_max] and can be
/// safely cast to the narrow type using the existing cast kernels.
__global__ void apply_mxfp_scaling_kernel_sm100(
    const __half* __restrict__ input,       // FP16 source data, M × K RowMajor
    __half* __restrict__ output,            // FP16 scaled data (can alias input)
    const uint8_t* __restrict__ sf_input,   // scale factors in SfAtom layout
    int64_t num_elements,
    int K,
    int num_k_tiles)
{
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;

    for (int64_t i = idx; i < num_elements; i += stride) {
        int m = static_cast<int>(i / K);
        int k = static_cast<int>(i % K);
        int k_block = k / 32;

        // Read scale factor from SfAtom layout
        int64_t sf_idx = sf_atom_offset(m, k_block, num_k_tiles);
        uint8_t biased_exp = sf_input[sf_idx];
        // scale = 2^(biased_exp - 127). Divide input by scale = multiply by 2^(127 - biased_exp)
        float inv_scale = ldexpf(1.0f, 127 - static_cast<int>(biased_exp));

        float val = __half2float(input[i]);
        float scaled = val * inv_scale;
        output[i] = __float2half(scaled);
    }
}

/// Launch MXFP scaling: reads FP16 input + scale factors, writes scaled FP16 output.
/// After this, the existing cast kernels (cast_fp16_to_fp6_*, cast_fp16_to_fp4_*) can
/// be used to convert the scaled FP16 to the packed sub-byte format.
inline void apply_mxfp_scaling_sm100(
    const __half* input,
    __half* output,
    const void* sf_buffer,
    int M, int K,
    cudaStream_t stream = nullptr)
{
    int64_t num_elements = static_cast<int64_t>(M) * K;
    int num_k_tiles = (K + 127) / 128;
    int64_t total_bytes = num_elements * 4;  // read FP16 (2B) + read SF (~0) + write FP16 (2B)
    TUNED_LAUNCH_1D(apply_mxfp_scaling_kernel_sm100, "apply_mxfp_scaling",
        num_elements, 1, total_bytes, stream,
        input, output, static_cast<const uint8_t*>(sf_buffer),
        num_elements, K, num_k_tiles);
}

/// Fused MXFP preprocessing kernel: read FP16 → compute SF → scale → cast → write packed + SF.
/// Replaces the 3-step pipeline (compute_mxfp_scale_factors → apply_mxfp_scaling → cast)
/// with a single kernel launch. Reads input FP16 once, writes packed sub-byte + scale factors.
/// Bandwidth: ~2.78 B/element (fused) vs ~8.75 B/element (3-step), 3.2x reduction.
///
/// Thread mapping: 1 thread per (m, k_block), each processes 32 consecutive K elements.
/// Grid: 2D with x = k_blocks, y = M rows (same as old SF kernel).
template <NarrowFormat Format, int BlockSize = 32>
__global__ void fused_mxfp_preprocess_kernel_sm100(
    const __half* __restrict__ input,       // FP16 data, M × K RowMajor
    uint8_t* __restrict__ narrow_output,    // packed sub-byte output
    uint8_t* __restrict__ sf_output,        // scale factors in SfAtom layout
    int M, int K,
    int num_k_tiles,                        // ceil(K / 128)
    float narrow_max)                       // max representable value of narrow type
{
    const int m = (blockIdx.y + blockIdx.z * gridDim.y) * blockDim.y + threadIdx.y;
    const int k_block = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_k_blocks = (K + BlockSize - 1) / BlockSize;

    if (m >= M || k_block >= total_k_blocks) return;

    // --- Phase 1: Read 32 FP16 elements and compute max_abs ---
    const int64_t base = static_cast<int64_t>(m) * K + k_block * BlockSize;
    const int valid = min(BlockSize, K - k_block * BlockSize);

    float vals[BlockSize];
    float max_abs = 0.0f;

    for (int i = 0; i < valid; ++i) {
        vals[i] = __half2float(input[base + i]);
        max_abs = fmaxf(max_abs, fabsf(vals[i]));
    }
    for (int i = valid; i < BlockSize; ++i) {
        vals[i] = 0.0f;
    }

    // --- Phase 2: Compute biased_exp scale factor ---
    uint8_t biased_exp;
    if (max_abs == 0.0f) {
        biased_exp = 0;
    } else {
        float ratio = max_abs / narrow_max;
        int exp_bits;
        frexpf(ratio, &exp_bits);
        float p2 = ldexpf(1.0f, exp_bits - 1);
        if (p2 < ratio) exp_bits++;
        int unbiased = exp_bits - 1;
        int biased = unbiased + 127;
        biased_exp = static_cast<uint8_t>(max(0, min(254, biased)));
    }

    // --- Phase 3: Write scale factor to SfAtom layout ---
    int64_t sf_idx = sf_atom_offset(m, k_block, num_k_tiles);
    sf_output[sf_idx] = biased_exp;

    // --- Phase 4: Scale each element and convert to narrow format ---
    float inv_scale = ldexpf(1.0f, 127 - static_cast<int>(biased_exp));

    uint8_t narrow_vals[BlockSize];
    for (int i = 0; i < BlockSize; ++i) {
        float scaled = vals[i] * inv_scale;
        narrow_vals[i] = convert_to_narrow<Format>(scaled);
    }

    // --- Phase 5: Pack narrow values and write output ---
    // Output offset for this block of 32 elements at (m, k_block * 32)
    if constexpr (Format == NarrowFormat::FP4_E2M1) {
        // FP4: 2 values → 1 byte (nibble packing), 32 → 16 bytes
        const int64_t out_base = static_cast<int64_t>(m) * ((K + 1) / 2) + k_block * (BlockSize / 2);
        const int out_count = (valid + 1) / 2;
        #pragma unroll
        for (int i = 0; i < 16; ++i) {
            if (i < out_count) {
                uint8_t hi = narrow_vals[2 * i];
                uint8_t lo = (2 * i + 1 < valid) ? narrow_vals[2 * i + 1] : 0;
                narrow_output[out_base + i] = hi | (lo << 4);
            }
        }
    } else {
        // FP6: 4 values → 3 bytes (MSB-first packing), 32 → 24 bytes
        const int64_t out_base = static_cast<int64_t>(m) * ((K * 3 + 3) / 4) + k_block * ((BlockSize * 3) / 4);
        const int num_groups = (valid + 3) / 4;
        #pragma unroll
        for (int g = 0; g < 8; ++g) {
            if (g < num_groups) {
                uint8_t v0 = narrow_vals[4 * g];
                uint8_t v1 = (4 * g + 1 < valid) ? narrow_vals[4 * g + 1] : 0;
                uint8_t v2 = (4 * g + 2 < valid) ? narrow_vals[4 * g + 2] : 0;
                uint8_t v3 = (4 * g + 3 < valid) ? narrow_vals[4 * g + 3] : 0;
                int64_t byte_off = out_base + g * 3;
                narrow_output[byte_off + 0] = v0 | (v1 << 6);
                narrow_output[byte_off + 1] = (v1 >> 2) | (v2 << 4);
                narrow_output[byte_off + 2] = (v2 >> 4) | (v3 << 2);
            }
        }
    }
}

/// Fused MXFP preprocessing for two independent matrices (F5-style paired).
/// Processes both operands in a single kernel launch, halving launch count.
/// Each thread handles one (m, k_block) pair and processes both matrices sequentially
/// to avoid doubling register pressure from 64 floats.
template <NarrowFormat Format, int BlockSize = 32>
__global__ void fused_mxfp_preprocess_paired_kernel_sm100(
    const __half* __restrict__ input1,
    const __half* __restrict__ input2,
    uint8_t* __restrict__ narrow_out1,
    uint8_t* __restrict__ narrow_out2,
    uint8_t* __restrict__ sf_out1,
    uint8_t* __restrict__ sf_out2,
    int M, int K,
    int num_k_tiles,
    float narrow_max,
    int M_per_batch = 0,            // 0 = flat mode (no batching)
    int64_t sf_stride_batch = 0,    // SF bytes between batch elements
    int64_t data_stride_batch = 0)  // narrow data bytes between batch elements
{
    const int m = (blockIdx.y + blockIdx.z * gridDim.y) * blockDim.y + threadIdx.y;
    const int k_block = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_k_blocks = (K + BlockSize - 1) / BlockSize;

    if (m >= M || k_block >= total_k_blocks) return;

    // Per-batch offset computation: when M_per_batch > 0, each batch element
    // has its own SF tile atoms and narrow data region. local_m is the row
    // within the batch element; offsets shift the SF and narrow output pointers.
    int local_m = m;
    int64_t sf_base_offset = 0;
    int64_t data_base_offset = 0;
    int num_k_tiles_eff = num_k_tiles;
    if (M_per_batch > 0) {
        int batch_idx = m / M_per_batch;
        local_m = m % M_per_batch;
        sf_base_offset = static_cast<int64_t>(batch_idx) * sf_stride_batch;
        data_base_offset = static_cast<int64_t>(batch_idx) * data_stride_batch;
        // Per-batch SF layout uses M_per_batch-based num_k_tiles
        // (num_k_tiles is computed from K, same for all batches)
    }

    // Input is flat contiguous: m indexes across all batches
    const int64_t base = static_cast<int64_t>(m) * K + k_block * BlockSize;
    const int valid = min(BlockSize, K - k_block * BlockSize);

    // Process operand 1
    {
        float vals[BlockSize];
        float max_abs = 0.0f;
        for (int i = 0; i < valid; ++i) {
            vals[i] = __half2float(input1[base + i]);
            max_abs = fmaxf(max_abs, fabsf(vals[i]));
        }
        for (int i = valid; i < BlockSize; ++i) vals[i] = 0.0f;

        uint8_t biased_exp;
        if (max_abs == 0.0f) {
            biased_exp = 0;
        } else {
            float ratio = max_abs / narrow_max;
            int exp_bits;
            frexpf(ratio, &exp_bits);
            float p2 = ldexpf(1.0f, exp_bits - 1);
            if (p2 < ratio) exp_bits++;
            int biased = (exp_bits - 1) + 127;
            biased_exp = static_cast<uint8_t>(max(0, min(254, biased)));
        }

        sf_out1[sf_base_offset + sf_atom_offset(local_m, k_block, num_k_tiles_eff)] = biased_exp;
        float inv_scale = ldexpf(1.0f, 127 - static_cast<int>(biased_exp));

        uint8_t narrow_vals[BlockSize];
        for (int i = 0; i < BlockSize; ++i)
            narrow_vals[i] = convert_to_narrow<Format>(vals[i] * inv_scale);

        if constexpr (Format == NarrowFormat::FP4_E2M1) {
            const int64_t out_base = data_base_offset
                + static_cast<int64_t>(local_m) * ((K + 1) / 2) + k_block * (BlockSize / 2);
            const int out_count = (valid + 1) / 2;
            #pragma unroll
            for (int i = 0; i < 16; ++i) {
                if (i < out_count) {
                    uint8_t hi = narrow_vals[2 * i];
                    uint8_t lo = (2 * i + 1 < valid) ? narrow_vals[2 * i + 1] : 0;
                    narrow_out1[out_base + i] = hi | (lo << 4);
                }
            }
        } else {
            const int64_t out_base = data_base_offset
                + static_cast<int64_t>(local_m) * ((K * 3 + 3) / 4) + k_block * ((BlockSize * 3) / 4);
            const int num_groups = (valid + 3) / 4;
            #pragma unroll
            for (int g = 0; g < 8; ++g) {
                if (g < num_groups) {
                    uint8_t v0 = narrow_vals[4 * g];
                    uint8_t v1 = (4 * g + 1 < valid) ? narrow_vals[4 * g + 1] : 0;
                    uint8_t v2 = (4 * g + 2 < valid) ? narrow_vals[4 * g + 2] : 0;
                    uint8_t v3 = (4 * g + 3 < valid) ? narrow_vals[4 * g + 3] : 0;
                    int64_t byte_off = out_base + g * 3;
                    narrow_out1[byte_off + 0] = v0 | (v1 << 6);
                    narrow_out1[byte_off + 1] = (v1 >> 2) | (v2 << 4);
                    narrow_out1[byte_off + 2] = (v2 >> 4) | (v3 << 2);
                }
            }
        }
    }

    // Process operand 2
    {
        float vals[BlockSize];
        float max_abs = 0.0f;
        for (int i = 0; i < valid; ++i) {
            vals[i] = __half2float(input2[base + i]);
            max_abs = fmaxf(max_abs, fabsf(vals[i]));
        }
        for (int i = valid; i < BlockSize; ++i) vals[i] = 0.0f;

        uint8_t biased_exp;
        if (max_abs == 0.0f) {
            biased_exp = 0;
        } else {
            float ratio = max_abs / narrow_max;
            int exp_bits;
            frexpf(ratio, &exp_bits);
            float p2 = ldexpf(1.0f, exp_bits - 1);
            if (p2 < ratio) exp_bits++;
            int biased = (exp_bits - 1) + 127;
            biased_exp = static_cast<uint8_t>(max(0, min(254, biased)));
        }

        sf_out2[sf_base_offset + sf_atom_offset(local_m, k_block, num_k_tiles_eff)] = biased_exp;
        float inv_scale = ldexpf(1.0f, 127 - static_cast<int>(biased_exp));

        uint8_t narrow_vals[BlockSize];
        for (int i = 0; i < BlockSize; ++i)
            narrow_vals[i] = convert_to_narrow<Format>(vals[i] * inv_scale);

        if constexpr (Format == NarrowFormat::FP4_E2M1) {
            const int64_t out_base = data_base_offset
                + static_cast<int64_t>(local_m) * ((K + 1) / 2) + k_block * (BlockSize / 2);
            const int out_count = (valid + 1) / 2;
            #pragma unroll
            for (int i = 0; i < 16; ++i) {
                if (i < out_count) {
                    uint8_t hi = narrow_vals[2 * i];
                    uint8_t lo = (2 * i + 1 < valid) ? narrow_vals[2 * i + 1] : 0;
                    narrow_out2[out_base + i] = hi | (lo << 4);
                }
            }
        } else {
            const int64_t out_base = data_base_offset
                + static_cast<int64_t>(local_m) * ((K * 3 + 3) / 4) + k_block * ((BlockSize * 3) / 4);
            const int num_groups = (valid + 3) / 4;
            #pragma unroll
            for (int g = 0; g < 8; ++g) {
                if (g < num_groups) {
                    uint8_t v0 = narrow_vals[4 * g];
                    uint8_t v1 = (4 * g + 1 < valid) ? narrow_vals[4 * g + 1] : 0;
                    uint8_t v2 = (4 * g + 2 < valid) ? narrow_vals[4 * g + 2] : 0;
                    uint8_t v3 = (4 * g + 3 < valid) ? narrow_vals[4 * g + 3] : 0;
                    int64_t byte_off = out_base + g * 3;
                    narrow_out2[byte_off + 0] = v0 | (v1 << 6);
                    narrow_out2[byte_off + 1] = (v1 >> 2) | (v2 << 4);
                    narrow_out2[byte_off + 2] = (v2 >> 4) | (v3 << 2);
                }
            }
        }
    }
}

/// Get the narrow_max constant for a given ComputePrecision.
inline float narrow_max_for_precision(ComputePrecision precision) {
    switch (precision) {
#ifdef COMPLEX_SM100_ENABLE_FP6
    case ComputePrecision::FP6_E3M2: return 28.0f;
    case ComputePrecision::FP6_E2M3: return 7.5f;
#endif
#ifdef COMPLEX_SM100_ENABLE_FP4
    case ComputePrecision::FP4_E2M1: return 6.0f;
#endif
    default:
        throw std::runtime_error("narrow_max_for_precision: unsupported precision");
    }
}

/// Compute a 3D grid that splits the row dimension (Y) into Y*Z when
/// the row-block count exceeds the CUDA grid Y limit (65535).
inline dim3 mxfp_grid_3d(int grid_x, int row_blocks) {
    constexpr int kMaxGridY = 65535;
    if (row_blocks <= kMaxGridY) {
        return dim3(grid_x, row_blocks, 1);
    }
    int grid_z = (row_blocks + kMaxGridY - 1) / kMaxGridY;
    int grid_y = (row_blocks + grid_z - 1) / grid_z;
    return dim3(grid_x, grid_y, grid_z);
}

/// Launch the fused MXFP preprocessing kernel for a single operand.
/// Replaces the old 3-step pipeline: compute_SF → apply_scaling → cast.
inline void preprocess_mxfp_sm100(
    const __half* input,        // FP16 source, M × K RowMajor
    void* narrow_output,        // packed sub-byte output
    void* sf_output,            // scale factor output (SfAtom layout)
    int M, int K,
    ComputePrecision precision,
    cudaStream_t stream = nullptr)
{
    const int total_k_blocks = (K + 31) / 32;
    const int num_k_tiles = (K + 127) / 128;
    float narrow_max = narrow_max_for_precision(precision);

    dim3 block(32, 8);
    dim3 grid = mxfp_grid_3d((total_k_blocks + block.x - 1) / block.x,
                              (M + block.y - 1) / block.y);

    switch (precision) {
#ifdef COMPLEX_SM100_ENABLE_FP6
    case ComputePrecision::FP6_E3M2:
        fused_mxfp_preprocess_kernel_sm100<NarrowFormat::FP6_E3M2>
            <<<grid, block, 0, stream>>>(
                input, static_cast<uint8_t*>(narrow_output),
                static_cast<uint8_t*>(sf_output),
                M, K, num_k_tiles, narrow_max);
        break;
    case ComputePrecision::FP6_E2M3:
        fused_mxfp_preprocess_kernel_sm100<NarrowFormat::FP6_E2M3>
            <<<grid, block, 0, stream>>>(
                input, static_cast<uint8_t*>(narrow_output),
                static_cast<uint8_t*>(sf_output),
                M, K, num_k_tiles, narrow_max);
        break;
#endif
#ifdef COMPLEX_SM100_ENABLE_FP4
    case ComputePrecision::FP4_E2M1:
        fused_mxfp_preprocess_kernel_sm100<NarrowFormat::FP4_E2M1>
            <<<grid, block, 0, stream>>>(
                input, static_cast<uint8_t*>(narrow_output),
                static_cast<uint8_t*>(sf_output),
                M, K, num_k_tiles, narrow_max);
        break;
#endif
    default:
        throw std::runtime_error("preprocess_mxfp_sm100: unsupported precision");
    }
    CUDA_CHECK(cudaGetLastError());
}

/// Launch the fused paired MXFP preprocessing kernel for two operands.
/// Halves the number of kernel launches (2 → 1) by processing both
/// operands in a single grid launch.
inline void preprocess_mxfp_paired_sm100(
    const __half* input1, const __half* input2,
    void* narrow_out1, void* narrow_out2,
    void* sf_out1, void* sf_out2,
    int M, int K,
    ComputePrecision precision,
    cudaStream_t stream = nullptr)
{
    const int total_k_blocks = (K + 31) / 32;
    const int num_k_tiles = (K + 127) / 128;
    float narrow_max = narrow_max_for_precision(precision);

    dim3 block(32, 8);
    dim3 grid = mxfp_grid_3d((total_k_blocks + block.x - 1) / block.x,
                              (M + block.y - 1) / block.y);

    switch (precision) {
#ifdef COMPLEX_SM100_ENABLE_FP6
    case ComputePrecision::FP6_E3M2:
        fused_mxfp_preprocess_paired_kernel_sm100<NarrowFormat::FP6_E3M2>
            <<<grid, block, 0, stream>>>(
                input1, input2,
                static_cast<uint8_t*>(narrow_out1), static_cast<uint8_t*>(narrow_out2),
                static_cast<uint8_t*>(sf_out1), static_cast<uint8_t*>(sf_out2),
                M, K, num_k_tiles, narrow_max);
        break;
    case ComputePrecision::FP6_E2M3:
        fused_mxfp_preprocess_paired_kernel_sm100<NarrowFormat::FP6_E2M3>
            <<<grid, block, 0, stream>>>(
                input1, input2,
                static_cast<uint8_t*>(narrow_out1), static_cast<uint8_t*>(narrow_out2),
                static_cast<uint8_t*>(sf_out1), static_cast<uint8_t*>(sf_out2),
                M, K, num_k_tiles, narrow_max);
        break;
#endif
#ifdef COMPLEX_SM100_ENABLE_FP4
    case ComputePrecision::FP4_E2M1:
        fused_mxfp_preprocess_paired_kernel_sm100<NarrowFormat::FP4_E2M1>
            <<<grid, block, 0, stream>>>(
                input1, input2,
                static_cast<uint8_t*>(narrow_out1), static_cast<uint8_t*>(narrow_out2),
                static_cast<uint8_t*>(sf_out1), static_cast<uint8_t*>(sf_out2),
                M, K, num_k_tiles, narrow_max);
        break;
#endif
    default:
        throw std::runtime_error("preprocess_mxfp_paired_sm100: unsupported precision");
    }
    CUDA_CHECK(cudaGetLastError());
}

/// Batched MXFP paired preprocessing: processes M_per_batch × K per batch element
/// across batch_count elements in a single kernel launch.
/// Replaces the per-batch loop when M_per_batch is not a multiple of 128.
/// Each batch element gets its own SF tile atoms (sf_stride_batch apart) and
/// narrow data region (data_stride_batch apart). Input is contiguous:
/// [batch_count × M_per_batch × K] in row-major order.
inline void preprocess_mxfp_paired_batched_sm100(
    const __half* input1, const __half* input2,
    void* narrow_out1, void* narrow_out2,
    void* sf_out1, void* sf_out2,
    int M_per_batch, int K, int batch_count,
    int64_t data_bytes_per_batch,   // narrow data bytes per batch element
    int64_t sf_bytes_per_batch,     // SF bytes per batch element
    ComputePrecision precision,
    cudaStream_t stream = nullptr)
{
    int M_total = M_per_batch * batch_count;
    const int total_k_blocks = (K + 31) / 32;
    const int num_k_tiles = (K + 127) / 128;
    float narrow_max = narrow_max_for_precision(precision);

    dim3 block(32, 8);
    dim3 grid = mxfp_grid_3d((total_k_blocks + block.x - 1) / block.x,
                              (M_total + block.y - 1) / block.y);

    switch (precision) {
#ifdef COMPLEX_SM100_ENABLE_FP6
    case ComputePrecision::FP6_E3M2:
        fused_mxfp_preprocess_paired_kernel_sm100<NarrowFormat::FP6_E3M2>
            <<<grid, block, 0, stream>>>(
                input1, input2,
                static_cast<uint8_t*>(narrow_out1), static_cast<uint8_t*>(narrow_out2),
                static_cast<uint8_t*>(sf_out1), static_cast<uint8_t*>(sf_out2),
                M_total, K, num_k_tiles, narrow_max,
                M_per_batch, sf_bytes_per_batch, data_bytes_per_batch);
        break;
    case ComputePrecision::FP6_E2M3:
        fused_mxfp_preprocess_paired_kernel_sm100<NarrowFormat::FP6_E2M3>
            <<<grid, block, 0, stream>>>(
                input1, input2,
                static_cast<uint8_t*>(narrow_out1), static_cast<uint8_t*>(narrow_out2),
                static_cast<uint8_t*>(sf_out1), static_cast<uint8_t*>(sf_out2),
                M_total, K, num_k_tiles, narrow_max,
                M_per_batch, sf_bytes_per_batch, data_bytes_per_batch);
        break;
#endif
#ifdef COMPLEX_SM100_ENABLE_FP4
    case ComputePrecision::FP4_E2M1:
        fused_mxfp_preprocess_paired_kernel_sm100<NarrowFormat::FP4_E2M1>
            <<<grid, block, 0, stream>>>(
                input1, input2,
                static_cast<uint8_t*>(narrow_out1), static_cast<uint8_t*>(narrow_out2),
                static_cast<uint8_t*>(sf_out1), static_cast<uint8_t*>(sf_out2),
                M_total, K, num_k_tiles, narrow_max,
                M_per_batch, sf_bytes_per_batch, data_bytes_per_batch);
        break;
#endif
    default:
        throw std::runtime_error("preprocess_mxfp_paired_batched_sm100: unsupported precision");
    }
    CUDA_CHECK(cudaGetLastError());
}

/// Fused deinterleave + MXFP preprocessing for interleaved complex FP16 input.
/// Reads interleaved [Re0,Im0,Re1,Im1,...] and produces two MXFP-packed outputs
/// (Re and Im) plus scale factors, in a single kernel. Eliminates the intermediate
/// planar FP16 buffers that a separate deinterleave → MXFP pipeline would require.
///
/// Bandwidth savings vs unfused: eliminates 2 × M × K × sizeof(__half) intermediate
/// planar writes and reads (the deinterleave output / MXFP input).
template <NarrowFormat Format, int BlockSize = 32>
__global__ void deinterleave_mxfp_preprocess_paired_kernel_sm100(
    const __half* __restrict__ interleaved,   // [M × K × 2] interleaved complex FP16
    uint8_t* __restrict__ narrow_out_re,      // MXFP-packed real output
    uint8_t* __restrict__ narrow_out_im,      // MXFP-packed imag output
    uint8_t* __restrict__ sf_out_re,          // Scale factors for real
    uint8_t* __restrict__ sf_out_im,          // Scale factors for imag
    int M, int K,
    int num_k_tiles,
    float narrow_max,
    int M_per_batch = 0,
    int64_t sf_stride_batch = 0,
    int64_t data_stride_batch = 0)
{
    const int m = (blockIdx.y + blockIdx.z * gridDim.y) * blockDim.y + threadIdx.y;
    const int k_block = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_k_blocks = (K + BlockSize - 1) / BlockSize;

    if (m >= M || k_block >= total_k_blocks) return;

    int local_m = m;
    int64_t sf_base_offset = 0;
    int64_t data_base_offset = 0;
    int num_k_tiles_eff = num_k_tiles;
    if (M_per_batch > 0) {
        int batch_idx = m / M_per_batch;
        local_m = m % M_per_batch;
        sf_base_offset = static_cast<int64_t>(batch_idx) * sf_stride_batch;
        data_base_offset = static_cast<int64_t>(batch_idx) * data_stride_batch;
    }

    // Input is interleaved: element (m, k) has Re at interleaved[2*(m*K + k)]
    //                                       and Im at interleaved[2*(m*K + k) + 1]
    const int64_t row_start = static_cast<int64_t>(m) * K + k_block * BlockSize;
    const int valid = min(BlockSize, K - k_block * BlockSize);

    // Process real part (operand 1)
    {
        float vals[BlockSize];
        float max_abs = 0.0f;
        for (int i = 0; i < valid; ++i) {
            vals[i] = __half2float(interleaved[2 * (row_start + i)]);
            max_abs = fmaxf(max_abs, fabsf(vals[i]));
        }
        for (int i = valid; i < BlockSize; ++i) vals[i] = 0.0f;

        uint8_t biased_exp;
        if (max_abs == 0.0f) {
            biased_exp = 0;
        } else {
            float ratio = max_abs / narrow_max;
            int exp_bits;
            frexpf(ratio, &exp_bits);
            float p2 = ldexpf(1.0f, exp_bits - 1);
            if (p2 < ratio) exp_bits++;
            int biased = (exp_bits - 1) + 127;
            biased_exp = static_cast<uint8_t>(max(0, min(254, biased)));
        }

        sf_out_re[sf_base_offset + sf_atom_offset(local_m, k_block, num_k_tiles_eff)] = biased_exp;
        float inv_scale = ldexpf(1.0f, 127 - static_cast<int>(biased_exp));

        uint8_t narrow_vals[BlockSize];
        for (int i = 0; i < BlockSize; ++i)
            narrow_vals[i] = convert_to_narrow<Format>(vals[i] * inv_scale);

        if constexpr (Format == NarrowFormat::FP4_E2M1) {
            const int64_t out_base = data_base_offset
                + static_cast<int64_t>(local_m) * ((K + 1) / 2) + k_block * (BlockSize / 2);
            const int out_count = (valid + 1) / 2;
            #pragma unroll
            for (int i = 0; i < 16; ++i) {
                if (i < out_count) {
                    uint8_t hi = narrow_vals[2 * i];
                    uint8_t lo = (2 * i + 1 < valid) ? narrow_vals[2 * i + 1] : 0;
                    narrow_out_re[out_base + i] = hi | (lo << 4);
                }
            }
        } else {
            const int64_t out_base = data_base_offset
                + static_cast<int64_t>(local_m) * ((K * 3 + 3) / 4) + k_block * ((BlockSize * 3) / 4);
            const int num_groups = (valid + 3) / 4;
            #pragma unroll
            for (int g = 0; g < 8; ++g) {
                if (g < num_groups) {
                    uint8_t v0 = narrow_vals[4 * g];
                    uint8_t v1 = (4 * g + 1 < valid) ? narrow_vals[4 * g + 1] : 0;
                    uint8_t v2 = (4 * g + 2 < valid) ? narrow_vals[4 * g + 2] : 0;
                    uint8_t v3 = (4 * g + 3 < valid) ? narrow_vals[4 * g + 3] : 0;
                    int64_t byte_off = out_base + g * 3;
                    narrow_out_re[byte_off + 0] = v0 | (v1 << 6);
                    narrow_out_re[byte_off + 1] = (v1 >> 2) | (v2 << 4);
                    narrow_out_re[byte_off + 2] = (v2 >> 4) | (v3 << 2);
                }
            }
        }
    }

    // Process imaginary part (operand 2)
    {
        float vals[BlockSize];
        float max_abs = 0.0f;
        for (int i = 0; i < valid; ++i) {
            vals[i] = __half2float(interleaved[2 * (row_start + i) + 1]);
            max_abs = fmaxf(max_abs, fabsf(vals[i]));
        }
        for (int i = valid; i < BlockSize; ++i) vals[i] = 0.0f;

        uint8_t biased_exp;
        if (max_abs == 0.0f) {
            biased_exp = 0;
        } else {
            float ratio = max_abs / narrow_max;
            int exp_bits;
            frexpf(ratio, &exp_bits);
            float p2 = ldexpf(1.0f, exp_bits - 1);
            if (p2 < ratio) exp_bits++;
            int biased = (exp_bits - 1) + 127;
            biased_exp = static_cast<uint8_t>(max(0, min(254, biased)));
        }

        sf_out_im[sf_base_offset + sf_atom_offset(local_m, k_block, num_k_tiles_eff)] = biased_exp;
        float inv_scale = ldexpf(1.0f, 127 - static_cast<int>(biased_exp));

        uint8_t narrow_vals[BlockSize];
        for (int i = 0; i < BlockSize; ++i)
            narrow_vals[i] = convert_to_narrow<Format>(vals[i] * inv_scale);

        if constexpr (Format == NarrowFormat::FP4_E2M1) {
            const int64_t out_base = data_base_offset
                + static_cast<int64_t>(local_m) * ((K + 1) / 2) + k_block * (BlockSize / 2);
            const int out_count = (valid + 1) / 2;
            #pragma unroll
            for (int i = 0; i < 16; ++i) {
                if (i < out_count) {
                    uint8_t hi = narrow_vals[2 * i];
                    uint8_t lo = (2 * i + 1 < valid) ? narrow_vals[2 * i + 1] : 0;
                    narrow_out_im[out_base + i] = hi | (lo << 4);
                }
            }
        } else {
            const int64_t out_base = data_base_offset
                + static_cast<int64_t>(local_m) * ((K * 3 + 3) / 4) + k_block * ((BlockSize * 3) / 4);
            const int num_groups = (valid + 3) / 4;
            #pragma unroll
            for (int g = 0; g < 8; ++g) {
                if (g < num_groups) {
                    uint8_t v0 = narrow_vals[4 * g];
                    uint8_t v1 = (4 * g + 1 < valid) ? narrow_vals[4 * g + 1] : 0;
                    uint8_t v2 = (4 * g + 2 < valid) ? narrow_vals[4 * g + 2] : 0;
                    uint8_t v3 = (4 * g + 3 < valid) ? narrow_vals[4 * g + 3] : 0;
                    int64_t byte_off = out_base + g * 3;
                    narrow_out_im[byte_off + 0] = v0 | (v1 << 6);
                    narrow_out_im[byte_off + 1] = (v1 >> 2) | (v2 << 4);
                    narrow_out_im[byte_off + 2] = (v2 >> 4) | (v3 << 2);
                }
            }
        }
    }
}

/// Launch fused deinterleave + MXFP preprocessing for interleaved complex input.
/// Reads interleaved FP16, produces MXFP-packed Re/Im + scale factors.
inline void deinterleave_preprocess_mxfp_paired_sm100(
    const __half* interleaved,      // Input: interleaved complex [M × K × 2]
    void* narrow_out_re, void* narrow_out_im,
    void* sf_out_re, void* sf_out_im,
    int M, int K,
    ComputePrecision precision,
    cudaStream_t stream = nullptr,
    int M_per_batch = 0,
    int64_t sf_stride_batch = 0,
    int64_t data_stride_batch = 0)
{
    const int total_k_blocks = (K + 31) / 32;
    const int num_k_tiles = (K + 127) / 128;
    float narrow_max = narrow_max_for_precision(precision);

    dim3 block(32, 8);
    dim3 grid = mxfp_grid_3d((total_k_blocks + block.x - 1) / block.x,
                              (M + block.y - 1) / block.y);

    switch (precision) {
#ifdef COMPLEX_SM100_ENABLE_FP6
    case ComputePrecision::FP6_E3M2:
        deinterleave_mxfp_preprocess_paired_kernel_sm100<NarrowFormat::FP6_E3M2>
            <<<grid, block, 0, stream>>>(
                interleaved,
                static_cast<uint8_t*>(narrow_out_re), static_cast<uint8_t*>(narrow_out_im),
                static_cast<uint8_t*>(sf_out_re), static_cast<uint8_t*>(sf_out_im),
                M, K, num_k_tiles, narrow_max,
                M_per_batch, sf_stride_batch, data_stride_batch);
        break;
    case ComputePrecision::FP6_E2M3:
        deinterleave_mxfp_preprocess_paired_kernel_sm100<NarrowFormat::FP6_E2M3>
            <<<grid, block, 0, stream>>>(
                interleaved,
                static_cast<uint8_t*>(narrow_out_re), static_cast<uint8_t*>(narrow_out_im),
                static_cast<uint8_t*>(sf_out_re), static_cast<uint8_t*>(sf_out_im),
                M, K, num_k_tiles, narrow_max,
                M_per_batch, sf_stride_batch, data_stride_batch);
        break;
#endif
#ifdef COMPLEX_SM100_ENABLE_FP4
    case ComputePrecision::FP4_E2M1:
        deinterleave_mxfp_preprocess_paired_kernel_sm100<NarrowFormat::FP4_E2M1>
            <<<grid, block, 0, stream>>>(
                interleaved,
                static_cast<uint8_t*>(narrow_out_re), static_cast<uint8_t*>(narrow_out_im),
                static_cast<uint8_t*>(sf_out_re), static_cast<uint8_t*>(sf_out_im),
                M, K, num_k_tiles, narrow_max,
                M_per_batch, sf_stride_batch, data_stride_batch);
        break;
#endif
    default:
        throw std::runtime_error("deinterleave_preprocess_mxfp_paired_sm100: unsupported precision");
    }
    CUDA_CHECK(cudaGetLastError());
}

#endif // COMPLEX_SM100_ENABLE_FP6 || COMPLEX_SM100_ENABLE_FP4


/// Precision-aware transposed cast dispatch. Reads FP16 ColMajor [rows × cols],
/// writes sub-byte ColMajor [cols × rows] (transposed, packed).
inline void cast_fp16_to_narrow_transposed_sm100(
    const __half* input,
    void* output,
    int rows, int cols,
    ComputePrecision precision,
    cudaStream_t stream = nullptr,
    int batch_count = 1)
{
    switch (precision) {
    case ComputePrecision::FP8_E4M3:
        cast_fp16_to_fp8_e4m3_transposed(
            input,
            reinterpret_cast<cutlass::float_e4m3_t*>(output),
            rows, cols, stream, batch_count);
        break;
#ifdef COMPLEX_SM100_ENABLE_FP6
    case ComputePrecision::FP6_E3M2:
        cast_fp16_to_fp6_e3m2_transposed_sm100(input, output, rows, cols, stream, batch_count);
        break;
    case ComputePrecision::FP6_E2M3:
        cast_fp16_to_fp6_e2m3_transposed_sm100(input, output, rows, cols, stream, batch_count);
        break;
#endif
#ifdef COMPLEX_SM100_ENABLE_FP4
    case ComputePrecision::FP4_E2M1:
        cast_fp16_to_fp4_e2m1_transposed_sm100(input, output, rows, cols, stream, batch_count);
        break;
#endif
    default:
        throw std::runtime_error("Unsupported precision for transposed cast");
    }
}


// ========================================================================================
// F1 Dispatch: Deinterleave + Cast (interleaved FP16 → 2× planar sub-byte)
// ========================================================================================

/// F1 linear dispatch: deinterleave interleaved complex FP16 → 2 planar sub-byte outputs
inline void deinterleave_cast_fp16_to_narrow_sm100(
    const __half* interleaved,
    void* out_real, void* out_imag,
    int64_t num_complex_elements,
    ComputePrecision precision,
    cudaStream_t stream = nullptr)
{
    switch (precision) {
    case ComputePrecision::FP8_E4M3:
        deinterleave_cast_fp16_to_fp8(
            interleaved,
            reinterpret_cast<__nv_fp8_e4m3*>(out_real),
            reinterpret_cast<__nv_fp8_e4m3*>(out_imag),
            num_complex_elements, stream);
        break;
#ifdef COMPLEX_SM100_ENABLE_FP6
    case ComputePrecision::FP6_E3M2:
        deinterleave_cast_fp16_to_fp6_e3m2_sm100(
            interleaved, out_real, out_imag, num_complex_elements, stream);
        break;
    case ComputePrecision::FP6_E2M3:
        deinterleave_cast_fp16_to_fp6_e2m3_sm100(
            interleaved, out_real, out_imag, num_complex_elements, stream);
        break;
#endif
#ifdef COMPLEX_SM100_ENABLE_FP4
    case ComputePrecision::FP4_E2M1:
        deinterleave_cast_fp16_to_fp4_e2m1_sm100(
            interleaved, out_real, out_imag, num_complex_elements, stream);
        break;
#endif
    default:
        throw std::runtime_error("Unsupported precision for deinterleave+cast");
    }
}

/// F1 transposed dispatch: deinterleave interleaved complex FP16 ColMajor
/// → 2 transposed planar sub-byte outputs
inline void deinterleave_cast_fp16_to_narrow_transposed_sm100(
    const __half* input,
    void* out_real, void* out_imag,
    int rows, int cols,
    ComputePrecision precision,
    cudaStream_t stream = nullptr,
    int batch_count = 1)
{
    switch (precision) {
    case ComputePrecision::FP8_E4M3:
        deinterleave_cast_fp16_to_fp8_transposed(
            input,
            reinterpret_cast<cutlass::float_e4m3_t*>(out_real),
            reinterpret_cast<cutlass::float_e4m3_t*>(out_imag),
            rows, cols, stream, batch_count);
        break;
#ifdef COMPLEX_SM100_ENABLE_FP6
    case ComputePrecision::FP6_E3M2:
        deinterleave_cast_fp16_to_fp6_e3m2_transposed_sm100(
            input, out_real, out_imag, rows, cols, stream, batch_count);
        break;
    case ComputePrecision::FP6_E2M3:
        deinterleave_cast_fp16_to_fp6_e2m3_transposed_sm100(
            input, out_real, out_imag, rows, cols, stream, batch_count);
        break;
#endif
#ifdef COMPLEX_SM100_ENABLE_FP4
    case ComputePrecision::FP4_E2M1:
        deinterleave_cast_fp16_to_fp4_e2m1_transposed_sm100(
            input, out_real, out_imag, rows, cols, stream, batch_count);
        break;
#endif
    default:
        throw std::runtime_error("Unsupported precision for transposed deinterleave+cast");
    }
}


// ========================================================================================
// F2 Dispatch: Dual-Output Cast (planar FP16 → 2× identical sub-byte outputs)
// ========================================================================================

/// F2 linear dispatch: 1 FP16 input → 2 identical sub-byte outputs
inline void cast_fp16_to_narrow_dual_sm100(
    const __half* input,
    void* output1, void* output2,
    int64_t num_elements,
    ComputePrecision precision,
    cudaStream_t stream = nullptr)
{
    switch (precision) {
    case ComputePrecision::FP8_E4M3:
        cast_fp16_to_fp8_e4m3_dual(
            input,
            reinterpret_cast<__nv_fp8_e4m3*>(output1),
            reinterpret_cast<__nv_fp8_e4m3*>(output2),
            num_elements, stream);
        break;
#ifdef COMPLEX_SM100_ENABLE_FP6
    case ComputePrecision::FP6_E3M2:
        cast_fp16_to_fp6_e3m2_dual_sm100(input, output1, output2, num_elements, stream);
        break;
    case ComputePrecision::FP6_E2M3:
        cast_fp16_to_fp6_e2m3_dual_sm100(input, output1, output2, num_elements, stream);
        break;
#endif
#ifdef COMPLEX_SM100_ENABLE_FP4
    case ComputePrecision::FP4_E2M1:
        cast_fp16_to_fp4_e2m1_dual_sm100(input, output1, output2, num_elements, stream);
        break;
#endif
    default:
        throw std::runtime_error("Unsupported precision for dual cast");
    }
}

/// F2 transposed dispatch: 1 FP16 input → 2 identical transposed sub-byte outputs
inline void cast_fp16_to_narrow_transposed_dual_sm100(
    const __half* input,
    void* output1, void* output2,
    int rows, int cols,
    ComputePrecision precision,
    cudaStream_t stream = nullptr,
    int batch_count = 1)
{
    switch (precision) {
    case ComputePrecision::FP8_E4M3:
        cast_fp16_to_fp8_e4m3_transposed_dual(
            input,
            reinterpret_cast<cutlass::float_e4m3_t*>(output1),
            reinterpret_cast<cutlass::float_e4m3_t*>(output2),
            rows, cols, stream, batch_count);
        break;
#ifdef COMPLEX_SM100_ENABLE_FP6
    case ComputePrecision::FP6_E3M2:
        cast_fp16_to_fp6_e3m2_transposed_dual_sm100(
            input, output1, output2, rows, cols, stream, batch_count);
        break;
    case ComputePrecision::FP6_E2M3:
        cast_fp16_to_fp6_e2m3_transposed_dual_sm100(
            input, output1, output2, rows, cols, stream, batch_count);
        break;
#endif
#ifdef COMPLEX_SM100_ENABLE_FP4
    case ComputePrecision::FP4_E2M1:
        cast_fp16_to_fp4_e2m1_transposed_dual_sm100(
            input, output1, output2, rows, cols, stream, batch_count);
        break;
#endif
    default:
        throw std::runtime_error("Unsupported precision for transposed dual cast");
    }
}


// ========================================================================================
// F5 Dispatch: Paired Cast (2 independent FP16 inputs → 2 independent sub-byte outputs)
// ========================================================================================

/// F5 linear dispatch: 2 FP16 inputs → 2 independent sub-byte outputs
inline void cast_fp16_to_narrow_paired_sm100(
    const __half* input1, const __half* input2,
    void* output1, void* output2,
    int64_t num_elements,
    ComputePrecision precision,
    cudaStream_t stream = nullptr)
{
    switch (precision) {
    case ComputePrecision::FP8_E4M3:
        cast_fp16_to_fp8_e4m3_paired_sm100(
            input1, input2,
            reinterpret_cast<__nv_fp8_e4m3*>(output1),
            reinterpret_cast<__nv_fp8_e4m3*>(output2),
            num_elements, stream);
        break;
#ifdef COMPLEX_SM100_ENABLE_FP6
    case ComputePrecision::FP6_E3M2:
        cast_fp16_to_fp6_e3m2_paired_sm100(
            input1, input2, output1, output2, num_elements, stream);
        break;
    case ComputePrecision::FP6_E2M3:
        cast_fp16_to_fp6_e2m3_paired_sm100(
            input1, input2, output1, output2, num_elements, stream);
        break;
#endif
#ifdef COMPLEX_SM100_ENABLE_FP4
    case ComputePrecision::FP4_E2M1:
        cast_fp16_to_fp4_e2m1_paired_sm100(
            input1, input2, output1, output2, num_elements, stream);
        break;
#endif
    default:
        throw std::runtime_error("Unsupported precision for paired cast");
    }
}

/// F5 transposed dispatch: 2 FP16 inputs → 2 independent transposed sub-byte outputs
inline void cast_fp16_to_narrow_transposed_paired_sm100(
    const __half* input1, const __half* input2,
    void* output1, void* output2,
    int rows, int cols,
    ComputePrecision precision,
    cudaStream_t stream = nullptr,
    int batch_count = 1)
{
    switch (precision) {
    case ComputePrecision::FP8_E4M3:
        cast_fp16_to_fp8_e4m3_transposed_paired_sm100(
            input1, input2,
            reinterpret_cast<cutlass::float_e4m3_t*>(output1),
            reinterpret_cast<cutlass::float_e4m3_t*>(output2),
            rows, cols, stream, batch_count);
        break;
#ifdef COMPLEX_SM100_ENABLE_FP6
    case ComputePrecision::FP6_E3M2:
        cast_fp16_to_fp6_e3m2_transposed_paired_sm100(
            input1, input2, output1, output2, rows, cols, stream, batch_count);
        break;
    case ComputePrecision::FP6_E2M3:
        cast_fp16_to_fp6_e2m3_transposed_paired_sm100(
            input1, input2, output1, output2, rows, cols, stream, batch_count);
        break;
#endif
#ifdef COMPLEX_SM100_ENABLE_FP4
    case ComputePrecision::FP4_E2M1:
        cast_fp16_to_fp4_e2m1_transposed_paired_sm100(
            input1, input2, output1, output2, rows, cols, stream, batch_count);
        break;
#endif
    default:
        throw std::runtime_error("Unsupported precision for transposed paired cast");
    }
}
