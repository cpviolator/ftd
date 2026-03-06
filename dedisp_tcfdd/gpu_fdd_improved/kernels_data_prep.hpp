// --- kernels_data_prep.hpp ---
// Data prep, phasor prep, and post-GEMM finalize kernels.
// Textual include — no #pragma once.

/**
 * @brief Optimized Transpose/Quantize Kernel for FP8 Hopper Layout.
 * Pure Transpose: [Batch, Nf, Nt_complex] -> [Nt_complex, Batch, Nf]
 * Uses Shared Memory Tiling for Coalesced Reads and Writes.
 */
template <typename T_OUT>
__global__ void kernel_fused_data_prep_fp8_opt(
    const cufftComplex* __restrict__ input, // [Batch, Nf, Nt_complex]
    T_OUT* __restrict__ out_real,           // [Nt_complex, Batch, Nf]
    T_OUT* __restrict__ out_imag,           // [Nt_complex, Batch, Nf]
    int batch_size, int Nf, int Nt_complex,
    float scale)
{
    // Shared memory for 32x32 tile of Complex (2 floats)
    // Padded to [32][33] to avoid bank conflicts
    __shared__ float2 tile[32][33];

    int b = blockIdx.z;

    // --- PHASE 1: COALESCED READ ---
    int k_in = blockIdx.y * 32 + threadIdx.x;
    int f_in = blockIdx.x * 32 + threadIdx.y;

    if (k_in < Nt_complex && f_in < Nf && b < batch_size) {
        size_t in_idx = (size_t)b * Nf * Nt_complex + (size_t)f_in * Nt_complex + k_in;
        cufftComplex val = input[in_idx];
        tile[threadIdx.y][threadIdx.x] = make_float2(val.x, val.y);
    }

    __syncthreads();

    // --- PHASE 2: COALESCED WRITE ---
    int f_out = blockIdx.x * 32 + threadIdx.x;
    int k_out = blockIdx.y * 32 + threadIdx.y;

    if (f_out < Nf && k_out < Nt_complex && b < batch_size) {
        float2 val = tile[threadIdx.x][threadIdx.y];

        float r = val.x * scale;
        float i = val.y * scale;

        // Output Addr: k * (Batch * Nf) + b * Nf + f
        size_t plane_offset = (size_t)k_out * batch_size * Nf;
        size_t out_idx = plane_offset + (size_t)b * Nf + f_out;

        // Saturate and Cast to FP8
        r = fminf(fmaxf(r, -448.0f), 448.0f);
        i = fminf(fmaxf(i, -448.0f), 448.0f);

        out_real[out_idx] = T_OUT(r);
        out_imag[out_idx] = T_OUT(i);
    }
}


/**
 * @brief 64×65 tile variant of kernel_fused_data_prep_fp8_opt.
 * Same transpose [Batch, Nf, Nt_complex] → [Nt_complex, Batch, Nf] with scale+cast,
 * but processes a 64×64 tile per block (4× fewer blocks, 4 elements/thread).
 * Block: (64, 16) = 1024 threads. SMEM: 64×65×8 = 33 KB.
 */
template <typename T_OUT>
__global__ void kernel_fused_data_prep_fp8_opt_64(
    const cufftComplex* __restrict__ input, // [Batch, Nf, Nt_complex]
    T_OUT* __restrict__ out_real,           // [Nt_complex, Batch, Nf]
    T_OUT* __restrict__ out_imag,           // [Nt_complex, Batch, Nf]
    int batch_size, int Nf, int Nt_complex,
    float scale)
{
    __shared__ float2 tile[64][65]; // Padded +1 to avoid bank conflicts

    int b = blockIdx.z;
    int k_base = blockIdx.y * 64;
    int f_base = blockIdx.x * 64;

    // --- PHASE 1: COALESCED READ (4 iterations over f) ---
    // tx (0..63) → k dimension (stride-1 in input), ty (0..15) → f, 4 iterations
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        int k_in = k_base + threadIdx.x;
        int f_in = f_base + threadIdx.y + i * 16;

        if (k_in < Nt_complex && f_in < Nf && b < batch_size) {
            size_t in_idx = (size_t)b * Nf * Nt_complex + (size_t)f_in * Nt_complex + k_in;
            cufftComplex val = input[in_idx];
            tile[threadIdx.y + i * 16][threadIdx.x] = make_float2(val.x, val.y);
        }
    }

    __syncthreads();

    // --- PHASE 2: COALESCED WRITE (4 iterations over k, transposed) ---
    // tx (0..63) → f dimension (stride-1 in output), ty (0..15) → k, 4 iterations
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        int f_out = f_base + threadIdx.x;
        int k_out = k_base + threadIdx.y + i * 16;

        if (f_out < Nf && k_out < Nt_complex && b < batch_size) {
            float2 val = tile[threadIdx.x][threadIdx.y + i * 16];

            float r = val.x * scale;
            float im = val.y * scale;

            size_t plane_offset = (size_t)k_out * batch_size * Nf;
            size_t out_idx = plane_offset + (size_t)b * Nf + f_out;

            r = fminf(fmaxf(r, -448.0f), 448.0f);
            im = fminf(fmaxf(im, -448.0f), 448.0f);

            out_real[out_idx] = T_OUT(r);
            out_imag[out_idx] = T_OUT(im);
        }
    }
}


template <typename T_OUT>
__global__ void kernel_fused_data_prep(
    const cufftComplex* __restrict__ input, // [Batch, Nf, Nt_complex]
    T_OUT* __restrict__ out_real,           // [Nt_complex, Nf, Batch]
    T_OUT* __restrict__ out_imag,           // [Nt_complex, Nf, Batch]
    int batch_size, int Nf, int Nt_complex,
    float scale)
{
    // Tile dimensions: 32x32. Padding +1 to avoid shared mem bank conflicts.
    __shared__ float tile_r[32][33];
    __shared__ float tile_i[32][33];

    // Grid Layout:
    // x -> k (Time)
    // y -> b (Batch)
    // z -> f (Frequency) - handled as separate blocks

    int k_origin = blockIdx.x * 32;
    int b_origin = blockIdx.y * 32;
    int f = blockIdx.z;

    // --- 1. COALESCED READ (Load into Shared Mem) ---
    // Map threadIdx.x to k (contiguous in Input)
    int k_in = k_origin + threadIdx.x;
    int b_in = b_origin + threadIdx.y;

    if (k_in < Nt_complex && b_in < batch_size) {
        // Input Index: [b][f][k] -> Stride is Nt_complex
        size_t in_idx = (size_t)b_in * Nf * Nt_complex + (size_t)f * Nt_complex + k_in;
        cufftComplex val = input[in_idx];

        // Store transposed in shared memory: tile[y][x]
        tile_r[threadIdx.y][threadIdx.x] = val.x;
        tile_i[threadIdx.y][threadIdx.x] = val.y;
    }

    __syncthreads();

    // --- 2. COALESCED WRITE (Store from Shared Mem) ---
    // Swap roles: threadIdx.x now handles Batch (usually contiguous in Output for FP32/16)
    int b_out = b_origin + threadIdx.x;
    int k_out = k_origin + threadIdx.y;

    if (b_out < batch_size && k_out < Nt_complex) {
        // Read from shared memory transposed: tile[x][y]
        float r = tile_r[threadIdx.x][threadIdx.y] * scale;
        float i = tile_i[threadIdx.x][threadIdx.y] * scale;

        // --- HOPPER FP8 LAYOUT CHANGE ---
        // FP8 on Hopper prefers 'A' to be Transposed (Row-Major).
        // Row-Major 'A' (M x K) means strides are [K, 1].
        // Memory Layout: b * Nf + f.

        size_t out_idx;

        if constexpr (IsFP8<T_OUT>::value) {
            // FP8: Store as [Batch, Nf] (Row-Major / Transposed A)
            // Stride between Time(k) planes is Batch * Nf
            size_t plane_offset = (size_t)k_out * Nf * batch_size;
            out_idx = plane_offset + (size_t)b_out * Nf + f;

            // Saturation logic for e4m3
            r = fminf(fmaxf(r, -448.0f), 448.0f);
            i = fminf(fmaxf(i, -448.0f), 448.0f);

            out_real[out_idx] = T_OUT(r);
            out_imag[out_idx] = T_OUT(i);
        } else {
            // FP16/FP32: Store as [Nf, Batch] (Col-Major A)
            // Stride between Time(k) planes is Batch * Nf
            size_t plane_offset = (size_t)k_out * Nf * batch_size;
            out_idx = plane_offset + (size_t)f * batch_size + b_out;

            out_real[out_idx] = (T_OUT)(r);
            out_imag[out_idx] = (T_OUT)(i);
        }
    }
}

// Kernel 2: Transpose [k, dm, f] (Interleaved) -> [k, f, dm] (Planar Col-Major Phasor)
// This prepares Matrix B (Nf x Ndm) for GEMM.
// Col-Major 'B' means stride-1 is Nf.
template <typename T_OUT>
__global__ void kernel_fused_phasor_prep(
    const cufftComplex* __restrict__ input, // [k, dm, f]
    T_OUT* __restrict__ out_real,
    T_OUT* __restrict__ out_imag,
    int Nf, int Ndm, int Nt_complex,
    float scale)
{
    int f = blockIdx.x * blockDim.x + threadIdx.x;
    int dm = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z;

    if (f < Nf && dm < Ndm && k < Nt_complex) {
        // Input Index: [k, dm, f]
        size_t in_idx = (size_t)k * Ndm * Nf + (size_t)dm * Nf + f;

        // Output Index (Col Major B): [k, f, dm]
        // For a specific 'k', matrix is [Nf, Ndm]. Stride 1 is f.
        size_t plane_offset = (size_t)k * Ndm * Nf;
        size_t out_idx = plane_offset + (size_t)dm * Nf + f;

	cufftComplex val = input[in_idx];
        float r = val.x * scale;
        float i = val.y * scale;

        if constexpr (IsFP8<T_OUT>::value) {
            // Saturate for FP8
            r = fminf(fmaxf(r, -448.0f), 448.0f);
            i = fminf(fmaxf(i, -448.0f), 448.0f);
            out_real[out_idx] = T_OUT(r);
            out_imag[out_idx] = T_OUT(i);
        } else {
            out_real[out_idx] = (T_OUT)(r);
            out_imag[out_idx] = (T_OUT)(i);
        }
    }
}



#ifdef HAS_CUTLASS_GEMM
// Kernel: Data Prep for CUTLASS path (32×33 tile, legacy)
// Converts [Batch, Nf, Nt_complex] (interleaved complex) -> [Nt_complex, Batch, Nf] (planar row-major A)
// Applies scale factor so data fits in FP8 E4M3 range (±448) after CUTLASS's internal FP16->FP8 cast.
// Uses 32x32 SMEM tile transpose for coalesced memory access.
__global__ void kernel_fused_data_prep_rowmajor(
    const cufftComplex* __restrict__ input,   // [batch_size, Nf, Nt_complex]
    __half* __restrict__ out_real,             // [Nt_complex, batch_size, Nf]
    __half* __restrict__ out_imag,             // [Nt_complex, batch_size, Nf]
    int batch_size, int Nf, int Nt_complex,
    float scale)
{
    // Same tile transpose pattern as kernel_fused_data_prep_fp8_opt
    __shared__ float2 tile[32][33]; // Padded to avoid bank conflicts

    int b = blockIdx.z;

    // PHASE 1: COALESCED READ -- threadIdx.x maps to k (contiguous in input)
    int k_in = blockIdx.y * 32 + threadIdx.x;
    int f_in = blockIdx.x * 32 + threadIdx.y;

    if (k_in < Nt_complex && f_in < Nf && b < batch_size) {
        size_t in_idx = (size_t)b * Nf * Nt_complex + (size_t)f_in * Nt_complex + k_in;
        cufftComplex val = input[in_idx];
        tile[threadIdx.y][threadIdx.x] = make_float2(val.x * scale, val.y * scale);
    }

    __syncthreads();

    // PHASE 2: COALESCED WRITE -- threadIdx.x maps to f (contiguous in output)
    int f_out = blockIdx.x * 32 + threadIdx.x;
    int k_out = blockIdx.y * 32 + threadIdx.y;

    if (f_out < Nf && k_out < Nt_complex && b < batch_size) {
        float2 val = tile[threadIdx.x][threadIdx.y];

        // Output: [k][b][f] row-major A with K=Nf stride-1
        size_t out_idx = (size_t)k_out * batch_size * Nf + (size_t)b * Nf + f_out;

        out_real[out_idx] = __float2half(val.x);
        out_imag[out_idx] = __float2half(val.y);
    }
}

/**
 * @brief 64×65 tile variant of kernel_fused_data_prep_rowmajor.
 * Same transpose [Batch, Nf, Nt_complex] → [Nt_complex, Batch, Nf] with scale + FP16 cast.
 * Block: (64, 16) = 1024 threads, 4 elements/thread. SMEM: 64×65×8 = 33 KB.
 */
__global__ void kernel_fused_data_prep_rowmajor_64(
    const cufftComplex* __restrict__ input,   // [batch_size, Nf, Nt_complex]
    __half* __restrict__ out_real,             // [Nt_complex, batch_size, Nf]
    __half* __restrict__ out_imag,             // [Nt_complex, batch_size, Nf]
    int batch_size, int Nf, int Nt_complex,
    float scale)
{
    __shared__ float2 tile[64][65];

    int b = blockIdx.z;
    int k_base = blockIdx.y * 64;
    int f_base = blockIdx.x * 64;

    #pragma unroll
    for (int i = 0; i < 4; i++) {
        int k_in = k_base + threadIdx.x;
        int f_in = f_base + threadIdx.y + i * 16;

        if (k_in < Nt_complex && f_in < Nf && b < batch_size) {
            size_t in_idx = (size_t)b * Nf * Nt_complex + (size_t)f_in * Nt_complex + k_in;
            cufftComplex val = input[in_idx];
            tile[threadIdx.y + i * 16][threadIdx.x] = make_float2(val.x * scale, val.y * scale);
        }
    }

    __syncthreads();

    #pragma unroll
    for (int i = 0; i < 4; i++) {
        int f_out = f_base + threadIdx.x;
        int k_out = k_base + threadIdx.y + i * 16;

        if (f_out < Nf && k_out < Nt_complex && b < batch_size) {
            float2 val = tile[threadIdx.x][threadIdx.y + i * 16];

            size_t out_idx = (size_t)k_out * batch_size * Nf + (size_t)b * Nf + f_out;

            out_real[out_idx] = __float2half(val.x);
            out_imag[out_idx] = __float2half(val.y);
        }
    }
}

// Kernel: Phasor Prep for CUTLASS path
// Converts [k, dm, f] (interleaved complex) -> [k, dm, f] (planar FP16)
// Applies scale factor so phasors utilise FP8 E4M3 dynamic range after CUTLASS's internal cast.
// Output layout: B[N x K] row-major = B[K x N] col-major (CUTLASS TN convention)
// where N=Ndm, K=Nf. Memory: dm * Nf + f (f stride-1).
__global__ void kernel_fused_phasor_prep_rowmajor(
    const cufftComplex* __restrict__ input,   // [Nt_complex, Ndm, Nf]
    __half* __restrict__ out_real,             // [Nt_complex, Ndm, Nf]
    __half* __restrict__ out_imag,             // [Nt_complex, Ndm, Nf]
    int Nf, int Ndm, int Nt_complex,
    float scale)
{
    int f = blockIdx.x * blockDim.x + threadIdx.x;
    int dm = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z;

    if (f < Nf && dm < Ndm && k < Nt_complex) {
        // Input: [k, dm, f] -- same layout as output
        size_t idx = (size_t)k * Ndm * Nf + (size_t)dm * Nf + f;

        cufftComplex val = input[idx];
        out_real[idx] = __float2half(val.x * scale);
        out_imag[idx] = __float2half(val.y * scale);
    }
}

// Kernel: Post-GEMM Finalize for CUTLASS path (legacy, NO SMEM tiling)
// Converts [k, b, dm] (planar FP32 row-major C) -> [b, dm, k] (interleaved cufftComplex)
// C layout from CUTLASS: row-major C[M x N] with N stride-1 (M=batch_size, N=Ndm)
// WARNING: Both reads and writes are strided — use kernel_post_gemm_finalize_cutlass_tiled instead.
__global__ void kernel_post_gemm_finalize_cutlass(
    const float* __restrict__ c_real,
    const float* __restrict__ c_imag,
    cufftComplex* __restrict__ output,
    int batch_size, int Ndm, int Nt_complex)
{
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    int dm = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z;

    if (b < batch_size && dm < Ndm && k < Nt_complex) {
        // Input: [k][b][dm] row-major C (N=Ndm stride-1)
        size_t in_idx = (size_t)k * batch_size * Ndm + (size_t)b * Ndm + dm;
        // Output: [b][dm][k] interleaved complex for IFFT
        size_t out_idx = (size_t)b * Ndm * Nt_complex + (size_t)dm * Nt_complex + k;

        output[out_idx].x = c_real[in_idx];
        output[out_idx].y = c_imag[in_idx];
    }
}

/**
 * @brief SMEM-tiled Post-GEMM Finalize for CUTLASS path.
 * Transposes [k, b, dm] (planar FP32) → [b, dm, k] (interleaved cufftComplex).
 *
 * Tiles over (dm, k) with 32×33 SMEM buffers.
 * - Read:  tx → dm (stride-1 in input [k][b][dm]) → COALESCED
 * - Write: tx → k  (stride-1 in output [b][dm][k]) → COALESCED
 * Grid: ((Ndm+31)/32, (Nt_complex+31)/32, batch_size), block: (32, 32).
 */
__global__ void kernel_post_gemm_finalize_cutlass_tiled(
    const float* __restrict__ c_real,
    const float* __restrict__ c_imag,
    cufftComplex* __restrict__ output,
    int batch_size, int Ndm, int Nt_complex)
{
    __shared__ float tile_r[32][33];
    __shared__ float tile_i[32][33];

    int b = blockIdx.z;
    int dm_base = blockIdx.x * 32;
    int k_base = blockIdx.y * 32;

    // Phase 1: Coalesced read — tx maps to dm (stride-1 in input)
    int dm_in = dm_base + threadIdx.x;
    int k_in = k_base + threadIdx.y;

    if (dm_in < Ndm && k_in < Nt_complex && b < batch_size) {
        size_t in_idx = (size_t)k_in * batch_size * Ndm + (size_t)b * Ndm + dm_in;
        tile_r[threadIdx.y][threadIdx.x] = c_real[in_idx];
        tile_i[threadIdx.y][threadIdx.x] = c_imag[in_idx];
    }

    __syncthreads();

    // Phase 2: Coalesced write — tx maps to k (stride-1 in output)
    int k_out = k_base + threadIdx.x;
    int dm_out = dm_base + threadIdx.y;

    if (k_out < Nt_complex && dm_out < Ndm && b < batch_size) {
        float r = tile_r[threadIdx.x][threadIdx.y];
        float im = tile_i[threadIdx.x][threadIdx.y];

        size_t out_idx = (size_t)b * Ndm * Nt_complex + (size_t)dm_out * Nt_complex + k_out;
        output[out_idx].x = r;
        output[out_idx].y = im;
    }
}
#ifdef HAS_CUFFT_LTO_CALLBACK
/**
 * @brief Transpose FP32-planar -> FP16-planar with device-side scaling.
 * Input:  d_re/d_im in [batch, Nf, Nt_complex] layout (FP32, from cuFFT callback)
 * Output: out_re/out_im in [Nt_complex, batch, Nf] layout (FP16, for CUTLASS)
 * Reads max from d_max_uint (IEEE float-as-uint from atomicMax).
 * Block: (64, 16) = 1024 threads, processes 64x64 tile, SMEM: 64x65x4 = 16.6 KB x 2.
 */
__global__ void kernel_transpose_scale_fp16(
    const float* __restrict__ d_re,       // [batch, Nf, Nt_complex] FP32
    const float* __restrict__ d_im,       // [batch, Nf, Nt_complex] FP32
    const unsigned int* __restrict__ d_max_uint,
    __half* __restrict__ out_re,          // [Nt_complex, batch, Nf] FP16
    __half* __restrict__ out_im,          // [Nt_complex, batch, Nf] FP16
    int batch_size, int Nf, int Nt_complex)
{
    __shared__ float tile_re[64][65];  // +1 padding for bank conflicts
    __shared__ float tile_im[64][65];

    // Compute scale from device-side max (all threads read the same value, L1 cached)
    float max_val = __uint_as_float(*d_max_uint);
    if (max_val < 1e-9f) max_val = 1.0f;
    float scale = 448.0f / max_val;

    int b = blockIdx.z;
    int k_base = blockIdx.y * 64;
    int f_base = blockIdx.x * 64;

    // Phase 1: Coalesced read from [batch, Nf, Nt_complex]
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        int k_in = k_base + threadIdx.x;
        int f_in = f_base + threadIdx.y + i * 16;
        if (k_in < Nt_complex && f_in < Nf && b < batch_size) {
            size_t idx = (size_t)b * Nf * Nt_complex + (size_t)f_in * Nt_complex + k_in;
            tile_re[threadIdx.y + i * 16][threadIdx.x] = d_re[idx];
            tile_im[threadIdx.y + i * 16][threadIdx.x] = d_im[idx];
        }
    }

    __syncthreads();

    // Phase 2: Coalesced write to [Nt_complex, batch, Nf] as FP16
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        int f_out = f_base + threadIdx.x;
        int k_out = k_base + threadIdx.y + i * 16;
        if (f_out < Nf && k_out < Nt_complex && b < batch_size) {
            float re = tile_re[threadIdx.x][threadIdx.y + i * 16] * scale;
            float im = tile_im[threadIdx.x][threadIdx.y + i * 16] * scale;

            size_t out_idx = (size_t)k_out * batch_size * Nf + (size_t)b * Nf + f_out;
            out_re[out_idx] = __float2half(fminf(fmaxf(re, -448.0f), 448.0f));
            out_im[out_idx] = __float2half(fminf(fmaxf(im, -448.0f), 448.0f));
        }
    }
}
#endif // HAS_CUFFT_LTO_CALLBACK

#endif // HAS_CUTLASS_GEMM

/**
 * @brief Generate phasors on-the-fly for a tile of Nt_complex, writing FP16 planar output directly.
 *
 * Computes exp(2πi * f_k * time_delay) for k in [k_offset, k_offset+Nt_tile),
 * writing cos/sin as scaled __half into B_re/B_im in [Nt_tile, Ndm, Nf] layout.
 * Avoids materializing the full cufftComplex phasor table (419 GB at production sizes).
 *
 * Grid: ((Nf+31)/32, (Ndm+31)/32, Nt_tile), Block: (32, 32)
 */
__global__ void kernel_generate_phasors_tiled_fp16(
    const float* __restrict__ d_f_k_values,   // [Nt_complex]
    const float* __restrict__ d_time_delays,   // [Ndm, Nf]
    __half* __restrict__ B_re,                 // [Nt_tile, Ndm, Nf]
    __half* __restrict__ B_im,                 // [Nt_tile, Ndm, Nf]
    int Nf, int Ndm, int Nt_tile, int k_offset,
    float scale, bool use_conjugate)
{
    int f = blockIdx.x * blockDim.x + threadIdx.x;
    int dm = blockIdx.y * blockDim.y + threadIdx.y;
    int k_local = blockIdx.z;

    if (f >= Nf || dm >= Ndm || k_local >= Nt_tile) return;

    double f_k_val = (double)d_f_k_values[k_offset + k_local];
    double time_delay = (double)d_time_delays[dm * Nf + f];
    double angle = 2.0 * 3.14159265358979323846 * f_k_val * time_delay;

    double s, c;
    sincos(angle, &s, &c);

    // Output: [k_local, dm, f] — same layout as kernel_fused_phasor_prep_rowmajor
    size_t out_idx = (size_t)k_local * Ndm * Nf + (size_t)dm * Nf + f;
    B_re[out_idx] = __float2half((float)(c * scale));
    // use_conjugate matches kernel_generate_phasors sign convention:
    // conjugate=true → +sin, conjugate=false → -sin
    B_im[out_idx] = __float2half((float)(use_conjugate ? s * scale : -s * scale));
}

// Kernel 3 (legacy): Merge [k, dm, b] (Planar) -> [b, dm, k] (Interleaved Result)
// De-quantizes and transposes for IFFT.
// WARNING: Writes are strided — use kernel_post_gemm_finalize_tiled instead.
template <typename T_IN>
__global__ void kernel_post_gemm_finalize(
    const float* __restrict__ c_real,
    const float* __restrict__ c_imag,
    cufftComplex* __restrict__ output,
    int batch_size, int Ndm, int Nt_complex,
    float unscale)
{
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    int dm = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z;

    if (b < batch_size && dm < Ndm && k < Nt_complex) {
        // Input Index (Col Major C): [k, dm, b]
        // For a specific 'k', matrix is [Batch, Ndm]. Stride 1 is b.
        size_t plane_offset = (size_t)k * Ndm * batch_size;
        size_t in_idx = plane_offset + (size_t)dm * batch_size + b;

        float re_val = float(c_real[in_idx]);
        float im_val = float(c_imag[in_idx]);

        // Output Index: [b, dm, k]
        size_t out_idx = (size_t)b * Ndm * Nt_complex + (size_t)dm * Nt_complex + k;

        output[out_idx].x = re_val * unscale;
        output[out_idx].y = im_val * unscale;
    }
}

/**
 * @brief SMEM-tiled Post-GEMM Finalize for cuBLASLt path.
 * Transposes [k, dm, b] (planar FP32 col-major) → [b, dm, k] (interleaved cufftComplex).
 * Applies unscale factor.
 *
 * Tiles over (b, k) with 32×33 SMEM buffers.
 * - Read:  tx → b (stride-1 in input [k][dm][b]) → COALESCED
 * - Write: tx → k (stride-1 in output [b][dm][k]) → COALESCED
 * Grid: ((batch+31)/32, (Nt_complex+31)/32, Ndm), block: (32, 32).
 */
template <typename T_IN>
__global__ void kernel_post_gemm_finalize_tiled(
    const float* __restrict__ c_real,
    const float* __restrict__ c_imag,
    cufftComplex* __restrict__ output,
    int batch_size, int Ndm, int Nt_complex,
    float unscale)
{
    __shared__ float tile_r[32][33];
    __shared__ float tile_i[32][33];

    int dm = blockIdx.z;
    int b_base = blockIdx.x * 32;
    int k_base = blockIdx.y * 32;

    // Phase 1: Coalesced read — tx maps to b (stride-1 in input [k][dm][b])
    int b_in = b_base + threadIdx.x;
    int k_in = k_base + threadIdx.y;

    if (b_in < batch_size && k_in < Nt_complex && dm < Ndm) {
        size_t in_idx = (size_t)k_in * Ndm * batch_size + (size_t)dm * batch_size + b_in;
        tile_r[threadIdx.y][threadIdx.x] = float(c_real[in_idx]);
        tile_i[threadIdx.y][threadIdx.x] = float(c_imag[in_idx]);
    }

    __syncthreads();

    // Phase 2: Coalesced write — tx maps to k (stride-1 in output [b][dm][k])
    int k_out = k_base + threadIdx.x;
    int b_out = b_base + threadIdx.y;

    if (k_out < Nt_complex && b_out < batch_size && dm < Ndm) {
        float r = tile_r[threadIdx.x][threadIdx.y] * unscale;
        float im = tile_i[threadIdx.x][threadIdx.y] * unscale;

        size_t out_idx = (size_t)b_out * Ndm * Nt_complex + (size_t)dm * Nt_complex + k_out;
        output[out_idx].x = r;
        output[out_idx].y = im;
    }
}
