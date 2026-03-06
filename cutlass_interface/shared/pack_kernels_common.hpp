// ========================================================================================
// Shared Pack/Triangle Kernels — Common between SM90 and SM100/SM120
// ========================================================================================
//
// Contains the 5 kernels + wrappers that are identical between architectures:
//   1. enforce_hermitian_kernel + enforce_hermitian_triangle()
//   2. antisymmetrize_to_triangle_kernel + antisymmetrize_to_triangle()
//   3. pack_triangle_kernel + pack_triangle()
//   4. antisymmetrize_pack_kernel + antisymmetrize_pack()
//   5. pack_antisymmetrize_triangle_kernel + pack_antisymmetrize_triangle()
//
// Also includes the lightweight enforce_real_diagonal_kernel.
//
// This header is #include'd from within each architecture's pack_kernels.hpp.

#pragma once

// ========================================================================================
// Triangular Enforcement Kernel (for HERK)
// ========================================================================================

/*
 * After computing the full Gram matrix (A·A^H or A^H·A), HERK requires:
 *   1. Force diagonal elements to be real: Im(C_ii) = 0
 *   2. Enforce exact Hermitian symmetry: C(j,i) = conj(C(i,j))
 *
 * Because FP8 arithmetic is not strictly associative (rounding depends on operand
 * order), the computed Ar·Ar^T is not bitwise equal to (Ar·Ar^T)^T. This kernel
 * picks one triangle as authoritative and copies it to the other with conjugation,
 * guaranteeing exact Hermitian symmetry in the output.
 *
 * For FillMode::Upper: upper triangle (j >= i) is authoritative → copied to lower
 * For FillMode::Lower: lower triangle (i >= j) is authoritative → copied to upper
 * For FillMode::Full:  lower triangle is authoritative (arbitrary choice) → copied to upper
 *
 * O(N²) memory pass — negligible vs the O(N²K) GEMM. Batched via blockIdx.z.
 */

__global__ void enforce_hermitian_kernel(
    __half* __restrict__ C_real,
    __half* __restrict__ C_imag,
    int N,
    int64_t batch_stride,
    int fill_mode)              // 0 = Upper (upper→lower), 1 = Lower (lower→upper)
{
    const int batch = blockIdx.z;
    __half* Cr = C_real + batch * batch_stride;
    __half* Ci = C_imag + batch * batch_stride;

    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row >= N || col >= N) return;

    int64_t idx = static_cast<int64_t>(row) * N + col;  // RowMajor

    if (row == col) {
        // Diagonal: force real (Im = 0)
        Ci[idx] = __float2half(0.0f);
    } else if (fill_mode == 0 && row > col) {
        // Upper authoritative: copy C(col,row) → C(row,col) with conjugation
        int64_t src = static_cast<int64_t>(col) * N + row;  // (col, row) in RowMajor
        Cr[idx] = Cr[src];            // Re(C(row,col)) = Re(C(col,row))
        Ci[idx] = __hneg(Ci[src]);    // Im(C(row,col)) = -Im(C(col,row))
    } else if (fill_mode == 1 && col > row) {
        // Lower authoritative: copy C(col,row) → C(row,col) with conjugation
        int64_t src = static_cast<int64_t>(col) * N + row;
        Cr[idx] = Cr[src];
        Ci[idx] = __hneg(Ci[src]);
    }
}

/// Launch helper for Hermitian enforcement (symmetrize + zero imag diagonal)
inline void enforce_hermitian_triangle(
    __half* C_real, __half* C_imag,
    int N, FillMode fill,
    cudaStream_t stream = nullptr,
    int batch_count = 1)
{
    int64_t batch_stride = static_cast<int64_t>(N) * N;
    // Full mode uses Lower as authoritative (arbitrary — both computed by GEMM)
    int mode_int = (fill == FillMode::Upper) ? 0 : 1;
    TUNED_LAUNCH_2D(enforce_hermitian_kernel, "enforce_hermitian",
        N, N, batch_count, static_cast<int64_t>(N) * N * batch_count * 4, stream,
        C_real, C_imag, N, batch_stride, mode_int);
    CUDA_CHECK(cudaGetLastError());
}

// ========================================================================================
// HERK Production Kernels — Triangle-Only Output
// ========================================================================================
//
// These kernels are used in production mode (COMPLEX_FP8_HERK_FULL_MATRIX=0) to:
//   1. Replace the 4th sub-GEMM with an O(N²) anti-symmetrize that writes only
//      the authoritative triangle of Im(C). This saves one full N×N×K GEMM.
//   2. Zero the imaginary diagonal without touching the rest of the matrix.
//

/*
 * Anti-symmetrize kernel for the imaginary part of HERK output.
 *
 * Given temp = Xi·Xr^T (a full N×N matrix from a single sub-GEMM), we need:
 *   Im(C)[i,j] = α·(temp[i,j] − temp[j,i]) + β·Im(C_old)[i,j]
 *
 * This exploits the fact that Im(C) = α·(Xi·Xr^T − Xr·Xi^T) + β·Ci_old
 * and (Xr·Xi^T) = (Xi·Xr^T)^T, so the second sub-GEMM is redundant.
 *
 * Only writes to the authoritative triangle:
 *   fill_mode=0 (Upper): writes where col >= row
 *   fill_mode=1 (Lower): writes where row >= col
 *
 * Diagonal is forced to zero (Hermitian requirement: Im(C_ii) = 0).
 */
__global__ void antisymmetrize_to_triangle_kernel(
    const __half* __restrict__ temp,       // Full Xi·Xr^T result [N × N] RowMajor
    const __half* __restrict__ C_imag_old, // Previous Im(C) for β accumulation
    __half* __restrict__ C_imag_out,       // Output Im(C) — only triangle written
    int N,
    int64_t batch_stride,
    float alpha, float beta,
    int fill_mode)                          // 0 = Upper, 1 = Lower
{
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int batch = blockIdx.z;

    if (row >= N || col >= N) return;

    // Only write the authoritative triangle
    if (fill_mode == 0 && col < row) return;   // Upper: skip below diagonal
    if (fill_mode == 1 && row < col) return;   // Lower: skip above diagonal

    const __half* t   = temp       + batch * batch_stride;
    const __half* old = C_imag_old + batch * batch_stride;
    __half*       out = C_imag_out + batch * batch_stride;

    const int64_t ij = static_cast<int64_t>(row) * N + col;

    if (row == col) {
        // Diagonal: Hermitian ⇒ Im(C_ii) = 0, regardless of α, β
        out[ij] = __float2half(0.0f);
    } else {
        const int64_t ji = static_cast<int64_t>(col) * N + row;
        float tij = __half2float(t[ij]);
        float tji = __half2float(t[ji]);
        float old_val = __half2float(old[ij]);
        out[ij] = __float2half(alpha * (tij - tji) + beta * old_val);
    }
}

/// Launch anti-symmetrize: temp[N×N] → triangle of C_imag
/// C_imag_old may alias C_imag_out (reads complete before writes in same triangle).
inline void antisymmetrize_to_triangle(
    const __half* temp,
    const __half* C_imag_old,
    __half* C_imag_out,
    int N,
    float alpha, float beta,
    FillMode fill,
    cudaStream_t stream = nullptr,
    int batch_count = 1)
{
    int64_t batch_stride = static_cast<int64_t>(N) * N;
    int mode_int = (fill == FillMode::Upper) ? 0 : 1;
    TUNED_LAUNCH_2D(antisymmetrize_to_triangle_kernel, "antisymmetrize_to_triangle",
        N, N, batch_count, static_cast<int64_t>(N) * N * batch_count * 6, stream,
        temp, C_imag_old, C_imag_out, N, batch_stride, alpha, beta, mode_int);
    CUDA_CHECK(cudaGetLastError());
}


/*
 * Lightweight diagonal-only enforcement: just zero Im(C_ii).
 *
 * Used in production HERK mode where the real part's authoritative triangle is
 * already correct from the 2 sub-GEMMs, but FP8 rounding may leave tiny nonzero
 * imaginary residuals on the diagonal of Re(C). (Re doesn't have this issue —
 * Xr·Xr^T + Xi·Xi^T gives the correct real diagonal intrinsically.)
 *
 * This is O(N) per batch — negligible.
 */
__global__ void enforce_real_diagonal_kernel(
    __half* __restrict__ C_imag,
    int N,
    int64_t batch_stride)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int batch = blockIdx.y;
    if (idx >= N) return;
    __half* ci = C_imag + batch * batch_stride;
    ci[static_cast<int64_t>(idx) * N + idx] = __float2half(0.0f);
}

/// Launch diagonal enforcement: zero Im(C_ii) for all batch elements
inline void enforce_real_diagonal(
    __half* C_imag,
    int N,
    cudaStream_t stream = nullptr,
    int batch_count = 1)
{
    constexpr int kBlock = 256;
    dim3 grid((N + kBlock - 1) / kBlock, batch_count);
    int64_t batch_stride = static_cast<int64_t>(N) * N;
    enforce_real_diagonal_kernel<<<grid, kBlock, 0, stream>>>(
        C_imag, N, batch_stride);
    CUDA_CHECK(cudaGetLastError());
}

// ========================================================================================
// F3: Pack Triangle + Fused Anti-Symmetrize+Pack
// ========================================================================================
//
// pack_triangle_kernel: Unified lower/upper triangle packing (full N×N → packed N*(N+1)/2).
//   Moved from example code into library for use by herk_planar_packed().
//
// antisymmetrize_pack_kernel: Fused anti-symmetrize + pack for Im(C) in HERK.
//   Reads full N×N temp (Xi·Xr^T), reads packed C_imag_old (for β accumulation),
//   computes α·(temp[i,j] − temp[j,i]) + β·old, writes packed C_imag output.
//   Eliminates the full N×N intermediate Im(C) buffer.
//
// Bandwidth savings: 5·N² bytes fused vs 8·N² bytes separate (38% reduction on Im(C) path).

/*
 * Pack a full N×N matrix into triangular packed format (N*(N+1)/2 elements).
 * Optionally accumulates with β: packed_out[idx] = full[row*N+col] + β·packed_old[idx]
 *
 * fill_mode=0 (Upper): packed index maps to (row, col) with col >= row
 *   idx = col*(col+1)/2 + row
 * fill_mode=1 (Lower): packed index maps to (row, col) with row >= col
 *   idx = row*(row+1)/2 + col
 *
 * When packed_old is null, β is ignored (pure pack).
 * packed_old may alias packed_out (each thread reads/writes the same index).
 */
__global__ void pack_triangle_kernel(
    const __half* __restrict__ full,
    const __half* __restrict__ packed_old,  // For β accumulation (null if β=0)
    __half* __restrict__ packed_out,
    int N,
    int64_t batch_stride_full,
    int64_t batch_stride_packed,
    float beta,
    int fill_mode)                          // 0 = Upper, 1 = Lower
{
    const int major = blockIdx.x;
    if (major >= N) return;
    const int batch = blockIdx.y;

    const __half* src = full       + batch * batch_stride_full;
    __half*       dst = packed_out + batch * batch_stride_packed;
    const int64_t packed_start = static_cast<int64_t>(major) * (major + 1) / 2;
    const int num_minor = major + 1;

    if (fill_mode == 1) {
        // Lower: row=major, iterate cols [0..major]
        const int64_t row_off = static_cast<int64_t>(major) * N;
        if (packed_old) {
            const __half* old = packed_old + batch * batch_stride_packed;
            for (int c = threadIdx.x; c < num_minor; c += blockDim.x) {
                float val = __half2float(src[row_off + c])
                          + beta * __half2float(old[packed_start + c]);
                dst[packed_start + c] = __float2half(val);
            }
        } else {
            for (int c = threadIdx.x; c < num_minor; c += blockDim.x) {
                dst[packed_start + c] = src[row_off + c];   // direct copy
            }
        }
    } else {
        // Upper: col=major, iterate rows [0..major]
        if (packed_old) {
            const __half* old = packed_old + batch * batch_stride_packed;
            for (int r = threadIdx.x; r < num_minor; r += blockDim.x) {
                float val = __half2float(src[static_cast<int64_t>(r) * N + major])
                          + beta * __half2float(old[packed_start + r]);
                dst[packed_start + r] = __float2half(val);
            }
        } else {
            for (int r = threadIdx.x; r < num_minor; r += blockDim.x) {
                dst[packed_start + r] = src[static_cast<int64_t>(r) * N + major];
            }
        }
    }
}

/// Launch triangle packing: full[N×N] → packed[N*(N+1)/2]
/// When packed_old is non-null, accumulates: out[i] = full[r*N+c] + β·old[i]
inline void pack_triangle(
    const __half* full,
    __half* packed,
    int N,
    FillMode fill,
    cudaStream_t stream = nullptr,
    int batch_count = 1,
    const __half* packed_old = nullptr,
    float beta = 0.0f)
{
    int64_t total = static_cast<int64_t>(N) * (N + 1) / 2;
    int64_t batch_stride_full   = static_cast<int64_t>(N) * N;
    int64_t batch_stride_packed = total;
    int mode_int = (fill == FillMode::Upper) ? 0 : 1;
    TUNED_LAUNCH_ROW(pack_triangle_kernel, "pack_triangle",
        N, batch_count,
        (batch_stride_full + total) * 2 * batch_count, stream,
        full, packed_old, packed, N, batch_stride_full, batch_stride_packed, beta, mode_int);
    CUDA_CHECK(cudaGetLastError());
}

/*
 * Fused anti-symmetrize + pack kernel for HERK Im(C) output.
 *
 * Reads temp[N×N] (full Xi·Xr^T result) and C_imag_old[packed] (for β accumulation),
 * writes C_imag_packed[packed].
 *
 * For each packed index, recovers (row, col) in the authoritative triangle:
 *   Im(C)[row,col] = α·(temp[row,col] − temp[col,row]) + β·C_imag_old[idx]
 *   Diagonal (row==col): forced to 0.
 *
 * C_imag_old may alias C_imag_packed (packed format, no read/write conflict since
 * each thread reads and writes the same packed index).
 */
__global__ void antisymmetrize_pack_kernel(
    const __half* __restrict__ temp,            // Full N×N row-major
    const __half* __restrict__ C_imag_old,      // Packed N*(N+1)/2 (for β accumulation)
    __half* __restrict__ C_imag_packed,         // Packed N*(N+1)/2 output
    int N,
    int64_t batch_stride_full,                  // N*N for full matrices
    int64_t batch_stride_packed,                // N*(N+1)/2 for packed
    float alpha, float beta,
    int fill_mode)                               // 0 = Upper, 1 = Lower
{
    const int major = blockIdx.x;
    if (major >= N) return;
    const int batch = blockIdx.y;

    const __half* t   = temp         + batch * batch_stride_full;
    const __half* old = C_imag_old   + batch * batch_stride_packed;
    __half*       out = C_imag_packed + batch * batch_stride_packed;
    const int64_t packed_start = static_cast<int64_t>(major) * (major + 1) / 2;
    const int num_minor = major + 1;

    if (fill_mode == 1) {
        // Lower: row=major, cols [0..major]
        const int row = major;
        const int64_t row_off = static_cast<int64_t>(row) * N;
        for (int col = threadIdx.x; col < num_minor; col += blockDim.x) {
            float result;
            if (row == col) {
                result = 0.0f;
            } else {
                float tij = __half2float(t[row_off + col]);
                float tji = __half2float(t[static_cast<int64_t>(col) * N + row]);
                result = alpha * (tij - tji) + beta * __half2float(old[packed_start + col]);
            }
            out[packed_start + col] = __float2half(result);
        }
    } else {
        // Upper: col=major, rows [0..major]
        const int col = major;
        for (int row = threadIdx.x; row < num_minor; row += blockDim.x) {
            float result;
            if (row == col) {
                result = 0.0f;
            } else {
                float tij = __half2float(t[static_cast<int64_t>(row) * N + col]);
                float tji = __half2float(t[static_cast<int64_t>(col) * N + row]);
                result = alpha * (tij - tji) + beta * __half2float(old[packed_start + row]);
            }
            out[packed_start + row] = __float2half(result);
        }
    }
}

/// Launch fused anti-symmetrize + pack: temp[N×N] + C_imag_old[packed] → C_imag_packed[packed]
inline void antisymmetrize_pack(
    const __half* temp,
    const __half* C_imag_old,
    __half* C_imag_packed,
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
    TUNED_LAUNCH_ROW(antisymmetrize_pack_kernel, "antisymmetrize_pack",
        N, batch_count,
        (batch_stride_full + 2 * total) * 2 * batch_count, stream,
        temp, C_imag_old, C_imag_packed, N,
        batch_stride_full, batch_stride_packed,
        alpha, beta, mode_int);
    CUDA_CHECK(cudaGetLastError());
}

/// Fused pack_triangle + antisymmetrize_pack kernel — processes both Re and Im
/// outputs in a single kernel launch (saves 1 launch vs separate kernels).
__global__ void pack_antisymmetrize_triangle_kernel(
    const __half* __restrict__ scratch_Re,
    const __half* __restrict__ temp_Im,
    const __half* __restrict__ old_Re,
    const __half* __restrict__ old_Im,
    __half* __restrict__ out_Re,
    __half* __restrict__ out_Im,
    int N,
    int64_t batch_stride_full,
    int64_t batch_stride_packed,
    float alpha, float beta,
    int fill_mode)
{
    const int major = blockIdx.x;
    if (major >= N) return;
    const int batch = blockIdx.y;

    const __half* re_src = scratch_Re + batch * batch_stride_full;
    const __half* im_src = temp_Im    + batch * batch_stride_full;
    __half*       re_dst = out_Re     + batch * batch_stride_packed;
    __half*       im_dst = out_Im     + batch * batch_stride_packed;
    const int64_t packed_start = static_cast<int64_t>(major) * (major + 1) / 2;
    const int num_minor = major + 1;

    if (fill_mode == 1) {
        const int64_t row_off = static_cast<int64_t>(major) * N;
        if (old_Re) {
            const __half* re_old = old_Re + batch * batch_stride_packed;
            const __half* im_old = old_Im + batch * batch_stride_packed;
            for (int c = threadIdx.x; c < num_minor; c += blockDim.x) {
                float re = __half2float(re_src[row_off + c])
                         + beta * __half2float(re_old[packed_start + c]);
                re_dst[packed_start + c] = __float2half(re);

                float im;
                if (major == c) {
                    im = 0.0f;
                } else {
                    float tij = __half2float(im_src[row_off + c]);
                    float tji = __half2float(im_src[static_cast<int64_t>(c) * N + major]);
                    im = alpha * (tij - tji) + beta * __half2float(im_old[packed_start + c]);
                }
                im_dst[packed_start + c] = __float2half(im);
            }
        } else {
            for (int c = threadIdx.x; c < num_minor; c += blockDim.x) {
                re_dst[packed_start + c] = re_src[row_off + c];

                float im;
                if (major == c) {
                    im = 0.0f;
                } else {
                    float tij = __half2float(im_src[row_off + c]);
                    float tji = __half2float(im_src[static_cast<int64_t>(c) * N + major]);
                    im = alpha * (tij - tji);
                }
                im_dst[packed_start + c] = __float2half(im);
            }
        }
    } else {
        if (old_Re) {
            const __half* re_old = old_Re + batch * batch_stride_packed;
            const __half* im_old = old_Im + batch * batch_stride_packed;
            for (int r = threadIdx.x; r < num_minor; r += blockDim.x) {
                float re = __half2float(re_src[static_cast<int64_t>(r) * N + major])
                         + beta * __half2float(re_old[packed_start + r]);
                re_dst[packed_start + r] = __float2half(re);

                float im;
                if (r == major) {
                    im = 0.0f;
                } else {
                    float tij = __half2float(im_src[static_cast<int64_t>(r) * N + major]);
                    float tji = __half2float(im_src[static_cast<int64_t>(major) * N + r]);
                    im = alpha * (tij - tji) + beta * __half2float(im_old[packed_start + r]);
                }
                im_dst[packed_start + r] = __float2half(im);
            }
        } else {
            for (int r = threadIdx.x; r < num_minor; r += blockDim.x) {
                re_dst[packed_start + r] = re_src[static_cast<int64_t>(r) * N + major];

                float im;
                if (r == major) {
                    im = 0.0f;
                } else {
                    float tij = __half2float(im_src[static_cast<int64_t>(r) * N + major]);
                    float tji = __half2float(im_src[static_cast<int64_t>(major) * N + r]);
                    im = alpha * (tij - tji);
                }
                im_dst[packed_start + r] = __float2half(im);
            }
        }
    }
}

// ========================================================================================
// Pack Scratch to Triangle — extracts lower triangle from N×N interleaved scratch
// ========================================================================================
//
// Copies the lower triangle from an N×N row-major interleaved complex scratch buffer
// to packed interleaved triangle format. Used by the direct HERK scratch output path
// to separate coalesced N×N writes (zero write amplification) from triangle packing.
//
// Grid: (N, batch_count). Each block processes one row of the lower triangle.
// Supports beta accumulation against an existing packed triangle buffer.
//

__global__ void pack_scratch_to_triangle_kernel(
    const __half* __restrict__ scratch,     // [batch × N × 2N] interleaved complex
    __half* __restrict__ packed,            // [batch × N*(N+1)] interleaved packed triangle
    const __half* __restrict__ old_packed,  // for beta accumulation (nullable)
    int N,
    int64_t scratch_batch_stride,      // N * N * 2
    int64_t packed_batch_stride,       // N * (N + 1)
    float beta)
{
    const int row = blockIdx.x;
    if (row >= N) return;
    const int batch = blockIdx.y;

    const __half* src = scratch + batch * scratch_batch_stride + row * 2 * N;
    __half* dst = packed + batch * packed_batch_stride;
    const int64_t packed_row_start = static_cast<int64_t>(row) * (row + 1);  // row*(row+1)/2 * 2

    if (beta != 0.0f && old_packed) {
        const __half* old = old_packed + batch * packed_batch_stride;
        for (int col = threadIdx.x; col <= row; col += blockDim.x) {
            int64_t src_re = col * 2;
            int64_t src_im = src_re + 1;
            int64_t dst_idx_re = packed_row_start + col * 2;
            int64_t dst_idx_im = dst_idx_re + 1;
            float re = __half2float(src[src_re]) + beta * __half2float(old[dst_idx_re]);
            float im = __half2float(src[src_im]) + beta * __half2float(old[dst_idx_im]);
            dst[dst_idx_re] = __float2half(re);
            dst[dst_idx_im] = __float2half(im);
        }
    } else {
        for (int col = threadIdx.x; col <= row; col += blockDim.x) {
            int64_t src_re = col * 2;
            int64_t src_im = src_re + 1;
            int64_t dst_idx_re = packed_row_start + col * 2;
            int64_t dst_idx_im = dst_idx_re + 1;
            dst[dst_idx_re] = src[src_re];
            dst[dst_idx_im] = src[src_im];
        }
    }
}

/// Launch pack_scratch_to_triangle: scratch[N×N×2] → packed[N*(N+1)]
inline void pack_scratch_to_triangle(
    const __half* scratch,
    __half* packed,
    const __half* old_packed,  // nullable — only used when beta != 0
    int N,
    int batch_count,
    float beta,
    cudaStream_t stream)
{
    int64_t scratch_batch_stride = static_cast<int64_t>(N) * N * 2;
    int64_t packed_batch_stride = static_cast<int64_t>(N) * (N + 1);
    int64_t bytes = (scratch_batch_stride + packed_batch_stride) * 2 * batch_count;
    TUNED_LAUNCH_ROW(pack_scratch_to_triangle_kernel, "pack_scratch_to_triangle",
        N, batch_count, bytes, stream,
        scratch, packed, old_packed, N,
        scratch_batch_stride, packed_batch_stride, beta);
    CUDA_CHECK(cudaGetLastError());
}

/// Launch fused pack + antisymmetrize: scratch_Re[N×N] + temp_Im[N×N] → packed Re + packed Im
inline void pack_antisymmetrize_triangle(
    const __half* scratch_Re,
    const __half* temp_Im,
    __half* out_Re,
    __half* out_Im,
    int N,
    float alpha, float beta,
    FillMode fill,
    cudaStream_t stream = nullptr,
    int batch_count = 1,
    const __half* old_Re = nullptr,
    const __half* old_Im = nullptr)
{
    int64_t batch_stride_full   = static_cast<int64_t>(N) * N;
    int64_t batch_stride_packed = static_cast<int64_t>(N) * (N + 1) / 2;
    int mode_int = (fill == FillMode::Upper) ? 0 : 1;
    TUNED_LAUNCH_ROW(pack_antisymmetrize_triangle_kernel, "pack_antisymmetrize_triangle",
        N, batch_count,
        (2 * batch_stride_full + 4 * batch_stride_packed) * 2 * batch_count, stream,
        scratch_Re, temp_Im, old_Re, old_Im, out_Re, out_Im,
        N, batch_stride_full, batch_stride_packed,
        alpha, beta, mode_int);
    CUDA_CHECK(cudaGetLastError());
}

// ========================================================================================
// FP32 scratch → packed triangle kernel
// ========================================================================================
//
// Same logic as pack_scratch_to_triangle_kernel but with float types.
// Used by the FP32 direct HERK scratch output path.

__global__ void pack_scratch_to_triangle_fp32_kernel(
    const float* __restrict__ scratch,      // [batch × N × 2N] interleaved complex
    float* __restrict__ packed,             // [batch × N*(N+1)] interleaved packed triangle
    const float* __restrict__ old_packed,   // for beta accumulation (nullable)
    int N,
    int64_t scratch_batch_stride,      // N * N * 2
    int64_t packed_batch_stride,       // N * (N + 1)
    float beta)
{
    const int row = blockIdx.x;
    if (row >= N) return;
    const int batch = blockIdx.y;

    const float* src = scratch + batch * scratch_batch_stride + row * 2 * N;
    float* dst = packed + batch * packed_batch_stride;
    const int64_t packed_row_start = static_cast<int64_t>(row) * (row + 1);  // row*(row+1)/2 * 2

    if (beta != 0.0f && old_packed) {
        const float* old = old_packed + batch * packed_batch_stride;
        for (int col = threadIdx.x; col <= row; col += blockDim.x) {
            int64_t src_re = col * 2;
            int64_t src_im = src_re + 1;
            int64_t dst_idx_re = packed_row_start + col * 2;
            int64_t dst_idx_im = dst_idx_re + 1;
            dst[dst_idx_re] = src[src_re] + beta * old[dst_idx_re];
            dst[dst_idx_im] = src[src_im] + beta * old[dst_idx_im];
        }
    } else {
        for (int col = threadIdx.x; col <= row; col += blockDim.x) {
            int64_t src_re = col * 2;
            int64_t src_im = src_re + 1;
            int64_t dst_idx_re = packed_row_start + col * 2;
            int64_t dst_idx_im = dst_idx_re + 1;
            dst[dst_idx_re] = src[src_re];
            dst[dst_idx_im] = src[src_im];
        }
    }
}

/// Launch pack_scratch_to_triangle_fp32: scratch[N×N×2 float] → packed[N*(N+1) float]
inline void pack_scratch_to_triangle_fp32(
    const float* scratch,
    float* packed,
    const float* old_packed,  // nullable — only used when beta != 0
    int N,
    int batch_count,
    float beta,
    cudaStream_t stream)
{
    int64_t scratch_batch_stride = static_cast<int64_t>(N) * N * 2;
    int64_t packed_batch_stride = static_cast<int64_t>(N) * (N + 1);
    int64_t bytes = (scratch_batch_stride + packed_batch_stride) * 4 * batch_count;
    TUNED_LAUNCH_ROW(pack_scratch_to_triangle_fp32_kernel, "pack_scratch_to_triangle_fp32",
        N, batch_count, bytes, stream,
        scratch, packed, old_packed, N,
        scratch_batch_stride, packed_batch_stride, beta);
    CUDA_CHECK(cudaGetLastError());
}
