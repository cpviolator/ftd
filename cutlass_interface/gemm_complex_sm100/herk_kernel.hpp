// ========================================================================================
// Direct HERK Kernel — SM100/SM120 Wrapper
// ========================================================================================
//
// Includes the shared HERK kernel implementation, plus the SM100-only FP16-input
// cast-in-loop kernel (herk_direct_fp8_kernel + load_tile_to_smem + launch_herk_direct_fp8).
//
// This is a textual include (no #pragma once) — included inside namespace gemm_complex_sm100.
// ========================================================================================

// ---- Shared code (all templated kernels, helpers, launchers) ----
#include "../shared/herk_kernel_common.hpp"

// ========================================================================================
// SM100-only: FP16-input cast-in-loop kernel
// ========================================================================================
//
// Original direct HERK that loads FP16 from global memory, casts to FP8 in registers.
// Retained for backward compatibility. The precast kernel (in shared code) is preferred.
//
// Uses double-buffered SMEM pipeline (2 buffers × 2 tiles × 4096 = 16 KB)
// vs the precast kernel's 3-buffer cp.async pipeline (24 KB).
//

// ---- Vectorized global → shared load: FP16 input → FP8 cast → shared memory ----
__device__ __forceinline__ void load_tile_to_smem(
    const __half* __restrict__ A_batch,
    unsigned char* __restrict__ smem,
    unsigned firstRecv,
    int kt, int K, int N, int k_remaining,
    int tid)
{
    int recv = tid / 4;        // 0..31
    int k_group = tid % 4;    // 0..3, each covers 16 complex (32 FP8)

    unsigned abs_recv = firstRecv + recv;

    // Destination in shared memory: 32 consecutive FP8 values = 8 unsigned ints
    unsigned* smem_dst = reinterpret_cast<unsigned*>(
        &smem[recv * HERK_SMEM_K + k_group * 32]);

    if (abs_recv >= (unsigned)N) {
        // Out of bounds receiver: zero-fill all 32 FP8
        #pragma unroll
        for (int i = 0; i < 8; i++) smem_dst[i] = 0;
        return;
    }

    // Base address in global memory for this thread's data
    // k_group covers complex K range [k_group*16 .. k_group*16+15]
    int k_base = k_group * 16;  // local complex offset within this chunk

    #pragma unroll
    for (int sub = 0; sub < 4; sub++) {
        int local_k = k_base + sub * 4;  // complex K offset within chunk
        int global_k = kt + local_k;     // absolute complex K offset

        if (local_k < k_remaining && global_k < K) {
            const __half* src = &A_batch[static_cast<int64_t>(abs_recv) * 2 * K
                                         + static_cast<int64_t>(global_k) * 2];
            int valid_complex = min(4, k_remaining - local_k);

            if (valid_complex >= 4) {
                // Full group: vectorized 128-bit load → 8 FP8 cast → 2 × 32-bit store
                uint4 data = *reinterpret_cast<const uint4*>(src);
                unsigned f0, f1;
                cast_8xfp16_to_fp8(data, f0, f1);
                smem_dst[sub * 2]     = f0;
                smem_dst[sub * 2 + 1] = f1;
            } else {
                // Partial group: scalar load with zero padding
                unsigned char fp8_bytes[8] = {0, 0, 0, 0, 0, 0, 0, 0};
                for (int i = 0; i < valid_complex * 2; i++) {
                    __nv_fp8_e4m3 f = __nv_fp8_e4m3(src[i]);
                    fp8_bytes[i] = *reinterpret_cast<unsigned char*>(&f);
                }
                smem_dst[sub * 2]     = fp8_bytes[0] | (fp8_bytes[1] << 8)
                                      | (fp8_bytes[2] << 16) | (fp8_bytes[3] << 24);
                smem_dst[sub * 2 + 1] = fp8_bytes[4] | (fp8_bytes[5] << 8)
                                      | (fp8_bytes[6] << 16) | (fp8_bytes[7] << 24);
            }
        } else {
            // Out of bounds K: zero-fill
            smem_dst[sub * 2]     = 0;
            smem_dst[sub * 2 + 1] = 0;
        }
    }
}

// ---- The direct HERK kernel (FP16 input, cast-in-loop, double-buffered) ----
__global__ void __launch_bounds__(HERK_THREADS, 4)
herk_direct_fp8_kernel(
    const __half* __restrict__ A,        // [batch, N, 2*K] interleaved complex FP16
    __half* __restrict__ C,              // [batch, N*(N+1)] interleaved packed triangle
    int N, int K,
    int batch_count,
    float alpha, float beta,
    int blocks_per_dim)
{
    // ---- 1. Map triangular block index → (blockY, blockX) ----
    const unsigned block = blockIdx.x;
    const unsigned batch = blockIdx.y;

    unsigned blockX = static_cast<unsigned>(__fsqrt_rn(8.0f * block + 1.0f) - 0.999f) * 0.5f;
    while (blockX * (blockX + 1) / 2 > block) --blockX;
    while ((blockX + 1) * (blockX + 2) / 2 <= block) ++blockX;
    unsigned blockY = block - blockX * (blockX + 1) / 2;

    const unsigned firstRecvRow = blockX * HERK_BLOCK_N;
    const unsigned firstRecvCol = blockY * HERK_BLOCK_N;
    const bool is_diagonal = (blockX == blockY);

    const int tid = threadIdx.x + threadIdx.y * 32 + threadIdx.z * 64;
    const int warp_id = tid / 32;
    const int lane = tid % 32;

    // ---- 2. Shared memory: double-buffered ----
    extern __shared__ char smem_raw[];
    constexpr int TILE_BYTES = HERK_BLOCK_N * HERK_SMEM_K;  // 32 × 128 = 4096 bytes
    const int tiles_per_buf = is_diagonal ? 1 : 2;

    unsigned char* smem_buf[2];
    smem_buf[0] = reinterpret_cast<unsigned char*>(smem_raw);
    smem_buf[1] = smem_buf[0] + tiles_per_buf * TILE_BYTES;

    // ---- 3. Initialize FP32 accumulators ----
    float acc[HERK_FRAGS_Y][HERK_FRAGS_X][4];
    #pragma unroll
    for (int fy = 0; fy < HERK_FRAGS_Y; fy++)
        #pragma unroll
        for (int fx = 0; fx < HERK_FRAGS_X; fx++)
            acc[fy][fx][0] = acc[fy][fx][1] = acc[fy][fx][2] = acc[fy][fx][3] = 0.0f;

    // ---- 4. Batch pointer ----
    const int64_t A_batch_stride = static_cast<int64_t>(N) * K * 2;
    const __half* A_batch = A + batch * A_batch_stride;

    // ---- 5. Software-pipelined K-loop (double-buffered, FP16 input) ----
    int buf_idx = 0;

    // ---- Prologue: load first chunk ----
    {
        int k_remaining = min(HERK_K_CHUNK, K);
        unsigned char* smem_row = smem_buf[0];
        unsigned char* smem_col = is_diagonal ? smem_row : (smem_row + TILE_BYTES);

        load_tile_to_smem(A_batch, smem_row, firstRecvRow, 0, K, N, k_remaining, tid);
        if (!is_diagonal) {
            load_tile_to_smem(A_batch, smem_col, firstRecvCol, 0, K, N, k_remaining, tid);
        }
        __syncthreads();
    }

    // ---- Main loop: pipelined load + compute ----
    for (int kt = HERK_K_CHUNK; kt < K; kt += HERK_K_CHUNK) {
        int next_buf = 1 - buf_idx;
        int k_remaining_next = min(HERK_K_CHUNK, K - kt);

        unsigned char* smem_row_next = smem_buf[next_buf];
        unsigned char* smem_col_next = is_diagonal ? smem_row_next : (smem_row_next + TILE_BYTES);

        load_tile_to_smem(A_batch, smem_row_next, firstRecvRow, kt, K, N, k_remaining_next, tid);
        if (!is_diagonal) {
            load_tile_to_smem(A_batch, smem_col_next, firstRecvCol, kt, K, N, k_remaining_next, tid);
        }

        // Compute on CURRENT buffer
        unsigned char* smem_row = smem_buf[buf_idx];
        unsigned char* smem_col = is_diagonal ? smem_row : (smem_row + TILE_BYTES);

        #pragma unroll
        for (int ks = 0; ks < HERK_K_SUBS; ks++) {
            int k_base = ks * 32;

            #pragma unroll
            for (int fy = 0; fy < HERK_FRAGS_Y; fy++) {
                if (fy != warp_id) continue;

                int recv_base_y = fy * HERK_RECV_PER_MMA_Y;

                unsigned a0, a1, a2, a3;
                {
                    int groupID = lane / 4;
                    int k_chunk = lane % 4;
                    int recv_local = groupID / 2;
                    int is_conj = groupID & 1;
                    int smem_recv0 = recv_base_y + recv_local;
                    int smem_recv1 = recv_base_y + recv_local + 4;
                    int k_off0 = k_base + k_chunk * 4;
                    int k_off1 = k_base + 16 + k_chunk * 4;

                    unsigned v0 = *reinterpret_cast<const unsigned*>(&smem_row[smem_recv0 * HERK_SMEM_K + k_off0]);
                    unsigned v1 = *reinterpret_cast<const unsigned*>(&smem_row[smem_recv0 * HERK_SMEM_K + k_off1]);
                    a0 = is_conj ? conj_perm_fp8(v0) : v0;
                    a2 = is_conj ? conj_perm_fp8(v1) : v1;

                    if (smem_recv1 < HERK_BLOCK_N) {
                        unsigned v2 = *reinterpret_cast<const unsigned*>(&smem_row[smem_recv1 * HERK_SMEM_K + k_off0]);
                        unsigned v3 = *reinterpret_cast<const unsigned*>(&smem_row[smem_recv1 * HERK_SMEM_K + k_off1]);
                        a1 = is_conj ? conj_perm_fp8(v2) : v2;
                        a3 = is_conj ? conj_perm_fp8(v3) : v3;
                    } else {
                        a1 = 0; a3 = 0;
                    }
                }

                #pragma unroll
                for (int fx = 0; fx < HERK_FRAGS_X; fx++) {
                    if (is_diagonal) {
                        unsigned row_start = firstRecvRow + fy * HERK_RECV_PER_MMA_Y;
                        unsigned col_start = firstRecvCol + fx * HERK_RECV_PER_MMA_X;
                        if (row_start < col_start) continue;
                    }

                    int recv_base_x = fx * HERK_RECV_PER_MMA_X;
                    unsigned b0, b1;
                    {
                        int recv_idx = recv_base_x + lane / 4;
                        int k_off0 = k_base + (lane % 4) * 4;
                        int k_off1 = k_base + 16 + (lane % 4) * 4;

                        if (recv_idx < HERK_BLOCK_N) {
                            b0 = *reinterpret_cast<const unsigned*>(&smem_col[recv_idx * HERK_SMEM_K + k_off0]);
                            b1 = *reinterpret_cast<const unsigned*>(&smem_col[recv_idx * HERK_SMEM_K + k_off1]);
                        } else {
                            b0 = 0; b1 = 0;
                        }
                    }

                    mma_fp8_m16n8k32(
                        acc[fy][fx][0], acc[fy][fx][1], acc[fy][fx][2], acc[fy][fx][3],
                        a0, a1, a2, a3,
                        b0, b1,
                        acc[fy][fx][0], acc[fy][fx][1], acc[fy][fx][2], acc[fy][fx][3]);
                }
            }
        }

        __syncthreads();
        buf_idx = next_buf;
    }

    // ---- Epilogue: compute last chunk ----
    {
        unsigned char* smem_row = smem_buf[buf_idx];
        unsigned char* smem_col = is_diagonal ? smem_row : (smem_row + TILE_BYTES);

        #pragma unroll
        for (int ks = 0; ks < HERK_K_SUBS; ks++) {
            int k_base = ks * 32;

            #pragma unroll
            for (int fy = 0; fy < HERK_FRAGS_Y; fy++) {
                if (fy != warp_id) continue;

                int recv_base_y = fy * HERK_RECV_PER_MMA_Y;

                unsigned a0, a1, a2, a3;
                {
                    int groupID = lane / 4;
                    int k_chunk = lane % 4;
                    int recv_local = groupID / 2;
                    int is_conj = groupID & 1;
                    int smem_recv0 = recv_base_y + recv_local;
                    int smem_recv1 = recv_base_y + recv_local + 4;
                    int k_off0 = k_base + k_chunk * 4;
                    int k_off1 = k_base + 16 + k_chunk * 4;

                    unsigned v0 = *reinterpret_cast<const unsigned*>(&smem_row[smem_recv0 * HERK_SMEM_K + k_off0]);
                    unsigned v1 = *reinterpret_cast<const unsigned*>(&smem_row[smem_recv0 * HERK_SMEM_K + k_off1]);
                    a0 = is_conj ? conj_perm_fp8(v0) : v0;
                    a2 = is_conj ? conj_perm_fp8(v1) : v1;

                    if (smem_recv1 < HERK_BLOCK_N) {
                        unsigned v2 = *reinterpret_cast<const unsigned*>(&smem_row[smem_recv1 * HERK_SMEM_K + k_off0]);
                        unsigned v3 = *reinterpret_cast<const unsigned*>(&smem_row[smem_recv1 * HERK_SMEM_K + k_off1]);
                        a1 = is_conj ? conj_perm_fp8(v2) : v2;
                        a3 = is_conj ? conj_perm_fp8(v3) : v3;
                    } else {
                        a1 = 0; a3 = 0;
                    }
                }

                #pragma unroll
                for (int fx = 0; fx < HERK_FRAGS_X; fx++) {
                    if (is_diagonal) {
                        unsigned row_start = firstRecvRow + fy * HERK_RECV_PER_MMA_Y;
                        unsigned col_start = firstRecvCol + fx * HERK_RECV_PER_MMA_X;
                        if (row_start < col_start) continue;
                    }

                    int recv_base_x = fx * HERK_RECV_PER_MMA_X;
                    unsigned b0, b1;
                    {
                        int recv_idx = recv_base_x + lane / 4;
                        int k_off0 = k_base + (lane % 4) * 4;
                        int k_off1 = k_base + 16 + (lane % 4) * 4;

                        if (recv_idx < HERK_BLOCK_N) {
                            b0 = *reinterpret_cast<const unsigned*>(&smem_col[recv_idx * HERK_SMEM_K + k_off0]);
                            b1 = *reinterpret_cast<const unsigned*>(&smem_col[recv_idx * HERK_SMEM_K + k_off1]);
                        } else {
                            b0 = 0; b1 = 0;
                        }
                    }

                    mma_fp8_m16n8k32(
                        acc[fy][fx][0], acc[fy][fx][1], acc[fy][fx][2], acc[fy][fx][3],
                        a0, a1, a2, a3,
                        b0, b1,
                        acc[fy][fx][0], acc[fy][fx][1], acc[fy][fx][2], acc[fy][fx][3]);
                }
            }
        }
    }

    // ---- 6. Store: FP32 accumulators → interleaved packed triangle ----
    const int64_t packed_batch_stride = static_cast<int64_t>(N) * (N + 1);
    __half* C_batch = C + batch * packed_batch_stride;

    #pragma unroll
    for (int fy = 0; fy < HERK_FRAGS_Y; fy++) {
        if (fy != warp_id) continue;

        #pragma unroll
        for (int fx = 0; fx < HERK_FRAGS_X; fx++) {
            if (is_diagonal) {
                unsigned row_start = firstRecvRow + fy * HERK_RECV_PER_MMA_Y;
                unsigned col_start = firstRecvCol + fx * HERK_RECV_PER_MMA_X;
                if (row_start < col_start) continue;
            }

            int mma_row0 = lane / 4;
            int mma_row1 = mma_row0 + 8;
            int mma_col0 = (lane % 4) * 2;
            int mma_col1 = mma_col0 + 1;

            float d_vals[4] = { acc[fy][fx][0], acc[fy][fx][1], acc[fy][fx][2], acc[fy][fx][3] };
            int mma_rows[4] = { mma_row0, mma_row0, mma_row1, mma_row1 };
            int mma_cols[4] = { mma_col0, mma_col1, mma_col0, mma_col1 };

            #pragma unroll
            for (int vi = 0; vi < 4; vi++) {
                int mr = mma_rows[vi];
                int mc = mma_cols[vi];

                int recv_local_row = mr / 2;
                int is_im = mr & 1;
                int recv_local_col = mc;

                unsigned recv_row = firstRecvRow + fy * HERK_RECV_PER_MMA_Y + recv_local_row;
                unsigned recv_col = firstRecvCol + fx * HERK_RECV_PER_MMA_X + recv_local_col;

                if (recv_row >= (unsigned)N || recv_col >= (unsigned)N) continue;
                if (recv_row < recv_col) continue;

                float val = d_vals[vi];
                if (recv_row == recv_col && is_im) val = 0.0f;
                val *= alpha;

                int64_t tri_idx = static_cast<int64_t>(recv_row) * (recv_row + 1) / 2 + recv_col;
                int64_t out_idx = tri_idx * 2 + is_im;

                if (beta != 0.0f) {
                    val += beta * __half2float(C_batch[out_idx]);
                }

                C_batch[out_idx] = __float2half(val);
            }
        }
    }
}

// ---- Host-side launch for cast-in-loop kernel ----
inline cutlass::Status launch_herk_direct_fp8(
    const __half* A,       // [batch, N, 2*K] interleaved complex FP16
    __half* C,             // [batch, N*(N+1)] interleaved packed triangle
    int N, int K,
    int batch_count,
    float alpha, float beta,
    FillMode fill,
    cudaStream_t stream)
{
    if (fill != FillMode::Lower) {
        return cutlass::Status::kErrorNotSupported;
    }

    int blocks_per_dim = (N + HERK_BLOCK_N - 1) / HERK_BLOCK_N;
    int tri_blocks = blocks_per_dim * (blocks_per_dim + 1) / 2;

    dim3 grid(tri_blocks, batch_count, 1);
    dim3 block(32, 2, 2);  // 128 threads = 4 warps

    // Shared memory: double-buffered, max 2 tiles per buffer (rectangle case)
    int smem_bytes = 2 * 2 * HERK_BLOCK_N * HERK_SMEM_K;  // 16384 bytes

    herk_direct_fp8_kernel<<<grid, block, smem_bytes, stream>>>(
        A, C, N, K, batch_count, alpha, beta, blocks_per_dim);
    CUDA_CHECK(cudaGetLastError());

    return cutlass::Status::kSuccess;
}
