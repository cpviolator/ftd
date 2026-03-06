// ========================================================================================
// Direct Complex GEMM Kernel — Single-Launch Complex GEMM via Conjugate Permutation
// ========================================================================================
//
// Custom CUDA kernel that performs complex GEMM C = alpha * A * B^T + beta * C
// in a single launch using the conjugate permutation trick:
//
//   With B_neg = [Br, -Bi] interleaved FP8 (imaginary negated during precast):
//     Normal A row × B_neg col:     Ar*Br + Ai*(-Bi) = Ar*Br - Ai*Bi = Re(C)
//     Conj_perm A row × B_neg col:  Ai*Br + (-Ar)*(-Bi) = Ai*Br + Ar*Bi = Im(C)
//
//   conj_perm(v) = __byte_perm(v ^ 0x00800080, 0, 0x2301)
//   Transforms interleaved FP8 [re, im] → [im, -re].
//   Loading A fragment with alternating normal/conj_perm rows makes a single MMA
//   produce BOTH Re(C) and Im(C) simultaneously.
//
// Key differences from HERK kernel:
//   - Rectangular M×N output grid (not triangular)
//   - Separate A and B input matrices (not A=B like HERK)
//   - B is pre-cast to interleaved FP8 with negated imaginary parts
//   - Output modes: PlanarComplex (separate Re/Im) or PowerDetect (Re²+Im²)
//
// Tile configurations:
//   GemmTile32x32  — 32×32  output tile, AI=64,  SMEM=24KB (3-buf) — backward-compatible
//   GemmTile32x128 — 32×128 output tile, AI=102, SMEM=60KB (3-buf) — wide-N
//   GemmTile64x64  — 64×64  output tile, AI=128, SMEM=48KB (3-buf) — balanced
//   GemmTile64x128 — 64×128 output tile, AI=171, SMEM=72KB (3-buf) — large
//
// Batch parallelism: gridDim.y = batch_count (all batches in one launch)
// Grid: gridDim.x = ceil(M/BLOCK_M) * ceil(N/BLOCK_N) (rectangular block mapping)
//
// Auto-selection: launch_gemm_direct_power/planar pick the largest tile that fits
// the problem dimensions to maximize arithmetic intensity.
//
// This is a textual include (no #pragma once) — included inside the arch-specific namespace.
// ========================================================================================

// ========================================================================================
// GemmTileConfig — compile-time tile dimensions for direct GEMM kernel
// ========================================================================================
//
// Parameterizes the output tile shape. The MMA is m16n8k32 with the conjugate
// permutation trick producing 8 output rows and 8 output columns per MMA.
// THREADS=128 (4 warps) is fixed. Wider tiles use multi-pass loading and
// strided warp mapping to cover more fragments.
//
// A_LOAD_PASSES: number of 32-row loads needed for the A tile
// B_LOAD_PASSES: number of 32-row loads needed for the B tile
// STORE_PHASES:  number of 32-column phases for the store step (bounds SMEM)
//

template <int BLOCK_M_, int BLOCK_N_>
struct GemmTileConfig {
    static constexpr int BLOCK_M = BLOCK_M_;
    static constexpr int BLOCK_N = BLOCK_N_;
    static constexpr int THREADS = 128;       // 4 warps, unchanged
    static constexpr int NUM_WARPS = 4;
    static constexpr int RECV_PER_MMA_Y = 8;  // m16n8k32: 16 rows / 2 (conj trick)
    static constexpr int RECV_PER_MMA_X = 8;  // m16n8k32: 8 columns
    static constexpr int FRAGS_Y = BLOCK_M / RECV_PER_MMA_Y;
    static constexpr int FRAGS_X = BLOCK_N / RECV_PER_MMA_X;
    static constexpr int A_LOAD_PASSES = BLOCK_M / 32;  // 32 rows per cp.async pass
    static constexpr int B_LOAD_PASSES = BLOCK_N / 32;  // 32 rows per cp.async pass
    static constexpr int STORE_PHASES  = BLOCK_N / 32;   // 32 cols per store phase
    // Per-warp accumulator count: each warp handles FRAGS_Y/NUM_WARPS fy values
    // using strided assignment (fy = warp_id, warp_id+4, ...).
    // The accumulator array is sized per-warp to avoid register spilling:
    //   32x32:  1 fy × 4 fx × 4 = 16 floats
    //   32x128: 1 fy × 16 fx × 4 = 64 floats
    //   64x64:  2 fy × 8 fx × 4 = 64 floats
    //   64x128: 2 fy × 16 fx × 4 = 128 floats
    static constexpr int FYS_PER_WARP = FRAGS_Y / NUM_WARPS;
};

using GemmTile32x32  = GemmTileConfig<32, 32>;    // Current, backward-compatible
using GemmTile32x128 = GemmTileConfig<32, 128>;   // Wide-N
using GemmTile64x64  = GemmTileConfig<64, 64>;    // Balanced
using GemmTile64x128 = GemmTileConfig<64, 128>;   // Large

// Legacy constants — kept for the non-templated HERK kernel compatibility
static constexpr int GEMM_BLOCK_M = 32;
static constexpr int GEMM_BLOCK_N = 32;
static constexpr int GEMM_THREADS = 128;
static constexpr int GEMM_RECV_PER_MMA_Y = 8;
static constexpr int GEMM_RECV_PER_MMA_X = 8;
static constexpr int GEMM_FRAGS_Y = GEMM_BLOCK_M / GEMM_RECV_PER_MMA_Y;
static constexpr int GEMM_FRAGS_X = GEMM_BLOCK_N / GEMM_RECV_PER_MMA_X;

// ========================================================================================
// Output mode enum
// ========================================================================================

enum class GemmOutputMode {
    PlanarComplex,      // Write separate Re(C) and Im(C) FP32 buffers
    PlanarComplexFP16,  // Write separate Re(C) and Im(C) FP16 buffers
    PowerDetect         // Write Re(C)^2 + Im(C)^2 to single FP32 buffer
};

// ========================================================================================
// Tile name helper — for debug/benchmark output
// ========================================================================================

template <typename Tile>
inline const char* gemm_tile_name() {
    if constexpr (Tile::BLOCK_M == 32 && Tile::BLOCK_N == 32)   return "32x32";
    if constexpr (Tile::BLOCK_M == 32 && Tile::BLOCK_N == 128)  return "32x128";
    if constexpr (Tile::BLOCK_M == 64 && Tile::BLOCK_N == 64)   return "64x64";
    if constexpr (Tile::BLOCK_M == 64 && Tile::BLOCK_N == 128)  return "64x128";
    return "unknown";
}

// ========================================================================================
// Templated store helpers — rectangular output (no triangle masking)
// ========================================================================================

// Store to planar complex output: separate Re(C) and Im(C) FP32 buffers.
// Uses SMEM staging with phased column processing for bounded SMEM usage.
// Each phase processes 32 columns; SMEM = BLOCK_M × 64 × sizeof(float) per phase.
template <typename Tile>
__device__ __forceinline__ void store_gemm_planar_t(
    float acc[Tile::FYS_PER_WARP][Tile::FRAGS_X][4],
    float* __restrict__ C_re_batch,
    float* __restrict__ C_im_batch,
    char* __restrict__ smem_raw,
    unsigned firstRow, unsigned firstCol,
    int M, int N, float alpha, float beta,
    int tid, int warp_id, int lane)
{
    float* smem_store = reinterpret_cast<float*>(smem_raw);

    // Process 32 columns per phase to keep SMEM at BLOCK_M × 64 floats
    #pragma unroll
    for (int phase = 0; phase < Tile::STORE_PHASES; phase++) {
        int fx_start = phase * 4;  // 4 x-fragments per 32 columns

        // Phase 1: Scatter accumulators to SMEM tile [BLOCK_M rows × 64 elements]
        // Layout: smem_store[tile_row * 64 + local_col * 2 + is_im]
        #pragma unroll
        for (int fy_local = 0; fy_local < Tile::FYS_PER_WARP; fy_local++) {
            int fy = warp_id + fy_local * Tile::NUM_WARPS;
            #pragma unroll
            for (int fx_off = 0; fx_off < 4 && fx_start + fx_off < Tile::FRAGS_X; fx_off++) {
                int fx = fx_start + fx_off;
                int mma_row0 = lane / 4;
                int mma_row1 = mma_row0 + 8;
                int mma_col0 = (lane % 4) * 2;
                int mma_col1 = mma_col0 + 1;

                float d_vals[4] = { acc[fy_local][fx][0], acc[fy_local][fx][1], acc[fy_local][fx][2], acc[fy_local][fx][3] };
                int mma_rows[4] = { mma_row0, mma_row0, mma_row1, mma_row1 };
                int mma_cols[4] = { mma_col0, mma_col1, mma_col0, mma_col1 };

                #pragma unroll
                for (int vi = 0; vi < 4; vi++) {
                    int mr = mma_rows[vi];
                    int mc = mma_cols[vi];
                    int recv_local_row = mr / 2;
                    int is_im = mr & 1;
                    int recv_local_col = mc;

                    int tile_row = fy * Tile::RECV_PER_MMA_Y + recv_local_row;
                    int local_col = fx_off * Tile::RECV_PER_MMA_X + recv_local_col;
                    smem_store[tile_row * 64 + local_col * 2 + is_im] = d_vals[vi];
                }
            }
        }

        __syncthreads();

        // Phase 2: Coalesced row-by-row flush to global memory.
        // 128 threads write 2 rows per iteration (64 elements per row = 32 Re + 32 Im).
        for (int r = 0; r < Tile::BLOCK_M; r += 2) {
            int my_row = r + (tid / 64);
            int my_elem = tid % 64;
            int local_col = my_elem / 2;
            int is_im = my_elem & 1;

            unsigned row = firstRow + my_row;
            unsigned col = firstCol + phase * 32 + local_col;

            if (row < (unsigned)M && col < (unsigned)N) {
                float val = smem_store[my_row * 64 + my_elem];
                val *= alpha;

                int64_t out_idx = static_cast<int64_t>(row) * N + col;

                if (is_im == 0) {
                    if (beta != 0.0f) val += beta * C_re_batch[out_idx];
                    C_re_batch[out_idx] = val;
                } else {
                    if (beta != 0.0f) val += beta * C_im_batch[out_idx];
                    C_im_batch[out_idx] = val;
                }
            }
        }

        if (Tile::STORE_PHASES > 1) __syncthreads();
    }
}

// Store with power detection: Re(C)^2 + Im(C)^2 → single FP32 buffer.
// Uses phased column processing to bound SMEM usage.
template <typename Tile>
__device__ __forceinline__ void store_gemm_power_t(
    float acc[Tile::FYS_PER_WARP][Tile::FRAGS_X][4],
    float* __restrict__ C_power_batch,
    char* __restrict__ smem_raw,
    unsigned firstRow, unsigned firstCol,
    int M, int N, float alpha, float beta,
    int tid, int warp_id, int lane)
{
    float* smem_store = reinterpret_cast<float*>(smem_raw);

    // Process 32 columns per phase
    #pragma unroll
    for (int phase = 0; phase < Tile::STORE_PHASES; phase++) {
        int fx_start = phase * 4;

        // Phase 1: Scatter accumulators to SMEM
        #pragma unroll
        for (int fy_local = 0; fy_local < Tile::FYS_PER_WARP; fy_local++) {
            int fy = warp_id + fy_local * Tile::NUM_WARPS;
            #pragma unroll
            for (int fx_off = 0; fx_off < 4 && fx_start + fx_off < Tile::FRAGS_X; fx_off++) {
                int fx = fx_start + fx_off;
                int mma_row0 = lane / 4;
                int mma_row1 = mma_row0 + 8;
                int mma_col0 = (lane % 4) * 2;
                int mma_col1 = mma_col0 + 1;

                float d_vals[4] = { acc[fy_local][fx][0], acc[fy_local][fx][1], acc[fy_local][fx][2], acc[fy_local][fx][3] };
                int mma_rows[4] = { mma_row0, mma_row0, mma_row1, mma_row1 };
                int mma_cols[4] = { mma_col0, mma_col1, mma_col0, mma_col1 };

                #pragma unroll
                for (int vi = 0; vi < 4; vi++) {
                    int mr = mma_rows[vi];
                    int mc = mma_cols[vi];
                    int recv_local_row = mr / 2;
                    int is_im = mr & 1;
                    int recv_local_col = mc;

                    int tile_row = fy * Tile::RECV_PER_MMA_Y + recv_local_row;
                    int local_col = fx_off * Tile::RECV_PER_MMA_X + recv_local_col;
                    smem_store[tile_row * 64 + local_col * 2 + is_im] = d_vals[vi];
                }
            }
        }

        __syncthreads();

        // Phase 2: Read Re/Im pairs from SMEM, compute power, write coalesced.
        for (int r = 0; r < Tile::BLOCK_M; r++) {
            int my_col = tid;
            if (my_col >= 32) continue;

            unsigned row = firstRow + r;
            unsigned col = firstCol + phase * 32 + my_col;

            if (row < (unsigned)M && col < (unsigned)N) {
                float re_val = smem_store[r * 64 + my_col * 2 + 0];
                float im_val = smem_store[r * 64 + my_col * 2 + 1];
                float power = alpha * (re_val * re_val + im_val * im_val);

                int64_t out_idx = static_cast<int64_t>(row) * N + col;
                if (beta != 0.0f) power += beta * C_power_batch[out_idx];
                C_power_batch[out_idx] = power;
            }
        }

        if (Tile::STORE_PHASES > 1) __syncthreads();
    }
}

// Store to planar complex output with FP16 conversion: separate Re(C) and Im(C) __half buffers.
// Same SMEM staging pattern as FP32 variant, but final write converts via __float2half().
template <typename Tile>
__device__ __forceinline__ void store_gemm_planar_fp16_t(
    float acc[Tile::FYS_PER_WARP][Tile::FRAGS_X][4],
    __half* __restrict__ C_re_batch,
    __half* __restrict__ C_im_batch,
    char* __restrict__ smem_raw,
    unsigned firstRow, unsigned firstCol,
    int M, int N, float alpha, float beta,
    int tid, int warp_id, int lane)
{
    float* smem_store = reinterpret_cast<float*>(smem_raw);

    #pragma unroll
    for (int phase = 0; phase < Tile::STORE_PHASES; phase++) {
        int fx_start = phase * 4;

        // Phase 1: Scatter accumulators to SMEM tile (same as FP32 variant)
        #pragma unroll
        for (int fy_local = 0; fy_local < Tile::FYS_PER_WARP; fy_local++) {
            int fy = warp_id + fy_local * Tile::NUM_WARPS;
            #pragma unroll
            for (int fx_off = 0; fx_off < 4 && fx_start + fx_off < Tile::FRAGS_X; fx_off++) {
                int fx = fx_start + fx_off;
                int mma_row0 = lane / 4;
                int mma_row1 = mma_row0 + 8;
                int mma_col0 = (lane % 4) * 2;
                int mma_col1 = mma_col0 + 1;

                float d_vals[4] = { acc[fy_local][fx][0], acc[fy_local][fx][1], acc[fy_local][fx][2], acc[fy_local][fx][3] };
                int mma_rows[4] = { mma_row0, mma_row0, mma_row1, mma_row1 };
                int mma_cols[4] = { mma_col0, mma_col1, mma_col0, mma_col1 };

                #pragma unroll
                for (int vi = 0; vi < 4; vi++) {
                    int mr = mma_rows[vi];
                    int mc = mma_cols[vi];
                    int recv_local_row = mr / 2;
                    int is_im = mr & 1;
                    int recv_local_col = mc;

                    int tile_row = fy * Tile::RECV_PER_MMA_Y + recv_local_row;
                    int local_col = fx_off * Tile::RECV_PER_MMA_X + recv_local_col;
                    smem_store[tile_row * 64 + local_col * 2 + is_im] = d_vals[vi];
                }
            }
        }

        __syncthreads();

        // Phase 2: Coalesced row-by-row flush to global memory with FP16 conversion.
        for (int r = 0; r < Tile::BLOCK_M; r += 2) {
            int my_row = r + (tid / 64);
            int my_elem = tid % 64;
            int local_col = my_elem / 2;
            int is_im = my_elem & 1;

            unsigned row = firstRow + my_row;
            unsigned col = firstCol + phase * 32 + local_col;

            if (row < (unsigned)M && col < (unsigned)N) {
                float val = smem_store[my_row * 64 + my_elem];
                val *= alpha;

                int64_t out_idx = static_cast<int64_t>(row) * N + col;

                if (is_im == 0) {
                    if (beta != 0.0f) val += beta * __half2float(C_re_batch[out_idx]);
                    C_re_batch[out_idx] = __float2half(val);
                } else {
                    if (beta != 0.0f) val += beta * __half2float(C_im_batch[out_idx]);
                    C_im_batch[out_idx] = __float2half(val);
                }
            }
        }

        if (Tile::STORE_PHASES > 1) __syncthreads();
    }
}

// Direct vectorized FP8 tile load for GEMM (small-K optimization).
// Loads a 32-row tile using uint4 vectorized reads instead of cp.async pipeline.
// Used by the persistent kernel when K <= K_CHUNK.
template <typename Traits>
__device__ __forceinline__ void gemm_load_tile_direct(
    const unsigned char* __restrict__ data_fp8,   // [total_rows, 2*K] interleaved FP8
    unsigned char* __restrict__ smem,
    unsigned firstRow,
    int kt, int K, int total_rows,
    int tid)
{
    int recv = tid / 4;
    int k_group = tid % 4;
    unsigned abs_row = firstRow + recv;

    constexpr int BYTES_PER_GROUP = Traits::K_PER_GROUP * 2;
    constexpr int K_PER_LOAD = 8;
    unsigned char* smem_dst = &smem[recv * Traits::SMEM_K + k_group * BYTES_PER_GROUP];

    if (abs_row < (unsigned)total_rows && kt + k_group * Traits::K_PER_GROUP < K) {
        int k_start = kt + k_group * Traits::K_PER_GROUP;
        int64_t global_offset = static_cast<int64_t>(abs_row) * 2 * K
                              + static_cast<int64_t>(k_start) * 2;
        const unsigned char* gmem_ptr = data_fp8 + global_offset;

        #pragma unroll
        for (int i = 0; i < BYTES_PER_GROUP / 16; i++) {
            if (k_start + i * K_PER_LOAD < K) {
                *reinterpret_cast<uint4*>(smem_dst + i * 16) =
                    *reinterpret_cast<const uint4*>(gmem_ptr + i * 16);
            } else {
                *reinterpret_cast<uint4*>(smem_dst + i * 16) = make_uint4(0, 0, 0, 0);
            }
        }
    } else {
        unsigned* dst = reinterpret_cast<unsigned*>(smem_dst);
        #pragma unroll
        for (int i = 0; i < BYTES_PER_GROUP / 4; i++) dst[i] = 0;
    }
}

// ========================================================================================
// Templated cp.async tile load for GEMM
// ========================================================================================
//
// Loads one 32-row pass of a tile from global FP8 to shared via cp.async.
// Called multiple times for tiles wider than 32 rows (A_LOAD_PASSES or B_LOAD_PASSES).
// smem_offset: byte offset into the tile SMEM for this pass (pass_idx * 32 * SMEM_K).
//
template <typename Traits>
__device__ __forceinline__ void gemm_load_tile_cp_async(
    const unsigned char* __restrict__ data_fp8,   // [total_rows, 2*K] interleaved FP8
    unsigned char* __restrict__ smem,
    unsigned firstRow,
    int kt, int K, int total_rows,
    int tid)
{
    int recv = tid / 4;
    int k_group = tid % 4;
    unsigned abs_row = firstRow + recv;

    constexpr int BYTES_PER_GROUP = Traits::K_PER_GROUP * 2;
    constexpr int K_PER_CP_ASYNC = 8;
    unsigned char* smem_dst = &smem[recv * Traits::SMEM_K + k_group * BYTES_PER_GROUP];

    if (abs_row < (unsigned)total_rows && kt + k_group * Traits::K_PER_GROUP < K) {
        int k_start = kt + k_group * Traits::K_PER_GROUP;
        int64_t global_offset = static_cast<int64_t>(abs_row) * 2 * K
                              + static_cast<int64_t>(k_start) * 2;
        const unsigned char* gmem_ptr = data_fp8 + global_offset;

        #pragma unroll
        for (int i = 0; i < Traits::CP_ASYNC_PER_THREAD; i++) {
            if (k_start + i * K_PER_CP_ASYNC < K) {
                cp_async_cg_16(smem_dst + i * 16, gmem_ptr + i * 16);
            } else {
                unsigned* dst = reinterpret_cast<unsigned*>(smem_dst + i * 16);
                dst[0] = 0; dst[1] = 0; dst[2] = 0; dst[3] = 0;
            }
        }
    } else {
        unsigned* dst = reinterpret_cast<unsigned*>(smem_dst);
        #pragma unroll
        for (int i = 0; i < BYTES_PER_GROUP / 4; i++) dst[i] = 0;
    }
}

// ========================================================================================
// Templated compute body for GEMM — generalized warp mapping
// ========================================================================================
//
// For BLOCK_M=32 (FRAGS_Y=4): each warp handles exactly 1 fy (backward-compatible).
// For BLOCK_M=64 (FRAGS_Y=8): each warp handles 2 fy values (strided by NUM_WARPS=4).
// A fragments are loaded from the appropriate offset in the wider A tile.
// B fragments are loaded from the appropriate offset in the wider B tile.
//
template <typename Traits, typename Tile>
__device__ __forceinline__ void compute_gemm_mma_block_t(
    unsigned char* __restrict__ smem_a,
    unsigned char* __restrict__ smem_b,
    float acc[Tile::FYS_PER_WARP][Tile::FRAGS_X][4],
    int warp_id, int lane)
{
    #pragma unroll
    for (int ks = 0; ks < Traits::K_SUBS; ks++) {
        int k_base = ks * 32;

        #pragma unroll
        for (int fy_local = 0; fy_local < Tile::FYS_PER_WARP; fy_local++) {
            int fy = warp_id + fy_local * Tile::NUM_WARPS;
            int recv_base_y = fy * Tile::RECV_PER_MMA_Y;

            // Load A fragment with conj_perm trick
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

                unsigned v0 = *reinterpret_cast<const unsigned*>(&smem_a[smem_recv0 * Traits::SMEM_K + k_off0]);
                unsigned v1 = *reinterpret_cast<const unsigned*>(&smem_a[smem_recv0 * Traits::SMEM_K + k_off1]);
                a0 = is_conj ? conj_perm_fp8(v0) : v0;
                a2 = is_conj ? conj_perm_fp8(v1) : v1;

                if (smem_recv1 < Tile::BLOCK_M) {
                    unsigned v2 = *reinterpret_cast<const unsigned*>(&smem_a[smem_recv1 * Traits::SMEM_K + k_off0]);
                    unsigned v3 = *reinterpret_cast<const unsigned*>(&smem_a[smem_recv1 * Traits::SMEM_K + k_off1]);
                    a1 = is_conj ? conj_perm_fp8(v2) : v2;
                    a3 = is_conj ? conj_perm_fp8(v3) : v3;
                } else {
                    a1 = 0; a3 = 0;
                }
            }

            // Load B fragments and issue MMAs
            #pragma unroll
            for (int fx = 0; fx < Tile::FRAGS_X; fx++) {
                int recv_base_x = fx * Tile::RECV_PER_MMA_X;
                unsigned b0, b1;
                {
                    int recv_idx = recv_base_x + lane / 4;
                    int k_off0 = k_base + (lane % 4) * 4;
                    int k_off1 = k_base + 16 + (lane % 4) * 4;

                    if (recv_idx < Tile::BLOCK_N) {
                        b0 = *reinterpret_cast<const unsigned*>(&smem_b[recv_idx * Traits::SMEM_K + k_off0]);
                        b1 = *reinterpret_cast<const unsigned*>(&smem_b[recv_idx * Traits::SMEM_K + k_off1]);
                    } else {
                        b0 = 0; b1 = 0;
                    }
                }

                mma_fp8_m16n8k32(
                    acc[fy_local][fx][0], acc[fy_local][fx][1], acc[fy_local][fx][2], acc[fy_local][fx][3],
                    a0, a1, a2, a3,
                    b0, b1,
                    acc[fy_local][fx][0], acc[fy_local][fx][1], acc[fy_local][fx][2], acc[fy_local][fx][3]);
            }
        }
    }
}

// ========================================================================================
// Direct Complex GEMM — Templated on Traits + Tile + OutputMode
// ========================================================================================
//
// A and B_neg are pre-cast to interleaved FP8 before kernel launch.
// B_neg has imaginary parts negated during precast.
// Uses NR_BUFS-buffer cp.async pipeline.
//
// Multi-pass tile loading: for tiles > 32 rows/cols, the load function is called
// multiple times per buffer with SMEM offsets for each 32-row pass.
//
template <typename Traits, typename Tile, GemmOutputMode OutputMode>
__global__ void __launch_bounds__(Tile::THREADS)
gemm_direct_fp8_kernel_t(
    const unsigned char* __restrict__ A_fp8,        // [batch, M, 2*K] interleaved FP8
    const unsigned char* __restrict__ B_neg_fp8,    // [batch, N, 2*K] interleaved FP8 (Im negated)
    float* __restrict__ C_re,                       // [batch, M, N] Re output (PlanarComplex)
    float* __restrict__ C_im,                       // [batch, M, N] Im output (nullptr for PowerDetect)
    int M, int N, int K,
    int batch_count,
    float alpha, float beta,
    int blocks_per_N)
{
    // ---- 1. Map block index to (blockRow, blockCol) — rectangular ----
    const unsigned block = blockIdx.x;
    const unsigned batch = blockIdx.y;

    const unsigned blockRow = block / blocks_per_N;
    const unsigned blockCol = block % blocks_per_N;

    const unsigned firstRow = blockRow * Tile::BLOCK_M;
    const unsigned firstCol = blockCol * Tile::BLOCK_N;

    const int tid = threadIdx.x + threadIdx.y * 32 + threadIdx.z * 64;
    const int warp_id = tid / 32;
    const int lane = tid % 32;

    // ---- 2. Shared memory: NR_BUFS-buffered, A + B tiles per buffer ----
    extern __shared__ char smem_raw[];
    constexpr int A_TILE_BYTES = Tile::BLOCK_M * Traits::SMEM_K;
    constexpr int B_TILE_BYTES = Tile::BLOCK_N * Traits::SMEM_K;
    constexpr int BUF_BYTES = A_TILE_BYTES + B_TILE_BYTES;

    unsigned char* smem_buf[Traits::NR_BUFS];
    smem_buf[0] = reinterpret_cast<unsigned char*>(smem_raw);
    for (int i = 1; i < Traits::NR_BUFS; i++)
        smem_buf[i] = smem_buf[i-1] + BUF_BYTES;

    // ---- 3. Initialize FP32 accumulators (per-warp sized to avoid register spill) ----
    // Each warp handles FYS_PER_WARP fy values with strided assignment.
    // Register counts: 32x32=16, 32x128=64, 64x64=64, 64x128=128 floats.
    float acc[Tile::FYS_PER_WARP][Tile::FRAGS_X][4];
    #pragma unroll
    for (int fy_local = 0; fy_local < Tile::FYS_PER_WARP; fy_local++)
        #pragma unroll
        for (int fx = 0; fx < Tile::FRAGS_X; fx++)
            acc[fy_local][fx][0] = acc[fy_local][fx][1] = acc[fy_local][fx][2] = acc[fy_local][fx][3] = 0.0f;

    // ---- 4. Batch pointers ----
    const int64_t A_batch_stride = static_cast<int64_t>(M) * K * 2;
    const int64_t B_batch_stride = static_cast<int64_t>(N) * K * 2;
    const unsigned char* A_batch = A_fp8 + batch * A_batch_stride;
    const unsigned char* B_batch = B_neg_fp8 + batch * B_batch_stride;

    // ---- 5. Multi-buffer cp.async pipeline ----
    const int num_chunks = (K + Traits::K_CHUNK - 1) / Traits::K_CHUNK;

    // Prologue: issue READ_AHEAD loads
    for (int p = 0; p < Traits::READ_AHEAD && p < num_chunks; ++p) {
        unsigned char* buf = smem_buf[p];
        unsigned char* smem_a = buf;
        unsigned char* smem_b = buf + A_TILE_BYTES;

        int kt_p = p * Traits::K_CHUNK;

        // Multi-pass A load (A_LOAD_PASSES × 32 rows)
        #pragma unroll
        for (int ap = 0; ap < Tile::A_LOAD_PASSES; ap++) {
            gemm_load_tile_cp_async<Traits>(
                A_batch, smem_a + ap * 32 * Traits::SMEM_K,
                firstRow + ap * 32, kt_p, K, M, tid);
        }
        // Multi-pass B load (B_LOAD_PASSES × 32 rows)
        #pragma unroll
        for (int bp = 0; bp < Tile::B_LOAD_PASSES; bp++) {
            gemm_load_tile_cp_async<Traits>(
                B_batch, smem_b + bp * 32 * Traits::SMEM_K,
                firstCol + bp * 32, kt_p, K, N, tid);
        }
        cp_async_commit();
    }

    // Main loop
    for (int kt_idx = 0; kt_idx < num_chunks; ++kt_idx) {
        int buf_idx = kt_idx % Traits::NR_BUFS;

        // Wait for current buffer to be ready
        if (kt_idx + 1 < num_chunks) {
            cp_async_wait<Traits::READ_AHEAD - 1>();
        } else {
            cp_async_wait<0>();  // Last chunk: drain pipeline
        }
        __syncthreads();

        // Issue next prefetch
        int next_p = kt_idx + Traits::READ_AHEAD;
        if (next_p < num_chunks) {
            int next_buf = next_p % Traits::NR_BUFS;
            unsigned char* buf = smem_buf[next_buf];
            unsigned char* smem_a_next = buf;
            unsigned char* smem_b_next = buf + A_TILE_BYTES;

            int kt_next = next_p * Traits::K_CHUNK;

            #pragma unroll
            for (int ap = 0; ap < Tile::A_LOAD_PASSES; ap++) {
                gemm_load_tile_cp_async<Traits>(
                    A_batch, smem_a_next + ap * 32 * Traits::SMEM_K,
                    firstRow + ap * 32, kt_next, K, M, tid);
            }
            #pragma unroll
            for (int bp = 0; bp < Tile::B_LOAD_PASSES; bp++) {
                gemm_load_tile_cp_async<Traits>(
                    B_batch, smem_b_next + bp * 32 * Traits::SMEM_K,
                    firstCol + bp * 32, kt_next, K, N, tid);
            }
            cp_async_commit();
        }

        // Compute on current buffer
        unsigned char* smem_a = smem_buf[buf_idx];
        unsigned char* smem_b = smem_a + A_TILE_BYTES;

        compute_gemm_mma_block_t<Traits, Tile>(smem_a, smem_b, acc, warp_id, lane);

        __syncthreads();
    }

    // ---- 6. Store output ----
    const int64_t C_batch_stride = static_cast<int64_t>(M) * N;

    if constexpr (OutputMode == GemmOutputMode::PlanarComplex) {
        float* C_re_batch = C_re + batch * C_batch_stride;
        float* C_im_batch = C_im + batch * C_batch_stride;
        store_gemm_planar_t<Tile>(acc, C_re_batch, C_im_batch, smem_raw,
                                  firstRow, firstCol, M, N, alpha, beta,
                                  tid, warp_id, lane);
    } else if constexpr (OutputMode == GemmOutputMode::PlanarComplexFP16) {
        __half* C_re_batch = reinterpret_cast<__half*>(C_re) + batch * C_batch_stride;
        __half* C_im_batch = reinterpret_cast<__half*>(C_im) + batch * C_batch_stride;
        store_gemm_planar_fp16_t<Tile>(acc, C_re_batch, C_im_batch, smem_raw,
                                       firstRow, firstCol, M, N, alpha, beta,
                                       tid, warp_id, lane);
    } else {
        float* C_power_batch = C_re + batch * C_batch_stride;
        store_gemm_power_t<Tile>(acc, C_power_batch, smem_raw,
                                 firstRow, firstCol, M, N, alpha, beta,
                                 tid, warp_id, lane);
    }
}

// ========================================================================================
// Host-side launch function — templated on Traits + Tile + OutputMode
// ========================================================================================

template <typename Traits, typename Tile, GemmOutputMode OutputMode>
inline cutlass::Status launch_gemm_direct_t(
    const __nv_fp8_e4m3* A_fp8,
    const __nv_fp8_e4m3* B_neg_fp8,
    float* C_re, float* C_im,
    int M, int N, int K,
    int batch_count,
    float alpha, float beta,
    cudaStream_t stream)
{
    int blocks_per_M = (M + Tile::BLOCK_M - 1) / Tile::BLOCK_M;
    int blocks_per_N = (N + Tile::BLOCK_N - 1) / Tile::BLOCK_N;

    dim3 grid(blocks_per_M * blocks_per_N, batch_count, 1);
    dim3 block(32, 2, 2);  // 128 threads = 4 warps

    // SMEM: max of pipeline buffers vs store staging
    constexpr int A_TILE_BYTES = Tile::BLOCK_M * Traits::SMEM_K;
    constexpr int B_TILE_BYTES = Tile::BLOCK_N * Traits::SMEM_K;
    constexpr int PIPELINE_SMEM = Traits::NR_BUFS * (A_TILE_BYTES + B_TILE_BYTES);
    constexpr int STORE_SMEM = Tile::BLOCK_M * 64 * static_cast<int>(sizeof(float));
    int smem_bytes = PIPELINE_SMEM > STORE_SMEM ? PIPELINE_SMEM : STORE_SMEM;

    // Opt-in to extended dynamic SMEM if needed (default limit is 48KB on SM90)
    if (smem_bytes > 48 * 1024) {
        CUDA_CHECK(cudaFuncSetAttribute(
            gemm_direct_fp8_kernel_t<Traits, Tile, OutputMode>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            smem_bytes));
    }

    gemm_direct_fp8_kernel_t<Traits, Tile, OutputMode><<<grid, block, smem_bytes, stream>>>(
        reinterpret_cast<const unsigned char*>(A_fp8),
        reinterpret_cast<const unsigned char*>(B_neg_fp8),
        C_re, C_im,
        M, N, K, batch_count, alpha, beta, blocks_per_N);
    CUDA_CHECK(cudaGetLastError());

    return cutlass::Status::kSuccess;
}

// ========================================================================================
// Auto-selection dispatch — picks the largest tile that fits the problem
// ========================================================================================
//
// Selection priority (largest tile first for maximum arithmetic intensity):
//   M >= 64 && N >= 128 → 64×128 (AI=171)
//   M >= 64 && N >= 64  → 64×64  (AI=128)
//   N >= 128            → 32×128 (AI=102)
//   fallback            → 32×32  (AI=64)
//

template <GemmOutputMode OutputMode>
inline cutlass::Status launch_gemm_direct_auto(
    const __nv_fp8_e4m3* A_fp8,
    const __nv_fp8_e4m3* B_neg_fp8,
    float* C_re, float* C_im,
    int M, int N, int K,
    int batch_count,
    float alpha, float beta,
    cudaStream_t stream)
{
    if (M >= 64 && N >= 128) {
        return launch_gemm_direct_t<TraitsK64_B3, GemmTile64x128, OutputMode>(
            A_fp8, B_neg_fp8, C_re, C_im, M, N, K, batch_count, alpha, beta, stream);
    } else if (M >= 64 && N >= 64) {
        return launch_gemm_direct_t<TraitsK64_B3, GemmTile64x64, OutputMode>(
            A_fp8, B_neg_fp8, C_re, C_im, M, N, K, batch_count, alpha, beta, stream);
    } else if (N >= 128) {
        return launch_gemm_direct_t<TraitsK64_B3, GemmTile32x128, OutputMode>(
            A_fp8, B_neg_fp8, C_re, C_im, M, N, K, batch_count, alpha, beta, stream);
    } else {
        return launch_gemm_direct_t<TraitsK64_B3, GemmTile32x32, OutputMode>(
            A_fp8, B_neg_fp8, C_re, C_im, M, N, K, batch_count, alpha, beta, stream);
    }
}

// ========================================================================================
// Convenience aliases — default auto-selection (backward-compatible API)
// ========================================================================================

inline cutlass::Status launch_gemm_direct_planar(
    const __nv_fp8_e4m3* A_fp8, const __nv_fp8_e4m3* B_neg_fp8,
    float* C_re, float* C_im,
    int M, int N, int K, int batch_count,
    float alpha, float beta, cudaStream_t stream)
{
    return launch_gemm_direct_auto<GemmOutputMode::PlanarComplex>(
        A_fp8, B_neg_fp8, C_re, C_im, M, N, K, batch_count, alpha, beta, stream);
}

inline cutlass::Status launch_gemm_direct_power(
    const __nv_fp8_e4m3* A_fp8, const __nv_fp8_e4m3* B_neg_fp8,
    float* C_power,
    int M, int N, int K, int batch_count,
    float alpha, float beta, cudaStream_t stream)
{
    return launch_gemm_direct_auto<GemmOutputMode::PowerDetect>(
        A_fp8, B_neg_fp8, C_power, nullptr, M, N, K, batch_count, alpha, beta, stream);
}

// ========================================================================================
// Persistent Direct Complex GEMM Kernel
// ========================================================================================
//
// Persistent variant of the direct GEMM kernel. Launches sm_count * 4 blocks
// that loop over work items, eliminating block scheduling overhead.
//
// Three optimizations (ported from persistent HERK kernel):
//   1. Persistent loop: gridDim.x = min(sm_count * 4, total_work)
//   2. Small-K direct load (K <= K_CHUNK): single SMEM buffer with vectorized
//      uint4 loads instead of 3-buffer cp.async pipeline
//   3. Batch grouping (BATCH_GROUP=4): amortizes grid index computation
//
// Work decomposition: total_work_groups = ceil(batch/BATCH_GROUP) * blocks_per_M * blocks_per_N
// Each work item is a (batch_group, blockRow, blockCol) triple.
//
template <typename Traits, typename Tile, GemmOutputMode OutputMode, int BATCH_GROUP_ = 4>
__global__ void __launch_bounds__(Tile::THREADS)
gemm_direct_fp8_persistent_kernel_t(
    const unsigned char* __restrict__ A_fp8,        // [batch, M, 2*K] interleaved FP8
    const unsigned char* __restrict__ B_neg_fp8,    // [batch, N, 2*K] interleaved FP8 (Im negated)
    float* __restrict__ C_re,                       // Output Re (or power)
    float* __restrict__ C_im,                       // Output Im (nullptr for PowerDetect)
    int M, int N, int K,
    int batch_count,
    float alpha, float beta,
    int blocks_per_N, int total_work_items)
{
    constexpr int A_TILE_BYTES = Tile::BLOCK_M * Traits::SMEM_K;
    constexpr int B_TILE_BYTES = Tile::BLOCK_N * Traits::SMEM_K;

    extern __shared__ char smem_raw[];

    const int tid = threadIdx.x + threadIdx.y * 32 + threadIdx.z * 64;
    const int warp_id = tid / 32;
    const int lane = tid % 32;

    const int64_t A_batch_stride = static_cast<int64_t>(M) * K * 2;
    const int64_t B_batch_stride = static_cast<int64_t>(N) * K * 2;
    const int64_t C_batch_stride = static_cast<int64_t>(M) * N;

    const int num_chunks = (K + Traits::K_CHUNK - 1) / Traits::K_CHUNK;
    const bool small_k = (K <= Traits::K_CHUNK);

    const int blocks_per_M = (M + Tile::BLOCK_M - 1) / Tile::BLOCK_M;
    const int mn_blocks = blocks_per_M * blocks_per_N;
    const int total_work_groups = ((batch_count + BATCH_GROUP_ - 1) / BATCH_GROUP_) * mn_blocks;

    for (int work = blockIdx.x; work < total_work_groups; work += gridDim.x) {
        int group = work / mn_blocks;
        int mn_block = work % mn_blocks;
        int batch_base = group * BATCH_GROUP_;

        const unsigned blockRow = mn_block / blocks_per_N;
        const unsigned blockCol = mn_block % blocks_per_N;
        const unsigned firstRow = blockRow * Tile::BLOCK_M;
        const unsigned firstCol = blockCol * Tile::BLOCK_N;

        for (int bi = 0; bi < BATCH_GROUP_ && batch_base + bi < batch_count; ++bi) {
            int batch = batch_base + bi;

            const unsigned char* A_batch = A_fp8 + batch * A_batch_stride;
            const unsigned char* B_batch = B_neg_fp8 + batch * B_batch_stride;

            // Initialize accumulators
            float acc[Tile::FYS_PER_WARP][Tile::FRAGS_X][4];
            #pragma unroll
            for (int fy_local = 0; fy_local < Tile::FYS_PER_WARP; fy_local++)
                #pragma unroll
                for (int fx = 0; fx < Tile::FRAGS_X; fx++)
                    acc[fy_local][fx][0] = acc[fy_local][fx][1] = acc[fy_local][fx][2] = acc[fy_local][fx][3] = 0.0f;

            if (small_k) {
                // Small-K path: single-buffer direct load (no cp.async pipeline)
                unsigned char* smem_a = reinterpret_cast<unsigned char*>(smem_raw);
                unsigned char* smem_b = smem_a + A_TILE_BYTES;

                #pragma unroll
                for (int ap = 0; ap < Tile::A_LOAD_PASSES; ap++) {
                    gemm_load_tile_direct<Traits>(
                        A_batch, smem_a + ap * 32 * Traits::SMEM_K,
                        firstRow + ap * 32, 0, K, M, tid);
                }
                #pragma unroll
                for (int bp = 0; bp < Tile::B_LOAD_PASSES; bp++) {
                    gemm_load_tile_direct<Traits>(
                        B_batch, smem_b + bp * 32 * Traits::SMEM_K,
                        firstCol + bp * 32, 0, K, N, tid);
                }
                __syncthreads();

                compute_gemm_mma_block_t<Traits, Tile>(smem_a, smem_b, acc, warp_id, lane);

                __syncthreads();
            } else {
                // Large-K path: multi-buffer cp.async pipeline
                constexpr int BUF_BYTES = A_TILE_BYTES + B_TILE_BYTES;
                unsigned char* smem_buf[Traits::NR_BUFS];
                smem_buf[0] = reinterpret_cast<unsigned char*>(smem_raw);
                for (int i = 1; i < Traits::NR_BUFS; i++)
                    smem_buf[i] = smem_buf[i-1] + BUF_BYTES;

                // Prologue: issue READ_AHEAD loads
                for (int p = 0; p < Traits::READ_AHEAD && p < num_chunks; ++p) {
                    unsigned char* buf = smem_buf[p];
                    unsigned char* smem_a = buf;
                    unsigned char* smem_b = buf + A_TILE_BYTES;
                    int kt_p = p * Traits::K_CHUNK;

                    #pragma unroll
                    for (int ap = 0; ap < Tile::A_LOAD_PASSES; ap++) {
                        gemm_load_tile_cp_async<Traits>(
                            A_batch, smem_a + ap * 32 * Traits::SMEM_K,
                            firstRow + ap * 32, kt_p, K, M, tid);
                    }
                    #pragma unroll
                    for (int bp = 0; bp < Tile::B_LOAD_PASSES; bp++) {
                        gemm_load_tile_cp_async<Traits>(
                            B_batch, smem_b + bp * 32 * Traits::SMEM_K,
                            firstCol + bp * 32, kt_p, K, N, tid);
                    }
                    cp_async_commit();
                }

                // Main K-loop
                for (int kt_idx = 0; kt_idx < num_chunks; ++kt_idx) {
                    int buf_idx = kt_idx % Traits::NR_BUFS;

                    if (kt_idx + 1 < num_chunks) {
                        cp_async_wait<Traits::READ_AHEAD - 1>();
                    } else {
                        cp_async_wait<0>();
                    }
                    __syncthreads();

                    int next_p = kt_idx + Traits::READ_AHEAD;
                    if (next_p < num_chunks) {
                        int next_buf = next_p % Traits::NR_BUFS;
                        unsigned char* buf = smem_buf[next_buf];
                        unsigned char* smem_a_next = buf;
                        unsigned char* smem_b_next = buf + A_TILE_BYTES;
                        int kt_next = next_p * Traits::K_CHUNK;

                        #pragma unroll
                        for (int ap = 0; ap < Tile::A_LOAD_PASSES; ap++) {
                            gemm_load_tile_cp_async<Traits>(
                                A_batch, smem_a_next + ap * 32 * Traits::SMEM_K,
                                firstRow + ap * 32, kt_next, K, M, tid);
                        }
                        #pragma unroll
                        for (int bp = 0; bp < Tile::B_LOAD_PASSES; bp++) {
                            gemm_load_tile_cp_async<Traits>(
                                B_batch, smem_b_next + bp * 32 * Traits::SMEM_K,
                                firstCol + bp * 32, kt_next, K, N, tid);
                        }
                        cp_async_commit();
                    }

                    unsigned char* smem_a = smem_buf[buf_idx];
                    unsigned char* smem_b = smem_a + A_TILE_BYTES;
                    compute_gemm_mma_block_t<Traits, Tile>(smem_a, smem_b, acc, warp_id, lane);

                    __syncthreads();
                }
            }

            // Store output
            if constexpr (OutputMode == GemmOutputMode::PlanarComplex) {
                float* C_re_batch = C_re + batch * C_batch_stride;
                float* C_im_batch = C_im + batch * C_batch_stride;
                store_gemm_planar_t<Tile>(acc, C_re_batch, C_im_batch, smem_raw,
                                          firstRow, firstCol, M, N, alpha, beta,
                                          tid, warp_id, lane);
            } else if constexpr (OutputMode == GemmOutputMode::PlanarComplexFP16) {
                __half* C_re_batch = reinterpret_cast<__half*>(C_re) + batch * C_batch_stride;
                __half* C_im_batch = reinterpret_cast<__half*>(C_im) + batch * C_batch_stride;
                store_gemm_planar_fp16_t<Tile>(acc, C_re_batch, C_im_batch, smem_raw,
                                               firstRow, firstCol, M, N, alpha, beta,
                                               tid, warp_id, lane);
            } else {
                float* C_power_batch = C_re + batch * C_batch_stride;
                store_gemm_power_t<Tile>(acc, C_power_batch, smem_raw,
                                         firstRow, firstCol, M, N, alpha, beta,
                                         tid, warp_id, lane);
            }

            __syncthreads();
        }
    }
}

// ========================================================================================
// Host-side persistent launch function
// ========================================================================================

template <typename Traits, typename Tile, GemmOutputMode OutputMode>
inline cutlass::Status launch_gemm_direct_persistent_t(
    const __nv_fp8_e4m3* A_fp8,
    const __nv_fp8_e4m3* B_neg_fp8,
    float* C_re, float* C_im,
    int M, int N, int K,
    int batch_count,
    float alpha, float beta,
    int sm_count,
    cudaStream_t stream)
{
    int blocks_per_M = (M + Tile::BLOCK_M - 1) / Tile::BLOCK_M;
    int blocks_per_N = (N + Tile::BLOCK_N - 1) / Tile::BLOCK_N;
    int mn_blocks = blocks_per_M * blocks_per_N;
    int total_work = mn_blocks * batch_count;

    constexpr int TARGET_OCCUPANCY = 4;
    int grid_x = std::min(sm_count * TARGET_OCCUPANCY, total_work);
    if (grid_x <= 0) return cutlass::Status::kSuccess;

    dim3 grid(grid_x, 1, 1);
    dim3 block(32, 2, 2);  // 128 threads

    constexpr int A_TILE_BYTES = Tile::BLOCK_M * Traits::SMEM_K;
    constexpr int B_TILE_BYTES = Tile::BLOCK_N * Traits::SMEM_K;
    constexpr int PIPELINE_SMEM = Traits::NR_BUFS * (A_TILE_BYTES + B_TILE_BYTES);
    constexpr int SINGLE_BUF_SMEM = A_TILE_BYTES + B_TILE_BYTES;
    constexpr int STORE_SMEM = Tile::BLOCK_M * 64 * static_cast<int>(sizeof(float));

    // K-gated SMEM: small-K uses single buffer, large-K uses full pipeline
    int smem_bytes;
    if (K <= Traits::K_CHUNK) {
        smem_bytes = SINGLE_BUF_SMEM > STORE_SMEM ? SINGLE_BUF_SMEM : STORE_SMEM;
    } else {
        smem_bytes = PIPELINE_SMEM > STORE_SMEM ? PIPELINE_SMEM : STORE_SMEM;
    }

    if (smem_bytes > 48 * 1024) {
        CUDA_CHECK(cudaFuncSetAttribute(
            gemm_direct_fp8_persistent_kernel_t<Traits, Tile, OutputMode>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            smem_bytes));
    }

    gemm_direct_fp8_persistent_kernel_t<Traits, Tile, OutputMode><<<grid, block, smem_bytes, stream>>>(
        reinterpret_cast<const unsigned char*>(A_fp8),
        reinterpret_cast<const unsigned char*>(B_neg_fp8),
        C_re, C_im,
        M, N, K, batch_count, alpha, beta, blocks_per_N, total_work);
    CUDA_CHECK(cudaGetLastError());

    return cutlass::Status::kSuccess;
}

// Auto-selection dispatch for persistent kernel (same tile priority as non-persistent)
template <GemmOutputMode OutputMode>
inline cutlass::Status launch_gemm_direct_persistent_auto(
    const __nv_fp8_e4m3* A_fp8,
    const __nv_fp8_e4m3* B_neg_fp8,
    float* C_re, float* C_im,
    int M, int N, int K,
    int batch_count,
    float alpha, float beta,
    int sm_count,
    cudaStream_t stream)
{
    if (M >= 64 && N >= 128) {
        return launch_gemm_direct_persistent_t<TraitsK64_B3, GemmTile64x128, OutputMode>(
            A_fp8, B_neg_fp8, C_re, C_im, M, N, K, batch_count, alpha, beta, sm_count, stream);
    } else if (M >= 64 && N >= 64) {
        return launch_gemm_direct_persistent_t<TraitsK64_B3, GemmTile64x64, OutputMode>(
            A_fp8, B_neg_fp8, C_re, C_im, M, N, K, batch_count, alpha, beta, sm_count, stream);
    } else if (N >= 128) {
        return launch_gemm_direct_persistent_t<TraitsK64_B3, GemmTile32x128, OutputMode>(
            A_fp8, B_neg_fp8, C_re, C_im, M, N, K, batch_count, alpha, beta, sm_count, stream);
    } else {
        return launch_gemm_direct_persistent_t<TraitsK64_B3, GemmTile32x32, OutputMode>(
            A_fp8, B_neg_fp8, C_re, C_im, M, N, K, batch_count, alpha, beta, sm_count, stream);
    }
}

// Convenience aliases for persistent variants
inline cutlass::Status launch_gemm_direct_persistent_planar(
    const __nv_fp8_e4m3* A_fp8, const __nv_fp8_e4m3* B_neg_fp8,
    float* C_re, float* C_im,
    int M, int N, int K, int batch_count,
    float alpha, float beta, int sm_count, cudaStream_t stream)
{
    return launch_gemm_direct_persistent_auto<GemmOutputMode::PlanarComplex>(
        A_fp8, B_neg_fp8, C_re, C_im, M, N, K, batch_count, alpha, beta, sm_count, stream);
}

inline cutlass::Status launch_gemm_direct_persistent_power(
    const __nv_fp8_e4m3* A_fp8, const __nv_fp8_e4m3* B_neg_fp8,
    float* C_power,
    int M, int N, int K, int batch_count,
    float alpha, float beta, int sm_count, cudaStream_t stream)
{
    return launch_gemm_direct_persistent_auto<GemmOutputMode::PowerDetect>(
        A_fp8, B_neg_fp8, C_power, nullptr, M, N, K, batch_count, alpha, beta, sm_count, stream);
}
