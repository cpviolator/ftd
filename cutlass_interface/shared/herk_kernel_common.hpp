// ========================================================================================
// Direct HERK Kernel — Shared Code for SM90 and SM100/SM120
// ========================================================================================
//
// Unified direct HERK kernel implementation. Pre-cast + cp.async approach:
//   Phase 1 (host): Pre-cast interleaved FP16 → interleaved FP8 (cast_fp16_to_fp8_e4m3)
//   Phase 2 (kernel): Load FP8 via cp.async global→shared, bypassing registers entirely
//
// The conjugate permutation trick (from TCC):
//   conj_perm(v) = __byte_perm(v ^ 0x00800080, 0, 0x2301)
//   Transforms interleaved FP8 [re, im] pairs into [im, -re].
//   Loading A fragment with alternating normal/conj_perm makes a single MMA
//   produce BOTH Re(C) and Im(C) simultaneously.
//
// Batch parallelism: gridDim.y = batch_count (all batches in one launch)
// Triangular indexing: gridDim.x = N/B * (N/B + 1) / 2 (lower triangle only)
//
// Constants:
//   BLOCK_N = 32, K_CHUNK = 64, K_SUBS = 4, THREADS = 128, NR_BUFS = 3, READ_AHEAD = 2
//
// Shared memory: 3-buffer pipeline, 3 × 2 tiles × 4096 = 24,576 bytes (rectangle)
//   Occupancy 4: 24,576 × 4 = 98,304 bytes (fits SM120 99 KB / SM90 228 KB SMEM)
//
// PTX requirements: mma.sync.aligned.m16n8k32 (SM90+), cp.async.cg (SM80+)
//
// This is a textual include (no #pragma once) — included inside the arch-specific namespace.
// ========================================================================================

// ---- Conjugate permutation: [re, im] → [im, -re] in FP8 ----
__device__ __forceinline__ unsigned conj_perm_fp8(unsigned v) {
    return __byte_perm(v ^ 0x00800080u, 0u, 0x2301);
}

// ---- PTX MMA wrapper: m16n8k32 FP8 E4M3 → FP32 ----
__device__ __forceinline__ void mma_fp8_m16n8k32(
    float &d0, float &d1, float &d2, float &d3,
    unsigned a0, unsigned a1, unsigned a2, unsigned a3,
    unsigned b0, unsigned b1,
    float c0, float c1, float c2, float c3)
{
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
        "{%0, %1, %2, %3}, "
        "{%4, %5, %6, %7}, "
        "{%8, %9}, "
        "{%10, %11, %12, %13};\n"
        : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
          "r"(b0), "r"(b1),
          "f"(c0), "f"(c1), "f"(c2), "f"(c3)
    );
}

// ---- Vectorized FP16→FP8 cast: convert 8 FP16 values → 8 FP8, return as 2 × uint32 ----
__device__ __forceinline__ void cast_8xfp16_to_fp8(
    const uint4& src,   // 8 FP16 values as 128 bits
    unsigned& out0,     // first 4 FP8 values
    unsigned& out1)     // next 4 FP8 values
{
    const __half* h = reinterpret_cast<const __half*>(&src);
    unsigned char fp8[8];
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        __nv_fp8_e4m3 f = __nv_fp8_e4m3(h[i]);
        fp8[i] = *reinterpret_cast<unsigned char*>(&f);
    }
    out0 = fp8[0] | (fp8[1] << 8) | (fp8[2] << 16) | (fp8[3] << 24);
    out1 = fp8[4] | (fp8[5] << 8) | (fp8[6] << 16) | (fp8[7] << 24);
}

// ========================================================================================
// Kernel constants
// ========================================================================================

static constexpr int HERK_BLOCK_N = 32;
static constexpr int HERK_K_CHUNK = 64;
static constexpr int HERK_THREADS = 128;
static constexpr int HERK_RECV_PER_MMA_Y = 8;
static constexpr int HERK_RECV_PER_MMA_X = 8;
static constexpr int HERK_FRAGS_Y = HERK_BLOCK_N / HERK_RECV_PER_MMA_Y;  // 4
static constexpr int HERK_FRAGS_X = HERK_BLOCK_N / HERK_RECV_PER_MMA_X;  // 4
static constexpr int HERK_K_PER_MMA = 16;
static constexpr int HERK_K_SUBS = HERK_K_CHUNK / HERK_K_PER_MMA;  // 4
static constexpr int HERK_SMEM_K = HERK_K_CHUNK * 2;  // 128 FP8 per receiver per chunk

// Pipeline constants (legacy — match TraitsK64_B3 default)
static constexpr int HERK_NR_BUFS = 3;
static constexpr int HERK_READ_AHEAD = 2;

// ========================================================================================
// DirectHerkTraits — compile-time traits for templated direct HERK kernels
// ========================================================================================
template <int K_CHUNK_, int NR_BUFS_>
struct DirectHerkTraits {
    static constexpr int K_CHUNK    = K_CHUNK_;
    static constexpr int NR_BUFS    = NR_BUFS_;
    static constexpr int K_SUBS     = K_CHUNK / HERK_K_PER_MMA;
    static constexpr int SMEM_K     = K_CHUNK * 2;
    static constexpr int READ_AHEAD = NR_BUFS - 1;
    static constexpr int TILE_BYTES = HERK_BLOCK_N * SMEM_K;
    static constexpr int SMEM_RECT  = NR_BUFS * 2 * TILE_BYTES;
    static constexpr int CP_ASYNC_PER_THREAD = K_CHUNK / 32;
    static constexpr int K_PER_GROUP = K_CHUNK / 4;
};

using TraitsK32_B3  = DirectHerkTraits<32, 3>;
using TraitsK64_B2  = DirectHerkTraits<64, 2>;
using TraitsK64_B3  = DirectHerkTraits<64, 3>;
using TraitsK128_B2 = DirectHerkTraits<128, 2>;

// ---- cp.async PTX helpers (SM80+) ----

__device__ __forceinline__ void cp_async_cg_16(void* smem, const void* gmem) {
    unsigned addr = __cvta_generic_to_shared(smem);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(addr), "l"(gmem));
}

__device__ __forceinline__ void cp_async_commit() {
    asm volatile("cp.async.commit_group;\n");
}

template<int N>
__device__ __forceinline__ void cp_async_wait() {
    asm volatile("cp.async.wait_group %0;\n" :: "n"(N));
}

// ========================================================================================
// cp.async tile load helper
// ========================================================================================

__device__ __forceinline__ void load_tile_cp_async(
    const unsigned char* __restrict__ A_fp8_batch,
    unsigned char* __restrict__ smem,
    unsigned firstRecv,
    int kt, int K, int N,
    int tid)
{
    int recv = tid / 4;
    int k_group = tid % 4;

    unsigned abs_recv = firstRecv + recv;

    unsigned char* smem_dst = &smem[recv * HERK_SMEM_K + k_group * 32];

    if (abs_recv < (unsigned)N && kt + k_group * 16 < K) {
        int64_t global_offset = static_cast<int64_t>(abs_recv) * 2 * K
                              + static_cast<int64_t>(kt + k_group * 16) * 2;
        const unsigned char* gmem_ptr = A_fp8_batch + global_offset;

        cp_async_cg_16(smem_dst, gmem_ptr);
        cp_async_cg_16(smem_dst + 16, gmem_ptr + 16);
    } else {
        unsigned* dst = reinterpret_cast<unsigned*>(smem_dst);
        #pragma unroll
        for (int i = 0; i < 8; i++) dst[i] = 0;
    }
}

// ========================================================================================
// Output type conversion helpers (for templated FP16/FP32 output)
// ========================================================================================

template <typename T> __device__ __forceinline__ T float_to_out(float val);
template <> __device__ __forceinline__ __half float_to_out<__half>(float val) { return __float2half(val); }
template <> __device__ __forceinline__ float float_to_out<float>(float val) { return val; }

template <typename T> __device__ __forceinline__ float out_to_float(T val);
template <> __device__ __forceinline__ float out_to_float<__half>(__half val) { return __half2float(val); }
template <> __device__ __forceinline__ float out_to_float<float>(float val) { return val; }

// ========================================================================================
// Pre-cast + cp.async direct HERK kernel (templated on output type)
// ========================================================================================

template <typename OutputType>
__global__ void __launch_bounds__(HERK_THREADS, 4)
herk_direct_fp8_precast_kernel(
    const unsigned char* __restrict__ A_fp8,
    OutputType* __restrict__ C,
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

    // ---- 2. Shared memory: 3-buffer pipeline ----
    extern __shared__ char smem_raw[];
    constexpr int TILE_BYTES = HERK_BLOCK_N * HERK_SMEM_K;  // 4096 bytes
    const int tiles_per_buf = is_diagonal ? 1 : 2;

    unsigned char* smem_buf[HERK_NR_BUFS];
    smem_buf[0] = reinterpret_cast<unsigned char*>(smem_raw);
    smem_buf[1] = smem_buf[0] + tiles_per_buf * TILE_BYTES;
    smem_buf[2] = smem_buf[1] + tiles_per_buf * TILE_BYTES;

    // ---- 3. Initialize FP32 accumulators ----
    float acc[HERK_FRAGS_Y][HERK_FRAGS_X][4];
    #pragma unroll
    for (int fy = 0; fy < HERK_FRAGS_Y; fy++)
        #pragma unroll
        for (int fx = 0; fx < HERK_FRAGS_X; fx++)
            acc[fy][fx][0] = acc[fy][fx][1] = acc[fy][fx][2] = acc[fy][fx][3] = 0.0f;

    // ---- 4. Batch pointer ----
    const int64_t A_batch_stride = static_cast<int64_t>(N) * K * 2;
    const unsigned char* A_batch = A_fp8 + batch * A_batch_stride;

    // ---- 5. 3-buffer pipelined K-loop with cp.async ----
    const int num_chunks = (K + HERK_K_CHUNK - 1) / HERK_K_CHUNK;

    // Prologue: issue READ_AHEAD loads
    for (int p = 0; p < HERK_READ_AHEAD && p < num_chunks; ++p) {
        unsigned char* buf = smem_buf[p];
        unsigned char* smem_row = buf;
        unsigned char* smem_col = is_diagonal ? smem_row : (smem_row + TILE_BYTES);

        int kt_p = p * HERK_K_CHUNK;
        load_tile_cp_async(A_batch, smem_row, firstRecvRow, kt_p, K, N, tid);
        if (!is_diagonal) {
            load_tile_cp_async(A_batch, smem_col, firstRecvCol, kt_p, K, N, tid);
        }
        cp_async_commit();
    }

    // Main loop
    for (int kt_idx = 0; kt_idx < num_chunks; ++kt_idx) {
        int buf_idx = kt_idx % HERK_NR_BUFS;

        // Wait until this chunk's cp.async group is done.
        // For pipelined iterations (not the last), wait<1> leaves 1 pending (correct).
        // For the last chunk (or single-chunk K ≤ K_CHUNK), wait<0> drains all.
        // Without this fix, wait<1> with ≤ 1 pending is a no-op (1 ≤ 1 = true),
        // reading uninitialized shared memory.
        if (kt_idx + 1 < num_chunks) {
            cp_async_wait<HERK_READ_AHEAD - 1>();
        } else {
            cp_async_wait<0>();
        }
        __syncthreads();

        // Issue next prefetch if there are more chunks
        int next_p = kt_idx + HERK_READ_AHEAD;
        if (next_p < num_chunks) {
            int next_buf_idx = next_p % HERK_NR_BUFS;
            unsigned char* next_buf = smem_buf[next_buf_idx];
            unsigned char* smem_row_next = next_buf;
            unsigned char* smem_col_next = is_diagonal ? smem_row_next : (smem_row_next + TILE_BYTES);

            int kt_next = next_p * HERK_K_CHUNK;
            load_tile_cp_async(A_batch, smem_row_next, firstRecvRow, kt_next, K, N, tid);
            if (!is_diagonal) {
                load_tile_cp_async(A_batch, smem_col_next, firstRecvCol, kt_next, K, N, tid);
            }
            cp_async_commit();
        }

        // Compute 4 MMA sub-iterations on current buffer
        unsigned char* smem_row = smem_buf[buf_idx];
        unsigned char* smem_col = is_diagonal ? smem_row : (smem_row + TILE_BYTES);

        #pragma unroll
        for (int ks = 0; ks < HERK_K_SUBS; ks++) {
            int k_base = ks * 32;

            // CRITICAL: Use unrolled loop with constant fy index to allow nvcc to
            // decompose acc[fy][fx][4] with compile-time constant indexing.
            // Using "const int fy = warp_id" causes dynamic array indexing which
            // forces nvcc to keep all 64 accumulator floats live simultaneously,
            // exceeding the register budget and causing massive spills to local memory.
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
    }

    // ---- 6. Store: SMEM-staged coalesced writes to packed triangle ----
    // Reuse pipeline SMEM for a 32×64 float store tile.
    // Phase 1: Each warp scatters its accumulator fragments into SMEM.
    // Phase 2: All 128 threads write rows sequentially (coalesced).
    // This eliminates L2 write amplification from scattered per-element stores.
    const int64_t packed_batch_stride = static_cast<int64_t>(N) * (N + 1);
    OutputType* C_batch = C + batch * packed_batch_stride;

    float* smem_store = reinterpret_cast<float*>(smem_raw);

    // Phase 1: Scatter accumulators to SMEM tile [32 rows × 64 elements]
    // Layout: smem_store[tile_row * 64 + tile_col * 2 + is_im]
    #pragma unroll
    for (int fy = 0; fy < HERK_FRAGS_Y; fy++) {
        if (fy != warp_id) continue;

        #pragma unroll
        for (int fx = 0; fx < HERK_FRAGS_X; fx++) {
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

                int tile_row = fy * HERK_RECV_PER_MMA_Y + recv_local_row;
                int tile_col = fx * HERK_RECV_PER_MMA_X + recv_local_col;
                smem_store[tile_row * 64 + tile_col * 2 + is_im] = d_vals[vi];
            }
        }
    }

    __syncthreads();

    // Phase 2: Coalesced row-by-row flush to global memory.
    // 128 threads write 2 rows per iteration (64 elements per row).
    // All threads within a warp write to the same row → coalesced transactions.
    for (int r = 0; r < HERK_BLOCK_N; r += 2) {
        int my_row = r + (tid / 64);
        int my_elem = tid % 64;
        int tile_col = my_elem / 2;
        int is_im = my_elem & 1;

        unsigned recv_row = firstRecvRow + my_row;
        unsigned recv_col = firstRecvCol + tile_col;

        if (recv_row < (unsigned)N && recv_col < (unsigned)N && recv_row >= recv_col) {
            float val = smem_store[my_row * 64 + my_elem];

            if (recv_row == recv_col && is_im) val = 0.0f;
            val *= alpha;

            int64_t tri_idx = static_cast<int64_t>(recv_row) * (recv_row + 1) / 2 + recv_col;
            int64_t out_idx = tri_idx * 2 + is_im;

            if (beta != 0.0f) {
                val += beta * out_to_float<OutputType>(C_batch[out_idx]);
            }

            C_batch[out_idx] = float_to_out<OutputType>(val);
        }
    }
}

// ========================================================================================
// Host-side launch for pre-cast + cp.async kernel (templated on output type)
// ========================================================================================

template <typename OutputType>
inline cutlass::Status launch_herk_direct_fp8_precast(
    const __nv_fp8_e4m3* A_fp8,
    OutputType* C,
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

    // Shared memory: 3-buffer pipeline, max 2 tiles per buffer (rectangle case)
    int smem_bytes = HERK_NR_BUFS * 2 * HERK_BLOCK_N * HERK_SMEM_K;  // 24576 bytes

    herk_direct_fp8_precast_kernel<OutputType><<<grid, block, smem_bytes, stream>>>(
        reinterpret_cast<const unsigned char*>(A_fp8),
        C, N, K, batch_count, alpha, beta, blocks_per_dim);
    CUDA_CHECK(cudaGetLastError());

    return cutlass::Status::kSuccess;
}

// ========================================================================================
// Direct load helper: vectorized global → SMEM for small-K path (K ≤ K_CHUNK)
// ========================================================================================
//
// Replaces cp.async with regular global loads for single-chunk K.
// Uses 2 × 128-bit (uint4) loads per thread, writing 32 FP8 bytes to SMEM.
// Thread mapping: recv = tid / 4, k_group = tid % 4 (same as cp.async version).
//
__device__ __forceinline__ void load_tile_direct_fp8(
    const unsigned char* __restrict__ A_fp8_batch,
    unsigned char* __restrict__ smem,
    unsigned firstRecv,
    int kt, int K, int N,
    int tid)
{
    int recv = tid / 4;
    int k_group = tid % 4;

    unsigned abs_recv = firstRecv + recv;
    unsigned char* smem_dst = &smem[recv * HERK_SMEM_K + k_group * 32];

    if (abs_recv < (unsigned)N && kt + k_group * 16 < K) {
        int64_t offset = static_cast<int64_t>(abs_recv) * 2 * K
                       + static_cast<int64_t>(kt + k_group * 16) * 2;
        const unsigned char* src = A_fp8_batch + offset;
        *reinterpret_cast<uint4*>(smem_dst) = *reinterpret_cast<const uint4*>(src);
        *reinterpret_cast<uint4*>(smem_dst + 16) = *reinterpret_cast<const uint4*>(src + 16);
    } else {
        unsigned* dst = reinterpret_cast<unsigned*>(smem_dst);
        #pragma unroll
        for (int i = 0; i < 8; i++) dst[i] = 0;
    }
}

// ========================================================================================
// Persistent Direct HERK Kernel
// ========================================================================================
//
// Optimizations over the standard per-block precast kernel:
//   1. Persistent: sm_count × occupancy blocks loop over work items (eliminates
//      block scheduling overhead — dominant cost at small K with millions of blocks)
//   2. Small-K direct load: single SMEM buffer with vectorized loads when K ≤ K_CHUNK,
//      skipping the 3-buffer cp.async pipeline entirely
//   3. Batch grouping: BATCH_GROUP consecutive batches share triangle index computation
//
// Work decomposition: batch-major ordering (work = batch_group * tri_blocks + tri_block)
// ensures consecutive work items for the same triangle position access the same A rows
// across batches, improving L2 locality for the A-side data.
//
// SMEM sizing:
//   Small-K path (K ≤ K_CHUNK): 1 buffer × 2 tiles × TILE_BYTES = 8 KB (BN=32)
//   Large-K path: 3 buffers × 2 tiles × TILE_BYTES = 24 KB (BN=32)
//
template <typename OutputType>
__global__ void __launch_bounds__(HERK_THREADS, 4)
herk_direct_fp8_persistent_kernel(
    const unsigned char* __restrict__ A_fp8,
    OutputType* __restrict__ C,
    int N, int K, int batch_count,
    float alpha, float beta,
    int tri_blocks, int total_work_items)
{
    constexpr int TILE_BYTES = HERK_BLOCK_N * HERK_SMEM_K;  // 4096 bytes
    constexpr int BATCH_GROUP = 4;

    extern __shared__ char smem_raw[];

    const int tid = threadIdx.x + threadIdx.y * 32 + threadIdx.z * 64;
    const int warp_id = tid / 32;
    const int lane = tid % 32;

    const int64_t A_batch_stride = static_cast<int64_t>(N) * K * 2;
    const int64_t packed_batch_stride = static_cast<int64_t>(N) * (N + 1);

    const int num_chunks = (K + HERK_K_CHUNK - 1) / HERK_K_CHUNK;
    const bool small_k = (K <= HERK_K_CHUNK);

    // Work grouping: process BATCH_GROUP batches per work item to amortize
    // triangle index computation across batches
    int total_work_groups = ((batch_count + BATCH_GROUP - 1) / BATCH_GROUP) * tri_blocks;

    // ---- Persistent loop: each block iterates over multiple work items ----
    for (int work = blockIdx.x; work < total_work_groups; work += gridDim.x) {
        int group = work / tri_blocks;
        int tri_block = work % tri_blocks;
        int batch_base = group * BATCH_GROUP;

        // ---- Triangular indexing (computed once per work item) ----
        unsigned blockX = static_cast<unsigned>(__fsqrt_rn(8.0f * tri_block + 1.0f) - 0.999f) * 0.5f;
        while (blockX * (blockX + 1) / 2 > (unsigned)tri_block) --blockX;
        while ((blockX + 1) * (blockX + 2) / 2 <= (unsigned)tri_block) ++blockX;
        unsigned blockY = tri_block - blockX * (blockX + 1) / 2;

        const unsigned firstRecvRow = blockX * HERK_BLOCK_N;
        const unsigned firstRecvCol = blockY * HERK_BLOCK_N;
        const bool is_diagonal = (blockX == blockY);

        // ---- Process BATCH_GROUP batches for this triangle position ----
        for (int bi = 0; bi < BATCH_GROUP && batch_base + bi < batch_count; ++bi) {
            int batch = batch_base + bi;

            const unsigned char* A_batch = A_fp8 + batch * A_batch_stride;
            OutputType* C_batch = C + batch * packed_batch_stride;

            // ---- Zero accumulators ----
            float acc[HERK_FRAGS_Y][HERK_FRAGS_X][4];
            #pragma unroll
            for (int fy = 0; fy < HERK_FRAGS_Y; fy++)
                #pragma unroll
                for (int fx = 0; fx < HERK_FRAGS_X; fx++)
                    acc[fy][fx][0] = acc[fy][fx][1] = acc[fy][fx][2] = acc[fy][fx][3] = 0.0f;

            if (small_k) {
                // ---- OPT 2: Small-K direct load — single buffer, no pipeline ----
                unsigned char* smem_row = reinterpret_cast<unsigned char*>(smem_raw);
                unsigned char* smem_col = is_diagonal ? smem_row : (smem_row + TILE_BYTES);

                load_tile_direct_fp8(A_batch, smem_row, firstRecvRow, 0, K, N, tid);
                if (!is_diagonal) {
                    load_tile_direct_fp8(A_batch, smem_col, firstRecvCol, 0, K, N, tid);
                }
                __syncthreads();

                // Compute ceil(K/16) MMA sub-iterations (K_SUBS or fewer)
                int actual_subs = (K + HERK_K_PER_MMA - 1) / HERK_K_PER_MMA;
                for (int ks = 0; ks < actual_subs; ks++) {
                    int k_base = ks * 32;

                    // Same unrolled fy pattern as above — required for register allocation.
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
                                a0, a1, a2, a3, b0, b1,
                                acc[fy][fx][0], acc[fy][fx][1], acc[fy][fx][2], acc[fy][fx][3]);
                        }
                    }
                }

                __syncthreads();  // Ensure SMEM reads complete before next iteration

            } else {
                // ---- Large-K: 3-buffer cp.async pipeline (same as non-persistent) ----
                const int tiles_per_buf = is_diagonal ? 1 : 2;

                unsigned char* smem_buf[HERK_NR_BUFS];
                smem_buf[0] = reinterpret_cast<unsigned char*>(smem_raw);
                smem_buf[1] = smem_buf[0] + tiles_per_buf * TILE_BYTES;
                smem_buf[2] = smem_buf[1] + tiles_per_buf * TILE_BYTES;

                // Prologue: issue READ_AHEAD loads
                for (int p = 0; p < HERK_READ_AHEAD && p < num_chunks; ++p) {
                    unsigned char* buf = smem_buf[p];
                    unsigned char* s_row = buf;
                    unsigned char* s_col = is_diagonal ? s_row : (s_row + TILE_BYTES);

                    int kt_p = p * HERK_K_CHUNK;
                    load_tile_cp_async(A_batch, s_row, firstRecvRow, kt_p, K, N, tid);
                    if (!is_diagonal) {
                        load_tile_cp_async(A_batch, s_col, firstRecvCol, kt_p, K, N, tid);
                    }
                    cp_async_commit();
                }

                // Main K-loop
                for (int kt_idx = 0; kt_idx < num_chunks; ++kt_idx) {
                    int buf_idx = kt_idx % HERK_NR_BUFS;

                    if (kt_idx + 1 < num_chunks) {
                        cp_async_wait<HERK_READ_AHEAD - 1>();
                    } else {
                        cp_async_wait<0>();
                    }
                    __syncthreads();

                    // Prefetch next chunk
                    int next_p = kt_idx + HERK_READ_AHEAD;
                    if (next_p < num_chunks) {
                        int next_buf_idx = next_p % HERK_NR_BUFS;
                        unsigned char* next_buf = smem_buf[next_buf_idx];
                        unsigned char* s_row_next = next_buf;
                        unsigned char* s_col_next = is_diagonal ? s_row_next : (s_row_next + TILE_BYTES);

                        int kt_next = next_p * HERK_K_CHUNK;
                        load_tile_cp_async(A_batch, s_row_next, firstRecvRow, kt_next, K, N, tid);
                        if (!is_diagonal) {
                            load_tile_cp_async(A_batch, s_col_next, firstRecvCol, kt_next, K, N, tid);
                        }
                        cp_async_commit();
                    }

                    // Compute on current buffer
                    unsigned char* smem_row = smem_buf[buf_idx];
                    unsigned char* smem_col = is_diagonal ? smem_row : (smem_row + TILE_BYTES);

                    #pragma unroll
                    for (int ks = 0; ks < HERK_K_SUBS; ks++) {
                        int k_base = ks * 32;

                        // Same unrolled fy pattern as above — required for register allocation.
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
                                    a0, a1, a2, a3, b0, b1,
                                    acc[fy][fx][0], acc[fy][fx][1], acc[fy][fx][2], acc[fy][fx][3]);
                            }
                        }
                    }

                    __syncthreads();
                }
            }

            // ---- Store: SMEM-staged coalesced writes to packed triangle ----
            {
                float* smem_store = reinterpret_cast<float*>(smem_raw);

                // Phase 1: Scatter accumulators to SMEM tile
                #pragma unroll
                for (int fy = 0; fy < HERK_FRAGS_Y; fy++) {
                    if (fy != warp_id) continue;

                    #pragma unroll
                    for (int fx = 0; fx < HERK_FRAGS_X; fx++) {
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

                            int tile_row = fy * HERK_RECV_PER_MMA_Y + recv_local_row;
                            int tile_col = fx * HERK_RECV_PER_MMA_X + recv_local_col;
                            smem_store[tile_row * 64 + tile_col * 2 + is_im] = d_vals[vi];
                        }
                    }
                }

                __syncthreads();

                // Phase 2: Coalesced row-by-row flush
                for (int r = 0; r < HERK_BLOCK_N; r += 2) {
                    int my_row = r + (tid / 64);
                    int my_elem = tid % 64;
                    int tile_col = my_elem / 2;
                    int is_im = my_elem & 1;

                    unsigned recv_row = firstRecvRow + my_row;
                    unsigned recv_col = firstRecvCol + tile_col;

                    if (recv_row < (unsigned)N && recv_col < (unsigned)N && recv_row >= recv_col) {
                        float val = smem_store[my_row * 64 + my_elem];

                        if (recv_row == recv_col && is_im) val = 0.0f;
                        val *= alpha;

                        int64_t tri_idx = static_cast<int64_t>(recv_row) * (recv_row + 1) / 2 + recv_col;
                        int64_t out_idx = tri_idx * 2 + is_im;

                        if (beta != 0.0f) {
                            val += beta * out_to_float<OutputType>(C_batch[out_idx]);
                        }

                        C_batch[out_idx] = float_to_out<OutputType>(val);
                    }
                }

                __syncthreads();  // Protect SMEM for next batch iteration
            }
        }  // end batch group loop
    }  // end persistent work loop
}

// ========================================================================================
// Host-side launch for persistent direct HERK kernel
// ========================================================================================

template <typename OutputType>
inline cutlass::Status launch_herk_direct_fp8_persistent(
    const __nv_fp8_e4m3* A_fp8,
    OutputType* C,
    int N, int K,
    int batch_count,
    float alpha, float beta,
    FillMode fill,
    int sm_count,
    cudaStream_t stream)
{
    if (fill != FillMode::Lower) {
        return cutlass::Status::kErrorNotSupported;
    }

    int blocks_per_dim = (N + HERK_BLOCK_N - 1) / HERK_BLOCK_N;
    int tri_blocks = blocks_per_dim * (blocks_per_dim + 1) / 2;
    int total_work = tri_blocks * batch_count;

    // Persistent grid: sm_count × target_occupancy, capped at total work items
    constexpr int TARGET_OCCUPANCY = 4;
    int grid_x = std::min(sm_count * TARGET_OCCUPANCY, total_work);

    // SMEM: small-K uses 1 buffer (2 tiles), large-K uses 3 buffers (2 tiles each)
    int smem_bytes;
    if (K <= HERK_K_CHUNK) {
        smem_bytes = 2 * HERK_BLOCK_N * HERK_SMEM_K;  // 8192 bytes
    } else {
        smem_bytes = HERK_NR_BUFS * 2 * HERK_BLOCK_N * HERK_SMEM_K;  // 24576 bytes
    }

    dim3 grid(grid_x, 1, 1);
    dim3 block(32, 2, 2);  // 128 threads = 4 warps

    herk_direct_fp8_persistent_kernel<OutputType><<<grid, block, smem_bytes, stream>>>(
        reinterpret_cast<const unsigned char*>(A_fp8),
        C, N, K, batch_count, alpha, beta, tri_blocks, total_work);
    CUDA_CHECK(cudaGetLastError());

    return cutlass::Status::kSuccess;
}

// ========================================================================================
// Templated cp.async tile load (parameterized by DirectHerkTraits)
// ========================================================================================
template <typename Traits>
__device__ __forceinline__ void load_tile_cp_async_t(
    const unsigned char* __restrict__ A_fp8_batch,
    unsigned char* __restrict__ smem,
    unsigned firstRecv,
    int kt, int K, int N,
    int tid)
{
    int recv = tid / 4;
    int k_group = tid % 4;
    unsigned abs_recv = firstRecv + recv;

    constexpr int BYTES_PER_GROUP = Traits::K_PER_GROUP * 2;
    constexpr int K_PER_CP_ASYNC = 8;  // each cp.async loads 16 bytes = 8 complex FP8
    unsigned char* smem_dst = &smem[recv * Traits::SMEM_K + k_group * BYTES_PER_GROUP];

    if (abs_recv < (unsigned)N && kt + k_group * Traits::K_PER_GROUP < K) {
        int k_start = kt + k_group * Traits::K_PER_GROUP;
        int64_t global_offset = static_cast<int64_t>(abs_recv) * 2 * K
                              + static_cast<int64_t>(k_start) * 2;
        const unsigned char* gmem_ptr = A_fp8_batch + global_offset;

        #pragma unroll
        for (int i = 0; i < Traits::CP_ASYNC_PER_THREAD; i++) {
            if (k_start + i * K_PER_CP_ASYNC < K) {
                cp_async_cg_16(smem_dst + i * 16, gmem_ptr + i * 16);
            } else {
                // Zero-fill sub-operations beyond valid K range
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
// Templated direct (non-cp.async) tile load for small-K persistent path
// ========================================================================================
template <typename Traits>
__device__ __forceinline__ void load_tile_direct_fp8_t(
    const unsigned char* __restrict__ A_fp8_batch,
    unsigned char* __restrict__ smem,
    unsigned firstRecv,
    int kt, int K, int N,
    int tid)
{
    int recv = tid / 4;
    int k_group = tid % 4;
    unsigned abs_recv = firstRecv + recv;

    constexpr int BYTES_PER_GROUP = Traits::K_PER_GROUP * 2;
    constexpr int K_PER_LOAD = 8;  // each uint4 load = 16 bytes = 8 complex FP8
    unsigned char* smem_dst = &smem[recv * Traits::SMEM_K + k_group * BYTES_PER_GROUP];

    if (abs_recv < (unsigned)N && kt + k_group * Traits::K_PER_GROUP < K) {
        int k_start = kt + k_group * Traits::K_PER_GROUP;
        int64_t offset = static_cast<int64_t>(abs_recv) * 2 * K
                       + static_cast<int64_t>(k_start) * 2;
        const unsigned char* src = A_fp8_batch + offset;
        // Vectorized 128-bit loads with per-sub-operation bounds check
        #pragma unroll
        for (int i = 0; i < BYTES_PER_GROUP / 16; i++) {
            if (k_start + i * K_PER_LOAD < K) {
                *reinterpret_cast<uint4*>(smem_dst + i * 16) =
                    *reinterpret_cast<const uint4*>(src + i * 16);
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
// Templated compute body — shared between precast and persistent kernels
// ========================================================================================
template <typename Traits>
__device__ __forceinline__ void compute_mma_block(
    unsigned char* __restrict__ smem_row,
    unsigned char* __restrict__ smem_col,
    float acc[HERK_FRAGS_Y][HERK_FRAGS_X][4],
    unsigned firstRecvRow, unsigned firstRecvCol,
    bool is_diagonal,
    int warp_id, int lane)
{
    #pragma unroll
    for (int ks = 0; ks < Traits::K_SUBS; ks++) {
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

                unsigned v0 = *reinterpret_cast<const unsigned*>(&smem_row[smem_recv0 * Traits::SMEM_K + k_off0]);
                unsigned v1 = *reinterpret_cast<const unsigned*>(&smem_row[smem_recv0 * Traits::SMEM_K + k_off1]);
                a0 = is_conj ? conj_perm_fp8(v0) : v0;
                a2 = is_conj ? conj_perm_fp8(v1) : v1;

                if (smem_recv1 < HERK_BLOCK_N) {
                    unsigned v2 = *reinterpret_cast<const unsigned*>(&smem_row[smem_recv1 * Traits::SMEM_K + k_off0]);
                    unsigned v3 = *reinterpret_cast<const unsigned*>(&smem_row[smem_recv1 * Traits::SMEM_K + k_off1]);
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
                        b0 = *reinterpret_cast<const unsigned*>(&smem_col[recv_idx * Traits::SMEM_K + k_off0]);
                        b1 = *reinterpret_cast<const unsigned*>(&smem_col[recv_idx * Traits::SMEM_K + k_off1]);
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

// ========================================================================================
// Templated store body — shared between precast and persistent kernels
// ========================================================================================
//
// SMEM-staged coalesced store: eliminates L2 write amplification from scattered
// per-element stores to packed triangle format.
//   Phase 1: Each warp scatters its accumulator fragments into a 32×64 SMEM tile.
//   Phase 2: All 128 threads write rows sequentially with coalesced transactions.
// Within each warp, all 32 lanes write to the same row → fully coalesced L2 writes.
// Requires: smem_raw has at least HERK_SMEM_STORE_BYTES (8 KB) available.
//
template <typename OutputType>
__device__ __forceinline__ void store_triangle_block(
    float acc[HERK_FRAGS_Y][HERK_FRAGS_X][4],
    OutputType* __restrict__ C_batch,
    char* __restrict__ smem_raw,
    unsigned firstRecvRow, unsigned firstRecvCol,
    bool is_diagonal,
    int N, float alpha, float beta,
    int tid, int warp_id, int lane)
{
    float* smem_store = reinterpret_cast<float*>(smem_raw);

    // Phase 1: Scatter accumulators to SMEM tile [32 rows × 64 elements]
    // Layout: smem_store[tile_row * 64 + tile_col * 2 + is_im]
    #pragma unroll
    for (int fy = 0; fy < HERK_FRAGS_Y; fy++) {
        if (fy != warp_id) continue;

        #pragma unroll
        for (int fx = 0; fx < HERK_FRAGS_X; fx++) {
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

                int tile_row = fy * HERK_RECV_PER_MMA_Y + recv_local_row;
                int tile_col = fx * HERK_RECV_PER_MMA_X + recv_local_col;
                smem_store[tile_row * 64 + tile_col * 2 + is_im] = d_vals[vi];
            }
        }
    }

    __syncthreads();

    // Phase 2: Coalesced row-by-row flush to global memory.
    // 128 threads write 2 rows per iteration (64 elements per row).
    // All threads within a warp write to the same row → coalesced transactions.
    for (int r = 0; r < HERK_BLOCK_N; r += 2) {
        int my_row = r + (tid / 64);
        int my_elem = tid % 64;
        int tile_col = my_elem / 2;
        int is_im = my_elem & 1;

        unsigned recv_row = firstRecvRow + my_row;
        unsigned recv_col = firstRecvCol + tile_col;

        if (recv_row < (unsigned)N && recv_col < (unsigned)N && recv_row >= recv_col) {
            float val = smem_store[my_row * 64 + my_elem];

            if (recv_row == recv_col && is_im) val = 0.0f;
            val *= alpha;

            int64_t tri_idx = static_cast<int64_t>(recv_row) * (recv_row + 1) / 2 + recv_col;
            int64_t out_idx = tri_idx * 2 + is_im;

            if (beta != 0.0f) {
                val += beta * out_to_float<OutputType>(C_batch[out_idx]);
            }

            C_batch[out_idx] = float_to_out<OutputType>(val);
        }
    }

    __syncthreads();  // Protect SMEM for next iteration (persistent kernel)
}

// ========================================================================================
// Templated scratch store — direct register-to-global N×N writes (FP16 only)
// ========================================================================================
//
// Writes accumulator fragments to an N×N row-major interleaved scratch buffer
// instead of packed triangle format. Eliminates:
//   - SMEM staging (no smem_raw usage, no __syncthreads)
//   - Triangle bounds checking (pack kernel extracts lower triangle later)
//   - Beta accumulation (deferred to pack kernel)
//
template <typename OutputType>
__device__ __forceinline__ void store_scratch_block(
    float acc[HERK_FRAGS_Y][HERK_FRAGS_X][4],
    OutputType* __restrict__ scratch_batch,  // N×N×2 interleaved scratch
    unsigned firstRecvRow, unsigned firstRecvCol,
    bool is_diagonal,
    int N, float alpha,
    int warp_id, int lane)
{
    #pragma unroll
    for (int fy = 0; fy < HERK_FRAGS_Y; fy++) {
        if (fy != warp_id) continue;

        #pragma unroll
        for (int fx = 0; fx < HERK_FRAGS_X; fx++) {
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

                float val = d_vals[vi];
                if (is_diagonal && recv_row == recv_col && is_im) val = 0.0f;
                val *= alpha;

                int64_t idx = static_cast<int64_t>(recv_row) * (2 * N) + recv_col * 2 + is_im;
                scratch_batch[idx] = float_to_out<OutputType>(val);
            }
        }
    }
}

// ========================================================================================
// Templated Pre-cast + cp.async direct HERK kernel
// ========================================================================================
template <typename Traits, typename OutputType, bool ScratchMode = false>
__global__ void __launch_bounds__(HERK_THREADS)
herk_direct_fp8_precast_kernel_t(
    const unsigned char* __restrict__ A_fp8,
    OutputType* __restrict__ C,
    int N, int K, int batch_count,
    float alpha, float beta,
    int blocks_per_dim,
    int64_t output_batch_stride)
{
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

    extern __shared__ char smem_raw[];
    constexpr int TILE_BYTES = Traits::TILE_BYTES;
    const int tiles_per_buf = is_diagonal ? 1 : 2;

    unsigned char* smem_buf[Traits::NR_BUFS];
    smem_buf[0] = reinterpret_cast<unsigned char*>(smem_raw);
    for (int i = 1; i < Traits::NR_BUFS; i++)
        smem_buf[i] = smem_buf[i-1] + tiles_per_buf * TILE_BYTES;

    float acc[HERK_FRAGS_Y][HERK_FRAGS_X][4];
    #pragma unroll
    for (int fy = 0; fy < HERK_FRAGS_Y; fy++)
        #pragma unroll
        for (int fx = 0; fx < HERK_FRAGS_X; fx++)
            acc[fy][fx][0] = acc[fy][fx][1] = acc[fy][fx][2] = acc[fy][fx][3] = 0.0f;

    const int64_t A_batch_stride = static_cast<int64_t>(N) * K * 2;
    const unsigned char* A_batch = A_fp8 + batch * A_batch_stride;

    const int num_chunks = (K + Traits::K_CHUNK - 1) / Traits::K_CHUNK;

    for (int p = 0; p < Traits::READ_AHEAD && p < num_chunks; ++p) {
        unsigned char* buf = smem_buf[p];
        unsigned char* smem_row = buf;
        unsigned char* smem_col = is_diagonal ? smem_row : (smem_row + TILE_BYTES);

        int kt_p = p * Traits::K_CHUNK;
        load_tile_cp_async_t<Traits>(A_batch, smem_row, firstRecvRow, kt_p, K, N, tid);
        if (!is_diagonal) {
            load_tile_cp_async_t<Traits>(A_batch, smem_col, firstRecvCol, kt_p, K, N, tid);
        }
        cp_async_commit();
    }

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
            int next_buf_idx = next_p % Traits::NR_BUFS;
            unsigned char* next_buf = smem_buf[next_buf_idx];
            unsigned char* smem_row_next = next_buf;
            unsigned char* smem_col_next = is_diagonal ? smem_row_next : (smem_row_next + TILE_BYTES);

            int kt_next = next_p * Traits::K_CHUNK;
            load_tile_cp_async_t<Traits>(A_batch, smem_row_next, firstRecvRow, kt_next, K, N, tid);
            if (!is_diagonal) {
                load_tile_cp_async_t<Traits>(A_batch, smem_col_next, firstRecvCol, kt_next, K, N, tid);
            }
            cp_async_commit();
        }

        unsigned char* smem_row = smem_buf[buf_idx];
        unsigned char* smem_col = is_diagonal ? smem_row : (smem_row + TILE_BYTES);

        compute_mma_block<Traits>(smem_row, smem_col, acc,
            firstRecvRow, firstRecvCol, is_diagonal, warp_id, lane);

        __syncthreads();
    }

    OutputType* C_batch = C + batch * output_batch_stride;
    if (ScratchMode) {
        store_scratch_block<OutputType>(acc, C_batch, firstRecvRow, firstRecvCol,
            is_diagonal, N, alpha, warp_id, lane);
    } else {
        store_triangle_block<OutputType>(acc, C_batch, smem_raw, firstRecvRow, firstRecvCol,
            is_diagonal, N, alpha, beta, tid, warp_id, lane);
    }
}

// ========================================================================================
// Templated Persistent Direct HERK Kernel
// ========================================================================================
template <typename Traits, int BATCH_GROUP_, typename OutputType, bool ScratchMode = false>
__global__ void __launch_bounds__(HERK_THREADS)
herk_direct_fp8_persistent_kernel_t(
    const unsigned char* __restrict__ A_fp8,
    OutputType* __restrict__ C,
    int N, int K, int batch_count,
    float alpha, float beta,
    int tri_blocks, int total_work_items,
    int64_t output_batch_stride)
{
    constexpr int TILE_BYTES = Traits::TILE_BYTES;
    extern __shared__ char smem_raw[];

    const int tid = threadIdx.x + threadIdx.y * 32 + threadIdx.z * 64;
    const int warp_id = tid / 32;
    const int lane = tid % 32;

    const int64_t A_batch_stride = static_cast<int64_t>(N) * K * 2;

    const int num_chunks = (K + Traits::K_CHUNK - 1) / Traits::K_CHUNK;
    const bool small_k = (K <= Traits::K_CHUNK);

    int total_work_groups = ((batch_count + BATCH_GROUP_ - 1) / BATCH_GROUP_) * tri_blocks;

    for (int work = blockIdx.x; work < total_work_groups; work += gridDim.x) {
        int group = work / tri_blocks;
        int tri_block = work % tri_blocks;
        int batch_base = group * BATCH_GROUP_;

        unsigned blockX = static_cast<unsigned>(__fsqrt_rn(8.0f * tri_block + 1.0f) - 0.999f) * 0.5f;
        while (blockX * (blockX + 1) / 2 > (unsigned)tri_block) --blockX;
        while ((blockX + 1) * (blockX + 2) / 2 <= (unsigned)tri_block) ++blockX;
        unsigned blockY = tri_block - blockX * (blockX + 1) / 2;

        const unsigned firstRecvRow = blockX * HERK_BLOCK_N;
        const unsigned firstRecvCol = blockY * HERK_BLOCK_N;
        const bool is_diagonal = (blockX == blockY);

        for (int bi = 0; bi < BATCH_GROUP_ && batch_base + bi < batch_count; ++bi) {
            int batch = batch_base + bi;

            const unsigned char* A_batch = A_fp8 + batch * A_batch_stride;
            OutputType* C_batch = C + batch * output_batch_stride;

            float acc[HERK_FRAGS_Y][HERK_FRAGS_X][4];
            #pragma unroll
            for (int fy = 0; fy < HERK_FRAGS_Y; fy++)
                #pragma unroll
                for (int fx = 0; fx < HERK_FRAGS_X; fx++)
                    acc[fy][fx][0] = acc[fy][fx][1] = acc[fy][fx][2] = acc[fy][fx][3] = 0.0f;

            if (small_k) {
                unsigned char* smem_row = reinterpret_cast<unsigned char*>(smem_raw);
                unsigned char* smem_col = is_diagonal ? smem_row : (smem_row + TILE_BYTES);

                load_tile_direct_fp8_t<Traits>(A_batch, smem_row, firstRecvRow, 0, K, N, tid);
                if (!is_diagonal) {
                    load_tile_direct_fp8_t<Traits>(A_batch, smem_col, firstRecvCol, 0, K, N, tid);
                }
                __syncthreads();

                int actual_subs = (K + HERK_K_PER_MMA - 1) / HERK_K_PER_MMA;
                for (int ks = 0; ks < actual_subs; ks++) {
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

                            unsigned v0 = *reinterpret_cast<const unsigned*>(&smem_row[smem_recv0 * Traits::SMEM_K + k_off0]);
                            unsigned v1 = *reinterpret_cast<const unsigned*>(&smem_row[smem_recv0 * Traits::SMEM_K + k_off1]);
                            a0 = is_conj ? conj_perm_fp8(v0) : v0;
                            a2 = is_conj ? conj_perm_fp8(v1) : v1;

                            if (smem_recv1 < HERK_BLOCK_N) {
                                unsigned v2 = *reinterpret_cast<const unsigned*>(&smem_row[smem_recv1 * Traits::SMEM_K + k_off0]);
                                unsigned v3 = *reinterpret_cast<const unsigned*>(&smem_row[smem_recv1 * Traits::SMEM_K + k_off1]);
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
                                    b0 = *reinterpret_cast<const unsigned*>(&smem_col[recv_idx * Traits::SMEM_K + k_off0]);
                                    b1 = *reinterpret_cast<const unsigned*>(&smem_col[recv_idx * Traits::SMEM_K + k_off1]);
                                } else {
                                    b0 = 0; b1 = 0;
                                }
                            }
                            mma_fp8_m16n8k32(
                                acc[fy][fx][0], acc[fy][fx][1], acc[fy][fx][2], acc[fy][fx][3],
                                a0, a1, a2, a3, b0, b1,
                                acc[fy][fx][0], acc[fy][fx][1], acc[fy][fx][2], acc[fy][fx][3]);
                        }
                    }
                }
                __syncthreads();
            } else {
                const int tiles_per_buf = is_diagonal ? 1 : 2;
                unsigned char* smem_buf[Traits::NR_BUFS];
                smem_buf[0] = reinterpret_cast<unsigned char*>(smem_raw);
                for (int i = 1; i < Traits::NR_BUFS; i++)
                    smem_buf[i] = smem_buf[i-1] + tiles_per_buf * TILE_BYTES;

                for (int p = 0; p < Traits::READ_AHEAD && p < num_chunks; ++p) {
                    unsigned char* buf = smem_buf[p];
                    unsigned char* s_row = buf;
                    unsigned char* s_col = is_diagonal ? s_row : (s_row + TILE_BYTES);
                    int kt_p = p * Traits::K_CHUNK;
                    load_tile_cp_async_t<Traits>(A_batch, s_row, firstRecvRow, kt_p, K, N, tid);
                    if (!is_diagonal) {
                        load_tile_cp_async_t<Traits>(A_batch, s_col, firstRecvCol, kt_p, K, N, tid);
                    }
                    cp_async_commit();
                }

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
                        int next_buf_idx = next_p % Traits::NR_BUFS;
                        unsigned char* next_buf = smem_buf[next_buf_idx];
                        unsigned char* s_row_next = next_buf;
                        unsigned char* s_col_next = is_diagonal ? s_row_next : (s_row_next + TILE_BYTES);
                        int kt_next = next_p * Traits::K_CHUNK;
                        load_tile_cp_async_t<Traits>(A_batch, s_row_next, firstRecvRow, kt_next, K, N, tid);
                        if (!is_diagonal) {
                            load_tile_cp_async_t<Traits>(A_batch, s_col_next, firstRecvCol, kt_next, K, N, tid);
                        }
                        cp_async_commit();
                    }

                    unsigned char* smem_row = smem_buf[buf_idx];
                    unsigned char* smem_col = is_diagonal ? smem_row : (smem_row + TILE_BYTES);
                    compute_mma_block<Traits>(smem_row, smem_col, acc,
                        firstRecvRow, firstRecvCol, is_diagonal, warp_id, lane);
                    __syncthreads();
                }
            }

            if (ScratchMode) {
                store_scratch_block<OutputType>(acc, C_batch, firstRecvRow, firstRecvCol,
                    is_diagonal, N, alpha, warp_id, lane);
            } else {
                store_triangle_block<OutputType>(acc, C_batch, smem_raw, firstRecvRow, firstRecvCol,
                    is_diagonal, N, alpha, beta, tid, warp_id, lane);
            }
        }
    }
}

// ========================================================================================
// Templated launch helpers
// ========================================================================================
template <typename Traits, typename OutputType, bool ScratchMode = false>
inline cutlass::Status launch_precast_t(
    const __nv_fp8_e4m3* A_fp8, OutputType* C,
    int N, int K, int batch_count,
    float alpha, float beta, FillMode fill,
    cudaStream_t stream)
{
    if (fill != FillMode::Lower) return cutlass::Status::kErrorNotSupported;
    int blocks_per_dim = (N + HERK_BLOCK_N - 1) / HERK_BLOCK_N;
    int tri_blocks = blocks_per_dim * (blocks_per_dim + 1) / 2;
    dim3 grid(tri_blocks, batch_count, 1);
    dim3 block(32, 2, 2);

    int64_t output_batch_stride = ScratchMode
        ? static_cast<int64_t>(N) * N * 2
        : static_cast<int64_t>(N) * (N + 1);

    // SMEM: ScratchMode doesn't need STORE_SMEM (no SMEM staging)
    int smem_bytes;
    if (ScratchMode) {
        smem_bytes = Traits::NR_BUFS * 2 * Traits::TILE_BYTES;
    } else {
        constexpr int STORE_SMEM = HERK_BLOCK_N * 64 * static_cast<int>(sizeof(float));
        smem_bytes = std::max(Traits::NR_BUFS * 2 * Traits::TILE_BYTES, STORE_SMEM);
    }

    herk_direct_fp8_precast_kernel_t<Traits, OutputType, ScratchMode><<<grid, block, smem_bytes, stream>>>(
        reinterpret_cast<const unsigned char*>(A_fp8),
        C, N, K, batch_count, alpha, beta, blocks_per_dim,
        output_batch_stride);
    CUDA_CHECK(cudaGetLastError());
    return cutlass::Status::kSuccess;
}

template <typename Traits, int BATCH_GROUP, typename OutputType, bool ScratchMode = false>
inline cutlass::Status launch_persistent_t(
    const __nv_fp8_e4m3* A_fp8, OutputType* C,
    int N, int K, int batch_count,
    float alpha, float beta, FillMode fill,
    int sm_count, cudaStream_t stream)
{
    if (fill != FillMode::Lower) return cutlass::Status::kErrorNotSupported;
    int blocks_per_dim = (N + HERK_BLOCK_N - 1) / HERK_BLOCK_N;
    int tri_blocks = blocks_per_dim * (blocks_per_dim + 1) / 2;
    int total_work = tri_blocks * batch_count;
    constexpr int TARGET_OCCUPANCY = 4;
    int grid_x = std::min(sm_count * TARGET_OCCUPANCY, total_work);

    int64_t output_batch_stride = ScratchMode
        ? static_cast<int64_t>(N) * N * 2
        : static_cast<int64_t>(N) * (N + 1);

    // SMEM: ScratchMode doesn't need STORE_SMEM (no SMEM staging)
    int smem_bytes;
    if (ScratchMode) {
        if (K <= Traits::K_CHUNK) {
            smem_bytes = 2 * Traits::TILE_BYTES;
        } else {
            smem_bytes = Traits::NR_BUFS * 2 * Traits::TILE_BYTES;
        }
    } else {
        constexpr int STORE_SMEM = HERK_BLOCK_N * 64 * static_cast<int>(sizeof(float));
        if (K <= Traits::K_CHUNK) {
            smem_bytes = std::max(2 * Traits::TILE_BYTES, STORE_SMEM);
        } else {
            smem_bytes = std::max(Traits::NR_BUFS * 2 * Traits::TILE_BYTES, STORE_SMEM);
        }
    }
    dim3 grid(grid_x, 1, 1);
    dim3 block(32, 2, 2);
    herk_direct_fp8_persistent_kernel_t<Traits, BATCH_GROUP, OutputType, ScratchMode>
        <<<grid, block, smem_bytes, stream>>>(
        reinterpret_cast<const unsigned char*>(A_fp8),
        C, N, K, batch_count, alpha, beta, tri_blocks, total_work,
        output_batch_stride);
    CUDA_CHECK(cudaGetLastError());
    return cutlass::Status::kSuccess;
}

// ========================================================================================
// (Large-K kernel and load_tile_cp_async_large removed — replaced by
//  TraitsK128_B2 template instantiation via launch_herk_direct_dispatch)

// ========================================================================================
// HerkTileConfig — compile-time tile dimensions for tiled direct HERK kernel
// ========================================================================================
//
// Parameterizes the output tile shape. HERK is symmetric (A=B) so BLOCK_M = BLOCK_N.
// The MMA is m16n8k32 with the conjugate permutation trick producing 8 output rows
// and 8 output columns per MMA. THREADS=128 (4 warps) is fixed.
//
// LOAD_PASSES:   number of 32-row loads needed per tile (same load function, different offsets)
// STORE_PHASES:  number of 32-column phases for the store step (bounds SMEM)
// FYS_PER_WARP:  number of fy values each warp handles (strided by NUM_WARPS)
//
// SMEM budget (K_CHUNK=64, NR_BUFS=3):
//   N32: pipeline=24KB, store=8KB  → max=24KB, SM120 OCC=4
//   N64: pipeline=48KB, store=16KB → max=48KB, SM120 OCC=2
//

template <int BLOCK_N_>
struct HerkTileConfig {
    static constexpr int BLOCK_N = BLOCK_N_;
    static constexpr int THREADS = 128;
    static constexpr int NUM_WARPS = 4;
    static constexpr int RECV_PER_MMA_Y = 8;
    static constexpr int RECV_PER_MMA_X = 8;
    static constexpr int FRAGS_Y = BLOCK_N / RECV_PER_MMA_Y;
    static constexpr int FRAGS_X = BLOCK_N / RECV_PER_MMA_X;
    static constexpr int LOAD_PASSES = BLOCK_N / 32;
    static constexpr int STORE_PHASES = BLOCK_N / 32;
    static constexpr int FYS_PER_WARP = FRAGS_Y / NUM_WARPS;
};

using HerkTile32 = HerkTileConfig<32>;  // Current: AI=64, 24KB (K64_B3), OCC=4
using HerkTile64 = HerkTileConfig<64>;  // New: AI=128, 48KB (K64_B3), OCC=2

// ========================================================================================
// Tiled compute body — strided warp mapping for HERK
// ========================================================================================
//
// For BLOCK_N=32 (FYS_PER_WARP=1): each warp handles 1 fy — identical to compute_mma_block.
// For BLOCK_N=64 (FYS_PER_WARP=2): each warp handles 2 fy values (strided by NUM_WARPS=4).
// Accumulator: acc[FYS_PER_WARP][FRAGS_X][4] — per-warp sized to avoid register spills.
//
template <typename Traits, typename Tile>
__device__ __forceinline__ void compute_mma_herk_tiled_t(
    unsigned char* __restrict__ smem_row,
    unsigned char* __restrict__ smem_col,
    float acc[Tile::FYS_PER_WARP][Tile::FRAGS_X][4],
    unsigned firstRecvRow, unsigned firstRecvCol,
    bool is_diagonal,
    int warp_id, int lane)
{
    #pragma unroll
    for (int ks = 0; ks < Traits::K_SUBS; ks++) {
        int k_base = ks * 32;

        // CRITICAL: #pragma unroll on fy_local ensures compile-time indexing into
        // acc[fy_local][...], preventing nvcc from keeping all accumulator floats live.
        #pragma unroll
        for (int fy_local = 0; fy_local < Tile::FYS_PER_WARP; fy_local++) {
            int fy = warp_id + fy_local * Tile::NUM_WARPS;
            int recv_base_y = fy * Tile::RECV_PER_MMA_Y;

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

                unsigned v0 = *reinterpret_cast<const unsigned*>(&smem_row[smem_recv0 * Traits::SMEM_K + k_off0]);
                unsigned v1 = *reinterpret_cast<const unsigned*>(&smem_row[smem_recv0 * Traits::SMEM_K + k_off1]);
                a0 = is_conj ? conj_perm_fp8(v0) : v0;
                a2 = is_conj ? conj_perm_fp8(v1) : v1;

                if (smem_recv1 < Tile::BLOCK_N) {
                    unsigned v2 = *reinterpret_cast<const unsigned*>(&smem_row[smem_recv1 * Traits::SMEM_K + k_off0]);
                    unsigned v3 = *reinterpret_cast<const unsigned*>(&smem_row[smem_recv1 * Traits::SMEM_K + k_off1]);
                    a1 = is_conj ? conj_perm_fp8(v2) : v2;
                    a3 = is_conj ? conj_perm_fp8(v3) : v3;
                } else {
                    a1 = 0; a3 = 0;
                }
            }

            #pragma unroll
            for (int fx = 0; fx < Tile::FRAGS_X; fx++) {
                if (is_diagonal) {
                    unsigned row_start = firstRecvRow + fy * Tile::RECV_PER_MMA_Y;
                    unsigned col_start = firstRecvCol + fx * Tile::RECV_PER_MMA_X;
                    if (row_start < col_start) continue;
                }

                int recv_base_x = fx * Tile::RECV_PER_MMA_X;
                unsigned b0, b1;
                {
                    int recv_idx = recv_base_x + lane / 4;
                    int k_off0 = k_base + (lane % 4) * 4;
                    int k_off1 = k_base + 16 + (lane % 4) * 4;

                    if (recv_idx < Tile::BLOCK_N) {
                        b0 = *reinterpret_cast<const unsigned*>(&smem_col[recv_idx * Traits::SMEM_K + k_off0]);
                        b1 = *reinterpret_cast<const unsigned*>(&smem_col[recv_idx * Traits::SMEM_K + k_off1]);
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
// Tiled triangle store — phased column processing for bounded SMEM
// ========================================================================================
//
// Uses SMEM staging with phased column processing (same pattern as GEMM store).
// Each phase processes 32 columns; SMEM = BLOCK_N × 64 × sizeof(float) per phase.
//   N32 (STORE_PHASES=1): single pass, 8 KB SMEM — identical to store_triangle_block.
//   N64 (STORE_PHASES=2): two passes of 32 cols, 16 KB SMEM.
//
template <typename Tile, typename OutputType>
__device__ __forceinline__ void store_triangle_herk_tiled_t(
    float acc[Tile::FYS_PER_WARP][Tile::FRAGS_X][4],
    OutputType* __restrict__ C_batch,
    char* __restrict__ smem_raw,
    unsigned firstRecvRow, unsigned firstRecvCol,
    bool is_diagonal,
    int N, float alpha, float beta,
    int tid, int warp_id, int lane)
{
    float* smem_store = reinterpret_cast<float*>(smem_raw);

    #pragma unroll
    for (int phase = 0; phase < Tile::STORE_PHASES; phase++) {
        int fx_start = phase * 4;  // 4 x-fragments per 32 columns

        // Phase 1: Scatter accumulators to SMEM tile [BLOCK_N rows × 64 elements]
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

        // Phase 2: Coalesced row-by-row flush to packed triangle.
        // 128 threads write 2 rows per iteration (64 elements per row).
        for (int r = 0; r < Tile::BLOCK_N; r += 2) {
            int my_row = r + (tid / 64);
            int my_elem = tid % 64;
            int local_col = my_elem / 2;
            int is_im = my_elem & 1;

            unsigned recv_row = firstRecvRow + my_row;
            unsigned recv_col = firstRecvCol + phase * 32 + local_col;

            if (recv_row < (unsigned)N && recv_col < (unsigned)N && recv_row >= recv_col) {
                float val = smem_store[my_row * 64 + my_elem];

                if (recv_row == recv_col && is_im) val = 0.0f;
                val *= alpha;

                int64_t tri_idx = static_cast<int64_t>(recv_row) * (recv_row + 1) / 2 + recv_col;
                int64_t out_idx = tri_idx * 2 + is_im;

                if (beta != 0.0f) {
                    val += beta * out_to_float<OutputType>(C_batch[out_idx]);
                }

                C_batch[out_idx] = float_to_out<OutputType>(val);
            }
        }

        if (Tile::STORE_PHASES > 1) __syncthreads();
    }

    // Final sync protects SMEM for next batch iteration (persistent kernel)
    if (Tile::STORE_PHASES == 1) __syncthreads();
}

// ========================================================================================
// Tiled scratch store — strided warp mapping, direct register-to-global writes
// ========================================================================================
template <typename Tile, typename OutputType>
__device__ __forceinline__ void store_scratch_herk_tiled_t(
    float acc[Tile::FYS_PER_WARP][Tile::FRAGS_X][4],
    OutputType* __restrict__ scratch_batch,
    unsigned firstRecvRow, unsigned firstRecvCol,
    bool is_diagonal,
    int N, float alpha,
    int warp_id, int lane)
{
    #pragma unroll
    for (int fy_local = 0; fy_local < Tile::FYS_PER_WARP; fy_local++) {
        int fy = warp_id + fy_local * Tile::NUM_WARPS;

        #pragma unroll
        for (int fx = 0; fx < Tile::FRAGS_X; fx++) {
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

                unsigned recv_row = firstRecvRow + fy * Tile::RECV_PER_MMA_Y + recv_local_row;
                unsigned recv_col = firstRecvCol + fx * Tile::RECV_PER_MMA_X + recv_local_col;

                if (recv_row >= (unsigned)N || recv_col >= (unsigned)N) continue;

                float val = d_vals[vi];
                if (is_diagonal && recv_row == recv_col && is_im) val = 0.0f;
                val *= alpha;

                int64_t idx = static_cast<int64_t>(recv_row) * (2 * N) + recv_col * 2 + is_im;
                scratch_batch[idx] = float_to_out<OutputType>(val);
            }
        }
    }
}

// ========================================================================================
// Tiled pre-cast + cp.async direct HERK kernel
// ========================================================================================
//
// Multi-pass loading: for BLOCK_N > 32, each tile load is split into LOAD_PASSES
// iterations of 32-row cp.async passes (same load function, offset SMEM pointers).
// Phased store: for BLOCK_N > 32, the store processes 32 columns per phase.
//
template <typename Traits, typename Tile, typename OutputType, bool ScratchMode = false>
__global__ void __launch_bounds__(Tile::THREADS)
herk_precast_tiled_kernel_t(
    const unsigned char* __restrict__ A_fp8,
    OutputType* __restrict__ C,
    int N, int K, int batch_count,
    float alpha, float beta,
    int blocks_per_dim,
    int64_t output_batch_stride)
{
    constexpr int TILE_BYTES = Tile::BLOCK_N * Traits::SMEM_K;

    const unsigned block = blockIdx.x;
    const unsigned batch = blockIdx.y;

    unsigned blockX = static_cast<unsigned>(__fsqrt_rn(8.0f * block + 1.0f) - 0.999f) * 0.5f;
    while (blockX * (blockX + 1) / 2 > block) --blockX;
    while ((blockX + 1) * (blockX + 2) / 2 <= block) ++blockX;
    unsigned blockY = block - blockX * (blockX + 1) / 2;

    const unsigned firstRecvRow = blockX * Tile::BLOCK_N;
    const unsigned firstRecvCol = blockY * Tile::BLOCK_N;
    const bool is_diagonal = (blockX == blockY);

    const int tid = threadIdx.x + threadIdx.y * 32 + threadIdx.z * 64;
    const int warp_id = tid / 32;
    const int lane = tid % 32;

    extern __shared__ char smem_raw[];
    const int tiles_per_buf = is_diagonal ? 1 : 2;

    unsigned char* smem_buf[Traits::NR_BUFS];
    smem_buf[0] = reinterpret_cast<unsigned char*>(smem_raw);
    for (int i = 1; i < Traits::NR_BUFS; i++)
        smem_buf[i] = smem_buf[i-1] + tiles_per_buf * TILE_BYTES;

    float acc[Tile::FYS_PER_WARP][Tile::FRAGS_X][4];
    #pragma unroll
    for (int fy_local = 0; fy_local < Tile::FYS_PER_WARP; fy_local++)
        #pragma unroll
        for (int fx = 0; fx < Tile::FRAGS_X; fx++)
            acc[fy_local][fx][0] = acc[fy_local][fx][1] = acc[fy_local][fx][2] = acc[fy_local][fx][3] = 0.0f;

    const int64_t A_batch_stride = static_cast<int64_t>(N) * K * 2;
    const unsigned char* A_batch = A_fp8 + batch * A_batch_stride;

    const int num_chunks = (K + Traits::K_CHUNK - 1) / Traits::K_CHUNK;

    // Prologue: issue READ_AHEAD loads with multi-pass tile loading
    for (int p = 0; p < Traits::READ_AHEAD && p < num_chunks; ++p) {
        unsigned char* buf = smem_buf[p];
        unsigned char* smem_row = buf;
        unsigned char* smem_col = is_diagonal ? smem_row : (smem_row + TILE_BYTES);
        int kt_p = p * Traits::K_CHUNK;

        #pragma unroll
        for (int lp = 0; lp < Tile::LOAD_PASSES; lp++) {
            load_tile_cp_async_t<Traits>(A_batch, smem_row + lp * 32 * Traits::SMEM_K,
                firstRecvRow + lp * 32, kt_p, K, N, tid);
        }
        if (!is_diagonal) {
            #pragma unroll
            for (int lp = 0; lp < Tile::LOAD_PASSES; lp++) {
                load_tile_cp_async_t<Traits>(A_batch, smem_col + lp * 32 * Traits::SMEM_K,
                    firstRecvCol + lp * 32, kt_p, K, N, tid);
            }
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

        // Prefetch next chunk
        int next_p = kt_idx + Traits::READ_AHEAD;
        if (next_p < num_chunks) {
            int next_buf_idx = next_p % Traits::NR_BUFS;
            unsigned char* next_buf = smem_buf[next_buf_idx];
            unsigned char* smem_row_next = next_buf;
            unsigned char* smem_col_next = is_diagonal ? smem_row_next : (smem_row_next + TILE_BYTES);
            int kt_next = next_p * Traits::K_CHUNK;

            #pragma unroll
            for (int lp = 0; lp < Tile::LOAD_PASSES; lp++) {
                load_tile_cp_async_t<Traits>(A_batch, smem_row_next + lp * 32 * Traits::SMEM_K,
                    firstRecvRow + lp * 32, kt_next, K, N, tid);
            }
            if (!is_diagonal) {
                #pragma unroll
                for (int lp = 0; lp < Tile::LOAD_PASSES; lp++) {
                    load_tile_cp_async_t<Traits>(A_batch, smem_col_next + lp * 32 * Traits::SMEM_K,
                        firstRecvCol + lp * 32, kt_next, K, N, tid);
                }
            }
            cp_async_commit();
        }

        unsigned char* smem_row = smem_buf[buf_idx];
        unsigned char* smem_col = is_diagonal ? smem_row : (smem_row + TILE_BYTES);

        compute_mma_herk_tiled_t<Traits, Tile>(smem_row, smem_col, acc,
            firstRecvRow, firstRecvCol, is_diagonal, warp_id, lane);

        __syncthreads();
    }

    // Store output
    OutputType* C_batch = C + batch * output_batch_stride;
    if (ScratchMode) {
        store_scratch_herk_tiled_t<Tile, OutputType>(acc, C_batch, firstRecvRow, firstRecvCol,
            is_diagonal, N, alpha, warp_id, lane);
    } else {
        store_triangle_herk_tiled_t<Tile, OutputType>(acc, C_batch, smem_raw, firstRecvRow, firstRecvCol,
            is_diagonal, N, alpha, beta, tid, warp_id, lane);
    }
}

// ========================================================================================
// Tiled persistent direct HERK kernel
// ========================================================================================
template <typename Traits, typename Tile, int BATCH_GROUP_, typename OutputType, bool ScratchMode = false>
__global__ void __launch_bounds__(Tile::THREADS)
herk_persistent_tiled_kernel_t(
    const unsigned char* __restrict__ A_fp8,
    OutputType* __restrict__ C,
    int N, int K, int batch_count,
    float alpha, float beta,
    int tri_blocks, int total_work_items,
    int64_t output_batch_stride)
{
    constexpr int TILE_BYTES = Tile::BLOCK_N * Traits::SMEM_K;
    extern __shared__ char smem_raw[];

    const int tid = threadIdx.x + threadIdx.y * 32 + threadIdx.z * 64;
    const int warp_id = tid / 32;
    const int lane = tid % 32;

    const int64_t A_batch_stride = static_cast<int64_t>(N) * K * 2;
    const int num_chunks = (K + Traits::K_CHUNK - 1) / Traits::K_CHUNK;
    const bool small_k = (K <= Traits::K_CHUNK);

    int total_work_groups = ((batch_count + BATCH_GROUP_ - 1) / BATCH_GROUP_) * tri_blocks;

    for (int work = blockIdx.x; work < total_work_groups; work += gridDim.x) {
        int group = work / tri_blocks;
        int tri_block = work % tri_blocks;
        int batch_base = group * BATCH_GROUP_;

        unsigned blockX = static_cast<unsigned>(__fsqrt_rn(8.0f * tri_block + 1.0f) - 0.999f) * 0.5f;
        while (blockX * (blockX + 1) / 2 > (unsigned)tri_block) --blockX;
        while ((blockX + 1) * (blockX + 2) / 2 <= (unsigned)tri_block) ++blockX;
        unsigned blockY = tri_block - blockX * (blockX + 1) / 2;

        const unsigned firstRecvRow = blockX * Tile::BLOCK_N;
        const unsigned firstRecvCol = blockY * Tile::BLOCK_N;
        const bool is_diagonal = (blockX == blockY);

        for (int bi = 0; bi < BATCH_GROUP_ && batch_base + bi < batch_count; ++bi) {
            int batch = batch_base + bi;

            const unsigned char* A_batch = A_fp8 + batch * A_batch_stride;
            OutputType* C_batch = C + batch * output_batch_stride;

            float acc[Tile::FYS_PER_WARP][Tile::FRAGS_X][4];
            #pragma unroll
            for (int fy_local = 0; fy_local < Tile::FYS_PER_WARP; fy_local++)
                #pragma unroll
                for (int fx = 0; fx < Tile::FRAGS_X; fx++)
                    acc[fy_local][fx][0] = acc[fy_local][fx][1] = acc[fy_local][fx][2] = acc[fy_local][fx][3] = 0.0f;

            if (small_k) {
                // ---- Small-K: single buffer, multi-pass direct loads ----
                unsigned char* smem_row = reinterpret_cast<unsigned char*>(smem_raw);
                unsigned char* smem_col = is_diagonal ? smem_row : (smem_row + TILE_BYTES);

                #pragma unroll
                for (int lp = 0; lp < Tile::LOAD_PASSES; lp++) {
                    load_tile_direct_fp8_t<Traits>(A_batch, smem_row + lp * 32 * Traits::SMEM_K,
                        firstRecvRow + lp * 32, 0, K, N, tid);
                }
                if (!is_diagonal) {
                    #pragma unroll
                    for (int lp = 0; lp < Tile::LOAD_PASSES; lp++) {
                        load_tile_direct_fp8_t<Traits>(A_batch, smem_col + lp * 32 * Traits::SMEM_K,
                            firstRecvCol + lp * 32, 0, K, N, tid);
                    }
                }
                __syncthreads();

                // Inline MMA with strided warp mapping and runtime actual_subs
                int actual_subs = (K + HERK_K_PER_MMA - 1) / HERK_K_PER_MMA;
                for (int ks = 0; ks < actual_subs; ks++) {
                    int k_base = ks * 32;

                    #pragma unroll
                    for (int fy_local = 0; fy_local < Tile::FYS_PER_WARP; fy_local++) {
                        int fy = warp_id + fy_local * Tile::NUM_WARPS;
                        int recv_base_y = fy * Tile::RECV_PER_MMA_Y;

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

                            unsigned v0 = *reinterpret_cast<const unsigned*>(&smem_row[smem_recv0 * Traits::SMEM_K + k_off0]);
                            unsigned v1 = *reinterpret_cast<const unsigned*>(&smem_row[smem_recv0 * Traits::SMEM_K + k_off1]);
                            a0 = is_conj ? conj_perm_fp8(v0) : v0;
                            a2 = is_conj ? conj_perm_fp8(v1) : v1;

                            if (smem_recv1 < Tile::BLOCK_N) {
                                unsigned v2 = *reinterpret_cast<const unsigned*>(&smem_row[smem_recv1 * Traits::SMEM_K + k_off0]);
                                unsigned v3 = *reinterpret_cast<const unsigned*>(&smem_row[smem_recv1 * Traits::SMEM_K + k_off1]);
                                a1 = is_conj ? conj_perm_fp8(v2) : v2;
                                a3 = is_conj ? conj_perm_fp8(v3) : v3;
                            } else {
                                a1 = 0; a3 = 0;
                            }
                        }

                        #pragma unroll
                        for (int fx = 0; fx < Tile::FRAGS_X; fx++) {
                            if (is_diagonal) {
                                unsigned row_start = firstRecvRow + fy * Tile::RECV_PER_MMA_Y;
                                unsigned col_start = firstRecvCol + fx * Tile::RECV_PER_MMA_X;
                                if (row_start < col_start) continue;
                            }
                            int recv_base_x = fx * Tile::RECV_PER_MMA_X;
                            unsigned b0, b1;
                            {
                                int recv_idx = recv_base_x + lane / 4;
                                int k_off0 = k_base + (lane % 4) * 4;
                                int k_off1 = k_base + 16 + (lane % 4) * 4;
                                if (recv_idx < Tile::BLOCK_N) {
                                    b0 = *reinterpret_cast<const unsigned*>(&smem_col[recv_idx * Traits::SMEM_K + k_off0]);
                                    b1 = *reinterpret_cast<const unsigned*>(&smem_col[recv_idx * Traits::SMEM_K + k_off1]);
                                } else {
                                    b0 = 0; b1 = 0;
                                }
                            }
                            mma_fp8_m16n8k32(
                                acc[fy_local][fx][0], acc[fy_local][fx][1], acc[fy_local][fx][2], acc[fy_local][fx][3],
                                a0, a1, a2, a3, b0, b1,
                                acc[fy_local][fx][0], acc[fy_local][fx][1], acc[fy_local][fx][2], acc[fy_local][fx][3]);
                        }
                    }
                }

                __syncthreads();

            } else {
                // ---- Large-K: multi-buffer cp.async pipeline with multi-pass loads ----
                const int tiles_per_buf = is_diagonal ? 1 : 2;
                unsigned char* smem_buf[Traits::NR_BUFS];
                smem_buf[0] = reinterpret_cast<unsigned char*>(smem_raw);
                for (int i = 1; i < Traits::NR_BUFS; i++)
                    smem_buf[i] = smem_buf[i-1] + tiles_per_buf * TILE_BYTES;

                for (int p = 0; p < Traits::READ_AHEAD && p < num_chunks; ++p) {
                    unsigned char* buf = smem_buf[p];
                    unsigned char* s_row = buf;
                    unsigned char* s_col = is_diagonal ? s_row : (s_row + TILE_BYTES);
                    int kt_p = p * Traits::K_CHUNK;
                    #pragma unroll
                    for (int lp = 0; lp < Tile::LOAD_PASSES; lp++) {
                        load_tile_cp_async_t<Traits>(A_batch, s_row + lp * 32 * Traits::SMEM_K,
                            firstRecvRow + lp * 32, kt_p, K, N, tid);
                    }
                    if (!is_diagonal) {
                        #pragma unroll
                        for (int lp = 0; lp < Tile::LOAD_PASSES; lp++) {
                            load_tile_cp_async_t<Traits>(A_batch, s_col + lp * 32 * Traits::SMEM_K,
                                firstRecvCol + lp * 32, kt_p, K, N, tid);
                        }
                    }
                    cp_async_commit();
                }

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
                        int next_buf_idx = next_p % Traits::NR_BUFS;
                        unsigned char* next_buf = smem_buf[next_buf_idx];
                        unsigned char* s_row_next = next_buf;
                        unsigned char* s_col_next = is_diagonal ? s_row_next : (s_row_next + TILE_BYTES);
                        int kt_next = next_p * Traits::K_CHUNK;
                        #pragma unroll
                        for (int lp = 0; lp < Tile::LOAD_PASSES; lp++) {
                            load_tile_cp_async_t<Traits>(A_batch, s_row_next + lp * 32 * Traits::SMEM_K,
                                firstRecvRow + lp * 32, kt_next, K, N, tid);
                        }
                        if (!is_diagonal) {
                            #pragma unroll
                            for (int lp = 0; lp < Tile::LOAD_PASSES; lp++) {
                                load_tile_cp_async_t<Traits>(A_batch, s_col_next + lp * 32 * Traits::SMEM_K,
                                    firstRecvCol + lp * 32, kt_next, K, N, tid);
                            }
                        }
                        cp_async_commit();
                    }

                    unsigned char* smem_row = smem_buf[buf_idx];
                    unsigned char* smem_col = is_diagonal ? smem_row : (smem_row + TILE_BYTES);
                    compute_mma_herk_tiled_t<Traits, Tile>(smem_row, smem_col, acc,
                        firstRecvRow, firstRecvCol, is_diagonal, warp_id, lane);
                    __syncthreads();
                }
            }

            // Store output
            if (ScratchMode) {
                store_scratch_herk_tiled_t<Tile, OutputType>(acc, C_batch, firstRecvRow, firstRecvCol,
                    is_diagonal, N, alpha, warp_id, lane);
            } else {
                store_triangle_herk_tiled_t<Tile, OutputType>(acc, C_batch, smem_raw, firstRecvRow, firstRecvCol,
                    is_diagonal, N, alpha, beta, tid, warp_id, lane);
            }
        }
    }
}

// ========================================================================================
// Tiled launch helpers
// ========================================================================================

template <typename Traits, typename Tile, typename OutputType, bool ScratchMode = false>
inline cutlass::Status launch_precast_tiled_t(
    const __nv_fp8_e4m3* A_fp8, OutputType* C,
    int N, int K, int batch_count,
    float alpha, float beta, FillMode fill,
    cudaStream_t stream)
{
    if (fill != FillMode::Lower) return cutlass::Status::kErrorNotSupported;

    int blocks_per_dim = (N + Tile::BLOCK_N - 1) / Tile::BLOCK_N;
    int tri_blocks = blocks_per_dim * (blocks_per_dim + 1) / 2;
    dim3 grid(tri_blocks, batch_count, 1);
    dim3 block(32, 2, 2);

    int64_t output_batch_stride = ScratchMode
        ? static_cast<int64_t>(N) * N * 2
        : static_cast<int64_t>(N) * (N + 1);

    constexpr int TILE_BYTES = Tile::BLOCK_N * Traits::SMEM_K;
    constexpr int PIPELINE_SMEM = Traits::NR_BUFS * 2 * TILE_BYTES;
    constexpr int STORE_SMEM = Tile::BLOCK_N * 64 * static_cast<int>(sizeof(float));
    int smem_bytes = ScratchMode ? PIPELINE_SMEM : std::max(PIPELINE_SMEM, STORE_SMEM);

    herk_precast_tiled_kernel_t<Traits, Tile, OutputType, ScratchMode><<<grid, block, smem_bytes, stream>>>(
        reinterpret_cast<const unsigned char*>(A_fp8),
        C, N, K, batch_count, alpha, beta, blocks_per_dim,
        output_batch_stride);
    CUDA_CHECK(cudaGetLastError());
    return cutlass::Status::kSuccess;
}

template <typename Traits, typename Tile, int BATCH_GROUP, typename OutputType, bool ScratchMode = false>
inline cutlass::Status launch_persistent_tiled_t(
    const __nv_fp8_e4m3* A_fp8, OutputType* C,
    int N, int K, int batch_count,
    float alpha, float beta, FillMode fill,
    int sm_count, cudaStream_t stream)
{
    if (fill != FillMode::Lower) return cutlass::Status::kErrorNotSupported;

    int blocks_per_dim = (N + Tile::BLOCK_N - 1) / Tile::BLOCK_N;
    int tri_blocks = blocks_per_dim * (blocks_per_dim + 1) / 2;
    int total_work = tri_blocks * batch_count;
    constexpr int TARGET_OCCUPANCY = 4;
    int grid_x = std::min(sm_count * TARGET_OCCUPANCY, total_work);

    int64_t output_batch_stride = ScratchMode
        ? static_cast<int64_t>(N) * N * 2
        : static_cast<int64_t>(N) * (N + 1);

    constexpr int TILE_BYTES = Tile::BLOCK_N * Traits::SMEM_K;
    constexpr int STORE_SMEM = Tile::BLOCK_N * 64 * static_cast<int>(sizeof(float));
    int smem_bytes;
    if (ScratchMode) {
        smem_bytes = (K <= Traits::K_CHUNK) ? (2 * TILE_BYTES)
                                            : (Traits::NR_BUFS * 2 * TILE_BYTES);
    } else {
        smem_bytes = (K <= Traits::K_CHUNK)
            ? std::max(2 * TILE_BYTES, STORE_SMEM)
            : std::max(Traits::NR_BUFS * 2 * TILE_BYTES, STORE_SMEM);
    }

    dim3 grid(grid_x, 1, 1);
    dim3 block(32, 2, 2);
    herk_persistent_tiled_kernel_t<Traits, Tile, BATCH_GROUP, OutputType, ScratchMode>
        <<<grid, block, smem_bytes, stream>>>(
        reinterpret_cast<const unsigned char*>(A_fp8),
        C, N, K, batch_count, alpha, beta, tri_blocks, total_work,
        output_batch_stride);
    CUDA_CHECK(cudaGetLastError());
    return cutlass::Status::kSuccess;
}

// ========================================================================================
// Tiled dispatch: inner tile dispatch (called per DirectHerkConfig)
// ========================================================================================

template <typename Traits, typename OutputType, bool ScratchMode>
inline cutlass::Status launch_precast_tile_dispatch(
    HerkTileSize tile,
    const __nv_fp8_e4m3* A_fp8, OutputType* C,
    int N, int K, int batch_count,
    float alpha, float beta, FillMode fill,
    cudaStream_t stream)
{
    switch (tile) {
    case HerkTileSize::N32: return launch_precast_tiled_t<Traits, HerkTile32, OutputType, ScratchMode>(A_fp8, C, N, K, batch_count, alpha, beta, fill, stream);
    case HerkTileSize::N64: return launch_precast_tiled_t<Traits, HerkTile64, OutputType, ScratchMode>(A_fp8, C, N, K, batch_count, alpha, beta, fill, stream);
    default: return cutlass::Status::kErrorNotSupported;
    }
}

template <typename Traits, typename OutputType, bool ScratchMode>
inline cutlass::Status launch_persistent_tile_dispatch(
    HerkTileSize tile,
    const __nv_fp8_e4m3* A_fp8, OutputType* C,
    int N, int K, int batch_count,
    float alpha, float beta, FillMode fill,
    int sm_count, cudaStream_t stream)
{
    switch (tile) {
    case HerkTileSize::N32: return launch_persistent_tiled_t<Traits, HerkTile32, 4, OutputType, ScratchMode>(A_fp8, C, N, K, batch_count, alpha, beta, fill, sm_count, stream);
    case HerkTileSize::N64: return launch_persistent_tiled_t<Traits, HerkTile64, 4, OutputType, ScratchMode>(A_fp8, C, N, K, batch_count, alpha, beta, fill, sm_count, stream);
    default: return cutlass::Status::kErrorNotSupported;
    }
}

// ========================================================================================
// Tiled dispatch: outer config + tile dispatch
// ========================================================================================

template <typename OutputType, bool ScratchMode = false>
inline cutlass::Status launch_herk_direct_tiled_dispatch(
    DirectHerkConfig config, HerkTileSize tile,
    const __nv_fp8_e4m3* A_fp8, OutputType* C,
    int N, int K, int batch_count,
    float alpha, float beta, FillMode fill,
    cudaStream_t stream)
{
    switch (config) {
    case DirectHerkConfig::K32_B3:  return launch_precast_tile_dispatch<TraitsK32_B3, OutputType, ScratchMode>(tile, A_fp8, C, N, K, batch_count, alpha, beta, fill, stream);
    case DirectHerkConfig::K64_B2:  return launch_precast_tile_dispatch<TraitsK64_B2, OutputType, ScratchMode>(tile, A_fp8, C, N, K, batch_count, alpha, beta, fill, stream);
    case DirectHerkConfig::K64_B3:  return launch_precast_tile_dispatch<TraitsK64_B3, OutputType, ScratchMode>(tile, A_fp8, C, N, K, batch_count, alpha, beta, fill, stream);
    case DirectHerkConfig::K128_B2: return launch_precast_tile_dispatch<TraitsK128_B2, OutputType, ScratchMode>(tile, A_fp8, C, N, K, batch_count, alpha, beta, fill, stream);
    default: return cutlass::Status::kErrorNotSupported;
    }
}

template <typename OutputType, bool ScratchMode = false>
inline cutlass::Status launch_herk_persistent_tiled_dispatch(
    DirectHerkConfig config, HerkTileSize tile,
    const __nv_fp8_e4m3* A_fp8, OutputType* C,
    int N, int K, int batch_count,
    float alpha, float beta, FillMode fill,
    int sm_count, cudaStream_t stream)
{
    switch (config) {
    case DirectHerkConfig::K32_B3:  return launch_persistent_tile_dispatch<TraitsK32_B3, OutputType, ScratchMode>(tile, A_fp8, C, N, K, batch_count, alpha, beta, fill, sm_count, stream);
    case DirectHerkConfig::K64_B2:  return launch_persistent_tile_dispatch<TraitsK64_B2, OutputType, ScratchMode>(tile, A_fp8, C, N, K, batch_count, alpha, beta, fill, sm_count, stream);
    case DirectHerkConfig::K64_B3:  return launch_persistent_tile_dispatch<TraitsK64_B3, OutputType, ScratchMode>(tile, A_fp8, C, N, K, batch_count, alpha, beta, fill, sm_count, stream);
    case DirectHerkConfig::K128_B2: return launch_persistent_tile_dispatch<TraitsK128_B2, OutputType, ScratchMode>(tile, A_fp8, C, N, K, batch_count, alpha, beta, fill, sm_count, stream);
    default: return cutlass::Status::kErrorNotSupported;
    }
}

// ========================================================================================
// DirectHerkConfig dispatch wrappers (backward-compatible: default tile = N32)
// ========================================================================================
template <typename OutputType, bool ScratchMode = false>
inline cutlass::Status launch_herk_direct_dispatch(
    DirectHerkConfig config,
    const __nv_fp8_e4m3* A_fp8, OutputType* C,
    int N, int K, int batch_count,
    float alpha, float beta, FillMode fill,
    cudaStream_t stream,
    HerkTileSize tile = HerkTileSize::N32)
{
    return launch_herk_direct_tiled_dispatch<OutputType, ScratchMode>(
        config, tile, A_fp8, C, N, K, batch_count, alpha, beta, fill, stream);
}

template <typename OutputType, bool ScratchMode = false>
inline cutlass::Status launch_herk_persistent_dispatch(
    DirectHerkConfig config,
    const __nv_fp8_e4m3* A_fp8, OutputType* C,
    int N, int K, int batch_count,
    float alpha, float beta, FillMode fill,
    int sm_count, cudaStream_t stream,
    HerkTileSize tile = HerkTileSize::N32)
{
    return launch_herk_persistent_tiled_dispatch<OutputType, ScratchMode>(
        config, tile, A_fp8, C, N, K, batch_count, alpha, beta, fill, sm_count, stream);
}

// ========================================================================================
// Warp-Specialized Direct HERK Kernel (Phase 2)
// ========================================================================================
//
// Replaces __syncthreads() with mbarrier-based producer-consumer coordination.
// Warp 0 = producer: issues cp.async loads, signals per-buffer full barriers.
// Warps 1-3 = consumers: wait on full barriers, execute MMAs, signal empty barriers.
//
// Benefits:
//   - Eliminates 2 __syncthreads() per K_CHUNK iteration (pipeline bubbles)
//   - Producer stays ahead of consumers (24 cp.async instr vs 80 MMA instr per chunk)
//   - Fully overlapped load-compute with no synchronization gaps
//
// Requirements: SM90+ (__CUDA_ARCH__ >= 900) for mbarrier PTX instructions.
// Falls back to sync-based kernel on SM80/SM89.
//

#include "mbarrier_helpers.hpp"

// ========================================================================================
// Warp-specialized precast kernel — mbarrier producer-consumer pipeline
// ========================================================================================
// The kernel is always defined (visible at host compilation), but the mbarrier
// PTX instructions inside are guarded by __CUDA_ARCH__ >= 900 in mbarrier_helpers.hpp.
// On pre-SM90, the stubs are no-ops and the kernel should never be launched.
template <typename Traits, typename Tile, typename OutputType, bool ScratchMode = false>
__global__ void __launch_bounds__(Tile::THREADS)
herk_warp_specialized_kernel_t(
    const unsigned char* __restrict__ A_fp8,
    OutputType* __restrict__ C,
    int N, int K, int batch_count,
    float alpha, float beta,
    int blocks_per_dim,
    int64_t output_batch_stride)
{
    constexpr int TILE_BYTES = Tile::BLOCK_N * Traits::SMEM_K;

    const unsigned block = blockIdx.x;
    const unsigned batch = blockIdx.y;

    unsigned blockX = static_cast<unsigned>(__fsqrt_rn(8.0f * block + 1.0f) - 0.999f) * 0.5f;
    while (blockX * (blockX + 1) / 2 > block) --blockX;
    while ((blockX + 1) * (blockX + 2) / 2 <= block) ++blockX;
    unsigned blockY = block - blockX * (blockX + 1) / 2;

    const unsigned firstRecvRow = blockX * Tile::BLOCK_N;
    const unsigned firstRecvCol = blockY * Tile::BLOCK_N;
    const bool is_diagonal = (blockX == blockY);

    const int tid = threadIdx.x + threadIdx.y * 32 + threadIdx.z * 64;
    const int warp_id = tid / 32;
    const int lane = tid % 32;
    const bool is_producer = (warp_id == 0);

    // SMEM layout:
    //   [0, pipeline_smem): NR_BUFS pipeline tile buffers
    //   [pipeline_smem, pipeline_smem + barrier_smem): mbarrier arrays
    //   [pipeline_smem + barrier_smem, ...): store scratch (reuses pipeline buffer space)
    extern __shared__ char smem_raw[];
    const int tiles_per_buf = is_diagonal ? 1 : 2;
    const int buf_bytes = tiles_per_buf * TILE_BYTES;
    const int pipeline_smem = Traits::NR_BUFS * buf_bytes;

    unsigned char* smem_buf[Traits::NR_BUFS];
    smem_buf[0] = reinterpret_cast<unsigned char*>(smem_raw);
    for (int i = 1; i < Traits::NR_BUFS; i++)
        smem_buf[i] = smem_buf[i-1] + buf_bytes;

    // Barriers: full[i] = "buffer i has data", empty[i] = "buffer i is consumed"
    // Aligned to 8 bytes in shared memory after pipeline buffers.
    uint64_t* full_barriers  = reinterpret_cast<uint64_t*>(smem_raw + pipeline_smem);
    uint64_t* empty_barriers = full_barriers + Traits::NR_BUFS;

    // Initialize barriers — only one thread does this.
    // full_barriers: producer lane 0 arrives once → expected=1.
    // empty_barriers: one lane 0 per consumer warp arrives → expected=NUM_WARPS-1.
    if (tid == 0) {
        for (int i = 0; i < Traits::NR_BUFS; i++) {
            mbarrier_init(&full_barriers[i], 1);                    // producer lane 0 arrives once
            mbarrier_init(&empty_barriers[i], Tile::NUM_WARPS - 1); // 3 consumer warp leaders
        }
    }
    __syncthreads();  // ensure all barriers are initialized before use

    // Accumulator sized for warp-specialized consumer distribution.
    // 3 consumers distribute FRAGS_Y cyclically, so max fragments per consumer =
    // ceil(FRAGS_Y / NUM_CONSUMERS).  BN=32: ceil(4/3)=2.  BN=64: ceil(8/3)=3.
    constexpr int NUM_CONSUMERS_K = Tile::NUM_WARPS - 1;  // 3
    constexpr int WS_FYS_PER_CONSUMER = (Tile::FRAGS_Y + NUM_CONSUMERS_K - 1) / NUM_CONSUMERS_K;

    float acc[WS_FYS_PER_CONSUMER][Tile::FRAGS_X][4];
    #pragma unroll
    for (int fy_local = 0; fy_local < WS_FYS_PER_CONSUMER; fy_local++)
        #pragma unroll
        for (int fx = 0; fx < Tile::FRAGS_X; fx++)
            acc[fy_local][fx][0] = acc[fy_local][fx][1] = acc[fy_local][fx][2] = acc[fy_local][fx][3] = 0.0f;

    const int64_t A_batch_stride = static_cast<int64_t>(N) * K * 2;
    const unsigned char* A_batch = A_fp8 + batch * A_batch_stride;
    const int num_chunks = (K + Traits::K_CHUNK - 1) / Traits::K_CHUNK;

    // Phase tracking for parity-based mbarrier wait
    int full_phase[Traits::NR_BUFS];
    int empty_phase[Traits::NR_BUFS];
    for (int i = 0; i < Traits::NR_BUFS; i++) {
        full_phase[i] = 0;
        empty_phase[i] = 0;
    }

    if (is_producer) {
        // ---- PRODUCER WARP (warp 0) ----
        // Issues all cp.async loads, signals full_barrier when each buffer's loads are committed.
        // Waits on empty_barrier before reusing a buffer.

        for (int kt_idx = 0; kt_idx < num_chunks; ++kt_idx) {
            int buf_idx = kt_idx % Traits::NR_BUFS;

            // Wait for buffer to be consumed (skip on first NR_BUFS iterations)
            if (kt_idx >= Traits::NR_BUFS) {
                mbarrier_wait_parity(&empty_barriers[buf_idx], empty_phase[buf_idx]);
                empty_phase[buf_idx] ^= 1;
            }

            // Issue cp.async loads for this buffer.
            // The producer warp has 32 threads, but load_tile_cp_async_t expects 128 threads
            // (tid/4 = 32 rows). With 32 threads, we only cover 8 rows per call, so we need
            // NUM_WARPS (4) inner passes per 32-row sub-tile to cover all rows.
            unsigned char* buf = smem_buf[buf_idx];
            unsigned char* smem_row = buf;
            unsigned char* smem_col = is_diagonal ? smem_row : (smem_row + TILE_BYTES);
            int kt = kt_idx * Traits::K_CHUNK;

            #pragma unroll
            for (int lp = 0; lp < Tile::LOAD_PASSES; lp++) {
                #pragma unroll
                for (int tp = 0; tp < Tile::NUM_WARPS; tp++) {
                    load_tile_cp_async_t<Traits>(A_batch, smem_row + lp * 32 * Traits::SMEM_K,
                        firstRecvRow + lp * 32, kt, K, N, lane + tp * 32);
                }
            }
            if (!is_diagonal) {
                #pragma unroll
                for (int lp = 0; lp < Tile::LOAD_PASSES; lp++) {
                    #pragma unroll
                    for (int tp = 0; tp < Tile::NUM_WARPS; tp++) {
                        load_tile_cp_async_t<Traits>(A_batch, smem_col + lp * 32 * Traits::SMEM_K,
                            firstRecvCol + lp * 32, kt, K, N, lane + tp * 32);
                    }
                }
            }
            cp_async_commit();
            cp_async_wait<0>();
            __syncwarp();

            // Signal: buffer is full (only lane 0 to match expected_count=1)
            if (lane == 0) {
                mbarrier_arrive(&full_barriers[buf_idx]);
            }
        }

    } else {
        // ---- CONSUMER WARPS (warps 1-3) ----
        // Wait for full_barrier, compute MMA, signal empty_barrier.
        // Consumer warp_id is adjusted to 0-based for accumulator indexing:
        // warp_id 1→0, 2→1, 3→2 for a 3-consumer setup.
        int consumer_warp = warp_id - 1;
        constexpr int NUM_CONSUMERS = Tile::NUM_WARPS - 1;  // 3

        for (int kt_idx = 0; kt_idx < num_chunks; ++kt_idx) {
            int buf_idx = kt_idx % Traits::NR_BUFS;

            // Wait for buffer to be filled
            mbarrier_wait_parity(&full_barriers[buf_idx], full_phase[buf_idx]);
            full_phase[buf_idx] ^= 1;

            unsigned char* smem_row = smem_buf[buf_idx];
            unsigned char* smem_col = is_diagonal ? smem_row : (smem_row + TILE_BYTES);

            // Each consumer warp processes a subset of fy fragments.
            // With 3 consumers and FRAGS_Y fragments, distribute cyclically:
            //   consumer 0: fy = 0, 3, 6, ...
            //   consumer 1: fy = 1, 4, 7, ...
            //   consumer 2: fy = 2, 5, 8, ...
            constexpr int K_SUBS = Traits::K_CHUNK / HERK_K_PER_MMA;

            for (int ks = 0; ks < K_SUBS; ks++) {
                int k_base = ks * 32;

                for (int fy = consumer_warp; fy < Tile::FRAGS_Y; fy += NUM_CONSUMERS) {
                    // Map fy to accumulator index
                    int acc_idx = fy / NUM_CONSUMERS;

                    int recv_base_y = fy * Tile::RECV_PER_MMA_Y;

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

                        unsigned v0 = *reinterpret_cast<const unsigned*>(&smem_row[smem_recv0 * Traits::SMEM_K + k_off0]);
                        unsigned v1 = *reinterpret_cast<const unsigned*>(&smem_row[smem_recv0 * Traits::SMEM_K + k_off1]);
                        a0 = is_conj ? conj_perm_fp8(v0) : v0;
                        a2 = is_conj ? conj_perm_fp8(v1) : v1;

                        if (smem_recv1 < Tile::BLOCK_N) {
                            unsigned v2 = *reinterpret_cast<const unsigned*>(&smem_row[smem_recv1 * Traits::SMEM_K + k_off0]);
                            unsigned v3 = *reinterpret_cast<const unsigned*>(&smem_row[smem_recv1 * Traits::SMEM_K + k_off1]);
                            a1 = is_conj ? conj_perm_fp8(v2) : v2;
                            a3 = is_conj ? conj_perm_fp8(v3) : v3;
                        } else {
                            a1 = 0; a3 = 0;
                        }
                    }

                    #pragma unroll
                    for (int fx = 0; fx < Tile::FRAGS_X; fx++) {
                        if (is_diagonal) {
                            unsigned row_start = firstRecvRow + fy * Tile::RECV_PER_MMA_Y;
                            unsigned col_start = firstRecvCol + fx * Tile::RECV_PER_MMA_X;
                            if (row_start < col_start) continue;
                        }
                        int recv_base_x = fx * Tile::RECV_PER_MMA_X;
                        unsigned b0, b1;
                        {
                            int recv_idx = recv_base_x + lane / 4;
                            int k_off0 = k_base + (lane % 4) * 4;
                            int k_off1 = k_base + 16 + (lane % 4) * 4;
                            if (recv_idx < Tile::BLOCK_N) {
                                b0 = *reinterpret_cast<const unsigned*>(&smem_col[recv_idx * Traits::SMEM_K + k_off0]);
                                b1 = *reinterpret_cast<const unsigned*>(&smem_col[recv_idx * Traits::SMEM_K + k_off1]);
                            } else {
                                b0 = 0; b1 = 0;
                            }
                        }
                        mma_fp8_m16n8k32(
                            acc[acc_idx][fx][0], acc[acc_idx][fx][1], acc[acc_idx][fx][2], acc[acc_idx][fx][3],
                            a0, a1, a2, a3, b0, b1,
                            acc[acc_idx][fx][0], acc[acc_idx][fx][1], acc[acc_idx][fx][2], acc[acc_idx][fx][3]);
                    }
                }
            }

            __syncwarp();

            // One thread per consumer warp signals buffer consumed
            if (lane == 0) {
                mbarrier_arrive(&empty_barriers[buf_idx]);
            }
        }
    }

    // ---- STORE PHASE ----
    // All 4 warps participate. Need a full sync before store since producer
    // warp has no accumulators and consumers need to be done.
    __syncthreads();

    if (!is_producer) {
        // Consumer warps store their portion.
        // Remap warp_id for store: consumers 0-2 have fragments distributed cyclically.
        // Use the triangle store with proper fragment indexing.
        int consumer_warp = warp_id - 1;
        constexpr int NUM_CONSUMERS = Tile::NUM_WARPS - 1;

        OutputType* C_batch = C + batch * output_batch_stride;

        if (ScratchMode) {
            // Direct register-to-global store (no SMEM needed)
            for (int fy = consumer_warp; fy < Tile::FRAGS_Y; fy += NUM_CONSUMERS) {
                int acc_idx = fy / NUM_CONSUMERS;

                #pragma unroll
                for (int fx = 0; fx < Tile::FRAGS_X; fx++) {
                    int mma_row0 = lane / 4;
                    int mma_row1 = mma_row0 + 8;
                    int mma_col0 = (lane % 4) * 2;
                    int mma_col1 = mma_col0 + 1;

                    float d_vals[4] = { acc[acc_idx][fx][0], acc[acc_idx][fx][1],
                                        acc[acc_idx][fx][2], acc[acc_idx][fx][3] };
                    int mma_rows[4] = { mma_row0, mma_row0, mma_row1, mma_row1 };
                    int mma_cols[4] = { mma_col0, mma_col1, mma_col0, mma_col1 };

                    #pragma unroll
                    for (int vi = 0; vi < 4; vi++) {
                        int mr = mma_rows[vi];
                        int mc = mma_cols[vi];
                        int recv_local_row = mr / 2;
                        int is_im = mr & 1;
                        int recv_local_col = mc;

                        unsigned recv_row = firstRecvRow + fy * Tile::RECV_PER_MMA_Y + recv_local_row;
                        unsigned recv_col = firstRecvCol + fx * Tile::RECV_PER_MMA_X + recv_local_col;

                        if (recv_row >= (unsigned)N || recv_col >= (unsigned)N) continue;

                        float val = d_vals[vi];
                        if (is_diagonal && recv_row == recv_col && is_im) val = 0.0f;
                        val *= alpha;

                        int64_t idx = static_cast<int64_t>(recv_row) * (2 * N) + recv_col * 2 + is_im;
                        C_batch[idx] = float_to_out<OutputType>(val);
                    }
                }
            }
        } else {
            // Triangle store via SMEM — consumer warps write their fragments to SMEM,
            // then all 3 consumer warps flush to packed triangle.
            float* smem_store = reinterpret_cast<float*>(smem_raw);

            #pragma unroll
            for (int phase = 0; phase < Tile::STORE_PHASES; phase++) {
                int fx_start = phase * 4;

                // Scatter accumulators to SMEM
                for (int fy = consumer_warp; fy < Tile::FRAGS_Y; fy += NUM_CONSUMERS) {
                    int acc_idx = fy / NUM_CONSUMERS;

                    #pragma unroll
                    for (int fx_off = 0; fx_off < 4 && fx_start + fx_off < Tile::FRAGS_X; fx_off++) {
                        int fx = fx_start + fx_off;
                        int mma_row0 = lane / 4;
                        int mma_row1 = mma_row0 + 8;
                        int mma_col0 = (lane % 4) * 2;
                        int mma_col1 = mma_col0 + 1;

                        float d_vals[4] = { acc[acc_idx][fx][0], acc[acc_idx][fx][1],
                                            acc[acc_idx][fx][2], acc[acc_idx][fx][3] };
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

                __syncthreads();  // ensure all consumer warps have written SMEM

                // Coalesced row-by-row flush to packed triangle (all 3 consumer warps participate)
                // 96 threads (warps 1-3) write rows cooperatively
                int consumer_tid = (warp_id - 1) * 32 + lane;
                int consumer_count = NUM_CONSUMERS * 32;  // 96

                for (int r = 0; r < Tile::BLOCK_N; r++) {
                    for (int elem = consumer_tid; elem < 64; elem += consumer_count) {
                        int local_col = elem / 2;
                        int is_im = elem & 1;

                        unsigned recv_row = firstRecvRow + r;
                        unsigned recv_col = firstRecvCol + phase * 32 + local_col;

                        if (recv_row < (unsigned)N && recv_col < (unsigned)N && recv_row >= recv_col) {
                            float val = smem_store[r * 64 + elem];

                            if (recv_row == recv_col && is_im) val = 0.0f;
                            val *= alpha;

                            int64_t tri_idx = static_cast<int64_t>(recv_row) * (recv_row + 1) / 2 + recv_col;
                            int64_t out_idx = tri_idx * 2 + is_im;

                            if (beta != 0.0f) {
                                val += beta * out_to_float<OutputType>(C_batch[out_idx]);
                            }

                            C_batch[out_idx] = float_to_out<OutputType>(val);
                        }
                    }
                }

                if (Tile::STORE_PHASES > 1) __syncthreads();
            }
        }
    }
}

// ========================================================================================
// Warp-specialized launch helpers
// ========================================================================================

template <typename Traits, typename Tile, typename OutputType, bool ScratchMode = false>
struct launch_ws_tiled_t {
    static cutlass::Status launch(
        const __nv_fp8_e4m3* A_fp8, OutputType* C,
        int N, int K, int batch_count,
        float alpha, float beta, FillMode fill,
        cudaStream_t stream)
    {
        const int blocks_per_dim = (N + Tile::BLOCK_N - 1) / Tile::BLOCK_N;
        const int tri_blocks = blocks_per_dim * (blocks_per_dim + 1) / 2;

        int64_t output_batch_stride;
        if (ScratchMode) {
            output_batch_stride = static_cast<int64_t>(N) * N * 2;
        } else {
            output_batch_stride = static_cast<int64_t>(N) * (N + 1);
        }

        constexpr int TILE_BYTES = Tile::BLOCK_N * Traits::SMEM_K;
        // Pipeline SMEM + barrier SMEM (2 × NR_BUFS × 8 bytes for mbarriers)
        int smem_pipeline = 2 * Traits::NR_BUFS * TILE_BYTES;  // worst case: non-diagonal
        int smem_barriers = 2 * Traits::NR_BUFS * sizeof(uint64_t);
        // Store SMEM: reuse pipeline buffer space (>= BLOCK_N * 64 * 4)
        int smem_store = Tile::BLOCK_N * 64 * sizeof(float);
        int smem_bytes = max(smem_pipeline + smem_barriers, smem_store);

        // Configure max dynamic shared memory
        cudaFuncSetAttribute(
            herk_warp_specialized_kernel_t<Traits, Tile, OutputType, ScratchMode>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            smem_bytes);

        dim3 grid(tri_blocks, batch_count);
        dim3 block_dim(32, 2, 2);  // 128 threads = 4 warps

        herk_warp_specialized_kernel_t<Traits, Tile, OutputType, ScratchMode>
            <<<grid, block_dim, smem_bytes, stream>>>(
                reinterpret_cast<const unsigned char*>(A_fp8), C,
                N, K, batch_count, alpha, beta,
                blocks_per_dim, output_batch_stride);

        cudaError_t err = cudaGetLastError();
        return err == cudaSuccess ? cutlass::Status::kSuccess : cutlass::Status::kErrorInternal;
    }
};

// Tile dispatch for warp-specialized kernel
template <typename Traits, typename OutputType, bool ScratchMode = false>
inline cutlass::Status launch_ws_tile_dispatch(
    HerkTileSize tile,
    const __nv_fp8_e4m3* A_fp8, OutputType* C,
    int N, int K, int batch_count,
    float alpha, float beta, FillMode fill,
    cudaStream_t stream)
{
    switch (tile) {
    case HerkTileSize::N32:
        return launch_ws_tiled_t<Traits, HerkTile32, OutputType, ScratchMode>::launch(
            A_fp8, C, N, K, batch_count, alpha, beta, fill, stream);
    case HerkTileSize::N64:
        return launch_ws_tiled_t<Traits, HerkTile64, OutputType, ScratchMode>::launch(
            A_fp8, C, N, K, batch_count, alpha, beta, fill, stream);
    default:
        return cutlass::Status::kErrorNotSupported;
    }
}

// Config dispatch for warp-specialized kernel
template <typename OutputType, bool ScratchMode = false>
inline cutlass::Status launch_herk_ws_dispatch(
    DirectHerkConfig config, HerkTileSize tile,
    const __nv_fp8_e4m3* A_fp8, OutputType* C,
    int N, int K, int batch_count,
    float alpha, float beta, FillMode fill,
    cudaStream_t stream)
{
    switch (config) {
    case DirectHerkConfig::K32_B3:  return launch_ws_tile_dispatch<TraitsK32_B3, OutputType, ScratchMode>(tile, A_fp8, C, N, K, batch_count, alpha, beta, fill, stream);
    case DirectHerkConfig::K64_B2:  return launch_ws_tile_dispatch<TraitsK64_B2, OutputType, ScratchMode>(tile, A_fp8, C, N, K, batch_count, alpha, beta, fill, stream);
    case DirectHerkConfig::K64_B3:  return launch_ws_tile_dispatch<TraitsK64_B3, OutputType, ScratchMode>(tile, A_fp8, C, N, K, batch_count, alpha, beta, fill, stream);
    case DirectHerkConfig::K128_B2: return launch_ws_tile_dispatch<TraitsK128_B2, OutputType, ScratchMode>(tile, A_fp8, C, N, K, batch_count, alpha, beta, fill, stream);
    default: return cutlass::Status::kErrorNotSupported;
    }
}
