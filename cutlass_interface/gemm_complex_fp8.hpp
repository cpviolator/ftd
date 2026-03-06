/*
 * gemm_complex_fp8.hpp
 * 
 * Senior CUDA / CUTLASS Integration: Complex FP8 GEMM for Hopper (SM90) & Blackwell (SM100)
 *
 * ========================================================================================
 * ARCHITECTURE OVERVIEW
 * ========================================================================================
 *
 * Problem:
 *   Input data is FP16 (half), but we want FP8 tensor core throughput for complex GEMM.
 *   We must support both standard C = α(A×B) + βC and Hermitian C = α(A×B^H) + βC.
 *
 * Key Insight — Why Complex FP8 Requires Decomposition:
 *   NVIDIA Hopper/Blackwell tensor cores do NOT have native complex-number FP8 instructions.
 *   Complex GEMM must be decomposed into real-valued FP8 GEMMs. We use the "4M" algorithm:
 *
 *     Given A = Ar + i·Ai, B = Br + i·Bi:
 *       Re(C) = Ar·Br − Ai·Bi
 *       Im(C) = Ar·Bi + Ai·Br
 *
 *   For Hermitian (B^H = conj(B)^T = Br^T − i·Bi^T):
 *       Re(C) = Ar·Br^T + Ai·Bi^T
 *       Im(C) = Ai·Br^T − Ar·Bi^T
 *
 *   This gives us 4 real FP8 GEMMs per complex GEMM. A Gauss-style 3M variant is possible
 *   but introduces numerical instability with FP8's limited dynamic range — we avoid it here.
 *
 * Data Layout Strategy (Planar Complex):
 *   We use PLANAR COMPLEX layout: real and imaginary parts are stored in separate contiguous
 *   buffers. This is superior to interleaved complex for FP8 because:
 *     1. TMA (Tensor Memory Accelerator) on Hopper can load contiguous FP8 tiles directly.
 *     2. No need to deinterleave inside shared memory — saves SMEM bandwidth.
 *     3. Each real-valued sub-GEMM is a standard dense GEMM, fully optimized by CUTLASS.
 *     4. FP16→FP8 on-the-fly conversion is simpler on contiguous real/imag streams.
 *
 * FP16 → FP8 Conversion Strategy:
 *   We use a SEPARATE PREPROCESS KERNEL rather than in-loader conversion because:
 *     1. CUTLASS 3.x TMA-based mainloops on SM90 expect the source data type to match
 *        ElementA/ElementB. The TMA descriptor is typed — you cannot ask TMA to load FP16
 *        and silently reinterpret as FP8.
 *     2. A dedicated cast kernel can apply STOCHASTIC ROUNDING or custom clamping to mitigate
 *        FP8's limited range (e4m3: ±448, e5m2: ±57344). This is critical for accuracy.
 *     3. The cast kernel runs once; the GEMM sub-kernels run 4 times. Amortized cost is low.
 *     4. Separation of concerns: easier to profile, debug, and swap FP8 variants.
 *
 *   Alternative considered (and rejected for SM90):
 *     CUTLASS 2.x style iterators *can* do on-the-fly conversion via custom `PredicatedTileAccessIterator`
 *     with a converter functor. However, CUTLASS 3.x/CuTe on SM90 uses TMA which bypasses
 *     these iterators entirely. On-the-fly conversion would require a post-TMA transform in
 *     shared memory, adding SMEM pressure and pipeline stalls. Not worth it.
 *
 * ========================================================================================
 * CUTLASS 3.x TEMPLATE PARAMETER GUIDE (SM90 Hopper)
 * ========================================================================================
 *
 * For CUTLASS 3.x, we use the "kernel builder" pattern:
 *   cutlass::gemm::kernel::GemmUniversal<ProblemShape, CollectiveMainloop, CollectiveEpilogue, TileScheduler>
 *
 * Key type aliases and why we chose them:
 *
 *   ElementA / ElementB = cutlass::float_e4m3_t (FP8 E4M3)
 *     - E4M3 has 4 exponent bits, 3 mantissa bits → range ±448, precision ~0.125 at 1.0
 *     - Better precision than E5M2 for most ML workloads (E5M2 is better for gradients)
 *     - Hopper wgmma.mma_async supports both e4m3 and e5m2 natively
 *
 *   ElementAccumulator = float (FP32)
 *     - Tensor cores accumulate in FP32 even when inputs are FP8. This is mandatory on Hopper.
 *     - Using FP16 accumulation with FP8 inputs would destroy numerical accuracy.
 *
 *   ElementC / ElementD = cutlass::half_t (FP16)
 *     - Output stays in FP16 to match the user's memory layout for complex components.
 *     - The epilogue handles FP32 accumulator → FP16 output conversion.
 *
 *   LayoutA = cutlass::layout::RowMajor
 *   LayoutB = cutlass::layout::ColumnMajor
 *     - This gives us the "TN" GEMM format, which is optimal for Hopper wgmma because:
 *       * A in row-major and B in column-major means both are accessed along the K dimension
 *         contiguously, which aligns with TMA's 2D tile descriptors.
 *       * For Hermitian mode, we need B^T — with B already column-major, B^T is row-major,
 *         so we can just swap the layout tag without physically transposing.
 *
 *   TileShape = cute::Shape<_128, _256, _128>  (configurable, see COMPLEX_FP8_TILE_*)
 *     - 128×256 output tile, K-tile of 128.
 *     - WHY 128×256: Benchmark-driven selection on GH200. Doubling N from 128→256
 *       doubles A-tile reuse (each A row multiplies 256 B columns), cutting A memory
 *       traffic in half. Measured 940 TFLOPS vs 867 for 128×128×128 at 4096³.
 *     - K=128 gives 4 wgmma k-iterations (K=32 per wgmma), providing enough ILP
 *       to hide the ~30-cycle wgmma latency.
 *
 *   ClusterShape = cute::Shape<_1, _1, _1>  (configurable, see COMPLEX_FP8_CLUSTER_*)
 *     - Single-SM cluster. Benchmark showed clusters (2×1, 1×2) don't help at 4096³
 *       due to insufficient tile count for TMA multicast amortization. At 8192³+,
 *       try _1×_2×_1 for B-tile multicast.
 *
 *   KernelSchedule = KernelTmaWarpSpecializedCooperativeFP8FastAccum  (configurable)
 *     - Persistent kernel: 1 TMA producer + 2 MMA consumer warp groups on same C tile.
 *     - FastAccum skips periodic FP32 promotion → higher throughput, slight accuracy loss.
 *     - Measured best at 4096³. PingpongFP8FastAccum expected to win at 8192³+.
 *
 *   StageCount = cutlass::gemm::collective::StageCountAutoCarveout<sizeof(EpilogueSharedStorage)>
 *     - Let CUTLASS auto-determine the number of pipeline stages based on available SMEM
 *       after reserving space for the epilogue. Typically yields 4-7 stages for 128×256×128.
 *
 * ========================================================================================
 * FP6 FEASIBILITY CHECK
 * ========================================================================================
 *
 * As of CUTLASS 3.5.x (mid-2024):
 *   - FP6 (E3M2 or E2M3) is NOT a standard CUTLASS data type.
 *   - Blackwell (SM100) introduces hardware support for "microscaling" (MX) formats including
 *     MXFP6 as part of the MX specification. This is block-scaled FP6 with a shared exponent
 *     per block of 32 elements.
 *   - CUTLASS will likely add support via types like `cutlass::float_e3m2_t` or 
 *     `cutlass::mx_float6_t` with corresponding collective mainloop specializations.
 *   - For now, to simulate FP6, one could:
 *       a) Pack two FP6 values into 12 bits and use custom shared memory unpacking.
 *       b) Use INT8 as a transport type and unpack in registers — but this breaks TMA typing.
 *       c) Wait for official CUTLASS MX format support for Blackwell.
 *   - Placeholder type alias is provided below for forward compatibility.
 *
 * ========================================================================================
 */

#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <cassert>

// ---- CUTLASS 3.x Core Includes ----
#include "cutlass/cutlass.h"
#include "cutlass/numeric_types.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/util/packed_stride.hpp"
#include "cutlass/kernel_hardware_info.hpp"

// ---- Grouped GEMM (SM90) ----
#include "cutlass/gemm/group_array_problem_shape.hpp"
#include "cutlass/epilogue/fusion/operations.hpp"

// ---- CuTe ----
#include "cute/tensor.hpp"

// ---- CUDA Runtime ----
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>

// ---- Kernel Autotuning Cache ----
#include "tune_cache.hpp"

// CUDA 12.9+ strongly recommended for:
//   - SM90a real-architecture codegen (native SASS, no JIT overhead)
//   - Improved wgmma async pipeline inlining (C7510 mitigation)
//   - --extra-device-vectorization for wider memory coalescing
//   - Device LTO improvements for template-heavy code
#if defined(__CUDACC_VER_MAJOR__) && defined(__CUDACC_VER_MINOR__)
#if __CUDACC_VER_MAJOR__ < 12 || (__CUDACC_VER_MAJOR__ == 12 && __CUDACC_VER_MINOR__ < 9)
#warning "gemm_complex_fp8.hpp: CUDA 12.9+ recommended. Older toolkits may exhibit wgmma serialization (C7510) and reduced performance."
#endif
#endif

namespace gemm_complex_fp8 {


// Free-standing kernels and utilities (textual includes)
#include "gemm_complex_fp8/config.hpp"
#include "gemm_complex_fp8/cast_kernels.hpp"
#include "gemm_complex_fp8/pack_kernels.hpp"
#include "gemm_complex_fp8/type_chains.hpp"
#include "gemm_complex_fp8/buffers.hpp"
#include "gemm_complex_fp8/herk_kernel.hpp"
#include "shared/gemm_kernel_common.hpp"

// ========================================================================================
// GemmComplexFP8 — The Public API
// ========================================================================================

/*
 * Usage:
 *   GemmComplexFP8 gemm;
 *
 *   // Planar complex layout: real and imaginary parts are separate contiguous buffers.
 *   // A_real, A_imag are each M×K in FP16.
 *   // B_real, B_imag are each K×N in FP16.
 *   // C_real, C_imag are each M×N in FP16 (input/output).
 *   gemm.run(A_real, A_imag, B_real, B_imag, C_real, C_imag,
 *            M, N, K, alpha, beta, ComplexMode::Hermitian);
 *
 * The simplified interface (interleaved half* pointers) is also provided:
 *   gemm.run(A, B, C, M, N, K, true);  // is_hermitian=true
 *   Here A/B/C are interleaved complex: [re0, im0, re1, im1, ...] in FP16.
 *   The wrapper deinterleaves internally.
 */
class GemmComplexFP8 {
public:
    GemmComplexFP8() = default;

    ~GemmComplexFP8() {
        prepared_b_.free_all();
        destroy_streams();
    }

    // Non-copyable (owns CUDA resources)
    GemmComplexFP8(const GemmComplexFP8&) = delete;
    GemmComplexFP8& operator=(const GemmComplexFP8&) = delete;
    GemmComplexFP8(GemmComplexFP8&&) = delete;
    GemmComplexFP8& operator=(GemmComplexFP8&&) = delete;

    /// Enable CUDA graph capture/replay for baseline HERK (non-triangle path).
    void set_herk_graph(bool enable) { use_herk_graph_ = enable; }
    bool herk_graph_enabled() const { return use_herk_graph_; }

    /// Enable/disable batch tiling in HERK_batched (L2 scratch optimization).
    /// When enabled (default), batch dimension is tiled to keep scratch in L2.
    void set_batch_tiling(bool enable) { use_batch_tiling_ = enable; }
    bool batch_tiling_enabled() const { return use_batch_tiling_; }

    /// HERK dispatch mode: Auto selects direct for small K, baseline for large K.
    void set_herk_mode(HerkMode mode) { herk_mode_ = mode; }
    HerkMode herk_mode() const { return herk_mode_; }

    /// Convenience setter: direct=true → ForceDirect, false → ForceBaseline.
    void set_direct_herk(bool enable) {
        herk_mode_ = enable ? HerkMode::ForceDirect : HerkMode::ForceBaseline;
    }
    bool direct_herk_enabled() const { return herk_mode_ != HerkMode::ForceBaseline; }

    /// Backward-compatible aliases (deprecated: use set_direct_herk / direct_herk_enabled).
    void set_fused_herk(bool enable) { set_direct_herk(enable); }
    bool fused_herk_enabled() const { return direct_herk_enabled(); }

    /// Persistent kernel mode for direct HERK: Auto selects based on work count.
    void set_persistent_mode(PersistentMode mode) { persistent_mode_ = mode; }
    PersistentMode persistent_mode() const { return persistent_mode_; }

    /// Set CUTLASS GEMM tile/cluster/schedule configuration for FP8 dispatch.
    /// Default uses the compile-time configured tile. Autotuner overrides this.
    /// Configs invalid for SM90 return kErrorNotSupported at dispatch.
    void set_gemm_config(GemmConfig config) { gemm_config_ = config; }
    GemmConfig gemm_config() const { return gemm_config_; }

    /// Set direct HERK kernel configuration (K_CHUNK/pipeline depth variant).
    /// Default is K64_B3. Autotuner sweeps all variants.
    void set_direct_herk_config(DirectHerkConfig config) { direct_herk_config_ = config; }
    DirectHerkConfig direct_herk_config() const { return direct_herk_config_; }

    /// Set direct HERK tile size (N32 or N64). Default is N32.
    void set_herk_tile_size(HerkTileSize tile) { herk_tile_size_ = tile; }
    HerkTileSize herk_tile_size() const { return herk_tile_size_; }

    /// Set direct HERK pipeline mode (Sync or WarpSpecialized). Default is Sync.
    void set_herk_pipeline_mode(HerkPipelineMode mode) { herk_pipeline_mode_ = mode; }
    HerkPipelineMode herk_pipeline_mode() const { return herk_pipeline_mode_; }

    /// Check if FP8 baseline GEMM kernels fit in device SMEM.
    /// On SM90, the FP8 baseline always fits (auto carveout), so this returns true.
    static bool fp8_baseline_available() {
        int max_smem = 0;
        cudaDeviceGetAttribute(&max_smem, cudaDevAttrMaxSharedMemoryPerBlockOptin, 0);
        return sizeof(typename GemmKernel::SharedStorage) <= static_cast<size_t>(max_smem);
    }

    /// ---------------------------------------------------------------
    /// Primary Interface: Planar Complex (recommended for performance)
    /// ---------------------------------------------------------------
    ///
    /// All pointers are device pointers in FP16.
    /// A_real, A_imag: [M × K] row-major
    /// B_real, B_imag: [K × N] column-major
    /// C_real, C_imag: [M × N] row-major (both input and output)
    ///
    /// Computes:
    ///   Standard:   C = α·(A × B) + β·C
    ///   Hermitian:  C = α·(A × B^H) + β·C
    ///
    /// where all multiplications are in complex arithmetic, decomposed into real FP8 GEMMs.
    ///
    #include "gemm_complex_fp8/api_impl.hpp"



    /// ---------------------------------------------------------------
    /// PreparedBState — pre-converted B data for the Prepare/Execute API
    /// ---------------------------------------------------------------
    struct PreparedBState {
        int N = 0, K = 0, batch_count = 0;
        cutlass::float_e4m3_t* B_real_fp8 = nullptr;
        cutlass::float_e4m3_t* B_imag_fp8 = nullptr;

        // Direct GEMM: B_neg interleaved FP8 with negated Im (owned)
        __nv_fp8_e4m3* B_neg_interleaved = nullptr;

        void* workspace_a = nullptr;
        void* workspace_b = nullptr;
        size_t workspace_size = 0;
        bool valid = false;

        void free_all() {
            if (B_real_fp8)        { cudaFree(B_real_fp8);        B_real_fp8 = nullptr; }
            if (B_imag_fp8)        { cudaFree(B_imag_fp8);        B_imag_fp8 = nullptr; }
            if (B_neg_interleaved) { cudaFree(B_neg_interleaved); B_neg_interleaved = nullptr; }
            if (workspace_a)       { cudaFree(workspace_a);       workspace_a = nullptr; }
            if (workspace_b)       { cudaFree(workspace_b);       workspace_b = nullptr; }
            workspace_size = 0;
            valid = false;
        }
    };

    /// Pre-convert B matrix (FP16→FP8) for reuse across multiple GEMM calls.
    /// Call once at precompute time. Subsequent calls via
    /// run_planar_batched_prepared_fp32out() skip B conversion entirely.
    void prepare_b_data(
        const __half* B_real, const __half* B_imag,
        int N, int K, int batch_count,
        cudaStream_t stream = nullptr)
    {
        prepared_b_.free_all();
        prepared_b_.N = N;
        prepared_b_.K = K;
        prepared_b_.batch_count = batch_count;

        ensure_streams();
        ensure_hw_info();

        int64_t size_B = static_cast<int64_t>(N) * K;
        int64_t total_B = size_B * batch_count;

        // Allocate persistent FP8 B buffers
        CUDA_CHECK(cudaMalloc(&prepared_b_.B_real_fp8,
                               total_B * sizeof(cutlass::float_e4m3_t)));
        CUDA_CHECK(cudaMalloc(&prepared_b_.B_imag_fp8,
                               total_B * sizeof(cutlass::float_e4m3_t)));

        // Convert FP16 → FP8 (planar, for 4M sub-GEMM path)
        cast_fp16_to_fp8_e4m3_paired(
            B_real, B_imag,
            reinterpret_cast<__nv_fp8_e4m3*>(prepared_b_.B_real_fp8),
            reinterpret_cast<__nv_fp8_e4m3*>(prepared_b_.B_imag_fp8),
            total_B, stream);

        // Also prepare B_neg interleaved for direct GEMM kernel:
        // [re, -im, re, -im, ...] interleaved FP8 with negated imaginary
        int64_t interleaved_size = total_B * 2;  // 2 FP8 per complex element
        CUDA_CHECK(cudaMalloc(&prepared_b_.B_neg_interleaved,
                               interleaved_size * sizeof(__nv_fp8_e4m3)));
        cast_fp16_planar_to_fp8_interleaved_negate_im(
            B_real, B_imag,
            prepared_b_.B_neg_interleaved,
            total_B, stream);

        prepared_b_.valid = true;
    }

    /// GEMM with pre-prepared B. Only converts A per-call.
    /// Lazy workspace allocation on first call (needs M to compute size).
    cutlass::Status run_planar_batched_prepared_fp32out(
        const __half* A_real, const __half* A_imag,
        float* C_real, float* C_imag,
        int M, int N, int K,
        int batch_count,
        float alpha, float beta,
        ComplexMode mode,
        cudaStream_t stream = nullptr)
    {
        if (!prepared_b_.valid)
            throw std::runtime_error("prepare_b_data() not called");
        if (N != prepared_b_.N || K != prepared_b_.K ||
            batch_count != prepared_b_.batch_count)
            throw std::runtime_error("Prepared B dimensions mismatch");

        ensure_streams();

        // Lazy workspace allocation (once, on first call with known M)
        if (prepared_b_.workspace_size == 0) {
            size_t ws = get_fp8_workspace_size_fp32out(M, N, K, batch_count);
            if (ws > 0) {
                CUDA_CHECK(cudaMalloc(&prepared_b_.workspace_a, ws));
                CUDA_CHECK(cudaMalloc(&prepared_b_.workspace_b, ws));
                prepared_b_.workspace_size = ws;
            }
        }

        // Only convert A (FP16 → FP8)
        int64_t size_A = static_cast<int64_t>(M) * K;
        int64_t total_A = size_A * batch_count;
        buffers_.ensure_capacity(total_A, 0, 0, stream);

        cast_fp16_to_fp8_e4m3_paired(
            A_real, A_imag,
            reinterpret_cast<__nv_fp8_e4m3*>(buffers_.A_real()),
            reinterpret_cast<__nv_fp8_e4m3*>(buffers_.A_imag()),
            total_A, stream);

        return run_subgemms_fp8_fp32out(
            buffers_.A_real(), buffers_.A_imag(),
            prepared_b_.B_real_fp8, prepared_b_.B_imag_fp8,
            C_real, C_imag, M, N, K,
            alpha, beta, mode, stream, batch_count,
            prepared_b_.workspace_a, prepared_b_.workspace_b,
            prepared_b_.workspace_size);
    }

    // ---------------------------------------------------------------
    // Batch-Fused M GEMM — Fuse batch elements along M for better TC util
    // ---------------------------------------------------------------

    /// Execute prepared GEMM with batch-fused M dimension (SM90, FP8 only).
    /// Fuses fuse_factor batch elements along M: M_fused = M * fuse_factor.
    /// B must be uniform across batches (only the first batch element's B is used).
    cutlass::Status run_planar_batched_prepared_fused_fp32out(
        const __half* A_real, const __half* A_imag,
        float* C_real, float* C_imag,
        int M, int N, int K,
        int batch_count,
        int fuse_factor,
        float alpha, float beta,
        ComplexMode mode,
        cudaStream_t stream = nullptr)
    {
        if (!prepared_b_.valid)
            throw std::runtime_error("prepare_b_data() not called");
        if (N != prepared_b_.N || K != prepared_b_.K)
            throw std::runtime_error("Prepared B dimensions mismatch (N or K)");
        if (fuse_factor <= 0 || batch_count % fuse_factor != 0)
            throw std::runtime_error("fuse_factor must be > 0 and divide batch_count evenly");

        int M_fused = M * fuse_factor;
        int effective_batch = batch_count / fuse_factor;

        ensure_streams();

        // Select tile config for fused M
        GemmConfig config = select_gemm_config_for_gemm(M_fused, N, K);

        // Lazy workspace allocation
        if (prepared_b_.workspace_size == 0) {
            size_t ws = get_fp8_workspace_size_fp32out(M_fused, N, K, effective_batch);
            if (ws > 0) {
                CUDA_CHECK(cudaMalloc(&prepared_b_.workspace_a, ws));
                CUDA_CHECK(cudaMalloc(&prepared_b_.workspace_b, ws));
                prepared_b_.workspace_size = ws;
            }
        }

        // Convert ALL A elements (full batch, contiguous in memory)
        // A layout: [batch_count, M, K] -> reinterpreted as [effective_batch, M_fused, K]
        int64_t total_A = (int64_t)M * K * batch_count;
        buffers_.ensure_capacity(total_A, 0, 0, stream);

        cast_fp16_to_fp8_e4m3_paired(
            A_real, A_imag,
            reinterpret_cast<__nv_fp8_e4m3*>(buffers_.A_real()),
            reinterpret_cast<__nv_fp8_e4m3*>(buffers_.A_imag()),
            total_A, stream);

        // Single batched call with M=M_fused, batch=effective_batch.
        // Internal batch strides: A = M_fused*K, B = N*K, C = M_fused*N.
        // B was prepared with prepared_b_.batch_count copies (all uniform),
        // and effective_batch <= prepared_b_.batch_count, so B[eb*N*K] is valid.
        return run_subgemms_fp8_fp32out(
            buffers_.A_real(), buffers_.A_imag(),
            prepared_b_.B_real_fp8, prepared_b_.B_imag_fp8,
            C_real, C_imag, M_fused, N, K,
            alpha, beta, mode, stream, effective_batch,
            prepared_b_.workspace_a, prepared_b_.workspace_b,
            prepared_b_.workspace_size);
    }

    // ---------------------------------------------------------------
    // Direct Complex GEMM — Single-launch kernel using conjugate permutation
    // ---------------------------------------------------------------

    /// Execute prepared GEMM via direct kernel (planar FP32 output).
    cutlass::Status run_direct_gemm_prepared_fp32out(
        const __half* A_real, const __half* A_imag,
        float* C_real, float* C_imag,
        int M, int N, int K,
        int batch_count,
        float alpha, float beta,
        cudaStream_t stream = nullptr)
    {
        if (!prepared_b_.valid)
            throw std::runtime_error("prepare_b_data() not called");
        if (N != prepared_b_.N || K != prepared_b_.K || batch_count != prepared_b_.batch_count)
            throw std::runtime_error("Prepared B dimensions mismatch");
        if (!prepared_b_.B_neg_interleaved)
            throw std::runtime_error("B_neg not available");

        ensure_streams();

        // Cast A from planar FP16 to interleaved FP8
        int64_t total_A = static_cast<int64_t>(M) * K * batch_count;
        ensure_gemm_a_interleaved(total_A * 2, stream);
        cast_fp16_planar_to_fp8_interleaved(
            A_real, A_imag,
            gemm_a_interleaved_buf_,
            total_A, stream);

        return launch_gemm_direct_planar(
            gemm_a_interleaved_buf_,
            prepared_b_.B_neg_interleaved,
            C_real, C_imag,
            M, N, K, batch_count,
            alpha, beta, stream);
    }

    /// Execute prepared GEMM with fused power detection.
    cutlass::Status run_direct_gemm_prepared_power(
        const __half* A_real, const __half* A_imag,
        float* C_power,
        int M, int N, int K,
        int batch_count,
        float alpha, float beta,
        cudaStream_t stream = nullptr)
    {
        if (!prepared_b_.valid)
            throw std::runtime_error("prepare_b_data() not called");
        if (N != prepared_b_.N || K != prepared_b_.K || batch_count != prepared_b_.batch_count)
            throw std::runtime_error("Prepared B dimensions mismatch");
        if (!prepared_b_.B_neg_interleaved)
            throw std::runtime_error("B_neg not available");

        ensure_streams();

        int64_t total_A = static_cast<int64_t>(M) * K * batch_count;
        ensure_gemm_a_interleaved(total_A * 2, stream);
        cast_fp16_planar_to_fp8_interleaved(
            A_real, A_imag,
            gemm_a_interleaved_buf_,
            total_A, stream);

        return launch_gemm_direct_power(
            gemm_a_interleaved_buf_,
            prepared_b_.B_neg_interleaved,
            C_power,
            M, N, K, batch_count,
            alpha, beta, stream);
    }

    /// Execute prepared GEMM with pre-cast FP8 A and fused power detection.
    cutlass::Status run_direct_gemm_prepared_power_fp8(
        const __nv_fp8_e4m3* A_fp8_interleaved,
        float* C_power,
        int M, int N, int K,
        int batch_count,
        float alpha, float beta,
        cudaStream_t stream = nullptr)
    {
        if (!prepared_b_.valid)
            throw std::runtime_error("prepare_b_data() not called");
        if (N != prepared_b_.N || K != prepared_b_.K || batch_count != prepared_b_.batch_count)
            throw std::runtime_error("Prepared B dimensions mismatch");
        if (!prepared_b_.B_neg_interleaved)
            throw std::runtime_error("B_neg not available");

        return launch_gemm_direct_power(
            A_fp8_interleaved,
            prepared_b_.B_neg_interleaved,
            C_power,
            M, N, K, batch_count,
            alpha, beta, stream);
    }

    /// Execute prepared GEMM via persistent direct kernel (planar FP32 output).
    cutlass::Status run_direct_gemm_prepared_persistent_fp32out(
        const __half* A_real, const __half* A_imag,
        float* C_real, float* C_imag,
        int M, int N, int K,
        int batch_count,
        float alpha, float beta,
        cudaStream_t stream = nullptr)
    {
        if (!prepared_b_.valid)
            throw std::runtime_error("prepare_b_data() not called");
        if (N != prepared_b_.N || K != prepared_b_.K || batch_count != prepared_b_.batch_count)
            throw std::runtime_error("Prepared B dimensions mismatch");
        if (!prepared_b_.B_neg_interleaved)
            throw std::runtime_error("B_neg not available");

        ensure_streams();
        ensure_hw_info();

        int64_t total_A = static_cast<int64_t>(M) * K * batch_count;
        ensure_gemm_a_interleaved(total_A * 2, stream);
        cast_fp16_planar_to_fp8_interleaved(
            A_real, A_imag,
            gemm_a_interleaved_buf_,
            total_A, stream);

        return launch_gemm_direct_persistent_planar(
            gemm_a_interleaved_buf_,
            prepared_b_.B_neg_interleaved,
            C_real, C_imag,
            M, N, K, batch_count,
            alpha, beta, hw_sm_count_, stream);
    }

    /// Execute prepared GEMM via persistent direct kernel (power detection output).
    cutlass::Status run_direct_gemm_prepared_persistent_power(
        const __half* A_real, const __half* A_imag,
        float* C_power,
        int M, int N, int K,
        int batch_count,
        float alpha, float beta,
        cudaStream_t stream = nullptr)
    {
        if (!prepared_b_.valid)
            throw std::runtime_error("prepare_b_data() not called");
        if (N != prepared_b_.N || K != prepared_b_.K || batch_count != prepared_b_.batch_count)
            throw std::runtime_error("Prepared B dimensions mismatch");
        if (!prepared_b_.B_neg_interleaved)
            throw std::runtime_error("B_neg not available");

        ensure_streams();
        ensure_hw_info();

        int64_t total_A = static_cast<int64_t>(M) * K * batch_count;
        ensure_gemm_a_interleaved(total_A * 2, stream);
        cast_fp16_planar_to_fp8_interleaved(
            A_real, A_imag,
            gemm_a_interleaved_buf_,
            total_A, stream);

        return launch_gemm_direct_persistent_power(
            gemm_a_interleaved_buf_,
            prepared_b_.B_neg_interleaved,
            C_power,
            M, N, K, batch_count,
            alpha, beta, hw_sm_count_, stream);
    }

    #include "gemm_complex_fp8/herk_impl.hpp"


    // ---------------------------------------------------------------
    // Public accessors for PIMPL pre-allocation (init_herk)
    // ---------------------------------------------------------------

    /// Lazily initialize CUDA streams and events.
    void ensure_streams_public() { ensure_streams(); }

    /// Lazily query hardware info (SM count, L2 cache size).
    void ensure_hw_info_public() { ensure_hw_info(); }

    /// Pre-allocate the FP8 precast buffer for direct HERK.
    void ensure_herk_precast_public(int64_t num_fp8_elements, cudaStream_t stream) {
        ensure_herk_precast(num_fp8_elements, stream);
    }

    /// Pre-allocate the scratch buffer for HERK intermediate results.
    void ensure_herk_scratch_public(int64_t num_elements, cudaStream_t stream) {
        ensure_herk_scratch(num_elements, stream);
    }

    /// Get L2 cache size in bytes (0 if not yet queried).
    int l2_cache_bytes() const { return l2_cache_bytes_; }

    /// Get the capture stream for CUDA graph recording.
    cudaStream_t capture_stream() const { return capture_stream_; }

private:
    FP8BufferManager buffers_;
    PreparedBState prepared_b_;

    // SM count, cached at first use. Persistent kernels (Cooperative, Pingpong)
    // need this to compute their grid size (= SM count for persistent scheduling).
    int hw_sm_count_ = 0;
    int l2_cache_bytes_ = 0;
    int persisting_l2_max_ = 0;

    // HERK production mode temp buffer: holds Xi·Xr^T [N×N] in FP16
    // before anti-symmetrization. Only allocated when herk_planar() is called
    // in production mode.
    __half* herk_imag_temp_ = nullptr;
    int64_t herk_imag_temp_capacity_ = 0;

    void ensure_herk_temp(int64_t num_elements, cudaStream_t stream = nullptr) {
        if (num_elements > herk_imag_temp_capacity_) {
            if (herk_imag_temp_) {
                stream ? cudaFreeAsync(herk_imag_temp_, stream) : cudaFree(herk_imag_temp_);
            }
            herk_imag_temp_capacity_ = num_elements;
            CUDA_CHECK(cudaMallocAsync(&herk_imag_temp_, num_elements * sizeof(__half), stream));
        }
    }

    // HERK packed mode: scratch for Re(C) sub-GEMM output (full N×N)
    // before packing to user's packed output buffer. Separate from
    // herk_imag_temp_ so Re (stream_a) and Im (stream_b) can run concurrently.
    __half* herk_real_temp_ = nullptr;
    int64_t herk_real_temp_capacity_ = 0;

    void ensure_herk_real_temp(int64_t num_elements, cudaStream_t stream = nullptr) {
        if (num_elements > herk_real_temp_capacity_) {
            if (herk_real_temp_) {
                stream ? cudaFreeAsync(herk_real_temp_, stream) : cudaFree(herk_real_temp_);
            }
            herk_real_temp_capacity_ = num_elements;
            CUDA_CHECK(cudaMallocAsync(&herk_real_temp_, num_elements * sizeof(__half), stream));
        }
    }

    void ensure_hw_info() {
        if (hw_sm_count_ > 0) return;
        hw_sm_count_ = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(0);
        cudaDeviceGetAttribute(&l2_cache_bytes_, cudaDevAttrL2CacheSize, 0);
        cudaDeviceGetAttribute(&persisting_l2_max_, cudaDevAttrMaxPersistingL2CacheSize, 0);
    }

    // ---------------------------------------------------------------
    // HERK CUDA graph cache (baseline path)
    // ---------------------------------------------------------------
    struct HerkGraphEntry {
        cudaGraphExec_t exec = nullptr;
        int cached_N = 0, cached_K = 0, cached_batch = 0;
        float cached_alpha = 0, cached_beta = 0;
        const __half* cached_A_real = nullptr;
        const __half* cached_A_imag = nullptr;
        __half* cached_C_real = nullptr;
        __half* cached_C_imag = nullptr;

        bool matches(int N, int K_val, int batch, float a, float b,
                     const __half* Ar, const __half* Ai,
                     __half* Cr, __half* Ci) const {
            return exec && N == cached_N && K_val == cached_K &&
                   batch == cached_batch &&
                   a == cached_alpha && b == cached_beta &&
                   Ar == cached_A_real && Ai == cached_A_imag &&
                   Cr == cached_C_real && Ci == cached_C_imag;
        }

        void store(int N, int K_val, int batch, float a, float b,
                   const __half* Ar, const __half* Ai,
                   __half* Cr, __half* Ci, cudaGraphExec_t e) {
            clear();
            exec = e;
            cached_N = N; cached_K = K_val; cached_batch = batch;
            cached_alpha = a; cached_beta = b;
            cached_A_real = Ar; cached_A_imag = Ai;
            cached_C_real = Cr; cached_C_imag = Ci;
        }

        void clear() {
            if (exec) { cudaGraphExecDestroy(exec); exec = nullptr; }
        }
    };

    static constexpr int kHerkGraphCacheSize = 4;
    HerkGraphEntry herk_graph_cache_[kHerkGraphCacheSize];
    bool use_herk_graph_ = false;
    bool use_batch_tiling_ = true;
    HerkMode herk_mode_ = HerkMode::Auto;
    PersistentMode persistent_mode_ = PersistentMode::Auto;
    GemmConfig gemm_config_ = GemmConfig::Default;
    DirectHerkConfig direct_herk_config_ = DirectHerkConfig::Default;
    HerkTileSize herk_tile_size_ = HerkTileSize::N32;
    HerkPipelineMode herk_pipeline_mode_ = HerkPipelineMode::Sync;

    // A interleaved FP8 buffer for direct GEMM kernel
    __nv_fp8_e4m3* gemm_a_interleaved_buf_ = nullptr;
    int64_t gemm_a_interleaved_capacity_ = 0;

    void ensure_gemm_a_interleaved(int64_t num_fp8_elements, cudaStream_t stream = nullptr) {
        if (num_fp8_elements > gemm_a_interleaved_capacity_) {
            if (gemm_a_interleaved_buf_) cudaFree(gemm_a_interleaved_buf_);
            CUDA_CHECK(cudaMalloc(&gemm_a_interleaved_buf_,
                                   num_fp8_elements * sizeof(__nv_fp8_e4m3)));
            gemm_a_interleaved_capacity_ = num_fp8_elements;
        }
    }

    // Pre-cast buffer for direct HERK (FP8 interleaved data)
    __nv_fp8_e4m3* herk_precast_buf_ = nullptr;
    int64_t herk_precast_capacity_ = 0;

    void ensure_herk_precast(int64_t num_fp8_elements, cudaStream_t stream = nullptr) {
        if (num_fp8_elements > herk_precast_capacity_) {
            if (herk_precast_buf_) cudaFree(herk_precast_buf_);
            CUDA_CHECK(cudaMalloc(&herk_precast_buf_, num_fp8_elements));
            herk_precast_capacity_ = num_fp8_elements;
        }
    }

    // N×N scratch buffer for direct HERK coalesced output (FP16 only)
    __half* herk_scratch_buf_ = nullptr;
    int64_t herk_scratch_capacity_ = 0;

    void ensure_herk_scratch(int64_t num_elements, cudaStream_t stream = nullptr) {
        if (num_elements > herk_scratch_capacity_) {
            if (herk_scratch_buf_) cudaFree(herk_scratch_buf_);
            CUDA_CHECK(cudaMalloc(&herk_scratch_buf_, num_elements * sizeof(__half)));
            herk_scratch_capacity_ = num_elements;
        }
    }

    // N×N scratch buffer for direct HERK coalesced output (FP32)
    float* herk_scratch_fp32_buf_ = nullptr;
    int64_t herk_scratch_fp32_capacity_ = 0;

    void ensure_herk_scratch_fp32(int64_t num_elements, cudaStream_t stream = nullptr) {
        if (num_elements > herk_scratch_fp32_capacity_) {
            if (herk_scratch_fp32_buf_) cudaFree(herk_scratch_fp32_buf_);
            CUDA_CHECK(cudaMalloc(&herk_scratch_fp32_buf_, num_elements * sizeof(float)));
            herk_scratch_fp32_capacity_ = num_elements;
        }
    }

    HerkGraphEntry* find_herk_graph(int N, int K_val, int batch, float a, float b,
                                     const __half* Ar, const __half* Ai,
                                     __half* Cr, __half* Ci) {
        for (int i = 0; i < kHerkGraphCacheSize; ++i) {
            if (herk_graph_cache_[i].matches(N, K_val, batch, a, b, Ar, Ai, Cr, Ci))
                return &herk_graph_cache_[i];
        }
        return nullptr;
    }

    HerkGraphEntry* alloc_herk_graph_slot() {
        for (int i = 0; i < kHerkGraphCacheSize; ++i) {
            if (!herk_graph_cache_[i].exec) return &herk_graph_cache_[i];
        }
        herk_graph_cache_[0].clear();
        for (int i = 0; i < kHerkGraphCacheSize - 1; ++i) {
            herk_graph_cache_[i] = herk_graph_cache_[i + 1];
        }
        herk_graph_cache_[kHerkGraphCacheSize - 1] = HerkGraphEntry{};
        return &herk_graph_cache_[kHerkGraphCacheSize - 1];
    }

    void clear_herk_graph_cache() {
        for (int i = 0; i < kHerkGraphCacheSize; ++i)
            herk_graph_cache_[i].clear();
    }

    // Pre-allocated GEMM workspaces for HERK graph capture (one per stream)
    void* herk_gemm_workspace_a_ = nullptr;
    void* herk_gemm_workspace_b_ = nullptr;
    size_t herk_gemm_workspace_size_ = 0;

    void ensure_herk_gemm_workspace(int N, int K, int batch_count) {
        size_t needed_a = get_fp8_workspace_size(N, N, 2 * K, batch_count);  // stacked Re GEMM
        size_t needed_b = get_fp8_workspace_size(N, N, K, batch_count);      // Im GEMM
        size_t needed = std::max(needed_a, needed_b);
        if (needed > herk_gemm_workspace_size_) {
            if (herk_gemm_workspace_a_) cudaFree(herk_gemm_workspace_a_);
            if (herk_gemm_workspace_b_) cudaFree(herk_gemm_workspace_b_);
            herk_gemm_workspace_a_ = nullptr;
            herk_gemm_workspace_b_ = nullptr;
            if (needed > 0) {
                CUDA_CHECK(cudaMalloc(&herk_gemm_workspace_a_, needed));
                CUDA_CHECK(cudaMalloc(&herk_gemm_workspace_b_, needed));
            }
            herk_gemm_workspace_size_ = needed;
        }
    }

    // ---------------------------------------------------------------
    // Stream parallelism infrastructure
    // ---------------------------------------------------------------
    //
    // Two internal streams allow the real-part and imag-part sub-GEMM pairs
    // to execute concurrently. Events synchronize preprocessing → GEMMs → caller.
    //
    // Timeline:
    //   caller_stream: [cast kernels] → record(preprocess_done)
    //   stream_a:      wait(preprocess_done) → GEMM1 → GEMM2 → record(stream_a_done)
    //   stream_b:      wait(preprocess_done) → GEMM3 → GEMM4 → record(stream_b_done)
    //   caller_stream: wait(stream_a_done) + wait(stream_b_done) → return

    cudaStream_t stream_a_ = nullptr;
    cudaStream_t stream_b_ = nullptr;
    cudaStream_t capture_stream_ = nullptr;
    cudaEvent_t preprocess_done_ = nullptr;
    cudaEvent_t stream_a_done_ = nullptr;
    cudaEvent_t stream_b_done_ = nullptr;
    bool streams_initialized_ = false;

    void ensure_streams() {
        if (streams_initialized_) return;
        ensure_hw_info();
        CUDA_CHECK(cudaStreamCreate(&stream_a_));
        CUDA_CHECK(cudaStreamCreate(&stream_b_));
        CUDA_CHECK(cudaStreamCreate(&capture_stream_));
        // Disable timing on events for lower overhead
        CUDA_CHECK(cudaEventCreateWithFlags(&preprocess_done_, cudaEventDisableTiming));
        CUDA_CHECK(cudaEventCreateWithFlags(&stream_a_done_, cudaEventDisableTiming));
        CUDA_CHECK(cudaEventCreateWithFlags(&stream_b_done_, cudaEventDisableTiming));
        streams_initialized_ = true;
    }

    void destroy_streams() {
        clear_herk_graph_cache();
        if (herk_gemm_workspace_a_) { cudaFree(herk_gemm_workspace_a_); herk_gemm_workspace_a_ = nullptr; }
        if (herk_gemm_workspace_b_) { cudaFree(herk_gemm_workspace_b_); herk_gemm_workspace_b_ = nullptr; }
        herk_gemm_workspace_size_ = 0;
        if (!streams_initialized_) return;
        cudaStreamDestroy(stream_a_);
        cudaStreamDestroy(stream_b_);
        cudaStreamDestroy(capture_stream_);
        cudaEventDestroy(preprocess_done_);
        cudaEventDestroy(stream_a_done_);
        cudaEventDestroy(stream_b_done_);
        streams_initialized_ = false;
        // Free HERK precast buffer
        if (herk_precast_buf_) {
            cudaFree(herk_precast_buf_);
            herk_precast_buf_ = nullptr;
            herk_precast_capacity_ = 0;
        }
        if (herk_scratch_buf_) {
            cudaFree(herk_scratch_buf_);
            herk_scratch_buf_ = nullptr;
            herk_scratch_capacity_ = 0;
        }
        if (herk_scratch_fp32_buf_) {
            cudaFree(herk_scratch_fp32_buf_);
            herk_scratch_fp32_buf_ = nullptr;
            herk_scratch_fp32_capacity_ = 0;
        }
        // Also free HERK temp buffers
        if (herk_imag_temp_) {
            cudaFree(herk_imag_temp_);
            herk_imag_temp_ = nullptr;
            herk_imag_temp_capacity_ = 0;
        }
        if (herk_real_temp_) {
            cudaFree(herk_real_temp_);
            herk_real_temp_ = nullptr;
            herk_real_temp_capacity_ = 0;
        }
        if (gemm_a_interleaved_buf_) {
            cudaFree(gemm_a_interleaved_buf_);
            gemm_a_interleaved_buf_ = nullptr;
            gemm_a_interleaved_capacity_ = 0;
        }
    }

    #include "gemm_complex_fp8/dispatch_impl.hpp"

    #include "gemm_complex_fp8/triangle_impl.hpp"


public:
    static void deinterleave_complex(
        const __half* interleaved, __half* real_out, __half* imag_out,
        int64_t num_complex_elements, cudaStream_t stream);

    static void interleave_complex(
        const __half* real_in, const __half* imag_in, __half* interleaved_out,
        int64_t num_complex_elements, cudaStream_t stream);
};

// ========================================================================================
// Free-standing __global__ kernels (cannot be class members in CUDA)
// ========================================================================================

__global__ void deinterleave_kernel(
    const __half* __restrict__ interleaved,
    __half* __restrict__ real_part,
    __half* __restrict__ imag_part,
    int64_t num_complex_elements)
{
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < num_complex_elements) {
        real_part[idx] = interleaved[2 * idx];
        imag_part[idx] = interleaved[2 * idx + 1];
    }
}

__global__ void interleave_kernel(
    const __half* __restrict__ real_part,
    const __half* __restrict__ imag_part,
    __half* __restrict__ interleaved,
    int64_t num_complex_elements)
{
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < num_complex_elements) {
        interleaved[2 * idx]     = real_part[idx];
        interleaved[2 * idx + 1] = imag_part[idx];
    }
}

// ---- Inline definitions of the static launcher methods ----

inline void GemmComplexFP8::deinterleave_complex(
    const __half* interleaved, __half* real_out, __half* imag_out,
    int64_t num_complex_elements, cudaStream_t stream)
{
    constexpr int kBlock = 256;
    int grid = static_cast<int>((num_complex_elements + kBlock - 1) / kBlock);
    deinterleave_kernel<<<grid, kBlock, 0, stream>>>(
        interleaved, real_out, imag_out, num_complex_elements);
    CUDA_CHECK(cudaGetLastError());
}

inline void GemmComplexFP8::interleave_complex(
    const __half* real_in, const __half* imag_in, __half* interleaved_out,
    int64_t num_complex_elements, cudaStream_t stream)
{
    constexpr int kBlock = 256;
    int grid = static_cast<int>((num_complex_elements + kBlock - 1) / kBlock);
    interleave_kernel<<<grid, kBlock, 0, stream>>>(
        real_in, imag_in, interleaved_out, num_complex_elements);
    CUDA_CHECK(cudaGetLastError());
}

}  // namespace gemm_complex_fp8
