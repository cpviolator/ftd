/*
 * gemm_complex_sm100.hpp
 *
 * Complex FP8 GEMM for Blackwell SM100 (GB200) & SM120 (GB10/RTX 50-series)
 * Requires: CUTLASS 4.0+, CUDA 13.x
 *
 * ========================================================================================
 * BLACKWELL vs HOPPER — ARCHITECTURAL DIFFERENCES
 * ========================================================================================
 *
 * This header is the Blackwell counterpart to gemm_complex_fp8.hpp (Hopper SM90).
 * The complex decomposition algorithm (4M), planar layout, and public API are IDENTICAL.
 * What changes is the underlying CUTLASS kernel type chain:
 *
 *   Hopper SM90                          Blackwell SM100
 *   ──────────────────────────────────────────────────────────────────────
 *   wgmma.mma_async (64×N×32)            tcgen05.mma (128×N×K, 1SM or 2SM)
 *   TileShape = CTA output tile          MmaTileShape = per-MMA-instruction tile
 *   Cooperative/Pingpong schedule         1Sm/2Sm warp-specialized schedule
 *   wgmma pipeline (C7510 issues)         TMEM double-buffered accumulators
 *   EpilogueScheduleAuto                  TmaWarpSpecialized1Sm / 2Sm
 *   Static cluster shapes                 Preferred + fallback dynamic clusters
 *   SMEM accumulator staging              TMEM (Tensor Memory) — dedicated HW buffer
 *
 * Key Blackwell advantages:
 *   1. tcgen05.mma is 2-4× faster than wgmma per instruction (wider + deeper pipeline)
 *   2. TMEM eliminates accumulator spill/fill overhead that limited Hopper occupancy
 *   3. 2SM MMA: two SMs cooperate on a single tile — doubles compute per cluster unit
 *   4. CLC (Cluster Launch Control) tile scheduler: hardware-assisted persistence
 *   5. No C7510 wgmma serialization — tcgen05 pipeline is compiler-managed differently
 *   6. Native MXFP8 block-scaled support (shared exponent per 32 elements)
 *
 * GB200 (SM100) vs GB10/RTX 50 (SM120):
 *   - SM100 (datacenter): TMA multicast, 2SM MMA, large clusters, full tcgen05
 *   - SM120 (consumer):   No multicast, cluster 1×1×1 only, TN layout only
 *   - Code compiled for sm100a will NOT run on SM120 devices
 *   - Use COMPLEX_FP8_SM100_TARGET_SM120 to build for GB10 development
 *
 * MXFP8 (Microscaling FP8):
 *   Blackwell introduces hardware block-scaled FP8. Instead of per-tensor scaling
 *   (what Hopper does), MXFP8 uses a shared 8-bit exponent per block of 32 elements.
 *   This dramatically improves numerical fidelity while maintaining FP8 throughput.
 *   MXFP8 is available as an optional code path via COMPLEX_FP8_SM100_USE_MXFP8.
 *
 * ========================================================================================
 * CUTLASS 4.x SM100 TEMPLATE PARAMETER GUIDE
 * ========================================================================================
 *
 * The builder API is structurally the same as SM90, but with key differences:
 *
 *   Architecture tag:   cutlass::arch::Sm100  (not Sm90)
 *
 *   MmaTileShape:  NOT the CTA output tile — it is the tile consumed by 1 or 2 SM
 *                  tcgen05.mma instructions. The CTA tile is derived by the builder.
 *                  Valid 1SM shapes for vanilla FP8 (TN layout):
 *                    128×64×128, 128×128×128, 128×256×128
 *                  Valid 2SM shapes:
 *                    256×64×128, 256×128×128, 256×256×128
 *                  For 2SM, ClusterShape M dimension must be a multiple of 2.
 *
 *   KernelSchedule:
 *     KernelTmaWarpSpecialized1SmSm100  — single SM per MMA instruction
 *     KernelTmaWarpSpecialized2SmSm100  — two SMs cooperate on one MMA
 *     KernelScheduleAuto                — builder auto-selects based on cluster shape
 *
 *   EpilogueSchedule:
 *     cutlass::epilogue::TmaWarpSpecialized1Sm   (pairs with 1SM mainloop)
 *     cutlass::epilogue::TmaWarpSpecialized2Sm   (pairs with 2SM mainloop)
 *
 *   BUILD ORDER: Epilogue MUST be built BEFORE mainloop. The mainloop builder takes
 *   sizeof(CollectiveEpilogue::SharedStorage) as a template parameter to compute
 *   the SMEM budget for pipeline stages.
 *
 *   Alignment: 16 bytes (= 16 / sizeof(element) in elements).
 *     FP8:  16 / 1 = 16 elements
 *     FP16: 16 / 2 = 8 elements
 *
 * ========================================================================================
 */

#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <cassert>
#include <algorithm>

// ---- Kernel Type Chains (shared with gemm_blockscaled_dispatch.cu) ----
// Includes CUTLASS framework headers, shared type config, and chain instantiations.
// Separated into its own header to allow multiple .cu TUs without ODR violations.
#include "gemm_sm100_type_chains.hpp"

// ---- Block-Scaled GEMM Dispatch ----
// Concrete (non-template) functions in a .cu TU for kernel registration fix.
// See CUTLASS issue #2478 — device_kernel<GemmKernel> fails to register when
// instantiated through deep template indirection in header-only code.
#include "gemm_blockscaled_dispatch.h"

// ---- CUDA Runtime ----
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>

// ---- Kernel Autotuning Cache ----
#include "tune_cache.hpp"

// ---- Compile-time CUDA version check ----
// CUDA 13.x required for SM100a/SM120a real-architecture codegen
#if defined(__CUDACC_VER_MAJOR__)
#if __CUDACC_VER_MAJOR__ < 13
#warning "gemm_complex_sm100.hpp: CUDA 13.x required for Blackwell SM100/SM120 codegen."
#endif
#endif


namespace gemm_complex_sm100 {


// Free-standing kernels and utilities (textual includes — no #pragma once)
#include "gemm_complex_sm100/config.hpp"
#include "gemm_complex_sm100/cast_kernels.hpp"
#include "gemm_complex_sm100/pack_kernels.hpp"
#include "gemm_complex_sm100/int4_kernels.hpp"
#include "gemm_complex_sm100/fp6_kernels.hpp"
#include "gemm_complex_sm100/fp4_kernels.hpp"
#include "gemm_complex_sm100/mxfp_kernels.hpp"
#include "gemm_complex_sm100/buffers.hpp"
#include "gemm_complex_sm100/herk_kernel.hpp"
#include "shared/gemm_kernel_common.hpp"



// ========================================================================================
// GemmComplexSm100 — The Public API
// ========================================================================================
//
// API-compatible with SM90 GemmComplexFP8. The complex decomposition algorithm is
// identical — only the underlying real FP8 GEMM kernel changes to use SM100 tcgen05.
//

class GemmComplexSm100 {
public:
    GemmComplexSm100() = default;
    ~GemmComplexSm100() { destroy_streams(); }

    GemmComplexSm100(const GemmComplexSm100&) = delete;
    GemmComplexSm100& operator=(const GemmComplexSm100&) = delete;
    GemmComplexSm100(GemmComplexSm100&&) = delete;
    GemmComplexSm100& operator=(GemmComplexSm100&&) = delete;

    /// Enable CUDA graph capture/replay for baseline HERK (non-triangle path).
    /// When enabled, herk_planar_packed() captures the full HERK body (cast + GEMMs + pack)
    /// as a CUDA graph on first call and replays it on subsequent calls with matching parameters.
    void set_herk_graph(bool enable) { use_herk_graph_ = enable; }
    bool herk_graph_enabled() const { return use_herk_graph_; }

    /// Enable/disable batch tiling in HERK_batched (L2 scratch optimization).
    /// When enabled (default), batch dimension is tiled to keep scratch in L2.
    void set_batch_tiling(bool enable) { use_batch_tiling_ = enable; }
    bool batch_tiling_enabled() const { return use_batch_tiling_; }

    /// Set HERK dispatch mode for FP8 NoTrans:
    ///   Auto          — direct for small K (K <= N/4), baseline otherwise
    ///   ForceDirect   — always use direct single-launch kernel
    ///   ForceBaseline — always use multi-launch CUTLASS path
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

    /// Set CUTLASS GEMM tile/cluster configuration for FP8 dispatch.
    /// Default uses the compile-time configured tile. Autotuner overrides this.
    /// Configs invalid for the current arch return kErrorNotSupported at dispatch.
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
    /// Returns false on builds where kernel SharedStorage exceeds device limits.
    /// SM120 uses 128×64 FP8 tile — 3-stage SharedStorage ~82 KB, fits in 99 KB.
    /// SM100 uses 128×128 FP8 tile — 3-stage SharedStorage ~110 KB, fits in 228 KB.
    static bool fp8_baseline_available() {
        int max_smem = 0;
        cudaDeviceGetAttribute(&max_smem, cudaDevAttrMaxSharedMemoryPerBlockOptin, 0);
        return sizeof(typename GemmKernel::SharedStorage) <= static_cast<size_t>(max_smem);
    }

    // ---------------------------------------------------------------
    // Primary Interface: Planar Complex
    // ---------------------------------------------------------------
#include "gemm_complex_sm100/api_impl.hpp"

#include "gemm_complex_sm100/herk_impl.hpp"



public:
    // ---------------------------------------------------------------
    // Prepared B state: pre-convert B matrix for reuse across GEMM calls
    // ---------------------------------------------------------------
    struct PreparedBState {
        ComputePrecision precision = ComputePrecision::FP8_E4M3;
        int N = 0, K = 0, batch_count = 0;

        // FP8: converted B data (owned)
        cutlass::float_e4m3_t* B_real_fp8 = nullptr;
        cutlass::float_e4m3_t* B_imag_fp8 = nullptr;

        // Direct GEMM: B_neg interleaved FP8 with negated Im (owned)
        // [batch, N, 2*K] interleaved FP8: [re, -im, re, -im, ...]
        __nv_fp8_e4m3* B_neg_interleaved = nullptr;

        // MXFP: packed data + scale factors (owned)
        void* B_real_packed = nullptr;
        void* B_imag_packed = nullptr;
        void* sf_B_real = nullptr;
        void* sf_B_imag = nullptr;

        // Pre-allocated workspaces (2: one per stream for concurrent sub-GEMMs)
        void* workspace_a = nullptr;
        void* workspace_b = nullptr;
        size_t workspace_size = 0;

        bool valid = false;

        void free_all() {
            if (B_real_fp8)        { cudaFree(B_real_fp8);        B_real_fp8 = nullptr; }
            if (B_imag_fp8)        { cudaFree(B_imag_fp8);        B_imag_fp8 = nullptr; }
            if (B_neg_interleaved) { cudaFree(B_neg_interleaved); B_neg_interleaved = nullptr; }
            if (B_real_packed)     { cudaFree(B_real_packed);     B_real_packed = nullptr; }
            if (B_imag_packed)     { cudaFree(B_imag_packed);     B_imag_packed = nullptr; }
            if (sf_B_real)         { cudaFree(sf_B_real);         sf_B_real = nullptr; }
            if (sf_B_imag)         { cudaFree(sf_B_imag);         sf_B_imag = nullptr; }
            if (workspace_a)       { cudaFree(workspace_a);       workspace_a = nullptr; }
            if (workspace_b)       { cudaFree(workspace_b);       workspace_b = nullptr; }
            workspace_size = 0;
            valid = false;
        }
    };

    /// Pre-convert B matrix to internal format (FP8 or MXFP) for reuse.
    /// Call once at precompute time. Subsequent GEMM calls via
    /// run_planar_batched_prepared_fp32out() skip B conversion entirely.
    void prepare_b_data(
        const __half* B_real, const __half* B_imag,
        int N, int K, int batch_count,
        ComputePrecision precision,
        cudaStream_t stream = nullptr)
    {
        prepared_b_.free_all();
        prepared_b_.precision = precision;
        prepared_b_.N = N;
        prepared_b_.K = K;
        prepared_b_.batch_count = batch_count;

        ensure_streams();
        ensure_hw_info();

        int64_t size_B = (int64_t)N * K;
        int64_t total_B = size_B * batch_count;

        if (precision == ComputePrecision::FP8_E4M3) {
            // Allocate persistent FP8 B buffers
            CUDA_CHECK(cudaMalloc(&prepared_b_.B_real_fp8,
                                   total_B * sizeof(cutlass::float_e4m3_t)));
            CUDA_CHECK(cudaMalloc(&prepared_b_.B_imag_fp8,
                                   total_B * sizeof(cutlass::float_e4m3_t)));

            // Convert FP16 -> FP8 (planar, for 4M sub-GEMM path)
            cast_fp16_to_fp8_e4m3_paired_sm100(
                B_real, B_imag,
                reinterpret_cast<__nv_fp8_e4m3*>(prepared_b_.B_real_fp8),
                reinterpret_cast<__nv_fp8_e4m3*>(prepared_b_.B_imag_fp8),
                total_B, stream);

            // Also prepare B_neg interleaved for direct GEMM kernel:
            // [re, -im, re, -im, ...] interleaved FP8 with negated imaginary
            int64_t interleaved_size = total_B * 2;  // 2 FP8 per complex element
            CUDA_CHECK(cudaMalloc(&prepared_b_.B_neg_interleaved,
                                   interleaved_size * sizeof(__nv_fp8_e4m3)));
            cast_fp16_planar_to_fp8_interleaved_negate_im_sm100(
                B_real, B_imag,
                prepared_b_.B_neg_interleaved,
                total_B, stream);
        } else {
            // MXFP path: convert + compute scale factors
            int64_t bytes_B = bytes_for_elements(total_B, precision);
            CUDA_CHECK(cudaMalloc(&prepared_b_.B_real_packed, bytes_B));
            CUDA_CHECK(cudaMalloc(&prepared_b_.B_imag_packed, bytes_B));

            // SF buffers
            bool per_batch_sf_B = (N % 128 != 0);
            int64_t sf_per_B = sf_buffer_bytes(N, K);
            int64_t sf_B_total = per_batch_sf_B
                ? (int64_t)batch_count * sf_per_B
                : sf_buffer_bytes(N * batch_count, K);
            CUDA_CHECK(cudaMalloc(&prepared_b_.sf_B_real, sf_B_total));
            CUDA_CHECK(cudaMalloc(&prepared_b_.sf_B_imag, sf_B_total));

            // Preprocess B
            if (per_batch_sf_B) {
                int64_t data_bytes_per = bytes_for_elements(size_B, precision);
                preprocess_mxfp_paired_batched_sm100(
                    B_real, B_imag,
                    prepared_b_.B_real_packed, prepared_b_.B_imag_packed,
                    prepared_b_.sf_B_real, prepared_b_.sf_B_imag,
                    N, K, batch_count,
                    data_bytes_per, sf_per_B,
                    precision, stream);
            } else {
                // Flat: N is a multiple of 128
                preprocess_mxfp_paired_sm100(
                    B_real, B_imag,
                    prepared_b_.B_real_packed, prepared_b_.B_imag_packed,
                    prepared_b_.sf_B_real, prepared_b_.sf_B_imag,
                    N * batch_count, K, precision, stream);
            }
        }

        // Workspace pre-allocation deferred to first GEMM call
        // (need M to compute workspace size)
        prepared_b_.valid = true;
    }

    /// Execute GEMM with pre-prepared B data. Only converts A per-call.
    /// B must have been prepared via prepare_b_data() with matching N, K, batch_count.
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
        if (N != prepared_b_.N || K != prepared_b_.K || batch_count != prepared_b_.batch_count)
            throw std::runtime_error("Prepared B dimensions mismatch");

        ensure_streams();
        ComputePrecision precision = prepared_b_.precision;

        // Auto-select optimal tile config for GEMM (not HERK) based on dimensions
        GemmConfig config = select_gemm_config_for_gemm(M, N, K);

        // Lazy workspace allocation (once, on first call with known M)
        if (prepared_b_.workspace_size == 0) {
            size_t ws = compute_workspace_size(M, N, K, batch_count, precision);
            if (ws > 0) {
                CUDA_CHECK(cudaMalloc(&prepared_b_.workspace_a, ws));
                CUDA_CHECK(cudaMalloc(&prepared_b_.workspace_b, ws));
                prepared_b_.workspace_size = ws;
            }
        }

        if (precision == ComputePrecision::FP8_E4M3) {
            // Only convert A (FP16 -> FP8)
            int64_t size_A = (int64_t)M * K;
            int64_t total_A = size_A * batch_count;
            buffers_.ensure_capacity(total_A, 0, 0, stream);

            cast_fp16_to_fp8_e4m3_paired_sm100(
                A_real, A_imag,
                reinterpret_cast<__nv_fp8_e4m3*>(buffers_.A_real()),
                reinterpret_cast<__nv_fp8_e4m3*>(buffers_.A_imag()),
                total_A, stream);

            // Dispatch sub-GEMMs with pre-converted B and pre-allocated workspace
            return run_subgemms_fp8_fp32out(
                buffers_.A_real(), buffers_.A_imag(),
                prepared_b_.B_real_fp8, prepared_b_.B_imag_fp8,
                C_real, C_imag, M, N, K,
                alpha, beta, mode, stream, batch_count,
                prepared_b_.workspace_a, prepared_b_.workspace_b,
                prepared_b_.workspace_size);

        } else {
            // MXFP: only convert A
            int64_t size_A = (int64_t)M * K;
            int64_t total_A = size_A * batch_count;
            int64_t bytes_A = bytes_for_elements(total_A, precision);
            lp_buffers_.ensure_capacity(bytes_A, 0, 0, stream);

            // Per-batch A SF allocation
            bool per_batch_sf_A = (M % 128 != 0);
            int64_t sf_per_A = sf_buffer_bytes(M, K);
            int64_t sf_A_total = per_batch_sf_A
                ? (int64_t)batch_count * sf_per_A
                : sf_buffer_bytes(M * batch_count, K);
            lp_buffers_.ensure_sf_capacity(sf_A_total, 0, 0, stream);

            // Preprocess A (batched or flat)
            if (per_batch_sf_A) {
                int64_t data_bytes_per = bytes_for_elements(size_A, precision);
                preprocess_mxfp_paired_batched_sm100(
                    A_real, A_imag,
                    lp_buffers_.A_real(), lp_buffers_.A_imag(),
                    lp_buffers_.sf_A_real(), lp_buffers_.sf_A_imag(),
                    M, K, batch_count,
                    data_bytes_per, sf_per_A,
                    precision, stream);
            } else {
                preprocess_mxfp_paired_sm100(
                    A_real, A_imag,
                    lp_buffers_.A_real(), lp_buffers_.A_imag(),
                    lp_buffers_.sf_A_real(), lp_buffers_.sf_A_imag(),
                    M * batch_count, K, precision, stream);
            }

            // Dispatch 4 sub-GEMMs using prepared B + pre-allocated workspace
            CUDA_CHECK(cudaEventRecord(preprocess_done_, stream));
            CUDA_CHECK(cudaStreamWaitEvent(stream_a_, preprocess_done_));
            CUDA_CHECK(cudaStreamWaitEvent(stream_b_, preprocess_done_));

            cutlass::Status status;

            // Stream A: Re(C) = alpha*Ar*Br - alpha*Ai*Bi + beta*Cr
            status = run_real_gemm_dispatch_fp32out(
                lp_buffers_.A_real(), prepared_b_.B_real_packed,
                lp_buffers_.sf_A_real(), prepared_b_.sf_B_real,
                C_real, M, N, K, alpha, beta, stream_a_, precision,
                config, batch_count, 0,
                prepared_b_.workspace_a, prepared_b_.workspace_size);
            if (status != cutlass::Status::kSuccess) return status;

            status = run_real_gemm_dispatch_fp32out(
                lp_buffers_.A_imag(), prepared_b_.B_imag_packed,
                lp_buffers_.sf_A_imag(), prepared_b_.sf_B_imag,
                C_real, M, N, K, -alpha, 1.0f, stream_a_, precision,
                config, batch_count, 0,
                prepared_b_.workspace_a, prepared_b_.workspace_size);
            if (status != cutlass::Status::kSuccess) return status;

            // Stream B: Im(C) = alpha*Ar*Bi + alpha*Ai*Br + beta*Ci
            status = run_real_gemm_dispatch_fp32out(
                lp_buffers_.A_real(), prepared_b_.B_imag_packed,
                lp_buffers_.sf_A_real(), prepared_b_.sf_B_imag,
                C_imag, M, N, K, alpha, beta, stream_b_, precision,
                config, batch_count, 0,
                prepared_b_.workspace_b, prepared_b_.workspace_size);
            if (status != cutlass::Status::kSuccess) return status;

            status = run_real_gemm_dispatch_fp32out(
                lp_buffers_.A_imag(), prepared_b_.B_real_packed,
                lp_buffers_.sf_A_imag(), prepared_b_.sf_B_real,
                C_imag, M, N, K, alpha, 1.0f, stream_b_, precision,
                config, batch_count, 0,
                prepared_b_.workspace_b, prepared_b_.workspace_size);
            if (status != cutlass::Status::kSuccess) return status;

            CUDA_CHECK(cudaEventRecord(stream_a_done_, stream_a_));
            CUDA_CHECK(cudaEventRecord(stream_b_done_, stream_b_));
            CUDA_CHECK(cudaStreamWaitEvent(stream, stream_a_done_));
            CUDA_CHECK(cudaStreamWaitEvent(stream, stream_b_done_));

            return cutlass::Status::kSuccess;
        }
    }

    // ---------------------------------------------------------------
    // Batch-Fused M GEMM — Fuse batch elements along M for better TC util
    // ---------------------------------------------------------------
    //
    // For small-M problems (M=32-256), wave quantization wastes tiles.
    // Fusing fuse_factor consecutive batch elements yields M_fused = M * fuse_factor
    // which fills more tile rows per wave. Requires B uniform across batches
    // (only one copy of B is used).
    //
    // A layout: [batch, M, K] contiguous -> reinterpreted as [effective_batch, M_fused, K]
    // C layout: [batch, M, N] contiguous -> reinterpreted as [effective_batch, M_fused, N]
    // B is the same for all batch elements (prepared with batch_count >= 1).
    //

    /// Execute prepared GEMM with batch-fused M dimension.
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
        ComputePrecision precision = prepared_b_.precision;

        // Select tile config: use GEMM-aware selection for the fused M
        GemmConfig config = select_gemm_config_for_gemm(M_fused, N, K);

        // Lazy workspace allocation (once, on first call with known M_fused)
        if (prepared_b_.workspace_size == 0) {
            size_t ws = compute_workspace_size(M_fused, N, K, effective_batch, precision);
            if (ws > 0) {
                CUDA_CHECK(cudaMalloc(&prepared_b_.workspace_a, ws));
                CUDA_CHECK(cudaMalloc(&prepared_b_.workspace_b, ws));
                prepared_b_.workspace_size = ws;
            }
        }

        if (precision == ComputePrecision::FP8_E4M3) {
            // Convert ALL A elements (full batch, contiguous in memory)
            // A layout: [batch_count, M, K] -> reinterpreted as [effective_batch, M_fused, K]
            int64_t total_A = (int64_t)M * K * batch_count;
            buffers_.ensure_capacity(total_A, 0, 0, stream);

            cast_fp16_to_fp8_e4m3_paired_sm100(
                A_real, A_imag,
                reinterpret_cast<__nv_fp8_e4m3*>(buffers_.A_real()),
                reinterpret_cast<__nv_fp8_e4m3*>(buffers_.A_imag()),
                total_A, stream);

            // Batched call with M_fused and effective_batch.
            //
            // The internal run_subgemms_fp8_fp32out computes batch strides as:
            //   A stride = M * K  -> with M=M_fused: M_fused * K (correct: A is contiguous)
            //   B stride = N * K  -> B was prepared with batch_count copies (all uniform),
            //                        and effective_batch <= prepared_b_.batch_count, so
            //                        B data at offset [eb * N * K] is valid for each eb.
            //   C stride = M * N  -> with M=M_fused: M_fused * N (correct: C is contiguous)
            //
            // This replaces the per-element loop with a single batched launch,
            // eliminating per-batch launch overhead entirely.
            return run_subgemms_fp8_fp32out(
                buffers_.A_real(), buffers_.A_imag(),
                prepared_b_.B_real_fp8, prepared_b_.B_imag_fp8,
                C_real, C_imag, M_fused, N, K,
                alpha, beta, mode, stream, effective_batch,
                prepared_b_.workspace_a, prepared_b_.workspace_b,
                prepared_b_.workspace_size);
        } else {
            // MXFP path: convert all A elements (full batch)
            int64_t total_A = (int64_t)M * K * batch_count;
            int64_t bytes_A = bytes_for_elements(total_A, precision);
            lp_buffers_.ensure_capacity(bytes_A, 0, 0, stream);

            // Per-batch A SF allocation with M_fused
            bool per_batch_sf_A = (M_fused % 128 != 0);
            int64_t sf_per_A = sf_buffer_bytes(M_fused, K);
            int64_t sf_A_total = per_batch_sf_A
                ? (int64_t)effective_batch * sf_per_A
                : sf_buffer_bytes(M_fused * effective_batch, K);
            lp_buffers_.ensure_sf_capacity(sf_A_total, 0, 0, stream);

            // Preprocess A: treat as (M_fused * effective_batch) x K
            if (per_batch_sf_A) {
                int64_t data_bytes_per = bytes_for_elements((int64_t)M_fused * K, precision);
                preprocess_mxfp_paired_batched_sm100(
                    A_real, A_imag,
                    lp_buffers_.A_real(), lp_buffers_.A_imag(),
                    lp_buffers_.sf_A_real(), lp_buffers_.sf_A_imag(),
                    M_fused, K, effective_batch,
                    data_bytes_per, sf_per_A,
                    precision, stream);
            } else {
                preprocess_mxfp_paired_sm100(
                    A_real, A_imag,
                    lp_buffers_.A_real(), lp_buffers_.A_imag(),
                    lp_buffers_.sf_A_real(), lp_buffers_.sf_A_imag(),
                    M_fused * effective_batch, K, precision, stream);
            }

            // Dispatch with per-element loop (B stride = 0)
            CUDA_CHECK(cudaEventRecord(preprocess_done_, stream));
            CUDA_CHECK(cudaStreamWaitEvent(stream_a_, preprocess_done_));
            CUDA_CHECK(cudaStreamWaitEvent(stream_b_, preprocess_done_));

            int64_t a_data_stride = bytes_for_elements((int64_t)M_fused * K, precision);
            int64_t c_stride = (int64_t)M_fused * N;
            int64_t sf_a_stride = per_batch_sf_A ? sf_per_A : sf_buffer_bytes(M_fused, K);

            for (int eb = 0; eb < effective_batch; ++eb) {
                auto* Ar = reinterpret_cast<void*>(
                    reinterpret_cast<uint8_t*>(lp_buffers_.A_real()) + eb * a_data_stride);
                auto* Ai = reinterpret_cast<void*>(
                    reinterpret_cast<uint8_t*>(lp_buffers_.A_imag()) + eb * a_data_stride);
                auto* sfAr = reinterpret_cast<void*>(
                    reinterpret_cast<uint8_t*>(lp_buffers_.sf_A_real()) + eb * sf_a_stride);
                auto* sfAi = reinterpret_cast<void*>(
                    reinterpret_cast<uint8_t*>(lp_buffers_.sf_A_imag()) + eb * sf_a_stride);
                float* Cr = C_real + eb * c_stride;
                float* Ci = C_imag + eb * c_stride;

                // Stream A: Re(C)
                auto status = run_real_gemm_dispatch_fp32out(
                    Ar, prepared_b_.B_real_packed,
                    sfAr, prepared_b_.sf_B_real,
                    Cr, M_fused, N, K, alpha, beta, stream_a_, precision,
                    config, 1, 0,
                    prepared_b_.workspace_a, prepared_b_.workspace_size);
                if (status != cutlass::Status::kSuccess) return status;

                status = run_real_gemm_dispatch_fp32out(
                    Ai, prepared_b_.B_imag_packed,
                    sfAi, prepared_b_.sf_B_imag,
                    Cr, M_fused, N, K, -alpha, 1.0f, stream_a_, precision,
                    config, 1, 0,
                    prepared_b_.workspace_a, prepared_b_.workspace_size);
                if (status != cutlass::Status::kSuccess) return status;

                // Stream B: Im(C)
                status = run_real_gemm_dispatch_fp32out(
                    Ar, prepared_b_.B_imag_packed,
                    sfAr, prepared_b_.sf_B_imag,
                    Ci, M_fused, N, K, alpha, beta, stream_b_, precision,
                    config, 1, 0,
                    prepared_b_.workspace_b, prepared_b_.workspace_size);
                if (status != cutlass::Status::kSuccess) return status;

                status = run_real_gemm_dispatch_fp32out(
                    Ai, prepared_b_.B_real_packed,
                    sfAi, prepared_b_.sf_B_real,
                    Ci, M_fused, N, K, alpha, 1.0f, stream_b_, precision,
                    config, 1, 0,
                    prepared_b_.workspace_b, prepared_b_.workspace_size);
                if (status != cutlass::Status::kSuccess) return status;
            }

            CUDA_CHECK(cudaEventRecord(stream_a_done_, stream_a_));
            CUDA_CHECK(cudaEventRecord(stream_b_done_, stream_b_));
            CUDA_CHECK(cudaStreamWaitEvent(stream, stream_a_done_));
            CUDA_CHECK(cudaStreamWaitEvent(stream, stream_b_done_));

            return cutlass::Status::kSuccess;
        }
    }

    // ---------------------------------------------------------------
    // Direct Complex GEMM — Single-launch kernel using conjugate permutation
    // ---------------------------------------------------------------
    //
    // Uses the direct GEMM kernel (gemm_kernel.hpp) with pre-cast B_neg.
    // Only converts A per-call. B_neg is prepared once in prepare_b_data().
    //

    /// Execute prepared GEMM via direct kernel (planar FP32 output).
    /// Falls back to 4M sub-GEMMs if B_neg_interleaved is not available.
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
            throw std::runtime_error("B_neg not available (only FP8 precision supports direct GEMM)");

        ensure_streams();

        // Cast A from planar FP16 to interleaved FP8
        int64_t total_A = static_cast<int64_t>(M) * K * batch_count;
        ensure_gemm_a_interleaved(total_A * 2, stream);  // 2 FP8 per complex element
        cast_fp16_planar_to_fp8_interleaved_sm100(
            A_real, A_imag,
            gemm_a_interleaved_buf_,
            total_A, stream);

        // Launch direct GEMM kernel
        return launch_gemm_direct_planar(
            gemm_a_interleaved_buf_,
            prepared_b_.B_neg_interleaved,
            C_real, C_imag,
            M, N, K, batch_count,
            alpha, beta, stream);
    }

    /// Execute prepared GEMM with fused power detection (Re²+Im² → single FP32 buffer).
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
            throw std::runtime_error("B_neg not available (only FP8 precision supports direct GEMM)");

        ensure_streams();

        // Cast A from planar FP16 to interleaved FP8
        int64_t total_A = static_cast<int64_t>(M) * K * batch_count;
        ensure_gemm_a_interleaved(total_A * 2, stream);
        cast_fp16_planar_to_fp8_interleaved_sm100(
            A_real, A_imag,
            gemm_a_interleaved_buf_,
            total_A, stream);

        // Launch direct GEMM kernel with power detection
        return launch_gemm_direct_power(
            gemm_a_interleaved_buf_,
            prepared_b_.B_neg_interleaved,
            C_power,
            M, N, K, batch_count,
            alpha, beta, stream);
    }

    /// Execute prepared GEMM with pre-cast FP8 A and fused power detection.
    /// Skips the internal FP16→FP8 A cast — caller provides FP8 interleaved A directly.
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
            throw std::runtime_error("B_neg not available (only FP8 precision supports direct GEMM)");

        // Launch direct GEMM kernel with power detection — A already in FP8 interleaved
        return launch_gemm_direct_power(
            A_fp8_interleaved,
            prepared_b_.B_neg_interleaved,
            C_power,
            M, N, K, batch_count,
            alpha, beta, stream);
    }

    /// Execute prepared GEMM via persistent direct kernel (planar FP32 output).
    /// K-gated: best when K <= K_CHUNK (64). Falls back to large-K pipeline otherwise.
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
        cast_fp16_planar_to_fp8_interleaved_sm100(
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
        cast_fp16_planar_to_fp8_interleaved_sm100(
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
    LowPrecisionBufferManager lp_buffers_;
    LowPrecisionBufferManager lp_buffers_b_;  // Second buffer set for pipeline double-buffering
    cudaEvent_t pipeline_preprocess_done_[2] = {nullptr, nullptr};
    cudaEvent_t pipeline_gemm_done_[2] = {nullptr, nullptr};
    bool pipeline_events_initialized_ = false;
    static constexpr int kPipelineKThreshold = 512;

    void ensure_pipeline_events() {
        if (pipeline_events_initialized_) return;
        for (int i = 0; i < 2; ++i) {
            CUDA_CHECK(cudaEventCreateWithFlags(&pipeline_preprocess_done_[i], cudaEventDisableTiming));
            CUDA_CHECK(cudaEventCreateWithFlags(&pipeline_gemm_done_[i], cudaEventDisableTiming));
        }
        pipeline_events_initialized_ = true;
    }

    void destroy_pipeline_events() {
        if (!pipeline_events_initialized_) return;
        for (int i = 0; i < 2; ++i) {
            if (pipeline_preprocess_done_[i]) cudaEventDestroy(pipeline_preprocess_done_[i]);
            if (pipeline_gemm_done_[i]) cudaEventDestroy(pipeline_gemm_done_[i]);
            pipeline_preprocess_done_[i] = nullptr;
            pipeline_gemm_done_[i] = nullptr;
        }
        pipeline_events_initialized_ = false;
    }

    int hw_sm_count_ = 0;
    int l2_cache_bytes_ = 0;
    int persisting_l2_max_ = 0;
    PreparedBState prepared_b_;

    /// Compute scale factor buffer size in bytes for a matrix of size (rows × cols).
    /// Uses the MXFP formula: roundup(rows, 128) × roundup(cols/32, 4) bytes.
    static int64_t sf_buffer_bytes(int rows, int cols) {
        int64_t padded_rows = ((rows + 127) / 128) * 128;
        int64_t k_blocks = (cols + 31) / 32;
        int64_t padded_k_blocks = ((k_blocks + 3) / 4) * 4;
        return padded_rows * padded_k_blocks;  // 1 byte per float_ue8m0_t
    }

    // A interleaved FP8 buffer for direct GEMM kernel (per-call, not pre-computed)
    __nv_fp8_e4m3* gemm_a_interleaved_buf_ = nullptr;
    int64_t gemm_a_interleaved_capacity_ = 0;

    void ensure_gemm_a_interleaved(int64_t num_fp8_elements, cudaStream_t stream) {
        if (num_fp8_elements > gemm_a_interleaved_capacity_) {
            if (gemm_a_interleaved_buf_) cudaFree(gemm_a_interleaved_buf_);
            CUDA_CHECK(cudaMalloc(&gemm_a_interleaved_buf_,
                                   num_fp8_elements * sizeof(__nv_fp8_e4m3)));
            gemm_a_interleaved_capacity_ = num_fp8_elements;
        }
    }

    // Pre-cast buffer for direct HERK cp.async path (FP16→FP8 done once before kernel)
    __nv_fp8_e4m3* herk_precast_buf_ = nullptr;
    int64_t herk_precast_capacity_ = 0;

    void ensure_herk_precast(int64_t num_fp8_elements, cudaStream_t stream) {
        if (num_fp8_elements > herk_precast_capacity_) {
            if (herk_precast_buf_) cudaFree(herk_precast_buf_);
            CUDA_CHECK(cudaMalloc(&herk_precast_buf_, num_fp8_elements));
            herk_precast_capacity_ = num_fp8_elements;
        }
    }

    // N×N scratch buffer for direct HERK coalesced output (FP16 only).
    // Direct kernel writes to full N×N row-major interleaved scratch (zero write amplification),
    // then pack_scratch_to_triangle extracts the lower triangle to packed format.
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

    // HERK production mode: temp buffer for Xi·Xr^T before anti-symmetrization
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
    // Block-scaled HERK: 3-GEMM path for FP6/FP4 (NoTrans only)
    // ---------------------------------------------------------------
    //
    // Exploits C = C^H symmetry: preprocess A once, reuse same buffers
    // for both A-side and B-side operands (self-product: A = B).
    //
    // Stream A — Re(C): 2 sub-GEMMs
    //   GEMM 1: Ar × Ar^T → scratch_Re  (alpha, beta=0)
    //   GEMM 2: Ai × Ai^T → scratch_Re  (alpha, beta=1, accumulate)
    //   pack_triangle → C_real_packed
    //
    // Stream B — Im(C): 1 sub-GEMM
    //   GEMM 3: Ai × Ar^T → temp_Im  (alpha=1, beta=0)
    //   antisymmetrize_pack → C_imag_packed
    //

    // ---------------------------------------------------------------
    // CUDA Graph cache for triangle slab launches (SM120 optimization)
    // ---------------------------------------------------------------
    // Caches per-slab kernel launch sequences as CUDA graphs for near-zero
    // replay overhead on repeated HERK calls. Uses a 4-entry cache because
    // triangle HERK alternates between parameter sets (e.g., beta=0 for Xr,
    // beta=1 for Xi) and may use different batch counts.
    struct TriangleGraphEntry {
        cudaGraphExec_t exec = nullptr;
        int cached_N = 0, cached_K = 0, cached_batch = 0, cached_slabs = 0;
        float cached_alpha = 0, cached_beta = 0;
        const void* cached_A = nullptr;
        const void* cached_B = nullptr;
        void* cached_C = nullptr;

        bool matches(int N, int K_val, int batch, int slabs, float a, float b,
                     const void* A, const void* B, void* C) const {
            return exec && N == cached_N && K_val == cached_K &&
                   batch == cached_batch && slabs == cached_slabs &&
                   a == cached_alpha && b == cached_beta &&
                   A == cached_A && B == cached_B && C == cached_C;
        }

        void store(int N, int K_val, int batch, int slabs, float a, float b,
                   const void* A, const void* B, void* C, cudaGraphExec_t e) {
            clear();
            exec = e;
            cached_N = N; cached_K = K_val; cached_batch = batch;
            cached_slabs = slabs; cached_alpha = a; cached_beta = b;
            cached_A = A; cached_B = B; cached_C = C;
        }

        void clear() {
            if (exec) { cudaGraphExecDestroy(exec); exec = nullptr; }
        }
    };

    static constexpr int kTriangleGraphCacheSize = 4;
    TriangleGraphEntry triangle_graph_cache_[kTriangleGraphCacheSize];

    // Find matching entry or return nullptr
    TriangleGraphEntry* find_triangle_graph(int N, int K_val, int batch, int slabs,
                                            float a, float b,
                                            const void* A, const void* B, void* C) {
        for (int i = 0; i < kTriangleGraphCacheSize; ++i) {
            if (triangle_graph_cache_[i].matches(N, K_val, batch, slabs, a, b, A, B, C))
                return &triangle_graph_cache_[i];
        }
        return nullptr;
    }

    // Find an empty slot, or evict the oldest (slot 0) and shift down
    TriangleGraphEntry* alloc_triangle_graph_slot() {
        for (int i = 0; i < kTriangleGraphCacheSize; ++i) {
            if (!triangle_graph_cache_[i].exec) return &triangle_graph_cache_[i];
        }
        // Evict slot 0 (oldest), shift entries down
        triangle_graph_cache_[0].clear();
        for (int i = 0; i < kTriangleGraphCacheSize - 1; ++i) {
            triangle_graph_cache_[i] = triangle_graph_cache_[i + 1];
        }
        triangle_graph_cache_[kTriangleGraphCacheSize - 1] = TriangleGraphEntry{};
        return &triangle_graph_cache_[kTriangleGraphCacheSize - 1];
    }

    void clear_triangle_graph_cache() {
        for (int i = 0; i < kTriangleGraphCacheSize; ++i)
            triangle_graph_cache_[i].clear();
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
        // Evict slot 0 (oldest), shift entries down
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
    // Stream parallelism (same as SM90 version)
    // ---------------------------------------------------------------
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
        CUDA_CHECK(cudaEventCreateWithFlags(&preprocess_done_, cudaEventDisableTiming));
        CUDA_CHECK(cudaEventCreateWithFlags(&stream_a_done_, cudaEventDisableTiming));
        CUDA_CHECK(cudaEventCreateWithFlags(&stream_b_done_, cudaEventDisableTiming));
        streams_initialized_ = true;
    }

    void destroy_streams() {
        clear_triangle_graph_cache();
        clear_herk_graph_cache();
        prepared_b_.free_all();
        destroy_pipeline_events();
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

    // ---------------------------------------------------------------
    // Core real-valued GEMM — templated on KernelTypeChain
    // ---------------------------------------------------------------
    //
    // run_real_gemm_impl<Chain>: The actual CUTLASS kernel launch, parameterized
    // on the kernel type chain (FP8, FP6, FP4). Inputs are typed pointers matching
    // the chain's ElementA/ElementB types.
    //
    // run_real_gemm: Backward-compatible thin wrapper that dispatches to ChainFP8.
    // All existing code continues to call this without changes.
    //
    // run_real_gemm_dispatch: Runtime dispatch based on ComputePrecision enum.
    // Takes void* inputs — used by the multi-precision path in run_planar().
    //

#include "gemm_complex_sm100/dispatch_impl.hpp"

#include "gemm_complex_sm100/triangle_impl.hpp"

};


// ========================================================================================
// Free-standing __global__ kernels
// ========================================================================================

__global__ void deinterleave_kernel_sm100(
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

__global__ void interleave_kernel_sm100(
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

inline void GemmComplexSm100::deinterleave_complex(
    const __half* interleaved, __half* real_out, __half* imag_out,
    int64_t num_complex_elements, cudaStream_t stream)
{
    constexpr int kBlock = 256;
    int grid = static_cast<int>((num_complex_elements + kBlock - 1) / kBlock);
    deinterleave_kernel_sm100<<<grid, kBlock, 0, stream>>>(
        interleaved, real_out, imag_out, num_complex_elements);
    CUDA_CHECK(cudaGetLastError());
}

inline void GemmComplexSm100::interleave_complex(
    const __half* real_in, const __half* imag_in, __half* interleaved_out,
    int64_t num_complex_elements, cudaStream_t stream)
{
    constexpr int kBlock = 256;
    int grid = static_cast<int>((num_complex_elements + kBlock - 1) / kBlock);
    interleave_kernel_sm100<<<grid, kBlock, 0, stream>>>(
        real_in, imag_in, interleaved_out, num_complex_elements);
    CUDA_CHECK(cudaGetLastError());
}

}  // namespace gemm_complex_sm100
