/*
 * gemm_sm100_type_chains.hpp
 *
 * CUTLASS kernel type chain definitions for SM100/SM120 multi-precision GEMM.
 *
 * This header is separated from gemm_complex_sm100.hpp so that it can be
 * safely included by multiple .cu compilation units without causing multiple-
 * definition linker errors (the main header contains __global__ CUDA kernels
 * that cannot appear in multiple TUs).
 *
 * Contains ONLY:
 *   - CUTLASS framework includes
 *   - Shared type configuration (layouts, strides, alignments, schedules)
 *   - KernelTypeChain template (FP8 OpClassTensorOp)
 *   - BlockScaledKernelTypeChain template (FP6/FP4 OpClassBlockScaledTensorOp)
 *   - Chain instantiations (ChainFP8, ChainFP6_E3M2, ChainFP6_E2M3, ChainFP4)
 *
 * Does NOT contain: __global__ kernels, class definitions, buffer managers.
 */

#pragma once

// ---- CUTLASS 4.x Core Includes ----
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

// ---- SM100 Specific Includes ----
#include "cutlass/epilogue/fusion/operations.hpp"

// ---- Block-Scaled (MXFP) Includes ----
#if defined(COMPLEX_SM100_ENABLE_FP6) || defined(COMPLEX_SM100_ENABLE_FP4)
#include "cutlass/detail/sm100_blockscaled_layout.hpp"
#endif

// ---- CuTe ----
#include "cute/tensor.hpp"

// ---- Standard Library ----
#include <type_traits>   // std::conditional_t, std::is_same_v (for StageCountTag)


namespace gemm_complex_sm100 {

// ========================================================================================
// CUTLASS 4.x SM100 Kernel Type Chains — Multi-Precision Support
// ========================================================================================
//
// Shared types (layouts, schedules, architecture) are defined once. The per-precision
// kernel type chain is parameterized via the KernelTypeChain template struct.
//
// FP8 E4M3 is always available. FP6 and FP4 require compile-time gates:
//   -DCOMPLEX_SM100_ENABLE_FP6=1  →  ChainFP6_E3M2, ChainFP6_E2M3
//   -DCOMPLEX_SM100_ENABLE_FP4=1  →  ChainFP4
//

// ----- Shared Type Configuration -----

using ElementAccumulator = float;     // tcgen05.mma always accumulates in FP32
using ElementC = cutlass::half_t;
using ElementD = cutlass::half_t;
using ElementCompute = float;         // Epilogue α/β computation precision

// ----- Layout Configuration -----
// TN layout: A=RowMajor, B=ColumnMajor (same as Hopper — optimal for TMA)
// SM100 vanilla FP8 supports all four layouts (TT, TN, NT, NN).
// SM120 (GeForce) is restricted to TN layout only.

using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::ColumnMajor;
using LayoutC = cutlass::layout::RowMajor;
using LayoutD = cutlass::layout::RowMajor;

using StrideA = cutlass::gemm::TagToStrideA_t<LayoutA>;
using StrideB = cutlass::gemm::TagToStrideB_t<LayoutB>;
using StrideC = cutlass::gemm::TagToStrideC_t<LayoutC>;
using StrideD = cutlass::gemm::TagToStrideC_t<LayoutD>;

// ----- Alignment (16 bytes = 128 bits, same as Hopper) -----
// For SM100: alignment is specified as 128 / bits_per_element in elements
static constexpr int AlignmentFP8  = 16;   // 128 / 8 = 16
static constexpr int AlignmentINT8 = 16;   // 128 / 8 = 16
static constexpr int AlignmentFP16 = 8;    // 128 / 16 = 8
#ifdef COMPLEX_SM100_ENABLE_FP6
static constexpr int AlignmentFP6  = 128;  // MXFP block-scaled: 128 elements × 6 bits = 96 bytes
#endif
#ifdef COMPLEX_SM100_ENABLE_FP4
static constexpr int AlignmentFP4  = 128;  // MXFP block-scaled: 128 elements × 4 bits = 64 bytes
#endif

// ----- MmaTileShape Configuration -----
//
// CRITICAL DIFFERENCE from Hopper: On SM100, the TileShape parameter to the
// CollectiveBuilder is the MmaTileShape — the tile consumed by 1 or 2 SM
// tcgen05.mma instructions — NOT the CTA output tile. The builder derives
// the actual CTA tile from MmaTileShape × number of MMA iterations.
//
// Valid MmaTileShapes for vanilla FP8 E4M3 in TN layout:
//
//   1SM (KernelTmaWarpSpecialized1SmSm100):
//     128×64×128, 128×128×128, 128×256×128
//
//   2SM (KernelTmaWarpSpecialized2SmSm100):
//     256×64×128, 256×128×128, 256×256×128
//     (ClusterShape M must be a multiple of 2)
//
// Default: 128×128×128 with 1SM (conservative, works on both SM100 and SM120)
// For GB200 at large problem sizes: try 256×256×128 with 2SM for maximum throughput
//
// Sub-byte types may require different K dimensions:
//   FP8: K=128 (default)
//   FP6: K=128 (same as FP8, hardware handles 6-bit packing)
//   FP4: K=256 (half the bits → hardware may want 2× elements per tile)
//

#ifndef COMPLEX_FP8_SM100_MMA_M
#define COMPLEX_FP8_SM100_MMA_M 128
#endif
#ifndef COMPLEX_FP8_SM100_MMA_N
#define COMPLEX_FP8_SM100_MMA_N 128
#endif
#ifndef COMPLEX_FP8_SM100_MMA_K
#define COMPLEX_FP8_SM100_MMA_K 128
#endif

// Per-precision K defaults (overridable via cmake)
#ifndef COMPLEX_SM100_FP6_MMA_K
#define COMPLEX_SM100_FP6_MMA_K 128
#endif
#ifndef COMPLEX_SM100_FP4_MMA_K
#define COMPLEX_SM100_FP4_MMA_K 256
#endif

// Per-precision M/N defaults for block-scaled kernels.
// Block-scaled kernels need extra SMEM for scale factors.
// On SM120 (99 KB SMEM), the default 128×128 tile may not fit with
// scale factor storage. Override to smaller values if initialize() fails
// with kErrorInternal (SMEM overflow).
#ifndef COMPLEX_SM100_BLKSCALED_MMA_M
#define COMPLEX_SM100_BLKSCALED_MMA_M COMPLEX_FP8_SM100_MMA_M
#endif
#ifndef COMPLEX_SM100_BLKSCALED_MMA_N
#define COMPLEX_SM100_BLKSCALED_MMA_N COMPLEX_FP8_SM100_MMA_N
#endif

// SM120 FP8 tile: 128×64×128 fits 3-stage pipeline in 99 KB SMEM (HERK-optimized).
// Default 128×128×128 needs 110 KB with 3 stages — exceeds SM120's 99 KB limit.
// The 128×64 tile halves B-operand SMEM per stage (8 KB vs 16 KB), bringing
// 3-stage total to ~82 KB. Listed as valid for 1SM FP8 in CUTLASS docs.
//
// For GEMM-heavy workloads (dedisp, beamformer), the wider 128×128 tile with
// auto-carveout (2 stages) may be faster. Override via -DCOMPLEX_SM100_FP8_TILE_N=128.
#ifdef COMPLEX_SM100_FP8_TILE_N
// Explicit FP8 tile N override from cmake (e.g. -DCOMPLEX_SM100_FP8_TILE_N=128 for GEMM)
using MmaTileShape = cute::Shape<
    cute::Int<COMPLEX_FP8_SM100_MMA_M>,
    cute::Int<COMPLEX_SM100_FP8_TILE_N>,
    cute::Int<COMPLEX_FP8_SM100_MMA_K>>;
#elif defined(COMPLEX_FP8_SM100_TARGET_SM120)
using MmaTileShape = cute::Shape<cute::_128, cute::_64, cute::Int<COMPLEX_FP8_SM100_MMA_K>>;
#else
using MmaTileShape = cute::Shape<
    cute::Int<COMPLEX_FP8_SM100_MMA_M>,
    cute::Int<COMPLEX_FP8_SM100_MMA_N>,
    cute::Int<COMPLEX_FP8_SM100_MMA_K>>;
#endif

// Actual FP8 tile dimensions (varies by arch: 128×64 on SM120, 128×128 on SM100)
static constexpr int kFP8TileM = cute::size<0>(MmaTileShape{});
static constexpr int kFP8TileN = cute::size<1>(MmaTileShape{});

#ifdef COMPLEX_SM100_ENABLE_FP6
using MmaTileShapeFP6 = cute::Shape<
    cute::Int<COMPLEX_SM100_BLKSCALED_MMA_M>,
    cute::Int<COMPLEX_SM100_BLKSCALED_MMA_N>,
    cute::Int<COMPLEX_SM100_FP6_MMA_K>>;
#endif

#ifdef COMPLEX_SM100_ENABLE_FP4
using MmaTileShapeFP4 = cute::Shape<
    cute::Int<COMPLEX_SM100_BLKSCALED_MMA_M>,
    cute::Int<COMPLEX_SM100_BLKSCALED_MMA_N>,
    cute::Int<COMPLEX_SM100_FP4_MMA_K>>;
#endif

// ----- Cluster Shape -----
//
// SM100 supports dynamic preferred + fallback clusters via CLC.
// For 2SM MMA, ClusterShape M must be a multiple of 2.
// SM120 (GeForce) is limited to 1×1×1.
//
#ifndef COMPLEX_FP8_SM100_CLUSTER_M
#define COMPLEX_FP8_SM100_CLUSTER_M 1
#endif
#ifndef COMPLEX_FP8_SM100_CLUSTER_N
#define COMPLEX_FP8_SM100_CLUSTER_N 1
#endif

using ClusterShape = cute::Shape<
    cute::Int<COMPLEX_FP8_SM100_CLUSTER_M>,
    cute::Int<COMPLEX_FP8_SM100_CLUSTER_N>,
    cute::_1>;

// ----- Kernel Schedule Selection -----
//
// SM100 replaces Hopper's Cooperative/Pingpong with 1Sm/2Sm schedules.
// The tcgen05 MMA instruction set eliminates the wgmma serialization problem (C7510)
// and uses TMEM (Tensor Memory) for accumulator double-buffering instead of SMEM.
//
// 1SM: Each SM executes its own MMA independently. Best for small/medium problems.
// 2SM: Two SMs cooperate on a single MMA instruction. Best for large problems where
//      the 2× compute per tile offsets the coordination overhead.
//

#if defined(COMPLEX_FP8_SM100_USE_2SM) && COMPLEX_FP8_SM100_USE_2SM
// 2SM schedule: requires ClusterShape M to be a multiple of 2 (SM100 only)
using KernelSchedule   = cutlass::gemm::KernelTmaWarpSpecialized2SmSm100;
using EpilogueSchedule = cutlass::epilogue::TmaWarpSpecialized2Sm;
#elif defined(COMPLEX_FP8_SM100_TARGET_SM120)
// SM120: let the CollectiveBuilder auto-select the correct dispatch policy.
// SM120 has its own kernel paths in CUTLASS 4.3+ and may not accept SM100
// schedule tags directly.
using KernelSchedule   = cutlass::gemm::collective::KernelScheduleAuto;
using EpilogueSchedule = cutlass::epilogue::collective::EpilogueScheduleAuto;
#else
// 1SM schedule (default for SM100)
using KernelSchedule   = cutlass::gemm::KernelTmaWarpSpecialized1SmSm100;
using EpilogueSchedule = cutlass::epilogue::TmaWarpSpecialized1Sm;
#endif

// ----- Architecture Tag -----
//
// SM100 = datacenter Blackwell (B200, GB200)
// SM120 = consumer Blackwell (RTX 5090, GB10 Spark)
// Code compiled for sm100a will NOT run on SM120 and vice versa.
//
// CUTLASS 4.3+ has native Sm120 support with dedicated CollectiveBuilder,
// epilogue builder (sm120_builder.inl), and dispatch policies. Although SM120
// shares the tcgen05 ISA, ptxas requires sm_12x-specific instruction encodings
// that differ from sm_100a. Using the wrong ArchTag causes ptxas to reject
// the generated PTX.
//
// The COMPLEX_FP8_SM100_TARGET_SM120 macro selects both the correct ArchTag
// and schedule types, and enforces SM120 constraints (no multicast, cluster 1×1×1).

// ArchTag: Use Sm120 for GB10/RTX 50-series, Sm100 for datacenter GB200.
#if defined(COMPLEX_FP8_SM100_TARGET_SM120)
using ArchTag = cutlass::arch::Sm120;
#else
using ArchTag = cutlass::arch::Sm100;
#endif

// ----- Epilogue Fusion -----
//
// Linear combination: D = α * Accumulator + β * C
// Same as Hopper but with explicit fusion operation for SM100 epilogue builder.

using FusionOperation = cutlass::epilogue::fusion::LinearCombination<
    ElementD,                  // Output element type
    ElementCompute,            // Compute precision (FP32)
    ElementC,                  // Source element type
    ElementCompute,            // Scalar type for alpha/beta
    cutlass::FloatRoundStyle::round_to_nearest  // Rounding mode
>;


// ========================================================================================
// Stage Count Tags — For runtime-selectable pipeline stage counts
// ========================================================================================
//
// KernelTypeChain and KernelTypeChainFP32Out accept an optional 5th template parameter
// to override the pipeline stage count:
//   UseDefaultStages  — Use the compile-time configured default (StageCount<N> or AutoCarveout)
//   UseAutoCarveout   — Always use StageCountAutoCarveout regardless of compile-time config
//
// This allows the same binary to contain both explicit-stage and auto-stage kernel variants
// (e.g. FP8_T128x64_C1x1_S3 vs FP8_T128x64_C1x1_SAuto).

struct UseDefaultStages {};   // Sentinel: use compile-time configured stages
struct UseAutoCarveout {};    // Force StageCountAutoCarveout regardless of config

// ========================================================================================
// KernelTypeChain — Parameterized kernel type chain for multi-precision support
// ========================================================================================
//
// Each precision has its own chain. All chains share the same epilogue (C/D = FP16,
// accumulator = FP32) and architecture config. Only the mainloop element types,
// alignment, and tile shape differ.
//
// Usage:
//   using Chain = KernelTypeChain<cutlass::float_e4m3_t, 16, MmaTileShape>;
//   typename Chain::DeviceGemm gemm_op;
//   typename Chain::GemmArguments arguments{ ... };
//   gemm_op.initialize(arguments, workspace, stream);
//   gemm_op.run(stream);
//
// Template parameters:
//   ElementAB_       — Element type for A and B operands (e.g. float_e4m3_t, float_e2m1_t)
//   AlignmentAB_     — Alignment in elements for A/B (128 bits / bits_per_element)
//   TileShape_       — MmaTileShape (may differ per precision due to K dimension)
//   ClusterShape_    — Cluster shape (default: compile-time configured ClusterShape)
//   StageCountTag_   — UseDefaultStages (default) or UseAutoCarveout
//
// The epilogue (C/D = FP16) and architecture config are shared across all chains.
// Only the mainloop element types and alignment change per precision.
//

template <typename ElementAB_, int AlignmentAB_, typename TileShape_,
          typename ClusterShape_ = ClusterShape,
          typename StageCountTag_ = UseDefaultStages>
struct KernelTypeChain {
    using ElementA = ElementAB_;
    using ElementB = ElementAB_;
    static constexpr int AlignmentAB = AlignmentAB_;
    using TileShape = TileShape_;

    // Epilogue: same for all precisions (C/D are always FP16)
    using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
        ArchTag,
        cutlass::arch::OpClassTensorOp,
        TileShape,
        ClusterShape_,
        cutlass::epilogue::collective::EpilogueTileAuto,
        ElementAccumulator,
        ElementCompute,
        ElementC, LayoutC, AlignmentFP16,
        ElementD, LayoutD, AlignmentFP16,
        EpilogueSchedule,
        FusionOperation
    >::CollectiveOp;

    // Pipeline stage count
    // UseAutoCarveout tag forces auto regardless of compile-time config.
    // UseDefaultStages uses the compile-time configured value.
    using _AutoCarveout = cutlass::gemm::collective::StageCountAutoCarveout<
        static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>;

#if defined(COMPLEX_SM100_FP8_TILE_N) && defined(COMPLEX_FP8_SM100_TARGET_SM120) && COMPLEX_SM100_FP8_TILE_N > 64
    using _DefaultStages = _AutoCarveout;  // SM120 tile override: forced auto
#elif defined(COMPLEX_FP8_SM100_STAGES) && COMPLEX_FP8_SM100_STAGES > 0
    using _DefaultStages = cutlass::gemm::collective::StageCount<COMPLEX_FP8_SM100_STAGES>;
#else
    using _DefaultStages = _AutoCarveout;  // No explicit stages: use auto
#endif

    using StageCountType = std::conditional_t<
        std::is_same_v<StageCountTag_, UseAutoCarveout>,
        _AutoCarveout,
        _DefaultStages>;

    // Mainloop: parameterized on element type and alignment
    using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
        ArchTag,
        cutlass::arch::OpClassTensorOp,
        ElementA, LayoutA, AlignmentAB,
        ElementB, LayoutB, AlignmentAB,
        ElementAccumulator,
        TileShape,
        ClusterShape_,
        StageCountType,
        KernelSchedule
    >::CollectiveOp;

    // Full kernel
    using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
        cute::Shape<int, int, int, int>,
        CollectiveMainloop,
        CollectiveEpilogue
    >;

    using DeviceGemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
    using GemmArguments = typename DeviceGemm::Arguments;

    using ClusterShapeType = ClusterShape_;
    static constexpr bool IsBlockScaled = false;
};

// ========================================================================================
// KernelTypeChainFP32Out — FP8 input, FP32 output (for large K / high dynamic range)
// ========================================================================================
//
// Identical to KernelTypeChain except C/D are float instead of half_t.
// FP32 output avoids FP16 overflow when K * max_A * max_B > 65504.
// Used by the dedisp FDD pipeline where K=512 and A/B are near FP8 saturation.
//

static constexpr int AlignmentFP32 = 4;   // 128 / 32 = 4 elements

using FusionOperationFP32Out = cutlass::epilogue::fusion::LinearCombination<
    float,                     // Output element type (FP32)
    ElementCompute,            // Compute precision (FP32)
    float,                     // Source element type (FP32)
    ElementCompute,            // Scalar type for alpha/beta
    cutlass::FloatRoundStyle::round_to_nearest
>;

template <typename ElementAB_, int AlignmentAB_, typename TileShape_,
          typename ClusterShape_ = ClusterShape,
          typename StageCountTag_ = UseDefaultStages>
struct KernelTypeChainFP32Out {
    using ElementA = ElementAB_;
    using ElementB = ElementAB_;
    static constexpr int AlignmentAB = AlignmentAB_;
    using TileShape = TileShape_;

    // Epilogue: FP32 accumulator → FP32 output
    using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
        ArchTag,
        cutlass::arch::OpClassTensorOp,
        TileShape,
        ClusterShape_,
        cutlass::epilogue::collective::EpilogueTileAuto,
        ElementAccumulator,
        ElementCompute,
        float, LayoutC, AlignmentFP32,
        float, LayoutD, AlignmentFP32,
        EpilogueSchedule,
        FusionOperationFP32Out
    >::CollectiveOp;

    // Pipeline stage count (same tag system as KernelTypeChain)
    using _AutoCarveout = cutlass::gemm::collective::StageCountAutoCarveout<
        static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>;

#if defined(COMPLEX_SM100_FP8_TILE_N) && defined(COMPLEX_FP8_SM100_TARGET_SM120) && COMPLEX_SM100_FP8_TILE_N > 64
    using _DefaultStages = _AutoCarveout;
#elif defined(COMPLEX_FP8_SM100_STAGES) && COMPLEX_FP8_SM100_STAGES > 0
    using _DefaultStages = cutlass::gemm::collective::StageCount<COMPLEX_FP8_SM100_STAGES>;
#else
    using _DefaultStages = _AutoCarveout;
#endif

    using StageCountType = std::conditional_t<
        std::is_same_v<StageCountTag_, UseAutoCarveout>,
        _AutoCarveout,
        _DefaultStages>;

    // Mainloop: same as KernelTypeChain (FP8 input)
    using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
        ArchTag,
        cutlass::arch::OpClassTensorOp,
        ElementA, LayoutA, AlignmentAB,
        ElementB, LayoutB, AlignmentAB,
        ElementAccumulator,
        TileShape,
        ClusterShape_,
        StageCountType,
        KernelSchedule
    >::CollectiveOp;

    // Full kernel
    using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
        cute::Shape<int, int, int, int>,
        CollectiveMainloop,
        CollectiveEpilogue
    >;

    using DeviceGemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
    using GemmArguments = typename DeviceGemm::Arguments;

    using ClusterShapeType = ClusterShape_;
    static constexpr bool IsBlockScaled = false;
};

// ========================================================================================
// IntegerKernelTypeChain — For INT8 with INT32 accumulation (SM100 only)
// ========================================================================================
//
// INT8 tensor cores use kind::i8 MMA atoms (SM100_MMA_S8_SS/TS), which mandate
// int32_t accumulators. Same throughput as FP8 (2048 ops/cycle/SM).
//
// SM120 (consumer Blackwell) only supports F8F6F4 element types — no INT8 tensor
// cores. This chain is gated behind #ifndef COMPLEX_FP8_SM100_TARGET_SM120.
//
// The epilogue converts INT32 → float (for alpha/beta computation) → FP16 output.
// This chain provides exact integer arithmetic for the GEMM, with no floating-point
// rounding in the MMA accumulation.
//

#ifndef COMPLEX_FP8_SM100_TARGET_SM120

template <typename ElementAB_, int AlignmentAB_, typename TileShape_,
          typename ClusterShape_ = ClusterShape>
struct IntegerKernelTypeChain {
    using ElementA = ElementAB_;
    using ElementB = ElementAB_;
    static constexpr int AlignmentAB = AlignmentAB_;
    using TileShape = TileShape_;

    using IntAccumulator = int32_t;

    // Epilogue: INT32 accumulator → FP16 output via FP32 compute
    using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
        ArchTag,
        cutlass::arch::OpClassTensorOp,
        TileShape,
        ClusterShape_,
        cutlass::epilogue::collective::EpilogueTileAuto,
        IntAccumulator,        // int32_t from MMA
        ElementCompute,        // FP32 for alpha/beta
        ElementC, LayoutC, AlignmentFP16,
        ElementD, LayoutD, AlignmentFP16,
        EpilogueSchedule,
        FusionOperation
    >::CollectiveOp;

    // Pipeline stage count
    using StageCountType = cutlass::gemm::collective::StageCountAutoCarveout<
        static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))
    >;

    // Mainloop: int8_t elements with int32_t accumulation
    using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
        ArchTag,
        cutlass::arch::OpClassTensorOp,
        ElementA, LayoutA, AlignmentAB,
        ElementB, LayoutB, AlignmentAB,
        IntAccumulator,        // int32_t accumulator
        TileShape,
        ClusterShape_,
        StageCountType,
        KernelSchedule
    >::CollectiveOp;

    // Full kernel
    using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
        cute::Shape<int, int, int, int>,
        CollectiveMainloop,
        CollectiveEpilogue
    >;

    using DeviceGemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
    using GemmArguments = typename DeviceGemm::Arguments;

    using ClusterShapeType = ClusterShape_;
    static constexpr bool IsBlockScaled = false;
};

#endif // !COMPLEX_FP8_SM100_TARGET_SM120


// ========================================================================================
// GroupedKernelTypeChain — Grouped GEMM for single-launch triangle decomposition (SM100)
// ========================================================================================
//
// Grouped GEMM packs multiple independent sub-GEMMs (one per slab × batch) into a
// single kernel launch. This eliminates per-slab launch overhead when using triangle-
// aware HERK decomposition.
//
// Key differences from standard KernelTypeChain:
//   - ProblemShape: GroupProblemShape<Shape<int,int,int>> (rank-3, no batch dim)
//   - Schedules: PtrArray variants (KernelPtrArrayTmaWarpSpecialized1SmSm100)
//   - Layouts: pointer-decorated (LayoutA *, LayoutB *, LayoutC *, LayoutD *)
//   - Strides/pointers: device arrays (one entry per group)
//
// SM120 does NOT support grouped GEMM (mma_builder has static_assert(!IsPtrArrayKernel)).
// This chain is gated behind #ifndef COMPLEX_FP8_SM100_TARGET_SM120.
//

#ifndef COMPLEX_FP8_SM100_TARGET_SM120

#include "cutlass/gemm/group_array_problem_shape.hpp"

template <typename ElementAB_, int AlignmentAB_, typename TileShape_,
          typename ClusterShape_ = ClusterShape>
struct GroupedKernelTypeChain {
    using ElementA = ElementAB_;
    using ElementB = ElementAB_;
    static constexpr int AlignmentAB = AlignmentAB_;
    using TileShape = TileShape_;

    // Grouped GEMM schedules (PtrArray variants)
    using GroupedEpilogueSchedule = cutlass::epilogue::PtrArrayTmaWarpSpecialized1Sm;
    using GroupedKernelSchedule = cutlass::gemm::KernelPtrArrayTmaWarpSpecialized1SmSm100;

    // Epilogue: pointer-decorated layouts (LayoutC *, LayoutD *)
    using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
        ArchTag, cutlass::arch::OpClassTensorOp,
        TileShape, ClusterShape_,
        cutlass::epilogue::collective::EpilogueTileAuto,
        ElementAccumulator, ElementCompute,
        ElementC, LayoutC *, AlignmentFP16,
        ElementD, LayoutD *, AlignmentFP16,
        GroupedEpilogueSchedule,
        FusionOperation
    >::CollectiveOp;

    using StageCountType = cutlass::gemm::collective::StageCountAutoCarveout<
        static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>;

    // Mainloop: pointer-decorated layouts (LayoutA *, LayoutB *)
    using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
        ArchTag, cutlass::arch::OpClassTensorOp,
        ElementA, LayoutA *, AlignmentAB,
        ElementB, LayoutB *, AlignmentAB,
        ElementAccumulator,
        TileShape, ClusterShape_,
        StageCountType,
        GroupedKernelSchedule
    >::CollectiveOp;

    // GroupProblemShape: rank-3 (M, N, K), no batch dimension
    using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
        cutlass::gemm::GroupProblemShape<cute::Shape<int, int, int>>,
        CollectiveMainloop, CollectiveEpilogue>;

    using DeviceGemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
    using GemmArguments = typename DeviceGemm::Arguments;

    // Stride types from the kernel (needed for device arrays)
    using InternalStrideA = typename GemmKernel::InternalStrideA;
    using InternalStrideB = typename GemmKernel::InternalStrideB;
    using InternalStrideC = typename GemmKernel::InternalStrideC;
    using InternalStrideD = typename GemmKernel::InternalStrideD;

    using ClusterShapeType = ClusterShape_;
    static constexpr bool IsBlockScaled = false;
};

using GroupedChainFP8 = GroupedKernelTypeChain<cutlass::float_e4m3_t, AlignmentFP8, MmaTileShape>;

#endif // !COMPLEX_FP8_SM100_TARGET_SM120


// ========================================================================================
// BlockScaledKernelTypeChain — For sub-byte MXFP types (FP6/FP4)
// ========================================================================================
//
// Sub-byte types (float_e2m1_t, float_e3m2_t, float_e2m3_t) CANNOT be used directly
// with OpClassTensorOp on SM100/SM120. They MUST be wrapped in mx_float4_t<> or
// mx_float6_t<> and use OpClassBlockScaledTensorOp.
//
// This chain produces kernels whose GemmArguments include mandatory scale factor
// pointers (ptr_SFA, layout_SFA, ptr_SFB, layout_SFB). The scale factors are
// per-block (32 elements) max-abs exponents stored as float_ue8m0_t.
//
// Template parameters:
//   MxWrapperType_  — mx_float4_t<float_e2m1_t>, mx_float6_t<float_e3m2_t>, etc.
//   AlignmentAB_    — Alignment in elements (128 for both FP6 and FP4 in MXFP mode)
//   TileShape_      — MmaTileShape (may differ per precision)
//

#if defined(COMPLEX_SM100_ENABLE_FP6) || defined(COMPLEX_SM100_ENABLE_FP4)

template <typename MxWrapperType_, int AlignmentAB_, typename TileShape_,
          typename ClusterShape_ = ClusterShape>
struct BlockScaledKernelTypeChain {
    using MxWrapperType = MxWrapperType_;
    using ElementA = MxWrapperType;
    using ElementB = MxWrapperType;
    using RawElementType = typename MxWrapperType::DataType;
    using ScaleFactorType = typename MxWrapperType::ScaleFactorType;
    static constexpr int AlignmentAB = AlignmentAB_;
    using TileShape = TileShape_;

    // Block-scaled operator class (required for sub-byte MXFP types)
    using OpClass = cutlass::arch::OpClassBlockScaledTensorOp;

    // Epilogue: same output types (C/D = FP16, accumulator = FP32)
    // Must use OpClassBlockScaledTensorOp + EpilogueScheduleAuto
    using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
        ArchTag,
        OpClass,
        TileShape,
        ClusterShape_,
        cutlass::epilogue::collective::EpilogueTileAuto,
        ElementAccumulator,
        ElementAccumulator,
        ElementC, LayoutC, AlignmentFP16,
        ElementD, LayoutD, AlignmentFP16,
        cutlass::epilogue::collective::EpilogueScheduleAuto
    >::CollectiveOp;

    // Pipeline stage count
    using StageCountType = cutlass::gemm::collective::StageCountAutoCarveout<
        static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))
    >;

    // Mainloop: block-scaled with MXFP wrapper types
    using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
        ArchTag,
        OpClass,
        ElementA, LayoutA, AlignmentAB,
        ElementB, LayoutB, AlignmentAB,
        ElementAccumulator,
        TileShape,
        ClusterShape_,
        StageCountType,
        cutlass::gemm::collective::KernelScheduleAuto
    >::CollectiveOp;

    // Scale factor layout types (complex CuTe layouts, not simple strides)
    using LayoutSFA = typename CollectiveMainloop::LayoutSFA;
    using LayoutSFB = typename CollectiveMainloop::LayoutSFB;
    using ElementSF = typename CollectiveMainloop::ElementSF;

    // Block-scaled config helper for computing scale factor layouts from problem shape
    using Sm1xxBlkScaledConfig = typename CollectiveMainloop::Sm1xxBlkScaledConfig;

    // Full kernel
    using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
        cute::Shape<int, int, int, int>,
        CollectiveMainloop,
        CollectiveEpilogue
    >;

    using DeviceGemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
    using GemmArguments = typename DeviceGemm::Arguments;

    using ClusterShapeType = ClusterShape_;
    static constexpr bool IsBlockScaled = true;
};

// ========================================================================================
// BlockScaledKernelTypeChainFP32Out — Sub-byte MXFP with FP32 output
// ========================================================================================
//
// Identical to BlockScaledKernelTypeChain but with float C/D output instead of half_t.
// FP32 output avoids FP16 overflow when K * max_A * max_B > 65504.
// Used by the dedisp FDD pipeline where K=512 and A/B are near saturation.
//

template <typename MxWrapperType_, int AlignmentAB_, typename TileShape_,
          typename ClusterShape_ = ClusterShape>
struct BlockScaledKernelTypeChainFP32Out {
    using MxWrapperType = MxWrapperType_;
    using ElementA = MxWrapperType;
    using ElementB = MxWrapperType;
    using RawElementType = typename MxWrapperType::DataType;
    using ScaleFactorType = typename MxWrapperType::ScaleFactorType;
    static constexpr int AlignmentAB = AlignmentAB_;
    using TileShape = TileShape_;

    // Block-scaled operator class (required for sub-byte MXFP types)
    using OpClass = cutlass::arch::OpClassBlockScaledTensorOp;

    // Epilogue: FP32 accumulator → FP32 output
    // IMPORTANT: Epilogue uses OpClassTensorOp (standard), NOT OpClassBlockScaledTensorOp.
    // The block-scaled op class is only for the mainloop. The epilogue operates on
    // FP32 accumulator tiles and writes FP32 output — this is standard tensor-op behavior.
    // Using OpClassBlockScaledTensorOp for the epilogue produces silently wrong results.
    // (Pattern confirmed by CUTLASS tests: sm120_bs_gemm_nvf4_nvf4_f32_f32.cu)
    using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
        ArchTag,
        cutlass::arch::OpClassTensorOp,
        TileShape,
        ClusterShape_,
        cutlass::epilogue::collective::EpilogueTileAuto,
        ElementAccumulator,
        ElementAccumulator,
        float, LayoutC, AlignmentFP32,
        float, LayoutD, AlignmentFP32,
        cutlass::epilogue::collective::EpilogueScheduleAuto
    >::CollectiveOp;

    // Pipeline stage count
    using StageCountType = cutlass::gemm::collective::StageCountAutoCarveout<
        static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))
    >;

    // Mainloop: block-scaled with MXFP wrapper types
    using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
        ArchTag,
        OpClass,
        ElementA, LayoutA, AlignmentAB,
        ElementB, LayoutB, AlignmentAB,
        ElementAccumulator,
        TileShape,
        ClusterShape_,
        StageCountType,
        cutlass::gemm::collective::KernelScheduleAuto
    >::CollectiveOp;

    // Scale factor layout types (complex CuTe layouts, not simple strides)
    using LayoutSFA = typename CollectiveMainloop::LayoutSFA;
    using LayoutSFB = typename CollectiveMainloop::LayoutSFB;
    using ElementSF = typename CollectiveMainloop::ElementSF;

    // Block-scaled config helper for computing scale factor layouts from problem shape
    using Sm1xxBlkScaledConfig = typename CollectiveMainloop::Sm1xxBlkScaledConfig;

    // Full kernel
    using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
        cute::Shape<int, int, int, int>,
        CollectiveMainloop,
        CollectiveEpilogue
    >;

    using DeviceGemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
    using GemmArguments = typename DeviceGemm::Arguments;

    using ClusterShapeType = ClusterShape_;
    static constexpr bool IsBlockScaled = true;
};

#endif // COMPLEX_SM100_ENABLE_FP6 || COMPLEX_SM100_ENABLE_FP4

// ========================================================================================
// Kernel Type Chain Instantiations
// ========================================================================================

// ----- FP8 E4M3 (always available — default, regression baseline) -----
using ChainFP8 = KernelTypeChain<cutlass::float_e4m3_t, AlignmentFP8, MmaTileShape>;

// ----- FP8 E4M3 with FP32 output (for large-K / high dynamic range applications) -----
using ChainFP8_FP32Out = KernelTypeChainFP32Out<cutlass::float_e4m3_t, AlignmentFP8, MmaTileShape>;

// ----- INT8 (exact integer arithmetic, INT32 accumulation) -----
// INT8 tensor cores (kind::i8) are only supported on SM100 (datacenter Blackwell).
// SM120 (consumer) only supports F8F6F4 element types in OpClassTensorOp.
#ifndef COMPLEX_FP8_SM100_TARGET_SM120
using ChainINT8 = IntegerKernelTypeChain<int8_t, AlignmentINT8, MmaTileShape>;
#endif

// Backward-compatible aliases (used throughout the class methods)
using ElementA_FP8 = cutlass::float_e4m3_t;
using ElementB_FP8 = cutlass::float_e4m3_t;
using CollectiveEpilogue = typename ChainFP8::CollectiveEpilogue;
using CollectiveMainloop = typename ChainFP8::CollectiveMainloop;
using GemmKernel = typename ChainFP8::GemmKernel;
using DeviceGemm = typename ChainFP8::DeviceGemm;
using GemmArguments = typename ChainFP8::GemmArguments;
#if defined(COMPLEX_SM100_FP8_TILE_N) && defined(COMPLEX_FP8_SM100_TARGET_SM120) && COMPLEX_SM100_FP8_TILE_N > 64
using StageCountType = cutlass::gemm::collective::StageCountAutoCarveout<
    static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))
>;
#elif defined(COMPLEX_FP8_SM100_STAGES) && COMPLEX_FP8_SM100_STAGES > 0
using StageCountType = cutlass::gemm::collective::StageCount<COMPLEX_FP8_SM100_STAGES>;
#else
using StageCountType = cutlass::gemm::collective::StageCountAutoCarveout<
    static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))
>;
#endif

// ----- FP6 E3M2 / E2M3 (compile-time gated, block-scaled MXFP) -----
#ifdef COMPLEX_SM100_ENABLE_FP6
using ChainFP6_E3M2 = BlockScaledKernelTypeChain<
    cutlass::mx_float6_t<cutlass::float_e3m2_t>, AlignmentFP6, MmaTileShapeFP6>;
using ChainFP6_E2M3 = BlockScaledKernelTypeChain<
    cutlass::mx_float6_t<cutlass::float_e2m3_t>, AlignmentFP6, MmaTileShapeFP6>;
#endif

// ----- FP4 E2M1 (compile-time gated, block-scaled MXFP) -----
#ifdef COMPLEX_SM100_ENABLE_FP4
using ChainFP4 = BlockScaledKernelTypeChain<
    cutlass::mx_float4_t<cutlass::float_e2m1_t>, AlignmentFP4, MmaTileShapeFP4>;
#endif

// ----- FP6 E3M2 with FP32 output (for large-K / high dynamic range) -----
#ifdef COMPLEX_SM100_ENABLE_FP6
using ChainFP6_E3M2_FP32Out = BlockScaledKernelTypeChainFP32Out<
    cutlass::mx_float6_t<cutlass::float_e3m2_t>, AlignmentFP6, MmaTileShapeFP6>;
#endif

// ----- FP4 E2M1 with FP32 output (for large-K / high dynamic range) -----
#ifdef COMPLEX_SM100_ENABLE_FP4
using ChainFP4_FP32Out = BlockScaledKernelTypeChainFP32Out<
    cutlass::mx_float4_t<cutlass::float_e2m1_t>, AlignmentFP4, MmaTileShapeFP4>;
#endif

// ========================================================================================
// Small-M Tile Instantiations (optimized for M ≤ 64 workloads)
// ========================================================================================
//
// When the problem M dimension is small (e.g., batch_size=32 in dedisp), the default
// 128-row block-scaled tiles waste 75% of compute (96 of 128 rows are padding).
// The 64-row tile reduces this to 50% — a 2× compute efficiency improvement.
//
// The SfAtom memory layout uses 128-row tiles regardless of MMA tile shape, so the
// same preprocessed data (scale factors + packed narrow values) works for both
// Default and SmallM configs.
//
// Gated behind COMPLEX_SM100_ENABLE_SMALL_M. Requires FP6 or FP4.
// Each chain adds ~15 min compile time.

// SmallM requires SM100 (datacenter) — SM120's block-scaled SfAtom TMA descriptor
// requires M_tile ≥ 128 (the SfAtom uses 128-row atoms and TMA cannot load partial atoms).
#if defined(COMPLEX_SM100_ENABLE_SMALL_M) && !defined(COMPLEX_FP8_SM100_TARGET_SM120)

#ifdef COMPLEX_SM100_ENABLE_FP6
// 64×128×K tile — halves M tile, 2× less M-waste for M≤64
using MmaTileShapeFP6_SmallM = cute::Shape<cute::_64, cute::_128, cute::Int<COMPLEX_SM100_FP6_MMA_K>>;

using ChainFP6_E3M2_SmallM = BlockScaledKernelTypeChain<
    cutlass::mx_float6_t<cutlass::float_e3m2_t>, AlignmentFP6, MmaTileShapeFP6_SmallM>;
using ChainFP6_E3M2_FP32Out_SmallM = BlockScaledKernelTypeChainFP32Out<
    cutlass::mx_float6_t<cutlass::float_e3m2_t>, AlignmentFP6, MmaTileShapeFP6_SmallM>;
#endif

#ifdef COMPLEX_SM100_ENABLE_FP4
using MmaTileShapeFP4_SmallM = cute::Shape<cute::_64, cute::_128, cute::Int<COMPLEX_SM100_FP4_MMA_K>>;

using ChainFP4_SmallM = BlockScaledKernelTypeChain<
    cutlass::mx_float4_t<cutlass::float_e2m1_t>, AlignmentFP4, MmaTileShapeFP4_SmallM>;
using ChainFP4_FP32Out_SmallM = BlockScaledKernelTypeChainFP32Out<
    cutlass::mx_float4_t<cutlass::float_e2m1_t>, AlignmentFP4, MmaTileShapeFP4_SmallM>;
#endif

#endif // COMPLEX_SM100_ENABLE_SMALL_M && !COMPLEX_FP8_SM100_TARGET_SM120


// ========================================================================================
// Runtime-Selectable Multi-Config Instantiations
// ========================================================================================
//
// ALL configurations are always compiled (no COMPLEX_SM100_MULTI_CONFIG gate).
// Runtime dispatch selects the optimal config via GemmConfig enum.
//
// Each new chain instantiation adds one CUTLASS kernel to compile (~15 min each).
// FP8 multi-config chains stay in the header (no nvcc stub issue).
// FP6/FP4 multi-config chains need concrete kernels in gemm_blockscaled_dispatch.cu.
//
// SM120: only 1×1 cluster, no 2SM, no multicast.
// SM100: full set including cluster and 2SM variants.

// ----- Named tile shapes (always defined, used by chain aliases below) -----
using MmaTileShape_128x64   = cute::Shape<cute::_128, cute::_64,  cute::Int<COMPLEX_FP8_SM100_MMA_K>>;
using MmaTileShape_128x128  = cute::Shape<cute::_128, cute::_128, cute::Int<COMPLEX_FP8_SM100_MMA_K>>;
using MmaTileShape_128x256  = cute::Shape<cute::_128, cute::_256, cute::Int<COMPLEX_FP8_SM100_MMA_K>>;

// ----- Named cluster shapes (always defined) -----
using ClusterShape_1x1 = cute::Shape<cute::_1, cute::_1, cute::_1>;
using ClusterShape_1x2 = cute::Shape<cute::_1, cute::_2, cute::_1>;
using ClusterShape_2x1 = cute::Shape<cute::_2, cute::_1, cute::_1>;
using ClusterShape_2x2 = cute::Shape<cute::_2, cute::_2, cute::_1>;

// ========================================================================================
// FP8 multi-tile chains (SM120 + SM100, 1SM 1×1 cluster)
// ========================================================================================
//
// Each tile shape has two stage-count variants:
//   (no suffix)  — Uses compile-time default stages (StageCount<3> when STAGES=3)
//   _Auto        — Forces StageCountAutoCarveout (fewer stages, smaller SMEM)
// Dispatch selects the variant matching the GemmConfig (S3 vs SAuto).

// FP8_T128x64_C1x1 — SM120 HERK-optimal
using ChainFP8_T128x64_C1x1 = KernelTypeChain<cutlass::float_e4m3_t, AlignmentFP8,
    MmaTileShape_128x64, ClusterShape_1x1>;
using ChainFP8_T128x64_C1x1_Auto = KernelTypeChain<cutlass::float_e4m3_t, AlignmentFP8,
    MmaTileShape_128x64, ClusterShape_1x1, UseAutoCarveout>;
using ChainFP8_T128x64_C1x1_FP32Out = KernelTypeChainFP32Out<cutlass::float_e4m3_t, AlignmentFP8,
    MmaTileShape_128x64, ClusterShape_1x1>;
using ChainFP8_T128x64_C1x1_Auto_FP32Out = KernelTypeChainFP32Out<cutlass::float_e4m3_t, AlignmentFP8,
    MmaTileShape_128x64, ClusterShape_1x1, UseAutoCarveout>;

// FP8_T128x128_C1x1 — GEMM-optimal
using ChainFP8_T128x128_C1x1 = KernelTypeChain<cutlass::float_e4m3_t, AlignmentFP8,
    MmaTileShape_128x128, ClusterShape_1x1>;
using ChainFP8_T128x128_C1x1_Auto = KernelTypeChain<cutlass::float_e4m3_t, AlignmentFP8,
    MmaTileShape_128x128, ClusterShape_1x1, UseAutoCarveout>;
using ChainFP8_T128x128_C1x1_FP32Out = KernelTypeChainFP32Out<cutlass::float_e4m3_t, AlignmentFP8,
    MmaTileShape_128x128, ClusterShape_1x1>;
using ChainFP8_T128x128_C1x1_Auto_FP32Out = KernelTypeChainFP32Out<cutlass::float_e4m3_t, AlignmentFP8,
    MmaTileShape_128x128, ClusterShape_1x1, UseAutoCarveout>;

// FP8_T128x256_C1x1 — Wide-N (SM100 only: auto-carveout computes 1 stage on SM120 → static assert)
#ifndef COMPLEX_FP8_SM100_TARGET_SM120
using ChainFP8_T128x256_C1x1 = KernelTypeChain<cutlass::float_e4m3_t, AlignmentFP8,
    MmaTileShape_128x256, ClusterShape_1x1>;
using ChainFP8_T128x256_C1x1_Auto = KernelTypeChain<cutlass::float_e4m3_t, AlignmentFP8,
    MmaTileShape_128x256, ClusterShape_1x1, UseAutoCarveout>;
using ChainFP8_T128x256_C1x1_FP32Out = KernelTypeChainFP32Out<cutlass::float_e4m3_t, AlignmentFP8,
    MmaTileShape_128x256, ClusterShape_1x1>;
using ChainFP8_T128x256_C1x1_Auto_FP32Out = KernelTypeChainFP32Out<cutlass::float_e4m3_t, AlignmentFP8,
    MmaTileShape_128x256, ClusterShape_1x1, UseAutoCarveout>;
#endif

// ========================================================================================
// FP8 cluster variants (SM100 only — SM120 limited to 1×1)
// ========================================================================================

#ifndef COMPLEX_FP8_SM100_TARGET_SM120

// FP8_T128x128_C1x2 — 1×2 cluster (TMA N-multicast)
using ChainFP8_T128x128_C1x2 = KernelTypeChain<cutlass::float_e4m3_t, AlignmentFP8,
    MmaTileShape_128x128, ClusterShape_1x2>;
using ChainFP8_T128x128_C1x2_FP32Out = KernelTypeChainFP32Out<cutlass::float_e4m3_t, AlignmentFP8,
    MmaTileShape_128x128, ClusterShape_1x2>;

// FP8_T128x128_C2x2 — 2×2 cluster
using ChainFP8_T128x128_C2x2 = KernelTypeChain<cutlass::float_e4m3_t, AlignmentFP8,
    MmaTileShape_128x128, ClusterShape_2x2>;
using ChainFP8_T128x128_C2x2_FP32Out = KernelTypeChainFP32Out<cutlass::float_e4m3_t, AlignmentFP8,
    MmaTileShape_128x128, ClusterShape_2x2>;

// FP8_T128x256_C1x2 — Wide-N + 1×2 cluster
using ChainFP8_T128x256_C1x2 = KernelTypeChain<cutlass::float_e4m3_t, AlignmentFP8,
    MmaTileShape_128x256, ClusterShape_1x2>;
using ChainFP8_T128x256_C1x2_FP32Out = KernelTypeChainFP32Out<cutlass::float_e4m3_t, AlignmentFP8,
    MmaTileShape_128x256, ClusterShape_1x2>;

// Grouped GEMM chains for multi-config triangle decomposition
using GroupedChainFP8_T128x128_C1x1 = GroupedKernelTypeChain<cutlass::float_e4m3_t, AlignmentFP8,
    MmaTileShape_128x128, ClusterShape_1x1>;
using GroupedChainFP8_T128x64_C1x1 = GroupedKernelTypeChain<cutlass::float_e4m3_t, AlignmentFP8,
    MmaTileShape_128x64, ClusterShape_1x1>;

// ----- FP6: Cluster variants (SM100 only) -----
#ifdef COMPLEX_SM100_ENABLE_FP6
using ChainFP6_E3M2_C1x2 = BlockScaledKernelTypeChain<
    cutlass::mx_float6_t<cutlass::float_e3m2_t>, AlignmentFP6, MmaTileShapeFP6, ClusterShape_1x2>;
using ChainFP6_E3M2_C2x2 = BlockScaledKernelTypeChain<
    cutlass::mx_float6_t<cutlass::float_e3m2_t>, AlignmentFP6, MmaTileShapeFP6, ClusterShape_2x2>;
using ChainFP6_E2M3_C1x2 = BlockScaledKernelTypeChain<
    cutlass::mx_float6_t<cutlass::float_e2m3_t>, AlignmentFP6, MmaTileShapeFP6, ClusterShape_1x2>;
using ChainFP6_E2M3_C2x2 = BlockScaledKernelTypeChain<
    cutlass::mx_float6_t<cutlass::float_e2m3_t>, AlignmentFP6, MmaTileShapeFP6, ClusterShape_2x2>;
#endif

// ----- FP4: Cluster variants (SM100 only) -----
#ifdef COMPLEX_SM100_ENABLE_FP4
using ChainFP4_C1x2 = BlockScaledKernelTypeChain<
    cutlass::mx_float4_t<cutlass::float_e2m1_t>, AlignmentFP4, MmaTileShapeFP4, ClusterShape_1x2>;
using ChainFP4_C2x2 = BlockScaledKernelTypeChain<
    cutlass::mx_float4_t<cutlass::float_e2m1_t>, AlignmentFP4, MmaTileShapeFP4, ClusterShape_2x2>;
#endif

#endif // !COMPLEX_FP8_SM100_TARGET_SM120

// ========================================================================================
// Backward-compatible chain aliases
// ========================================================================================
//
// ChainFP8_128x256, ChainFP8_C1x2, ChainFP8_C2x2 map to the new naming convention.
// Old dispatch code (COMPLEX_SM100_MULTI_CONFIG) used these names.

#ifndef COMPLEX_FP8_SM100_TARGET_SM120
using ChainFP8_128x256 = ChainFP8_T128x256_C1x1;
using ChainFP8_C1x2    = ChainFP8_T128x128_C1x2;
using ChainFP8_C2x2    = ChainFP8_T128x128_C2x2;
#endif

} // namespace gemm_complex_sm100
