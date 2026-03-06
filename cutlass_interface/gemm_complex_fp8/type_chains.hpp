// ========================================================================================

/*
 * This defines a single real-valued FP8 GEMM kernel using CUTLASS 3.x's builder pattern.
 * We instantiate this kernel and call it 4 times for the complex decomposition.
 */

// ----- Type Configuration -----

/// FP8 E4M3 as compute type — this is what the tensor cores see.
using ElementA_FP8 = cutlass::float_e4m3_t;
using ElementB_FP8 = cutlass::float_e4m3_t;

/// FP32 accumulator — mandatory for FP8 inputs on Hopper.
/// The tensor core MMA accumulates into FP32 registers regardless; using FP16 here
/// would require an extra truncation that CUTLASS doesn't optimize away.
using ElementAccumulator = float;

/// Output is FP16 to match the caller's planar complex buffers.
using ElementC = cutlass::half_t;
using ElementD = cutlass::half_t;

// ----- Layout Configuration -----

/// A is Row-Major, B is Column-Major → "TN" GEMM format.
///
/// Why TN for FP8 on Hopper?
///   - wgmma.mma_async instructions expect A in row-major (M×K) and B in column-major (K×N)
///     layout in shared memory. TMA loads these directly without any shared memory transpose.
///   - For the Hermitian case (B^T), since B is stored column-major, B^T is effectively
///     row-major. We handle this by swapping the layout tag to RowMajor when configuring
///     the transposed sub-GEMMs, avoiding any physical data movement.
using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::ColumnMajor;
using LayoutC = cutlass::layout::RowMajor;
using LayoutD = cutlass::layout::RowMajor;

// CuTe stride types for the 3.x API (rank-2 strides for M×K, N×K, M×N matrices)
using StrideA = cutlass::gemm::TagToStrideA_t<LayoutA>;
using StrideB = cutlass::gemm::TagToStrideB_t<LayoutB>;
using StrideC = cutlass::gemm::TagToStrideC_t<LayoutC>;
using StrideD = cutlass::gemm::TagToStrideC_t<LayoutD>;

// ----- Tile Shape Configuration -----

/// Thread block tile shape (M × N × K), configurable at compile time.
///
/// Override at build time with:
///   -DCOMPLEX_FP8_TILE_M=128 -DCOMPLEX_FP8_TILE_N=256 -DCOMPLEX_FP8_TILE_K=128
///
/// Default: 128×256×128, selected by benchmark_configs sweep on GH200:
///
///   | TileShape      | TFLOPS (4096³) | Notes                               |
///   |----------------|----------------|-------------------------------------|
///   | 128×256×128    | 940            | WINNER — best compute/load ratio    |
///   | 256×128×128    | 897            | Good when M >> N                    |
///   | 128×128×128    | 867            | Previous default, safe fallback     |
///
/// WHY 128×256×128 WINS:
///   Doubling N from 128→256 means each CTA computes 128×256 = 32K output elements
///   per K-step, but the A-tile is still only 128×128 = 16KB. The B-tile grows to
///   128×256 = 32KB, but B is loaded via TMA which hides the extra latency. Net effect:
///   A-tile reuse doubles (each A row multiplies 256 B columns instead of 128),
///   cutting A's memory traffic contribution in half. The larger tile also reduces
///   the total number of output tiles (M/128 × N/256 vs M/128 × N/128), improving
///   wave efficiency on the SM grid.
///
///   K=128 remains optimal: 4 wgmma k-iterations per K-tile provides enough ILP
///   to hide the 30-cycle wgmma latency, and keeps per-stage SMEM at 48KB
///   (16KB A + 32KB B), fitting ~4-5 pipeline stages in SM90's 228KB SMEM.
///
#ifndef COMPLEX_FP8_TILE_M
#define COMPLEX_FP8_TILE_M 128
#endif
#ifndef COMPLEX_FP8_TILE_N
#define COMPLEX_FP8_TILE_N 256
#endif
#ifndef COMPLEX_FP8_TILE_K
#define COMPLEX_FP8_TILE_K 128
#endif

using TileShape = cute::Shape<
    cute::Int<COMPLEX_FP8_TILE_M>,
    cute::Int<COMPLEX_FP8_TILE_N>,
    cute::Int<COMPLEX_FP8_TILE_K>>;

/// Cluster shape: configurable at compile time.
///
/// Override: -DCOMPLEX_FP8_CLUSTER_M=1 -DCOMPLEX_FP8_CLUSTER_N=1
///
/// Default: 1×1×1. Benchmark showed clusters (2×1, 1×2) did not help at 4096³
/// due to insufficient tile count for TMA multicast amortization. At 8192³+
/// with many output tiles, try 1×2×1 for B-tile multicast across N dimension.
///
#ifndef COMPLEX_FP8_CLUSTER_M
#define COMPLEX_FP8_CLUSTER_M 1
#endif
#ifndef COMPLEX_FP8_CLUSTER_N
#define COMPLEX_FP8_CLUSTER_N 1
#endif

using ClusterShape = cute::Shape<
    cute::Int<COMPLEX_FP8_CLUSTER_M>,
    cute::Int<COMPLEX_FP8_CLUSTER_N>,
    cute::_1>;

/// Kernel schedule selection.
///
/// Override: -DCOMPLEX_FP8_USE_PINGPONG=1 or -DCOMPLEX_FP8_USE_FAST_ACCUM=0
///
/// Benchmark results on GH200 at 4096³ (single real FP8 GEMM):
///
///   | Schedule                     | TFLOPS | Notes                          |
///   |------------------------------|--------|--------------------------------|
///   | CooperativeFP8FastAccum      | 940    | Best at 4096³ with 128×256     |
///   | Cooperative (non-FastAccum)  | 871    | Better accuracy, ~8% slower    |
///   | PingpongFP8FastAccum         | 795    | Needs 8192³+ to overtake Coop  |
///   | Auto                         | 867    | Builder picks Coop non-Fast    |
///
/// CooperativeFP8FastAccum: 2 consumer warp groups cooperate on the SAME C tile.
///   FastAccum skips periodic FP32 promotion of the FP8 accumulator, trading
///   slight accuracy loss for higher throughput. Good for inference.
///
/// PingpongFP8FastAccum: 2 consumers work on SEPARATE C tiles in ping-pong
///   fashion — one does MMA while the other epilogues. Deepest pipeline,
///   highest theoretical throughput, but needs large tile counts to fill.
///   Expected to win at 8192³+ where there are enough tiles.
///
/// Default: CooperativeFP8FastAccum (best measured throughput).
///   Define COMPLEX_FP8_USE_PINGPONG=1 for Pingpong at large problem sizes.
///   Define COMPLEX_FP8_USE_FAST_ACCUM=0 for better accuracy (training).
///

#if defined(COMPLEX_FP8_USE_PINGPONG) && COMPLEX_FP8_USE_PINGPONG
using KernelSchedule = cutlass::gemm::KernelTmaWarpSpecializedPingpongFP8FastAccum;
#elif defined(COMPLEX_FP8_USE_FAST_ACCUM) && !COMPLEX_FP8_USE_FAST_ACCUM
using KernelSchedule = cutlass::gemm::KernelTmaWarpSpecializedCooperative;
#else
using KernelSchedule = cutlass::gemm::KernelTmaWarpSpecializedCooperativeFP8FastAccum;
#endif

// ----- Grouped GEMM schedules (PtrArray variants for triangle decomposition) -----
//
// Mirror the standard schedule selection but use PtrArray kernel/epilogue schedules
// required for grouped GEMM (multiple independent sub-GEMMs in a single kernel launch).
//
#if defined(CUTLASS_ARCH_MMA_MODIFIABLE_TMA_SM90_SUPPORTED)
#if defined(COMPLEX_FP8_USE_PINGPONG) && COMPLEX_FP8_USE_PINGPONG
using GroupedKernelSchedule = cutlass::gemm::KernelPtrArrayTmaWarpSpecializedPingpongFP8FastAccum;
using GroupedEpilogueSchedule = cutlass::epilogue::PtrArrayTmaWarpSpecializedPingpong;
#elif defined(COMPLEX_FP8_USE_FAST_ACCUM) && !COMPLEX_FP8_USE_FAST_ACCUM
using GroupedKernelSchedule = cutlass::gemm::KernelPtrArrayTmaWarpSpecializedCooperative;
using GroupedEpilogueSchedule = cutlass::epilogue::PtrArrayTmaWarpSpecializedCooperative;
#else
using GroupedKernelSchedule = cutlass::gemm::KernelPtrArrayTmaWarpSpecializedCooperativeFP8FastAccum;
using GroupedEpilogueSchedule = cutlass::epilogue::PtrArrayTmaWarpSpecializedCooperative;
#endif
#endif // CUTLASS_ARCH_MMA_MODIFIABLE_TMA_SM90_SUPPORTED

// ----- Build the Collective Mainloop (handles TMA loads + wgmma MMA) -----

/// Alignment: 128-bit (16 bytes) access width.
/// FP8 = 1 byte → 16 elements; FP16 = 2 bytes → 8 elements.
static constexpr int AlignmentFP8  = 16;  // 16 × 1B = 128-bit
static constexpr int AlignmentFP16 = 8;   // 8 × 2B  = 128-bit

using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm90,                     // Target architecture
    cutlass::arch::OpClassTensorOp,          // Use tensor core operations
    TileShape,                                // Must match mainloop tile shape
    ClusterShape,                             // Must match mainloop cluster shape
    cutlass::epilogue::collective::EpilogueTileAuto,  // Auto-select epilogue tile
    ElementAccumulator,                       // Accumulator type (FP32)
    ElementAccumulator,                       // Compute type for epilogue (FP32 alpha/beta)
    ElementC, LayoutC, AlignmentFP16,         // C matrix: FP16, 8-element alignment
    ElementD, LayoutD, AlignmentFP16,         // D matrix: FP16, 8-element alignment
    cutlass::epilogue::collective::EpilogueScheduleAuto  // Auto-select schedule
>::CollectiveOp;

using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm90,                     // SM90 Hopper architecture
    cutlass::arch::OpClassTensorOp,          // Tensor core operation class
    ElementA_FP8, LayoutA, AlignmentFP8,      // A matrix: FP8 E4M3, Row-Major, 16-elem align
    ElementB_FP8, LayoutB, AlignmentFP8,      // B matrix: FP8 E4M3, Column-Major, 16-elem align
    ElementAccumulator,                       // FP32 accumulator (mandatory for FP8)
    TileShape,                                // Tile shape (default 128×256×128)
    ClusterShape,                             // Cluster shape (default 1×1×1)
    cutlass::gemm::collective::StageCountAutoCarveout<
        static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))
    >,                                        // Auto pipeline stages after epilogue SMEM carveout
    KernelSchedule                            // Explicit schedule (default CooperativeFP8FastAccum)
>::CollectiveOp;

// ----- Assemble the Full Kernel -----

using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    cute::Shape<int, int, int, int>,          // ProblemShape: <M, N, K, L> (L=batch)
    CollectiveMainloop,
    CollectiveEpilogue
    // TileScheduler defaults to void → uses static persistent tile scheduler
>;

/// Device-level GEMM adapter — provides initialize() + run() interface
using DeviceGemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

// Argument types
using GemmArguments = typename DeviceGemm::Arguments;

// ----- FP32 Output Type Chain (avoids FP16 overflow for large K) -----
//
// Identical to the FP16 chain except C/D are float.
// FP32 output avoids overflow when K * max_A * max_B > 65504.

static constexpr int AlignmentFP32 = 4;  // 4 × 4B = 128-bit

using FusionOperationFP32Out = cutlass::epilogue::fusion::LinearCombination<
    float, ElementAccumulator, float, ElementAccumulator,
    cutlass::FloatRoundStyle::round_to_nearest>;

using CollectiveEpilogueFP32Out = typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
    TileShape, ClusterShape,
    cutlass::epilogue::collective::EpilogueTileAuto,
    ElementAccumulator, ElementAccumulator,
    float, LayoutC, AlignmentFP32,
    float, LayoutD, AlignmentFP32,
    cutlass::epilogue::collective::EpilogueScheduleAuto,
    FusionOperationFP32Out
>::CollectiveOp;

using CollectiveMainloopFP32Out = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
    ElementA_FP8, LayoutA, AlignmentFP8,
    ElementB_FP8, LayoutB, AlignmentFP8,
    ElementAccumulator, TileShape, ClusterShape,
    cutlass::gemm::collective::StageCountAutoCarveout<
        static_cast<int>(sizeof(typename CollectiveEpilogueFP32Out::SharedStorage))>,
    KernelSchedule
>::CollectiveOp;

using GemmKernelFP32Out = cutlass::gemm::kernel::GemmUniversal<
    cute::Shape<int, int, int, int>, CollectiveMainloopFP32Out, CollectiveEpilogueFP32Out>;
using DeviceGemmFP32Out = cutlass::gemm::device::GemmUniversalAdapter<GemmKernelFP32Out>;
using GemmArgumentsFP32Out = typename DeviceGemmFP32Out::Arguments;

// ========================================================================================
// SM90 Multi-Config Chain Templates
// ========================================================================================
//
// Parameterized chain templates for runtime-selectable tile shape, cluster shape,
// and kernel schedule. Each instantiation yields a complete CUTLASS kernel type chain.
// The existing non-template types above remain as backward-compatible aliases for
// the compile-time-configured default.
//

// ----- Named tile shapes (always available) -----
using TileShape_128x256 = cute::Shape<cute::_128, cute::_256, cute::_128>;
using TileShape_128x128 = cute::Shape<cute::_128, cute::_128, cute::_128>;
using TileShape_64x128  = cute::Shape<cute::_64,  cute::_128, cute::_128>;

// ----- Named cluster shapes -----
using ClusterShape_1x1 = cute::Shape<cute::_1, cute::_1, cute::_1>;
using ClusterShape_1x2 = cute::Shape<cute::_1, cute::_2, cute::_1>;

// ----- Named kernel schedules -----
using SchedCoopFP8  = cutlass::gemm::KernelTmaWarpSpecializedCooperativeFP8FastAccum;
using SchedPP_FP8   = cutlass::gemm::KernelTmaWarpSpecializedPingpongFP8FastAccum;
using SchedBasicFP8 = cutlass::gemm::KernelTmaWarpSpecializedFP8FastAccum;  // no M>=128 constraint

/// SM90 FP8 chain template (FP16 output). Parameterized on tile, cluster, schedule.
/// Provides the same interface as SM100's KernelTypeChain for generic dispatch.
template <typename TileShape_, typename ClusterShape_, typename KernelSchedule_>
struct SM90FP8Chain {
    using TileShape = TileShape_;
    using ClusterShapeType = ClusterShape_;
    using ElementA = cutlass::float_e4m3_t;
    using ElementB = cutlass::float_e4m3_t;

    using CollectiveEpilogue_ = typename cutlass::epilogue::collective::CollectiveBuilder<
        cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
        TileShape_, ClusterShape_,
        cutlass::epilogue::collective::EpilogueTileAuto,
        ElementAccumulator, ElementAccumulator,
        ElementC, LayoutC, AlignmentFP16,
        ElementD, LayoutD, AlignmentFP16,
        cutlass::epilogue::collective::EpilogueScheduleAuto
    >::CollectiveOp;

    using CollectiveMainloop_ = typename cutlass::gemm::collective::CollectiveBuilder<
        cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
        ElementA_FP8, LayoutA, AlignmentFP8,
        ElementB_FP8, LayoutB, AlignmentFP8,
        ElementAccumulator, TileShape_, ClusterShape_,
        cutlass::gemm::collective::StageCountAutoCarveout<
            static_cast<int>(sizeof(typename CollectiveEpilogue_::SharedStorage))>,
        KernelSchedule_
    >::CollectiveOp;

    using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
        cute::Shape<int, int, int, int>, CollectiveMainloop_, CollectiveEpilogue_>;
    using DeviceGemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
    using GemmArguments = typename DeviceGemm::Arguments;
};

/// SM90 FP8 chain template (FP32 output). Same structure with float C/D.
template <typename TileShape_, typename ClusterShape_, typename KernelSchedule_>
struct SM90FP8ChainFP32Out {
    using TileShape = TileShape_;
    using ClusterShapeType = ClusterShape_;
    using ElementA = cutlass::float_e4m3_t;
    using ElementB = cutlass::float_e4m3_t;

    using FusionOp_ = cutlass::epilogue::fusion::LinearCombination<
        float, ElementAccumulator, float, ElementAccumulator,
        cutlass::FloatRoundStyle::round_to_nearest>;

    using CollectiveEpilogue_ = typename cutlass::epilogue::collective::CollectiveBuilder<
        cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
        TileShape_, ClusterShape_,
        cutlass::epilogue::collective::EpilogueTileAuto,
        ElementAccumulator, ElementAccumulator,
        float, LayoutC, AlignmentFP32,
        float, LayoutD, AlignmentFP32,
        cutlass::epilogue::collective::EpilogueScheduleAuto,
        FusionOp_
    >::CollectiveOp;

    using CollectiveMainloop_ = typename cutlass::gemm::collective::CollectiveBuilder<
        cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
        ElementA_FP8, LayoutA, AlignmentFP8,
        ElementB_FP8, LayoutB, AlignmentFP8,
        ElementAccumulator, TileShape_, ClusterShape_,
        cutlass::gemm::collective::StageCountAutoCarveout<
            static_cast<int>(sizeof(typename CollectiveEpilogue_::SharedStorage))>,
        KernelSchedule_
    >::CollectiveOp;

    using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
        cute::Shape<int, int, int, int>, CollectiveMainloop_, CollectiveEpilogue_>;
    using DeviceGemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
    using GemmArguments = typename DeviceGemm::Arguments;
};

// ----- Chain aliases for all SM90 GemmConfig values -----
// FP16 output
using ChainFP8_T128x256_C1x1_Coop = SM90FP8Chain<TileShape_128x256, ClusterShape_1x1, SchedCoopFP8>;
using ChainFP8_T128x128_C1x1_Coop = SM90FP8Chain<TileShape_128x128, ClusterShape_1x1, SchedCoopFP8>;
using ChainFP8_T128x256_C1x1_PP   = SM90FP8Chain<TileShape_128x256, ClusterShape_1x1, SchedPP_FP8>;
using ChainFP8_T128x128_C1x1_PP   = SM90FP8Chain<TileShape_128x128, ClusterShape_1x1, SchedPP_FP8>;
using ChainFP8_T128x256_C1x2_Coop = SM90FP8Chain<TileShape_128x256, ClusterShape_1x2, SchedCoopFP8>;
using ChainFP8_T64x128_C1x1_Coop  = SM90FP8Chain<TileShape_64x128,  ClusterShape_1x1, SchedBasicFP8>;  // Basic schedule (Coop requires M>=128)

// FP32 output
using ChainFP8_T128x256_C1x1_Coop_FP32Out = SM90FP8ChainFP32Out<TileShape_128x256, ClusterShape_1x1, SchedCoopFP8>;
using ChainFP8_T128x128_C1x1_Coop_FP32Out = SM90FP8ChainFP32Out<TileShape_128x128, ClusterShape_1x1, SchedCoopFP8>;
using ChainFP8_T128x256_C1x1_PP_FP32Out   = SM90FP8ChainFP32Out<TileShape_128x256, ClusterShape_1x1, SchedPP_FP8>;
using ChainFP8_T128x128_C1x1_PP_FP32Out   = SM90FP8ChainFP32Out<TileShape_128x128, ClusterShape_1x1, SchedPP_FP8>;
using ChainFP8_T128x256_C1x2_Coop_FP32Out = SM90FP8ChainFP32Out<TileShape_128x256, ClusterShape_1x2, SchedCoopFP8>;
using ChainFP8_T64x128_C1x1_Coop_FP32Out  = SM90FP8ChainFP32Out<TileShape_64x128,  ClusterShape_1x1, SchedBasicFP8>;  // Basic schedule (Coop requires M>=128)


// ----- Grouped GEMM Type Chain (SM90 triangle decomposition) -----
//
// Packs multiple independent sub-GEMMs into a single kernel launch using
// CUTLASS Grouped GEMM (PtrArray schedules). Used by run_real_gemm_lower_triangle_grouped().
//
// Key differences from standard types:
//   - ProblemShape: GroupProblemShape<Shape<int,int,int>> (rank-3, no batch dim)
//   - Schedules: PtrArray variants (KernelPtrArrayTmaWarpSpecialized{Cooperative,Pingpong}...)
//   - Layouts: pointer-decorated (LayoutA *, LayoutB *, LayoutC *, LayoutD *)
//   - Explicit LinearCombination fusion operation (required for PtrArray epilogue)
//
#if defined(CUTLASS_ARCH_MMA_MODIFIABLE_TMA_SM90_SUPPORTED)

using GroupedFusionOperation = cutlass::epilogue::fusion::LinearCombination<
    ElementD,                  // Output element type (FP16)
    ElementAccumulator,        // Compute precision (FP32)
    ElementC,                  // Source element type (FP16)
    ElementAccumulator,        // Scalar type for alpha/beta (FP32)
    cutlass::FloatRoundStyle::round_to_nearest
>;

using GroupedCollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
    TileShape, ClusterShape,
    cutlass::epilogue::collective::EpilogueTileAuto,
    ElementAccumulator, ElementAccumulator,
    ElementC, LayoutC *, AlignmentFP16,
    ElementD, LayoutD *, AlignmentFP16,
    GroupedEpilogueSchedule,
    GroupedFusionOperation
>::CollectiveOp;

using GroupedCollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
    ElementA_FP8, LayoutA *, AlignmentFP8,
    ElementB_FP8, LayoutB *, AlignmentFP8,
    ElementAccumulator,
    TileShape, ClusterShape,
    cutlass::gemm::collective::StageCountAutoCarveout<
        static_cast<int>(sizeof(typename GroupedCollectiveEpilogue::SharedStorage))
    >,
    GroupedKernelSchedule
>::CollectiveOp;

using GroupedGemmKernel = cutlass::gemm::kernel::GemmUniversal<
    cutlass::gemm::GroupProblemShape<cute::Shape<int, int, int>>,
    GroupedCollectiveMainloop,
    GroupedCollectiveEpilogue
>;

using GroupedDeviceGemm = cutlass::gemm::device::GemmUniversalAdapter<GroupedGemmKernel>;
using GroupedGemmArguments = typename GroupedDeviceGemm::Arguments;

// Stride types from the kernel (needed for device arrays in grouped launch)
using GroupedInternalStrideA = typename GroupedGemmKernel::InternalStrideA;
using GroupedInternalStrideB = typename GroupedGemmKernel::InternalStrideB;
using GroupedInternalStrideC = typename GroupedGemmKernel::InternalStrideC;
using GroupedInternalStrideD = typename GroupedGemmKernel::InternalStrideD;

#endif // CUTLASS_ARCH_MMA_MODIFIABLE_TMA_SM90_SUPPORTED


// ========================================================================================
// NOTE: Previous versions defined a separate DeviceGemm_BT with LayoutB=RowMajor for
// Hermitian B^T. This triggered the RS (Register-Shared) mainloop on SM90, which is
// significantly slower than the SS (Shared-Shared) path used for ColumnMajor B.
//
// New approach: physically transpose B into ColumnMajor scratch buffers using a lightweight
// kernel, then run ALL sub-GEMMs through the fast SS DeviceGemm. The transpose cost is
// O(K×N) per buffer — negligible vs the O(M×N×K) GEMM at any reasonable problem size.
// ========================================================================================


// ========================================================================================
// Epilogue Accumulation Kernel (for combining sub-GEMM results)
// ========================================================================================

/*
 * After the 4 sub-GEMMs, we need to combine results:
 *   Re(C) = α·(sub1 − sub2) + β·Re(C_old)    or    α·(sub1 + sub2) + β·Re(C_old)
 *   Im(C) = α·(sub3 + sub4) + β·Im(C_old)    or    α·(sub3 − sub4) + β·Im(C_old)
 *
 * We handle this by using the CUTLASS epilogue's built-in α/β scaling:
 *   - First sub-GEMM: α=+α, β=β   (initializes with β·C_old)
 *   - Second sub-GEMM: α=−α, β=1.0 (accumulates with sign flip, preserving previous result)
 *                    or α=+α, β=1.0 (for terms that add)
 *
 * This avoids a separate combination kernel entirely! The epilogue fusion handles it.
 */

