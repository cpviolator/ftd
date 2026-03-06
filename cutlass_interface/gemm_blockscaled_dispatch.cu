/*
 * gemm_blockscaled_dispatch.cu
 *
 * Concrete (non-template) implementations of block-scaled GEMM dispatch functions
 * for FP6 and FP4 precision.
 *
 * WHY THIS FILE EXISTS:
 * CUTLASS issue #2478 — nvcc fails to generate proper device stubs for
 * cutlass::device_kernel<GemmKernel> when the GemmKernel type is deeply nested
 * (as in BlockScaledKernelTypeChain). The template specialization in nvcc's
 * generated .stub.c file fails to match the template declaration.
 *
 * The fix is to define CONCRETE (non-template) __global__ kernel functions that
 * do the same thing as device_kernel<GemmKernel> but without template indirection.
 * We then bypass GemmUniversal::initialize()/run() (which internally reference
 * device_kernel<GemmKernel>) and manually:
 *   1. Initialize workspace via GemmKernel::initialize_workspace()
 *   2. Convert arguments via GemmKernel::to_underlying_arguments()
 *   3. Call cudaFuncSetAttribute() on our concrete kernel
 *   4. Launch our concrete kernel via cudaLaunchKernel()
 *
 * The FP8 path uses a simpler type chain (OpClassTensorOp) and is unaffected.
 * It remains using GemmUniversal::initialize()/run() in the header.
 */

#include "gemm_sm100_type_chains.hpp"
#include "gemm_blockscaled_dispatch.h"
#include "cutlass/arch/synclog.hpp"

namespace gemm_complex_sm100 {

// ============================================================================
// Concrete __global__ kernel functions — one per precision × config
// ============================================================================
//
// These are functionally identical to cutlass::device_kernel<GemmKernel> but
// defined as concrete (non-template) functions. nvcc generates proper device
// stubs for these without the template matching issues.
//

#ifdef COMPLEX_SM100_ENABLE_FP6

using FP6E3M2Kernel = ChainFP6_E3M2::GemmKernel;

__global__
__launch_bounds__(FP6E3M2Kernel::MaxThreadsPerBlock, FP6E3M2Kernel::MinBlocksPerMultiprocessor)
void blockscaled_kernel_fp6_e3m2(CUTLASS_GRID_CONSTANT FP6E3M2Kernel::Params const params)
{
    extern __shared__ char smem[];
    FP6E3M2Kernel op;
    op(params, smem);
    cutlass::arch::synclog_print();
}

using FP6E2M3Kernel = ChainFP6_E2M3::GemmKernel;

__global__
__launch_bounds__(FP6E2M3Kernel::MaxThreadsPerBlock, FP6E2M3Kernel::MinBlocksPerMultiprocessor)
void blockscaled_kernel_fp6_e2m3(CUTLASS_GRID_CONSTANT FP6E2M3Kernel::Params const params)
{
    extern __shared__ char smem[];
    FP6E2M3Kernel op;
    op(params, smem);
    cutlass::arch::synclog_print();
}

// ----- FP6 Cluster variants (SM100 only — SM120 limited to 1x1 cluster) -----
#ifndef COMPLEX_FP8_SM100_TARGET_SM120

// ----- FP6 E3M2: Cluster 1x2 -----
using FP6E3M2Kernel_C1x2 = ChainFP6_E3M2_C1x2::GemmKernel;

__global__
__launch_bounds__(FP6E3M2Kernel_C1x2::MaxThreadsPerBlock, FP6E3M2Kernel_C1x2::MinBlocksPerMultiprocessor)
void blockscaled_kernel_fp6_e3m2_c1x2(CUTLASS_GRID_CONSTANT FP6E3M2Kernel_C1x2::Params const params)
{
    extern __shared__ char smem[];
    FP6E3M2Kernel_C1x2 op;
    op(params, smem);
    cutlass::arch::synclog_print();
}

// ----- FP6 E3M2: Cluster 2x2 -----
using FP6E3M2Kernel_C2x2 = ChainFP6_E3M2_C2x2::GemmKernel;

__global__
__launch_bounds__(FP6E3M2Kernel_C2x2::MaxThreadsPerBlock, FP6E3M2Kernel_C2x2::MinBlocksPerMultiprocessor)
void blockscaled_kernel_fp6_e3m2_c2x2(CUTLASS_GRID_CONSTANT FP6E3M2Kernel_C2x2::Params const params)
{
    extern __shared__ char smem[];
    FP6E3M2Kernel_C2x2 op;
    op(params, smem);
    cutlass::arch::synclog_print();
}

// ----- FP6 E2M3: Cluster 1x2 -----
using FP6E2M3Kernel_C1x2 = ChainFP6_E2M3_C1x2::GemmKernel;

__global__
__launch_bounds__(FP6E2M3Kernel_C1x2::MaxThreadsPerBlock, FP6E2M3Kernel_C1x2::MinBlocksPerMultiprocessor)
void blockscaled_kernel_fp6_e2m3_c1x2(CUTLASS_GRID_CONSTANT FP6E2M3Kernel_C1x2::Params const params)
{
    extern __shared__ char smem[];
    FP6E2M3Kernel_C1x2 op;
    op(params, smem);
    cutlass::arch::synclog_print();
}

// ----- FP6 E2M3: Cluster 2x2 -----
using FP6E2M3Kernel_C2x2 = ChainFP6_E2M3_C2x2::GemmKernel;

__global__
__launch_bounds__(FP6E2M3Kernel_C2x2::MaxThreadsPerBlock, FP6E2M3Kernel_C2x2::MinBlocksPerMultiprocessor)
void blockscaled_kernel_fp6_e2m3_c2x2(CUTLASS_GRID_CONSTANT FP6E2M3Kernel_C2x2::Params const params)
{
    extern __shared__ char smem[];
    FP6E2M3Kernel_C2x2 op;
    op(params, smem);
    cutlass::arch::synclog_print();
}

#endif // !COMPLEX_FP8_SM100_TARGET_SM120

#endif // COMPLEX_SM100_ENABLE_FP6


#ifdef COMPLEX_SM100_ENABLE_FP4

using FP4Kernel = ChainFP4::GemmKernel;

__global__
__launch_bounds__(FP4Kernel::MaxThreadsPerBlock, FP4Kernel::MinBlocksPerMultiprocessor)
void blockscaled_kernel_fp4(CUTLASS_GRID_CONSTANT FP4Kernel::Params const params)
{
    extern __shared__ char smem[];
    FP4Kernel op;
    op(params, smem);
    cutlass::arch::synclog_print();
}

// ----- FP4 Cluster variants (SM100 only) -----
#ifndef COMPLEX_FP8_SM100_TARGET_SM120

// ----- FP4: Cluster 1x2 -----
using FP4Kernel_C1x2 = ChainFP4_C1x2::GemmKernel;

__global__
__launch_bounds__(FP4Kernel_C1x2::MaxThreadsPerBlock, FP4Kernel_C1x2::MinBlocksPerMultiprocessor)
void blockscaled_kernel_fp4_c1x2(CUTLASS_GRID_CONSTANT FP4Kernel_C1x2::Params const params)
{
    extern __shared__ char smem[];
    FP4Kernel_C1x2 op;
    op(params, smem);
    cutlass::arch::synclog_print();
}

// ----- FP4: Cluster 2x2 -----
using FP4Kernel_C2x2 = ChainFP4_C2x2::GemmKernel;

__global__
__launch_bounds__(FP4Kernel_C2x2::MaxThreadsPerBlock, FP4Kernel_C2x2::MinBlocksPerMultiprocessor)
void blockscaled_kernel_fp4_c2x2(CUTLASS_GRID_CONSTANT FP4Kernel_C2x2::Params const params)
{
    extern __shared__ char smem[];
    FP4Kernel_C2x2 op;
    op(params, smem);
    cutlass::arch::synclog_print();
}

#endif // !COMPLEX_FP8_SM100_TARGET_SM120

#endif // COMPLEX_SM100_ENABLE_FP4


// ============================================================================
// FP32-output concrete __global__ kernel functions
// ============================================================================

#ifdef COMPLEX_SM100_ENABLE_FP6

using FP6E3M2Kernel_FP32Out = ChainFP6_E3M2_FP32Out::GemmKernel;

__global__
__launch_bounds__(FP6E3M2Kernel_FP32Out::MaxThreadsPerBlock, FP6E3M2Kernel_FP32Out::MinBlocksPerMultiprocessor)
void blockscaled_kernel_fp6_e3m2_fp32out(CUTLASS_GRID_CONSTANT FP6E3M2Kernel_FP32Out::Params const params)
{
    extern __shared__ char smem[];
    FP6E3M2Kernel_FP32Out op;
    op(params, smem);
    cutlass::arch::synclog_print();
}

#endif // COMPLEX_SM100_ENABLE_FP6

#ifdef COMPLEX_SM100_ENABLE_FP4

using FP4Kernel_FP32Out = ChainFP4_FP32Out::GemmKernel;

__global__
__launch_bounds__(FP4Kernel_FP32Out::MaxThreadsPerBlock, FP4Kernel_FP32Out::MinBlocksPerMultiprocessor)
void blockscaled_kernel_fp4_fp32out(CUTLASS_GRID_CONSTANT FP4Kernel_FP32Out::Params const params)
{
    extern __shared__ char smem[];
    FP4Kernel_FP32Out op;
    op(params, smem);
    cutlass::arch::synclog_print();
}

#endif // COMPLEX_SM100_ENABLE_FP4


// ============================================================================
// Small-M concrete __global__ kernel functions (64×128 tiles for M≤64)
// ============================================================================

#if defined(COMPLEX_SM100_ENABLE_SMALL_M) && !defined(COMPLEX_FP8_SM100_TARGET_SM120)

#ifdef COMPLEX_SM100_ENABLE_FP6

// ----- FP6 E3M2 SmallM: FP16 output -----
using FP6E3M2Kernel_SmallM = ChainFP6_E3M2_SmallM::GemmKernel;

__global__
__launch_bounds__(FP6E3M2Kernel_SmallM::MaxThreadsPerBlock, FP6E3M2Kernel_SmallM::MinBlocksPerMultiprocessor)
void blockscaled_kernel_fp6_e3m2_smallm(CUTLASS_GRID_CONSTANT FP6E3M2Kernel_SmallM::Params const params)
{
    extern __shared__ char smem[];
    FP6E3M2Kernel_SmallM op;
    op(params, smem);
    cutlass::arch::synclog_print();
}

// ----- FP6 E3M2 SmallM: FP32 output -----
using FP6E3M2Kernel_FP32Out_SmallM = ChainFP6_E3M2_FP32Out_SmallM::GemmKernel;

__global__
__launch_bounds__(FP6E3M2Kernel_FP32Out_SmallM::MaxThreadsPerBlock, FP6E3M2Kernel_FP32Out_SmallM::MinBlocksPerMultiprocessor)
void blockscaled_kernel_fp6_e3m2_fp32out_smallm(CUTLASS_GRID_CONSTANT FP6E3M2Kernel_FP32Out_SmallM::Params const params)
{
    extern __shared__ char smem[];
    FP6E3M2Kernel_FP32Out_SmallM op;
    op(params, smem);
    cutlass::arch::synclog_print();
}

#endif // COMPLEX_SM100_ENABLE_FP6

#ifdef COMPLEX_SM100_ENABLE_FP4

// ----- FP4 SmallM: FP16 output -----
using FP4Kernel_SmallM = ChainFP4_SmallM::GemmKernel;

__global__
__launch_bounds__(FP4Kernel_SmallM::MaxThreadsPerBlock, FP4Kernel_SmallM::MinBlocksPerMultiprocessor)
void blockscaled_kernel_fp4_smallm(CUTLASS_GRID_CONSTANT FP4Kernel_SmallM::Params const params)
{
    extern __shared__ char smem[];
    FP4Kernel_SmallM op;
    op(params, smem);
    cutlass::arch::synclog_print();
}

// ----- FP4 SmallM: FP32 output -----
using FP4Kernel_FP32Out_SmallM = ChainFP4_FP32Out_SmallM::GemmKernel;

__global__
__launch_bounds__(FP4Kernel_FP32Out_SmallM::MaxThreadsPerBlock, FP4Kernel_FP32Out_SmallM::MinBlocksPerMultiprocessor)
void blockscaled_kernel_fp4_fp32out_smallm(CUTLASS_GRID_CONSTANT FP4Kernel_FP32Out_SmallM::Params const params)
{
    extern __shared__ char smem[];
    FP4Kernel_FP32Out_SmallM op;
    op(params, smem);
    cutlass::arch::synclog_print();
}

#endif // COMPLEX_SM100_ENABLE_FP4

#endif // COMPLEX_SM100_ENABLE_SMALL_M


#if defined(COMPLEX_SM100_ENABLE_FP6) || defined(COMPLEX_SM100_ENABLE_FP4)

// ============================================================================
// Common implementation — manually replicates GemmUniversal::initialize()/run()
// but uses our concrete kernel instead of device_kernel<GemmKernel>
// ============================================================================

template <typename Chain>
static cutlass::Status run_blockscaled_gemm_impl(
    const void* kernel_fn,
    const void* A,
    const void* B,
    const void* SFA,
    const void* SFB,
    cutlass::half_t* C,
    int M, int N, int K,
    float alpha, float beta,
    cudaStream_t stream,
    int hw_sm_count,
    int batch_count,
    int ld_C,
    void* ext_workspace = nullptr,
    size_t ext_workspace_size = 0)
{
    static_assert(Chain::IsBlockScaled, "This function is only for BlockScaledKernelTypeChain");
    using GemmKernel = typename Chain::GemmKernel;

    auto stride_A = cutlass::make_cute_packed_stride(
        StrideA{}, cute::make_shape(M, K, batch_count));
    auto stride_B = cutlass::make_cute_packed_stride(
        StrideB{}, cute::make_shape(N, K, batch_count));
    int ld_C_eff = (ld_C > 0) ? ld_C : N;
    auto stride_C = cutlass::make_cute_packed_stride(
        StrideC{}, cute::make_shape(M, ld_C_eff, batch_count));
    auto stride_D = stride_C;

    // Compute scale factor layouts from problem shape
    auto layout_SFA = Chain::Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(
        cute::make_shape(M, N, K, batch_count));
    auto layout_SFB = Chain::Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(
        cute::make_shape(M, N, K, batch_count));

    cutlass::KernelHardwareInfo hw_info;
    hw_info.device_id = 0;
    hw_info.sm_count = hw_sm_count;

    typename Chain::GemmArguments arguments{
        cutlass::gemm::GemmUniversalMode::kGemm,
        {M, N, K, batch_count},
        {   // Mainloop arguments: data + scale factors
            static_cast<const typename Chain::RawElementType*>(A),
            stride_A,
            static_cast<const typename Chain::RawElementType*>(B),
            stride_B,
            static_cast<const typename Chain::ElementSF*>(SFA),
            layout_SFA,
            static_cast<const typename Chain::ElementSF*>(SFB),
            layout_SFB
        },
        {   // Epilogue arguments
            {alpha, beta}, C, stride_C, C, stride_D
        },
        hw_info
    };

    // --- can_implement (host-only, no kernel binary reference) ---
    {
        typename Chain::DeviceGemm gemm_check;
        auto status = gemm_check.can_implement(arguments);
        if (status != cutlass::Status::kSuccess) {
            static bool printed = false;
            if (!printed) {
                printed = true;
                fprintf(stderr, "[CUTLASS] can_implement FAILED (status=%d) for blockscaled GEMM(%d,%d,%d)\n",
                        static_cast<int>(status), M, N, K);
            }
            return status;
        }
    }

    // --- Workspace allocation ---
    size_t workspace_size = Chain::DeviceGemm::get_workspace_size(arguments);
    void* workspace = nullptr;
    bool own_workspace = false;
    if (workspace_size > 0) {
        if (ext_workspace && ext_workspace_size >= workspace_size) {
            workspace = ext_workspace;
        } else {
            cudaError_t alloc_err = cudaMallocAsync(&workspace, workspace_size, stream);
            if (alloc_err != cudaSuccess) {
                fprintf(stderr, "[CUTLASS] workspace allocation failed: %s\n",
                        cudaGetErrorString(alloc_err));
                return cutlass::Status::kErrorInternal;
            }
            own_workspace = true;
        }
    }

    // --- Initialize workspace (host-only, no kernel binary reference) ---
    auto status = GemmKernel::initialize_workspace(arguments, workspace, stream, nullptr);
    if (status != cutlass::Status::kSuccess) {
        fprintf(stderr, "[CUTLASS] initialize_workspace FAILED (status=%d)\n",
                static_cast<int>(status));
        if (own_workspace && workspace) cudaFreeAsync(workspace, stream);
        return status;
    }

    // --- Convert arguments to kernel params (host-only) ---
    auto params = GemmKernel::to_underlying_arguments(arguments, workspace);

    // --- Set dynamic shared memory on OUR concrete kernel ---
    // (This is the critical difference: we use our concrete kernel_fn,
    //  not cutlass::device_kernel<GemmKernel> which fails stub generation)
    constexpr int smem_size = GemmKernel::SharedStorageSize;

    // Print pre-launch diagnostics on first call per chain type
    static bool printed_diag = false;
    if (!printed_diag) {
        printed_diag = true;
        int device_smem_limit = 0;
        cudaDeviceGetAttribute(&device_smem_limit,
                               cudaDevAttrMaxSharedMemoryPerBlockOptin, 0);
        fprintf(stderr, "[DIAG] Block-scaled GEMM tile: %dx%dx%d  cluster: %dx%dx%d\n",
                int(cute::size<0>(typename Chain::TileShape{})),
                int(cute::size<1>(typename Chain::TileShape{})),
                int(cute::size<2>(typename Chain::TileShape{})),
                int(cute::size<0>(typename Chain::ClusterShapeType{})),
                int(cute::size<1>(typename Chain::ClusterShapeType{})),
                int(cute::size<2>(typename Chain::ClusterShapeType{})));
        fprintf(stderr, "[DIAG] Block-scaled SharedStorageSize: %d bytes (%d KB), device max: %d bytes (%d KB)\n",
                smem_size, smem_size / 1024, device_smem_limit, device_smem_limit / 1024);
    }

    if (smem_size >= (48 << 10)) {
        cudaError_t result = cudaFuncSetAttribute(
            kernel_fn,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            smem_size);
        if (cudaSuccess != result) {
            fprintf(stderr, "[CUTLASS] cudaFuncSetAttribute FAILED (err=%d/%s) for concrete kernel, smem=%d\n",
                    static_cast<int>(result), cudaGetErrorString(result), smem_size);
            result = cudaGetLastError(); // clear error bit
            if (own_workspace && workspace) cudaFreeAsync(workspace, stream);
            return cutlass::Status::kErrorInternal;
        }
    }

    // --- Launch OUR concrete kernel (bypasses device_kernel<GemmKernel>) ---
    dim3 block = GemmKernel::get_block_shape();
    dim3 grid = GemmKernel::get_grid_shape(params);

    void* kernel_args[] = {const_cast<void*>(static_cast<const void*>(&params))};
    cudaError_t launch_err = cudaLaunchKernel(kernel_fn, grid, block,
                                               kernel_args, smem_size, stream);
    if (launch_err != cudaSuccess) {
        fprintf(stderr, "[CUTLASS] cudaLaunchKernel FAILED (err=%d/%s) for blockscaled GEMM(%d,%d,%d)\n",
                static_cast<int>(launch_err), cudaGetErrorString(launch_err), M, N, K);
        if (own_workspace && workspace) cudaFreeAsync(workspace, stream);
        return cutlass::Status::kErrorInternal;
    }

    if (own_workspace && workspace) cudaFreeAsync(workspace, stream);
    return cutlass::Status::kSuccess;
}

// ============================================================================
// FP32-output implementation — same as above but float* C instead of half_t*
// ============================================================================

template <typename Chain>
static cutlass::Status run_blockscaled_gemm_impl_fp32out(
    const void* kernel_fn,
    const void* A,
    const void* B,
    const void* SFA,
    const void* SFB,
    float* C,
    int M, int N, int K,
    float alpha, float beta,
    cudaStream_t stream,
    int hw_sm_count,
    int batch_count,
    int ld_C,
    void* ext_workspace = nullptr,
    size_t ext_workspace_size = 0)
{
    static_assert(Chain::IsBlockScaled, "This function is only for BlockScaledKernelTypeChainFP32Out");
    using GemmKernel = typename Chain::GemmKernel;

    auto stride_A = cutlass::make_cute_packed_stride(
        StrideA{}, cute::make_shape(M, K, batch_count));
    auto stride_B = cutlass::make_cute_packed_stride(
        StrideB{}, cute::make_shape(N, K, batch_count));
    int ld_C_eff = (ld_C > 0) ? ld_C : N;
    auto stride_C = cutlass::make_cute_packed_stride(
        StrideC{}, cute::make_shape(M, ld_C_eff, batch_count));
    auto stride_D = stride_C;

    // Compute scale factor layouts from problem shape
    auto layout_SFA = Chain::Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(
        cute::make_shape(M, N, K, batch_count));
    auto layout_SFB = Chain::Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(
        cute::make_shape(M, N, K, batch_count));

    cutlass::KernelHardwareInfo hw_info;
    hw_info.device_id = 0;
    hw_info.sm_count = hw_sm_count;

    typename Chain::GemmArguments arguments{
        cutlass::gemm::GemmUniversalMode::kGemm,
        {M, N, K, batch_count},
        {   // Mainloop arguments: data + scale factors
            static_cast<const typename Chain::RawElementType*>(A),
            stride_A,
            static_cast<const typename Chain::RawElementType*>(B),
            stride_B,
            static_cast<const typename Chain::ElementSF*>(SFA),
            layout_SFA,
            static_cast<const typename Chain::ElementSF*>(SFB),
            layout_SFB
        },
        {   // Epilogue arguments (float C/D)
            {alpha, beta}, C, stride_C, C, stride_D
        },
        hw_info
    };

    // --- can_implement ---
    {
        typename Chain::DeviceGemm gemm_check;
        auto status = gemm_check.can_implement(arguments);
        if (status != cutlass::Status::kSuccess) {
            static bool printed = false;
            if (!printed) {
                printed = true;
                fprintf(stderr, "[CUTLASS FP32Out] can_implement FAILED (status=%d) for blockscaled GEMM(%d,%d,%d)\n",
                        static_cast<int>(status), M, N, K);
            }
            return status;
        }
    }

    // --- Workspace allocation ---
    size_t workspace_size = Chain::DeviceGemm::get_workspace_size(arguments);
    void* workspace = nullptr;
    bool own_workspace = false;
    if (workspace_size > 0) {
        if (ext_workspace && ext_workspace_size >= workspace_size) {
            workspace = ext_workspace;
        } else {
            cudaError_t alloc_err = cudaMallocAsync(&workspace, workspace_size, stream);
            if (alloc_err != cudaSuccess) {
                fprintf(stderr, "[CUTLASS FP32Out] workspace allocation failed: %s\n",
                        cudaGetErrorString(alloc_err));
                return cutlass::Status::kErrorInternal;
            }
            own_workspace = true;
        }
    }

    // --- Initialize workspace ---
    auto status = GemmKernel::initialize_workspace(arguments, workspace, stream, nullptr);
    if (status != cutlass::Status::kSuccess) {
        fprintf(stderr, "[CUTLASS FP32Out] initialize_workspace FAILED (status=%d)\n",
                static_cast<int>(status));
        if (own_workspace && workspace) cudaFreeAsync(workspace, stream);
        return status;
    }

    // --- Convert arguments to kernel params ---
    auto params = GemmKernel::to_underlying_arguments(arguments, workspace);

    // --- Set dynamic shared memory on concrete kernel ---
    constexpr int smem_size = GemmKernel::SharedStorageSize;

    static bool printed_diag = false;
    if (!printed_diag) {
        printed_diag = true;
        int device_smem_limit = 0;
        cudaDeviceGetAttribute(&device_smem_limit,
                               cudaDevAttrMaxSharedMemoryPerBlockOptin, 0);
        fprintf(stderr, "[DIAG] Block-scaled FP32Out GEMM tile: %dx%dx%d  cluster: %dx%dx%d\n",
                int(cute::size<0>(typename Chain::TileShape{})),
                int(cute::size<1>(typename Chain::TileShape{})),
                int(cute::size<2>(typename Chain::TileShape{})),
                int(cute::size<0>(typename Chain::ClusterShapeType{})),
                int(cute::size<1>(typename Chain::ClusterShapeType{})),
                int(cute::size<2>(typename Chain::ClusterShapeType{})));
        fprintf(stderr, "[DIAG] Block-scaled FP32Out SharedStorageSize: %d bytes (%d KB), device max: %d bytes (%d KB)\n",
                smem_size, smem_size / 1024, device_smem_limit, device_smem_limit / 1024);
    }

    if (smem_size >= (48 << 10)) {
        cudaError_t result = cudaFuncSetAttribute(
            kernel_fn,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            smem_size);
        if (cudaSuccess != result) {
            fprintf(stderr, "[CUTLASS FP32Out] cudaFuncSetAttribute FAILED (err=%d/%s) smem=%d\n",
                    static_cast<int>(result), cudaGetErrorString(result), smem_size);
            result = cudaGetLastError();
            if (own_workspace && workspace) cudaFreeAsync(workspace, stream);
            return cutlass::Status::kErrorInternal;
        }
    }

    // --- Launch concrete kernel ---
    dim3 block = GemmKernel::get_block_shape();
    dim3 grid = GemmKernel::get_grid_shape(params);

    void* kernel_args[] = {const_cast<void*>(static_cast<const void*>(&params))};
    cudaError_t launch_err = cudaLaunchKernel(kernel_fn, grid, block,
                                               kernel_args, smem_size, stream);
    if (launch_err != cudaSuccess) {
        fprintf(stderr, "[CUTLASS FP32Out] cudaLaunchKernel FAILED (err=%d/%s) for blockscaled GEMM(%d,%d,%d)\n",
                static_cast<int>(launch_err), cudaGetErrorString(launch_err), M, N, K);
        if (own_workspace && workspace) cudaFreeAsync(workspace, stream);
        return cutlass::Status::kErrorInternal;
    }

    if (own_workspace && workspace) cudaFreeAsync(workspace, stream);
    return cutlass::Status::kSuccess;
}

#endif // COMPLEX_SM100_ENABLE_FP6 || COMPLEX_SM100_ENABLE_FP4


// ============================================================================
// Public concrete dispatch functions
// ============================================================================

#ifdef COMPLEX_SM100_ENABLE_FP6

cutlass::Status run_blockscaled_gemm_fp6_e3m2(
    const void* A, const void* B,
    const void* SFA, const void* SFB,
    cutlass::half_t* C,
    int M, int N, int K,
    float alpha, float beta,
    cudaStream_t stream,
    int hw_sm_count,
    GemmConfig config,
    int batch_count,
    int ld_C,
    void* ext_workspace,
    size_t ext_workspace_size)
{
    switch (config) {
#if defined(COMPLEX_SM100_ENABLE_SMALL_M) && !defined(COMPLEX_FP8_SM100_TARGET_SM120)
    case GemmConfig::SmallM:
        return run_blockscaled_gemm_impl<ChainFP6_E3M2_SmallM>(
            (const void*)blockscaled_kernel_fp6_e3m2_smallm,
            A, B, SFA, SFB, C, M, N, K, alpha, beta, stream, hw_sm_count, batch_count, ld_C,
            ext_workspace, ext_workspace_size);
#endif
#ifndef COMPLEX_FP8_SM100_TARGET_SM120
    // Cluster variants (SM100 only — block-scaled uses same tile, different cluster)
    case GemmConfig::FP8_T128x128_C1x2:
    case GemmConfig::FP8_T128x256_C1x2:
        return run_blockscaled_gemm_impl<ChainFP6_E3M2_C1x2>(
            (const void*)blockscaled_kernel_fp6_e3m2_c1x2,
            A, B, SFA, SFB, C, M, N, K, alpha, beta, stream, hw_sm_count, batch_count, ld_C,
            ext_workspace, ext_workspace_size);
    case GemmConfig::FP8_T128x128_C2x2:
        return run_blockscaled_gemm_impl<ChainFP6_E3M2_C2x2>(
            (const void*)blockscaled_kernel_fp6_e3m2_c2x2,
            A, B, SFA, SFB, C, M, N, K, alpha, beta, stream, hw_sm_count, batch_count, ld_C,
            ext_workspace, ext_workspace_size);
#endif
    default:
        // All 1x1 cluster FP8 configs use the default block-scaled tile
        return run_blockscaled_gemm_impl<ChainFP6_E3M2>(
            (const void*)blockscaled_kernel_fp6_e3m2,
            A, B, SFA, SFB, C, M, N, K, alpha, beta, stream, hw_sm_count, batch_count, ld_C,
            ext_workspace, ext_workspace_size);
    }
}

size_t get_blockscaled_workspace_size_fp6_e3m2(
    int M, int N, int K, int hw_sm_count, int batch_count)
{
    auto stride_A = cutlass::make_cute_packed_stride(
        StrideA{}, cute::make_shape(M, K, batch_count));
    auto stride_B = cutlass::make_cute_packed_stride(
        StrideB{}, cute::make_shape(N, K, batch_count));
    auto stride_C = cutlass::make_cute_packed_stride(
        StrideC{}, cute::make_shape(M, N, batch_count));
    auto layout_SFA = ChainFP6_E3M2::Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(
        cute::make_shape(M, N, K, batch_count));
    auto layout_SFB = ChainFP6_E3M2::Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(
        cute::make_shape(M, N, K, batch_count));
    cutlass::KernelHardwareInfo hw_info;
    hw_info.device_id = 0;
    hw_info.sm_count = hw_sm_count;
    typename ChainFP6_E3M2::GemmArguments arguments{
        cutlass::gemm::GemmUniversalMode::kGemm,
        {M, N, K, batch_count},
        {nullptr, stride_A, nullptr, stride_B, nullptr, layout_SFA, nullptr, layout_SFB},
        {{1.0f, 0.0f}, nullptr, stride_C, nullptr, stride_C},
        hw_info
    };
    return ChainFP6_E3M2::DeviceGemm::get_workspace_size(arguments);
}

cutlass::Status run_blockscaled_gemm_fp6_e2m3(
    const void* A, const void* B,
    const void* SFA, const void* SFB,
    cutlass::half_t* C,
    int M, int N, int K,
    float alpha, float beta,
    cudaStream_t stream,
    int hw_sm_count,
    GemmConfig config,
    int batch_count,
    int ld_C,
    void* ext_workspace,
    size_t ext_workspace_size)
{
    switch (config) {
#if defined(COMPLEX_SM100_ENABLE_SMALL_M) && !defined(COMPLEX_FP8_SM100_TARGET_SM120)
    case GemmConfig::SmallM:
        // E2M3 SmallM not instantiated — fall back to default
        return run_blockscaled_gemm_impl<ChainFP6_E2M3>(
            (const void*)blockscaled_kernel_fp6_e2m3,
            A, B, SFA, SFB, C, M, N, K, alpha, beta, stream, hw_sm_count, batch_count, ld_C,
            ext_workspace, ext_workspace_size);
#endif
#ifndef COMPLEX_FP8_SM100_TARGET_SM120
    // Cluster variants (SM100 only — block-scaled uses same tile, different cluster)
    case GemmConfig::FP8_T128x128_C1x2:
    case GemmConfig::FP8_T128x256_C1x2:
        return run_blockscaled_gemm_impl<ChainFP6_E2M3_C1x2>(
            (const void*)blockscaled_kernel_fp6_e2m3_c1x2,
            A, B, SFA, SFB, C, M, N, K, alpha, beta, stream, hw_sm_count, batch_count, ld_C,
            ext_workspace, ext_workspace_size);
    case GemmConfig::FP8_T128x128_C2x2:
        return run_blockscaled_gemm_impl<ChainFP6_E2M3_C2x2>(
            (const void*)blockscaled_kernel_fp6_e2m3_c2x2,
            A, B, SFA, SFB, C, M, N, K, alpha, beta, stream, hw_sm_count, batch_count, ld_C,
            ext_workspace, ext_workspace_size);
#endif
    default:
        // All 1x1 cluster FP8 configs use the default block-scaled tile
        return run_blockscaled_gemm_impl<ChainFP6_E2M3>(
            (const void*)blockscaled_kernel_fp6_e2m3,
            A, B, SFA, SFB, C, M, N, K, alpha, beta, stream, hw_sm_count, batch_count, ld_C,
            ext_workspace, ext_workspace_size);
    }
}

size_t get_blockscaled_workspace_size_fp6_e2m3(
    int M, int N, int K, int hw_sm_count, int batch_count)
{
    auto stride_A = cutlass::make_cute_packed_stride(
        StrideA{}, cute::make_shape(M, K, batch_count));
    auto stride_B = cutlass::make_cute_packed_stride(
        StrideB{}, cute::make_shape(N, K, batch_count));
    auto stride_C = cutlass::make_cute_packed_stride(
        StrideC{}, cute::make_shape(M, N, batch_count));
    auto layout_SFA = ChainFP6_E2M3::Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(
        cute::make_shape(M, N, K, batch_count));
    auto layout_SFB = ChainFP6_E2M3::Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(
        cute::make_shape(M, N, K, batch_count));
    cutlass::KernelHardwareInfo hw_info;
    hw_info.device_id = 0;
    hw_info.sm_count = hw_sm_count;
    typename ChainFP6_E2M3::GemmArguments arguments{
        cutlass::gemm::GemmUniversalMode::kGemm,
        {M, N, K, batch_count},
        {nullptr, stride_A, nullptr, stride_B, nullptr, layout_SFA, nullptr, layout_SFB},
        {{1.0f, 0.0f}, nullptr, stride_C, nullptr, stride_C},
        hw_info
    };
    return ChainFP6_E2M3::DeviceGemm::get_workspace_size(arguments);
}

#endif // COMPLEX_SM100_ENABLE_FP6


#ifdef COMPLEX_SM100_ENABLE_FP4

cutlass::Status run_blockscaled_gemm_fp4(
    const void* A, const void* B,
    const void* SFA, const void* SFB,
    cutlass::half_t* C,
    int M, int N, int K,
    float alpha, float beta,
    cudaStream_t stream,
    int hw_sm_count,
    GemmConfig config,
    int batch_count,
    int ld_C,
    void* ext_workspace,
    size_t ext_workspace_size)
{
    switch (config) {
#if defined(COMPLEX_SM100_ENABLE_SMALL_M) && !defined(COMPLEX_FP8_SM100_TARGET_SM120)
    case GemmConfig::SmallM:
        return run_blockscaled_gemm_impl<ChainFP4_SmallM>(
            (const void*)blockscaled_kernel_fp4_smallm,
            A, B, SFA, SFB, C, M, N, K, alpha, beta, stream, hw_sm_count, batch_count, ld_C,
            ext_workspace, ext_workspace_size);
#endif
#ifndef COMPLEX_FP8_SM100_TARGET_SM120
    // Cluster variants (SM100 only — block-scaled uses same tile, different cluster)
    case GemmConfig::FP8_T128x128_C1x2:
    case GemmConfig::FP8_T128x256_C1x2:
        return run_blockscaled_gemm_impl<ChainFP4_C1x2>(
            (const void*)blockscaled_kernel_fp4_c1x2,
            A, B, SFA, SFB, C, M, N, K, alpha, beta, stream, hw_sm_count, batch_count, ld_C,
            ext_workspace, ext_workspace_size);
    case GemmConfig::FP8_T128x128_C2x2:
        return run_blockscaled_gemm_impl<ChainFP4_C2x2>(
            (const void*)blockscaled_kernel_fp4_c2x2,
            A, B, SFA, SFB, C, M, N, K, alpha, beta, stream, hw_sm_count, batch_count, ld_C,
            ext_workspace, ext_workspace_size);
#endif
    default:
        // All 1x1 cluster FP8 configs use the default block-scaled tile
        return run_blockscaled_gemm_impl<ChainFP4>(
            (const void*)blockscaled_kernel_fp4,
            A, B, SFA, SFB, C, M, N, K, alpha, beta, stream, hw_sm_count, batch_count, ld_C,
            ext_workspace, ext_workspace_size);
    }
}

size_t get_blockscaled_workspace_size_fp4(
    int M, int N, int K, int hw_sm_count, int batch_count)
{
    auto stride_A = cutlass::make_cute_packed_stride(
        StrideA{}, cute::make_shape(M, K, batch_count));
    auto stride_B = cutlass::make_cute_packed_stride(
        StrideB{}, cute::make_shape(N, K, batch_count));
    auto stride_C = cutlass::make_cute_packed_stride(
        StrideC{}, cute::make_shape(M, N, batch_count));
    auto layout_SFA = ChainFP4::Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(
        cute::make_shape(M, N, K, batch_count));
    auto layout_SFB = ChainFP4::Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(
        cute::make_shape(M, N, K, batch_count));
    cutlass::KernelHardwareInfo hw_info;
    hw_info.device_id = 0;
    hw_info.sm_count = hw_sm_count;
    typename ChainFP4::GemmArguments arguments{
        cutlass::gemm::GemmUniversalMode::kGemm,
        {M, N, K, batch_count},
        {nullptr, stride_A, nullptr, stride_B, nullptr, layout_SFA, nullptr, layout_SFB},
        {{1.0f, 0.0f}, nullptr, stride_C, nullptr, stride_C},
        hw_info
    };
    return ChainFP4::DeviceGemm::get_workspace_size(arguments);
}

#endif // COMPLEX_SM100_ENABLE_FP4


// ============================================================================
// FP32-output public dispatch functions
// ============================================================================

#ifdef COMPLEX_SM100_ENABLE_FP6

cutlass::Status run_blockscaled_gemm_fp6_e3m2_fp32out(
    const void* A, const void* B,
    const void* SFA, const void* SFB,
    float* C,
    int M, int N, int K,
    float alpha, float beta,
    cudaStream_t stream,
    int hw_sm_count,
    GemmConfig config,
    int batch_count,
    int ld_C,
    void* ext_workspace,
    size_t ext_workspace_size)
{
    switch (config) {
#if defined(COMPLEX_SM100_ENABLE_SMALL_M) && !defined(COMPLEX_FP8_SM100_TARGET_SM120)
    case GemmConfig::SmallM:
        return run_blockscaled_gemm_impl_fp32out<ChainFP6_E3M2_FP32Out_SmallM>(
            (const void*)blockscaled_kernel_fp6_e3m2_fp32out_smallm,
            A, B, SFA, SFB, C, M, N, K, alpha, beta, stream, hw_sm_count, batch_count, ld_C,
            ext_workspace, ext_workspace_size);
#endif
    default:
        // All 1x1 cluster FP8 configs use the default block-scaled tile;
        // FP32Out cluster variants not instantiated — fall back to default
        return run_blockscaled_gemm_impl_fp32out<ChainFP6_E3M2_FP32Out>(
            (const void*)blockscaled_kernel_fp6_e3m2_fp32out,
            A, B, SFA, SFB, C, M, N, K, alpha, beta, stream, hw_sm_count, batch_count, ld_C,
            ext_workspace, ext_workspace_size);
    }
}

size_t get_blockscaled_workspace_size_fp6_e3m2_fp32out(
    int M, int N, int K, int hw_sm_count, int batch_count)
{
    auto stride_A = cutlass::make_cute_packed_stride(
        StrideA{}, cute::make_shape(M, K, batch_count));
    auto stride_B = cutlass::make_cute_packed_stride(
        StrideB{}, cute::make_shape(N, K, batch_count));
    auto stride_C = cutlass::make_cute_packed_stride(
        StrideC{}, cute::make_shape(M, N, batch_count));
    auto layout_SFA = ChainFP6_E3M2_FP32Out::Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(
        cute::make_shape(M, N, K, batch_count));
    auto layout_SFB = ChainFP6_E3M2_FP32Out::Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(
        cute::make_shape(M, N, K, batch_count));
    cutlass::KernelHardwareInfo hw_info;
    hw_info.device_id = 0;
    hw_info.sm_count = hw_sm_count;
    typename ChainFP6_E3M2_FP32Out::GemmArguments arguments{
        cutlass::gemm::GemmUniversalMode::kGemm,
        {M, N, K, batch_count},
        {nullptr, stride_A, nullptr, stride_B, nullptr, layout_SFA, nullptr, layout_SFB},
        {{1.0f, 0.0f}, nullptr, stride_C, nullptr, stride_C},
        hw_info
    };
    return ChainFP6_E3M2_FP32Out::DeviceGemm::get_workspace_size(arguments);
}

#endif // COMPLEX_SM100_ENABLE_FP6


#ifdef COMPLEX_SM100_ENABLE_FP4

cutlass::Status run_blockscaled_gemm_fp4_fp32out(
    const void* A, const void* B,
    const void* SFA, const void* SFB,
    float* C,
    int M, int N, int K,
    float alpha, float beta,
    cudaStream_t stream,
    int hw_sm_count,
    GemmConfig config,
    int batch_count,
    int ld_C,
    void* ext_workspace,
    size_t ext_workspace_size)
{
    switch (config) {
#if defined(COMPLEX_SM100_ENABLE_SMALL_M) && !defined(COMPLEX_FP8_SM100_TARGET_SM120)
    case GemmConfig::SmallM:
        return run_blockscaled_gemm_impl_fp32out<ChainFP4_FP32Out_SmallM>(
            (const void*)blockscaled_kernel_fp4_fp32out_smallm,
            A, B, SFA, SFB, C, M, N, K, alpha, beta, stream, hw_sm_count, batch_count, ld_C,
            ext_workspace, ext_workspace_size);
#endif
    default:
        // All 1x1 cluster FP8 configs use the default block-scaled tile;
        // FP32Out cluster variants not instantiated — fall back to default
        return run_blockscaled_gemm_impl_fp32out<ChainFP4_FP32Out>(
            (const void*)blockscaled_kernel_fp4_fp32out,
            A, B, SFA, SFB, C, M, N, K, alpha, beta, stream, hw_sm_count, batch_count, ld_C,
            ext_workspace, ext_workspace_size);
    }
}

size_t get_blockscaled_workspace_size_fp4_fp32out(
    int M, int N, int K, int hw_sm_count, int batch_count)
{
    auto stride_A = cutlass::make_cute_packed_stride(
        StrideA{}, cute::make_shape(M, K, batch_count));
    auto stride_B = cutlass::make_cute_packed_stride(
        StrideB{}, cute::make_shape(N, K, batch_count));
    auto stride_C = cutlass::make_cute_packed_stride(
        StrideC{}, cute::make_shape(M, N, batch_count));
    auto layout_SFA = ChainFP4_FP32Out::Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(
        cute::make_shape(M, N, K, batch_count));
    auto layout_SFB = ChainFP4_FP32Out::Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(
        cute::make_shape(M, N, K, batch_count));
    cutlass::KernelHardwareInfo hw_info;
    hw_info.device_id = 0;
    hw_info.sm_count = hw_sm_count;
    typename ChainFP4_FP32Out::GemmArguments arguments{
        cutlass::gemm::GemmUniversalMode::kGemm,
        {M, N, K, batch_count},
        {nullptr, stride_A, nullptr, stride_B, nullptr, layout_SFA, nullptr, layout_SFB},
        {{1.0f, 0.0f}, nullptr, stride_C, nullptr, stride_C},
        hw_info
    };
    return ChainFP4_FP32Out::DeviceGemm::get_workspace_size(arguments);
}

#endif // COMPLEX_SM100_ENABLE_FP4

} // namespace gemm_complex_sm100
