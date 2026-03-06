/*
 * gemm_blockscaled_dispatch.h
 *
 * Forward declarations for concrete (non-template) block-scaled GEMM dispatch
 * functions. These are implemented in gemm_blockscaled_dispatch.cu as non-template
 * functions in a .cu compilation unit, which forces nvcc to properly generate and
 * register the device_kernel<GemmKernel> binary.
 *
 * Background: CUTLASS issue #2478 — when device_kernel<GemmKernel> is instantiated
 * only through deep template indirection in header-only code, nvcc fails to register
 * the kernel binary with the CUDA runtime. Moving the GEMM initialization and launch
 * code into a .cu file resolves this.
 *
 * The block-scaled type chain (mx_float6_t/mx_float4_t → BlockScaledKernelTypeChain
 * → OpClassBlockScaledTensorOp → GemmUniversal → device_kernel) is deeper than the
 * FP8 chain and triggers this nvcc limitation. FP8 kernels are unaffected.
 */

#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/numeric_types.h"
#include <cuda_runtime.h>
#include <vector>

namespace gemm_complex_sm100 {

// GemmConfig enum, config_name(), select_gemm_config(), is_config_valid_for_arch(),
// and all_baseline_configs() are defined in shared/config_common.hpp (included via
// gemm_complex_sm100/config.hpp → shared/config_common.hpp in the namespace).
//
// For standalone .cu files that include this header directly (like
// gemm_blockscaled_dispatch.cu), config_common.hpp is included inside
// the namespace below if GemmConfig is not yet defined.
#include "shared/config_common.hpp"


/// Compute strategy for INT4 sign-magnitude complex input.
/// Selects which tensor core precision to use for the sub-GEMMs.
enum class ComputeStrategy {
    FP8_E4M3,   // INT4 → FP8: exact, FP32 accumulation
#ifndef COMPLEX_FP8_SM100_TARGET_SM120
    INT8,        // INT4 → INT8: exact, INT32 accumulation (SM100 only — SM120 lacks INT8 TC)
#endif
#ifdef COMPLEX_SM100_ENABLE_FP6
    FP6_E3M2,   // INT4 → FP16 → MXFP FP6: exact, FP32 accumulation, 25% less bandwidth
#endif
#ifdef COMPLEX_SM100_ENABLE_FP4
    FP4_E2M1,   // INT4 → FP16 → MXFP FP4: lossy (5→4, 7→8), 2x throughput
#endif
};

inline const char* strategy_name(ComputeStrategy s) {
    switch (s) {
    case ComputeStrategy::FP8_E4M3: return "FP8_E4M3";
#ifndef COMPLEX_FP8_SM100_TARGET_SM120
    case ComputeStrategy::INT8: return "INT8";
#endif
#ifdef COMPLEX_SM100_ENABLE_FP6
    case ComputeStrategy::FP6_E3M2: return "FP6_E3M2";
#endif
#ifdef COMPLEX_SM100_ENABLE_FP4
    case ComputeStrategy::FP4_E2M1: return "FP4_E2M1";
#endif
    default: return "Unknown";
    }
}

// Each function wraps run_real_gemm_blockscaled_impl<Chain> for one concrete chain type.
// Parameters match the template version, plus hw_sm_count (caller must ensure_hw_info first).

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
    int batch_count = 1,
    int ld_C = 0,
    void* ext_workspace = nullptr,
    size_t ext_workspace_size = 0);

/// Query workspace size for FP6 E3M2 FP16-output block-scaled GEMM.
size_t get_blockscaled_workspace_size_fp6_e3m2(
    int M, int N, int K, int hw_sm_count, int batch_count = 1);

cutlass::Status run_blockscaled_gemm_fp6_e2m3(
    const void* A, const void* B,
    const void* SFA, const void* SFB,
    cutlass::half_t* C,
    int M, int N, int K,
    float alpha, float beta,
    cudaStream_t stream,
    int hw_sm_count,
    GemmConfig config,
    int batch_count = 1,
    int ld_C = 0,
    void* ext_workspace = nullptr,
    size_t ext_workspace_size = 0);

/// Query workspace size for FP6 E2M3 FP16-output block-scaled GEMM.
size_t get_blockscaled_workspace_size_fp6_e2m3(
    int M, int N, int K, int hw_sm_count, int batch_count = 1);

#endif // COMPLEX_SM100_ENABLE_FP6

// INT8 GEMM dispatch — SM100 only (SM120 lacks INT8 tensor cores)
#ifndef COMPLEX_FP8_SM100_TARGET_SM120
cutlass::Status run_int8_gemm(
    const int8_t* A, const int8_t* B,
    cutlass::half_t* C,
    int M, int N, int K,
    float alpha, float beta,
    cudaStream_t stream,
    int hw_sm_count,
    GemmConfig config,
    int batch_count = 1);
#endif

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
    int batch_count = 1,
    int ld_C = 0,
    void* ext_workspace = nullptr,
    size_t ext_workspace_size = 0);

/// Query workspace size for FP4 FP16-output block-scaled GEMM.
size_t get_blockscaled_workspace_size_fp4(
    int M, int N, int K, int hw_sm_count, int batch_count = 1);

#endif // COMPLEX_SM100_ENABLE_FP4

// ============================================================================
// FP32-output block-scaled dispatch (for large-K / high dynamic range)
// ============================================================================
// Same as above but with float* C output instead of half_t* C.
// Avoids FP16 overflow when K * max_A * max_B > 65504.

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
    int batch_count = 1,
    int ld_C = 0,
    void* ext_workspace = nullptr,
    size_t ext_workspace_size = 0);

/// Query workspace size for FP6 E3M2 FP32-output block-scaled GEMM.
size_t get_blockscaled_workspace_size_fp6_e3m2_fp32out(
    int M, int N, int K, int hw_sm_count, int batch_count = 1);

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
    int batch_count = 1,
    int ld_C = 0,
    void* ext_workspace = nullptr,
    size_t ext_workspace_size = 0);

/// Query workspace size for FP4 FP32-output block-scaled GEMM.
size_t get_blockscaled_workspace_size_fp4_fp32out(
    int M, int N, int K, int hw_sm_count, int batch_count = 1);

#endif // COMPLEX_SM100_ENABLE_FP4

} // namespace gemm_complex_sm100
