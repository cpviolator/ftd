/**
   @file quda_define.h
   @brief Macros defined set by the cmake build system.  This file
   should not be edited manually.
 */

/**
 * @def   __COMPUTE_CAPABILITY__
 * @brief This macro sets the target GPU architecture, which is
 * defined on both host and device.
 */
#define __COMPUTE_CAPABILITY__ 1210

/**
 * @def   MAX_MULTI_RHS
 * @brief This macro sets the limit of RHS the multi-blas
 * and multi-reduce kernels
 */
#define MAX_MULTI_RHS 64

#define GGP_HETEROGENEOUS_ATOMIC
#ifdef GGP_HETEROGENEOUS_ATOMIC
/**
 * @def   HETEROGENEOUS_ATOMIC
 * @brief This macro sets whether we are compiling GGP with heterogeneous atomic
 * support enabled or not
 */
#define HETEROGENEOUS_ATOMIC
#undef GGP_HETEROGENEOUS_ATOMIC
#endif

#define GGP_LARGE_KERNEL_ARG


/**
 * @def GGP_ORDER_FP
 * @brief This macro sets the data ordering for Wilson, gauge
 * (recon-8/9) and clover fixed-point fields
 */
#define GGP_ORDER_FP 8

#ifdef __cplusplus
static_assert(GGP_ORDER_FP == 4 || GGP_ORDER_FP == 8, "invalid GGP_ORDER_FP");
#endif

/**
 * @def GGP_ORDER_SP
 * @brief This macro sets the data ordering for single-precision multigrid fields
 */
#define GGP_ORDER_SP_MG 2

#ifdef __cplusplus
static_assert(GGP_ORDER_SP_MG == 2 || GGP_ORDER_SP_MG == 4, "invalid GGP_ORDER_SP_MG");
#endif

/**
 * @def GGP_ORDER_FP_MG
 * @brief This macro sets the data ordering for fixed-point multigrid fields
 */
#define GGP_ORDER_FP_MG 2

#ifdef __cplusplus
static_assert(GGP_ORDER_FP_MG == 2 || GGP_ORDER_FP_MG == 4 || GGP_ORDER_FP_MG == 8, "invalid GGP_ORDER_FP_MG");
#endif

/**
 * @def GGP_BUILD_NATIVE_FFT
 * @brief This macro is set by CMake if the native FFT library is used
 */
#define GGP_BUILD_NATIVE_FFT ON

/**
 * @def GGP_TARGET_CUDA
 * @brief This macro is set by CMake if the CUDA Build Target is selected
 */
#define GGP_TARGET_CUDA ON

/**
 * @def GGP_TARGET_HIP
 * @brief This macro is set by CMake if the HIP Build target is selected
 */
/* #undef GGP_TARGET_HIP */

/**
 * @def GGP_TARGET_SYCL
 * @brief This macro is set by CMake if the SYCL Build target is selected
 */
/* #undef GGP_TARGET_SYCL */

#if !defined(GGP_TARGET_CUDA) && !defined(GGP_TARGET_HIP) && !defined(GGP_TARGET_SYCL)
#error "No GGP_TARGET selected"
#endif
