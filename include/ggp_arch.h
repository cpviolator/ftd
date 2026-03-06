#pragma once
#include <ggp_define.h>

#if defined(GGP_TARGET_CUDA)
#include <cuda.h>
#include <cuda_runtime.h>

#if (__COMPUTE_CAPABILITY__ >= 700) && defined(GGP_ENABLE_MMA)
#define GGP_MMA_AVAILABLE
#endif

#elif defined(GGP_TARGET_HIP)
#include <hip/hip_runtime.h>

#elif defined(GGP_TARGET_SYCL)
#include <targets/sycl/GGP_sycl.h>
#endif

#ifdef GGP_OPENMP
#include <omp.h>
#endif
