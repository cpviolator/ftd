// --- config.hpp ---
// Macros, constants, and type traits for gpu_fdd_improved.
// Textual include — no #pragma once.

// Helper: Check if type is FP8
template <typename T> struct IsFP8 : std::false_type {};
template <> struct IsFP8<__nv_fp8_e4m3> : std::true_type {};

// Check for CUDA errors
#define CUBLASLT_CHECK(call)                                                  \
  do {                                                                        \
    cublasStatus_t err = (call);                                              \
    if (err != CUBLAS_STATUS_SUCCESS) {                                       \
      fprintf(stderr, "cuBLASLt Error (Status %d) at %s:%d\n", (int)err,      \
              __FILE__, __LINE__);                                            \
      exit(EXIT_FAILURE);                                                     \
    }                                                                         \
  } while (0)

#define CUDA_CHECK(call)                                                      \
  do {                                                                        \
    cudaError_t err = (call);                                                 \
    if (err != cudaSuccess) {                                                 \
      fprintf(stderr, "CUDA Error at %s:%d: %s\n", __FILE__, __LINE__,        \
              cudaGetErrorString(err));                                       \
      exit(EXIT_FAILURE);                                                     \
    }                                                                         \
  } while (0)

#define CURAND_CHECK(call)                                                    \
  do {                                                                        \
    curandStatus_t err = (call);                                              \
    if (err != CURAND_STATUS_SUCCESS) {                                       \
      fprintf(stderr, "cuRAND Error (Status %d) at %s:%d\n", (int)err,        \
              __FILE__, __LINE__);                                            \
      exit(EXIT_FAILURE);                                                     \
    }                                                                         \
  } while (0)

#define CUBLAS_CHECK(call)                                                    \
  do {                                                                        \
    cublasStatus_t err = (call);                                              \
    if (err != CUBLAS_STATUS_SUCCESS) {                                       \
      fprintf(stderr, "cuBLAS Error at %s:%d\n", __FILE__, __LINE__);          \
      exit(EXIT_FAILURE);                                                     \
    }                                                                         \
  } while (0)


// --- Global Constants ---
const double DISPERSION_CONSTANT = 4.15e-3;
__device__ const double PI_D = 3.14159265358979323846;
const int TILE_DIM = 32; // For transpose kernel

// --- HELPER TYPES ---
// Wrapper to help templates select the storage type
template <typename T> struct StorageType { using type = T; };
template <> struct StorageType<__nv_fp8_e4m3> { using type = __nv_fp8_e4m3; };
// Note: cuBLASLt expects FP8 matrices to be void* but strictly typed in descriptors
