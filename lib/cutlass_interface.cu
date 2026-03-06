/**
 * @file cutlass_interface.cu
 * @brief CUTLASS CUDA-only backend for FTD XEngine correlation.
 *
 * When built with -DGGP_CUTLASS=ON -DGGP_CUTLASS_GEMM_API_DIR=<path>,
 * this file uses the pre-built CUTLASS GEMM API library (PIMPL pattern)
 * which requires only cuda_fp16.h and cuda_runtime.h -- no CUTLASS headers.
 *
 * Provides two levels of integration:
 *  1. stridedBatchGEMMCutlass() -- drop-in replacement for the generic BLAS
 *     dispatch path (operates on already-promoted FP32 complex data).
 *  2. herkBatchedCutlassQC() / herkBatchedCutlassFP16() -- direct HERK from
 *     native QC (INT4 sign-magnitude) or FP16 interleaved complex input,
 *     producing packed lower triangular output in one call. Used by the
 *     optimized XEngine compute path to eliminate promoteData + cuBLAS +
 *     triangulateFromHerm.
 */
#include <ggp.h>
#include <timer.h>
#include <algorithm>

// ======================================================================
// New CUDA-only backend via CUTLASS GEMM API (PIMPL)
// ======================================================================
#ifdef CUTLASS_GEMM_API

#include "cutlass_gemm_api.h"
using cutlass_gemm_api::InputPrecision;
using cutlass_gemm_api::ComputePrecision;
using cutlass_gemm_api::OutputPrecision;
using cutlass_gemm_api::OutputFormat;
#include <cstdio>
#include <cstdint>

namespace {

/// @brief Singleton accessor for the CUTLASS GEMM API instance.
cutlass_gemm_api::CutlassComplexGemm& get_cutlass_api() {
    static cutlass_gemm_api::CutlassComplexGemm instance;
    return instance;
}

/// @brief Whether HERK autotuning (strategy sweep) is enabled.
static bool g_herk_tune_enabled = false;

/// @brief Whether GEMM autotuning (GemmConfig sweep) is enabled.
static bool g_gemm_tune_enabled = false;

/// @brief CUDA kernel: narrow FP32 to FP16, element-wise.
__global__ void narrow_fp32_to_fp16_kernel(
    const float* __restrict__ input,
    __half* __restrict__ output,
    int64_t n)
{
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
    for (int64_t i = idx; i < n; i += stride) {
        output[i] = __float2half(input[i]);
    }
}

/// @brief Launch FP32-to-FP16 narrowing conversion.
void narrow_fp32_to_fp16(
    const float* input, __half* output,
    int64_t n, cudaStream_t stream)
{
    if (n == 0) return;
    const int block = 256;
    const int64_t grid = min((n + block - 1) / block, static_cast<int64_t>(1024));
    narrow_fp32_to_fp16_kernel<<<grid, block, 0, stream>>>(input, output, n);
}

/// @brief CUDA kernel: widen FP16 to FP32, element-wise.
__global__ void widen_fp16_to_fp32_kernel(
    const __half* __restrict__ input,
    float* __restrict__ output,
    int64_t n)
{
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
    for (int64_t i = idx; i < n; i += stride) {
        output[i] = __half2float(input[i]);
    }
}

/// @brief Launch FP16-to-FP32 widening conversion.
void widen_fp16_to_fp32(
    const __half* input, float* output,
    int64_t n, cudaStream_t stream)
{
    if (n == 0) return;
    const int block = 256;
    const int64_t grid = min((n + block - 1) / block, static_cast<int64_t>(1024));
    widen_fp16_to_fp32_kernel<<<grid, block, 0, stream>>>(input, output, n);
}

/// @brief CUDA kernel: deinterleave FP32 complex -> FP16 planar Re/Im.
__global__ void deinterleave_narrow_kernel(
    const float* __restrict__ input,
    __half* __restrict__ out_re,
    __half* __restrict__ out_im,
    int64_t n_complex,
    bool negate_im)
{
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
    for (int64_t i = idx; i < n_complex; i += stride) {
        float re = input[2 * i];
        float im = input[2 * i + 1];
        out_re[i] = __float2half(re);
        out_im[i] = __float2half(negate_im ? -im : im);
    }
}

/// @brief CUDA kernel: interleave FP32 planar Re/Im -> FP32 interleaved complex.
__global__ void interleave_fp32_kernel(
    const float* __restrict__ in_re,
    const float* __restrict__ in_im,
    float* __restrict__ output,
    int64_t n_complex)
{
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
    for (int64_t i = idx; i < n_complex; i += stride) {
        output[2 * i]     = in_re[i];
        output[2 * i + 1] = in_im[i];
    }
}

} // anonymous namespace

// HERK functions live in namespace ggp (called from VisibilityBeamformer)
namespace ggp
{

/// @brief Initialize the CUTLASS API singleton with tuning configuration.
///
/// Call once before any HERK/GEMM calls (typically from VisibilityPipeline
/// constructor). Safe to call multiple times; the last call wins.
///
/// @param kernel_tune_verbosity  Kernel-level tuning verbosity (0-3).
///                               Levels >= 2 trigger blockDim/gridDim sweep.
/// @param kernel_tune_cache_path  File path for kernel tune cache (nullptr = default).
/// @param strategy_tune_cache_path  File path for strategy/GEMM tune cache (nullptr = default).
/// @param herk_tune  If true, HERK and GEMM calls will auto-tune strategy
///                   (HerkMode, GemmConfig, etc.) on first call per problem size.
/// @param strategy_tune_verbosity  Strategy-level tuning verbosity (0-3) for both
///                                  HERK and GEMM autotuning.
///                                  0=silent, 1=summary (default), 2=sweep, 3=full.
void initCutlassApi(int kernel_tune_verbosity,
                    const char* kernel_tune_cache_path,
                    const char* strategy_tune_cache_path,
                    bool herk_tune,
                    int strategy_tune_verbosity)
{
    auto& api = get_cutlass_api();
    if (kernel_tune_verbosity > 0)
        api.set_kernel_tune_verbosity(kernel_tune_verbosity);
    api.set_strategy_tune_verbosity(strategy_tune_verbosity);
    api.set_gemm_tune_verbosity(strategy_tune_verbosity);
    if (kernel_tune_cache_path)
        api.set_kernel_tune_cache_path(kernel_tune_cache_path);
    if (strategy_tune_cache_path) {
        api.set_tune_cache_path(strategy_tune_cache_path);
        api.set_gemm_tune_cache_path(strategy_tune_cache_path);
    }
    g_herk_tune_enabled = herk_tune;
    g_gemm_tune_enabled = herk_tune;
}

/// @brief Batched HERK from native QC (INT4 sign-magnitude) input.
///
/// The CUTLASS fused HERK kernel's cp.async load path requires K >= 16
/// (each receiver's FP8 row must be >= 32 bytes for the 2x16-byte
/// vectorized loads).  When K < 16, we pad the input along K with zero
/// bytes so that the fused kernel operates correctly.  Zero-padded
/// elements contribute 0 to the HERK accumulation, preserving correctness.
int herkBatchedCutlassQC(
    const void* qc_data,
    void* tri_output,
    int N, int K, int batch,
    cudaStream_t stream)
{
    auto& api = get_cutlass_api();

    // Minimum K for the fused pre-cast HERK kernel (HERK_K_PER_MMA = 16).
    // Below this, the cp.async 32-byte loads read beyond receiver row
    // boundaries.  Pad with zero QC bytes (Re=0, Im=0) which don't
    // contribute to the HERK result.
    static constexpr int K_MIN_FUSED = 16;

    if (K < K_MIN_FUSED) {
        const int K_padded = K_MIN_FUSED;
        const size_t padded_size = static_cast<size_t>(batch) * N * K_padded;

        void* d_padded = nullptr;
        cudaMallocAsync(&d_padded, padded_size, stream);
        cudaMemsetAsync(d_padded, 0, padded_size, stream);

        // Scatter each receiver's K bytes into K_padded-strided rows.
        // Source: [batch*N] rows × K  bytes, pitch = K
        // Dest:   [batch*N] rows × K_padded bytes, pitch = K_padded
        cudaMemcpy2DAsync(d_padded, K_padded,
                          qc_data, K,
                          K, static_cast<size_t>(batch) * N,
                          cudaMemcpyDeviceToDevice, stream);

        int ret = api.herk(
            d_padded,
            static_cast<float*>(tri_output),
            N, K_padded, batch,
            InputPrecision::INT4, ComputePrecision::FP8,
            OutputPrecision::FP32, OutputFormat::PackedTriangle,
            1.0f, 0.0f,
            stream,
            /*tune=*/g_herk_tune_enabled);

        cudaFreeAsync(d_padded, stream);
        return ret;
    }

    return api.herk(
        qc_data,
        static_cast<float*>(tri_output),
        N, K, batch,
        InputPrecision::INT4, ComputePrecision::FP8,
        OutputPrecision::FP32, OutputFormat::PackedTriangle,
        1.0f, 0.0f,
        stream,
        /*tune=*/g_herk_tune_enabled);
}

/// @brief Batched HERK from native QC (INT4) input with FP16 output.
///
/// Same as herkBatchedCutlassQC() but outputs __half packed triangle
/// instead of float. Halves the triangle output memory (2 bytes/element
/// vs 4), enabling larger batch counts and more memory-efficient
/// imaging pipelines.
///
/// @warning FP16 output can overflow for large K with large input values.
///          For INT4 input (max magnitude 7), worst case is K*7^2*2 = 98*K.
///          Safe for K <= ~668 (result < 65504 = FP16 max).
int herkBatchedCutlassQC_FP16(
    const void* qc_data,
    void* tri_output,
    int N, int K, int batch,
    cudaStream_t stream)
{
    auto& api = get_cutlass_api();

    static constexpr int K_MIN_FUSED = 16;

    if (K < K_MIN_FUSED) {
        const int K_padded = K_MIN_FUSED;
        const size_t padded_size = static_cast<size_t>(batch) * N * K_padded;

        void* d_padded = nullptr;
        cudaMallocAsync(&d_padded, padded_size, stream);
        cudaMemsetAsync(d_padded, 0, padded_size, stream);

        cudaMemcpy2DAsync(d_padded, K_padded,
                          qc_data, K,
                          K, static_cast<size_t>(batch) * N,
                          cudaMemcpyDeviceToDevice, stream);

        int ret = api.herk(
            d_padded,
            tri_output,
            N, K_padded, batch,
            InputPrecision::INT4, ComputePrecision::FP8,
            OutputPrecision::FP16, OutputFormat::PackedTriangle,
            1.0f, 0.0f,
            stream,
            /*tune=*/g_herk_tune_enabled);

        cudaFreeAsync(d_padded, stream);
        return ret;
    }

    return api.herk(
        qc_data,
        tri_output,
        N, K, batch,
        InputPrecision::INT4, ComputePrecision::FP8,
        OutputPrecision::FP16, OutputFormat::PackedTriangle,
        1.0f, 0.0f,
        stream,
        /*tune=*/g_herk_tune_enabled);
}

/// @brief Batched HERK from interleaved FP16 complex input.
int herkBatchedCutlassFP16(
    const void* fp16_data,
    void* tri_output,
    int N, int K, int batch,
    cudaStream_t stream)
{
    auto& api = get_cutlass_api();
    return api.herk(
        fp16_data, tri_output,
        N, K, batch,
        InputPrecision::FP16, ComputePrecision::FP8,
        OutputPrecision::FP16, OutputFormat::PackedTriangle,
        1.0f, 0.0f,
        stream,
        /*tune=*/g_herk_tune_enabled);
}

} // namespace ggp

// BLAS-interface functions live in namespace quda (called from blas_lapack_cublas.cpp)
namespace quda
{

/// @brief Generic strided batch GEMM via CUTLASS (drop-in for cuBLAS path).
long long stridedBatchGEMMCutlass(
    void *A_data, void *B_data, void *C_data,
    QudaBLASParam param, QudaFieldLocation location)
{
    auto& api = get_cutlass_api();

    const int M = param.m;
    const int N = param.n;
    const int K = param.k;
    const int batch = param.batch_count;

    // Detect HERK pattern: A == B, trans_a = conj transpose, trans_b = no transpose
    const bool is_herk = (A_data == B_data) &&
                         (param.trans_a == QUDA_BLAS_OP_C) &&
                         (param.trans_b == QUDA_BLAS_OP_N) &&
                         (M == N);

    if (is_herk && (param.data_type == QUDA_BLAS_DATATYPE_C)) {
        // FP32 interleaved complex HERK
        // Step 1: Narrow FP32 interleaved -> FP16 interleaved
        const int64_t total_fp16 = static_cast<int64_t>(N) * K * 2 * batch;
        __half* A_fp16 = nullptr;
        cudaMalloc(&A_fp16, total_fp16 * sizeof(__half));
        narrow_fp32_to_fp16(
            static_cast<const float*>(A_data), A_fp16,
            total_fp16, nullptr);

        // Step 2: HERK -> packed lower triangle FP16
        const int64_t packed = static_cast<int64_t>(N) * (N + 1) / 2;
        const int64_t tri_fp16_elems = packed * 2 * batch;
        __half* C_tri_fp16 = nullptr;
        cudaMalloc(&C_tri_fp16, tri_fp16_elems * sizeof(__half));

        int ret = api.herk(
            A_fp16, C_tri_fp16,
            N, K, batch,
            InputPrecision::FP16, ComputePrecision::FP8,
            OutputPrecision::FP16, OutputFormat::PackedTriangle,
            1.0f, 0.0f,
            nullptr,
            /*tune=*/g_herk_tune_enabled);

        // Step 3: Widen packed triangle FP16 -> FP32 and write to output
        if (ret == 0) {
            widen_fp16_to_fp32(C_tri_fp16, static_cast<float*>(C_data),
                               tri_fp16_elems, nullptr);
        }

        cudaFree(A_fp16);
        cudaFree(C_tri_fp16);
        cudaDeviceSynchronize();

        if (ret != 0) return -1;
        return static_cast<long long>(batch) * 8LL * M * N * K;
    }

    // General complex GEMM path (non-HERK)
    if (param.data_type == QUDA_BLAS_DATATYPE_C) {
        errorQuda("CUTLASS GEMM API: general complex GEMM not yet implemented via this interface. "
                  "Use the HERK path (trans_a=CONJ_TRANS, A==B) or cuBLAS.");
        return -1;
    }

    errorQuda("CUTLASS GEMM API: unsupported data type %d", param.data_type);
    return -1;
}

/// @brief General batched complex GEMM via CUTLASS planar API.
long long gemmBatchedCutlass(
    void *A_data, void *B_data, void *C_data,
    int M, int N, int K, int batch_count,
    bool conjugate_b)
{
    auto& api = get_cutlass_api();
    const int block = 256;

    // Element counts
    const int64_t A_complex = static_cast<int64_t>(batch_count) * M * K;
    const int64_t B_complex = static_cast<int64_t>(batch_count) * N * K;
    const int64_t C_complex = static_cast<int64_t>(batch_count) * M * N;

    // Allocate FP16 scratch for A and B (planar), FP32 scratch for C (planar)
    __half *A_re = nullptr, *A_im = nullptr;
    __half *B_re = nullptr, *B_im = nullptr;
    float *C_re = nullptr, *C_im = nullptr;

    cudaMalloc(&A_re, A_complex * sizeof(__half));
    cudaMalloc(&A_im, A_complex * sizeof(__half));
    cudaMalloc(&B_re, B_complex * sizeof(__half));
    cudaMalloc(&B_im, B_complex * sizeof(__half));
    cudaMalloc(&C_re, C_complex * sizeof(float));
    cudaMalloc(&C_im, C_complex * sizeof(float));

    // Deinterleave + narrow A -> {A_re, A_im}
    {
        int64_t grid = std::min((A_complex + block - 1) / block, static_cast<int64_t>(1024));
        deinterleave_narrow_kernel<<<grid, block>>>(
            static_cast<const float*>(A_data), A_re, A_im, A_complex, false);
    }

    // Deinterleave + narrow B -> {B_re, +/-B_im}
    {
        int64_t grid = std::min((B_complex + block - 1) / block, static_cast<int64_t>(1024));
        deinterleave_narrow_kernel<<<grid, block>>>(
            static_cast<const float*>(B_data), B_re, B_im, B_complex, !conjugate_b);
    }

    // CUTLASS planar batched GEMM: C = A x B^T (TN layout)
    int ret = api.gemm(
        A_re, A_im, B_re, B_im, C_re, C_im,
        M, N, K, batch_count,
        ComputePrecision::FP8, OutputPrecision::FP32,
        1.0f, 0.0f, nullptr,
        /*tune=*/g_gemm_tune_enabled);

    // Re-interleave C: {C_re, C_im} -> FP32 interleaved
    if (ret == 0) {
        int64_t grid = std::min((C_complex + block - 1) / block, static_cast<int64_t>(1024));
        interleave_fp32_kernel<<<grid, block>>>(
            C_re, C_im, static_cast<float*>(C_data), C_complex);
    }

    cudaFree(A_re);
    cudaFree(A_im);
    cudaFree(B_re);
    cudaFree(B_im);
    cudaFree(C_re);
    cudaFree(C_im);

    if (ret != 0) return -1;
    return static_cast<long long>(batch_count) * 8LL * M * N * K;
}

} // namespace quda

// ======================================================================
// Legacy CUTLASS backend (CUTLASS 2.x headers, SM80 only)
// ======================================================================
#elif defined(CUTLASS_LIB)

#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm_complex.h"
#include "cutlass/util/device_memory.h"

#define CUTLASS_CHECK(status)                                                   \
  {                                                                             \
    cutlass::Status error = status;                                             \
    if (error != cutlass::Status::kSuccess) {                                   \
      std::cerr << "Got cutlass error: " << cutlassGetStatusString(error)       \
                << " at: " << __LINE__ << std::endl;                            \
      exit(EXIT_FAILURE);                                                       \
    }                                                                           \
  }

namespace ggp
{

void initCutlassApi(int, const char*, const char*, bool, int) {
    // No-op: CUTLASS_GEMM_API not compiled.
}

int herkBatchedCutlassQC(const void*, void*, int, int, int, cudaStream_t) {
    errorQuda("herkBatchedCutlassQC requires CUTLASS_GEMM_API build (-DGGP_CUTLASS_GEMM_API_DIR=...)");
    return -1;
}

int herkBatchedCutlassQC_FP16(const void*, void*, int, int, int, cudaStream_t) {
    errorQuda("herkBatchedCutlassQC_FP16 requires CUTLASS_GEMM_API build (-DGGP_CUTLASS_GEMM_API_DIR=...)");
    return -1;
}

int herkBatchedCutlassFP16(const void*, void*, int, int, int, cudaStream_t) {
    errorQuda("herkBatchedCutlassFP16 requires CUTLASS_GEMM_API build (-DGGP_CUTLASS_GEMM_API_DIR=...)");
    return -1;
}

} // namespace ggp

namespace quda
{

long long gemmBatchedCutlass(void *A_data, void *B_data, void *C_data,
                              int M, int N, int K, int batch_count, bool conjugate_b)
{
    errorQuda("gemmBatchedCutlass requires CUTLASS_GEMM_API build (-DGGP_CUTLASS_GEMM_API_DIR=...)");
    return -1;
}

long long stridedBatchGEMMCutlass(void *A_data, void *B_data, void *C_data,
                                   QudaBLASParam param, QudaFieldLocation location)
{
    // Legacy CUTLASS 2.x path (FP32 complex, SM80, single GEMM only)
    using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<64, 64, 16>;
    using ShapeMMAWarp = cutlass::gemm::GemmShape<32, 32, 16>;
    using ShapeMMAOp = cutlass::gemm::GemmShape<16, 8, 8>;
    using SwizzleThreadBlock = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;
    using EpilogueOp = cutlass::epilogue::thread::LinearCombination<
        cutlass::complex<float>, 1, cutlass::complex<float>, cutlass::complex<float>>;
    constexpr int NumStages = 3;
    constexpr cutlass::ComplexTransform TransformA = cutlass::ComplexTransform::kNone;
    constexpr cutlass::ComplexTransform TransformB = cutlass::ComplexTransform::kConjugate;

    using Gemm = cutlass::gemm::device::GemmComplex<
        cutlass::complex<float>, cutlass::layout::RowMajor,
        cutlass::complex<float>, cutlass::layout::RowMajor,
        cutlass::complex<float>, cutlass::layout::RowMajor,
        cutlass::complex<float>, cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
        ShapeMMAThreadBlock, ShapeMMAWarp, ShapeMMAOp, EpilogueOp,
        SwizzleThreadBlock, NumStages, TransformA, TransformB,
        cutlass::arch::OpMultiplyAddComplexFastF32>;

    cutlass::complex<float> alpha({1.0f, 0.0f});
    cutlass::complex<float> beta({0.0f, 0.0f});
    cutlass::gemm::GemmCoord problem_size({param.m, param.n, param.k});

    cutlass::TensorRef<cutlass::complex<float> const, cutlass::layout::RowMajor> ref_A(
        (cutlass::complex<float>*)A_data, param.lda);
    cutlass::TensorRef<cutlass::complex<float> const, cutlass::layout::RowMajor> ref_B(
        (cutlass::complex<float>*)B_data, param.ldb);
    cutlass::TensorRef<cutlass::complex<float>, cutlass::layout::RowMajor> ref_C(
        (cutlass::complex<float>*)C_data, param.ldc);

    typename Gemm::Arguments arguments{problem_size, ref_A, ref_B, ref_C, ref_C,
                                        {alpha, beta}, 1};
    size_t workspace_size = Gemm::get_workspace_size(arguments);
    cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

    Gemm gemm_op;
    CUTLASS_CHECK(gemm_op.can_implement(arguments));
    CUTLASS_CHECK(gemm_op.initialize(arguments, workspace.get()));
    CUTLASS_CHECK(gemm_op());
    cudaDeviceSynchronize();

    return 8LL * param.m * param.n * param.k;
}

} // namespace quda

// ======================================================================
// No CUTLASS at all
// ======================================================================
#else

namespace ggp
{

void initCutlassApi(int, const char*, const char*, bool, int) {
    // No-op: CUTLASS not linked.
}

int herkBatchedCutlassQC(const void*, void*, int, int, int, cudaStream_t) {
    errorQuda("CUTLASS not linked, please revise build options");
    return -1;
}

int herkBatchedCutlassQC_FP16(const void*, void*, int, int, int, cudaStream_t) {
    errorQuda("CUTLASS not linked, please revise build options");
    return -1;
}

int herkBatchedCutlassFP16(const void*, void*, int, int, int, cudaStream_t) {
    errorQuda("CUTLASS not linked, please revise build options");
    return -1;
}

} // namespace ggp

namespace quda
{

long long gemmBatchedCutlass(void *A_data, void *B_data, void *C_data,
                              int M, int N, int K, int batch_count, bool conjugate_b)
{
    errorQuda("CUTLASS not linked, please revise build options");
    return -1;
}

long long stridedBatchGEMMCutlass(void *A_data, void *B_data, void *C_data,
                                   QudaBLASParam param, QudaFieldLocation location)
{
    errorQuda("CUTLASS not linked, please revise build options");
    return -1;
}

} // namespace quda

#endif
