/**
 * @file cutlass_gemm_api.h
 * @brief Lightweight PIMPL API for CUTLASS complex FP8 GEMM and HERK operations.
 *
 * This header has **no CUTLASS dependencies**. Consumers (e.g. dedisp, DSA-2000 FTD)
 * include only this header and link against @c libcutlass_gemm_api.a, which contains
 * the heavy CUTLASS template instantiations compiled in @c cutlass_gemm_api.cu.
 *
 * Supported architectures:
 *  - SM90  (Hopper)   -- FP8 GEMM + HERK + INT4 GEMM
 *  - SM100 (Blackwell datacenter) -- FP8/FP6/FP4 GEMM + HERK + INT4 HERK/GEMM
 *  - SM120 (Blackwell consumer)   -- FP8/FP6/FP4 GEMM + HERK + INT4 HERK/GEMM
 *
 * The target architecture is selected at compile time via CMake
 * (@c COMPLEX_FP8_TARGET_SM90 define).
 *
 * @see cutlass_gemm_api.cu for the PIMPL implementation.
 */
#pragma once
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <string>

/**
 * @namespace cutlass_gemm_api
 * @brief Public namespace for the CUTLASS complex GEMM/HERK library API.
 */
namespace cutlass_gemm_api {

// =====================================================================
/// @name Precision and Format Enums
/// @brief Runtime selection of input/compute/output precision and format.
// =====================================================================
///@{

/// @brief Input data precision.
enum class InputPrecision {
    FP16,   ///< Interleaved complex FP16 (__half pairs)
    INT4    ///< Packed INT4 sign-magnitude (1 byte per complex, DSA-2000 QC format)
};

/// @brief Tensor core compute precision (internal FP8/FP6/FP4 conversion).
enum class ComputePrecision {
    FP8,    ///< FP8 E4M3 -- lossless for INT4 input, highest bandwidth
    FP6,    ///< FP6 E3M2 -- lossless for INT4, 25% less bandwidth (SM100/120 only)
    FP4     ///< FP4 E2M1 -- lossy (5->4, 7->8), 2x throughput (SM100/120 only)
};

/// @brief Output data precision.
enum class OutputPrecision {
    FP16,   ///< FP16 output (may overflow at large K with large inputs)
    FP32    ///< FP32 output (native -- no FP16 intermediate)
};

/// @brief HERK output format.
enum class OutputFormat {
    PackedTriangle   ///< N*(N+1)/2 complex elements per batch (lower triangle)
};

/// @brief HERK dispatch mode for direct vs baseline kernel selection.
enum class HerkMode {
    Auto,           ///< K-adaptive: direct for small K, baseline for large K (default)
    ForceDirect,    ///< Always use direct single-launch PTX kernel
    ForceBaseline   ///< Always use multi-launch CUTLASS baseline path
};

/// @brief CUTLASS tile/cluster configuration for the baseline GEMM path.
///
/// Controls which kernel configuration the CUTLASS baseline path uses.
/// All valid configurations are compiled into a single binary; this enum
/// selects among them at runtime. The autotuner (tune=true) benchmarks
/// across all valid configs and overrides this setting with the fastest.
///
/// @note Does NOT affect the direct HERK kernel (which uses hand-written PTX).
///       Only impacts the baseline CUTLASS GEMM path.
enum class TileConfig {
    Auto,           ///< Arch-dependent default; autotuner overrides if tune=true (default)
    HerkOptimal,    ///< SM120: 128x64 3-stage (~82 KB); SM100: 128x128 3-stage; SM90: 128x256 Coop
    GemmOptimal,    ///< 128x128 auto-stages (GEMM-throughput-optimal)
    WideN,          ///< 128x256 auto-stages (large N benefit, SM100/SM90 only)
    Cluster1x2,     ///< 128x128 with 1x2 cluster (TMA N-multicast, SM100 only)
    Cluster2x2,     ///< 128x128 with 2x2 cluster (SM100 only)
    SmallTile,      ///< 128x64 auto-stages (SM120+SM100) or 64x128 Coop (SM90)
    Pingpong,       ///< 128x256 PingpongFP8FastAccum (SM90 only)
    PingpongSmall   ///< 128x128 PingpongFP8FastAccum (SM90 only)
};

/// @brief Power GEMM dispatch mode (direct kernel vs 4M CUTLASS + reduction).
///
/// Controls how gemm_prepared_power(), gemm_prepared_power_fp8(), and
/// gemm_prepared_power_int4() compute C_power = |Re(A*B^T)|^2 + |Im(A*B^T)|^2.
///
/// - **Direct**: Single fused PTX kernel. Low overhead, ~52 TFLOPS on GB10.
///   Best for small batch or small N.
/// - **4M**: 4 real CUTLASS sub-GEMMs (TMA + warp-specialized) producing
///   separate Re/Im FP32 outputs, followed by a lightweight power reduction
///   kernel. Reaches 112+ TFLOPS at batch>=4. Best for large problems.
enum class PowerMode {
    Auto,        ///< 4M for batch*N >= threshold, direct otherwise (default)
    ForceDirect, ///< Always use direct PTX kernel (fused power)
    Force4M      ///< Always use 4M CUTLASS sub-GEMMs + power reduction kernel
};

/// @brief Direct HERK kernel tile configuration (K_CHUNK x NR_BUFS).
///
/// Controls the shared memory tile size and pipeline depth of the direct
/// single-launch PTX HERK kernel. Affects SMEM usage and SM occupancy.
/// The autotuner (tune=true) benchmarks all variants and overrides this
/// setting with the fastest for each problem size.
///
/// @note Only affects the direct HERK kernel (HerkMode::ForceDirect or Auto
///       when K is small). Does NOT affect the CUTLASS baseline path.
enum class DirectTileConfig {
    Auto,     ///< Default (K64_B3 -- current production default)
    K32_B3,   ///< K_CHUNK=32, 3-buffer, 12 KB SMEM, occupancy 8 (memory-bound sweet spot)
    K64_B2,   ///< K_CHUNK=64, 2-buffer, 16 KB SMEM, occupancy 6
    K64_B3,   ///< K_CHUNK=64, 3-buffer, 24 KB SMEM, occupancy 4 (default)
    K128_B2   ///< K_CHUNK=128, 2-buffer, 32 KB SMEM, occupancy 3 (compute-bound sweet spot)
};

/// @brief GEMM dispatch mode: direct PTX kernel vs 4M CUTLASS sub-GEMMs.
///
/// Controls which kernel path is used for planar complex GEMM output.
/// The direct PTX kernel uses mma.sync.aligned.m16n8k32 with conjugate
/// permutation trick (immune to C7510 wgmma serialization).
/// The 4M path decomposes complex GEMM into 4 real FP8 sub-GEMMs via CUTLASS.
///
/// - **Direct**: Single-launch kernel, fused complex output, lower overhead.
///   Best for small-to-medium problems or when C7510 is a bottleneck.
/// - **4M**: 4 CUTLASS sub-GEMMs with TMA + warp-specialized scheduling.
///   Best for large problems where CUTLASS pipeline efficiency dominates.
///
/// @see set_gemm_mode(), PowerMode (for power detection output)
enum class GemmMode {
    Auto,        ///< Autotuner cache first, then roofline heuristic (default)
    ForceDirect, ///< Always use direct PTX kernel
    Force4M      ///< Always use 4 CUTLASS sub-GEMMs
};

///@}

/**
 * @class CutlassComplexGemm
 * @brief PIMPL-wrapped interface to CUTLASS complex-number GEMM and HERK.
 *
 * Decomposes complex GEMM into 4 real FP8 sub-GEMMs (Gauss's 4M algorithm)
 * and dispatches via CUTLASS 3.x grouped/batched kernels. HERK (Hermitian
 * Rank-K Update) auto-dispatches between a direct PTX kernel (small K) and
 * the CUTLASS baseline (large K).
 *
 * All device pointers must reside in GPU memory. The class manages internal
 * scratch buffers and CUDA streams; it is **not** thread-safe (use one
 * instance per thread or synchronize externally).
 *
 * @note Non-copyable, non-movable. Use a pointer or unique_ptr for ownership.
 */
class CutlassComplexGemm {
public:
    /// @brief Construct a new CutlassComplexGemm instance.
    ///
    /// Allocates internal state (CUDA streams, scratch buffers) on the
    /// current CUDA device. The first GEMM/HERK call may trigger additional
    /// lazy allocation.
    CutlassComplexGemm();

    /// @brief Destroy the instance and free all internal GPU resources.
    ~CutlassComplexGemm();

    // =================================================================
    /// @name Planar Complex GEMM
    /// @brief C = alpha * A * B + beta * C, complex-valued, batched.
    ///
    /// Uses **planar layout**: separate device buffers for real and imaginary
    /// parts. Layout is TN (A row-major, B row-major, C row-major). Internally
    /// B is transposed to column-major per CUTLASS convention.
    // =================================================================
    ///@{

    /// @brief Unified batched planar complex GEMM with runtime precision selection.
    ///
    /// C = alpha * A * B + beta * C (planar complex, batched, TN layout).
    /// Compute precision and output precision are selected at runtime via
    /// enum parameters.
    ///
    /// @param A_re     Device pointer to real part of A  [batch_count x M x K].
    ///                 When input==INT4: reinterpreted as const uint8_t*, M*K bytes/batch
    ///                 (packed INT4 sign-magnitude, high nibble=Re, low nibble=Im).
    /// @param A_im     Device pointer to imag part of A  [batch_count x M x K].
    ///                 Ignored when input==INT4.
    /// @param B_re     Device pointer to real part of B  [batch_count x N x K].
    /// @param B_im     Device pointer to imag part of B  [batch_count x N x K].
    /// @param C_re     Device pointer to real part of C  [batch_count x M x N] (in/out).
    ///                 __half* when output==FP16, float* when output==FP32.
    /// @param C_im     Device pointer to imag part of C  [batch_count x M x N] (in/out).
    ///                 __half* when output==FP16, float* when output==FP32.
    /// @param M        Number of rows of A and C.
    /// @param N        Number of columns of C (rows of B in TN layout).
    /// @param K        Inner dimension (columns of A, columns of B in TN layout).
    /// @param batch_count  Number of independent GEMM instances.
    /// @param compute  Tensor core compute precision (default FP8).
    /// @param output   Output data precision (default FP16).
    /// @param alpha    Scalar multiplier (default 1.0f).
    /// @param beta     Scalar accumulator for existing C (default 0.0f).
    /// @param stream   CUDA stream for async execution (default nullptr = default stream).
    /// @param tune     Auto-tune GemmConfig for this problem size (default true).
    ///                 On the first call for a given problem size, runs a ~15-40ms
    ///                 benchmark sweep. Results are cached to disk. Pass false to
    ///                 skip autotuning and use the default config.
    /// @param input    Input data precision (default FP16). When INT4, A_re is
    ///                 reinterpreted as packed QC data and A_im is ignored.
    /// @return 0 on success, non-zero CUTLASS status code on error.
    ///
    /// @note FP6 and FP4 compute are SM100/SM120 only. Throws @c std::runtime_error on SM90.
    /// @note FP6+FP16 and FP4+FP16 output combinations are supported internally.
    int gemm(
        const __half* A_re, const __half* A_im,
        const __half* B_re, const __half* B_im,
        void* C_re, void* C_im,
        int M, int N, int K, int batch_count,
        ComputePrecision compute = ComputePrecision::FP8,
        OutputPrecision output = OutputPrecision::FP16,
        float alpha = 1.0f, float beta = 0.0f,
        cudaStream_t stream = nullptr,
        bool tune = true,
        InputPrecision input = InputPrecision::FP16);

    ///@}

    // =================================================================
    /// @name Prepare/Execute Pattern
    /// @brief Pre-convert B once, reuse across multiple GEMM calls.
    // =================================================================
    ///@{

    /// @brief Pre-convert B matrix to internal format for reuse.
    ///
    /// Converts FP16 B to FP8/MXFP once. Subsequent calls to
    /// gemm_prepared() skip B conversion, reducing per-call overhead
    /// by 120-265 ms at typical problem sizes.
    ///
    /// @param B_re     Device pointer to real part of B  [batch_count x N x K].
    /// @param B_im     Device pointer to imag part of B  [batch_count x N x K].
    /// @param N        Number of rows of B (columns of C).
    /// @param K        Inner dimension.
    /// @param batch_count  Number of independent GEMM instances.
    /// @param compute  Compute precision for the prepared B data.
    /// @param stream   CUDA stream (default nullptr).
    ///
    /// @note On SM90, only ComputePrecision::FP8 is supported.
    /// @see gemm_prepared()
    void prepare_b(
        const __half* B_re, const __half* B_im,
        int N, int K, int batch_count,
        ComputePrecision compute,
        cudaStream_t stream = nullptr);

    /// @brief Execute GEMM with pre-prepared B.
    ///
    /// Uses the precision and pre-converted B data set by prepare_b().
    /// Only converts A per-call. Workspaces are lazily pre-allocated on
    /// the first call, eliminating @c cudaMallocAsync overhead thereafter.
    ///
    /// @param A_re     Device pointer to real part of A  [batch_count x M x K].
    ///                 When input==INT4: reinterpreted as const uint8_t*, M*K bytes/batch
    ///                 (packed INT4 sign-magnitude, high nibble=Re, low nibble=Im).
    /// @param A_im     Device pointer to imag part of A  [batch_count x M x K].
    ///                 Ignored when input==INT4.
    /// @param C_re     Device pointer to real part of C  [batch_count x M x N] (out).
    ///                 __half* when output==FP16, float* when output==FP32.
    /// @param C_im     Device pointer to imag part of C  [batch_count x M x N] (out).
    ///                 __half* when output==FP16, float* when output==FP32.
    /// @param M        Number of rows of A and C.
    /// @param N        Number of columns of C (must match prepare_b()).
    /// @param K        Inner dimension (must match prepare_b()).
    /// @param batch_count  Number of instances (must match prepare_b()).
    /// @param output   Output data precision (default FP32).
    /// @param alpha    Scalar multiplier (default 1.0f).
    /// @param beta     Scalar accumulator (default 0.0f).
    /// @param stream   CUDA stream (default nullptr).
    /// @param tune     Auto-tune GemmConfig for this problem size (default true).
    ///                 On the first call for a given problem size, runs a ~15-40ms
    ///                 benchmark sweep. Results are cached to disk. Pass false to
    ///                 skip autotuning and use the default config.
    /// @param input    Input data precision (default FP16). When INT4, A_re is
    ///                 reinterpreted as packed QC data and A_im is ignored.
    /// @return 0 on success, non-zero on error.
    ///
    /// @pre prepare_b() must have been called with matching N, K, batch_count.
    /// @see prepare_b()
    int gemm_prepared(
        const __half* A_re, const __half* A_im,
        void* C_re, void* C_im,
        int M, int N, int K, int batch_count,
        OutputPrecision output = OutputPrecision::FP32,
        float alpha = 1.0f, float beta = 0.0f,
        cudaStream_t stream = nullptr,
        bool tune = true,
        InputPrecision input = InputPrecision::FP16);

    /// @brief Execute prepared GEMM via direct single-launch kernel (FP32 output).
    ///
    /// Uses the direct complex GEMM kernel with conjugate permutation trick
    /// for ~1.5-2x speedup over the 4M sub-GEMM baseline. B must have been
    /// prepared with ComputePrecision::FP8. Only converts A per-call.
    ///
    /// @param A_re     Device pointer to real part of A  [batch_count x M x K].
    /// @param A_im     Device pointer to imag part of A  [batch_count x M x K].
    /// @param C_re     Device pointer to real part of C  [batch_count x M x N] (out, float*).
    /// @param C_im     Device pointer to imag part of C  [batch_count x M x N] (out, float*).
    /// @param M        Number of rows of A and C.
    /// @param N        Number of columns of C (must match prepare_b()).
    /// @param K        Inner dimension (must match prepare_b()).
    /// @param batch_count  Number of instances (must match prepare_b()).
    /// @param alpha    Scalar multiplier (default 1.0f).
    /// @param beta     Scalar accumulator (default 0.0f).
    /// @param stream   CUDA stream (default nullptr).
    /// @return 0 on success, non-zero on error.
    ///
    /// @pre prepare_b() must have been called with ComputePrecision::FP8.
    /// @see prepare_b()
    int gemm_prepared_direct(
        const __half* A_re, const __half* A_im,
        float* C_re, float* C_im,
        int M, int N, int K, int batch_count,
        float alpha = 1.0f, float beta = 0.0f,
        cudaStream_t stream = nullptr);

    /// @brief Execute prepared GEMM with fused power detection (direct kernel).
    ///
    /// Computes C_power = alpha * (Re(A*B^T)² + Im(A*B^T)²) + beta * C_power
    /// in a single kernel launch. Eliminates 4 FP32 intermediate buffers
    /// and the separate power computation pass.
    ///
    /// @param A_re     Device pointer to real part of A  [batch_count x M x K].
    /// @param A_im     Device pointer to imag part of A  [batch_count x M x K].
    /// @param C_power  Device pointer to power output  [batch_count x M x N] (float*, in/out).
    /// @param M        Number of rows of A and C.
    /// @param N        Number of columns of C (must match prepare_b()).
    /// @param K        Inner dimension (must match prepare_b()).
    /// @param batch_count  Number of instances (must match prepare_b()).
    /// @param alpha    Scalar multiplier (default 1.0f).
    /// @param beta     Scalar accumulator (default 0.0f).
    /// @param stream   CUDA stream (default nullptr).
    /// @return 0 on success, non-zero on error.
    ///
    /// @pre prepare_b() must have been called with ComputePrecision::FP8.
    /// @see prepare_b()
    int gemm_prepared_power(
        const __half* A_re, const __half* A_im,
        float* C_power,
        int M, int N, int K, int batch_count,
        float alpha = 1.0f, float beta = 0.0f,
        cudaStream_t stream = nullptr);

    /// @brief Execute prepared GEMM with pre-cast FP8 A and fused power detection.
    ///
    /// Same as gemm_prepared_power() but accepts pre-cast FP8 interleaved A,
    /// skipping the internal FP16→FP8 cast. Use when the caller has already
    /// converted voltages to FP8 interleaved format (e.g. via the
    /// qc_to_fp8_interleaved_polsplit kernel).
    ///
    /// @param A_fp8_interleaved  Device pointer to FP8 interleaved A
    ///                           [batch_count x M x 2*K] bytes (Re, Im pairs).
    /// @param C_power  Device pointer to power output  [batch_count x M x N] (float*, in/out).
    /// @param M        Number of rows of A and C.
    /// @param N        Number of columns of C (must match prepare_b()).
    /// @param K        Inner dimension (must match prepare_b()).
    /// @param batch_count  Number of instances (must match prepare_b()).
    /// @param alpha    Scalar multiplier (default 1.0f).
    /// @param beta     Scalar accumulator (default 0.0f).
    /// @param stream   CUDA stream (default nullptr).
    /// @return 0 on success, non-zero on error.
    ///
    /// @pre prepare_b() must have been called with ComputePrecision::FP8.
    /// @see prepare_b(), gemm_prepared_power()
    int gemm_prepared_power_fp8(
        const void* A_fp8_interleaved,
        float* C_power,
        int M, int N, int K, int batch_count,
        float alpha = 1.0f, float beta = 0.0f,
        cudaStream_t stream = nullptr);

    /// @brief Execute prepared GEMM with INT4 A input and fused power detection.
    ///
    /// Converts INT4 QC data to FP8 interleaved internally, then runs the
    /// direct GEMM kernel with fused power detection. Eliminates the need
    /// for consumers to maintain custom QC->FP8 conversion kernels.
    ///
    /// @param A_int4    Device pointer to packed INT4 QC data [batch_count x M x K] bytes.
    /// @param C_power   Device pointer to power output [batch_count x M x N] (float*, in/out).
    /// @param M, N, K, batch_count  Problem dimensions (must match prepare_b()).
    /// @param alpha, beta  Scalars (default 1.0f, 0.0f).
    /// @param stream    CUDA stream (default nullptr).
    /// @return 0 on success, non-zero on error.
    ///
    /// @pre prepare_b() must have been called with ComputePrecision::FP8.
    /// @see prepare_b(), gemm_prepared_power_fp8()
    int gemm_prepared_power_int4(
        const void* A_int4, float* C_power,
        int M, int N, int K, int batch_count,
        float alpha = 1.0f, float beta = 0.0f,
        cudaStream_t stream = nullptr);

    /// @brief Execute prepared direct GEMM with INT4 A input (FP32 output).
    ///
    /// Converts INT4 QC data to FP16 planar internally, then runs the
    /// direct complex GEMM kernel for separate Re/Im FP32 output.
    ///
    /// @param A_int4    Device pointer to packed INT4 QC data [batch_count x M x K] bytes.
    /// @param C_re      Device pointer to real part of C [batch_count x M x N] (out, float*).
    /// @param C_im      Device pointer to imag part of C [batch_count x M x N] (out, float*).
    /// @param M, N, K, batch_count  Problem dimensions (must match prepare_b()).
    /// @param alpha, beta  Scalars (default 1.0f, 0.0f).
    /// @param stream    CUDA stream (default nullptr).
    /// @return 0 on success, non-zero on error.
    ///
    /// @pre prepare_b() must have been called with ComputePrecision::FP8.
    /// @see prepare_b(), gemm_prepared_direct()
    int gemm_prepared_direct_int4(
        const void* A_int4,
        float* C_re, float* C_im,
        int M, int N, int K, int batch_count,
        float alpha = 1.0f, float beta = 0.0f,
        cudaStream_t stream = nullptr);

    /// @brief Execute prepared GEMM with batch-fused M dimension.
    ///
    /// Fuses @p fuse_factor consecutive batch elements along the M dimension,
    /// computing a single GEMM of size (M * fuse_factor) x N x K for each
    /// group of @p fuse_factor batches. This improves tensor core utilization
    /// for small-M problems (e.g. M=128 beamformer, M=32 dedisp) by reducing
    /// wave quantization waste.
    ///
    /// The input A arrays must be contiguous along the batch dimension with
    /// stride M*K per batch element (standard batch layout). The output C
    /// arrays must be contiguous with stride M*N per batch element. The fused
    /// GEMM writes to M_fused*N elements of C, which maps back to fuse_factor
    /// consecutive M*N output blocks.
    ///
    /// @param A_re     Device pointer to real part of A  [batch_count x M x K].
    /// @param A_im     Device pointer to imag part of A  [batch_count x M x K].
    /// @param C_re     Device pointer to real part of C  [batch_count x M x N] (out, float*).
    /// @param C_im     Device pointer to imag part of C  [batch_count x M x N] (out, float*).
    /// @param M        Number of rows of A and C per batch element.
    /// @param N        Number of columns of C (must match prepare_b()).
    /// @param K        Inner dimension (must match prepare_b()).
    /// @param batch_count  Total number of batch elements (must be divisible by fuse_factor).
    /// @param fuse_factor  Number of batch elements to fuse along M.
    ///                     0 = auto-select targeting M_fused in [128, 1024].
    ///                     Must divide batch_count evenly.
    /// @param output   Output data precision (default FP32).
    /// @param alpha    Scalar multiplier (default 1.0f).
    /// @param beta     Scalar accumulator (default 0.0f).
    /// @param stream   CUDA stream (default nullptr).
    /// @return 0 on success, non-zero on error.
    ///
    /// @pre prepare_b() must have been called. B must be uniform across all
    ///      batch elements (same B for each). The prepared batch_count must
    ///      be >= 1 (only one copy of B is used).
    /// @note Only valid when B is the same for all batch elements (e.g. voltage
    ///       beamformer weights). If B varies per batch, use gemm_prepared().
    /// @see prepare_b(), gemm_prepared()
    int gemm_prepared_fused(
        const __half* A_re, const __half* A_im,
        void* C_re, void* C_im,
        int M, int N, int K, int batch_count,
        int fuse_factor = 0,
        OutputPrecision output = OutputPrecision::FP32,
        float alpha = 1.0f, float beta = 0.0f,
        cudaStream_t stream = nullptr);

    ///@}

    // =================================================================
    /// @name Diagnostics
    // =================================================================
    ///@{

    /// @brief Diagnostic: run a single real FP6 GEMM and verify correctness.
    ///
    /// Runs FP6 E3M2 GEMM with both FP16 and FP32 output, compares each
    /// against a CPU reference, and prints results to stderr.
    ///
    /// @param M     Number of rows.
    /// @param N     Number of columns.
    /// @param K     Inner dimension.
    /// @param stream CUDA stream (default nullptr).
    /// @return 0 if both outputs match reference within tolerance, non-zero on failure.
    /// @note SM100/SM120 only. Returns -1 on SM90.
    int debug_fp6_real_gemm(int M, int N, int K, cudaStream_t stream = nullptr);

    ///@}

    // =================================================================
    /// @name Strategy Autotuning
    /// @brief Configuration for automatic strategy-level tuning.
    // =================================================================
    ///@{

    /// @brief Set the path for the strategy tune cache file.
    ///
    /// The strategy cache stores the optimal internal strategy (HerkMode,
    /// HerkStrategy, CUDA graph, PersistentMode, herk_graph, batch_tiling)
    /// for each (N, K, batch_count, precision) problem size, determined by
    /// runtime benchmarking.
    ///
    /// Default: auto-generated as
    /// @c cutlass_strategy_cache_{build_fingerprint}.txt in the current
    /// directory (e.g. @c cutlass_strategy_cache_sm120_m128x128x128_stg3.txt).
    /// Each build configuration gets its own file.
    ///
    /// @param path  File path for the strategy cache. The file is created
    ///              on first write if it does not exist. Overrides the
    ///              auto-generated filename.
    void set_tune_cache_path(const std::string& path);

    /// @brief Set the HERK strategy-level autotuning verbosity.
    ///
    /// Controls output from the HERK strategy autotuning sweep, which benchmarks
    /// HerkMode x HerkStrategy x CUDA graph x PersistentMode combinations.
    ///
    /// @param level  Verbosity level:
    ///   - 0: Silent (no output -- production use).
    ///   - 1: Summary (cache hit/miss + winner only, 2-3 lines).
    ///   - 2: Sweep detail (per-candidate timing during autotune).
    ///   - 3: Full diagnostics (sweep + skip reasons).
    ///
    /// Default: 1 (summary).
    /// @note Strategy tuning is triggered by herk(..., tune=true).
    void set_strategy_tune_verbosity(int level);

    /// @brief Set the GEMM strategy-level autotuning verbosity.
    ///
    /// Controls output from the GEMM strategy autotuning sweep, which benchmarks
    /// all valid GemmConfig tile/cluster/schedule configurations.
    ///
    /// @param level  Verbosity level (same as HERK strategy verbosity).
    /// Default: 1 (summary).
    /// @note GEMM tuning is triggered by gemm(..., tune=true).
    void set_gemm_tune_verbosity(int level);

    /// @brief Set the path for the GEMM strategy tune cache file.
    ///
    /// The GEMM strategy cache stores the optimal GemmConfig for each
    /// (M, N, K, batch_count, precision) problem size. GEMM entries share
    /// the same file as HERK strategy entries, distinguished by a "GEMM"
    /// prefix on each data line.
    ///
    /// Default: same file as the HERK strategy cache
    /// (@c cutlass_strategy_cache_{build_fingerprint}.txt).
    ///
    /// @param path  File path for the GEMM strategy cache.
    void set_gemm_tune_cache_path(const std::string& path);

    ///@}

    // =================================================================
    /// @name Kernel-Level Tuning
    /// @brief Configuration for automatic blockDim/gridDim tuning of
    ///        overhead kernels (cast, pack, deinterleave, MXFP preprocess).
    ///
    /// Kernel-level tuning optimizes the launch parameters (blockDim and
    /// gridDim) of the non-GEMM kernels used for data format conversion
    /// and preprocessing. This benefits both the GEMM and HERK paths.
    ///
    /// Tuning results are cached to a file and reused across sessions.
    /// The first call with tuning enabled runs a ~30-60 second sweep;
    /// subsequent calls are free (cache hit).
    // =================================================================
    ///@{

    /// @brief Set the kernel-level tuning verbosity.
    ///
    /// @param level  Verbosity level:
    ///   - 0: Silent (no tuning, no output).
    ///   - 1: Show cached/default params per kernel (no tuning).
    ///   - 2: Tune kernels and show one-line summary per kernel.
    ///   - 3: Tune kernels with full per-config timing detail.
    ///
    /// Levels >= 2 trigger the tuning sweep on the first GEMM/HERK call.
    /// Default: 0 (silent).
    void set_kernel_tune_verbosity(int level);

    /// @brief Set the file path for the kernel-level tune cache.
    ///
    /// Default: "cutlass_kernel_cache_{build_fingerprint}.txt" in the current
    /// directory (one file per build configuration, mirroring strategy cache).
    ///
    /// @param path  File path for the kernel tune cache. Created on first
    ///              write if it does not exist.
    void set_kernel_tune_cache_path(const std::string& path);

    ///@}

    // =================================================================
    /// @name HERK Pipeline Init/End
    /// @brief Pre-allocate buffers for repeated HERK calls.
    // =================================================================
    ///@{

    /// @brief Pre-allocate internal buffers for repeated HERK calls.
    ///
    /// Eliminates per-call cudaMallocAsync/cudaFreeAsync overhead for the
    /// INT4->FP16 conversion buffer.
    /// Subsequent herk() calls with dimensions <= the init'd sizes
    /// reuse these buffers instead of allocating per call.
    ///
    /// Also triggers internal lazy allocations (streams, mode selection)
    /// so the first real HERK call has no hidden setup cost.
    ///
    /// @param N           Matrix dimension (antennas).
    /// @param K           Inner dimension (time samples).
    /// @param batch_count Number of batch elements (channels).
    /// @param stream      Optional CUDA stream for allocation.
    /// @note Call end_herk() to free pre-allocated buffers.
    /// @note Safe to call multiple times; previous buffers are freed first.
    void init_herk(int N, int K, int batch_count, cudaStream_t stream = nullptr);

    /// @brief Free pre-allocated HERK buffers.
    ///
    /// After this call, herk() reverts to per-call allocation.
    /// Safe to call even if init_herk() was never called (no-op).
    /// Also called automatically by the destructor.
    void end_herk();

    ///@}

    // =================================================================
    /// @name HERK -- Hermitian Rank-K Update
    /// @brief Computes C = alpha * A * A^H + beta * C (batched).
    ///
    /// Unified API with runtime precision and format selection. Auto-dispatches
    /// between a direct PTX kernel (K <= threshold) and the CUTLASS baseline
    /// path (K > threshold) for optimal performance.
    // =================================================================
    ///@{

    /// @brief Set the HERK dispatch mode (direct vs baseline kernel selection).
    ///
    /// By default, HerkMode::Auto selects the direct single-launch PTX kernel
    /// for small K values and the multi-launch CUTLASS baseline for large K.
    /// Use ForceDirect or ForceBaseline to override the automatic selection.
    ///
    /// @param mode  The desired HERK dispatch mode.
    /// @see HerkMode
    void set_herk_mode(HerkMode mode);

    /// @brief Set the CUTLASS tile/cluster configuration for the baseline GEMM path.
    ///
    /// Controls which compiled-in kernel configuration is used for CUTLASS
    /// baseline GEMMs. TileConfig::Auto uses the architecture-dependent default
    /// (overridden by the autotuner when tune=true).
    ///
    /// @param config  The desired tile/cluster configuration.
    /// @see TileConfig
    void set_tile_config(TileConfig config);

    /// @brief Set the direct HERK kernel tile configuration (K_CHUNK x NR_BUFS).
    ///
    /// Controls shared memory tile size and pipeline depth for the direct
    /// single-launch PTX kernel. DirectTileConfig::Auto uses the default
    /// (K64_B3, overridden by autotuner when tune=true).
    ///
    /// @param config  The desired direct tile configuration.
    /// @see DirectTileConfig
    void set_direct_tile_config(DirectTileConfig config);

    /// @brief Enable CUDA graph capture for the direct HERK path.
    ///
    /// When enabled, the first herk() call with HerkMode::ForceDirect
    /// captures the full pipeline (INT4->FP8 cast + HERK kernel) as a
    /// CUDA graph. Subsequent calls with matching parameters replay the
    /// graph, eliminating per-call kernel launch overhead (~20us savings).
    ///
    /// @param enable  true to enable graph capture/replay, false to disable.
    /// @pre init_herk() should be called first to pre-allocate buffers
    ///      (cudaMallocAsync is not graph-capturable).
    void set_direct_herk_graph(bool enable);

    /// @brief Set the power GEMM dispatch mode (direct vs 4M path).
    ///
    /// Controls how gemm_prepared_power*() methods compute fused power
    /// detection. PowerMode::Auto (default) selects 4M for large problems
    /// (batch_count >= 4 && N >= 256) and direct otherwise.
    ///
    /// @param mode  The desired power dispatch mode.
    /// @see PowerMode
    void set_power_mode(PowerMode mode);

    /// @brief Set the planar GEMM dispatch mode (direct vs 4M path).
    ///
    /// Controls how gemm_prepared() dispatches planar complex GEMM output.
    /// GemmMode::Auto (default) checks the autotuner cache first, then falls
    /// back to a roofline heuristic. ForceDirect always uses the direct PTX
    /// kernel; Force4M always uses 4 CUTLASS sub-GEMMs.
    ///
    /// The direct kernel is immune to C7510 wgmma serialization (Hopper)
    /// and has lower launch overhead for small-to-medium problems.
    ///
    /// @param mode  The desired GEMM dispatch mode.
    /// @see GemmMode
    void set_gemm_mode(GemmMode mode);

    /// @brief Get the current planar GEMM dispatch mode.
    /// @return The current GemmMode setting.
    GemmMode get_gemm_mode() const;

    /// @brief Enable or disable automatic GEMM/HERK config autotuning.
    ///
    /// When enabled (default), the first GEMM/HERK call for each problem
    /// size runs a ~15-40ms benchmark sweep and caches the optimal
    /// configuration. Set to false to use default configs without
    /// autotuning overhead.
    ///
    /// This controls the internal autotuning for power GEMM methods
    /// (which don't have an explicit tune parameter). For gemm(), herk(),
    /// and gemm_prepared(), use their tune parameter instead.
    ///
    /// @param enable  true to enable autotuning (default), false to disable.
    void set_auto_tune(bool enable);

    /// @brief Unified batched HERK with runtime precision and format selection.
    ///
    /// C = alpha * A * A^H + beta * C (batched)
    ///
    /// @param A       Device pointer to input data.
    ///                - FP16: [batch x N x K x 2] __half (interleaved complex)
    ///                - INT4: [batch x N x K] bytes (packed sign-magnitude)
    /// @param C       Device pointer to output.
    ///                - PackedTriangle + FP16: [batch x N*(N+1)] __half
    ///                - PackedTriangle + FP32: [batch x N*(N+1)] float
    /// @param N       Matrix dimension (antennas/receivers).
    /// @param K       Inner dimension (time samples).
    /// @param batch_count  Number of independent HERKs.
    /// @param input   Input data precision (default FP16).
    /// @param compute Tensor core compute precision (default FP8).
    /// @param output  Output data precision (default FP16).
    /// @param format  Output format (default PackedTriangle).
    /// @param alpha   Scalar multiplier (default 1.0f).
    /// @param beta    Scalar accumulator (default 0.0f).
    /// @param stream  CUDA stream (default nullptr).
    /// @param tune    Auto-tune strategy for this problem size (default true).
    ///                On the first call for a given problem size, runs a ~5-second
    ///                benchmark sweep. Results are cached to disk. Pass false to
    ///                skip autotuning and use the default strategy.
    /// @return 0 on success, non-zero on error.
    ///
    /// @note FP6/FP4 compute are SM100/SM120 only. Throws on SM90.
    /// @note Only PackedTriangle output is currently supported.
    int herk(
        const void* A,
        void* C,
        int N, int K, int batch_count,
        InputPrecision input = InputPrecision::FP16,
        ComputePrecision compute = ComputePrecision::FP8,
        OutputPrecision output = OutputPrecision::FP16,
        OutputFormat format = OutputFormat::PackedTriangle,
        float alpha = 1.0f, float beta = 0.0f,
        cudaStream_t stream = nullptr,
        bool tune = true);

    ///@}

    // =================================================================
    /// @name Build and System Information
    // =================================================================
    ///@{

    /// @brief Print GPU system info and compile-time build configuration to stdout.
    ///
    /// Prints CUDA version, GPU name, SM count, peak TFLOPS, memory bandwidth,
    /// and all compile-time feature flags (architecture, tile shapes, pipeline
    /// stages, precision features, codegen flags). Useful for debugging and
    /// reproducing benchmark results.
    ///
    /// @note This is a static method -- no instance needed.
    static void print_build_info();

    ///@}

    /// @cond INTERNAL
    CutlassComplexGemm(const CutlassComplexGemm&) = delete;
    CutlassComplexGemm& operator=(const CutlassComplexGemm&) = delete;
    /// @endcond

private:
    struct Impl;  ///< Opaque PIMPL handle (defined in cutlass_gemm_api.cu).
    Impl* impl_;  ///< Pointer to implementation (owns CUTLASS state + scratch).
};

} // namespace cutlass_gemm_api
