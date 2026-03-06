/**
 * @file cutlass_gemm_api.cu
 * @brief PIMPL implementation for the CUTLASS complex FP8 GEMM / HERK API.
 *
 * This translation unit includes the heavy CUTLASS template headers
 * (~15 min compile). External consumers link against the compiled static
 * library (@c libcutlass_gemm_api.a) and include only the lightweight
 * @c cutlass_gemm_api.h header.
 *
 * Architecture selection is compile-time: define @c COMPLEX_FP8_TARGET_SM90
 * (set by CMake when @c ARCH_FAMILY == "hopper") to use the SM90 (Hopper)
 * implementation; otherwise the SM100/SM120 (Blackwell) implementation is used.
 *
 * @see cutlass_gemm_api.h for the public API documentation.
 */
#include "cutlass_gemm_api.h"

#ifdef COMPLEX_FP8_TARGET_SM90
  #include "gemm_complex_fp8.hpp"
  using GemmImpl = gemm_complex_fp8::GemmComplexFP8;
  using namespace gemm_complex_fp8;
  /// @brief Alias to disambiguate internal HerkMode from cutlass_gemm_api::HerkMode.
  using InternalHerkMode = gemm_complex_fp8::HerkMode;
  /// @brief Alias for internal PersistentMode.
  using InternalPersistentMode = gemm_complex_fp8::PersistentMode;
  /// @brief Alias for internal GemmConfig.
  using InternalGemmConfig = gemm_complex_fp8::GemmConfig;
  /// @brief Alias for internal DirectHerkConfig.
  using InternalDirectHerkConfig = gemm_complex_fp8::DirectHerkConfig;
  /// @brief Alias for internal HerkTileSize.
  using InternalHerkTileSize = gemm_complex_fp8::HerkTileSize;
  /// @brief Alias for internal HerkPipelineMode.
  using InternalHerkPipelineMode = gemm_complex_fp8::HerkPipelineMode;
  /// @brief SM90 has no ComputePrecision enum (FP8 only) — use void sentinel.
  using InternalComputePrecision = void;
#else
  #include "gemm_complex_sm100.hpp"
  using GemmImpl = gemm_complex_sm100::GemmComplexSm100;
  using namespace gemm_complex_sm100;
  /// @brief Alias to disambiguate internal HerkMode from cutlass_gemm_api::HerkMode.
  using InternalHerkMode = gemm_complex_sm100::HerkMode;
  /// @brief Alias for internal PersistentMode.
  using InternalPersistentMode = gemm_complex_sm100::PersistentMode;
  /// @brief Alias for internal GemmConfig.
  using InternalGemmConfig = gemm_complex_sm100::GemmConfig;
  /// @brief Alias for internal DirectHerkConfig.
  using InternalDirectHerkConfig = gemm_complex_sm100::DirectHerkConfig;
  /// @brief Alias for internal HerkTileSize.
  using InternalHerkTileSize = gemm_complex_sm100::HerkTileSize;
  /// @brief Alias for internal HerkPipelineMode.
  using InternalHerkPipelineMode = gemm_complex_sm100::HerkPipelineMode;
  /// @brief Alias for internal ComputePrecision.
  using InternalComputePrecision = gemm_complex_sm100::ComputePrecision;
#endif

#include "strategy_cache.hpp"
#include "system_info.hpp"

#include <vector>
#include <cstdlib>
#include <cmath>
#include <stdexcept>

namespace cutlass_gemm_api {

// =====================================================================
// Forward declarations for INT4 conversion kernels (defined below)
// =====================================================================
static void cast_int4_to_fp8_interleaved(
    const uint8_t* input, __nv_fp8_e4m3* output,
    int64_t num_complex_elements, cudaStream_t stream);
static void cast_int4_to_fp16_planar(
    const uint8_t* input, __half* out_real, __half* out_imag,
    int64_t num_complex_elements, cudaStream_t stream);

// =====================================================================
// Internal precision mapping helpers
// =====================================================================

#ifndef COMPLEX_FP8_TARGET_SM90
/// @brief Map public ComputePrecision enum to internal (SM100) ComputePrecision.
///
/// The public API uses short names (FP8, FP6, FP4) while the internal
/// gemm_complex_sm100 namespace uses explicit format names (FP8_E4M3, etc.).
///
/// @param p  Public ComputePrecision value.
/// @return Corresponding internal ComputePrecision value.
static gemm_complex_sm100::ComputePrecision map_compute(ComputePrecision p) {
    switch (p) {
    case ComputePrecision::FP6: return gemm_complex_sm100::ComputePrecision::FP6_E3M2;
    case ComputePrecision::FP4: return gemm_complex_sm100::ComputePrecision::FP4_E2M1;
    default:                    return gemm_complex_sm100::ComputePrecision::FP8_E4M3;
    }
}
#endif

/// @brief Map public HerkMode enum to internal (SM90/SM100) HerkMode.
///
/// Both the SM90 and SM100/SM120 namespaces define the same HerkMode enum
/// (Auto, ForceDirect, ForceBaseline). InternalHerkMode aliases the correct
/// internal type to avoid collision with cutlass_gemm_api::HerkMode.
///
/// @param m  Public cutlass_gemm_api::HerkMode value.
/// @return Corresponding internal HerkMode value.
static InternalHerkMode map_herk_mode(cutlass_gemm_api::HerkMode m) {
    switch (m) {
    case cutlass_gemm_api::HerkMode::ForceDirect:  return InternalHerkMode::ForceDirect;
    case cutlass_gemm_api::HerkMode::ForceBaseline: return InternalHerkMode::ForceBaseline;
    default: return InternalHerkMode::Auto;
    }
}

/// @brief Map public ComputePrecision to an integer cache key for strategy tuning.
///
/// @param p  Public ComputePrecision value.
/// @return Integer key: 0=FP8, 1=FP6, 2=FP4.
static int precision_cache_key(ComputePrecision p) {
    switch (p) {
    case ComputePrecision::FP8: return 0;
    case ComputePrecision::FP6: return 1;
    case ComputePrecision::FP4: return 2;
    default: return 0;
    }
}

/// @brief Map public TileConfig enum to internal GemmConfig.
///
/// TileConfig provides architecture-agnostic names (HerkOptimal, GemmOptimal,
/// etc.) that map to the correct internal GemmConfig based on the compiled
/// architecture. TileConfig::Auto maps to the arch-dependent default.
///
/// @param tc  Public cutlass_gemm_api::TileConfig value.
/// @return Corresponding internal GemmConfig value.
static InternalGemmConfig map_tile_config(cutlass_gemm_api::TileConfig tc) {
    switch (tc) {
    case cutlass_gemm_api::TileConfig::Auto:
        return InternalGemmConfig::Default;
    case cutlass_gemm_api::TileConfig::HerkOptimal:
#if defined(COMPLEX_FP8_TARGET_SM90)
        return InternalGemmConfig::FP8_T128x256_C1x1_Coop;
#elif defined(COMPLEX_FP8_SM100_TARGET_SM120)
        return InternalGemmConfig::FP8_T128x64_C1x1_S3;
#else
        return InternalGemmConfig::FP8_T128x128_C1x1_S3;
#endif
    case cutlass_gemm_api::TileConfig::GemmOptimal:
        return InternalGemmConfig::FP8_T128x128_C1x1_SAuto;
    case cutlass_gemm_api::TileConfig::WideN:
#if defined(COMPLEX_FP8_TARGET_SM90)
        return InternalGemmConfig::FP8_T128x256_C1x1_Coop;
#else
        return InternalGemmConfig::FP8_T128x256_C1x1_SAuto;
#endif
    case cutlass_gemm_api::TileConfig::Cluster1x2:
        return InternalGemmConfig::FP8_T128x128_C1x2;
    case cutlass_gemm_api::TileConfig::Cluster2x2:
        return InternalGemmConfig::FP8_T128x128_C2x2;
    case cutlass_gemm_api::TileConfig::SmallTile:
#if defined(COMPLEX_FP8_TARGET_SM90)
        return InternalGemmConfig::FP8_T64x128_C1x1_Coop;
#else
        return InternalGemmConfig::FP8_T128x64_C1x1_SAuto;
#endif
    case cutlass_gemm_api::TileConfig::Pingpong:
        return InternalGemmConfig::FP8_T128x256_C1x1_PP;
    case cutlass_gemm_api::TileConfig::PingpongSmall:
        return InternalGemmConfig::FP8_T128x128_C1x1_PP;
    default:
        return InternalGemmConfig::Default;
    }
}

/// @brief Map public DirectTileConfig to internal DirectHerkConfig.
///
/// DirectTileConfig::Auto maps to DirectHerkConfig::Default (K64_B3).
///
/// @param dtc  Public cutlass_gemm_api::DirectTileConfig value.
/// @return Corresponding internal DirectHerkConfig value.
static InternalDirectHerkConfig map_direct_tile_config(cutlass_gemm_api::DirectTileConfig dtc) {
    switch (dtc) {
    case cutlass_gemm_api::DirectTileConfig::Auto:
        return InternalDirectHerkConfig::Default;
    case cutlass_gemm_api::DirectTileConfig::K32_B3:
        return InternalDirectHerkConfig::K32_B3;
    case cutlass_gemm_api::DirectTileConfig::K64_B2:
        return InternalDirectHerkConfig::K64_B2;
    case cutlass_gemm_api::DirectTileConfig::K64_B3:
        return InternalDirectHerkConfig::K64_B3;
    case cutlass_gemm_api::DirectTileConfig::K128_B2:
        return InternalDirectHerkConfig::K128_B2;
    default:
        return InternalDirectHerkConfig::Default;
    }
}

// =====================================================================
// Impl struct
// =====================================================================

/// @brief LRU cache entry for a captured CUDA graph of the direct HERK pipeline.
///
/// Caches the full INT4→FP8 cast + direct HERK kernel as a single CUDA graph.
/// Keyed on (N, K, batch, alpha, beta, A_ptr, C_ptr, input, output) so that
/// graph replay is only used when all parameters match exactly.
struct DirectHerkGraphEntry {
    cudaGraphExec_t exec = nullptr;
    int cached_N = 0, cached_K = 0, cached_batch = 0;
    float cached_alpha = 0, cached_beta = 0;
    const void* cached_A = nullptr;
    void* cached_C = nullptr;
    int cached_input = 0, cached_output = 0;

    bool matches(int N, int K, int batch, float a, float b,
                 const void* A, void* C, int inp, int out) const {
        return exec && cached_N == N && cached_K == K && cached_batch == batch
            && cached_alpha == a && cached_beta == b
            && cached_A == A && cached_C == C
            && cached_input == inp && cached_output == out;
    }

    void store(int N, int K, int batch, float a, float b,
               const void* A, void* C, int inp, int out, cudaGraphExec_t e) {
        if (exec) cudaGraphExecDestroy(exec);
        exec = e;
        cached_N = N; cached_K = K; cached_batch = batch;
        cached_alpha = a; cached_beta = b;
        cached_A = A; cached_C = C;
        cached_input = inp; cached_output = out;
    }

    void clear() {
        if (exec) { cudaGraphExecDestroy(exec); exec = nullptr; }
        cached_N = cached_K = cached_batch = 0;
        cached_alpha = cached_beta = 0;
        cached_A = nullptr; cached_C = nullptr;
        cached_input = cached_output = 0;
    }
};

/// @brief Opaque implementation holding the CUTLASS GEMM engine and all
///        internal state (CUDA streams, scratch buffers, tune cache, etc.).
struct CutlassComplexGemm::Impl {
    GemmImpl gemm;  ///< The underlying SM90 or SM100/SM120 GEMM class instance.

    // Pre-allocated HERK pipeline buffers (set by init_herk, freed by end_herk)
    __half* herk_A_fp16 = nullptr;      ///< INT4->FP16 conversion temp (baseline path)
    int64_t herk_A_fp16_cap = 0;        ///< capacity in __half elements
    __nv_fp8_e4m3* herk_A_fp8 = nullptr; ///< INT4->FP8 conversion temp (direct path)
    int64_t herk_A_fp8_cap = 0;         ///< capacity in FP8 elements

    // Pre-allocated GEMM FP8 buffer (persistent, grown on demand)
    __nv_fp8_e4m3* gemm_A_fp8 = nullptr; ///< INT4->FP8 interleaved temp for GEMM power
    int64_t gemm_A_fp8_cap = 0;          ///< capacity in FP8 bytes (= 2 * num_complex_elements)

    // Temp FP32 Re/Im buffers for 4M power path (persistent, grown on demand)
    float* power_C_re = nullptr;    ///< Temp Re output for 4M sub-GEMMs
    float* power_C_im = nullptr;    ///< Temp Im output for 4M sub-GEMMs
    int64_t power_C_cap = 0;        ///< Capacity in float elements (= M*N*batch)

    // Temp FP16 planar buffers for FP8->FP16 deinterleave (4M power_fp8 path)
    __half* power_A_re = nullptr;   ///< Temp Re for FP8 deinterleave
    __half* power_A_im = nullptr;   ///< Temp Im for FP8 deinterleave
    int64_t power_A_cap = 0;        ///< Capacity in __half elements (= M*K*batch)

    // Power dispatch mode
    PowerMode power_mode = PowerMode::Auto;

    // Planar GEMM dispatch mode
    GemmMode gemm_mode = GemmMode::Auto;

    // Auto-tune flag for power GEMM paths (no explicit tune param)
    bool auto_tune_enabled = true;

    // Dimensions from init_herk (used for graph cache management)
    int init_N = 0;
    int init_K = 0;
    int init_batch = 0;

    // Direct HERK graph cache (4-entry LRU, mirrors HerkGraphEntry pattern)
    bool use_direct_herk_graph = false;
    static constexpr int kDirectGraphCacheSize = 4;
    DirectHerkGraphEntry direct_graph_cache[kDirectGraphCacheSize];
    int direct_graph_lru[kDirectGraphCacheSize] = {0, 1, 2, 3};

    /// Find a matching graph in the cache, or nullptr if no match.
    DirectHerkGraphEntry* find_direct_graph(
        int N, int K, int batch, float a, float b,
        const void* A, void* C, int inp, int out) {
        for (int i = 0; i < kDirectGraphCacheSize; ++i) {
            if (direct_graph_cache[direct_graph_lru[i]].matches(N, K, batch, a, b, A, C, inp, out)) {
                // Move to front (MRU)
                int hit = direct_graph_lru[i];
                for (int j = i; j > 0; --j) direct_graph_lru[j] = direct_graph_lru[j-1];
                direct_graph_lru[0] = hit;
                return &direct_graph_cache[hit];
            }
        }
        return nullptr;
    }

    /// Allocate the LRU slot (evicting oldest) and return it.
    DirectHerkGraphEntry* alloc_direct_graph_slot() {
        int victim = direct_graph_lru[kDirectGraphCacheSize - 1];
        // Move to front
        for (int j = kDirectGraphCacheSize - 1; j > 0; --j)
            direct_graph_lru[j] = direct_graph_lru[j-1];
        direct_graph_lru[0] = victim;
        return &direct_graph_cache[victim];
    }

    void clear_direct_graph_cache() {
        for (int i = 0; i < kDirectGraphCacheSize; ++i)
            direct_graph_cache[i].clear();
    }
};

CutlassComplexGemm::CutlassComplexGemm()
    : impl_(new Impl())
{
}

CutlassComplexGemm::~CutlassComplexGemm() {
    end_herk();
    if (impl_->gemm_A_fp8) {
        cudaFree(impl_->gemm_A_fp8);
        impl_->gemm_A_fp8 = nullptr;
        impl_->gemm_A_fp8_cap = 0;
    }
    if (impl_->power_C_re) { cudaFree(impl_->power_C_re); impl_->power_C_re = nullptr; }
    if (impl_->power_C_im) { cudaFree(impl_->power_C_im); impl_->power_C_im = nullptr; }
    impl_->power_C_cap = 0;
    if (impl_->power_A_re) { cudaFree(impl_->power_A_re); impl_->power_A_re = nullptr; }
    if (impl_->power_A_im) { cudaFree(impl_->power_A_im); impl_->power_A_im = nullptr; }
    impl_->power_A_cap = 0;
    delete impl_;
}

void CutlassComplexGemm::set_tune_cache_path(const std::string& path) {
    strategy_cache::StrategyCache::instance().set_file_path(path);
}

void CutlassComplexGemm::set_kernel_tune_verbosity(int level) {
    tune_cache::TuneCache::instance().set_verbosity(level);
}

void CutlassComplexGemm::set_strategy_tune_verbosity(int level) {
    strategy_cache::StrategyCache::instance().set_verbosity(level);
}

void CutlassComplexGemm::set_gemm_tune_verbosity(int level) {
    strategy_cache::GemmStrategyCache::instance().set_verbosity(level);
}

void CutlassComplexGemm::set_gemm_tune_cache_path(const std::string& path) {
    strategy_cache::GemmStrategyCache::instance().set_file_path(path);
}

void CutlassComplexGemm::set_kernel_tune_cache_path(const std::string& path) {
    tune_cache::TuneCache::instance().set_file_path(path);
}

/// @copydoc CutlassComplexGemm::print_build_info
void CutlassComplexGemm::print_build_info() {
    auto sysinfo = cutlass_complex::query_system_info();
    cutlass_complex::print_system_info(sysinfo);
    cutlass_complex::print_build_config();
    std::cout.flush();
}

/// @copydoc CutlassComplexGemm::set_herk_mode
void CutlassComplexGemm::set_herk_mode(HerkMode mode) {
    impl_->gemm.set_herk_mode(map_herk_mode(mode));
}

/// @copydoc CutlassComplexGemm::set_tile_config
void CutlassComplexGemm::set_tile_config(TileConfig config) {
    impl_->gemm.set_gemm_config(map_tile_config(config));
}

/// @copydoc CutlassComplexGemm::set_direct_tile_config
void CutlassComplexGemm::set_direct_tile_config(DirectTileConfig config) {
    impl_->gemm.set_direct_herk_config(map_direct_tile_config(config));
}

/// @copydoc CutlassComplexGemm::set_direct_herk_graph
void CutlassComplexGemm::set_direct_herk_graph(bool enable) {
    impl_->use_direct_herk_graph = enable;
}

/// @copydoc CutlassComplexGemm::set_power_mode
void CutlassComplexGemm::set_power_mode(PowerMode mode) {
    impl_->power_mode = mode;
}

/// @copydoc CutlassComplexGemm::set_gemm_mode
void CutlassComplexGemm::set_gemm_mode(GemmMode mode) {
    impl_->gemm_mode = mode;
}

/// @copydoc CutlassComplexGemm::get_gemm_mode
GemmMode CutlassComplexGemm::get_gemm_mode() const {
    return impl_->gemm_mode;
}

/// @copydoc CutlassComplexGemm::set_auto_tune
void CutlassComplexGemm::set_auto_tune(bool enable) {
    impl_->auto_tune_enabled = enable;
}

// =====================================================================
// Power Reduction Kernel + Helpers (4M power path)
// =====================================================================

/// @brief Fused power reduction: C_power[i] = alpha*(re[i]^2+im[i]^2) + beta*C_power[i]
///
/// Memory-bound elementwise kernel, negligible vs GEMM (<1% of total time).
/// Uses grid-stride loop for arbitrary sizes.
///
/// @param C_re          FP32 real output from 4M sub-GEMMs.
/// @param C_im          FP32 imaginary output from 4M sub-GEMMs.
/// @param C_power       Power output (in/out when beta != 0).
/// @param num_elements  Total elements (M * N * batch_count).
/// @param alpha         Scalar multiplier.
/// @param beta          Accumulator scalar.
__global__ static void power_reduce_kernel(
    const float* __restrict__ C_re, const float* __restrict__ C_im,
    float* __restrict__ C_power,
    int64_t num_elements, float alpha, float beta)
{
    int64_t idx = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    int64_t stride = gridDim.x * (int64_t)blockDim.x;
    for (int64_t i = idx; i < num_elements; i += stride) {
        float re = C_re[i], im = C_im[i];
        float p = alpha * (re * re + im * im);
        if (beta != 0.0f) p += beta * C_power[i];
        C_power[i] = p;
    }
}

/// @brief Launch power reduction kernel.
static void launch_power_reduce(
    const float* C_re, const float* C_im, float* C_power,
    int64_t total, float alpha, float beta, cudaStream_t stream)
{
    if (total == 0) return;
    const int block = 256;
    const int64_t grid = std::min((total + block - 1) / block, (int64_t)1024);
    power_reduce_kernel<<<(int)grid, block, 0, stream>>>(
        C_re, C_im, C_power, total, alpha, beta);
}

/// @brief Deinterleave FP8 [re,im,re,im,...] to separate FP16 Re/Im arrays.
///
/// Even bytes → Re (as FP16), odd bytes → Im (as FP16).
/// Uses grid-stride loop with uint4 vectorized loads (16 complex elements/iter).
///
/// @param input       FP8 interleaved input (2 bytes per complex element).
/// @param out_re      FP16 real output.
/// @param out_im      FP16 imaginary output.
/// @param num_complex Total complex elements.
__global__ static void deinterleave_fp8_to_fp16_kernel(
    const uint8_t* __restrict__ input,
    __half* __restrict__ out_re, __half* __restrict__ out_im,
    int64_t num_complex)
{
    int64_t idx = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    int64_t stride = gridDim.x * (int64_t)blockDim.x;
    for (int64_t i = idx; i < num_complex; i += stride) {
        // Each complex element is 2 FP8 bytes: [Re, Im]
        float re = float(*reinterpret_cast<const __nv_fp8_e4m3*>(input + 2 * i));
        float im = float(*reinterpret_cast<const __nv_fp8_e4m3*>(input + 2 * i + 1));
        out_re[i] = __float2half(re);
        out_im[i] = __float2half(im);
    }
}

/// @brief Launch FP8 interleaved to FP16 planar deinterleave.
static void deinterleave_fp8_to_fp16(
    const void* fp8_interleaved,
    __half* out_re, __half* out_im,
    int64_t num_complex, cudaStream_t stream)
{
    if (num_complex == 0) return;
    const int block = 256;
    const int64_t grid = std::min((num_complex + block - 1) / block, (int64_t)1024);
    deinterleave_fp8_to_fp16_kernel<<<(int)grid, block, 0, stream>>>(
        static_cast<const uint8_t*>(fp8_interleaved), out_re, out_im,
        num_complex);
}

/// @brief Determine whether to use 4M power path based on PowerMode and problem size.
///
/// Auto heuristic: use 4M when batch_count >= 4 && N >= 256. This matches
/// empirical results where CUTLASS 4M overtakes the direct kernel.
///
/// @param mode         Current PowerMode setting.
/// @param M, N, K      Problem dimensions.
/// @param batch_count  Batch size.
/// @return true to use 4M path, false for direct.
static bool should_use_4m_power(PowerMode mode, int M, int N, int K, int batch_count) {
    switch (mode) {
    case PowerMode::ForceDirect: return false;
    case PowerMode::Force4M:     return true;
    case PowerMode::Auto:
    default:
        return (batch_count >= 4 && N >= 256);
    }
}

/// @brief Determine whether to use direct PTX kernel for planar GEMM.
///
/// Auto heuristic: check autotuner cache first. If no cache hit, use heuristic
/// based on problem size — direct for small M*N (launch overhead dominated),
/// 4M for large M*N (CUTLASS pipeline efficiency dominated).
///
static bool should_use_direct_gemm(GemmMode mode, int M, int N, int K, int batch_count, int precision) {
    switch (mode) {
    case GemmMode::ForceDirect: return true;
    case GemmMode::Force4M:     return false;
    case GemmMode::Auto:
    default: {
        // Check autotuner cache for a direct/4M decision
        strategy_cache::GemmStrategyEntry cached;
        if (strategy_cache::GemmStrategyCache::instance().lookup(M, N, K, batch_count, precision, cached)) {
            return cached.use_direct;
        }
        // Heuristic: direct is better for small problems or small batch
        // 4M has ~4x launch overhead but better pipeline utilization for large tiles
        return (batch_count <= 2 || M * N <= 4096);
    }
    }
}

/// @brief Ensure a pair of float buffers have sufficient capacity.
static void ensure_float_pair(float*& buf_a, float*& buf_b, int64_t& cap,
                               int64_t needed) {
    if (needed > cap) {
        if (buf_a) cudaFree(buf_a);
        if (buf_b) cudaFree(buf_b);
        CUDA_CHECK(cudaMalloc(&buf_a, needed * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&buf_b, needed * sizeof(float)));
        cap = needed;
    }
}

/// @brief Ensure a pair of __half buffers have sufficient capacity.
static void ensure_half_pair(__half*& buf_a, __half*& buf_b, int64_t& cap,
                              int64_t needed) {
    if (needed > cap) {
        if (buf_a) cudaFree(buf_a);
        if (buf_b) cudaFree(buf_b);
        CUDA_CHECK(cudaMalloc(&buf_a, needed * sizeof(__half)));
        CUDA_CHECK(cudaMalloc(&buf_b, needed * sizeof(__half)));
        cap = needed;
    }
}

// =====================================================================
// GEMM Strategy Autotuner Helper
// =====================================================================

/// @brief Run the GEMM strategy autotuning sweep and apply cached settings.
///
/// Delegates to strategy_cache::run_gemm_autotune() which benchmarks all valid
/// GemmConfig values for the current architecture and caches the fastest.
/// Applies the winning GemmConfig to the engine.
///
/// @param gemm         The underlying GEMM engine instance.
/// @param M            Number of rows of A and C.
/// @param N            Number of columns of C.
/// @param K            Inner dimension.
/// @param batch_count  Batch size.
/// @param precision    Integer cache key (0=FP8, 1=FP6, 2=FP4).
/// @param stream       CUDA stream.
static void tune_gemm(GemmImpl& gemm, int M, int N, int K, int batch_count,
                       int precision, cudaStream_t stream) {
    // Build viable config list from all_baseline_configs()
    auto viable = all_baseline_configs();
    std::vector<int> viable_ints;
    for (auto c : viable) viable_ints.push_back(static_cast<int>(c));

    auto entry = strategy_cache::run_gemm_autotune<
        GemmImpl, InternalGemmConfig, ComplexMode, InternalComputePrecision>(
            gemm, M, N, K, batch_count, precision, stream, viable_ints);

    // Apply the winning config
    strategy_cache::apply_gemm_cached_settings<GemmImpl, InternalGemmConfig>(
        gemm, entry);
}

// =====================================================================
/// @name Unified Planar Complex GEMM
// =====================================================================

/// @copydoc CutlassComplexGemm::gemm
int CutlassComplexGemm::gemm(
    const __half* A_re, const __half* A_im,
    const __half* B_re, const __half* B_im,
    void* C_re, void* C_im,
    int M, int N, int K, int batch_count,
    ComputePrecision compute,
    OutputPrecision output,
    float alpha, float beta,
    cudaStream_t stream,
    bool tune,
    InputPrecision input)
{
    // INT4 A input: convert to FP16 planar, then delegate to FP16 path
    if (input == InputPrecision::INT4) {
        const uint8_t* A_qc = reinterpret_cast<const uint8_t*>(A_re);
        int64_t total = static_cast<int64_t>(M) * K * batch_count;
        __half *Ar = nullptr, *Ai = nullptr;
        CUDA_CHECK(cudaMallocAsync(&Ar, total * sizeof(__half), stream));
        CUDA_CHECK(cudaMallocAsync(&Ai, total * sizeof(__half), stream));
        cast_int4_to_fp16_planar(A_qc, Ar, Ai, total, stream);
        int ret = gemm(Ar, Ai, B_re, B_im, C_re, C_im,
                        M, N, K, batch_count, compute, output,
                        alpha, beta, stream, tune, InputPrecision::FP16);
        cudaFreeAsync(Ar, stream);
        cudaFreeAsync(Ai, stream);
        return ret;
    }

    // Run GEMM autotuning if requested (sweeps GemmConfig, caches result)
    if (tune) {
        int prec_key = precision_cache_key(compute);
        tune_gemm(impl_->gemm, M, N, K, batch_count, prec_key, stream);
    }

    if (output == OutputPrecision::FP32) {
        float* fC_re = static_cast<float*>(C_re);
        float* fC_im = static_cast<float*>(C_im);

#ifdef COMPLEX_FP8_TARGET_SM90
        if (compute != ComputePrecision::FP8)
            throw std::runtime_error("SM90 only supports FP8 compute precision");
        auto status = impl_->gemm.run_planar_batched_fp32out(
            A_re, A_im, B_re, B_im, fC_re, fC_im,
            M, N, K, batch_count, alpha, beta,
            ComplexMode::Standard, stream);
#else
        auto status = impl_->gemm.run_planar_batched_fp32out(
            A_re, A_im, B_re, B_im, fC_re, fC_im,
            M, N, K, batch_count, alpha, beta,
            ComplexMode::Standard, map_compute(compute), stream);
#endif
        return static_cast<int>(status);
    } else {
        // FP16 output — only FP8 compute is supported (internal run_planar_batched
        // is FP8-only; FP6/FP4 with FP16 output would require block-scaled dispatch)
        if (compute != ComputePrecision::FP8)
            throw std::runtime_error("FP16 output only supports FP8 compute precision; use FP32 output for FP6/FP4");

        __half* hC_re = static_cast<__half*>(C_re);
        __half* hC_im = static_cast<__half*>(C_im);

        auto status = impl_->gemm.run_planar_batched(
            A_re, A_im, B_re, B_im, hC_re, hC_im,
            M, N, K, batch_count, alpha, beta,
            ComplexMode::Standard, stream);
        return static_cast<int>(status);
    }
}

// =====================================================================
/// @name Prepare/Execute
// =====================================================================

/// @copydoc CutlassComplexGemm::prepare_b
void CutlassComplexGemm::prepare_b(
    const __half* B_re, const __half* B_im,
    int N, int K, int batch_count,
    ComputePrecision compute, cudaStream_t stream)
{
#ifdef COMPLEX_FP8_TARGET_SM90
    if (compute != ComputePrecision::FP8)
        throw std::runtime_error("SM90 only supports FP8 precision for prepare_b");
    impl_->gemm.prepare_b_data(B_re, B_im, N, K, batch_count, stream);
#else
    impl_->gemm.prepare_b_data(B_re, B_im, N, K, batch_count,
                                map_compute(compute), stream);
#endif
}

/// @copydoc CutlassComplexGemm::gemm_prepared
int CutlassComplexGemm::gemm_prepared(
    const __half* A_re, const __half* A_im,
    void* C_re, void* C_im,
    int M, int N, int K, int batch_count,
    OutputPrecision output,
    float alpha, float beta, cudaStream_t stream,
    bool tune,
    InputPrecision input)
{
    if (output != OutputPrecision::FP32)
        throw std::runtime_error("gemm_prepared: only FP32 output is currently supported");

    // INT4 A input: convert to FP16 planar, then delegate to FP16 path
    if (input == InputPrecision::INT4) {
        const uint8_t* A_qc = reinterpret_cast<const uint8_t*>(A_re);
        int64_t total = static_cast<int64_t>(M) * K * batch_count;
        __half *Ar = nullptr, *Ai = nullptr;
        CUDA_CHECK(cudaMallocAsync(&Ar, total * sizeof(__half), stream));
        CUDA_CHECK(cudaMallocAsync(&Ai, total * sizeof(__half), stream));
        cast_int4_to_fp16_planar(A_qc, Ar, Ai, total, stream);
        int ret = gemm_prepared(Ar, Ai, C_re, C_im,
                                 M, N, K, batch_count, output,
                                 alpha, beta, stream, tune, InputPrecision::FP16);
        cudaFreeAsync(Ar, stream);
        cudaFreeAsync(Ai, stream);
        return ret;
    }

    // Run GEMM autotuning if requested (FP8 compute since prepare_b was FP8)
    if (tune) {
        tune_gemm(impl_->gemm, M, N, K, batch_count, 0 /*FP8*/, stream);
    }

    // Dispatch based on GemmMode
    bool use_direct = should_use_direct_gemm(impl_->gemm_mode, M, N, K, batch_count, 0);

    if (use_direct) {
        // Check if persistent variant is optimal (from cache or K-gated heuristic)
        bool use_persistent = false;
        strategy_cache::GemmStrategyEntry cached;
        if (strategy_cache::GemmStrategyCache::instance().lookup(M, N, K, batch_count, 0, cached)) {
            use_persistent = cached.use_persistent;
        } else {
            use_persistent = (K <= 64);  // K-gated heuristic
        }

        cutlass::Status status;
        if (use_persistent) {
            status = impl_->gemm.run_direct_gemm_prepared_persistent_fp32out(
                A_re, A_im,
                static_cast<float*>(C_re), static_cast<float*>(C_im),
                M, N, K, batch_count,
                alpha, beta, stream);
        } else {
            status = impl_->gemm.run_direct_gemm_prepared_fp32out(
                A_re, A_im,
                static_cast<float*>(C_re), static_cast<float*>(C_im),
                M, N, K, batch_count,
                alpha, beta, stream);
        }
        return static_cast<int>(status);
    }

    auto status = impl_->gemm.run_planar_batched_prepared_fp32out(
        A_re, A_im,
        static_cast<float*>(C_re), static_cast<float*>(C_im),
        M, N, K, batch_count,
        alpha, beta, ComplexMode::Standard, stream);
    return static_cast<int>(status);
}

/// @copydoc CutlassComplexGemm::gemm_prepared_direct
int CutlassComplexGemm::gemm_prepared_direct(
    const __half* A_re, const __half* A_im,
    float* C_re, float* C_im,
    int M, int N, int K, int batch_count,
    float alpha, float beta, cudaStream_t stream)
{
    auto status = impl_->gemm.run_direct_gemm_prepared_fp32out(
        A_re, A_im, C_re, C_im,
        M, N, K, batch_count,
        alpha, beta, stream);
    return static_cast<int>(status);
}

/// @copydoc CutlassComplexGemm::gemm_prepared_power
int CutlassComplexGemm::gemm_prepared_power(
    const __half* A_re, const __half* A_im,
    float* C_power,
    int M, int N, int K, int batch_count,
    float alpha, float beta, cudaStream_t stream)
{
    bool use_4m = should_use_4m_power(impl_->power_mode, M, N, K, batch_count);
    if (use_4m) {
        int64_t total = (int64_t)M * N * batch_count;
        ensure_float_pair(impl_->power_C_re, impl_->power_C_im, impl_->power_C_cap, total);

        // Run GEMM autotuning if enabled (first-time only, cached)
        if (impl_->auto_tune_enabled) {
            tune_gemm(impl_->gemm, M, N, K, batch_count, 0 /*FP8*/, stream);
        }

        // 4M sub-GEMMs → separate Re, Im FP32 outputs
        auto status = impl_->gemm.run_planar_batched_prepared_fp32out(
            A_re, A_im,
            impl_->power_C_re, impl_->power_C_im,
            M, N, K, batch_count,
            1.0f, 0.0f, ComplexMode::Standard, stream);
        if (status != cutlass::Status::kSuccess)
            return static_cast<int>(status);

        // Power reduction: C_power = alpha*(re^2+im^2) + beta*C_power
        launch_power_reduce(impl_->power_C_re, impl_->power_C_im, C_power,
                            total, alpha, beta, stream);
        return 0;
    } else {
        auto status = impl_->gemm.run_direct_gemm_prepared_power(
            A_re, A_im, C_power,
            M, N, K, batch_count,
            alpha, beta, stream);
        return static_cast<int>(status);
    }
}

/// @copydoc CutlassComplexGemm::gemm_prepared_power_fp8
int CutlassComplexGemm::gemm_prepared_power_fp8(
    const void* A_fp8_interleaved,
    float* C_power,
    int M, int N, int K, int batch_count,
    float alpha, float beta, cudaStream_t stream)
{
    bool use_4m = should_use_4m_power(impl_->power_mode, M, N, K, batch_count);
    if (use_4m) {
        int64_t total_complex = (int64_t)M * K * batch_count;
        int64_t total_output = (int64_t)M * N * batch_count;

        // Deinterleave FP8 [Re,Im,...] → FP16 planar Re/Im
        ensure_half_pair(impl_->power_A_re, impl_->power_A_im, impl_->power_A_cap, total_complex);
        deinterleave_fp8_to_fp16(A_fp8_interleaved,
                                  impl_->power_A_re, impl_->power_A_im,
                                  total_complex, stream);

        // Ensure output temp buffers
        ensure_float_pair(impl_->power_C_re, impl_->power_C_im, impl_->power_C_cap, total_output);

        // Run GEMM autotuning if enabled
        if (impl_->auto_tune_enabled) {
            tune_gemm(impl_->gemm, M, N, K, batch_count, 0 /*FP8*/, stream);
        }

        // 4M sub-GEMMs → separate Re, Im FP32 outputs
        auto status = impl_->gemm.run_planar_batched_prepared_fp32out(
            impl_->power_A_re, impl_->power_A_im,
            impl_->power_C_re, impl_->power_C_im,
            M, N, K, batch_count,
            1.0f, 0.0f, ComplexMode::Standard, stream);
        if (status != cutlass::Status::kSuccess)
            return static_cast<int>(status);

        // Power reduction
        launch_power_reduce(impl_->power_C_re, impl_->power_C_im, C_power,
                            total_output, alpha, beta, stream);
        return 0;
    } else {
        auto status = impl_->gemm.run_direct_gemm_prepared_power_fp8(
            reinterpret_cast<const __nv_fp8_e4m3*>(A_fp8_interleaved),
            C_power,
            M, N, K, batch_count,
            alpha, beta, stream);
        return static_cast<int>(status);
    }
}

/// @copydoc CutlassComplexGemm::gemm_prepared_power_int4
int CutlassComplexGemm::gemm_prepared_power_int4(
    const void* A_int4, float* C_power,
    int M, int N, int K, int batch_count,
    float alpha, float beta, cudaStream_t stream)
{
    const uint8_t* A_qc = static_cast<const uint8_t*>(A_int4);
    int64_t total = static_cast<int64_t>(M) * K * batch_count;

    bool use_4m = should_use_4m_power(impl_->power_mode, M, N, K, batch_count);
    if (use_4m) {
        int64_t total_output = (int64_t)M * N * batch_count;

        // INT4 → FP16 planar for 4M GEMM path
        ensure_half_pair(impl_->power_A_re, impl_->power_A_im, impl_->power_A_cap,
                          total);
        cast_int4_to_fp16_planar(A_qc, impl_->power_A_re, impl_->power_A_im,
                                  total, stream);

        // Ensure output temp buffers
        ensure_float_pair(impl_->power_C_re, impl_->power_C_im, impl_->power_C_cap, total_output);

        // Run GEMM autotuning if enabled
        if (impl_->auto_tune_enabled) {
            tune_gemm(impl_->gemm, M, N, K, batch_count, 0 /*FP8*/, stream);
        }

        // 4M sub-GEMMs → separate Re, Im FP32 outputs
        auto status = impl_->gemm.run_planar_batched_prepared_fp32out(
            impl_->power_A_re, impl_->power_A_im,
            impl_->power_C_re, impl_->power_C_im,
            M, N, K, batch_count,
            1.0f, 0.0f, ComplexMode::Standard, stream);
        if (status != cutlass::Status::kSuccess)
            return static_cast<int>(status);

        // Power reduction
        launch_power_reduce(impl_->power_C_re, impl_->power_C_im, C_power,
                            total_output, alpha, beta, stream);
        return 0;
    } else {
        // Direct path: INT4 → FP8 interleaved → direct power kernel
        int64_t needed = total * 2;  // 2 FP8 bytes per complex element
        if (needed > impl_->gemm_A_fp8_cap) {
            if (impl_->gemm_A_fp8) cudaFree(impl_->gemm_A_fp8);
            CUDA_CHECK(cudaMalloc(&impl_->gemm_A_fp8, needed * sizeof(__nv_fp8_e4m3)));
            impl_->gemm_A_fp8_cap = needed;
        }

        cast_int4_to_fp8_interleaved(A_qc, impl_->gemm_A_fp8, total, stream);
        auto status = impl_->gemm.run_direct_gemm_prepared_power_fp8(
            impl_->gemm_A_fp8, C_power, M, N, K, batch_count, alpha, beta, stream);
        return static_cast<int>(status);
    }
}

/// @copydoc CutlassComplexGemm::gemm_prepared_direct_int4
int CutlassComplexGemm::gemm_prepared_direct_int4(
    const void* A_int4,
    float* C_re, float* C_im,
    int M, int N, int K, int batch_count,
    float alpha, float beta, cudaStream_t stream)
{
    const uint8_t* A_qc = static_cast<const uint8_t*>(A_int4);
    int64_t total = static_cast<int64_t>(M) * K * batch_count;
    __half *Ar = nullptr, *Ai = nullptr;
    CUDA_CHECK(cudaMallocAsync(&Ar, total * sizeof(__half), stream));
    CUDA_CHECK(cudaMallocAsync(&Ai, total * sizeof(__half), stream));
    cast_int4_to_fp16_planar(A_qc, Ar, Ai, total, stream);
    auto status = impl_->gemm.run_direct_gemm_prepared_fp32out(
        Ar, Ai, C_re, C_im, M, N, K, batch_count, alpha, beta, stream);
    cudaFreeAsync(Ar, stream);
    cudaFreeAsync(Ai, stream);
    return static_cast<int>(status);
}

// =====================================================================
/// @name Batch-Fused M GEMM
// =====================================================================

/// @brief Auto-select fuse_factor targeting M_fused in [128, 1024].
///
/// Picks the largest power-of-2 fuse_factor such that:
///   (a) M * fuse_factor >= 128  (fill at least one tile row fully)
///   (b) M * fuse_factor <= 1024  (diminishing returns beyond this)
///   (c) fuse_factor divides batch_count evenly
///
/// Falls back to 1 if no valid factor exists.
///
/// @param M            Per-element row count.
/// @param batch_count  Total batch size.
/// @return Selected fuse factor (always >= 1).
static int auto_fuse_factor(int M, int batch_count) {
    if (M <= 0 || batch_count <= 0) return 1;
    // If M is already >= 256, no benefit from fusing
    if (M >= 256) return 1;

    // Target M_fused in [128, 1024]
    int best_ff = 1;
    for (int ff = 2; ff <= 32; ff *= 2) {
        int M_fused = M * ff;
        if (M_fused > 1024) break;
        if (batch_count % ff != 0) continue;
        best_ff = ff;
    }
    // If M * best_ff < 128, try to get to at least 128
    if (M * best_ff < 128) {
        for (int ff = best_ff + 1; ff <= 128; ++ff) {
            if (batch_count % ff != 0) continue;
            if (M * ff >= 128) { best_ff = ff; break; }
        }
    }
    return best_ff;
}

/// @copydoc CutlassComplexGemm::gemm_prepared_fused
int CutlassComplexGemm::gemm_prepared_fused(
    const __half* A_re, const __half* A_im,
    void* C_re, void* C_im,
    int M, int N, int K, int batch_count,
    int fuse_factor,
    OutputPrecision output,
    float alpha, float beta, cudaStream_t stream)
{
    if (output != OutputPrecision::FP32)
        throw std::runtime_error("gemm_prepared_fused: only FP32 output is currently supported");

    // Auto-select fuse_factor if 0
    if (fuse_factor <= 0) {
        fuse_factor = auto_fuse_factor(M, batch_count);
    }

    // Validate
    if (fuse_factor < 1)
        throw std::runtime_error("gemm_prepared_fused: fuse_factor must be >= 1");
    if (batch_count % fuse_factor != 0)
        throw std::runtime_error("gemm_prepared_fused: fuse_factor must divide batch_count evenly");

    // If fuse_factor == 1, just delegate to regular prepared path
    if (fuse_factor == 1) {
        return gemm_prepared(A_re, A_im, C_re, C_im,
                             M, N, K, batch_count, output, alpha, beta, stream);
    }

    auto status = impl_->gemm.run_planar_batched_prepared_fused_fp32out(
        A_re, A_im,
        static_cast<float*>(C_re), static_cast<float*>(C_im),
        M, N, K, batch_count, fuse_factor,
        alpha, beta, ComplexMode::Standard, stream);
    return static_cast<int>(status);
}

// =====================================================================
// Diagnostics
// =====================================================================

int CutlassComplexGemm::debug_fp6_real_gemm(int M, int N, int K, cudaStream_t stream)
{
#ifdef COMPLEX_FP8_TARGET_SM90
    fprintf(stderr, "debug_fp6_real_gemm: not available on SM90 (Hopper)\n");
    return -1;
#else
    fprintf(stderr, "\n=== debug_fp6_real_gemm M=%d N=%d K=%d ===\n", M, N, K);

    int64_t sA = (int64_t)M * K;
    int64_t sB = (int64_t)N * K;
    int64_t sC = (int64_t)M * N;

    // Allocate and fill host data with small integers [-7,7]
    std::vector<float> h_A(sA), h_B(sB), h_C_ref(sC, 0);
    srand(42);
    for (int64_t i = 0; i < sA; i++) h_A[i] = (float)(rand() % 15 - 7);
    for (int64_t i = 0; i < sB; i++) h_B[i] = (float)(rand() % 15 - 7);

    // Reference C = A * B^T (TN layout)
    #pragma omp parallel for schedule(dynamic)
    for (int m = 0; m < M; m++)
        for (int n = 0; n < N; n++) {
            float sum = 0;
            for (int k = 0; k < K; k++) sum += h_A[m*K+k] * h_B[n*K+k];
            h_C_ref[m*N+n] = sum;
        }

    // Upload as FP16
    std::vector<__half> h_A_h(sA), h_B_h(sB);
    #pragma omp parallel for
    for (int64_t i = 0; i < sA; i++) h_A_h[i] = __float2half(h_A[i]);
    #pragma omp parallel for
    for (int64_t i = 0; i < sB; i++) h_B_h[i] = __float2half(h_B[i]);

    __half *d_A, *d_B;
    cudaMalloc(&d_A, sA * sizeof(__half));
    cudaMalloc(&d_B, sB * sizeof(__half));
    cudaMemcpy(d_A, h_A_h.data(), sA * sizeof(__half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B_h.data(), sB * sizeof(__half), cudaMemcpyHostToDevice);

    int hw_sm = 0;
    cudaDeviceGetAttribute(&hw_sm, cudaDevAttrMultiProcessorCount, 0);

    // ---------------------------------------------------------------
    // TEST 1: Through run_planar (complex with imag=0) -> FP16 output
    // ---------------------------------------------------------------
    fprintf(stderr, "\n--- TEST 1: run_planar(Standard, FP6) complex with imag=0 -> FP16 ---\n");
    {
        __half* d_zeros;
        cudaMalloc(&d_zeros, std::max(sA, sB) * sizeof(__half));
        cudaMemset(d_zeros, 0, std::max(sA, sB) * sizeof(__half));

        __half *d_Cre, *d_Cim;
        cudaMalloc(&d_Cre, sC * sizeof(__half));
        cudaMalloc(&d_Cim, sC * sizeof(__half));
        cudaMemset(d_Cre, 0, sC * sizeof(__half));
        cudaMemset(d_Cim, 0, sC * sizeof(__half));

        auto st = impl_->gemm.run_planar(
            d_A, d_zeros,
            d_B, d_zeros,
            d_Cre, d_Cim,
            M, N, K,
            1.0f, 0.0f,
            ComplexMode::Standard,
            gemm_complex_sm100::ComputePrecision::FP6_E3M2,
            stream);
        cudaDeviceSynchronize();

        std::vector<__half> h_Cre(sC);
        cudaMemcpy(h_Cre.data(), d_Cre, sC * sizeof(__half), cudaMemcpyDeviceToHost);

        float max_ref = 0, max_err = 0;
        #pragma omp parallel for reduction(max:max_ref)
        for (int i = 0; i < (int)sC; i++) max_ref = fmaxf(max_ref, fabsf(h_C_ref[i]));
        #pragma omp parallel for reduction(max:max_err)
        for (int i = 0; i < (int)sC; i++)
            max_err = fmaxf(max_err, fabsf(__half2float(h_Cre[i]) - h_C_ref[i]));

        fprintf(stderr, "  First 5: ");
        for (int i = 0; i < 5; i++)
            fprintf(stderr, "ref=%.0f got=%.0f  ", h_C_ref[i], __half2float(h_Cre[i]));
        fprintf(stderr, "\n  max_ref=%.1f  max_err=%.2f  rel=%.4f  %s\n",
                max_ref, max_err, max_err/max_ref,
                (max_err/max_ref < 0.02) ? "PASS" : "FAIL");
        fprintf(stderr, "  status=%d\n", (int)st);

        cudaFree(d_zeros); cudaFree(d_Cre); cudaFree(d_Cim);
    }

    // ---------------------------------------------------------------
    // TEST 2: Direct block-scaled dispatch with NON-PAIRED preprocessing
    // ---------------------------------------------------------------
    fprintf(stderr, "\n--- TEST 2: preprocess_mxfp_sm100 (non-paired) -> direct dispatch -> FP16 ---\n");
    {
        int64_t bytes_A = bytes_for_elements(sA, gemm_complex_sm100::ComputePrecision::FP6_E3M2);
        int64_t bytes_B = bytes_for_elements(sB, gemm_complex_sm100::ComputePrecision::FP6_E3M2);
        auto sf_buf = [](int rows, int cols) -> int64_t {
            int64_t pr = ((rows + 127) / 128) * 128;
            int64_t kb = (cols + 31) / 32;
            int64_t pkb = ((kb + 3) / 4) * 4;
            return pr * pkb;
        };
        int64_t sf_A = sf_buf(M, K);
        int64_t sf_B = sf_buf(N, K);

        void *d_A6, *d_B6, *d_sfA, *d_sfB;
        cudaMalloc(&d_A6, bytes_A);
        cudaMalloc(&d_B6, bytes_B);
        cudaMalloc(&d_sfA, sf_A);
        cudaMalloc(&d_sfB, sf_B);

        preprocess_mxfp_sm100(d_A, d_A6, d_sfA, M, K, gemm_complex_sm100::ComputePrecision::FP6_E3M2, stream);
        preprocess_mxfp_sm100(d_B, d_B6, d_sfB, N, K, gemm_complex_sm100::ComputePrecision::FP6_E3M2, stream);
        cudaDeviceSynchronize();

        // Verify SFs
        std::vector<uint8_t> h_sfA(sf_A), h_sfB(sf_B);
        cudaMemcpy(h_sfA.data(), d_sfA, sf_A, cudaMemcpyDeviceToHost);
        cudaMemcpy(h_sfB.data(), d_sfB, sf_B, cudaMemcpyDeviceToHost);

        fprintf(stderr, "  SF_A row0: ");
        int num_k_tiles = (K + 127) / 128;
        int total_k_blocks = (K + 31) / 32;
        for (int kb = 0; kb < std::min(total_k_blocks, 4); kb++) {
            int tile_m = 0, tile_k = kb / 4;
            int local_k_block = kb % 4;
            int m_inner = 0, m_outer = 0;
            int64_t sf_idx = (int64_t)(tile_m * num_k_tiles + tile_k) * 512
                           + m_inner * 16 + m_outer * 4 + local_k_block;
            fprintf(stderr, "kb%d=0x%02X(2^%d) ", kb, h_sfA[sf_idx],
                    (int)h_sfA[sf_idx] - 127);
        }
        fprintf(stderr, "\n");

        // FP6 -> FP16 output
        cutlass::half_t* d_C16;
        cudaMalloc(&d_C16, sC * sizeof(cutlass::half_t));
        cudaMemset(d_C16, 0, sC * sizeof(cutlass::half_t));
        auto st = run_blockscaled_gemm_fp6_e3m2(
            d_A6, d_B6, d_sfA, d_sfB, d_C16,
            M, N, K, 1.0f, 0.0f, stream, hw_sm, GemmConfig::Default, 1, 0);
        cudaDeviceSynchronize();

        std::vector<cutlass::half_t> h_C16(sC);
        cudaMemcpy(h_C16.data(), d_C16, sC * sizeof(cutlass::half_t), cudaMemcpyDeviceToHost);

        float max_ref = 0, max_err = 0;
        #pragma omp parallel for reduction(max:max_ref)
        for (int i = 0; i < (int)sC; i++) max_ref = fmaxf(max_ref, fabsf(h_C_ref[i]));
        #pragma omp parallel for reduction(max:max_err)
        for (int i = 0; i < (int)sC; i++)
            max_err = fmaxf(max_err, fabsf((float)h_C16[i] - h_C_ref[i]));

        fprintf(stderr, "  First 5: ");
        for (int i = 0; i < 5; i++)
            fprintf(stderr, "ref=%.0f got=%.0f  ", h_C_ref[i], (float)h_C16[i]);
        fprintf(stderr, "\n  max_ref=%.1f  max_err=%.2f  rel=%.4f  %s\n",
                max_ref, max_err, max_err/max_ref,
                (max_err/max_ref < 0.02) ? "PASS" : "FAIL");
        fprintf(stderr, "  status=%d\n", (int)st);

        // FP6 -> FP32 output
        float* d_C32;
        cudaMalloc(&d_C32, sC * sizeof(float));
        cudaMemset(d_C32, 0, sC * sizeof(float));
        auto st2 = run_blockscaled_gemm_fp6_e3m2_fp32out(
            d_A6, d_B6, d_sfA, d_sfB, d_C32,
            M, N, K, 1.0f, 0.0f, stream, hw_sm, GemmConfig::Default, 1, 0);
        cudaDeviceSynchronize();

        std::vector<float> h_C32(sC);
        cudaMemcpy(h_C32.data(), d_C32, sC * sizeof(float), cudaMemcpyDeviceToHost);

        float max_err32 = 0;
        for (int i = 0; i < (int)sC; i++)
            max_err32 = fmaxf(max_err32, fabsf(h_C32[i] - h_C_ref[i]));
        fprintf(stderr, "  FP32: first 5: ");
        for (int i = 0; i < 5; i++)
            fprintf(stderr, "ref=%.0f got=%.1f  ", h_C_ref[i], h_C32[i]);
        fprintf(stderr, "\n  FP32 max_err=%.2f  rel=%.4f  %s\n",
                max_err32, max_err32/max_ref,
                (max_err32/max_ref < 0.02) ? "PASS" : "FAIL");

        cudaFree(d_A6); cudaFree(d_B6); cudaFree(d_sfA); cudaFree(d_sfB);
        cudaFree(d_C16); cudaFree(d_C32);
    }

    // ---------------------------------------------------------------
    // TEST 3: Through run_planar (complex with imag=0) -> FP32 output
    // ---------------------------------------------------------------
    fprintf(stderr, "\n--- TEST 3: run_planar_batched_fp32out(FP6) complex with imag=0 -> FP32 ---\n");
    {
        __half* d_zeros;
        cudaMalloc(&d_zeros, std::max(sA, sB) * sizeof(__half));
        cudaMemset(d_zeros, 0, std::max(sA, sB) * sizeof(__half));

        float *d_Cre, *d_Cim;
        cudaMalloc(&d_Cre, sC * sizeof(float));
        cudaMalloc(&d_Cim, sC * sizeof(float));
        cudaMemset(d_Cre, 0, sC * sizeof(float));
        cudaMemset(d_Cim, 0, sC * sizeof(float));

        auto st = impl_->gemm.run_planar_batched_fp32out(
            d_A, d_zeros,
            d_B, d_zeros,
            d_Cre, d_Cim,
            M, N, K,
            1,
            1.0f, 0.0f,
            ComplexMode::Standard,
            gemm_complex_sm100::ComputePrecision::FP6_E3M2,
            stream);
        cudaDeviceSynchronize();

        std::vector<float> h_Cre(sC);
        cudaMemcpy(h_Cre.data(), d_Cre, sC * sizeof(float), cudaMemcpyDeviceToHost);

        float max_ref = 0, max_err = 0;
        #pragma omp parallel for reduction(max:max_ref)
        for (int i = 0; i < (int)sC; i++) max_ref = fmaxf(max_ref, fabsf(h_C_ref[i]));
        #pragma omp parallel for reduction(max:max_err)
        for (int i = 0; i < (int)sC; i++)
            max_err = fmaxf(max_err, fabsf(h_Cre[i] - h_C_ref[i]));

        fprintf(stderr, "  First 5: ");
        for (int i = 0; i < 5; i++)
            fprintf(stderr, "ref=%.0f got=%.1f  ", h_C_ref[i], h_Cre[i]);
        fprintf(stderr, "\n  max_ref=%.1f  max_err=%.2f  rel=%.4f  %s\n",
                max_ref, max_err, max_err/max_ref,
                (max_err/max_ref < 0.02) ? "PASS" : "FAIL");
        fprintf(stderr, "  status=%d\n", (int)st);

        cudaFree(d_zeros); cudaFree(d_Cre); cudaFree(d_Cim);
    }

    cudaFree(d_A); cudaFree(d_B);
    fprintf(stderr, "\n=== debug_fp6_real_gemm done ===\n");
    return 0;
#endif  // COMPLEX_FP8_TARGET_SM90
}

// ===================================================================
/// @name HERK API -- Hermitian Rank-K Update
/// @brief Internal helpers and PIMPL method implementations for HERK.
// ===================================================================

/// @brief Build a CutlassParams with only the CUDA stream field set.
///
/// CutlassParams is a POD-like struct with default-initialized fields.
/// This helper avoids designated-initializer syntax that may not compile
/// under all nvcc C++ standard modes.
///
/// @param stream  CUDA stream to embed in the returned params.
/// @return A CutlassParams instance with @c stream set and all other
///         fields at their default values.
static CutlassParams make_params(cudaStream_t stream) {
    CutlassParams p;
    p.stream = stream;
    return p;
}

/// @brief Ensure the GemmImpl is configured with a viable HerkMode.
///
/// When the FP8 baseline GEMM kernel exceeds device SMEM (e.g. stg3 on SM120:
/// 110 KB kernel vs 99 KB device), HerkMode::Auto would select Baseline which
/// cannot launch. This function forces Direct mode in that case.
///
/// @param gemm  The underlying GEMM engine instance.
static void ensure_viable_herk_mode(GemmImpl& gemm) {
    if (!GemmImpl::fp8_baseline_available()) {
        gemm.set_herk_mode(InternalHerkMode::ForceDirect);
    }
}

/// @copydoc CutlassComplexGemm::init_herk
void CutlassComplexGemm::init_herk(int N, int K, int batch_count, cudaStream_t stream) {
    end_herk();  // free any existing buffers

    // INT4->FP16 conversion buffer: N*K*batch complex elements, 2 halfs each
    int64_t A_elems = static_cast<int64_t>(N) * K * batch_count * 2;
    if (A_elems > 0) {
        CUDA_CHECK(cudaMalloc(&impl_->herk_A_fp16, A_elems * sizeof(__half)));
        impl_->herk_A_fp16_cap = A_elems;
    }

    // INT4->FP8 interleaved buffer for direct HERK path
    int64_t fp8_needed = static_cast<int64_t>(N) * K * batch_count * 2;
    if (fp8_needed > 0) {
        CUDA_CHECK(cudaMalloc(&impl_->herk_A_fp8, fp8_needed * sizeof(__nv_fp8_e4m3)));
        impl_->herk_A_fp8_cap = fp8_needed;
    }

    // Force internal lazy allocations (mode selection, streams, HW info)
    ensure_viable_herk_mode(impl_->gemm);
    impl_->gemm.ensure_streams_public();
    impl_->gemm.ensure_hw_info_public();

    // Pre-allocate precast buffer for direct HERK (FP16→FP8 done once)
    int64_t total_fp8 = static_cast<int64_t>(N) * 2 * K * batch_count;
    if (total_fp8 > 0) {
        impl_->gemm.ensure_herk_precast_public(total_fp8, stream);
    }

    // Pre-allocate scratch for batch-tiled direct HERK (tile count fits in L2)
    int l2 = impl_->gemm.l2_cache_bytes();
    if (l2 > 0) {
        int64_t scratch_per_batch = static_cast<int64_t>(N) * N * 2;
        int64_t scratch_bytes_per_batch = scratch_per_batch * static_cast<int64_t>(sizeof(__half));
        if (scratch_bytes_per_batch <= l2) {
            int scratch_batch_tile = batch_count;
            if (scratch_bytes_per_batch * batch_count > l2) {
                scratch_batch_tile = std::max(1, static_cast<int>(l2 / scratch_bytes_per_batch));
            }
            impl_->gemm.ensure_herk_scratch_public(scratch_per_batch * scratch_batch_tile, stream);
        }
    }

    impl_->init_N = N;
    impl_->init_K = K;
    impl_->init_batch = batch_count;
}

/// @copydoc CutlassComplexGemm::end_herk
void CutlassComplexGemm::end_herk() {
    if (!impl_) return;
    if (impl_->herk_A_fp16) {
        cudaFree(impl_->herk_A_fp16);
        impl_->herk_A_fp16 = nullptr;
    }
    impl_->herk_A_fp16_cap = 0;
    if (impl_->herk_A_fp8) {
        cudaFree(impl_->herk_A_fp8);
        impl_->herk_A_fp8 = nullptr;
    }
    impl_->herk_A_fp8_cap = 0;
    impl_->clear_direct_graph_cache();
    impl_->init_N = impl_->init_K = impl_->init_batch = 0;
}

/// @brief Build a CutlassParams from a cached strategy entry.
///
/// Sets herk_strategy and triangle_config.use_cuda_graph in the params.
/// Engine-instance settings (herk_mode, persistent_mode, herk_graph, batch_tiling)
/// are applied separately via strategy_cache::apply_cached_settings().
///
/// @param entry   The strategy entry from the cache.
/// @param stream  CUDA stream.
/// @return A CutlassParams configured with the optimal strategy.
static CutlassParams make_tuned_params(const strategy_cache::StrategyEntry& entry,
                                       cudaStream_t stream) {
    CutlassParams p;
    p.stream = stream;
    p.herk_strategy = (entry.herk_strategy == 1)
        ? HerkStrategy::TriangleAware
        : HerkStrategy::Baseline;
    p.triangle_config.use_cuda_graph = entry.use_cuda_graph;
    p.config = static_cast<GemmConfig>(entry.gemm_config);
    return p;
}

/// @brief Run the full strategy autotuning sweep and apply all cached settings.
///
/// Delegates to strategy_cache::run_autotune() which benchmarks ~12 strategy
/// combinations (Direct/Baseline x Baseline/TriangleAware x graph x persistent x
/// herk_graph) and caches the fastest. Applies ALL cached settings to the engine.
///
/// @param gemm         The underlying GEMM engine instance.
/// @param N            Matrix dimension.
/// @param K            Inner dimension.
/// @param batch_count  Batch size.
/// @param precision    Integer cache key (0=FP8, 1=FP6, 2=FP4).
/// @param stream       CUDA stream.
/// @return CutlassParams configured with the optimal strategy.
static CutlassParams tune_herk(GemmImpl& gemm, int N, int K, int batch_count,
                               int precision, cudaStream_t stream) {
    // Build viable config list from all_baseline_configs() for GemmConfig sweep
    auto viable = all_baseline_configs();
    std::vector<int> viable_ints;
    for (auto c : viable) viable_ints.push_back(static_cast<int>(c));

    auto entry = strategy_cache::run_autotune<
        GemmImpl, InternalHerkMode, InternalPersistentMode, InternalDirectHerkConfig,
        InternalHerkTileSize, InternalHerkPipelineMode,
        HerkOp, FillMode, HerkStrategy, CutlassParams, TriangleConfig>(
            gemm, N, K, batch_count, precision, stream, viable_ints);

    // Apply ALL cached settings to the engine instance
    strategy_cache::apply_cached_settings<GemmImpl, InternalHerkMode, InternalPersistentMode,
                                          InternalGemmConfig, InternalDirectHerkConfig,
                                          InternalHerkTileSize, InternalHerkPipelineMode>(
        gemm, entry);

    return make_tuned_params(entry, stream);
}

// ===================================================================
// INT4 -> FP8/FP16 conversion kernels (used by herk/gemm with INT4 input)
// ===================================================================
//
// All conversion kernels below use a 256-entry LUT (qc_to_fp8_lut_256)
// that maps each possible input byte directly to its 2-byte FP8 output,
// eliminating all per-nibble float arithmetic.  The LUT is indexed by the
// full byte value (high nibble = Re, low nibble = Im) and produces a
// uint16_t containing [Re_fp8 | Im_fp8 << 8] in little-endian order.
//
// For FP16 output, a matching 256-entry LUT (qc_to_fp16_lut_256) maps
// each input byte to a uint32_t containing [Re_fp16 | Im_fp16 << 16].
//
// These LUTs reside in __constant__ memory (256 bytes / 1 KB respectively)
// which is broadcast to all threads in a warp when they access the same
// address, and cached in a dedicated per-SM constant cache (8 KB on
// Blackwell, 64 KB on Hopper).  Since the LUT is only 256 entries, it
// fits entirely in the constant cache after the first access.
//
// Vectorization: 128-bit loads (uint4, 16 QC bytes) and 128-bit stores
// (uint4, 16 FP8 pairs = 32 bytes).  Each thread processes 16 complex
// elements per iteration (16 bytes in, 32 bytes out for FP8; 16 bytes in,
// 64 bytes out for FP16 planar).
// ===================================================================

/// @brief 256-entry byte->FP8-pair LUT for INT4 sign-magnitude to FP8 E4M3.
///
/// Index: full QC byte (0x00..0xFF). Value: uint16_t [Re_fp8 | Im_fp8 << 8].
/// Sign-magnitude nibble encoding: bit 3 = sign, bits 2:0 = magnitude.
/// FP8 E4M3 exactly represents all integers in [-7, +7].
///
/// The LUT is constructed from the 16-entry per-nibble table:
///   nibble 0..7  -> +0..+7: FP8 {0x00, 0x38, 0x40, 0x44, 0x48, 0x4A, 0x4C, 0x4E}
///   nibble 8..15 -> -0..-7: FP8 {0x80, 0xB8, 0xC0, 0xC4, 0xC8, 0xCA, 0xCC, 0xCE}
__device__ __constant__ static uint16_t qc_to_fp8_lut_256[256] = {
    // Generated: for each byte b in 0..255:
    //   re_nib = (b >> 4) & 0xF, im_nib = b & 0xF
    //   fp8_re = fp8_table[re_nib], fp8_im = fp8_table[im_nib]
    //   entry = fp8_re | (fp8_im << 8)
    // where fp8_table[16] = {0x00,0x38,0x40,0x44,0x48,0x4A,0x4C,0x4E,
    //                        0x80,0xB8,0xC0,0xC4,0xC8,0xCA,0xCC,0xCE}
    #define FP8_TBL(n) ((uint8_t[]){0x00,0x38,0x40,0x44,0x48,0x4A,0x4C,0x4E,0x80,0xB8,0xC0,0xC4,0xC8,0xCA,0xCC,0xCE}[(n)])
    #define QC_FP8(b) ((uint16_t)(FP8_TBL((b)>>4)) | ((uint16_t)(FP8_TBL((b)&0xF)) << 8))
    // Row 0x00..0x0F (Re nibble = 0, Re_fp8 = 0x00)
    QC_FP8(0x00), QC_FP8(0x01), QC_FP8(0x02), QC_FP8(0x03),
    QC_FP8(0x04), QC_FP8(0x05), QC_FP8(0x06), QC_FP8(0x07),
    QC_FP8(0x08), QC_FP8(0x09), QC_FP8(0x0A), QC_FP8(0x0B),
    QC_FP8(0x0C), QC_FP8(0x0D), QC_FP8(0x0E), QC_FP8(0x0F),
    // Row 0x10..0x1F
    QC_FP8(0x10), QC_FP8(0x11), QC_FP8(0x12), QC_FP8(0x13),
    QC_FP8(0x14), QC_FP8(0x15), QC_FP8(0x16), QC_FP8(0x17),
    QC_FP8(0x18), QC_FP8(0x19), QC_FP8(0x1A), QC_FP8(0x1B),
    QC_FP8(0x1C), QC_FP8(0x1D), QC_FP8(0x1E), QC_FP8(0x1F),
    // Row 0x20..0x2F
    QC_FP8(0x20), QC_FP8(0x21), QC_FP8(0x22), QC_FP8(0x23),
    QC_FP8(0x24), QC_FP8(0x25), QC_FP8(0x26), QC_FP8(0x27),
    QC_FP8(0x28), QC_FP8(0x29), QC_FP8(0x2A), QC_FP8(0x2B),
    QC_FP8(0x2C), QC_FP8(0x2D), QC_FP8(0x2E), QC_FP8(0x2F),
    // Row 0x30..0x3F
    QC_FP8(0x30), QC_FP8(0x31), QC_FP8(0x32), QC_FP8(0x33),
    QC_FP8(0x34), QC_FP8(0x35), QC_FP8(0x36), QC_FP8(0x37),
    QC_FP8(0x38), QC_FP8(0x39), QC_FP8(0x3A), QC_FP8(0x3B),
    QC_FP8(0x3C), QC_FP8(0x3D), QC_FP8(0x3E), QC_FP8(0x3F),
    // Row 0x40..0x4F
    QC_FP8(0x40), QC_FP8(0x41), QC_FP8(0x42), QC_FP8(0x43),
    QC_FP8(0x44), QC_FP8(0x45), QC_FP8(0x46), QC_FP8(0x47),
    QC_FP8(0x48), QC_FP8(0x49), QC_FP8(0x4A), QC_FP8(0x4B),
    QC_FP8(0x4C), QC_FP8(0x4D), QC_FP8(0x4E), QC_FP8(0x4F),
    // Row 0x50..0x5F
    QC_FP8(0x50), QC_FP8(0x51), QC_FP8(0x52), QC_FP8(0x53),
    QC_FP8(0x54), QC_FP8(0x55), QC_FP8(0x56), QC_FP8(0x57),
    QC_FP8(0x58), QC_FP8(0x59), QC_FP8(0x5A), QC_FP8(0x5B),
    QC_FP8(0x5C), QC_FP8(0x5D), QC_FP8(0x5E), QC_FP8(0x5F),
    // Row 0x60..0x6F
    QC_FP8(0x60), QC_FP8(0x61), QC_FP8(0x62), QC_FP8(0x63),
    QC_FP8(0x64), QC_FP8(0x65), QC_FP8(0x66), QC_FP8(0x67),
    QC_FP8(0x68), QC_FP8(0x69), QC_FP8(0x6A), QC_FP8(0x6B),
    QC_FP8(0x6C), QC_FP8(0x6D), QC_FP8(0x6E), QC_FP8(0x6F),
    // Row 0x70..0x7F
    QC_FP8(0x70), QC_FP8(0x71), QC_FP8(0x72), QC_FP8(0x73),
    QC_FP8(0x74), QC_FP8(0x75), QC_FP8(0x76), QC_FP8(0x77),
    QC_FP8(0x78), QC_FP8(0x79), QC_FP8(0x7A), QC_FP8(0x7B),
    QC_FP8(0x7C), QC_FP8(0x7D), QC_FP8(0x7E), QC_FP8(0x7F),
    // Row 0x80..0x8F
    QC_FP8(0x80), QC_FP8(0x81), QC_FP8(0x82), QC_FP8(0x83),
    QC_FP8(0x84), QC_FP8(0x85), QC_FP8(0x86), QC_FP8(0x87),
    QC_FP8(0x88), QC_FP8(0x89), QC_FP8(0x8A), QC_FP8(0x8B),
    QC_FP8(0x8C), QC_FP8(0x8D), QC_FP8(0x8E), QC_FP8(0x8F),
    // Row 0x90..0x9F
    QC_FP8(0x90), QC_FP8(0x91), QC_FP8(0x92), QC_FP8(0x93),
    QC_FP8(0x94), QC_FP8(0x95), QC_FP8(0x96), QC_FP8(0x97),
    QC_FP8(0x98), QC_FP8(0x99), QC_FP8(0x9A), QC_FP8(0x9B),
    QC_FP8(0x9C), QC_FP8(0x9D), QC_FP8(0x9E), QC_FP8(0x9F),
    // Row 0xA0..0xAF
    QC_FP8(0xA0), QC_FP8(0xA1), QC_FP8(0xA2), QC_FP8(0xA3),
    QC_FP8(0xA4), QC_FP8(0xA5), QC_FP8(0xA6), QC_FP8(0xA7),
    QC_FP8(0xA8), QC_FP8(0xA9), QC_FP8(0xAA), QC_FP8(0xAB),
    QC_FP8(0xAC), QC_FP8(0xAD), QC_FP8(0xAE), QC_FP8(0xAF),
    // Row 0xB0..0xBF
    QC_FP8(0xB0), QC_FP8(0xB1), QC_FP8(0xB2), QC_FP8(0xB3),
    QC_FP8(0xB4), QC_FP8(0xB5), QC_FP8(0xB6), QC_FP8(0xB7),
    QC_FP8(0xB8), QC_FP8(0xB9), QC_FP8(0xBA), QC_FP8(0xBB),
    QC_FP8(0xBC), QC_FP8(0xBD), QC_FP8(0xBE), QC_FP8(0xBF),
    // Row 0xC0..0xCF
    QC_FP8(0xC0), QC_FP8(0xC1), QC_FP8(0xC2), QC_FP8(0xC3),
    QC_FP8(0xC4), QC_FP8(0xC5), QC_FP8(0xC6), QC_FP8(0xC7),
    QC_FP8(0xC8), QC_FP8(0xC9), QC_FP8(0xCA), QC_FP8(0xCB),
    QC_FP8(0xCC), QC_FP8(0xCD), QC_FP8(0xCE), QC_FP8(0xCF),
    // Row 0xD0..0xDF
    QC_FP8(0xD0), QC_FP8(0xD1), QC_FP8(0xD2), QC_FP8(0xD3),
    QC_FP8(0xD4), QC_FP8(0xD5), QC_FP8(0xD6), QC_FP8(0xD7),
    QC_FP8(0xD8), QC_FP8(0xD9), QC_FP8(0xDA), QC_FP8(0xDB),
    QC_FP8(0xDC), QC_FP8(0xDD), QC_FP8(0xDE), QC_FP8(0xDF),
    // Row 0xE0..0xEF
    QC_FP8(0xE0), QC_FP8(0xE1), QC_FP8(0xE2), QC_FP8(0xE3),
    QC_FP8(0xE4), QC_FP8(0xE5), QC_FP8(0xE6), QC_FP8(0xE7),
    QC_FP8(0xE8), QC_FP8(0xE9), QC_FP8(0xEA), QC_FP8(0xEB),
    QC_FP8(0xEC), QC_FP8(0xED), QC_FP8(0xEE), QC_FP8(0xEF),
    // Row 0xF0..0xFF
    QC_FP8(0xF0), QC_FP8(0xF1), QC_FP8(0xF2), QC_FP8(0xF3),
    QC_FP8(0xF4), QC_FP8(0xF5), QC_FP8(0xF6), QC_FP8(0xF7),
    QC_FP8(0xF8), QC_FP8(0xF9), QC_FP8(0xFA), QC_FP8(0xFB),
    QC_FP8(0xFC), QC_FP8(0xFD), QC_FP8(0xFE), QC_FP8(0xFF),
    #undef QC_FP8
    #undef FP8_TBL
};

/// @brief 16-entry nibble -> FP16 bit-pattern LUT for INT4 sign-magnitude.
///
/// Maps a 4-bit sign-magnitude nibble to the uint16_t bit pattern of the
/// corresponding __half value.  Used to build FP16 output without going
/// through float arithmetic.
///
/// Nibble values 0..7 map to +0.0..+7.0 in FP16:
///   0x0000, 0x3C00, 0x4000, 0x4200, 0x4400, 0x4500, 0x4600, 0x4700
/// Nibble values 8..15 map to -0.0..-7.0 in FP16:
///   0x8000, 0xBC00, 0xC000, 0xC200, 0xC400, 0xC500, 0xC600, 0xC700
__device__ __constant__ static uint16_t qc_to_fp16_nib_lut[16] = {
    0x0000, 0x3C00, 0x4000, 0x4200, 0x4400, 0x4500, 0x4600, 0x4700,  // +0..+7
    0x8000, 0xBC00, 0xC000, 0xC200, 0xC400, 0xC500, 0xC600, 0xC700   // -0..-7
};

// ---- INT4 -> FP8 interleaved (LUT-based, 128-bit vectorized) ----

/// @brief CUDA kernel: convert packed INT4 sign-magnitude complex data directly
///        to interleaved FP8 E4M3 using a 256-entry constant-memory LUT.
///
/// Each input byte maps to 2 output FP8 bytes via a single LUT lookup,
/// eliminating all per-nibble float arithmetic, int-to-float conversion,
/// and FP8 constructor calls.  Processes 16 complex elements per thread
/// per iteration using 128-bit vectorized loads and stores.
///
/// @param input                Device pointer to packed INT4 bytes.
/// @param output               Device pointer to FP8 E4M3 output (2 * num_complex_elements bytes).
/// @param num_complex_elements Total number of complex elements to convert.
__global__ static void cast_int4_to_fp8_interleaved_kernel(
    const uint8_t* __restrict__ input,
    uint8_t* __restrict__ output,
    int64_t num_complex_elements)
{
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;

    // Process 16 complex elements per iteration: 16 bytes in (uint4), 32 bytes out (2 x uint4)
    for (int64_t i = idx * 16; i < num_complex_elements; i += stride * 16) {
        if (i + 15 < num_complex_elements) {
            // 128-bit vectorized load: 16 QC bytes
            uint4 packed = *reinterpret_cast<const uint4*>(input + i);
            const uint8_t* bytes = reinterpret_cast<const uint8_t*>(&packed);

            // LUT decode: each byte -> 2 FP8 bytes (uint16_t)
            uint16_t pairs[16];
            #pragma unroll
            for (int j = 0; j < 16; ++j) {
                pairs[j] = qc_to_fp8_lut_256[bytes[j]];
            }

            // 128-bit vectorized stores: 32 output bytes = 2 x uint4
            *reinterpret_cast<uint4*>(output + 2 * i)      = *reinterpret_cast<const uint4*>(&pairs[0]);
            *reinterpret_cast<uint4*>(output + 2 * i + 16)  = *reinterpret_cast<const uint4*>(&pairs[8]);
        } else {
            // Scalar tail: handle remaining elements
            for (int64_t j = i; j < num_complex_elements && j < i + 16; ++j) {
                uint16_t fp8_pair = qc_to_fp8_lut_256[input[j]];
                output[2 * j]     = static_cast<uint8_t>(fp8_pair & 0xFF);
                output[2 * j + 1] = static_cast<uint8_t>(fp8_pair >> 8);
            }
        }
    }
}

/// @brief Launch INT4 sign-magnitude to FP8 E4M3 interleaved conversion.
///
/// Output: 2 bytes per complex element in [Re_fp8, Im_fp8] interleaved layout,
/// directly consumable by the direct HERK kernel's precast buffer.
/// Uses a 256-entry constant-memory LUT for zero-arithmetic conversion.
///
/// @param input                Device pointer to packed INT4 bytes.
/// @param output               Device pointer to FP8 E4M3 output (2 * num_complex_elements bytes).
/// @param num_complex_elements Number of complex elements (= number of input bytes).
/// @param stream               CUDA stream for async execution.
static void cast_int4_to_fp8_interleaved(
    const uint8_t* input, __nv_fp8_e4m3* output,
    int64_t num_complex_elements, cudaStream_t stream)
{
    if (num_complex_elements == 0) return;
    const int block = 256;
    // Each thread processes 16 elements per iteration
    const int64_t grid = std::min((num_complex_elements + block * 16 - 1) / (block * 16),
                                  static_cast<int64_t>(1024));
    cast_int4_to_fp8_interleaved_kernel<<<grid, block, 0, stream>>>(
        input, reinterpret_cast<uint8_t*>(output), num_complex_elements);
}

// ---- INT4 -> FP16 planar (LUT-based, 128-bit vectorized) ----

/// @brief CUDA kernel: convert packed INT4 sign-magnitude complex data directly
///        to planar FP16 (separate Re and Im buffers) using a nibble LUT.
///
/// Each nibble maps to its FP16 bit pattern via a 16-entry LUT lookup,
/// eliminating per-element int-to-float and float-to-half conversions.
/// Processes 16 complex elements per thread per iteration using 128-bit
/// vectorized loads and stores.
///
/// @param input                Device pointer to packed INT4 bytes.
/// @param out_real             Device pointer to FP16 real output (num_complex_elements values).
/// @param out_imag             Device pointer to FP16 imag output (num_complex_elements values).
/// @param num_complex_elements Total number of complex elements to convert.
__global__ static void cast_int4_to_fp16_planar_kernel(
    const uint8_t* __restrict__ input,
    __half* __restrict__ out_real,
    __half* __restrict__ out_imag,
    int64_t num_complex_elements)
{
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;

    // Process 8 complex elements per iteration: 8 bytes in, 16 bytes Re out, 16 bytes Im out
    for (int64_t i = idx * 8; i < num_complex_elements; i += stride * 8) {
        if (i + 7 < num_complex_elements) {
            // 64-bit vectorized load: 8 QC bytes
            uint2 packed = *reinterpret_cast<const uint2*>(input + i);
            const uint8_t* bytes = reinterpret_cast<const uint8_t*>(&packed);

            uint16_t re[8], im[8];
            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                re[j] = qc_to_fp16_nib_lut[(bytes[j] >> 4) & 0xF];
                im[j] = qc_to_fp16_nib_lut[bytes[j] & 0xF];
            }

            // 128-bit vectorized stores: 8 x FP16 = 16 bytes = uint4
            *reinterpret_cast<uint4*>(out_real + i) = *reinterpret_cast<const uint4*>(re);
            *reinterpret_cast<uint4*>(out_imag + i) = *reinterpret_cast<const uint4*>(im);
        } else {
            for (int64_t j = i; j < num_complex_elements && j < i + 8; ++j) {
                uint16_t re_bits = qc_to_fp16_nib_lut[(input[j] >> 4) & 0xF];
                uint16_t im_bits = qc_to_fp16_nib_lut[input[j] & 0xF];
                out_real[j] = *reinterpret_cast<const __half*>(&re_bits);
                out_imag[j] = *reinterpret_cast<const __half*>(&im_bits);
            }
        }
    }
}

/// @brief Launch the INT4 sign-magnitude to planar FP16 conversion kernel.
///
/// Produces separate Re and Im FP16 buffers directly, skipping the interleaved
/// intermediate and subsequent deinterleave pass.  Uses LUT-based conversion
/// without float arithmetic.
///
/// @param input                Device pointer to packed INT4 bytes.
/// @param out_real             Device pointer to FP16 real output.
/// @param out_imag             Device pointer to FP16 imag output.
/// @param num_complex_elements Number of complex elements (= number of input bytes).
/// @param stream               CUDA stream for async execution.
static void cast_int4_to_fp16_planar(
    const uint8_t* input, __half* out_real, __half* out_imag,
    int64_t num_complex_elements, cudaStream_t stream)
{
    if (num_complex_elements == 0) return;
    const int block = 256;
    // Each thread processes 8 elements per iteration
    const int64_t grid = std::min((num_complex_elements + block * 8 - 1) / (block * 8),
                                  static_cast<int64_t>(1024));
    cast_int4_to_fp16_planar_kernel<<<grid, block, 0, stream>>>(
        input, out_real, out_imag, num_complex_elements);
}

// ---- INT4 -> FP8 planar (LUT-based, for baseline GEMM path) ----

// ===================================================================
// Unified HERK
// ===================================================================

/// @copydoc CutlassComplexGemm::herk
int CutlassComplexGemm::herk(
    const void* A, void* C,
    int N, int K, int batch_count,
    InputPrecision input, ComputePrecision compute,
    OutputPrecision output, OutputFormat format,
    float alpha, float beta,
    cudaStream_t stream, bool tune)
{
    // Validate compute precision
#ifdef COMPLEX_FP8_TARGET_SM90
    if (compute != ComputePrecision::FP8)
        throw std::runtime_error("SM90 HERK only supports FP8 compute precision");
#endif

    // FP32 output with block-scaled (FP6/FP4) not yet implemented in baseline path
    if (output == OutputPrecision::FP32 && compute != ComputePrecision::FP8)
        throw std::runtime_error("FP32 output HERK currently only supports FP8 compute precision");

    // Strategy tuning / mode selection
    int prec_key = precision_cache_key(compute);
    CutlassParams params;
    if (tune) {
        params = tune_herk(impl_->gemm, N, K, batch_count, prec_key, stream);
    } else {
        ensure_viable_herk_mode(impl_->gemm);
        params = make_params(stream);
    }
#ifndef COMPLEX_FP8_TARGET_SM90
    params.precision = map_compute(compute);
#endif

    // ---- CUDA graph replay: check cache before doing any work ----
    // Graph captures the full INT4→FP8 cast + HERK dispatch pipeline.
    // Requires init_herk() pre-allocation (no cudaMallocAsync during capture).
    bool is_direct = (impl_->gemm.herk_mode() == InternalHerkMode::ForceDirect);
    if (impl_->use_direct_herk_graph && is_direct && compute == ComputePrecision::FP8) {
        int inp_key = static_cast<int>(input);
        int out_key = static_cast<int>(output);
        auto* entry = impl_->find_direct_graph(
            N, K, batch_count, alpha, beta, A, C, inp_key, out_key);
        if (entry) {
            CUDA_CHECK(cudaGraphLaunch(entry->exec, stream));
            return 0;
        }

        // Cache miss — capture graph on the capture stream
        // All buffers must be pre-allocated (init_herk)
        impl_->gemm.ensure_streams_public();
        cudaStream_t cap = impl_->gemm.capture_stream();
        CUDA_CHECK(cudaStreamBeginCapture(cap, cudaStreamCaptureModeRelaxed));

        // INT4→FP8 interleaved cast (or FP16 precast for direct path)
        if (input == InputPrecision::INT4) {
            int64_t total_complex = static_cast<int64_t>(N) * K * batch_count;
            cast_int4_to_fp8_interleaved(
                static_cast<const uint8_t*>(A), impl_->herk_A_fp8,
                total_complex, cap);
        }

        // Dispatch HERK on capture stream
        CutlassParams cap_params = params;
        cap_params.stream = cap;
        const __nv_fp8_e4m3* cap_fp8 = (input == InputPrecision::INT4)
            ? impl_->herk_A_fp8 : nullptr;
        const __half* cap_fp16 = (input == InputPrecision::FP16)
            ? static_cast<const __half*>(A) : nullptr;

        cutlass::Status cap_status;
        if (output == OutputPrecision::FP32) {
            cap_status = impl_->gemm.HERK_batched_fp32out(
                cap_fp16, static_cast<float*>(C), N, K, batch_count,
                alpha, beta,
                HerkOp::NoTrans, FillMode::Lower,
                cap_params, false, cap_fp8);
        } else {
            cap_status = impl_->gemm.HERK_batched(
                cap_fp16, static_cast<__half*>(C), N, K, batch_count,
                alpha, beta,
                HerkOp::NoTrans, FillMode::Lower,
                cap_params, false, cap_fp8);
        }

        if (cap_status != cutlass::Status::kSuccess) {
            cudaGraph_t abandoned;
            cudaStreamEndCapture(cap, &abandoned);
            if (abandoned) cudaGraphDestroy(abandoned);
            return static_cast<int>(cap_status);
        }

        cudaGraph_t graph;
        CUDA_CHECK(cudaStreamEndCapture(cap, &graph));
        cudaGraphExec_t exec;
        CUDA_CHECK(cudaGraphInstantiateWithFlags(&exec, graph, 0));
        cudaGraphDestroy(graph);

        auto* slot = impl_->alloc_direct_graph_slot();
        slot->store(N, K, batch_count, alpha, beta, A, C, inp_key, out_key, exec);

        CUDA_CHECK(cudaGraphLaunch(exec, stream));
        return 0;
    }

    // Handle INT4 input: produce only the format(s) needed by the selected path.
    //
    // Direct HERK path:   needs FP8 interleaved only (no FP16)
    // Baseline HERK path: needs FP16 planar only (no FP8 interleaved)
    // Auto mode:          needs both (direct may be selected at dispatch time)
    //
    // All conversions use constant-memory LUTs for zero-arithmetic decode.
    const __half* A_fp16 = nullptr;
    const __nv_fp8_e4m3* A_fp8 = nullptr;
    __half* A_alloc = nullptr;  // non-null if we allocated FP16 temp (must free)
    bool pre_deinterleaved = false;
    if (input == InputPrecision::INT4) {
        int64_t total_complex = static_cast<int64_t>(N) * K * batch_count;
        auto herk_mode = impl_->gemm.herk_mode();

        // Only produce FP8 interleaved if direct path might be used
        bool need_fp8 = (herk_mode != InternalHerkMode::ForceBaseline);
        if (need_fp8) {
            int64_t fp8_needed = total_complex * 2;  // 2 FP8 bytes per complex
            if (!impl_->herk_A_fp8 || impl_->herk_A_fp8_cap < fp8_needed) {
                if (impl_->herk_A_fp8) cudaFree(impl_->herk_A_fp8);
                CUDA_CHECK(cudaMalloc(&impl_->herk_A_fp8, fp8_needed * sizeof(__nv_fp8_e4m3)));
                impl_->herk_A_fp8_cap = fp8_needed;
            }
            cast_int4_to_fp8_interleaved(
                static_cast<const uint8_t*>(A), impl_->herk_A_fp8,
                total_complex, stream);
            A_fp8 = impl_->herk_A_fp8;
        }

        // Only produce FP16 planar if baseline path might be used
        bool need_fp16 = (herk_mode != InternalHerkMode::ForceDirect);
        if (need_fp16) {
            CUDA_CHECK(cudaMallocAsync(&A_alloc, total_complex * 2 * sizeof(__half), stream));
            cast_int4_to_fp16_planar(
                static_cast<const uint8_t*>(A), A_alloc, A_alloc + total_complex,
                total_complex, stream);
            A_fp16 = A_alloc;
            pre_deinterleaved = true;
        }
    } else {
        A_fp16 = static_cast<const __half*>(A);
    }

    // Dispatch based on output precision
    int result;
    try {
        if (output == OutputPrecision::FP32) {
            auto status = impl_->gemm.HERK_batched_fp32out(
                A_fp16, static_cast<float*>(C), N, K, batch_count,
                alpha, beta,
                HerkOp::NoTrans, FillMode::Lower,
                params, pre_deinterleaved, A_fp8);
            result = static_cast<int>(status);
        } else {
            auto status = impl_->gemm.HERK_batched(
                A_fp16, static_cast<__half*>(C), N, K, batch_count,
                alpha, beta,
                HerkOp::NoTrans, FillMode::Lower,
                params, pre_deinterleaved, A_fp8);
            result = static_cast<int>(status);
        }
    } catch (...) {
        if (A_alloc) cudaFreeAsync(A_alloc, stream);
        throw;
    }

    if (A_alloc) cudaFreeAsync(A_alloc, stream);
    return result;
}

} // namespace cutlass_gemm_api
