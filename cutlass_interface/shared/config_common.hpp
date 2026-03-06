// ========================================================================================
// Shared Configuration — Common enums and macros for SM90 and SM100/SM120
// ========================================================================================
//
// This header is #include'd (not textually pasted) from within each architecture's
// config.hpp, which is itself textually included inside the arch-specific namespace
// (gemm_complex_fp8 or gemm_complex_sm100). The enums defined here therefore live
// in whichever namespace is active at the inclusion site.
//
// Since SM90 and SM100 are never compiled together (PIMPL dispatches at build time),
// there is no ODR conflict.

#pragma once

#include <cuda_runtime.h>
#include <vector>

// ========================================================================================
// Error handling utilities
// ========================================================================================

#ifndef GEMM_COMPLEX_ERROR_MACROS_DEFINED
#define GEMM_COMPLEX_ERROR_MACROS_DEFINED

#define CUTLASS_CHECK(status)                                                         \
  do {                                                                                \
    cutlass::Status _s = (status);                                                    \
    if (_s != cutlass::Status::kSuccess) {                                            \
      throw std::runtime_error(                                                       \
          std::string("CUTLASS error: ") + cutlassGetStatusString(_s) +               \
          " at " + __FILE__ + ":" + std::to_string(__LINE__));                        \
    }                                                                                 \
  } while (0)

#define CUDA_CHECK(call)                                                              \
  do {                                                                                \
    cudaError_t _e = (call);                                                          \
    if (_e != cudaSuccess) {                                                          \
      throw std::runtime_error(                                                       \
          std::string("CUDA error: ") + cudaGetErrorString(_e) +                      \
          " at " + __FILE__ + ":" + std::to_string(__LINE__));                        \
    }                                                                                 \
  } while (0)

#endif // GEMM_COMPLEX_ERROR_MACROS_DEFINED

// ========================================================================================
// Configuration Enums
// ========================================================================================

/// Which FP8 variant to use for compute
enum class FP8Variant {
    E4M3,   // cutlass::float_e4m3_t — better precision, range ±448
    E5M2    // cutlass::float_e5m2_t — better range ±57344, less precision
};

/// Complex GEMM operation mode
enum class ComplexMode {
    Standard,    // C = α(A × B) + βC
    Hermitian    // C = α(A × B^H) + βC,  where B^H = conj(B)^T
};

/// Gram matrix (self-product) operation mode
enum class GramMode {
    AAH,    // C = α(A × A^H) + βC,  result is [M × M], Hermitian positive semi-definite
    AHA     // C = α(A^H × A) + βC,  result is [K × K], Hermitian positive semi-definite
};

/// Fill mode for triangular/Hermitian storage (matches BLAS conventions)
enum class FillMode {
    Upper,   // Store upper triangle: C(i,j) valid for j >= i
    Lower,   // Store lower triangle: C(i,j) valid for i >= j
    Full     // Store entire matrix (but still enforce real diagonal)
};

/// HERK operation variant (matches BLAS UPLO+TRANS convention)
enum class HerkOp {
    NoTrans,     // C = α·A·A^H + β·C,  A is [N × K], C is [N × N]
    ConjTrans    // C = α·A^H·A + β·C,  A is [K × N], C is [N × N]
};

/// HERK compute strategy
enum class HerkStrategy {
    Baseline,       // 3 full-matrix sub-GEMMs (current production mode)
    TriangleAware   // 2 lower-triangle + 1 full sub-GEMM
                    // 33% fewer FLOPs via block-row decomposition
};

/// HERK dispatch mode for FP8 NoTrans
enum class HerkMode {
    Auto,           // Auto-select: direct for small K, baseline for large K
    ForceDirect,    // Always use direct single-launch kernel (FP8 NoTrans only)
    ForceBaseline,  // Always use baseline multi-launch path
    ForceFused = ForceDirect  // Backward compat alias
};

/// GEMM dispatch mode: direct PTX kernel vs 4M CUTLASS sub-GEMMs.
/// Direct kernel uses mma.sync.aligned.m16n8k32 with conjugate permutation trick.
/// 4M path decomposes complex GEMM into 4 real FP8 sub-GEMMs via CUTLASS.
enum class GemmMode {
    Auto,           // Roofline-based: autotuner cache first, then heuristic
    ForceDirect,    // Always use direct PTX kernel
    Force4M         // Always use 4 CUTLASS sub-GEMMs
};

/// Persistent kernel dispatch mode for direct HERK
enum class PersistentMode {
    Auto,    // Auto-select: persistent when work > sm_count*16 AND K <= K_CHUNK (64).
             // The persistent kernel's single-buffer direct load is optimal for small K.
             // At K > K_CHUNK, cp.async pipeline restart overhead per work item makes
             // persistent ~1.5-2x slower than the non-persistent precast kernel.
    ForceOn, // Always use persistent kernel for direct HERK
    ForceOff // Always use standard per-block kernel
};

/// Triangle decomposition tuning parameters.
/// Controls how the lower-triangle HERK block-row decomposition partitions N.
struct TriangleConfig {
    int target_slabs = 0;      // 0 = auto (maximize slabs while maintaining occupancy)
    int min_slab_height = 0;   // 0 = auto (occupancy-aware)
    bool graduated = false;    // sqrt-spaced slab boundaries (experimental)
    bool verbose = false;      // print slab decomposition details
    bool use_cuda_graph = false; // cache slab launches as CUDA graph for replay
};


// ========================================================================================
// GemmConfig — Runtime-Selectable CUTLASS Tile/Cluster/Schedule Configurations
// ========================================================================================
//
// All configurations are always present in the enum (no #ifdef guards). Dispatch
// returns kErrorNotSupported for configs not valid on the current architecture.
// Use is_config_valid_for_arch() to check validity, all_baseline_configs() to get
// the set of FP8 configs available for autotuning.
//
// Naming: FP8_T{M}x{N}_C{cm}x{cn}[_S{stages}|_SAuto][_Coop|_PP][_2SM]
//   T = tile MxNxK (K always 128)
//   C = cluster shape
//   S3 = StageCount<3>, SAuto = StageCountAutoCarveout
//   Coop = CooperativeFP8FastAccum (SM90)
//   PP = PingpongFP8FastAccum (SM90)
//   2SM = 2SM cooperative (SM100 only)

enum class GemmConfig : int {
    // === SM120 + SM100: 1SM, 1x1 cluster (always compiled on Blackwell) ===
    FP8_T128x64_C1x1_S3,       // SM120 HERK-optimal: 128×64 tile, 3-stage (~82 KB)
    FP8_T128x64_C1x1_SAuto,    // SM120: 128×64 tile, auto-carveout stages
    FP8_T128x128_C1x1_SAuto,   // SM120/SM100 GEMM-optimal: 128×128, auto stages (~55 KB@2stg)
    FP8_T128x128_C1x1_S3,      // SM100 default: 128×128 tile, 3-stage (~164 KB, SM100 only)
    FP8_T128x256_C1x1_SAuto,   // Wide-N: 128×256 tile, auto stages (SM100 only)

    // === SM100 only: cluster variants ===
    FP8_T128x128_C1x2,         // 1×2 cluster (TMA N-multicast)
    FP8_T128x128_C2x2,         // 2×2 cluster
    FP8_T128x256_C1x2,         // Wide-N + 1×2 cluster
    FP8_T256x128_C2x1_2SM,     // 2SM cooperative (M-doubled)
    FP8_T256x256_C2x2_2SM,     // 2SM cooperative (large tile)

    // === SM90: Hopper FP8 configs ===
    FP8_T128x256_C1x1_Coop,    // SM90 default: CooperativeFP8FastAccum
    FP8_T128x128_C1x1_Coop,    // SM90 smaller tile: Cooperative
    FP8_T128x256_C1x1_PP,      // SM90 Pingpong: PingpongFP8FastAccum
    FP8_T128x128_C1x1_PP,      // SM90 smaller tile: Pingpong
    FP8_T128x256_C1x2_Coop,    // SM90 cluster 1×2: Cooperative
    FP8_T64x128_C1x1_Coop,     // SM90 small-M tile: Cooperative

    // === Block-scaled tile selection ===
    SmallM,                     // 64×128 block-scaled tile for M≤64 (SM100 only)

    // Sentinel
    NUM_CONFIGS,

    // Architecture-dependent default alias
#if defined(COMPLEX_FP8_TARGET_SM90)
    Default = FP8_T128x256_C1x1_Coop,
#elif defined(COMPLEX_FP8_SM100_TARGET_SM120)
    Default = FP8_T128x64_C1x1_S3,
#else
    Default = FP8_T128x128_C1x1_S3,
#endif
};

/// Human-readable config name for logging and cache files.
inline const char* config_name(GemmConfig c) {
    switch (c) {
    case GemmConfig::FP8_T128x64_C1x1_S3:    return "FP8_T128x64_C1x1_S3";
    case GemmConfig::FP8_T128x64_C1x1_SAuto:  return "FP8_T128x64_C1x1_SAuto";
    case GemmConfig::FP8_T128x128_C1x1_SAuto: return "FP8_T128x128_C1x1_SAuto";
    case GemmConfig::FP8_T128x128_C1x1_S3:    return "FP8_T128x128_C1x1_S3";
    case GemmConfig::FP8_T128x256_C1x1_SAuto: return "FP8_T128x256_C1x1_SAuto";
    case GemmConfig::FP8_T128x128_C1x2:       return "FP8_T128x128_C1x2";
    case GemmConfig::FP8_T128x128_C2x2:       return "FP8_T128x128_C2x2";
    case GemmConfig::FP8_T128x256_C1x2:       return "FP8_T128x256_C1x2";
    case GemmConfig::FP8_T256x128_C2x1_2SM:   return "FP8_T256x128_C2x1_2SM";
    case GemmConfig::FP8_T256x256_C2x2_2SM:   return "FP8_T256x256_C2x2_2SM";
    case GemmConfig::FP8_T128x256_C1x1_Coop:  return "FP8_T128x256_C1x1_Coop";
    case GemmConfig::FP8_T128x128_C1x1_Coop:  return "FP8_T128x128_C1x1_Coop";
    case GemmConfig::FP8_T128x256_C1x1_PP:    return "FP8_T128x256_C1x1_PP";
    case GemmConfig::FP8_T128x128_C1x1_PP:    return "FP8_T128x128_C1x1_PP";
    case GemmConfig::FP8_T128x256_C1x2_Coop:  return "FP8_T128x256_C1x2_Coop";
    case GemmConfig::FP8_T64x128_C1x1_Coop:   return "FP8_T64x128_C1x1_Coop";
    case GemmConfig::SmallM:                   return "SmallM_64x128";
    default: return "Unknown";
    }
}

/// Check if a GemmConfig is valid for the current compile-time architecture.
inline bool is_config_valid_for_arch(GemmConfig c) {
#if defined(COMPLEX_FP8_TARGET_SM90)
    switch (c) {
    case GemmConfig::FP8_T128x256_C1x1_Coop:
    case GemmConfig::FP8_T128x128_C1x1_Coop:
    case GemmConfig::FP8_T128x256_C1x1_PP:
    case GemmConfig::FP8_T128x128_C1x1_PP:
    case GemmConfig::FP8_T128x256_C1x2_Coop:
    case GemmConfig::FP8_T64x128_C1x1_Coop:
        return true;
    default:
        return false;
    }
#elif defined(COMPLEX_FP8_SM100_TARGET_SM120)
    // SM120: 1×1 cluster only, no 2SM, no SmallM, 99 KB SMEM limit
    switch (c) {
    case GemmConfig::FP8_T128x64_C1x1_S3:
    case GemmConfig::FP8_T128x64_C1x1_SAuto:
    case GemmConfig::FP8_T128x128_C1x1_SAuto:
        return true;
    default:
        return false;
    }
#else
    // SM100: all Blackwell configs (1SM + cluster + 2SM + SmallM)
    switch (c) {
    case GemmConfig::FP8_T128x64_C1x1_S3:
    case GemmConfig::FP8_T128x64_C1x1_SAuto:
    case GemmConfig::FP8_T128x128_C1x1_SAuto:
    case GemmConfig::FP8_T128x128_C1x1_S3:
    case GemmConfig::FP8_T128x256_C1x1_SAuto:
    case GemmConfig::FP8_T128x128_C1x2:
    case GemmConfig::FP8_T128x128_C2x2:
    case GemmConfig::FP8_T128x256_C1x2:
    case GemmConfig::FP8_T256x128_C2x1_2SM:
    case GemmConfig::FP8_T256x256_C2x2_2SM:
    case GemmConfig::SmallM:
        return true;
    default:
        return false;
    }
#endif
}

/// Return all FP8 baseline GemmConfigs valid for the current architecture.
/// Used by the strategy autotuner to enumerate configs for benchmarking.
/// Does NOT include SmallM (block-scaled only) or NUM_CONFIGS/Default.
inline std::vector<GemmConfig> all_baseline_configs() {
    std::vector<GemmConfig> configs;
#if defined(COMPLEX_FP8_TARGET_SM90)
    configs.push_back(GemmConfig::FP8_T128x256_C1x1_Coop);
    configs.push_back(GemmConfig::FP8_T128x128_C1x1_Coop);
    configs.push_back(GemmConfig::FP8_T128x256_C1x1_PP);
    configs.push_back(GemmConfig::FP8_T128x128_C1x1_PP);
    configs.push_back(GemmConfig::FP8_T128x256_C1x2_Coop);
    configs.push_back(GemmConfig::FP8_T64x128_C1x1_Coop);
#elif defined(COMPLEX_FP8_SM100_TARGET_SM120)
    configs.push_back(GemmConfig::FP8_T128x64_C1x1_S3);
    configs.push_back(GemmConfig::FP8_T128x64_C1x1_SAuto);
    configs.push_back(GemmConfig::FP8_T128x128_C1x1_SAuto);
#else
    configs.push_back(GemmConfig::FP8_T128x128_C1x1_S3);
    configs.push_back(GemmConfig::FP8_T128x64_C1x1_S3);
    configs.push_back(GemmConfig::FP8_T128x64_C1x1_SAuto);
    configs.push_back(GemmConfig::FP8_T128x128_C1x1_SAuto);
    configs.push_back(GemmConfig::FP8_T128x256_C1x1_SAuto);
    configs.push_back(GemmConfig::FP8_T128x128_C1x2);
    configs.push_back(GemmConfig::FP8_T128x128_C2x2);
    configs.push_back(GemmConfig::FP8_T128x256_C1x2);
#endif
    return configs;
}

/// Select optimal GemmConfig based on problem dimensions.
/// Returns SmallM when M ≤ 64 and compiled for SM100 (not SM120).
/// For simple heuristic use; autotuner overrides with benchmarked choice.
inline GemmConfig select_gemm_config(int M, int N = 0, int K = 0,
                                      int batch = 0, int sm_count = 0) {
#if defined(COMPLEX_SM100_ENABLE_SMALL_M) && !defined(COMPLEX_FP8_SM100_TARGET_SM120)
    if (M > 0 && M <= 64) return GemmConfig::SmallM;
#endif
    (void)M; (void)N; (void)K; (void)batch; (void)sm_count;
    return GemmConfig::Default;
}

/// Select optimal GemmConfig for GEMM (not HERK) based on problem dimensions.
///
/// For GEMM workloads, wider N tiles can improve throughput when N is large,
/// because the larger tile reduces the number of output tile columns and
/// improves data reuse of A operand. On SM120, the default HERK tile is
/// 128x64 (optimal for HERK scratch fitting in SMEM), but for GEMM the
/// 128x128 tile avoids wave quantization waste when N >> 64.
///
/// Selection logic:
///   - SmallM (64x128): M <= 64 (SM100 only, block-scaled)
///   - 128x128 (SAuto): N >= 256 on SM120 (overrides HERK-default 128x64)
///   - Default: architecture default otherwise
///
/// The autotuner overrides this with a benchmarked choice when active.
inline GemmConfig select_gemm_config_for_gemm(int M, int N = 0, int K = 0,
                                               int batch = 0, int sm_count = 0) {
#if defined(COMPLEX_SM100_ENABLE_SMALL_M) && !defined(COMPLEX_FP8_SM100_TARGET_SM120)
    if (M > 0 && M <= 64) return GemmConfig::SmallM;
#endif
#if defined(COMPLEX_FP8_SM100_TARGET_SM120)
    // SM120: default HERK tile is 128x64 (S3), but for GEMM the 128x128 SAuto
    // tile avoids 2x wave quantization waste when N >= 256 (4+ output columns
    // per wave vs 2 per wave with 128x64). At N < 256 the 128x64 tile is fine.
    if (N >= 256) return GemmConfig::FP8_T128x128_C1x1_SAuto;
#elif !defined(COMPLEX_FP8_TARGET_SM90)
    // SM100: default is 128x128 S3. For very large N (>= 1024), the 128x256
    // wide-N tile can improve throughput by reducing output tile columns.
    if (N >= 1024) return GemmConfig::FP8_T128x256_C1x1_SAuto;
#else
    // SM90: default is 128x256 Coop, already wide. No override needed.
#endif
    (void)M; (void)N; (void)K; (void)batch; (void)sm_count;
    return GemmConfig::Default;
}

/// Return a multi-line string summarizing all compiled FP8 configs for --help output.
/// Each line: "  [ordinal] config_name  (annotation)"
inline std::string compiled_configs_summary() {
    std::string s;
    auto configs = all_baseline_configs();
    GemmConfig def = GemmConfig::Default;
    for (auto c : configs) {
        char buf[128];
        snprintf(buf, sizeof(buf), "  [%2d] %-30s%s\n",
                 static_cast<int>(c), config_name(c),
                 (c == def) ? " (default)" : "");
        s += buf;
    }
    return s;
}


// ========================================================================================
// DirectHerkConfig — Runtime-Selectable Direct HERK Kernel Variants
// ========================================================================================
//
// The direct HERK kernel uses hand-written PTX (mma.sync.aligned.m16n8k32) with
// tunable SMEM/pipeline parameters. These configs trade off SMEM usage (occupancy)
// vs compute-per-sync (K_CHUNK) and pipeline depth (NR_BUFS).
//
// Naming: K{chunk}_B{bufs}
//   K = K_CHUNK (complex samples per iteration)
//   B = NR_BUFS (cp.async pipeline buffers)

enum class DirectHerkConfig : int {
    K32_B3,    // K_CHUNK=32, 3-buf, 12 KB SMEM, OCC=8  (memory-bound sweet spot)
    K64_B2,    // K_CHUNK=64, 2-buf, 16 KB SMEM, OCC=6
    K64_B3,    // K_CHUNK=64, 3-buf, 24 KB SMEM, OCC=4  (current default)
    K128_B2,   // K_CHUNK=128, 2-buf, 32 KB SMEM, OCC=3 (compute-bound sweet spot)
    NUM_CONFIGS,
    Default = K64_B3,
};

/// Human-readable config name for logging and cache files.
inline const char* direct_herk_config_name(DirectHerkConfig c) {
    switch (c) {
    case DirectHerkConfig::K32_B3:  return "K32_B3";
    case DirectHerkConfig::K64_B2:  return "K64_B2";
    case DirectHerkConfig::K64_B3:  return "K64_B3";
    case DirectHerkConfig::K128_B2: return "K128_B2";
    default: return "Unknown";
    }
}

/// Return the K_CHUNK value for a given DirectHerkConfig.
inline int direct_herk_k_chunk(DirectHerkConfig c) {
    switch (c) {
    case DirectHerkConfig::K32_B3:  return 32;
    case DirectHerkConfig::K64_B2:
    case DirectHerkConfig::K64_B3:  return 64;
    case DirectHerkConfig::K128_B2: return 128;
    default: return 64;
    }
}

/// Return all DirectHerkConfig variants for autotuning sweeps.
inline std::vector<DirectHerkConfig> all_direct_herk_configs() {
    return {
        DirectHerkConfig::K32_B3,
        DirectHerkConfig::K64_B2,
        DirectHerkConfig::K64_B3,
        DirectHerkConfig::K128_B2,
    };
}

/// Human-readable direct config name from integer ordinal (for strategy cache).
inline const char* direct_config_name_from_int(int dc) {
    if (dc >= 0 && dc < static_cast<int>(DirectHerkConfig::NUM_CONFIGS))
        return direct_herk_config_name(static_cast<DirectHerkConfig>(dc));
    return "Unknown";
}

// ========================================================================================
// HerkTileSize — output tile dimension for direct HERK kernel
// ========================================================================================
//
// N32: 32×32 tile, AI=64, 24 KB SMEM (K64_B3), OCC=4 on SM120  (default, current)
// N64: 64×64 tile, AI=128, 48 KB SMEM (K64_B3), OCC=2 on SM120 (bandwidth-optimized)
//
// Multi-pass loading for N64: 2 × 32-row cp.async passes per buffer fill.
// Phased store: 2 phases of 32 columns each (bounds store SMEM to 16 KB).

enum class HerkTileSize : int {
    N32,           // Current: 32×32 output tile
    N64,           // New: 64×64 output tile (2x arithmetic intensity)
    NUM_SIZES,
    Auto = N32,    // Default to N32 for now (autotuner will sweep both)
};

inline const char* herk_tile_name(HerkTileSize t) {
    switch (t) {
    case HerkTileSize::N32: return "N32";
    case HerkTileSize::N64: return "N64";
    default: return "Unknown";
    }
}

inline std::vector<HerkTileSize> all_herk_tile_sizes() {
    return { HerkTileSize::N32, HerkTileSize::N64 };
}

// ========================================================================================
// HerkPipelineMode — pipeline synchronization strategy for direct HERK kernel
// ========================================================================================
//
// Sync:            __syncthreads()-based pipeline (current, all arches)
// WarpSpecialized: mbarrier-based producer-consumer (SM90+ only, eliminates sync bubbles)

enum class HerkPipelineMode : int {
    Sync,             // __syncthreads()-based (default, backward-compatible)
    WarpSpecialized,  // mbarrier-based producer-consumer (SM90+)
    NUM_MODES,
    Auto = Sync,      // Default to Sync (autotuner will sweep both when available)
};

inline const char* herk_pipeline_name(HerkPipelineMode m) {
    switch (m) {
    case HerkPipelineMode::Sync:            return "Sync";
    case HerkPipelineMode::WarpSpecialized: return "WarpSpec";
    default: return "Unknown";
    }
}

// ========================================================================================
// HERK Production vs Debug Mode
// ========================================================================================
//
// COMPLEX_FP8_HERK_FULL_MATRIX controls whether HERK computes the full symmetric matrix
// (debug/validation mode) or only the authoritative triangle (production mode).
//
//   0 = PRODUCTION (default):
//       - 3 real sub-GEMMs instead of 4 (25% fewer FLOPs)
//       - Imaginary part uses 1 sub-GEMM + O(N²) anti-symmetrize kernel
//       - Only the triangle specified by FillMode is written; other triangle is UNDEFINED
//       - Diagonal Im(C_ii) is forced to zero
//       - Non-authoritative triangle contains stale/garbage data — do NOT read it
//       - This matches BLAS ZHERK semantics (only the specified UPLO is meaningful)
//
//   1 = DEBUG:
//       - 4 real sub-GEMMs (original code path)
//       - Full matrix computed, then symmetrized by copying authoritative → non-authoritative
//       - Both triangles are valid and exactly conjugate-symmetric
//       - Use for correctness verification against reference implementations
//
// Override at compile time: -DCOMPLEX_FP8_HERK_FULL_MATRIX=1
//
#ifndef COMPLEX_FP8_HERK_FULL_MATRIX
#define COMPLEX_FP8_HERK_FULL_MATRIX 0
#endif
