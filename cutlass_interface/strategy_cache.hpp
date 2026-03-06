// ========================================================================================
// Strategy-Level Autotuning Cache
// ========================================================================================
//
// Caches the optimal *strategy combination* for each (N, K, batch_count, precision)
// problem size. Separate from the kernel-level tune_cache.hpp, which tunes individual
// kernel launch parameters (blockDim/gridDim).
//
// This cache stores the best HerkMode, HerkStrategy, CUDA graph, PersistentMode,
// herk_graph, and batch_tiling settings determined by benchmarking all viable
// combinations on the current hardware.
//
// Two separate cache files per build configuration:
//
//   HERK: cutlass_herk_strategy_cache_{build_fingerprint}.txt
//   GEMM: cutlass_gemm_strategy_cache_{build_fingerprint}.txt
//
// HERK format (tab-separated text, one entry per line):
//   # N	K	batch	precision	mode	strategy	graph	persistent	herk_graph	tile	gemm_config	tile_name	time_ms	tflops	bandwidth_gbs	direct_config	direct_name
//   3328	128	32	0	Direct	Baseline	0	1	0	1	0	FP8_T128x64_C1x1_S3	1.234	123.4	456.7	0	K32_B3
//
// GEMM format (tab-separated text, one entry per line):
//   # M	N	K	batch	precision	gemm_config	tile_name	time_ms	tflops	bandwidth_gbs
//   128	4000	1664	128	0	0	FP8_T128x64_C1x1_S3	44.1	19.8	91.5
//
// The tile_name column is informational (derived from gemm_config via config_name()).
// On load, tile_name is read and discarded — the integer gemm_config is authoritative.

#pragma once

#include <algorithm>
#include <string>
#include <type_traits>
#include <vector>
#include <unordered_map>
#include <fstream>
#include <sstream>
#include <cstdio>
#include <cuda_runtime.h>
#include <cuda_fp8.h>

#include "tune_cache.hpp"  // for build_config_fingerprint(), build_config_header_string()

namespace strategy_cache {

/// Human-readable name for a GemmConfig ordinal. Mirrors config_name() in
/// shared/config_common.hpp but avoids the namespace dependency (GemmConfig
/// lives inside the arch-specific namespace, strategy_cache is independent).
inline const char* config_name_from_int(int gc) {
    switch (gc) {
    case  0: return "FP8_T128x64_C1x1_S3";
    case  1: return "FP8_T128x64_C1x1_SAuto";
    case  2: return "FP8_T128x128_C1x1_SAuto";
    case  3: return "FP8_T128x128_C1x1_S3";
    case  4: return "FP8_T128x256_C1x1_SAuto";
    case  5: return "FP8_T128x128_C1x2";
    case  6: return "FP8_T128x128_C2x2";
    case  7: return "FP8_T128x256_C1x2";
    case  8: return "FP8_T256x128_C2x1_2SM";
    case  9: return "FP8_T256x256_C2x2_2SM";
    case 10: return "FP8_T128x256_C1x1_Coop";
    case 11: return "FP8_T128x128_C1x1_Coop";
    case 12: return "FP8_T128x256_C1x1_PP";
    case 13: return "FP8_T128x128_C1x1_PP";
    case 14: return "FP8_T128x256_C1x2_Coop";
    case 15: return "FP8_T64x128_C1x1_Coop";
    case 16: return "SmallM_64x128";
    default: return "Unknown";
    }
}

/// Human-readable name for a DirectHerkConfig ordinal.
inline const char* direct_config_name_from_int(int dc) {
    switch (dc) {
    case 0: return "K32_B3";
    case 1: return "K64_B2";
    case 2: return "K64_B3";
    case 3: return "K128_B2";
    default: return "Unknown";
    }
}

/// Return the K_CHUNK value for a DirectHerkConfig ordinal.
/// Mirrors direct_herk_k_chunk() from config_common.hpp (avoids namespace dependency).
inline int direct_config_k_chunk(int dc) {
    switch (dc) {
    case 0: return 32;   // K32_B3
    case 1: return 64;   // K64_B2
    case 2: return 64;   // K64_B3
    case 3: return 128;  // K128_B2
    default: return 64;
    }
}

/// Number of DirectHerkConfig variants (for candidate generation).
static constexpr int NUM_DIRECT_HERK_CONFIGS = 4;

/// HERK tile size enum. Mirrors HerkTileSize from config_common.hpp (avoids namespace dependency).
enum class HerkTileSize { N32 = 0, N64 = 1, NUM_SIZES = 2 };

/// Human-readable name for a HerkTileSize.
inline const char* herk_tile_name(HerkTileSize t) {
    switch (t) {
    case HerkTileSize::N32: return "N32";
    case HerkTileSize::N64: return "N64";
    default: return "N32";
    }
}

/// Return all HerkTileSize values for candidate generation.
inline std::vector<HerkTileSize> all_herk_tile_sizes() {
    return { HerkTileSize::N32, HerkTileSize::N64 };
}

/// HERK pipeline mode enum. Mirrors HerkPipelineMode from config_common.hpp.
enum class HerkPipelineMode { Sync = 0, WarpSpecialized = 1, NUM_MODES = 2 };

/// Human-readable name for a HerkPipelineMode.
inline const char* herk_pipeline_name(HerkPipelineMode m) {
    switch (m) {
    case HerkPipelineMode::Sync:            return "Sync";
    case HerkPipelineMode::WarpSpecialized: return "WarpSpec";
    default: return "Sync";
    }
}

struct StrategyEntry {
    int herk_mode;        // 0 = ForceDirect, 1 = ForceBaseline
    int herk_strategy;    // 0 = Baseline, 1 = TriangleAware
    bool use_cuda_graph;  // CUDA graph for triangle slabs
    int persistent_mode;  // 0 = ForceOff, 1 = ForceOn
    bool herk_graph;      // baseline HERK graph capture
    bool batch_tiling;    // L2-aware batch tiling
    int gemm_config;      // GemmConfig ordinal (0 = Default)
    int direct_config;    // DirectHerkConfig ordinal (2 = K64_B3 default)
    int herk_tile;        // HerkTileSize ordinal (0 = N32, 1 = N64)
    int herk_pipeline;    // HerkPipelineMode ordinal (0 = Sync, 1 = WarpSpecialized)
    float time_ms;        // Best time in ms
    float tflops;         // Achieved TFLOPS
    float bandwidth_gbs;  // External I/O bandwidth in GB/s
};

class StrategyCache {
public:
    static StrategyCache& instance() {
        static StrategyCache inst;
        return inst;
    }

    ~StrategyCache() {
        if (dirty_) save();
    }

    void set_file_path(const std::string& path) {
        file_path_ = path;
        custom_path_ = true;
        loaded_ = false;
        dirty_ = false;
        cache_.clear();
    }

    const std::string& file_path() {
        ensure_file_path();
        return file_path_;
    }

    /// Set strategy cache verbosity level.
    /// 0=silent, 1=summary (cache hit/miss + winner), 2=per-candidate timing, 3=full diagnostics.
    void set_verbosity(int level) { verbosity_ = level; }
    int  verbosity() const { return verbosity_; }

    /// Look up optimal strategy. Returns true if found.
    bool lookup(int N, int K, int batch_count, int precision, StrategyEntry& out) {
        if (!loaded_) load();
        auto key = make_key(N, K, batch_count, precision);
        auto it = cache_.find(key);
        if (it != cache_.end()) {
            out = it->second;
            return true;
        }
        return false;
    }

    /// Store tuning result.
    void store(int N, int K, int batch_count, int precision, const StrategyEntry& entry) {
        if (!loaded_) load();
        auto key = make_key(N, K, batch_count, precision);
        cache_[key] = entry;
        dirty_ = true;
        save();
    }

    void load() {
        if (loaded_) return;
        loaded_ = true;
        ensure_gpu_name();
        ensure_file_path();

        std::ifstream f(file_path_);
        if (!f.is_open()) return;

        std::string line;
        bool gpu_ok = true;

        while (std::getline(f, line)) {
            if (line.empty()) continue;
            if (line[0] == '#') {
                // Check GPU header
                if (line.find("# GPU: ") == 0) {
                    std::string file_gpu = line.substr(7);
                    if (file_gpu != gpu_name_) {
                        gpu_ok = false;
                        break;
                    }
                }
                continue;
            }

            if (!gpu_ok) break;

            std::istringstream iss(line);

            // Format: N  K  batch  precision  mode  strategy  graph  persistent  herk_graph  tile  gemm_config  tile_name  time_ms  tflops  bandwidth_gbs  direct_config  direct_name
            std::string first_token;
            if (!(iss >> first_token)) continue;

            // Skip GEMM entries (from legacy combined files)
            if (first_token == "GEMM") continue;

            int n, k, batch, prec, graph, persistent, hg, tile, gc;
            std::string mode_str, strategy_str, tile_name;
            float time_ms, tflops;

            try { n = std::stoi(first_token); } catch (...) { continue; }

            if (!(iss >> k >> batch >> prec >> mode_str >> strategy_str
                      >> graph >> persistent >> hg >> tile >> gc >> tile_name
                      >> time_ms >> tflops)) {
                continue;
            }

            StrategyEntry entry;
            entry.herk_mode = (mode_str == "Direct") ? 0 : 1;
            entry.herk_strategy = (strategy_str == "TriangleAware") ? 1 : 0;
            entry.use_cuda_graph = (graph != 0);
            entry.persistent_mode = persistent;
            entry.herk_graph = (hg != 0);
            entry.batch_tiling = (tile != 0);
            entry.gemm_config = gc;
            entry.direct_config = 2;  // default: K64_B3
            entry.time_ms = time_ms;
            entry.tflops = tflops;

            // Try reading bandwidth_gbs (optional)
            float bw = 0.0f;
            if (iss >> bw) {
                entry.bandwidth_gbs = bw;
            }

            // Try reading direct_config + direct_name (optional)
            int dc = 2;  // default: K64_B3
            std::string dc_name;
            if (iss >> dc >> dc_name) {
                entry.direct_config = dc;
            }

            // Try reading herk_tile + herk_tile_name (optional, backward compat: default N32)
            int ht = 0;  // default: N32
            std::string ht_name;
            if (iss >> ht >> ht_name) {
                entry.herk_tile = ht;
            } else {
                entry.herk_tile = 0;
            }

            // Try reading herk_pipeline + pipeline_name (optional, backward compat: default Sync)
            int hp = 0;  // default: Sync
            std::string hp_name;
            if (iss >> hp >> hp_name) {
                entry.herk_pipeline = hp;
            } else {
                entry.herk_pipeline = 0;
            }

            cache_[make_key(n, k, batch, prec)] = entry;
        }
    }

    void save() {
        if (!dirty_) return;
        ensure_gpu_name();

        // Always write to the new HERK-specific filename, even if we loaded from legacy
        std::string save_path = file_path_;
        if (!custom_path_) {
            save_path = tune_cache::cache_dir_prefix() +
                        "cutlass_herk_strategy_cache_" +
                        tune_cache::build_config_fingerprint() + ".txt";
            file_path_ = save_path;  // update for future loads
        }

        std::string tmp_path = save_path + ".tmp";
        std::ofstream f(tmp_path);
        if (!f.is_open()) {
            if (verbosity_ >= 1)
                fprintf(stderr, "[StrategyCache] WARNING: cannot write %s\n", tmp_path.c_str());
            return;
        }

        std::string current_fp = tune_cache::build_config_fingerprint();

        f << "# CUTLASS HERK Strategy Tune Cache\n";
        f << "# GPU: " << gpu_name_ << "\n";
        f << "# Build: " << current_fp << "\n";
        f << tune_cache::build_config_header_string();
        f << "#\n";
        f << "# N\tK\tbatch\tprecision\tmode\tstrategy\tgraph\tpersistent\therk_graph\ttile\tgemm_config\ttile_name\ttime_ms\ttflops\tbandwidth_gbs\tdirect_config\tdirect_name\therk_tile\therk_tile_name\therk_pipeline\tpipeline_name\n";

        // Write all HERK entries
        for (const auto& kv : cache_) {
            const auto& e = kv.second;

            // Parse key back into components
            int n, k, batch, prec;
            if (sscanf(kv.first.c_str(), "%d:%d:%d:%d", &n, &k, &batch, &prec) != 4)
                continue;

            // Derive human-readable names from config ordinals
            const char* tile_name = config_name_from_int(e.gemm_config);
            const char* direct_name = direct_config_name_from_int(e.direct_config);
            const char* htile_name = herk_tile_name(static_cast<HerkTileSize>(e.herk_tile));
            const char* hpipe_name = herk_pipeline_name(static_cast<HerkPipelineMode>(e.herk_pipeline));

            char buf[512];
            snprintf(buf, sizeof(buf),
                "%d\t%d\t%d\t%d\t%s\t%s\t%d\t%d\t%d\t%d\t%d\t%s\t%.3f\t%.1f\t%.1f\t%d\t%s\t%d\t%s\t%d\t%s\n",
                n, k, batch, prec,
                e.herk_mode == 0 ? "Direct" : "Baseline",
                e.herk_strategy == 1 ? "TriangleAware" : "Baseline",
                e.use_cuda_graph ? 1 : 0,
                e.persistent_mode,
                e.herk_graph ? 1 : 0,
                e.batch_tiling ? 1 : 0,
                e.gemm_config,
                tile_name,
                e.time_ms,
                e.tflops,
                e.bandwidth_gbs,
                e.direct_config,
                direct_name,
                e.herk_tile,
                htile_name,
                e.herk_pipeline,
                hpipe_name);
            f << buf;
        }

        f.close();
        std::rename(tmp_path.c_str(), save_path.c_str());
        dirty_ = false;
    }

private:
    StrategyCache() = default;

    std::string make_key(int N, int K, int batch, int prec) {
        char buf[64];
        snprintf(buf, sizeof(buf), "%d:%d:%d:%d", N, K, batch, prec);
        return std::string(buf);
    }

    void ensure_gpu_name() {
        if (!gpu_name_.empty()) return;
        cudaDeviceProp prop;
        if (cudaGetDeviceProperties(&prop, 0) == cudaSuccess) {
            gpu_name_ = prop.name;
        } else {
            gpu_name_ = "unknown";
        }
    }

    void ensure_file_path() {
        if (!custom_path_ && file_path_.empty()) {
            file_path_ = tune_cache::cache_dir_prefix() +
                          "cutlass_herk_strategy_cache_" +
                          tune_cache::build_config_fingerprint() + ".txt";
        }
    }

    std::unordered_map<std::string, StrategyEntry> cache_;
    std::string file_path_;   // empty until first use (auto-generated) or set_file_path()
    std::string gpu_name_;
    int verbosity_ = 1;       // 0=silent, 1=summary, 2=sweep, 3=detail
    bool custom_path_ = false;
    bool loaded_ = false;
    bool dirty_ = false;
};

// ========================================================================================
// Templated autotune helpers — usable from both PIMPL API and example binaries
// ========================================================================================

/// @brief Apply all cached settings from a StrategyEntry to a GEMM engine instance.
///
/// Sets herk_mode, persistent_mode, herk_graph, batch_tiling, herk_tile_size, and
/// herk_pipeline_mode on the engine.
/// HerkStrategy and triangle_config.use_cuda_graph are set in CutlassParams, not here.
///
/// @tparam GemmImpl             The GEMM engine class (GemmComplexFP8 or GemmComplexSm100).
/// @tparam InternalHerkMode     The internal HerkMode enum from the engine's namespace.
/// @tparam InternalPersistentMode The internal PersistentMode enum from the engine's namespace.
/// @tparam InternalGemmConfig   The internal GemmConfig enum from the engine's namespace.
/// @tparam InternalDirectHerkConfig The internal DirectHerkConfig enum from the engine's namespace.
/// @tparam InternalHerkTileSize The internal HerkTileSize enum from the engine's namespace.
/// @tparam InternalHerkPipelineMode The internal HerkPipelineMode enum from the engine's namespace.
template <typename GemmImpl, typename InternalHerkMode, typename InternalPersistentMode,
          typename InternalGemmConfig, typename InternalDirectHerkConfig,
          typename InternalHerkTileSize, typename InternalHerkPipelineMode>
void apply_cached_settings(GemmImpl& gemm, const StrategyEntry& entry) {
    gemm.set_herk_mode(entry.herk_mode == 0
        ? InternalHerkMode::ForceDirect
        : InternalHerkMode::ForceBaseline);
    gemm.set_persistent_mode(entry.persistent_mode == 1
        ? InternalPersistentMode::ForceOn
        : InternalPersistentMode::ForceOff);
    gemm.set_herk_graph(entry.herk_graph);
    gemm.set_batch_tiling(entry.batch_tiling);

    // Validate gemm_config is valid for current arch before applying.
    // Stale cache entries from a different arch would crash dispatch.
    auto cfg = static_cast<InternalGemmConfig>(entry.gemm_config);
    if (!is_config_valid_for_arch(cfg))
        cfg = InternalGemmConfig::Default;
    gemm.set_gemm_config(cfg);

    // Apply direct HERK config (K_CHUNK/NR_BUFS template selection).
    auto dc = static_cast<InternalDirectHerkConfig>(entry.direct_config);
    if (static_cast<int>(dc) < 0 || static_cast<int>(dc) >= static_cast<int>(InternalDirectHerkConfig::NUM_CONFIGS))
        dc = InternalDirectHerkConfig::Default;
    gemm.set_direct_herk_config(dc);

    // Apply HERK tile size (N32/N64).
    auto ht = static_cast<InternalHerkTileSize>(entry.herk_tile);
    if (static_cast<int>(ht) < 0 || static_cast<int>(ht) >= static_cast<int>(InternalHerkTileSize::NUM_SIZES))
        ht = InternalHerkTileSize::N32;
    gemm.set_herk_tile_size(ht);

    // Apply HERK pipeline mode (Sync/WarpSpecialized).
    auto hp = static_cast<InternalHerkPipelineMode>(entry.herk_pipeline);
    if (static_cast<int>(hp) < 0 || static_cast<int>(hp) >= static_cast<int>(InternalHerkPipelineMode::NUM_MODES))
        hp = InternalHerkPipelineMode::Sync;
    gemm.set_herk_pipeline_mode(hp);
}

/// @brief Strategy sweep candidate for HERK autotuning.
struct TuneCandidate {
    int herk_mode;       // 0 = ForceDirect, 1 = ForceBaseline
    int herk_strategy;   // 0 = Baseline, 1 = TriangleAware
    bool use_graph;      // triangle slab CUDA graph
    int persistent;      // 0 = ForceOff, 1 = ForceOn
    bool herk_graph;     // baseline HERK graph capture
    int gemm_config;     // GemmConfig ordinal (0 = Default)
    int direct_config;   // DirectHerkConfig ordinal (2 = K64_B3 default)
    int herk_tile;       // HerkTileSize ordinal (0 = N32, 1 = N64)
    int herk_pipeline;   // HerkPipelineMode ordinal (0 = Sync, 1 = WarpSpecialized)
};

/// @brief Run strategy autotuning sweep for a given HERK problem size.
///
/// Benchmarks strategy combinations (Direct/Baseline x Baseline/TriangleAware x
/// graph x persistent x herk_graph x GemmConfig) and caches the fastest. Subsequent
/// calls with the same (N, K, batch_count, precision) hit the cache and skip the sweep.
///
/// Direct HERK uses hand-written PTX (not CUTLASS GEMMs), so GemmConfig only affects
/// Baseline candidates. DirectHerkConfig (K_CHUNK/NR_BUFS) only affects Direct candidates.
/// The viable_configs parameter lists FP8 GemmConfig ordinals to sweep for baseline;
/// if empty, only Default (0) is tested. All DirectHerkConfig variants are always swept.
///
/// Batch tiling is always ON (not a sweep axis — disabling it wastes L2).
///
/// @tparam GemmImpl             GEMM engine class.
/// @tparam InternalHerkMode     Internal HerkMode enum.
/// @tparam InternalPersistentMode Internal PersistentMode enum.
/// @tparam InternalDirectHerkConfig Internal DirectHerkConfig enum.
/// @tparam InternalHerkTileSize Internal HerkTileSize enum.
/// @tparam HerkOp               HerkOp enum (NoTrans/ConjTrans).
/// @tparam FillMode             FillMode enum (Lower/Upper).
/// @tparam HerkStrategy         HerkStrategy enum (Baseline/TriangleAware).
/// @tparam CutlassParams        CutlassParams struct from the engine's namespace.
/// @tparam TriangleConfig       TriangleConfig struct from the engine's namespace.
/// @param gemm            The underlying GEMM engine instance.
/// @param N               Matrix dimension.
/// @param K               Inner dimension.
/// @param batch_count     Batch size.
/// @param precision       Integer cache key (0=FP8, 1=FP6, 2=FP4).
/// @param stream          CUDA stream.
/// @param viable_configs  GemmConfig ordinals to sweep for baseline (default: {0}).
/// @return The optimal StrategyEntry (from cache or fresh sweep).
template <typename GemmImpl, typename InternalHerkMode, typename InternalPersistentMode,
          typename InternalDirectHerkConfig, typename InternalHerkTileSize,
          typename InternalHerkPipelineMode,
          typename HerkOp, typename FillMode, typename HerkStrategy, typename CutlassParams,
          typename TriangleConfig>
StrategyEntry run_autotune(GemmImpl& gemm, int N, int K, int batch_count,
                           int precision, cudaStream_t stream,
                           const std::vector<int>& viable_configs = {0}) {
    // 1. Check cache — return immediately on hit
    int strat_verb = StrategyCache::instance().verbosity();

    StrategyEntry cached;
    if (StrategyCache::instance().lookup(N, K, batch_count, precision, cached)) {
        if (strat_verb >= 1)
            fprintf(stderr, "[StrategyTune] Cache hit for N=%d K=%d batch=%d prec=%d: "
                    "%s+%s+graph=%d+persistent=%d+herk_graph=%d+tile=%d+cfg=%d(%s)+dcfg=%d(%s)+htile=%s+pipe=%s (%.1f TFLOPS, %.1f GB/s)\n",
                    N, K, batch_count, precision,
                    cached.herk_mode == 0 ? "Direct" : "Baseline",
                    cached.herk_strategy == 1 ? "TriangleAware" : "Baseline",
                    cached.use_cuda_graph, cached.persistent_mode,
                    cached.herk_graph, cached.batch_tiling, cached.gemm_config,
                    config_name_from_int(cached.gemm_config),
                    cached.direct_config,
                    direct_config_name_from_int(cached.direct_config),
                    herk_tile_name(static_cast<HerkTileSize>(cached.herk_tile)),
                    herk_pipeline_name(static_cast<HerkPipelineMode>(cached.herk_pipeline)),
                    cached.tflops, cached.bandwidth_gbs);
        return cached;
    }

    // 2. Ensure kernel cache is loaded before any launches (avoid lazy load during timing)
    int saved_verbosity = tune_cache::TuneCache::instance().verbosity();
    tune_cache::TuneCache::instance().set_verbosity(std::max(saved_verbosity, 2));
    tune_cache::TuneCache::instance().load();

    if (strat_verb >= 1)
        fprintf(stderr, "[StrategyTune] Sweeping strategies for N=%d K=%d batch=%d precision=%d...\n",
                N, K, batch_count, precision);

    // 3. Allocate test data
    int64_t total_complex = static_cast<int64_t>(N) * K * batch_count;
    int64_t packed_elems = static_cast<int64_t>(N) * (N + 1) / 2;
    int64_t total_halfs_out = packed_elems * 2 * batch_count;

    // Check available memory before allocating
    size_t free_mem = 0, total_mem = 0;
    cudaMemGetInfo(&free_mem, &total_mem);
    int64_t fp8_elems = total_complex * 2;  // 2 FP8 bytes per complex element
    size_t needed = total_complex * 2 * sizeof(__half)       // d_A_test
                  + total_halfs_out * sizeof(__half)          // d_C_test
                  + fp8_elems * sizeof(__nv_fp8_e4m3)        // d_A_fp8_test
                  + total_complex * 2;                       // internal FP8 buffers
    if (needed > free_mem * 0.8) {
        if (strat_verb >= 1)
            fprintf(stderr, "[StrategyTune] Skipping autotune: need %.1f GB, only %.1f GB free\n",
                    needed / 1e9, free_mem / 1e9);
        StrategyEntry entry{};
        return entry;
    }

    __half* d_A_test = nullptr;
    __half* d_C_test = nullptr;
    auto alloc_ok = cudaMalloc(&d_A_test, total_complex * 2 * sizeof(__half));
    if (alloc_ok == cudaSuccess) alloc_ok = cudaMalloc(&d_C_test, total_halfs_out * sizeof(__half));

    __nv_fp8_e4m3* d_A_fp8_test = nullptr;
    if (alloc_ok == cudaSuccess) alloc_ok = cudaMalloc(&d_A_fp8_test, fp8_elems * sizeof(__nv_fp8_e4m3));

    if (alloc_ok != cudaSuccess) {
        if (strat_verb >= 1)
            fprintf(stderr, "[StrategyTune] Skipping autotune: cudaMalloc failed (%s)\n",
                    cudaGetErrorString(alloc_ok));
        cudaFree(d_A_test); cudaFree(d_C_test); cudaFree(d_A_fp8_test);
        cudaGetLastError();  // clear sticky error
        StrategyEntry entry{};
        return entry;
    }

    // Zero-fill — timing matters, not values
    cudaMemset(d_A_test, 0, total_complex * 2 * sizeof(__half));

    // Pre-cast FP8 buffer for direct HERK candidates.
    // The direct kernel expects interleaved FP8 [re,im] pairs — providing this
    // pre-cast buffer avoids the FP16→FP8 cast overhead in the timed path,
    // matching the production pipeline where INT4→FP8 is done before HERK.
    cudaMemset(d_A_fp8_test, 0, fp8_elems * sizeof(__nv_fp8_e4m3));

    // 4. Generate candidates
    std::vector<TuneCandidate> candidates;

    // Direct mode candidates: sweep all DirectHerkConfig × HerkTileSize variants.
    // GemmConfig uses first viable as safe default (only matters when cached entry
    // is later applied to baseline GEMM calls).
    int direct_gemm_cfg = viable_configs.empty() ? 0 : viable_configs[0];

    auto all_tiles = all_herk_tile_sizes();

    for (int dci = 0; dci < NUM_DIRECT_HERK_CONFIGS; dci++) {
        // Skip configs where K_CHUNK greatly exceeds K — wasteful (most loads are zero-padded).
        // The kernel handles partial loads correctly (per-sub-op bounds check), but configs
        // where > 75% of each tile is zero-fill are unlikely to win.  Threshold: k_chunk > 4*K.
        int k_chunk = direct_config_k_chunk(dci);
        if (k_chunk > 4 * K) continue;

        for (auto htile : all_tiles) {
            int hti = static_cast<int>(htile);
            // N64 tile requires N >= 64 to be useful
            if (htile == HerkTileSize::N64 && N < 64) continue;

            // Sweep pipeline modes: Sync (0) and WarpSpecialized (1)
            for (int hpi = 0; hpi < static_cast<int>(HerkPipelineMode::NUM_MODES); hpi++) {
                candidates.push_back({0, 0, false, 0, false, direct_gemm_cfg, dci, hti, hpi}); // Direct+Base+pers=off
                candidates.push_back({0, 0, false, 1, false, direct_gemm_cfg, dci, hti, hpi}); // Direct+Base+pers=on
                candidates.push_back({0, 1, false, 0, false, direct_gemm_cfg, dci, hti, hpi}); // Direct+Tri+pers=off
                candidates.push_back({0, 1, false, 1, false, direct_gemm_cfg, dci, hti, hpi}); // Direct+Tri+pers=on
                candidates.push_back({0, 1, true,  0, false, direct_gemm_cfg, dci, hti, hpi}); // Direct+Tri+graph+pers=off
                candidates.push_back({0, 1, true,  1, false, direct_gemm_cfg, dci, hti, hpi}); // Direct+Tri+graph+pers=on
            }
        }
    }

    // Baseline mode candidates: (strategy x graph x herk_graph) × viable GemmConfigs
    // Only if FP8 baseline kernel fits in device SMEM.
    // Baseline does not use direct HERK kernel, so direct_config=Default (2), herk_tile=0 (N32).
    if (GemmImpl::fp8_baseline_available()) {
        for (int cfg : viable_configs) {
            candidates.push_back({1, 0, false, 0, false, cfg, 2, 0, 0}); // Baseline+Base
            candidates.push_back({1, 1, false, 0, false, cfg, 2, 0, 0}); // Baseline+Tri
            candidates.push_back({1, 1, true,  0, false, cfg, 2, 0, 0}); // Baseline+Tri+graph
            candidates.push_back({1, 0, false, 0, true,  cfg, 2, 0, 0}); // Baseline+Base+hgraph
            candidates.push_back({1, 1, false, 0, true,  cfg, 2, 0, 0}); // Baseline+Tri+hgraph
            candidates.push_back({1, 1, true,  0, true,  cfg, 2, 0, 0}); // Baseline+Tri+graph+hgraph
        }
    }

    int num_candidates = static_cast<int>(candidates.size());

    // 5. Benchmark each candidate
    cudaEvent_t ev_start, ev_stop;
    cudaEventCreate(&ev_start);
    cudaEventCreate(&ev_stop);

    float best_time = 1e9f;
    int best_idx = 0;

    for (int ci = 0; ci < num_candidates; ci++) {
        auto& c = candidates[ci];

        // Configure the engine
        gemm.set_herk_mode(c.herk_mode == 0
            ? InternalHerkMode::ForceDirect
            : InternalHerkMode::ForceBaseline);
        gemm.set_persistent_mode(c.persistent == 1
            ? InternalPersistentMode::ForceOn
            : InternalPersistentMode::ForceOff);
        gemm.set_herk_graph(c.herk_graph);
        gemm.set_batch_tiling(true);  // always ON
        gemm.set_direct_herk_config(static_cast<InternalDirectHerkConfig>(c.direct_config));
        gemm.set_herk_tile_size(static_cast<InternalHerkTileSize>(c.herk_tile));
        gemm.set_herk_pipeline_mode(static_cast<InternalHerkPipelineMode>(c.herk_pipeline));

        CutlassParams p;
        p.stream = stream;
        p.herk_strategy = (c.herk_strategy == 1)
            ? HerkStrategy::TriangleAware
            : HerkStrategy::Baseline;
        p.triangle_config.use_cuda_graph = c.use_graph;
        p.config = static_cast<decltype(p.config)>(c.gemm_config);

        const char* mode_name = c.herk_mode == 0 ? "Direct" : "Baseline";
        const char* strat_name = c.herk_strategy == 1 ? "Tri" : "Base";

        // 2 warmup runs
        const char* htile_str = herk_tile_name(static_cast<HerkTileSize>(c.herk_tile));
        const char* hpipe_str = herk_pipeline_name(static_cast<HerkPipelineMode>(c.herk_pipeline));

        bool failed = false;
        for (int w = 0; w < 2; w++) {
            cudaMemset(d_C_test, 0, total_halfs_out * sizeof(__half));
            try {
                auto ws = gemm.HERK_batched(
                    d_A_test, d_C_test, N, K, batch_count,
                    1.0f, 0.0f,
                    HerkOp::NoTrans, FillMode::Lower, p,
                    false, d_A_fp8_test);
                if (static_cast<int>(ws) != 0) { failed = true; break; }
            } catch (...) {
                failed = true;
                break;
            }
        }
        if (failed) {
            if (strat_verb >= 2)
                fprintf(stderr, "  [%d] %s+%s+graph=%d+pers=%d+hgraph=%d+cfg=%d+dcfg=%d+htile=%s+pipe=%s: SKIPPED (warmup failed)\n",
                        ci, mode_name, strat_name, c.use_graph, c.persistent, c.herk_graph, c.gemm_config, c.direct_config, htile_str, hpipe_str);
            cudaDeviceSynchronize();
            cudaGetLastError();  // clear sticky error from kernel crash
            continue;
        }
        cudaDeviceSynchronize();

        // 5 timed runs
        const int timed_runs = 5;
        float total_ms = 0;

        for (int t = 0; t < timed_runs; t++) {
            cudaMemset(d_C_test, 0, total_halfs_out * sizeof(__half));
            cudaEventRecord(ev_start, stream);
            try {
                auto ws = gemm.HERK_batched(
                    d_A_test, d_C_test, N, K, batch_count,
                    1.0f, 0.0f,
                    HerkOp::NoTrans, FillMode::Lower, p,
                    false, d_A_fp8_test);
                if (static_cast<int>(ws) != 0) { failed = true; break; }
            } catch (...) {
                failed = true;
                break;
            }
            cudaEventRecord(ev_stop, stream);
            cudaEventSynchronize(ev_stop);
            float ms;
            cudaEventElapsedTime(&ms, ev_start, ev_stop);
            total_ms += ms;
        }

        if (failed) {
            if (strat_verb >= 2)
                fprintf(stderr, "  [%d] %s+%s+graph=%d+pers=%d+hgraph=%d+cfg=%d+dcfg=%d+htile=%s+pipe=%s: SKIPPED (timed run failed)\n",
                        ci, mode_name, strat_name, c.use_graph, c.persistent, c.herk_graph, c.gemm_config, c.direct_config, htile_str, hpipe_str);
            cudaDeviceSynchronize();
            cudaGetLastError();  // clear sticky error from kernel crash
            continue;
        }

        float avg_ms = total_ms / timed_runs;
        double flops = 6.0 * N * N * K * batch_count;  // HERK = 3 sub-GEMMs × 2NNK
        double tflops = flops / (avg_ms * 1e9);

        if (strat_verb >= 2)
            fprintf(stderr, "  [%d] %s+%s+graph=%d+pers=%d+hgraph=%d+cfg=%d(%s)+dcfg=%d(%s)+htile=%s+pipe=%s: %.3f ms (%.1f TFLOPS)%s\n",
                    ci, mode_name, strat_name, c.use_graph, c.persistent, c.herk_graph,
                    c.gemm_config, config_name_from_int(c.gemm_config),
                    c.direct_config, direct_config_name_from_int(c.direct_config),
                    htile_str, hpipe_str,
                    avg_ms, tflops,
                    avg_ms < best_time ? " <-- best" : "");

        if (avg_ms < best_time) {
            best_time = avg_ms;
            best_idx = ci;
        }
    }

    cudaEventDestroy(ev_start);
    cudaEventDestroy(ev_stop);
    cudaFree(d_A_test);
    cudaFree(d_A_fp8_test);
    cudaFree(d_C_test);

    // 6. Store result
    auto& winner = candidates[best_idx];
    double flops = 6.0 * N * N * K * batch_count;  // HERK = 3 sub-GEMMs × 2NNK
    double tflops_val = flops / (best_time * 1e9);
    double ext_bytes = batch_count * (4.0 * N * K + 2.0 * N * (N + 1));
    double bw_gbs = ext_bytes / (best_time * 1e-3) / 1e9;

    StrategyEntry entry;
    entry.herk_mode = winner.herk_mode;
    entry.herk_strategy = winner.herk_strategy;
    entry.use_cuda_graph = winner.use_graph;
    entry.persistent_mode = winner.persistent;
    entry.herk_graph = winner.herk_graph;
    entry.batch_tiling = true;  // always ON
    entry.gemm_config = winner.gemm_config;
    entry.direct_config = winner.direct_config;
    entry.herk_tile = winner.herk_tile;
    entry.herk_pipeline = winner.herk_pipeline;
    entry.time_ms = best_time;
    entry.tflops = static_cast<float>(tflops_val);
    entry.bandwidth_gbs = static_cast<float>(bw_gbs);

    StrategyCache::instance().store(N, K, batch_count, precision, entry);

    // 7. Save kernel-level tune cache and restore verbosity
    tune_cache::TuneCache::instance().save();
    tune_cache::TuneCache::instance().set_verbosity(saved_verbosity);

    if (strat_verb >= 1)
        fprintf(stderr, "[StrategyTune] Winner: %s+%s+graph=%d+persistent=%d+herk_graph=%d+tile=1+cfg=%d(%s)+dcfg=%d(%s)+htile=%s+pipe=%s "
                "(%.3f ms, %.1f TFLOPS, %.1f GB/s)\n",
                winner.herk_mode == 0 ? "Direct" : "Baseline",
                winner.herk_strategy == 1 ? "TriangleAware" : "Baseline",
                winner.use_graph, winner.persistent, winner.herk_graph,
                winner.gemm_config, config_name_from_int(winner.gemm_config),
                winner.direct_config, direct_config_name_from_int(winner.direct_config),
                herk_tile_name(static_cast<HerkTileSize>(winner.herk_tile)),
                herk_pipeline_name(static_cast<HerkPipelineMode>(winner.herk_pipeline)),
                best_time, tflops_val, bw_gbs);

    return entry;
}

// ========================================================================================
// GEMM Strategy Autotuner
// ========================================================================================
//
// Simpler than the HERK autotuner: GEMM only varies GemmConfig (tile/cluster/schedule).
// No herk_mode, herk_strategy, persistent_mode, direct_config, herk_graph, or triangle
// config -- those are HERK-specific.
//
// GEMM also sweeps stacked-K (Strategy 5D) and L2 persistence (Strategy 4B), which are
// auto-selected internally based on problem dimensions. The autotuner tests with and
// without these optimizations by calling run_planar_batched_fp32out() which includes
// them automatically.
//
// Cache entries use a "GEMM:" prefix in the key to distinguish from HERK entries.
// Same cache file, same StrategyCache singleton.
//
// FLOPS formula: complex GEMM = 4 real sub-GEMMs, each 2*M*N*K = 8*M*N*K total.

/// @brief GEMM strategy cache entry.
struct GemmStrategyEntry {
    int gemm_config;       // GemmConfig ordinal (4M path)
    float time_ms;         // Best time in ms
    float tflops;          // Achieved TFLOPS
    float bandwidth_gbs;   // External I/O bandwidth in GB/s
    bool use_direct;       // Winner uses direct PTX kernel?
    int direct_tile;       // DirectGemmTileConfig ordinal (when use_direct=true)
    bool use_persistent;   // Persistent variant? (K-gated)
};

/// @brief GEMM strategy sweep candidate.
struct GemmTuneCandidate {
    int gemm_config;       // GemmConfig ordinal (for 4M path)
    bool use_direct;       // true = direct PTX kernel
    int direct_tile;       // DirectGemmTileConfig ordinal (for direct path)
    bool use_persistent;   // true = persistent variant (K-gated)
};

/// @brief Apply a GEMM strategy entry to a GEMM engine instance.
///
/// Sets the GemmConfig (for 4M path) or direct kernel settings on the engine.
/// Validates against the current architecture.
///
/// @tparam GemmImpl           The GEMM engine class.
/// @tparam InternalGemmConfig The internal GemmConfig enum.
template <typename GemmImpl, typename InternalGemmConfig>
void apply_gemm_cached_settings(GemmImpl& gemm, const GemmStrategyEntry& entry) {
    if (!entry.use_direct) {
        // 4M sub-GEMM path: apply GemmConfig
        auto cfg = static_cast<InternalGemmConfig>(entry.gemm_config);
        if (!is_config_valid_for_arch(cfg))
            cfg = InternalGemmConfig::Default;
        gemm.set_gemm_config(cfg);
    }
    // Direct/persistent settings are applied at the PIMPL level via GemmMode,
    // not on the internal engine (which doesn't know about direct PTX kernels).
}

/// @brief Singleton cache for GEMM strategy entries.
///
/// Uses the same file as StrategyCache but with "GEMM:" prefix in keys to avoid
/// collision with HERK entries. Shares the same GPU name and build fingerprint
/// validation.
class GemmStrategyCache {
public:
    static GemmStrategyCache& instance() {
        static GemmStrategyCache inst;
        return inst;
    }

    ~GemmStrategyCache() {
        if (dirty_) save();
    }

    void set_file_path(const std::string& path) {
        file_path_ = path;
        custom_path_ = true;
        loaded_ = false;
        dirty_ = false;
        cache_.clear();
    }

    const std::string& file_path() {
        ensure_file_path();
        return file_path_;
    }

    void set_verbosity(int level) { verbosity_ = level; }
    int  verbosity() const { return verbosity_; }

    /// Look up optimal GEMM strategy. Returns true if found.
    bool lookup(int M, int N, int K, int batch_count, int precision, GemmStrategyEntry& out) {
        if (!loaded_) load();
        auto key = make_key(M, N, K, batch_count, precision);
        auto it = cache_.find(key);
        if (it != cache_.end()) {
            out = it->second;
            return true;
        }
        return false;
    }

    /// Store GEMM tuning result.
    void store(int M, int N, int K, int batch_count, int precision, const GemmStrategyEntry& entry) {
        if (!loaded_) load();
        auto key = make_key(M, N, K, batch_count, precision);
        cache_[key] = entry;
        dirty_ = true;
        save();
    }

    void load() {
        if (loaded_) return;
        loaded_ = true;
        ensure_gpu_name();
        ensure_file_path();

        std::ifstream f(file_path_);
        if (!f.is_open()) return;

        std::string line;
        bool gpu_ok = true;

        while (std::getline(f, line)) {
            if (line.empty()) continue;
            if (line[0] == '#') {
                if (line.find("# GPU: ") == 0) {
                    std::string file_gpu = line.substr(7);
                    if (file_gpu != gpu_name_) {
                        gpu_ok = false;
                        break;
                    }
                }
                continue;
            }
            if (!gpu_ok) break;

            // Format: M  N  K  batch  precision  gemm_config  tile_name  time_ms  tflops  bandwidth_gbs  [direct  tile  persistent]
            std::istringstream iss(line);

            int m, n, k, batch, prec, gc;
            std::string tile_name;
            float time_ms, tflops, bw;

            if (!(iss >> m >> n >> k >> batch >> prec >> gc >> tile_name
                      >> time_ms >> tflops >> bw)) {
                continue;
            }

            GemmStrategyEntry entry;
            entry.gemm_config = gc;
            entry.time_ms = time_ms;
            entry.tflops = tflops;
            entry.bandwidth_gbs = bw;
            entry.use_direct = false;
            entry.direct_tile = 0;
            entry.use_persistent = false;

            // Try to read optional direct/tile/persistent columns (backward compat)
            int direct_flag = 0, dtile = 0, persistent_flag = 0;
            if (iss >> direct_flag >> dtile >> persistent_flag) {
                entry.use_direct = (direct_flag != 0);
                entry.direct_tile = dtile;
                entry.use_persistent = (persistent_flag != 0);
            }

            cache_[make_key(m, n, k, batch, prec)] = entry;
        }
    }

    void save() {
        if (!dirty_) return;
        ensure_gpu_name();

        // Always write to the new GEMM-specific filename
        std::string save_path = file_path_;
        if (!custom_path_) {
            save_path = tune_cache::cache_dir_prefix() +
                        "cutlass_gemm_strategy_cache_" +
                        tune_cache::build_config_fingerprint() + ".txt";
            file_path_ = save_path;  // update for future loads
        }

        std::string tmp_path = save_path + ".tmp";
        std::ofstream f(tmp_path);
        if (!f.is_open()) {
            if (verbosity_ >= 1)
                fprintf(stderr, "[GemmStrategyCache] WARNING: cannot write %s\n", tmp_path.c_str());
            return;
        }

        std::string current_fp = tune_cache::build_config_fingerprint();
        f << "# CUTLASS GEMM Strategy Tune Cache\n";
        f << "# GPU: " << gpu_name_ << "\n";
        f << "# Build: " << current_fp << "\n";
        f << tune_cache::build_config_header_string();
        f << "#\n";
        f << "# M\tN\tK\tbatch\tprecision\tgemm_config\ttile_name\ttime_ms\ttflops\tbandwidth_gbs\tdirect\ttile\tpersistent\n";

        // Write GEMM entries
        for (const auto& kv : cache_) {
            const auto& e = kv.second;
            int m, n, k, batch, prec;
            if (sscanf(kv.first.c_str(), "GEMM:%d:%d:%d:%d:%d", &m, &n, &k, &batch, &prec) != 5)
                continue;

            const char* tile_name = e.use_direct ? "Direct" : config_name_from_int(e.gemm_config);

            char buf[320];
            snprintf(buf, sizeof(buf),
                "%d\t%d\t%d\t%d\t%d\t%d\t%s\t%.3f\t%.1f\t%.1f\t%d\t%d\t%d\n",
                m, n, k, batch, prec,
                e.gemm_config, tile_name,
                e.time_ms, e.tflops, e.bandwidth_gbs,
                e.use_direct ? 1 : 0, e.direct_tile, e.use_persistent ? 1 : 0);
            f << buf;
        }

        f.close();
        std::rename(tmp_path.c_str(), save_path.c_str());
        dirty_ = false;
    }

private:
    GemmStrategyCache() = default;

    std::string make_key(int M, int N, int K, int batch, int prec) {
        char buf[96];
        snprintf(buf, sizeof(buf), "GEMM:%d:%d:%d:%d:%d", M, N, K, batch, prec);
        return std::string(buf);
    }

    void ensure_gpu_name() {
        if (!gpu_name_.empty()) return;
        cudaDeviceProp prop;
        if (cudaGetDeviceProperties(&prop, 0) == cudaSuccess) {
            gpu_name_ = prop.name;
        } else {
            gpu_name_ = "unknown";
        }
    }

    void ensure_file_path() {
        if (!custom_path_ && file_path_.empty()) {
            file_path_ = tune_cache::cache_dir_prefix() +
                          "cutlass_gemm_strategy_cache_" +
                          tune_cache::build_config_fingerprint() + ".txt";
        }
    }

    std::unordered_map<std::string, GemmStrategyEntry> cache_;
    std::string file_path_;
    std::string gpu_name_;
    int verbosity_ = 1;
    bool custom_path_ = false;
    bool loaded_ = false;
    bool dirty_ = false;
};

/// @brief Run GEMM strategy autotuning sweep for a given problem size.
///
/// Benchmarks all valid GemmConfig values for the current architecture
/// and caches the fastest. Subsequent calls with the same (M, N, K,
/// batch_count, precision) hit the cache and skip the sweep.
///
/// Complex GEMM FLOPS = 8 * M * N * K * batch (4 real sub-GEMMs, each 2MNK).
///
/// @tparam GemmImpl                  GEMM engine class (GemmComplexFP8 or GemmComplexSm100).
/// @tparam InternalGemmConfig        Internal GemmConfig enum.
/// @tparam ComplexModeT              ComplexMode enum from the engine's namespace.
/// @tparam InternalComputePrecision  ComputePrecision enum from the engine's namespace.
///                                   Pass `void` for SM90 (FP8-only, no precision param).
/// @param gemm            The underlying GEMM engine instance.
/// @param M               Number of rows of A and C.
/// @param N               Number of columns of C.
/// @param K               Inner dimension.
/// @param batch_count     Batch size.
/// @param precision       Integer cache key (0=FP8, 1=FP6, 2=FP4).
/// @param stream          CUDA stream.
/// @param viable_configs  GemmConfig ordinals to sweep (from all_baseline_configs()).
///                        Must not be empty; callers populate this from the arch-specific
///                        namespace before calling.
/// @return The optimal GemmStrategyEntry (from cache or fresh sweep).
template <typename GemmImpl, typename InternalGemmConfig, typename ComplexModeT,
          typename InternalComputePrecision = void>
GemmStrategyEntry run_gemm_autotune(GemmImpl& gemm, int M, int N, int K,
                                    int batch_count, int precision,
                                    cudaStream_t stream,
                                    const std::vector<int>& viable_configs) {
    int strat_verb = GemmStrategyCache::instance().verbosity();

    // 1. Check cache
    GemmStrategyEntry cached;
    if (GemmStrategyCache::instance().lookup(M, N, K, batch_count, precision, cached)) {
        if (strat_verb >= 1)
            fprintf(stderr, "[GemmTune] Cache hit for M=%d N=%d K=%d batch=%d prec=%d: "
                    "cfg=%d(%s) (%.1f TFLOPS, %.1f GB/s)\n",
                    M, N, K, batch_count, precision,
                    cached.gemm_config, config_name_from_int(cached.gemm_config),
                    cached.tflops, cached.bandwidth_gbs);
        return cached;
    }

    // 2. Ensure kernel cache is loaded
    int saved_verbosity = tune_cache::TuneCache::instance().verbosity();
    tune_cache::TuneCache::instance().set_verbosity(std::max(saved_verbosity, 2));
    tune_cache::TuneCache::instance().load();

    if (strat_verb >= 1)
        fprintf(stderr, "[GemmTune] Sweeping GemmConfig for M=%d N=%d K=%d batch=%d precision=%d...\n",
                M, N, K, batch_count, precision);

    // 3. Build candidates list from caller-provided config ordinals + direct kernel.
    std::vector<GemmTuneCandidate> candidates;
    // 3a. 4M sub-GEMM candidates (from all_baseline_configs())
    for (int cfg : viable_configs) {
        GemmTuneCandidate tc;
        tc.gemm_config = cfg;
        tc.use_direct = false;
        tc.direct_tile = 0;
        tc.use_persistent = false;
        candidates.push_back(tc);
    }
    // 3b. Direct PTX kernel candidates (auto-tile selection only, not per-tile)
    //     The direct kernel auto-selects tile based on M/N, so we sweep:
    //     - non-persistent direct (works for all K)
    //     - persistent direct (K-gated: optimal when K <= 64)
    {
        GemmTuneCandidate tc_direct;
        tc_direct.gemm_config = 0;
        tc_direct.use_direct = true;
        tc_direct.direct_tile = 0;  // Auto tile selection
        tc_direct.use_persistent = false;
        candidates.push_back(tc_direct);

        // Only add persistent candidate when it has a chance to be faster (K <= 64)
        if (K <= 64) {
            GemmTuneCandidate tc_persistent;
            tc_persistent.gemm_config = 0;
            tc_persistent.use_direct = true;
            tc_persistent.direct_tile = 0;
            tc_persistent.use_persistent = true;
            candidates.push_back(tc_persistent);
        }
    }

    if (candidates.empty()) {
        if (strat_verb >= 1)
            fprintf(stderr, "[GemmTune] No viable configs -- returning Default\n");
        GemmStrategyEntry entry;
        entry.gemm_config = 0;
        entry.time_ms = 0;
        entry.tflops = 0;
        entry.bandwidth_gbs = 0;
        entry.use_direct = false;
        entry.direct_tile = 0;
        entry.use_persistent = false;
        return entry;
    }

    // 4. Allocate test data (planar complex FP16)
    int64_t sA = static_cast<int64_t>(M) * K * batch_count;
    int64_t sB = static_cast<int64_t>(N) * K * batch_count;
    int64_t sC = static_cast<int64_t>(M) * N * batch_count;

    // Check available memory before allocating (need ~3x for test data + internal FP8 + workspace)
    size_t free_mem = 0, total_mem = 0;
    cudaMemGetInfo(&free_mem, &total_mem);
    size_t needed = (sA * 2 + sB * 2) * sizeof(__half) + sC * 2 * sizeof(float)  // test data
                  + (sA * 2 + sB * 2) * sizeof(uint8_t);  // internal FP8 buffers
    if (needed > free_mem * 0.8) {
        if (strat_verb >= 1)
            fprintf(stderr, "[GemmTune] Skipping autotune: need %.1f GB, only %.1f GB free\n",
                    needed / 1e9, free_mem / 1e9);
        GemmStrategyEntry entry;
        entry.gemm_config = viable_configs.empty() ? 0 : viable_configs[0];
        entry.time_ms = 0;
        entry.tflops = 0;
        entry.bandwidth_gbs = 0;
        entry.use_direct = false;
        entry.direct_tile = 0;
        entry.use_persistent = false;
        return entry;
    }

    __half *d_Ar = nullptr, *d_Ai = nullptr;
    __half *d_Br = nullptr, *d_Bi = nullptr;
    float  *d_Cr = nullptr, *d_Ci = nullptr;

    auto alloc_ok = cudaMalloc(&d_Ar, sA * sizeof(__half));
    if (alloc_ok == cudaSuccess) alloc_ok = cudaMalloc(&d_Ai, sA * sizeof(__half));
    if (alloc_ok == cudaSuccess) alloc_ok = cudaMalloc(&d_Br, sB * sizeof(__half));
    if (alloc_ok == cudaSuccess) alloc_ok = cudaMalloc(&d_Bi, sB * sizeof(__half));
    if (alloc_ok == cudaSuccess) alloc_ok = cudaMalloc(&d_Cr, sC * sizeof(float));
    if (alloc_ok == cudaSuccess) alloc_ok = cudaMalloc(&d_Ci, sC * sizeof(float));

    if (alloc_ok != cudaSuccess) {
        if (strat_verb >= 1)
            fprintf(stderr, "[GemmTune] Skipping autotune: cudaMalloc failed (%s)\n",
                    cudaGetErrorString(alloc_ok));
        cudaFree(d_Ar); cudaFree(d_Ai);
        cudaFree(d_Br); cudaFree(d_Bi);
        cudaFree(d_Cr); cudaFree(d_Ci);
        cudaGetLastError();  // clear sticky error
        GemmStrategyEntry entry;
        entry.gemm_config = viable_configs.empty() ? 0 : viable_configs[0];
        entry.time_ms = 0;
        entry.tflops = 0;
        entry.bandwidth_gbs = 0;
        entry.use_direct = false;
        entry.direct_tile = 0;
        entry.use_persistent = false;
        return entry;
    }

    // Zero-fill -- timing matters, not values
    cudaMemset(d_Ar, 0, sA * sizeof(__half));
    cudaMemset(d_Ai, 0, sA * sizeof(__half));
    cudaMemset(d_Br, 0, sB * sizeof(__half));
    cudaMemset(d_Bi, 0, sB * sizeof(__half));

    // 4b. Prepare B for direct kernel candidates (one-time)
    bool has_direct_candidates = false;
    for (auto& c : candidates) {
        if (c.use_direct) { has_direct_candidates = true; break; }
    }
    if (has_direct_candidates) {
        try {
            if constexpr (!std::is_void_v<InternalComputePrecision>)
                gemm.prepare_b_data(d_Br, d_Bi, N, K, batch_count,
                                    InternalComputePrecision::FP8_E4M3, stream);
            else
                gemm.prepare_b_data(d_Br, d_Bi, N, K, batch_count, stream);
            cudaDeviceSynchronize();
        } catch (...) {
            // If prepare_b fails, remove all direct candidates
            if (strat_verb >= 1)
                fprintf(stderr, "[GemmTune] prepare_b_data() failed, skipping direct candidates\n");
            candidates.erase(
                std::remove_if(candidates.begin(), candidates.end(),
                               [](const GemmTuneCandidate& c) { return c.use_direct; }),
                candidates.end());
        }
    }

    // 5. Benchmark each candidate
    cudaEvent_t ev_start, ev_stop;
    cudaEventCreate(&ev_start);
    cudaEventCreate(&ev_stop);

    float best_time = 1e9f;
    int best_idx = 0;
    int num_candidates = static_cast<int>(candidates.size());

    for (int ci = 0; ci < num_candidates; ci++) {
        auto& c = candidates[ci];

        const char* candidate_name;
        if (c.use_direct) {
            candidate_name = c.use_persistent ? "Direct-Persistent" : "Direct";
        } else {
            candidate_name = config_name_from_int(c.gemm_config);
        }

        if (!c.use_direct) {
            // 4M candidate: validate config
            auto cfg = static_cast<InternalGemmConfig>(c.gemm_config);
            if (!is_config_valid_for_arch(cfg)) {
                if (strat_verb >= 2)
                    fprintf(stderr, "  [%d] %s: SKIPPED (not valid for arch)\n",
                            ci, candidate_name);
                continue;
            }
            gemm.set_gemm_config(cfg);
        }

        // Lambda to run one iteration
        auto run_one = [&]() -> cutlass::Status {
            if (c.use_direct) {
                if (c.use_persistent) {
                    return gemm.run_direct_gemm_prepared_persistent_fp32out(
                        d_Ar, d_Ai, d_Cr, d_Ci,
                        M, N, K, batch_count,
                        1.0f, 0.0f, stream);
                } else {
                    return gemm.run_direct_gemm_prepared_fp32out(
                        d_Ar, d_Ai, d_Cr, d_Ci,
                        M, N, K, batch_count,
                        1.0f, 0.0f, stream);
                }
            } else {
                return gemm.run_planar_batched_fp32out(
                    d_Ar, d_Ai, d_Br, d_Bi, d_Cr, d_Ci,
                    M, N, K, batch_count,
                    1.0f, 0.0f, ComplexModeT::Standard, stream);
            }
        };

        // 2 warmup runs
        bool failed = false;
        for (int w = 0; w < 2; w++) {
            cudaMemset(d_Cr, 0, sC * sizeof(float));
            cudaMemset(d_Ci, 0, sC * sizeof(float));
            try {
                auto status = run_one();
                if (status != cutlass::Status::kSuccess) { failed = true; break; }
            } catch (...) {
                failed = true;
                break;
            }
        }
        if (failed) {
            if (strat_verb >= 2)
                fprintf(stderr, "  [%d] %s: SKIPPED (warmup failed)\n",
                        ci, candidate_name);
            cudaDeviceSynchronize();
            continue;
        }
        cudaDeviceSynchronize();

        // 5 timed runs
        const int timed_runs = 5;
        float total_ms = 0;

        for (int t = 0; t < timed_runs; t++) {
            cudaMemset(d_Cr, 0, sC * sizeof(float));
            cudaMemset(d_Ci, 0, sC * sizeof(float));
            cudaEventRecord(ev_start, stream);
            try {
                auto status = run_one();
                if (status != cutlass::Status::kSuccess) { failed = true; break; }
            } catch (...) {
                failed = true;
                break;
            }
            cudaEventRecord(ev_stop, stream);
            cudaEventSynchronize(ev_stop);
            float ms;
            cudaEventElapsedTime(&ms, ev_start, ev_stop);
            total_ms += ms;
        }

        if (failed) {
            if (strat_verb >= 2)
                fprintf(stderr, "  [%d] %s: SKIPPED (timed run failed)\n",
                        ci, candidate_name);
            continue;
        }

        float avg_ms = total_ms / timed_runs;
        double flops = 8.0 * M * N * K * batch_count;
        double tflops = flops / (avg_ms * 1e9);

        if (strat_verb >= 2)
            fprintf(stderr, "  [%d] %s: %.3f ms (%.1f TFLOPS)%s\n",
                    ci, candidate_name,
                    avg_ms, tflops,
                    avg_ms < best_time ? " <-- best" : "");

        if (avg_ms < best_time) {
            best_time = avg_ms;
            best_idx = ci;
        }
    }

    cudaEventDestroy(ev_start);
    cudaEventDestroy(ev_stop);
    cudaFree(d_Ar); cudaFree(d_Ai);
    cudaFree(d_Br); cudaFree(d_Bi);
    cudaFree(d_Cr); cudaFree(d_Ci);

    // 6. Store result
    auto& winner = candidates[best_idx];
    double flops = 8.0 * M * N * K * batch_count;
    double tflops_val = flops / (best_time * 1e9);
    // External I/O: read A(2MK) + B(2NK) + write C(2MN), all elements x batch x 2 bytes (FP16 input) or 4 (FP32 output)
    double ext_bytes = (double)batch_count * (2.0 * M * K * 2 + 2.0 * N * K * 2 + 2.0 * M * N * 4);
    double bw_gbs = ext_bytes / (best_time * 1e-3) / 1e9;

    GemmStrategyEntry entry;
    entry.gemm_config = winner.gemm_config;
    entry.time_ms = best_time;
    entry.tflops = static_cast<float>(tflops_val);
    entry.bandwidth_gbs = static_cast<float>(bw_gbs);
    entry.use_direct = winner.use_direct;
    entry.direct_tile = winner.direct_tile;
    entry.use_persistent = winner.use_persistent;

    GemmStrategyCache::instance().store(M, N, K, batch_count, precision, entry);

    // 7. Save kernel-level tune cache and restore verbosity
    tune_cache::TuneCache::instance().save();
    tune_cache::TuneCache::instance().set_verbosity(saved_verbosity);

    const char* winner_name = winner.use_direct
        ? (winner.use_persistent ? "Direct-Persistent" : "Direct")
        : config_name_from_int(winner.gemm_config);

    if (strat_verb >= 1)
        fprintf(stderr, "[GemmTune] Winner: %s "
                "(%.3f ms, %.1f TFLOPS, %.1f GB/s)\n",
                winner_name, best_time, tflops_val, bw_gbs);

    return entry;
}

} // namespace strategy_cache
