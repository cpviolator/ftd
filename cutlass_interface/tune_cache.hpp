// ========================================================================================
// CUDA Kernel Autotuning Cache
// ========================================================================================
//
// Provides automatic blockDim/gridDim tuning for overhead kernels (cast, pack, etc.).
//
// Usage:
//   1. Set verbosity:  tune_cache::TuneCache::instance().set_verbosity(level);
//        0 = silent:     no tuning, no output
//        1 = info:       no tuning, shows cached/default params per kernel
//        2 = tune:       sweeps configs, shows one-line result per kernel
//        3 = tune+detail: sweeps configs with full per-config timing output
//   2. Replace kernel<<<grid, block, 0, stream>>>(args...) with:
//        TUNED_LAUNCH_1D(kernel, "name", total_elems, ept, bytes, stream, args...)
//   3. On first call (verbosity >= 2), sweeps blockDim × gridDim, picks fastest, caches.
//   4. Subsequent calls use cached params if file exists.
//
// Three launch patterns:
//   TUNED_LAUNCH_1D     — 1D grid-stride kernels (cast, deinterleave, stack)
//   TUNED_LAUNCH_ROW    — row-per-block kernels (pack_triangle, antisymmetrize_pack)
//   TUNED_LAUNCH_2D     — 2D element-wise without shared memory (enforce_hermitian)
//
// Cache file: one per build configuration, named
//   cutlass_kernel_cache_{build_fingerprint}.txt
// Example: cutlass_kernel_cache_sm120_m128x64x128_fp6k128_fp4k256_bs128x128_stg3.txt
//
// Format (text, one entry per line, no per-row build fingerprint):
//   # CUTLASS Kernel Tune Cache
//   # GPU: NVIDIA GB10
//   # Build: sm120_m128x64x128_fp6k128_fp4k256_bs128x128_stg3
//   #
//   # key  block_x block_y block_z  grid_x grid_y grid_z  bandwidth_GBps time_us
//   cast_fp16_to_fp8:65536  128 1 1  64 1 1  240.4  418.70
//
// Backward compat: loads legacy cutlass_tunecache.txt (per-row build fingerprint
// format) by filtering to rows matching the current build.
//
// Note: 2D transposed kernels with shared memory tiles are NOT tunable via this system
// (their blockDim is coupled to __shared__ array dimensions).

#pragma once

#include <string>
#include <unordered_map>
#include <vector>
#include <set>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sys/stat.h>
#include <cuda_runtime.h>

namespace tune_cache {

// ========================================================================================
// Cache directory from environment
// ========================================================================================
//
// Returns the directory prefix for all tune/strategy cache files.
// If CUTLASS_TUNECACHE_PATH is set, returns its value (with trailing '/').
// Otherwise returns "" (current working directory).

inline std::string cache_dir_prefix() {
    const char* env = std::getenv("CUTLASS_TUNECACHE_PATH");
    if (env && env[0] != '\0') {
        struct stat st;
        if (stat(env, &st) != 0 || !S_ISDIR(st.st_mode)) {
            fprintf(stderr,
                    "ERROR: CUTLASS_TUNECACHE_PATH='%s' does not exist or is not a directory.\n"
                    "  Create the directory or unset the variable.\n", env);
            exit(1);
        }
        std::string dir(env);
        if (dir.back() != '/') dir += '/';
        return dir;
    }
    return "";
}

// ========================================================================================
// Build configuration fingerprint
// ========================================================================================
//
// Constructs a canonical string from compile-time #ifdef/#define checks so that
// each build configuration's tuning entries are uniquely identified within the
// shared cache file.
//
// SM100/SM120 format: sm120_allcfg_fp6k128_fp4k256_bs128x128_stg3_vec
// SM90 format:        sm90_allcfg_vec
//
// Structure: {arch}_allcfg[_fp6k{K}][_fp4k{K}][_bs{M}x{N}][_stg{N}][_vec][_lto][_reg{N}]
// Note: "_allcfg" indicates all FP8 tile/cluster/stage configs are compiled in
// (runtime-selectable via GemmConfig). Block-scaled tiles are still cmake-parameterized.

inline std::string build_config_fingerprint() {
    std::string fp;

    // Architecture
#if defined(COMPLEX_FP8_TARGET_SM90)
    fp = "sm90";
#elif defined(COMPLEX_FP8_SM100_TARGET_SM120)
    fp = "sm120";
#else
    fp = "sm100";
#endif

    // All FP8 tile/cluster/schedule configs are always compiled in
    fp += "_allcfg";

#if !defined(COMPLEX_FP8_TARGET_SM90)

    // FP6/FP4 precision
#if defined(COMPLEX_SM100_ENABLE_FP6)
    fp += "_fp6k" + std::to_string(COMPLEX_SM100_FP6_MMA_K);
#endif
#if defined(COMPLEX_SM100_ENABLE_FP4)
    fp += "_fp4k" + std::to_string(COMPLEX_SM100_FP4_MMA_K);
#endif

    // Block-scaled tile (only if FP6 or FP4 enabled)
#if defined(COMPLEX_SM100_ENABLE_FP6) || defined(COMPLEX_SM100_ENABLE_FP4)
    fp += "_bs" + std::to_string(COMPLEX_SM100_BLKSCALED_MMA_M)
        + "x" + std::to_string(COMPLEX_SM100_BLKSCALED_MMA_N);
#endif

    // Pipeline stages (only if explicitly set)
#if defined(COMPLEX_FP8_SM100_STAGES)
    fp += "_stg" + std::to_string(COMPLEX_FP8_SM100_STAGES);
#endif

#endif  // SM90 vs SM100/SM120

    // Codegen flags (cross-architecture)
#if defined(COMPLEX_FP8_EXTRA_VECTORIZATION_ENABLED)
    fp += "_vec";
#endif
#if defined(COMPLEX_FP8_DEVICE_LTO_ENABLED)
    fp += "_lto";
#endif
#if defined(COMPLEX_FP8_MAX_REGISTERS_VALUE)
    fp += "_reg" + std::to_string(COMPLEX_FP8_MAX_REGISTERS_VALUE);
#endif

    return fp;
}

// Human-readable multi-line build config for cache file header (informational)
inline std::string build_config_header_string() {
    std::string h;

    // Architecture
#if defined(COMPLEX_FP8_TARGET_SM90)
    h += "# Architecture: SM90\n";
    h += "# FP8 configs: all tile/cluster/schedule combos compiled (runtime-selectable)\n";
#elif defined(COMPLEX_FP8_SM100_TARGET_SM120)
    h += "# Architecture: SM120\n";
    h += "# FP8 configs: all tile/stage combos compiled (runtime-selectable)\n";
#else
    h += "# Architecture: SM100\n";
    h += "# FP8 configs: all tile/cluster/stage combos compiled (runtime-selectable)\n";
#endif

#if !defined(COMPLEX_FP8_TARGET_SM90)

#if defined(COMPLEX_SM100_ENABLE_FP6)
    h += "# FP6: enabled (MMA_K=" + std::to_string(COMPLEX_SM100_FP6_MMA_K) + ")\n";
#else
    h += "# FP6: disabled\n";
#endif

#if defined(COMPLEX_SM100_ENABLE_FP4)
    h += "# FP4: enabled (MMA_K=" + std::to_string(COMPLEX_SM100_FP4_MMA_K) + ")\n";
#else
    h += "# FP4: disabled\n";
#endif

#if defined(COMPLEX_SM100_ENABLE_FP6) || defined(COMPLEX_SM100_ENABLE_FP4)
    h += "# Block-scaled tile: " + std::to_string(COMPLEX_SM100_BLKSCALED_MMA_M) + "x"
       + std::to_string(COMPLEX_SM100_BLKSCALED_MMA_N) + "\n";
#endif

#if defined(COMPLEX_FP8_SM100_STAGES)
    h += "# Pipeline stages: " + std::to_string(COMPLEX_FP8_SM100_STAGES) + "\n";
#else
    h += "# Pipeline stages: auto\n";
#endif

#endif  // !SM90

    // Codegen flags
#if defined(COMPLEX_FP8_EXTRA_VECTORIZATION_ENABLED)
    h += "# Extra vectorization: ON\n";
#else
    h += "# Extra vectorization: OFF\n";
#endif

#if defined(COMPLEX_FP8_DEVICE_LTO_ENABLED)
    h += "# Device LTO: ON\n";
#else
    h += "# Device LTO: OFF\n";
#endif

#if defined(COMPLEX_FP8_MAX_REGISTERS_VALUE)
    h += "# Max registers: " + std::to_string(COMPLEX_FP8_MAX_REGISTERS_VALUE) + "\n";
#else
    h += "# Max registers: default\n";
#endif

    return h;
}

struct TuneEntry {
    dim3 block = {256, 1, 1};
    dim3 grid  = {1, 1, 1};
    float bandwidth_gbps = 0.0f;
    float time_us = 0.0f;
};

class TuneCache {
public:
    static TuneCache& instance() {
        static TuneCache inst;
        return inst;
    }

    // Configuration — verbosity levels:
    //   0 = silent:      no tuning, no output
    //   1 = info:        no tuning, shows cached/default params per kernel (once per unique key)
    //   2 = tune:        sweeps configs, shows one-line result per kernel
    //   3 = tune+detail: sweeps configs with full per-config timing output
    void set_verbosity(int level) { verbosity_ = level; }
    int  verbosity() const { return verbosity_; }
    bool tuning_enabled() const { return verbosity_ >= 2; }

    // Backward compatibility
    void set_tuning_enabled(bool e) { if (e && verbosity_ < 2) verbosity_ = 2; }
    void set_verbose(bool v) { if (v && verbosity_ < 3) verbosity_ = 3; }

    void set_file_path(const std::string& path) {
        file_path_ = path;
        custom_path_ = true;
        loaded_ = false;
        dirty_ = false;
        cache_.clear();
        printed_keys_.clear();
    }

    // ================================================================================
    // File I/O
    // ================================================================================

    void load() {
        if (loaded_) return;
        loaded_ = true;
        ensure_gpu_name();
        ensure_file_path();

        std::string current_fp = build_config_fingerprint();

        // Try the per-build file first, then fall back to legacy cutlass_tunecache.txt
        if (!load_from_file(file_path_, current_fp)) {
            if (!custom_path_) {
                std::string legacy = cache_dir_prefix() + "cutlass_tunecache.txt";
                if (legacy != file_path_) {
                    load_from_file(legacy, current_fp);
                }
            }
        }

        if (verbosity_ >= 1) {
            printf("[TuneCache] Loaded %zu cached entries for build '%s' from %s\n",
                   cache_.size(), current_fp.c_str(), file_path_.c_str());
        }
    }

    void save() {
        if (!dirty_) return;
        ensure_gpu_name();
        ensure_file_path();

        std::string tmp_path = file_path_ + ".tmp";
        std::ofstream f(tmp_path);
        if (!f.is_open()) {
            fprintf(stderr, "[TuneCache] Failed to open %s for writing\n", tmp_path.c_str());
            return;
        }

        std::string current_fp = build_config_fingerprint();

        f << "# CUTLASS Kernel Tune Cache\n";
        f << "# GPU: " << gpu_name_ << "\n";
        f << "# Build: " << current_fp << "\n";
        f << build_config_header_string();
        f << "#\n";
        f << "# key  block_x block_y block_z  grid_x grid_y grid_z  bandwidth_GBps time_us\n";

        // Sort by key for deterministic output
        std::vector<std::pair<std::string, TuneEntry>> sorted(cache_.begin(), cache_.end());
        std::sort(sorted.begin(), sorted.end(),
                  [](const auto& a, const auto& b) { return a.first < b.first; });

        for (const auto& kv : sorted) {
            char buf[512];
            snprintf(buf, sizeof(buf), "%-60s %4u %4u %4u %6u %6u %6u %8.1f %8.2f",
                     kv.first.c_str(),
                     kv.second.block.x, kv.second.block.y, kv.second.block.z,
                     kv.second.grid.x, kv.second.grid.y, kv.second.grid.z,
                     kv.second.bandwidth_gbps, kv.second.time_us);
            f << buf << "\n";
        }

        f.close();
        std::rename(tmp_path.c_str(), file_path_.c_str());
        dirty_ = false;
        if (verbosity_ >= 2) {
            printf("[TuneCache] Saved %zu entries to %s\n", cache_.size(), file_path_.c_str());
        }
    }

    // ================================================================================
    // 1D grid-stride kernel launch (cast, deinterleave, stack, etc.)
    // ================================================================================
    //
    // Sweeps blockDim.x ∈ {32,64,128,256,512,1024} and gridDim.x capped at
    // {sm_count, 2×sm, 4×sm, 8×sm, full_coverage}.

    template<typename LaunchFn>
    void launch_tuned_1d(const char* name, int64_t total_elems, int elems_per_thread,
                         int64_t total_bytes, cudaStream_t stream, LaunchFn launch) {
        ensure_loaded();

        std::string key = make_key_1d(name, total_elems);

        // If in graph capture, use cached or default (can't synchronize)
        if (is_capturing(stream)) {
            auto it = cache_.find(key);
            if (it != cache_.end()) {
                if (verbosity_ >= 1 && printed_keys_.insert(key).second) {
                    printf("[TuneCache] %-45s cached  block=(%u,%u,%u) grid=(%u,%u,%u) [graph]\n",
                           key.c_str(), it->second.block.x, it->second.block.y, it->second.block.z,
                           it->second.grid.x, it->second.grid.y, it->second.grid.z);
                }
                launch(it->second.grid, it->second.block, stream);
            } else {
                dim3 dg(default_grid_1d(total_elems, 256, elems_per_thread));
                if (verbosity_ >= 1 && printed_keys_.insert(key).second) {
                    printf("[TuneCache] %-45s default block=(256,1,1) grid=(%u,1,1) [graph]\n",
                           key.c_str(), dg.x);
                }
                launch(dg, dim3(256), stream);
            }
            return;
        }

        // Check cache
        auto it = cache_.find(key);
        if (it != cache_.end()) {
            if (verbosity_ >= 1 && printed_keys_.insert(key).second) {
                printf("[TuneCache] %-45s cached  block=(%u,%u,%u) grid=(%u,%u,%u)\n",
                       key.c_str(), it->second.block.x, it->second.block.y, it->second.block.z,
                       it->second.grid.x, it->second.grid.y, it->second.grid.z);
            }
            launch(it->second.grid, it->second.block, stream);
            return;
        }

        // No cache hit — use defaults if tuning not enabled
        if (verbosity_ < 2) {
            dim3 dg(default_grid_1d(total_elems, 256, elems_per_thread));
            if (verbosity_ >= 1 && printed_keys_.insert(key).second) {
                printf("[TuneCache] %-45s default block=(256,1,1) grid=(%u,1,1)\n",
                       key.c_str(), dg.x);
            }
            launch(dg, dim3(256), stream);
            return;
        }

        // Tune
        TuneEntry best = sweep_1d(name, total_elems, elems_per_thread, total_bytes, stream, launch);
        cache_[key] = best;
        dirty_ = true;
        save();  // persist after each new entry

        launch(best.grid, best.block, stream);
    }

    // ================================================================================
    // Row-per-block kernel launch (pack_triangle, antisymmetrize_pack, etc.)
    // ================================================================================
    //
    // Grid = (N, batch_count), sweeps blockDim.x ∈ {32,64,128,256,512,1024}.

    template<typename LaunchFn>
    void launch_tuned_row(const char* name, int N, int batch_count,
                          int64_t total_bytes, cudaStream_t stream, LaunchFn launch) {
        ensure_loaded();

        std::string key = make_key_2d(name, N, batch_count, 0);

        if (is_capturing(stream)) {
            auto it = cache_.find(key);
            if (it != cache_.end()) {
                if (verbosity_ >= 1 && printed_keys_.insert(key).second) {
                    printf("[TuneCache] %-45s cached  block=(%u,%u,%u) grid=(%u,%u,%u) [graph]\n",
                           key.c_str(), it->second.block.x, it->second.block.y, it->second.block.z,
                           it->second.grid.x, it->second.grid.y, it->second.grid.z);
                }
                launch(it->second.grid, it->second.block, stream);
            } else {
                if (verbosity_ >= 1 && printed_keys_.insert(key).second) {
                    printf("[TuneCache] %-45s default block=(256,1,1) grid=(%u,%u,1) [graph]\n",
                           key.c_str(), (unsigned)N, (unsigned)batch_count);
                }
                launch(dim3(N, batch_count), dim3(256), stream);
            }
            return;
        }

        auto it = cache_.find(key);
        if (it != cache_.end()) {
            if (verbosity_ >= 1 && printed_keys_.insert(key).second) {
                printf("[TuneCache] %-45s cached  block=(%u,%u,%u) grid=(%u,%u,%u)\n",
                       key.c_str(), it->second.block.x, it->second.block.y, it->second.block.z,
                       it->second.grid.x, it->second.grid.y, it->second.grid.z);
            }
            launch(it->second.grid, it->second.block, stream);
            return;
        }

        if (verbosity_ < 2) {
            if (verbosity_ >= 1 && printed_keys_.insert(key).second) {
                printf("[TuneCache] %-45s default block=(256,1,1) grid=(%u,%u,1)\n",
                       key.c_str(), (unsigned)N, (unsigned)batch_count);
            }
            launch(dim3(N, batch_count), dim3(256), stream);
            return;
        }

        TuneEntry best = sweep_row(name, N, batch_count, total_bytes, stream, launch);
        cache_[key] = best;
        dirty_ = true;
        save();

        launch(best.grid, best.block, stream);
    }

    // ================================================================================
    // 2D element-wise kernel launch (enforce_hermitian, antisymmetrize_to_triangle)
    // ================================================================================
    //
    // Sweeps block shapes: (8,8), (16,16), (32,32), (8,16), (16,8), (16,32), (32,16).
    // Grid derived as ceil(dims/block). No shared memory constraints.

    template<typename LaunchFn>
    void launch_tuned_2d(const char* name, int rows, int cols, int batch_count,
                         int64_t total_bytes, cudaStream_t stream, LaunchFn launch) {
        ensure_loaded();

        std::string key = make_key_2d(name, rows, cols, batch_count);

        if (is_capturing(stream)) {
            auto it = cache_.find(key);
            if (it != cache_.end()) {
                if (verbosity_ >= 1 && printed_keys_.insert(key).second) {
                    printf("[TuneCache] %-45s cached  block=(%u,%u,%u) grid=(%u,%u,%u) [graph]\n",
                           key.c_str(), it->second.block.x, it->second.block.y, it->second.block.z,
                           it->second.grid.x, it->second.grid.y, it->second.grid.z);
                }
                launch(it->second.grid, it->second.block, stream);
            } else {
                dim3 db(16, 16);
                dim3 dg((cols + 15) / 16, (rows + 15) / 16, batch_count);
                if (verbosity_ >= 1 && printed_keys_.insert(key).second) {
                    printf("[TuneCache] %-45s default block=(16,16,1) grid=(%u,%u,%u) [graph]\n",
                           key.c_str(), dg.x, dg.y, dg.z);
                }
                launch(dg, db, stream);
            }
            return;
        }

        auto it = cache_.find(key);
        if (it != cache_.end()) {
            if (verbosity_ >= 1 && printed_keys_.insert(key).second) {
                printf("[TuneCache] %-45s cached  block=(%u,%u,%u) grid=(%u,%u,%u)\n",
                       key.c_str(), it->second.block.x, it->second.block.y, it->second.block.z,
                       it->second.grid.x, it->second.grid.y, it->second.grid.z);
            }
            launch(it->second.grid, it->second.block, stream);
            return;
        }

        if (verbosity_ < 2) {
            dim3 db(16, 16);
            dim3 dg((cols + 15) / 16, (rows + 15) / 16, batch_count);
            if (verbosity_ >= 1 && printed_keys_.insert(key).second) {
                printf("[TuneCache] %-45s default block=(16,16,1) grid=(%u,%u,%u)\n",
                       key.c_str(), dg.x, dg.y, dg.z);
            }
            launch(dg, db, stream);
            return;
        }

        TuneEntry best = sweep_2d(name, rows, cols, batch_count, total_bytes, stream, launch);
        cache_[key] = best;
        dirty_ = true;
        save();

        launch(best.grid, best.block, stream);
    }

private:
    TuneCache() = default;
    ~TuneCache() { save(); }

    TuneCache(const TuneCache&) = delete;
    TuneCache& operator=(const TuneCache&) = delete;

    std::unordered_map<std::string, TuneEntry> cache_;
    std::set<std::string> printed_keys_;  // dedup verbosity=1 output (print once per key)

    std::string file_path_;   // empty until first use (auto-generated) or set_file_path()
    int verbosity_ = 0;      // 0=silent, 1=info, 2=tune, 3=tune+detail
    bool custom_path_ = false;
    bool loaded_ = false;
    bool dirty_ = false;
    int sm_count_ = 0;
    std::string gpu_name_;

    // ================================================================================
    // Helpers
    // ================================================================================

    void ensure_loaded() {
        if (!loaded_) load();
    }

    void ensure_file_path() {
        if (!custom_path_ && file_path_.empty()) {
            file_path_ = cache_dir_prefix() +
                          "cutlass_kernel_cache_" +
                          build_config_fingerprint() + ".txt";
        }
    }

    /// Load entries from a file, filtering old per-row-tagged formats to current build.
    /// Returns true if the file was opened (even if no matching entries were found).
    bool load_from_file(const std::string& path, const std::string& current_fp) {
        std::ifstream f(path);
        if (!f.is_open()) return false;

        std::string line;
        bool gpu_ok = true;
        while (std::getline(f, line)) {
            if (line.empty()) continue;
            if (line[0] == '#') {
                if (line.find("# GPU: ") == 0) {
                    std::string file_gpu = line.substr(7);
                    if (file_gpu != gpu_name_) {
                        if (verbosity_ >= 1) {
                            printf("[TuneCache] GPU mismatch: file='%s' current='%s' — ignoring %s\n",
                                   file_gpu.c_str(), gpu_name_.c_str(), path.c_str());
                        }
                        gpu_ok = false;
                        break;
                    }
                }
                continue;
            }

            // Detect format: first token starting with "sm90_"/"sm100_"/"sm120_"
            // indicates old per-row-tagged 10-column format.
            char first_tok[256];
            if (sscanf(line.c_str(), "%255s", first_tok) != 1) continue;

            bool has_build_prefix = (strncmp(first_tok, "sm90_", 5) == 0 ||
                                     strncmp(first_tok, "sm100_", 6) == 0 ||
                                     strncmp(first_tok, "sm120_", 6) == 0);

            char key[256];
            int bx, by, bz, gx, gy, gz;
            float bw, t;

            if (has_build_prefix) {
                // Old 10-column format: build_fp key bx by bz gx gy gz bw t
                char build_fp[256];
                if (sscanf(line.c_str(), "%255s %255s %d %d %d %d %d %d %f %f",
                           build_fp, key, &bx, &by, &bz, &gx, &gy, &gz, &bw, &t) == 10) {
                    if (std::string(build_fp) != current_fp) continue;  // skip other builds
                    TuneEntry entry;
                    entry.block = dim3(bx, by, bz);
                    entry.grid  = dim3(gx, gy, gz);
                    entry.bandwidth_gbps = bw;
                    entry.time_us = t;
                    cache_[std::string(key)] = entry;
                }
            } else {
                // New 9-column format (or legacy 9-column): key bx by bz gx gy gz bw t
                if (sscanf(line.c_str(), "%255s %d %d %d %d %d %d %f %f",
                           key, &bx, &by, &bz, &gx, &gy, &gz, &bw, &t) == 9) {
                    TuneEntry entry;
                    entry.block = dim3(bx, by, bz);
                    entry.grid  = dim3(gx, gy, gz);
                    entry.bandwidth_gbps = bw;
                    entry.time_us = t;
                    cache_[std::string(key)] = entry;
                }
            }
        }

        if (!gpu_ok) {
            cache_.clear();
        }
        return true;
    }

    void ensure_sm_count() {
        if (sm_count_ == 0) {
            cudaDeviceGetAttribute(&sm_count_, cudaDevAttrMultiProcessorCount, 0);
        }
    }

    void ensure_gpu_name() {
        if (gpu_name_.empty()) {
            cudaDeviceProp prop;
            if (cudaGetDeviceProperties(&prop, 0) == cudaSuccess) {
                gpu_name_ = prop.name;
            } else {
                gpu_name_ = "Unknown";
            }
        }
        ensure_sm_count();
    }

    static bool is_capturing(cudaStream_t stream) {
        cudaStreamCaptureStatus status;
        cudaGetLastError();  // clear prior errors
        cudaError_t err = cudaStreamIsCapturing(stream, &status);
        if (err != cudaSuccess) {
            cudaGetLastError();  // consume the error
            return false;
        }
        return status != cudaStreamCaptureStatusNone;
    }

    static int default_grid_1d(int64_t total, int block_size, int elems_per_thread) {
        int64_t elems_per_block = static_cast<int64_t>(block_size) * elems_per_thread;
        return static_cast<int>(std::min<int64_t>((total + elems_per_block - 1) / elems_per_block, 65535));
    }

    std::string make_key_1d(const char* name, int64_t total) {
        char buf[256];
        snprintf(buf, sizeof(buf), "%s:%lld", name, (long long)total);
        return std::string(buf);
    }

    std::string make_key_2d(const char* name, int d1, int d2, int d3) {
        char buf[256];
        snprintf(buf, sizeof(buf), "%s:%dx%dx%d", name, d1, d2, d3);
        return std::string(buf);
    }

    // ================================================================================
    // Timing helpers
    // ================================================================================

    static constexpr int kWarmup = 3;
    static constexpr int kTrials = 10;

    struct TimingResult {
        float avg_us;
        float bw_gbps;
    };

    template<typename LaunchFn>
    TimingResult time_config(dim3 grid, dim3 block, int64_t total_bytes,
                             cudaStream_t stream, LaunchFn& launch) {
        // Warmup
        for (int w = 0; w < kWarmup; ++w) {
            launch(grid, block, stream);
        }
        cudaStreamSynchronize(stream);

        // Check for errors from warmup
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            return {1e9f, 0.0f};  // invalid config
        }

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        cudaEventRecord(start, stream);
        for (int t = 0; t < kTrials; ++t) {
            launch(grid, block, stream);
        }
        cudaEventRecord(stop, stream);
        cudaEventSynchronize(stop);

        err = cudaGetLastError();
        if (err != cudaSuccess) {
            cudaEventDestroy(start);
            cudaEventDestroy(stop);
            return {1e9f, 0.0f};
        }

        float ms;
        cudaEventElapsedTime(&ms, start, stop);
        float avg_us = ms * 1000.0f / kTrials;
        float bw = (avg_us > 0) ? (total_bytes / 1e9f) / (avg_us / 1e6f) : 0.0f;

        cudaEventDestroy(start);
        cudaEventDestroy(stop);

        return {avg_us, bw};
    }

    // ================================================================================
    // Sweep functions
    // ================================================================================

    template<typename LaunchFn>
    TuneEntry sweep_1d(const char* name, int64_t total_elems, int elems_per_thread,
                       int64_t total_bytes, cudaStream_t stream, LaunchFn& launch) {
        ensure_sm_count();

        static const int block_sizes[] = {32, 64, 128, 256, 512, 1024};
        static const int grid_mults[] = {1, 2, 4, 8, 0};  // 0 = unlimited

        if (verbosity_ >= 2) {
            printf("[TuneCache] Tuning 1D '%s' (total=%lld, ept=%d, bytes=%lld)\n",
                   name, (long long)total_elems, elems_per_thread, (long long)total_bytes);
        }

        TuneEntry best;
        best.time_us = 1e9f;

        for (int bs : block_sizes) {
            int full_grid = default_grid_1d(total_elems, bs, elems_per_thread);
            if (full_grid <= 0) continue;

            // Generate unique grid sizes
            std::set<int> grid_vals;
            grid_vals.insert(full_grid);
            for (int mult : grid_mults) {
                if (mult == 0) continue;
                grid_vals.insert(std::min(mult * sm_count_, full_grid));
            }

            for (int g : grid_vals) {
                if (g <= 0) continue;
                dim3 grid(g);
                dim3 block(bs);

                auto result = time_config(grid, block, total_bytes, stream, launch);

                if (verbosity_ >= 3 && result.avg_us < 1e8f) {
                    printf("  block=%4d  grid=%6d  -> %8.2f us  %7.1f GB/s\n",
                           bs, g, result.avg_us, result.bw_gbps);
                }

                if (result.avg_us < best.time_us) {
                    best.block = block;
                    best.grid = grid;
                    best.time_us = result.avg_us;
                    best.bandwidth_gbps = result.bw_gbps;
                }
            }
        }

        if (verbosity_ >= 2) {
            printf("[TuneCache] %-45s tuned   block=(%u,%u,%u) grid=(%u,%u,%u) %.2f us %.1f GB/s\n",
                   make_key_1d(name, total_elems).c_str(),
                   best.block.x, best.block.y, best.block.z,
                   best.grid.x, best.grid.y, best.grid.z,
                   best.time_us, best.bandwidth_gbps);
        }

        return best;
    }

    template<typename LaunchFn>
    TuneEntry sweep_row(const char* name, int N, int batch_count,
                        int64_t total_bytes, cudaStream_t stream, LaunchFn& launch) {
        static const int block_sizes[] = {32, 64, 128, 256, 512, 1024};

        if (verbosity_ >= 2) {
            printf("[TuneCache] Tuning ROW '%s' (N=%d, batch=%d, bytes=%lld)\n",
                   name, N, batch_count, (long long)total_bytes);
        }

        TuneEntry best;
        best.time_us = 1e9f;

        for (int bs : block_sizes) {
            dim3 grid(N, batch_count);
            dim3 block(bs);

            auto result = time_config(grid, block, total_bytes, stream, launch);

            if (verbosity_ >= 3 && result.avg_us < 1e8f) {
                printf("  block=%4d  -> %8.2f us  %7.1f GB/s\n",
                       bs, result.avg_us, result.bw_gbps);
            }

            if (result.avg_us < best.time_us) {
                best.block = block;
                best.grid = grid;
                best.time_us = result.avg_us;
                best.bandwidth_gbps = result.bw_gbps;
            }
        }

        if (verbosity_ >= 2) {
            printf("[TuneCache] %-45s tuned   block=(%u,%u,%u) grid=(%u,%u,%u) %.2f us %.1f GB/s\n",
                   make_key_2d(name, N, batch_count, 0).c_str(),
                   best.block.x, best.block.y, best.block.z,
                   best.grid.x, best.grid.y, best.grid.z,
                   best.time_us, best.bandwidth_gbps);
        }

        return best;
    }

    template<typename LaunchFn>
    TuneEntry sweep_2d(const char* name, int rows, int cols, int batch_count,
                       int64_t total_bytes, cudaStream_t stream, LaunchFn& launch) {
        // Block shape candidates (all with total threads ≤ 1024)
        struct BlockShape { int x, y; };
        static const BlockShape shapes[] = {
            {8, 8}, {16, 8}, {8, 16}, {16, 16},
            {32, 8}, {8, 32}, {32, 16}, {16, 32}, {32, 32}
        };

        if (verbosity_ >= 2) {
            printf("[TuneCache] Tuning 2D '%s' (%dx%dx%d, bytes=%lld)\n",
                   name, rows, cols, batch_count, (long long)total_bytes);
        }

        TuneEntry best;
        best.time_us = 1e9f;

        for (const auto& s : shapes) {
            dim3 block(s.x, s.y);
            dim3 grid((cols + s.x - 1) / s.x, (rows + s.y - 1) / s.y, batch_count);

            auto result = time_config(grid, block, total_bytes, stream, launch);

            if (verbosity_ >= 3 && result.avg_us < 1e8f) {
                printf("  block=(%2d,%2d) grid=(%4u,%4u,%u) -> %8.2f us  %7.1f GB/s\n",
                       s.x, s.y, grid.x, grid.y, grid.z,
                       result.avg_us, result.bw_gbps);
            }

            if (result.avg_us < best.time_us) {
                best.block = block;
                best.grid = grid;
                best.time_us = result.avg_us;
                best.bandwidth_gbps = result.bw_gbps;
            }
        }

        if (verbosity_ >= 2) {
            printf("[TuneCache] %-45s tuned   block=(%u,%u,%u) grid=(%u,%u,%u) %.2f us %.1f GB/s\n",
                   make_key_2d(name, rows, cols, batch_count).c_str(),
                   best.block.x, best.block.y, best.block.z,
                   best.grid.x, best.grid.y, best.grid.z,
                   best.time_us, best.bandwidth_gbps);
        }

        return best;
    }
};

} // namespace tune_cache


// ========================================================================================
// Convenience macros for kernel launchers
// ========================================================================================
//
// These replace the manual <<<grid, block, 0, stream>>> kernel launch with an
// auto-tuned launch. The kernel function and its arguments are passed through.
//
// Usage:
//   // Before:
//   int num_blocks = min((total + 256*8 - 1) / (256*8), 65535);
//   my_kernel<<<num_blocks, 256, 0, stream>>>(arg1, arg2, arg3);
//
//   // After:
//   TUNED_LAUNCH_1D(my_kernel, "my_kernel", total, 8, total_bytes, stream, arg1, arg2, arg3);
//
// The CUDA_CHECK(cudaGetLastError()) should follow after the macro call.

#define TUNED_LAUNCH_1D(kernel, name, total_elems, ept, total_bytes, stream, ...) \
    ::tune_cache::TuneCache::instance().launch_tuned_1d( \
        name, total_elems, ept, total_bytes, stream, \
        [&](dim3 _tc_g, dim3 _tc_b, cudaStream_t _tc_s) { \
            kernel<<<_tc_g, _tc_b, 0, _tc_s>>>(__VA_ARGS__); \
        })

#define TUNED_LAUNCH_ROW(kernel, name, N, batch, total_bytes, stream, ...) \
    ::tune_cache::TuneCache::instance().launch_tuned_row( \
        name, N, batch, total_bytes, stream, \
        [&](dim3 _tc_g, dim3 _tc_b, cudaStream_t _tc_s) { \
            kernel<<<_tc_g, _tc_b, 0, _tc_s>>>(__VA_ARGS__); \
        })

#define TUNED_LAUNCH_2D(kernel, name, rows, cols, batch, total_bytes, stream, ...) \
    ::tune_cache::TuneCache::instance().launch_tuned_2d( \
        name, rows, cols, batch, total_bytes, stream, \
        [&](dim3 _tc_g, dim3 _tc_b, cudaStream_t _tc_s) { \
            kernel<<<_tc_g, _tc_b, 0, _tc_s>>>(__VA_ARGS__); \
        })
