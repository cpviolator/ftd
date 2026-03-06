// system_info.hpp — Shared GPU/build info for all binaries
//
// Header-only, no CUTLASS dependencies. Prints system info, peak TFLOPS,
// and compiled feature flags at startup.
//
#pragma once

#include <cuda_runtime.h>
#include <iostream>
#include <iomanip>
#include <string>

namespace cutlass_complex {

struct SystemInfo {
    std::string gpu_name;
    int sm_count = 0;
    int smem_per_sm_kb = 0;
    int compute_major = 0;
    int compute_minor = 0;
    int clock_mhz = 0;
    int cuda_runtime_version = 0;
    int cuda_driver_version = 0;

    // Peak TFLOPS (tensor core)
    double peak_fp8_tflops = 0;
    double peak_fp6_tflops = 0;
    double peak_fp4_tflops = 0;

    // Memory bandwidth
    double memory_bw_gbs = 0;   // Theoretical peak GB/s from cudaDeviceProp

    // L2 cache
    int l2_cache_bytes = 0;     // L2 cache size in bytes (from cudaDevAttrL2CacheSize)
};

inline SystemInfo query_system_info(int device = 0) {
    SystemInfo info;

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);

    info.gpu_name = prop.name;
    info.sm_count = prop.multiProcessorCount;
    info.smem_per_sm_kb = prop.sharedMemPerMultiprocessor / 1024;
    info.compute_major = prop.major;
    info.compute_minor = prop.minor;

    int clock_khz = 0;
    cudaDeviceGetAttribute(&clock_khz, cudaDevAttrClockRate, device);
    info.clock_mhz = clock_khz / 1000;

    cudaRuntimeGetVersion(&info.cuda_runtime_version);
    cudaDriverGetVersion(&info.cuda_driver_version);

    // Tensor core ops per cycle per SM (FP8/FP6/FP4, dense, non-sparse)
    // SM90 (Hopper):  8192 FP8 ops/cycle/SM (4th-gen tensor cores, no FP6/FP4)
    //   H100 SXM5: 132 SMs x 1830 MHz x 8192 = 1979 TFLOPS
    //   GH200:     132 SMs x 1980 MHz x 8192 = 2141 TFLOPS
    // SM100/SM120 (Blackwell): 2048 FP8, 2048 FP6, 4096 FP4
    //   GB10:       48 SMs x 2418 MHz x 2048 =  238 TFLOPS (confirmed via mmapeak)
    double fp8_ops = 0, fp6_ops = 0, fp4_ops = 0;
    int sm = prop.major * 10 + prop.minor;
    if (sm >= 120) {
        // Blackwell consumer (SM120/SM121)
        fp8_ops = 2048.0; fp6_ops = 2048.0; fp4_ops = 4096.0;
    } else if (sm >= 100) {
        // Blackwell datacenter (SM100)
        fp8_ops = 2048.0; fp6_ops = 2048.0; fp4_ops = 4096.0;
    } else if (sm >= 90) {
        // Hopper (SM90) — 4th-gen tensor cores: 8192 FP8 ops/cycle/SM
        fp8_ops = 8192.0;
    }

    double freq_ghz = clock_khz / 1.0e6;
    info.peak_fp8_tflops = info.sm_count * freq_ghz * fp8_ops * 1e-3;
    info.peak_fp6_tflops = info.sm_count * freq_ghz * fp6_ops * 1e-3;
    info.peak_fp4_tflops = info.sm_count * freq_ghz * fp4_ops * 1e-3;

    // Memory bandwidth: memoryClockRate (kHz) * 2 (DDR) * busWidth/8 (bytes)
    int mem_clock_khz = 0, mem_bus_width = 0;
    cudaDeviceGetAttribute(&mem_clock_khz, cudaDevAttrMemoryClockRate, device);
    cudaDeviceGetAttribute(&mem_bus_width, cudaDevAttrGlobalMemoryBusWidth, device);
    info.memory_bw_gbs = mem_clock_khz * 2.0 * (mem_bus_width / 8.0) / 1.0e6;

    // L2 cache size
    cudaDeviceGetAttribute(&info.l2_cache_bytes, cudaDevAttrL2CacheSize, device);

    return info;
}

inline std::string format_cuda_version(int v) {
    return std::to_string(v / 1000) + "." + std::to_string((v % 1000) / 10);
}

inline void print_system_info(const SystemInfo& info) {
    std::cout << "CUDA Runtime: " << format_cuda_version(info.cuda_runtime_version)
              << "  Driver: " << format_cuda_version(info.cuda_driver_version) << "\n"
              << "GPU: " << info.gpu_name
              << " (sm_" << info.compute_major << info.compute_minor
              << ", " << info.sm_count << " SMs, "
              << info.smem_per_sm_kb << " KB SMEM/SM, "
              << info.clock_mhz << " MHz)\n"
              << std::fixed << std::setprecision(1)
              << "Peak FP8: ~" << info.peak_fp8_tflops << " TFLOPS";
    if (info.peak_fp6_tflops > 0)
        std::cout << "  FP6: ~" << info.peak_fp6_tflops << " TFLOPS";
    if (info.peak_fp4_tflops > 0)
        std::cout << "  FP4: ~" << info.peak_fp4_tflops << " TFLOPS";
    std::cout << "\n";
    if (info.memory_bw_gbs > 0)
        std::cout << "Memory BW: ~" << info.memory_bw_gbs << " GB/s (theoretical)";
    if (info.l2_cache_bytes > 0)
        std::cout << "  L2: " << (info.l2_cache_bytes / (1024 * 1024)) << " MB";
    if (info.memory_bw_gbs > 0 || info.l2_cache_bytes > 0)
        std::cout << "\n";
}

inline void print_build_config() {
#if defined(CUTLASS_LINKED_MAJOR)
    std::cout << "CUTLASS: " << CUTLASS_LINKED_MAJOR << "."
              << CUTLASS_LINKED_MINOR << "." << CUTLASS_LINKED_PATCH << "\n";
#endif
    std::cout << "Features:";

    // Architecture
#if defined(COMPLEX_FP8_TARGET_SM90)
    std::cout << " SM90";
#elif defined(COMPLEX_FP8_SM100_TARGET_SM120)
    std::cout << " SM120";
#else
    std::cout << " SM100";
#endif

    // Precision features
#if defined(COMPLEX_SM100_ENABLE_FP6)
    std::cout << " FP6";
#endif
#if defined(COMPLEX_SM100_ENABLE_FP4)
    std::cout << " FP4";
#endif
#if defined(COMPLEX_SM100_ENABLE_SMALL_M)
    std::cout << " SmallM";
#endif
    std::cout << " AllCfg";  // All FP8 tile/cluster configs compiled (runtime-selectable)

    // Grouped GEMM
#if defined(COMPLEX_FP8_DISABLE_GROUPED_GEMM)
    std::cout << " GroupedGEMM:OFF";
#elif !defined(COMPLEX_FP8_SM100_TARGET_SM120)
    std::cout << " GroupedGEMM";
#endif

    // Codegen flags
#ifdef COMPLEX_FP8_EXTRA_VECTORIZATION_ENABLED
    std::cout << " ExtraVec";
#endif
#ifdef COMPLEX_FP8_DEVICE_LTO_ENABLED
    std::cout << " LTO";
#endif
#ifdef COMPLEX_FP8_MAX_REGISTERS_VALUE
    std::cout << " MaxReg=" << COMPLEX_FP8_MAX_REGISTERS_VALUE;
#endif

    // HERK mode
#if COMPLEX_FP8_HERK_FULL_MATRIX
    std::cout << " HERK:debug";
#else
    std::cout << " HERK:prod";
#endif

    // SM90-specific features
#ifdef COMPLEX_FP8_USE_PINGPONG
    std::cout << " PingPong";
#endif
#if defined(COMPLEX_FP8_USE_FAST_ACCUM) && COMPLEX_FP8_USE_FAST_ACCUM == 0
    std::cout << " NoFastAccum";
#endif

    std::cout << "\n";

    // FP8 tile configs (all compiled, runtime-selectable via GemmConfig)
#if defined(COMPLEX_FP8_SM100_MMA_M) && !defined(COMPLEX_FP8_TARGET_SM90)
    std::cout << "FP8 tiles: all compiled (runtime-selectable)";
#ifdef COMPLEX_FP8_SM100_STAGES
    std::cout << "  DefaultStages: " << COMPLEX_FP8_SM100_STAGES;
#endif
    std::cout << "\n";
#endif

#if defined(COMPLEX_FP8_TARGET_SM90)
    std::cout << "FP8 tiles: all compiled (runtime-selectable, Coop+Pingpong)\n";
#endif

    // Block-scaled tile (FP6/FP4)
#if defined(COMPLEX_SM100_BLKSCALED_MMA_M) && defined(COMPLEX_SM100_BLKSCALED_MMA_N)
    if (COMPLEX_SM100_BLKSCALED_MMA_M != COMPLEX_FP8_SM100_MMA_M ||
        COMPLEX_SM100_BLKSCALED_MMA_N != COMPLEX_FP8_SM100_MMA_N) {
        std::cout << "BlockScaled Tile: " << COMPLEX_SM100_BLKSCALED_MMA_M << "x"
                  << COMPLEX_SM100_BLKSCALED_MMA_N << "\n";
    }
#endif

    // FP6/FP4 MMA_K values
#if defined(COMPLEX_SM100_FP6_MMA_K) || defined(COMPLEX_SM100_FP4_MMA_K)
    std::cout << "MMA_K:";
#ifdef COMPLEX_SM100_FP6_MMA_K
    std::cout << " FP6=" << COMPLEX_SM100_FP6_MMA_K;
#endif
#ifdef COMPLEX_SM100_FP4_MMA_K
    std::cout << " FP4=" << COMPLEX_SM100_FP4_MMA_K;
#endif
    std::cout << "\n";
#endif
}

// ---- Bandwidth calculation helpers ----

// External I/O bytes for complex GEMM (interleaved FP16 I/O)
// Read A(2MK×2) + B(2KN×2) + Write C(2MN×2) = 4(MK + KN + MN)
inline double gemm_external_bytes(int64_t M, int64_t N, int64_t K, int batch = 1) {
    return 4.0 * batch * (M * K + K * N + M * N);
}

// External I/O bytes for HERK (interleaved FP16 in, packed triangle out)
// Read A(2NK×2) + Write C_packed(N(N+1)/2 complex × 4 bytes)
inline double herk_external_bytes(int64_t N, int64_t K, int batch = 1) {
    return (double)batch * (4.0 * N * K + 2.0 * N * ((int64_t)N + 1));
}

// Roofline ridge point (FLOPs/byte)
inline double ridge_point(double peak_tflops, double peak_bw_gbs) {
    return (peak_bw_gbs > 0) ? (peak_tflops * 1e3 / peak_bw_gbs) : 0;
}

// Internal bytes estimate for batched HERK baseline (NoTrans, Stacked-K)
inline double herk_internal_bytes(int64_t N, int64_t K, int batch) {
    double B = batch;
    double deinterleave = 8.0 * N * K * B;         // read+write planar
    double cast         = 8.0 * N * K * B;         // FP16->FP8 stacked+separate
    double re_gemm      = B * N * (4.0*K + 2.0*N); // read FP8 + write FP16
    double im_gemm      = B * N * (2.0*K + 2.0*N);
    double pack         = B * (4.0*N*N + 2.0*N*(N+1)); // read scratch + write packed
    double interleave   = 4.0 * N * ((int64_t)N+1) / 2 * B; // read+write
    return deinterleave + cast + re_gemm + im_gemm + pack + interleave;
}

// Compute optimal batch tile size for HERK_batched scratch L2 reuse.
// HERK scratch = 2 × N² × sizeof(__half) = 4N² bytes per batch element.
// Returns batch_tile such that scratch fits roughly in L2.
// Returns 0 when tiling is unnecessary (scratch for full batch fits in L2).
inline int compute_herk_batch_tile(int N, int batch_count, int l2_cache_bytes) {
    if (l2_cache_bytes <= 0 || N <= 0 || batch_count <= 0) return 0;
    int64_t scratch_per_batch = 4LL * N * N;  // 2 buffers × N² × 2 bytes
    int64_t total_scratch = scratch_per_batch * batch_count;
    if (total_scratch <= l2_cache_bytes) return 0;  // everything fits, no tiling
    int tile = static_cast<int>(l2_cache_bytes / scratch_per_batch);
    return std::max(tile, 1);
}

} // namespace cutlass_complex
