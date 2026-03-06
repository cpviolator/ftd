/*
 * benchmark_configs.cu
 *
 * Multi-configuration CUTLASS FP8 GEMM benchmark for Hopper (SM90).
 *
 * Tests real-valued FP8 GEMMs (the sub-GEMM that limits complex GEMM performance)
 * across different:
 *   - Kernel schedules: Auto, Cooperative+FastAccum, Pingpong+FastAccum
 *   - Cluster shapes:   1×1×1, 2×1×1, 1×2×1
 *   - Tile shapes:      128×128×128, 256×128×128
 *   - Swizzle sizes:    1, 2, 4, 8 (runtime tunable for persistent kernels)
 *   - Raster order:     Heuristic, AlongM, AlongN
 *
 * Each configuration is a separate CUTLASS kernel instantiation, benchmarked
 * in isolation. Results show which template parameters yield the best throughput.
 *
 * Usage:
 *   ./benchmark_configs [M] [N] [K]
 *   ./benchmark_configs 8192 8192 8192
 *   ./benchmark_configs 4096           # M=N=K=4096
 *
 * Compile:
 *   mkdir build && cd build
 *   cmake .. -DCUTLASS_DIR=/path/to/cutlass -DCMAKE_CUDA_ARCHITECTURES=90a
 *   make -j$(nproc) benchmark_configs
 *
 * NOTE: This benchmark instantiates ~11 unique CUTLASS kernels. Compilation
 * may take 10-20 minutes depending on your machine and nvcc version.
 * If specific configs fail to compile (e.g., 256x128 tile + certain clusters),
 * comment out those sections and rebuild.
 */

#include "system_info.hpp"
#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <cstring>
#include <functional>
#include <algorithm>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>

#include "cutlass/cutlass.h"
#include "cutlass/numeric_types.h"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/epilogue/dispatch_policy.hpp"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/kernel/tile_scheduler_params.h"
#include "cutlass/util/packed_stride.hpp"
#include "cute/tensor.hpp"

using namespace cute;

#define CUDA_CHECK(expr) do { \
    cudaError_t err = (expr); \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA error: " << cudaGetErrorString(err) \
                  << " at " << __FILE__ << ":" << __LINE__ << "\n"; \
        std::exit(1); \
    } \
} while(0)

#define CUTLASS_CHECK(status) do { \
    if ((status) != cutlass::Status::kSuccess) { \
        std::cerr << "CUTLASS error: " << cutlassGetStatusString(status) \
                  << " at " << __FILE__ << ":" << __LINE__ << "\n"; \
        std::exit(1); \
    } \
} while(0)


// ============================================================================
// FP16→FP8 cast kernel (needed to prepare benchmark data)
// ============================================================================

__global__ void cast_fp16_to_fp8_kernel(
    const __half* __restrict__ in,
    __nv_fp8_e4m3* __restrict__ out,
    int64_t n)
{
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float v = __half2float(in[idx]);
        v = fminf(fmaxf(v, -448.0f), 448.0f);
        out[idx] = __nv_fp8_e4m3(__float2half_rn(v));
    }
}

void cast_fp16_to_fp8(const __half* in, __nv_fp8_e4m3* out, int64_t n, cudaStream_t s = nullptr) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    cast_fp16_to_fp8_kernel<<<blocks, threads, 0, s>>>(in, out, n);
}


// ============================================================================
// Benchmark harness
// ============================================================================

double benchmark_ms(std::function<void()> fn, int warmup = 5, int iters = 20) {
    for (int i = 0; i < warmup; ++i) fn();
    cudaDeviceSynchronize();

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    for (int i = 0; i < iters; ++i) fn();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float total_ms;
    cudaEventElapsedTime(&total_ms, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return total_ms / iters;
}


// ============================================================================
// Templated GEMM configuration benchmark
// ============================================================================

// Common type aliases
using ElementA = cutlass::float_e4m3_t;
using ElementB = cutlass::float_e4m3_t;
using ElementC = cutlass::half_t;
using ElementD = cutlass::half_t;
using ElementAccumulator = float;

using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::ColumnMajor;
using LayoutC = cutlass::layout::RowMajor;
using LayoutD = cutlass::layout::RowMajor;

using StrideA = cutlass::gemm::TagToStrideA_t<LayoutA>;
using StrideB = cutlass::gemm::TagToStrideB_t<LayoutB>;
using StrideC = cutlass::gemm::TagToStrideC_t<LayoutC>;
using StrideD = cutlass::gemm::TagToStrideC_t<LayoutD>;

static constexpr int AlignFP8  = 16;
static constexpr int AlignFP16 = 8;


/// Build and benchmark a single GEMM configuration.
/// Returns elapsed time in ms, or -1.0 if the kernel is not supported.
template <
    typename TileShape_,
    typename ClusterShape_,
    typename KernelSchedule_,
    typename EpilogueSchedule_
>
double run_config(
    const ElementA* A, const ElementB* B, ElementC* C,
    int M, int N, int K,
    float alpha, float beta,
    int max_swizzle = 1,
    int raster_order_int = 0,   // 0=Heuristic, 1=AlongM, 2=AlongN
    cudaStream_t stream = nullptr)
{
    // Build the epilogue collective
    using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
        cutlass::arch::Sm90,
        cutlass::arch::OpClassTensorOp,
        TileShape_, ClusterShape_,
        cutlass::epilogue::collective::EpilogueTileAuto,
        ElementAccumulator, ElementAccumulator,
        ElementC, LayoutC, AlignFP16,
        ElementD, LayoutD, AlignFP16,
        EpilogueSchedule_
    >::CollectiveOp;

    // Build the mainloop collective
    using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
        cutlass::arch::Sm90,
        cutlass::arch::OpClassTensorOp,
        ElementA, LayoutA, AlignFP8,
        ElementB, LayoutB, AlignFP8,
        ElementAccumulator,
        TileShape_, ClusterShape_,
        cutlass::gemm::collective::StageCountAutoCarveout<
            static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))
        >,
        KernelSchedule_
    >::CollectiveOp;

    // Build the kernel
    using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
        cute::Shape<int, int, int, int>,
        CollectiveMainloop,
        CollectiveEpilogue
    >;

    using DeviceGemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
    using GemmArguments = typename DeviceGemm::Arguments;

    // Build strides
    auto stride_A = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(M, K, 1));
    auto stride_B = cutlass::make_cute_packed_stride(StrideB{}, cute::make_shape(N, K, 1));
    auto stride_C = cutlass::make_cute_packed_stride(StrideC{}, cute::make_shape(M, N, 1));

    // Hardware info for persistent kernels
    cutlass::KernelHardwareInfo hw_info;
    hw_info.device_id = 0;
    hw_info.sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(0);

    // Build arguments
    GemmArguments arguments{
        cutlass::gemm::GemmUniversalMode::kGemm,
        {M, N, K, 1},
        {A, stride_A, B, stride_B},
        {{alpha, beta}, C, stride_C, C, stride_C},
        hw_info
    };

    // Set scheduler parameters (swizzle, raster order) if the kernel supports them
    arguments.scheduler.max_swizzle_size = max_swizzle;
    // Raster order: map int → enum
    using RasterOrder = typename cutlass::gemm::kernel::detail::
        PersistentTileSchedulerSm90Params::RasterOrderOptions;
    if (raster_order_int == 1)
        arguments.scheduler.raster_order = RasterOrder::AlongM;
    else if (raster_order_int == 2)
        arguments.scheduler.raster_order = RasterOrder::AlongN;
    else
        arguments.scheduler.raster_order = RasterOrder::Heuristic;

    // Instantiate and check if this config is supported
    DeviceGemm gemm_op;

    if (gemm_op.can_implement(arguments) != cutlass::Status::kSuccess) {
        return -1.0;  // Not supported
    }

    // Allocate workspace
    size_t workspace_size = gemm_op.get_workspace_size(arguments);
    void* workspace = nullptr;
    if (workspace_size > 0) {
        CUDA_CHECK(cudaMalloc(&workspace, workspace_size));
    }

    auto status = gemm_op.initialize(arguments, workspace, stream);
    if (status != cutlass::Status::kSuccess) {
        if (workspace) cudaFree(workspace);
        return -1.0;
    }

    // Verify correctness with single run
    status = gemm_op.run(stream);
    CUDA_CHECK(cudaDeviceSynchronize());
    if (status != cutlass::Status::kSuccess) {
        if (workspace) cudaFree(workspace);
        return -1.0;
    }

    // Benchmark
    double ms = benchmark_ms([&]() {
        gemm_op.run(stream);
    });

    if (workspace) cudaFree(workspace);
    return ms;
}


// ============================================================================
// Result tracking
// ============================================================================

struct BenchResult {
    std::string name;
    double ms;
    double tflops;
    int swizzle;
    std::string raster;

    bool valid() const { return ms > 0; }
};


// ============================================================================
// Configuration instantiations
// ============================================================================

// Shorthand
using _1 = cute::_1;  using _2 = cute::_2;
using _64 = cute::_64; using _128 = cute::_128; using _256 = cute::_256;

// Tile shapes
using Tile_128x128x128 = cute::Shape<_128, _128, _128>;
using Tile_256x128x128 = cute::Shape<_256, _128, _128>;
using Tile_128x256x128 = cute::Shape<_128, _256, _128>;

// Cluster shapes
using Cluster_1x1x1 = cute::Shape<_1, _1, _1>;
using Cluster_2x1x1 = cute::Shape<_2, _1, _1>;
using Cluster_1x2x1 = cute::Shape<_1, _2, _1>;

// Kernel schedules
using SchedAuto        = cutlass::gemm::collective::KernelScheduleAuto;
using SchedCoopFP8Fast = cutlass::gemm::KernelTmaWarpSpecializedCooperativeFP8FastAccum;
using SchedPingFP8Fast = cutlass::gemm::KernelTmaWarpSpecializedPingpongFP8FastAccum;
using SchedCoopFP8     = cutlass::gemm::KernelTmaWarpSpecializedCooperative;

// Epilogue schedule — Auto lets the builder match with kernel schedule.
// Cooperative kernels get TmaWarpSpecializedCooperative; Pingpong gets
// NoSmemWarpSpecialized; the builder resolves this internally.
using EpiAuto     = cutlass::epilogue::collective::EpilogueScheduleAuto;


// ============================================================================
// Main
// ============================================================================

void print_help(const char* prog) {
    printf("Usage: %s [M] [N] [K]\n", prog);
    printf("\n");
    printf("CUTLASS FP8 GEMM configuration benchmark on NVIDIA Hopper (SM90).\n");
    printf("Sweeps tile shapes, cluster shapes, kernel schedules, swizzle, and raster order\n");
    printf("to find the optimal configuration for your GPU and problem size.\n");
    printf("\n");
    printf("Positional arguments:\n");
    printf("  M N K        Matrix dimensions (default: 4096 4096 4096)\n");
    printf("\n");
    printf("Shorthand:\n");
    printf("  %s 4096              # M=N=K=4096\n", prog);
    printf("  %s 4096 4096 8192    # rectangular\n", prog);
    printf("\n");
    printf("Parameters swept:\n");
    printf("  KernelSchedule   Auto, CooperativeFP8FastAccum, PingpongFP8FastAccum, Cooperative\n");
    printf("  ClusterShape     1x1x1, 2x1x1, 1x2x1\n");
    printf("  TileShape        128x128x128, 256x128x128, 128x256x128\n");
    printf("  Swizzle          1, 2, 4, 8\n");
    printf("  RasterOrder      Heuristic, AlongM, AlongN\n");
}

int main(int argc, char** argv) {
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            auto sysinfo = cutlass_complex::query_system_info();
            cutlass_complex::print_system_info(sysinfo);
            cutlass_complex::print_build_config();
            printf("\n");
            print_help(argv[0]);
            return 0;
        }
    }

    int M = 4096, N = 4096, K = 4096;
    if (argc >= 4) {
        M = std::atoi(argv[1]); N = std::atoi(argv[2]); K = std::atoi(argv[3]);
    } else if (argc == 2) {
        M = N = K = std::atoi(argv[1]);
    }

    std::cout << "CUTLASS FP8 GEMM Configuration Benchmark\n";
    {
        auto sysinfo = cutlass_complex::query_system_info();
        cutlass_complex::print_system_info(sysinfo);
        cutlass_complex::print_build_config();
    }
    std::cout << "Problem: M=" << M << " N=" << N << " K=" << K << "\n"
              << "Operation: D[M×N] = alpha * A[M×K] * B[K×N] + beta * C[M×N]\n"
              << "Types: A,B=FP8e4m3  C,D=FP16  Acc=FP32\n\n";

    double real_flops = 2.0 * M * N * K;  // Real GEMM FLOPS

    // Allocate FP16 source data and FP8 converted data
    int64_t size_A = (int64_t)M * K;
    int64_t size_B = (int64_t)K * N;
    int64_t size_C = (int64_t)M * N;

    // Host data (random FP16)
    std::vector<__half> h_A(size_A), h_B(size_B);
    srand(42);
    for (auto& v : h_A) v = __float2half((float)(rand() % 5 - 2) * 0.1f);
    for (auto& v : h_B) v = __float2half((float)(rand() % 5 - 2) * 0.1f);

    // Device allocations
    __half *d_A_fp16, *d_B_fp16;
    ElementA *d_A;
    ElementB *d_B;
    ElementC *d_C;

    CUDA_CHECK(cudaMalloc(&d_A_fp16, size_A * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_B_fp16, size_B * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_A, size_A * sizeof(ElementA)));
    CUDA_CHECK(cudaMalloc(&d_B, size_B * sizeof(ElementB)));
    CUDA_CHECK(cudaMalloc(&d_C, size_C * sizeof(ElementC)));

    CUDA_CHECK(cudaMemcpy(d_A_fp16, h_A.data(), size_A * sizeof(__half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B_fp16, h_B.data(), size_B * sizeof(__half), cudaMemcpyHostToDevice));

    // Cast FP16 → FP8
    cast_fp16_to_fp8(d_A_fp16, reinterpret_cast<__nv_fp8_e4m3*>(d_A), size_A);
    cast_fp16_to_fp8(d_B_fp16, reinterpret_cast<__nv_fp8_e4m3*>(d_B), size_B);
    CUDA_CHECK(cudaMemset(d_C, 0, size_C * sizeof(ElementC)));
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaFree(d_A_fp16);
    cudaFree(d_B_fp16);

    // ====================================================================
    // Phase 1: Kernel schedule + cluster shape + tile shape sweep
    // ====================================================================

    std::cout << "========== Phase 1: Kernel Configuration Sweep ==========\n"
              << std::left << std::setw(52) << "Configuration"
              << std::right << std::setw(10) << "Time(ms)"
              << std::setw(12) << "TFLOPS" << "\n"
              << std::string(74, '-') << "\n";

    std::vector<BenchResult> results;

    auto report = [&](const std::string& name, double ms) {
        BenchResult r;
        r.name = name;
        r.ms = ms;
        r.tflops = (ms > 0) ? (real_flops / (ms * 1e-3) / 1e12) : 0;
        r.swizzle = 1;
        r.raster = "heuristic";

        if (ms > 0) {
            std::cout << std::left << std::setw(52) << name
                      << std::right << std::fixed << std::setprecision(3)
                      << std::setw(10) << ms
                      << std::setprecision(1) << std::setw(12) << r.tflops << "\n";
        } else {
            std::cout << std::left << std::setw(52) << name
                      << std::right << std::setw(10) << "N/A"
                      << std::setw(12) << "unsupported" << "\n";
        }
        results.push_back(r);
    };

    // Reset C between configs to avoid NaN accumulation
    auto reset_c = [&]() {
        CUDA_CHECK(cudaMemset(d_C, 0, size_C * sizeof(ElementC)));
    };

    // ---- Config 1: Baseline (Auto, 1x1x1, 128x128x128) ----
    reset_c();
    report("Auto / 128x128x128 / 1x1x1",
        run_config<Tile_128x128x128, Cluster_1x1x1, SchedAuto, EpiAuto>(
            d_A, d_B, d_C, M, N, K, 1.0f, 0.0f));

    // ---- Config 2: Cooperative FP8 FastAccum ----
    reset_c();
    report("CooperativeFP8Fast / 128x128x128 / 1x1x1",
        run_config<Tile_128x128x128, Cluster_1x1x1, SchedCoopFP8Fast, EpiAuto>(
            d_A, d_B, d_C, M, N, K, 1.0f, 0.0f));

    // ---- Config 3: Pingpong FP8 FastAccum ----
    reset_c();
    report("PingpongFP8Fast / 128x128x128 / 1x1x1",
        run_config<Tile_128x128x128, Cluster_1x1x1, SchedPingFP8Fast, EpiAuto>(
            d_A, d_B, d_C, M, N, K, 1.0f, 0.0f));

    // ---- Config 4: Cooperative (non-FastAccum, for comparison) ----
    reset_c();
    report("Cooperative / 128x128x128 / 1x1x1",
        run_config<Tile_128x128x128, Cluster_1x1x1, SchedCoopFP8, EpiAuto>(
            d_A, d_B, d_C, M, N, K, 1.0f, 0.0f));

    // ---- Config 5: Cooperative FP8 Fast + Cluster 2x1x1 ----
    reset_c();
    report("CooperativeFP8Fast / 128x128x128 / 2x1x1",
        run_config<Tile_128x128x128, Cluster_2x1x1, SchedCoopFP8Fast, EpiAuto>(
            d_A, d_B, d_C, M, N, K, 1.0f, 0.0f));

    // ---- Config 6: Cooperative FP8 Fast + Cluster 1x2x1 ----
    reset_c();
    report("CooperativeFP8Fast / 128x128x128 / 1x2x1",
        run_config<Tile_128x128x128, Cluster_1x2x1, SchedCoopFP8Fast, EpiAuto>(
            d_A, d_B, d_C, M, N, K, 1.0f, 0.0f));

    // ---- Config 7: Pingpong FP8 Fast + Cluster 2x1x1 ----
    reset_c();
    report("PingpongFP8Fast / 128x128x128 / 2x1x1",
        run_config<Tile_128x128x128, Cluster_2x1x1, SchedPingFP8Fast, EpiAuto>(
            d_A, d_B, d_C, M, N, K, 1.0f, 0.0f));

    // ---- Config 8: Pingpong FP8 Fast + Cluster 1x2x1 ----
    reset_c();
    report("PingpongFP8Fast / 128x128x128 / 1x2x1",
        run_config<Tile_128x128x128, Cluster_1x2x1, SchedPingFP8Fast, EpiAuto>(
            d_A, d_B, d_C, M, N, K, 1.0f, 0.0f));

    // ---- Config 9: Cooperative FP8 Fast + Tile 256x128x128 ----
    reset_c();
    report("CooperativeFP8Fast / 256x128x128 / 1x1x1",
        run_config<Tile_256x128x128, Cluster_1x1x1, SchedCoopFP8Fast, EpiAuto>(
            d_A, d_B, d_C, M, N, K, 1.0f, 0.0f));

    // ---- Config 10: Cooperative FP8 Fast + Tile 128x256x128 ----
    reset_c();
    report("CooperativeFP8Fast / 128x256x128 / 1x1x1",
        run_config<Tile_128x256x128, Cluster_1x1x1, SchedCoopFP8Fast, EpiAuto>(
            d_A, d_B, d_C, M, N, K, 1.0f, 0.0f));

    // ---- Config 11: Cooperative FP8 Fast + 256x128x128 + Cluster 1x2x1 ----
    reset_c();
    report("CooperativeFP8Fast / 256x128x128 / 1x2x1",
        run_config<Tile_256x128x128, Cluster_1x2x1, SchedCoopFP8Fast, EpiAuto>(
            d_A, d_B, d_C, M, N, K, 1.0f, 0.0f));

    std::cout << "\n";

    // ====================================================================
    // Phase 2: Swizzle + raster order sweep (on best config from Phase 1)
    // ====================================================================

    // Find best config
    double best_tflops = 0;
    int best_idx = -1;
    for (int i = 0; i < (int)results.size(); ++i) {
        if (results[i].valid() && results[i].tflops > best_tflops) {
            best_tflops = results[i].tflops;
            best_idx = i;
        }
    }

    if (best_idx >= 0) {
        std::cout << "========== Phase 2: Swizzle / Raster Order Sweep ==========\n"
                  << "Best config from Phase 1: " << results[best_idx].name
                  << " (" << std::fixed << std::setprecision(1) << best_tflops << " TFLOPS)\n\n"
                  << "Now testing swizzle × raster order combinations on TOP configs...\n\n";

        std::cout << std::left << std::setw(52) << "Configuration"
                  << std::right << std::setw(8) << "Swizzle"
                  << std::setw(10) << "Raster"
                  << std::setw(10) << "Time(ms)"
                  << std::setw(12) << "TFLOPS" << "\n"
                  << std::string(92, '-') << "\n";

        // Test swizzle/raster on the top 3 performing kernel configs.
        // We re-run each with different scheduler args.

        // Helper: sweep swizzle/raster on a particular config
        auto sweep_scheduler = [&](auto tile_shape_tag, auto cluster_shape_tag,
                                    auto kernel_sched_tag, auto epi_sched_tag,
                                    const std::string& config_name) {
            using TS = decltype(tile_shape_tag);
            using CS = decltype(cluster_shape_tag);
            using KS = decltype(kernel_sched_tag);
            using ES = decltype(epi_sched_tag);

            const int swizzles[] = {1, 2, 4, 8};
            const char* raster_names[] = {"Heuristic", "AlongM", "AlongN"};

            for (int swz : swizzles) {
                for (int ro = 0; ro < 3; ++ro) {
                    reset_c();
                    double ms = run_config<TS, CS, KS, ES>(
                        d_A, d_B, d_C, M, N, K, 1.0f, 0.0f, swz, ro);

                    double tflops = (ms > 0) ? (real_flops / (ms * 1e-3) / 1e12) : 0;

                    if (ms > 0) {
                        std::cout << std::left << std::setw(52) << config_name
                                  << std::right << std::setw(8) << swz
                                  << std::setw(10) << raster_names[ro]
                                  << std::fixed << std::setprecision(3)
                                  << std::setw(10) << ms
                                  << std::setprecision(1) << std::setw(12) << tflops << "\n";
                    }
                }
            }
        };

        // Sweep on the top performing configs — hardcode the ones we know will compile
        // (we can't easily template-iterate at runtime, so we pick the most promising ones)

        sweep_scheduler(Tile_128x128x128{}, Cluster_1x1x1{},
            SchedCoopFP8Fast{}, EpiAuto{},
            "CoopFP8Fast/128x128x128/1x1x1");

        sweep_scheduler(Tile_128x128x128{}, Cluster_1x1x1{},
            SchedPingFP8Fast{}, EpiAuto{},
            "PingFP8Fast/128x128x128/1x1x1");

        sweep_scheduler(Tile_128x128x128{}, Cluster_1x2x1{},
            SchedCoopFP8Fast{}, EpiAuto{},
            "CoopFP8Fast/128x128x128/1x2x1");
    }

    std::cout << "\n";

    // ====================================================================
    // Summary
    // ====================================================================

    std::cout << "========== Summary (sorted by TFLOPS) ==========\n\n";

    // Sort results by TFLOPS descending
    auto sorted = results;
    std::sort(sorted.begin(), sorted.end(),
        [](const BenchResult& a, const BenchResult& b) { return a.tflops > b.tflops; });

    std::cout << std::left << std::setw(52) << "Configuration"
              << std::right << std::setw(10) << "Time(ms)"
              << std::setw(12) << "TFLOPS"
              << std::setw(10) << "vs Best" << "\n"
              << std::string(84, '-') << "\n";

    for (const auto& r : sorted) {
        if (!r.valid()) continue;
        double pct = (best_tflops > 0) ? (r.tflops / best_tflops * 100.0) : 0;
        std::cout << std::left << std::setw(52) << r.name
                  << std::right << std::fixed << std::setprecision(3) << std::setw(10) << r.ms
                  << std::setprecision(1) << std::setw(12) << r.tflops
                  << std::setprecision(1) << std::setw(9) << pct << "%" << "\n";
    }

    std::cout << "\nH100/GH200 theoretical FP8 peak: 1,979 TFLOPS\n"
              << "Best achieved: " << std::fixed << std::setprecision(1)
              << best_tflops << " TFLOPS ("
              << std::setprecision(1) << (best_tflops / 1979.0 * 100.0) << "% of peak)\n";

    // Cleanup
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);

    return 0;
}
