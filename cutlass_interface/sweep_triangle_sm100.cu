/*
 * sweep_triangle_sm100.cu
 *
 * Sweeps triangle decomposition options for batched HERK to find
 * the optimal configuration. Reports actual FLOPs saved and speedup
 * over baseline.
 *
 * Usage:
 *   ./sweep_triangle_sm100 [N] [K] [batch] [graph=0|1]
 *   ./sweep_triangle_sm100 4096 4096 128
 *   ./sweep_triangle_sm100 4096 4096 128 graph=1   # include CUDA graph comparison
 */

#include "gemm_complex_sm100.hpp"
#include "system_info.hpp"
#include <iostream>
#include <iomanip>
#include <vector>
#include <random>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <algorithm>

using namespace gemm_complex_sm100;

// ============================================================================
// Helpers
// ============================================================================

void fill_random_fp16(std::vector<__half>& buf, float scale = 0.3f) {
    static std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-scale, scale);
    for (auto& v : buf) v = __float2half(dist(rng));
}

template <typename Fn>
double benchmark_ms(Fn&& fn, int warmup = 2, int iters = 5) {
    for (int i = 0; i < warmup; ++i) fn();
    cudaDeviceSynchronize();
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iters; ++i) fn();
    cudaDeviceSynchronize();
    auto t1 = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::milli>(t1 - t0).count() / iters;
}

// Compute the actual triangle FLOPs fraction for a given config.
// HERK does: 2 triangle sub-GEMMs for Re(C) + 1 full sub-GEMM for Im(C).
// Returns: (2*triangle_flops + 1*full_flops) / (3*full_flops)
// where triangle_flops = sum over slabs of M_block * N_block * K
// and   full_flops = N * N * K
double compute_flops_fraction(int N, int K, int batch_count,
                               const TriangleConfig& tri, int sm_count) {
    constexpr int kTileM = kFP8TileM;
    constexpr int kTileN = kFP8TileN;

    // Determine target_slabs (same logic as run_real_gemm_lower_triangle)
    int target_slabs;
    if (tri.target_slabs > 0) {
        target_slabs = tri.target_slabs;
    } else {
        target_slabs = 2;
        for (int T = 32; T >= 2; --T) {
            int S = ((N + T - 1) / T);
            S = ((S + kTileM - 1) / kTileM) * kTileM;
            int tiles_m = (S + kTileM - 1) / kTileM;
            int tiles_n = (S + kTileN - 1) / kTileN;
            int64_t tiles_slab0 = static_cast<int64_t>(tiles_m) * tiles_n;
            if (tiles_slab0 >= sm_count) {
                target_slabs = T;
                break;
            }
        }
    }

    // Determine min_slab_height
    int min_slab_height;
    if (tri.min_slab_height > 0) {
        min_slab_height = tri.min_slab_height;
    } else {
        double ratio = static_cast<double>(kTileM) / kTileN;
        int tiles_needed = static_cast<int>(std::ceil(
            std::sqrt(static_cast<double>(sm_count) / ratio)));
        min_slab_height = kTileM * std::max(tiles_needed, 1);
    }

    // Compute slab boundaries
    std::vector<int> boundaries;
    int num_slabs;

    if (tri.graduated) {
        int slab_height_uniform = (N + target_slabs - 1) / target_slabs;
        slab_height_uniform = ((slab_height_uniform + kTileM - 1) / kTileM) * kTileM;
        slab_height_uniform = std::max(slab_height_uniform, min_slab_height);
        num_slabs = (N + slab_height_uniform - 1) / slab_height_uniform;
        boundaries.resize(num_slabs + 1);
        boundaries[0] = 0;
        for (int i = 1; i <= num_slabs; ++i) {
            double frac = std::sqrt(static_cast<double>(i) / num_slabs);
            int b = static_cast<int>(std::round(frac * N));
            b = ((b + kTileM - 1) / kTileM) * kTileM;
            b = std::min(b, N);
            boundaries[i] = b;
        }
        boundaries[num_slabs] = N;
        std::vector<int> deduped;
        deduped.push_back(0);
        for (int i = 1; i <= num_slabs; ++i) {
            if (boundaries[i] > deduped.back()) deduped.push_back(boundaries[i]);
        }
        if (deduped.back() != N) deduped.push_back(N);
        boundaries = deduped;
        num_slabs = static_cast<int>(boundaries.size()) - 1;
    } else {
        int slab_height = (N + target_slabs - 1) / target_slabs;
        slab_height = ((slab_height + kTileM - 1) / kTileM) * kTileM;
        slab_height = std::max(slab_height, min_slab_height);
        num_slabs = (N + slab_height - 1) / slab_height;
        boundaries.resize(num_slabs + 1);
        for (int i = 0; i <= num_slabs; ++i) {
            boundaries[i] = std::min(i * slab_height, N);
        }
    }

    // Sum triangle FLOPs
    int64_t triangle_flops = 0;
    for (int i = 0; i < num_slabs; ++i) {
        int M_block = boundaries[i + 1] - boundaries[i];
        int N_block = boundaries[i + 1];
        triangle_flops += static_cast<int64_t>(M_block) * N_block * K;
    }
    int64_t full_flops = static_cast<int64_t>(N) * N * K;

    // HERK: 2 triangle + 1 full, vs baseline 3 full
    // fraction = (2*triangle + 1*full) / (3*full)
    return (2.0 * triangle_flops + 1.0 * full_flops) / (3.0 * full_flops);
}

int compute_actual_slabs(int N, int batch_count, const TriangleConfig& tri, int sm_count) {
    constexpr int kTileM = kFP8TileM;
    constexpr int kTileN = kFP8TileN;

    int target_slabs;
    if (tri.target_slabs > 0) {
        target_slabs = tri.target_slabs;
    } else {
        target_slabs = 2;
        for (int T = 32; T >= 2; --T) {
            int S = ((N + T - 1) / T);
            S = ((S + kTileM - 1) / kTileM) * kTileM;
            int tiles_m = (S + kTileM - 1) / kTileM;
            int tiles_n = (S + kTileN - 1) / kTileN;
            int64_t tiles_slab0 = static_cast<int64_t>(tiles_m) * tiles_n;
            if (tiles_slab0 >= sm_count) {
                target_slabs = T;
                break;
            }
        }
    }

    int min_slab_height;
    if (tri.min_slab_height > 0) {
        min_slab_height = tri.min_slab_height;
    } else {
        double ratio = static_cast<double>(kTileM) / kTileN;
        int tiles_needed = static_cast<int>(std::ceil(
            std::sqrt(static_cast<double>(sm_count) / ratio)));
        min_slab_height = kTileM * std::max(tiles_needed, 1);
    }

    if (tri.graduated) {
        int slab_height_uniform = (N + target_slabs - 1) / target_slabs;
        slab_height_uniform = ((slab_height_uniform + kTileM - 1) / kTileM) * kTileM;
        slab_height_uniform = std::max(slab_height_uniform, min_slab_height);
        int num_slabs = (N + slab_height_uniform - 1) / slab_height_uniform;
        // Count unique boundaries after dedup
        std::vector<int> boundaries(num_slabs + 1);
        boundaries[0] = 0;
        for (int i = 1; i <= num_slabs; ++i) {
            double frac = std::sqrt(static_cast<double>(i) / num_slabs);
            int b = static_cast<int>(std::round(frac * N));
            b = ((b + kTileM - 1) / kTileM) * kTileM;
            b = std::min(b, N);
            boundaries[i] = b;
        }
        boundaries[num_slabs] = N;
        std::vector<int> deduped;
        deduped.push_back(0);
        for (int i = 1; i <= num_slabs; ++i) {
            if (boundaries[i] > deduped.back()) deduped.push_back(boundaries[i]);
        }
        if (deduped.back() != N) deduped.push_back(N);
        return static_cast<int>(deduped.size()) - 1;
    } else {
        int slab_height = (N + target_slabs - 1) / target_slabs;
        slab_height = ((slab_height + kTileM - 1) / kTileM) * kTileM;
        slab_height = std::max(slab_height, min_slab_height);
        return (N + slab_height - 1) / slab_height;
    }
}


// ============================================================================
// Main
// ============================================================================

void print_help(const char* prog) {
    printf("Usage: %s [N] [K] [batch] [graph=0|1]\n", prog);
    printf("\n");
    printf("Sweeps triangle decomposition parameters for batched HERK on Blackwell\n");
    printf("(SM100/SM120). Reports FLOPs saved and speedup vs baseline for each\n");
    printf("slab count and spacing option.\n");
    printf("\n");
    printf("Positional arguments:\n");
    printf("  N            Matrix dimension (default: 4096)\n");
    printf("  K            Inner dimension (default: N)\n");
    printf("  batch        Batch count (default: 128)\n");
    printf("\n");
    printf("Key=value options:\n");
    printf("  graph=<0|1>  Include CUDA graph comparison sweep (default: 0)\n");
    printf("\n");
    printf("Shorthand:\n");
    printf("  %s 4096              # N=K=4096, batch=128\n", prog);
    printf("  %s 4096 128          # N=K=4096, batch=128\n", prog);
    printf("  %s 4096 4096 128     # N=4096, K=4096, batch=128\n", prog);
    printf("  %s 4096 graph=1      # include CUDA graph comparison\n", prog);
    printf("\n");
    printf("Configurations swept:\n");
    printf("  Auto-adaptive (slabs=0), uniform and graduated\n");
    printf("  Explicit slabs: 2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 24, 32\n");
    printf("  Optional: with/without CUDA graph capture (graph=1)\n");
}

bool parse_graph_flag(int argc, char** argv) {
    for (int i = 1; i < argc; ++i) {
        std::string arg(argv[i]);
        auto eq = arg.find('=');
        if (eq == std::string::npos) continue;
        std::string key = arg.substr(0, eq);
        std::string val = arg.substr(eq + 1);
        if (key == "graph") return (val == "1" || val == "true");
    }
    return false;
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

    int N = 4096, K = 4096, batch = 128;
    if (argc >= 4) {
        // Skip key=value args for positional parsing
        std::vector<int> positional;
        for (int i = 1; i < argc; ++i) {
            std::string arg(argv[i]);
            if (arg.find('=') == std::string::npos) {
                positional.push_back(std::atoi(argv[i]));
            }
        }
        if (positional.size() >= 3) {
            N = positional[0]; K = positional[1]; batch = positional[2];
        } else if (positional.size() >= 2) {
            N = K = positional[0]; batch = positional[1];
        } else if (positional.size() >= 1) {
            N = K = positional[0];
        }
    } else if (argc >= 2) {
        N = K = std::atoi(argv[1]);
        if (argc >= 3) batch = std::atoi(argv[2]);
    }
    bool sweep_graph = parse_graph_flag(argc, argv);

    auto sysinfo = cutlass_complex::query_system_info();
    int sm_count = sysinfo.sm_count;

    printf("=== Triangle Decomposition Sweep ===\n");
    cutlass_complex::print_system_info(sysinfo);
    cutlass_complex::print_build_config();
    printf("Problem: N=%d, K=%d, batch=%d\n", N, K, batch);
    printf("Tile: %dx%d\n", kFP8TileM, kFP8TileN);
    printf("\n");

    // Allocate data
    int64_t per_A = static_cast<int64_t>(N) * K;
    int64_t packed_elems = static_cast<int64_t>(N) * (N + 1) / 2;

    size_t bytes_A = 2 * per_A * batch * sizeof(__half);
    size_t bytes_C = 2 * packed_elems * batch * sizeof(__half);

    printf("Memory: A=%.1f GB, C=%.1f GB\n",
           bytes_A / 1e9, bytes_C / 1e9);

    // Allocate host data for one A matrix, then replicate on device
    std::vector<__half> h_A(2 * per_A);
    fill_random_fp16(h_A, 0.3f);

    __half *d_A, *d_C;
    cudaMalloc(&d_A, bytes_A);
    cudaMalloc(&d_C, bytes_C);

    // Fill all batches with same data
    {
        __half* d_tmp;
        cudaMalloc(&d_tmp, 2 * per_A * sizeof(__half));
        cudaMemcpy(d_tmp, h_A.data(), 2 * per_A * sizeof(__half), cudaMemcpyHostToDevice);
        for (int b = 0; b < batch; ++b) {
            cudaMemcpy(d_A + b * 2 * per_A, d_tmp,
                       2 * per_A * sizeof(__half), cudaMemcpyDeviceToDevice);
        }
        cudaFree(d_tmp);
    }

    GemmComplexSm100 gemm;

    // Baseline FLOPs: 3 full sub-GEMMs * batch
    double baseline_gflops = 3.0 * batch * 2.0 * N * N * K / 1e9;

    // ---- Baseline ----
    printf("Running baseline...\n");
    {
        CutlassParams bp;
        bp.herk_strategy = HerkStrategy::Baseline;
        double ms = benchmark_ms([&]() {
            gemm.HERK_batched(d_A, d_C, N, K, batch,
                              1.0f, 0.0f, HerkOp::NoTrans, FillMode::Lower, bp);
        });
        double tflops = baseline_gflops / ms;
        printf("\n  BASELINE:  %8.2f ms  %6.1f TFLOPS (3 full sub-GEMMs x %d)\n\n",
               ms, tflops, batch);
    }

    double baseline_ms;
    {
        CutlassParams bp;
        bp.herk_strategy = HerkStrategy::Baseline;
        baseline_ms = benchmark_ms([&]() {
            gemm.HERK_batched(d_A, d_C, N, K, batch,
                              1.0f, 0.0f, HerkOp::NoTrans, FillMode::Lower, bp);
        });
    }

    // ---- Sweep ----
    printf("%-6s %-5s %8s %10s %10s %8s %8s\n",
           "Slabs", "Grad", "Actual", "FLOPs%%", "Time(ms)", "TFLOPS", "Speedup");
    printf("%-6s %-5s %8s %10s %10s %8s %8s\n",
           "-----", "-----", "--------", "----------", "----------", "--------", "--------");

    struct Result {
        int slabs;
        bool graduated;
        int actual_slabs;
        double flops_pct;
        double ms;
        double tflops;
        double speedup;
    };
    std::vector<Result> results;

    auto run_config = [&](int slabs, bool graduated) {
        TriangleConfig tri;
        tri.target_slabs = slabs;
        tri.graduated = graduated;
        tri.verbose = false;

        double flops_frac = compute_flops_fraction(N, K, batch, tri, sm_count);
        int actual = compute_actual_slabs(N, batch, tri, sm_count);

        CutlassParams tp;
        tp.herk_strategy = HerkStrategy::TriangleAware;
        tp.triangle_config = tri;

        double ms = benchmark_ms([&]() {
            gemm.HERK_batched(d_A, d_C, N, K, batch,
                              1.0f, 0.0f, HerkOp::NoTrans, FillMode::Lower, tp);
        });

        // Report TFLOPS based on actual work done (baseline FLOPs as reference)
        double tflops = baseline_gflops / ms;
        double speedup = baseline_ms / ms;

        Result r{slabs, graduated, actual, flops_frac * 100.0, ms, tflops, speedup};
        results.push_back(r);

        printf("%-6d %-5s %8d %9.1f%% %10.2f %8.1f %7.2fx\n",
               slabs, graduated ? "yes" : "no",
               actual, r.flops_pct, ms, tflops, speedup);
    };

    // Auto-adaptive
    printf("\n--- Auto-adaptive (slabs=0) ---\n");
    {
        TriangleConfig tri;
        tri.verbose = false;

        double flops_frac = compute_flops_fraction(N, K, batch, tri, sm_count);
        int actual = compute_actual_slabs(N, batch, tri, sm_count);

        CutlassParams tp;
        tp.herk_strategy = HerkStrategy::TriangleAware;
        tp.triangle_config = tri;

        double ms = benchmark_ms([&]() {
            gemm.HERK_batched(d_A, d_C, N, K, batch,
                              1.0f, 0.0f, HerkOp::NoTrans, FillMode::Lower, tp);
        });

        double tflops = baseline_gflops / ms;
        double speedup = baseline_ms / ms;

        printf("%-6s %-5s %8d %9.1f%% %10.2f %8.1f %7.2fx\n",
               "auto", "no", actual, flops_frac * 100.0, ms, tflops, speedup);

        // Graduated auto
        tri.graduated = true;
        flops_frac = compute_flops_fraction(N, K, batch, tri, sm_count);
        actual = compute_actual_slabs(N, batch, tri, sm_count);

        tp.triangle_config = tri;
        ms = benchmark_ms([&]() {
            gemm.HERK_batched(d_A, d_C, N, K, batch,
                              1.0f, 0.0f, HerkOp::NoTrans, FillMode::Lower, tp);
        });
        tflops = baseline_gflops / ms;
        speedup = baseline_ms / ms;

        printf("%-6s %-5s %8d %9.1f%% %10.2f %8.1f %7.2fx\n",
               "auto", "yes", actual, flops_frac * 100.0, ms, tflops, speedup);
    }

    // Explicit slab counts: uniform
    printf("\n--- Uniform spacing ---\n");
    for (int s : {2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 24, 32}) {
        run_config(s, false);
    }

    // Explicit slab counts: graduated
    printf("\n--- Graduated spacing ---\n");
    for (int s : {2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 24, 32}) {
        run_config(s, true);
    }

    // ---- CUDA Graph comparison (SM120 only, opt-in) ----
    if (sweep_graph) {
        printf("\n--- CUDA Graph comparison (graph=1) ---\n");
        printf("%-6s %-5s %-5s %8s %10s %10s %8s\n",
               "Slabs", "Grad", "Graph", "Actual", "Time(ms)", "vs NoGr", "Speedup");
        printf("%-6s %-5s %-5s %8s %10s %10s %8s\n",
               "-----", "-----", "-----", "--------", "----------", "----------", "--------");

        // Test auto-adaptive with and without graph
        for (bool use_graph : {false, true}) {
            TriangleConfig tri;
            tri.verbose = false;
            tri.use_cuda_graph = use_graph;

            int actual = compute_actual_slabs(N, batch, tri, sm_count);

            CutlassParams tp;
            tp.herk_strategy = HerkStrategy::TriangleAware;
            tp.triangle_config = tri;

            double ms = benchmark_ms([&]() {
                gemm.HERK_batched(d_A, d_C, N, K, batch,
                                  1.0f, 0.0f, HerkOp::NoTrans, FillMode::Lower, tp);
            });

            double speedup = baseline_ms / ms;
            printf("%-6s %-5s %-5s %8d %10.2f %10s %7.2fx\n",
                   "auto", "no", use_graph ? "yes" : "no",
                   actual, ms, use_graph ? "" : "(ref)", speedup);
        }

        // Test a few explicit slab counts with graph
        for (int s : {2, 5, 8, 16, 32}) {
            double no_graph_ms = 0, graph_ms = 0;
            int actual = 0;

            for (bool use_graph : {false, true}) {
                TriangleConfig tri;
                tri.target_slabs = s;
                tri.verbose = false;
                tri.use_cuda_graph = use_graph;

                actual = compute_actual_slabs(N, batch, tri, sm_count);

                CutlassParams tp;
                tp.herk_strategy = HerkStrategy::TriangleAware;
                tp.triangle_config = tri;

                double ms = benchmark_ms([&]() {
                    gemm.HERK_batched(d_A, d_C, N, K, batch,
                                      1.0f, 0.0f, HerkOp::NoTrans, FillMode::Lower, tp);
                });

                if (!use_graph) no_graph_ms = ms;
                else graph_ms = ms;

                double speedup = baseline_ms / ms;
                char vs_buf[32] = "(ref)";
                if (use_graph && no_graph_ms > 0) {
                    snprintf(vs_buf, sizeof(vs_buf), "%.2fx", no_graph_ms / graph_ms);
                }
                printf("%-6d %-5s %-5s %8d %10.2f %10s %7.2fx\n",
                       s, "no", use_graph ? "yes" : "no",
                       actual, ms, vs_buf, speedup);
            }
        }
    }

    // ---- Find best ----
    printf("\n=== Summary ===\n");
    printf("Baseline: %.2f ms\n", baseline_ms);

    // Find best uniform and graduated
    double best_uniform_ms = 1e20, best_grad_ms = 1e20;
    int best_uniform_s = 0, best_grad_s = 0;
    for (auto& r : results) {
        if (!r.graduated && r.ms < best_uniform_ms) {
            best_uniform_ms = r.ms;
            best_uniform_s = r.slabs;
        }
        if (r.graduated && r.ms < best_grad_ms) {
            best_grad_ms = r.ms;
            best_grad_s = r.slabs;
        }
    }

    printf("Best uniform:   slabs=%d  %.2f ms  %.2fx speedup  (%.1f%% FLOPs)\n",
           best_uniform_s, best_uniform_ms, baseline_ms / best_uniform_ms,
           compute_flops_fraction(N, K, batch, {best_uniform_s, 0, false, false}, sm_count) * 100.0);
    printf("Best graduated: slabs=%d  %.2f ms  %.2fx speedup  (%.1f%% FLOPs)\n",
           best_grad_s, best_grad_ms, baseline_ms / best_grad_ms,
           compute_flops_fraction(N, K, batch, {best_grad_s, 0, true, false}, sm_count) * 100.0);

    double overall_best_ms = std::min(best_uniform_ms, best_grad_ms);
    bool overall_best_grad = (best_grad_ms < best_uniform_ms);
    int overall_best_s = overall_best_grad ? best_grad_s : best_uniform_s;
    double flops_saved = (1.0 - compute_flops_fraction(N, K, batch,
        {overall_best_s, 0, overall_best_grad, false}, sm_count)) * 100.0;

    printf("\nBest overall: slabs=%d graduated=%s\n",
           overall_best_s, overall_best_grad ? "yes" : "no");
    printf("  Time:    %.2f ms (vs %.2f ms baseline)\n", overall_best_ms, baseline_ms);
    printf("  Speedup: %.2fx\n", baseline_ms / overall_best_ms);
    printf("  FLOPs saved: %.1f%%\n", flops_saved);

    printf("\nCommand-line for best config:\n");
    printf("  ./example_complex_sm100 %d %d %d %d slabs=%d%s\n",
           N, N, K, batch, overall_best_s,
           overall_best_grad ? " graduated=1" : "");

    cudaFree(d_A);
    cudaFree(d_C);
    return 0;
}
