#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <vector>
#include <random>
#include <numeric>
#include <inttypes.h>

#include <test.h>
#include <misc.h>
#include <ggp.h>
#include <visibility_pipeline.h>

// if "--enable-testing true" is passed, we run the tests defined in here
#include <cutlass_cublas_benchmark_gtest.hpp>

namespace quda {
  extern void setTransferGPU(bool);
}

// Forward declarations from visibility_bf_kernels.cu / cutlass_interface.cu
namespace ggp {
  int herkBatchedCutlassQC(const void* qc_data, void* tri_output,
                           int N, int K, int batch, cudaStream_t stream = 0);
  void promoteQcSmToFp32(void *output, const void *input,
                         unsigned long long int N, int stream_idx);
}

// Forward declaration from visibility_bf_kernels.cu
namespace ggp {
  int cublas_herk_batched_qc(const float* promoted_data, float* result_data,
                             float* tri_output, int N, int K, int batch,
                             cudaStream_t stream);
  // QC sign-magnitude -> TCC two's complement nibble conversion (in-place)
  void convertQcSmToFtd(void *data, unsigned long long int N, int stream_idx);
}

// Forward declaration from tcc_interface.cu
namespace quda {
  void correlateTCC(void *tri_output, const void *raw_input,
                    unsigned n_ant, unsigned n_chan, unsigned n_time,
                    unsigned n_time_inner, unsigned n_pol, int stream_idx);
}

// ============================================================================
// CPU reference: decode QC byte to complex float
// ============================================================================
static inline void decode_qc_byte(uint8_t byte, float &re, float &im) {
  uint8_t re_raw = (byte >> 4) & 0x0F;
  int re_sign = (re_raw & 0x8) ? -1 : 1;
  int re_mag  = re_raw & 0x7;
  re = static_cast<float>(re_sign * re_mag);

  uint8_t im_raw = byte & 0x0F;
  int im_sign = (im_raw & 0x8) ? -1 : 1;
  int im_mag  = im_raw & 0x7;
  im = static_cast<float>(im_sign * im_mag);
}

// CPU HERK reference: compute C = A * A^H for one batch
// A is [N x K] complex, row-major: A[antenna][time] = h_qc[batch_offset + antenna*K + time]
// C[i][j] = sum_k A[i][k] * conj(A[j][k])
// Output: packed lower triangle [n_baselines x 2] floats
static void cpu_herk_one_batch(
    const uint8_t* qc_bytes, int batch_offset,
    float* tri_out,
    int N, int K)
{
  // Promote input to FP32 complex for this batch
  std::vector<float> Ar(N * K), Ai(N * K);
  for (int n = 0; n < N; n++) {
    for (int k = 0; k < K; k++) {
      int idx = batch_offset + n * K + k;
      decode_qc_byte(qc_bytes[idx], Ar[n * K + k], Ai[n * K + k]);
    }
  }

  // Compute lower triangle of C = A * A^H
  int tri_idx = 0;
  for (int i = 0; i < N; i++) {
    for (int j = 0; j <= i; j++) {
      double c_re = 0.0, c_im = 0.0;
      for (int k = 0; k < K; k++) {
        // A[i][k] * conj(A[j][k])
        double ai_re = Ar[i * K + k], ai_im = Ai[i * K + k];
        double aj_re = Ar[j * K + k], aj_im = Ai[j * K + k];
        c_re += ai_re * aj_re + ai_im * aj_im;
        c_im += ai_im * aj_re - ai_re * aj_im;
      }
      tri_out[tri_idx * 2]     = static_cast<float>(c_re);
      tri_out[tri_idx * 2 + 1] = static_cast<float>(c_im);
      tri_idx++;
    }
  }
}

// FTD-aware CPU HERK reference: data in [time][antenna][batch_element] order
// For batch b, antenna a, time k: byte at k * (N * total_batch) + a * total_batch + b
// This matches how TCC (via the reorder kernel) interprets the data.
static void cpu_herk_one_batch_ftd(
    const uint8_t* qc_bytes, int batch_idx, int total_batch,
    float* tri_out,
    int N, int K)
{
  // Promote input to FP32 complex for this batch (FTD layout)
  std::vector<float> Ar(N * K), Ai(N * K);
  for (int n = 0; n < N; n++) {
    for (int k = 0; k < K; k++) {
      int idx = k * (N * total_batch) + n * total_batch + batch_idx;
      decode_qc_byte(qc_bytes[idx], Ar[n * K + k], Ai[n * K + k]);
    }
  }

  // Compute lower triangle of C = A * A^H
  int tri_idx = 0;
  for (int i = 0; i < N; i++) {
    for (int j = 0; j <= i; j++) {
      double c_re = 0.0, c_im = 0.0;
      for (int k = 0; k < K; k++) {
        double ai_re = Ar[i * K + k], ai_im = Ai[i * K + k];
        double aj_re = Ar[j * K + k], aj_im = Ai[j * K + k];
        c_re += ai_re * aj_re + ai_im * aj_im;
        c_im += ai_im * aj_re - ai_re * aj_im;
      }
      tri_out[tri_idx * 2]     = static_cast<float>(c_re);
      tri_out[tri_idx * 2 + 1] = static_cast<float>(c_im);
      tri_idx++;
    }
  }
}

// Run correctness verification with CPU reference (single small config)
// Returns the max relative error across cuBLAS/CUTLASS vs CPU (all batches).
static double run_verification(int N, int K, int batch) {
  printfQuda("\n======== CPU Reference Verification (N=%d, K=%d, batch=%d) ========\n", N, K, batch);

  const uint64_t in_size = (uint64_t)batch * N * K;
  const uint64_t n_baselines = (uint64_t)(N + 1) * N / 2;
  const uint64_t tri_elems = (uint64_t)batch * n_baselines * 2;
  const uint64_t tri_bytes = tri_elems * sizeof(float);
  const uint64_t promoted_bytes = in_size * 2 * sizeof(float);
  const uint64_t full_bytes = (uint64_t)batch * N * N * 2 * sizeof(float);

  // Random input
  std::vector<uint8_t> h_input(in_size);
  std::mt19937 rng(42);
  for (auto &val : h_input) val = rng() & 0xFF;

  // GPU paths
  void *d_qc = nullptr;
  float *d_promoted = nullptr, *d_full = nullptr;
  float *d_tri_cublas = nullptr, *d_tri_cutlass = nullptr;
  cudaMalloc(&d_qc, in_size);
  cudaMalloc(&d_promoted, promoted_bytes);
  cudaMalloc(&d_full, full_bytes);
  cudaMalloc(&d_tri_cublas, tri_bytes);
  cudaMalloc(&d_tri_cutlass, tri_bytes);
  cudaMemcpy(d_qc, h_input.data(), in_size, cudaMemcpyHostToDevice);

  // cuBLAS path
  ggp::promoteQcSmToFp32(d_promoted, d_qc, in_size, 0);
  ggp::cublas_herk_batched_qc(d_promoted, d_full, d_tri_cublas, N, K, batch, nullptr);
  cudaDeviceSynchronize();

  // CUTLASS path
  ggp::herkBatchedCutlassQC(d_qc, d_tri_cutlass, N, K, batch);
  cudaDeviceSynchronize();

  // Download GPU results
  std::vector<float> h_cublas(tri_elems), h_cutlass(tri_elems);
  cudaMemcpy(h_cublas.data(), d_tri_cublas, tri_bytes, cudaMemcpyDeviceToHost);
  cudaMemcpy(h_cutlass.data(), d_tri_cutlass, tri_bytes, cudaMemcpyDeviceToHost);

  // Compare ALL batches against CPU reference (OpenMP-parallelized)
  double overall_max_err = 0.0;
  auto compare_all_batches = [&](const char* label, const std::vector<float>& gpu) -> double {
    double max_rel = 0.0;
    int total_mismatch = 0;
    int worst_batch = -1;
    int worst_idx = -1;

    #pragma omp parallel
    {
      double local_max = 0.0;
      int local_mis = 0;
      int local_wb = -1, local_wi = -1;

      #pragma omp for schedule(dynamic)
      for (int b = 0; b < batch; b++) {
        std::vector<float> cpu_tri(n_baselines * 2);
        cpu_herk_one_batch(h_input.data(), b * N * K, cpu_tri.data(), N, K);

        const float *gpu_batch = gpu.data() + (uint64_t)b * n_baselines * 2;
        for (uint64_t i = 0; i < n_baselines * 2; i++) {
          double ref = cpu_tri[i];
          double test = gpu_batch[i];
          if (fabs(ref) > 1e-6) {
            double rel = fabs(ref - test) / fabs(ref);
            if (rel > local_max) {
              local_max = rel;
              local_wb = b;
              local_wi = (int)i;
            }
            if (rel > 0.01) local_mis++;
          } else if (fabs(test) > 1e-6) {
            local_mis++;
          }
        }
      }

      #pragma omp critical
      {
        if (local_max > max_rel) {
          max_rel = local_max;
          worst_batch = local_wb;
          worst_idx = local_wi;
        }
        total_mismatch += local_mis;
      }
    }

    printfQuda("  %s vs CPU (all %d batches): max_rel_err=%e  mismatches=%d/%llu",
               label, batch, max_rel, total_mismatch,
               (unsigned long long)(n_baselines * 2 * batch));
    if (worst_idx >= 0) {
      // Recompute worst batch CPU ref for diagnostic printing
      std::vector<float> worst_cpu(n_baselines * 2);
      cpu_herk_one_batch(h_input.data(), worst_batch * N * K, worst_cpu.data(), N, K);
      int tri_k = worst_idx / 2;
      int component = worst_idx % 2;
      int row = (int)((sqrt(8.0 * tri_k + 1.0) - 1.0) * 0.5);
      int col = tri_k - row * (row + 1) / 2;
      printfQuda("  worst at batch %d tri[%d,%d].%s: cpu=%f gpu=%f",
                 worst_batch, row, col, component ? "im" : "re",
                 worst_cpu[worst_idx],
                 gpu[(uint64_t)worst_batch * n_baselines * 2 + worst_idx]);
    }
    printfQuda("\n");
    return max_rel;
  };

  overall_max_err = fmax(overall_max_err, compare_all_batches("cuBLAS", h_cublas));
  overall_max_err = fmax(overall_max_err, compare_all_batches("CUTLASS", h_cutlass));

  // TCC path (only if K is TCC-aligned and batch is decomposable)
  // TCC requires n_pol=2 and K % 128 == 0
  bool tcc_aligned = (K % 128 == 0) && (batch % 2 == 0);
  std::vector<float> h_tcc;
  if (tcc_aligned) {
    unsigned tcc_n_chan = batch / 2;  // batch = n_chan * n_pol, n_pol=2
    unsigned tcc_n_pol = 2;

    // TCC needs the data in FTD nibble convention: convert a copy
    void *d_qc_tcc = nullptr;
    float *d_tri_tcc = nullptr;
    cudaMalloc(&d_qc_tcc, in_size);
    cudaMalloc(&d_tri_tcc, tri_bytes);
    cudaMemcpy(d_qc_tcc, d_qc, in_size, cudaMemcpyDeviceToDevice);

    // Convert sign-magnitude -> two's complement nibbles
    ggp::convertQcSmToFtd(d_qc_tcc, in_size, 0);

    // Run TCC: n_time=K, n_time_inner=1 (integrate all samples)
    quda::correlateTCC(d_tri_tcc, d_qc_tcc,
                        static_cast<unsigned>(N),
                        tcc_n_chan,
                        static_cast<unsigned>(K),
                        /*n_time_inner=*/1,
                        tcc_n_pol,
                        0);
    cudaDeviceSynchronize();

    h_tcc.resize(tri_elems);
    cudaMemcpy(h_tcc.data(), d_tri_tcc, tri_bytes, cudaMemcpyDeviceToHost);

    // TCC uses FTD layout [time][antenna][channel][pol], which is different
    // from the batch-first layout [batch][antenna][K] used by cuBLAS/CUTLASS.
    // Compare TCC against an FTD-aware CPU reference for ALL batches (OpenMP).
    // Note: convertQcSmToFtd converts the nibble encoding but not the memory
    // layout. The CPU reference must use the pre-conversion (sign-magnitude)
    // data with the FTD indexing pattern.
    {
      double tcc_max_rel = 0.0;
      int tcc_mismatch = 0;
      int tcc_worst_batch = -1;
      int tcc_worst_idx = -1;

      #pragma omp parallel
      {
        double local_max = 0.0;
        int local_mis = 0;
        int local_wb = -1, local_wi = -1;

        #pragma omp for schedule(dynamic)
        for (int b = 0; b < batch; b++) {
          std::vector<float> h_ftd_tri(n_baselines * 2);
          cpu_herk_one_batch_ftd(h_input.data(), b, batch,
                                 h_ftd_tri.data(), N, K);

          const float *tcc_batch = h_tcc.data() + (uint64_t)b * n_baselines * 2;
          for (uint64_t i = 0; i < n_baselines * 2; i++) {
            double ref = h_ftd_tri[i];
            double test = tcc_batch[i];
            if (fabs(ref) > 1e-6) {
              double rel = fabs(ref - test) / fabs(ref);
              if (rel > local_max) {
                local_max = rel;
                local_wb = b;
                local_wi = (int)i;
              }
              if (rel > 0.01) local_mis++;
            } else if (fabs(test) > 1e-6) {
              local_mis++;
            }
          }
        }

        #pragma omp critical
        {
          if (local_max > tcc_max_rel) {
            tcc_max_rel = local_max;
            tcc_worst_batch = local_wb;
            tcc_worst_idx = local_wi;
          }
          tcc_mismatch += local_mis;
        }
      }

      printfQuda("  TCC vs FTD-CPU (all %d batches): max_rel_err=%e  mismatches=%d/%llu",
                 batch, tcc_max_rel, tcc_mismatch,
                 (unsigned long long)(n_baselines * 2 * batch));
      if (tcc_worst_idx >= 0) {
        // Recompute worst batch CPU ref for diagnostic printing
        std::vector<float> worst_cpu(n_baselines * 2);
        cpu_herk_one_batch_ftd(h_input.data(), tcc_worst_batch, batch,
                               worst_cpu.data(), N, K);
        int tri_k = tcc_worst_idx / 2;
        int component = tcc_worst_idx % 2;
        int row = (int)((sqrt(8.0 * tri_k + 1.0) - 1.0) * 0.5);
        int col = tri_k - row * (row + 1) / 2;
        printfQuda("  worst at batch %d tri[%d,%d].%s: ftd_cpu=%f tcc=%f",
                   tcc_worst_batch, row, col, component ? "im" : "re",
                   worst_cpu[tcc_worst_idx],
                   h_tcc[(uint64_t)tcc_worst_batch * n_baselines * 2 + tcc_worst_idx]);
      }
      printfQuda("\n");
    }

    cudaFree(d_qc_tcc);
    cudaFree(d_tri_tcc);
  } else {
    printfQuda("  TCC: SKIPPED (K=%d, need K%%128==0 and even batch)\n", K);
  }

  // Also compare cuBLAS vs CUTLASS directly (all batches)
  {
    double max_rel = 0.0;
    for (uint64_t i = 0; i < tri_elems; i++) {
      double a = h_cublas[i], b = h_cutlass[i];
      if (fabs(a) > 1e-6) {
        double rel = fabs(a - b) / fabs(a);
        if (rel > max_rel) max_rel = rel;
      }
    }
    printfQuda("  cuBLAS vs CUTLASS (all %d batches): max_rel_err=%e\n", batch, max_rel);
  }

  cudaFree(d_qc);
  cudaFree(d_promoted);
  cudaFree(d_full);
  cudaFree(d_tri_cublas);
  cudaFree(d_tri_cutlass);

  return overall_max_err;
}

// ============================================================================
// GTest wrapper functions for verification tests
// ============================================================================
double VerificationTest_Small()  { return run_verification(32, 8, 4); }
double VerificationTest_Medium() { return run_verification(64, 8, 4); }
double VerificationTest_Large()  { return run_verification(128, 8, 4); }
double VerificationTest_LargeK() { return run_verification(256, 32, 4); }
double VerificationTest_TCC()    { return run_verification(32, 128, 4); }

// ============================================================================
// Benchmark parameters
// ============================================================================
struct BenchConfig {
  int n_antennae;
  int n_channels;
  int n_time;
  int n_time_inner;
  int n_polarizations;
  int n_warmup;
  int n_iters;
};

// ============================================================================
// Run a single benchmark configuration
// ============================================================================
struct BenchResult {
  double cublas_ms;
  double cutlass_ms;
  double tcc_ms;           // -1.0 if TCC not available (K not aligned)
  double cutlass_speedup;  // cuBLAS / CUTLASS
  double tcc_vs_cutlass;   // TCC / CUTLASS (-1.0 if N/A)
};

BenchResult run_benchmark(const BenchConfig& cfg) {
  const int N = cfg.n_antennae;
  const int K = cfg.n_time / cfg.n_time_inner;
  const int batch = cfg.n_channels * cfg.n_polarizations * cfg.n_time_inner;

  // TCC alignment check: K must be a multiple of 128 (nrTimesPerBlock for i4)
  const bool tcc_available = (K % 128 == 0);

  // Input size: QC format = 1 byte per complex element
  const uint64_t in_size = (uint64_t)N * cfg.n_channels * cfg.n_time * cfg.n_polarizations;
  // Triangle output: batch * n_baselines * 2 * sizeof(float)
  const uint64_t n_baselines = (uint64_t)(N + 1) * N / 2;
  const uint64_t tri_elems = (uint64_t)batch * n_baselines * 2;
  const uint64_t tri_bytes = tri_elems * sizeof(float);
  // Full NxN for cuBLAS
  const uint64_t full_elems = (uint64_t)batch * N * N * 2;
  const uint64_t full_bytes = full_elems * sizeof(float);
  // Promoted data
  const uint64_t promoted_elems = in_size * 2;  // re,im per QC element
  const uint64_t promoted_bytes = promoted_elems * sizeof(float);

  printfQuda("\n--- N=%d, Nf=%d, T=%d, Ti=%d, batch=%d, K=%d ---\n",
             N, cfg.n_channels, cfg.n_time, cfg.n_time_inner, batch, K);
  printfQuda("    Input:     %.2f MB (QC bytes)\n", in_size / 1e6);
  printfQuda("    Promoted:  %.2f MB (FP32 complex)\n", promoted_bytes / 1e6);
  printfQuda("    Full NxN:  %.2f MB (FP32 complex)\n", full_bytes / 1e6);
  printfQuda("    Triangle:  %.2f MB (FP32 complex)\n", tri_bytes / 1e6);
  if (!tcc_available) {
    printfQuda("    TCC:       N/A (K=%d not a multiple of 128)\n", K);
  }

  // Allocate device memory
  void *d_qc_input = nullptr;
  float *d_promoted = nullptr;
  float *d_full_result = nullptr;
  float *d_tri_cublas = nullptr;
  float *d_tri_cutlass = nullptr;

  cudaMalloc(&d_qc_input, in_size);
  cudaMalloc(&d_promoted, promoted_bytes);
  cudaMalloc(&d_full_result, full_bytes);
  cudaMalloc(&d_tri_cublas, tri_bytes);
  cudaMalloc(&d_tri_cutlass, tri_bytes);

  // TCC buffers (only if TCC is available)
  void *d_input_tcc = nullptr;
  float *d_tri_tcc = nullptr;
  if (tcc_available) {
    cudaMalloc(&d_input_tcc, in_size);
    cudaMalloc(&d_tri_tcc, tri_bytes);
  }

  // Fill input with random data
  std::vector<unsigned char> h_input(in_size);
  std::mt19937 rng(42);
  for (auto &val : h_input) val = rng() & 0xFF;
  cudaMemcpy(d_qc_input, h_input.data(), in_size, cudaMemcpyHostToDevice);

  // CUDA events for timing
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  // ======================================================================
  // cuBLAS path: promoteData + cuBLAS GEMM + triangulate
  // ======================================================================
  // Warm up
  for (int i = 0; i < cfg.n_warmup; i++) {
    ggp::promoteQcSmToFp32(d_promoted, d_qc_input, in_size, 0);
    ggp::cublas_herk_batched_qc(d_promoted, d_full_result, d_tri_cublas, N, K, batch, nullptr);
  }
  cudaDeviceSynchronize();

  // Timed iterations
  cudaEventRecord(start);
  for (int i = 0; i < cfg.n_iters; i++) {
    ggp::promoteQcSmToFp32(d_promoted, d_qc_input, in_size, 0);
    ggp::cublas_herk_batched_qc(d_promoted, d_full_result, d_tri_cublas, N, K, batch, nullptr);
  }
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);

  float cublas_total_ms = 0.0f;
  cudaEventElapsedTime(&cublas_total_ms, start, stop);
  double cublas_avg_ms = cublas_total_ms / cfg.n_iters;

  // ======================================================================
  // CUTLASS path: raw QC -> packed triangle in one call
  // ======================================================================
  // Warm up
  for (int i = 0; i < cfg.n_warmup; i++) {
    ggp::herkBatchedCutlassQC(d_qc_input, d_tri_cutlass, N, K, batch);
  }
  cudaDeviceSynchronize();

  // Timed iterations
  cudaEventRecord(start);
  for (int i = 0; i < cfg.n_iters; i++) {
    ggp::herkBatchedCutlassQC(d_qc_input, d_tri_cutlass, N, K, batch);
  }
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);

  float cutlass_total_ms = 0.0f;
  cudaEventElapsedTime(&cutlass_total_ms, start, stop);
  double cutlass_avg_ms = cutlass_total_ms / cfg.n_iters;

  // ======================================================================
  // TCC path: convert nibbles + correlateTCC
  // ======================================================================
  double tcc_avg_ms = -1.0;
  if (tcc_available) {
    // Warm up
    for (int i = 0; i < cfg.n_warmup; i++) {
      cudaMemcpy(d_input_tcc, d_qc_input, in_size, cudaMemcpyDeviceToDevice);
      ggp::convertQcSmToFtd(d_input_tcc, in_size, 0);
      quda::correlateTCC(d_tri_tcc, d_input_tcc,
                          static_cast<unsigned>(N),
                          static_cast<unsigned>(cfg.n_channels),
                          static_cast<unsigned>(cfg.n_time),
                          static_cast<unsigned>(cfg.n_time_inner),
                          static_cast<unsigned>(cfg.n_polarizations), 0);
    }
    cudaDeviceSynchronize();

    // Timed iterations
    cudaEventRecord(start);
    for (int i = 0; i < cfg.n_iters; i++) {
      cudaMemcpy(d_input_tcc, d_qc_input, in_size, cudaMemcpyDeviceToDevice);
      ggp::convertQcSmToFtd(d_input_tcc, in_size, 0);
      quda::correlateTCC(d_tri_tcc, d_input_tcc,
                          static_cast<unsigned>(N),
                          static_cast<unsigned>(cfg.n_channels),
                          static_cast<unsigned>(cfg.n_time),
                          static_cast<unsigned>(cfg.n_time_inner),
                          static_cast<unsigned>(cfg.n_polarizations), 0);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float tcc_total_ms = 0.0f;
    cudaEventElapsedTime(&tcc_total_ms, start, stop);
    tcc_avg_ms = tcc_total_ms / cfg.n_iters;
  }

  double cutlass_speedup = cublas_avg_ms / cutlass_avg_ms;
  double tcc_vs_cutlass = tcc_available ? (tcc_avg_ms / cutlass_avg_ms) : -1.0;

  // Compute TFLOPS (complex HERK: 8*M*N*K flops per batch element)
  double flops = (double)batch * 8.0 * N * N * K;
  double cublas_tflops = flops / (cublas_avg_ms * 1e-3) / 1e12;
  double cutlass_tflops = flops / (cutlass_avg_ms * 1e-3) / 1e12;

  printfQuda("    cuBLAS:   %8.3f ms  (%6.2f TFLOPS)\n", cublas_avg_ms, cublas_tflops);
  printfQuda("    CUTLASS:  %8.3f ms  (%6.2f TFLOPS)\n", cutlass_avg_ms, cutlass_tflops);
  if (tcc_available) {
    double tcc_tflops = flops / (tcc_avg_ms * 1e-3) / 1e12;
    printfQuda("    TCC:      %8.3f ms  (%6.2f TFLOPS)\n", tcc_avg_ms, tcc_tflops);
  } else {
    printfQuda("    TCC:          N/A\n");
  }
  printfQuda("    CUTLASS/cuBLAS: %.2fx\n", cutlass_speedup);
  if (tcc_available) {
    printfQuda("    CUTLASS/TCC:    %.2fx\n", 1.0 / tcc_vs_cutlass);
  }

  // Verify output agreement
  std::vector<float> h_cublas(tri_elems), h_cutlass(tri_elems);
  cudaMemcpy(h_cublas.data(), d_tri_cublas, tri_bytes, cudaMemcpyDeviceToHost);
  cudaMemcpy(h_cutlass.data(), d_tri_cutlass, tri_bytes, cudaMemcpyDeviceToHost);

  double max_rel_err = 0.0;
  int n_checked = 0;
  for (uint64_t i = 0; i < tri_elems && n_checked < 10000; i++, n_checked++) {
    double ref = (double)h_cublas[i];
    double test = (double)h_cutlass[i];
    if (fabs(ref) > 1e-6) {
      double rel = fabs(ref - test) / fabs(ref);
      if (rel > max_rel_err) max_rel_err = rel;
    }
  }
  printfQuda("    Max relative error (CUTLASS vs cuBLAS): %e\n", max_rel_err);

  // TCC vs cuBLAS agreement (if available)
  if (tcc_available) {
    std::vector<float> h_tcc(tri_elems);
    cudaMemcpy(h_tcc.data(), d_tri_tcc, tri_bytes, cudaMemcpyDeviceToHost);
    double tcc_max_rel = 0.0;
    n_checked = 0;
    for (uint64_t i = 0; i < tri_elems && n_checked < 10000; i++, n_checked++) {
      double ref = (double)h_cublas[i];
      double test = (double)h_tcc[i];
      if (fabs(ref) > 1e-6) {
        double rel = fabs(ref - test) / fabs(ref);
        if (rel > tcc_max_rel) tcc_max_rel = rel;
      }
    }
    printfQuda("    Max relative error (TCC vs cuBLAS):    %e\n", tcc_max_rel);
  }

  // Cleanup
  cudaFree(d_qc_input);
  cudaFree(d_promoted);
  cudaFree(d_full_result);
  cudaFree(d_tri_cublas);
  cudaFree(d_tri_cutlass);
  if (d_input_tcc) cudaFree(d_input_tcc);
  if (d_tri_tcc) cudaFree(d_tri_tcc);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  return {cublas_avg_ms, cutlass_avg_ms, tcc_avg_ms, cutlass_speedup, tcc_vs_cutlass};
}

// ============================================================================
// Main
// ============================================================================

int BM_n_antennae = 1664;
int BM_n_channels = 4;
int BM_n_time = 16;
int BM_n_time_inner = 2;
int BM_n_warmup = 3;
int BM_n_iters = 10;

struct cutlass_cublas_benchmark : quda_test {
  void add_command_line_group(std::shared_ptr<GGPApp> app) const override {
    quda_test::add_command_line_group(app);
    auto opgroup = app->add_option_group("Benchmark", "Benchmark options");
    opgroup->add_option("--BM-n-antennae", BM_n_antennae, "Number of antennas (default 1664)");
    opgroup->add_option("--BM-n-channels", BM_n_channels, "Number of channels (default 4)");
    opgroup->add_option("--BM-n-time", BM_n_time, "Time samples (default 16)");
    opgroup->add_option("--BM-n-time-inner", BM_n_time_inner, "Time inner (default 2)");
    opgroup->add_option("--BM-n-warmup", BM_n_warmup, "Warmup iterations (default 3)");
    opgroup->add_option("--BM-n-iters", BM_n_iters, "Benchmark iterations (default 10)");
  }
  cutlass_cublas_benchmark(int argc, char **argv)
    : quda_test("CUTLASS vs cuBLAS vs TCC Benchmark", argc, argv) {}
};

int main(int argc, char **argv) {
  cutlass_cublas_benchmark test(argc, argv);
  test.init();

  int result = 0;
  if (enable_testing) {
    // Run GTest verification suite defined in cutlass_cublas_benchmark_gtest.hpp
    result = test.execute();
    if (result) warningQuda("Google tests for CUTLASS/cuBLAS benchmark failed.");
  } else {

  printfQuda("\n");
  printfQuda("==========================================================\n");
  printfQuda("  CUTLASS vs cuBLAS vs TCC Visibility Correlation Benchmark\n");
  printfQuda("  DSA-2000 Configuration\n");
  printfQuda("==========================================================\n");
  printfQuda("  GPU: NVIDIA GB10 (SM121, Blackwell)\n");
  printfQuda("  Warmup: %d iterations, Timed: %d iterations\n", BM_n_warmup, BM_n_iters);
  printfQuda("  cuBLAS path:  promoteQC -> cuBLAS CGEMM -> triangulate\n");
  printfQuda("  CUTLASS path: herkBatchedCutlassQC (QC -> triangle)\n");
  printfQuda("  TCC path:     convertQcSmToFtd -> correlateTCC\n");
  printfQuda("  Note: TCC requires K %% 128 == 0 (N/A otherwise)\n");
  printfQuda("==========================================================\n");

  // ================================================================
  // Correctness verification against CPU reference
  // ================================================================
  run_verification(32, 8, 4);    // Small (should pass)
  run_verification(64, 8, 4);    // Medium (should pass after K<16 fix)
  run_verification(128, 8, 4);   // Start testing larger N
  run_verification(256, 8, 4);   // Where mismatch was observed
  run_verification(256, 32, 4);  // Large N, larger K
  run_verification(32, 128, 4);  // TCC-aligned (K=128, K%128==0, even batch)

  // ================================================================
  // Test 1: DSA-2000 nominal (1664 antennas, vary time integration)
  // ================================================================
  printfQuda("\n======== Time Integration Sweep (N=%d) ========\n", BM_n_antennae);

  struct SweepResult {
    int param;
    double cublas_ms, cutlass_ms, tcc_ms;
    double cutlass_speedup, tcc_vs_cutlass;
  };
  std::vector<SweepResult> time_results;

  for (int n_time : {16, 32, 64, 128, 256, 512}) {
    if (n_time < BM_n_time_inner) continue;  // skip invalid K=0 configs
    BenchConfig cfg = {BM_n_antennae, BM_n_channels, n_time, BM_n_time_inner, 2, BM_n_warmup, BM_n_iters};
    auto res = run_benchmark(cfg);
    time_results.push_back({n_time, res.cublas_ms, res.cutlass_ms, res.tcc_ms,
                            res.cutlass_speedup, res.tcc_vs_cutlass});
  }

  // ================================================================
  // Test 2: DSA-2000 n_time_inner=32 sweep (K=16, 32, 64)
  // ================================================================
  printfQuda("\n======== DSA-2000 Inner Time=32 Sweep (N=%d) ========\n", BM_n_antennae);

  std::vector<SweepResult> dsa2k_results;
  for (int n_time : {512, 1024, 2048}) {
    // n_time_inner=32: K = n_time/32 = {16, 32, 64}
    BenchConfig cfg = {BM_n_antennae, BM_n_channels, n_time, 32, 2, BM_n_warmup, BM_n_iters};
    auto res = run_benchmark(cfg);
    dsa2k_results.push_back({n_time, res.cublas_ms, res.cutlass_ms, res.tcc_ms,
                              res.cutlass_speedup, res.tcc_vs_cutlass});
  }

  // ================================================================
  // Test 3: Antenna count scaling (T=16, 4 channels)
  // ================================================================
  printfQuda("\n======== Antenna Count Scaling (T=16) ========\n");

  std::vector<SweepResult> ant_results;
  for (int n_ant : {256, 512, 1024, 1664, 2048}) {
    BenchConfig cfg = {n_ant, BM_n_channels, 16, BM_n_time_inner, 2, BM_n_warmup, BM_n_iters};
    auto res = run_benchmark(cfg);
    ant_results.push_back({n_ant, res.cublas_ms, res.cutlass_ms, res.tcc_ms,
                           res.cutlass_speedup, res.tcc_vs_cutlass});
  }

  // ================================================================
  // Summary tables
  // ================================================================
  printfQuda("\n==========================================================\n");
  printfQuda("  SUMMARY\n");
  printfQuda("==========================================================\n");

  printfQuda("\nTime Integration Sweep (N=%d, Nf=%d):\n", BM_n_antennae, BM_n_channels);
  printfQuda("  %-8s  %12s  %12s  %12s  %12s  %12s\n",
             "N_time", "cuBLAS(ms)", "CUTLASS(ms)", "TCC(ms)", "CUT/cuBLAS", "CUT/TCC");
  printfQuda("  %-8s  %12s  %12s  %12s  %12s  %12s\n",
             "------", "----------", "-----------", "------", "----------", "-------");
  for (auto &r : time_results) {
    if (r.tcc_ms > 0) {
      printfQuda("  %-8d  %12.3f  %12.3f  %12.3f  %11.2fx  %11.2fx\n",
                 r.param, r.cublas_ms, r.cutlass_ms, r.tcc_ms,
                 r.cutlass_speedup, 1.0 / r.tcc_vs_cutlass);
    } else {
      printfQuda("  %-8d  %12.3f  %12.3f  %12s  %11.2fx  %12s\n",
                 r.param, r.cublas_ms, r.cutlass_ms, "N/A",
                 r.cutlass_speedup, "N/A");
    }
  }

  printfQuda("\nDSA-2000 Inner Time=32 Sweep (N=%d, Nf=%d):\n", BM_n_antennae, BM_n_channels);
  printfQuda("  %-8s  %6s  %12s  %12s  %12s  %12s  %12s\n",
             "N_time", "K", "cuBLAS(ms)", "CUTLASS(ms)", "TCC(ms)", "CUT/cuBLAS", "CUT/TCC");
  printfQuda("  %-8s  %6s  %12s  %12s  %12s  %12s  %12s\n",
             "------", "---", "----------", "-----------", "------", "----------", "-------");
  for (auto &r : dsa2k_results) {
    int K_val = r.param / 32;
    if (r.tcc_ms > 0) {
      printfQuda("  %-8d  %6d  %12.3f  %12.3f  %12.3f  %11.2fx  %11.2fx\n",
                 r.param, K_val, r.cublas_ms, r.cutlass_ms, r.tcc_ms,
                 r.cutlass_speedup, 1.0 / r.tcc_vs_cutlass);
    } else {
      printfQuda("  %-8d  %6d  %12.3f  %12.3f  %12s  %11.2fx  %12s\n",
                 r.param, K_val, r.cublas_ms, r.cutlass_ms, "N/A",
                 r.cutlass_speedup, "N/A");
    }
  }

  printfQuda("\nAntenna Count Scaling (T=16, Nf=%d):\n", BM_n_channels);
  printfQuda("  %-8s  %12s  %12s  %12s  %12s  %12s\n",
             "N_ant", "cuBLAS(ms)", "CUTLASS(ms)", "TCC(ms)", "CUT/cuBLAS", "CUT/TCC");
  printfQuda("  %-8s  %12s  %12s  %12s  %12s  %12s\n",
             "------", "----------", "-----------", "------", "----------", "-------");
  for (auto &r : ant_results) {
    if (r.tcc_ms > 0) {
      printfQuda("  %-8d  %12.3f  %12.3f  %12.3f  %11.2fx  %11.2fx\n",
                 r.param, r.cublas_ms, r.cutlass_ms, r.tcc_ms,
                 r.cutlass_speedup, 1.0 / r.tcc_vs_cutlass);
    } else {
      printfQuda("  %-8d  %12.3f  %12.3f  %12s  %11.2fx  %12s\n",
                 r.param, r.cublas_ms, r.cutlass_ms, "N/A",
                 r.cutlass_speedup, "N/A");
    }
  }

  printfQuda("\n==========================================================\n");

  } // end else (non-testing mode)

  return result;
}
