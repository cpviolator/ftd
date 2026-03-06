/**
 * @file bench_pipeline.cu
 * @brief Production-scale pipeline benchmark with per-stage CUDA event timing.
 *
 * Benchmarks three DSA-2000 FTD pipeline stages:
 *   - Voltage Beamformer (VoltBF): QC INT4 transpose -> INT4 GEMM x2 (power) -> time integrate -> corner turn
 *   - Visibility Beamformer (VisBF): QC INT4 -> HERK -> pol reduce/sum -> imaging -> corner turn
 *   - Dedispersion (Dedisp): FP32 filterbank -> dedisp_api (CuBLAS_FP32 vs CUTLASS_FP8)
 *
 * Build: via imaging/CMakeLists.txt (bench_pipeline target)
 * Usage: bench_pipeline [suite] [key=value ...]
 */

// CMake defines: STANDALONE_TEST, CUTLASS_GEMM_API, DEDISP_API_LIB

#include "cutlass_gemm_api.h"
#include <imaging_pipeline.h>
#include <corner_turn.h>
#include "dedisp_api.h"

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <curand.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cstdint>
#include <string>
#include <vector>
#include <algorithm>
#include <numeric>
#include <chrono>

// ========================================================================
// Error-checking macros
// ========================================================================

#define CHECK_CUDA(x) do {                                                   \
    cudaError_t err = (x);                                                   \
    if (err != cudaSuccess) {                                                \
        fprintf(stderr, "CUDA error at %s:%d: %s\n",                        \
                __FILE__, __LINE__, cudaGetErrorString(err));                \
        exit(1);                                                             \
    }                                                                        \
} while(0)

#define CHECK_CURAND(x) do {                                                 \
    curandStatus_t st = (x);                                                 \
    if (st != CURAND_STATUS_SUCCESS) {                                       \
        fprintf(stderr, "cuRAND error %d at %s:%d\n",                       \
                (int)st, __FILE__, __LINE__);                               \
        exit(1);                                                             \
    }                                                                        \
} while(0)

// ========================================================================
// StageTimer — per-stage CUDA event timing with multi-iteration stats
// ========================================================================

struct StageTimer {
    struct Stage {
        const char* name;
        cudaEvent_t start, stop;
    };

    std::vector<Stage> stages;
    cudaStream_t stream;
    int cur = -1;

    explicit StageTimer(cudaStream_t s) : stream(s) {}

    void begin(const char* name) {
        if (cur >= 0) end();
        Stage s;
        s.name = name;
        CHECK_CUDA(cudaEventCreate(&s.start));
        CHECK_CUDA(cudaEventCreate(&s.stop));
        CHECK_CUDA(cudaEventRecord(s.start, stream));
        stages.push_back(s);
        cur = (int)stages.size() - 1;
    }

    void end() {
        if (cur >= 0 && cur < (int)stages.size()) {
            CHECK_CUDA(cudaEventRecord(stages[cur].stop, stream));
        }
        cur = -1;
    }

    /// Synchronize and return per-stage times in ms.
    std::vector<float> collect() {
        if (cur >= 0) end();
        CHECK_CUDA(cudaEventSynchronize(stages.back().stop));
        std::vector<float> times(stages.size());
        for (size_t i = 0; i < stages.size(); ++i) {
            CHECK_CUDA(cudaEventElapsedTime(&times[i], stages[i].start, stages[i].stop));
        }
        return times;
    }

    std::vector<const char*> names() const {
        std::vector<const char*> n;
        for (auto& s : stages) n.push_back(s.name);
        return n;
    }

    void reset() {
        for (auto& s : stages) {
            cudaEventDestroy(s.start);
            cudaEventDestroy(s.stop);
        }
        stages.clear();
        cur = -1;
    }
};

/// Print a timing table from multi-iteration results.
static void print_timing_table(
    const std::vector<const char*>& stage_names,
    const std::vector<std::vector<float>>& all_times,
    int runs)
{
    const int n_stages = (int)stage_names.size();

    // Compute per-stage min/mean/std
    std::vector<float> mins(n_stages, 1e30f), means(n_stages, 0), stds(n_stages, 0);
    for (int s = 0; s < n_stages; ++s) {
        for (int r = 0; r < runs; ++r) {
            float t = all_times[r][s];
            if (t < mins[s]) mins[s] = t;
            means[s] += t;
        }
        means[s] /= runs;
        for (int r = 0; r < runs; ++r) {
            float d = all_times[r][s] - means[s];
            stds[s] += d * d;
        }
        stds[s] = sqrtf(stds[s] / runs);
    }

    float total_mean = 0;
    for (int s = 0; s < n_stages; ++s) total_mean += means[s];

    printf("\n  %-24s %10s %11s %10s %8s\n",
           "Stage", "Min (ms)", "Mean (ms)", "Std (ms)", "% Total");
    printf("  %-24s %10s %11s %10s %8s\n",
           "------------------------", "--------", "---------", "--------", "-------");

    float total_min = 0;
    for (int s = 0; s < n_stages; ++s) {
        total_min += mins[s];
        printf("  %-24s %10.2f %11.2f %10.2f %7.1f%%\n",
               stage_names[s], mins[s], means[s], stds[s],
               total_mean > 0 ? 100.0f * means[s] / total_mean : 0.0f);
    }

    float total_std = 0;
    for (int r = 0; r < runs; ++r) {
        float t = 0;
        for (int s = 0; s < n_stages; ++s) t += all_times[r][s];
        float d = t - total_mean;
        total_std += d * d;
    }
    total_std = sqrtf(total_std / runs);

    printf("  %-24s %10s %11s %10s\n",
           "------------------------", "--------", "---------", "--------");
    printf("  %-24s %10.2f %11.2f %10.2f\n",
           "Total", total_min, total_mean, total_std);
}

// ========================================================================
// Anonymous namespace: duplicated kernels from voltage_pipeline.cu
// ========================================================================

namespace {

/// Transpose + pol-split for QC INT4 data (no decode — keeps INT4 byte format).
/// Input:  [n_ch x 2 x n_ant x n_time] bytes (QC format)
/// Output: [n_ch x n_time x n_ant] bytes (INT4 QC), per pol
__global__ void qc_transpose_polsplit_kernel(
    const uint8_t* __restrict__ qc_data,
    uint8_t* __restrict__ int4_pol0,
    uint8_t* __restrict__ int4_pol1,
    int n_ant, int n_time, int n_ch)
{
    const int64_t total = (int64_t)n_ch * n_time * n_ant;
    for (int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
         i < total;
         i += (int64_t)gridDim.x * blockDim.x)
    {
        const int ant  = (int)(i % n_ant);
        const int time = (int)((i / n_ant) % n_time);
        const int chan = (int)(i / ((int64_t)n_ant * n_time));

        const int64_t ch_offset = (int64_t)chan * 2 * n_ant * n_time;
        const int64_t elem_offset = (int64_t)ant * n_time + time;
        const int64_t pol_stride = (int64_t)n_ant * n_time;

        // Just copy bytes — no decode, keep INT4 format
        int4_pol0[i] = qc_data[ch_offset + elem_offset];
        int4_pol1[i] = qc_data[ch_offset + pol_stride + elem_offset];
    }
}

/// Input:  d_power [n_ch x n_time x n_beam] FP32 (Stokes I power per sample)
/// Output: d_filterbank [n_ch x n_beam x n_time_out] FP32
__global__ void time_integrate_fp32_kernel(
    const float* __restrict__ power,
    float* __restrict__ output,
    int n_beam, int n_time, int n_ch,
    int n_time_power_sum)
{
    const int n_time_out = n_time / n_time_power_sum;
    const int64_t total = (int64_t)n_ch * n_beam * n_time_out;

    for (int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
         i < total;
         i += (int64_t)gridDim.x * blockDim.x)
    {
        const int t_out = (int)(i % n_time_out);
        const int beam  = (int)((i / n_time_out) % n_beam);
        const int chan  = (int)(i / ((int64_t)n_time_out * n_beam));

        float acc = 0.0f;
        for (int dt = 0; dt < n_time_power_sum; ++dt) {
            const int t = t_out * n_time_power_sum + dt;
            const int64_t idx = (int64_t)chan * n_time * n_beam
                              + (int64_t)t * n_beam + beam;
            acc += power[idx];
        }

        output[i] = acc;
    }
}

/// Device kernel: deinterleave FP32 [Re,Im,...] -> FP16 planar Re[], Im[].
__global__ void deinterleave_fp32_to_fp16_kernel(
    const float* __restrict__ input,
    __half* __restrict__ out_re,
    __half* __restrict__ out_im,
    int64_t n_complex)
{
    for (int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
         i < n_complex;
         i += (int64_t)gridDim.x * blockDim.x)
    {
        out_re[i] = __float2half(input[2 * i]);
        out_im[i] = __float2half(input[2 * i + 1]);
    }
}

/// Reduce 2-pol packed triangle into 1-pol (sum pol=0 + pol=1).
/// Input:  [n_ch * 2 * n_time_inner, n_bl_2] FP32
/// Output: [n_ch * n_time_inner, n_bl_2] FP32
__global__ void pol_reduce_triangle_kernel(
    const float* __restrict__ tri_in,
    float* __restrict__ tri_out,
    int n_bl_2,
    int n_ch, int n_time_inner)
{
    const int64_t total = (int64_t)n_ch * n_time_inner * n_bl_2;
    for (int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
         i < total;
         i += (int64_t)gridDim.x * blockDim.x)
    {
        const int elem    = (int)(i % n_bl_2);
        const int t_inner = (int)((i / n_bl_2) % n_time_inner);
        const int ch      = (int)(i / ((int64_t)n_bl_2 * n_time_inner));

        const int64_t in_base = (int64_t)ch * 2 * n_time_inner * n_bl_2;
        const float val0 = tri_in[in_base + (int64_t)t_inner * n_bl_2 + elem];
        const float val1 = tri_in[in_base + (int64_t)(n_time_inner + t_inner) * n_bl_2 + elem];

        tri_out[i] = val0 + val1;
    }
}

/// FP16 variant of pol_reduce_triangle_kernel.
/// Input:  [n_ch * 2 * n_time_inner, n_bl_2] __half
/// Output: [n_ch * n_time_inner, n_bl_2] __half
__global__ void pol_reduce_triangle_kernel_fp16(
    const __half* __restrict__ tri_in,
    __half* __restrict__ tri_out,
    int n_bl_2,
    int n_ch, int n_time_inner)
{
    const int64_t total = (int64_t)n_ch * n_time_inner * n_bl_2;
    for (int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
         i < total;
         i += (int64_t)gridDim.x * blockDim.x)
    {
        const int elem    = (int)(i % n_bl_2);
        const int t_inner = (int)((i / n_bl_2) % n_time_inner);
        const int ch      = (int)(i / ((int64_t)n_bl_2 * n_time_inner));

        const int64_t in_base = (int64_t)ch * 2 * n_time_inner * n_bl_2;
        const __half val0 = tri_in[in_base + (int64_t)t_inner * n_bl_2 + elem];
        const __half val1 = tri_in[in_base + (int64_t)(n_time_inner + t_inner) * n_bl_2 + elem];

        tri_out[i] = __hadd(val0, val1);
    }
}

/// Gather-transpose QC data: split pols and reorder for batched HERK.
/// Input:  [n_ch, 2, n_ant, n_time] bytes (QC)
/// Output: [n_ch * n_time_inner, n_ant, K] per pol (contiguous batch elements for HERK)
__global__ void qc_pol_gather_transpose_kernel(
    const uint8_t* __restrict__ qc_data,
    uint8_t* __restrict__ pol0_buf,
    uint8_t* __restrict__ pol1_buf,
    int n_ant, int n_time, int n_ch, int n_time_inner)
{
    const int K = n_time / n_time_inner;
    const int64_t total = (int64_t)n_ch * n_time_inner * n_ant * K;
    for (int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
         i < total;
         i += (int64_t)gridDim.x * blockDim.x)
    {
        const int k       = (int)(i % K);
        const int ant     = (int)((i / K) % n_ant);
        const int t_inner = (int)((i / ((int64_t)K * n_ant)) % n_time_inner);
        const int ch      = (int)(i / ((int64_t)K * n_ant * n_time_inner));

        const int64_t src_base = (int64_t)ch * 2 * n_ant * n_time;
        const int64_t elem = (int64_t)ant * n_time + t_inner * K + k;

        pol0_buf[i] = qc_data[src_base + elem];
        pol1_buf[i] = qc_data[src_base + (int64_t)n_ant * n_time + elem];
    }
}

// INT4 sign-magnitude nibble -> FP8 E4M3 LUT (same as in voltage_pipeline.cu)
__device__ __constant__ uint8_t int4_to_fp8_lut[16] = {
    0x00, 0x38, 0x40, 0x44, 0x48, 0x4A, 0x4C, 0x4E,  // 0..7
    0x00, 0xB8, 0xC0, 0xC4, 0xC8, 0xCA, 0xCC, 0xCE   // -0..-7
};

/// Fused QC transpose + pol-split + ch-fuse + payload-batch + INT4->FP8.
/// (duplicated from voltage_pipeline.cu for standalone bench_pipeline build)
__global__ void qc_fused_transpose_fp8_kernel(
    const uint8_t* __restrict__ qc_data,
    uint8_t* __restrict__ fp8_pol0,
    uint8_t* __restrict__ fp8_pol1,
    int n_ant, int n_time, int n_ch,
    int ch_fuse, int n_payloads,
    int64_t payload_stride)
{
    const int batch_fused = n_ch / ch_fuse;
    const int M_total = n_time * ch_fuse * n_payloads;
    const int64_t total = (int64_t)batch_fused * M_total * n_ant;

    for (int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
         i < total;
         i += (int64_t)gridDim.x * blockDim.x)
    {
        const int ant       = (int)(i % n_ant);
        const int m_idx     = (int)((i / n_ant) % M_total);
        const int batch_idx = (int)(i / ((int64_t)n_ant * M_total));

        const int time        = m_idx % n_time;
        const int ch_in_group = (m_idx / n_time) % ch_fuse;
        const int payload     = m_idx / (n_time * ch_fuse);

        const int chan = batch_idx * ch_fuse + ch_in_group;

        const int64_t payload_offset = (int64_t)payload * payload_stride;
        const int64_t ch_offset = (int64_t)chan * 2 * n_ant * n_time;
        const int64_t elem_offset = (int64_t)ant * n_time + time;
        const int64_t pol_stride = (int64_t)n_ant * n_time;

        const uint8_t byte_0 = qc_data[payload_offset + ch_offset + elem_offset];
        const uint8_t byte_1 = qc_data[payload_offset + ch_offset + pol_stride + elem_offset];

        const uint8_t re_fp8_0 = int4_to_fp8_lut[(byte_0 >> 4) & 0x0F];
        const uint8_t im_fp8_0 = int4_to_fp8_lut[byte_0 & 0x0F];
        const uint8_t re_fp8_1 = int4_to_fp8_lut[(byte_1 >> 4) & 0x0F];
        const uint8_t im_fp8_1 = int4_to_fp8_lut[byte_1 & 0x0F];

        const int64_t out_idx = ((int64_t)batch_idx * M_total + m_idx) * (2 * n_ant)
                              + 2 * ant;

        fp8_pol0[out_idx]     = re_fp8_0;
        fp8_pol0[out_idx + 1] = im_fp8_0;
        fp8_pol1[out_idx]     = re_fp8_1;
        fp8_pol1[out_idx + 1] = im_fp8_1;
    }
}

/// Mask random bytes to valid QC sign-magnitude format.
__global__ void generate_random_qc_kernel(uint8_t* output, const uint8_t* random_bytes,
                                           int64_t total)
{
    for (int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
         i < total; i += (int64_t)gridDim.x * blockDim.x)
    {
        uint8_t r = random_bytes[i];
        output[i] = (r >> 4 << 4) | (r & 0x0F);
    }
}

} // anonymous namespace

// ========================================================================
// Random data generation helpers
// ========================================================================

static void generate_random_qc(uint8_t* d_qc, int64_t total_bytes, unsigned seed)
{
    uint8_t* d_rand = nullptr;
    CHECK_CUDA(cudaMalloc(&d_rand, total_bytes));
    curandGenerator_t gen;
    CHECK_CURAND(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));
    CHECK_CURAND(curandSetPseudoRandomGeneratorSeed(gen, seed));
    int64_t n_uint32 = (total_bytes + 3) / 4;
    CHECK_CURAND(curandGenerate(gen, (unsigned int*)d_rand, n_uint32));
    const int block = 256;
    int grid = (int)std::min((total_bytes + block - 1) / block, (int64_t)1024);
    generate_random_qc_kernel<<<grid, block>>>(d_qc, d_rand, total_bytes);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CURAND(curandDestroyGenerator(gen));
    CHECK_CUDA(cudaFree(d_rand));
}

static void generate_random_weights(float* d_weights, int64_t n_complex, unsigned seed)
{
    curandGenerator_t gen;
    CHECK_CURAND(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));
    CHECK_CURAND(curandSetPseudoRandomGeneratorSeed(gen, seed));
    CHECK_CURAND(curandGenerateNormal(gen, d_weights, n_complex * 2, 0.0f, 1.0f));
    CHECK_CURAND(curandDestroyGenerator(gen));
}

static void generate_noise_filterbank(float* d_fb, int64_t total, float mean, float stddev,
                                       unsigned seed)
{
    curandGenerator_t gen;
    CHECK_CURAND(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));
    CHECK_CURAND(curandSetPseudoRandomGeneratorSeed(gen, seed));
    int64_t n_gen = (total + 1) & ~1LL;
    CHECK_CURAND(curandGenerateNormal(gen, d_fb, n_gen, mean, stddev));
    CHECK_CURAND(curandDestroyGenerator(gen));
}

// ========================================================================
// Memory estimation
// ========================================================================

static void print_memory_estimate(const char* label,
                                   size_t bytes_needed,
                                   bool& skip)
{
    size_t free_mem = 0, total_mem = 0;
    CHECK_CUDA(cudaMemGetInfo(&free_mem, &total_mem));
    printf("  Memory: %.1f MB needed, %.1f MB free (%.0f%%)\n",
           bytes_needed / 1e6, free_mem / 1e6,
           100.0 * bytes_needed / free_mem);
    if (bytes_needed > (size_t)(free_mem * 0.9)) {
        printf("  WARNING: %s would use >90%% of GPU memory — SKIPPING\n", label);
        skip = true;
    } else {
        skip = false;
    }
}

// ========================================================================
// Voltage Beamformer Benchmark
// ========================================================================

static void bench_voltbf(
    int n_ant, int n_beam, int n_ch, int n_time, int n_tps,
    int warmup, int runs,
    bool tune, int tune_verb, int ktune,
    cudaStream_t stream)
{
    using namespace cutlass_gemm_api;

    const int n_time_out = n_time / n_tps;

    printf("\n================================================================\n");
    printf("  Voltage Beamformer Benchmark\n");
    printf("  n_ant=%d  n_beam=%d  n_ch=%d  n_time=%d  n_tps=%d\n",
           n_ant, n_beam, n_ch, n_time, n_tps);
    printf("  GEMM: M=%d N=%d K=%d batch=%d\n", n_time, n_beam, n_ant, n_ch);

    // Memory estimate
    const size_t qc_bytes = (size_t)n_ch * 2 * n_ant * n_time;
    const size_t int4_bytes = (size_t)n_ch * n_time * n_ant;  // per pol (1 byte/complex)
    const size_t power_bytes = (size_t)n_ch * n_time * n_beam * sizeof(float);
    const size_t fb_chan_bytes = (size_t)n_ch * n_beam * n_time_out * sizeof(float);
    const size_t fb_beam_bytes = fb_chan_bytes;
    const size_t weight_bytes = (size_t)n_ch * n_beam * n_ant * 2 * sizeof(float);
    const size_t weight_fp16_bytes = (size_t)n_ch * n_beam * n_ant * sizeof(__half) * 2;
    const size_t total_bytes = qc_bytes + int4_bytes * 2 + power_bytes +
                               fb_chan_bytes + fb_beam_bytes + weight_bytes + weight_fp16_bytes;
    bool skip = false;
    print_memory_estimate("VoltBF", total_bytes, skip);
    if (skip) return;

    printf("  Memory: %.1f MB allocated\n", total_bytes / 1e6);
    printf("================================================================\n");

    // Allocate
    uint8_t *d_qc = nullptr, *d_int4_pol0 = nullptr, *d_int4_pol1 = nullptr;
    float *d_power = nullptr, *d_fb_chan = nullptr, *d_fb_beam = nullptr;
    float *d_weights = nullptr;
    __half *d_wt_re = nullptr, *d_wt_im = nullptr;

    CHECK_CUDA(cudaMalloc(&d_qc, qc_bytes));
    CHECK_CUDA(cudaMalloc(&d_int4_pol0, int4_bytes));
    CHECK_CUDA(cudaMalloc(&d_int4_pol1, int4_bytes));
    CHECK_CUDA(cudaMalloc(&d_power, power_bytes));
    CHECK_CUDA(cudaMalloc(&d_fb_chan, fb_chan_bytes));
    CHECK_CUDA(cudaMalloc(&d_fb_beam, fb_beam_bytes));
    CHECK_CUDA(cudaMalloc(&d_weights, weight_bytes));
    CHECK_CUDA(cudaMalloc(&d_wt_re, (size_t)n_ch * n_beam * n_ant * sizeof(__half)));
    CHECK_CUDA(cudaMalloc(&d_wt_im, (size_t)n_ch * n_beam * n_ant * sizeof(__half)));

    // Random data
    generate_random_qc(d_qc, qc_bytes, 123);
    generate_random_weights(d_weights, (int64_t)n_ch * n_beam * n_ant, 42);

    // CUTLASS API singleton
    CutlassComplexGemm api;
    if (ktune > 0) api.set_kernel_tune_verbosity(ktune);
    if (tune_verb >= 0) {
        api.set_strategy_tune_verbosity(tune_verb);
        api.set_gemm_tune_verbosity(tune_verb);
    }

    // Prepare weights: deinterleave FP32 -> FP16 planar, then prepare_b
    {
        const int64_t n_complex = (int64_t)n_ch * n_beam * n_ant;
        const int block = 256;
        const int grid = (int)std::min((n_complex + block - 1) / block, (int64_t)1024);
        deinterleave_fp32_to_fp16_kernel<<<grid, block, 0, stream>>>(
            d_weights, d_wt_re, d_wt_im, n_complex);
        CHECK_CUDA(cudaGetLastError());
        api.prepare_b(d_wt_re, d_wt_im, n_beam, n_ant, n_ch,
                      ComputePrecision::FP8, stream);
    }
    CHECK_CUDA(cudaStreamSynchronize(stream));

    // Run iterations
    std::vector<std::vector<float>> all_times;

    for (int iter = 0; iter < warmup + runs; ++iter) {
        StageTimer timer(stream);

        // Stage: QC transpose + pol-split (INT4, no decode)
        timer.begin("qc_transpose");
        {
            const int64_t total = (int64_t)n_ch * n_time * n_ant;
            const int block = 256;
            const int grid = (int)std::min((total + block - 1) / block, (int64_t)1024);
            qc_transpose_polsplit_kernel<<<grid, block, 0, stream>>>(
                d_qc, d_int4_pol0, d_int4_pol1, n_ant, n_time, n_ch);
        }

        // Stage: GEMM pol0 (INT4 -> FP8 conversion inside PIMPL)
        timer.begin("gemm_pol0");
        api.gemm_prepared_power_int4(d_int4_pol0, d_power,
            n_time, n_beam, n_ant, n_ch, 1.0f, 0.0f, stream);

        // Stage: GEMM pol1
        timer.begin("gemm_pol1");
        api.gemm_prepared_power_int4(d_int4_pol1, d_power,
            n_time, n_beam, n_ant, n_ch, 1.0f, 1.0f, stream);

        // Stage: time integrate
        timer.begin("time_integrate");
        {
            const int64_t total = (int64_t)n_ch * n_beam * n_time_out;
            const int block = 256;
            const int grid = (int)std::min((total + block - 1) / block, (int64_t)1024);
            time_integrate_fp32_kernel<<<grid, block, 0, stream>>>(
                d_power, d_fb_chan, n_beam, n_time, n_ch, n_tps);
        }

        // Stage: corner turn
        timer.begin("corner_turn");
        ggp::corner_turn_nf_nb(d_fb_beam, d_fb_chan, n_ch, n_beam, n_time_out, stream);
        timer.end();

        auto times = timer.collect();
        if (iter >= warmup) {
            all_times.push_back(times);
        }
        timer.reset();
    }

    // Report
    auto names = StageTimer(stream).names();
    // Reconstruct names from the last iteration's pattern
    std::vector<const char*> stage_names = {"qc_transpose", "gemm_pol0", "gemm_pol1",
                                             "time_integrate", "corner_turn"};
    print_timing_table(stage_names, all_times, runs);

    // GEMM TFLOPS: 2 pols x 8 real FLOPs per complex multiply-add x M x N x K x batch
    float total_mean = 0;
    float gemm_mean = 0;
    for (int r = 0; r < runs; ++r) {
        for (int s = 0; s < (int)all_times[r].size(); ++s) total_mean += all_times[r][s];
        gemm_mean += all_times[r][1] + all_times[r][2];  // pol0 + pol1
    }
    total_mean /= runs;
    gemm_mean /= runs;

    double gemm_flops = 2.0 * 8.0 * n_time * (double)n_beam * n_ant * n_ch;
    double tflops = gemm_flops / (gemm_mean * 1e-3) / 1e12;
    double throughput = 1000.0 / total_mean;

    double beams_per_ch_sec = (double)n_beam * throughput;
    printf("  TFLOPS: %.1f  |  Throughput: %.1f payloads/sec  |  Beams/ch/sec: %.2e\n",
           tflops, throughput, beams_per_ch_sec);

    // Cleanup
    cudaFree(d_qc); cudaFree(d_int4_pol0); cudaFree(d_int4_pol1);
    cudaFree(d_power); cudaFree(d_fb_chan); cudaFree(d_fb_beam);
    cudaFree(d_weights); cudaFree(d_wt_re); cudaFree(d_wt_im);
}

// ========================================================================
// Fused Voltage Beamformer: Channel Fusion + Payload Batching
// ========================================================================
//
// Eliminates per-call overhead for short-integration (small n_time) by:
//   1. Fusing ch_fuse consecutive channels along M (B weights shared)
//   2. Batching n_payloads_batch payloads along M (same weights)
//   3. Single-pass fused kernel: QC transpose + pol-split + ch-fuse + FP8 conversion
//   4. Calling gemm_prepared_power_fp8() directly (skips INT4->FP8 re-conversion)

static void bench_voltbf_fused(
    int n_ant, int n_beam, int n_ch, int n_time, int n_tps,
    int ch_fuse, int n_payloads_batch,
    int warmup, int runs,
    bool tune, int tune_verb, int ktune,
    cudaStream_t stream)
{
    using namespace cutlass_gemm_api;

    if (n_ch % ch_fuse != 0) {
        fprintf(stderr, "  ERROR: n_ch (%d) not divisible by ch_fuse (%d)\n", n_ch, ch_fuse);
        return;
    }

    const int batch_fused = n_ch / ch_fuse;
    const int M_total = n_time * ch_fuse * n_payloads_batch;
    const int n_time_out = n_time / n_tps;

    printf("\n================================================================\n");
    printf("  Fused Voltage Beamformer Benchmark\n");
    printf("  n_ant=%d  n_beam=%d  n_ch=%d  n_time=%d  n_tps=%d\n",
           n_ant, n_beam, n_ch, n_time, n_tps);
    printf("  ch_fuse=%d  n_payloads_batch=%d\n", ch_fuse, n_payloads_batch);
    printf("  Fused GEMM: M=%d N=%d K=%d batch=%d\n", M_total, n_beam, n_ant, batch_fused);

    // Memory estimate
    const size_t qc_bytes = (size_t)n_ch * 2 * n_ant * n_time * n_payloads_batch;
    const size_t fp8_bytes = (size_t)batch_fused * M_total * 2 * n_ant;  // per pol
    const size_t power_bytes = (size_t)batch_fused * M_total * n_beam * sizeof(float);
    const size_t fb_chan_bytes = (size_t)batch_fused * (M_total / n_tps) * n_beam * sizeof(float);
    const size_t weight_bytes = (size_t)batch_fused * n_beam * n_ant * 2 * sizeof(float);
    const size_t weight_fp16_bytes = (size_t)batch_fused * n_beam * n_ant * sizeof(__half) * 2;
    const size_t total_bytes = qc_bytes + fp8_bytes * 2 + power_bytes +
                               fb_chan_bytes + weight_bytes + weight_fp16_bytes;
    bool skip = false;
    print_memory_estimate("VoltBF-Fused", total_bytes, skip);
    if (skip) return;

    printf("  Memory: %.1f MB allocated\n", total_bytes / 1e6);
    printf("================================================================\n");

    // Allocate
    uint8_t *d_qc = nullptr;
    uint8_t *d_fp8_pol0 = nullptr, *d_fp8_pol1 = nullptr;
    float *d_power = nullptr, *d_fb_chan = nullptr;
    float *d_weights = nullptr;
    __half *d_wt_re = nullptr, *d_wt_im = nullptr;

    CHECK_CUDA(cudaMalloc(&d_qc, qc_bytes));
    CHECK_CUDA(cudaMalloc(&d_fp8_pol0, fp8_bytes));
    CHECK_CUDA(cudaMalloc(&d_fp8_pol1, fp8_bytes));
    CHECK_CUDA(cudaMalloc(&d_power, power_bytes));
    CHECK_CUDA(cudaMalloc(&d_fb_chan, fb_chan_bytes));
    CHECK_CUDA(cudaMalloc(&d_weights, weight_bytes));
    CHECK_CUDA(cudaMalloc(&d_wt_re, (size_t)batch_fused * n_beam * n_ant * sizeof(__half)));
    CHECK_CUDA(cudaMalloc(&d_wt_im, (size_t)batch_fused * n_beam * n_ant * sizeof(__half)));

    // Random data
    generate_random_qc(d_qc, qc_bytes, 123);
    generate_random_weights(d_weights, (int64_t)batch_fused * n_beam * n_ant, 42);

    // CUTLASS API
    CutlassComplexGemm api;
    if (ktune > 0) api.set_kernel_tune_verbosity(ktune);
    if (tune_verb >= 0) {
        api.set_strategy_tune_verbosity(tune_verb);
        api.set_gemm_tune_verbosity(tune_verb);
    }

    // Prepare weights (batch = n_ch / ch_fuse)
    {
        const int64_t n_complex = (int64_t)batch_fused * n_beam * n_ant;
        const int block = 256;
        const int grid = (int)std::min((n_complex + block - 1) / block, (int64_t)1024);
        deinterleave_fp32_to_fp16_kernel<<<grid, block, 0, stream>>>(
            d_weights, d_wt_re, d_wt_im, n_complex);
        CHECK_CUDA(cudaGetLastError());
        api.prepare_b(d_wt_re, d_wt_im, n_beam, n_ant, batch_fused,
                      ComputePrecision::FP8, stream);
    }
    CHECK_CUDA(cudaStreamSynchronize(stream));

    // Run iterations
    const int64_t payload_stride = (int64_t)n_ch * 2 * n_ant * n_time;
    std::vector<std::vector<float>> all_times;

    for (int iter = 0; iter < warmup + runs; ++iter) {
        StageTimer timer(stream);

        // Stage: Fused QC transpose + FP8 conversion (single pass)
        timer.begin("fused_transpose_fp8");
        {
            const int64_t total = (int64_t)batch_fused * M_total * n_ant;
            const int block = 256;
            const int grid = (int)std::min((total + block - 1) / block, (int64_t)1024);
            qc_fused_transpose_fp8_kernel<<<grid, block, 0, stream>>>(
                d_qc, d_fp8_pol0, d_fp8_pol1,
                n_ant, n_time, n_ch, ch_fuse, n_payloads_batch,
                payload_stride);
        }

        // Stage: GEMM pol0 (already FP8 — no internal conversion)
        timer.begin("gemm_pol0");
        api.gemm_prepared_power_fp8(d_fp8_pol0, d_power,
            M_total, n_beam, n_ant, batch_fused, 1.0f, 0.0f, stream);

        // Stage: GEMM pol1
        timer.begin("gemm_pol1");
        api.gemm_prepared_power_fp8(d_fp8_pol1, d_power,
            M_total, n_beam, n_ant, batch_fused, 1.0f, 1.0f, stream);

        // Stage: time integrate
        timer.begin("time_integrate");
        {
            const int64_t total = (int64_t)batch_fused * (M_total / n_tps) * n_beam;
            const int block = 256;
            const int grid = (int)std::min((total + block - 1) / block, (int64_t)1024);
            time_integrate_fp32_kernel<<<grid, block, 0, stream>>>(
                d_power, d_fb_chan, n_beam, M_total, batch_fused, n_tps);
        }
        timer.end();

        auto times = timer.collect();
        if (iter >= warmup) {
            all_times.push_back(times);
        }
        timer.reset();
    }

    // Report
    std::vector<const char*> stage_names = {"fused_transpose_fp8", "gemm_pol0", "gemm_pol1",
                                             "time_integrate"};
    print_timing_table(stage_names, all_times, runs);

    // GEMM TFLOPS: includes all n_payloads * all channels
    float total_mean = 0;
    float gemm_mean = 0;
    for (int r = 0; r < runs; ++r) {
        for (int s = 0; s < (int)all_times[r].size(); ++s) total_mean += all_times[r][s];
        gemm_mean += all_times[r][1] + all_times[r][2];
    }
    total_mean /= runs;
    gemm_mean /= runs;

    // 2 pols x 8 real FLOPs x M_total x N x K x batch_fused
    double gemm_flops = 2.0 * 8.0 * M_total * (double)n_beam * n_ant * batch_fused;
    double tflops = gemm_flops / (gemm_mean * 1e-3) / 1e12;
    double throughput = 1000.0 * n_payloads_batch / total_mean;

    double beams_per_ch_sec = (double)n_beam * throughput;
    printf("  TFLOPS: %.1f  |  Throughput: %.1f payloads/sec  (%d payloads/call)  |  Beams/ch/sec: %.2e\n",
           tflops, throughput, n_payloads_batch, beams_per_ch_sec);

    // Compare vs unfused
    printf("\n  vs unfused M=%d batch=%d: M increased %.0fx, batch reduced %.0fx\n",
           n_time, n_ch,
           (double)M_total / n_time,
           (double)n_ch / batch_fused);

    // Cleanup
    cudaFree(d_qc); cudaFree(d_fp8_pol0); cudaFree(d_fp8_pol1);
    cudaFree(d_power); cudaFree(d_fb_chan);
    cudaFree(d_weights); cudaFree(d_wt_re); cudaFree(d_wt_im);
}

// ========================================================================
// Multi-Stream Voltage Beamformer: 2-Stream Overlap
// ========================================================================
//
// Overlaps compute-bound and memory-bound stages using 2 CUDA streams:
//   Stream A (compute): QC cast + GEMM pol0 + GEMM pol1
//   Stream B (postproc): time_integrate + corner_turn
//
// Stream B waits for GEMM completion via event before starting postproc.
// With very short n_time (8-16), all stages are tiny so overlap is minimal,
// but this measures the overhead of multi-stream coordination.

static void bench_voltbf_multistream(
    int n_ant, int n_beam, int n_ch, int n_time, int n_tps,
    int /*n_ch_tile*/,
    int warmup, int runs,
    bool tune, int tune_verb, int ktune)
{
    using namespace cutlass_gemm_api;

    const int n_time_out = n_time / n_tps;

    printf("\n================================================================\n");
    printf("  Multi-Stream Voltage Beamformer Benchmark\n");
    printf("  n_ant=%d  n_beam=%d  n_ch=%d  n_time=%d  n_tps=%d\n",
           n_ant, n_beam, n_ch, n_time, n_tps);
    printf("  GEMM: M=%d N=%d K=%d batch=%d\n", n_time, n_beam, n_ant, n_ch);
    printf("  Strategy: GEMM on stream_a, postproc on stream_b\n");
    printf("================================================================\n");

    // Memory estimate
    const size_t qc_bytes = (size_t)n_ch * 2 * n_ant * n_time;
    const size_t int4_bytes = (size_t)n_ch * n_time * n_ant;  // per pol (1 byte/complex)
    const size_t power_bytes = (size_t)n_ch * n_time * n_beam * sizeof(float);
    const size_t fb_chan_bytes = (size_t)n_ch * n_beam * n_time_out * sizeof(float);
    const size_t fb_beam_bytes = fb_chan_bytes;
    const size_t weight_bytes = (size_t)n_ch * n_beam * n_ant * 2 * sizeof(float);
    const size_t weight_fp16_bytes = (size_t)n_ch * n_beam * n_ant * sizeof(__half) * 2;
    const size_t total_bytes = qc_bytes + int4_bytes * 2 + power_bytes +
                               fb_chan_bytes + fb_beam_bytes + weight_bytes + weight_fp16_bytes;
    bool skip = false;
    print_memory_estimate("VoltBF-MS", total_bytes, skip);
    if (skip) return;

    printf("  Memory: %.1f MB allocated\n", total_bytes / 1e6);

    // Allocate
    uint8_t *d_qc = nullptr, *d_int4_pol0 = nullptr, *d_int4_pol1 = nullptr;
    float *d_power = nullptr, *d_fb_chan = nullptr, *d_fb_beam = nullptr;
    float *d_weights = nullptr;
    __half *d_wt_re = nullptr, *d_wt_im = nullptr;

    CHECK_CUDA(cudaMalloc(&d_qc, qc_bytes));
    CHECK_CUDA(cudaMalloc(&d_int4_pol0, int4_bytes));
    CHECK_CUDA(cudaMalloc(&d_int4_pol1, int4_bytes));
    CHECK_CUDA(cudaMalloc(&d_power, power_bytes));
    CHECK_CUDA(cudaMalloc(&d_fb_chan, fb_chan_bytes));
    CHECK_CUDA(cudaMalloc(&d_fb_beam, fb_beam_bytes));
    CHECK_CUDA(cudaMalloc(&d_weights, weight_bytes));
    CHECK_CUDA(cudaMalloc(&d_wt_re, (size_t)n_ch * n_beam * n_ant * sizeof(__half)));
    CHECK_CUDA(cudaMalloc(&d_wt_im, (size_t)n_ch * n_beam * n_ant * sizeof(__half)));

    // Random data
    generate_random_qc(d_qc, qc_bytes, 123);
    generate_random_weights(d_weights, (int64_t)n_ch * n_beam * n_ant, 42);

    // Two streams
    cudaStream_t stream_compute, stream_postproc;
    CHECK_CUDA(cudaStreamCreate(&stream_compute));
    CHECK_CUDA(cudaStreamCreate(&stream_postproc));

    CutlassComplexGemm api;
    if (ktune > 0) api.set_kernel_tune_verbosity(ktune);
    if (tune_verb >= 0) {
        api.set_strategy_tune_verbosity(tune_verb);
        api.set_gemm_tune_verbosity(tune_verb);
    }

    // Prepare weights on stream_compute
    {
        const int64_t n_complex = (int64_t)n_ch * n_beam * n_ant;
        const int block = 256;
        const int grid = (int)std::min((n_complex + block - 1) / block, (int64_t)1024);
        deinterleave_fp32_to_fp16_kernel<<<grid, block, 0, stream_compute>>>(
            d_weights, d_wt_re, d_wt_im, n_complex);
        CHECK_CUDA(cudaGetLastError());
        api.prepare_b(d_wt_re, d_wt_im, n_beam, n_ant, n_ch,
                      ComputePrecision::FP8, stream_compute);
    }
    CHECK_CUDA(cudaStreamSynchronize(stream_compute));

    // Run iterations
    std::vector<float> iter_times;

    for (int iter = 0; iter < warmup + runs; ++iter) {
        cudaEvent_t ev_start, ev_stop, ev_gemm_done;
        CHECK_CUDA(cudaEventCreate(&ev_start));
        CHECK_CUDA(cudaEventCreate(&ev_stop));
        CHECK_CUDA(cudaEventCreate(&ev_gemm_done));

        CHECK_CUDA(cudaEventRecord(ev_start, stream_compute));

        // Stream A: QC transpose + GEMM pol0 + GEMM pol1
        {
            const int64_t total = (int64_t)n_ch * n_time * n_ant;
            const int block = 256;
            const int grid = (int)std::min((total + block - 1) / block, (int64_t)1024);
            qc_transpose_polsplit_kernel<<<grid, block, 0, stream_compute>>>(
                d_qc, d_int4_pol0, d_int4_pol1, n_ant, n_time, n_ch);
        }
        api.gemm_prepared_power_int4(d_int4_pol0, d_power,
            n_time, n_beam, n_ant, n_ch, 1.0f, 0.0f, stream_compute);
        api.gemm_prepared_power_int4(d_int4_pol1, d_power,
            n_time, n_beam, n_ant, n_ch, 1.0f, 1.0f, stream_compute);

        // Signal GEMM completion
        CHECK_CUDA(cudaEventRecord(ev_gemm_done, stream_compute));

        // Stream B: wait for GEMM, then time_integrate + corner_turn
        CHECK_CUDA(cudaStreamWaitEvent(stream_postproc, ev_gemm_done));
        {
            const int64_t total = (int64_t)n_ch * n_beam * n_time_out;
            const int block = 256;
            const int grid = (int)std::min((total + block - 1) / block, (int64_t)1024);
            time_integrate_fp32_kernel<<<grid, block, 0, stream_postproc>>>(
                d_power, d_fb_chan, n_beam, n_time, n_ch, n_tps);
        }
        ggp::corner_turn_nf_nb(d_fb_beam, d_fb_chan, n_ch, n_beam, n_time_out, stream_postproc);

        CHECK_CUDA(cudaEventRecord(ev_stop, stream_postproc));
        CHECK_CUDA(cudaEventSynchronize(ev_stop));

        float ms = 0;
        CHECK_CUDA(cudaEventElapsedTime(&ms, ev_start, ev_stop));
        if (iter >= warmup) iter_times.push_back(ms);

        cudaEventDestroy(ev_start);
        cudaEventDestroy(ev_stop);
        cudaEventDestroy(ev_gemm_done);
    }

    // Report
    float min_ms = *std::min_element(iter_times.begin(), iter_times.end());
    float mean_ms = std::accumulate(iter_times.begin(), iter_times.end(), 0.0f) / runs;
    float var = 0;
    for (float t : iter_times) { float d = t - mean_ms; var += d * d; }
    float std_ms = sqrtf(var / runs);

    double gemm_flops = 2.0 * 8.0 * n_time * (double)n_beam * n_ant * n_ch;
    double tflops = gemm_flops / (mean_ms * 1e-3) / 1e12;

    printf("\n  %-24s %10s %11s %10s\n", "Pipeline", "Min (ms)", "Mean (ms)", "Std (ms)");
    printf("  %-24s %10s %11s %10s\n", "------------------------", "--------", "---------", "--------");
    printf("  %-24s %10.2f %11.2f %10.2f\n", "Multi-stream total", min_ms, mean_ms, std_ms);
    double throughput_ms = 1000.0 / mean_ms;
    double beams_per_ch_sec = (double)n_beam * throughput_ms;
    printf("  Effective TFLOPS: %.1f  |  Throughput: %.1f payloads/sec  |  Beams/ch/sec: %.2e\n",
           tflops, throughput_ms, beams_per_ch_sec);

    // Cleanup
    cudaFree(d_qc); cudaFree(d_int4_pol0); cudaFree(d_int4_pol1);
    cudaFree(d_power);
    cudaFree(d_fb_chan); cudaFree(d_fb_beam);
    cudaFree(d_weights); cudaFree(d_wt_re); cudaFree(d_wt_im);
    cudaStreamDestroy(stream_compute);
    cudaStreamDestroy(stream_postproc);
}

// ========================================================================
// Multi-Stream VisBF: Overlap HERK with Imaging
// ========================================================================
//
// Imaging processes Nf_eff channels in tiles internally (typically
// Nf_tile channels per grid_and_image tile). We can overlap:
//   Stream A: HERK (produces packed triangle for all channels at once)
//   Stream B: Imaging (processes channel tiles as they become available)
//
// However, HERK produces ALL batch outputs before any are available (it's
// one big batched call). So the overlap opportunity is limited to:
//   1. Overlap pol_reduce with the end of HERK (marginal)
//   2. In mode B: overlap HERK_pol1 with imaging of HERK_pol0's output
//      (but they write to the same tri buffer with beta accumulation)
//
// The main win is: in mode A, we can overlap the pol_reduce kernel
// (41 ms) with the start of imaging. Let's test this.

static void bench_visbf_multistream(
    int n_ant, int n_beam, int n_ch, int n_time, int n_time_inner,
    int Ng, char pol_mode,
    int warmup, int runs,
    bool tune, int tune_verb, int ktune,
    bool vis_prec_fp16, bool fft_prec_fp16)
{
    using namespace cutlass_gemm_api;

    const int K = n_time / n_time_inner;
    const int n_baselines = n_ant * (n_ant + 1) / 2;
    const int n_bl_2 = n_baselines * 2;
    const int Nf_eff = n_ch * n_time_inner;
    const int herk_batch = pol_mode == 'A'
        ? n_ch * 2 * n_time_inner : n_ch * n_time_inner;
    const size_t vis_elem_size = vis_prec_fp16 ? sizeof(__half) : sizeof(float);
    const auto out_prec = vis_prec_fp16 ? OutputPrecision::FP16 : OutputPrecision::FP32;
    const auto vis_prec = vis_prec_fp16
        ? imaging_pipeline::VisPrecision::FP16
        : imaging_pipeline::VisPrecision::FP32;
    const auto fft_prec = fft_prec_fp16
        ? imaging_pipeline::FftPrecision::FP16
        : imaging_pipeline::FftPrecision::FP32;

    printf("\n================================================================\n");
    printf("  Multi-Stream Visibility Beamformer (pol_mode=%c)\n", pol_mode);
    printf("  n_ant=%d  n_ch=%d  n_time=%d  n_time_inner=%d\n",
           n_ant, n_ch, n_time, n_time_inner);
    printf("  HERK: N=%d K=%d batch=%d  output=%s\n", n_ant, K, herk_batch,
           vis_prec_fp16 ? "FP16" : "FP32");
    printf("  Imaging: Nf_eff=%d  Ng=%d  n_beam=%d  fft=%s\n",
           Nf_eff, Ng, n_beam, fft_prec_fp16 ? "FP16" : "FP32");
    printf("  Strategy: HERK on stream_a, pol_reduce+imaging on stream_b\n");
    printf("================================================================\n");

    // Allocate
    const size_t qc_bytes = (size_t)n_ch * 2 * n_ant * n_time;
    const size_t tri_full_bytes = (size_t)herk_batch * n_bl_2 * vis_elem_size;
    const size_t tri_reduced_bytes = pol_mode == 'A'
        ? (size_t)Nf_eff * n_bl_2 * vis_elem_size : 0;
    const size_t beam_bytes = (size_t)Nf_eff * n_beam * sizeof(float);
    const size_t fb_beam_bytes = (size_t)n_beam * Nf_eff * sizeof(float);
    const size_t total_bytes = qc_bytes + tri_full_bytes + tri_reduced_bytes + beam_bytes + fb_beam_bytes;

    bool skip = false;
    print_memory_estimate("VisBF-MS", total_bytes, skip);
    if (skip) return;

    uint8_t *d_qc = nullptr;
    void *d_tri_full = nullptr, *d_tri_reduced = nullptr;
    float *d_beam_output = nullptr, *d_fb_beam = nullptr;

    CHECK_CUDA(cudaMalloc(&d_qc, qc_bytes));
    CHECK_CUDA(cudaMalloc(&d_tri_full, tri_full_bytes));
    if (pol_mode == 'A')
        CHECK_CUDA(cudaMalloc(&d_tri_reduced, (size_t)Nf_eff * n_bl_2 * vis_elem_size));
    CHECK_CUDA(cudaMalloc(&d_beam_output, beam_bytes));
    CHECK_CUDA(cudaMalloc(&d_fb_beam, fb_beam_bytes));

    generate_random_qc(d_qc, qc_bytes, 123);

    // Two streams
    cudaStream_t stream_herk, stream_img;
    CHECK_CUDA(cudaStreamCreate(&stream_herk));
    CHECK_CUDA(cudaStreamCreate(&stream_img));

    CutlassComplexGemm api;
    if (ktune > 0) api.set_kernel_tune_verbosity(ktune);
    if (tune_verb >= 0) {
        api.set_strategy_tune_verbosity(tune_verb);
        api.set_gemm_tune_verbosity(tune_verb);
    }

    // Imaging pipeline (configured on stream_img)
    imaging_pipeline::ImagingPipeline imager;
    std::vector<float> baseline_uv(n_baselines * 2);
    srand(42);
    for (int i = 0; i < n_baselines * 2; ++i)
        baseline_uv[i] = ((float)rand() / RAND_MAX - 0.5f) * 2000.0f;
    std::vector<double> freqs(Nf_eff);
    for (int i = 0; i < Nf_eff; ++i)
        freqs[i] = 700e6 + (1500e6 - 700e6) * i / std::max(Nf_eff - 1, 1);
    std::vector<int> beam_pixels(n_beam * 2);
    for (int i = 0; i < n_beam * 2; ++i)
        beam_pixels[i] = rand() % Ng;

    auto active_fft_prec_ms = fft_prec;
    {
        int ret = imager.configure(n_ant, Nf_eff, Ng, n_beam, active_fft_prec_ms, stream_img);
        if (ret != 0 && active_fft_prec_ms == imaging_pipeline::FftPrecision::FP16) {
            printf("  WARNING: FP16 FFT failed — falling back to FP32\n");
            active_fft_prec_ms = imaging_pipeline::FftPrecision::FP32;
            imager.configure(n_ant, Nf_eff, Ng, n_beam, active_fft_prec_ms, stream_img);
        }
    }
    imager.set_baseline_uv(baseline_uv.data(), n_baselines);
    imager.set_frequencies(freqs.data());
    imager.set_cell_size(1e-4f);
    imager.set_beam_pixels(beam_pixels.data(), n_beam);
    CHECK_CUDA(cudaStreamSynchronize(stream_img));

    std::vector<float> iter_times;

    for (int iter = 0; iter < warmup + runs; ++iter) {
        cudaEvent_t ev_start, ev_stop;
        CHECK_CUDA(cudaEventCreate(&ev_start));
        CHECK_CUDA(cudaEventCreate(&ev_stop));

        CHECK_CUDA(cudaEventRecord(ev_start, stream_herk));

        if (pol_mode == 'A') {
            // HERK on stream_herk
            api.herk(d_qc, d_tri_full, n_ant, K,
                     herk_batch,
                     InputPrecision::INT4, ComputePrecision::FP8,
                     out_prec, OutputFormat::PackedTriangle,
                     1.0f, 0.0f, stream_herk, tune && iter == 0);

            // Signal HERK done
            cudaEvent_t ev_herk_done;
            CHECK_CUDA(cudaEventCreate(&ev_herk_done));
            CHECK_CUDA(cudaEventRecord(ev_herk_done, stream_herk));

            // stream_img waits for HERK, then runs pol_reduce + imaging
            CHECK_CUDA(cudaStreamWaitEvent(stream_img, ev_herk_done));
            cudaEventDestroy(ev_herk_done);

            // Pol reduce on stream_img
            {
                const int64_t total = (int64_t)n_ch * n_time_inner * n_bl_2;
                const int block = 256;
                const int grid = (int)std::min((total + block - 1) / block, (int64_t)1024);
                if (vis_prec_fp16) {
                    pol_reduce_triangle_kernel_fp16<<<grid, block, 0, stream_img>>>(
                        static_cast<const __half*>(d_tri_full),
                        static_cast<__half*>(d_tri_reduced), n_bl_2, n_ch, n_time_inner);
                } else {
                    pol_reduce_triangle_kernel<<<grid, block, 0, stream_img>>>(
                        static_cast<const float*>(d_tri_full),
                        static_cast<float*>(d_tri_reduced), n_bl_2, n_ch, n_time_inner);
                }
            }

            // Imaging on stream_img
            imager.grid_and_image(d_tri_reduced, d_beam_output,
                                  vis_prec, stream_img);
        } else {
            // Mode B
            api.herk(d_qc, d_tri_full, n_ant, K,
                     herk_batch,
                     InputPrecision::INT4, ComputePrecision::FP8,
                     out_prec, OutputFormat::PackedTriangle,
                     1.0f, 0.0f, stream_herk, tune && iter == 0);

            cudaEvent_t ev_herk_done;
            CHECK_CUDA(cudaEventCreate(&ev_herk_done));
            CHECK_CUDA(cudaEventRecord(ev_herk_done, stream_herk));
            CHECK_CUDA(cudaStreamWaitEvent(stream_img, ev_herk_done));
            cudaEventDestroy(ev_herk_done);

            imager.grid_and_image(d_tri_full, d_beam_output,
                                  vis_prec, stream_img);
        }

        // Corner turn on stream_img
        ggp::corner_turn_nf_nb(d_fb_beam, d_beam_output, Nf_eff, n_beam, 1, stream_img);

        CHECK_CUDA(cudaEventRecord(ev_stop, stream_img));
        CHECK_CUDA(cudaEventSynchronize(ev_stop));

        float ms = 0;
        CHECK_CUDA(cudaEventElapsedTime(&ms, ev_start, ev_stop));
        if (iter >= warmup) iter_times.push_back(ms);

        cudaEventDestroy(ev_start);
        cudaEventDestroy(ev_stop);
    }

    float min_ms = *std::min_element(iter_times.begin(), iter_times.end());
    float mean_ms = std::accumulate(iter_times.begin(), iter_times.end(), 0.0f) / runs;
    float var = 0;
    for (float t : iter_times) { float d = t - mean_ms; var += d * d; }
    float std_ms = sqrtf(var / runs);

    printf("\n  %-24s %10s %11s %10s\n", "Pipeline", "Min (ms)", "Mean (ms)", "Std (ms)");
    printf("  %-24s %10s %11s %10s\n", "------------------------", "--------", "---------", "--------");
    printf("  %-24s %10.2f %11.2f %10.2f\n", "Multi-stream total", min_ms, mean_ms, std_ms);
    double throughput_ms = 1000.0 / mean_ms;
    double beams_per_ch_sec = (double)n_beam * throughput_ms;
    printf("  Throughput: %.1f payloads/sec  |  Beams/ch/sec: %.2e\n", throughput_ms, beams_per_ch_sec);

    cudaFree(d_qc); cudaFree(d_tri_full);
    if (d_tri_reduced) cudaFree(d_tri_reduced);
    cudaFree(d_beam_output); cudaFree(d_fb_beam);
    cudaStreamDestroy(stream_herk);
    cudaStreamDestroy(stream_img);
}

// ========================================================================
// Visibility Beamformer Benchmark
// ========================================================================

static void bench_visbf(
    int n_ant, int n_beam, int n_ch, int n_time, int n_time_inner,
    int Ng, char pol_mode,
    int warmup, int runs,
    bool tune, int tune_verb, int ktune,
    bool vis_prec_fp16, bool fft_prec_fp16,
    cudaStream_t stream)
{
    using namespace cutlass_gemm_api;

    const int K = n_time / n_time_inner;
    const int n_baselines = n_ant * (n_ant + 1) / 2;
    const int n_bl_2 = n_baselines * 2;  // real + imag elements per packed triangle
    const size_t vis_elem_size = vis_prec_fp16 ? sizeof(__half) : sizeof(float);
    const auto out_prec = vis_prec_fp16 ? OutputPrecision::FP16 : OutputPrecision::FP32;
    const auto vis_prec = vis_prec_fp16
        ? imaging_pipeline::VisPrecision::FP16
        : imaging_pipeline::VisPrecision::FP32;
    const auto fft_prec = fft_prec_fp16
        ? imaging_pipeline::FftPrecision::FP16
        : imaging_pipeline::FftPrecision::FP32;

    printf("\n================================================================\n");
    printf("  Visibility Beamformer Benchmark (pol_mode=%c)\n", pol_mode);
    printf("  n_ant=%d  n_ch=%d  n_time=%d  n_time_inner=%d\n",
           n_ant, n_ch, n_time, n_time_inner);
    printf("  HERK: N=%d K=%d batch=%d  n_baselines=%d  output=%s\n",
           n_ant, K,
           pol_mode == 'A' ? n_ch * 2 * n_time_inner : n_ch * n_time_inner,
           n_baselines, vis_prec_fp16 ? "FP16" : "FP32");

    const int Nf_eff = n_ch * n_time_inner;

    // Memory estimate
    const size_t qc_bytes = (size_t)n_ch * 2 * n_ant * n_time;
    const size_t herk_batch = pol_mode == 'A'
        ? (size_t)n_ch * 2 * n_time_inner
        : (size_t)n_ch * n_time_inner;
    const size_t tri_full_bytes = herk_batch * n_bl_2 * vis_elem_size;
    const size_t tri_reduced_bytes = pol_mode == 'A'
        ? (size_t)Nf_eff * n_bl_2 * vis_elem_size
        : 0;
    const size_t pol_buf_bytes = pol_mode == 'B'
        ? (size_t)n_ch * n_time_inner * n_ant * K * 2  // 2 pols
        : 0;
    const size_t beam_bytes = (size_t)Nf_eff * n_beam * sizeof(float);
    const size_t fb_beam_bytes = (size_t)n_beam * Nf_eff * sizeof(float);
    const size_t total_bytes = qc_bytes + tri_full_bytes + tri_reduced_bytes +
                               pol_buf_bytes + beam_bytes + fb_beam_bytes;

    bool skip = false;
    print_memory_estimate("VisBF", total_bytes, skip);
    if (skip) return;

    printf("  Memory: %.1f MB allocated\n", total_bytes / 1e6);
    printf("  Imaging: Nf_eff=%d  Ng=%d  n_beam=%d  fft=%s\n",
           Nf_eff, Ng, n_beam, fft_prec_fp16 ? "FP16" : "FP32");
    printf("================================================================\n");

    // Allocate
    uint8_t *d_qc = nullptr;
    void *d_tri_full = nullptr, *d_tri_reduced = nullptr;
    float *d_beam_output = nullptr, *d_fb_beam = nullptr;
    uint8_t *d_pol0 = nullptr, *d_pol1 = nullptr;

    CHECK_CUDA(cudaMalloc(&d_qc, qc_bytes));
    CHECK_CUDA(cudaMalloc(&d_tri_full, tri_full_bytes));
    if (pol_mode == 'A') {
        CHECK_CUDA(cudaMalloc(&d_tri_reduced, (size_t)Nf_eff * n_bl_2 * vis_elem_size));
    }
    if (pol_mode == 'B') {
        CHECK_CUDA(cudaMalloc(&d_pol0, (size_t)n_ch * n_time_inner * n_ant * K));
        CHECK_CUDA(cudaMalloc(&d_pol1, (size_t)n_ch * n_time_inner * n_ant * K));
    }
    CHECK_CUDA(cudaMalloc(&d_beam_output, beam_bytes));
    CHECK_CUDA(cudaMalloc(&d_fb_beam, fb_beam_bytes));

    // Random QC data
    generate_random_qc(d_qc, qc_bytes, 123);

    // CUTLASS API
    CutlassComplexGemm api;
    if (ktune > 0) api.set_kernel_tune_verbosity(ktune);
    if (tune_verb >= 0) {
        api.set_strategy_tune_verbosity(tune_verb);
        api.set_gemm_tune_verbosity(tune_verb);
    }

    // Configure imaging pipeline
    imaging_pipeline::ImagingPipeline imager;

    // Generate synthetic imaging config
    std::vector<float> baseline_uv(n_baselines * 2);
    srand(42);
    for (int i = 0; i < n_baselines * 2; ++i)
        baseline_uv[i] = ((float)rand() / RAND_MAX - 0.5f) * 2000.0f;

    std::vector<double> freqs(Nf_eff);
    for (int i = 0; i < Nf_eff; ++i)
        freqs[i] = 700e6 + (1500e6 - 700e6) * i / std::max(Nf_eff - 1, 1);

    std::vector<int> beam_pixels(n_beam * 2);
    for (int i = 0; i < n_beam * 2; ++i)
        beam_pixels[i] = rand() % Ng;

    auto active_fft_prec = fft_prec;
    int img_ret = imager.configure(n_ant, Nf_eff, Ng, n_beam, active_fft_prec, stream);
    if (img_ret != 0 && active_fft_prec == imaging_pipeline::FftPrecision::FP16) {
        printf("  WARNING: FP16 FFT plan failed — falling back to FP32 FFT\n");
        active_fft_prec = imaging_pipeline::FftPrecision::FP32;
        img_ret = imager.configure(n_ant, Nf_eff, Ng, n_beam, active_fft_prec, stream);
    }
    if (img_ret != 0) {
        printf("  ERROR: imaging configure failed (%d) — skipping imaging stages\n", img_ret);
    }
    imager.set_baseline_uv(baseline_uv.data(), n_baselines);
    imager.set_frequencies(freqs.data());
    imager.set_cell_size(1e-4f);
    imager.set_beam_pixels(beam_pixels.data(), n_beam);
    CHECK_CUDA(cudaStreamSynchronize(stream));

    // Enable imaging sub-stage timing
    imager.set_timing_enabled(true);

    // Run iterations
    std::vector<std::vector<float>> all_times;
    std::vector<const char*> stage_names;

    for (int iter = 0; iter < warmup + runs; ++iter) {
        StageTimer timer(stream);

        if (pol_mode == 'A') {
            // Mode A: single HERK + pol reduce
            timer.begin("herk");
            api.herk(d_qc, d_tri_full, n_ant, K,
                     n_ch * 2 * n_time_inner,
                     InputPrecision::INT4, ComputePrecision::FP8,
                     out_prec, OutputFormat::PackedTriangle,
                     1.0f, 0.0f, stream, tune && iter == 0);

            timer.begin("pol_reduce");
            {
                const int64_t total = (int64_t)n_ch * n_time_inner * n_bl_2;
                const int block = 256;
                const int grid = (int)std::min((total + block - 1) / block, (int64_t)1024);
                if (vis_prec_fp16) {
                    pol_reduce_triangle_kernel_fp16<<<grid, block, 0, stream>>>(
                        static_cast<const __half*>(d_tri_full),
                        static_cast<__half*>(d_tri_reduced), n_bl_2, n_ch, n_time_inner);
                } else {
                    pol_reduce_triangle_kernel<<<grid, block, 0, stream>>>(
                        static_cast<const float*>(d_tri_full),
                        static_cast<float*>(d_tri_reduced), n_bl_2, n_ch, n_time_inner);
                }
            }

            timer.begin("imaging");
            imager.grid_and_image(d_tri_reduced, d_beam_output,
                                  vis_prec, stream);

            timer.begin("corner_turn");
            ggp::corner_turn_nf_nb(d_fb_beam, d_beam_output, Nf_eff, n_beam, 1, stream);
            timer.end();

            if (iter == warmup) {
                stage_names = {"herk", "pol_reduce", "img_scatter",
                               "img_fft", "img_beam", "corner_turn"};
            }
        } else {
            // Mode B: pol gather + two HERKs + imaging
            timer.begin("pol_gather");
            {
                const int64_t total = (int64_t)n_ch * n_time_inner * n_ant * K;
                const int block = 256;
                const int grid = (int)std::min((total + block - 1) / block, (int64_t)1024);
                qc_pol_gather_transpose_kernel<<<grid, block, 0, stream>>>(
                    d_qc, d_pol0, d_pol1, n_ant, n_time, n_ch, n_time_inner);
            }

            timer.begin("herk_pol0");
            api.herk(d_pol0, d_tri_full, n_ant, K,
                     n_ch * n_time_inner,
                     InputPrecision::INT4, ComputePrecision::FP8,
                     out_prec, OutputFormat::PackedTriangle,
                     1.0f, 0.0f, stream, tune && iter == 0);

            timer.begin("herk_pol1");
            api.herk(d_pol1, d_tri_full, n_ant, K,
                     n_ch * n_time_inner,
                     InputPrecision::INT4, ComputePrecision::FP8,
                     out_prec, OutputFormat::PackedTriangle,
                     1.0f, 1.0f, stream, tune && iter == 0);

            timer.begin("imaging");
            imager.grid_and_image(d_tri_full, d_beam_output,
                                  vis_prec, stream);

            timer.begin("corner_turn");
            ggp::corner_turn_nf_nb(d_fb_beam, d_beam_output, Nf_eff, n_beam, 1, stream);
            timer.end();

            if (iter == warmup) {
                stage_names = {"pol_gather", "herk_pol0", "herk_pol1",
                               "img_scatter", "img_fft", "img_beam",
                               "corner_turn"};
            }
        }

        auto raw_times = timer.collect();
        if (iter >= warmup) {
            // Expand the "imaging" timer stage into sub-stages from
            // the imaging pipeline's internal CUDA event timing.
            auto img = imager.get_stage_times();
            const int img_idx = (pol_mode == 'A') ? 2 : 3;
            std::vector<float> expanded;
            for (int i = 0; i < (int)raw_times.size(); ++i) {
                if (i == img_idx) {
                    expanded.push_back(img.scatter_ms);
                    expanded.push_back(img.fft_ms);
                    expanded.push_back(img.beam_ms);
                } else {
                    expanded.push_back(raw_times[i]);
                }
            }
            all_times.push_back(expanded);
        }
        timer.reset();
    }

    print_timing_table(stage_names, all_times, runs);

    // HERK TFLOPS and pipeline throughput
    {
        float total_mean = 0;
        for (auto& t : all_times) for (float v : t) total_mean += v;
        total_mean /= runs;
        if (total_mean > 0) {
            int herk_batch = pol_mode == 'A' ? n_ch * 2 * n_time_inner : n_ch * n_time_inner;
            // HERK FLOPs: batch * 8 * N * N * K (complex FP8, 4 real sub-GEMMs × 2 FMA)
            double herk_flops = (double)herk_batch * 8.0 * n_ant * n_ant * K;
            double tflops = herk_flops / (total_mean * 1e9);
            double throughput = 1000.0 / total_mean;
            double beams_per_ch_sec = (double)n_beam * throughput;
            printf("  TFLOPS: %.1f  |  Throughput: %.1f payloads/sec  |  Beams/ch/sec: %.2e\n",
                   tflops, throughput, beams_per_ch_sec);
        }
    }

    // Cleanup
    cudaFree(d_qc); cudaFree(d_tri_full);
    if (d_tri_reduced) cudaFree(d_tri_reduced);
    if (d_pol0) cudaFree(d_pol0);
    if (d_pol1) cudaFree(d_pol1);
    cudaFree(d_beam_output); cudaFree(d_fb_beam);
}

// ========================================================================
// Dedispersion Benchmark
// ========================================================================

static void bench_dedisp(
    int n_beam, int n_ch, int n_time_out, int n_dm,
    const char* dedisp_mode_str,
    float f_min_mhz, float f_max_mhz, float max_dm,
    int warmup, int runs,
    int ktune,
    cudaStream_t stream)
{
    using namespace dedisp_api;

    printf("\n================================================================\n");
    printf("  Dedispersion Benchmark\n");
    printf("  n_beam=%d  n_ch=%d  n_time=%d  n_dm=%d\n",
           n_beam, n_ch, n_time_out, n_dm);
    printf("  f_min=%.1f MHz  f_max=%.1f MHz  max_dm=%.0f\n",
           f_min_mhz, f_max_mhz, max_dm);
    printf("================================================================\n");

    // Determine which modes to run
    bool run_cublas = (strcmp(dedisp_mode_str, "cublas") == 0 ||
                       strcmp(dedisp_mode_str, "both") == 0);
    bool run_cutlass = (strcmp(dedisp_mode_str, "cutlass") == 0 ||
                        strcmp(dedisp_mode_str, "both") == 0);

    // Input filterbank: [n_beam x n_ch x n_time_out]
    const size_t fb_bytes = (size_t)n_beam * n_ch * n_time_out * sizeof(float);
    // Output: [n_beam x n_dm x n_time_out]
    const size_t dd_bytes = (size_t)n_beam * n_dm * n_time_out * sizeof(float);
    const size_t total_bytes = fb_bytes + dd_bytes * 2;  // two modes

    bool skip = false;
    print_memory_estimate("Dedisp", total_bytes, skip);
    if (skip) return;

    float *d_fb = nullptr, *d_dd_cublas = nullptr, *d_dd_cutlass = nullptr;
    CHECK_CUDA(cudaMalloc(&d_fb, fb_bytes));
    if (run_cublas) CHECK_CUDA(cudaMalloc(&d_dd_cublas, dd_bytes));
    if (run_cutlass) CHECK_CUDA(cudaMalloc(&d_dd_cutlass, dd_bytes));

    // Random filterbank data
    generate_noise_filterbank(d_fb, (int64_t)n_beam * n_ch * n_time_out,
                               100.0f, 10.0f, 999);

    struct ModeResult {
        const char* name;
        float min_ms, mean_ms, std_ms;
    };
    std::vector<ModeResult> results;

    auto run_mode = [&](ComputeMode mode, const char* name, float* d_out) {
        DedispConfig cfg;
        cfg.Nf = n_ch;
        cfg.Nt = n_time_out;
        cfg.Ndm = n_dm;
        cfg.f_min_MHz = f_min_mhz;
        cfg.f_max_MHz = f_max_mhz;
        cfg.max_dm = max_dm;
        cfg.total_obs_time_s = 1.0f;
        cfg.compute_mode = mode;
        cfg.max_batch_size = n_beam;
        cfg.kernel_tune_verbosity = ktune;

        DedispPipeline pipeline(cfg);
        int ret = pipeline.initialize(stream);
        if (ret != 0) {
            printf("  %s: initialize FAILED (%d) — SKIPPING\n", name, ret);
            return;
        }

        std::vector<float> times;
        for (int iter = 0; iter < warmup + runs; ++iter) {
            cudaEvent_t ev_start, ev_stop;
            CHECK_CUDA(cudaEventCreate(&ev_start));
            CHECK_CUDA(cudaEventCreate(&ev_stop));

            CHECK_CUDA(cudaEventRecord(ev_start, stream));
            pipeline.dedisperse(d_fb, d_out, n_beam, stream);
            CHECK_CUDA(cudaEventRecord(ev_stop, stream));
            CHECK_CUDA(cudaEventSynchronize(ev_stop));

            float ms = 0;
            CHECK_CUDA(cudaEventElapsedTime(&ms, ev_start, ev_stop));
            if (iter >= warmup) times.push_back(ms);

            cudaEventDestroy(ev_start);
            cudaEventDestroy(ev_stop);
        }

        float min_ms = *std::min_element(times.begin(), times.end());
        float mean_ms = std::accumulate(times.begin(), times.end(), 0.0f) / runs;
        float var = 0;
        for (float t : times) { float d = t - mean_ms; var += d * d; }
        float std_ms = sqrtf(var / runs);

        results.push_back({name, min_ms, mean_ms, std_ms});
    };

    if (run_cublas) {
        run_mode(ComputeMode::CuBLAS_FP32, "CuBLAS_FP32", d_dd_cublas);
    }
    if (run_cutlass) {
        run_mode(ComputeMode::CUTLASS_FP8, "CUTLASS_FP8", d_dd_cutlass);
    }

    // Print results
    printf("\n  %-18s %10s %11s %10s\n", "Mode", "Min (ms)", "Mean (ms)", "Std (ms)");
    printf("  %-18s %10s %11s %10s\n", "------------------", "--------", "---------", "--------");
    for (auto& r : results) {
        printf("  %-18s %10.2f %11.2f %10.2f\n",
               r.name, r.min_ms, r.mean_ms, r.std_ms);
    }
    if (results.size() == 2 && results[1].mean_ms > 0) {
        printf("  Speedup: %.2fx\n", results[0].mean_ms / results[1].mean_ms);
    }

    // Print TFLOPS and throughput for the best result (CUTLASS if available, else CuBLAS)
    const auto& best = results.back();
    if (best.mean_ms > 0) {
        // GEMM: M=n_time, N=n_dm, K=n_ch, batch=n_beam, 2 FLOPs per MAC
        double gemm_flops = 2.0 * (double)n_time_out * n_dm * n_ch * n_beam;
        double tflops = gemm_flops / (best.mean_ms * 1e9);
        double throughput = 1000.0 * n_beam / best.mean_ms;
        printf("  TFLOPS: %.1f\n", tflops);
        printf("  Throughput: %.1f beams/s\n", throughput);
    }

    // Cleanup
    cudaFree(d_fb);
    if (d_dd_cublas) cudaFree(d_dd_cublas);
    if (d_dd_cutlass) cudaFree(d_dd_cutlass);
}

// ========================================================================
// Pipelined Benchmarks: N payloads with P-deep concurrent streams
// ========================================================================

/// Wall-clock helper (multi-stream timing can't use single-stream CUDA events)
static double wall_ms() {
    return std::chrono::duration<double, std::milli>(
        std::chrono::high_resolution_clock::now().time_since_epoch()).count();
}

static void bench_voltbf_pipelined(
    int n_ant, int n_beam, int n_ch, int n_time, int n_tps,
    int n_payloads, int pipeline_depth, int ktune)
{
    using namespace cutlass_gemm_api;
    const int P = std::min(pipeline_depth, n_payloads);
    const int n_time_out = n_time / n_tps;

    printf("\n================================================================\n");
    printf("  VoltBF Pipelined: %d payloads, %d-deep pipeline\n", n_payloads, P);
    printf("  GEMM: M=%d N=%d K=%d batch=%d\n", n_time, n_beam, n_ant, n_ch);

    const size_t qc_bytes = (size_t)n_ch * 2 * n_ant * n_time;
    const size_t int4_bytes = (size_t)n_ch * n_time * n_ant;  // per pol (1 byte/complex)
    const size_t power_bytes = (size_t)n_ch * n_time * n_beam * sizeof(float);
    const size_t fb_bytes = (size_t)n_ch * n_beam * n_time_out * sizeof(float);
    const size_t wt_bytes = (size_t)n_ch * n_beam * n_ant * 2 * sizeof(float);
    const size_t per_stream_bytes = int4_bytes * 2 + power_bytes + fb_bytes +
                                     n_ch * n_beam * n_ant * 2;  // prepared B ~FP8
    const size_t total_bytes = qc_bytes + fb_bytes + wt_bytes +
                               (size_t)n_ch * n_beam * n_ant * sizeof(__half) * 2 +
                               per_stream_bytes * P;
    bool skip = false;
    print_memory_estimate("VoltBF-Pipeline", total_bytes, skip);
    if (skip) return;
    printf("================================================================\n");

    // Shared
    uint8_t* d_qc; float* d_fb_beam;
    CHECK_CUDA(cudaMalloc(&d_qc, qc_bytes));
    CHECK_CUDA(cudaMalloc(&d_fb_beam, fb_bytes));
    generate_random_qc(d_qc, qc_bytes, 123);

    // Weights: FP32 -> FP16 -> prepare_b per stream -> free source
    float* d_wt; __half *d_wt_re, *d_wt_im;
    CHECK_CUDA(cudaMalloc(&d_wt, wt_bytes));
    CHECK_CUDA(cudaMalloc(&d_wt_re, (size_t)n_ch * n_beam * n_ant * sizeof(__half)));
    CHECK_CUDA(cudaMalloc(&d_wt_im, (size_t)n_ch * n_beam * n_ant * sizeof(__half)));
    generate_random_weights(d_wt, (int64_t)n_ch * n_beam * n_ant, 42);
    {
        int64_t nc = (int64_t)n_ch * n_beam * n_ant;
        int grid = (int)std::min((nc + 255) / 256, (int64_t)1024);
        deinterleave_fp32_to_fp16_kernel<<<grid, 256>>>(d_wt, d_wt_re, d_wt_im, nc);
        CHECK_CUDA(cudaDeviceSynchronize());
    }
    cudaFree(d_wt);

    // Per-stream contexts
    struct Ctx {
        cudaStream_t stream; cudaEvent_t done;
        CutlassComplexGemm* api;
        uint8_t *int4_p0, *int4_p1; float *power, *fb_chan;
    };
    std::vector<Ctx> ctx(P);
    for (int i = 0; i < P; ++i) {
        CHECK_CUDA(cudaStreamCreate(&ctx[i].stream));
        CHECK_CUDA(cudaEventCreate(&ctx[i].done));
        ctx[i].api = new CutlassComplexGemm();
        if (ktune > 0) ctx[i].api->set_kernel_tune_verbosity(i == 0 ? ktune : 0);
        CHECK_CUDA(cudaMalloc(&ctx[i].int4_p0, int4_bytes));
        CHECK_CUDA(cudaMalloc(&ctx[i].int4_p1, int4_bytes));
        CHECK_CUDA(cudaMalloc(&ctx[i].power, power_bytes));
        CHECK_CUDA(cudaMalloc(&ctx[i].fb_chan, fb_bytes));
        ctx[i].api->prepare_b(d_wt_re, d_wt_im, n_beam, n_ant, n_ch,
                               ComputePrecision::FP8, ctx[i].stream);
        CHECK_CUDA(cudaStreamSynchronize(ctx[i].stream));
    }
    cudaFree(d_wt_re); cudaFree(d_wt_im);

    auto run_payload = [&](Ctx& c) {
        int64_t total = (int64_t)n_ch * n_time * n_ant;
        int grid = (int)std::min((total + 255) / 256, (int64_t)1024);
        qc_transpose_polsplit_kernel<<<grid, 256, 0, c.stream>>>(
            d_qc, c.int4_p0, c.int4_p1, n_ant, n_time, n_ch);
        c.api->gemm_prepared_power_int4(c.int4_p0, c.power,
            n_time, n_beam, n_ant, n_ch, 1.0f, 0.0f, c.stream);
        c.api->gemm_prepared_power_int4(c.int4_p1, c.power,
            n_time, n_beam, n_ant, n_ch, 1.0f, 1.0f, c.stream);
        int64_t ti = (int64_t)n_ch * n_beam * n_time_out;
        int tg = (int)std::min((ti + 255) / 256, (int64_t)1024);
        time_integrate_fp32_kernel<<<tg, 256, 0, c.stream>>>(
            c.power, c.fb_chan, n_beam, n_time, n_ch, n_tps);
        ggp::corner_turn_nf_nb(d_fb_beam, c.fb_chan, n_ch, n_beam, n_time_out, c.stream);
    };

    // Warmup
    for (int i = 0; i < P; ++i) run_payload(ctx[i]);
    CHECK_CUDA(cudaDeviceSynchronize());

    // Serial
    CHECK_CUDA(cudaDeviceSynchronize());
    double t0 = wall_ms();
    for (int p = 0; p < n_payloads; ++p) run_payload(ctx[0]);
    CHECK_CUDA(cudaDeviceSynchronize());
    float serial_ms = (float)(wall_ms() - t0);

    // Pipelined
    CHECK_CUDA(cudaDeviceSynchronize());
    double t2 = wall_ms();
    for (int p = 0; p < n_payloads; ++p) {
        int s = p % P;
        if (p >= P) CHECK_CUDA(cudaEventSynchronize(ctx[s].done));
        run_payload(ctx[s]);
        CHECK_CUDA(cudaEventRecord(ctx[s].done, ctx[s].stream));
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    float pipe_ms = (float)(wall_ms() - t2);

    double gemm_flops = 2.0 * 8.0 * n_time * (double)n_beam * n_ant * n_ch * n_payloads;
    printf("\n  %-16s %10s %10s %12s %10s\n",
           "Mode", "Total (ms)", "Per-PL", "Throughput", "TFLOPS");
    printf("  %-16s %10s %10s %12s %10s\n",
           "----------------", "--------", "--------", "----------", "--------");
    printf("  %-16s %10.1f %10.2f %10.1f Hz %10.1f\n",
           "Serial", serial_ms, serial_ms / n_payloads,
           1000.0 * n_payloads / serial_ms,
           gemm_flops / (serial_ms * 1e-3) / 1e12);
    printf("  %-16s %10.1f %10.2f %10.1f Hz %10.1f\n",
           "Pipelined", pipe_ms, pipe_ms / n_payloads,
           1000.0 * n_payloads / pipe_ms,
           gemm_flops / (pipe_ms * 1e-3) / 1e12);
    printf("  Speedup: %.2fx\n", serial_ms / pipe_ms);

    cudaFree(d_qc); cudaFree(d_fb_beam);
    for (auto& c : ctx) {
        cudaFree(c.int4_p0); cudaFree(c.int4_p1);
        cudaFree(c.power); cudaFree(c.fb_chan);
        delete c.api; cudaStreamDestroy(c.stream); cudaEventDestroy(c.done);
    }
}

static void bench_visbf_pipelined(
    int n_ant, int n_beam, int n_ch, int n_time, int n_time_inner,
    int Ng, char pol_mode,
    int n_payloads, int pipeline_depth, int ktune,
    bool vis_prec_fp16, bool fft_prec_fp16)
{
    using namespace cutlass_gemm_api;
    const int K = n_time / n_time_inner;
    const int n_baselines = n_ant * (n_ant + 1) / 2;
    const int n_bl_2 = n_baselines * 2;
    const int Nf_eff = n_ch * n_time_inner;
    const int herk_batch = pol_mode == 'A'
        ? n_ch * 2 * n_time_inner : n_ch * n_time_inner;
    const size_t vis_elem_size = vis_prec_fp16 ? sizeof(__half) : sizeof(float);
    const auto out_prec = vis_prec_fp16 ? OutputPrecision::FP16 : OutputPrecision::FP32;
    const auto vis_prec = vis_prec_fp16
        ? imaging_pipeline::VisPrecision::FP16
        : imaging_pipeline::VisPrecision::FP32;
    const auto fft_prec = fft_prec_fp16
        ? imaging_pipeline::FftPrecision::FP16
        : imaging_pipeline::FftPrecision::FP32;

    // Memory-limit pipeline depth
    size_t tri_bytes = (size_t)herk_batch * n_bl_2 * vis_elem_size;
    size_t tri_red_bytes = pol_mode == 'A' ? (size_t)Nf_eff * n_bl_2 * vis_elem_size : 0;
    size_t beam_bytes = (size_t)Nf_eff * n_beam * sizeof(float);
    size_t per_stream = tri_bytes + tri_red_bytes + beam_bytes + 2ULL * 1024 * 1024 * 1024;  // ~2GB imaging
    size_t free_mem = 0, total_mem = 0;
    cudaMemGetInfo(&free_mem, &total_mem);
    size_t overhead = 4ULL * 1024 * 1024 * 1024;  // 4GB OS/driver
    int max_P = std::max(1, (int)((free_mem > overhead ? free_mem - overhead : 0) / per_stream));
    int P = std::min({pipeline_depth, n_payloads, max_P});

    printf("\n================================================================\n");
    printf("  VisBF Pipelined: %d payloads, %d-deep pipeline (pol_mode=%c)\n",
           n_payloads, P, pol_mode);
    if (P < pipeline_depth)
        printf("  (limited from %d by memory: %.1f GB/stream, %.1f GB free)\n",
               pipeline_depth, per_stream / 1e9, free_mem / 1e9);
    printf("  HERK: N=%d K=%d batch=%d  output=%s  |  Imaging: Nf=%d Ng=%d fft=%s\n",
           n_ant, K, herk_batch, vis_prec_fp16 ? "FP16" : "FP32",
           Nf_eff, Ng, fft_prec_fp16 ? "FP16" : "FP32");
    printf("================================================================\n");

    size_t qc_bytes = (size_t)n_ch * 2 * n_ant * n_time;
    uint8_t* d_qc;
    CHECK_CUDA(cudaMalloc(&d_qc, qc_bytes));
    generate_random_qc(d_qc, qc_bytes, 123);

    // Imaging setup data (shared across instances)
    std::vector<float> baseline_uv(n_baselines * 2);
    srand(42);
    for (int i = 0; i < n_baselines * 2; ++i)
        baseline_uv[i] = ((float)rand() / RAND_MAX - 0.5f) * 2000.0f;
    std::vector<double> freqs(Nf_eff);
    for (int i = 0; i < Nf_eff; ++i)
        freqs[i] = 700e6 + (1500e6 - 700e6) * i / std::max(Nf_eff - 1, 1);
    std::vector<int> beam_pixels(n_beam * 2);
    for (int i = 0; i < n_beam * 2; ++i)
        beam_pixels[i] = rand() % Ng;

    struct Ctx {
        cudaStream_t stream; cudaEvent_t done;
        CutlassComplexGemm* api;
        imaging_pipeline::ImagingPipeline* imager;
        void *tri_full, *tri_reduced;
        float *beam_out, *fb_beam;
    };
    std::vector<Ctx> ctx(P);
    printf("  Setting up %d pipeline slots...\n", P);
    for (int i = 0; i < P; ++i) {
        CHECK_CUDA(cudaStreamCreate(&ctx[i].stream));
        CHECK_CUDA(cudaEventCreate(&ctx[i].done));
        ctx[i].api = new CutlassComplexGemm();
        if (ktune > 0) ctx[i].api->set_kernel_tune_verbosity(i == 0 ? ktune : 0);
        CHECK_CUDA(cudaMalloc(&ctx[i].tri_full, tri_bytes));
        ctx[i].tri_reduced = nullptr;
        if (pol_mode == 'A')
            CHECK_CUDA(cudaMalloc(&ctx[i].tri_reduced, tri_red_bytes));
        CHECK_CUDA(cudaMalloc(&ctx[i].beam_out, beam_bytes));
        CHECK_CUDA(cudaMalloc(&ctx[i].fb_beam, beam_bytes));

        ctx[i].imager = new imaging_pipeline::ImagingPipeline();
        {
            auto slot_fft = fft_prec;
            int ret = ctx[i].imager->configure(n_ant, Nf_eff, Ng, n_beam,
                                                slot_fft, ctx[i].stream);
            if (ret != 0 && slot_fft == imaging_pipeline::FftPrecision::FP16) {
                if (i == 0) printf("  WARNING: FP16 FFT failed — falling back to FP32\n");
                slot_fft = imaging_pipeline::FftPrecision::FP32;
                ctx[i].imager->configure(n_ant, Nf_eff, Ng, n_beam,
                                          slot_fft, ctx[i].stream);
            }
        }
        ctx[i].imager->set_baseline_uv(baseline_uv.data(), n_baselines);
        ctx[i].imager->set_frequencies(freqs.data());
        ctx[i].imager->set_cell_size(1e-4f);
        ctx[i].imager->set_beam_pixels(beam_pixels.data(), n_beam);
        CHECK_CUDA(cudaStreamSynchronize(ctx[i].stream));
        printf("    Slot %d ready\n", i);
    }

    auto run_payload = [&](Ctx& c) {
        if (pol_mode == 'A') {
            c.api->herk(d_qc, c.tri_full, n_ant, K, herk_batch,
                        InputPrecision::INT4, ComputePrecision::FP8,
                        out_prec, OutputFormat::PackedTriangle,
                        1.0f, 0.0f, c.stream, false);
            int64_t total = (int64_t)n_ch * n_time_inner * n_bl_2;
            int grid = (int)std::min((total + 255) / 256, (int64_t)1024);
            if (vis_prec_fp16) {
                pol_reduce_triangle_kernel_fp16<<<grid, 256, 0, c.stream>>>(
                    static_cast<const __half*>(c.tri_full),
                    static_cast<__half*>(c.tri_reduced), n_bl_2, n_ch, n_time_inner);
            } else {
                pol_reduce_triangle_kernel<<<grid, 256, 0, c.stream>>>(
                    static_cast<const float*>(c.tri_full),
                    static_cast<float*>(c.tri_reduced), n_bl_2, n_ch, n_time_inner);
            }
            c.imager->grid_and_image(c.tri_reduced, c.beam_out,
                                      vis_prec, c.stream);
        } else {
            c.api->herk(d_qc, c.tri_full, n_ant, K, herk_batch,
                        InputPrecision::INT4, ComputePrecision::FP8,
                        out_prec, OutputFormat::PackedTriangle,
                        1.0f, 0.0f, c.stream, false);
            c.imager->grid_and_image(c.tri_full, c.beam_out,
                                      vis_prec, c.stream);
        }
        ggp::corner_turn_nf_nb(c.fb_beam, c.beam_out, Nf_eff, n_beam, 1, c.stream);
    };

    // Warmup
    run_payload(ctx[0]);
    CHECK_CUDA(cudaDeviceSynchronize());

    // Serial
    CHECK_CUDA(cudaDeviceSynchronize());
    double t0 = wall_ms();
    for (int p = 0; p < n_payloads; ++p) run_payload(ctx[0]);
    CHECK_CUDA(cudaDeviceSynchronize());
    float serial_ms = (float)(wall_ms() - t0);

    // Pipelined
    CHECK_CUDA(cudaDeviceSynchronize());
    double t2 = wall_ms();
    for (int p = 0; p < n_payloads; ++p) {
        int s = p % P;
        if (p >= P) CHECK_CUDA(cudaEventSynchronize(ctx[s].done));
        run_payload(ctx[s]);
        CHECK_CUDA(cudaEventRecord(ctx[s].done, ctx[s].stream));
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    float pipe_ms = (float)(wall_ms() - t2);

    printf("\n  %-16s %10s %12s %12s\n", "Mode", "Total (ms)", "Per-PL (ms)", "Throughput");
    printf("  %-16s %10s %12s %12s\n",
           "----------------", "--------", "----------", "----------");
    printf("  %-16s %10.1f %12.2f %10.2f Hz\n",
           "Serial", serial_ms, serial_ms / n_payloads, 1000.0 * n_payloads / serial_ms);
    printf("  %-16s %10.1f %12.2f %10.2f Hz\n",
           "Pipelined", pipe_ms, pipe_ms / n_payloads, 1000.0 * n_payloads / pipe_ms);
    printf("  Speedup: %.2fx\n", serial_ms / pipe_ms);

    cudaFree(d_qc);
    for (auto& c : ctx) {
        cudaFree(c.tri_full);
        if (c.tri_reduced) cudaFree(c.tri_reduced);
        cudaFree(c.beam_out); cudaFree(c.fb_beam);
        delete c.api; delete c.imager;
        cudaStreamDestroy(c.stream); cudaEventDestroy(c.done);
    }
}

static void bench_dedisp_pipelined(
    int n_beam, int n_ch, int n_time_out, int n_dm,
    float f_min_mhz, float f_max_mhz, float max_dm,
    int n_payloads, int pipeline_depth, int ktune)
{
    using namespace dedisp_api;
    int P = std::min(pipeline_depth, n_payloads);

    printf("\n================================================================\n");
    printf("  Dedisp Pipelined: %d payloads, %d-deep pipeline\n", n_payloads, P);
    printf("  n_beam=%d  n_ch=%d  n_time=%d  n_dm=%d\n", n_beam, n_ch, n_time_out, n_dm);
    printf("================================================================\n");

    const size_t fb_bytes = (size_t)n_beam * n_ch * n_time_out * sizeof(float);
    const size_t dd_bytes = (size_t)n_beam * n_dm * n_time_out * sizeof(float);

    // Shared input filterbank
    float* d_fb;
    CHECK_CUDA(cudaMalloc(&d_fb, fb_bytes));
    generate_noise_filterbank(d_fb, (int64_t)n_beam * n_ch * n_time_out, 100.0f, 10.0f, 999);

    struct Ctx {
        cudaStream_t stream; cudaEvent_t done;
        DedispPipeline* pipeline;
        float* dd;
    };
    std::vector<Ctx> ctx(P);
    printf("  Setting up %d pipeline slots...\n", P);
    for (int i = 0; i < P; ++i) {
        CHECK_CUDA(cudaStreamCreate(&ctx[i].stream));
        CHECK_CUDA(cudaEventCreate(&ctx[i].done));
        CHECK_CUDA(cudaMalloc(&ctx[i].dd, dd_bytes));

        DedispConfig cfg;
        cfg.Nf = n_ch; cfg.Nt = n_time_out; cfg.Ndm = n_dm;
        cfg.f_min_MHz = f_min_mhz; cfg.f_max_MHz = f_max_mhz;
        cfg.max_dm = max_dm; cfg.total_obs_time_s = 1.0f;
        cfg.compute_mode = ComputeMode::CUTLASS_FP8;
        cfg.max_batch_size = n_beam;
        cfg.kernel_tune_verbosity = (i == 0 ? ktune : 0);
        ctx[i].pipeline = new DedispPipeline(cfg);
        int ret = ctx[i].pipeline->initialize(ctx[i].stream);
        if (ret != 0) {
            printf("  Slot %d: init FAILED (%d)\n", i, ret);
            // Fall back to fewer slots
            P = i; break;
        }
        printf("    Slot %d ready\n", i);
    }
    if (P == 0) { printf("  No slots initialized — SKIPPING\n"); cudaFree(d_fb); return; }
    CHECK_CUDA(cudaDeviceSynchronize());

    auto run_payload = [&](Ctx& c) {
        c.pipeline->dedisperse(d_fb, c.dd, n_beam, c.stream);
    };

    // Warmup
    run_payload(ctx[0]);
    CHECK_CUDA(cudaDeviceSynchronize());

    // Serial
    CHECK_CUDA(cudaDeviceSynchronize());
    double t0 = wall_ms();
    for (int p = 0; p < n_payloads; ++p) run_payload(ctx[0]);
    CHECK_CUDA(cudaDeviceSynchronize());
    float serial_ms = (float)(wall_ms() - t0);

    // Pipelined
    CHECK_CUDA(cudaDeviceSynchronize());
    double t2 = wall_ms();
    for (int p = 0; p < n_payloads; ++p) {
        int s = p % P;
        if (p >= P) CHECK_CUDA(cudaEventSynchronize(ctx[s].done));
        run_payload(ctx[s]);
        CHECK_CUDA(cudaEventRecord(ctx[s].done, ctx[s].stream));
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    float pipe_ms = (float)(wall_ms() - t2);

    printf("\n  %-16s %10s %12s %12s\n", "Mode", "Total (ms)", "Per-PL (ms)", "Throughput");
    printf("  %-16s %10s %12s %12s\n",
           "----------------", "--------", "----------", "----------");
    printf("  %-16s %10.1f %12.2f %10.2f Hz\n",
           "Serial", serial_ms, serial_ms / n_payloads, 1000.0 * n_payloads / serial_ms);
    printf("  %-16s %10.1f %12.2f %10.2f Hz\n",
           "Pipelined", pipe_ms, pipe_ms / n_payloads, 1000.0 * n_payloads / pipe_ms);
    printf("  Speedup: %.2fx\n", serial_ms / pipe_ms);

    cudaFree(d_fb);
    for (int i = 0; i < P; ++i) {
        cudaFree(ctx[i].dd);
        delete ctx[i].pipeline;
        cudaStreamDestroy(ctx[i].stream); cudaEventDestroy(ctx[i].done);
    }
}

// ========================================================================
// CLI parsing and main
// ========================================================================

static void print_usage() {
    printf("Usage: bench_pipeline [suite] [key=value ...]\n\n");
    printf("Suites:\n");
    printf("  voltbf    Voltage beamformer only\n");
    printf("  visbf     Visibility beamformer only\n");
    printf("  dedisp    Dedispersion only\n");
    printf("  all       All three (default)\n\n");
    printf("Dimension keys:\n");
    printf("  n_ant=1664          Antennas\n");
    printf("  n_beam=4000         Output beams (VoltBF/Dedisp batch dim)\n");
    printf("  n_ch=1600           Frequency channels (Band 2)\n");
    printf("  n_time=9000         Time samples per payload\n");
    printf("  n_tps=1             Time power sum factor\n");
    printf("  n_time_inner=2      HERK time factoring (1,2,4,8)\n");
    printf("  Ng=4096             UV grid size for imaging\n");
    printf("  n_dm=2000           DM trials\n");
    printf("  f_min_mhz=1312      Min frequency (MHz)\n");
    printf("  f_max_mhz=1508      Max frequency (MHz)\n");
    printf("  max_dm=2600         Max dispersion measure\n\n");
    printf("VisBF keys:\n");
    printf("  pol_mode=A          A (single HERK+reduce) or B (two HERKs)\n");
    printf("  vis_prec=fp16       HERK output precision: fp16 (default) or fp32\n");
    printf("  fft_prec=fp16       FFT precision: fp16 (default) or fp32\n\n");
    printf("Dedisp keys:\n");
    printf("  dedisp_mode=both    cublas, cutlass, or both\n\n");
    printf("Timing keys:\n");
    printf("  warmup=3            Warmup iterations\n");
    printf("  runs=10             Timed iterations\n\n");
    printf("Fusion keys (VoltBF short-integration):\n");
    printf("  ch_fuse=1           Channel fusion factor (1/2/4/8)\n");
    printf("  n_batch=1           Payload batch count (>1 batches along M)\n\n");
    printf("Multi-stream keys:\n");
    printf("  streams=0           0=single-stream, 1=multi-stream, 2=both\n");
    printf("  ch_tile=32          Channel tile size for multi-stream VoltBF\n\n");
    printf("Pipeline keys:\n");
    printf("  n_payloads=32       Payloads to process (0=skip pipelined)\n");
    printf("  pipeline=8          Pipeline depth (concurrent streams)\n\n");
    printf("Tuning keys:\n");
    printf("  tune=true           Enable HERK/GEMM strategy autotuning (default: on)\n");
    printf("  tune_verb=1         Strategy tune verbosity\n");
    printf("  ktune=0             Kernel tune verbosity\n");
}

enum Suite { ALL, VOLTBF, VISBF, DEDISP_SUITE };

int main(int argc, char* argv[]) {
    // Defaults
    Suite suite = ALL;
    // Defaults sized for ~12 GB GPUs (e.g. GB10 Spark).
    // Production DSA-2000: n_beam=4000 n_ch=1600 n_time=9000 n_dm=2000
    // (require 85+ GB for weights alone; use run_benchmarks.sh for production sweeps).
    int n_ant = 1664, n_beam = 256, n_ch = 32, n_time = 256;
    int n_tps = 1, n_time_inner = 2, Ng = 4096, n_dm = 512;
    float f_min_mhz = 1312.0f, f_max_mhz = 1508.0f, max_dm = 2600.0f;
    char pol_mode = 'A';
    const char* dedisp_mode = "both";
    int warmup = 3, runs = 10;
    bool tune = true;
    int tune_verb = 1, ktune = 0;
    int streams_mode = 0;   // 0=single, 1=multi, 2=both
    int ch_tile = 32;
    int n_payloads = 0;     // 0 = skip pipelined benchmarks
    int pipeline = 8;
    int ch_fuse = 1;        // Channel fusion factor (1/2/4/8)
    int n_batch = 1;        // Payload batching along M (for fused benchmark)
    bool vis_prec_fp16 = true;   // default: FP16 HERK output (production)
    bool fft_prec_fp16 = true;   // default: FP16 FFT (production)

    for (int i = 1; i < argc; ++i) {
        std::string arg(argv[i]);

        // Positional suite
        if (arg == "voltbf" || arg == "volt")    { suite = VOLTBF; continue; }
        if (arg == "visbf" || arg == "vis")      { suite = VISBF; continue; }
        if (arg == "dedisp")                     { suite = DEDISP_SUITE; continue; }
        if (arg == "all")                        { suite = ALL; continue; }
        if (arg == "--help" || arg == "-h")      { print_usage(); return 0; }

        // key=value
        auto eq = arg.find('=');
        if (eq != std::string::npos) {
            std::string key = arg.substr(0, eq);
            std::string val = arg.substr(eq + 1);

            if      (key == "n_ant")         n_ant = atoi(val.c_str());
            else if (key == "n_beam")        n_beam = atoi(val.c_str());
            else if (key == "n_ch")          n_ch = atoi(val.c_str());
            else if (key == "n_time")        n_time = atoi(val.c_str());
            else if (key == "n_tps")         n_tps = atoi(val.c_str());
            else if (key == "n_time_inner")  n_time_inner = atoi(val.c_str());
            else if (key == "Ng")            Ng = atoi(val.c_str());
            else if (key == "n_dm")          n_dm = atoi(val.c_str());
            else if (key == "f_min_mhz")    f_min_mhz = atof(val.c_str());
            else if (key == "f_max_mhz")    f_max_mhz = atof(val.c_str());
            else if (key == "max_dm")       max_dm = atof(val.c_str());
            else if (key == "pol_mode")      pol_mode = val[0];
            else if (key == "dedisp_mode")   dedisp_mode = argv[i] + eq + 1;
            else if (key == "warmup")        warmup = atoi(val.c_str());
            else if (key == "runs")          runs = atoi(val.c_str());
            else if (key == "tune")          tune = (val == "1" || val == "true");
            else if (key == "tune_verb")     tune_verb = atoi(val.c_str());
            else if (key == "ktune")         ktune = atoi(val.c_str());
            else if (key == "streams")       streams_mode = atoi(val.c_str());
            else if (key == "ch_tile")       ch_tile = atoi(val.c_str());
            else if (key == "n_payloads")    n_payloads = atoi(val.c_str());
            else if (key == "pipeline")      pipeline = atoi(val.c_str());
            else if (key == "ch_fuse")       ch_fuse = atoi(val.c_str());
            else if (key == "n_batch")       n_batch = atoi(val.c_str());
            else if (key == "vis_prec")      vis_prec_fp16 = (val == "fp16" || val == "FP16");
            else if (key == "fft_prec")      fft_prec_fp16 = (val == "fp16" || val == "FP16");
            else {
                fprintf(stderr, "Unknown key: %s\n", key.c_str());
                return 1;
            }
        }
    }

    // Validate
    if (n_time % n_tps != 0) {
        fprintf(stderr, "n_time (%d) must be divisible by n_tps (%d)\n", n_time, n_tps);
        return 1;
    }
    if (n_time % n_time_inner != 0) {
        fprintf(stderr, "n_time (%d) must be divisible by n_time_inner (%d)\n",
                n_time, n_time_inner);
        return 1;
    }
    if (pol_mode != 'A' && pol_mode != 'B') {
        fprintf(stderr, "pol_mode must be A or B, got '%c'\n", pol_mode);
        return 1;
    }

    // Print GPU info
    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, 0));
    printf("GPU: %s  (SM %d.%d, %d SMs, %.0f MB)\n",
           prop.name, prop.major, prop.minor,
           prop.multiProcessorCount,
           prop.totalGlobalMem / 1e6);

    cudaStream_t stream;
    CHECK_CUDA(cudaStreamCreate(&stream));

    const int n_time_out = n_time / n_tps;

    if (suite == ALL || suite == VOLTBF) {
        if (streams_mode == 0 || streams_mode == 2) {
            bench_voltbf(n_ant, n_beam, n_ch, n_time, n_tps,
                         warmup, runs, tune, tune_verb, ktune, stream);
        }
        if (streams_mode == 1 || streams_mode == 2) {
            bench_voltbf_multistream(n_ant, n_beam, n_ch, n_time, n_tps,
                                     ch_tile, warmup, runs, tune, tune_verb, ktune);
        }
        // Fused benchmark: only when ch_fuse > 1 or n_batch > 1
        if (ch_fuse > 1 || n_batch > 1) {
            bench_voltbf_fused(n_ant, n_beam, n_ch, n_time, n_tps,
                               ch_fuse, n_batch,
                               warmup, runs, tune, tune_verb, ktune, stream);
        }
    }

    if (suite == ALL || suite == VISBF) {
        if (streams_mode == 0 || streams_mode == 2) {
            bench_visbf(n_ant, n_beam, n_ch, n_time, n_time_inner,
                        Ng, pol_mode,
                        warmup, runs, tune, tune_verb, ktune,
                        vis_prec_fp16, fft_prec_fp16, stream);
        }
        if (streams_mode == 1 || streams_mode == 2) {
            bench_visbf_multistream(n_ant, n_beam, n_ch, n_time, n_time_inner,
                                    Ng, pol_mode,
                                    warmup, runs, tune, tune_verb, ktune,
                                    vis_prec_fp16, fft_prec_fp16);
        }
    }

    if (suite == ALL || suite == DEDISP_SUITE) {
        bench_dedisp(n_beam, n_ch, n_time_out, n_dm,
                     dedisp_mode, f_min_mhz, f_max_mhz, max_dm,
                     warmup, runs, ktune, stream);
    }

    // Pipelined benchmarks
    if (n_payloads > 0) {
        if (suite == ALL || suite == VOLTBF) {
            bench_voltbf_pipelined(n_ant, n_beam, n_ch, n_time, n_tps,
                                    n_payloads, pipeline, ktune);
        }
        if (suite == ALL || suite == VISBF) {
            bench_visbf_pipelined(n_ant, n_beam, n_ch, n_time, n_time_inner,
                                   Ng, pol_mode,
                                   n_payloads, pipeline, ktune,
                                   vis_prec_fp16, fft_prec_fp16);
        }
        if (suite == ALL || suite == DEDISP_SUITE) {
            bench_dedisp_pipelined(n_beam, n_ch, n_time_out, n_dm,
                                    f_min_mhz, f_max_mhz, max_dm,
                                    n_payloads, pipeline, ktune);
        }
    }

    CHECK_CUDA(cudaStreamDestroy(stream));
    printf("\nDone.\n");
    return 0;
}
