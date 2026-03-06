/**
 * @file bench_transpose_kernels.cu
 * @brief Standalone benchmark: old (32×33) vs new (64×65 / SMEM-tiled) transpose kernels.
 *
 * Usage: ./bench_transpose_kernels [Nf] [Nt_complex] [Ndm] [batch]
 *        Defaults: Nf=512, Nt_complex=4097, Ndm=512, batch=128
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cufft.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <algorithm>

#define CUDA_CHECK(call)                                                      \
  do {                                                                        \
    cudaError_t err = (call);                                                 \
    if (err != cudaSuccess) {                                                 \
      fprintf(stderr, "CUDA Error at %s:%d: %s\n", __FILE__, __LINE__,        \
              cudaGetErrorString(err));                                       \
      exit(EXIT_FAILURE);                                                     \
    }                                                                         \
  } while (0)

// ======================================================================
// Kernel definitions (self-contained, no external includes)
// ======================================================================

// --- Data Prep FP8: OLD 32×33 tile ---
template <typename T_OUT>
__global__ void data_prep_fp8_32(
    const cufftComplex* __restrict__ input,
    T_OUT* __restrict__ out_real,
    T_OUT* __restrict__ out_imag,
    int batch_size, int Nf, int Nt_complex, float scale)
{
    __shared__ float2 tile[32][33];
    int b = blockIdx.z;
    int k_in = blockIdx.y * 32 + threadIdx.x;
    int f_in = blockIdx.x * 32 + threadIdx.y;
    if (k_in < Nt_complex && f_in < Nf && b < batch_size) {
        size_t in_idx = (size_t)b * Nf * Nt_complex + (size_t)f_in * Nt_complex + k_in;
        cufftComplex val = input[in_idx];
        tile[threadIdx.y][threadIdx.x] = make_float2(val.x, val.y);
    }
    __syncthreads();
    int f_out = blockIdx.x * 32 + threadIdx.x;
    int k_out = blockIdx.y * 32 + threadIdx.y;
    if (f_out < Nf && k_out < Nt_complex && b < batch_size) {
        float2 val = tile[threadIdx.x][threadIdx.y];
        float r = fminf(fmaxf(val.x * scale, -448.0f), 448.0f);
        float im = fminf(fmaxf(val.y * scale, -448.0f), 448.0f);
        size_t out_idx = (size_t)k_out * batch_size * Nf + (size_t)b * Nf + f_out;
        out_real[out_idx] = T_OUT(r);
        out_imag[out_idx] = T_OUT(im);
    }
}

// --- Data Prep FP8: NEW 64×65 tile ---
template <typename T_OUT>
__global__ void data_prep_fp8_64(
    const cufftComplex* __restrict__ input,
    T_OUT* __restrict__ out_real,
    T_OUT* __restrict__ out_imag,
    int batch_size, int Nf, int Nt_complex, float scale)
{
    __shared__ float2 tile[64][65];
    int b = blockIdx.z;
    int k_base = blockIdx.y * 64;
    int f_base = blockIdx.x * 64;
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        int k_in = k_base + threadIdx.x;
        int f_in = f_base + threadIdx.y + i * 16;
        if (k_in < Nt_complex && f_in < Nf && b < batch_size) {
            size_t in_idx = (size_t)b * Nf * Nt_complex + (size_t)f_in * Nt_complex + k_in;
            cufftComplex val = input[in_idx];
            tile[threadIdx.y + i * 16][threadIdx.x] = make_float2(val.x, val.y);
        }
    }
    __syncthreads();
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        int f_out = f_base + threadIdx.x;
        int k_out = k_base + threadIdx.y + i * 16;
        if (f_out < Nf && k_out < Nt_complex && b < batch_size) {
            float2 val = tile[threadIdx.x][threadIdx.y + i * 16];
            float r = fminf(fmaxf(val.x * scale, -448.0f), 448.0f);
            float im = fminf(fmaxf(val.y * scale, -448.0f), 448.0f);
            size_t out_idx = (size_t)k_out * batch_size * Nf + (size_t)b * Nf + f_out;
            out_real[out_idx] = T_OUT(r);
            out_imag[out_idx] = T_OUT(im);
        }
    }
}

// --- Data Prep FP16 (CUTLASS): OLD 32×33 tile ---
__global__ void data_prep_f16_32(
    const cufftComplex* __restrict__ input,
    __half* __restrict__ out_real,
    __half* __restrict__ out_imag,
    int batch_size, int Nf, int Nt_complex, float scale)
{
    __shared__ float2 tile[32][33];
    int b = blockIdx.z;
    int k_in = blockIdx.y * 32 + threadIdx.x;
    int f_in = blockIdx.x * 32 + threadIdx.y;
    if (k_in < Nt_complex && f_in < Nf && b < batch_size) {
        size_t in_idx = (size_t)b * Nf * Nt_complex + (size_t)f_in * Nt_complex + k_in;
        cufftComplex val = input[in_idx];
        tile[threadIdx.y][threadIdx.x] = make_float2(val.x * scale, val.y * scale);
    }
    __syncthreads();
    int f_out = blockIdx.x * 32 + threadIdx.x;
    int k_out = blockIdx.y * 32 + threadIdx.y;
    if (f_out < Nf && k_out < Nt_complex && b < batch_size) {
        float2 val = tile[threadIdx.x][threadIdx.y];
        size_t out_idx = (size_t)k_out * batch_size * Nf + (size_t)b * Nf + f_out;
        out_real[out_idx] = __float2half(val.x);
        out_imag[out_idx] = __float2half(val.y);
    }
}

// --- Data Prep FP16 (CUTLASS): NEW 64×65 tile ---
__global__ void data_prep_f16_64(
    const cufftComplex* __restrict__ input,
    __half* __restrict__ out_real,
    __half* __restrict__ out_imag,
    int batch_size, int Nf, int Nt_complex, float scale)
{
    __shared__ float2 tile[64][65];
    int b = blockIdx.z;
    int k_base = blockIdx.y * 64;
    int f_base = blockIdx.x * 64;
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        int k_in = k_base + threadIdx.x;
        int f_in = f_base + threadIdx.y + i * 16;
        if (k_in < Nt_complex && f_in < Nf && b < batch_size) {
            size_t in_idx = (size_t)b * Nf * Nt_complex + (size_t)f_in * Nt_complex + k_in;
            cufftComplex val = input[in_idx];
            tile[threadIdx.y + i * 16][threadIdx.x] = make_float2(val.x * scale, val.y * scale);
        }
    }
    __syncthreads();
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        int f_out = f_base + threadIdx.x;
        int k_out = k_base + threadIdx.y + i * 16;
        if (f_out < Nf && k_out < Nt_complex && b < batch_size) {
            float2 val = tile[threadIdx.x][threadIdx.y + i * 16];
            size_t out_idx = (size_t)k_out * batch_size * Nf + (size_t)b * Nf + f_out;
            out_real[out_idx] = __float2half(val.x);
            out_imag[out_idx] = __float2half(val.y);
        }
    }
}

// --- Post-GEMM Finalize (cuBLASLt): OLD no-SMEM ---
// Input: [k][dm][b] col-major (b stride-1). Output: [b][dm][k] (k stride-1).
__global__ void finalize_cublaslt_old(
    const float* __restrict__ c_real,
    const float* __restrict__ c_imag,
    cufftComplex* __restrict__ output,
    int batch_size, int Ndm, int Nt_complex, float unscale)
{
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    int dm = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z;
    if (b < batch_size && dm < Ndm && k < Nt_complex) {
        size_t in_idx = (size_t)k * Ndm * batch_size + (size_t)dm * batch_size + b;
        size_t out_idx = (size_t)b * Ndm * Nt_complex + (size_t)dm * Nt_complex + k;
        output[out_idx].x = float(c_real[in_idx]) * unscale;
        output[out_idx].y = float(c_imag[in_idx]) * unscale;
    }
}

// --- Post-GEMM Finalize (cuBLASLt): NEW SMEM-tiled ---
__global__ void finalize_cublaslt_tiled(
    const float* __restrict__ c_real,
    const float* __restrict__ c_imag,
    cufftComplex* __restrict__ output,
    int batch_size, int Ndm, int Nt_complex, float unscale)
{
    __shared__ float tile_r[32][33];
    __shared__ float tile_i[32][33];
    int dm = blockIdx.z;
    int b_base = blockIdx.x * 32;
    int k_base = blockIdx.y * 32;
    int b_in = b_base + threadIdx.x;
    int k_in = k_base + threadIdx.y;
    if (b_in < batch_size && k_in < Nt_complex && dm < Ndm) {
        size_t in_idx = (size_t)k_in * Ndm * batch_size + (size_t)dm * batch_size + b_in;
        tile_r[threadIdx.y][threadIdx.x] = c_real[in_idx];
        tile_i[threadIdx.y][threadIdx.x] = c_imag[in_idx];
    }
    __syncthreads();
    int k_out = k_base + threadIdx.x;
    int b_out = b_base + threadIdx.y;
    if (k_out < Nt_complex && b_out < batch_size && dm < Ndm) {
        float r = tile_r[threadIdx.x][threadIdx.y] * unscale;
        float im = tile_i[threadIdx.x][threadIdx.y] * unscale;
        size_t out_idx = (size_t)b_out * Ndm * Nt_complex + (size_t)dm * Nt_complex + k_out;
        output[out_idx].x = r;
        output[out_idx].y = im;
    }
}

// --- Post-GEMM Finalize (CUTLASS): OLD no-SMEM ---
// Input: [k][b][dm] row-major (dm stride-1). Output: [b][dm][k] (k stride-1).
__global__ void finalize_cutlass_old(
    const float* __restrict__ c_real,
    const float* __restrict__ c_imag,
    cufftComplex* __restrict__ output,
    int batch_size, int Ndm, int Nt_complex)
{
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    int dm = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z;
    if (b < batch_size && dm < Ndm && k < Nt_complex) {
        size_t in_idx = (size_t)k * batch_size * Ndm + (size_t)b * Ndm + dm;
        size_t out_idx = (size_t)b * Ndm * Nt_complex + (size_t)dm * Nt_complex + k;
        output[out_idx].x = c_real[in_idx];
        output[out_idx].y = c_imag[in_idx];
    }
}

// --- Post-GEMM Finalize (CUTLASS): NEW SMEM-tiled ---
__global__ void finalize_cutlass_tiled(
    const float* __restrict__ c_real,
    const float* __restrict__ c_imag,
    cufftComplex* __restrict__ output,
    int batch_size, int Ndm, int Nt_complex)
{
    __shared__ float tile_r[32][33];
    __shared__ float tile_i[32][33];
    int b = blockIdx.z;
    int dm_base = blockIdx.x * 32;
    int k_base = blockIdx.y * 32;
    int dm_in = dm_base + threadIdx.x;
    int k_in = k_base + threadIdx.y;
    if (dm_in < Ndm && k_in < Nt_complex && b < batch_size) {
        size_t in_idx = (size_t)k_in * batch_size * Ndm + (size_t)b * Ndm + dm_in;
        tile_r[threadIdx.y][threadIdx.x] = c_real[in_idx];
        tile_i[threadIdx.y][threadIdx.x] = c_imag[in_idx];
    }
    __syncthreads();
    int k_out = k_base + threadIdx.x;
    int dm_out = dm_base + threadIdx.y;
    if (k_out < Nt_complex && dm_out < Ndm && b < batch_size) {
        float r = tile_r[threadIdx.x][threadIdx.y];
        float im = tile_i[threadIdx.x][threadIdx.y];
        size_t out_idx = (size_t)b * Ndm * Nt_complex + (size_t)dm_out * Nt_complex + k_out;
        output[out_idx].x = r;
        output[out_idx].y = im;
    }
}

// ======================================================================
// Timing helpers
// ======================================================================

typedef void (*KernelLauncher)(cudaStream_t stream, void* args);

float bench(void (*fn)(cudaStream_t), int warmup = 5, int iters = 20) {
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    for (int i = 0; i < warmup; i++) fn(stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    CUDA_CHECK(cudaEventRecord(start, stream));
    for (int i = 0; i < iters; i++) fn(stream);
    CUDA_CHECK(cudaEventRecord(stop, stream));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaStreamDestroy(stream));
    return ms / iters;
}

// ======================================================================
// Global params (set in main, used by launcher functions)
// ======================================================================
static int g_Nf, g_Nt, g_Ndm, g_batch;
static float g_scale = 1.0f, g_unscale = 1.0f;
static cufftComplex* g_input;
static __nv_fp8_e4m3 *g_fp8_re, *g_fp8_im;
static __half *g_f16_re, *g_f16_im;
static float *g_c_real, *g_c_imag;
static cufftComplex* g_output;

// ---- Launcher functions ----
void launch_fp8_32(cudaStream_t s) {
    dim3 b(32,32), g((g_Nf+31)/32, (g_Nt+31)/32, g_batch);
    data_prep_fp8_32<__nv_fp8_e4m3><<<g,b,0,s>>>(g_input, g_fp8_re, g_fp8_im, g_batch, g_Nf, g_Nt, g_scale);
}
void launch_fp8_64(cudaStream_t s) {
    dim3 b(64,16), g((g_Nf+63)/64, (g_Nt+63)/64, g_batch);
    data_prep_fp8_64<__nv_fp8_e4m3><<<g,b,0,s>>>(g_input, g_fp8_re, g_fp8_im, g_batch, g_Nf, g_Nt, g_scale);
}
void launch_f16_32(cudaStream_t s) {
    dim3 b(32,32), g((g_Nf+31)/32, (g_Nt+31)/32, g_batch);
    data_prep_f16_32<<<g,b,0,s>>>(g_input, g_f16_re, g_f16_im, g_batch, g_Nf, g_Nt, g_scale);
}
void launch_f16_64(cudaStream_t s) {
    dim3 b(64,16), g((g_Nf+63)/64, (g_Nt+63)/64, g_batch);
    data_prep_f16_64<<<g,b,0,s>>>(g_input, g_f16_re, g_f16_im, g_batch, g_Nf, g_Nt, g_scale);
}
void launch_fin_cublaslt_old(cudaStream_t s) {
    dim3 b(32,32), g((g_batch+31)/32, (g_Ndm+31)/32, g_Nt);
    finalize_cublaslt_old<<<g,b,0,s>>>(g_c_real, g_c_imag, g_output, g_batch, g_Ndm, g_Nt, g_unscale);
}
void launch_fin_cublaslt_tiled(cudaStream_t s) {
    dim3 b(32,32), g((g_batch+31)/32, (g_Nt+31)/32, g_Ndm);
    finalize_cublaslt_tiled<<<g,b,0,s>>>(g_c_real, g_c_imag, g_output, g_batch, g_Ndm, g_Nt, g_unscale);
}
void launch_fin_cutlass_old(cudaStream_t s) {
    dim3 b(32,32), g((g_batch+31)/32, (g_Ndm+31)/32, g_Nt);
    finalize_cutlass_old<<<g,b,0,s>>>(g_c_real, g_c_imag, g_output, g_batch, g_Ndm, g_Nt);
}
void launch_fin_cutlass_tiled(cudaStream_t s) {
    dim3 b(32,32), g((g_Ndm+31)/32, (g_Nt+31)/32, g_batch);
    finalize_cutlass_tiled<<<g,b,0,s>>>(g_c_real, g_c_imag, g_output, g_batch, g_Ndm, g_Nt);
}

void print_row(const char* name, float ms, double bytes) {
    double gbps = bytes / (ms * 1e6);
    printf("  %-44s  %8.3f ms  %7.1f GB/s\n", name, ms, gbps);
}

int main(int argc, char** argv) {
    g_Nf = 512; g_Nt = 4097; g_Ndm = 512; g_batch = 128;
    if (argc > 1) g_Nf = atoi(argv[1]);
    if (argc > 2) g_Nt = atoi(argv[2]);
    if (argc > 3) g_Ndm = atoi(argv[3]);
    if (argc > 4) g_batch = atoi(argv[4]);

    printf("=== Transpose Kernel Benchmark ===\n");
    printf("Nf=%d  Nt_complex=%d  Ndm=%d  batch=%d\n\n", g_Nf, g_Nt, g_Ndm, g_batch);

    // Memory sizes
    size_t sz_in  = (size_t)g_batch * g_Nf * g_Nt * sizeof(cufftComplex);
    size_t sz_fp8 = (size_t)g_batch * g_Nf * g_Nt;
    size_t sz_f16 = (size_t)g_batch * g_Nf * g_Nt * sizeof(__half);
    size_t sz_cr  = (size_t)g_batch * g_Ndm * g_Nt * sizeof(float);
    size_t sz_out = (size_t)g_batch * g_Ndm * g_Nt * sizeof(cufftComplex);
    size_t total  = sz_in + sz_fp8*2 + sz_f16*2 + sz_cr*2 + sz_out;

    // Check GPU memory
    size_t free_mem, total_mem;
    CUDA_CHECK(cudaMemGetInfo(&free_mem, &total_mem));
    if (total > free_mem * 0.9) {
        int new_batch = std::max(1, (int)(g_batch * free_mem * 0.7 / total));
        printf("NOTE: Reducing batch %d -> %d to fit in %.1f GB GPU memory\n\n",
               g_batch, new_batch, free_mem/1e9);
        g_batch = new_batch;
        sz_in  = (size_t)g_batch * g_Nf * g_Nt * sizeof(cufftComplex);
        sz_fp8 = (size_t)g_batch * g_Nf * g_Nt;
        sz_f16 = (size_t)g_batch * g_Nf * g_Nt * sizeof(__half);
        sz_cr  = (size_t)g_batch * g_Ndm * g_Nt * sizeof(float);
        sz_out = (size_t)g_batch * g_Ndm * g_Nt * sizeof(cufftComplex);
        total  = sz_in + sz_fp8*2 + sz_f16*2 + sz_cr*2 + sz_out;
    }

    printf("GPU memory: %.1f MB allocated (%.1f GB free)\n\n", total/1e6, free_mem/1e9);

    CUDA_CHECK(cudaMalloc(&g_input,  sz_in));
    CUDA_CHECK(cudaMalloc(&g_fp8_re, sz_fp8));
    CUDA_CHECK(cudaMalloc(&g_fp8_im, sz_fp8));
    CUDA_CHECK(cudaMalloc(&g_f16_re, sz_f16));
    CUDA_CHECK(cudaMalloc(&g_f16_im, sz_f16));
    CUDA_CHECK(cudaMalloc(&g_c_real, sz_cr));
    CUDA_CHECK(cudaMalloc(&g_c_imag, sz_cr));
    CUDA_CHECK(cudaMalloc(&g_output, sz_out));
    CUDA_CHECK(cudaMemset(g_input, 0x42, sz_in));
    CUDA_CHECK(cudaMemset(g_c_real, 0x42, sz_cr));
    CUDA_CHECK(cudaMemset(g_c_imag, 0x42, sz_cr));

    // Bytes moved per kernel
    double dp_fp8_bytes = (double)g_batch * g_Nf * g_Nt * (sizeof(cufftComplex) + 2);
    double dp_f16_bytes = (double)g_batch * g_Nf * g_Nt * (sizeof(cufftComplex) + 2*sizeof(__half));
    double fin_bytes    = (double)g_batch * g_Ndm * g_Nt * (2*sizeof(float) + sizeof(cufftComplex));

    // ---- Benchmark 1: Data Prep FP8 ----
    printf("[1] Data Prep FP8: [B,Nf,k] -> [k,B,Nf] + scale + FP8 cast\n");
    float fp8_32 = bench(launch_fp8_32);
    float fp8_64 = bench(launch_fp8_64);
    print_row("OLD: 32x33 tile, block(32,32)", fp8_32, dp_fp8_bytes);
    print_row("NEW: 64x65 tile, block(64,16)", fp8_64, dp_fp8_bytes);
    printf("  => %.2fx %s\n\n",
           fp8_32 > fp8_64 ? fp8_32/fp8_64 : fp8_64/fp8_32,
           fp8_32 > fp8_64 ? "SPEEDUP" : "slowdown");

    // ---- Benchmark 2: Data Prep FP16 ----
    printf("[2] Data Prep FP16 (CUTLASS): [B,Nf,k] -> [k,B,Nf] + scale + FP16 cast\n");
    float f16_32 = bench(launch_f16_32);
    float f16_64 = bench(launch_f16_64);
    print_row("OLD: 32x33 tile, block(32,32)", f16_32, dp_f16_bytes);
    print_row("NEW: 64x65 tile, block(64,16)", f16_64, dp_f16_bytes);
    printf("  => %.2fx %s\n\n",
           f16_32 > f16_64 ? f16_32/f16_64 : f16_64/f16_32,
           f16_32 > f16_64 ? "SPEEDUP" : "slowdown");

    // ---- Benchmark 3: Finalize cuBLASLt ----
    printf("[3] Post-GEMM Finalize (cuBLASLt): [k,dm,b] planar -> [b,dm,k] interleaved\n");
    float fcl_old = bench(launch_fin_cublaslt_old);
    float fcl_new = bench(launch_fin_cublaslt_tiled);
    print_row("OLD: No SMEM (coalesced reads, strided writes)", fcl_old, fin_bytes);
    print_row("NEW: SMEM-tiled (coalesced reads + writes)",     fcl_new, fin_bytes);
    printf("  => %.2fx %s\n\n",
           fcl_old > fcl_new ? fcl_old/fcl_new : fcl_new/fcl_old,
           fcl_old > fcl_new ? "SPEEDUP" : "slowdown");

    // ---- Benchmark 4: Finalize CUTLASS ----
    printf("[4] Post-GEMM Finalize (CUTLASS): [k,b,dm] planar -> [b,dm,k] interleaved\n");
    float fcu_old = bench(launch_fin_cutlass_old);
    float fcu_new = bench(launch_fin_cutlass_tiled);
    print_row("OLD: No SMEM (strided reads + writes)",      fcu_old, fin_bytes);
    print_row("NEW: SMEM-tiled (coalesced reads + writes)", fcu_new, fin_bytes);
    printf("  => %.2fx %s\n\n",
           fcu_old > fcu_new ? fcu_old/fcu_new : fcu_new/fcu_old,
           fcu_old > fcu_new ? "SPEEDUP" : "slowdown");

    // ---- Summary ----
    printf("=== Summary (Nf=%d, Nt=%d, Ndm=%d, batch=%d) ===\n", g_Nf, g_Nt, g_Ndm, g_batch);
    printf("  %-30s  %8s  %8s  %8s\n", "Kernel", "Old(ms)", "New(ms)", "Speedup");
    printf("  %-30s  %8.3f  %8.3f  %7.2fx\n", "Data Prep FP8",          fp8_32,  fp8_64,  fp8_32/fp8_64);
    printf("  %-30s  %8.3f  %8.3f  %7.2fx\n", "Data Prep FP16",         f16_32,  f16_64,  f16_32/f16_64);
    printf("  %-30s  %8.3f  %8.3f  %7.2fx\n", "Finalize cuBLASLt",      fcl_old, fcl_new, fcl_old/fcl_new);
    printf("  %-30s  %8.3f  %8.3f  %7.2fx\n", "Finalize CUTLASS",       fcu_old, fcu_new, fcu_old/fcu_new);

    CUDA_CHECK(cudaFree(g_input));
    CUDA_CHECK(cudaFree(g_fp8_re));
    CUDA_CHECK(cudaFree(g_fp8_im));
    CUDA_CHECK(cudaFree(g_f16_re));
    CUDA_CHECK(cudaFree(g_f16_im));
    CUDA_CHECK(cudaFree(g_c_real));
    CUDA_CHECK(cudaFree(g_c_imag));
    CUDA_CHECK(cudaFree(g_output));
    return 0;
}
