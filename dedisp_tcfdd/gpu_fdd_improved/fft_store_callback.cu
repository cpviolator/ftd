// --- fft_store_callback.cu ---
// cuFFT LTO store callback: deinterleaves complex FFT output to FP32 planar
// buffers and accumulates max via warp reduction + atomicMax.
// Compiled to a fatbin (LTO) and embedded into the main binary via bin2c.

#include <cufft.h>

/// Context struct passed to the callback via callerInfo (lives in device memory).
struct FFTStoreCallbackInfo {
    float* d_planar_re;       // [batch*Nf*Nt_complex] FP32, FFT-native layout
    float* d_planar_im;       // [batch*Nf*Nt_complex] FP32, FFT-native layout
    unsigned int* d_max_uint; // atomicMax target (IEEE 754 float-as-uint)
};

/// cuFFT LTO store callback: deinterleaves complex->planar FP32 + accumulates max.
/// Called once per FFT output element. Replaces cuFFT's default DRAM store.
__device__ void fft_store_deinterleave_max(
    void* dataOut,
    unsigned long long offset,
    cufftComplex element,
    void* callerInfo,
    void* sharedPointer)
{
    FFTStoreCallbackInfo* ctx = static_cast<FFTStoreCallbackInfo*>(callerInfo);

    // Write deinterleaved FP32 (coalesced: offset is contiguous within each transform)
    ctx->d_planar_re[offset] = element.x;
    ctx->d_planar_im[offset] = element.y;

    // Warp-level max reduction to minimize atomicMax contention
    float abs_val = fmaxf(fabsf(element.x), fabsf(element.y));
    unsigned int mask = __activemask();
    for (int off = 16; off > 0; off >>= 1)
        abs_val = fmaxf(abs_val, __shfl_xor_sync(mask, abs_val, off));

    // Only lane 0 of each warp does the atomic (reduces contention 32x)
    if ((threadIdx.x & 31) == 0) {
        atomicMax(ctx->d_max_uint, __float_as_uint(abs_val));
    }
}
