// ========================================================================================
// Kernel type chains (shared types, templates, and instantiations) are defined in
// gemm_sm100_type_chains.hpp — included at the top of this file. That header is also
// included by gemm_blockscaled_dispatch.cu, enabling multi-TU builds without ODR
// violations from __global__ kernel definitions in this header.
// ========================================================================================


// ========================================================================================
// FP8 Buffer Manager (shared with SM90)
// ========================================================================================

#include "../shared/buffers_common.hpp"


// ========================================================================================
// Low-Precision Buffer Manager — Sub-Byte Types (FP6, FP4)
// ========================================================================================
//
// Manages void* GPU buffers sized in bytes, not elements. Used for FP6 (0.75 bytes/elem)
// and FP4 (0.5 bytes/elem) where element-typed pointers don't exist.
//
// The FP8 path continues to use FP8BufferManager for zero-overhead backward compatibility.

class LowPrecisionBufferManager {
public:
    LowPrecisionBufferManager() = default;
    ~LowPrecisionBufferManager() { free_all(); }

    LowPrecisionBufferManager(const LowPrecisionBufferManager&) = delete;
    LowPrecisionBufferManager& operator=(const LowPrecisionBufferManager&) = delete;

    /// Ensure data buffers have at least `bytes_A` and `bytes_B` capacity.
    /// Reallocates only if current capacity is insufficient.
    void ensure_capacity(int64_t bytes_A, int64_t bytes_B, int64_t bytes_BT = 0,
                         cudaStream_t stream = nullptr) {
        if (bytes_A > cap_A_bytes_ || bytes_B > cap_B_bytes_) {
            free_main(stream);
            cap_A_bytes_ = std::max(bytes_A, cap_A_bytes_);
            cap_B_bytes_ = std::max(bytes_B, cap_B_bytes_);
            CUDA_CHECK(cudaMallocAsync(&A_real_, cap_A_bytes_, stream));
            CUDA_CHECK(cudaMallocAsync(&A_imag_, cap_A_bytes_, stream));
            CUDA_CHECK(cudaMallocAsync(&B_real_, cap_B_bytes_, stream));
            CUDA_CHECK(cudaMallocAsync(&B_imag_, cap_B_bytes_, stream));
        }
        if (bytes_BT > 0 && bytes_BT > cap_BT_bytes_) {
            free_transpose(stream);
            cap_BT_bytes_ = bytes_BT;
            CUDA_CHECK(cudaMallocAsync(&BT_real_, bytes_BT, stream));
            CUDA_CHECK(cudaMallocAsync(&BT_imag_, bytes_BT, stream));
        }
    }

    /// Ensure scale factor buffers for MXFP block-scaled types.
    /// Each operand (Ar, Ai, Br, Bi) needs its own scale factor buffer.
    /// Scale factor count is computed from problem shape via Sm1xxBlkScaledConfig.
    void ensure_sf_capacity(int64_t sf_bytes_A, int64_t sf_bytes_B,
                            int64_t sf_bytes_BT = 0,
                            cudaStream_t stream = nullptr) {
        if (sf_bytes_A > cap_sf_A_bytes_ || sf_bytes_B > cap_sf_B_bytes_) {
            free_sf_main(stream);
            cap_sf_A_bytes_ = std::max(sf_bytes_A, cap_sf_A_bytes_);
            cap_sf_B_bytes_ = std::max(sf_bytes_B, cap_sf_B_bytes_);
            CUDA_CHECK(cudaMallocAsync(&sf_A_real_, cap_sf_A_bytes_, stream));
            CUDA_CHECK(cudaMallocAsync(&sf_A_imag_, cap_sf_A_bytes_, stream));
            CUDA_CHECK(cudaMallocAsync(&sf_B_real_, cap_sf_B_bytes_, stream));
            CUDA_CHECK(cudaMallocAsync(&sf_B_imag_, cap_sf_B_bytes_, stream));
        }
        if (sf_bytes_BT > 0 && sf_bytes_BT > cap_sf_BT_bytes_) {
            free_sf_transpose(stream);
            cap_sf_BT_bytes_ = sf_bytes_BT;
            CUDA_CHECK(cudaMallocAsync(&sf_BT_real_, sf_bytes_BT, stream));
            CUDA_CHECK(cudaMallocAsync(&sf_BT_imag_, sf_bytes_BT, stream));
        }
    }

    void* A_real()  { return A_real_; }
    void* A_imag()  { return A_imag_; }
    void* B_real()  { return B_real_; }
    void* B_imag()  { return B_imag_; }
    void* BT_real() { return BT_real_; }
    void* BT_imag() { return BT_imag_; }

    // Scale factor accessors (float_ue8m0_t buffers for MXFP)
    void* sf_A_real()  { return sf_A_real_; }
    void* sf_A_imag()  { return sf_A_imag_; }
    void* sf_B_real()  { return sf_B_real_; }
    void* sf_B_imag()  { return sf_B_imag_; }
    void* sf_BT_real() { return sf_BT_real_; }
    void* sf_BT_imag() { return sf_BT_imag_; }

private:
    // Stream-ordered free for reallocation paths (avoids implicit cudaDeviceSynchronize).
    // Destructor passes stream=nullptr which falls back to synchronous cudaFree.
    void free_main(cudaStream_t stream = nullptr) {
        if (A_real_)  { stream ? cudaFreeAsync(A_real_, stream) : cudaFree(A_real_);  A_real_ = nullptr; }
        if (A_imag_)  { stream ? cudaFreeAsync(A_imag_, stream) : cudaFree(A_imag_);  A_imag_ = nullptr; }
        if (B_real_)  { stream ? cudaFreeAsync(B_real_, stream) : cudaFree(B_real_);  B_real_ = nullptr; }
        if (B_imag_)  { stream ? cudaFreeAsync(B_imag_, stream) : cudaFree(B_imag_);  B_imag_ = nullptr; }
    }
    void free_transpose(cudaStream_t stream = nullptr) {
        if (BT_real_) { stream ? cudaFreeAsync(BT_real_, stream) : cudaFree(BT_real_); BT_real_ = nullptr; }
        if (BT_imag_) { stream ? cudaFreeAsync(BT_imag_, stream) : cudaFree(BT_imag_); BT_imag_ = nullptr; }
    }
    void free_sf_main(cudaStream_t stream = nullptr) {
        if (sf_A_real_)  { stream ? cudaFreeAsync(sf_A_real_, stream) : cudaFree(sf_A_real_);  sf_A_real_ = nullptr; }
        if (sf_A_imag_)  { stream ? cudaFreeAsync(sf_A_imag_, stream) : cudaFree(sf_A_imag_);  sf_A_imag_ = nullptr; }
        if (sf_B_real_)  { stream ? cudaFreeAsync(sf_B_real_, stream) : cudaFree(sf_B_real_);  sf_B_real_ = nullptr; }
        if (sf_B_imag_)  { stream ? cudaFreeAsync(sf_B_imag_, stream) : cudaFree(sf_B_imag_);  sf_B_imag_ = nullptr; }
    }
    void free_sf_transpose(cudaStream_t stream = nullptr) {
        if (sf_BT_real_) { stream ? cudaFreeAsync(sf_BT_real_, stream) : cudaFree(sf_BT_real_); sf_BT_real_ = nullptr; }
        if (sf_BT_imag_) { stream ? cudaFreeAsync(sf_BT_imag_, stream) : cudaFree(sf_BT_imag_); sf_BT_imag_ = nullptr; }
    }
    void free_all() { free_main(); free_transpose(); free_sf_main(); free_sf_transpose(); }

    // Data buffers
    void* A_real_ = nullptr;
    void* A_imag_ = nullptr;
    void* B_real_ = nullptr;
    void* B_imag_ = nullptr;
    void* BT_real_ = nullptr;
    void* BT_imag_ = nullptr;
    int64_t cap_A_bytes_ = 0;
    int64_t cap_B_bytes_ = 0;
    int64_t cap_BT_bytes_ = 0;

    // Scale factor buffers (float_ue8m0_t, 1 byte per 32-element block)
    void* sf_A_real_ = nullptr;
    void* sf_A_imag_ = nullptr;
    void* sf_B_real_ = nullptr;
    void* sf_B_imag_ = nullptr;
    void* sf_BT_real_ = nullptr;
    void* sf_BT_imag_ = nullptr;
    int64_t cap_sf_A_bytes_ = 0;
    int64_t cap_sf_B_bytes_ = 0;
    int64_t cap_sf_BT_bytes_ = 0;
};
