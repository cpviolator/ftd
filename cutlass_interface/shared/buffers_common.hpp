// ========================================================================================
// FP8 Buffer Manager — Shared between SM90 and SM100/SM120
// ========================================================================================
//
// RAII wrapper for FP8 scratch buffers used by the 4M complex GEMM decomposition.
// Manages A/B operand buffers, transpose scratch, stacked-K buffers, and
// stacked-GEMM buffers (Strategy 5D).
//
// This header is #include'd from within each architecture's buffers.hpp.

#pragma once

class FP8BufferManager {
public:
    FP8BufferManager() = default;
    ~FP8BufferManager() { free(); }

    FP8BufferManager(const FP8BufferManager&) = delete;
    FP8BufferManager& operator=(const FP8BufferManager&) = delete;
    FP8BufferManager(FP8BufferManager&& other) noexcept { swap(other); }
    FP8BufferManager& operator=(FP8BufferManager&& other) noexcept { swap(other); return *this; }

    /// Ensure FP8 buffers have at least `need_A` and `need_B` element capacity.
    /// For batched operations, pass total elements: e.g. need_A = M*K*batch_count.
    /// If need_BT > 0, also allocates transpose scratch buffers.
    void ensure_capacity(int64_t need_A, int64_t need_B, int64_t need_BT = 0,
                         cudaStream_t stream = nullptr) {
        if (need_A > capacity_A_ || need_B > capacity_B_) {
            free_main(stream);
            capacity_A_ = std::max(need_A, capacity_A_);
            capacity_B_ = std::max(need_B, capacity_B_);
            CUDA_CHECK(cudaMallocAsync(&A_real_fp8_, capacity_A_ * sizeof(cutlass::float_e4m3_t), stream));
            CUDA_CHECK(cudaMallocAsync(&A_imag_fp8_, capacity_A_ * sizeof(cutlass::float_e4m3_t), stream));
            CUDA_CHECK(cudaMallocAsync(&B_real_fp8_, capacity_B_ * sizeof(cutlass::float_e4m3_t), stream));
            CUDA_CHECK(cudaMallocAsync(&B_imag_fp8_, capacity_B_ * sizeof(cutlass::float_e4m3_t), stream));
        }
        if (need_BT > 0 && need_BT > capacity_BT_) {
            free_transpose(stream);
            capacity_BT_ = need_BT;
            CUDA_CHECK(cudaMallocAsync(&BT_real_fp8_, need_BT * sizeof(cutlass::float_e4m3_t), stream));
            CUDA_CHECK(cudaMallocAsync(&BT_imag_fp8_, need_BT * sizeof(cutlass::float_e4m3_t), stream));
        }
    }

    cutlass::float_e4m3_t* A_real() { return A_real_fp8_; }
    cutlass::float_e4m3_t* A_imag() { return A_imag_fp8_; }
    cutlass::float_e4m3_t* B_real() { return B_real_fp8_; }
    cutlass::float_e4m3_t* B_imag() { return B_imag_fp8_; }
    cutlass::float_e4m3_t* BT_real() { return BT_real_fp8_; }
    cutlass::float_e4m3_t* BT_imag() { return BT_imag_fp8_; }

    /// Stacked-K buffers: stacked M×2K + stacked_b (duplicate for TMA L2 fix) + separate Xi M×K + Xr M×K
    void ensure_stacked_capacity(int64_t stacked_elems, int64_t separate_elems,
                                 cudaStream_t stream = nullptr) {
        if (stacked_elems > stacked_capacity_ || separate_elems > separate_capacity_) {
            free_stacked(stream);
            stacked_capacity_ = std::max(stacked_elems, stacked_capacity_);
            separate_capacity_ = std::max(separate_elems, separate_capacity_);
            CUDA_CHECK(cudaMallocAsync(&stacked_fp8_, stacked_capacity_ * sizeof(cutlass::float_e4m3_t), stream));
            CUDA_CHECK(cudaMallocAsync(&stacked_b_fp8_, stacked_capacity_ * sizeof(cutlass::float_e4m3_t), stream));
            CUDA_CHECK(cudaMallocAsync(&xi_separate_fp8_, separate_capacity_ * sizeof(cutlass::float_e4m3_t), stream));
            CUDA_CHECK(cudaMallocAsync(&xr_separate_fp8_, separate_capacity_ * sizeof(cutlass::float_e4m3_t), stream));
        }
    }

    cutlass::float_e4m3_t* stacked() { return stacked_fp8_; }
    cutlass::float_e4m3_t* stacked_b() { return stacked_b_fp8_; }
    cutlass::float_e4m3_t* xi_separate() { return xi_separate_fp8_; }
    cutlass::float_e4m3_t* xr_separate() { return xr_separate_fp8_; }

    /// Stacked-K GEMM buffers (Strategy 5D): 4 operands stacked along K dimension
    void ensure_stacked_gemm_capacity(int64_t a_stacked_elems, int64_t b_stacked_elems,
                                       cudaStream_t stream = nullptr) {
        if (a_stacked_elems > stacked_gemm_cap_A_ || b_stacked_elems > stacked_gemm_cap_B_) {
            free_stacked_gemm(stream);
            stacked_gemm_cap_A_ = std::max(a_stacked_elems, stacked_gemm_cap_A_);
            stacked_gemm_cap_B_ = std::max(b_stacked_elems, stacked_gemm_cap_B_);
            CUDA_CHECK(cudaMallocAsync(&A_re_stacked_, stacked_gemm_cap_A_ * sizeof(cutlass::float_e4m3_t), stream));
            CUDA_CHECK(cudaMallocAsync(&A_im_stacked_, stacked_gemm_cap_A_ * sizeof(cutlass::float_e4m3_t), stream));
            CUDA_CHECK(cudaMallocAsync(&B_re_stacked_, stacked_gemm_cap_B_ * sizeof(cutlass::float_e4m3_t), stream));
            CUDA_CHECK(cudaMallocAsync(&B_im_stacked_, stacked_gemm_cap_B_ * sizeof(cutlass::float_e4m3_t), stream));
        }
    }

    cutlass::float_e4m3_t* A_re_stacked() { return A_re_stacked_; }
    cutlass::float_e4m3_t* A_im_stacked() { return A_im_stacked_; }
    cutlass::float_e4m3_t* B_re_stacked() { return B_re_stacked_; }
    cutlass::float_e4m3_t* B_im_stacked() { return B_im_stacked_; }

private:
    // Stream-ordered free for reallocation paths (avoids implicit cudaDeviceSynchronize).
    // Destructor passes stream=nullptr which falls back to synchronous cudaFree.
    void free_main(cudaStream_t stream = nullptr) {
        if (A_real_fp8_) { stream ? cudaFreeAsync(A_real_fp8_, stream) : cudaFree(A_real_fp8_); A_real_fp8_ = nullptr; }
        if (A_imag_fp8_) { stream ? cudaFreeAsync(A_imag_fp8_, stream) : cudaFree(A_imag_fp8_); A_imag_fp8_ = nullptr; }
        if (B_real_fp8_) { stream ? cudaFreeAsync(B_real_fp8_, stream) : cudaFree(B_real_fp8_); B_real_fp8_ = nullptr; }
        if (B_imag_fp8_) { stream ? cudaFreeAsync(B_imag_fp8_, stream) : cudaFree(B_imag_fp8_); B_imag_fp8_ = nullptr; }
    }
    void free_transpose(cudaStream_t stream = nullptr) {
        if (BT_real_fp8_) { stream ? cudaFreeAsync(BT_real_fp8_, stream) : cudaFree(BT_real_fp8_); BT_real_fp8_ = nullptr; }
        if (BT_imag_fp8_) { stream ? cudaFreeAsync(BT_imag_fp8_, stream) : cudaFree(BT_imag_fp8_); BT_imag_fp8_ = nullptr; }
    }
    void free_stacked(cudaStream_t stream = nullptr) {
        if (stacked_fp8_) { stream ? cudaFreeAsync(stacked_fp8_, stream) : cudaFree(stacked_fp8_); stacked_fp8_ = nullptr; }
        if (stacked_b_fp8_) { stream ? cudaFreeAsync(stacked_b_fp8_, stream) : cudaFree(stacked_b_fp8_); stacked_b_fp8_ = nullptr; }
        if (xi_separate_fp8_) { stream ? cudaFreeAsync(xi_separate_fp8_, stream) : cudaFree(xi_separate_fp8_); xi_separate_fp8_ = nullptr; }
        if (xr_separate_fp8_) { stream ? cudaFreeAsync(xr_separate_fp8_, stream) : cudaFree(xr_separate_fp8_); xr_separate_fp8_ = nullptr; }
    }
    void free_stacked_gemm(cudaStream_t stream = nullptr) {
        if (A_re_stacked_) { stream ? cudaFreeAsync(A_re_stacked_, stream) : cudaFree(A_re_stacked_); A_re_stacked_ = nullptr; }
        if (A_im_stacked_) { stream ? cudaFreeAsync(A_im_stacked_, stream) : cudaFree(A_im_stacked_); A_im_stacked_ = nullptr; }
        if (B_re_stacked_) { stream ? cudaFreeAsync(B_re_stacked_, stream) : cudaFree(B_re_stacked_); B_re_stacked_ = nullptr; }
        if (B_im_stacked_) { stream ? cudaFreeAsync(B_im_stacked_, stream) : cudaFree(B_im_stacked_); B_im_stacked_ = nullptr; }
    }
    void free() { free_main(); free_transpose(); free_stacked(); free_stacked_gemm(); }
    void swap(FP8BufferManager& other) noexcept {
        std::swap(A_real_fp8_, other.A_real_fp8_);
        std::swap(A_imag_fp8_, other.A_imag_fp8_);
        std::swap(B_real_fp8_, other.B_real_fp8_);
        std::swap(B_imag_fp8_, other.B_imag_fp8_);
        std::swap(BT_real_fp8_, other.BT_real_fp8_);
        std::swap(BT_imag_fp8_, other.BT_imag_fp8_);
        std::swap(stacked_fp8_, other.stacked_fp8_);
        std::swap(stacked_b_fp8_, other.stacked_b_fp8_);
        std::swap(xi_separate_fp8_, other.xi_separate_fp8_);
        std::swap(xr_separate_fp8_, other.xr_separate_fp8_);
        std::swap(capacity_A_, other.capacity_A_);
        std::swap(capacity_B_, other.capacity_B_);
        std::swap(capacity_BT_, other.capacity_BT_);
        std::swap(stacked_capacity_, other.stacked_capacity_);
        std::swap(separate_capacity_, other.separate_capacity_);
        std::swap(A_re_stacked_, other.A_re_stacked_);
        std::swap(A_im_stacked_, other.A_im_stacked_);
        std::swap(B_re_stacked_, other.B_re_stacked_);
        std::swap(B_im_stacked_, other.B_im_stacked_);
        std::swap(stacked_gemm_cap_A_, other.stacked_gemm_cap_A_);
        std::swap(stacked_gemm_cap_B_, other.stacked_gemm_cap_B_);
    }

    cutlass::float_e4m3_t* A_real_fp8_ = nullptr;
    cutlass::float_e4m3_t* A_imag_fp8_ = nullptr;
    cutlass::float_e4m3_t* B_real_fp8_ = nullptr;
    cutlass::float_e4m3_t* B_imag_fp8_ = nullptr;
    cutlass::float_e4m3_t* BT_real_fp8_ = nullptr;
    cutlass::float_e4m3_t* BT_imag_fp8_ = nullptr;
    cutlass::float_e4m3_t* stacked_fp8_ = nullptr;
    cutlass::float_e4m3_t* stacked_b_fp8_ = nullptr;
    cutlass::float_e4m3_t* xi_separate_fp8_ = nullptr;
    cutlass::float_e4m3_t* xr_separate_fp8_ = nullptr;
    int64_t capacity_A_ = 0;
    int64_t capacity_B_ = 0;
    int64_t capacity_BT_ = 0;
    int64_t stacked_capacity_ = 0;
    int64_t separate_capacity_ = 0;
    cutlass::float_e4m3_t* A_re_stacked_ = nullptr;
    cutlass::float_e4m3_t* A_im_stacked_ = nullptr;
    cutlass::float_e4m3_t* B_re_stacked_ = nullptr;
    cutlass::float_e4m3_t* B_im_stacked_ = nullptr;
    int64_t stacked_gemm_cap_A_ = 0;
    int64_t stacked_gemm_cap_B_ = 0;
};
