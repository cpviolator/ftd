    // ---------------------------------------------------------------
    // Core real-valued FP8 GEMM (always uses the fast SS mainloop)
    // ---------------------------------------------------------------

    /// Real-valued FP8 GEMM with optional batching.
    /// When batch_count > 1, executes batch_count independent GEMMs in a single
    /// kernel launch using CUTLASS's native L (batch) dimension.
    ///
    /// Memory layout for batched operation (strided batching):
    ///   A[b] starts at A + b * M * K    (each batch element is contiguous)
    ///   B[b] starts at B + b * K * N
    ///   C[b] starts at C + b * M * N
    ///
    /// This is vastly more efficient than launching batch_count separate kernels
    /// because: (1) single kernel launch overhead, (2) the tile scheduler distributes
    /// all batch_count × tile_count tiles across SMs, (3) better GPU utilization for
    /// small per-batch problems that wouldn't fill the GPU alone.

    // ---------------------------------------------------------------
    // Strategy 5D: Stacked-K GEMM (reduce 4 sub-GEMMs to 2)
    // ---------------------------------------------------------------
    //
    // Stacks operands along K-dimension: K' = 2K, reducing 4 launches to 2.
    // Standard: Re(C) = [Ar|neg(Ai)] × [Br|Bi]^T, Im(C) = [Ar|Ai] × [Bi|Br]^T
    // Hermitian: Re(C) = [Ar|Ai] × [BrT|BiT]^T, Im(C) = [Ai|neg(Ar)] × [BrT|BiT]^T
    // Beneficial when M*N > (M+N)*K (eliminates C read-modify-write from beta=1 GEMMs).
    //
    cutlass::Status run_subgemms_fp8_stacked(
        cutlass::float_e4m3_t* A_real_fp8,
        cutlass::float_e4m3_t* A_imag_fp8,
        cutlass::float_e4m3_t* Br_fp8,
        cutlass::float_e4m3_t* Bi_fp8,
        __half* C_real, __half* C_imag,
        int M, int N, int K,
        float alpha, float beta,
        ComplexMode mode,
        cudaStream_t stream,
        int batch_count = 1)
    {
        int K2 = 2 * K;
        int64_t a_stacked = static_cast<int64_t>(M) * K2 * batch_count;
        int64_t b_stacked = static_cast<int64_t>(N) * K2 * batch_count;
        buffers_.ensure_stacked_gemm_capacity(a_stacked, b_stacked, stream);

        int M_total = M * batch_count;
        size_t bk = static_cast<size_t>(N) * K;  // B block size per batch (bytes)

        // Column-major B stacking: [B1; B2] concatenation per batch
        auto stack_B_colmajor = [&](cutlass::float_e4m3_t* b1, cutlass::float_e4m3_t* b2,
                                     cutlass::float_e4m3_t* dst) {
            if (batch_count == 1) {
                CUDA_CHECK(cudaMemcpyAsync(dst, b1, bk, cudaMemcpyDeviceToDevice, stream));
                CUDA_CHECK(cudaMemcpyAsync(dst + bk, b2, bk, cudaMemcpyDeviceToDevice, stream));
            } else {
                CUDA_CHECK(cudaMemcpy2DAsync(dst, 2 * bk, b1, bk, bk, batch_count,
                                              cudaMemcpyDeviceToDevice, stream));
                CUDA_CHECK(cudaMemcpy2DAsync(dst + bk, 2 * bk, b2, bk, bk, batch_count,
                                              cudaMemcpyDeviceToDevice, stream));
            }
        };

        if (mode == ComplexMode::Standard) {
            // A_re_stacked = [Ar | neg(Ai)] per row (row-major M×2K)
            negate_and_stack_fp8(A_real_fp8, A_imag_fp8, buffers_.A_re_stacked(), M_total, K, stream);
            // A_im_stacked = [Ar | Ai] per row
            stack_fp8(A_real_fp8, A_imag_fp8, buffers_.A_im_stacked(), M_total, K, stream);
            // B_re_stacked = [Br ; Bi] column-concat (column-major N×2K)
            stack_B_colmajor(Br_fp8, Bi_fp8, buffers_.B_re_stacked());
            // B_im_stacked = [Bi ; Br] column-concat
            stack_B_colmajor(Bi_fp8, Br_fp8, buffers_.B_im_stacked());
        } else {
            // Hermitian: B^H, operands already transposed
            // A_re_stacked = [Ar | Ai] per row
            stack_fp8(A_real_fp8, A_imag_fp8, buffers_.A_re_stacked(), M_total, K, stream);
            // A_im_stacked = [Ai | neg(Ar)] per row
            negate_and_stack_fp8(A_imag_fp8, A_real_fp8, buffers_.A_im_stacked(), M_total, K, stream);
            // B_re_stacked = [BrT ; BiT] column-concat (used for both GEMMs)
            stack_B_colmajor(Br_fp8, Bi_fp8, buffers_.B_re_stacked());
        }

        // Record stacking complete, sync GEMM streams
        CUDA_CHECK(cudaEventRecord(preprocess_done_, stream));
        CUDA_CHECK(cudaStreamWaitEvent(stream_a_, preprocess_done_));
        CUDA_CHECK(cudaStreamWaitEvent(stream_b_, preprocess_done_));

        // Strategy 4B: L2 hints for stacked operands
        ensure_hw_info();
        int64_t fp8_working_set = (static_cast<int64_t>(M) * K2 +
                                    static_cast<int64_t>(N) * K2) * batch_count;
        bool use_l2_hints = (l2_cache_bytes_ > 0 && fp8_working_set <= l2_cache_bytes_);
        if (use_l2_hints) {
            size_t persist_max = std::min(static_cast<size_t>(fp8_working_set),
                                          static_cast<size_t>(persisting_l2_max_));
            cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, persist_max);
            auto set_persist = [](void* ptr, size_t bytes, cudaStream_t s) {
                cudaStreamAttrValue attr = {};
                attr.accessPolicyWindow.base_ptr = ptr;
                attr.accessPolicyWindow.num_bytes = bytes;
                attr.accessPolicyWindow.hitRatio = 1.0f;
                attr.accessPolicyWindow.hitProp = cudaAccessPropertyPersisting;
                attr.accessPolicyWindow.missProp = cudaAccessPropertyStreaming;
                cudaStreamSetAttribute(s, cudaStreamAttributeAccessPolicyWindow, &attr);
            };
            size_t a_bytes = static_cast<size_t>(M) * K2 * batch_count;
            size_t b_bytes = static_cast<size_t>(N) * K2 * batch_count;
            set_persist(buffers_.A_re_stacked(), a_bytes, stream_a_);
            set_persist(buffers_.B_re_stacked(), b_bytes, stream_b_);
        }

        cutlass::Status status;

        // Stream A: Re(C) = A_re_stacked × B_re_stacked  (M, N, K'=2K)
        status = run_real_gemm(
            buffers_.A_re_stacked(), buffers_.B_re_stacked(),
            reinterpret_cast<cutlass::half_t*>(C_real),
            M, N, K2, alpha, beta, stream_a_, batch_count);
        if (status != cutlass::Status::kSuccess) return status;

        // Stream B: Im(C) = A_im_stacked × B_im_stacked  (M, N, K'=2K)
        auto* B_im = (mode == ComplexMode::Hermitian) ?
            buffers_.B_re_stacked() : buffers_.B_im_stacked();
        status = run_real_gemm(
            buffers_.A_im_stacked(), B_im,
            reinterpret_cast<cutlass::half_t*>(C_imag),
            M, N, K2, alpha, beta, stream_b_, batch_count);
        if (status != cutlass::Status::kSuccess) return status;

        CUDA_CHECK(cudaEventRecord(stream_a_done_, stream_a_));
        CUDA_CHECK(cudaEventRecord(stream_b_done_, stream_b_));
        CUDA_CHECK(cudaStreamWaitEvent(stream, stream_a_done_));
        CUDA_CHECK(cudaStreamWaitEvent(stream, stream_b_done_));

        if (use_l2_hints) {
            cudaStreamAttrValue attr = {};
            attr.accessPolicyWindow.num_bytes = 0;
            cudaStreamSetAttribute(stream_a_, cudaStreamAttributeAccessPolicyWindow, &attr);
            cudaStreamSetAttribute(stream_b_, cudaStreamAttributeAccessPolicyWindow, &attr);
        }

        return cutlass::Status::kSuccess;
    }

    // ---------------------------------------------------------------
    // Sub-GEMM dispatch core (shared by run_planar and fused paths)
    // ---------------------------------------------------------------
    //
    // Executes 4 real FP8 sub-GEMMs with stream parallelism.
    // All FP8 inputs must already be cast and in the correct layout.
    // C_real / C_imag are FP16 (for β accumulation and output).
    //
    // This is the compute core extracted from run_planar() so that both
    // the planar path (separate cast → run_subgemms) and fused interleaved
    // path (F1 deinterleave+cast → run_subgemms) can share it.
    //
    cutlass::Status run_subgemms_fp8(
        cutlass::float_e4m3_t* A_real_fp8,
        cutlass::float_e4m3_t* A_imag_fp8,
        cutlass::float_e4m3_t* Br_fp8,     // B (Standard) or B^T (Hermitian)
        cutlass::float_e4m3_t* Bi_fp8,
        __half* C_real, __half* C_imag,     // FP16 in/out
        int M, int N, int K,
        float alpha, float beta,
        ComplexMode mode,
        cudaStream_t stream,
        int batch_count = 1)
    {
        // Strategy 5D: Auto-select stacked-K when beneficial (saves C read-modify-write)
        if (static_cast<int64_t>(M) * N > static_cast<int64_t>(M + N) * K) {
            return run_subgemms_fp8_stacked(A_real_fp8, A_imag_fp8, Br_fp8, Bi_fp8,
                                             C_real, C_imag, M, N, K, alpha, beta,
                                             mode, stream, batch_count);
        }

        // Record event: preprocessing complete on caller's stream
        CUDA_CHECK(cudaEventRecord(preprocess_done_, stream));
        CUDA_CHECK(cudaStreamWaitEvent(stream_a_, preprocess_done_));
        CUDA_CHECK(cudaStreamWaitEvent(stream_b_, preprocess_done_));

        // Strategy 4B: L2 cache persistence hints for FP8 operand reuse.
        // Each operand (Ar, Ai, Br, Bi) is read by 2 of the 4 sub-GEMMs.
        // Hinting persistence keeps them in L2 between the first and second read.
        ensure_hw_info();
        int64_t fp8_working_set = (static_cast<int64_t>(M) * K * 2 +
                                    static_cast<int64_t>(N) * K * 2) * batch_count;
        bool use_l2_hints = (l2_cache_bytes_ > 0 && fp8_working_set <= l2_cache_bytes_);
        if (use_l2_hints) {
            size_t persist_max = std::min(static_cast<size_t>(fp8_working_set),
                                          static_cast<size_t>(persisting_l2_max_));
            cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, persist_max);
            auto set_persist = [](void* ptr, size_t bytes, cudaStream_t s) {
                cudaStreamAttrValue attr = {};
                attr.accessPolicyWindow.base_ptr = ptr;
                attr.accessPolicyWindow.num_bytes = bytes;
                attr.accessPolicyWindow.hitRatio = 1.0f;
                attr.accessPolicyWindow.hitProp = cudaAccessPropertyPersisting;
                attr.accessPolicyWindow.missProp = cudaAccessPropertyStreaming;
                cudaStreamSetAttribute(s, cudaStreamAttributeAccessPolicyWindow, &attr);
            };
            size_t a_bytes = static_cast<size_t>(M) * K * batch_count;
            size_t b_bytes = static_cast<size_t>(N) * K * batch_count;
            set_persist(A_real_fp8, a_bytes, stream_a_);
            set_persist(Br_fp8, b_bytes, stream_b_);
        }

        cutlass::Status status;

        if (mode == ComplexMode::Standard) {
            // Stream A: Re(C) = α·Ar·Br − α·Ai·Bi + β·Cr
            status = run_real_gemm(
                A_real_fp8, Br_fp8,
                reinterpret_cast<cutlass::half_t*>(C_real),
                M, N, K, alpha, beta, stream_a_, batch_count);
            if (status != cutlass::Status::kSuccess) return status;

            status = run_real_gemm(
                A_imag_fp8, Bi_fp8,
                reinterpret_cast<cutlass::half_t*>(C_real),
                M, N, K, -alpha, 1.0f, stream_a_, batch_count);
            if (status != cutlass::Status::kSuccess) return status;

            // Stream B: Im(C) = α·Ar·Bi + α·Ai·Br + β·Ci
            status = run_real_gemm(
                A_real_fp8, Bi_fp8,
                reinterpret_cast<cutlass::half_t*>(C_imag),
                M, N, K, alpha, beta, stream_b_, batch_count);
            if (status != cutlass::Status::kSuccess) return status;

            status = run_real_gemm(
                A_imag_fp8, Br_fp8,
                reinterpret_cast<cutlass::half_t*>(C_imag),
                M, N, K, alpha, 1.0f, stream_b_, batch_count);
            if (status != cutlass::Status::kSuccess) return status;

        } else {
            // Hermitian: Br_fp8/Bi_fp8 are already B^T
            // Stream A: Re(C) = α·Ar·Br^T + α·Ai·Bi^T + β·Cr
            status = run_real_gemm(
                A_real_fp8, Br_fp8,
                reinterpret_cast<cutlass::half_t*>(C_real),
                M, N, K, alpha, beta, stream_a_, batch_count);
            if (status != cutlass::Status::kSuccess) return status;

            status = run_real_gemm(
                A_imag_fp8, Bi_fp8,
                reinterpret_cast<cutlass::half_t*>(C_real),
                M, N, K, alpha, 1.0f, stream_a_, batch_count);
            if (status != cutlass::Status::kSuccess) return status;

            // Stream B: Im(C) = α·Ai·Br^T − α·Ar·Bi^T + β·Ci
            status = run_real_gemm(
                A_imag_fp8, Br_fp8,
                reinterpret_cast<cutlass::half_t*>(C_imag),
                M, N, K, alpha, beta, stream_b_, batch_count);
            if (status != cutlass::Status::kSuccess) return status;

            status = run_real_gemm(
                A_real_fp8, Bi_fp8,
                reinterpret_cast<cutlass::half_t*>(C_imag),
                M, N, K, -alpha, 1.0f, stream_b_, batch_count);
            if (status != cutlass::Status::kSuccess) return status;
        }

        // Synchronize: make the caller's stream wait for both sub-GEMM streams
        CUDA_CHECK(cudaEventRecord(stream_a_done_, stream_a_));
        CUDA_CHECK(cudaEventRecord(stream_b_done_, stream_b_));
        CUDA_CHECK(cudaStreamWaitEvent(stream, stream_a_done_));
        CUDA_CHECK(cudaStreamWaitEvent(stream, stream_b_done_));

        // Reset L2 persistence hints
        if (use_l2_hints) {
            cudaStreamAttrValue attr = {};
            attr.accessPolicyWindow.num_bytes = 0;
            cudaStreamSetAttribute(stream_a_, cudaStreamAttributeAccessPolicyWindow, &attr);
            cudaStreamSetAttribute(stream_b_, cudaStreamAttributeAccessPolicyWindow, &attr);
        }

        return cutlass::Status::kSuccess;
    }

    // ---------------------------------------------------------------
    // FP32-output FP8 GEMM — templatized on chain for multi-config dispatch
    // ---------------------------------------------------------------
    template <typename FP32OutChain>
    cutlass::Status run_real_gemm_impl_fp32out(
        const typename FP32OutChain::ElementA* A,
        const typename FP32OutChain::ElementB* B,
        float* C,
        int M, int N, int K,
        float alpha, float beta,
        cudaStream_t stream,
        int batch_count = 1,
        void* ext_workspace = nullptr,
        size_t ext_workspace_size = 0)
    {
        ensure_hw_info();

        auto stride_A = cutlass::make_cute_packed_stride(
            StrideA{}, cute::make_shape(M, K, batch_count));
        auto stride_B = cutlass::make_cute_packed_stride(
            StrideB{}, cute::make_shape(N, K, batch_count));
        auto stride_C = cutlass::make_cute_packed_stride(
            StrideC{}, cute::make_shape(M, N, batch_count));
        auto stride_D = stride_C;

        cutlass::KernelHardwareInfo hw_info;
        hw_info.device_id = 0;
        hw_info.sm_count = hw_sm_count_;

        typename FP32OutChain::GemmArguments arguments{
            cutlass::gemm::GemmUniversalMode::kGemm,
            {M, N, K, batch_count},
            {A, stride_A, B, stride_B},
            {{alpha, beta}, C, stride_C, C, stride_D},
            hw_info
        };

        typename FP32OutChain::DeviceGemm gemm_op;

        size_t workspace_size = gemm_op.get_workspace_size(arguments);
        void* workspace = nullptr;
        bool own_workspace = false;
        if (workspace_size > 0) {
            if (ext_workspace && ext_workspace_size >= workspace_size) {
                workspace = ext_workspace;
            } else {
                CUDA_CHECK(cudaMallocAsync(&workspace, workspace_size, stream));
                own_workspace = true;
            }
        }

        auto status = gemm_op.can_implement(arguments);
        if (status != cutlass::Status::kSuccess) {
            if (own_workspace && workspace) cudaFreeAsync(workspace, stream);
            return status;
        }

        status = gemm_op.initialize(arguments, workspace, stream);
        if (status != cutlass::Status::kSuccess) {
            if (own_workspace && workspace) cudaFreeAsync(workspace, stream);
            return status;
        }

        status = gemm_op.run(stream);
        if (own_workspace && workspace) cudaFreeAsync(workspace, stream);
        return status;
    }

    /// FP8 FP32-output dispatch with GemmConfig selection.
    cutlass::Status run_real_gemm_fp8_dispatch_fp32out(
        const cutlass::float_e4m3_t* A,
        const cutlass::float_e4m3_t* B,
        float* C,
        int M, int N, int K,
        float alpha, float beta,
        cudaStream_t stream,
        GemmConfig config,
        int batch_count = 1,
        void* ext_workspace = nullptr,
        size_t ext_workspace_size = 0)
    {
        switch (config) {
        case GemmConfig::FP8_T128x256_C1x1_Coop:
            return run_real_gemm_impl_fp32out<ChainFP8_T128x256_C1x1_Coop_FP32Out>(A, B, C, M, N, K, alpha, beta, stream, batch_count,
                                                                                     ext_workspace, ext_workspace_size);
        case GemmConfig::FP8_T128x128_C1x1_Coop:
            return run_real_gemm_impl_fp32out<ChainFP8_T128x128_C1x1_Coop_FP32Out>(A, B, C, M, N, K, alpha, beta, stream, batch_count,
                                                                                     ext_workspace, ext_workspace_size);
        case GemmConfig::FP8_T128x256_C1x1_PP:
            return run_real_gemm_impl_fp32out<ChainFP8_T128x256_C1x1_PP_FP32Out>(A, B, C, M, N, K, alpha, beta, stream, batch_count,
                                                                                   ext_workspace, ext_workspace_size);
        case GemmConfig::FP8_T128x128_C1x1_PP:
            return run_real_gemm_impl_fp32out<ChainFP8_T128x128_C1x1_PP_FP32Out>(A, B, C, M, N, K, alpha, beta, stream, batch_count,
                                                                                   ext_workspace, ext_workspace_size);
        case GemmConfig::FP8_T128x256_C1x2_Coop:
            return run_real_gemm_impl_fp32out<ChainFP8_T128x256_C1x2_Coop_FP32Out>(A, B, C, M, N, K, alpha, beta, stream, batch_count,
                                                                                     ext_workspace, ext_workspace_size);
        case GemmConfig::FP8_T64x128_C1x1_Coop:
            return run_real_gemm_impl_fp32out<ChainFP8_T64x128_C1x1_Coop_FP32Out>(A, B, C, M, N, K, alpha, beta, stream, batch_count,
                                                                                    ext_workspace, ext_workspace_size);
        default:
            throw std::runtime_error(
                std::string("Unsupported GemmConfig for SM90 FP8 FP32Out dispatch: ") +
                config_name(config));
        }
    }

    /// Backward-compatible FP32-output FP8 GEMM — uses class member gemm_config_.
    cutlass::Status run_real_gemm_fp32out(
        const cutlass::float_e4m3_t* A,
        const cutlass::float_e4m3_t* B,
        float* C,
        int M, int N, int K,
        float alpha, float beta,
        cudaStream_t stream,
        int batch_count = 1,
        void* ext_workspace = nullptr,
        size_t ext_workspace_size = 0)
    {
        return run_real_gemm_fp8_dispatch_fp32out(A, B, C, M, N, K, alpha, beta, stream, gemm_config_, batch_count,
                                                   ext_workspace, ext_workspace_size);
    }

    /// Query workspace size for FP8 FP32-output GEMM.
    size_t get_fp8_workspace_size_fp32out(int M, int N, int K, int batch_count = 1) {
        ensure_hw_info();
        auto stride_A = cutlass::make_cute_packed_stride(
            StrideA{}, cute::make_shape(M, K, batch_count));
        auto stride_B = cutlass::make_cute_packed_stride(
            StrideB{}, cute::make_shape(N, K, batch_count));
        auto stride_C = cutlass::make_cute_packed_stride(
            StrideC{}, cute::make_shape(M, N, batch_count));
        cutlass::KernelHardwareInfo hw_info;
        hw_info.device_id = 0;
        hw_info.sm_count = hw_sm_count_;
        GemmArgumentsFP32Out arguments{
            cutlass::gemm::GemmUniversalMode::kGemm,
            {M, N, K, batch_count},
            {nullptr, stride_A, nullptr, stride_B},
            {{1.0f, 0.0f}, nullptr, stride_C, nullptr, stride_C},
            hw_info
        };
        return DeviceGemmFP32Out::get_workspace_size(arguments);
    }

    // ---------------------------------------------------------------
    // Strategy 5D: Stacked-K GEMM with FP32 output
    // ---------------------------------------------------------------
    cutlass::Status run_subgemms_fp8_stacked_fp32out(
        cutlass::float_e4m3_t* A_real_fp8, cutlass::float_e4m3_t* A_imag_fp8,
        cutlass::float_e4m3_t* Br_fp8, cutlass::float_e4m3_t* Bi_fp8,
        float* C_real, float* C_imag,
        int M, int N, int K,
        float alpha, float beta,
        ComplexMode mode, cudaStream_t stream,
        int batch_count = 1,
        void* workspace_a = nullptr,
        void* workspace_b = nullptr,
        size_t workspace_size = 0)
    {
        int K2 = 2 * K;
        int64_t a_stacked = static_cast<int64_t>(M) * K2 * batch_count;
        int64_t b_stacked = static_cast<int64_t>(N) * K2 * batch_count;
        buffers_.ensure_stacked_gemm_capacity(a_stacked, b_stacked, stream);

        int M_total = M * batch_count;
        size_t bk = static_cast<size_t>(N) * K;

        auto stack_B_colmajor = [&](cutlass::float_e4m3_t* b1, cutlass::float_e4m3_t* b2,
                                     cutlass::float_e4m3_t* dst) {
            if (batch_count == 1) {
                CUDA_CHECK(cudaMemcpyAsync(dst, b1, bk, cudaMemcpyDeviceToDevice, stream));
                CUDA_CHECK(cudaMemcpyAsync(dst + bk, b2, bk, cudaMemcpyDeviceToDevice, stream));
            } else {
                CUDA_CHECK(cudaMemcpy2DAsync(dst, 2 * bk, b1, bk, bk, batch_count,
                                              cudaMemcpyDeviceToDevice, stream));
                CUDA_CHECK(cudaMemcpy2DAsync(dst + bk, 2 * bk, b2, bk, bk, batch_count,
                                              cudaMemcpyDeviceToDevice, stream));
            }
        };

        if (mode == ComplexMode::Standard) {
            negate_and_stack_fp8(A_real_fp8, A_imag_fp8, buffers_.A_re_stacked(), M_total, K, stream);
            stack_fp8(A_real_fp8, A_imag_fp8, buffers_.A_im_stacked(), M_total, K, stream);
            stack_B_colmajor(Br_fp8, Bi_fp8, buffers_.B_re_stacked());
            stack_B_colmajor(Bi_fp8, Br_fp8, buffers_.B_im_stacked());
        } else {
            stack_fp8(A_real_fp8, A_imag_fp8, buffers_.A_re_stacked(), M_total, K, stream);
            negate_and_stack_fp8(A_imag_fp8, A_real_fp8, buffers_.A_im_stacked(), M_total, K, stream);
            stack_B_colmajor(Br_fp8, Bi_fp8, buffers_.B_re_stacked());
        }

        CUDA_CHECK(cudaEventRecord(preprocess_done_, stream));
        CUDA_CHECK(cudaStreamWaitEvent(stream_a_, preprocess_done_));
        CUDA_CHECK(cudaStreamWaitEvent(stream_b_, preprocess_done_));

        ensure_hw_info();
        int64_t fp8_working_set = (static_cast<int64_t>(M) * K2 +
                                    static_cast<int64_t>(N) * K2) * batch_count;
        bool use_l2_hints = (l2_cache_bytes_ > 0 && fp8_working_set <= l2_cache_bytes_);
        if (use_l2_hints) {
            size_t persist_max = std::min(static_cast<size_t>(fp8_working_set),
                                          static_cast<size_t>(persisting_l2_max_));
            cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, persist_max);
            auto set_persist = [](void* ptr, size_t bytes, cudaStream_t s) {
                cudaStreamAttrValue attr = {};
                attr.accessPolicyWindow.base_ptr = ptr;
                attr.accessPolicyWindow.num_bytes = bytes;
                attr.accessPolicyWindow.hitRatio = 1.0f;
                attr.accessPolicyWindow.hitProp = cudaAccessPropertyPersisting;
                attr.accessPolicyWindow.missProp = cudaAccessPropertyStreaming;
                cudaStreamSetAttribute(s, cudaStreamAttributeAccessPolicyWindow, &attr);
            };
            size_t a_bytes = static_cast<size_t>(M) * K2 * batch_count;
            size_t b_bytes = static_cast<size_t>(N) * K2 * batch_count;
            set_persist(buffers_.A_re_stacked(), a_bytes, stream_a_);
            set_persist(buffers_.B_re_stacked(), b_bytes, stream_b_);
        }

        cutlass::Status status;

        status = run_real_gemm_fp32out(
            buffers_.A_re_stacked(), buffers_.B_re_stacked(), C_real,
            M, N, K2, alpha, beta, stream_a_, batch_count,
            workspace_a, workspace_size);
        if (status != cutlass::Status::kSuccess) return status;

        auto* B_im = (mode == ComplexMode::Hermitian) ?
            buffers_.B_re_stacked() : buffers_.B_im_stacked();
        status = run_real_gemm_fp32out(
            buffers_.A_im_stacked(), B_im, C_imag,
            M, N, K2, alpha, beta, stream_b_, batch_count,
            workspace_b, workspace_size);
        if (status != cutlass::Status::kSuccess) return status;

        CUDA_CHECK(cudaEventRecord(stream_a_done_, stream_a_));
        CUDA_CHECK(cudaEventRecord(stream_b_done_, stream_b_));
        CUDA_CHECK(cudaStreamWaitEvent(stream, stream_a_done_));
        CUDA_CHECK(cudaStreamWaitEvent(stream, stream_b_done_));

        if (use_l2_hints) {
            cudaStreamAttrValue attr = {};
            attr.accessPolicyWindow.num_bytes = 0;
            cudaStreamSetAttribute(stream_a_, cudaStreamAttributeAccessPolicyWindow, &attr);
            cudaStreamSetAttribute(stream_b_, cudaStreamAttributeAccessPolicyWindow, &attr);
        }

        return cutlass::Status::kSuccess;
    }

    // ---------------------------------------------------------------
    // 4M complex sub-GEMM dispatch with FP32 output
    // ---------------------------------------------------------------
    cutlass::Status run_subgemms_fp8_fp32out(
        cutlass::float_e4m3_t* A_real_fp8, cutlass::float_e4m3_t* A_imag_fp8,
        cutlass::float_e4m3_t* Br_fp8, cutlass::float_e4m3_t* Bi_fp8,
        float* C_real, float* C_imag,
        int M, int N, int K,
        float alpha, float beta,
        ComplexMode mode, cudaStream_t stream,
        int batch_count = 1,
        void* workspace_a = nullptr,
        void* workspace_b = nullptr,
        size_t workspace_size = 0)
    {
        // Strategy 5D: Auto-select stacked-K when beneficial
        if (static_cast<int64_t>(M) * N > static_cast<int64_t>(M + N) * K) {
            return run_subgemms_fp8_stacked_fp32out(A_real_fp8, A_imag_fp8, Br_fp8, Bi_fp8,
                                                     C_real, C_imag, M, N, K, alpha, beta,
                                                     mode, stream, batch_count,
                                                     workspace_a, workspace_b, workspace_size);
        }

        CUDA_CHECK(cudaEventRecord(preprocess_done_, stream));
        CUDA_CHECK(cudaStreamWaitEvent(stream_a_, preprocess_done_));
        CUDA_CHECK(cudaStreamWaitEvent(stream_b_, preprocess_done_));

        // Strategy 4B: L2 cache persistence hints
        ensure_hw_info();
        int64_t fp8_working_set = (static_cast<int64_t>(M) * K * 2 +
                                    static_cast<int64_t>(N) * K * 2) * batch_count;
        bool use_l2_hints = (l2_cache_bytes_ > 0 && fp8_working_set <= l2_cache_bytes_);
        if (use_l2_hints) {
            size_t persist_max = std::min(static_cast<size_t>(fp8_working_set),
                                          static_cast<size_t>(persisting_l2_max_));
            cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, persist_max);
            auto set_persist = [](void* ptr, size_t bytes, cudaStream_t s) {
                cudaStreamAttrValue attr = {};
                attr.accessPolicyWindow.base_ptr = ptr;
                attr.accessPolicyWindow.num_bytes = bytes;
                attr.accessPolicyWindow.hitRatio = 1.0f;
                attr.accessPolicyWindow.hitProp = cudaAccessPropertyPersisting;
                attr.accessPolicyWindow.missProp = cudaAccessPropertyStreaming;
                cudaStreamSetAttribute(s, cudaStreamAttributeAccessPolicyWindow, &attr);
            };
            size_t a_bytes = static_cast<size_t>(M) * K * batch_count;
            size_t b_bytes = static_cast<size_t>(N) * K * batch_count;
            set_persist(A_real_fp8, a_bytes, stream_a_);
            set_persist(Br_fp8, b_bytes, stream_b_);
        }

        cutlass::Status status;

        if (mode == ComplexMode::Standard) {
            // Stream A: Re(C) = α·Ar·Br − α·Ai·Bi + β·Cr
            status = run_real_gemm_fp32out(
                A_real_fp8, Br_fp8, C_real,
                M, N, K, alpha, beta, stream_a_, batch_count,
                workspace_a, workspace_size);
            if (status != cutlass::Status::kSuccess) return status;

            status = run_real_gemm_fp32out(
                A_imag_fp8, Bi_fp8, C_real,
                M, N, K, -alpha, 1.0f, stream_a_, batch_count,
                workspace_a, workspace_size);
            if (status != cutlass::Status::kSuccess) return status;

            // Stream B: Im(C) = α·Ar·Bi + α·Ai·Br + β·Ci
            status = run_real_gemm_fp32out(
                A_real_fp8, Bi_fp8, C_imag,
                M, N, K, alpha, beta, stream_b_, batch_count,
                workspace_b, workspace_size);
            if (status != cutlass::Status::kSuccess) return status;

            status = run_real_gemm_fp32out(
                A_imag_fp8, Br_fp8, C_imag,
                M, N, K, alpha, 1.0f, stream_b_, batch_count,
                workspace_b, workspace_size);
            if (status != cutlass::Status::kSuccess) return status;
        } else {
            // Hermitian mode
            status = run_real_gemm_fp32out(
                A_real_fp8, Br_fp8, C_real,
                M, N, K, alpha, beta, stream_a_, batch_count,
                workspace_a, workspace_size);
            if (status != cutlass::Status::kSuccess) return status;

            status = run_real_gemm_fp32out(
                A_imag_fp8, Bi_fp8, C_real,
                M, N, K, alpha, 1.0f, stream_a_, batch_count,
                workspace_a, workspace_size);
            if (status != cutlass::Status::kSuccess) return status;

            status = run_real_gemm_fp32out(
                A_imag_fp8, Br_fp8, C_imag,
                M, N, K, alpha, beta, stream_b_, batch_count,
                workspace_b, workspace_size);
            if (status != cutlass::Status::kSuccess) return status;

            status = run_real_gemm_fp32out(
                A_real_fp8, Bi_fp8, C_imag,
                M, N, K, -alpha, 1.0f, stream_b_, batch_count,
                workspace_b, workspace_size);
            if (status != cutlass::Status::kSuccess) return status;
        }

        CUDA_CHECK(cudaEventRecord(stream_a_done_, stream_a_));
        CUDA_CHECK(cudaEventRecord(stream_b_done_, stream_b_));
        CUDA_CHECK(cudaStreamWaitEvent(stream, stream_a_done_));
        CUDA_CHECK(cudaStreamWaitEvent(stream, stream_b_done_));

        // Reset L2 persistence hints
        if (use_l2_hints) {
            cudaStreamAttrValue attr = {};
            attr.accessPolicyWindow.num_bytes = 0;
            cudaStreamSetAttribute(stream_a_, cudaStreamAttributeAccessPolicyWindow, &attr);
            cudaStreamSetAttribute(stream_b_, cudaStreamAttributeAccessPolicyWindow, &attr);
        }

        return cutlass::Status::kSuccess;
    }

    /// Gram sub-GEMM dispatch: 4 real sub-GEMMs for X·X^H decomposition.
    /// Xr_A/Xi_A are A-side FP8, Xr_B/Xi_B are B-side FP8 (TMA-safe duplicates).
    /// C_real/C_imag are FP16 in/out. All pointers are device memory.
    cutlass::Status run_gram_subgemms_fp8(
        cutlass::float_e4m3_t* Xr_A, cutlass::float_e4m3_t* Xi_A,
        cutlass::float_e4m3_t* Xr_B, cutlass::float_e4m3_t* Xi_B,
        __half* C_real, __half* C_imag,
        int gemm_M, int gemm_N, int gemm_K,
        float alpha, float beta,
        cudaStream_t stream,
        int batch_count = 1)
    {
        CUDA_CHECK(cudaEventRecord(preprocess_done_, stream));
        CUDA_CHECK(cudaStreamWaitEvent(stream_a_, preprocess_done_));
        CUDA_CHECK(cudaStreamWaitEvent(stream_b_, preprocess_done_));

        cutlass::Status status;

        // Stream A: Re(C) = α·Xr·Xr^T + α·Xi·Xi^T + β·Cr
        status = run_real_gemm(Xr_A, Xr_B,
            reinterpret_cast<cutlass::half_t*>(C_real),
            gemm_M, gemm_N, gemm_K, alpha, beta, stream_a_, batch_count);
        if (status != cutlass::Status::kSuccess) return status;
        status = run_real_gemm(Xi_A, Xi_B,
            reinterpret_cast<cutlass::half_t*>(C_real),
            gemm_M, gemm_N, gemm_K, alpha, 1.0f, stream_a_, batch_count);
        if (status != cutlass::Status::kSuccess) return status;

        // Stream B: Im(C) = α·Xi·Xr^T − α·Xr·Xi^T + β·Ci
        status = run_real_gemm(Xi_A, Xr_B,
            reinterpret_cast<cutlass::half_t*>(C_imag),
            gemm_M, gemm_N, gemm_K, alpha, beta, stream_b_, batch_count);
        if (status != cutlass::Status::kSuccess) return status;
        status = run_real_gemm(Xr_A, Xi_B,
            reinterpret_cast<cutlass::half_t*>(C_imag),
            gemm_M, gemm_N, gemm_K, -alpha, 1.0f, stream_b_, batch_count);
        if (status != cutlass::Status::kSuccess) return status;

        CUDA_CHECK(cudaEventRecord(stream_a_done_, stream_a_));
        CUDA_CHECK(cudaEventRecord(stream_b_done_, stream_b_));
        CUDA_CHECK(cudaStreamWaitEvent(stream, stream_a_done_));
        CUDA_CHECK(cudaStreamWaitEvent(stream, stream_b_done_));

        return cutlass::Status::kSuccess;
    }

    /// Generic FP8 GEMM implementation — templatized on chain for multi-config dispatch.
    template <typename Chain>
    cutlass::Status run_real_gemm_impl(
        const typename Chain::ElementA* A,
        const typename Chain::ElementB* B,
        cutlass::half_t* C,
        int M, int N, int K,
        float alpha, float beta,
        cudaStream_t stream,
        int batch_count = 1,
        void* ext_workspace = nullptr,
        size_t ext_workspace_size = 0)
    {
        ensure_hw_info();

        auto stride_A = cutlass::make_cute_packed_stride(
            StrideA{}, cute::make_shape(M, K, batch_count));
        auto stride_B = cutlass::make_cute_packed_stride(
            StrideB{}, cute::make_shape(N, K, batch_count));
        auto stride_C = cutlass::make_cute_packed_stride(
            StrideC{}, cute::make_shape(M, N, batch_count));
        auto stride_D = stride_C;

        cutlass::KernelHardwareInfo hw_info;
        hw_info.device_id = 0;
        hw_info.sm_count = hw_sm_count_;

        typename Chain::GemmArguments arguments{
            cutlass::gemm::GemmUniversalMode::kGemm,
            {M, N, K, batch_count},
            {A, stride_A, B, stride_B},
            {{alpha, beta}, C, stride_C, C, stride_D},
            hw_info
        };

        typename Chain::DeviceGemm gemm_op;

        size_t workspace_size = gemm_op.get_workspace_size(arguments);
        void* workspace = nullptr;
        bool own_workspace = false;
        if (workspace_size > 0) {
            if (ext_workspace && ext_workspace_size >= workspace_size) {
                workspace = ext_workspace;
            } else {
                CUDA_CHECK(cudaMallocAsync(&workspace, workspace_size, stream));
                own_workspace = true;
            }
        }

        auto status = gemm_op.can_implement(arguments);
        if (status != cutlass::Status::kSuccess) {
            if (own_workspace && workspace) cudaFreeAsync(workspace, stream);
            return status;
        }

        status = gemm_op.initialize(arguments, workspace, stream);
        if (status != cutlass::Status::kSuccess) {
            if (own_workspace && workspace) cudaFreeAsync(workspace, stream);
            return status;
        }

        status = gemm_op.run(stream);
        if (own_workspace && workspace) cudaFreeAsync(workspace, stream);
        return status;
    }

    /// FP8 FP16-output dispatch with GemmConfig selection.
    cutlass::Status run_real_gemm_fp8_dispatch(
        const cutlass::float_e4m3_t* A,
        const cutlass::float_e4m3_t* B,
        cutlass::half_t* C,
        int M, int N, int K,
        float alpha, float beta,
        cudaStream_t stream,
        GemmConfig config,
        int batch_count = 1,
        void* ext_workspace = nullptr,
        size_t ext_workspace_size = 0)
    {
        switch (config) {
        case GemmConfig::FP8_T128x256_C1x1_Coop:
            return run_real_gemm_impl<ChainFP8_T128x256_C1x1_Coop>(A, B, C, M, N, K, alpha, beta, stream, batch_count,
                                                                     ext_workspace, ext_workspace_size);
        case GemmConfig::FP8_T128x128_C1x1_Coop:
            return run_real_gemm_impl<ChainFP8_T128x128_C1x1_Coop>(A, B, C, M, N, K, alpha, beta, stream, batch_count,
                                                                     ext_workspace, ext_workspace_size);
        case GemmConfig::FP8_T128x256_C1x1_PP:
            return run_real_gemm_impl<ChainFP8_T128x256_C1x1_PP>(A, B, C, M, N, K, alpha, beta, stream, batch_count,
                                                                   ext_workspace, ext_workspace_size);
        case GemmConfig::FP8_T128x128_C1x1_PP:
            return run_real_gemm_impl<ChainFP8_T128x128_C1x1_PP>(A, B, C, M, N, K, alpha, beta, stream, batch_count,
                                                                   ext_workspace, ext_workspace_size);
        case GemmConfig::FP8_T128x256_C1x2_Coop:
            return run_real_gemm_impl<ChainFP8_T128x256_C1x2_Coop>(A, B, C, M, N, K, alpha, beta, stream, batch_count,
                                                                     ext_workspace, ext_workspace_size);
        case GemmConfig::FP8_T64x128_C1x1_Coop:
            return run_real_gemm_impl<ChainFP8_T64x128_C1x1_Coop>(A, B, C, M, N, K, alpha, beta, stream, batch_count,
                                                                    ext_workspace, ext_workspace_size);
        default:
            throw std::runtime_error(
                std::string("Unsupported GemmConfig for SM90 FP8 dispatch: ") +
                config_name(config));
        }
    }

    /// Backward-compatible FP8 GEMM — uses class member gemm_config_.
    cutlass::Status run_real_gemm(
        const cutlass::float_e4m3_t* A,
        const cutlass::float_e4m3_t* B,
        cutlass::half_t* C,
        int M, int N, int K,
        float alpha, float beta,
        cudaStream_t stream,
        int batch_count = 1,
        void* ext_workspace = nullptr,
        size_t ext_workspace_size = 0)
    {
        return run_real_gemm_fp8_dispatch(A, B, C, M, N, K, alpha, beta, stream, gemm_config_, batch_count,
                                          ext_workspace, ext_workspace_size);
    }

    /// Query workspace size for FP8 FP16-output GEMM (used by HERK graph pre-allocation).
    size_t get_fp8_workspace_size(int M, int N, int K, int batch_count = 1) {
        ensure_hw_info();
        auto stride_A = cutlass::make_cute_packed_stride(
            StrideA{}, cute::make_shape(M, K, batch_count));
        auto stride_B = cutlass::make_cute_packed_stride(
            StrideB{}, cute::make_shape(N, K, batch_count));
        auto stride_C = cutlass::make_cute_packed_stride(
            StrideC{}, cute::make_shape(M, N, batch_count));
        cutlass::KernelHardwareInfo hw_info;
        hw_info.device_id = 0;
        hw_info.sm_count = hw_sm_count_;
        GemmArguments arguments{
            cutlass::gemm::GemmUniversalMode::kGemm,
            {M, N, K, batch_count},
            {nullptr, stride_A, nullptr, stride_B},
            {{1.0f, 0.0f}, nullptr, stride_C, nullptr, stride_C},
            hw_info
        };
        return DeviceGemm::get_workspace_size(arguments);
    }
