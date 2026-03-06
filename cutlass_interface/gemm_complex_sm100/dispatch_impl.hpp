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
        GemmConfig config = GemmConfig::Default,
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
            negate_and_stack_fp8_sm100(A_real_fp8, A_imag_fp8, buffers_.A_re_stacked(), M_total, K, stream);
            // A_im_stacked = [Ar | Ai] per row
            stack_fp8_sm100(A_real_fp8, A_imag_fp8, buffers_.A_im_stacked(), M_total, K, stream);
            // B_re_stacked = [Br ; Bi] column-concat (column-major N×2K)
            stack_B_colmajor(Br_fp8, Bi_fp8, buffers_.B_re_stacked());
            // B_im_stacked = [Bi ; Br] column-concat
            stack_B_colmajor(Bi_fp8, Br_fp8, buffers_.B_im_stacked());
        } else {
            // Hermitian: B^H, operands already transposed
            // A_re_stacked = [Ar | Ai] per row
            stack_fp8_sm100(A_real_fp8, A_imag_fp8, buffers_.A_re_stacked(), M_total, K, stream);
            // A_im_stacked = [Ai | neg(Ar)] per row
            negate_and_stack_fp8_sm100(A_imag_fp8, A_real_fp8, buffers_.A_im_stacked(), M_total, K, stream);
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
            M, N, K2, alpha, beta, stream_a_, batch_count, config);
        if (status != cutlass::Status::kSuccess) return status;

        // Stream B: Im(C) = A_im_stacked × B_im_stacked  (M, N, K'=2K)
        auto* B_im = (mode == ComplexMode::Hermitian) ?
            buffers_.B_re_stacked() : buffers_.B_im_stacked();
        status = run_real_gemm(
            buffers_.A_im_stacked(), B_im,
            reinterpret_cast<cutlass::half_t*>(C_imag),
            M, N, K2, alpha, beta, stream_b_, batch_count, config);
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

    cutlass::Status run_subgemms_fp8(
        cutlass::float_e4m3_t* A_real_fp8, cutlass::float_e4m3_t* A_imag_fp8,
        cutlass::float_e4m3_t* Br_fp8, cutlass::float_e4m3_t* Bi_fp8,
        __half* C_real, __half* C_imag,
        int M, int N, int K,
        float alpha, float beta,
        ComplexMode mode, cudaStream_t stream,
        GemmConfig config = GemmConfig::Default,
        int batch_count = 1)
    {
        // Strategy 5D: Auto-select stacked-K when beneficial (saves C read-modify-write)
        if (static_cast<int64_t>(M) * N > static_cast<int64_t>(M + N) * K) {
            return run_subgemms_fp8_stacked(A_real_fp8, A_imag_fp8, Br_fp8, Bi_fp8,
                                             C_real, C_imag, M, N, K, alpha, beta,
                                             mode, stream, config, batch_count);
        }

        CUDA_CHECK(cudaEventRecord(preprocess_done_, stream));
        CUDA_CHECK(cudaStreamWaitEvent(stream_a_, preprocess_done_));
        CUDA_CHECK(cudaStreamWaitEvent(stream_b_, preprocess_done_));

        // Strategy 4B: L2 cache persistence hints for FP8 operand reuse
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
                M, N, K, alpha, beta, stream_a_, batch_count, config);
            if (status != cutlass::Status::kSuccess) return status;

            status = run_real_gemm(
                A_imag_fp8, Bi_fp8,
                reinterpret_cast<cutlass::half_t*>(C_real),
                M, N, K, -alpha, 1.0f, stream_a_, batch_count, config);
            if (status != cutlass::Status::kSuccess) return status;

            // Stream B: Im(C) = α·Ar·Bi + α·Ai·Br + β·Ci
            status = run_real_gemm(
                A_real_fp8, Bi_fp8,
                reinterpret_cast<cutlass::half_t*>(C_imag),
                M, N, K, alpha, beta, stream_b_, batch_count, config);
            if (status != cutlass::Status::kSuccess) return status;

            status = run_real_gemm(
                A_imag_fp8, Br_fp8,
                reinterpret_cast<cutlass::half_t*>(C_imag),
                M, N, K, alpha, 1.0f, stream_b_, batch_count, config);
            if (status != cutlass::Status::kSuccess) return status;

        } else {
            // Hermitian: B^H, signs adjusted for conjugation
            // Stream A: Re(C) = α·Ar·Br^T + α·Ai·Bi^T + β·Cr
            status = run_real_gemm(
                A_real_fp8, Br_fp8,
                reinterpret_cast<cutlass::half_t*>(C_real),
                M, N, K, alpha, beta, stream_a_, batch_count, config);
            if (status != cutlass::Status::kSuccess) return status;

            status = run_real_gemm(
                A_imag_fp8, Bi_fp8,
                reinterpret_cast<cutlass::half_t*>(C_real),
                M, N, K, alpha, 1.0f, stream_a_, batch_count, config);
            if (status != cutlass::Status::kSuccess) return status;

            // Stream B: Im(C) = α·Ai·Br^T − α·Ar·Bi^T + β·Ci
            status = run_real_gemm(
                A_imag_fp8, Br_fp8,
                reinterpret_cast<cutlass::half_t*>(C_imag),
                M, N, K, alpha, beta, stream_b_, batch_count, config);
            if (status != cutlass::Status::kSuccess) return status;

            status = run_real_gemm(
                A_real_fp8, Bi_fp8,
                reinterpret_cast<cutlass::half_t*>(C_imag),
                M, N, K, -alpha, 1.0f, stream_b_, batch_count, config);
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

    // ---------------------------------------------------------------
    // Sub-GEMM dispatch core (INT8 inputs → FP16 output, INT32 accum)
    // ---------------------------------------------------------------
    //
    // SM100 only — SM120 lacks INT8 tensor cores.
    // Executes 4 real INT8 sub-GEMMs with stream parallelism.
    // Same 4M complex decomposition as run_subgemms_fp8 but with:
    //   - int8_t inputs, int32_t accumulation, half_t output
    //   - Uses ChainINT8 (IntegerKernelTypeChain)
    //
#ifndef COMPLEX_FP8_SM100_TARGET_SM120
    cutlass::Status run_subgemms_int8(
        int8_t* A_real_i8, int8_t* A_imag_i8,
        int8_t* Br_i8, int8_t* Bi_i8,
        __half* C_real, __half* C_imag,
        int M, int N, int K,
        float alpha, float beta,
        ComplexMode mode, cudaStream_t stream,
        GemmConfig config = GemmConfig::Default)
    {
        CUDA_CHECK(cudaEventRecord(preprocess_done_, stream));
        CUDA_CHECK(cudaStreamWaitEvent(stream_a_, preprocess_done_));
        CUDA_CHECK(cudaStreamWaitEvent(stream_b_, preprocess_done_));

        cutlass::Status status;

        if (mode == ComplexMode::Standard) {
            // Stream A: Re(C) = α·Ar·Br − α·Ai·Bi + β·Cr
            status = run_real_gemm_impl<ChainINT8>(
                reinterpret_cast<const int8_t*>(A_real_i8),
                reinterpret_cast<const int8_t*>(Br_i8),
                reinterpret_cast<cutlass::half_t*>(C_real),
                M, N, K, alpha, beta, stream_a_);
            if (status != cutlass::Status::kSuccess) return status;

            status = run_real_gemm_impl<ChainINT8>(
                reinterpret_cast<const int8_t*>(A_imag_i8),
                reinterpret_cast<const int8_t*>(Bi_i8),
                reinterpret_cast<cutlass::half_t*>(C_real),
                M, N, K, -alpha, 1.0f, stream_a_);
            if (status != cutlass::Status::kSuccess) return status;

            // Stream B: Im(C) = α·Ar·Bi + α·Ai·Br + β·Ci
            status = run_real_gemm_impl<ChainINT8>(
                reinterpret_cast<const int8_t*>(A_real_i8),
                reinterpret_cast<const int8_t*>(Bi_i8),
                reinterpret_cast<cutlass::half_t*>(C_imag),
                M, N, K, alpha, beta, stream_b_);
            if (status != cutlass::Status::kSuccess) return status;

            status = run_real_gemm_impl<ChainINT8>(
                reinterpret_cast<const int8_t*>(A_imag_i8),
                reinterpret_cast<const int8_t*>(Br_i8),
                reinterpret_cast<cutlass::half_t*>(C_imag),
                M, N, K, alpha, 1.0f, stream_b_);
            if (status != cutlass::Status::kSuccess) return status;
        } else {
            // Hermitian: B^H, signs adjusted for conjugation
            // Stream A: Re(C) = α·Ar·Br^T + α·Ai·Bi^T + β·Cr
            status = run_real_gemm_impl<ChainINT8>(
                reinterpret_cast<const int8_t*>(A_real_i8),
                reinterpret_cast<const int8_t*>(Br_i8),
                reinterpret_cast<cutlass::half_t*>(C_real),
                M, N, K, alpha, beta, stream_a_);
            if (status != cutlass::Status::kSuccess) return status;

            status = run_real_gemm_impl<ChainINT8>(
                reinterpret_cast<const int8_t*>(A_imag_i8),
                reinterpret_cast<const int8_t*>(Bi_i8),
                reinterpret_cast<cutlass::half_t*>(C_real),
                M, N, K, alpha, 1.0f, stream_a_);
            if (status != cutlass::Status::kSuccess) return status;

            // Stream B: Im(C) = −α·Ar·Bi^T + α·Ai·Br^T + β·Ci
            status = run_real_gemm_impl<ChainINT8>(
                reinterpret_cast<const int8_t*>(A_real_i8),
                reinterpret_cast<const int8_t*>(Bi_i8),
                reinterpret_cast<cutlass::half_t*>(C_imag),
                M, N, K, -alpha, beta, stream_b_);
            if (status != cutlass::Status::kSuccess) return status;

            status = run_real_gemm_impl<ChainINT8>(
                reinterpret_cast<const int8_t*>(A_imag_i8),
                reinterpret_cast<const int8_t*>(Br_i8),
                reinterpret_cast<cutlass::half_t*>(C_imag),
                M, N, K, alpha, 1.0f, stream_b_);
            if (status != cutlass::Status::kSuccess) return status;
        }

        CUDA_CHECK(cudaEventRecord(stream_a_done_, stream_a_));
        CUDA_CHECK(cudaEventRecord(stream_b_done_, stream_b_));
        CUDA_CHECK(cudaStreamWaitEvent(stream, stream_a_done_));
        CUDA_CHECK(cudaStreamWaitEvent(stream, stream_b_done_));

        return cutlass::Status::kSuccess;
    }
#endif // !COMPLEX_FP8_SM100_TARGET_SM120

    // ---------------------------------------------------------------
    // INT4 via FP16 helper (for FP6/FP4 MXFP pipeline)
    // ---------------------------------------------------------------
    //
    // Deinterleaves INT4 → FP16 planar, then delegates to run_planar()
    // which handles MXFP preprocessing (scale factors + quantization).
    //
    cutlass::Status run_int4_via_fp16(
        const uint8_t* A, const uint8_t* B, __half* C,
        int M, int N, int K,
        float alpha, float beta,
        ComputePrecision precision,
        cudaStream_t stream,
        GemmConfig config)
    {
        int64_t size_A = static_cast<int64_t>(M) * K;
        int64_t size_B = static_cast<int64_t>(K) * N;
        int64_t size_C = static_cast<int64_t>(M) * N;

        __half *Ar, *Ai, *Br, *Bi, *Cr, *Ci;
        CUDA_CHECK(cudaMallocAsync(&Ar, size_A * sizeof(__half), stream));
        CUDA_CHECK(cudaMallocAsync(&Ai, size_A * sizeof(__half), stream));
        CUDA_CHECK(cudaMallocAsync(&Br, size_B * sizeof(__half), stream));
        CUDA_CHECK(cudaMallocAsync(&Bi, size_B * sizeof(__half), stream));
        CUDA_CHECK(cudaMallocAsync(&Cr, size_C * sizeof(__half), stream));
        CUDA_CHECK(cudaMallocAsync(&Ci, size_C * sizeof(__half), stream));

        // INT4 → FP16 planar
        deinterleave_int4_to_fp16(A, Ar, Ai, size_A, stream);
        deinterleave_int4_to_fp16(B, Br, Bi, size_B, stream);
        if (beta != 0.0f) {
            deinterleave_complex(C, Cr, Ci, size_C, stream);
        }

        auto mode = ComplexMode::Standard;
        auto status = run_planar(Ar, Ai, Br, Bi, Cr, Ci,
                                 M, N, K, alpha, beta, mode, precision, stream, config);

        if (status == cutlass::Status::kSuccess) {
            interleave_complex(Cr, Ci, C, size_C, stream);
        }

        cudaFreeAsync(Ar, stream);
        cudaFreeAsync(Ai, stream);
        cudaFreeAsync(Br, stream);
        cudaFreeAsync(Bi, stream);
        cudaFreeAsync(Cr, stream);
        cudaFreeAsync(Ci, stream);

        return status;
    }

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
            // Print diagnostic once per chain type to avoid log spam
            static bool printed_can_impl_fail = false;
            if (!printed_can_impl_fail) {
                printed_can_impl_fail = true;
                int max_smem_per_block = 0;
                cudaDeviceGetAttribute(&max_smem_per_block,
                    cudaDevAttrMaxSharedMemoryPerBlockOptin, 0);
                if (sizeof(typename Chain::GemmKernel::SharedStorage) <= static_cast<size_t>(max_smem_per_block)) {
                    fprintf(stderr, "[CUTLASS] can_implement FAILED (status=%d) for GEMM(%d,%d,%d)\n",
                            static_cast<int>(status), M, N, K);
                    fprintf(stderr, "[CUTLASS] Device max SMEM per block (optin): %d bytes (%d KB)\n",
                            max_smem_per_block, max_smem_per_block / 1024);
                    fprintf(stderr, "[CUTLASS] Kernel SMEM requirement: %zu bytes (%zu KB)\n",
                            sizeof(typename Chain::GemmKernel::SharedStorage),
                            sizeof(typename Chain::GemmKernel::SharedStorage) / 1024);
                }
            }
            if (own_workspace && workspace) cudaFreeAsync(workspace, stream);
            return status;
        }

        status = gemm_op.initialize(arguments, workspace, stream);
        if (status != cutlass::Status::kSuccess) {
            static bool printed_init_fail = false;
            if (!printed_init_fail) {
                printed_init_fail = true;
                int max_smem_per_block = 0;
                cudaDeviceGetAttribute(&max_smem_per_block,
                    cudaDevAttrMaxSharedMemoryPerBlockOptin, 0);
                if (sizeof(typename Chain::GemmKernel::SharedStorage) <= static_cast<size_t>(max_smem_per_block)) {
                    fprintf(stderr, "[CUTLASS] initialize FAILED (status=%d) for GEMM(%d,%d,%d)\n",
                            static_cast<int>(status), M, N, K);
                    fprintf(stderr, "[CUTLASS] Device max SMEM per block (optin): %d bytes (%d KB)\n",
                            max_smem_per_block, max_smem_per_block / 1024);
                    fprintf(stderr, "[CUTLASS] Kernel SMEM requirement: %zu bytes (%zu KB)\n",
                            sizeof(typename Chain::GemmKernel::SharedStorage),
                            sizeof(typename Chain::GemmKernel::SharedStorage) / 1024);
                }
            }
            if (own_workspace && workspace) cudaFreeAsync(workspace, stream);
            return status;
        }

        // Print pre-launch diagnostics on first call (per chain type)
        static bool printed_diag = false;
        if (!printed_diag) {
            printed_diag = true;
            int max_smem_per_block = 0;
            cudaDeviceGetAttribute(&max_smem_per_block,
                cudaDevAttrMaxSharedMemoryPerBlockOptin, 0);
            fprintf(stderr, "[DIAG] Device max SMEM per block (optin): %d bytes (%d KB)\n",
                    max_smem_per_block, max_smem_per_block / 1024);
            fprintf(stderr, "[DIAG] Kernel SharedStorage size: %zu bytes (%zu KB)\n",
                    sizeof(typename Chain::GemmKernel::SharedStorage),
                    sizeof(typename Chain::GemmKernel::SharedStorage) / 1024);
            fprintf(stderr, "[DIAG] GEMM tile: %dx%dx%d  cluster: %dx%dx%d\n",
                    int(cute::size<0>(typename Chain::TileShape{})),
                    int(cute::size<1>(typename Chain::TileShape{})),
                    int(cute::size<2>(typename Chain::TileShape{})),
                    int(cute::size<0>(typename Chain::ClusterShapeType{})),
                    int(cute::size<1>(typename Chain::ClusterShapeType{})),
                    int(cute::size<2>(typename Chain::ClusterShapeType{})));
        }

        status = gemm_op.run(stream);
        // NOTE: no cudaStreamSynchronize — caller batches multiple launches
        // on stream_a_/stream_b_ for 2-stream overlap. Errors are caught at
        // the end-of-operation sync points (event waits, triangle sync, etc.).
        if (own_workspace && workspace) cudaFreeAsync(workspace, stream);
        return status;
    }

    /// Backward-compatible FP8 GEMM — delegates to run_real_gemm_fp8_dispatch.
    cutlass::Status run_real_gemm(
        const cutlass::float_e4m3_t* A,
        const cutlass::float_e4m3_t* B,
        cutlass::half_t* C,
        int M, int N, int K,
        float alpha, float beta,
        cudaStream_t stream,
        int batch_count = 1,
        GemmConfig config = GemmConfig::Default,
        void* ext_workspace = nullptr,
        size_t ext_workspace_size = 0)
    {
        return run_real_gemm_fp8_dispatch(A, B, C, M, N, K, alpha, beta, stream, config, batch_count,
                                          ext_workspace, ext_workspace_size);
    }

public:
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
            static bool printed_can_impl_fail_fp32 = false;
            if (!printed_can_impl_fail_fp32) {
                printed_can_impl_fail_fp32 = true;
                int max_smem_per_block = 0;
                cudaDeviceGetAttribute(&max_smem_per_block,
                    cudaDevAttrMaxSharedMemoryPerBlockOptin, 0);
                if (sizeof(typename FP32OutChain::GemmKernel::SharedStorage) <= static_cast<size_t>(max_smem_per_block)) {
                    fprintf(stderr, "[CUTLASS FP32Out] can_implement FAILED (status=%d) for GEMM(%d,%d,%d)\n",
                            static_cast<int>(status), M, N, K);
                }
            }
            if (own_workspace && workspace) cudaFreeAsync(workspace, stream);
            return status;
        }

        status = gemm_op.initialize(arguments, workspace, stream);
        if (status != cutlass::Status::kSuccess) {
            static bool printed_init_fail_fp32 = false;
            if (!printed_init_fail_fp32) {
                printed_init_fail_fp32 = true;
                int max_smem_per_block = 0;
                cudaDeviceGetAttribute(&max_smem_per_block,
                    cudaDevAttrMaxSharedMemoryPerBlockOptin, 0);
                if (sizeof(typename FP32OutChain::GemmKernel::SharedStorage) <= static_cast<size_t>(max_smem_per_block)) {
                    fprintf(stderr, "[CUTLASS FP32Out] initialize FAILED (status=%d) for GEMM(%d,%d,%d)\n",
                            static_cast<int>(status), M, N, K);
                }
            }
            if (own_workspace && workspace) cudaFreeAsync(workspace, stream);
            return status;
        }

        status = gemm_op.run(stream);
        if (own_workspace && workspace) cudaFreeAsync(workspace, stream);
        return status;
    }

    /// Backward-compatible FP8 FP32-output GEMM — delegates to default chain.
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
        return run_real_gemm_impl_fp32out<ChainFP8_FP32Out>(
            A, B, C, M, N, K, alpha, beta, stream, batch_count,
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
        typename ChainFP8_FP32Out::GemmArguments arguments{
            cutlass::gemm::GemmUniversalMode::kGemm,
            {M, N, K, batch_count},
            {nullptr, stride_A, nullptr, stride_B},
            {{1.0f, 0.0f}, nullptr, stride_C, nullptr, stride_C},
            hw_info
        };
        return ChainFP8_FP32Out::DeviceGemm::get_workspace_size(arguments);
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
        typename ChainFP8::GemmArguments arguments{
            cutlass::gemm::GemmUniversalMode::kGemm,
            {M, N, K, batch_count},
            {nullptr, stride_A, nullptr, stride_B},
            {{1.0f, 0.0f}, nullptr, stride_C, nullptr, stride_C},
            hw_info
        };
        return ChainFP8::DeviceGemm::get_workspace_size(arguments);
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
            negate_and_stack_fp8_sm100(A_real_fp8, A_imag_fp8, buffers_.A_re_stacked(), M_total, K, stream);
            stack_fp8_sm100(A_real_fp8, A_imag_fp8, buffers_.A_im_stacked(), M_total, K, stream);
            stack_B_colmajor(Br_fp8, Bi_fp8, buffers_.B_re_stacked());
            stack_B_colmajor(Bi_fp8, Br_fp8, buffers_.B_im_stacked());
        } else {
            stack_fp8_sm100(A_real_fp8, A_imag_fp8, buffers_.A_re_stacked(), M_total, K, stream);
            negate_and_stack_fp8_sm100(A_imag_fp8, A_real_fp8, buffers_.A_im_stacked(), M_total, K, stream);
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

    // ---------------------------------------------------------------
    // Planar batched complex GEMM with FP32 output
    // ---------------------------------------------------------------
    // Accepts FP16 input, internally converts to FP8, outputs FP32.
    // Avoids FP16 overflow for large-K problems.
    //
    cutlass::Status run_planar_batched_fp32out(
        const __half* A_real, const __half* A_imag,
        const __half* B_real, const __half* B_imag,
        float* C_real, float* C_imag,
        int M, int N, int K,
        int batch_count,
        float alpha = 1.0f, float beta = 0.0f,
        ComplexMode mode = ComplexMode::Standard,
        cudaStream_t stream = nullptr)
    {
        if (batch_count <= 0) return cutlass::Status::kSuccess;
        bool is_hermitian = (mode == ComplexMode::Hermitian);
        ensure_streams();

        int64_t size_A = (int64_t)M * K;
        int64_t size_B = (int64_t)N * K;
        int64_t total_A = size_A * batch_count;
        int64_t total_B = size_B * batch_count;

        buffers_.ensure_capacity(total_A, total_B, is_hermitian ? total_B : 0, stream);

        // Cast all batch elements FP16→FP8 (element-wise, layout-agnostic)
        cast_fp16_to_fp8_e4m3_paired_sm100(
            A_real, A_imag,
            reinterpret_cast<__nv_fp8_e4m3*>(buffers_.A_real()),
            reinterpret_cast<__nv_fp8_e4m3*>(buffers_.A_imag()),
            total_A, stream);

        cutlass::float_e4m3_t* Br_ptr;
        cutlass::float_e4m3_t* Bi_ptr;

        if (!is_hermitian) {
            cast_fp16_to_fp8_e4m3_paired_sm100(
                B_real, B_imag,
                reinterpret_cast<__nv_fp8_e4m3*>(buffers_.B_real()),
                reinterpret_cast<__nv_fp8_e4m3*>(buffers_.B_imag()),
                total_B, stream);
            Br_ptr = buffers_.B_real();
            Bi_ptr = buffers_.B_imag();
        } else {
            cast_fp16_to_fp8_e4m3_transposed_paired_sm100(
                B_real, B_imag,
                buffers_.BT_real(), buffers_.BT_imag(),
                K, N, stream, batch_count);
            Br_ptr = buffers_.BT_real();
            Bi_ptr = buffers_.BT_imag();
        }

        return run_subgemms_fp8_fp32out(
            buffers_.A_real(), buffers_.A_imag(),
            Br_ptr, Bi_ptr,
            C_real, C_imag,
            M, N, K, alpha, beta, mode, stream, batch_count);
    }

    // ---------------------------------------------------------------
    // Multi-Precision Planar Batched Complex GEMM with FP32 output
    // ---------------------------------------------------------------
    // Same as run_planar_batched_fp32out but with selectable compute precision.
    // For FP8: delegates to the existing FP8 run_planar_batched_fp32out path.
    // For FP6/FP4: uses LowPrecisionBufferManager + MXFP preprocessing,
    //   then dispatches 4 sub-GEMMs via run_real_gemm_dispatch_fp32out.
    //
    cutlass::Status run_planar_batched_fp32out(
        const __half* A_real, const __half* A_imag,
        const __half* B_real, const __half* B_imag,
        float* C_real, float* C_imag,
        int M, int N, int K,
        int batch_count,
        float alpha, float beta,
        ComplexMode mode,
        ComputePrecision precision,
        cudaStream_t stream = nullptr)
    {
        // FP8 fast path: delegate to existing optimized method
        if (precision == ComputePrecision::FP8_E4M3) {
            return run_planar_batched_fp32out(A_real, A_imag, B_real, B_imag,
                                              C_real, C_imag, M, N, K, batch_count,
                                              alpha, beta, mode, stream);
        }

        // MXFP block-scaled path (FP6/FP4)
        if (batch_count <= 0) return cutlass::Status::kSuccess;

        ensure_streams();
        ensure_hw_info();

        int64_t size_A = (int64_t)M * K;
        int64_t size_B = (int64_t)N * K;
        int64_t total_A = size_A * batch_count;
        int64_t total_B = size_B * batch_count;

        // Allocate sub-byte data buffers (flat layout — batch stride = M*K or N*K elements)
        int64_t bytes_A = bytes_for_elements(total_A, precision);
        int64_t bytes_B = bytes_for_elements(total_B, precision);
        lp_buffers_.ensure_capacity(bytes_A, bytes_B, 0, stream);

        // Allocate scale factor buffers.
        // SF tile atoms are 128-row blocks. When a dimension is not a multiple of 128,
        // flat (M*batch) preprocessing interleaves batches within tile atoms, but the
        // batched GEMM (tile_atom_to_shape_SFA) expects each batch to have its own
        // padded tile atoms. Use per-batch SF allocation+preprocessing in that case.
        bool per_batch_sf_A = (M % 128 != 0);
        bool per_batch_sf_B = (N % 128 != 0);
        int64_t sf_per_A = sf_buffer_bytes(M, K);  // SF bytes per single batch for A
        int64_t sf_per_B = sf_buffer_bytes(N, K);   // SF bytes per single batch for B
        int64_t sf_A_bytes = per_batch_sf_A ? (int64_t)batch_count * sf_per_A
                                            : sf_buffer_bytes(M * batch_count, K);
        int64_t sf_B_bytes = per_batch_sf_B ? (int64_t)batch_count * sf_per_B
                                            : sf_buffer_bytes(N * batch_count, K);
        lp_buffers_.ensure_sf_capacity(sf_A_bytes, sf_B_bytes, 0, stream);

        // Preprocess A: fused SF + scale + cast for Ar and Ai
        if (per_batch_sf_A) {
            // Batched preprocessing: single kernel launch for all batch elements
            int64_t data_bytes_per = bytes_for_elements(size_A, precision);
            preprocess_mxfp_paired_batched_sm100(
                A_real, A_imag,
                lp_buffers_.A_real(), lp_buffers_.A_imag(),
                lp_buffers_.sf_A_real(), lp_buffers_.sf_A_imag(),
                M, K, batch_count,
                data_bytes_per, sf_per_A,
                precision, stream);
        } else {
            // Flat: M is a multiple of 128, so tile atoms align across batches
            preprocess_mxfp_paired_sm100(
                A_real, A_imag,
                lp_buffers_.A_real(), lp_buffers_.A_imag(),
                lp_buffers_.sf_A_real(), lp_buffers_.sf_A_imag(),
                M * batch_count, K, precision, stream);
        }

        // Preprocess B: fused SF + scale + cast for Br and Bi
        if (per_batch_sf_B) {
            // Batched preprocessing: single kernel launch for all batch elements
            int64_t data_bytes_per = bytes_for_elements(size_B, precision);
            preprocess_mxfp_paired_batched_sm100(
                B_real, B_imag,
                lp_buffers_.B_real(), lp_buffers_.B_imag(),
                lp_buffers_.sf_B_real(), lp_buffers_.sf_B_imag(),
                N, K, batch_count,
                data_bytes_per, sf_per_B,
                precision, stream);
        } else {
            // Flat: N is a multiple of 128, so tile atoms align across batches
            preprocess_mxfp_paired_sm100(
                B_real, B_imag,
                lp_buffers_.B_real(), lp_buffers_.B_imag(),
                lp_buffers_.sf_B_real(), lp_buffers_.sf_B_imag(),
                N * batch_count, K, precision, stream);
        }

        // Auto-select optimal tile config based on M dimension
        GemmConfig config = select_gemm_config(M);

        // Dispatch 4 sub-GEMMs with FP32 output
        CUDA_CHECK(cudaEventRecord(preprocess_done_, stream));
        CUDA_CHECK(cudaStreamWaitEvent(stream_a_, preprocess_done_));
        CUDA_CHECK(cudaStreamWaitEvent(stream_b_, preprocess_done_));

        cutlass::Status status;

        // Stream A: Re(C) = alpha*Ar*Br - alpha*Ai*Bi + beta*Cr
        status = run_real_gemm_dispatch_fp32out(
            lp_buffers_.A_real(), lp_buffers_.B_real(),
            lp_buffers_.sf_A_real(), lp_buffers_.sf_B_real(),
            C_real, M, N, K, alpha, beta, stream_a_, precision,
            config, batch_count);
        if (status != cutlass::Status::kSuccess) return status;

        status = run_real_gemm_dispatch_fp32out(
            lp_buffers_.A_imag(), lp_buffers_.B_imag(),
            lp_buffers_.sf_A_imag(), lp_buffers_.sf_B_imag(),
            C_real, M, N, K, -alpha, 1.0f, stream_a_, precision,
            config, batch_count);
        if (status != cutlass::Status::kSuccess) return status;

        // Stream B: Im(C) = alpha*Ar*Bi + alpha*Ai*Br + beta*Ci
        status = run_real_gemm_dispatch_fp32out(
            lp_buffers_.A_real(), lp_buffers_.B_imag(),
            lp_buffers_.sf_A_real(), lp_buffers_.sf_B_imag(),
            C_imag, M, N, K, alpha, beta, stream_b_, precision,
            config, batch_count);
        if (status != cutlass::Status::kSuccess) return status;

        status = run_real_gemm_dispatch_fp32out(
            lp_buffers_.A_imag(), lp_buffers_.B_real(),
            lp_buffers_.sf_A_imag(), lp_buffers_.sf_B_real(),
            C_imag, M, N, K, alpha, 1.0f, stream_b_, precision,
            config, batch_count);
        if (status != cutlass::Status::kSuccess) return status;

        CUDA_CHECK(cudaEventRecord(stream_a_done_, stream_a_));
        CUDA_CHECK(cudaEventRecord(stream_b_done_, stream_b_));
        CUDA_CHECK(cudaStreamWaitEvent(stream, stream_a_done_));
        CUDA_CHECK(cudaStreamWaitEvent(stream, stream_b_done_));

        return cutlass::Status::kSuccess;
    }

    /// Runtime-dispatched GEMM for multi-precision paths. Takes void* inputs.
    /// For FP8: delegates to run_real_gemm_impl (no scale factors).
    /// For FP6/FP4: delegates to concrete functions in gemm_blockscaled_dispatch.cu.
    ///   These live in a .cu TU to fix kernel registration (CUTLASS issue #2478).
    cutlass::Status run_real_gemm_dispatch(
        const void* A, const void* B,
        const void* SFA, const void* SFB,
        cutlass::half_t* C,
        int M, int N, int K,
        float alpha, float beta,
        cudaStream_t stream,
        ComputePrecision precision,
        GemmConfig config = GemmConfig::Default,
        int batch_count = 1,
        int ld_C = 0,
        void* ext_workspace = nullptr,
        size_t ext_workspace_size = 0)
    {
        ensure_hw_info();
        switch (precision) {
        case ComputePrecision::FP8_E4M3:
            return run_real_gemm_fp8_dispatch(
                static_cast<const cutlass::float_e4m3_t*>(A),
                static_cast<const cutlass::float_e4m3_t*>(B),
                C, M, N, K, alpha, beta, stream, config, batch_count,
                ext_workspace, ext_workspace_size);
#ifdef COMPLEX_SM100_ENABLE_FP6
        case ComputePrecision::FP6_E3M2:
            return run_blockscaled_gemm_fp6_e3m2(
                A, B, SFA, SFB,
                C, M, N, K, alpha, beta, stream, hw_sm_count_, config, batch_count, ld_C,
                ext_workspace, ext_workspace_size);
        case ComputePrecision::FP6_E2M3:
            return run_blockscaled_gemm_fp6_e2m3(
                A, B, SFA, SFB,
                C, M, N, K, alpha, beta, stream, hw_sm_count_, config, batch_count, ld_C,
                ext_workspace, ext_workspace_size);
#endif
#ifdef COMPLEX_SM100_ENABLE_FP4
        case ComputePrecision::FP4_E2M1:
            return run_blockscaled_gemm_fp4(
                A, B, SFA, SFB,
                C, M, N, K, alpha, beta, stream, hw_sm_count_, config, batch_count, ld_C,
                ext_workspace, ext_workspace_size);
#endif
        default:
            throw std::runtime_error(
                "Unsupported compute precision in run_real_gemm_dispatch. "
                "Enable at compile time: -DCOMPLEX_SM100_ENABLE_FP6=1 or -DCOMPLEX_SM100_ENABLE_FP4=1");
        }
    }

    /// Runtime-dispatched GEMM for multi-precision paths with FP32 output.
    /// For FP8: delegates to run_real_gemm_fp32out (no scale factors).
    /// For FP6/FP4: delegates to FP32-output concrete functions in gemm_blockscaled_dispatch.cu.
    cutlass::Status run_real_gemm_dispatch_fp32out(
        const void* A, const void* B,
        const void* SFA, const void* SFB,
        float* C,
        int M, int N, int K,
        float alpha, float beta,
        cudaStream_t stream,
        ComputePrecision precision,
        GemmConfig config = GemmConfig::Default,
        int batch_count = 1,
        int ld_C = 0,
        void* ext_workspace = nullptr,
        size_t ext_workspace_size = 0)
    {
        ensure_hw_info();
        switch (precision) {
        case ComputePrecision::FP8_E4M3:
            return run_real_gemm_fp8_dispatch_fp32out(
                static_cast<const cutlass::float_e4m3_t*>(A),
                static_cast<const cutlass::float_e4m3_t*>(B),
                C, M, N, K, alpha, beta, stream, config, batch_count,
                ext_workspace, ext_workspace_size);
#ifdef COMPLEX_SM100_ENABLE_FP6
        case ComputePrecision::FP6_E3M2:
            return run_blockscaled_gemm_fp6_e3m2_fp32out(
                A, B, SFA, SFB,
                C, M, N, K, alpha, beta, stream, hw_sm_count_, config, batch_count, ld_C,
                ext_workspace, ext_workspace_size);
        case ComputePrecision::FP6_E2M3:
            throw std::runtime_error(
                "FP6 E2M3 with FP32 output is not yet implemented. "
                "Use FP6 E3M2 or FP8 for FP32 output.");
#endif
#ifdef COMPLEX_SM100_ENABLE_FP4
        case ComputePrecision::FP4_E2M1:
            return run_blockscaled_gemm_fp4_fp32out(
                A, B, SFA, SFB,
                C, M, N, K, alpha, beta, stream, hw_sm_count_, config, batch_count, ld_C,
                ext_workspace, ext_workspace_size);
#endif
        default:
            throw std::runtime_error(
                "Unsupported compute precision in run_real_gemm_dispatch_fp32out. "
                "Enable at compile time: -DCOMPLEX_SM100_ENABLE_FP6=1 or -DCOMPLEX_SM100_ENABLE_FP4=1");
        }
    }

    /// Query workspace size for any precision, suitable for pre-allocation.
    size_t compute_workspace_size(int M, int N, int K, int batch_count,
                                  ComputePrecision precision) {
        ensure_hw_info();
        switch (precision) {
        case ComputePrecision::FP8_E4M3:
            return get_fp8_workspace_size_fp32out(M, N, K, batch_count);
#ifdef COMPLEX_SM100_ENABLE_FP6
        case ComputePrecision::FP6_E3M2:
            return get_blockscaled_workspace_size_fp6_e3m2_fp32out(
                M, N, K, hw_sm_count_, batch_count);
        case ComputePrecision::FP6_E2M3:
            return 0;  // FP6 E2M3 FP32 output not yet implemented
#endif
#ifdef COMPLEX_SM100_ENABLE_FP4
        case ComputePrecision::FP4_E2M1:
            return get_blockscaled_workspace_size_fp4_fp32out(
                M, N, K, hw_sm_count_, batch_count);
#endif
        default:
            return 0;
        }
    }

    /// Query FP16-output workspace size for any precision, suitable for pre-allocation.
    size_t compute_workspace_size_fp16(int M, int N, int K, int batch_count,
                                       ComputePrecision precision) {
        ensure_hw_info();
        switch (precision) {
        case ComputePrecision::FP8_E4M3:
            return get_fp8_workspace_size(M, N, K, batch_count);
#ifdef COMPLEX_SM100_ENABLE_FP6
        case ComputePrecision::FP6_E3M2:
            return get_blockscaled_workspace_size_fp6_e3m2(
                M, N, K, hw_sm_count_, batch_count);
        case ComputePrecision::FP6_E2M3:
            return get_blockscaled_workspace_size_fp6_e2m3(
                M, N, K, hw_sm_count_, batch_count);
#endif
#ifdef COMPLEX_SM100_ENABLE_FP4
        case ComputePrecision::FP4_E2M1:
            return get_blockscaled_workspace_size_fp4(
                M, N, K, hw_sm_count_, batch_count);
#endif
        default:
            return 0;
        }
    }

    /// FP8 FP16-output dispatch with GemmConfig selection (stays in header — no nvcc stub issue).
    /// All configs are always compiled. Invalid configs for the current arch throw at runtime.
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
        // --- SM120 + SM100: 1×1 cluster, default stages (StageCount<3> when configured) ---
        case GemmConfig::FP8_T128x64_C1x1_S3:
            return run_real_gemm_impl<ChainFP8_T128x64_C1x1>(A, B, C, M, N, K, alpha, beta, stream, batch_count,
                                                               ext_workspace, ext_workspace_size);
        // --- SM120 + SM100: 1×1 cluster, auto-carveout stages ---
        case GemmConfig::FP8_T128x64_C1x1_SAuto:
            return run_real_gemm_impl<ChainFP8_T128x64_C1x1_Auto>(A, B, C, M, N, K, alpha, beta, stream, batch_count,
                                                                    ext_workspace, ext_workspace_size);
        case GemmConfig::FP8_T128x128_C1x1_SAuto:
            return run_real_gemm_impl<ChainFP8_T128x128_C1x1_Auto>(A, B, C, M, N, K, alpha, beta, stream, batch_count,
                                                                     ext_workspace, ext_workspace_size);
        // --- SM100 only: configs that exceed SM120's 99 KB SMEM or require clusters ---
#ifndef COMPLEX_FP8_SM100_TARGET_SM120
        case GemmConfig::FP8_T128x128_C1x1_S3:
            return run_real_gemm_impl<ChainFP8_T128x128_C1x1>(A, B, C, M, N, K, alpha, beta, stream, batch_count,
                                                                ext_workspace, ext_workspace_size);
        case GemmConfig::FP8_T128x256_C1x1_SAuto:
            return run_real_gemm_impl<ChainFP8_T128x256_C1x1_Auto>(A, B, C, M, N, K, alpha, beta, stream, batch_count,
                                                                     ext_workspace, ext_workspace_size);
        case GemmConfig::FP8_T128x128_C1x2:
            return run_real_gemm_impl<ChainFP8_T128x128_C1x2>(A, B, C, M, N, K, alpha, beta, stream, batch_count,
                                                                ext_workspace, ext_workspace_size);
        case GemmConfig::FP8_T128x128_C2x2:
            return run_real_gemm_impl<ChainFP8_T128x128_C2x2>(A, B, C, M, N, K, alpha, beta, stream, batch_count,
                                                                ext_workspace, ext_workspace_size);
        case GemmConfig::FP8_T128x256_C1x2:
            return run_real_gemm_impl<ChainFP8_T128x256_C1x2>(A, B, C, M, N, K, alpha, beta, stream, batch_count,
                                                                ext_workspace, ext_workspace_size);
#endif
        // 2SM configs (FP8_T256x128_C2x1_2SM, FP8_T256x256_C2x2_2SM) require
        // 2SM-schedule chains (not yet implemented — reserved for future work).

        default:
            throw std::runtime_error(
                std::string("Unsupported GemmConfig for SM100/SM120 FP8 dispatch: ") +
                config_name(config));
        }
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
        case GemmConfig::FP8_T128x64_C1x1_S3:
            return run_real_gemm_impl_fp32out<ChainFP8_T128x64_C1x1_FP32Out>(A, B, C, M, N, K, alpha, beta, stream, batch_count,
                                                                               ext_workspace, ext_workspace_size);
        case GemmConfig::FP8_T128x64_C1x1_SAuto:
            return run_real_gemm_impl_fp32out<ChainFP8_T128x64_C1x1_Auto_FP32Out>(A, B, C, M, N, K, alpha, beta, stream, batch_count,
                                                                                    ext_workspace, ext_workspace_size);
        case GemmConfig::FP8_T128x128_C1x1_SAuto:
            return run_real_gemm_impl_fp32out<ChainFP8_T128x128_C1x1_Auto_FP32Out>(A, B, C, M, N, K, alpha, beta, stream, batch_count,
                                                                                     ext_workspace, ext_workspace_size);
        // --- SM100 only: configs that exceed SM120's 99 KB SMEM or require clusters ---
#ifndef COMPLEX_FP8_SM100_TARGET_SM120
        case GemmConfig::FP8_T128x128_C1x1_S3:
            return run_real_gemm_impl_fp32out<ChainFP8_T128x128_C1x1_FP32Out>(A, B, C, M, N, K, alpha, beta, stream, batch_count,
                                                                                ext_workspace, ext_workspace_size);
        case GemmConfig::FP8_T128x256_C1x1_SAuto:
            return run_real_gemm_impl_fp32out<ChainFP8_T128x256_C1x1_Auto_FP32Out>(A, B, C, M, N, K, alpha, beta, stream, batch_count,
                                                                                     ext_workspace, ext_workspace_size);
        case GemmConfig::FP8_T128x128_C1x2:
            return run_real_gemm_impl_fp32out<ChainFP8_T128x128_C1x2_FP32Out>(A, B, C, M, N, K, alpha, beta, stream, batch_count,
                                                                                ext_workspace, ext_workspace_size);
        case GemmConfig::FP8_T128x128_C2x2:
            return run_real_gemm_impl_fp32out<ChainFP8_T128x128_C2x2_FP32Out>(A, B, C, M, N, K, alpha, beta, stream, batch_count,
                                                                                ext_workspace, ext_workspace_size);
        case GemmConfig::FP8_T128x256_C1x2:
            return run_real_gemm_impl_fp32out<ChainFP8_T128x256_C1x2_FP32Out>(A, B, C, M, N, K, alpha, beta, stream, batch_count,
                                                                                ext_workspace, ext_workspace_size);
#endif
        default:
            throw std::runtime_error(
                std::string("Unsupported GemmConfig for SM100/SM120 FP8 FP32Out dispatch: ") +
                config_name(config));
        }
    }

