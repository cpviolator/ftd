    cutlass::Status run_planar(
        const __half* A_real, const __half* A_imag,
        const __half* B_real, const __half* B_imag,
        __half* C_real, __half* C_imag,
        int M, int N, int K,
        float alpha = 1.0f,
        float beta  = 0.0f,
        ComplexMode mode = ComplexMode::Standard,
        cudaStream_t stream = nullptr,
        GemmConfig config = GemmConfig::Default)
    {
        bool is_hermitian = (mode == ComplexMode::Hermitian);
        ensure_streams();

        int64_t size_A = static_cast<int64_t>(M) * K;
        int64_t size_B = static_cast<int64_t>(K) * N;
        buffers_.ensure_capacity(size_A, size_B, is_hermitian ? size_B : 0, stream);

        // Preprocess: FP16 → FP8
        // F5: Fused paired cast — Re+Im in one launch
        cast_fp16_to_fp8_e4m3_paired_sm100(
            A_real, A_imag,
            reinterpret_cast<__nv_fp8_e4m3*>(buffers_.A_real()),
            reinterpret_cast<__nv_fp8_e4m3*>(buffers_.A_imag()),
            size_A, stream);

        cutlass::float_e4m3_t* Br_ptr;
        cutlass::float_e4m3_t* Bi_ptr;

        if (!is_hermitian) {
            // F5: Fused paired cast — Re+Im in one launch
            cast_fp16_to_fp8_e4m3_paired_sm100(
                B_real, B_imag,
                reinterpret_cast<__nv_fp8_e4m3*>(buffers_.B_real()),
                reinterpret_cast<__nv_fp8_e4m3*>(buffers_.B_imag()),
                size_B, stream);
            Br_ptr = buffers_.B_real();
            Bi_ptr = buffers_.B_imag();
        } else {
            // F5: Fused paired cast+transpose — Re+Im in one launch
            cast_fp16_to_fp8_e4m3_transposed_paired_sm100(
                B_real, B_imag,
                buffers_.BT_real(), buffers_.BT_imag(),
                K, N, stream);
            Br_ptr = buffers_.BT_real();
            Bi_ptr = buffers_.BT_imag();
        }

        return run_subgemms_fp8(
            buffers_.A_real(), buffers_.A_imag(),
            Br_ptr, Bi_ptr,
            C_real, C_imag,
            M, N, K, alpha, beta, mode, stream, config);
    }

    // ---------------------------------------------------------------
    // Batched Planar Complex (FP8 only)
    // ---------------------------------------------------------------
    //
    // Same as run_planar() but processes batch_count independent GEMMs
    // in a single launch. Input/output arrays are contiguous with
    // stride M*K (A), N*K (B), M*N (C) between batch elements.
    //
    // FP16→FP8 cast is done over the total element count (layout-agnostic),
    // then batch_count is forwarded to run_subgemms_fp8 → run_real_gemm.
    //
    cutlass::Status run_planar_batched(
        const __half* A_real, const __half* A_imag,
        const __half* B_real, const __half* B_imag,
        __half* C_real, __half* C_imag,
        int M, int N, int K,
        int batch_count,
        float alpha = 1.0f, float beta = 0.0f,
        ComplexMode mode = ComplexMode::Standard,
        cudaStream_t stream = nullptr,
        GemmConfig config = GemmConfig::Default)
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

        return run_subgemms_fp8(
            buffers_.A_real(), buffers_.A_imag(),
            Br_ptr, Bi_ptr,
            reinterpret_cast<__half*>(C_real),
            reinterpret_cast<__half*>(C_imag),
            M, N, K, alpha, beta, mode, stream, config, batch_count);
    }

    // ---------------------------------------------------------------
    // Multi-Precision Planar Complex
    // ---------------------------------------------------------------
    //
    // Same as run_planar() but with selectable compute precision.
    // For FP8, delegates to the existing run_planar() path.
    // For FP6/FP4, uses the LowPrecisionBufferManager and dispatches
    // to the appropriate CUTLASS kernel chain.
    //
    // Note: Only the basic linear cast path is implemented in Stage 1.
    // Transposed cast (Hermitian mode) for FP6/FP4 is added in Stage 2.
    //
    cutlass::Status run_planar(
        const __half* A_real, const __half* A_imag,
        const __half* B_real, const __half* B_imag,
        __half* C_real, __half* C_imag,
        int M, int N, int K,
        float alpha, float beta,
        ComplexMode mode,
        ComputePrecision precision,
        cudaStream_t stream = nullptr,
        GemmConfig config = GemmConfig::Default)
    {
        // FP8 fast path: use existing optimized path
        if (precision == ComputePrecision::FP8_E4M3) {
            return run_planar(A_real, A_imag, B_real, B_imag,
                              C_real, C_imag, M, N, K, alpha, beta, mode, stream, config);
        }

        // MXFP sub-byte precision path (FP6/FP4)
        // Sub-byte types on SM100/SM120 require block-scaled (MXFP) infrastructure:
        //   1. Compute per-32-element scale factors from FP16 input
        //   2. Scale FP16 data by inverse of scale factors
        //   3. Cast scaled FP16 to sub-byte packed format
        //   4. Run block-scaled GEMM with data + scale factor pointers

        bool is_hermitian = (mode == ComplexMode::Hermitian);

        ensure_streams();

        int64_t size_A = static_cast<int64_t>(M) * K;
        int64_t size_B = static_cast<int64_t>(K) * N;

        // Allocate sub-byte data buffers
        int64_t bytes_A = bytes_for_elements(size_A, precision);
        int64_t bytes_B = bytes_for_elements(size_B, precision);
        lp_buffers_.ensure_capacity(bytes_A, bytes_B,
                                    is_hermitian ? bytes_B : 0, stream);

        // Allocate scale factor buffers (SfAtom layout)
        // A is M×K RowMajor, B is N×K ColumnMajor (K-major for both)
        int64_t sf_A_bytes = sf_buffer_bytes(M, K);
        int64_t sf_B_bytes = sf_buffer_bytes(N, K);  // B is N×K
        lp_buffers_.ensure_sf_capacity(sf_A_bytes, sf_B_bytes,
                                       is_hermitian ? sf_B_bytes : 0, stream);

        // Preprocess A: fused SF + scale + cast for both Ar and Ai in 1 launch
        preprocess_mxfp_paired_sm100(
            A_real, A_imag,
            lp_buffers_.A_real(), lp_buffers_.A_imag(),
            lp_buffers_.sf_A_real(), lp_buffers_.sf_A_imag(),
            M, K, precision, stream);

        // Preprocess B: depends on mode
        void *Br_ptr, *Bi_ptr, *sf_Br_ptr, *sf_Bi_ptr;

        if (!is_hermitian) {
            // Standard: B is N×K ColumnMajor — fused paired preprocessing
            preprocess_mxfp_paired_sm100(
                B_real, B_imag,
                lp_buffers_.B_real(), lp_buffers_.B_imag(),
                lp_buffers_.sf_B_real(), lp_buffers_.sf_B_imag(),
                N, K, precision, stream);
            Br_ptr = lp_buffers_.B_real();
            Bi_ptr = lp_buffers_.B_imag();
            sf_Br_ptr = lp_buffers_.sf_B_real();
            sf_Bi_ptr = lp_buffers_.sf_B_imag();
        } else {
            // Hermitian: B^H = conj(B)^T — fused paired preprocessing
            // Scale factors are computed for the N×K layout.
            preprocess_mxfp_paired_sm100(
                B_real, B_imag,
                lp_buffers_.BT_real(), lp_buffers_.BT_imag(),
                lp_buffers_.sf_BT_real(), lp_buffers_.sf_BT_imag(),
                N, K, precision, stream);
            Br_ptr = lp_buffers_.BT_real();
            Bi_ptr = lp_buffers_.BT_imag();
            sf_Br_ptr = lp_buffers_.sf_BT_real();
            sf_Bi_ptr = lp_buffers_.sf_BT_imag();
        }

        // Dispatch 4 sub-GEMMs with scale factor pointers
        CUDA_CHECK(cudaEventRecord(preprocess_done_, stream));
        CUDA_CHECK(cudaStreamWaitEvent(stream_a_, preprocess_done_));
        CUDA_CHECK(cudaStreamWaitEvent(stream_b_, preprocess_done_));

        cutlass::Status status;

        // Stream A: Re(C) = α·Ar·Br − α·Ai·Bi + β·Cr
        status = run_real_gemm_dispatch(
            lp_buffers_.A_real(), Br_ptr,
            lp_buffers_.sf_A_real(), sf_Br_ptr,
            reinterpret_cast<cutlass::half_t*>(C_real),
            M, N, K, alpha, beta, stream_a_, precision, config);
        if (status != cutlass::Status::kSuccess) return status;

        status = run_real_gemm_dispatch(
            lp_buffers_.A_imag(), Bi_ptr,
            lp_buffers_.sf_A_imag(), sf_Bi_ptr,
            reinterpret_cast<cutlass::half_t*>(C_real),
            M, N, K, -alpha, 1.0f, stream_a_, precision, config);
        if (status != cutlass::Status::kSuccess) return status;

        // Stream B: Im(C) = α·Ar·Bi + α·Ai·Br + β·Ci
        status = run_real_gemm_dispatch(
            lp_buffers_.A_real(), Bi_ptr,
            lp_buffers_.sf_A_real(), sf_Bi_ptr,
            reinterpret_cast<cutlass::half_t*>(C_imag),
            M, N, K, alpha, beta, stream_b_, precision, config);
        if (status != cutlass::Status::kSuccess) return status;

        status = run_real_gemm_dispatch(
            lp_buffers_.A_imag(), Br_ptr,
            lp_buffers_.sf_A_imag(), sf_Br_ptr,
            reinterpret_cast<cutlass::half_t*>(C_imag),
            M, N, K, alpha, 1.0f, stream_b_, precision, config);
        if (status != cutlass::Status::kSuccess) return status;

        CUDA_CHECK(cudaEventRecord(stream_a_done_, stream_a_));
        CUDA_CHECK(cudaEventRecord(stream_b_done_, stream_b_));
        CUDA_CHECK(cudaStreamWaitEvent(stream, stream_a_done_));
        CUDA_CHECK(cudaStreamWaitEvent(stream, stream_b_done_));

        return cutlass::Status::kSuccess;
    }

    // ---------------------------------------------------------------
    // Convenience Interface: Interleaved Complex
    // ---------------------------------------------------------------
    //
    // F1 fusion: deinterleave+cast A and B directly from interleaved FP16 to
    // planar FP8, eliminating intermediate FP16 planar buffers for A and B.
    // C is still deinterleaved to FP16 (read-write during β-accumulation).
    //
    cutlass::Status run(
        __half* A, __half* B, __half* C,
        int M, int N, int K,
        bool is_hermitian = false,
        float alpha = 1.0f,
        float beta  = 0.0f,
        cudaStream_t stream = nullptr,
        GemmConfig config = GemmConfig::Default)
    {
        ensure_streams();

        int64_t size_A = static_cast<int64_t>(M) * K;
        int64_t size_B = static_cast<int64_t>(K) * N;
        int64_t size_C = static_cast<int64_t>(M) * N;

        // Allocate FP8 scratch buffers (no FP16 planar temps for A/B!)
        buffers_.ensure_capacity(size_A, size_B, is_hermitian ? size_B : 0, stream);

        // F1: Fused deinterleave + cast for A (interleaved FP16 → planar FP8)
        deinterleave_cast_fp16_to_fp8(
            A,
            reinterpret_cast<__nv_fp8_e4m3*>(buffers_.A_real()),
            reinterpret_cast<__nv_fp8_e4m3*>(buffers_.A_imag()),
            size_A, stream);

        // F1: Fused deinterleave + cast (+ transpose) for B
        cutlass::float_e4m3_t* Br_ptr;
        cutlass::float_e4m3_t* Bi_ptr;

        if (!is_hermitian) {
            deinterleave_cast_fp16_to_fp8(
                B,
                reinterpret_cast<__nv_fp8_e4m3*>(buffers_.B_real()),
                reinterpret_cast<__nv_fp8_e4m3*>(buffers_.B_imag()),
                size_B, stream);
            Br_ptr = buffers_.B_real();
            Bi_ptr = buffers_.B_imag();
        } else {
            // Fused deinterleave + cast + transpose: interleaved ColMajor [K×N] → 2 × FP8 ColMajor [N×K]
            deinterleave_cast_fp16_to_fp8_transposed(
                B, buffers_.BT_real(), buffers_.BT_imag(),
                K, N, stream);
            Br_ptr = buffers_.BT_real();
            Bi_ptr = buffers_.BT_imag();
        }

        // C still needs FP16 planar for β accumulation and sub-GEMM output
        __half *Cr, *Ci;
        CUDA_CHECK(cudaMallocAsync(&Cr, size_C * sizeof(__half), stream));
        CUDA_CHECK(cudaMallocAsync(&Ci, size_C * sizeof(__half), stream));
        deinterleave_complex(C, Cr, Ci, size_C, stream);

        // Dispatch 4 sub-GEMMs via shared compute core
        auto mode = is_hermitian ? ComplexMode::Hermitian : ComplexMode::Standard;
        auto status = run_subgemms_fp8(
            buffers_.A_real(), buffers_.A_imag(),
            Br_ptr, Bi_ptr,
            Cr, Ci,
            M, N, K, alpha, beta, mode, stream, config);

        // Reinterleave result back to C
        if (status == cutlass::Status::kSuccess) {
            interleave_complex(Cr, Ci, C, size_C, stream);
        }

        cudaFreeAsync(Cr, stream);
        cudaFreeAsync(Ci, stream);

        return status;
    }

    /// Multi-precision interleaved complex GEMM.
    /// For FP8: uses optimized F1 deinterleave+cast path.
    /// For FP6/FP4: deinterleaves to FP16 planar, then delegates to multi-precision
    /// run_planar() which handles MXFP preprocessing (scale factors + quantization).
    cutlass::Status run(
        __half* A, __half* B, __half* C,
        int M, int N, int K,
        bool is_hermitian,
        float alpha, float beta,
        ComputePrecision precision,
        cudaStream_t stream = nullptr,
        GemmConfig config = GemmConfig::Default)
    {
        // FP8 fast path: optimized F1 deinterleave+cast
        if (precision == ComputePrecision::FP8_E4M3) {
            return run(A, B, C, M, N, K, is_hermitian, alpha, beta, stream, config);
        }

        // Sub-byte MXFP: deinterleave to FP16 planar → delegate to run_planar
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

        deinterleave_complex(A, Ar, Ai, size_A, stream);
        deinterleave_complex(B, Br, Bi, size_B, stream);
        deinterleave_complex(C, Cr, Ci, size_C, stream);

        auto mode = is_hermitian ? ComplexMode::Hermitian : ComplexMode::Standard;
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

    // ---------------------------------------------------------------
    // INT4 Complex GEMM — Sign-Magnitude INT4 Input
    // ---------------------------------------------------------------
    //
    // Input: packed INT4 interleaved complex (1 byte = 1 complex element).
    //   High nibble = Re (sign-magnitude), Low nibble = Im (sign-magnitude).
    //   Range: [-7, +7] per component.
    //
    // Dispatches to one of four compute strategies:
    //   FP8_E4M3  — INT4 → FP8, FP32 accumulation (exact)
    //   INT8      — INT4 → INT8, INT32 accumulation (exact)
    //   FP6_E3M2  — INT4 → FP16 → MXFP FP6, FP32 accumulation (exact)
    //   FP4_E2M1  — INT4 → FP16 → MXFP FP4, FP32 accumulation (lossy: 5→4, 7→8)
    //
    cutlass::Status run_int4(
        const uint8_t* A,     // INT4 interleaved complex [M × K], row-major
        const uint8_t* B,     // INT4 interleaved complex [N × K], row-major (TN layout)
        __half* C,            // FP16 interleaved complex [M × N], row-major (in/out)
        int M, int N, int K,
        float alpha = 1.0f,
        float beta  = 0.0f,
        ComputeStrategy strategy = ComputeStrategy::FP8_E4M3,
        cudaStream_t stream = nullptr,
        GemmConfig config = GemmConfig::Default)
    {
        int64_t size_A = static_cast<int64_t>(M) * K;
        int64_t size_B = static_cast<int64_t>(K) * N;
        int64_t size_C = static_cast<int64_t>(M) * N;

        // --- FP6/FP4: deinterleave INT4 → FP16, delegate to MXFP pipeline ---
#ifdef COMPLEX_SM100_ENABLE_FP6
        if (strategy == ComputeStrategy::FP6_E3M2) {
            return run_int4_via_fp16(A, B, C, M, N, K, alpha, beta,
                                     ComputePrecision::FP6_E3M2, stream, config);
        }
#endif
#ifdef COMPLEX_SM100_ENABLE_FP4
        if (strategy == ComputeStrategy::FP4_E2M1) {
            return run_int4_via_fp16(A, B, C, M, N, K, alpha, beta,
                                     ComputePrecision::FP4_E2M1, stream, config);
        }
#endif

        // --- FP8 / INT8: direct conversion paths ---
        ensure_streams();

        // Reuse FP8 buffer manager (INT8 is also 1 byte/element)
        buffers_.ensure_capacity(size_A, size_B, 0, stream);

        // Deinterleave C for beta accumulation
        __half *Cr, *Ci;
        CUDA_CHECK(cudaMallocAsync(&Cr, size_C * sizeof(__half), stream));
        CUDA_CHECK(cudaMallocAsync(&Ci, size_C * sizeof(__half), stream));
        if (beta != 0.0f) {
            deinterleave_complex(C, Cr, Ci, size_C, stream);
        }

        auto mode = ComplexMode::Standard;
        cutlass::Status status = cutlass::Status::kErrorNotSupported;

        if (strategy == ComputeStrategy::FP8_E4M3) {
            // Strategy A: INT4 → FP8 (exact)
            deinterleave_int4_to_fp8(
                A,
                reinterpret_cast<__nv_fp8_e4m3*>(buffers_.A_real()),
                reinterpret_cast<__nv_fp8_e4m3*>(buffers_.A_imag()),
                size_A, stream);
            deinterleave_int4_to_fp8(
                B,
                reinterpret_cast<__nv_fp8_e4m3*>(buffers_.B_real()),
                reinterpret_cast<__nv_fp8_e4m3*>(buffers_.B_imag()),
                size_B, stream);

            status = run_subgemms_fp8(
                buffers_.A_real(), buffers_.A_imag(),
                buffers_.B_real(), buffers_.B_imag(),
                Cr, Ci, M, N, K, alpha, beta, mode, stream, config);

        }
#ifndef COMPLEX_FP8_SM100_TARGET_SM120
        else if (strategy == ComputeStrategy::INT8) {
            // Strategy B: INT4 → INT8 (exact, INT32 accumulation) — SM100 only
            deinterleave_int4_to_int8(
                A,
                reinterpret_cast<int8_t*>(buffers_.A_real()),
                reinterpret_cast<int8_t*>(buffers_.A_imag()),
                size_A, stream);
            deinterleave_int4_to_int8(
                B,
                reinterpret_cast<int8_t*>(buffers_.B_real()),
                reinterpret_cast<int8_t*>(buffers_.B_imag()),
                size_B, stream);

            status = run_subgemms_int8(
                reinterpret_cast<int8_t*>(buffers_.A_real()),
                reinterpret_cast<int8_t*>(buffers_.A_imag()),
                reinterpret_cast<int8_t*>(buffers_.B_real()),
                reinterpret_cast<int8_t*>(buffers_.B_imag()),
                Cr, Ci, M, N, K, alpha, beta, mode, stream, config);
        }
#endif

        if (status == cutlass::Status::kSuccess) {
            interleave_complex(Cr, Ci, C, size_C, stream);
        }

        cudaFreeAsync(Cr, stream);
        cudaFreeAsync(Ci, stream);
        return status;
    }

    // ---------------------------------------------------------------
    // Gram Matrix: Planar Complex
    // ---------------------------------------------------------------
    cutlass::Status run_gram_planar(
        const __half* A_real, const __half* A_imag,
        __half* C_real, __half* C_imag,
        int M, int K,
        float alpha = 1.0f,
        float beta  = 0.0f,
        GramMode mode = GramMode::AAH,
        cudaStream_t stream = nullptr,
        GemmConfig config = GemmConfig::Default)
    {
        ensure_streams();

        int64_t size_A = static_cast<int64_t>(M) * K;

        // TMA L2 contention fix: separate A-side and B-side buffers
        cutlass::float_e4m3_t *Xr_A, *Xi_A, *Xr_B, *Xi_B;
        int gemm_M, gemm_N, gemm_K;

        if (mode == GramMode::AAH) {
            gemm_M = M; gemm_N = M; gemm_K = K;
            buffers_.ensure_capacity(size_A, size_A, 0, stream);

            // F2: Fused quad cast + duplicate — single kernel writes to all 4 outputs
            cast_fp16_to_fp8_e4m3_quad(A_real, A_imag,
                reinterpret_cast<__nv_fp8_e4m3*>(buffers_.A_real()),
                reinterpret_cast<__nv_fp8_e4m3*>(buffers_.B_real()),
                reinterpret_cast<__nv_fp8_e4m3*>(buffers_.A_imag()),
                reinterpret_cast<__nv_fp8_e4m3*>(buffers_.B_imag()), size_A, stream);

            Xr_A = buffers_.A_real();  Xi_A = buffers_.A_imag();
            Xr_B = buffers_.B_real();  Xi_B = buffers_.B_imag();
        } else {
            gemm_M = K; gemm_N = K; gemm_K = M;
            buffers_.ensure_capacity(size_A, 0, size_A, stream);

            // F2: Fused quad cast + transpose + duplicate
            cast_fp16_to_fp8_e4m3_transposed_quad(A_real, A_imag,
                buffers_.BT_real(), buffers_.A_real(),
                buffers_.BT_imag(), buffers_.A_imag(), K, M, stream);

            Xr_A = buffers_.BT_real();  Xi_A = buffers_.BT_imag();
            Xr_B = buffers_.A_real();   Xi_B = buffers_.A_imag();
        }

        CUDA_CHECK(cudaEventRecord(preprocess_done_, stream));
        CUDA_CHECK(cudaStreamWaitEvent(stream_a_, preprocess_done_));
        CUDA_CHECK(cudaStreamWaitEvent(stream_b_, preprocess_done_));

        cutlass::Status status;
        auto* Cr = reinterpret_cast<cutlass::half_t*>(C_real);
        auto* Ci = reinterpret_cast<cutlass::half_t*>(C_imag);

        // Stream A: Re(C) — A_ptr != B_ptr for self-products
        status = run_real_gemm(Xr_A, Xr_B, Cr, gemm_M, gemm_N, gemm_K, alpha, beta, stream_a_, 1, config);
        if (status != cutlass::Status::kSuccess) return status;
        status = run_real_gemm(Xi_A, Xi_B, Cr, gemm_M, gemm_N, gemm_K, alpha, 1.0f, stream_a_, 1, config);
        if (status != cutlass::Status::kSuccess) return status;

        // Stream B: Im(C) — cross-terms already use different buffers
        status = run_real_gemm(Xi_A, Xr_B, Ci, gemm_M, gemm_N, gemm_K, alpha, beta, stream_b_, 1, config);
        if (status != cutlass::Status::kSuccess) return status;
        status = run_real_gemm(Xr_A, Xi_B, Ci, gemm_M, gemm_N, gemm_K, -alpha, 1.0f, stream_b_, 1, config);
        if (status != cutlass::Status::kSuccess) return status;

        CUDA_CHECK(cudaEventRecord(stream_a_done_, stream_a_));
        CUDA_CHECK(cudaEventRecord(stream_b_done_, stream_b_));
        CUDA_CHECK(cudaStreamWaitEvent(stream, stream_a_done_));
        CUDA_CHECK(cudaStreamWaitEvent(stream, stream_b_done_));

        return cutlass::Status::kSuccess;
    }

    /// Multi-precision Gram matrix.
    /// For MXFP (FP6/FP4), delegates to run_planar with Hermitian mode.
    /// AAH = A × A^H: pass A as both inputs with Hermitian mode.
    /// AHA = A^H × A: transpose, then pass as Hermitian.
    cutlass::Status run_gram_planar(
        const __half* A_real, const __half* A_imag,
        __half* C_real, __half* C_imag,
        int M, int K,
        float alpha, float beta,
        GramMode mode,
        ComputePrecision precision,
        cudaStream_t stream = nullptr,
        GemmConfig config = GemmConfig::Default)
    {
        if (precision == ComputePrecision::FP8_E4M3) {
            return run_gram_planar(A_real, A_imag, C_real, C_imag,
                                   M, K, alpha, beta, mode, stream, config);
        }

        // For MXFP sub-byte types, delegate through run_planar() which handles
        // all MXFP preprocessing. This is simpler than duplicating F2 dual-cast
        // MXFP logic for self-product paths.
        if (mode == GramMode::AAH) {
            return run_planar(A_real, A_imag, A_real, A_imag, C_real, C_imag,
                              M, M, K, alpha, beta, ComplexMode::Hermitian,
                              precision, stream, config);
        } else {
            return run_planar(A_real, A_imag, A_real, A_imag, C_real, C_imag,
                              K, K, M, alpha, beta, ComplexMode::Hermitian,
                              precision, stream, config);
        }
    }


    // ====================================================================
    // Strategy 3A: FP8 Native Input API
    // ====================================================================

    /// FP8 interleaved complex GEMM (single).
    cutlass::Status run_fp8_interleaved(
        const __nv_fp8_e4m3* A_interleaved,
        const __nv_fp8_e4m3* B_interleaved,
        __half* C_real, __half* C_imag,
        int M, int N, int K,
        float alpha = 1.0f, float beta = 0.0f,
        ComplexMode mode = ComplexMode::Standard,
        cudaStream_t stream = nullptr,
        GemmConfig config = GemmConfig::Default)
    {
        bool is_hermitian = (mode == ComplexMode::Hermitian);
        ensure_streams();

        int64_t size_A = static_cast<int64_t>(M) * K;
        int64_t size_B = static_cast<int64_t>(K) * N;

        buffers_.ensure_capacity(size_A, size_B, is_hermitian ? size_B : 0, stream);

        deinterleave_fp8_sm100(
            A_interleaved,
            reinterpret_cast<__nv_fp8_e4m3*>(buffers_.A_real()),
            reinterpret_cast<__nv_fp8_e4m3*>(buffers_.A_imag()),
            size_A, stream);

        cutlass::float_e4m3_t* Br_ptr;
        cutlass::float_e4m3_t* Bi_ptr;

        if (!is_hermitian) {
            deinterleave_fp8_sm100(
                B_interleaved,
                reinterpret_cast<__nv_fp8_e4m3*>(buffers_.B_real()),
                reinterpret_cast<__nv_fp8_e4m3*>(buffers_.B_imag()),
                size_B, stream);
            Br_ptr = buffers_.B_real();
            Bi_ptr = buffers_.B_imag();
        } else {
            deinterleave_fp8_sm100(
                B_interleaved,
                reinterpret_cast<__nv_fp8_e4m3*>(buffers_.B_real()),
                reinterpret_cast<__nv_fp8_e4m3*>(buffers_.B_imag()),
                size_B, stream);
            transpose_fp8(buffers_.B_real(), buffers_.BT_real(), K, N, stream);
            transpose_fp8(buffers_.B_imag(), buffers_.BT_imag(), K, N, stream);
            Br_ptr = buffers_.BT_real();
            Bi_ptr = buffers_.BT_imag();
        }

        return run_subgemms_fp8(
            buffers_.A_real(), buffers_.A_imag(),
            Br_ptr, Bi_ptr,
            C_real, C_imag,
            M, N, K, alpha, beta, mode, stream, config);
    }

    /// FP8 interleaved complex batched GEMM.
    cutlass::Status run_fp8_interleaved_batched(
        const __nv_fp8_e4m3* A_interleaved,
        const __nv_fp8_e4m3* B_interleaved,
        __half* C_real, __half* C_imag,
        int M, int N, int K,
        int batch_count,
        float alpha = 1.0f, float beta = 0.0f,
        ComplexMode mode = ComplexMode::Standard,
        cudaStream_t stream = nullptr,
        GemmConfig config = GemmConfig::Default)
    {
        if (batch_count <= 0) return cutlass::Status::kSuccess;
        bool is_hermitian = (mode == ComplexMode::Hermitian);
        ensure_streams();

        int64_t per_A = static_cast<int64_t>(M) * K;
        int64_t per_B = static_cast<int64_t>(K) * N;
        int64_t total_A = per_A * batch_count;
        int64_t total_B = per_B * batch_count;

        buffers_.ensure_capacity(total_A, total_B, is_hermitian ? total_B : 0, stream);

        deinterleave_fp8_sm100(
            A_interleaved,
            reinterpret_cast<__nv_fp8_e4m3*>(buffers_.A_real()),
            reinterpret_cast<__nv_fp8_e4m3*>(buffers_.A_imag()),
            total_A, stream);

        cutlass::float_e4m3_t* Br_ptr;
        cutlass::float_e4m3_t* Bi_ptr;

        if (!is_hermitian) {
            deinterleave_fp8_sm100(
                B_interleaved,
                reinterpret_cast<__nv_fp8_e4m3*>(buffers_.B_real()),
                reinterpret_cast<__nv_fp8_e4m3*>(buffers_.B_imag()),
                total_B, stream);
            Br_ptr = buffers_.B_real();
            Bi_ptr = buffers_.B_imag();
        } else {
            deinterleave_fp8_sm100(
                B_interleaved,
                reinterpret_cast<__nv_fp8_e4m3*>(buffers_.B_real()),
                reinterpret_cast<__nv_fp8_e4m3*>(buffers_.B_imag()),
                total_B, stream);
            for (int b = 0; b < batch_count; ++b) {
                transpose_fp8(buffers_.B_real() + b * per_B,
                              buffers_.BT_real() + b * per_B, K, N, stream);
                transpose_fp8(buffers_.B_imag() + b * per_B,
                              buffers_.BT_imag() + b * per_B, K, N, stream);
            }
            Br_ptr = buffers_.BT_real();
            Bi_ptr = buffers_.BT_imag();
        }

        return run_subgemms_fp8(
            buffers_.A_real(), buffers_.A_imag(),
            Br_ptr, Bi_ptr,
            C_real, C_imag,
            M, N, K, alpha, beta, mode, stream, config, batch_count);
    }


public:
    // ====================================================================
    // BLAS3-style API — interleaved complex FP16 in/out
    // ====================================================================

    /// GEMM — C = alpha * A * B + beta * C  (Standard)
    ///         C = alpha * A * B^H + beta * C  (Hermitian)
    cutlass::Status GEMM(
        const __half* A,
        const __half* B,
        __half* C,
        int M, int N, int K,
        float alpha = 1.0f,
        float beta  = 0.0f,
        ComplexMode mode = ComplexMode::Standard,
        CutlassParams params = {})
    {
        bool is_herm = (mode == ComplexMode::Hermitian);
        if (params.precision == ComputePrecision::FP8_E4M3) {
            return run(const_cast<__half*>(A), const_cast<__half*>(B),
                       C, M, N, K, is_herm, alpha, beta, params.stream, params.config);
        }
        return run(const_cast<__half*>(A), const_cast<__half*>(B),
                   C, M, N, K, is_herm, alpha, beta,
                   params.precision, params.stream, params.config);
    }

    /// Batched GEMM — implemented as a loop (no native batched support on SM100).
    cutlass::Status GEMM_batched(
        const __half* A,
        const __half* B,
        __half* C,
        int M, int N, int K,
        int batch_count,
        float alpha = 1.0f,
        float beta  = 0.0f,
        ComplexMode mode = ComplexMode::Standard,
        CutlassParams params = {})
    {
        int64_t stride_A = static_cast<int64_t>(M) * K * 2;  // interleaved
        int64_t stride_B = static_cast<int64_t>(K) * N * 2;
        int64_t stride_C = static_cast<int64_t>(M) * N * 2;
        for (int b = 0; b < batch_count; ++b) {
            auto status = GEMM(
                A + b * stride_A,
                B + b * stride_B,
                C + b * stride_C,
                M, N, K, alpha, beta, mode, params);
            if (status != cutlass::Status::kSuccess) return status;
        }
        return cutlass::Status::kSuccess;
    }

    /// GEMM_int4 — Complex GEMM with INT4 sign-magnitude input.
    ///
    /// Input:  A, B are packed INT4 interleaved complex (1 byte per complex element).
    /// Output: C is interleaved complex FP16 (same as standard GEMM output).
    ///
    /// Dispatches to one of four compute strategies selected by params.strategy:
    ///   FP8_E4M3  — exact, FP32 accumulation
    ///   INT8      — exact, INT32 accumulation
    ///   FP6_E3M2  — exact, FP32 accumulation, 25% less bandwidth
    ///   FP4_E2M1  — lossy (5→4, 7→8), 2x tensor core throughput
    struct CutlassParamsInt4 {
        cudaStream_t stream;
        ComputeStrategy strategy;
        GemmConfig config;
        CutlassParamsInt4()
            : stream(nullptr), strategy(ComputeStrategy::FP8_E4M3), config(GemmConfig::Default) {}
    };

    cutlass::Status GEMM_int4(
        const uint8_t* A,
        const uint8_t* B,
        __half* C,
        int M, int N, int K,
        float alpha = 1.0f,
        float beta  = 0.0f,
        CutlassParamsInt4 params = CutlassParamsInt4{})
    {
        return run_int4(A, B, C, M, N, K, alpha, beta,
                        params.strategy, params.stream, params.config);
    }

    /// Batched GEMM_int4 — loop over batches.
    cutlass::Status GEMM_int4_batched(
        const uint8_t* A,
        const uint8_t* B,
        __half* C,
        int M, int N, int K,
        int batch_count,
        float alpha = 1.0f,
        float beta  = 0.0f,
        CutlassParamsInt4 params = CutlassParamsInt4{})
    {
        int64_t stride_A = static_cast<int64_t>(M) * K;   // 1 byte per complex element
        int64_t stride_B = static_cast<int64_t>(K) * N;
        int64_t stride_C = static_cast<int64_t>(M) * N * 2;  // interleaved FP16
        for (int b = 0; b < batch_count; ++b) {
            auto status = GEMM_int4(
                A + b * stride_A,
                B + b * stride_B,
                C + b * stride_C,
                M, N, K, alpha, beta, params);
            if (status != cutlass::Status::kSuccess) return status;
        }
        return cutlass::Status::kSuccess;
    }

    /// HERK — Hermitian Rank-K Update with packed triangular output.
    ///
    /// A is interleaved complex FP16.
    /// C is interleaved packed triangular: N*(N+1)/2 interleaved complex
    /// elements, i.e. N*(N+1) __half values total.
    cutlass::Status HERK(
        const __half* A,
        __half* C,
        int N, int K,
        float alpha = 1.0f,
        float beta  = 0.0f,
        HerkOp op = HerkOp::NoTrans,
        FillMode fill = FillMode::Lower,
        CutlassParams params = {})
    {
        cudaStream_t stream = params.stream;

        // Direct HERK: single-launch path (same as batched, with batch_count=1)
        // Uses pre-cast + cp.async kernel: FP16→FP8 done once, then cp.async pipeline
        if (params.precision == ComputePrecision::FP8_E4M3
            && op == HerkOp::NoTrans)
        {
            bool should_use_direct = false;
            if (herk_mode_ == HerkMode::ForceDirect) {
                should_use_direct = true;
            } else if (herk_mode_ == HerkMode::Auto) {
                // Lower bound: below K_CHUNK (64), output writes dominate and
                // scattered packed-triangle stores cause ~6x write amplification.
                // Baseline writes coalesced to N×N scratch, avoiding this.
                should_use_direct = (K >= HERK_K_CHUNK && K <= N / 4);
            }
            if (should_use_direct) {
                ensure_hw_info();

                int64_t total_fp8 = static_cast<int64_t>(N) * 2 * K;
                ensure_herk_precast(total_fp8, stream);
                cast_fp16_to_fp8_e4m3(A, herk_precast_buf_, total_fp8, stream);

                // Persistent kernel dispatch: reduces block scheduling overhead
                int blocks_per_dim = (N + HERK_BLOCK_N - 1) / HERK_BLOCK_N;
                int tri_blocks = blocks_per_dim * (blocks_per_dim + 1) / 2;
                bool use_persistent = false;
                if (persistent_mode_ == PersistentMode::ForceOn) {
                    use_persistent = true;
                } else if (persistent_mode_ == PersistentMode::Auto) {
                    use_persistent = (tri_blocks > hw_sm_count_ * 16)
                                  && (K <= direct_herk_k_chunk(direct_herk_config_));
                }

                // L2-tiled scratch output: write coalesced N×N to scratch (L2-resident),
                // then pack to triangle. Eliminates per-block SMEM staging overhead.
                int64_t scratch_elems = static_cast<int64_t>(N) * N * 2;
                ensure_herk_scratch(scratch_elems, stream);

                cutlass::Status status;
                if (herk_pipeline_mode_ == HerkPipelineMode::WarpSpecialized) {
                    status = launch_herk_ws_dispatch<__half, true>(
                        direct_herk_config_, herk_tile_size_,
                        herk_precast_buf_, herk_scratch_buf_, N, K, 1,
                        alpha, 0.0f, fill, stream);
                } else if (use_persistent) {
                    status = launch_herk_persistent_dispatch<__half, true>(
                        direct_herk_config_,
                        herk_precast_buf_, herk_scratch_buf_, N, K, 1,
                        alpha, 0.0f, fill, hw_sm_count_, stream, herk_tile_size_);
                } else {
                    status = launch_herk_direct_dispatch<__half, true>(
                        direct_herk_config_,
                        herk_precast_buf_, herk_scratch_buf_, N, K, 1,
                        alpha, 0.0f, fill, stream, herk_tile_size_);
                }
                if (status != cutlass::Status::kSuccess) return status;

                pack_scratch_to_triangle(
                    herk_scratch_buf_, C,
                    (beta != 0.0f) ? C : nullptr,
                    N, 1, beta, stream);
                return cutlass::Status::kSuccess;
            }
        }

        int A_rows = (op == HerkOp::NoTrans) ? N : K;
        int A_cols = (op == HerkOp::NoTrans) ? K : N;
        int64_t size_A = static_cast<int64_t>(A_rows) * A_cols;
        int64_t packed_elems = static_cast<int64_t>(N) * (N + 1) / 2;

        __half *Ar, *Ai, *Cr_packed, *Ci_packed;
        CUDA_CHECK(cudaMallocAsync(&Ar, size_A * sizeof(__half), stream));
        CUDA_CHECK(cudaMallocAsync(&Ai, size_A * sizeof(__half), stream));
        CUDA_CHECK(cudaMallocAsync(&Cr_packed, packed_elems * sizeof(__half), stream));
        CUDA_CHECK(cudaMallocAsync(&Ci_packed, packed_elems * sizeof(__half), stream));

        deinterleave_complex(A, Ar, Ai, size_A, stream);

        if (beta != 0.0f) {
            deinterleave_complex(C, Cr_packed, Ci_packed, packed_elems, stream);
        }

        cutlass::Status status;
        if (params.precision != ComputePrecision::FP8_E4M3) {
            status = herk_planar_packed(
                Ar, Ai, Cr_packed, Ci_packed,
                N, K, alpha, beta,
                op, fill, params.herk_strategy,
                params.precision, stream, params.config, params.triangle_config);
        } else {
            status = herk_planar_packed(
                Ar, Ai, Cr_packed, Ci_packed,
                N, K, alpha, beta,
                op, fill, params.herk_strategy, stream,
                params.config, params.triangle_config);
        }

        if (status == cutlass::Status::kSuccess) {
            interleave_complex(Cr_packed, Ci_packed, C, packed_elems, stream);
        }

        cudaFreeAsync(Ar, stream);
        cudaFreeAsync(Ai, stream);
        cudaFreeAsync(Cr_packed, stream);
        cudaFreeAsync(Ci_packed, stream);

        return status;
    }

    /// Batched HERK — batch_count independent HERKs on contiguous arrays.
    ///
    /// A[b] at A + b*2*rows*cols (rows/cols depend on op).
    /// C[b] at C + b*2*packed_elems where packed_elems = N*(N+1)/2.
    /// Supports all precisions: FP8 (batched kernel), FP6/FP4 (batched block-scaled).
    cutlass::Status HERK_batched(
        const __half* A,
        __half* C,
        int N, int K,
        int batch_count,
        float alpha = 1.0f,
        float beta  = 0.0f,
        HerkOp op = HerkOp::NoTrans,
        FillMode fill = FillMode::Lower,
        CutlassParams params = {},
        bool pre_deinterleaved = false,
        const __nv_fp8_e4m3* A_fp8 = nullptr)
    {
        cudaStream_t stream = params.stream;
        int A_rows = (op == HerkOp::NoTrans) ? N : K;
        int A_cols = (op == HerkOp::NoTrans) ? K : N;
        int64_t per_A = static_cast<int64_t>(A_rows) * A_cols;
        int64_t total_A = per_A * batch_count;
        int64_t packed_elems = static_cast<int64_t>(N) * (N + 1) / 2;
        int64_t total_packed = packed_elems * batch_count;

        // ---- Direct HERK kernel: single-launch path for FP8 NoTrans ----
        // Uses pre-cast + cp.async kernel: FP16→FP8 done once, then cp.async pipeline
        // eliminates ~50% of K-loop instruction overhead (no register-based casting).
        // Auto threshold scales with batch tiling: when batch_tile < batch_count,
        // baseline loops with per-tile overhead while direct uses one gridDim.y launch.
        if (params.precision == ComputePrecision::FP8_E4M3
            && op == HerkOp::NoTrans)
        {
            bool should_use_direct = false;
            if (herk_mode_ == HerkMode::ForceDirect) {
                should_use_direct = true;
            } else if (herk_mode_ == HerkMode::Auto) {
                // Lower bound: below K_CHUNK (64), output writes dominate and
                // scattered packed-triangle stores cause ~6x write amplification.
                // Baseline writes coalesced to N×N scratch, avoiding this.
                int threshold_k = N / 4;
                if (batch_count > 1 && l2_cache_bytes_ > 0) {
                    int64_t scratch_per_batch = 4LL * N * N;
                    int batch_tile = std::max(1, static_cast<int>(l2_cache_bytes_ / scratch_per_batch));
                    if (batch_count > batch_tile)
                        threshold_k = N / 2;
                }
                should_use_direct = (K >= HERK_K_CHUNK && K <= threshold_k);
            }
            if (should_use_direct) {
                ensure_hw_info();

                // Prepare FP8 interleaved precast buffer for the direct HERK kernel.
                int64_t total_fp8 = static_cast<int64_t>(N) * 2 * K * batch_count;
                const __nv_fp8_e4m3* precast;
                if (A_fp8) {
                    // INT4 path: caller already produced FP8 interleaved — use directly
                    precast = A_fp8;
                } else if (pre_deinterleaved) {
                    // Planar FP16 [Ar|Ai] — re-interleave then cast
                    ensure_herk_precast(total_fp8, stream);
                    int64_t total_A_loc = static_cast<int64_t>(N) * K * batch_count;
                    __half* A_interleaved;
                    CUDA_CHECK(cudaMallocAsync(&A_interleaved, total_fp8 * sizeof(__half), stream));
                    interleave_complex(A, A + total_A_loc, A_interleaved, total_A_loc, stream);
                    cast_fp16_to_fp8_e4m3(A_interleaved, herk_precast_buf_, total_fp8, stream);
                    CUDA_CHECK(cudaFreeAsync(A_interleaved, stream));
                    precast = herk_precast_buf_;
                } else {
                    // Interleaved FP16 — cast to FP8
                    ensure_herk_precast(total_fp8, stream);
                    cast_fp16_to_fp8_e4m3(A, herk_precast_buf_, total_fp8, stream);
                    precast = herk_precast_buf_;
                }

                // Scratch output: use L2-resident scratch when entire batch fits,
                // avoiding per-block SMEM staging overhead. Otherwise direct
                // triangle output (single launch via gridDim.y parallelism).
                int blocks_per_dim = (N + HERK_BLOCK_N - 1) / HERK_BLOCK_N;
                int tri_blocks = blocks_per_dim * (blocks_per_dim + 1) / 2;
                int64_t total_work = static_cast<int64_t>(tri_blocks) * batch_count;

                int64_t scratch_per_batch = static_cast<int64_t>(N) * N * 2;  // elements
                int64_t scratch_bytes_per_batch = scratch_per_batch * sizeof(__half);  // bytes
                bool use_scratch = (l2_cache_bytes_ > 0)
                                && (scratch_bytes_per_batch <= l2_cache_bytes_);

                // Batch-tiled scratch: process scratch_batch_tile batches at a time
                // so that scratch fits in L2 even when total scratch exceeds L2.
                int scratch_batch_tile = batch_count;
                if (use_scratch && scratch_bytes_per_batch * batch_count > l2_cache_bytes_) {
                    scratch_batch_tile = std::max(1, static_cast<int>(l2_cache_bytes_ / scratch_bytes_per_batch));
                }

                bool use_persistent = false;
                if (persistent_mode_ == PersistentMode::ForceOn) {
                    use_persistent = true;
                } else if (persistent_mode_ == PersistentMode::Auto) {
                    use_persistent = (total_work > hw_sm_count_ * 16)
                                  && (K <= direct_herk_k_chunk(direct_herk_config_));
                }

                cutlass::Status status;
                if (use_scratch) {
                    ensure_herk_scratch(scratch_per_batch * scratch_batch_tile, stream);

                    // L2 persistence on FP8 precast buffer: keep input resident
                    // across tiles so each tile's compute hits L2 for repeated K reads.
                    bool use_l2_precast = (persisting_l2_max_ > 0);
                    if (use_l2_precast) {
                        int64_t fp8_bytes = static_cast<int64_t>(N) * 2 * K * batch_count;
                        size_t persist_max = std::min(
                            static_cast<size_t>(fp8_bytes),
                            static_cast<size_t>(persisting_l2_max_) / 2);
                        cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, persist_max);
                        cudaStreamAttrValue attr = {};
                        attr.accessPolicyWindow.base_ptr = const_cast<__nv_fp8_e4m3*>(precast);
                        attr.accessPolicyWindow.num_bytes = std::min(
                            static_cast<size_t>(fp8_bytes), persist_max);
                        attr.accessPolicyWindow.hitRatio = 1.0f;
                        attr.accessPolicyWindow.hitProp = cudaAccessPropertyPersisting;
                        attr.accessPolicyWindow.missProp = cudaAccessPropertyStreaming;
                        cudaStreamSetAttribute(stream, cudaStreamAttributeAccessPolicyWindow, &attr);
                    }

                    status = cutlass::Status::kSuccess;
                    for (int tile_start = 0; tile_start < batch_count && status == cutlass::Status::kSuccess;
                         tile_start += scratch_batch_tile) {
                        int actual_tile = std::min(scratch_batch_tile, batch_count - tile_start);

                        // Offset FP8 input: each batch has N * 2K bytes (FP8 interleaved)
                        const __nv_fp8_e4m3* precast_tile = precast
                            + static_cast<int64_t>(tile_start) * N * 2 * K;

                        if (herk_pipeline_mode_ == HerkPipelineMode::WarpSpecialized) {
                            status = launch_herk_ws_dispatch<__half, true>(
                                direct_herk_config_, herk_tile_size_,
                                precast_tile, herk_scratch_buf_, N, K, actual_tile,
                                alpha, 0.0f, fill, stream);
                        } else if (use_persistent) {
                            status = launch_herk_persistent_dispatch<__half, true>(
                                direct_herk_config_,
                                precast_tile, herk_scratch_buf_, N, K, actual_tile,
                                alpha, 0.0f, fill, hw_sm_count_, stream, herk_tile_size_);
                        } else {
                            status = launch_herk_direct_dispatch<__half, true>(
                                direct_herk_config_,
                                precast_tile, herk_scratch_buf_, N, K, actual_tile,
                                alpha, 0.0f, fill, stream, herk_tile_size_);
                        }
                        if (status != cutlass::Status::kSuccess) break;

                        // Pack this tile's scratch → packed triangle output
                        __half* C_tile = C + static_cast<int64_t>(tile_start) * N * (N + 1);
                        pack_scratch_to_triangle(
                            herk_scratch_buf_, C_tile,
                            (beta != 0.0f) ? C_tile : nullptr,
                            N, actual_tile, beta, stream);
                    }

                    // Reset L2 persistence hints
                    if (use_l2_precast) {
                        cudaStreamAttrValue attr = {};
                        attr.accessPolicyWindow.num_bytes = 0;
                        cudaStreamSetAttribute(stream, cudaStreamAttributeAccessPolicyWindow, &attr);
                    }

                    if (status != cutlass::Status::kSuccess) return status;
                } else {
                    if (herk_pipeline_mode_ == HerkPipelineMode::WarpSpecialized) {
                        status = launch_herk_ws_dispatch(
                            direct_herk_config_, herk_tile_size_,
                            precast, C, N, K, batch_count,
                            alpha, beta, fill, stream);
                    } else if (use_persistent) {
                        status = launch_herk_persistent_dispatch(
                            direct_herk_config_,
                            precast, C, N, K, batch_count,
                            alpha, beta, fill, hw_sm_count_, stream, herk_tile_size_);
                    } else {
                        status = launch_herk_direct_dispatch(
                            direct_herk_config_,
                            precast, C, N, K, batch_count,
                            alpha, beta, fill, stream, herk_tile_size_);
                    }
                }
                return status;
            }
        }

        // Check if we can use the fused pack+interleave path:
        // block-scaled NoTrans writes directly to interleaved C, eliminating
        // Cr_packed/Ci_packed intermediates and the deinterleave/interleave of C.
        bool use_fused_pack_interleave = (params.precision != ComputePrecision::FP8_E4M3
                                          && op == HerkOp::NoTrans);

        // Temporary planar buffers for A — only needed for non-fused paths.
        // The fused path reads interleaved A directly in deinterleave_preprocess_mxfp.
        __half *Ar = nullptr, *Ai = nullptr;
        bool owns_Ar_Ai = false;
        if (!use_fused_pack_interleave) {
            if (pre_deinterleaved) {
                Ar = const_cast<__half*>(A);
                Ai = const_cast<__half*>(A) + total_A;
            } else {
                CUDA_CHECK(cudaMallocAsync(&Ar, total_A * sizeof(__half), stream));
                CUDA_CHECK(cudaMallocAsync(&Ai, total_A * sizeof(__half), stream));
                owns_Ar_Ai = true;
            }
        }

        // Planar packed C buffers — only needed for non-fused paths
        __half *Cr_packed = nullptr, *Ci_packed = nullptr;
        if (!use_fused_pack_interleave) {
            CUDA_CHECK(cudaMallocAsync(&Cr_packed, total_packed * sizeof(__half), stream));
            CUDA_CHECK(cudaMallocAsync(&Ci_packed, total_packed * sizeof(__half), stream));
        }

        // Deinterleave all A batches (only for non-fused paths;
        // fused path uses deinterleave_preprocess_mxfp to read interleaved A directly)
        if (!use_fused_pack_interleave && !pre_deinterleaved) {
            deinterleave_complex(A, Ar, Ai, total_A, stream);
        }

        // Deinterleave packed C if beta != 0 (only for non-fused paths;
        // fused kernel reads old C directly from interleaved format)
        if (beta != 0.0f && !use_fused_pack_interleave) {
            deinterleave_complex(C, Cr_packed, Ci_packed, total_packed, stream);
        }

        // ---- Batch tiling: process batch_tile elements at a time ----
        // Reduces scratch from 2×N²×batch to 2×N²×batch_tile, improving
        // L2 cache reuse when N² × batch exceeds L2.
        // Skip tiling when beta == 0: all sub-GEMMs use beta=0 internally,
        // so scratch is write-only and L2 caching provides no benefit.
        // Skip tiling when compute-bound (K > N): at large K, GEMM occupancy
        // from full batch_count matters more than L2 caching of N×N scratch.
        ensure_hw_info();
        int batch_tile = batch_count;
        if (use_batch_tiling_ && beta != 0.0f && K <= N) {
            int64_t scratch_per_batch = 4LL * N * N;  // 2 buffers × N² × sizeof(__half)
            if (l2_cache_bytes_ > 0 && scratch_per_batch * batch_count > l2_cache_bytes_) {
                batch_tile = std::max(1, static_cast<int>(l2_cache_bytes_ / scratch_per_batch));
            }
        }

        // C is interleaved packed: each batch element has packed_elems complex values
        // = packed_elems * 2 __half values
        int64_t interleaved_stride = packed_elems * 2;

        // Determine number of batch tiles
        int num_tiles = (batch_count + batch_tile - 1) / batch_tile;

        // Pipeline: overlap MXFP preprocessing of tile N+1 with GEMMs of tile N.
        // Requires fused deinterleave+MXFP (Phase 3), K >= threshold, and ≥ 2 tiles.
        bool use_pipeline = use_fused_pack_interleave
                            && K >= kPipelineKThreshold
                            && num_tiles >= 2;

        cutlass::Status status = cutlass::Status::kSuccess;

        if (use_pipeline) {
            // ---- PIPELINED BATCH TILES ----
            // Double-buffer MXFP data + SF: while GEMMs run on tile N's data,
            // preprocess tile N+1's data on the main stream.
            ensure_streams();
            ensure_hw_info();
            ensure_pipeline_events();

            // Allocate double-buffered MXFP data + SF buffers
            int max_tile = batch_tile;  // largest tile (first tiles are full-sized)
            int64_t per_A_tile = static_cast<int64_t>(N) * K;
            int64_t bytes_A_tile = bytes_for_elements(per_A_tile * max_tile, params.precision);
            int64_t sf_tile_bytes = sf_buffer_bytes(N * max_tile, K);

            LowPrecisionBufferManager* lp_bufs[2] = { &lp_buffers_, &lp_buffers_b_ };
            for (int i = 0; i < 2; ++i) {
                lp_bufs[i]->ensure_capacity(bytes_A_tile, 0, 0, stream);
                lp_bufs[i]->ensure_sf_capacity(sf_tile_bytes, 0, 0, stream);
            }

            // Allocate N×N scratch buffers for intermediate results
            int64_t output_per_batch = static_cast<int64_t>(N) * N;
            int64_t max_output = output_per_batch * max_tile;
            ensure_herk_real_temp(max_output, stream);
            ensure_herk_temp(max_output, stream);

            // Pre-allocate GEMM workspaces
            size_t gemm_ws_size = compute_workspace_size_fp16(N, N, K, max_tile, params.precision);
            void* gemm_ws_a = nullptr;
            void* gemm_ws_b = nullptr;
            if (gemm_ws_size > 0) {
                CUDA_CHECK(cudaMallocAsync(&gemm_ws_a, gemm_ws_size, stream));
                CUDA_CHECK(cudaMallocAsync(&gemm_ws_b, gemm_ws_size, stream));
            }

            // Preprocess first tile into buf[0]
            int first_tile_size = std::min(batch_tile, batch_count);
            const __half* A_tile_0 = A;  // interleaved
            deinterleave_preprocess_mxfp_paired_sm100(
                A_tile_0,
                lp_bufs[0]->A_real(), lp_bufs[0]->A_imag(),
                lp_bufs[0]->sf_A_real(), lp_bufs[0]->sf_A_imag(),
                N * first_tile_size, K, params.precision, stream);
            CUDA_CHECK(cudaEventRecord(pipeline_preprocess_done_[0], stream));

            for (int tile_idx = 0; tile_idx < num_tiles && status == cutlass::Status::kSuccess; ++tile_idx) {
                int tile_start = tile_idx * batch_tile;
                int actual_tile = std::min(batch_tile, batch_count - tile_start);
                int buf_idx = tile_idx % 2;
                int next_buf = 1 - buf_idx;

                // Wait for this tile's preprocessing to complete on GEMM streams
                CUDA_CHECK(cudaStreamWaitEvent(stream_a_, pipeline_preprocess_done_[buf_idx]));
                CUDA_CHECK(cudaStreamWaitEvent(stream_b_, pipeline_preprocess_done_[buf_idx]));

                // Launch GEMMs for this tile
                void* Ar_data = lp_bufs[buf_idx]->A_real();
                void* Ai_data = lp_bufs[buf_idx]->A_imag();
                void* sf_Ar   = lp_bufs[buf_idx]->sf_A_real();
                void* sf_Ai   = lp_bufs[buf_idx]->sf_A_imag();
                auto* scratch_Re = reinterpret_cast<cutlass::half_t*>(herk_real_temp_);
                auto* temp_Im    = reinterpret_cast<cutlass::half_t*>(herk_imag_temp_);

                // Re(C) = Ar × Ar^T + Ai × Ai^T on stream_a_
                status = run_real_gemm_dispatch(
                    Ar_data, Ar_data, sf_Ar, sf_Ar,
                    scratch_Re,
                    N, N, K, alpha, 0.0f, stream_a_, params.precision, params.config, actual_tile,
                    /*ld_C=*/0, gemm_ws_a, gemm_ws_size);
                if (status != cutlass::Status::kSuccess) break;

                status = run_real_gemm_dispatch(
                    Ai_data, Ai_data, sf_Ai, sf_Ai,
                    scratch_Re,
                    N, N, K, alpha, 1.0f, stream_a_, params.precision, params.config, actual_tile,
                    /*ld_C=*/0, gemm_ws_a, gemm_ws_size);
                if (status != cutlass::Status::kSuccess) break;

                // Im(C) = Ai × Ar^T on stream_b_
                status = run_real_gemm_dispatch(
                    Ai_data, Ar_data, sf_Ai, sf_Ar,
                    temp_Im,
                    N, N, K, 1.0f, 0.0f, stream_b_, params.precision, params.config, actual_tile,
                    /*ld_C=*/0, gemm_ws_b, gemm_ws_size);
                if (status != cutlass::Status::kSuccess) break;

                // Record GEMM completion for this buffer (allows next preprocess to reuse it)
                CUDA_CHECK(cudaEventRecord(pipeline_gemm_done_[buf_idx], stream_a_));

                // If there's a next tile, preprocess it into the other buffer
                int next_tile_idx = tile_idx + 1;
                if (next_tile_idx < num_tiles) {
                    int next_start = next_tile_idx * batch_tile;
                    int next_tile_size = std::min(batch_tile, batch_count - next_start);
                    const __half* A_next = A + next_start * per_A * 2;

                    // Wait for previous GEMMs using next_buf to finish before overwriting
                    if (tile_idx >= 1) {
                        CUDA_CHECK(cudaStreamWaitEvent(stream, pipeline_gemm_done_[next_buf]));
                    }

                    deinterleave_preprocess_mxfp_paired_sm100(
                        A_next,
                        lp_bufs[next_buf]->A_real(), lp_bufs[next_buf]->A_imag(),
                        lp_bufs[next_buf]->sf_A_real(), lp_bufs[next_buf]->sf_A_imag(),
                        N * next_tile_size, K, params.precision, stream);
                    CUDA_CHECK(cudaEventRecord(pipeline_preprocess_done_[next_buf], stream));
                }

                // Sync GEMM streams, then pack+interleave on main stream
                CUDA_CHECK(cudaEventRecord(stream_a_done_, stream_a_));
                CUDA_CHECK(cudaEventRecord(stream_b_done_, stream_b_));
                CUDA_CHECK(cudaStreamWaitEvent(stream, stream_a_done_));
                CUDA_CHECK(cudaStreamWaitEvent(stream, stream_b_done_));

                __half* C_tile = C + tile_start * interleaved_stride;
                pack_antisymmetrize_interleave(
                    herk_real_temp_, herk_imag_temp_,
                    C_tile,
                    N, alpha, beta, fill, stream, actual_tile,
                    (beta != 0.0f) ? C_tile : nullptr);
            }

            if (gemm_ws_a) cudaFreeAsync(gemm_ws_a, stream);
            if (gemm_ws_b) cudaFreeAsync(gemm_ws_b, stream);
        } else {
            // ---- SEQUENTIAL BATCH TILES (non-pipelined) ----
            for (int tile_start = 0; tile_start < batch_count && status == cutlass::Status::kSuccess;
                 tile_start += batch_tile) {
                int actual_tile = std::min(batch_tile, batch_count - tile_start);

                const __half* Ar_tile = Ar ? Ar + tile_start * per_A : nullptr;
                const __half* Ai_tile = Ai ? Ai + tile_start * per_A : nullptr;

                if (use_fused_pack_interleave) {
                    // Block-scaled batched HERK with fused deinterleave+MXFP and pack+interleave:
                    // reads interleaved A directly, writes directly to interleaved C.
                    // Eliminates Ar/Ai planar buffers and Cr_packed/Ci_packed intermediates.
                    __half* C_tile = C + tile_start * interleaved_stride;
                    const __half* A_tile = A + tile_start * per_A * 2;  // interleaved: 2 halfs per complex
                    status = herk_planar_packed_batched_blockscaled(
                        nullptr, nullptr, nullptr, nullptr,
                        N, K, actual_tile, alpha, beta,
                        fill, params.precision, stream, params.config,
                        params.herk_strategy, params.triangle_config,
                        C_tile, A_tile);
                } else if (params.precision != ComputePrecision::FP8_E4M3) {
                // ConjTrans block-scaled: per-batch loop through non-batched path
                __half* Cr_tile = Cr_packed + tile_start * packed_elems;
                __half* Ci_tile = Ci_packed + tile_start * packed_elems;
                for (int b = 0; b < actual_tile && status == cutlass::Status::kSuccess; ++b) {
                    status = herk_planar_packed(
                        Ar_tile + b * per_A, Ai_tile + b * per_A,
                        Cr_tile + b * packed_elems, Ci_tile + b * packed_elems,
                        N, K, alpha, beta,
                        op, fill, params.herk_strategy,
                        params.precision, stream, params.config, params.triangle_config);
                }
            } else {
                // FP8: batched path with tile-sized batch
                __half* Cr_tile = Cr_packed + tile_start * packed_elems;
                __half* Ci_tile = Ci_packed + tile_start * packed_elems;
                status = herk_planar_packed_batched(
                    Ar_tile, Ai_tile, Cr_tile, Ci_tile,
                    N, K, actual_tile, alpha, beta,
                    op, fill, params.herk_strategy, stream, params.triangle_config,
                    params.config);
            }
            }
        }

        // Interleave all packed output batches back to C (only for non-fused paths)
        if (status == cutlass::Status::kSuccess && !use_fused_pack_interleave) {
            interleave_complex(Cr_packed, Ci_packed, C, total_packed, stream);
        }

        if (owns_Ar_Ai) { cudaFreeAsync(Ar, stream); cudaFreeAsync(Ai, stream); }
        if (Cr_packed) cudaFreeAsync(Cr_packed, stream);
        if (Ci_packed) cudaFreeAsync(Ci_packed, stream);

        return status;
    }

    /// HERK_batched_int4 — Batched HERK from native INT4 sign-magnitude complex input.
    ///
    /// Input:  A is packed INT4 interleaved complex (1 byte per complex element).
    ///         High nibble = Re (sign-magnitude), Low nibble = Im (sign-magnitude).
    ///         Layout: [batch_count × N × K] bytes, row-major.
    ///
    /// Output: C is interleaved packed triangular FP16: N*(N+1)/2 complex elements
    ///         per batch = N*(N+1) __half values per batch.
    ///
    /// Pipeline: INT4 → FP16 interleaved → HERK_batched (direct or baseline auto-dispatch).
    cutlass::Status HERK_batched_int4(
        const uint8_t* A,
        __half* C,
        int N, int K,
        int batch_count,
        float alpha = 1.0f,
        float beta  = 0.0f,
        HerkOp op = HerkOp::NoTrans,
        FillMode fill = FillMode::Lower,
        CutlassParams params = {})
    {
        cudaStream_t stream = params.stream;
        int64_t total_complex = static_cast<int64_t>(N) * K * batch_count;

        // Allocate FP16 interleaved buffer: 2 __half per complex element
        __half* A_fp16;
        CUDA_CHECK(cudaMallocAsync(&A_fp16, total_complex * 2 * sizeof(__half), stream));

        // INT4 sign-magnitude → FP16 interleaved
        cast_int4_to_fp16_interleaved(A, A_fp16, total_complex, stream);

        // Dispatch to HERK_batched (auto-selects direct vs baseline)
        auto status = HERK_batched(A_fp16, C, N, K, batch_count,
                                   alpha, beta, op, fill, params);

        CUDA_CHECK(cudaFreeAsync(A_fp16, stream));
        return status;
    }

    /// HERK_batched_fp32out — Batched HERK with native FP32 output.
    ///
    /// Input:  A is interleaved FP16 complex: [batch_count × N × K × 2] __half.
    /// Output: C is interleaved packed triangular FP32: N*(N+1)/2 complex elements
    ///         per batch = N*(N+1) float values per batch.
    ///
    /// Eliminates the FP16 intermediate and widen_fp16_to_fp32 conversion.
    /// Direct kernel path: single-launch PTX kernel writes FP32 directly.
    /// Baseline path: 3 real FP8 sub-GEMMs with FP32 accumulation + FP32 pack kernel.
    cutlass::Status HERK_batched_fp32out(
        const __half* A,
        float* C,
        int N, int K,
        int batch_count,
        float alpha = 1.0f,
        float beta  = 0.0f,
        HerkOp op = HerkOp::NoTrans,
        FillMode fill = FillMode::Lower,
        CutlassParams params = {},
        bool pre_deinterleaved = false,
        const __nv_fp8_e4m3* A_fp8 = nullptr)
    {
        cudaStream_t stream = params.stream;
        int64_t per_A = static_cast<int64_t>(N) * K;
        int64_t total_A = per_A * batch_count;
        int64_t packed_elems = static_cast<int64_t>(N) * (N + 1) / 2;
        // Interleaved packed stride: packed_elems * 2 floats per batch element
        int64_t interleaved_stride = packed_elems * 2;

        // ---- Direct HERK kernel: single-launch path for FP8 NoTrans ----
        if (params.precision == ComputePrecision::FP8_E4M3
            && op == HerkOp::NoTrans)
        {
            bool should_use_direct = false;
            if (herk_mode_ == HerkMode::ForceDirect) {
                should_use_direct = true;
            } else if (herk_mode_ == HerkMode::Auto) {
                // Lower bound: below K_CHUNK (64), output writes dominate and
                // scattered packed-triangle stores cause ~6x write amplification.
                // Baseline writes coalesced to N×N scratch, avoiding this.
                int threshold_k = N / 4;
                if (batch_count > 1 && l2_cache_bytes_ > 0) {
                    // FP32 scratch is 2× larger: 8N² bytes per batch
                    int64_t scratch_per_batch = 8LL * N * N;
                    int batch_tile = std::max(1, static_cast<int>(l2_cache_bytes_ / scratch_per_batch));
                    if (batch_count > batch_tile)
                        threshold_k = N / 2;
                }
                should_use_direct = (K >= HERK_K_CHUNK && K <= threshold_k);
            }
            if (should_use_direct) {
                ensure_hw_info();

                // Prepare FP8 interleaved precast buffer for the direct HERK kernel.
                int64_t total_fp8 = static_cast<int64_t>(N) * 2 * K * batch_count;
                const __nv_fp8_e4m3* precast;
                if (A_fp8) {
                    // INT4 path: caller already produced FP8 interleaved — use directly
                    precast = A_fp8;
                } else if (pre_deinterleaved) {
                    // Planar FP16 [Ar|Ai] — re-interleave then cast
                    ensure_herk_precast(total_fp8, stream);
                    int64_t total_A_loc = static_cast<int64_t>(N) * K * batch_count;
                    __half* A_interleaved;
                    CUDA_CHECK(cudaMallocAsync(&A_interleaved, total_fp8 * sizeof(__half), stream));
                    interleave_complex(A, A + total_A_loc, A_interleaved, total_A_loc, stream);
                    cast_fp16_to_fp8_e4m3(A_interleaved, herk_precast_buf_, total_fp8, stream);
                    CUDA_CHECK(cudaFreeAsync(A_interleaved, stream));
                    precast = herk_precast_buf_;
                } else {
                    // Interleaved FP16 — cast to FP8
                    ensure_herk_precast(total_fp8, stream);
                    cast_fp16_to_fp8_e4m3(A, herk_precast_buf_, total_fp8, stream);
                    precast = herk_precast_buf_;
                }

                // Scratch output: FP32 scratch uses 8N² bytes per batch.
                int blocks_per_dim = (N + HERK_BLOCK_N - 1) / HERK_BLOCK_N;
                int tri_blocks = blocks_per_dim * (blocks_per_dim + 1) / 2;
                int64_t total_work = static_cast<int64_t>(tri_blocks) * batch_count;

                int64_t scratch_per_batch_fp32 = static_cast<int64_t>(N) * N * 2;  // elements
                int64_t scratch_bytes_per_batch = scratch_per_batch_fp32 * sizeof(float);  // bytes
                bool use_scratch = (l2_cache_bytes_ > 0)
                                && (scratch_bytes_per_batch <= l2_cache_bytes_);

                int scratch_batch_tile = batch_count;
                if (use_scratch && scratch_bytes_per_batch * batch_count > l2_cache_bytes_) {
                    scratch_batch_tile = std::max(1, static_cast<int>(l2_cache_bytes_ / scratch_bytes_per_batch));
                }

                bool use_persistent = false;
                if (persistent_mode_ == PersistentMode::ForceOn) {
                    use_persistent = true;
                } else if (persistent_mode_ == PersistentMode::Auto) {
                    use_persistent = (total_work > hw_sm_count_ * 16)
                                  && (K <= direct_herk_k_chunk(direct_herk_config_));
                }

                if (use_scratch) {
                    ensure_herk_scratch_fp32(scratch_per_batch_fp32 * scratch_batch_tile, stream);

                    // L2 persistence on FP8 precast buffer
                    bool use_l2_precast = (persisting_l2_max_ > 0);
                    if (use_l2_precast) {
                        size_t persist_max = std::min(
                            static_cast<size_t>(total_fp8),
                            static_cast<size_t>(persisting_l2_max_) / 2);
                        cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, persist_max);
                        cudaStreamAttrValue attr = {};
                        attr.accessPolicyWindow.base_ptr = const_cast<__nv_fp8_e4m3*>(precast);
                        attr.accessPolicyWindow.num_bytes = std::min(
                            static_cast<size_t>(total_fp8), persist_max);
                        attr.accessPolicyWindow.hitRatio = 1.0f;
                        attr.accessPolicyWindow.hitProp = cudaAccessPropertyPersisting;
                        attr.accessPolicyWindow.missProp = cudaAccessPropertyStreaming;
                        cudaStreamSetAttribute(stream, cudaStreamAttributeAccessPolicyWindow, &attr);
                    }

                    cutlass::Status status = cutlass::Status::kSuccess;
                    for (int tile_start = 0; tile_start < batch_count && status == cutlass::Status::kSuccess;
                         tile_start += scratch_batch_tile) {
                        int actual_tile = std::min(scratch_batch_tile, batch_count - tile_start);

                        const __nv_fp8_e4m3* precast_tile = precast
                            + static_cast<int64_t>(tile_start) * N * 2 * K;

                        if (herk_pipeline_mode_ == HerkPipelineMode::WarpSpecialized) {
                            status = launch_herk_ws_dispatch<float, true>(
                                direct_herk_config_, herk_tile_size_,
                                precast_tile, herk_scratch_fp32_buf_, N, K, actual_tile,
                                alpha, 0.0f, fill, stream);
                        } else if (use_persistent) {
                            status = launch_herk_persistent_dispatch<float, true>(
                                direct_herk_config_,
                                precast_tile, herk_scratch_fp32_buf_, N, K, actual_tile,
                                alpha, 0.0f, fill, hw_sm_count_, stream, herk_tile_size_);
                        } else {
                            status = launch_herk_direct_dispatch<float, true>(
                                direct_herk_config_,
                                precast_tile, herk_scratch_fp32_buf_, N, K, actual_tile,
                                alpha, 0.0f, fill, stream, herk_tile_size_);
                        }
                        if (status != cutlass::Status::kSuccess) break;

                        // Pack this tile's scratch → packed FP32 triangle output
                        float* C_tile = C + static_cast<int64_t>(tile_start) * N * (N + 1);
                        pack_scratch_to_triangle_fp32(
                            herk_scratch_fp32_buf_, C_tile,
                            (beta != 0.0f) ? C_tile : nullptr,
                            N, actual_tile, beta, stream);
                    }

                    // Reset L2 persistence hints
                    if (use_l2_precast) {
                        cudaStreamAttrValue attr = {};
                        attr.accessPolicyWindow.num_bytes = 0;
                        cudaStreamSetAttribute(stream, cudaStreamAttributeAccessPolicyWindow, &attr);
                    }
                    return status;
                }

                // Non-scratch fallback: direct triangle output
                cutlass::Status status;
                if (herk_pipeline_mode_ == HerkPipelineMode::WarpSpecialized) {
                    status = launch_herk_ws_dispatch<float>(
                        direct_herk_config_, herk_tile_size_,
                        precast, C, N, K, batch_count,
                        alpha, beta, fill, stream);
                } else if (use_persistent) {
                    status = launch_herk_persistent_dispatch<float>(
                        direct_herk_config_,
                        precast, C, N, K, batch_count,
                        alpha, beta, fill, hw_sm_count_, stream, herk_tile_size_);
                } else {
                    status = launch_herk_direct_dispatch<float>(
                        direct_herk_config_,
                        precast, C, N, K, batch_count,
                        alpha, beta, fill, stream, herk_tile_size_);
                }
                return status;
            }
        }

        // ---- Baseline FP8 path with FP32 output ----
        // Only FP8 baseline is supported for FP32 output (block-scaled TBD).
        if (params.precision != ComputePrecision::FP8_E4M3) {
            throw std::runtime_error(
                "HERK_batched_fp32out: only FP8 compute is supported for FP32 output. "
                "Block-scaled FP32 output is not yet implemented.");
        }
        // Deinterleave A → planar FP16, then FP8 cast, 3 sub-GEMMs with FP32 output,
        // then fused pack+antisymmetrize+interleave to interleaved FP32 packed.

        __half *Ar = nullptr, *Ai = nullptr;
        bool owns_Ar_Ai = false;
        if (pre_deinterleaved) {
            Ar = const_cast<__half*>(A);
            Ai = const_cast<__half*>(A) + total_A;
        } else {
            CUDA_CHECK(cudaMallocAsync(&Ar, total_A * sizeof(__half), stream));
            CUDA_CHECK(cudaMallocAsync(&Ai, total_A * sizeof(__half), stream));
            owns_Ar_Ai = true;
            deinterleave_complex(A, Ar, Ai, total_A, stream);
        }

        // Batch tiling: FP32 scratch is 2× larger than FP16
        // Skip tiling when beta == 0: sub-GEMMs are write-only, no L2 benefit.
        // Skip tiling when compute-bound (K > N): full batch occupancy matters more.
        ensure_hw_info();
        int batch_tile = batch_count;
        if (use_batch_tiling_ && beta != 0.0f && K <= N) {
            int64_t scratch_per_batch = 8LL * N * N;  // 2 × N² × sizeof(float) = 8N²
            if (l2_cache_bytes_ > 0 && scratch_per_batch * batch_count > l2_cache_bytes_) {
                batch_tile = std::max(1, static_cast<int>(l2_cache_bytes_ / scratch_per_batch));
            }
        }

        cutlass::Status status = cutlass::Status::kSuccess;

        for (int tile_start = 0; tile_start < batch_count && status == cutlass::Status::kSuccess;
             tile_start += batch_tile) {
            int actual_tile = std::min(batch_tile, batch_count - tile_start);
            const __half* Ar_tile = Ar + tile_start * per_A;
            const __half* Ai_tile = Ai + tile_start * per_A;

            int64_t output_per_batch = static_cast<int64_t>(N) * N;
            int64_t total_output = output_per_batch * actual_tile;

            // Allocate float-sized scratch: pass 2× half-element count to get sizeof(float) bytes
            ensure_herk_real_temp(total_output * 2, stream);
            ensure_herk_temp(total_output * 2, stream);

            ensure_streams();

            // FP16 → FP8 stacked-K cast (NoTrans path)
            int64_t per_A_tile = per_A * actual_tile;
            int64_t stacked_total = per_A_tile * 2;
            buffers_.ensure_stacked_capacity(stacked_total, per_A_tile, stream);
            cast_fp16_to_fp8_stacked_and_separate(Ar_tile, Ai_tile,
                reinterpret_cast<__nv_fp8_e4m3*>(buffers_.stacked()),
                reinterpret_cast<__nv_fp8_e4m3*>(buffers_.xi_separate()),
                reinterpret_cast<__nv_fp8_e4m3*>(buffers_.xr_separate()),
                N * actual_tile, K, stream);
            auto* Stacked = buffers_.stacked();
            auto* Xi_sep  = buffers_.xi_separate();
            auto* Xr_sep  = buffers_.xr_separate();

            // Duplicate stacked buffer for TMA L2 fix
            auto* Stacked_B = buffers_.stacked_b();
            CUDA_CHECK(cudaMemcpyAsync(Stacked_B, Stacked,
                stacked_total * sizeof(cutlass::float_e4m3_t),
                cudaMemcpyDeviceToDevice, stream));

            CUDA_CHECK(cudaEventRecord(preprocess_done_, stream));
            CUDA_CHECK(cudaStreamWaitEvent(stream_a_, preprocess_done_));
            CUDA_CHECK(cudaStreamWaitEvent(stream_b_, preprocess_done_));

            auto* scratch_Re = reinterpret_cast<float*>(herk_real_temp_);
            auto* temp_Im    = reinterpret_cast<float*>(herk_imag_temp_);

            // Re(C) = α·(Xr·Xr^T + Xi·Xi^T) via stacked-K GEMM with FP32 output
            status = run_real_gemm_fp32out(Stacked, Stacked_B, scratch_Re,
                                            N, N, 2 * K, alpha, 0.0f, stream_a_, actual_tile);
            if (status != cutlass::Status::kSuccess) break;

            // Im temp = Xi·Xr^T (FP32 output)
            status = run_real_gemm_fp32out(Xi_sep, Xr_sep, temp_Im,
                                            N, N, K, 1.0f, 0.0f, stream_b_, actual_tile);
            if (status != cutlass::Status::kSuccess) break;

            // Sync both streams
            CUDA_CHECK(cudaEventRecord(stream_a_done_, stream_a_));
            CUDA_CHECK(cudaEventRecord(stream_b_done_, stream_b_));
            CUDA_CHECK(cudaStreamWaitEvent(stream, stream_a_done_));
            CUDA_CHECK(cudaStreamWaitEvent(stream, stream_b_done_));

            // Fused pack + antisymmetrize + interleave → interleaved FP32 packed
            float* C_tile = C + tile_start * interleaved_stride;
            pack_antisymmetrize_interleave_fp32(
                scratch_Re, temp_Im,
                C_tile,
                N, alpha, beta, fill, stream, actual_tile,
                (beta != 0.0f) ? C_tile : nullptr);
        }

        if (owns_Ar_Ai) {
            CUDA_CHECK(cudaFreeAsync(Ar, stream));
            CUDA_CHECK(cudaFreeAsync(Ai, stream));
        }

        return status;
    }

    // ---------------------------------------------------------------
    // Interleave / deinterleave helpers
    // ---------------------------------------------------------------
    static void deinterleave_complex(
        const __half* interleaved, __half* real_out, __half* imag_out,
        int64_t num_complex_elements, cudaStream_t stream);

    static void interleave_complex(
        const __half* real_in, const __half* imag_in, __half* interleaved_out,
        int64_t num_complex_elements, cudaStream_t stream);
