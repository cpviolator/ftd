    cutlass::Status herk_planar(
        const __half* A_real, const __half* A_imag,
        __half* C_real, __half* C_imag,
        int N, int K,
        float alpha, float beta,
        HerkOp op, FillMode fill,
        HerkStrategy strategy,
        ComputePrecision precision,
        cudaStream_t stream = nullptr,
        GemmConfig config = GemmConfig::Default,
        const TriangleConfig& tri = {})
    {
        if (precision == ComputePrecision::FP8_E4M3) {
            return herk_planar(A_real, A_imag, C_real, C_imag,
                               N, K, alpha, beta, op, fill, strategy, stream, config, tri);
        }

        // NoTrans block-scaled: 3-GEMM path (preprocess A once, reuse for B)
        if (op == HerkOp::NoTrans) {
            return herk_planar_blockscaled(
                A_real, A_imag, C_real, C_imag,
                N, K, alpha, beta, fill, precision, stream, config,
                strategy, tri);
        }

        // ConjTrans: fall back to 4-GEMM Gram + symmetrize
        int gram_M = K, gram_K = N;
        auto status = run_gram_planar(A_real, A_imag, C_real, C_imag,
                                       gram_M, gram_K, alpha, beta, GramMode::AHA,
                                       precision, stream, config);
        if (status != cutlass::Status::kSuccess) return status;
        enforce_hermitian_triangle(C_real, C_imag, N, fill, stream);
        return cutlass::Status::kSuccess;
    }

    /// Multi-precision HERK with packed output.
    /// NoTrans block-scaled: 3-GEMM path (preprocess once, reuse for A and B).
    /// ConjTrans block-scaled: 4-GEMM fallback (no transposed MXFP preprocessing).
    cutlass::Status herk_planar_packed(
        const __half* A_real, const __half* A_imag,
        __half* C_real_packed, __half* C_imag_packed,
        int N, int K,
        float alpha, float beta,
        HerkOp op, FillMode fill,
        HerkStrategy strategy,
        ComputePrecision precision,
        cudaStream_t stream = nullptr,
        GemmConfig config = GemmConfig::Default,
        const TriangleConfig& tri = {})
    {
        if (precision == ComputePrecision::FP8_E4M3) {
            return herk_planar_packed(A_real, A_imag, C_real_packed, C_imag_packed,
                                      N, K, alpha, beta, op, fill, strategy, stream, config, tri);
        }

        // NoTrans block-scaled: 3-GEMM path (preprocess A once, reuse for B)
        if (op == HerkOp::NoTrans) {
            return herk_planar_packed_blockscaled(
                A_real, A_imag, C_real_packed, C_imag_packed,
                N, K, alpha, beta, fill, precision, stream, config,
                strategy, tri);
        }

        // ConjTrans: fall back to 4-GEMM Gram + pack
        int64_t full_size = static_cast<int64_t>(N) * N;
        __half *d_Cr, *d_Ci;
        CUDA_CHECK(cudaMallocAsync(&d_Cr, full_size * sizeof(__half), stream));
        CUDA_CHECK(cudaMallocAsync(&d_Ci, full_size * sizeof(__half), stream));

        auto status = herk_planar(A_real, A_imag, d_Cr, d_Ci,
                                   N, K, alpha, beta, op, fill, strategy,
                                   precision, stream, config, tri);
        if (status == cutlass::Status::kSuccess) {
            pack_triangle(d_Cr, C_real_packed, N, fill, stream);
            pack_triangle(d_Ci, C_imag_packed, N, fill, stream);
        }

        cudaFreeAsync(d_Cr, stream);
        cudaFreeAsync(d_Ci, stream);
        return status;
    }

    // ---------------------------------------------------------------
    // HERK — Hermitian Rank-K Update
    // ---------------------------------------------------------------
    // PRODUCTION (COMPLEX_FP8_HERK_FULL_MATRIX=0): 3 sub-GEMMs + anti-symmetrize
    // DEBUG (COMPLEX_FP8_HERK_FULL_MATRIX=1): 4 sub-GEMMs + full symmetrize
    cutlass::Status herk_planar(
        const __half* A_real, const __half* A_imag,
        __half* C_real, __half* C_imag,
        int N, int K,
        float alpha = 1.0f,
        float beta  = 0.0f,
        HerkOp op = HerkOp::NoTrans,
        FillMode fill = FillMode::Lower,
        HerkStrategy strategy = HerkStrategy::Baseline,
        cudaStream_t stream = nullptr,
        GemmConfig config = GemmConfig::Default,
        const TriangleConfig& tri = {})
    {
#if COMPLEX_FP8_HERK_FULL_MATRIX
        // ---- DEBUG: 4 sub-GEMMs via Gram, then symmetrize ----
        int gram_M, gram_K;
        GramMode gram_mode;
        if (op == HerkOp::NoTrans) {
            gram_M = N; gram_K = K; gram_mode = GramMode::AAH;
        } else {
            gram_M = K; gram_K = N; gram_mode = GramMode::AHA;
        }
        auto status = run_gram_planar(A_real, A_imag, C_real, C_imag,
                                       gram_M, gram_K, alpha, beta, gram_mode, stream, config);
        if (status != cutlass::Status::kSuccess) return status;
        enforce_hermitian_triangle(C_real, C_imag, N, fill, stream);
        return cutlass::Status::kSuccess;

#else
        // ---- PRODUCTION: 3 sub-GEMMs + anti-symmetrize ----
        // (Baseline or TriangleAware depending on strategy)
        ensure_streams();
        ensure_hw_info();

        // Skip triangle decomposition when K >> N — grouped GEMM overhead
        // exceeds the 25% FLOP savings due to heterogeneous group load imbalance.
        if (strategy == HerkStrategy::TriangleAware && K > 4 * N) {
            strategy = HerkStrategy::Baseline;
        }

        int64_t size_A = static_cast<int64_t>(N) * K;

        // Stacked-K pointer (NoTrans) or separate A/B pointers (ConjTrans)
        cutlass::float_e4m3_t *Stacked = nullptr, *Xi_sep = nullptr, *Xr_sep = nullptr;
        cutlass::float_e4m3_t *Xr_A = nullptr, *Xi_A = nullptr, *Xr_B = nullptr, *Xi_B = nullptr;

        if (op == HerkOp::NoTrans) {
            // Stacked-K: cast to N×2K stacked + separate Xi,Xr
            int64_t stacked_size = size_A * 2;  // N × 2K
            buffers_.ensure_stacked_capacity(stacked_size, size_A, stream);
            cast_fp16_to_fp8_stacked_and_separate(A_real, A_imag,
                reinterpret_cast<__nv_fp8_e4m3*>(buffers_.stacked()),
                reinterpret_cast<__nv_fp8_e4m3*>(buffers_.xi_separate()),
                reinterpret_cast<__nv_fp8_e4m3*>(buffers_.xr_separate()),
                N, K, stream);
            Stacked = buffers_.stacked();
            Xi_sep  = buffers_.xi_separate();
            Xr_sep  = buffers_.xr_separate();

            // Duplicate stacked buffer to eliminate TMA L2 cache contention
            // when the same pointer is used for both A and B operands.
            auto* Stacked_B_tmp = buffers_.stacked_b();
            CUDA_CHECK(cudaMemcpyAsync(Stacked_B_tmp, Stacked,
                stacked_size * sizeof(cutlass::float_e4m3_t),
                cudaMemcpyDeviceToDevice, stream));
        } else {
            // F2: Fused quad cast + transpose + duplicate — AHA path
            buffers_.ensure_capacity(size_A, 0, size_A, stream);
            cast_fp16_to_fp8_e4m3_transposed_quad(A_real, A_imag,
                buffers_.BT_real(), buffers_.A_real(),
                buffers_.BT_imag(), buffers_.A_imag(), K, N, stream);
            Xr_A = buffers_.BT_real();  Xi_A = buffers_.BT_imag();
            Xr_B = buffers_.A_real();   Xi_B = buffers_.A_imag();
        }

        int64_t output_size = static_cast<int64_t>(N) * N;
        ensure_herk_temp(output_size, stream);

        CUDA_CHECK(cudaEventRecord(preprocess_done_, stream));
        CUDA_CHECK(cudaStreamWaitEvent(stream_a_, preprocess_done_));
        CUDA_CHECK(cudaStreamWaitEvent(stream_b_, preprocess_done_));

        cutlass::Status status;
        auto* Cr = reinterpret_cast<cutlass::half_t*>(C_real);

        if (op == HerkOp::NoTrans) {
            auto* Stacked_B = buffers_.stacked_b();
            if (strategy == HerkStrategy::TriangleAware) {
                // Stacked-K triangle: single launch at K_eff=2K
                status = run_real_gemm_lower_triangle_dispatched(
                    Stacked, Stacked_B, Cr, N, 2 * K, alpha, beta, stream_a_, 1, tri, hw_sm_count_, config);
                if (status != cutlass::Status::kSuccess) return status;
            } else {
                // Stacked-K: single GEMM with K_eff=2K computes Re(C) = Xr·Xr^T + Xi·Xi^T
                status = run_real_gemm(Stacked, Stacked_B, Cr, N, N, 2 * K, alpha, beta, stream_a_, 1, config);
                if (status != cutlass::Status::kSuccess) return status;
            }

            // Stream B: temp = Xi·Xr^T (Im path, separate buffers, original K)
            auto* temp_ptr = reinterpret_cast<cutlass::half_t*>(herk_imag_temp_);
            status = run_real_gemm(Xi_sep, Xr_sep, temp_ptr, N, N, K, 1.0f, 0.0f, stream_b_, 1, config);
            if (status != cutlass::Status::kSuccess) return status;
        } else {
            if (strategy == HerkStrategy::TriangleAware) {
                status = run_real_gemm_lower_triangle_dispatched(
                    Xr_A, Xr_B, Cr, N, K, alpha, beta, stream_a_, 1, tri, hw_sm_count_, config);
                if (status != cutlass::Status::kSuccess) return status;
                status = run_real_gemm_lower_triangle_dispatched(
                    Xi_A, Xi_B, Cr, N, K, alpha, 1.0f, stream_a_, 1, tri, hw_sm_count_, config);
                if (status != cutlass::Status::kSuccess) return status;
            } else {
                status = run_real_gemm(Xr_A, Xr_B, Cr, N, N, K, alpha, beta, stream_a_, 1, config);
                if (status != cutlass::Status::kSuccess) return status;
                status = run_real_gemm(Xi_A, Xi_B, Cr, N, N, K, alpha, 1.0f, stream_a_, 1, config);
                if (status != cutlass::Status::kSuccess) return status;
            }

            // Stream B: temp = Xi·Xr^T (full matrix — anti-symmetrize reads both triangles)
            auto* temp_ptr = reinterpret_cast<cutlass::half_t*>(herk_imag_temp_);
            status = run_real_gemm(Xi_A, Xr_B, temp_ptr, N, N, K, 1.0f, 0.0f, stream_b_, 1, config);
            if (status != cutlass::Status::kSuccess) return status;
        }

        antisymmetrize_to_triangle(
            herk_imag_temp_, C_imag, C_imag,
            N, alpha, beta, fill, stream_b_);

        CUDA_CHECK(cudaEventRecord(stream_a_done_, stream_a_));
        CUDA_CHECK(cudaEventRecord(stream_b_done_, stream_b_));
        CUDA_CHECK(cudaStreamWaitEvent(stream, stream_a_done_));
        CUDA_CHECK(cudaStreamWaitEvent(stream, stream_b_done_));

        return cutlass::Status::kSuccess;
#endif
    }

    /// Planar complex HERK with packed triangular output (F3 fusion).
    ///
    /// Like herk_planar() but outputs directly to packed N*(N+1)/2 format.
    /// F3: anti-symmetrize writes directly to packed Im(C), eliminating N×N intermediate.
    /// Re(C) is packed via library-side pack kernel after sub-GEMM output.
    cutlass::Status herk_planar_packed(
        const __half* A_real, const __half* A_imag,
        __half* C_real_packed, __half* C_imag_packed,
        int N, int K,
        float alpha = 1.0f,
        float beta  = 0.0f,
        HerkOp op = HerkOp::NoTrans,
        FillMode fill = FillMode::Lower,
        HerkStrategy strategy = HerkStrategy::Baseline,
        cudaStream_t stream = nullptr,
        GemmConfig config = GemmConfig::Default,
        const TriangleConfig& tri = {})
    {
#if COMPLEX_FP8_HERK_FULL_MATRIX
        // ---- DEBUG MODE: 4 sub-GEMMs via Gram → full N×N → pack ----
        int64_t full_size = static_cast<int64_t>(N) * N;
        __half *d_Cr, *d_Ci;
        CUDA_CHECK(cudaMallocAsync(&d_Cr, full_size * sizeof(__half), stream));
        CUDA_CHECK(cudaMallocAsync(&d_Ci, full_size * sizeof(__half), stream));

        int gram_M, gram_K;
        GramMode gram_mode;
        if (op == HerkOp::NoTrans) {
            gram_M = N; gram_K = K; gram_mode = GramMode::AAH;
        } else {
            gram_M = K; gram_K = N; gram_mode = GramMode::AHA;
        }
        auto status = run_gram_planar(A_real, A_imag, d_Cr, d_Ci,
                                       gram_M, gram_K, alpha, beta, gram_mode, stream, config);
        if (status != cutlass::Status::kSuccess) {
            cudaFreeAsync(d_Cr, stream); cudaFreeAsync(d_Ci, stream);
            return status;
        }
        enforce_hermitian_triangle(d_Cr, d_Ci, N, fill, stream);
        pack_triangle(d_Cr, C_real_packed, N, fill, stream);
        pack_triangle(d_Ci, C_imag_packed, N, fill, stream);

        cudaFreeAsync(d_Cr, stream);
        cudaFreeAsync(d_Ci, stream);
        return cutlass::Status::kSuccess;

#else
        // ---- PRODUCTION MODE: 3 sub-GEMMs + F3 fused anti-symmetrize+pack ----
        ensure_streams();
        ensure_hw_info();

        // Skip triangle decomposition when K >> N — grouped GEMM overhead
        // exceeds the 25% FLOP savings due to heterogeneous group load imbalance.
        if (strategy == HerkStrategy::TriangleAware && K > 4 * N) {
            strategy = HerkStrategy::Baseline;
        }

        // ---- CUDA Graph path: capture entire HERK as graph for replay ----
        // Only for NoTrans + Baseline (the dedisp hot path at small K).
        if (use_herk_graph_ && op == HerkOp::NoTrans && strategy == HerkStrategy::Baseline) {
            // Check cache for existing graph
            auto* entry = find_herk_graph(N, K, 1, alpha, beta, A_real, A_imag,
                                           C_real_packed, C_imag_packed);
            if (entry) {
                CUDA_CHECK(cudaGraphLaunch(entry->exec, stream));
                return cutlass::Status::kSuccess;
            }

            // Pre-allocate ALL buffers before capture (cudaMallocAsync is NOT capturable)
            int64_t size_A = static_cast<int64_t>(N) * K;
            int64_t output_size = static_cast<int64_t>(N) * N;
            int64_t stacked_size = static_cast<int64_t>(N) * K * 2;
            buffers_.ensure_stacked_capacity(stacked_size, size_A, stream);
            ensure_herk_real_temp(output_size, stream);
            ensure_herk_temp(output_size, stream);
            ensure_herk_gemm_workspace(N, K, 1);

            // Sync to ensure all async allocations are complete before capture
            CUDA_CHECK(cudaStreamSynchronize(stream));

            auto* Stacked_g  = buffers_.stacked();
            auto* Stacked_b_g = buffers_.stacked_b();
            auto* Xi_sep_g   = buffers_.xi_separate();
            auto* Xr_sep_g   = buffers_.xr_separate();
            auto* scratch_Re_g = reinterpret_cast<cutlass::half_t*>(herk_real_temp_);
            auto* temp_ptr_g   = reinterpret_cast<cutlass::half_t*>(herk_imag_temp_);

            // Use dedicated capture stream (null/legacy stream cannot be captured).
            // Relaxed mode allows non-stream CUDA API calls (e.g. cudaFuncSetAttribute
            // inside CUTLASS run()) during capture.
            cudaStream_t cap = capture_stream_;
            CUDA_CHECK(cudaStreamBeginCapture(cap, cudaStreamCaptureModeRelaxed));

            // Cast FP16 → FP8 stacked + separate
            cast_fp16_to_fp8_stacked_and_separate(A_real, A_imag,
                reinterpret_cast<__nv_fp8_e4m3*>(Stacked_g),
                reinterpret_cast<__nv_fp8_e4m3*>(Xi_sep_g),
                reinterpret_cast<__nv_fp8_e4m3*>(Xr_sep_g), N, K, cap);

            // Duplicate stacked buffer for TMA L2 fix
            CUDA_CHECK(cudaMemcpyAsync(Stacked_b_g, Stacked_g,
                stacked_size * sizeof(cutlass::float_e4m3_t),
                cudaMemcpyDeviceToDevice, cap));

            // Fork to two streams via event
            CUDA_CHECK(cudaEventRecord(preprocess_done_, cap));
            CUDA_CHECK(cudaStreamWaitEvent(stream_a_, preprocess_done_));
            CUDA_CHECK(cudaStreamWaitEvent(stream_b_, preprocess_done_));

            // Stream A: Re(C) = Stacked × Stacked_B^T with K_eff=2K
            auto status = run_real_gemm(Stacked_g, Stacked_b_g, scratch_Re_g,
                                         N, N, 2 * K, alpha, 0.0f, stream_a_, 1, config,
                                         herk_gemm_workspace_a_, herk_gemm_workspace_size_);
            if (status != cutlass::Status::kSuccess) {
                cudaGraph_t abandoned; cudaStreamEndCapture(cap, &abandoned);
                if (abandoned) cudaGraphDestroy(abandoned);
                return status;
            }

            // Stream B: Im temp = Xi × Xr^T with original K
            status = run_real_gemm(Xi_sep_g, Xr_sep_g, temp_ptr_g,
                                    N, N, K, 1.0f, 0.0f, stream_b_, 1, config,
                                    herk_gemm_workspace_b_, herk_gemm_workspace_size_);
            if (status != cutlass::Status::kSuccess) {
                cudaGraph_t abandoned; cudaStreamEndCapture(cap, &abandoned);
                if (abandoned) cudaGraphDestroy(abandoned);
                return status;
            }

            // Join streams back to capture stream
            CUDA_CHECK(cudaEventRecord(stream_a_done_, stream_a_));
            CUDA_CHECK(cudaEventRecord(stream_b_done_, stream_b_));
            CUDA_CHECK(cudaStreamWaitEvent(cap, stream_a_done_));
            CUDA_CHECK(cudaStreamWaitEvent(cap, stream_b_done_));

            // Fused pack on capture stream
            if (beta != 0.0f) {
                pack_antisymmetrize_triangle(
                    herk_real_temp_, herk_imag_temp_,
                    C_real_packed, C_imag_packed,
                    N, alpha, beta, fill, cap, 1,
                    C_real_packed, C_imag_packed);
            } else {
                pack_antisymmetrize_triangle(
                    herk_real_temp_, herk_imag_temp_,
                    C_real_packed, C_imag_packed,
                    N, alpha, 0.0f, fill, cap);
            }

            // End capture, instantiate, store, launch on CALLER's stream
            cudaGraph_t graph;
            CUDA_CHECK(cudaStreamEndCapture(cap, &graph));
            cudaGraphExec_t exec;
            CUDA_CHECK(cudaGraphInstantiateWithFlags(&exec, graph, 0));
            cudaGraphDestroy(graph);

            auto* slot = alloc_herk_graph_slot();
            slot->store(N, K, 1, alpha, beta, A_real, A_imag,
                        C_real_packed, C_imag_packed, exec);

            CUDA_CHECK(cudaGraphLaunch(exec, stream));
            return cutlass::Status::kSuccess;
        }

        // ---- Standard (non-graph) path ----
        int64_t size_A = static_cast<int64_t>(N) * K;
        int64_t output_size = static_cast<int64_t>(N) * N;

        // Stacked-K pointer (NoTrans) or separate A/B pointers (ConjTrans)
        cutlass::float_e4m3_t *Stacked = nullptr, *Xi_sep = nullptr, *Xr_sep = nullptr;
        cutlass::float_e4m3_t *Xr_A = nullptr, *Xi_A = nullptr, *Xr_B = nullptr, *Xi_B = nullptr;

        if (op == HerkOp::NoTrans) {
            // Stacked-K: cast to N×2K stacked + separate Xi,Xr
            int64_t stacked_size = size_A * 2;  // N × 2K
            buffers_.ensure_stacked_capacity(stacked_size, size_A, stream);
            cast_fp16_to_fp8_stacked_and_separate(A_real, A_imag,
                reinterpret_cast<__nv_fp8_e4m3*>(buffers_.stacked()),
                reinterpret_cast<__nv_fp8_e4m3*>(buffers_.xi_separate()),
                reinterpret_cast<__nv_fp8_e4m3*>(buffers_.xr_separate()),
                N, K, stream);
            Stacked = buffers_.stacked();
            Xi_sep  = buffers_.xi_separate();
            Xr_sep  = buffers_.xr_separate();

            // Duplicate stacked buffer for TMA L2 fix
            auto* Stacked_B_tmp = buffers_.stacked_b();
            CUDA_CHECK(cudaMemcpyAsync(Stacked_B_tmp, Stacked,
                stacked_size * sizeof(cutlass::float_e4m3_t),
                cudaMemcpyDeviceToDevice, stream));
        } else {
            buffers_.ensure_capacity(size_A, 0, size_A, stream);
            cast_fp16_to_fp8_e4m3_transposed_quad(A_real, A_imag,
                buffers_.BT_real(), buffers_.A_real(),
                buffers_.BT_imag(), buffers_.A_imag(), K, N, stream);
            Xr_A = buffers_.BT_real();  Xi_A = buffers_.BT_imag();
            Xr_B = buffers_.A_real();   Xi_B = buffers_.A_imag();
        }

        ensure_herk_real_temp(output_size, stream);
        ensure_herk_temp(output_size, stream);

        CUDA_CHECK(cudaEventRecord(preprocess_done_, stream));
        CUDA_CHECK(cudaStreamWaitEvent(stream_a_, preprocess_done_));
        CUDA_CHECK(cudaStreamWaitEvent(stream_b_, preprocess_done_));

        cutlass::Status status;
        auto* scratch_Re = reinterpret_cast<cutlass::half_t*>(herk_real_temp_);

        if (op == HerkOp::NoTrans) {
            auto* Stacked_B = buffers_.stacked_b();
            if (strategy == HerkStrategy::TriangleAware) {
                // Stacked-K triangle: single launch at K_eff=2K
                status = run_real_gemm_lower_triangle_dispatched(
                    Stacked, Stacked_B, scratch_Re, N, 2 * K, alpha, 0.0f, stream_a_, 1, tri, hw_sm_count_, config);
                if (status != cutlass::Status::kSuccess) return status;
            } else {
                // Stacked-K: single GEMM with K_eff=2K computes Re(C)
                status = run_real_gemm(Stacked, Stacked_B, scratch_Re,
                                       N, N, 2 * K, alpha, 0.0f, stream_a_, 1, config);
                if (status != cutlass::Status::kSuccess) return status;
            }

            // Stream B: temp = Xi·Xr^T (Im path, separate buffers, original K)
            auto* temp_ptr = reinterpret_cast<cutlass::half_t*>(herk_imag_temp_);
            status = run_real_gemm(Xi_sep, Xr_sep, temp_ptr,
                                   N, N, K, 1.0f, 0.0f, stream_b_, 1, config);
            if (status != cutlass::Status::kSuccess) return status;
        } else {
            if (strategy == HerkStrategy::TriangleAware) {
                status = run_real_gemm_lower_triangle_dispatched(
                    Xr_A, Xr_B, scratch_Re, N, K, alpha, 0.0f, stream_a_, 1, tri, hw_sm_count_, config);
                if (status != cutlass::Status::kSuccess) return status;
                status = run_real_gemm_lower_triangle_dispatched(
                    Xi_A, Xi_B, scratch_Re, N, K, alpha, 1.0f, stream_a_, 1, tri, hw_sm_count_, config);
                if (status != cutlass::Status::kSuccess) return status;
            } else {
                status = run_real_gemm(Xr_A, Xr_B, scratch_Re,
                                       N, N, K, alpha, 0.0f, stream_a_, 1, config);
                if (status != cutlass::Status::kSuccess) return status;
                status = run_real_gemm(Xi_A, Xi_B, scratch_Re,
                                       N, N, K, alpha, 1.0f, stream_a_, 1, config);
                if (status != cutlass::Status::kSuccess) return status;
            }

            // Stream B: temp = Xi·Xr^T
            auto* temp_ptr = reinterpret_cast<cutlass::half_t*>(herk_imag_temp_);
            status = run_real_gemm(Xi_A, Xr_B, temp_ptr,
                                   N, N, K, 1.0f, 0.0f, stream_b_, 1, config);
            if (status != cutlass::Status::kSuccess) return status;
        }

        // Sync both streams to caller, then fused pack on caller stream
        CUDA_CHECK(cudaEventRecord(stream_a_done_, stream_a_));
        CUDA_CHECK(cudaEventRecord(stream_b_done_, stream_b_));
        CUDA_CHECK(cudaStreamWaitEvent(stream, stream_a_done_));
        CUDA_CHECK(cudaStreamWaitEvent(stream, stream_b_done_));

        // Fused pack Re(C) + antisymmetrize+pack Im(C) on caller stream
        if (beta != 0.0f) {
            pack_antisymmetrize_triangle(
                herk_real_temp_, herk_imag_temp_,
                C_real_packed, C_imag_packed,
                N, alpha, beta, fill, stream, 1,
                C_real_packed, C_imag_packed);
        } else {
            pack_antisymmetrize_triangle(
                herk_real_temp_, herk_imag_temp_,
                C_real_packed, C_imag_packed,
                N, alpha, 0.0f, fill, stream);
        }

        return cutlass::Status::kSuccess;
#endif
    }

    /// Batched planar complex HERK with packed triangular output (F3 fusion).
    ///
    /// FP8 path only — for block-scaled (FP6/FP4), use
    /// herk_planar_packed_batched_blockscaled() instead.
    cutlass::Status herk_planar_packed_batched(
        const __half* A_real, const __half* A_imag,
        __half* C_real_packed, __half* C_imag_packed,   // Packed N*(N+1)/2 × batch_count
        int N, int K,
        int batch_count,
        float alpha = 1.0f,
        float beta  = 0.0f,
        HerkOp op = HerkOp::NoTrans,
        FillMode fill = FillMode::Lower,
        HerkStrategy strategy = HerkStrategy::Baseline,
        cudaStream_t stream = nullptr,
        const TriangleConfig& tri = {},
        GemmConfig config = GemmConfig::Default)
    {
#if COMPLEX_FP8_HERK_FULL_MATRIX
        // ---- DEBUG MODE: via Gram → full N×N → pack, looped over batches ----
        int64_t per_output = static_cast<int64_t>(N) * N;
        int64_t full_size = per_output * batch_count;
        int64_t per_A = static_cast<int64_t>(N) * K;
        __half *d_Cr, *d_Ci;
        CUDA_CHECK(cudaMallocAsync(&d_Cr, full_size * sizeof(__half), stream));
        CUDA_CHECK(cudaMallocAsync(&d_Ci, full_size * sizeof(__half), stream));

        int gram_M, gram_K;
        GramMode gram_mode;
        if (op == HerkOp::NoTrans) {
            gram_M = N; gram_K = K; gram_mode = GramMode::AAH;
        } else {
            gram_M = K; gram_K = N; gram_mode = GramMode::AHA;
        }
        for (int b = 0; b < batch_count; ++b) {
            auto status = run_gram_planar(
                A_real + b * per_A, A_imag + b * per_A,
                d_Cr + b * per_output, d_Ci + b * per_output,
                gram_M, gram_K, alpha, beta, gram_mode, stream);
            if (status != cutlass::Status::kSuccess) {
                cudaFreeAsync(d_Cr, stream); cudaFreeAsync(d_Ci, stream);
                return status;
            }
        }
        enforce_hermitian_triangle(d_Cr, d_Ci, N, fill, stream, batch_count);
        pack_triangle(d_Cr, C_real_packed, N, fill, stream, batch_count);
        pack_triangle(d_Ci, C_imag_packed, N, fill, stream, batch_count);

        cudaFreeAsync(d_Cr, stream);
        cudaFreeAsync(d_Ci, stream);
        return cutlass::Status::kSuccess;

#else
        // ---- PRODUCTION MODE: 3 batched sub-GEMMs + F3 fused anti-symmetrize+pack ----
        if (batch_count <= 0) return cutlass::Status::kSuccess;
        ensure_streams();
        ensure_hw_info();

        // Skip triangle decomposition when K >> N — grouped GEMM overhead
        // exceeds the 25% FLOP savings due to heterogeneous group load imbalance.
        if (strategy == HerkStrategy::TriangleAware && K > 4 * N) {
            strategy = HerkStrategy::Baseline;
        }

        // ---- CUDA Graph path: capture entire batched HERK as graph for replay ----
        if (use_herk_graph_ && op == HerkOp::NoTrans && strategy == HerkStrategy::Baseline) {
            auto* entry = find_herk_graph(N, K, batch_count, alpha, beta, A_real, A_imag,
                                           C_real_packed, C_imag_packed);
            if (entry) {
                CUDA_CHECK(cudaGraphLaunch(entry->exec, stream));
                return cutlass::Status::kSuccess;
            }

            // Pre-allocate ALL buffers before capture
            int64_t per_A = static_cast<int64_t>(N) * K;
            int64_t total_A = per_A * batch_count;
            int64_t total_output = static_cast<int64_t>(N) * N * batch_count;
            int64_t stacked_total = total_A * 2;
            buffers_.ensure_stacked_capacity(stacked_total, total_A, stream);
            ensure_herk_real_temp(total_output, stream);
            ensure_herk_temp(total_output, stream);
            ensure_herk_gemm_workspace(N, K, batch_count);

            CUDA_CHECK(cudaStreamSynchronize(stream));

            auto* Stacked_g  = buffers_.stacked();
            auto* Stacked_b_g = buffers_.stacked_b();
            auto* Xi_sep_g   = buffers_.xi_separate();
            auto* Xr_sep_g   = buffers_.xr_separate();
            auto* scratch_Re_g = reinterpret_cast<cutlass::half_t*>(herk_real_temp_);
            auto* temp_ptr_g   = reinterpret_cast<cutlass::half_t*>(herk_imag_temp_);

            // Use dedicated capture stream (null/legacy stream cannot be captured).
            cudaStream_t cap = capture_stream_;
            CUDA_CHECK(cudaStreamBeginCapture(cap, cudaStreamCaptureModeRelaxed));

            cast_fp16_to_fp8_stacked_and_separate(A_real, A_imag,
                reinterpret_cast<__nv_fp8_e4m3*>(Stacked_g),
                reinterpret_cast<__nv_fp8_e4m3*>(Xi_sep_g),
                reinterpret_cast<__nv_fp8_e4m3*>(Xr_sep_g),
                N * batch_count, K, cap);

            // Duplicate stacked buffer for TMA L2 fix
            CUDA_CHECK(cudaMemcpyAsync(Stacked_b_g, Stacked_g,
                stacked_total * sizeof(cutlass::float_e4m3_t),
                cudaMemcpyDeviceToDevice, cap));

            CUDA_CHECK(cudaEventRecord(preprocess_done_, cap));
            CUDA_CHECK(cudaStreamWaitEvent(stream_a_, preprocess_done_));
            CUDA_CHECK(cudaStreamWaitEvent(stream_b_, preprocess_done_));

            auto status = run_real_gemm(Stacked_g, Stacked_b_g, scratch_Re_g,
                                         N, N, 2 * K, alpha, 0.0f, stream_a_, batch_count,
                                         config,
                                         herk_gemm_workspace_a_, herk_gemm_workspace_size_);
            if (status != cutlass::Status::kSuccess) {
                cudaGraph_t abandoned; cudaStreamEndCapture(cap, &abandoned);
                if (abandoned) cudaGraphDestroy(abandoned);
                return status;
            }

            status = run_real_gemm(Xi_sep_g, Xr_sep_g, temp_ptr_g,
                                    N, N, K, 1.0f, 0.0f, stream_b_, batch_count,
                                    config,
                                    herk_gemm_workspace_b_, herk_gemm_workspace_size_);
            if (status != cutlass::Status::kSuccess) {
                cudaGraph_t abandoned; cudaStreamEndCapture(cap, &abandoned);
                if (abandoned) cudaGraphDestroy(abandoned);
                return status;
            }

            CUDA_CHECK(cudaEventRecord(stream_a_done_, stream_a_));
            CUDA_CHECK(cudaEventRecord(stream_b_done_, stream_b_));
            CUDA_CHECK(cudaStreamWaitEvent(cap, stream_a_done_));
            CUDA_CHECK(cudaStreamWaitEvent(cap, stream_b_done_));

            if (beta != 0.0f) {
                pack_antisymmetrize_triangle(
                    herk_real_temp_, herk_imag_temp_,
                    C_real_packed, C_imag_packed,
                    N, alpha, beta, fill, cap, batch_count,
                    C_real_packed, C_imag_packed);
            } else {
                pack_antisymmetrize_triangle(
                    herk_real_temp_, herk_imag_temp_,
                    C_real_packed, C_imag_packed,
                    N, alpha, 0.0f, fill, cap, batch_count);
            }

            cudaGraph_t graph;
            CUDA_CHECK(cudaStreamEndCapture(cap, &graph));
            cudaGraphExec_t exec;
            CUDA_CHECK(cudaGraphInstantiateWithFlags(&exec, graph, 0));
            cudaGraphDestroy(graph);

            auto* slot = alloc_herk_graph_slot();
            slot->store(N, K, batch_count, alpha, beta, A_real, A_imag,
                        C_real_packed, C_imag_packed, exec);

            CUDA_CHECK(cudaGraphLaunch(exec, stream));
            return cutlass::Status::kSuccess;
        }

        // ---- Standard (non-graph) batched path ----
        int64_t per_A = static_cast<int64_t>(N) * K;
        int64_t total_A = per_A * batch_count;
        int64_t output_per_batch = static_cast<int64_t>(N) * N;
        int64_t total_output = output_per_batch * batch_count;

        // Stacked-K pointer (NoTrans) or separate A/B pointers (ConjTrans)
        cutlass::float_e4m3_t *Stacked = nullptr, *Xi_sep = nullptr, *Xr_sep = nullptr;
        cutlass::float_e4m3_t *Xr_A = nullptr, *Xi_A = nullptr, *Xr_B = nullptr, *Xi_B = nullptr;

        if (op == HerkOp::NoTrans) {
            // Stacked-K: cast all batches to (N*batch)×2K stacked + separate Xi,Xr
            int64_t stacked_total = total_A * 2;  // N*K*batch * 2 = N*2K*batch
            buffers_.ensure_stacked_capacity(stacked_total, total_A, stream);
            cast_fp16_to_fp8_stacked_and_separate(A_real, A_imag,
                reinterpret_cast<__nv_fp8_e4m3*>(buffers_.stacked()),
                reinterpret_cast<__nv_fp8_e4m3*>(buffers_.xi_separate()),
                reinterpret_cast<__nv_fp8_e4m3*>(buffers_.xr_separate()),
                N * batch_count, K, stream);
            Stacked = buffers_.stacked();
            Xi_sep  = buffers_.xi_separate();
            Xr_sep  = buffers_.xr_separate();

            // Duplicate stacked buffer for TMA L2 fix
            auto* Stacked_B_tmp = buffers_.stacked_b();
            CUDA_CHECK(cudaMemcpyAsync(Stacked_B_tmp, Stacked,
                stacked_total * sizeof(cutlass::float_e4m3_t),
                cudaMemcpyDeviceToDevice, stream));
        } else {
            buffers_.ensure_capacity(total_A, 0, total_A, stream);
            cast_fp16_to_fp8_e4m3_transposed_quad(A_real, A_imag,
                buffers_.BT_real(), buffers_.A_real(),
                buffers_.BT_imag(), buffers_.A_imag(), K, N, stream, batch_count);
            Xr_A = buffers_.BT_real();  Xi_A = buffers_.BT_imag();
            Xr_B = buffers_.A_real();   Xi_B = buffers_.A_imag();
        }

        ensure_herk_real_temp(total_output, stream);
        ensure_herk_temp(total_output, stream);

        CUDA_CHECK(cudaEventRecord(preprocess_done_, stream));
        CUDA_CHECK(cudaStreamWaitEvent(stream_a_, preprocess_done_));
        CUDA_CHECK(cudaStreamWaitEvent(stream_b_, preprocess_done_));

        // Pre-allocate GEMM workspaces (eliminates per-GEMM cudaMallocAsync/Free)
        ensure_herk_gemm_workspace(N, K, batch_count);

        cutlass::Status status;
        auto* scratch_Re = reinterpret_cast<cutlass::half_t*>(herk_real_temp_);

        if (op == HerkOp::NoTrans) {
            auto* Stacked_B = buffers_.stacked_b();
            // Stacked-K: single GEMM with K_eff=2K computes Re(C) = Xr·Xr^T + Xi·Xi^T (batched)
            if (strategy == HerkStrategy::TriangleAware) {
                status = run_real_gemm_lower_triangle_dispatched(
                    Stacked, Stacked_B, scratch_Re, N, 2 * K, alpha, 0.0f, stream_a_, batch_count, tri, hw_sm_count_, config);
                if (status != cutlass::Status::kSuccess) return status;
            } else {
                status = run_real_gemm(Stacked, Stacked_B, scratch_Re,
                                       N, N, 2 * K, alpha, 0.0f, stream_a_, batch_count,
                                       config, herk_gemm_workspace_a_, herk_gemm_workspace_size_);
                if (status != cutlass::Status::kSuccess) return status;
            }

            // Stream B: temp = Xi·Xr^T (Im path, separate buffers, original K, batched)
            auto* temp_ptr = reinterpret_cast<cutlass::half_t*>(herk_imag_temp_);
            status = run_real_gemm(Xi_sep, Xr_sep, temp_ptr,
                                   N, N, K, 1.0f, 0.0f, stream_b_, batch_count,
                                   config, herk_gemm_workspace_b_, herk_gemm_workspace_size_);
            if (status != cutlass::Status::kSuccess) return status;
        } else {
            // ConjTrans: original 2-GEMM path (batched)
            if (strategy == HerkStrategy::TriangleAware) {
                status = run_real_gemm_lower_triangle_dispatched(
                    Xr_A, Xr_B, scratch_Re, N, K, alpha, 0.0f, stream_a_, batch_count, tri, hw_sm_count_, config);
                if (status != cutlass::Status::kSuccess) return status;
                status = run_real_gemm_lower_triangle_dispatched(
                    Xi_A, Xi_B, scratch_Re, N, K, alpha, 1.0f, stream_a_, batch_count, tri, hw_sm_count_, config);
                if (status != cutlass::Status::kSuccess) return status;
            } else {
                status = run_real_gemm(Xr_A, Xr_B, scratch_Re, N, N, K, alpha, 0.0f, stream_a_, batch_count,
                                       config, herk_gemm_workspace_a_, herk_gemm_workspace_size_);
                if (status != cutlass::Status::kSuccess) return status;
                status = run_real_gemm(Xi_A, Xi_B, scratch_Re, N, N, K, alpha, 1.0f, stream_a_, batch_count,
                                       config, herk_gemm_workspace_a_, herk_gemm_workspace_size_);
                if (status != cutlass::Status::kSuccess) return status;
            }

            // Stream B: temp = Xi·Xr^T (batched, always full N×N)
            auto* temp_ptr = reinterpret_cast<cutlass::half_t*>(herk_imag_temp_);
            status = run_real_gemm(Xi_A, Xr_B, temp_ptr, N, N, K, 1.0f, 0.0f, stream_b_, batch_count,
                                   config, herk_gemm_workspace_b_, herk_gemm_workspace_size_);
            if (status != cutlass::Status::kSuccess) return status;
        }

        // Sync both streams to caller, then fused pack on caller stream
        CUDA_CHECK(cudaEventRecord(stream_a_done_, stream_a_));
        CUDA_CHECK(cudaEventRecord(stream_b_done_, stream_b_));
        CUDA_CHECK(cudaStreamWaitEvent(stream, stream_a_done_));
        CUDA_CHECK(cudaStreamWaitEvent(stream, stream_b_done_));

        // Fused pack Re(C) + antisymmetrize+pack Im(C) on caller stream (batched)
        if (beta != 0.0f) {
            pack_antisymmetrize_triangle(
                herk_real_temp_, herk_imag_temp_,
                C_real_packed, C_imag_packed,
                N, alpha, beta, fill, stream, batch_count,
                C_real_packed, C_imag_packed);
        } else {
            pack_antisymmetrize_triangle(
                herk_real_temp_, herk_imag_temp_,
                C_real_packed, C_imag_packed,
                N, alpha, 0.0f, fill, stream, batch_count);
        }

        return cutlass::Status::kSuccess;
#endif
    }

    cutlass::Status herk_planar_packed_blockscaled(
        const __half* A_real, const __half* A_imag,
        __half* C_real_packed, __half* C_imag_packed,
        int N, int K,
        float alpha, float beta,
        FillMode fill,
        ComputePrecision precision,
        cudaStream_t stream,
        GemmConfig config,
        HerkStrategy strategy = HerkStrategy::Baseline,
        const TriangleConfig& tri = {})
    {
        ensure_streams();
        ensure_hw_info();

        int64_t size_A = static_cast<int64_t>(N) * K;
        int64_t output_size = static_cast<int64_t>(N) * N;

        // Allocate sub-byte data buffers — A only (B reuses A's buffers)
        int64_t bytes_A = bytes_for_elements(size_A, precision);
        lp_buffers_.ensure_capacity(bytes_A, 0, 0, stream);

        // Allocate scale factor buffers — A only (B reuses A's)
        int64_t sf_A_bytes = sf_buffer_bytes(N, K);
        lp_buffers_.ensure_sf_capacity(sf_A_bytes, 0, 0, stream);

        // Preprocess A once: fused SF + scale + cast for Ar and Ai
        preprocess_mxfp_paired_sm100(
            A_real, A_imag,
            lp_buffers_.A_real(), lp_buffers_.A_imag(),
            lp_buffers_.sf_A_real(), lp_buffers_.sf_A_imag(),
            N, K, precision, stream);

        // Scratch buffers for full N×N intermediate results
        ensure_herk_real_temp(output_size, stream);
        ensure_herk_temp(output_size, stream);

        CUDA_CHECK(cudaEventRecord(preprocess_done_, stream));
        CUDA_CHECK(cudaStreamWaitEvent(stream_a_, preprocess_done_));
        CUDA_CHECK(cudaStreamWaitEvent(stream_b_, preprocess_done_));

        cutlass::Status status;
        auto* scratch_Re = reinterpret_cast<cutlass::half_t*>(herk_real_temp_);

        // Self-product: pass same preprocessed A-side buffers for both operands.
        // A is N×K (RowMajor), B is N×K (ColumnMajor = A^T). Same MXFP layout.
        // GEMM only reads, so aliasing A=B is safe.
        void* Ar_data = lp_buffers_.A_real();
        void* Ai_data = lp_buffers_.A_imag();
        void* sf_Ar   = lp_buffers_.sf_A_real();
        void* sf_Ai   = lp_buffers_.sf_A_imag();

        // Skip triangle at very small K — per-slab launch overhead exceeds FLOPs savings
        if (strategy == HerkStrategy::TriangleAware && K <= 64) {
            strategy = HerkStrategy::Baseline;
        }

        if (strategy == HerkStrategy::TriangleAware) {
            // ---- TRIANGLE-AWARE: block-row decomposition for Re(C) ----
            //
            // Decompose the lower triangle into block-rows. Block-row i
            // computes C[row_start : row_end, 0 : row_end], an M_block × N_block × K
            // rectangular GEMM. Total FLOPs ≈ N²K (half of full 2·N²K).
            //
            // Sub-byte data pointer arithmetic (FP6: 0.75 B/elem, FP4: 0.5 B/elem):
            //   A slab offset: bytes_for_elements(row_start * K, precision)
            //   B: always starts at row 0 (no offset)
            //
            // Scale factor pointer arithmetic (SfAtom layout):
            //   num_k_tiles = padded_k_blocks / 4
            //   A SF slab offset: (row_start / 128) * num_k_tiles * 512 bytes
            //   B SF: no offset
            //
            // C pointer (FP16, scratch buffer): row_start * N

            constexpr int kTileM = kFP8TileM;
            constexpr int kTileN = kFP8TileN;
            int sm_count = hw_sm_count_;

            // --- Determine target_slabs ---
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
                // Cap slabs at small K — reduce launch overhead when GEMMs are tiny
                int max_slabs_for_K = std::max(2, K / 64);
                target_slabs = std::min(target_slabs, max_slabs_for_K);
            }

            // --- Determine min_slab_height ---
            int min_slab_height;
            if (tri.min_slab_height > 0) {
                min_slab_height = tri.min_slab_height;
            } else {
                double ratio = static_cast<double>(kTileM) / kTileN;
                int tiles_needed = static_cast<int>(std::ceil(
                    std::sqrt(static_cast<double>(sm_count) / ratio)));
                min_slab_height = kTileM * std::max(tiles_needed, 1);
            }

            // --- Compute slab boundaries ---
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
                    if (boundaries[i] > deduped.back()) {
                        deduped.push_back(boundaries[i]);
                    }
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

            if (tri.verbose) {
                fprintf(stderr, "[Triangle BS] N=%d K=%d sm_count=%d tile=%dx%d\n",
                        N, K, sm_count, kTileM, kTileN);
                fprintf(stderr, "  target_slabs=%d (auto=%s) min_slab=%d (auto=%s) graduated=%s\n",
                        target_slabs, tri.target_slabs > 0 ? "no" : "yes",
                        min_slab_height, tri.min_slab_height > 0 ? "no" : "yes",
                        tri.graduated ? "yes" : "no");
                fprintf(stderr, "  num_slabs=%d  boundaries:", num_slabs);
                for (int i = 0; i <= num_slabs; ++i) fprintf(stderr, " %d", boundaries[i]);
                fprintf(stderr, "\n");
                int64_t total_flops = 0;
                for (int i = 0; i < num_slabs; ++i) {
                    int row_start = boundaries[i];
                    int M_block = boundaries[i + 1] - row_start;
                    int N_block = boundaries[i + 1];
                    int64_t flops = 2LL * M_block * N_block * K;
                    total_flops += flops;
                    fprintf(stderr, "  slab %d: rows [%d..%d) M=%d N=%d\n",
                            i, row_start, boundaries[i + 1], M_block, N_block);
                }
                int64_t full_flops = 2LL * N * N * K;
                fprintf(stderr, "  total FLOPs: %.1f%% of full GEMM\n",
                        100.0 * total_flops / full_flops);
            }

            // SF stride per 128-row tile: num_k_tiles × 512 bytes (SfAtom layout)
            int64_t k_blocks = (K + 31) / 32;
            int64_t padded_k_blocks = ((k_blocks + 3) / 4) * 4;
            int64_t sf_tile_row_stride = (padded_k_blocks / 4) * 512;

            // --- Per-slab loop for Re(C) ---
            for (int i = 0; i < num_slabs; ++i) {
                int row_start = boundaries[i];
                int M_block = boundaries[i + 1] - row_start;
                int N_block = boundaries[i + 1];

                // Sub-byte data pointers: offset A by row_start rows
                // A slab: skip row_start * K elements in sub-byte format
                const char* A_slab_r = static_cast<const char*>(Ar_data)
                    + bytes_for_elements(static_cast<int64_t>(row_start) * K, precision);
                const char* A_slab_i = static_cast<const char*>(Ai_data)
                    + bytes_for_elements(static_cast<int64_t>(row_start) * K, precision);

                // B always starts at row 0 (full matrix)
                // B_data = Ar_data / Ai_data (no offset)

                // Scale factor pointers: offset A SF by row_start/128 tile-rows
                // SfAtom stride per 128-row tile = (padded_k_blocks / 4) tiles × 512 bytes/tile
                const char* sf_A_slab_r = static_cast<const char*>(sf_Ar)
                    + (row_start / 128) * sf_tile_row_stride;
                const char* sf_A_slab_i = static_cast<const char*>(sf_Ai)
                    + (row_start / 128) * sf_tile_row_stride;

                // B SF: no offset (always starts at row 0)

                // C pointer: offset into scratch buffer (FP16, N columns stride)
                auto* C_slab = scratch_Re + row_start * N;

                // First sub-GEMM: Ar_slab × Ar^T → C_slab
                float beta_first = 0.0f;  // scratch always starts clean
                status = run_real_gemm_dispatch(
                    A_slab_r, Ar_data, sf_A_slab_r, sf_Ar,
                    C_slab,
                    M_block, N_block, K, alpha, beta_first, stream_a_, precision, config,
                    /*batch_count=*/1, /*ld_C=*/N);
                if (status != cutlass::Status::kSuccess) return status;

                // Second sub-GEMM: Ai_slab × Ai^T → C_slab (accumulate)
                status = run_real_gemm_dispatch(
                    A_slab_i, Ai_data, sf_A_slab_i, sf_Ai,
                    C_slab,
                    M_block, N_block, K, alpha, 1.0f, stream_a_, precision, config,
                    /*batch_count=*/1, /*ld_C=*/N);
                if (status != cutlass::Status::kSuccess) return status;
            }
        } else {
            // ---- BASELINE: full N×N sub-GEMMs for Re(C) ----
            status = run_real_gemm_dispatch(
                Ar_data, Ar_data, sf_Ar, sf_Ar,
                scratch_Re,
                N, N, K, alpha, 0.0f, stream_a_, precision, config);
            if (status != cutlass::Status::kSuccess) return status;

            status = run_real_gemm_dispatch(
                Ai_data, Ai_data, sf_Ai, sf_Ai,
                scratch_Re,
                N, N, K, alpha, 1.0f, stream_a_, precision, config);
            if (status != cutlass::Status::kSuccess) return status;
        }

        // Stream B: temp = Ai · Ar^T  (alpha=1, beta=0)
        // Im(C) stays as full N×N — anti-symmetrize needs both temp[i,j] and temp[j,i]
        auto* temp_ptr = reinterpret_cast<cutlass::half_t*>(herk_imag_temp_);
        status = run_real_gemm_dispatch(
            Ai_data, Ar_data, sf_Ai, sf_Ar,
            temp_ptr,
            N, N, K, 1.0f, 0.0f, stream_b_, precision, config);
        if (status != cutlass::Status::kSuccess) return status;

        // Sync both streams to caller, then fused pack on caller stream
        CUDA_CHECK(cudaEventRecord(stream_a_done_, stream_a_));
        CUDA_CHECK(cudaEventRecord(stream_b_done_, stream_b_));
        CUDA_CHECK(cudaStreamWaitEvent(stream, stream_a_done_));
        CUDA_CHECK(cudaStreamWaitEvent(stream, stream_b_done_));

        // Fused pack Re(C) + antisymmetrize+pack Im(C) on caller stream
        if (beta != 0.0f) {
            pack_antisymmetrize_triangle(
                herk_real_temp_, herk_imag_temp_,
                C_real_packed, C_imag_packed,
                N, alpha, beta, fill, stream, 1,
                C_real_packed, C_imag_packed);
        } else {
            pack_antisymmetrize_triangle(
                herk_real_temp_, herk_imag_temp_,
                C_real_packed, C_imag_packed,
                N, alpha, 0.0f, fill, stream);
        }

        return cutlass::Status::kSuccess;
    }

    // ---------------------------------------------------------------
    // Batched block-scaled HERK: 3-GEMM path for FP6/FP4 (NoTrans only)
    // ---------------------------------------------------------------
    //
    // Same algorithm as herk_planar_packed_blockscaled() but with batch support.
    // Preprocesses all batch elements at once as (N*batch_count) × K.
    //
    // Baseline: 3 sub-GEMMs with batch_count (single kernel launch each).
    // TriangleAware: per-batch slab loop for Re(C), batched Im(C).
    //   Re on stream_a_, Im on stream_b_ — overlap hides Re slab overhead.
    //
    cutlass::Status herk_planar_packed_batched_blockscaled(
        const __half* A_real, const __half* A_imag,
        __half* C_real_packed, __half* C_imag_packed,
        int N, int K,
        int batch_count,
        float alpha, float beta,
        FillMode fill,
        ComputePrecision precision,
        cudaStream_t stream,
        GemmConfig config,
        HerkStrategy strategy = HerkStrategy::Baseline,
        const TriangleConfig& tri = {},
        __half* C_interleaved = nullptr,
        const __half* A_interleaved = nullptr)
    {
        ensure_streams();
        ensure_hw_info();

        int64_t per_A = static_cast<int64_t>(N) * K;
        int64_t total_A = per_A * batch_count;
        int64_t output_per_batch = static_cast<int64_t>(N) * N;
        int64_t total_output = output_per_batch * batch_count;

        // Allocate sub-byte data buffers — A only (B reuses A's buffers, self-product)
        int64_t bytes_total_A = bytes_for_elements(total_A, precision);
        lp_buffers_.ensure_capacity(bytes_total_A, 0, 0, stream);

        // Allocate scale factor buffers — (N*batch_count) rows × K cols
        int64_t sf_total_bytes = sf_buffer_bytes(N * batch_count, K);
        lp_buffers_.ensure_sf_capacity(sf_total_bytes, 0, 0, stream);

        // Preprocess all batch elements at once: (N*batch_count) × K
        // Since N is always tile-aligned (multiple of 128), the MXFP tile
        // boundaries align correctly across batch boundaries.
        if (A_interleaved) {
            // Fused deinterleave + MXFP: reads interleaved A directly,
            // eliminates planar FP16 intermediate buffers
            deinterleave_preprocess_mxfp_paired_sm100(
                A_interleaved,
                lp_buffers_.A_real(), lp_buffers_.A_imag(),
                lp_buffers_.sf_A_real(), lp_buffers_.sf_A_imag(),
                N * batch_count, K, precision, stream);
        } else {
            preprocess_mxfp_paired_sm100(
                A_real, A_imag,
                lp_buffers_.A_real(), lp_buffers_.A_imag(),
                lp_buffers_.sf_A_real(), lp_buffers_.sf_A_imag(),
                N * batch_count, K, precision, stream);
        }

        // Scratch buffers for full N×N × batch_count intermediate results
        ensure_herk_real_temp(total_output, stream);
        ensure_herk_temp(total_output, stream);

        // Pre-allocate GEMM workspaces (one per stream) to eliminate
        // cudaMallocAsync/cudaFreeAsync per sub-GEMM call.
        size_t gemm_ws_size = compute_workspace_size_fp16(N, N, K, batch_count, precision);
        void* gemm_ws_a = nullptr;
        void* gemm_ws_b = nullptr;
        if (gemm_ws_size > 0) {
            CUDA_CHECK(cudaMallocAsync(&gemm_ws_a, gemm_ws_size, stream));
            CUDA_CHECK(cudaMallocAsync(&gemm_ws_b, gemm_ws_size, stream));
        }

        CUDA_CHECK(cudaEventRecord(preprocess_done_, stream));
        CUDA_CHECK(cudaStreamWaitEvent(stream_a_, preprocess_done_));
        CUDA_CHECK(cudaStreamWaitEvent(stream_b_, preprocess_done_));

        cutlass::Status status;
        auto* scratch_Re = reinterpret_cast<cutlass::half_t*>(herk_real_temp_);

        void* Ar_data = lp_buffers_.A_real();
        void* Ai_data = lp_buffers_.A_imag();
        void* sf_Ar   = lp_buffers_.sf_A_real();
        void* sf_Ai   = lp_buffers_.sf_A_imag();

        // Skip triangle at very small K — per-slab launch overhead exceeds FLOPs savings
        if (strategy == HerkStrategy::TriangleAware && K <= 64) {
            strategy = HerkStrategy::Baseline;
        }

        if (strategy == HerkStrategy::TriangleAware) {
            // ---- TRIANGLE-AWARE: per-batch slab loop for Re(C) ----
            //
            // Can't batch the per-slab calls because run_real_gemm_dispatch()
            // computes batch stride from (M_block, K), but our data has stride
            // (N, K). So we loop per-batch for Re. Im is always a full N×N GEMM
            // so it CAN be batched (stride matches).

            constexpr int kTileM = kFP8TileM;
            constexpr int kTileN = kFP8TileN;
            int sm_count = hw_sm_count_;

            // --- Determine target_slabs ---
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
                // Cap slabs at small K — reduce launch overhead when GEMMs are tiny
                int max_slabs_for_K = std::max(2, K / 64);
                target_slabs = std::min(target_slabs, max_slabs_for_K);
            }

            // --- Determine min_slab_height ---
            int min_slab_height;
            if (tri.min_slab_height > 0) {
                min_slab_height = tri.min_slab_height;
            } else {
                double ratio = static_cast<double>(kTileM) / kTileN;
                int tiles_needed = static_cast<int>(std::ceil(
                    std::sqrt(static_cast<double>(sm_count) / ratio)));
                min_slab_height = kTileM * std::max(tiles_needed, 1);
            }

            // --- Compute slab boundaries ---
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
                    if (boundaries[i] > deduped.back()) {
                        deduped.push_back(boundaries[i]);
                    }
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

            // SF stride per 128-row tile (SfAtom layout)
            int64_t k_blocks = (K + 31) / 32;
            int64_t padded_k_blocks = ((k_blocks + 3) / 4) * 4;
            int64_t sf_tile_row_stride = (padded_k_blocks / 4) * 512;

            // Strides per batch element in preprocessed data
            int64_t data_stride_per_batch = bytes_for_elements(per_A, precision);
            int64_t sf_stride_per_batch = sf_buffer_bytes(N, K);

            // --- Per-batch slab loop for Re(C) on stream_a_ ---
            for (int b = 0; b < batch_count; ++b) {
                const char* Ar_b = static_cast<const char*>(Ar_data) + b * data_stride_per_batch;
                const char* Ai_b = static_cast<const char*>(Ai_data) + b * data_stride_per_batch;
                const char* sf_Ar_b = static_cast<const char*>(sf_Ar) + b * sf_stride_per_batch;
                const char* sf_Ai_b = static_cast<const char*>(sf_Ai) + b * sf_stride_per_batch;
                auto* scratch_Re_b = scratch_Re + b * output_per_batch;

                for (int i = 0; i < num_slabs; ++i) {
                    int row_start = boundaries[i];
                    int M_block = boundaries[i + 1] - row_start;
                    int N_block = boundaries[i + 1];

                    const char* A_slab_r = Ar_b
                        + bytes_for_elements(static_cast<int64_t>(row_start) * K, precision);
                    const char* A_slab_i = Ai_b
                        + bytes_for_elements(static_cast<int64_t>(row_start) * K, precision);
                    const char* sf_A_slab_r = sf_Ar_b + (row_start / 128) * sf_tile_row_stride;
                    const char* sf_A_slab_i = sf_Ai_b + (row_start / 128) * sf_tile_row_stride;
                    auto* C_slab = scratch_Re_b + row_start * N;

                    status = run_real_gemm_dispatch(
                        A_slab_r, Ar_b, sf_A_slab_r, sf_Ar_b,
                        C_slab,
                        M_block, N_block, K, alpha, 0.0f, stream_a_, precision, config,
                        /*batch_count=*/1, /*ld_C=*/N,
                        gemm_ws_a, gemm_ws_size);
                    if (status != cutlass::Status::kSuccess) goto cleanup_ws;

                    status = run_real_gemm_dispatch(
                        A_slab_i, Ai_b, sf_A_slab_i, sf_Ai_b,
                        C_slab,
                        M_block, N_block, K, alpha, 1.0f, stream_a_, precision, config,
                        /*batch_count=*/1, /*ld_C=*/N,
                        gemm_ws_a, gemm_ws_size);
                    if (status != cutlass::Status::kSuccess) goto cleanup_ws;
                }
            }
        } else {
            // ---- BASELINE: full N×N sub-GEMMs with batch_count ----
            status = run_real_gemm_dispatch(
                Ar_data, Ar_data, sf_Ar, sf_Ar,
                scratch_Re,
                N, N, K, alpha, 0.0f, stream_a_, precision, config, batch_count,
                /*ld_C=*/0, gemm_ws_a, gemm_ws_size);
            if (status != cutlass::Status::kSuccess) goto cleanup_ws;

            status = run_real_gemm_dispatch(
                Ai_data, Ai_data, sf_Ai, sf_Ai,
                scratch_Re,
                N, N, K, alpha, 1.0f, stream_a_, precision, config, batch_count,
                /*ld_C=*/0, gemm_ws_a, gemm_ws_size);
            if (status != cutlass::Status::kSuccess) goto cleanup_ws;
        }

        // Stream B: Im(C) = Ai × Ar^T — always full N×N, batched (stride matches)
        status = run_real_gemm_dispatch(
            Ai_data, Ar_data, sf_Ai, sf_Ar,
            reinterpret_cast<cutlass::half_t*>(herk_imag_temp_),
            N, N, K, 1.0f, 0.0f, stream_b_, precision, config, batch_count,
            /*ld_C=*/0, gemm_ws_b, gemm_ws_size);
        if (status != cutlass::Status::kSuccess) goto cleanup_ws;

        // Sync both streams to caller, then fused pack on caller stream
        CUDA_CHECK(cudaEventRecord(stream_a_done_, stream_a_));
        CUDA_CHECK(cudaEventRecord(stream_b_done_, stream_b_));
        CUDA_CHECK(cudaStreamWaitEvent(stream, stream_a_done_));
        CUDA_CHECK(cudaStreamWaitEvent(stream, stream_b_done_));

        // Fused pack Re(C) + antisymmetrize+pack Im(C) on caller stream (batched)
        if (C_interleaved) {
            // Fused pack + interleave: write directly to interleaved output,
            // skipping Cr_packed/Ci_packed intermediates and the separate interleave pass
            pack_antisymmetrize_interleave(
                herk_real_temp_, herk_imag_temp_,
                C_interleaved,
                N, alpha, beta, fill, stream, batch_count,
                (beta != 0.0f) ? C_interleaved : nullptr);
        } else if (beta != 0.0f) {
            pack_antisymmetrize_triangle(
                herk_real_temp_, herk_imag_temp_,
                C_real_packed, C_imag_packed,
                N, alpha, beta, fill, stream, batch_count,
                C_real_packed, C_imag_packed);
        } else {
            pack_antisymmetrize_triangle(
                herk_real_temp_, herk_imag_temp_,
                C_real_packed, C_imag_packed,
                N, alpha, 0.0f, fill, stream, batch_count);
        }

    cleanup_ws:
        if (gemm_ws_a) cudaFreeAsync(gemm_ws_a, stream);
        if (gemm_ws_b) cudaFreeAsync(gemm_ws_b, stream);
        return status;
    }

    // ---------------------------------------------------------------
    // Block-scaled HERK: 3-GEMM path for FP6/FP4, unpacked output (NoTrans only)
    // ---------------------------------------------------------------
    cutlass::Status herk_planar_blockscaled(
        const __half* A_real, const __half* A_imag,
        __half* C_real, __half* C_imag,
        int N, int K,
        float alpha, float beta,
        FillMode fill,
        ComputePrecision precision,
        cudaStream_t stream,
        GemmConfig config,
        HerkStrategy /*strategy*/ = HerkStrategy::Baseline,
        const TriangleConfig& /*tri*/ = {})
    {
        ensure_streams();

        int64_t size_A = static_cast<int64_t>(N) * K;
        int64_t output_size = static_cast<int64_t>(N) * N;

        // Allocate sub-byte data buffers — A only (B reuses A's buffers)
        int64_t bytes_A = bytes_for_elements(size_A, precision);
        lp_buffers_.ensure_capacity(bytes_A, 0, 0, stream);

        // Allocate scale factor buffers — A only (B reuses A's)
        int64_t sf_A_bytes = sf_buffer_bytes(N, K);
        lp_buffers_.ensure_sf_capacity(sf_A_bytes, 0, 0, stream);

        // Preprocess A once: fused SF + scale + cast for Ar and Ai
        preprocess_mxfp_paired_sm100(
            A_real, A_imag,
            lp_buffers_.A_real(), lp_buffers_.A_imag(),
            lp_buffers_.sf_A_real(), lp_buffers_.sf_A_imag(),
            N, K, precision, stream);

        // Temp buffer for Im(C) before anti-symmetrization
        ensure_herk_temp(output_size, stream);

        CUDA_CHECK(cudaEventRecord(preprocess_done_, stream));
        CUDA_CHECK(cudaStreamWaitEvent(stream_a_, preprocess_done_));
        CUDA_CHECK(cudaStreamWaitEvent(stream_b_, preprocess_done_));

        cutlass::Status status;

        // Self-product: pass same preprocessed A-side buffers for both operands.
        void* Ar_data = lp_buffers_.A_real();
        void* Ai_data = lp_buffers_.A_imag();
        void* sf_Ar   = lp_buffers_.sf_A_real();
        void* sf_Ai   = lp_buffers_.sf_A_imag();

        // Stream A: Re(C) = alpha * (Ar · Ar^T + Ai · Ai^T) + beta * Re(C)
        status = run_real_gemm_dispatch(
            Ar_data, Ar_data, sf_Ar, sf_Ar,
            reinterpret_cast<cutlass::half_t*>(C_real),
            N, N, K, alpha, beta, stream_a_, precision, config);
        if (status != cutlass::Status::kSuccess) return status;

        status = run_real_gemm_dispatch(
            Ai_data, Ai_data, sf_Ai, sf_Ai,
            reinterpret_cast<cutlass::half_t*>(C_real),
            N, N, K, alpha, 1.0f, stream_a_, precision, config);
        if (status != cutlass::Status::kSuccess) return status;

        // Stream B: temp = Ai · Ar^T  (alpha=1, beta=0)
        auto* temp_ptr = reinterpret_cast<cutlass::half_t*>(herk_imag_temp_);
        status = run_real_gemm_dispatch(
            Ai_data, Ar_data, sf_Ai, sf_Ar,
            temp_ptr,
            N, N, K, 1.0f, 0.0f, stream_b_, precision, config);
        if (status != cutlass::Status::kSuccess) return status;

        // Stream B: Im(C) = alpha * (temp - temp^T) + beta * Im(C)
        antisymmetrize_to_triangle(
            herk_imag_temp_, C_imag, C_imag,
            N, alpha, beta, fill, stream_b_);

        CUDA_CHECK(cudaEventRecord(stream_a_done_, stream_a_));
        CUDA_CHECK(cudaEventRecord(stream_b_done_, stream_b_));
        CUDA_CHECK(cudaStreamWaitEvent(stream, stream_a_done_));
        CUDA_CHECK(cudaStreamWaitEvent(stream, stream_b_done_));

        return cutlass::Status::kSuccess;
    }

    // ---------------------------------------------------------------
    // Sub-GEMM dispatch core (FP8 inputs → FP16 output)
    // ---------------------------------------------------------------
    //
    // Executes 4 real FP8 sub-GEMMs with stream parallelism.
    // All FP8 inputs must already be cast and in the correct layout.
    // C_real / C_imag are FP16 (for β accumulation and output).
    //
    // This is the compute core shared by:
    //   - run_planar() (separate cast → run_subgemms_fp8)
    //   - run()        (F1 fused deinterleave+cast → run_subgemms_fp8)
    //
