    /// ===============================================================
    /// HERK — Hermitian Rank-K Update (BLAS Level 3)
    /// ===============================================================
    ///
    /// Computes C = α·op(A)·op(A)^H + β·C, where:
    ///   - C is Hermitian [N × N], only the triangle specified by `fill` is stored
    ///   - Diagonal of C is guaranteed real (Im(C_ii) = 0)
    ///
    /// HerkOp::NoTrans:   op(A) = A,    A is [N × K] row-major
    ///   → C[N×N] = α·A·A^H + β·C
    ///
    /// HerkOp::ConjTrans:  op(A) = A^H,  A is [K × N] row-major
    ///   → C[N×N] = α·A^H·A + β·C
    ///
    /// BLAS Correspondence:
    ///   zherk(uplo, trans, N, K, alpha, A, lda, beta, C, ldc)
    ///   where uplo → fill, trans → op
    ///
    /// IMPLEMENTATION:
    ///   Production mode (COMPLEX_FP8_HERK_FULL_MATRIX=0):
    ///     3 real sub-GEMMs + O(N²) anti-symmetrize. 25% fewer FLOPs.
    ///     Only the authoritative triangle (FillMode) is written.
    ///     Non-authoritative triangle is UNDEFINED — do not read.
    ///   Debug mode (COMPLEX_FP8_HERK_FULL_MATRIX=1):
    ///     4 real sub-GEMMs + full symmetrize. Both triangles valid.
    ///

    /// Planar complex HERK (single problem)
    cutlass::Status herk_planar(
        const __half* A_real, const __half* A_imag,     // A (see HerkOp for dimensions)
        __half* C_real, __half* C_imag,                 // C [N × N] row-major (in/out)
        int N, int K,
        float alpha = 1.0f,
        float beta  = 0.0f,
        HerkOp op = HerkOp::NoTrans,
        FillMode fill = FillMode::Lower,
        HerkStrategy strategy = HerkStrategy::Baseline,
        cudaStream_t stream = nullptr,
        const TriangleConfig& tri = {})
    {
#if COMPLEX_FP8_HERK_FULL_MATRIX
        // ---- DEBUG MODE: 4 sub-GEMMs via Gram, then symmetrize both triangles ----
        int gram_M, gram_K;
        GramMode gram_mode;
        if (op == HerkOp::NoTrans) {
            gram_M = N; gram_K = K; gram_mode = GramMode::AAH;
        } else {
            gram_M = K; gram_K = N; gram_mode = GramMode::AHA;
        }
        auto status = run_gram_planar(A_real, A_imag, C_real, C_imag,
                                       gram_M, gram_K, alpha, beta, gram_mode, stream);
        if (status != cutlass::Status::kSuccess) return status;
        enforce_hermitian_triangle(C_real, C_imag, N, fill, stream);
        return cutlass::Status::kSuccess;

#else
        // ---- PRODUCTION MODE: 3 sub-GEMMs + anti-symmetrize ----
        //
        // Saves 1 full N×N×K GEMM by exploiting:
        //   Im(C) = α·(Xi·Xr^T − Xr·Xi^T) + β·Ci_old
        //         = α·(temp − temp^T) + β·Ci_old    where temp = Xi·Xr^T
        //
        // Stream A (2 GEMMs): Re(C) = α·Xr·Xr^T + α·Xi·Xi^T + β·Cr  (full matrix)
        // Stream B (1 GEMM):  temp  = Xi·Xr^T                         (full matrix)
        //          (kernel):  Ci    = α·(temp − temp^T) + β·Ci_old    (triangle only)

        ensure_streams();

        // Skip triangle decomposition when K >> N — grouped GEMM overhead
        // exceeds the 25% FLOP savings due to heterogeneous group load imbalance.
        if (strategy == HerkStrategy::TriangleAware && K > 4 * N) {
            strategy = HerkStrategy::Baseline;
        }

        // Both NoTrans and ConjTrans map to GEMM(N, N, K) after the
        // self-product pointer trick (NoTrans) or fused transpose (ConjTrans).
        // The `K` parameter to this function is always the reduction dimension.
        //
        // CRITICAL: TMA L2 CACHE CONTENTION ON SELF-PRODUCT GEMMS
        //
        // When run_real_gemm(X, X, ...) receives the same pointer for both A and B,
        // TMA's two DMA engines fetch tiles from the same physical buffer
        // simultaneously, causing L2 cache line thrashing and ~2× slowdown.
        //
        // Measured on H100 at 8192³:
        //   Self-product (A_ptr == B_ptr): ~2.23ms per GEMM
        //   Normal       (A_ptr != B_ptr): ~1.06ms per GEMM
        //
        // Fix: allocate SEPARATE A-side and B-side FP8 buffers with identical
        // content. The cudaMemcpyAsync cost (~0.04ms for 64MB) is negligible
        // vs the ~1.1ms saved per GEMM.

        int64_t size_A = static_cast<int64_t>(N) * K;  // same for NoTrans (N*K) and ConjTrans (K*N)

        // A-side pointers (used as GEMM A operand)
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
            auto* Stacked_B = buffers_.stacked_b();
            CUDA_CHECK(cudaMemcpyAsync(Stacked_B, Stacked,
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

        // Temp buffer for Xi·Xr^T result (N×N FP16)
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
                ensure_hw_info();
                status = run_real_gemm_lower_triangle(
                    Stacked, Stacked_B, Cr, N, 2 * K, alpha, beta, stream_a_, 1, tri, hw_sm_count_);
                if (status != cutlass::Status::kSuccess) return status;
            } else {
                // Stacked-K: single GEMM with K_eff=2K computes Re(C) = Xr·Xr^T + Xi·Xi^T
                status = run_real_gemm(Stacked, Stacked_B, Cr,
                                       N, N, 2 * K, alpha, beta, stream_a_);
                if (status != cutlass::Status::kSuccess) return status;
            }

            // Stream B: temp = Xi·Xr^T (Im path, separate buffers, original K)
            auto* temp_ptr = reinterpret_cast<cutlass::half_t*>(herk_imag_temp_);
            status = run_real_gemm(Xi_sep, Xr_sep, temp_ptr,
                                   N, N, K, 1.0f, 0.0f, stream_b_);
            if (status != cutlass::Status::kSuccess) return status;
        } else {
            if (strategy == HerkStrategy::TriangleAware) {
                ensure_hw_info();
                status = run_real_gemm_lower_triangle(
                    Xr_A, Xr_B, Cr, N, K, alpha, beta, stream_a_, 1, tri, hw_sm_count_);
                if (status != cutlass::Status::kSuccess) return status;
                status = run_real_gemm_lower_triangle(
                    Xi_A, Xi_B, Cr, N, K, alpha, 1.0f, stream_a_, 1, tri, hw_sm_count_);
                if (status != cutlass::Status::kSuccess) return status;
            } else {
                status = run_real_gemm(Xr_A, Xr_B, Cr,
                                       N, N, K, alpha, beta, stream_a_);
                if (status != cutlass::Status::kSuccess) return status;
                status = run_real_gemm(Xi_A, Xi_B, Cr,
                                       N, N, K, alpha, 1.0f, stream_a_);
                if (status != cutlass::Status::kSuccess) return status;
            }

            // Stream B: temp = Xi·Xr^T (full matrix — anti-symmetrize reads both triangles)
            auto* temp_ptr = reinterpret_cast<cutlass::half_t*>(herk_imag_temp_);
            status = run_real_gemm(Xi_A, Xr_B, temp_ptr,
                                   N, N, K, 1.0f, 0.0f, stream_b_);
            if (status != cutlass::Status::kSuccess) return status;
        }
        if (status != cutlass::Status::kSuccess) return status;

        //   Step 2: Anti-symmetrize into authoritative triangle only
        //   Ci[i,j] = α·(temp[i,j] − temp[j,i]) + β·Ci_old[i,j]  for (i,j) in triangle
        //   Diagonal = 0
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
    /// Like herk_planar() but outputs directly to packed N*(N+1)/2 format,
    /// eliminating the user-side N×N → packed packing step.
    ///
    /// F3 fusion for Im(C): the anti-symmetrize kernel writes directly to packed
    /// output, eliminating the full N×N intermediate Im(C) buffer (38% BW savings).
    /// Re(C) is packed via a library-side pack kernel after sub-GEMM output.
    ///
    /// Internal scratch: 2× N×N FP16 buffers (one for Re sub-GEMM output, one for
    /// Im temp). These are allocated lazily and cached across calls.
    cutlass::Status herk_planar_packed(
        const __half* A_real, const __half* A_imag,
        __half* C_real_packed, __half* C_imag_packed,   // Packed N*(N+1)/2 output
        int N, int K,
        float alpha = 1.0f,
        float beta  = 0.0f,
        HerkOp op = HerkOp::NoTrans,
        FillMode fill = FillMode::Lower,
        HerkStrategy strategy = HerkStrategy::Baseline,
        cudaStream_t stream = nullptr,
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
                                       gram_M, gram_K, alpha, beta, gram_mode, stream);
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
        //
        // Stream A: Re(C) sub-GEMMs → scratch_Re[N×N] → pack → C_real_packed
        // Stream B: Im temp GEMM → scratch_Im[N×N] → F3 antisymmetrize+pack → C_imag_packed

        ensure_streams();
        ensure_hw_info();

        // Skip triangle decomposition when K >> N — grouped GEMM overhead
        // exceeds the 25% FLOP savings due to heterogeneous group load imbalance.
        if (strategy == HerkStrategy::TriangleAware && K > 4 * N) {
            strategy = HerkStrategy::Baseline;
        }

        // ---- CUDA Graph path: capture entire HERK as graph for replay ----
        if (use_herk_graph_ && op == HerkOp::NoTrans && strategy == HerkStrategy::Baseline) {
            auto* entry = find_herk_graph(N, K, 1, alpha, beta, A_real, A_imag,
                                           C_real_packed, C_imag_packed);
            if (entry) {
                CUDA_CHECK(cudaGraphLaunch(entry->exec, stream));
                return cutlass::Status::kSuccess;
            }

            int64_t size_A = static_cast<int64_t>(N) * K;
            int64_t output_size = static_cast<int64_t>(N) * N;
            int64_t stacked_size = static_cast<int64_t>(N) * K * 2;
            buffers_.ensure_stacked_capacity(stacked_size, size_A, stream);
            ensure_herk_real_temp(output_size, stream);
            ensure_herk_temp(output_size, stream);
            ensure_herk_gemm_workspace(N, K, 1);

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

            cast_fp16_to_fp8_stacked_and_separate(A_real, A_imag,
                reinterpret_cast<__nv_fp8_e4m3*>(Stacked_g),
                reinterpret_cast<__nv_fp8_e4m3*>(Xi_sep_g),
                reinterpret_cast<__nv_fp8_e4m3*>(Xr_sep_g), N, K, cap);

            // Duplicate stacked buffer for TMA L2 fix
            CUDA_CHECK(cudaMemcpyAsync(Stacked_b_g, Stacked_g,
                stacked_size * sizeof(cutlass::float_e4m3_t),
                cudaMemcpyDeviceToDevice, cap));

            CUDA_CHECK(cudaEventRecord(preprocess_done_, cap));
            CUDA_CHECK(cudaStreamWaitEvent(stream_a_, preprocess_done_));
            CUDA_CHECK(cudaStreamWaitEvent(stream_b_, preprocess_done_));

            auto status = run_real_gemm(Stacked_g, Stacked_b_g, scratch_Re_g,
                                         N, N, 2 * K, alpha, 0.0f, stream_a_, 1,
                                         herk_gemm_workspace_a_, herk_gemm_workspace_size_);
            if (status != cutlass::Status::kSuccess) {
                cudaGraph_t abandoned; cudaStreamEndCapture(cap, &abandoned);
                if (abandoned) cudaGraphDestroy(abandoned);
                return status;
            }

            status = run_real_gemm(Xi_sep_g, Xr_sep_g, temp_ptr_g,
                                    N, N, K, 1.0f, 0.0f, stream_b_, 1,
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
                    N, alpha, beta, fill, cap, 1,
                    C_real_packed, C_imag_packed);
            } else {
                pack_antisymmetrize_triangle(
                    herk_real_temp_, herk_imag_temp_,
                    C_real_packed, C_imag_packed,
                    N, alpha, 0.0f, fill, cap);
            }

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
            // F2: Fused quad cast + transpose + duplicate — AHA path
            buffers_.ensure_capacity(size_A, 0, size_A, stream);
            cast_fp16_to_fp8_e4m3_transposed_quad(A_real, A_imag,
                buffers_.BT_real(), buffers_.A_real(),
                buffers_.BT_imag(), buffers_.A_imag(), K, N, stream);
            Xr_A = buffers_.BT_real();  Xi_A = buffers_.BT_imag();
            Xr_B = buffers_.A_real();   Xi_B = buffers_.A_imag();
        }

        // Scratch buffers: Re on stream_a, Im on stream_b (concurrent)
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
                status = run_real_gemm_lower_triangle(
                    Stacked, Stacked_B, scratch_Re, N, 2 * K, alpha, 0.0f, stream_a_, 1, tri, hw_sm_count_);
                if (status != cutlass::Status::kSuccess) return status;
            } else {
                // Stacked-K: single GEMM with K_eff=2K computes Re(C)
                status = run_real_gemm(Stacked, Stacked_B, scratch_Re,
                                       N, N, 2 * K, alpha, 0.0f, stream_a_);
                if (status != cutlass::Status::kSuccess) return status;
            }

            // Stream B: temp = Xi·Xr^T (Im path, separate buffers, original K)
            auto* temp_ptr = reinterpret_cast<cutlass::half_t*>(herk_imag_temp_);
            status = run_real_gemm(Xi_sep, Xr_sep, temp_ptr,
                                   N, N, K, 1.0f, 0.0f, stream_b_);
            if (status != cutlass::Status::kSuccess) return status;
        } else {
            if (strategy == HerkStrategy::TriangleAware) {
                status = run_real_gemm_lower_triangle(
                    Xr_A, Xr_B, scratch_Re, N, K, alpha, 0.0f, stream_a_, 1, tri, hw_sm_count_);
                if (status != cutlass::Status::kSuccess) return status;
                status = run_real_gemm_lower_triangle(
                    Xi_A, Xi_B, scratch_Re, N, K, alpha, 1.0f, stream_a_, 1, tri, hw_sm_count_);
                if (status != cutlass::Status::kSuccess) return status;
            } else {
                status = run_real_gemm(Xr_A, Xr_B, scratch_Re,
                                       N, N, K, alpha, 0.0f, stream_a_);
                if (status != cutlass::Status::kSuccess) return status;
                status = run_real_gemm(Xi_A, Xi_B, scratch_Re,
                                       N, N, K, alpha, 1.0f, stream_a_);
                if (status != cutlass::Status::kSuccess) return status;
            }

            // Stream B: temp = Xi·Xr^T
            auto* temp_ptr = reinterpret_cast<cutlass::half_t*>(herk_imag_temp_);
            status = run_real_gemm(Xi_A, Xr_B, temp_ptr,
                                   N, N, K, 1.0f, 0.0f, stream_b_);
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
        const TriangleConfig& tri = {})
    {
#if COMPLEX_FP8_HERK_FULL_MATRIX
        // ---- DEBUG MODE: via batched Gram → full N×N → pack ----
        int64_t full_size = static_cast<int64_t>(N) * N * batch_count;
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
        auto status = run_gram_planar_batched(A_real, A_imag, d_Cr, d_Ci,
                                               gram_M, gram_K, batch_count,
                                               alpha, beta, gram_mode, stream);
        if (status != cutlass::Status::kSuccess) {
            cudaFreeAsync(d_Cr, stream); cudaFreeAsync(d_Ci, stream);
            return status;
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
                                         herk_gemm_workspace_a_, herk_gemm_workspace_size_);
            if (status != cutlass::Status::kSuccess) {
                cudaGraph_t abandoned; cudaStreamEndCapture(cap, &abandoned);
                if (abandoned) cudaGraphDestroy(abandoned);
                return status;
            }

            status = run_real_gemm(Xi_sep_g, Xr_sep_g, temp_ptr_g,
                                    N, N, K, 1.0f, 0.0f, stream_b_, batch_count,
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
                status = run_real_gemm_lower_triangle(
                    Stacked, Stacked_B, scratch_Re, N, 2 * K, alpha, 0.0f, stream_a_, batch_count, tri, hw_sm_count_);
                if (status != cutlass::Status::kSuccess) return status;
            } else {
                status = run_real_gemm(Stacked, Stacked_B, scratch_Re,
                                       N, N, 2 * K, alpha, 0.0f, stream_a_, batch_count,
                                       herk_gemm_workspace_a_, herk_gemm_workspace_size_);
                if (status != cutlass::Status::kSuccess) return status;
            }

            // Stream B: temp = Xi·Xr^T (Im path, separate buffers, original K, batched)
            auto* temp_ptr = reinterpret_cast<cutlass::half_t*>(herk_imag_temp_);
            status = run_real_gemm(Xi_sep, Xr_sep, temp_ptr,
                                   N, N, K, 1.0f, 0.0f, stream_b_, batch_count,
                                   herk_gemm_workspace_b_, herk_gemm_workspace_size_);
            if (status != cutlass::Status::kSuccess) return status;
        } else {
            // ConjTrans: original 2-GEMM path (batched)
            if (strategy == HerkStrategy::TriangleAware) {
                status = run_real_gemm_lower_triangle(
                    Xr_A, Xr_B, scratch_Re, N, K, alpha, 0.0f, stream_a_, batch_count, tri, hw_sm_count_);
                if (status != cutlass::Status::kSuccess) return status;
                status = run_real_gemm_lower_triangle(
                    Xi_A, Xi_B, scratch_Re, N, K, alpha, 1.0f, stream_a_, batch_count, tri, hw_sm_count_);
                if (status != cutlass::Status::kSuccess) return status;
            } else {
                status = run_real_gemm(Xr_A, Xr_B, scratch_Re, N, N, K, alpha, 0.0f, stream_a_, batch_count,
                                       herk_gemm_workspace_a_, herk_gemm_workspace_size_);
                if (status != cutlass::Status::kSuccess) return status;
                status = run_real_gemm(Xi_A, Xi_B, scratch_Re, N, N, K, alpha, 1.0f, stream_a_, batch_count,
                                       herk_gemm_workspace_a_, herk_gemm_workspace_size_);
                if (status != cutlass::Status::kSuccess) return status;
            }

            // Stream B: temp = Xi·Xr^T (batched, always full N×N)
            auto* temp_ptr = reinterpret_cast<cutlass::half_t*>(herk_imag_temp_);
            status = run_real_gemm(Xi_A, Xr_B, temp_ptr, N, N, K, 1.0f, 0.0f, stream_b_, batch_count,
                                   herk_gemm_workspace_b_, herk_gemm_workspace_size_);
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

    /// Interleaved complex HERK (convenience)
    ///
    /// F1 fusion: deinterleave+cast A directly from interleaved FP16 to planar FP8,
    /// eliminating intermediate FP16 planar buffers for A.
    /// C is still deinterleaved to FP16 (read-write during β-accumulation).
    cutlass::Status herk(
        __half* A, __half* C,
        int N, int K,
        float alpha = 1.0f,
        float beta  = 0.0f,
        HerkOp op = HerkOp::NoTrans,
        FillMode fill = FillMode::Lower,
        HerkStrategy strategy = HerkStrategy::Baseline,
        cudaStream_t stream = nullptr,
        const TriangleConfig& tri = {})
    {
#if COMPLEX_FP8_HERK_FULL_MATRIX
        // ---- DEBUG MODE: delegate through herk_planar (not performance-critical) ----
        int A_rows = (op == HerkOp::NoTrans) ? N : K;
        int A_cols = (op == HerkOp::NoTrans) ? K : N;
        int64_t size_A = static_cast<int64_t>(A_rows) * A_cols;
        int64_t size_C = static_cast<int64_t>(N) * N;

        __half *Ar, *Ai, *Cr, *Ci;
        CUDA_CHECK(cudaMallocAsync(&Ar, size_A * sizeof(__half), stream));
        CUDA_CHECK(cudaMallocAsync(&Ai, size_A * sizeof(__half), stream));
        CUDA_CHECK(cudaMallocAsync(&Cr, size_C * sizeof(__half), stream));
        CUDA_CHECK(cudaMallocAsync(&Ci, size_C * sizeof(__half), stream));

        deinterleave_complex(A, Ar, Ai, size_A, stream);
        deinterleave_complex(C, Cr, Ci, size_C, stream);

        auto status = herk_planar(Ar, Ai, Cr, Ci, N, K, alpha, beta, op, fill, strategy, stream);

        if (status == cutlass::Status::kSuccess) {
            interleave_complex(Cr, Ci, C, size_C, stream);
        }

        cudaFreeAsync(Ar, stream); cudaFreeAsync(Ai, stream);
        cudaFreeAsync(Cr, stream); cudaFreeAsync(Ci, stream);

        return status;

#else
        // ---- PRODUCTION MODE: F1 fused deinterleave+cast + 3 sub-GEMMs ----
        ensure_streams();

        int64_t size_A = static_cast<int64_t>(N) * K;
        int64_t size_C = static_cast<int64_t>(N) * N;

        // F2: Fused deinterleave + cast + duplicate for A (interleaved self-product)
        cutlass::float_e4m3_t *Xr_A, *Xi_A, *Xr_B, *Xi_B;

        if (op == HerkOp::NoTrans) {
            buffers_.ensure_capacity(size_A, size_A, 0, stream);
            deinterleave_cast_fp16_to_fp8_dual(
                A,
                reinterpret_cast<__nv_fp8_e4m3*>(buffers_.A_real()),
                reinterpret_cast<__nv_fp8_e4m3*>(buffers_.A_imag()),
                reinterpret_cast<__nv_fp8_e4m3*>(buffers_.B_real()),
                reinterpret_cast<__nv_fp8_e4m3*>(buffers_.B_imag()),
                size_A, stream);
            Xr_A = buffers_.A_real();  Xi_A = buffers_.A_imag();
            Xr_B = buffers_.B_real();  Xi_B = buffers_.B_imag();
        } else {
            buffers_.ensure_capacity(size_A, 0, size_A, stream);
            deinterleave_cast_fp16_to_fp8_transposed_dual(
                A, buffers_.BT_real(), buffers_.BT_imag(),
                buffers_.A_real(), buffers_.A_imag(),
                K, N, stream);
            Xr_A = buffers_.BT_real();  Xi_A = buffers_.BT_imag();
            Xr_B = buffers_.A_real();   Xi_B = buffers_.A_imag();
        }

        // C still needs FP16 planar for β-accumulation
        __half *Cr, *Ci;
        CUDA_CHECK(cudaMallocAsync(&Cr, size_C * sizeof(__half), stream));
        CUDA_CHECK(cudaMallocAsync(&Ci, size_C * sizeof(__half), stream));
        deinterleave_complex(C, Cr, Ci, size_C, stream);

        // Temp buffer for Xi·Xr^T result (N×N FP16)
        int64_t output_size = size_C;
        ensure_herk_temp(output_size, stream);

        CUDA_CHECK(cudaEventRecord(preprocess_done_, stream));
        CUDA_CHECK(cudaStreamWaitEvent(stream_a_, preprocess_done_));
        CUDA_CHECK(cudaStreamWaitEvent(stream_b_, preprocess_done_));

        cutlass::Status status;
        auto* Cr_ht = reinterpret_cast<cutlass::half_t*>(Cr);

        if (strategy == HerkStrategy::TriangleAware) {
            // Stream A: Re(C) via lower-triangle block-row decomposition
            ensure_hw_info();
            status = run_real_gemm_lower_triangle(
                Xr_A, Xr_B, Cr_ht, N, K, alpha, beta, stream_a_, 1, tri, hw_sm_count_);
            if (status != cutlass::Status::kSuccess) { cudaFreeAsync(Cr, stream); cudaFreeAsync(Ci, stream); return status; }
            status = run_real_gemm_lower_triangle(
                Xi_A, Xi_B, Cr_ht, N, K, alpha, 1.0f, stream_a_, 1, tri, hw_sm_count_);
            if (status != cutlass::Status::kSuccess) { cudaFreeAsync(Cr, stream); cudaFreeAsync(Ci, stream); return status; }
        } else {
            // Stream A: Re(C) = α·Xr·Xr^T + α·Xi·Xi^T + β·Cr
            status = run_real_gemm(Xr_A, Xr_B, Cr_ht,
                                   N, N, K, alpha, beta, stream_a_);
            if (status != cutlass::Status::kSuccess) { cudaFreeAsync(Cr, stream); cudaFreeAsync(Ci, stream); return status; }
            status = run_real_gemm(Xi_A, Xi_B, Cr_ht,
                                   N, N, K, alpha, 1.0f, stream_a_);
            if (status != cutlass::Status::kSuccess) { cudaFreeAsync(Cr, stream); cudaFreeAsync(Ci, stream); return status; }
        }

        // Stream B: temp = Xi·Xr^T
        auto* temp_ptr = reinterpret_cast<cutlass::half_t*>(herk_imag_temp_);
        status = run_real_gemm(Xi_A, Xr_B, temp_ptr,
                               N, N, K, 1.0f, 0.0f, stream_b_);
        if (status != cutlass::Status::kSuccess) { cudaFreeAsync(Cr, stream); cudaFreeAsync(Ci, stream); return status; }

        // Anti-symmetrize into authoritative triangle
        antisymmetrize_to_triangle(
            herk_imag_temp_, Ci, Ci,
            N, alpha, beta, fill, stream_b_);

        CUDA_CHECK(cudaEventRecord(stream_a_done_, stream_a_));
        CUDA_CHECK(cudaEventRecord(stream_b_done_, stream_b_));
        CUDA_CHECK(cudaStreamWaitEvent(stream, stream_a_done_));
        CUDA_CHECK(cudaStreamWaitEvent(stream, stream_b_done_));

        // Reinterleave result back to C
        interleave_complex(Cr, Ci, C, size_C, stream);

        cudaFreeAsync(Cr, stream);
        cudaFreeAsync(Ci, stream);

        return cutlass::Status::kSuccess;
#endif
    }


    /// Batched planar complex HERK
    cutlass::Status herk_planar_batched(
        const __half* A_real, const __half* A_imag,
        __half* C_real, __half* C_imag,
        int N, int K,
        int batch_count,
        float alpha = 1.0f,
        float beta  = 0.0f,
        HerkOp op = HerkOp::NoTrans,
        FillMode fill = FillMode::Lower,
        cudaStream_t stream = nullptr)
    {
#if COMPLEX_FP8_HERK_FULL_MATRIX
        // ---- DEBUG: 4 sub-GEMMs via batched Gram ----
        int gram_M, gram_K;
        GramMode gram_mode;
        if (op == HerkOp::NoTrans) {
            gram_M = N; gram_K = K; gram_mode = GramMode::AAH;
        } else {
            gram_M = K; gram_K = N; gram_mode = GramMode::AHA;
        }
        auto status = run_gram_planar_batched(A_real, A_imag, C_real, C_imag,
                                               gram_M, gram_K, batch_count,
                                               alpha, beta, gram_mode, stream);
        if (status != cutlass::Status::kSuccess) return status;
        enforce_hermitian_triangle(C_real, C_imag, N, fill, stream, batch_count);
        return cutlass::Status::kSuccess;

#else
        // ---- PRODUCTION: 3 sub-GEMMs + anti-symmetrize (batched) ----
        if (batch_count <= 0) return cutlass::Status::kSuccess;
        ensure_streams();

        int64_t per_A = static_cast<int64_t>(N) * K;  // same for both ops
        int64_t total_A = per_A * batch_count;

        cutlass::float_e4m3_t *Xr_A, *Xi_A, *Xr_B, *Xi_B;

        if (op == HerkOp::NoTrans) {
            // F2: Fused quad cast + duplicate — batched AAH
            buffers_.ensure_capacity(total_A, total_A, 0, stream);
            cast_fp16_to_fp8_e4m3_quad(A_real, A_imag,
                reinterpret_cast<__nv_fp8_e4m3*>(buffers_.A_real()),
                reinterpret_cast<__nv_fp8_e4m3*>(buffers_.B_real()),
                reinterpret_cast<__nv_fp8_e4m3*>(buffers_.A_imag()),
                reinterpret_cast<__nv_fp8_e4m3*>(buffers_.B_imag()), total_A, stream);
            Xr_A = buffers_.A_real();  Xi_A = buffers_.A_imag();
            Xr_B = buffers_.B_real();  Xi_B = buffers_.B_imag();
        } else {
            // F2: Fused quad cast + transpose + duplicate — batched AHA
            buffers_.ensure_capacity(total_A, 0, total_A, stream);
            cast_fp16_to_fp8_e4m3_transposed_quad(A_real, A_imag,
                buffers_.BT_real(), buffers_.A_real(),
                buffers_.BT_imag(), buffers_.A_imag(), K, N, stream, batch_count);
            Xr_A = buffers_.BT_real();  Xi_A = buffers_.BT_imag();
            Xr_B = buffers_.A_real();   Xi_B = buffers_.A_imag();
        }

        int64_t output_per_batch = static_cast<int64_t>(N) * N;
        int64_t total_output = output_per_batch * batch_count;
        ensure_herk_temp(total_output, stream);

        CUDA_CHECK(cudaEventRecord(preprocess_done_, stream));
        CUDA_CHECK(cudaStreamWaitEvent(stream_a_, preprocess_done_));
        CUDA_CHECK(cudaStreamWaitEvent(stream_b_, preprocess_done_));

        cutlass::Status status;
        auto* Cr = reinterpret_cast<cutlass::half_t*>(C_real);

        // Stream A: Re(C) — A_ptr != B_ptr, no TMA contention
        status = run_real_gemm(Xr_A, Xr_B, Cr, N, N, K, alpha, beta, stream_a_, batch_count);
        if (status != cutlass::Status::kSuccess) return status;
        status = run_real_gemm(Xi_A, Xi_B, Cr, N, N, K, alpha, 1.0f, stream_a_, batch_count);
        if (status != cutlass::Status::kSuccess) return status;

        // Stream B: temp = Xi·Xr^T (Xi_A and Xr_B already different allocations)
        auto* temp_ptr = reinterpret_cast<cutlass::half_t*>(herk_imag_temp_);
        status = run_real_gemm(Xi_A, Xr_B, temp_ptr, N, N, K, 1.0f, 0.0f, stream_b_, batch_count);
        if (status != cutlass::Status::kSuccess) return status;

        antisymmetrize_to_triangle(
            herk_imag_temp_, C_imag, C_imag,
            N, alpha, beta, fill, stream_b_, batch_count);

        CUDA_CHECK(cudaEventRecord(stream_a_done_, stream_a_));
        CUDA_CHECK(cudaEventRecord(stream_b_done_, stream_b_));
        CUDA_CHECK(cudaStreamWaitEvent(stream, stream_a_done_));
        CUDA_CHECK(cudaStreamWaitEvent(stream, stream_b_done_));

        return cutlass::Status::kSuccess;
#endif
    }


    /// Batched interleaved complex HERK (convenience)
    ///
    /// F1 fusion: deinterleave+cast A directly from interleaved FP16 to planar FP8,
    /// eliminating intermediate FP16 planar buffers for A.
    cutlass::Status herk_batched(
        __half* A, __half* C,
        int N, int K,
        int batch_count,
        float alpha = 1.0f,
        float beta  = 0.0f,
        HerkOp op = HerkOp::NoTrans,
        FillMode fill = FillMode::Lower,
        cudaStream_t stream = nullptr)
    {
#if COMPLEX_FP8_HERK_FULL_MATRIX
        // ---- DEBUG MODE: delegate through herk_planar_batched ----
        int A_rows = (op == HerkOp::NoTrans) ? N : K;
        int A_cols = (op == HerkOp::NoTrans) ? K : N;
        int64_t size_A = static_cast<int64_t>(A_rows) * A_cols * batch_count;
        int64_t size_C = static_cast<int64_t>(N) * N * batch_count;

        __half *Ar, *Ai, *Cr, *Ci;
        CUDA_CHECK(cudaMallocAsync(&Ar, size_A * sizeof(__half), stream));
        CUDA_CHECK(cudaMallocAsync(&Ai, size_A * sizeof(__half), stream));
        CUDA_CHECK(cudaMallocAsync(&Cr, size_C * sizeof(__half), stream));
        CUDA_CHECK(cudaMallocAsync(&Ci, size_C * sizeof(__half), stream));

        deinterleave_complex(A, Ar, Ai, size_A, stream);
        deinterleave_complex(C, Cr, Ci, size_C, stream);

        auto status = herk_planar_batched(Ar, Ai, Cr, Ci, N, K, batch_count,
                                           alpha, beta, op, fill, stream);

        if (status == cutlass::Status::kSuccess) {
            interleave_complex(Cr, Ci, C, size_C, stream);
        }

        cudaFreeAsync(Ar, stream); cudaFreeAsync(Ai, stream);
        cudaFreeAsync(Cr, stream); cudaFreeAsync(Ci, stream);

        return status;

#else
        // ---- PRODUCTION MODE: F1 fused deinterleave+cast + 3 batched sub-GEMMs ----
        if (batch_count <= 0) return cutlass::Status::kSuccess;
        ensure_streams();

        int64_t per_A = static_cast<int64_t>(N) * K;
        int64_t total_A = per_A * batch_count;
        int64_t size_C = static_cast<int64_t>(N) * N * batch_count;

        // F2: Fused deinterleave + cast + duplicate for A (batched interleaved self-product)
        cutlass::float_e4m3_t *Xr_A, *Xi_A, *Xr_B, *Xi_B;

        if (op == HerkOp::NoTrans) {
            buffers_.ensure_capacity(total_A, total_A, 0, stream);
            deinterleave_cast_fp16_to_fp8_dual(
                A,
                reinterpret_cast<__nv_fp8_e4m3*>(buffers_.A_real()),
                reinterpret_cast<__nv_fp8_e4m3*>(buffers_.A_imag()),
                reinterpret_cast<__nv_fp8_e4m3*>(buffers_.B_real()),
                reinterpret_cast<__nv_fp8_e4m3*>(buffers_.B_imag()),
                total_A, stream);
            Xr_A = buffers_.A_real();  Xi_A = buffers_.A_imag();
            Xr_B = buffers_.B_real();  Xi_B = buffers_.B_imag();
        } else {
            buffers_.ensure_capacity(total_A, 0, total_A, stream);
            deinterleave_cast_fp16_to_fp8_transposed_dual(
                A, buffers_.BT_real(), buffers_.BT_imag(),
                buffers_.A_real(), buffers_.A_imag(),
                K, N, stream, batch_count);
            Xr_A = buffers_.BT_real();  Xi_A = buffers_.BT_imag();
            Xr_B = buffers_.A_real();   Xi_B = buffers_.A_imag();
        }

        // C still needs FP16 planar for β-accumulation
        __half *Cr, *Ci;
        CUDA_CHECK(cudaMallocAsync(&Cr, size_C * sizeof(__half), stream));
        CUDA_CHECK(cudaMallocAsync(&Ci, size_C * sizeof(__half), stream));
        deinterleave_complex(C, Cr, Ci, size_C, stream);

        int64_t output_per_batch = static_cast<int64_t>(N) * N;
        int64_t total_output = output_per_batch * batch_count;
        ensure_herk_temp(total_output, stream);

        CUDA_CHECK(cudaEventRecord(preprocess_done_, stream));
        CUDA_CHECK(cudaStreamWaitEvent(stream_a_, preprocess_done_));
        CUDA_CHECK(cudaStreamWaitEvent(stream_b_, preprocess_done_));

        cutlass::Status status;
        auto* Cr_ht = reinterpret_cast<cutlass::half_t*>(Cr);

        // Stream A: Re(C) = α·Xr·Xr^T + α·Xi·Xi^T + β·Cr
        status = run_real_gemm(Xr_A, Xr_B, Cr_ht, N, N, K, alpha, beta, stream_a_, batch_count);
        if (status != cutlass::Status::kSuccess) { cudaFreeAsync(Cr, stream); cudaFreeAsync(Ci, stream); return status; }
        status = run_real_gemm(Xi_A, Xi_B, Cr_ht, N, N, K, alpha, 1.0f, stream_a_, batch_count);
        if (status != cutlass::Status::kSuccess) { cudaFreeAsync(Cr, stream); cudaFreeAsync(Ci, stream); return status; }

        // Stream B: temp = Xi·Xr^T
        auto* temp_ptr = reinterpret_cast<cutlass::half_t*>(herk_imag_temp_);
        status = run_real_gemm(Xi_A, Xr_B, temp_ptr, N, N, K, 1.0f, 0.0f, stream_b_, batch_count);
        if (status != cutlass::Status::kSuccess) { cudaFreeAsync(Cr, stream); cudaFreeAsync(Ci, stream); return status; }

        antisymmetrize_to_triangle(
            herk_imag_temp_, Ci, Ci,
            N, alpha, beta, fill, stream_b_, batch_count);

        CUDA_CHECK(cudaEventRecord(stream_a_done_, stream_a_));
        CUDA_CHECK(cudaEventRecord(stream_b_done_, stream_b_));
        CUDA_CHECK(cudaStreamWaitEvent(stream, stream_a_done_));
        CUDA_CHECK(cudaStreamWaitEvent(stream, stream_b_done_));

        // Reinterleave result back to C
        interleave_complex(Cr, Ci, C, size_C, stream);

        cudaFreeAsync(Cr, stream);
        cudaFreeAsync(Ci, stream);

        return cutlass::Status::kSuccess;
#endif
    }

