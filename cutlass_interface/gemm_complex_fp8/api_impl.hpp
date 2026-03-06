    cutlass::Status run_planar(
        const __half* A_real, const __half* A_imag,     // A = Ar + i·Ai
        const __half* B_real, const __half* B_imag,     // B = Br + i·Bi
        __half* C_real, __half* C_imag,                 // C = Cr + i·Ci (in/out)
        int M, int N, int K,
        float alpha = 1.0f,
        float beta  = 0.0f,
        ComplexMode mode = ComplexMode::Standard,
        cudaStream_t stream = nullptr)
    {
        bool is_hermitian = (mode == ComplexMode::Hermitian);

        // Lazy-init internal streams and events for parallel sub-GEMM execution
        ensure_streams();

        // Step 1: Allocate FP8 scratch buffers (+ transpose buffers if Hermitian)
        int64_t size_A = static_cast<int64_t>(M) * K;
        int64_t size_B = static_cast<int64_t>(K) * N;
        buffers_.ensure_capacity(size_A, size_B, is_hermitian ? size_B : 0, stream);

        // Step 2: Preprocessing — cast FP16→FP8 (all on the caller's stream)

        // F5: Fused paired cast — Re+Im in one launch
        cast_fp16_to_fp8_e4m3_paired(
            A_real, A_imag,
            reinterpret_cast<__nv_fp8_e4m3*>(buffers_.A_real()),
            reinterpret_cast<__nv_fp8_e4m3*>(buffers_.A_imag()),
            size_A, stream);

        // B: Standard mode uses plain cast; Hermitian uses FUSED cast+transpose.
        // The fused kernel reads FP16 ColMajor [K×N] and writes FP8 ColMajor [N×K]
        // (transposed) in a single pass — eliminating the separate transpose kernel.
        cutlass::float_e4m3_t* Br_ptr;
        cutlass::float_e4m3_t* Bi_ptr;

        if (!is_hermitian) {
            // F5: Fused paired cast — Re+Im in one launch
            cast_fp16_to_fp8_e4m3_paired(
                B_real, B_imag,
                reinterpret_cast<__nv_fp8_e4m3*>(buffers_.B_real()),
                reinterpret_cast<__nv_fp8_e4m3*>(buffers_.B_imag()),
                size_B, stream);
            Br_ptr = buffers_.B_real();
            Bi_ptr = buffers_.B_imag();
        } else {
            // Fused FP16→FP8 + transpose: B[K×N] → BT[N×K], both ColMajor
            // F5: Fused paired cast+transpose — Re+Im in one launch
            cast_fp16_to_fp8_e4m3_transposed_paired(
                B_real, B_imag,
                buffers_.BT_real(), buffers_.BT_imag(),
                K, N, stream);
            Br_ptr = buffers_.BT_real();
            Bi_ptr = buffers_.BT_imag();
        }

        // Step 3: Dispatch sub-GEMMs via shared compute core
        return run_subgemms_fp8(
            buffers_.A_real(), buffers_.A_imag(),
            Br_ptr, Bi_ptr,
            C_real, C_imag,
            M, N, K, alpha, beta, mode, stream);
    }


    /// ---------------------------------------------------------------
    /// Convenience Interface: Interleaved Complex
    /// ---------------------------------------------------------------
    ///
    /// A, B, C are interleaved complex FP16: [re0, im0, re1, im1, ...]
    ///
    /// F1 FUSION: Reads interleaved A/B and writes directly to FP8 planar
    /// buffers in one kernel per matrix, eliminating intermediate FP16 planar
    /// allocations for A and B.  C is still deinterleaved to FP16 planar for
    /// β accumulation (the sub-GEMMs write FP16 output).
    ///
    /// Pipeline (12 → 8 kernels vs unfused):
    ///   1. deinterleave_cast(A → FP8 Ar, Ai)     ← 1 fused kernel
    ///   2. deinterleave_cast(B → FP8 Br, Bi)     ← 1 fused kernel (or +transpose)
    ///   3. deinterleave(C → FP16 Cr, Ci)          ← 1 kernel (for β accumulation)
    ///   4–7. 4 × run_real_gemm                    ← 4 kernels
    ///   8. reinterleave(Cr, Ci → C)               ← 1 kernel
    ///
    cutlass::Status run(
        __half* A, __half* B, __half* C,
        int M, int N, int K,
        bool is_hermitian = false,
        float alpha = 1.0f,
        float beta  = 0.0f,
        cudaStream_t stream = nullptr)
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
            M, N, K, alpha, beta, mode, stream);

        // Reinterleave result back to C
        if (status == cutlass::Status::kSuccess) {
            interleave_complex(Cr, Ci, C, size_C, stream);
        }

        cudaFreeAsync(Cr, stream);
        cudaFreeAsync(Ci, stream);

        return status;
    }


    /// ---------------------------------------------------------------
    /// Gram Matrix: Planar Complex
    /// ---------------------------------------------------------------
    ///
    /// Computes self-product Gram matrices using a single input matrix A.
    ///
    /// A_real, A_imag: [M × K] row-major, device pointers in FP16.
    ///
    /// GramMode::AAH:
    ///   C = α·(A × A^H) + β·C,   C is [M × M] row-major
    ///   Common use: sample covariance matrix, beamforming correlation
    ///
    /// GramMode::AHA:
    ///   C = α·(A^H × A) + β·C,   C is [K × K] row-major
    ///   Common use: feature Gram matrix, normal equations (A^H·A·x = A^H·b)
    ///
    /// ---------------------------------------------------------------
    /// MEMORY LAYOUT TRICK (why A·A^H needs NO transpose):
    ///
    /// Our GEMM expects A=RowMajor [M×K] and B=ColMajor [K×N].
    /// For A·A^H, B = A^H = conj(A)^T, which has shape [K×M].
    ///
    /// B in ColMajor [K×M] means: b(k,m) is stored at ptr_B[k + m*K].
    /// A in RowMajor [M×K] means: a(m,k) is stored at ptr_A[m*K + k].
    /// But k + m*K == m*K + k, so THE SAME pointer works for both!
    ///
    /// We pass A's FP8 buffer as both the A and B operand. The conjugation
    /// (negating imaginary part) is handled by sign flips on α in the sub-GEMMs.
    ///
    /// For A^H·A, the trick works after transposing: we fused-cast-transpose A
    /// into AT [K×M] RowMajor, then pass AT as both operands with GEMM(K,K,M).
    /// ---------------------------------------------------------------
    ///
    /// Complex decomposition for X·X^H (where X is the effective input):
    ///   (Xr + i·Xi)(Xr^T − i·Xi^T) =
    ///     Re(C) = Xr·Xr^T + Xi·Xi^T     (both terms ADD — from −i·i = +1)
    ///     Im(C) = Xi·Xr^T − Xr·Xi^T     (anti-symmetric: C is Hermitian)
    ///
    cutlass::Status run_gram_planar(
        const __half* A_real, const __half* A_imag,     // A = Ar + i·Ai, [M × K]
        __half* C_real, __half* C_imag,                 // C = Cr + i·Ci (in/out)
        int M, int K,
        float alpha = 1.0f,
        float beta  = 0.0f,
        GramMode mode = GramMode::AAH,
        cudaStream_t stream = nullptr)
    {
        ensure_streams();

        int64_t size_A = static_cast<int64_t>(M) * K;

        // TMA L2 CACHE CONTENTION FIX:
        // Self-product GEMMs (Xr·Xr^T, Xi·Xi^T) need physically separate A-side
        // and B-side buffers. Cross-terms (Xi·Xr^T, Xr·Xi^T) already use different
        // buffers (real vs imag) so they're fine.
        //
        // A-side: used as GEMM A operand (RowMajor read)
        // B-side: used as GEMM B operand (ColMajor read) — separate allocation

        cutlass::float_e4m3_t *Xr_A, *Xi_A;   // A-side
        cutlass::float_e4m3_t *Xr_B, *Xi_B;   // B-side (separate physical memory)
        int gemm_M, gemm_N, gemm_K;

        if (mode == GramMode::AAH) {
            gemm_M = M; gemm_N = M; gemm_K = K;

            // Cast into A buffers, duplicate into B buffers
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

            // F2: Fused quad cast + transpose + duplicate — writes to both BT-side and A-side
            buffers_.ensure_capacity(size_A, 0, size_A, stream);

            cast_fp16_to_fp8_e4m3_transposed_quad(A_real, A_imag,
                buffers_.BT_real(), buffers_.A_real(),
                buffers_.BT_imag(), buffers_.A_imag(), K, M, stream);

            Xr_A = buffers_.BT_real();  Xi_A = buffers_.BT_imag();
            Xr_B = buffers_.A_real();   Xi_B = buffers_.A_imag();
        }

        return run_gram_subgemms_fp8(
            Xr_A, Xi_A, Xr_B, Xi_B, C_real, C_imag,
            gemm_M, gemm_N, gemm_K, alpha, beta, stream);
    }


    /// ---------------------------------------------------------------
    /// Gram Matrix: Interleaved Complex (convenience)
    /// ---------------------------------------------------------------
    ///
    /// A is interleaved complex FP16: [re0, im0, re1, im1, ...], M×K complex elements.
    /// C is interleaved complex FP16: M×M (AAH) or K×K (AHA) complex elements.
    ///
    cutlass::Status run_gram(
        __half* A, __half* C,
        int M, int K,
        GramMode mode = GramMode::AAH,
        float alpha = 1.0f,
        float beta  = 0.0f,
        cudaStream_t stream = nullptr)
    {
        ensure_streams();

        int64_t size_A = static_cast<int64_t>(M) * K;
        int C_dim = (mode == GramMode::AAH) ? M : K;
        int64_t size_C = static_cast<int64_t>(C_dim) * C_dim;

        // F1: Fused deinterleave + cast for A (interleaved FP16 → planar FP8)
        // Eliminates intermediate FP16 planar buffers for A.
        cutlass::float_e4m3_t *Xr_A, *Xi_A, *Xr_B, *Xi_B;
        int gemm_M, gemm_N, gemm_K;

        if (mode == GramMode::AAH) {
            gemm_M = M; gemm_N = M; gemm_K = K;
            buffers_.ensure_capacity(size_A, size_A, 0, stream);

            // F2: Fused deinterleave + cast + duplicate — writes to A-side and B-side
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
            gemm_M = K; gemm_N = K; gemm_K = M;
            buffers_.ensure_capacity(size_A, 0, size_A, stream);

            // F2: Fused deinterleave + cast + transpose + duplicate
            deinterleave_cast_fp16_to_fp8_transposed_dual(
                A, buffers_.BT_real(), buffers_.BT_imag(),
                buffers_.A_real(), buffers_.A_imag(),
                K, M, stream);

            Xr_A = buffers_.BT_real();  Xi_A = buffers_.BT_imag();
            Xr_B = buffers_.A_real();   Xi_B = buffers_.A_imag();
        }

        // C still needs FP16 planar for β-accumulation and sub-GEMM output
        __half *Cr, *Ci;
        CUDA_CHECK(cudaMallocAsync(&Cr, size_C * sizeof(__half), stream));
        CUDA_CHECK(cudaMallocAsync(&Ci, size_C * sizeof(__half), stream));
        deinterleave_complex(C, Cr, Ci, size_C, stream);

        auto status = run_gram_subgemms_fp8(
            Xr_A, Xi_A, Xr_B, Xi_B, Cr, Ci,
            gemm_M, gemm_N, gemm_K, alpha, beta, stream);

        if (status == cutlass::Status::kSuccess) {
            interleave_complex(Cr, Ci, C, size_C, stream);
        }

        cudaFreeAsync(Cr, stream);
        cudaFreeAsync(Ci, stream);

        return status;
    }


    /// ---------------------------------------------------------------
    /// Batched Standard/Hermitian GEMM: Planar Complex
    /// ---------------------------------------------------------------
    ///
    /// Executes batch_count independent complex GEMMs in a single CUTLASS launch.
    ///
    /// Memory layout (strided batching — all batches contiguous):
    ///   A_real[b] starts at A_real + b * M * K   (each [M × K], row-major)
    ///   B_real[b] starts at B_real + b * K * N   (each [K × N], col-major)
    ///   C_real[b] starts at C_real + b * M * N   (each [M × N], row-major)
    ///   Same for A_imag, B_imag, C_imag.
    ///
    /// WHY BATCHED > LOOPED:
    ///   A single CUTLASS kernel launch with L=batch_count distributes ALL tiles
    ///   (batch_count × ceil(M/128) × ceil(N/128)) across SMs in one scheduling pass.
    ///   For small per-batch problems (e.g. 512×512 × 64 batches), this fills the GPU
    ///   far better than 64 sequential kernel launches that each under-utilize the SMs.
    ///
    cutlass::Status run_planar_batched(
        const __half* A_real, const __half* A_imag,
        const __half* B_real, const __half* B_imag,
        __half* C_real, __half* C_imag,
        int M, int N, int K,
        int batch_count,
        float alpha = 1.0f,
        float beta  = 0.0f,
        ComplexMode mode = ComplexMode::Standard,
        cudaStream_t stream = nullptr)
    {
        if (batch_count <= 0) return cutlass::Status::kSuccess;

        bool is_hermitian = (mode == ComplexMode::Hermitian);
        ensure_streams();

        int64_t per_A = static_cast<int64_t>(M) * K;
        int64_t per_B = static_cast<int64_t>(K) * N;
        int64_t total_A = per_A * batch_count;
        int64_t total_B = per_B * batch_count;

        buffers_.ensure_capacity(total_A, total_B, is_hermitian ? total_B : 0, stream);

        // Cast entire batch contiguously (cast kernel doesn't care about matrix boundaries)
        // F5: Fused paired cast — Re+Im in one launch
        cast_fp16_to_fp8_e4m3_paired(
            A_real, A_imag,
            reinterpret_cast<__nv_fp8_e4m3*>(buffers_.A_real()),
            reinterpret_cast<__nv_fp8_e4m3*>(buffers_.A_imag()),
            total_A, stream);

        cutlass::float_e4m3_t* Br_ptr;
        cutlass::float_e4m3_t* Bi_ptr;

        if (!is_hermitian) {
            // F5: Fused paired cast — Re+Im in one launch
            cast_fp16_to_fp8_e4m3_paired(
                B_real, B_imag,
                reinterpret_cast<__nv_fp8_e4m3*>(buffers_.B_real()),
                reinterpret_cast<__nv_fp8_e4m3*>(buffers_.B_imag()),
                total_B, stream);
            Br_ptr = buffers_.B_real();
            Bi_ptr = buffers_.B_imag();
        } else {
            // Fused cast+transpose each batch: B[K×N] → BT[N×K], using blockIdx.z for batching
            // F5: Fused paired cast+transpose — Re+Im in one launch
            cast_fp16_to_fp8_e4m3_transposed_paired(
                B_real, B_imag,
                buffers_.BT_real(), buffers_.BT_imag(),
                K, N, stream, batch_count);
            Br_ptr = buffers_.BT_real();
            Bi_ptr = buffers_.BT_imag();
        }

        CUDA_CHECK(cudaEventRecord(preprocess_done_, stream));
        CUDA_CHECK(cudaStreamWaitEvent(stream_a_, preprocess_done_));
        CUDA_CHECK(cudaStreamWaitEvent(stream_b_, preprocess_done_));

        cutlass::Status status;
        auto* Cr = reinterpret_cast<cutlass::half_t*>(C_real);
        auto* Ci = reinterpret_cast<cutlass::half_t*>(C_imag);

        if (mode == ComplexMode::Standard) {
            status = run_real_gemm(buffers_.A_real(), Br_ptr, Cr, M, N, K, alpha, beta, stream_a_, batch_count);
            if (status != cutlass::Status::kSuccess) return status;
            status = run_real_gemm(buffers_.A_imag(), Bi_ptr, Cr, M, N, K, -alpha, 1.0f, stream_a_, batch_count);
            if (status != cutlass::Status::kSuccess) return status;

            status = run_real_gemm(buffers_.A_real(), Bi_ptr, Ci, M, N, K, alpha, beta, stream_b_, batch_count);
            if (status != cutlass::Status::kSuccess) return status;
            status = run_real_gemm(buffers_.A_imag(), Br_ptr, Ci, M, N, K, alpha, 1.0f, stream_b_, batch_count);
            if (status != cutlass::Status::kSuccess) return status;
        } else {
            status = run_real_gemm(buffers_.A_real(), Br_ptr, Cr, M, N, K, alpha, beta, stream_a_, batch_count);
            if (status != cutlass::Status::kSuccess) return status;
            status = run_real_gemm(buffers_.A_imag(), Bi_ptr, Cr, M, N, K, alpha, 1.0f, stream_a_, batch_count);
            if (status != cutlass::Status::kSuccess) return status;

            status = run_real_gemm(buffers_.A_imag(), Br_ptr, Ci, M, N, K, alpha, beta, stream_b_, batch_count);
            if (status != cutlass::Status::kSuccess) return status;
            status = run_real_gemm(buffers_.A_real(), Bi_ptr, Ci, M, N, K, -alpha, 1.0f, stream_b_, batch_count);
            if (status != cutlass::Status::kSuccess) return status;
        }

        CUDA_CHECK(cudaEventRecord(stream_a_done_, stream_a_));
        CUDA_CHECK(cudaEventRecord(stream_b_done_, stream_b_));
        CUDA_CHECK(cudaStreamWaitEvent(stream, stream_a_done_));
        CUDA_CHECK(cudaStreamWaitEvent(stream, stream_b_done_));

        return cutlass::Status::kSuccess;
    }


    /// ---------------------------------------------------------------
    /// Batched Planar Complex GEMM with FP32 output
    /// ---------------------------------------------------------------
    ///
    /// Same as run_planar_batched() but outputs FP32 instead of FP16.
    /// Avoids FP16 overflow for large K or high dynamic range inputs.
    ///
    cutlass::Status run_planar_batched_fp32out(
        const __half* A_real, const __half* A_imag,
        const __half* B_real, const __half* B_imag,
        float* C_real, float* C_imag,
        int M, int N, int K,
        int batch_count,
        float alpha = 1.0f,
        float beta  = 0.0f,
        ComplexMode mode = ComplexMode::Standard,
        cudaStream_t stream = nullptr)
    {
        if (batch_count <= 0) return cutlass::Status::kSuccess;

        bool is_hermitian = (mode == ComplexMode::Hermitian);
        ensure_streams();

        int64_t per_A = static_cast<int64_t>(M) * K;
        int64_t per_B = static_cast<int64_t>(K) * N;
        int64_t total_A = per_A * batch_count;
        int64_t total_B = per_B * batch_count;

        buffers_.ensure_capacity(total_A, total_B, is_hermitian ? total_B : 0, stream);

        // Cast entire batch contiguously FP16→FP8
        cast_fp16_to_fp8_e4m3_paired(
            A_real, A_imag,
            reinterpret_cast<__nv_fp8_e4m3*>(buffers_.A_real()),
            reinterpret_cast<__nv_fp8_e4m3*>(buffers_.A_imag()),
            total_A, stream);

        cutlass::float_e4m3_t* Br_ptr;
        cutlass::float_e4m3_t* Bi_ptr;

        if (!is_hermitian) {
            cast_fp16_to_fp8_e4m3_paired(
                B_real, B_imag,
                reinterpret_cast<__nv_fp8_e4m3*>(buffers_.B_real()),
                reinterpret_cast<__nv_fp8_e4m3*>(buffers_.B_imag()),
                total_B, stream);
            Br_ptr = buffers_.B_real();
            Bi_ptr = buffers_.B_imag();
        } else {
            cast_fp16_to_fp8_e4m3_transposed_paired(
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
    /// ---------------------------------------------------------------
    /// Batched Standard/Hermitian GEMM: Interleaved Complex (convenience)
    /// ---------------------------------------------------------------
    ///
    /// A, B, C are interleaved complex: [re0,im0,re1,im1,...] per batch element.
    ///   A[b] starts at A + b * 2*M*K  (2× for interleaved re/im)
    ///   B[b] starts at B + b * 2*K*N
    ///   C[b] starts at C + b * 2*M*N
    ///
    cutlass::Status run_batched(
        __half* A, __half* B, __half* C,
        int M, int N, int K,
        int batch_count,
        bool is_hermitian = false,
        float alpha = 1.0f,
        float beta  = 0.0f,
        cudaStream_t stream = nullptr)
    {
        ensure_streams();

        int64_t total_A = static_cast<int64_t>(M) * K * batch_count;
        int64_t total_B = static_cast<int64_t>(K) * N * batch_count;
        int64_t size_C  = static_cast<int64_t>(M) * N * batch_count;

        buffers_.ensure_capacity(total_A, total_B, is_hermitian ? total_B : 0, stream);

        // F1: Fused deinterleave + cast for A (entire batch is contiguous)
        deinterleave_cast_fp16_to_fp8(
            A,
            reinterpret_cast<__nv_fp8_e4m3*>(buffers_.A_real()),
            reinterpret_cast<__nv_fp8_e4m3*>(buffers_.A_imag()),
            total_A, stream);

        // F1: Fused deinterleave + cast (+ transpose) for B
        cutlass::float_e4m3_t* Br_ptr;
        cutlass::float_e4m3_t* Bi_ptr;

        if (!is_hermitian) {
            deinterleave_cast_fp16_to_fp8(
                B,
                reinterpret_cast<__nv_fp8_e4m3*>(buffers_.B_real()),
                reinterpret_cast<__nv_fp8_e4m3*>(buffers_.B_imag()),
                total_B, stream);
            Br_ptr = buffers_.B_real();
            Bi_ptr = buffers_.B_imag();
        } else {
            deinterleave_cast_fp16_to_fp8_transposed(
                B, buffers_.BT_real(), buffers_.BT_imag(),
                K, N, stream, batch_count);
            Br_ptr = buffers_.BT_real();
            Bi_ptr = buffers_.BT_imag();
        }

        // C still needs FP16 planar for β-accumulation
        __half *Cr, *Ci;
        CUDA_CHECK(cudaMallocAsync(&Cr, size_C * sizeof(__half), stream));
        CUDA_CHECK(cudaMallocAsync(&Ci, size_C * sizeof(__half), stream));
        deinterleave_complex(C, Cr, Ci, size_C, stream);

        auto mode = is_hermitian ? ComplexMode::Hermitian : ComplexMode::Standard;
        auto status = run_subgemms_fp8(
            buffers_.A_real(), buffers_.A_imag(),
            Br_ptr, Bi_ptr,
            Cr, Ci,
            M, N, K, alpha, beta, mode, stream, batch_count);

        if (status == cutlass::Status::kSuccess) {
            interleave_complex(Cr, Ci, C, size_C, stream);
        }

        cudaFreeAsync(Cr, stream);
        cudaFreeAsync(Ci, stream);

        return status;
    }


    /// ---------------------------------------------------------------
    /// Batched Gram Matrix: Planar Complex
    /// ---------------------------------------------------------------
    ///
    /// Memory layout (strided):
    ///   A_real[b] at A_real + b * M * K    (each [M × K], row-major)
    ///   C_real[b] at C_real + b * D * D    (D = M for AAH, K for AHA)
    ///
    cutlass::Status run_gram_planar_batched(
        const __half* A_real, const __half* A_imag,
        __half* C_real, __half* C_imag,
        int M, int K,
        int batch_count,
        float alpha = 1.0f,
        float beta  = 0.0f,
        GramMode mode = GramMode::AAH,
        cudaStream_t stream = nullptr)
    {
        if (batch_count <= 0) return cutlass::Status::kSuccess;
        ensure_streams();

        int64_t per_A = static_cast<int64_t>(M) * K;
        int64_t total_A = per_A * batch_count;

        cutlass::float_e4m3_t *Xr_A, *Xi_A, *Xr_B, *Xi_B;
        int gemm_M, gemm_N, gemm_K;

        if (mode == GramMode::AAH) {
            gemm_M = M; gemm_N = M; gemm_K = K;

            buffers_.ensure_capacity(total_A, total_A, 0, stream);

            // F2: Fused quad cast + duplicate — single kernel writes to all 4 outputs
            cast_fp16_to_fp8_e4m3_quad(A_real, A_imag,
                reinterpret_cast<__nv_fp8_e4m3*>(buffers_.A_real()),
                reinterpret_cast<__nv_fp8_e4m3*>(buffers_.B_real()),
                reinterpret_cast<__nv_fp8_e4m3*>(buffers_.A_imag()),
                reinterpret_cast<__nv_fp8_e4m3*>(buffers_.B_imag()), total_A, stream);

            Xr_A = buffers_.A_real();  Xi_A = buffers_.A_imag();
            Xr_B = buffers_.B_real();  Xi_B = buffers_.B_imag();
        } else {
            gemm_M = K; gemm_N = K; gemm_K = M;

            buffers_.ensure_capacity(total_A, 0, total_A, stream);

            // F2: Fused quad cast + transpose + duplicate
            cast_fp16_to_fp8_e4m3_transposed_quad(A_real, A_imag,
                buffers_.BT_real(), buffers_.A_real(),
                buffers_.BT_imag(), buffers_.A_imag(), K, M, stream, batch_count);

            Xr_A = buffers_.BT_real();  Xi_A = buffers_.BT_imag();
            Xr_B = buffers_.A_real();   Xi_B = buffers_.A_imag();
        }

        return run_gram_subgemms_fp8(
            Xr_A, Xi_A, Xr_B, Xi_B, C_real, C_imag,
            gemm_M, gemm_N, gemm_K, alpha, beta, stream, batch_count);
    }


    /// ---------------------------------------------------------------
    /// Batched Gram Matrix: Interleaved Complex (convenience)
    /// ---------------------------------------------------------------
    cutlass::Status run_gram_batched(
        __half* A, __half* C,
        int M, int K,
        int batch_count,
        GramMode mode = GramMode::AAH,
        float alpha = 1.0f,
        float beta  = 0.0f,
        cudaStream_t stream = nullptr)
    {
        ensure_streams();

        int64_t total_A = static_cast<int64_t>(M) * K * batch_count;
        int C_dim = (mode == GramMode::AAH) ? M : K;
        int64_t size_C = static_cast<int64_t>(C_dim) * C_dim * batch_count;

        // F2: Fused deinterleave + cast + duplicate for A
        cutlass::float_e4m3_t *Xr_A, *Xi_A, *Xr_B, *Xi_B;
        int gemm_M, gemm_N, gemm_K;

        if (mode == GramMode::AAH) {
            gemm_M = M; gemm_N = M; gemm_K = K;
            buffers_.ensure_capacity(total_A, total_A, 0, stream);

            // F2: Fused deinterleave + cast + duplicate
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
            gemm_M = K; gemm_N = K; gemm_K = M;
            buffers_.ensure_capacity(total_A, 0, total_A, stream);

            // F2: Fused deinterleave + cast + transpose + duplicate
            deinterleave_cast_fp16_to_fp8_transposed_dual(
                A, buffers_.BT_real(), buffers_.BT_imag(),
                buffers_.A_real(), buffers_.A_imag(),
                K, M, stream, batch_count);

            Xr_A = buffers_.BT_real();  Xi_A = buffers_.BT_imag();
            Xr_B = buffers_.A_real();   Xi_B = buffers_.A_imag();
        }

        // C stays FP16 planar
        __half *Cr, *Ci;
        CUDA_CHECK(cudaMallocAsync(&Cr, size_C * sizeof(__half), stream));
        CUDA_CHECK(cudaMallocAsync(&Ci, size_C * sizeof(__half), stream));
        deinterleave_complex(C, Cr, Ci, size_C, stream);

        auto status = run_gram_subgemms_fp8(
            Xr_A, Xi_A, Xr_B, Xi_B, Cr, Ci,
            gemm_M, gemm_N, gemm_K, alpha, beta, stream, batch_count);

        if (status == cutlass::Status::kSuccess) {
            interleave_complex(Cr, Ci, C, size_C, stream);
        }

        cudaFreeAsync(Cr, stream);
        cudaFreeAsync(Ci, stream);

        return status;
    }
    // ====================================================================
    // Strategy 3A: FP8 Native Input API
    // ====================================================================
    //
    // Optional API for when input data is already in FP8 E4M3 format.
    // Skips the FP16→FP8 cast entirely — only a lightweight byte-shuffle
    // deinterleave is needed to split interleaved complex into planar Re/Im.

    /// FP8 interleaved complex GEMM (single).
    /// A_interleaved, B_interleaved: FP8 E4M3 interleaved [Re0,Im0,Re1,Im1,...].
    /// C_real, C_imag: FP16 planar output.
    cutlass::Status run_fp8_interleaved(
        const __nv_fp8_e4m3* A_interleaved,   // M*K complex elements
        const __nv_fp8_e4m3* B_interleaved,   // K*N complex elements
        __half* C_real, __half* C_imag,
        int M, int N, int K,
        float alpha = 1.0f, float beta = 0.0f,
        ComplexMode mode = ComplexMode::Standard,
        cudaStream_t stream = nullptr)
    {
        bool is_hermitian = (mode == ComplexMode::Hermitian);
        ensure_streams();

        int64_t size_A = static_cast<int64_t>(M) * K;
        int64_t size_B = static_cast<int64_t>(K) * N;

        buffers_.ensure_capacity(size_A, size_B, is_hermitian ? size_B : 0, stream);

        // Byte-shuffle deinterleave: FP8 interleaved → FP8 planar
        deinterleave_fp8(
            A_interleaved,
            reinterpret_cast<__nv_fp8_e4m3*>(buffers_.A_real()),
            reinterpret_cast<__nv_fp8_e4m3*>(buffers_.A_imag()),
            size_A, stream);

        cutlass::float_e4m3_t* Br_ptr;
        cutlass::float_e4m3_t* Bi_ptr;

        if (!is_hermitian) {
            deinterleave_fp8(
                B_interleaved,
                reinterpret_cast<__nv_fp8_e4m3*>(buffers_.B_real()),
                reinterpret_cast<__nv_fp8_e4m3*>(buffers_.B_imag()),
                size_B, stream);
            Br_ptr = buffers_.B_real();
            Bi_ptr = buffers_.B_imag();
        } else {
            // Deinterleave to planar, then transpose
            deinterleave_fp8(
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
            M, N, K, alpha, beta, mode, stream);
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
        cudaStream_t stream = nullptr)
    {
        if (batch_count <= 0) return cutlass::Status::kSuccess;
        bool is_hermitian = (mode == ComplexMode::Hermitian);
        ensure_streams();

        int64_t per_A = static_cast<int64_t>(M) * K;
        int64_t per_B = static_cast<int64_t>(K) * N;
        int64_t total_A = per_A * batch_count;
        int64_t total_B = per_B * batch_count;

        buffers_.ensure_capacity(total_A, total_B, is_hermitian ? total_B : 0, stream);

        // Byte-shuffle deinterleave entire batch (layout-agnostic)
        deinterleave_fp8(
            A_interleaved,
            reinterpret_cast<__nv_fp8_e4m3*>(buffers_.A_real()),
            reinterpret_cast<__nv_fp8_e4m3*>(buffers_.A_imag()),
            total_A, stream);

        cutlass::float_e4m3_t* Br_ptr;
        cutlass::float_e4m3_t* Bi_ptr;

        if (!is_hermitian) {
            deinterleave_fp8(
                B_interleaved,
                reinterpret_cast<__nv_fp8_e4m3*>(buffers_.B_real()),
                reinterpret_cast<__nv_fp8_e4m3*>(buffers_.B_imag()),
                total_B, stream);
            Br_ptr = buffers_.B_real();
            Bi_ptr = buffers_.B_imag();
        } else {
            deinterleave_fp8(
                B_interleaved,
                reinterpret_cast<__nv_fp8_e4m3*>(buffers_.B_real()),
                reinterpret_cast<__nv_fp8_e4m3*>(buffers_.B_imag()),
                total_B, stream);
            // Transpose each batch element
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
            M, N, K, alpha, beta, mode, stream, batch_count);
    }


    // ====================================================================
    // BLAS3-style API — interleaved complex FP16 in/out
    // ====================================================================
    //
    // These are thin wrappers over the existing methods. All casting,
    // deinterleaving, and packing is handled internally.

    /// GEMM — C = alpha * A * B + beta * C  (Standard)
    ///         C = alpha * A * B^H + beta * C  (Hermitian)
    ///
    /// A, B, C are interleaved complex FP16: [Re0,Im0,Re1,Im1,...].
    /// A is M x K, B is K x N (Standard) or N x K (Hermitian), C is M x N.
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
        return run(
            const_cast<__half*>(A),
            const_cast<__half*>(B),
            C, M, N, K,
            mode == ComplexMode::Hermitian,
            alpha, beta,
            params.stream);
    }

    /// Batched GEMM — batch_count independent GEMMs on contiguous arrays.
    ///
    /// A[b] at A + b*2*M*K, B[b] at B + b*2*K*N, C[b] at C + b*2*M*N.
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
        return run_batched(
            const_cast<__half*>(A),
            const_cast<__half*>(B),
            C, M, N, K, batch_count,
            mode == ComplexMode::Hermitian,
            alpha, beta,
            params.stream);
    }

    /// HERK — Hermitian Rank-K Update with packed triangular output.
    ///
    /// C = alpha * A * A^H + beta * C   (NoTrans, A is N x K)
    /// C = alpha * A^H * A + beta * C   (ConjTrans, A is K x N)
    ///
    /// A is interleaved complex FP16: [Re0,Im0,Re1,Im1,...].
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

        // ---- Direct HERK dispatch (NoTrans only) ----
        if (op == HerkOp::NoTrans) {
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
                ensure_streams();

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

        // Temporary planar buffers
        __half *Ar, *Ai, *Cr_packed, *Ci_packed;
        CUDA_CHECK(cudaMallocAsync(&Ar, size_A * sizeof(__half), stream));
        CUDA_CHECK(cudaMallocAsync(&Ai, size_A * sizeof(__half), stream));
        CUDA_CHECK(cudaMallocAsync(&Cr_packed, packed_elems * sizeof(__half), stream));
        CUDA_CHECK(cudaMallocAsync(&Ci_packed, packed_elems * sizeof(__half), stream));

        // Deinterleave A
        deinterleave_complex(A, Ar, Ai, size_A, stream);

        // Deinterleave packed C if beta != 0
        if (beta != 0.0f) {
            deinterleave_complex(C, Cr_packed, Ci_packed, packed_elems, stream);
        }

        auto status = herk_planar_packed(
            Ar, Ai, Cr_packed, Ci_packed,
            N, K, alpha, beta,
            op, fill, params.herk_strategy, stream,
            params.triangle_config);

        // Interleave planar packed output back to C
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

        // ---- Direct HERK dispatch (NoTrans only) ----
        // Auto threshold scales with batch tiling: when batch_tile < batch_count,
        // baseline loops with per-tile overhead while direct uses one gridDim.y launch.
        if (op == HerkOp::NoTrans) {
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
                ensure_streams();

                // Prepare FP8 interleaved precast buffer for the direct HERK kernel.
                int64_t total_fp8 = static_cast<int64_t>(N) * 2 * K * batch_count;
                const __nv_fp8_e4m3* precast;
                if (A_fp8) {
                    // INT4 path: caller already produced FP8 interleaved — use directly
                    precast = A_fp8;
                } else if (pre_deinterleaved) {
                    // Planar FP16 [Ar|Ai] — re-interleave then cast
                    ensure_herk_precast(total_fp8, stream);
                    int64_t total_A = static_cast<int64_t>(N) * K * batch_count;
                    __half* A_interleaved;
                    CUDA_CHECK(cudaMallocAsync(&A_interleaved, total_fp8 * sizeof(__half), stream));
                    interleave_complex(A, A + total_A, A_interleaved, total_A, stream);
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
                ensure_hw_info();
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

        int A_rows = (op == HerkOp::NoTrans) ? N : K;
        int A_cols = (op == HerkOp::NoTrans) ? K : N;
        int64_t per_A = static_cast<int64_t>(A_rows) * A_cols;
        int64_t total_A = per_A * batch_count;
        int64_t packed_elems = static_cast<int64_t>(N) * (N + 1) / 2;
        int64_t total_packed = packed_elems * batch_count;

        // Temporary planar buffers (all batches contiguous)
        __half *Ar = nullptr, *Ai = nullptr, *Cr_packed, *Ci_packed;
        bool owns_Ar_Ai = false;
        if (pre_deinterleaved) {
            // Caller already produced planar layout: Ar at A, Ai at A + total_A
            Ar = const_cast<__half*>(A);
            Ai = const_cast<__half*>(A) + total_A;
        } else {
            CUDA_CHECK(cudaMallocAsync(&Ar, total_A * sizeof(__half), stream));
            CUDA_CHECK(cudaMallocAsync(&Ai, total_A * sizeof(__half), stream));
            owns_Ar_Ai = true;
            deinterleave_complex(A, Ar, Ai, total_A, stream);
        }
        CUDA_CHECK(cudaMallocAsync(&Cr_packed, total_packed * sizeof(__half), stream));
        CUDA_CHECK(cudaMallocAsync(&Ci_packed, total_packed * sizeof(__half), stream));

        // Deinterleave packed C if beta != 0
        if (beta != 0.0f) {
            deinterleave_complex(C, Cr_packed, Ci_packed, total_packed, stream);
        }

        // ---- Batch tiling: process batch_tile elements at a time ----
        // Reduces scratch from 2×N²×batch to 2×N²×batch_tile, improving
        // L2 cache reuse when N² × batch exceeds L2.
        // Skip tiling when beta == 0: all sub-GEMMs use beta=0 internally,
        // so scratch is write-only and L2 caching provides no benefit.
        // Skip tiling when compute-bound (K > N): full batch occupancy matters more.
        ensure_hw_info();
        int batch_tile = batch_count;
        if (use_batch_tiling_ && beta != 0.0f && K <= N) {
            int64_t scratch_per_batch = 4LL * N * N;  // 2 buffers × N² × sizeof(__half)
            if (l2_cache_bytes_ > 0 && scratch_per_batch * batch_count > l2_cache_bytes_) {
                batch_tile = std::max(1, static_cast<int>(l2_cache_bytes_ / scratch_per_batch));
            }
        }

        cutlass::Status status = cutlass::Status::kSuccess;
        for (int tile_start = 0; tile_start < batch_count && status == cutlass::Status::kSuccess;
             tile_start += batch_tile) {
            int actual_tile = std::min(batch_tile, batch_count - tile_start);

            status = herk_planar_packed_batched(
                Ar + tile_start * per_A,
                Ai + tile_start * per_A,
                Cr_packed + tile_start * packed_elems,
                Ci_packed + tile_start * packed_elems,
                N, K, actual_tile, alpha, beta,
                op, fill, params.herk_strategy, stream,
                params.triangle_config);
        }

        // Interleave all packed output batches back to C
        if (status == cutlass::Status::kSuccess) {
            interleave_complex(Cr_packed, Ci_packed, C, total_packed, stream);
        }

        if (owns_Ar_Ai) {
            cudaFreeAsync(Ar, stream);
            cudaFreeAsync(Ai, stream);
        }
        cudaFreeAsync(Cr_packed, stream);
        cudaFreeAsync(Ci_packed, stream);

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

        // ---- Direct HERK kernel: single-launch path for NoTrans ----
        if (op == HerkOp::NoTrans) {
            bool should_use_direct = false;
            if (herk_mode_ == HerkMode::ForceDirect) {
                should_use_direct = true;
            } else if (herk_mode_ == HerkMode::Auto) {
                // Lower bound: below K_CHUNK (64), output writes dominate and
                // scattered packed-triangle stores cause ~6x write amplification.
                // Baseline writes coalesced to N×N scratch, avoiding this.
                int threshold_k = N / 4;
                if (batch_count > 1 && l2_cache_bytes_ > 0) {
                    int64_t scratch_per_batch = 8LL * N * N;
                    int batch_tile = std::max(1, static_cast<int>(l2_cache_bytes_ / scratch_per_batch));
                    if (batch_count > batch_tile)
                        threshold_k = N / 2;
                }
                should_use_direct = (K >= HERK_K_CHUNK && K <= threshold_k);
            }

            if (should_use_direct) {
                ensure_streams();

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
                ensure_hw_info();
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
        // The baseline FP32 path does not support beta accumulation because it would
        // require deinterleaving the existing packed FP32 C output into planar form.
        // The direct HERK kernel handles beta natively. If you need beta != 0 with
        // FP32 output, use the direct path (HerkMode::ForceDirect).
        if (beta != 0.0f) {
            throw std::runtime_error(
                "HERK_batched_fp32out baseline path does not support beta != 0. "
                "Use HerkMode::ForceDirect (direct=1) for beta accumulation with FP32 output.");
        }

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

        // Planar packed FP32 buffers for intermediate pack output
        int64_t total_packed = packed_elems * batch_count;
        float *Cr_packed_fp32 = nullptr, *Ci_packed_fp32 = nullptr;
        CUDA_CHECK(cudaMallocAsync(&Cr_packed_fp32, total_packed * sizeof(float), stream));
        CUDA_CHECK(cudaMallocAsync(&Ci_packed_fp32, total_packed * sizeof(float), stream));

        // Batch tiling — skip when beta == 0 (sub-GEMMs are write-only, no L2 benefit)
        // Skip tiling when compute-bound (K > N): full batch occupancy matters more.
        ensure_hw_info();
        int batch_tile = batch_count;
        if (use_batch_tiling_ && beta != 0.0f && K <= N) {
            int64_t scratch_per_batch = 8LL * N * N;
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

            // Float-sized scratch
            ensure_herk_real_temp(total_output * 2, stream);
            ensure_herk_temp(total_output * 2, stream);

            ensure_streams();

            // FP16 → FP8 stacked-K cast
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

            // Re(C) via stacked-K GEMM with FP32 output
            status = run_real_gemm_fp32out(Stacked, Stacked_B, scratch_Re,
                                            N, N, 2 * K, alpha, 0.0f, stream_a_, actual_tile);
            if (status != cutlass::Status::kSuccess) break;

            // Im temp = Xi·Xr^T with FP32 output
            status = run_real_gemm_fp32out(Xi_sep, Xr_sep, temp_Im,
                                            N, N, K, 1.0f, 0.0f, stream_b_, actual_tile);
            if (status != cutlass::Status::kSuccess) break;

            CUDA_CHECK(cudaEventRecord(stream_a_done_, stream_a_));
            CUDA_CHECK(cudaEventRecord(stream_b_done_, stream_b_));
            CUDA_CHECK(cudaStreamWaitEvent(stream, stream_a_done_));
            CUDA_CHECK(cudaStreamWaitEvent(stream, stream_b_done_));

            // Pack + antisymmetrize → planar FP32 packed
            float* Cr_tile = Cr_packed_fp32 + tile_start * packed_elems;
            float* Ci_tile = Ci_packed_fp32 + tile_start * packed_elems;
            pack_antisymmetrize_triangle_fp32(
                scratch_Re, temp_Im,
                Cr_tile, Ci_tile,
                N, alpha, 0.0f, fill, stream, actual_tile);
        }

        // Interleave planar FP32 packed → interleaved FP32 packed
        if (status == cutlass::Status::kSuccess) {
            interleave_complex_fp32(Cr_packed_fp32, Ci_packed_fp32, C,
                                     total_packed, stream);
        }

        if (owns_Ar_Ai) {
            CUDA_CHECK(cudaFreeAsync(Ar, stream));
            CUDA_CHECK(cudaFreeAsync(Ai, stream));
        }
        CUDA_CHECK(cudaFreeAsync(Cr_packed_fp32, stream));
        CUDA_CHECK(cudaFreeAsync(Ci_packed_fp32, stream));

        return status;
    }
