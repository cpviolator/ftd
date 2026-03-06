    // ---------------------------------------------------------------
    // Grouped GEMM — single-launch multiple sub-GEMMs (SM90)
    // ---------------------------------------------------------------
    // Packs num_groups independent sub-GEMMs into a single CUTLASS Grouped GEMM
    // kernel launch. Each group has its own problem shape, pointers, and strides.
    //
    // All groups share the same alpha/beta scalars.
    //
    // Host arrays are copied to device before launch. Device memory is freed
    // asynchronously after the kernel is enqueued.
    //
#if defined(CUTLASS_ARCH_MMA_MODIFIABLE_TMA_SM90_SUPPORTED)
    cutlass::Status run_real_gemm_grouped(
        int num_groups,
        const std::vector<cute::Shape<int, int, int>>& problem_shapes_host,
        const std::vector<const cutlass::float_e4m3_t*>& ptr_A_host,
        const std::vector<const cutlass::float_e4m3_t*>& ptr_B_host,
        const std::vector<cutlass::half_t*>& ptr_C_host,
        const std::vector<GroupedInternalStrideA>& stride_A_host,
        const std::vector<GroupedInternalStrideB>& stride_B_host,
        const std::vector<GroupedInternalStrideC>& stride_C_host,
        float alpha, float beta,
        cudaStream_t stream)
    {
        ensure_hw_info();

        using ProblemShapeType = cute::Shape<int, int, int>;

        // --- Allocate device arrays ---
        ProblemShapeType* d_problem_shapes = nullptr;
        const cutlass::float_e4m3_t** d_ptr_A = nullptr;
        const cutlass::float_e4m3_t** d_ptr_B = nullptr;
        const cutlass::half_t** d_ptr_C = nullptr;
        cutlass::half_t** d_ptr_D = nullptr;
        GroupedInternalStrideA* d_stride_A = nullptr;
        GroupedInternalStrideB* d_stride_B = nullptr;
        GroupedInternalStrideC* d_stride_C = nullptr;
        GroupedInternalStrideD* d_stride_D = nullptr;

        size_t sz_shapes  = num_groups * sizeof(ProblemShapeType);
        size_t sz_ptr_A   = num_groups * sizeof(const cutlass::float_e4m3_t*);
        size_t sz_ptr_B   = num_groups * sizeof(const cutlass::float_e4m3_t*);
        size_t sz_ptr_C   = num_groups * sizeof(const cutlass::half_t*);
        size_t sz_ptr_D   = num_groups * sizeof(cutlass::half_t*);
        size_t sz_str_A   = num_groups * sizeof(GroupedInternalStrideA);
        size_t sz_str_B   = num_groups * sizeof(GroupedInternalStrideB);
        size_t sz_str_C   = num_groups * sizeof(GroupedInternalStrideC);
        size_t sz_str_D   = num_groups * sizeof(GroupedInternalStrideD);

        CUDA_CHECK(cudaMallocAsync(&d_problem_shapes, sz_shapes, stream));
        CUDA_CHECK(cudaMallocAsync(&d_ptr_A, sz_ptr_A, stream));
        CUDA_CHECK(cudaMallocAsync(&d_ptr_B, sz_ptr_B, stream));
        CUDA_CHECK(cudaMallocAsync(&d_ptr_C, sz_ptr_C, stream));
        CUDA_CHECK(cudaMallocAsync(&d_ptr_D, sz_ptr_D, stream));
        CUDA_CHECK(cudaMallocAsync(&d_stride_A, sz_str_A, stream));
        CUDA_CHECK(cudaMallocAsync(&d_stride_B, sz_str_B, stream));
        CUDA_CHECK(cudaMallocAsync(&d_stride_C, sz_str_C, stream));
        CUDA_CHECK(cudaMallocAsync(&d_stride_D, sz_str_D, stream));

        // --- Copy host arrays to device ---
        CUDA_CHECK(cudaMemcpyAsync(d_problem_shapes, problem_shapes_host.data(), sz_shapes, cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(d_ptr_A, ptr_A_host.data(), sz_ptr_A, cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(d_ptr_B, ptr_B_host.data(), sz_ptr_B, cudaMemcpyHostToDevice, stream));
        // C and D alias (in-place update): copy C pointers to both
        CUDA_CHECK(cudaMemcpyAsync(d_ptr_C, ptr_C_host.data(), sz_ptr_C, cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(d_ptr_D, ptr_C_host.data(), sz_ptr_D, cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(d_stride_A, stride_A_host.data(), sz_str_A, cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(d_stride_B, stride_B_host.data(), sz_str_B, cudaMemcpyHostToDevice, stream));
        // stride_C == stride_D (in-place)
        CUDA_CHECK(cudaMemcpyAsync(d_stride_C, stride_C_host.data(), sz_str_C, cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(d_stride_D, stride_C_host.data(), sz_str_D, cudaMemcpyHostToDevice, stream));

        // --- Construct grouped GEMM arguments ---
        cutlass::KernelHardwareInfo hw_info;
        hw_info.device_id = 0;
        hw_info.sm_count = hw_sm_count_;

        // Construct arguments, then set fusion params on the epilogue thread args.
        // (Avoids naming the fusion params type, which differs across CUTLASS versions.)
        GroupedGemmArguments arguments{
            cutlass::gemm::GemmUniversalMode::kGrouped,
            {num_groups, d_problem_shapes, problem_shapes_host.data()},
            {d_ptr_A, d_stride_A, d_ptr_B, d_stride_B},
            {{}, d_ptr_C, d_stride_C, d_ptr_D, d_stride_D},
            hw_info
        };
        // Scalar alpha/beta shared across all groups
        arguments.epilogue.thread.alpha = alpha;
        arguments.epilogue.thread.beta = beta;
        arguments.epilogue.thread.alpha_ptr = nullptr;
        arguments.epilogue.thread.beta_ptr = nullptr;
        arguments.epilogue.thread.alpha_ptr_array = nullptr;
        arguments.epilogue.thread.beta_ptr_array = nullptr;
        arguments.epilogue.thread.dAlpha = {cute::_0{}, cute::_0{}, 0};
        arguments.epilogue.thread.dBeta = {cute::_0{}, cute::_0{}, 0};

        GroupedDeviceGemm gemm_op;

        size_t workspace_size = gemm_op.get_workspace_size(arguments);
        void* workspace = nullptr;
        if (workspace_size > 0) {
            CUDA_CHECK(cudaMallocAsync(&workspace, workspace_size, stream));
        }

        auto status = gemm_op.can_implement(arguments);
        if (status != cutlass::Status::kSuccess) {
            fprintf(stderr, "[CUTLASS SM90] can_implement FAILED for grouped GEMM (%d groups)\n", num_groups);
            goto cleanup;
        }

        status = gemm_op.initialize(arguments, workspace, stream);
        if (status != cutlass::Status::kSuccess) {
            fprintf(stderr, "[CUTLASS SM90] initialize FAILED for grouped GEMM (%d groups)\n", num_groups);
            goto cleanup;
        }

        status = gemm_op.run(stream);

    cleanup:
        if (workspace) cudaFreeAsync(workspace, stream);
        cudaFreeAsync(d_problem_shapes, stream);
        cudaFreeAsync(d_ptr_A, stream);
        cudaFreeAsync(d_ptr_B, stream);
        cudaFreeAsync(d_ptr_C, stream);
        cudaFreeAsync(d_ptr_D, stream);
        cudaFreeAsync(d_stride_A, stream);
        cudaFreeAsync(d_stride_B, stream);
        cudaFreeAsync(d_stride_C, stream);
        cudaFreeAsync(d_stride_D, stream);

        return status;
    }
#endif // CUTLASS_ARCH_MMA_MODIFIABLE_TMA_SM90_SUPPORTED


    // ---------------------------------------------------------------
    // Sub-Matrix GEMM with custom output leading dimension
    // ---------------------------------------------------------------
    // Computes C_sub[M_block × N_block] = α · A · B^T + β · C_sub
    // where C_sub is embedded within a larger ld_C-wide matrix.
    //
    // A: M_block × K, packed RowMajor
    // B: N_block × K, packed ColumnMajor (contiguous from B start)
    // C: M_block × N_block output, but stride is ld_C (not N_block)
    //
    cutlass::Status run_real_gemm_submatrix(
        const cutlass::float_e4m3_t* A,
        const cutlass::float_e4m3_t* B,
        cutlass::half_t* C,
        int M_block, int N_block, int K,
        int ld_C,      // leading dimension of C (full matrix width)
        float alpha, float beta,
        cudaStream_t stream,
        int batch_count = 1,
        int64_t batch_stride_A = 0,   // 0 → M_block * K (packed default)
        int64_t batch_stride_B = 0,   // 0 → N_block * K (packed default)
        int64_t batch_stride_C = 0)   // 0 → M_block * ld_C (packed default)
    {
        ensure_hw_info();

        // A and B are packed sub-matrices — standard stride derivation
        auto stride_A = cutlass::make_cute_packed_stride(
            StrideA{}, cute::make_shape(M_block, K, batch_count));
        auto stride_B = cutlass::make_cute_packed_stride(
            StrideB{}, cute::make_shape(N_block, K, batch_count));

        // C/D stride: use ld_C as leading dimension (may differ from N_block)
        auto stride_C = cutlass::make_cute_packed_stride(
            StrideC{}, cute::make_shape(M_block, ld_C, batch_count));
        auto stride_D = stride_C;

        // Override batch strides when caller provides custom values.
        if (batch_stride_A != 0) cute::get<2>(stride_A) = batch_stride_A;
        if (batch_stride_B != 0) cute::get<2>(stride_B) = batch_stride_B;
        if (batch_stride_C != 0) {
            cute::get<2>(stride_C) = batch_stride_C;
            stride_D = stride_C;
        }

        cutlass::KernelHardwareInfo hw_info;
        hw_info.device_id = 0;
        hw_info.sm_count = hw_sm_count_;

        GemmArguments arguments{
            cutlass::gemm::GemmUniversalMode::kGemm,
            {M_block, N_block, K, batch_count},  // problem shape — actual compute extent
            {A, stride_A, B, stride_B},
            {{alpha, beta}, C, stride_C, C, stride_D},
            hw_info
        };

        DeviceGemm gemm_op;

        size_t workspace_size = gemm_op.get_workspace_size(arguments);
        void* workspace = nullptr;
        if (workspace_size > 0) {
            CUDA_CHECK(cudaMallocAsync(&workspace, workspace_size, stream));
        }

        auto status = gemm_op.can_implement(arguments);
        if (status != cutlass::Status::kSuccess) {
            fprintf(stderr, "[CUTLASS] can_implement FAILED for submatrix GEMM(%d,%d,%d) ld_C=%d\n",
                    M_block, N_block, K, ld_C);
            if (workspace) cudaFreeAsync(workspace, stream);
            return status;
        }

        status = gemm_op.initialize(arguments, workspace, stream);
        if (status != cutlass::Status::kSuccess) {
            fprintf(stderr, "[CUTLASS] initialize FAILED for submatrix GEMM(%d,%d,%d) ld_C=%d\n",
                    M_block, N_block, K, ld_C);
            if (workspace) cudaFreeAsync(workspace, stream);
            return status;
        }

        status = gemm_op.run(stream);
        // NOTE: no cudaStreamSynchronize — caller batches multiple launches

        if (workspace) cudaFreeAsync(workspace, stream);
        return status;
    }


    // ---------------------------------------------------------------
    // Lower-Triangle via Grouped GEMM — single-launch (SM90)
    // ---------------------------------------------------------------
    // Replaces the per-slab launch loop with a single CUTLASS Grouped GEMM.
    // Total groups = num_slabs * batch_count. All groups share alpha/beta.
    //
    // The grouped kernel's tile scheduler distributes tiles from ALL slabs
    // across all SMs, enabling more slabs (T=5+) even when individual slabs
    // would have too few tiles to fill the GPU. This unlocks much deeper
    // triangle decomposition compared to the per-slab path.
    //
    // On GH200 (132 SMs) at N=4096 with 128x256 tiles:
    //   Per-slab: T=2 (75% FLOPs) — slab0 has only 128 tiles < 132 SMs
    //   Grouped:  T~32 (51.6% FLOPs) — 264 total tiles across all slabs
    //
#if defined(CUTLASS_ARCH_MMA_MODIFIABLE_TMA_SM90_SUPPORTED)
    cutlass::Status run_real_gemm_lower_triangle_grouped(
        const cutlass::float_e4m3_t* A,
        const cutlass::float_e4m3_t* B,
        cutlass::half_t* C,
        int N_out, int K,
        float alpha, float beta,
        cudaStream_t stream,
        int batch_count = 1,
        const TriangleConfig& tri = {},
        int sm_count = 0)
    {
        constexpr int kTileM = COMPLEX_FP8_TILE_M;
        constexpr int kTileN = COMPLEX_FP8_TILE_N;

        if (sm_count <= 0) {
            ensure_hw_info();
            sm_count = hw_sm_count_;
        }

        // --- Determine target_slabs ---
        // KEY DIFFERENCE from per-slab path: check TOTAL tiles across all slabs
        // (not per-slab tiles). The grouped kernel distributes tiles from ALL groups
        // across all SMs, so small individual slabs are fine as long as the total
        // tile count provides enough work for latency hiding (4x SM count).
        int target_slabs;
        if (tri.target_slabs > 0) {
            target_slabs = tri.target_slabs;
        } else {
            target_slabs = 2;
            for (int T = 32; T >= 2; --T) {
                int S = ((N_out + T - 1) / T);
                S = ((S + kTileM - 1) / kTileM) * kTileM;
                // Compute total tiles across all slabs for this T
                int64_t total_tiles = 0;
                int pos = 0;
                for (int slab = 0; slab < T && pos < N_out; ++slab) {
                    int row_end = std::min(pos + S, N_out);
                    int M_block = row_end - pos;
                    int N_block = row_end;
                    int tiles_m = (M_block + kTileM - 1) / kTileM;
                    int tiles_n = (N_block + kTileN - 1) / kTileN;
                    total_tiles += static_cast<int64_t>(tiles_m) * tiles_n;
                    pos = row_end;
                }
                total_tiles *= batch_count;  // grouped kernel sees all batches
                if (total_tiles >= 4 * sm_count) {
                    target_slabs = T;
                    break;
                }
            }
        }

        // min_slab_height: kTileM (one tile row). The grouped kernel handles
        // small groups fine — no per-slab occupancy requirement.
        int min_slab_height;
        if (tri.min_slab_height > 0) {
            min_slab_height = tri.min_slab_height;
        } else {
            min_slab_height = kTileM;
        }

        // --- Compute slab boundaries ---
        std::vector<int> boundaries;
        int num_slabs;

        if (tri.graduated) {
            int slab_height_uniform = (N_out + target_slabs - 1) / target_slabs;
            slab_height_uniform = ((slab_height_uniform + kTileM - 1) / kTileM) * kTileM;
            slab_height_uniform = std::max(slab_height_uniform, min_slab_height);
            num_slabs = (N_out + slab_height_uniform - 1) / slab_height_uniform;
            boundaries.resize(num_slabs + 1);
            boundaries[0] = 0;
            for (int i = 1; i <= num_slabs; ++i) {
                double frac = std::sqrt(static_cast<double>(i) / num_slabs);
                int b = static_cast<int>(std::round(frac * N_out));
                b = ((b + kTileM - 1) / kTileM) * kTileM;
                b = std::min(b, N_out);
                boundaries[i] = b;
            }
            boundaries[num_slabs] = N_out;
            std::vector<int> deduped;
            deduped.push_back(0);
            for (int i = 1; i <= num_slabs; ++i) {
                if (boundaries[i] > deduped.back()) {
                    deduped.push_back(boundaries[i]);
                }
            }
            if (deduped.back() != N_out) deduped.push_back(N_out);
            boundaries = deduped;
            num_slabs = static_cast<int>(boundaries.size()) - 1;
        } else {
            int slab_height = (N_out + target_slabs - 1) / target_slabs;
            slab_height = ((slab_height + kTileM - 1) / kTileM) * kTileM;
            slab_height = std::max(slab_height, min_slab_height);
            num_slabs = (N_out + slab_height - 1) / slab_height;
            boundaries.resize(num_slabs + 1);
            for (int i = 0; i <= num_slabs; ++i) {
                boundaries[i] = std::min(i * slab_height, N_out);
            }
        }

        if (tri.verbose) {
            fprintf(stderr, "[Triangle SM90 Grouped] N=%d K=%d batch=%d sm_count=%d tile=%dx%d\n",
                    N_out, K, batch_count, sm_count, kTileM, kTileN);
            fprintf(stderr, "  target_slabs=%d (auto=%s) min_slab=%d (auto=%s) graduated=%s\n",
                    target_slabs, tri.target_slabs > 0 ? "no" : "yes",
                    min_slab_height, tri.min_slab_height > 0 ? "no" : "yes",
                    tri.graduated ? "yes" : "no");
            fprintf(stderr, "  num_slabs=%d  boundaries:", num_slabs);
            for (int i = 0; i <= num_slabs; ++i) fprintf(stderr, " %d", boundaries[i]);
            fprintf(stderr, "\n");
            int64_t total_flops = 0;
            int64_t total_tiles = 0;
            for (int i = 0; i < num_slabs; ++i) {
                int row_start = boundaries[i];
                int M_block = boundaries[i + 1] - row_start;
                int N_block = boundaries[i + 1];
                int tiles_m = (M_block + kTileM - 1) / kTileM;
                int tiles_n = (N_block + kTileN - 1) / kTileN;
                int64_t tiles = static_cast<int64_t>(tiles_m) * tiles_n * batch_count;
                int64_t flops = 2LL * M_block * N_block * K * batch_count;
                total_flops += flops;
                total_tiles += tiles;
                fprintf(stderr, "  slab %d: rows [%d..%d) M=%d N=%d tiles=%lld\n",
                        i, row_start, boundaries[i + 1], M_block, N_block,
                        static_cast<long long>(tiles));
            }
            int64_t full_flops = 2LL * N_out * N_out * K * batch_count;
            fprintf(stderr, "  total FLOPs: %.1f%% of full GEMM\n",
                    100.0 * total_flops / full_flops);
            fprintf(stderr, "  total tiles: %lld  GPU waves: %.1f\n",
                    static_cast<long long>(total_tiles),
                    static_cast<double>(total_tiles) / sm_count);
            fprintf(stderr, "  groups: %d (2*%d-1=%d per batch × %d batches) -> single kernel launch\n",
                    (2 * num_slabs - 1) * batch_count, num_slabs,
                    2 * num_slabs - 1, batch_count);
        }

        // --- Strategy 2C: Build two-phase group descriptors ---
        // Each slab generates 2 groups (rectangle + diagonal) except slab 0 (diagonal only).
        // Total groups = (2*num_slabs - 1) * batch_count.
        int groups_per_batch = 2 * num_slabs - 1;
        int num_groups = groups_per_batch * batch_count;

        std::vector<cute::Shape<int, int, int>> problem_shapes_host(num_groups);
        std::vector<const cutlass::float_e4m3_t*> ptr_A_host(num_groups);
        std::vector<const cutlass::float_e4m3_t*> ptr_B_host(num_groups);
        std::vector<cutlass::half_t*> ptr_C_host(num_groups);
        std::vector<GroupedInternalStrideA> stride_A_host(num_groups);
        std::vector<GroupedInternalStrideB> stride_B_host(num_groups);
        std::vector<GroupedInternalStrideC> stride_C_host(num_groups);

        int64_t batch_stride_AB = static_cast<int64_t>(N_out) * K;
        int64_t batch_stride_C  = static_cast<int64_t>(N_out) * N_out;

        for (int batch = 0; batch < batch_count; ++batch) {
            int g = batch * groups_per_batch;
            for (int slab = 0; slab < num_slabs; ++slab) {
                int row_start = boundaries[slab];
                int M_block = boundaries[slab + 1] - row_start;

                // Phase 1: Rectangle (zero waste) — skip for slab 0
                if (row_start > 0) {
                    problem_shapes_host[g] = cute::make_shape(M_block, row_start, K);
                    ptr_A_host[g] = A + batch * batch_stride_AB + static_cast<int64_t>(row_start) * K;
                    ptr_B_host[g] = B + batch * batch_stride_AB;
                    ptr_C_host[g] = C + batch * batch_stride_C + static_cast<int64_t>(row_start) * N_out;
                    stride_A_host[g] = cutlass::make_cute_packed_stride(
                        GroupedInternalStrideA{}, cute::make_shape(M_block, K, 1));
                    stride_B_host[g] = cutlass::make_cute_packed_stride(
                        GroupedInternalStrideB{}, cute::make_shape(row_start, K, 1));
                    stride_C_host[g] = cutlass::make_cute_packed_stride(
                        GroupedInternalStrideC{}, cute::make_shape(M_block, N_out, 1));
                    ++g;
                }

                // Phase 2: Diagonal square (~50% waste)
                problem_shapes_host[g] = cute::make_shape(M_block, M_block, K);
                ptr_A_host[g] = A + batch * batch_stride_AB + static_cast<int64_t>(row_start) * K;
                ptr_B_host[g] = B + batch * batch_stride_AB + static_cast<int64_t>(row_start) * K;
                ptr_C_host[g] = C + batch * batch_stride_C + static_cast<int64_t>(row_start) * N_out + row_start;
                stride_A_host[g] = cutlass::make_cute_packed_stride(
                    GroupedInternalStrideA{}, cute::make_shape(M_block, K, 1));
                stride_B_host[g] = cutlass::make_cute_packed_stride(
                    GroupedInternalStrideB{}, cute::make_shape(M_block, K, 1));
                stride_C_host[g] = cutlass::make_cute_packed_stride(
                    GroupedInternalStrideC{}, cute::make_shape(M_block, N_out, 1));
                ++g;
            }
        }

        // --- Launch grouped GEMM ---
        auto status = run_real_gemm_grouped(
            num_groups, problem_shapes_host,
            ptr_A_host, ptr_B_host, ptr_C_host,
            stride_A_host, stride_B_host, stride_C_host,
            alpha, beta, stream);

        if (status != cutlass::Status::kSuccess) {
            fprintf(stderr, "[HERK Triangle SM90 Grouped] FAILED: groups=%d slabs=%d batch=%d\n",
                    num_groups, num_slabs, batch_count);
            return status;
        }

        // NOTE: no cudaStreamSynchronize — caller uses event-based sync
        // (stream_a_done_ / stream_b_done_) for completion.
        return cutlass::Status::kSuccess;
    }
#endif // CUTLASS_ARCH_MMA_MODIFIABLE_TMA_SM90_SUPPORTED


    // ---------------------------------------------------------------
    // Lower-Triangle GEMM via Block-Row Decomposition
    // ---------------------------------------------------------------
    // Computes only the lower triangle of C[N×N] = α · A[N×K] · B[N×K]^T + β · C
    //
    // Decomposes into T = ceil(N / slab_height) rectangular sub-GEMMs:
    //   Block-row i: C[i*S : (i+1)*S,  0 : (i+1)*S] = α · A_rows · B_cols^T + β · C_sub
    //   where S = slab_height.
    //
    // Total FLOPs ≈ N²K (half of the full 2·N²K GEMM), with T kernel launches.
    //
    // A: N×K, RowMajor (stride K). Block-row i → A_ptr + i*S*K
    // B: N×K, ColumnMajor (stride K). Block-row i → B_ptr + 0, N_block = (i+1)*S
    // C: N×N, RowMajor (stride N). Block-row i → C_ptr + i*S*N, ld_C = N
    //
    cutlass::Status run_real_gemm_lower_triangle(
        const cutlass::float_e4m3_t* A,
        const cutlass::float_e4m3_t* B,
        cutlass::half_t* C,
        int N_out, int K,
        float alpha, float beta,
        cudaStream_t stream,
        int batch_count = 1,
        const TriangleConfig& tri = {},
        int sm_count = 0)
    {
#if defined(CUTLASS_ARCH_MMA_MODIFIABLE_TMA_SM90_SUPPORTED) && \
    !defined(COMPLEX_FP8_DISABLE_GROUPED_GEMM)
        // Grouped GEMM path: single kernel launch for all slabs.
        // Distributes tiles from ALL slabs across all SMs, enabling
        // deeper triangle decomposition (T=5+ vs T=2 with per-slab).
        return run_real_gemm_lower_triangle_grouped(
            A, B, C, N_out, K, alpha, beta, stream, batch_count, tri, sm_count);
#else
        // Per-slab loop fallback (also used when grouped GEMM is disabled).
        constexpr int kTileM = COMPLEX_FP8_TILE_M;  // typically 128
        constexpr int kTileN = COMPLEX_FP8_TILE_N;  // typically 256

        // Use provided sm_count, or fall back to cached hw_sm_count_
        if (sm_count <= 0) {
            ensure_hw_info();
            sm_count = hw_sm_count_;
        }

        // --- Determine target_slabs ---
        int target_slabs;
        if (tri.target_slabs > 0) {
            target_slabs = tri.target_slabs;
        } else {
            // Adaptive: find maximum T where slab 0's per-batch tile count
            // fills the GPU. We use per-batch tiles (not total tiles) because
            // each slab is a separate kernel launch — small per-slab GEMMs
            // run inefficiently regardless of how many batches are processed.
            target_slabs = 2;
            for (int T = 32; T >= 2; --T) {
                int S = ((N_out + T - 1) / T);
                S = ((S + kTileM - 1) / kTileM) * kTileM;
                int tiles_m = (S + kTileM - 1) / kTileM;
                int tiles_n = (S + kTileN - 1) / kTileN;
                int64_t tiles_slab0 = static_cast<int64_t>(tiles_m) * tiles_n;
                if (tiles_slab0 >= sm_count) {
                    target_slabs = T;
                    break;
                }
            }
        }

        // --- Determine min_slab_height ---
        int min_slab_height;
        if (tri.min_slab_height > 0) {
            min_slab_height = tri.min_slab_height;
        } else {
            // Occupancy-aware: ensure smallest slab generates >= sm_count
            // per-batch tiles (same reasoning as target_slabs above)
            double ratio = static_cast<double>(kTileM) / kTileN;
            int tiles_needed = static_cast<int>(std::ceil(
                std::sqrt(static_cast<double>(sm_count) / ratio)));
            min_slab_height = kTileM * std::max(tiles_needed, 1);
        }

        // --- Compute slab boundaries ---
        std::vector<int> boundaries;
        int num_slabs;

        if (tri.graduated) {
            int slab_height_uniform = (N_out + target_slabs - 1) / target_slabs;
            slab_height_uniform = ((slab_height_uniform + kTileM - 1) / kTileM) * kTileM;
            slab_height_uniform = std::max(slab_height_uniform, min_slab_height);
            num_slabs = (N_out + slab_height_uniform - 1) / slab_height_uniform;
            boundaries.resize(num_slabs + 1);
            boundaries[0] = 0;
            for (int i = 1; i <= num_slabs; ++i) {
                double frac = std::sqrt(static_cast<double>(i) / num_slabs);
                int b = static_cast<int>(std::round(frac * N_out));
                b = ((b + kTileM - 1) / kTileM) * kTileM;
                b = std::min(b, N_out);
                boundaries[i] = b;
            }
            boundaries[num_slabs] = N_out;
            std::vector<int> deduped;
            deduped.push_back(0);
            for (int i = 1; i <= num_slabs; ++i) {
                if (boundaries[i] > deduped.back()) {
                    deduped.push_back(boundaries[i]);
                }
            }
            if (deduped.back() != N_out) deduped.push_back(N_out);
            boundaries = deduped;
            num_slabs = static_cast<int>(boundaries.size()) - 1;
        } else {
            int slab_height = (N_out + target_slabs - 1) / target_slabs;
            slab_height = ((slab_height + kTileM - 1) / kTileM) * kTileM;
            slab_height = std::max(slab_height, min_slab_height);
            num_slabs = (N_out + slab_height - 1) / slab_height;
            boundaries.resize(num_slabs + 1);
            for (int i = 0; i <= num_slabs; ++i) {
                boundaries[i] = std::min(i * slab_height, N_out);
            }
        }

        if (tri.verbose) {
            fprintf(stderr, "[Triangle SM90] N=%d K=%d batch=%d sm_count=%d tile=%dx%d\n",
                    N_out, K, batch_count, sm_count, kTileM, kTileN);
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
                int tiles_m = (M_block + kTileM - 1) / kTileM;
                int tiles_n = (N_block + kTileN - 1) / kTileN;
                int64_t tiles = static_cast<int64_t>(tiles_m) * tiles_n * batch_count;
                int64_t flops = 2LL * M_block * N_block * K * batch_count;
                total_flops += flops;
                fprintf(stderr, "  slab %d: rows [%d..%d) M=%d N=%d tiles=%lld\n",
                        i, row_start, boundaries[i + 1], M_block, N_block,
                        static_cast<long long>(tiles));
            }
            int64_t full_flops = 2LL * N_out * N_out * K * batch_count;
            fprintf(stderr, "  total FLOPs: %.1f%% of full GEMM\n",
                    100.0 * total_flops / full_flops);
        }

        // Batch strides reflect the full matrix, not the sub-block dimensions.
        int64_t batch_stride_AB = static_cast<int64_t>(N_out) * K;
        int64_t batch_stride_C  = static_cast<int64_t>(N_out) * N_out;

        // --- Optimized per-slab loop: single DeviceGemm, pre-allocated workspace ---
        DeviceGemm gemm_op;

        cutlass::KernelHardwareInfo hw_info;
        hw_info.device_id = 0;
        hw_info.sm_count = hw_sm_count_;

        // Pre-check can_implement with slab 0 dimensions (largest M) and N_out (largest N)
        {
            int pre_M = boundaries[1] - boundaries[0];
            int pre_N = N_out;
            auto sa = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(pre_M, K, batch_count));
            auto sb = cutlass::make_cute_packed_stride(StrideB{}, cute::make_shape(pre_N, K, batch_count));
            auto sc = cutlass::make_cute_packed_stride(StrideC{}, cute::make_shape(pre_M, N_out, batch_count));
            if (batch_count > 1) {
                cute::get<2>(sa) = batch_stride_AB;
                cute::get<2>(sb) = batch_stride_AB;
                cute::get<2>(sc) = batch_stride_C;
            }
            GemmArguments pre_args{
                cutlass::gemm::GemmUniversalMode::kGemm,
                {pre_M, pre_N, K, batch_count},
                {A, sa, B, sb},
                {{alpha, beta}, C, sc, C, sc},
                hw_info
            };
            auto check = gemm_op.can_implement(pre_args);
            if (check != cutlass::Status::kSuccess) {
                fprintf(stderr, "[HERK Triangle] can_implement FAILED for pre-check GEMM(%d,%d,%d)\n",
                        pre_M, pre_N, K);
                return check;
            }
        }

        // Workspace: compute max size across all slabs, allocate once
        size_t max_workspace = 0;
        for (int i = 0; i < num_slabs; ++i) {
            int M_block = boundaries[i + 1] - boundaries[i];
            int N_block = boundaries[i + 1];
            auto sa = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(M_block, K, batch_count));
            auto sb = cutlass::make_cute_packed_stride(StrideB{}, cute::make_shape(N_block, K, batch_count));
            auto sc = cutlass::make_cute_packed_stride(StrideC{}, cute::make_shape(M_block, N_out, batch_count));
            if (batch_count > 1) {
                cute::get<2>(sa) = batch_stride_AB;
                cute::get<2>(sb) = batch_stride_AB;
                cute::get<2>(sc) = batch_stride_C;
            }
            GemmArguments args{
                cutlass::gemm::GemmUniversalMode::kGemm,
                {M_block, N_block, K, batch_count},
                {A, sa, B, sb},
                {{alpha, beta}, C, sc, C, sc},
                hw_info
            };
            size_t ws = gemm_op.get_workspace_size(args);
            if (ws > max_workspace) max_workspace = ws;
        }

        void* workspace = nullptr;
        if (max_workspace > 0) {
            CUDA_CHECK(cudaMallocAsync(&workspace, max_workspace, stream));
        }

        // Strategy 2C: Two-phase per-slab launch (rectangle + diagonal)
        // Phase 1 — Rectangle (columns 0..row_start-1): zero waste
        // Phase 2 — Diagonal square (columns row_start..row_start+M_block-1): ~50% waste
        // Slab 0 has row_start=0 so only the diagonal phase runs.
        cutlass::Status status = cutlass::Status::kSuccess;
        for (int i = 0; i < num_slabs; ++i) {
            int row_start = boundaries[i];
            int M_block = boundaries[i + 1] - row_start;

            // Phase 1: Rectangle (zero waste)
            if (row_start > 0) {
                const auto* A_rect = A + static_cast<int64_t>(row_start) * K;
                const auto* B_rect = B;
                auto* C_rect = C + static_cast<int64_t>(row_start) * N_out;

                auto sa = cutlass::make_cute_packed_stride(
                    StrideA{}, cute::make_shape(M_block, K, batch_count));
                auto sb = cutlass::make_cute_packed_stride(
                    StrideB{}, cute::make_shape(row_start, K, batch_count));
                auto sc = cutlass::make_cute_packed_stride(
                    StrideC{}, cute::make_shape(M_block, N_out, batch_count));
                if (batch_count > 1) {
                    cute::get<2>(sa) = batch_stride_AB;
                    cute::get<2>(sb) = batch_stride_AB;
                    cute::get<2>(sc) = batch_stride_C;
                }

                GemmArguments rect_args{
                    cutlass::gemm::GemmUniversalMode::kGemm,
                    {M_block, row_start, K, batch_count},
                    {A_rect, sa, B_rect, sb},
                    {{alpha, beta}, C_rect, sc, C_rect, sc},
                    hw_info
                };

                status = gemm_op.initialize(rect_args, workspace, stream);
                if (status != cutlass::Status::kSuccess) {
                    fprintf(stderr, "[HERK Triangle] rect init FAILED: slab %d/%d GEMM(%d,%d,%d)\n",
                            i, num_slabs, M_block, row_start, K);
                    if (workspace) cudaFreeAsync(workspace, stream);
                    return status;
                }
                status = gemm_op.run(stream);
                if (status != cutlass::Status::kSuccess) {
                    fprintf(stderr, "[HERK Triangle] rect run FAILED: slab %d/%d GEMM(%d,%d,%d)\n",
                            i, num_slabs, M_block, row_start, K);
                    if (workspace) cudaFreeAsync(workspace, stream);
                    return status;
                }
            }

            // Phase 2: Diagonal square (~50% waste)
            {
                const auto* A_diag = A + static_cast<int64_t>(row_start) * K;
                const auto* B_diag = B + static_cast<int64_t>(row_start) * K;
                auto* C_diag = C + static_cast<int64_t>(row_start) * N_out + row_start;

                auto sa = cutlass::make_cute_packed_stride(
                    StrideA{}, cute::make_shape(M_block, K, batch_count));
                auto sb = cutlass::make_cute_packed_stride(
                    StrideB{}, cute::make_shape(M_block, K, batch_count));
                auto sc = cutlass::make_cute_packed_stride(
                    StrideC{}, cute::make_shape(M_block, N_out, batch_count));
                if (batch_count > 1) {
                    cute::get<2>(sa) = batch_stride_AB;
                    cute::get<2>(sb) = batch_stride_AB;
                    cute::get<2>(sc) = batch_stride_C;
                }

                GemmArguments diag_args{
                    cutlass::gemm::GemmUniversalMode::kGemm,
                    {M_block, M_block, K, batch_count},
                    {A_diag, sa, B_diag, sb},
                    {{alpha, beta}, C_diag, sc, C_diag, sc},
                    hw_info
                };

                status = gemm_op.initialize(diag_args, workspace, stream);
                if (status != cutlass::Status::kSuccess) {
                    fprintf(stderr, "[HERK Triangle] diag init FAILED: slab %d/%d GEMM(%d,%d,%d)\n",
                            i, num_slabs, M_block, M_block, K);
                    if (workspace) cudaFreeAsync(workspace, stream);
                    return status;
                }
                status = gemm_op.run(stream);
                if (status != cutlass::Status::kSuccess) {
                    fprintf(stderr, "[HERK Triangle] diag run FAILED: slab %d/%d GEMM(%d,%d,%d)\n",
                            i, num_slabs, M_block, M_block, K);
                    if (workspace) cudaFreeAsync(workspace, stream);
                    return status;
                }
            }
        }

        if (workspace) cudaFreeAsync(workspace, stream);

        // NOTE: no cudaStreamSynchronize — caller uses event-based sync
        // (stream_a_done_ / stream_b_done_) for completion.
        return cutlass::Status::kSuccess;
#endif // !CUTLASS_ARCH_MMA_MODIFIABLE_TMA_SM90_SUPPORTED || COMPLEX_FP8_DISABLE_GROUPED_GEMM
    }
