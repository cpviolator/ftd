// --- pipeline_impl.hpp ---
// FddGpuPipeline method implementations.
// Textual include inside the FddGpuPipeline class body — no #pragma once.

template <typename T_IO, typename T_ACC = float>
void run_cublaslt_4split_generic(int M, int N, int K, int batch_count, cudaStream_t stream) {
    // M = Batch Size
    // N = Ndm (Number of DM trials)
    // K = Nf (Number of Freq Channels) - Contraction dimension

    cublasLtMatmulDesc_t opDesc = nullptr;
    cublasLtMatrixLayout_t Adesc = nullptr, Bdesc = nullptr, Cdesc = nullptr;
    cublasLtMatmulPreference_t pref = nullptr;

    // 1. Setup Operation Descriptor
    // -----------------------------
    cublasComputeType_t computeType = CUBLAS_COMPUTE_32F;
    cudaDataType_t scaleType = CUDA_R_32F;
    CUBLASLT_CHECK(cublasLtMatmulDescCreate(&opDesc, computeType, scaleType));

    // Pointer Mode: HOST for Alpha/Beta, but SCALES are DEVICE pointers (set below)
    cublasLtPointerMode_t ptrMode = CUBLASLT_POINTER_MODE_HOST;
    CUBLASLT_CHECK(cublasLtMatmulDescSetAttribute(opDesc, CUBLASLT_MATMUL_DESC_POINTER_MODE, &ptrMode, sizeof(ptrMode)));

    // [FIX] Explicitly set Default Epilogue to resolve ambiguity for WGMMA kernels
    cublasLtEpilogue_t epilogue = CUBLASLT_EPILOGUE_DEFAULT;
    CUBLASLT_CHECK(cublasLtMatmulDescSetAttribute(opDesc, CUBLASLT_MATMUL_DESC_EPILOGUE, &epilogue, sizeof(epilogue)));

    // 2. Configure Specific Paths (FP8 vs FP16)
    // -----------------------------------------
    if constexpr (std::is_same_v<T_IO, __nv_fp8_e4m3>) {
        // --- FP8 HOPPER PATH (CANONICAL) ---

        // A. Operation: C = A^T * B
        cublasOperation_t opT = CUBLAS_OP_T;
        cublasOperation_t opN = CUBLAS_OP_N;
        CUBLASLT_CHECK(cublasLtMatmulDescSetAttribute(opDesc, CUBLASLT_MATMUL_DESC_TRANSA, &opT, sizeof(opT)));
        CUBLASLT_CHECK(cublasLtMatmulDescSetAttribute(opDesc, CUBLASLT_MATMUL_DESC_TRANSB, &opN, sizeof(opN)));

        // B. Scaling Pointers (STRICTLY REQUIRED FOR FP8)
        char* d_base = d_fp8_scales_.get<char>();
        float* p_scale_A = (float*)(d_base + 0);
        float* p_scale_B = (float*)(d_base + 16);

        CUBLASLT_CHECK(cublasLtMatmulDescSetAttribute(opDesc, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &p_scale_A, sizeof(p_scale_A)));
        CUBLASLT_CHECK(cublasLtMatmulDescSetAttribute(opDesc, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &p_scale_B, sizeof(p_scale_B)));

        // C. Matrix Layouts
        CUBLASLT_CHECK(cublasLtMatrixLayoutCreate(&Adesc, CUDA_R_8F_E4M3, K, M, K));
        CUBLASLT_CHECK(cublasLtMatrixLayoutCreate(&Bdesc, CUDA_R_8F_E4M3, K, N, K));

    } else {
        // --- FP16 LEGACY PATH ---

        // Operation: C = A * B (Standard Column Major)
        cublasOperation_t opN = CUBLAS_OP_N;
        CUBLASLT_CHECK(cublasLtMatmulDescSetAttribute(opDesc, CUBLASLT_MATMUL_DESC_TRANSA, &opN, sizeof(opN)));
        CUBLASLT_CHECK(cublasLtMatmulDescSetAttribute(opDesc, CUBLASLT_MATMUL_DESC_TRANSB, &opN, sizeof(opN)));

        // Layouts
        CUBLASLT_CHECK(cublasLtMatrixLayoutCreate(&Adesc, CUDA_R_16F, M, K, M));
        CUBLASLT_CHECK(cublasLtMatrixLayoutCreate(&Bdesc, CUDA_R_16F, K, N, K));
    }

    // Common Output Layout C: [M, N] (Batch x Ndm) - Always FP32 Col-Major
    CUBLASLT_CHECK(cublasLtMatrixLayoutCreate(&Cdesc, CUDA_R_32F, M, N, M));


    // Batch Strides
    int64_t strA = (int64_t)M * (int64_t)K;
    int64_t strB = (int64_t)K * (int64_t)N;
    int64_t strC = (int64_t)M * (int64_t)N;

    CUBLASLT_CHECK(cublasLtMatrixLayoutSetAttribute(Adesc, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch_count, sizeof(batch_count)));
    CUBLASLT_CHECK(cublasLtMatrixLayoutSetAttribute(Adesc, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &strA, sizeof(strA)));
    CUBLASLT_CHECK(cublasLtMatrixLayoutSetAttribute(Bdesc, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch_count, sizeof(batch_count)));
    CUBLASLT_CHECK(cublasLtMatrixLayoutSetAttribute(Bdesc, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &strB, sizeof(strB)));
    CUBLASLT_CHECK(cublasLtMatrixLayoutSetAttribute(Cdesc, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch_count, sizeof(batch_count)));
    CUBLASLT_CHECK(cublasLtMatrixLayoutSetAttribute(Cdesc, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &strC, sizeof(strC)));

    // 3. Algorithm Selection (Heuristic)
    // ----------------------------------
    const cublasLtMatmulAlgo_t* algo = nullptr;

    if (algo_cached_) {
        algo = &cached_algo_struct_;
    } else {
      CUBLASLT_CHECK(cublasLtMatmulPreferenceCreate(&pref));
      uint64_t max_workspace = 256ULL * 1024ULL * 1024ULL; // 256 MB

      if (d_workspace_.get() == nullptr || workspace_size_ < max_workspace) {
	d_workspace_.allocate(max_workspace);
	workspace_size_ = max_workspace;
      }
      CUBLASLT_CHECK(cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &max_workspace, sizeof(max_workspace)));

      cublasLtMatmulHeuristicResult_t heuristicResult = {};
      int returnedResults = 0;

      cublasStatus_t status = cublasLtMatmulAlgoGetHeuristic(
							     lt_handle_, opDesc, Adesc, Bdesc, Cdesc, Cdesc,
							     pref, 1, &heuristicResult, &returnedResults);

      if (status != CUBLAS_STATUS_SUCCESS || returnedResults == 0) {
	std::cerr << "[FATAL] cuBLASLt Heuristic failed to find an FP8 algorithm. "
		  << "Status: " << (int)status
		  << ". Ensure you are running on Ada/Hopper (sm_89/sm_90) and inputs are 32-byte aligned." << std::endl;
	exit(1);
      } else {
	if constexpr (std::is_same_v<T_IO, __nv_fp8_e4m3>) {
	  std::cout << "[SUCCESS] FP8 Algorithm Found!" << std::endl;
	}

	cached_algo_struct_ = heuristicResult.algo;
	algo_cached_ = true;
	algo = &cached_algo_struct_;

	std::cout << "[SUCCESS] Algorithm Found!" << std::endl;

	int algoId, tile, stages, splitK, reduction, swizzle, customOption;
	size_t sizeWritten;

	cublasLtMatmulAlgoConfigGetAttribute(algo, CUBLASLT_ALGO_CONFIG_ID, &algoId, sizeof(algoId), &sizeWritten);
	cublasLtMatmulAlgoConfigGetAttribute(algo, CUBLASLT_ALGO_CONFIG_TILE_ID, &tile, sizeof(tile), &sizeWritten);
	cublasLtMatmulAlgoConfigGetAttribute(algo, CUBLASLT_ALGO_CONFIG_STAGES_ID, &stages, sizeof(stages), &sizeWritten);
	cublasLtMatmulAlgoConfigGetAttribute(algo, CUBLASLT_ALGO_CONFIG_SPLITK_NUM, &splitK, sizeof(splitK), &sizeWritten);
	cublasLtMatmulAlgoConfigGetAttribute(algo, CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME, &reduction, sizeof(reduction), &sizeWritten);
	cublasLtMatmulAlgoConfigGetAttribute(algo, CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING, &swizzle, sizeof(swizzle), &sizeWritten);
	cublasLtMatmulAlgoConfigGetAttribute(algo, CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION, &customOption, sizeof(customOption), &sizeWritten);

	std::cout << "  Algo ID: " << algoId << "\n"
		  << "  Tile ID: " << tile << "\n"
		  << "  Stages:  " << stages << "\n"
		  << "  SplitK:  " << splitK << "\n"
		  << "  Reduct:  " << reduction << "\n"
		  << "  Swizzle: " << swizzle << "\n"
		  << "  Custom:  " << customOption << std::endl;
	}
      cublasLtMatmulPreferenceDestroy(pref);
    }

    // 4. Execution (4-Split GEMM)
    // ---------------------------
    float one = 1.0f;
    float zero = 0.0f;
    float neg_one = -1.0f;

    // 1. Cre = Are * Bre
    CUBLASLT_CHECK(cublasLtMatmul(lt_handle_, opDesc, &one,
        p_planar_A_re_, Adesc,
        d_planar_B_re_.get(), Bdesc,
        &zero,
        p_planar_C_re_, Cdesc,
        p_planar_C_re_, Cdesc,
        algo, d_workspace_.get(), workspace_size_, stream));

    // 2. Cre = Cre - Aim * Bim
    CUBLASLT_CHECK(cublasLtMatmul(lt_handle_, opDesc, &neg_one,
        p_planar_A_im_, Adesc,
        d_planar_B_im_.get(), Bdesc,
        &one,
        p_planar_C_re_, Cdesc,
        p_planar_C_re_, Cdesc,
        algo, d_workspace_.get(), workspace_size_, stream));

    // 3. Cim = Are * Bim
    CUBLASLT_CHECK(cublasLtMatmul(lt_handle_, opDesc, &one,
        p_planar_A_re_, Adesc,
        d_planar_B_im_.get(), Bdesc,
        &zero,
        p_planar_C_im_, Cdesc,
        p_planar_C_im_, Cdesc,
        algo, d_workspace_.get(), workspace_size_, stream));

    // 4. Cim = Cim + Aim * Bre
    CUBLASLT_CHECK(cublasLtMatmul(lt_handle_, opDesc, &one,
        p_planar_A_im_, Adesc,
        d_planar_B_re_.get(), Bdesc,
        &one,
        p_planar_C_im_, Cdesc,
        p_planar_C_im_, Cdesc,
        algo, d_workspace_.get(), workspace_size_, stream));

    // Cleanup
    if(opDesc) cublasLtMatmulDescDestroy(opDesc);
    if(Adesc) cublasLtMatrixLayoutDestroy(Adesc);
    if(Bdesc) cublasLtMatrixLayoutDestroy(Bdesc);
    if(Cdesc) cublasLtMatrixLayoutDestroy(Cdesc);
}


  // --- UPDATED: Execute with Stream ---
  void execute(const Real* d_in_real, const ComplexType* d_phasors,
               Real* d_out_real, int current_batch_size, cudaStream_t stream) {

    // [PATCH] Map Pointers
    p_real_in_ = d_pool_1_.get<Real>();
    p_comp_fft_ = d_pool_2_.get<ComplexType>(); // FFT writes to Pool 2

    // FP16/FP8 Layouts
    size_t b_size = (fdd_compute_mode_ == "cublas_lt_fp8") ? sizeof(__nv_fp8_e4m3) : sizeof(__half);
    size_t size_A_plane = (size_t)current_batch_size * Nf_ * Nt_complex_ * b_size;
    // Align planes
    size_A_plane = (size_A_plane + 255) & ~255;

    // Planar A goes into Pool 1 (Overwrites Real In, which is done)
    p_planar_A_re_ = d_pool_1_.get<void>();
    p_planar_A_im_ = (void*)((char*)d_pool_1_.get() + size_A_plane);

    // Planar C goes into Pool 2 (Overwrites Comp FFT, which is done)
    size_t c_elem_size = sizeof(float);
    size_t size_C_plane = (size_t)current_batch_size * Ndm_ * Nt_complex_ * c_elem_size;
    size_C_plane = (size_C_plane + 255) & ~255;
    p_planar_C_re_ = d_pool_2_.get<void>();
    p_planar_C_im_ = (void*)((char*)d_pool_2_.get() + size_C_plane);

    // IFFT Input goes into Pool 1 (Overwrites Planar A)
    p_comp_ifft_ = d_pool_1_.get<ComplexType>();
    // Real Out goes into Pool 2 (Overwrites Planar C)
    p_real_out_ = d_pool_2_.get<Real>();

    // [PATCH] Legacy pointer mapping
    p_gemm_out_ = d_pool_2_.get<void>();

    int device = 0;
    CUDA_CHECK(cudaGetDevice(&device));

    // [PATCH] Only prefetch if we haven't cached the phasors yet
    bool using_cached_phasors = (phasors_ready_ &&
				 (fdd_compute_mode_ == "cublas_lt_fp16" || fdd_compute_mode_ == "cublas_lt_fp8"
				  || fdd_compute_mode_ == "cutlass" || fdd_compute_mode_ == "cutlass_fp6"
				  || fdd_compute_mode_ == "cutlass_fp4"));

    if (!using_cached_phasors) {
      size_t phasor_size = (size_t)Nt_complex_ * Ndm_ * Nf_ * sizeof(ComplexType);

      int device_id = 0;
      CUDA_CHECK(cudaGetDevice(&device_id));

    cudaMemLocation location;
    location.type = cudaMemLocationTypeDevice;
    location.id = device_id;

    CUDA_CHECK(cudaMemPrefetchAsync(d_phasors, phasor_size, location, 0, stream));
    }

    if (current_batch_size > max_batch_size_) return;

    // Set Library Streams
    cufftSetStream(fft_plan_fwd_, stream);
    cufftSetStream(fft_plan_inv_, stream);
    if (blas_handle_) cublasSetStream(blas_handle_, stream);

    float fft_fwd_ms = 0.0f;
    float transpose_ms = 0.0f;
    float kernel_gemm_ms = 0.0f;
    float gemm_setup_ms = 0.0f;
    float cublas_gemm_ms = 0.0f;
    float transpose2_ms = 0.0f;

    // 1. Copy Pad Real (Add stream to launch)
    dim3 grid_pad(current_batch_size * Nf_, Nt_padded_);
    dim3 block_pad(256);
    grid_pad.y = (Nt_padded_ + block_pad.x - 1) / block_pad.x;

    kernel_copy_pad_real<Real><<<grid_pad, block_pad, 0, stream>>>(
	    d_in_real, p_real_in_, current_batch_size, Nf_, Nt_, Nt_padded_);

    // 2. Forward FFT (R2C)
#ifdef HAS_CUFFT_LTO_CALLBACK
    // Reset device-side max before FFT (callback will accumulate during FFT)
    if (callback_registered_) {
        CUDA_CHECK(cudaMemsetAsync(d_max_uint_.get(), 0, sizeof(unsigned int), stream));
    }
#endif
    CUDA_CHECK(cudaEventRecord(start_event_, stream));
    cufftResult_t err;
    if constexpr (std::is_same_v<Real, float>) {
      err = cufftExecR2C(fft_plan_fwd_, (Real*)p_real_in_, (cufftComplex*)p_comp_fft_);
    } else {
      err = cufftExecD2Z(fft_plan_fwd_, (double*)p_real_in_, (cufftDoubleComplex*)p_comp_fft_);
    }

    if (err != CUFFT_SUCCESS) exit(1);
    CUDA_CHECK(cudaEventRecord(stop_event_, stream));
    CUDA_CHECK(cudaEventSynchronize(stop_event_));
    CUDA_CHECK(cudaEventElapsedTime(&fft_fwd_ms, start_event_, stop_event_));

    // 3. Transpose & GEMM
    if (fdd_compute_mode_ == "kernel") {
      // Transpose
      CUDA_CHECK(cudaEventRecord(start_event_, stream));
      dim3 block_T(TILE_DIM, TILE_DIM);
      dim3 grid_T((Nf_ + TILE_DIM - 1) / TILE_DIM,
                  (Nt_complex_ + TILE_DIM - 1) / TILE_DIM, current_batch_size);

      kernel_transpose_f_k<Real, ComplexType><<<grid_T, block_T, 0, stream>>>(
          p_comp_fft_, p_comp_ifft_,
          current_batch_size, Nf_, Nt_complex_);

      CUDA_CHECK(cudaEventRecord(stop_event_, stream));
      CUDA_CHECK(cudaEventSynchronize(stop_event_));
      CUDA_CHECK(cudaEventElapsedTime(&transpose_ms, start_event_, stop_event_));

      // GEMM
      CUDA_CHECK(cudaEventRecord(start_event_, stream));
      dim3 grid_gemm(current_batch_size, Nt_complex_, Ndm_);
      dim3 block_gemm(256);

      kernel_fdd_gemm_transpose<Real, ComplexType><<<grid_gemm, block_gemm, 0, stream>>>(
          d_phasors, p_comp_ifft_,
          (ComplexType*)p_gemm_out_, current_batch_size, Nf_, Nt_complex_, Ndm_);

      CUDA_CHECK(cudaEventRecord(stop_event_, stream));
      CUDA_CHECK(cudaEventSynchronize(stop_event_));
      CUDA_CHECK(cudaEventElapsedTime(&kernel_gemm_ms, start_event_, stop_event_));

    } else if (fdd_compute_mode_ == "cublas") {
      if constexpr (std::is_same_v<Real, float>) {
        // Transpose
        CUDA_CHECK(cudaEventRecord(start_event_, stream));
	dim3 block_T1(32, 32);
        dim3 grid_T1((Nt_complex_ + 31) / 32,
                     (current_batch_size + 31) / 32,
                     Nf_);

	kernel_transpose_bfk_to_kfb<ComplexType><<<grid_T1, block_T1, 0, stream>>>(
            p_comp_fft_,
            p_comp_ifft_,
            current_batch_size, Nf_, Nt_complex_);

        CUDA_CHECK(cudaEventRecord(stop_event_, stream));
        CUDA_CHECK(cudaEventSynchronize(stop_event_));
        CUDA_CHECK(cudaEventElapsedTime(&transpose_ms, start_event_, stop_event_));

        // Pointers
        CUDA_CHECK(cudaEventRecord(start_event_, stream));
        dim3 grid_ptr((Nt_complex_ + 255) / 256);
        dim3 block_ptr(256);

	kernel_setup_gemm_pointers<ComplexType><<<grid_ptr, block_ptr, 0, stream>>>(
	    d_A_pointers_.get<void*>(),
	    d_B_pointers_.get<void*>(),
	    d_C_pointers_.get<void*>(),
	    d_phasors,
            p_comp_ifft_,
            (ComplexType*)p_gemm_out_,
            current_batch_size, Nf_, Ndm_, Nt_complex_);

        CUDA_CHECK(cudaEventRecord(stop_event_, stream));
        CUDA_CHECK(cudaEventSynchronize(stop_event_));
        CUDA_CHECK(cudaEventElapsedTime(&gemm_setup_ms, start_event_, stop_event_));

        // cuBLAS
        CUDA_CHECK(cudaEventRecord(start_event_, stream));
        const cuComplex alpha = {1.0f, 0.0f};
        const cuComplex beta = {0.0f, 0.0f};
        int M = current_batch_size;
        int N = Ndm_;
        int K = Nf_;

        CUBLAS_CHECK(cublasCgemmBatched(
            blas_handle_, CUBLAS_OP_N, CUBLAS_OP_N, M, N, K, &alpha,
	    d_B_pointers_.get<cuComplex*>(), M,
            d_A_pointers_.get<cuComplex*>(), K, &beta,
            d_C_pointers_.get<cuComplex*>(), M, Nt_complex_));
        CUDA_CHECK(cudaEventRecord(stop_event_, stream));
        CUDA_CHECK(cudaEventSynchronize(stop_event_));
        CUDA_CHECK(cudaEventElapsedTime(&cublas_gemm_ms, start_event_, stop_event_));

        // Transpose Back
        CUDA_CHECK(cudaEventRecord(start_event_, stream));
        dim3 block_T2(TILE_DIM, TILE_DIM);
        dim3 grid_T2((current_batch_size + TILE_DIM - 1) / TILE_DIM,
                     (Nt_complex_ + TILE_DIM - 1) / TILE_DIM, Ndm_);

        kernel_transpose_kdb_to_bdk<ComplexType><<<grid_T2, block_T2, 0, stream>>>(
            (ComplexType*)p_gemm_out_, p_comp_ifft_,
            current_batch_size, Ndm_, Nt_complex_);

        CUDA_CHECK(cudaEventRecord(stop_event_, stream));
        CUDA_CHECK(cudaEventSynchronize(stop_event_));
        CUDA_CHECK(cudaEventElapsedTime(&transpose2_ms, start_event_, stop_event_));

      }
    } else if (fdd_compute_mode_ == "cublas_lt_fp8" || fdd_compute_mode_ == "cublas_lt_fp16") {
      // [PATCH] Compile-time guard: This block is strictly for Single Precision (float).
      if constexpr (std::is_same_v<Real, float>) {
	CUDA_CHECK(cudaEventRecord(start_event_, stream));


	// --- SHARED DYNAMIC SCALING LOGIC (FP8 & FP16) ---
         float max_val = 0.0f;

	// Compute actual max of FFT output for correct FP8/FP16 quantization scaling
	{
	  size_t total_elements = (size_t)current_batch_size * Nf_ * Nt_complex_;
	  auto policy = thrust::cuda::par.on(stream);
	  if constexpr (std::is_same_v<Real, float>) {
	    thrust::device_ptr<float> d_ptr((float*)p_comp_fft_);
	    max_val = thrust::transform_reduce(
					       policy, d_ptr, d_ptr + (total_elements * 2),
					       AbsMaxFunctor(), 0.0f, thrust::maximum<float>()
					       );
	  }
	}



         // --- DEBUG: Print scaling info once per run to verify numerical stability ---
         static bool debug_printed = false;
         if (!debug_printed) {
             std::cout << "[DEBUG] Max Val in FFT: " << max_val << std::endl;
         }

         if (max_val < 1e-9f) max_val = 1.0f;

	 float target_max = 0.0f;
         float scale_B = 1.0f; // Scale used during Kernel Prep

         if (fdd_compute_mode_ == "cublas_lt_fp8") {
             target_max = 448.0f;
             scale_B = 448.0f; // Matches prepare_phasors
         } else {
             target_max = 4096.0f;
             scale_B = 1.0f;
         }

         float scale_A = target_max / max_val;

         float scale_dequant_A = 1.0f / scale_A;
         float scale_dequant_B = 1.0f / scale_B;

         float unscale_finalize = 1.0f;

         if (fdd_compute_mode_ == "cublas_lt_fp8") {
             CUDA_CHECK(cudaMemcpyAsync((char*)d_fp8_scales_.get() + 0,  &scale_dequant_A, sizeof(float), cudaMemcpyHostToDevice, stream));
             CUDA_CHECK(cudaMemcpyAsync((char*)d_fp8_scales_.get() + 16, &scale_dequant_B, sizeof(float), cudaMemcpyHostToDevice, stream));

             unscale_finalize = 1.0f; // cuBLAS handles it
         } else {
             unscale_finalize = 1.0f / (scale_A * scale_B);
         }

         if (!debug_printed) {
             std::cout << "[DEBUG] Quant Scale A: " << scale_A << " | Quant Scale B: " << scale_B << std::endl;
         }

         // 3. Launch Kernels
         dim3 block(32, 32);
	 dim3 grid_d((Nt_complex_ + 31)/32, (current_batch_size + 31)/32, Nf_);
         dim3 grid_p((Nf_ + 31)/32, (Ndm_ + 31)/32, Nt_complex_);
         dim3 grid_c((current_batch_size + 31)/32, (Ndm_ + 31)/32, Nt_complex_);

         if (fdd_compute_mode_ == "cublas_lt_fp8") {

	   // 64×65 tile: block (64,16) = 1024 threads, 4× fewer blocks
	   dim3 block_opt(64, 16);
             dim3 grid_opt(
                 (Nf_ + 63) / 64,
                 (Nt_complex_ + 63) / 64,
                 current_batch_size
             );

             kernel_fused_data_prep_fp8_opt_64<__nv_fp8_e4m3><<<grid_opt, block_opt, 0, stream>>>(
											       (cufftComplex*)p_comp_fft_,
											       (__nv_fp8_e4m3*)p_planar_A_re_,
											       (__nv_fp8_e4m3*)p_planar_A_im_,
											       current_batch_size, Nf_, Nt_complex_, scale_A);

             // Capture timing for Transpose 1
             CUDA_CHECK(cudaEventRecord(stop_event_, stream));
             CUDA_CHECK(cudaEventSynchronize(stop_event_));
             CUDA_CHECK(cudaEventElapsedTime(&transpose_ms, start_event_, stop_event_));

             // Restart timer for GEMM
             CUDA_CHECK(cudaEventRecord(start_event_, stream));

	     // OPTIMIZATION: Only convert phasors once!
             if (!phasors_ready_) {
	       if (fdd_compute_mode_ == "cublas_lt_fp8") {
		 kernel_fused_phasor_prep<__nv_fp8_e4m3><<<grid_p, block, 0, stream>>>(
                         (cufftComplex*)d_phasors,
                         d_planar_B_re_.get<__nv_fp8_e4m3>(),
                         d_planar_B_im_.get<__nv_fp8_e4m3>(),
                         Nf_, Ndm_, Nt_complex_, scale_B);
                 } else {
                      kernel_fused_phasor_prep<__half><<<grid_p, block, 0, stream>>>(
                         (cufftComplex*)d_phasors,
                         d_planar_B_re_.get<__half>(),
                         d_planar_B_im_.get<__half>(),
                         Nf_, Ndm_, Nt_complex_, scale_B);
                 }
	       phasors_ready_ = true;
             }

	     // Run GEMM
             run_cublaslt_4split_generic<__nv_fp8_e4m3, float>(current_batch_size, Ndm_, Nf_, Nt_complex_, stream);

	     // Finalize (SMEM-tiled: coalesced reads on b, coalesced writes on k)
	     {
	       dim3 grid_fin((current_batch_size + 31)/32, (Nt_complex_ + 31)/32, Ndm_);
	       kernel_post_gemm_finalize_tiled<float><<<grid_fin, block, 0, stream>>>(
                 (float*)p_planar_C_re_,
                 (float*)p_planar_C_im_,
		 (cufftComplex*)p_comp_ifft_,
                 current_batch_size, Ndm_, Nt_complex_, unscale_finalize);
	     }

             // [PATCH] ADD MISSING TIMING RECORDING
             CUDA_CHECK(cudaEventRecord(stop_event_, stream));
             CUDA_CHECK(cudaEventSynchronize(stop_event_));
             CUDA_CHECK(cudaEventElapsedTime(&cublas_gemm_ms, start_event_, stop_event_));

         } else { // FP16 Mode

	   // 1. Data Prep (Planar Split)
	   kernel_fused_data_prep<__half><<<grid_d, block, 0, stream>>>(
                 (cufftComplex*)p_comp_fft_,
                 (__half*)p_planar_A_re_,
                 (__half*)p_planar_A_im_,
                 current_batch_size, Nf_, Nt_complex_, scale_A);

             // Capture timing for Transpose 1
             CUDA_CHECK(cudaEventRecord(stop_event_, stream));
             CUDA_CHECK(cudaEventSynchronize(stop_event_));
             CUDA_CHECK(cudaEventElapsedTime(&transpose_ms, start_event_, stop_event_));

             // Restart timer for GEMM
             CUDA_CHECK(cudaEventRecord(start_event_, stream));

             // 2. Phasor Prep (Planar Split)
	     if (!phasors_ready_) {
	       kernel_fused_phasor_prep<__half><<<grid_p, block, 0, stream>>>(
									      (cufftComplex*)d_phasors,
									      d_planar_B_re_.get<__half>(),
									      d_planar_B_im_.get<__half>(),
									      Nf_, Ndm_, Nt_complex_, scale_B);
	       phasors_ready_ = true;
	     }

             // 3. Run 4-Split GEMM
             run_cublaslt_4split_generic<__half, float>(
                 current_batch_size, Ndm_, Nf_, Nt_complex_, stream);

	     // 4. Finalize (SMEM-tiled: coalesced reads on b, coalesced writes on k)
	     {
	       dim3 grid_fin((current_batch_size + 31)/32, (Nt_complex_ + 31)/32, Ndm_);
	       kernel_post_gemm_finalize_tiled<float><<<grid_fin, block, 0, stream>>>(
                 (float*)p_planar_C_re_,
                 (float*)p_planar_C_im_,
                 p_comp_ifft_,
                 current_batch_size, Ndm_, Nt_complex_, unscale_finalize);
	     }

             // Capture GEMM Time
             CUDA_CHECK(cudaEventRecord(stop_event_, stream));
             CUDA_CHECK(cudaEventSynchronize(stop_event_));
             CUDA_CHECK(cudaEventElapsedTime(&cublas_gemm_ms, start_event_, stop_event_));

	 }
      }
    }
#ifdef HAS_CUTLASS_GEMM
    else if (fdd_compute_mode_ == "cutlass" || fdd_compute_mode_ == "cutlass_fp6"
             || fdd_compute_mode_ == "cutlass_fp4") {
      if constexpr (std::is_same_v<Real, float>) {
        CUDA_CHECK(cudaEventRecord(start_event_, stream));

        float alpha_gemm;

#ifdef HAS_CUFFT_LTO_CALLBACK
        if (callback_registered_) {
            // LTO callback path: FFT already wrote FP32 planar to Pool 2
            // and accumulated max in d_max_uint_. Just transpose + scale + FP16.
            dim3 block_d(64, 16);
            dim3 grid_d(
                (Nf_ + 63) / 64,
                (Nt_complex_ + 63) / 64,
                current_batch_size);

            kernel_transpose_scale_fp16<<<grid_d, block_d, 0, stream>>>(
                d_fft_planar_re_, d_fft_planar_im_,
                (unsigned int*)d_max_uint_.get(),
                (__half*)p_planar_A_re_, (__half*)p_planar_A_im_,
                current_batch_size, Nf_, Nt_complex_);

            CUDA_CHECK(cudaEventRecord(stop_event_, stream));
            CUDA_CHECK(cudaEventSynchronize(stop_event_));
            CUDA_CHECK(cudaEventElapsedTime(&transpose_ms, start_event_, stop_event_));

            // Read max_val from device (free — already synced by timer)
            unsigned int max_uint;
            CUDA_CHECK(cudaMemcpy(&max_uint, d_max_uint_.get(),
                                  sizeof(unsigned int), cudaMemcpyDeviceToHost));
            float max_val = *reinterpret_cast<float*>(&max_uint);
            if (max_val < 1e-9f) max_val = 1.0f;
            float scale_A = 448.0f / max_val;
            alpha_gemm = 1.0f / (scale_A * 448.0f);

        } else
#endif
        {
            // Fallback: original thrust + data prep path
            float max_val = 0.0f;
            {
              size_t total_elements = (size_t)current_batch_size * Nf_ * Nt_complex_;
              auto policy = thrust::cuda::par.on(stream);
              thrust::device_ptr<float> d_ptr((float*)p_comp_fft_);
              max_val = thrust::transform_reduce(
                  policy, d_ptr, d_ptr + (total_elements * 2),
                  AbsMaxFunctor(), 0.0f, thrust::maximum<float>());
            }
            if (max_val < 1e-9f) max_val = 1.0f;

            float scale_A = 448.0f / max_val;
            alpha_gemm = 1.0f / (scale_A * 448.0f);

            // Data Prep (64x65 tile: block (64,16), 4x fewer blocks)
            dim3 block_d(64, 16);
            dim3 grid_d(
                (Nf_ + 63) / 64,
                (Nt_complex_ + 63) / 64,
                current_batch_size);

            kernel_fused_data_prep_rowmajor_64<<<grid_d, block_d, 0, stream>>>(
                (cufftComplex*)p_comp_fft_,
                (__half*)p_planar_A_re_,
                (__half*)p_planar_A_im_,
                current_batch_size, Nf_, Nt_complex_, scale_A);

            CUDA_CHECK(cudaEventRecord(stop_event_, stream));
            CUDA_CHECK(cudaEventSynchronize(stop_event_));
            CUDA_CHECK(cudaEventElapsedTime(&transpose_ms, start_event_, stop_event_));
        }

        CUDA_CHECK(cudaEventRecord(start_event_, stream));

        if (is_tiled()) {
          // --- TILED GEMM PATH ---
          // Generate B phasors on-the-fly per tile, call gemm() per tile.
          auto compute = cutlass_gemm_api::ComputePrecision::FP8;
          if (fdd_compute_mode_ == "cutlass_fp6") compute = cutlass_gemm_api::ComputePrecision::FP6;
          if (fdd_compute_mode_ == "cutlass_fp4") compute = cutlass_gemm_api::ComputePrecision::FP4;

          for (int k_off = 0; k_off < Nt_complex_; k_off += Nt_tile_) {
            int tile_sz = std::min(Nt_tile_, Nt_complex_ - k_off);

            // Generate B phasors for this tile
            generate_phasors_tile(k_off, tile_sz, stream);

            // Slice A and C pointers for this tile
            __half* A_re_tile = (__half*)p_planar_A_re_ + (size_t)k_off * current_batch_size * Nf_;
            __half* A_im_tile = (__half*)p_planar_A_im_ + (size_t)k_off * current_batch_size * Nf_;
            float*  C_re_tile = (float*)p_planar_C_re_ + (size_t)k_off * current_batch_size * Ndm_;
            float*  C_im_tile = (float*)p_planar_C_im_ + (size_t)k_off * current_batch_size * Ndm_;

            // GEMM per tile (B converted internally each time)
            int ret = cutlass_gemm_->gemm(
                A_re_tile, A_im_tile,
                d_planar_B_re_.get<__half>(), d_planar_B_im_.get<__half>(),
                C_re_tile, C_im_tile,
                current_batch_size, Ndm_, Nf_, tile_sz,
                compute,
                cutlass_gemm_api::OutputPrecision::FP32,
                alpha_gemm, 0.0f, stream,
                /*tune=*/gemm_tune_);
            if (ret != 0) {
              fprintf(stderr, "CUTLASS tiled GEMM failed (status=%d) tile k_off=%d\n", ret, k_off);
            }
          }
        } else {
          // --- ORIGINAL NON-TILED PATH ---
          // 2. Phasor prep (one-time)
          if (!phasors_ready_) {
            dim3 block_p(32, 32);
            dim3 grid_p((Nf_ + 31)/32, (Ndm_ + 31)/32, Nt_complex_);
            kernel_fused_phasor_prep_rowmajor<<<grid_p, block_p, 0, stream>>>(
                (cufftComplex*)d_phasors,
                d_planar_B_re_.get<__half>(), d_planar_B_im_.get<__half>(),
                Nf_, Ndm_, Nt_complex_, 448.0f);
            phasors_ready_ = true;
          }

          // 3. CUTLASS batched complex GEMM
          int ret;
          ret = cutlass_gemm_->gemm_prepared(
              (__half*)p_planar_A_re_, (__half*)p_planar_A_im_,
              (float*)p_planar_C_re_, (float*)p_planar_C_im_,
              current_batch_size, Ndm_, Nf_, Nt_complex_,
              cutlass_gemm_api::OutputPrecision::FP32,
              alpha_gemm, 0.0f, stream,
              /*tune=*/gemm_tune_);
          if (ret != 0) {
            fprintf(stderr, "CUTLASS GEMM failed (status=%d) mode=%s\n", ret, fdd_compute_mode_.c_str());
          }
        }

        // 4. Finalize (SMEM-tiled: coalesced reads on dm, coalesced writes on k)
        dim3 block_c(32, 32);
        dim3 grid_c((Ndm_ + 31)/32, (Nt_complex_ + 31)/32, current_batch_size);
        kernel_post_gemm_finalize_cutlass_tiled<<<grid_c, block_c, 0, stream>>>(
            (float*)p_planar_C_re_,
            (float*)p_planar_C_im_,
            (cufftComplex*)p_comp_ifft_,
            current_batch_size, Ndm_, Nt_complex_);

        CUDA_CHECK(cudaEventRecord(stop_event_, stream));
        CUDA_CHECK(cudaEventSynchronize(stop_event_));
        CUDA_CHECK(cudaEventElapsedTime(&cublas_gemm_ms, start_event_, stop_event_));
      }
    }
#endif

    // 4. Inverse FFT (C2R)
    float fft_inv_ms = 0.0f;
    CUDA_CHECK(cudaEventRecord(start_event_, stream));
    if constexpr (std::is_same_v<Real, float>) {
      err = cufftExecC2R(fft_plan_inv_, (cufftComplex*)p_comp_ifft_, (Real*)p_real_out_);
    } else {
      err = cufftExecZ2D(fft_plan_inv_, (cufftDoubleComplex*)p_comp_ifft_, (Real*)p_real_out_);
    }
    if (err != CUFFT_SUCCESS) exit(1);
    CUDA_CHECK(cudaEventRecord(stop_event_, stream));
    CUDA_CHECK(cudaEventSynchronize(stop_event_));
    CUDA_CHECK(cudaEventElapsedTime(&fft_inv_ms, start_event_, stop_event_));

    // 5. Extract & Scale (Real)
    dim3 grid_extract(current_batch_size * Ndm_, Nt_);
    dim3 block_extract(256);
    grid_extract.y = (Nt_ + block_extract.x - 1) / block_extract.x;

    kernel_extract_scale_real<Real><<<grid_extract, block_extract, 0, stream>>>(
	  p_real_out_, d_out_real, current_batch_size, Ndm_, Nt_, Nt_padded_);

    print_performance_metrics(fft_fwd_ms, transpose_ms, kernel_gemm_ms,
			      gemm_setup_ms, cublas_gemm_ms, transpose2_ms, fft_inv_ms, current_batch_size);

  }


private:
  void print_performance_metrics(float fft_fwd_ms, float transpose_ms,
                                 float kernel_gemm_ms, float gemm_setup_ms,
                                 float cublas_gemm_ms, float transpose2_ms,
                                 float fft_inv_ms, int current_batch_size) {

    double to_gflops = 1e-6;
    double to_gbs = 1e-6;

    double fft_ops = 2.5 * Nt_padded_ * std::log2(Nt_padded_) * ((double)current_batch_size * Nf_);

    double fft_bytes = ((size_t)current_batch_size * Nf_) * (Nt_padded_ * sizeof(Real) + Nt_complex_ * sizeof(ComplexType));

    if (!verbose_) return;
    std::cout << "--- FDD GPU Pipeline Performance (R2C) ---" << std::endl;
    std::cout << "  [Forward FFT (R2C)]" << std::endl;
    std::cout << "    Time:    " << fft_fwd_ms << " ms" << std::endl;
    std::cout << "    Compute: " << (fft_ops / fft_fwd_ms) * to_gflops << " GFLOPS" << std::endl;
    std::cout << "    Memory:  " << (fft_bytes / fft_fwd_ms) * to_gbs << " GB/s" << std::endl;

    double transp_bytes = 2.0 * ((size_t)current_batch_size * Nf_ * Nt_complex_ * sizeof(ComplexType));

    std::cout << "  [Transpose 1 (Batch <-> Freq)]" << std::endl;
    std::cout << "    Time:    " << transpose_ms << " ms" << std::endl;
    std::cout << "    Memory:  " << (transp_bytes / transpose_ms) * to_gbs << " GB/s" << std::endl;

    double gemm_ops = 8.0 * (double)current_batch_size * Ndm_ * Nf_ * Nt_complex_;

    if (fdd_compute_mode_ == "kernel") {
        std::cout << "  [GEMM Kernel]" << std::endl;
        std::cout << "    Time:    " << kernel_gemm_ms << " ms" << std::endl;
        std::cout << "    Compute: " << (gemm_ops / kernel_gemm_ms) * to_gflops << " GFLOPS" << std::endl;

        double kern_bytes = ((size_t)Nt_complex_ * Ndm_ * Nf_ * sizeof(ComplexType)) +
                            ((size_t)Nt_complex_ * Nf_ * current_batch_size * sizeof(ComplexType)) +
                            ((size_t)Nt_complex_ * Ndm_ * current_batch_size * sizeof(ComplexType));

        std::cout << "    Memory:  " << (kern_bytes / kernel_gemm_ms) * to_gbs << " GB/s" << std::endl;
    } else {
        std::cout << "  [cuBLAS Path]" << std::endl;
        if (gemm_setup_ms > 0)
            std::cout << "    Setup:   " << gemm_setup_ms << " ms" << std::endl;
        std::cout << "    GEMM:    " << cublas_gemm_ms << " ms" << std::endl;
        std::cout << "    Compute: " << (gemm_ops / cublas_gemm_ms) * to_gflops << " GFLOPS" << std::endl;
    }

    if (transpose2_ms > 0.0f) {
        double transp2_bytes = 2.0 * ((size_t)current_batch_size * Ndm_ * Nt_complex_ * sizeof(ComplexType));
        std::cout << "  [Transpose 2 (Batch <-> DM)]" << std::endl;
        std::cout << "    Time:    " << transpose2_ms << " ms" << std::endl;
        std::cout << "    Memory:  " << (transp2_bytes / transpose2_ms) * to_gbs << " GB/s" << std::endl;
    }

    double ifft_bytes = ((size_t)current_batch_size * Ndm_) * (Nt_complex_ * sizeof(ComplexType) + Nt_padded_ * sizeof(Real));
    double ifft_ops = 2.5 * Nt_padded_ * std::log2(Nt_padded_) * ((double)current_batch_size * Ndm_);

    std::cout << "  [Inverse FFT (C2R)]" << std::endl;
    std::cout << "    Time:    " << fft_inv_ms << " ms" << std::endl;
    std::cout << "    Compute: " << (ifft_ops / fft_inv_ms) * to_gflops << " GFLOPS" << std::endl;
    std::cout << "    Memory:  " << (ifft_bytes / fft_inv_ms) * to_gbs << " GB/s" << std::endl;

    std::cout << "------------------------------------" << std::endl;
  }
