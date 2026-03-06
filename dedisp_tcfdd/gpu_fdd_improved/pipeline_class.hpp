// --- pipeline_class.hpp ---
// FddGpuPipeline class definition (extracted from umbrella).
// Textual include — no #pragma once.

#ifdef HAS_CUFFT_LTO_CALLBACK
#include <cufftXt.h>                    // cufftXtSetJITCallback, CUFFT_CB_ST_COMPLEX
#include "fft_store_callback_fatbin.h"  // Generated: fft_callback_fatbin_data[]

/// Context struct for cuFFT LTO store callback (must match fft_store_callback.cu).
struct FFTStoreCallbackInfo {
    float* d_planar_re;
    float* d_planar_im;
    unsigned int* d_max_uint;
};
#endif

/**
 * @brief Manages all persistent GPU state for the FDD pipeline (R2C Version).
 */
template <typename Real>
class FddGpuPipeline {
 public:
  using ComplexType =
      typename std::conditional<std::is_same<Real, float>::value, cufftComplex,
                                cufftDoubleComplex>::type;

  FddGpuPipeline(int max_batch_size, int Nf, int Nt, int Ndm,
		 int Nt_padded, std::string fdd_compute_mode,
		 int kernel_tune_verbosity = 0,
		 const char* kernel_tune_cache_path = nullptr,
		 int Nt_tile = 0,
		 int strategy_tune_verbosity = 1,
		 const char* strategy_tune_cache_path = nullptr,
		 bool gemm_tune = true)
      : max_batch_size_(max_batch_size),
        Nf_(Nf),
        Nt_(Nt),
        Ndm_(Ndm),
        Nt_padded_(Nt_padded),
	fdd_compute_mode_(fdd_compute_mode),
	blas_handle_(nullptr),
        start_event_(nullptr),
        stop_event_(nullptr),
	d_A_pointers_(),
	d_B_pointers_(),
	d_C_pointers_(),
	lt_handle_(nullptr),
        d_workspace_(),
        d_planar_A_re_(), d_planar_A_im_(),
        d_planar_B_re_(), d_planar_B_im_(),
        d_planar_C_re_(), d_planar_C_im_(),
	phasors_ready_(false),
	algo_cached_(false),
	Nt_tile_(Nt_tile) {

    // Calculate complex size for R2C: N/2 + 1
    Nt_complex_ = Nt_padded_ / 2 + 1;

    cufftResult_t err;
    int rank = 1;
    int n[] = {Nt_padded_};

    // Forward: R2C (Real -> Complex)
    int batch_fwd = max_batch_size_ * Nf_;

#ifdef HAS_CUFFT_LTO_CALLBACK
    if constexpr (std::is_same_v<Real, float>) {
      if (fdd_compute_mode_ == "cutlass" || fdd_compute_mode_ == "cutlass_fp6"
          || fdd_compute_mode_ == "cutlass_fp4") {
        // Split plan creation to register LTO callback before finalization
        err = cufftCreate(&fft_plan_fwd_);
        if (err != CUFFT_SUCCESS) { fprintf(stderr, "cufftCreate failed\n"); exit(1); }

        // Allocate callback context in device memory
        d_max_uint_.allocate(sizeof(unsigned int));
        d_callback_info_.allocate(sizeof(FFTStoreCallbackInfo));

        // Register LTO callback (must be before cufftMakePlanMany)
        void* d_info_ptr = d_callback_info_.get();
        auto cb_result = cufftXtSetJITCallback(
            fft_plan_fwd_,
            "fft_store_deinterleave_max",
            fft_callback_fatbin_data,
            sizeof(fft_callback_fatbin_data),
            CUFFT_CB_ST_COMPLEX,
            (void**)&d_info_ptr);
        if (cb_result != CUFFT_SUCCESS) {
          fprintf(stderr, "Warning: cuFFT LTO callback registration failed (%d), "
                  "falling back to standard path\n", (int)cb_result);
          cufftDestroy(fft_plan_fwd_);
          // Fall through to standard cufftPlanMany below
        } else {
          size_t work_size;
          err = cufftMakePlanMany(fft_plan_fwd_, rank, n,
                                  NULL, 1, Nt_padded_,
                                  NULL, 1, Nt_complex_,
                                  CUFFT_R2C, batch_fwd, &work_size);
          if (err != CUFFT_SUCCESS) { fprintf(stderr, "cufftMakePlanMany failed\n"); exit(1); }
          callback_registered_ = true;
          std::cout << "[FFT] cuFFT LTO store callback registered successfully" << std::endl;
        }
      }
    }
    if (!callback_registered_)
#endif
    {
      err = cufftPlanMany(&fft_plan_fwd_, rank, n,
                          NULL, 1, Nt_padded_,
                          NULL, 1, Nt_complex_,
                          (sizeof(Real) == 4) ? CUFFT_R2C : CUFFT_D2Z, batch_fwd);
      if (err != CUFFT_SUCCESS) exit(1);
    }

    // Inverse: C2R (Complex -> Real)
    int batch_inv = max_batch_size_ * Ndm_;
    err = cufftPlanMany(&fft_plan_inv_, rank, n,
                        NULL, 1, Nt_complex_,
                        NULL, 1, Nt_padded_,
                        (sizeof(Real) == 4) ? CUFFT_C2R : CUFFT_Z2D, batch_inv);
    if (err != CUFFT_SUCCESS) exit(1);

    // [PATCH] MEMORY OPTIMIZATION: PING-PONG POOLS
    size_t sz_real_io = (size_t)max_batch_size_ * std::max(Nf_, Ndm_) * Nt_padded_ * sizeof(Real);
    size_t sz_comp_io = (size_t)max_batch_size_ * std::max(Nf_, Ndm_) * Nt_complex_ * sizeof(ComplexType);

    size_t sz_planar_A_half = 0;
    size_t sz_planar_C_float = 0;

    if (fdd_compute_mode_ == "cublas_lt_fp8" || fdd_compute_mode_ == "cublas_lt_fp16") {
        size_t b_size = (fdd_compute_mode_ == "cublas_lt_fp8") ? sizeof(__nv_fp8_e4m3) : sizeof(__half);
        sz_planar_A_half = (size_t)max_batch_size_ * Nf_ * Nt_complex_ * b_size;
        sz_planar_C_float = (size_t)max_batch_size_ * Ndm_ * Nt_complex_ * sizeof(float);
    }
    if (fdd_compute_mode_ == "cutlass" || fdd_compute_mode_ == "cutlass_fp6"
        || fdd_compute_mode_ == "cutlass_fp4") {
        sz_planar_A_half = (size_t)max_batch_size_ * Nf_ * Nt_complex_ * sizeof(__half);
        sz_planar_C_float = (size_t)max_batch_size_ * Ndm_ * Nt_complex_ * sizeof(float);
    }

    size_t pool1_size = std::max({sz_real_io, sz_comp_io, sz_planar_A_half * 2});
    size_t pool2_size = std::max({sz_real_io, sz_comp_io, sz_planar_C_float * 2});

    size_t sz_gemm_out = (size_t)max_batch_size_ * Ndm_ * Nt_complex_ * sizeof(ComplexType);
    pool2_size = std::max(pool2_size, sz_gemm_out);

    std::cout << "[Memory] Allocating Transient Pools: "
              << (pool1_size / 1e9) << " GB and " << (pool2_size / 1e9) << " GB." << std::endl;

    size_t total_pool_req = pool1_size + pool2_size;
    check_memory_availability(total_pool_req);

    d_pool_1_.allocate(pool1_size);
    check_memory_availability(pool2_size);
    d_pool_2_.allocate(pool2_size);

#ifdef HAS_CUFFT_LTO_CALLBACK
    // Populate callback info now that Pool 2 is allocated
    if (callback_registered_) {
        size_t fft_elements = (size_t)max_batch_size_ * Nf_ * Nt_complex_;
        d_fft_planar_re_ = (float*)d_pool_2_.get();
        d_fft_planar_im_ = (float*)d_pool_2_.get() + fft_elements;

        FFTStoreCallbackInfo h_info;
        h_info.d_planar_re = d_fft_planar_re_;
        h_info.d_planar_im = d_fft_planar_im_;
        h_info.d_max_uint = (unsigned int*)d_max_uint_.get();
        CUDA_CHECK(cudaMemcpy(d_callback_info_.get(), &h_info,
                              sizeof(FFTStoreCallbackInfo), cudaMemcpyHostToDevice));
    }
#endif

    CUDA_CHECK(cudaEventCreate(&start_event_));
    CUDA_CHECK(cudaEventCreate(&stop_event_));

    if (fdd_compute_mode_ == "cublas") {
      CUBLAS_CHECK(cublasCreate(&blas_handle_));
      size_t ptr_array_size = (size_t)Nt_complex_ * sizeof(void*) * 3;
      d_A_pointers_.allocate(ptr_array_size);
      d_B_pointers_.allocate(ptr_array_size);
      d_C_pointers_.allocate(ptr_array_size);
    }

    if (fdd_compute_mode_ == "cublas_lt_fp8" || fdd_compute_mode_ == "cublas_lt_fp16") {
      if constexpr (!std::is_same_v<Real, float>) {
         fprintf(stderr, "Error: Low precision modes require single precision host types.\n"); exit(1);
      }

      d_constants_.allocate(3 * sizeof(float));
      float consts[] = {1.0f, 0.0f, -1.0f};
      CUDA_CHECK(cudaMemcpy(d_constants_.get(), consts, sizeof(consts), cudaMemcpyHostToDevice));

      if (fdd_compute_mode_ == "cublas_lt_fp8") {
        size_t max_blocks = ((Nf_ + 31) / 32) * ((Nt_complex_ + 31) / 32) * max_batch_size_;
        d_partial_maxes_.allocate(max_blocks * sizeof(float));
        d_final_max_val_.allocate(sizeof(float));
        running_max_val_ = 4096.0f;

        d_fp8_scales_.allocate(128);

        float scale_val = 1.0f;
        CUDA_CHECK(cudaMemcpy((char*)d_fp8_scales_.get() + 0, &scale_val, sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy((char*)d_fp8_scales_.get() + 16, &scale_val, sizeof(float), cudaMemcpyHostToDevice));
      }

      if constexpr (std::is_same_v<Real, float>) {
          cublasLtCreate(&lt_handle_);
      d_workspace_.allocate(workspace_size_);

      size_t b_size;
      if (fdd_compute_mode_ == "cublas_lt_fp8") {
          b_size = sizeof(__nv_fp8_e4m3);
      } else {
          b_size = sizeof(__half);
      }

      size_t size_planar_B = (size_t)Nf_ * Ndm_ * Nt_complex_ * b_size;
      d_planar_B_re_.allocate(size_planar_B);
      d_planar_B_im_.allocate(size_planar_B);
      }
    }

#ifdef HAS_CUTLASS_GEMM
    if (fdd_compute_mode_ == "cutlass" || fdd_compute_mode_ == "cutlass_fp6"
        || fdd_compute_mode_ == "cutlass_fp4") {
      if constexpr (!std::is_same_v<Real, float>) {
        fprintf(stderr, "Error: CUTLASS mode requires single precision host types.\n"); exit(1);
      }
      if constexpr (std::is_same_v<Real, float>) {
        cutlass_gemm_ = std::make_unique<cutlass_gemm_api::CutlassComplexGemm>();
        gemm_tune_ = gemm_tune;

        if (kernel_tune_verbosity > 0)
          cutlass_gemm_->set_kernel_tune_verbosity(kernel_tune_verbosity);
        if (kernel_tune_cache_path)
          cutlass_gemm_->set_kernel_tune_cache_path(kernel_tune_cache_path);
        cutlass_gemm_->set_strategy_tune_verbosity(strategy_tune_verbosity);
        cutlass_gemm_->set_gemm_tune_verbosity(strategy_tune_verbosity);
        if (strategy_tune_cache_path) {
          cutlass_gemm_->set_tune_cache_path(strategy_tune_cache_path);
          cutlass_gemm_->set_gemm_tune_cache_path(strategy_tune_cache_path);
        }

        // Tile-sized B allocation: only Nt_tile_ batches instead of full Nt_complex_
        int b_batch = (Nt_tile_ > 0 && Nt_tile_ < Nt_complex_) ? Nt_tile_ : Nt_complex_;
        size_t size_planar_B = (size_t)Nf_ * Ndm_ * b_batch * sizeof(__half);
        d_planar_B_re_.allocate(size_planar_B);
        d_planar_B_im_.allocate(size_planar_B);

        if (Nt_tile_ > 0 && Nt_tile_ < Nt_complex_) {
          std::cout << "[Memory] CUTLASS B tile: " << (size_planar_B * 2 / 1e9)
                    << " GB (Nt_tile=" << Nt_tile_ << " of " << Nt_complex_ << ")" << std::endl;
        }
      }
    }
#endif
  }

  // [PATCH] Pre-load phasors logic
  void prepare_phasors(const ComplexType* d_big_phasors_in, cudaStream_t stream) {
    if (phasors_ready_) return;

    // Tiled mode generates phasors on-the-fly — skip full prepare_b()
    if (is_tiled()) return;

    if (fdd_compute_mode_ == "cublas" || fdd_compute_mode_ == "kernel") return;
    float scale_B = (fdd_compute_mode_ == "cublas_lt_fp8") ? 448.0f : 1.0f;
    dim3 block(32, 32);
    dim3 grid_p((Nf_ + 31)/32, (Ndm_ + 31)/32, Nt_complex_);

    if (fdd_compute_mode_ == "cublas_lt_fp8") {
      kernel_fused_phasor_prep<__nv_fp8_e4m3><<<grid_p, block, 0, stream>>>(
									    (cufftComplex*)d_big_phasors_in,
									    d_planar_B_re_.get<__nv_fp8_e4m3>(), d_planar_B_im_.get<__nv_fp8_e4m3>(),
									    Nf_, Ndm_, Nt_complex_, scale_B);
    }
#ifdef HAS_CUTLASS_GEMM
    else if (fdd_compute_mode_ == "cutlass" || fdd_compute_mode_ == "cutlass_fp6"
             || fdd_compute_mode_ == "cutlass_fp4") {
      kernel_fused_phasor_prep_rowmajor<<<grid_p, block, 0, stream>>>(
          (cufftComplex*)d_big_phasors_in,
          d_planar_B_re_.get<__half>(), d_planar_B_im_.get<__half>(),
          Nf_, Ndm_, Nt_complex_, 448.0f);

      auto compute = cutlass_gemm_api::ComputePrecision::FP8;
      if (fdd_compute_mode_ == "cutlass_fp6") compute = cutlass_gemm_api::ComputePrecision::FP6;
      if (fdd_compute_mode_ == "cutlass_fp4") compute = cutlass_gemm_api::ComputePrecision::FP4;
      cutlass_gemm_->prepare_b(
          d_planar_B_re_.get<__half>(), d_planar_B_im_.get<__half>(),
          Ndm_, Nf_, Nt_complex_, compute, stream);
    }
#endif
    else {
      kernel_fused_phasor_prep<__half><<<grid_p, block, 0, stream>>>(
								     (cufftComplex*)d_big_phasors_in,
								     d_planar_B_re_.get<__half>(), d_planar_B_im_.get<__half>(),
								     Nf_, Ndm_, Nt_complex_, scale_B);
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));

#ifdef HAS_CUTLASS_GEMM
    // Free FP16 B buffers after prepare_b() has copied to internal FP8/MXFP storage.
    // At production sizes (2048x2048x2048), this frees ~34 GB of device memory,
    // preventing OOM on unified-memory systems like GB10 (120 GB).
    if (fdd_compute_mode_ == "cutlass" || fdd_compute_mode_ == "cutlass_fp6"
        || fdd_compute_mode_ == "cutlass_fp4") {
      size_t freed = d_planar_B_re_.ptr ? (size_t)Nf_ * Ndm_ * Nt_complex_ * sizeof(__half) * 2 : 0;
      d_planar_B_re_.free();
      d_planar_B_im_.free();
      if (freed > 0)
        printf("[Memory] Freed FP16 B buffers: %.1f GB (prepare_b completed)\n", freed / 1e9);
    }
#endif

    phasors_ready_ = true;
    std::cout << "[Memory] Persistent Phasors cached. App source can be freed." << std::endl;
  }

  /// Store precomputation table pointers for on-the-fly tiled phasor generation.
  /// Called instead of prepare_phasors() when Nt_tile tiling is active.
  void set_precomp_tables(const float* d_time_delays, const float* d_f_k_values,
                          bool use_conjugate) {
    d_time_delays_ptr_ = d_time_delays;
    d_f_k_values_ptr_ = d_f_k_values;
    use_conjugate_ = use_conjugate;
    scale_B_ = 448.0f;  // FP8 E4M3 range matching
    phasors_ready_ = true;  // Signal that phasor config is ready (generated per-tile)
    std::cout << "[Memory] Precomp tables stored for tiled phasor generation "
              << "(Nt_tile=" << Nt_tile_ << ")" << std::endl;
  }

  /// Generate phasors for a single tile [k_offset, k_offset+tile_size) into B_re/B_im.
  void generate_phasors_tile(int k_offset, int tile_size, cudaStream_t stream) {
    dim3 block(32, 32);
    dim3 grid((Nf_ + 31)/32, (Ndm_ + 31)/32, tile_size);

    kernel_generate_phasors_tiled_fp16<<<grid, block, 0, stream>>>(
        d_f_k_values_ptr_, d_time_delays_ptr_,
        d_planar_B_re_.get<__half>(), d_planar_B_im_.get<__half>(),
        Nf_, Ndm_, tile_size, k_offset,
        scale_B_, use_conjugate_);
  }

  /// Whether time-domain tiling is active for this pipeline instance.
  bool is_tiled() const { return Nt_tile_ > 0 && Nt_tile_ < Nt_complex_; }
  int Nt_tile() const { return Nt_tile_; }

  ~FddGpuPipeline() {
    cufftDestroy(fft_plan_fwd_);
    cufftDestroy(fft_plan_inv_);

    CUDA_CHECK(cudaEventDestroy(start_event_));
    CUDA_CHECK(cudaEventDestroy(stop_event_));

    if (blas_handle_) CUBLAS_CHECK(cublasDestroy(blas_handle_));
    if (lt_handle_) cublasLtDestroy(lt_handle_);
  }

template <typename T>
  void inspect_device_buffer(const char* name, T* d_ptr, size_t count, size_t offset_elements = 0) {
      std::vector<T> host_buf(count);
      T* src_ptr = d_ptr + offset_elements;

      CUDA_CHECK(cudaMemcpy(host_buf.data(), src_ptr, count * sizeof(T), cudaMemcpyDeviceToHost));

      std::cout << "[DEBUG] " << name << " (Offset " << offset_elements << "): ";
      for(size_t i=0; i<std::min(count, (size_t)5); ++i) {
          float val;
          if constexpr (std::is_same_v<T, __half>) {
              val = __half2float(host_buf[i]);
          } else if constexpr (std::is_same_v<T, __nv_fp8_e4m3>) {
              val = float(host_buf[i]);
          } else {
              val = (float)host_buf[i];
          }
          std::cout << std::fixed << std::setprecision(4) << val << " ";
      }
      std::cout << std::endl;
  }

  // Accessors for PIMPL API
  int Nt_padded() const { return Nt_padded_; }
  int Nt_complex() const { return Nt_complex_; }
  void set_verbose(bool v) { verbose_ = v; }
  bool verbose() const { return verbose_; }

  // Class-body textual include for pipeline method implementations
  #include "pipeline_impl.hpp"

  // (print_performance_metrics is private, defined at the end of pipeline_impl.hpp)

private:
  int max_batch_size_;
  int Nf_, Nt_, Ndm_, Nt_padded_, Nt_complex_;
  std::string fdd_compute_mode_;

  cufftHandle fft_plan_fwd_;
  cufftHandle fft_plan_inv_;
  cublasHandle_t blas_handle_;

  cublasLtHandle_t lt_handle_;
  size_t workspace_size_ = 256 * 1024 * 1024; // 32MB workspace

  // [PATCH] Memory Pools
  DeviceBuffer d_pool_1_;
  DeviceBuffer d_pool_2_;

  // Transient Pointers (Views into pools)
  Real* p_real_in_ = nullptr;
  Real* p_real_out_ = nullptr;
  ComplexType* p_comp_fft_ = nullptr;
  ComplexType* p_comp_ifft_ = nullptr;

  // Raw pointers for Planar/GEMM operations
  void* p_planar_A_re_ = nullptr;
  void* p_planar_A_im_ = nullptr;
  void* p_planar_C_re_ = nullptr;
  void* p_planar_C_im_ = nullptr;
  void* p_gemm_out_ = nullptr; // Legacy FP32

  DeviceBuffer d_A_pointers_;
  DeviceBuffer d_B_pointers_;
  DeviceBuffer d_C_pointers_;

  // cuBLASLt Buffers
  DeviceBuffer d_workspace_;
  DeviceBuffer d_planar_A_re_;
  DeviceBuffer d_planar_A_im_;
  DeviceBuffer d_planar_B_re_;
  DeviceBuffer d_planar_B_im_;
  DeviceBuffer d_planar_C_re_;
  DeviceBuffer d_planar_C_im_;

  bool phasors_ready_;
  bool algo_cached_ = false;
  bool verbose_ = true;
  cublasLtMatmulAlgo_t cached_algo_struct_;

  cudaEvent_t start_event_;
  cudaEvent_t stop_event_;

  DeviceBuffer d_fp8_scales_;
  DeviceBuffer d_constants_;

  // [PATCH] Fused Reduction State
  DeviceBuffer d_partial_maxes_;
  DeviceBuffer d_final_max_val_;
  float running_max_val_ = 0.0f;

#ifdef HAS_CUTLASS_GEMM
  std::unique_ptr<cutlass_gemm_api::CutlassComplexGemm> cutlass_gemm_;
  bool gemm_tune_ = true;  ///< Whether to pass tune=true to GEMM calls
#endif

  // Time-domain tiling state
  int Nt_tile_ = 0;                          // 0 = no tiling (full Nt_complex at once)
  const float* d_time_delays_ptr_ = nullptr; // [Ndm, Nf] precomp table (owned by Impl)
  const float* d_f_k_values_ptr_ = nullptr;  // [Nt_complex] precomp table (owned by Impl)
  bool use_conjugate_ = true;
  float scale_B_ = 448.0f;

#ifdef HAS_CUFFT_LTO_CALLBACK
  DeviceBuffer d_callback_info_;     // FFTStoreCallbackInfo on device
  DeviceBuffer d_max_uint_;          // unsigned int for atomicMax
  float* d_fft_planar_re_ = nullptr; // Alias into Pool 2 (FP32 Re)
  float* d_fft_planar_im_ = nullptr; // Alias into Pool 2 (FP32 Im)
  bool callback_registered_ = false;
#endif

};
