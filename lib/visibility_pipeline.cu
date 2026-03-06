#include <visibility_pipeline.h>
#include <algorithm.h>
#include <corner_turn.h>
#include <timer.h>
#include <cuda_fp16.h>

#ifdef DEDISP_API_LIB
#include <dedisp_api.h>
#endif

// Forward declarations for CUTLASS functions (defined in cutlass_interface.cu)
namespace ggp {
  void initCutlassApi(int kernel_tune_verbosity,
                      const char* kernel_tune_cache_path,
                      const char* strategy_tune_cache_path,
                      bool herk_tune,
                      int strategy_tune_verbosity = 1);
  int herkBatchedCutlassQC(const void* qc_data, void* tri_output,
                           int N, int K, int batch, cudaStream_t stream = 0);
  int herkBatchedCutlassQC_FP16(const void* qc_data, void* tri_output,
                                int N, int K, int batch, cudaStream_t stream = 0);
}

namespace ggp {

  VisibilityPipeline::VisibilityPipeline(const Config &config) {

    n_channels_ = config.n_channels;
    n_beams_ = config.n_beams;
    n_time_ = config.n_time_per_payload;
    n_antennae_ = config.n_antennae;
    n_time_inner_ = config.n_time_inner;
    n_polarizations_ = config.n_polarizations;
    engine_ = config.engine;
    Ng_ = config.Ng;
    n_beam_extract_ = config.n_beam_pixels;
    fft_prec_ = config.fft_prec;
    herk_output_fp16_ = config.herk_output_fp16;

    // Validate n_time_inner divides n_time evenly
    if (n_time_inner_ == 0) {
      errorQuda("n_time_inner must be > 0");
    }
    if (n_time_ % n_time_inner_ != 0) {
      errorQuda("n_time (%llu) must be divisible by n_time_inner (%llu)",
                (unsigned long long)n_time_, (unsigned long long)n_time_inner_);
    }
    uint64_t K = n_time_ / n_time_inner_;
    if (K < 1) {
      errorQuda("K = n_time / n_time_inner = %llu must be >= 1", (unsigned long long)K);
    }

    // Compute buffer sizes for CUTLASS path
    in_payload_size_ = n_antennae_ * n_channels_ * n_time_ * n_polarizations_;
    uint64_t n_baselines = (n_antennae_ + 1) * n_antennae_ / 2;
    uint64_t batch = n_channels_ * n_polarizations_ * n_time_inner_;
    size_t tri_elem_size = herk_output_fp16_ ? sizeof(__half) : sizeof(float);
    tri_output_size_ = batch * n_baselines * 2 * tri_elem_size;

    logQuda(QUDA_VERBOSE, "VisibilityPipeline: N=%llu, K=%llu, batch=%llu, n_time_inner=%llu\n",
            (unsigned long long)n_antennae_, (unsigned long long)K,
            (unsigned long long)batch, (unsigned long long)n_time_inner_);

    if (engine_ == QUDA_BLAS_ENGINE_CUTLASS) {
      // CUTLASS path: configure tuning on the singleton, no XEngine needed
      initCutlassApi(config.kernel_tune_verbosity,
                     config.kernel_tune_cache_path,
                     config.strategy_tune_cache_path,
                     config.herk_tune,
                     config.strategy_tune_verbosity);
      xengine_ = nullptr;
    } else {
      // cuBLAS path: use XEngine
      xe_param_ = newXEngineParam();
      xe_param_.packet_format = config.packet_format;
      xe_param_.n_payload = 1;
      xe_param_.n_antennae_per_payload = config.n_antennae;
      xe_param_.n_channels_per_payload = config.n_channels;
      xe_param_.n_time_per_payload = config.n_time_per_payload;
      xe_param_.n_time_inner = config.n_time_inner;
      xe_param_.n_polarizations = config.n_polarizations;
      xe_param_.compute_prec = config.compute_prec;
      xe_param_.output_prec = config.compute_prec;
      xe_param_.engine = config.engine;
      xe_param_.data_type = QUDA_BLAS_DATATYPE_QC;
      xe_param_.data_order = QUDA_BLAS_DATAORDER_ROW;
      xe_param_.verbosity = QUDA_SILENT;
      xe_param_.format = QUDA_XENGINE_MAT_TRI;
      xe_param_.in_location = QUDA_CPU_FIELD_LOCATION;
      xe_param_.out_location = QUDA_CPU_FIELD_LOCATION;
      xe_param_.struct_size = sizeof(xe_param_);

      xengine_ = XEngine::create(xe_param_);
    }

#ifdef DEDISP_API_LIB
    // Create dedispersion pipeline
    dedisp_api::DedispConfig dedisp_config;
    dedisp_config.Nf = static_cast<int>(config.n_channels);
    dedisp_config.Nt = static_cast<int>(config.n_time_per_payload);
    dedisp_config.Ndm = config.n_dm_trials;
    dedisp_config.f_min_MHz = config.f_min_MHz;
    dedisp_config.f_max_MHz = config.f_max_MHz;
    dedisp_config.max_dm = config.max_dm;
    dedisp_config.total_obs_time_s = config.total_obs_time_s;
    dedisp_config.compute_mode = dedisp_api::ComputeMode::CuBLAS_FP32;
    dedisp_config.max_batch_size = static_cast<int>(config.n_beams);
    dedisp_config.kernel_tune_verbosity = config.kernel_tune_verbosity;
    dedisp_config.kernel_tune_cache_path = config.kernel_tune_cache_path;

    auto *pipeline = new dedisp_api::DedispPipeline(dedisp_config);
    pipeline->initialize();
    dedisp_pipeline_ = static_cast<void *>(pipeline);
#endif

    // Create imaging pipeline
    imager_ = new imaging_pipeline::ImagingPipeline();

    if (config.profile) set_profile(true);

    init_memory();
    init_streams();
  }

  VisibilityPipeline::~VisibilityPipeline() {
    if (profile_tp_) {
      delete profile_tp_;
      profile_tp_ = nullptr;
    }
    destroy_streams();
    destroy_memory();
    if (xengine_) {
      delete xengine_;
      xengine_ = nullptr;
    }
    if (imager_) {
      delete imager_;
      imager_ = nullptr;
    }
#ifdef DEDISP_API_LIB
    if (dedisp_pipeline_) {
      delete static_cast<dedisp_api::DedispPipeline *>(dedisp_pipeline_);
      dedisp_pipeline_ = nullptr;
    }
#endif
  }

  void VisibilityPipeline::init_streams() {
    if (!streams_init_ && engine_ == QUDA_BLAS_ENGINE_CUTLASS) {
      cudaStreamCreateWithFlags(&stream_h2d_, cudaStreamNonBlocking);
      cudaStreamCreateWithFlags(&stream_compute_, cudaStreamNonBlocking);
      cudaStreamCreateWithFlags(&stream_d2h_, cudaStreamNonBlocking);
      cudaEventCreateWithFlags(&ev_h2d_done_, cudaEventDisableTiming);
      cudaEventCreateWithFlags(&ev_compute_done_, cudaEventDisableTiming);
      for (int i = 0; i < 2; i++)
        cudaEventCreateWithFlags(&ev_compute_done_db_[i], cudaEventDisableTiming);
      streams_init_ = true;
    }
  }

  void VisibilityPipeline::destroy_streams() {
    if (streams_init_) {
      // Double-buffer device memory
      for (int i = 0; i < 2; i++) {
        if (d_qc_input_db_[i]) { cudaFree(d_qc_input_db_[i]); d_qc_input_db_[i] = nullptr; }
        if (d_tri_output_db_[i]) { cudaFree(d_tri_output_db_[i]); d_tri_output_db_[i] = nullptr; }
        if (ev_compute_done_db_[i]) { cudaEventDestroy(ev_compute_done_db_[i]); ev_compute_done_db_[i] = nullptr; }
      }
      // Pinned staging
      if (h_staging_in_) { cudaFreeHost(h_staging_in_); h_staging_in_ = nullptr; }
      if (h_staging_out_) { cudaFreeHost(h_staging_out_); h_staging_out_ = nullptr; }
      // Streams and events
      if (stream_h2d_) { cudaStreamDestroy(stream_h2d_); stream_h2d_ = nullptr; }
      if (stream_compute_) { cudaStreamDestroy(stream_compute_); stream_compute_ = nullptr; }
      if (stream_d2h_) { cudaStreamDestroy(stream_d2h_); stream_d2h_ = nullptr; }
      if (ev_h2d_done_) { cudaEventDestroy(ev_h2d_done_); ev_h2d_done_ = nullptr; }
      if (ev_compute_done_) { cudaEventDestroy(ev_compute_done_); ev_compute_done_ = nullptr; }
      streams_init_ = false;
    }
  }

  void VisibilityPipeline::init_memory() {
    if (!mem_init_) {
      size_t fb_elems = n_channels_ * n_beams_ * n_time_;
      cudaMalloc(&d_filterbank_chanmajor_, fb_elems * sizeof(float));
      cudaMalloc(&d_filterbank_beammajor_, fb_elems * sizeof(float));

      // Beam output buffer for imaging pipeline
      if (n_beam_extract_ > 0) {
        size_t beam_elems = n_channels_ * n_beam_extract_;
        cudaMalloc(&d_beam_output_, beam_elems * sizeof(float));
      }

      // Allocate CUTLASS-specific device buffers
      if (engine_ == QUDA_BLAS_ENGINE_CUTLASS) {
        cudaError_t err1 = cudaMalloc(&d_qc_input_, in_payload_size_);
        cudaError_t err2 = cudaMalloc(&d_tri_output_, tri_output_size_);
        if (err1 != cudaSuccess || err2 != cudaSuccess) {
          errorQuda("CUTLASS buffer allocation failed: input=%s, output=%s",
                    cudaGetErrorString(err1), cudaGetErrorString(err2));
        }
        logQuda(QUDA_VERBOSE,
                "VisibilityPipeline CUTLASS: allocated input=%lu bytes, output=%lu bytes\n",
                in_payload_size_, tri_output_size_);
      }

      mem_init_ = true;
    }
  }

  void VisibilityPipeline::destroy_memory() {
    if (mem_init_) {
      if (d_filterbank_chanmajor_) {
        cudaFree(d_filterbank_chanmajor_);
        d_filterbank_chanmajor_ = nullptr;
      }
      if (d_filterbank_beammajor_) {
        cudaFree(d_filterbank_beammajor_);
        d_filterbank_beammajor_ = nullptr;
      }
      if (d_beam_output_) {
        cudaFree(d_beam_output_);
        d_beam_output_ = nullptr;
      }
      if (d_qc_input_) {
        cudaFree(d_qc_input_);
        d_qc_input_ = nullptr;
      }
      if (d_tri_output_) {
        cudaFree(d_tri_output_);
        d_tri_output_ = nullptr;
      }
      mem_init_ = false;
    }
  }

  void VisibilityPipeline::set_profile(bool enabled) {
    if (enabled && !profile_tp_) {
      profile_tp_ = new quda::TimeProfile("VisibilityPipeline", false);
    } else if (!enabled && profile_tp_) {
      delete profile_tp_;
      profile_tp_ = nullptr;
    }
  }

  void VisibilityPipeline::print_profile() {
    if (profile_tp_) {
      profile_tp_->Print();
      profile_tp_->TPRESET();
    }
  }

  void VisibilityPipeline::correlate(void *vis_output, void *raw_input) {
    if (engine_ == QUDA_BLAS_ENGINE_CUTLASS) {
      // CUTLASS path: raw QC bytes -> packed triangle FP32 in one call

      if (profile_tp_) profile_tp_->TPSTART(quda::QUDA_PROFILE_H2D);

      // Async H2D on compute stream
      cudaMemcpyAsync(d_qc_input_, raw_input, in_payload_size_,
                      cudaMemcpyHostToDevice, stream_compute_);

      if (profile_tp_) {
        profile_tp_->TPSTOP(quda::QUDA_PROFILE_H2D);
        profile_tp_->TPSTART(quda::QUDA_PROFILE_COMPUTE);
      }

      int N = static_cast<int>(n_antennae_);
      int K = static_cast<int>(n_time_ / n_time_inner_);
      int batch = static_cast<int>(n_channels_ * n_polarizations_ * n_time_inner_);

      logQuda(QUDA_VERBOSE, "CUTLASS correlate: N=%d, K=%d, batch=%d, in=%p, out=%p\n",
              N, K, batch, d_qc_input_, d_tri_output_);

      int err = herk_output_fp16_
          ? herkBatchedCutlassQC_FP16(d_qc_input_, d_tri_output_, N, K, batch, stream_compute_)
          : herkBatchedCutlassQC(d_qc_input_, d_tri_output_, N, K, batch, stream_compute_);
      if (err != 0) {
        errorQuda("herkBatchedCutlassQC failed with error %d", err);
      }

      if (profile_tp_) {
        profile_tp_->TPSTOP(quda::QUDA_PROFILE_COMPUTE);
        profile_tp_->TPSTART(quda::QUDA_PROFILE_D2H);
      }

      // Async D2H on compute stream (naturally ordered after HERK)
      cudaMemcpyAsync(vis_output, d_tri_output_, tri_output_size_,
                      cudaMemcpyDeviceToHost, stream_compute_);

      // Sync only this stream (not global device sync)
      cudaStreamSynchronize(stream_compute_);

      // Check for CUDA errors
      cudaError_t cerr = cudaGetLastError();
      if (cerr != cudaSuccess) {
        errorQuda("CUDA error after CUTLASS HERK: %s", cudaGetErrorString(cerr));
      }

      if (profile_tp_) {
        profile_tp_->TPSTOP(quda::QUDA_PROFILE_D2H);
        print_profile();
      }
    } else {
      // cuBLAS path: delegate to XEngine (promote -> GEMM -> triangulate)
      xengine_->compute(vis_output, raw_input);
    }
  }

  void VisibilityPipeline::corner_turn(float *d_out, const float *d_in) {
    corner_turn_nf_nb(d_out, d_in,
                      static_cast<int>(n_channels_),
                      static_cast<int>(n_beams_),
                      static_cast<int>(n_time_));
    // No global sync — caller manages synchronization via streams
  }

  int VisibilityPipeline::dedisperse(float *d_output, const float *d_input, int batch_size) {
#ifdef DEDISP_API_LIB
    if (!dedisp_pipeline_) return -1;
    auto *pipeline = static_cast<dedisp_api::DedispPipeline *>(dedisp_pipeline_);
    return pipeline->dedisperse(d_input, d_output, batch_size);
#else
    (void)d_output;
    (void)d_input;
    (void)batch_size;
    return -1; // dedisp_api not linked
#endif
  }

  void VisibilityPipeline::configure_imaging(
      const float* baseline_uv_m, int n_baselines,
      const double* freq_hz,
      const int* beam_pixels, int n_beam,
      float cell_size_rad,
      cudaStream_t stream)
  {
    if (!imager_) {
      errorQuda("ImagingPipeline not created");
    }

    int ret = imager_->configure(
        static_cast<int>(n_antennae_),
        static_cast<int>(n_channels_),
        Ng_, n_beam, fft_prec_, stream);
    if (ret != 0) {
      errorQuda("ImagingPipeline::configure() failed with error %d", ret);
    }

    ret = imager_->set_baseline_uv(baseline_uv_m, n_baselines);
    if (ret != 0) errorQuda("set_baseline_uv failed: %d", ret);

    ret = imager_->set_frequencies(freq_hz);
    if (ret != 0) errorQuda("set_frequencies failed: %d", ret);

    imager_->set_cell_size(cell_size_rad);

    if (beam_pixels && n_beam > 0) {
      ret = imager_->set_beam_pixels(beam_pixels, n_beam);
      if (ret != 0) errorQuda("set_beam_pixels failed: %d", ret);
    }

    n_beam_extract_ = n_beam;
    imaging_configured_ = true;
  }

  void VisibilityPipeline::correlate_device_only_(void *raw_input) {
    // H2D + HERK on stream_compute_, no D2H
    cudaMemcpyAsync(d_qc_input_, raw_input, in_payload_size_,
                    cudaMemcpyHostToDevice, stream_compute_);

    int N = static_cast<int>(n_antennae_);
    int K = static_cast<int>(n_time_ / n_time_inner_);
    int batch = static_cast<int>(n_channels_ * n_polarizations_ * n_time_inner_);

    int err = herk_output_fp16_
        ? herkBatchedCutlassQC_FP16(d_qc_input_, d_tri_output_, N, K, batch, stream_compute_)
        : herkBatchedCutlassQC(d_qc_input_, d_tri_output_, N, K, batch, stream_compute_);
    if (err != 0) {
      errorQuda("herkBatchedCutlassQC failed with error %d", err);
    }
  }

  void VisibilityPipeline::compute(void *output, void *input) {
    if (engine_ == QUDA_BLAS_ENGINE_CUTLASS) {
      // Stage 1: H2D + HERK (no wasted D2H — output stays on device)
      if (profile_tp_) profile_tp_->TPSTART(quda::QUDA_PROFILE_COMPUTE);
      correlate_device_only_(input);
      cudaEventRecord(ev_compute_done_, stream_compute_);
      if (profile_tp_) profile_tp_->TPSTOP(quda::QUDA_PROFILE_COMPUTE);

      // Optional: D2H of visibility output overlaps with imaging
      if (output) {
        cudaStreamWaitEvent(stream_d2h_, ev_compute_done_);
        cudaMemcpyAsync(output, d_tri_output_, tri_output_size_,
                        cudaMemcpyDeviceToHost, stream_d2h_);
      }

      // Stage 2: Imaging (grid + IFFT + beam extraction)
      if (imaging_configured_ && imager_ && n_beam_extract_ > 0) {
        if (profile_tp_) profile_tp_->TPSTART(quda::QUDA_PROFILE_PREAMBLE);

        // Imaging on stream_compute_ (naturally ordered after HERK)
        auto vis_prec = herk_output_fp16_
            ? imaging_pipeline::VisPrecision::FP16
            : imaging_pipeline::VisPrecision::FP32;
        int ret = imager_->grid_and_image(
            d_tri_output_, d_beam_output_,
            vis_prec,
            stream_compute_);
        if (ret != 0) {
          errorQuda("grid_and_image failed with error %d", ret);
        }

        size_t beam_bytes = n_channels_ * n_beam_extract_ * sizeof(float);
        cudaMemcpyAsync(d_filterbank_chanmajor_, d_beam_output_,
                        beam_bytes, cudaMemcpyDeviceToDevice, stream_compute_);

        if (profile_tp_) {
          profile_tp_->TPSTOP(quda::QUDA_PROFILE_PREAMBLE);
          profile_tp_->TPSTART(quda::QUDA_PROFILE_EPILOGUE);
        }

        // Stage 3: Corner turn [Nf, Nb, Nt] -> [Nb, Nf, Nt]
        corner_turn_nf_nb(d_filterbank_beammajor_, d_filterbank_chanmajor_,
                          static_cast<int>(n_channels_),
                          static_cast<int>(n_beams_),
                          static_cast<int>(n_time_),
                          stream_compute_);

        if (profile_tp_) {
          profile_tp_->TPSTOP(quda::QUDA_PROFILE_EPILOGUE);
          print_profile();
        }
      }

      // Sync compute stream (imaging + corner turn done)
      cudaStreamSynchronize(stream_compute_);
      // Sync D2H if we issued a visibility copy
      if (output) cudaStreamSynchronize(stream_d2h_);

    } else {
      // cuBLAS path: delegate to XEngine
      if (profile_tp_) profile_tp_->TPSTART(quda::QUDA_PROFILE_COMPUTE);
      correlate(output, input);
      if (profile_tp_) profile_tp_->TPSTOP(quda::QUDA_PROFILE_COMPUTE);

      if (imaging_configured_ && imager_ && n_beam_extract_ > 0) {
        if (profile_tp_) profile_tp_->TPSTART(quda::QUDA_PROFILE_PREAMBLE);

        auto vis_prec_cublas = herk_output_fp16_
            ? imaging_pipeline::VisPrecision::FP16
            : imaging_pipeline::VisPrecision::FP32;
        int ret = imager_->grid_and_image(
            d_tri_output_, d_beam_output_,
            vis_prec_cublas);
        if (ret != 0) {
          errorQuda("grid_and_image failed with error %d", ret);
        }

        size_t beam_bytes = n_channels_ * n_beam_extract_ * sizeof(float);
        cudaMemcpy(d_filterbank_chanmajor_, d_beam_output_,
                   beam_bytes, cudaMemcpyDeviceToDevice);

        if (profile_tp_) {
          profile_tp_->TPSTOP(quda::QUDA_PROFILE_PREAMBLE);
          profile_tp_->TPSTART(quda::QUDA_PROFILE_EPILOGUE);
        }

        corner_turn(d_filterbank_beammajor_, d_filterbank_chanmajor_);

        if (profile_tp_) {
          profile_tp_->TPSTOP(quda::QUDA_PROFILE_EPILOGUE);
          print_profile();
        }
      }
    }

    // Stage 4: Dedispersion (caller invokes dedisperse() separately
    // after accumulating enough time samples into the filterbank buffer)
  }

  void VisibilityPipeline::correlate_pipelined(void **vis_outputs,
                                                void *const *raw_inputs,
                                                int n_payloads) {
    if (engine_ != QUDA_BLAS_ENGINE_CUTLASS) {
      // Fallback: sequential correlate for non-CUTLASS engines
      for (int i = 0; i < n_payloads; i++) {
        correlate(vis_outputs[i], raw_inputs[i]);
      }
      return;
    }

    // Lazy-allocate double buffers and pinned staging
    if (!d_qc_input_db_[0]) {
      for (int i = 0; i < 2; i++) {
        cudaMalloc(&d_qc_input_db_[i], in_payload_size_);
        cudaMalloc(&d_tri_output_db_[i], tri_output_size_);
      }
      cudaMallocHost(&h_staging_in_, in_payload_size_);
      cudaMallocHost(&h_staging_out_, tri_output_size_);
    }

    int N = static_cast<int>(n_antennae_);
    int K = static_cast<int>(n_time_ / n_time_inner_);
    int batch = static_cast<int>(n_channels_ * n_polarizations_ * n_time_inner_);

    for (int p = 0; p < n_payloads; p++) {
      int buf = p & 1;  // ping-pong buffer index

      // --- H2D on stream_h2d_ ---
      // Copy to pinned staging first (ensures truly async DMA)
      memcpy(h_staging_in_, raw_inputs[p], in_payload_size_);
      cudaMemcpyAsync(d_qc_input_db_[buf], h_staging_in_, in_payload_size_,
                      cudaMemcpyHostToDevice, stream_h2d_);
      cudaEventRecord(ev_h2d_done_, stream_h2d_);

      // --- Compute on stream_compute_ ---
      // Wait for H2D of this payload to complete
      cudaStreamWaitEvent(stream_compute_, ev_h2d_done_);
      // Wait for D2H of previous payload (same buffer) to complete
      if (p >= 2) {
        cudaStreamWaitEvent(stream_compute_, ev_compute_done_db_[buf]);
      }
      int err = herk_output_fp16_
          ? herkBatchedCutlassQC_FP16(d_qc_input_db_[buf], d_tri_output_db_[buf],
                                       N, K, batch, stream_compute_)
          : herkBatchedCutlassQC(d_qc_input_db_[buf], d_tri_output_db_[buf],
                                  N, K, batch, stream_compute_);
      if (err != 0) {
        errorQuda("herkBatchedCutlassQC failed with error %d (payload %d)", err, p);
      }
      cudaEventRecord(ev_compute_done_db_[buf], stream_compute_);

      // --- D2H on stream_d2h_ ---
      cudaStreamWaitEvent(stream_d2h_, ev_compute_done_db_[buf]);
      cudaMemcpyAsync(h_staging_out_, d_tri_output_db_[buf], tri_output_size_,
                      cudaMemcpyDeviceToHost, stream_d2h_);
      // Sync D2H before overwriting staging buffer on next iteration
      cudaStreamSynchronize(stream_d2h_);
      memcpy(vis_outputs[p], h_staging_out_, tri_output_size_);
    }

    // Final sync: ensure all work is done
    cudaStreamSynchronize(stream_compute_);
    cudaStreamSynchronize(stream_d2h_);
  }

} // namespace ggp
