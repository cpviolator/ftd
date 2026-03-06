#include <ggp_internal.h>
#include <timer.h>
#include <algorithm.h>
#include <blas_lapack.h>
#include <dsa.h>
#include <cufft.h>
#include <cmath>
#include <tcc_interface.h>

using namespace quda;
using namespace device;
using namespace blas_lapack;

// Host-side kernel launch wrappers (defined in visibility_bf_kernels.cu)
// and CUTLASS HERK function (defined in cutlass_interface.cu)
namespace ggp {
  // CUTLASS HERK: raw QC bytes -> packed triangle FP32 complex
  int herkBatchedCutlassQC(const void* qc_data, void* tri_output, int N, int K, int batch, cudaStream_t stream = 0);
  // cuBLAS HERK: promoted FP32 -> full NxN -> packed triangle
  int cublas_herk_batched_qc(const float* promoted_data, float* result_data,
                             float* tri_output, int N, int K, int batch, cudaStream_t stream);
  // QC INT4 sign-magnitude -> FP32 complex interleaved (TunableKernel1D)
  void promoteQcSmToFp32(void *output, const void *input,
                         unsigned long long int N, int stream_idx);
  // Pillbox grid scatter (TunableKernel1D, scratch buffer used during tuning)
  void pillboxGridScatter(const void *vis_tri, const void *baseline_uv_m,
                          const void *freq_hz, void *uv_grid,
                          int n_baselines, int Ng, int Nf_tile,
                          int freq_offset, float cell_size_rad, int stream_idx);
  // UV taper (TunableKernel1D)
  void applyUvTaper(void *uv_grid, const void *taper,
                    int Ng, int Nf_tile, int stream_idx);
  // Beam extraction (TunableKernel1D)
  void extractBeamIntensity(const void *image, const void *beam_pixels,
                            void *beam_output, int Ng, int n_beam,
                            int Nf_tile, float norm, int stream_idx);
  // Quantise beams (TunableKernel1D)
  void quantiseBeams(const void *beam_intensity, void *output,
                     unsigned long long int N, float scale, int stream_idx);
  // QC sign-magnitude -> FTD/TCC two's complement nibble conversion (in-place)
  void convertQcSmToFtd(void *data, unsigned long long int N, int stream_idx);
}

namespace ggp {

  // ---------------------------------------------------------------
  // Constructor
  // ---------------------------------------------------------------
  VisibilityBeamformer::VisibilityBeamformer(BeamformerParam &param) {
    getProfile().TPSTART(QUDA_PROFILE_INIT);

    engine = param.engine;
    packet_format = param.packet_format;
    compute_prec = param.compute_prec;

    // Core dimensions
    n_antennae = param.n_antennae_per_payload;
    n_channels = param.n_channels_per_payload;
    n_time     = param.n_time_per_payload;
    n_pol      = param.n_polarizations;
    n_beam     = param.n_beam;
    n_baselines = (n_antennae * (n_antennae + 1)) / 2;

    // UV grid parameters
    n_grid = param.n_grid > 0 ? param.n_grid : 1024;  // default 1024
    cell_size_rad = param.cell_size_rad > 0.0 ? param.cell_size_rad : 1.0;  // placeholder
    freq_start_hz = param.freq_start_hz;
    freq_step_hz  = param.freq_step_hz;

    // User-specified tile size (0 = auto)
    n_channels_per_tile = param.n_channels_per_tile;

    // Input payload size: QC format = 1 byte per complex element
    // [n_channels * n_antennae * n_time * n_pol] bytes
    in_payload_size = n_channels * n_antennae * n_time * n_pol;

    // Visibility output: packed lower triangle, FP32 complex per channel
    // For CUTLASS QC path: one HERK per call, batch = n_channels * n_pol
    // Output = batch * n_baselines * 2 * sizeof(float)
    uint64_t herk_batch = n_channels * n_pol;
    visibility_data_size = herk_batch * n_baselines * 2 * sizeof(float);

    // Beam output
    beam_output_size = n_channels * n_beam * sizeof(float);

    logQuda(QUDA_VERBOSE, "VisibilityBeamformer: N=%lu, Nf=%lu, K=%lu, Npol=%lu\n",
            n_antennae, n_channels, n_time, n_pol);
    logQuda(QUDA_VERBOSE, "  n_baselines=%lu, n_grid=%lu, n_beam=%lu\n",
            n_baselines, n_grid, n_beam);
    logQuda(QUDA_VERBOSE, "  in_payload_size=%lu, visibility_data_size=%lu\n",
            in_payload_size, visibility_data_size);

    // Compute frequency tile size based on available memory
    compute_tile_size();

    // Allocate memory
    init_memory();

    // Create cuFFT plan
    init_fft_plan();

    getProfile().TPSTOP(QUDA_PROFILE_INIT);
  }

  // ---------------------------------------------------------------
  // Compute frequency tile size to fit in GPU memory
  // ---------------------------------------------------------------
  void VisibilityBeamformer::compute_tile_size() {
    if (n_channels_per_tile > 0) {
      // User specified a tile size
      n_tiles = (n_channels + n_channels_per_tile - 1) / n_channels_per_tile;
      return;
    }

    size_t free_mem = 0, total_mem = 0;
    cudaMemGetInfo(&free_mem, &total_mem);
    size_t usable = static_cast<size_t>(free_mem * 0.7);  // reserve for cuFFT workspace + other

    // Fixed memory: visibility output + baseline UVs + beam pixels + freq array
    size_t fixed = visibility_data_size +
                   n_baselines * 2 * sizeof(float) +
                   n_beam * 2 * sizeof(int) +
                   n_channels * sizeof(double) +
                   beam_output_size +
                   in_payload_size;

    // Per-channel memory: UV grid (Ng x Ng x 2 floats)
    size_t per_ch = static_cast<size_t>(n_grid) * n_grid * 2 * sizeof(float);

    if (usable > fixed) {
      n_channels_per_tile = std::max(static_cast<uint64_t>(1), static_cast<uint64_t>((usable - fixed) / per_ch));
    } else {
      n_channels_per_tile = 1;
    }
    n_channels_per_tile = std::min(n_channels_per_tile, n_channels);
    n_tiles = (n_channels + n_channels_per_tile - 1) / n_channels_per_tile;

    logQuda(QUDA_VERBOSE, "  Frequency tiling: %lu channels/tile, %lu tiles (free=%zu MB, per_ch=%zu KB)\n",
            n_channels_per_tile, n_tiles, free_mem / (1024*1024), per_ch / 1024);
  }

  // ---------------------------------------------------------------
  // Memory allocation
  // ---------------------------------------------------------------
  void VisibilityBeamformer::init_memory() {
    if (mem_init) return;

    // Input buffer for QC data on device
    input_copy = device_pinned_malloc(in_payload_size);

    // Visibility output from HERK: [n_channels * n_pol * n_baselines * 2] floats
    visibility_data = device_pinned_malloc(visibility_data_size);

    // UV grid: [n_channels_per_tile * Ng * Ng * 2] floats
    uv_grid_size = n_channels_per_tile * n_grid * n_grid * 2 * sizeof(float);
    uv_grid = device_malloc(uv_grid_size);

    // Beam output: [n_channels * n_beam] floats
    beam_output = device_pinned_malloc(beam_output_size);

    // Baseline UVs: [n_baselines * 2] floats (set via set_baseline_uvw)
    baseline_uv_m = static_cast<float*>(device_pinned_malloc(n_baselines * 2 * sizeof(float)));

    // Beam pixel map: [n_beam * 2] ints (set via set_beam_pixels)
    beam_pixel_map = static_cast<int*>(device_pinned_malloc(n_beam * 2 * sizeof(int)));

    // Frequency array: [n_channels] doubles (set via set_frequencies)
    freq_hz_d = static_cast<double*>(device_pinned_malloc(n_channels * sizeof(double)));

    // Taper weights: allocated lazily in set_taper()
    taper_weights = nullptr;

    // cuBLAS path: needs promoted FP32 buffer and full NxN result buffer
    if (engine == QUDA_BLAS_ENGINE_CUBLAS) {
      // Promoted data: [batch x N x K] complex = [batch x N x K x 2] floats
      uint64_t herk_batch = n_channels * n_pol;
      promoted_data = static_cast<float*>(device_malloc(herk_batch * n_antennae * n_time * 2 * sizeof(float)));
      // Full NxN result: [batch x N x N] complex = [batch x N x N x 2] floats
      result_data = static_cast<float*>(device_malloc(herk_batch * n_antennae * n_antennae * 2 * sizeof(float)));
    }

    mem_init = true;
  }

  // ---------------------------------------------------------------
  // cuFFT plan lifecycle
  // ---------------------------------------------------------------
  void VisibilityBeamformer::init_fft_plan() {
    if (fft_init) return;

    int rank = 2;
    int n[2] = {static_cast<int>(n_grid), static_cast<int>(n_grid)};
    int batch = static_cast<int>(n_channels_per_tile);

    cufftResult err = cufftPlanMany(
        &fft_plan, rank, n,
        NULL, 1, 0,    // contiguous input
        NULL, 1, 0,    // contiguous output
        CUFFT_C2C, batch);
    if (err != CUFFT_SUCCESS) {
      errorQuda("cufftPlanMany failed with error %d", err);
    }

    fft_init = true;
    logQuda(QUDA_VERBOSE, "  cuFFT 2D C2C plan: %lux%lu, batch=%lu\n", n_grid, n_grid, n_channels_per_tile);
  }

  void VisibilityBeamformer::destroy_fft_plan() {
    if (fft_init) {
      cufftDestroy(fft_plan);
      fft_plan = 0;
      fft_init = false;
    }
  }

  // ---------------------------------------------------------------
  // Configuration methods (call once per pointing / antenna config)
  // ---------------------------------------------------------------
  void VisibilityBeamformer::set_baseline_uvw(const float *baseline_uv_metres_host, int n_bl) {
    if (n_bl != static_cast<int>(n_baselines)) {
      errorQuda("set_baseline_uvw: n_baselines mismatch (%d vs %lu)", n_bl, n_baselines);
    }
    qudaMemcpy(baseline_uv_m, baseline_uv_metres_host,
              n_baselines * 2 * sizeof(float), qudaMemcpyHostToDevice);
    logQuda(QUDA_VERBOSE, "VisibilityBeamformer: set %lu baseline UV coordinates\n", n_baselines);
  }

  void VisibilityBeamformer::set_beam_pixels(const int *pixel_coords_host, int n_beams) {
    if (n_beams != static_cast<int>(n_beam)) {
      errorQuda("set_beam_pixels: n_beam mismatch (%d vs %lu)", n_beams, n_beam);
    }
    qudaMemcpy(beam_pixel_map, pixel_coords_host,
              n_beam * 2 * sizeof(int), qudaMemcpyHostToDevice);
    logQuda(QUDA_VERBOSE, "VisibilityBeamformer: set %lu beam pixel coordinates\n", n_beam);
  }

  void VisibilityBeamformer::set_taper(const float *taper_host) {
    if (!taper_weights) {
      taper_weights = static_cast<float*>(device_pinned_malloc(n_grid * n_grid * sizeof(float)));
    }
    qudaMemcpy(taper_weights, taper_host,
              n_grid * n_grid * sizeof(float), qudaMemcpyHostToDevice);
    logQuda(QUDA_VERBOSE, "VisibilityBeamformer: set %lux%lu UV taper\n", n_grid, n_grid);
  }

  void VisibilityBeamformer::set_frequencies(const double *freq_hz_host) {
    qudaMemcpy(freq_hz_d, freq_hz_host,
              n_channels * sizeof(double), qudaMemcpyHostToDevice);
    logQuda(QUDA_VERBOSE, "VisibilityBeamformer: set %lu channel frequencies\n", n_channels);
  }

  // ---------------------------------------------------------------
  // Main pipeline: compute()
  // ---------------------------------------------------------------
  void VisibilityBeamformer::compute(void *output_data, void *input_data) {
    getProfile().TPSTART(QUDA_PROFILE_COMPUTE);

    // -------------------------------------------
    // Stage 1: HERK -- correlate QC input
    // -------------------------------------------
    if (engine == QUDA_BLAS_ENGINE_CUTLASS) {
      // CUTLASS path: copy QC data to device, call herkBatchedCutlassQC directly.
      // This takes raw QC bytes -> packed lower triangle FP32 complex.
      qudaMemcpy(input_copy, input_data, in_payload_size, qudaMemcpyHostToDevice);

      int N = static_cast<int>(n_antennae);
      int K = static_cast<int>(n_time);
      int batch = static_cast<int>(n_channels * n_pol);
      int err = herkBatchedCutlassQC(input_copy, visibility_data, N, K, batch);
      if (err != 0) {
        errorQuda("herkBatchedCutlassQC failed with error %d", err);
      }
    } else if (engine == QUDA_BLAS_ENGINE_TCC) {
      // TCC path: convert QC nibble convention, then call TCC correlator
      qudaMemcpy(input_copy, input_data, in_payload_size, qudaMemcpyHostToDevice);

      // Convert sign-magnitude (high=Re, low=Im) -> two's complement (low=Re, high=Im)
      convertQcSmToFtd(input_copy, in_payload_size, 0);

      // TCC correlate: n_time_inner=1 (beamformer integrates all time)
      correlateTCC(visibility_data, input_copy,
                   static_cast<unsigned>(n_antennae),
                   static_cast<unsigned>(n_channels),
                   static_cast<unsigned>(n_time),
                   /*n_time_inner=*/1,
                   static_cast<unsigned>(n_pol), 0);
    } else {
      // cuBLAS path: promote QC -> FP32, cuBLAS batched GEMM, extract triangle
      qudaMemcpy(input_copy, input_data, in_payload_size, qudaMemcpyHostToDevice);

      // Promote QC INT4 sign-magnitude -> FP32 complex interleaved
      promoteQcSmToFp32(promoted_data, input_copy, in_payload_size, 0);

      int N = static_cast<int>(n_antennae);
      int K = static_cast<int>(n_time);
      int batch = static_cast<int>(n_channels * n_pol);
      int err = cublas_herk_batched_qc(
          promoted_data, result_data,
          static_cast<float*>(visibility_data), N, K, batch, nullptr);
      if (err != 0) {
        errorQuda("cublas_herk_batched_qc failed with error %d", err);
      }
    }

    // Synchronize after HERK before gridding
    qudaDeviceSynchronize();

    // -------------------------------------------
    // Stages 2-5: Frequency-tiled pipeline
    // -------------------------------------------
    const uint64_t tri_stride = n_baselines * 2;  // floats per channel in packed triangle
    const float norm = 1.0f / (static_cast<float>(n_grid) * static_cast<float>(n_grid));

    for (uint64_t tile = 0; tile < n_tiles; tile++) {
      const uint64_t f_start = tile * n_channels_per_tile;
      const uint64_t Nf_tile = std::min(n_channels_per_tile, n_channels - f_start);

      // For tiles smaller than the plan batch, we need a temp cuFFT plan
      bool need_temp_plan = (Nf_tile < n_channels_per_tile) && fft_init;
      int temp_fft_plan = 0;

      if (need_temp_plan) {
        int rank = 2;
        int n_fft[2] = {static_cast<int>(n_grid), static_cast<int>(n_grid)};
        cufftResult err = cufftPlanMany(
            &temp_fft_plan, rank, n_fft,
            NULL, 1, 0, NULL, 1, 0,
            CUFFT_C2C, static_cast<int>(Nf_tile));
        if (err != CUFFT_SUCCESS) {
          errorQuda("cufftPlanMany (temp) failed with error %d", err);
        }
      }

      int active_plan = need_temp_plan ? temp_fft_plan : fft_plan;

      // Stage 2: Zero UV grid and scatter visibilities
      cudaMemset(uv_grid, 0, Nf_tile * n_grid * n_grid * 2 * sizeof(float));

      // Pointer to visibility data for this tile's channels
      // Layout: [n_channels * n_pol * n_baselines * 2] floats
      // Treat n_pol as folded into channels for the baseline index
      const float *vis_tile = static_cast<const float*>(visibility_data) + f_start * n_pol * tri_stride;

      pillboxGridScatter(
          vis_tile, baseline_uv_m, freq_hz_d, uv_grid,
          static_cast<int>(n_baselines * n_pol),
          static_cast<int>(n_grid),
          static_cast<int>(Nf_tile),
          static_cast<int>(f_start),
          static_cast<float>(cell_size_rad),
          0);

      // Stage 3: UV taper (optional)
      if (taper_weights) {
        applyUvTaper(uv_grid, taper_weights,
                     static_cast<int>(n_grid),
                     static_cast<int>(Nf_tile), 0);
      }

      // Stage 4: 2D IFFT (in-place)
      cufftResult fft_err = cufftExecC2C(
          active_plan,
          reinterpret_cast<cufftComplex*>(uv_grid),
          reinterpret_cast<cufftComplex*>(uv_grid),
          CUFFT_INVERSE);
      if (fft_err != CUFFT_SUCCESS) {
        errorQuda("cufftExecC2C failed with error %d", fft_err);
      }

      // Stage 5: Beam extraction
      float *beam_tile = static_cast<float*>(beam_output) + f_start * n_beam;
      extractBeamIntensity(
          uv_grid, beam_pixel_map, beam_tile,
          static_cast<int>(n_grid),
          static_cast<int>(n_beam),
          static_cast<int>(Nf_tile),
          norm, 0);

      if (need_temp_plan) {
        cufftDestroy(temp_fft_plan);
      }
    }

    // Copy beam output to host
    qudaDeviceSynchronize();
    qudaMemcpy(output_data, beam_output, beam_output_size, qudaMemcpyDeviceToHost);

    getProfile().TPSTOP(QUDA_PROFILE_COMPUTE);
  }

  // ---------------------------------------------------------------
  // Memory cleanup
  // ---------------------------------------------------------------
  void VisibilityBeamformer::destroy_memory() {
    if (mem_init) {
      destroy_fft_plan();

      if (input_copy) device_pinned_free(input_copy);
      if (visibility_data) device_pinned_free(visibility_data);
      if (uv_grid) device_free(uv_grid);
      if (beam_output) device_pinned_free(beam_output);
      if (baseline_uv_m) device_pinned_free(baseline_uv_m);
      if (beam_pixel_map) device_pinned_free(beam_pixel_map);
      if (taper_weights) device_pinned_free(taper_weights);
      if (freq_hz_d) device_pinned_free(freq_hz_d);
      if (promoted_data) device_free(promoted_data);
      if (result_data) device_free(result_data);

      input_copy = nullptr;
      visibility_data = nullptr;
      uv_grid = nullptr;
      beam_output = nullptr;
      baseline_uv_m = nullptr;
      beam_pixel_map = nullptr;
      taper_weights = nullptr;
      freq_hz_d = nullptr;
      promoted_data = nullptr;
      result_data = nullptr;

      mem_init = false;
    }
  }

  VisibilityBeamformer::~VisibilityBeamformer() {
    destroy_memory();
  }

}
