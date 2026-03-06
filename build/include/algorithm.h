#pragma once

#include <vector>
#include "memory"
#include "ggp.h"
#include "ggp_internal.h"
#include "data_structures.h"
#include "timer.h"

namespace ggp {

  // Static instance of PipelineArrays that can
  // be shared among all classes and algorithms
  static PipelineArrays *pipeline_arrays = nullptr;
  static bool pipeline_arrays_init = false;
  static void init_pipeline_arrays() {
    if(!pipeline_arrays_init) {
      pipeline_arrays = new PipelineArrays();
      pipeline_arrays_init = true;
    }
  }

  static void destroy_pipeline_arrays() {
    if(pipeline_arrays_init) {
      pipeline_arrays->destroy();
      pipeline_arrays_init = false;
    }
  }
    
  class Dedispersion {

  private:
    ChpcDedispersionLib dedisp_lib;    
    
  protected:
  public:

    Dedispersion(DedispParam &param);

    static Dedispersion *create(DedispParam &param) { return new Dedispersion(param); }

    // Compute cycle
    void compute(void *output_data, void *input_data, DedispParam &param);
    
    ~Dedispersion();
    
  };
  
  class XEngine {
    
  private:
    bool mem_init = false;
    bool blas_init = false;
    QudaBLASParam blas_param;
    QudaPacketFormat packet_format;
    QudaXEngineMatFormat mat_format;

    QudaFieldLocation in_location;
    QudaFieldLocation out_location;

    // Stored for TCC engine path
    unsigned n_pols_stored = 0;
    unsigned n_channels_stored = 0;
    unsigned n_antennae_stored = 0;
    unsigned n_time_stored = 0;
    unsigned n_time_inner_stored = 0;
    
    // Points to part of input and output arrays
    // for the current computation
    void *current_input_data_ptr;
    void *current_output_data_ptr;
    
    size_t output_data_size;
    size_t result_data_size;
    int compute_data_size;
    QudaPrecision compute_prec;
    QudaPrecision output_prec;
    
  protected:

    // Planar input arrays for half prec blas
    quda::quda_ptr input_data_real_ptr;
    quda::quda_ptr input_data_imag_ptr;
    // Interleaved input array
    quda::quda_ptr input_data_full_ptr;
    
    void *result_real = nullptr;
    void *result_imag = nullptr;
    void *input_copy = nullptr;
    void *output_copy = nullptr;
    void *tri_output = nullptr;
    void *result_data = nullptr;
    
    uint64_t n_payload;
    uint64_t in_payload_size;
    uint64_t out_payload_size;
    uint64_t tri_output_payload_size;
    uint64_t result_payload_size;
    
    void init_memory();
    void destroy_memory();
    void populate_blas_param(XEngineParam &param);
    
  public:

    // constructor
    XEngine(XEngineParam &param);

    static XEngine *create(XEngineParam &param) { return new XEngine(param); }
    
    // Compute cycle
    void compute(void *output_data, void *input_data);

    //void compute_half_prec();
    
    ~XEngine();
    
  };
  
  class VoltageBeamformer {
    
  private:
    bool mem_init = false;
    bool blas_init = false;
    QudaBLASParam blas_param;
    QudaBLASEngine engine;
    QudaPacketFormat packet_format;
    
    size_t output_data_size;
    size_t result_data_size;
    size_t compute_data_size;
    QudaPrecision compute_prec;
    QudaPrecision output_prec;

  protected:

    void init_memory();
    void populate_blas_param(BeamformerParam &param);

    // Interleaved input array
    quda::quda_ptr input_data_full_ptr_A;
    quda::quda_ptr input_data_full_ptr_B;

    void *result_data_A = nullptr;
    void *result_data_B = nullptr;
    void *weights_data_A = nullptr;
    void *weights_data_B = nullptr;    
    void *flagants_data = nullptr;
    
    void *ib_sum_data = nullptr;
    void *ps_data = nullptr;
    void *input_copy = nullptr;
    void *output_copy = nullptr;
    void *input_copy_110 = nullptr;
    
    uint64_t n_payload;
    uint64_t in_payload_size;
    uint64_t out_payload_size;
    uint64_t result_payload_size;

    uint64_t n_arm;
    uint64_t n_pol;
    uint64_t n_freq;
    uint64_t n_pos;
    uint64_t n_calib;
    uint64_t n_weights_per_pol;
    uint64_t n_time_per_payload;
    uint64_t n_antennae_per_payload;
    uint64_t n_channels_per_payload;
    
    double sfreq;
    double wfreq;
    double declination;
    
    uint64_t ib_sum_elems;
    uint64_t n_thread_ib_sum;
    uint64_t batches_ib_sum;
    
    uint64_t ps_data_elems;
    uint64_t n_beam;
    uint64_t n_time_power_sum;;

    uint64_t n_thread_sict;
    uint64_t n_channels_inner;
    uint64_t n_time_inner;
    
    // Load new voltaage data
    void refresh_input(void *input_data_host, uint64_t stream_idx);
    
  public:

    VoltageBeamformer() = default;
    
    // constructor
    VoltageBeamformer(BeamformerParam &param);

    // Compute cycle
    void compute(void *output_data_host, void *input_data_host);
    
    // Load new weights data from host
    void refresh_weights(void *weights_A, void *weights_B);

    // Compute weights from raw input
    void compute_weights(void *input_data_host, void *flagants_host, int n_flags, float dec, float sfreq);
    
    // Load new flagants data from host
    void refresh_flagants(void *flagants_data);

    void destroy_memory();
    ~VoltageBeamformer();
  };

  class VisibilityBeamformer {

  private:
    bool mem_init = false;
    bool fft_init = false;
    QudaBLASEngine engine;
    QudaPacketFormat packet_format;
    QudaPrecision compute_prec;

    // Dimensions
    uint64_t n_antennae, n_channels, n_time, n_pol;
    uint64_t n_baselines;           // N*(N+1)/2
    uint64_t n_grid;                // Ng
    uint64_t n_beam;
    uint64_t n_channels_per_tile;   // frequency tile size
    uint64_t n_tiles;               // ceil(n_channels / n_channels_per_tile)
    double cell_size_rad;
    double freq_start_hz, freq_step_hz;

    // Size tracking
    uint64_t in_payload_size;
    uint64_t visibility_data_size;  // packed triangle output bytes
    uint64_t uv_grid_size;          // per tile, bytes
    uint64_t beam_output_size;      // total across all channels, bytes

    // Device buffers
    void *input_copy = nullptr;
    void *visibility_data = nullptr;    // [Nf x n_baselines x 2] float from HERK
    void *uv_grid = nullptr;            // [Nf_tile x Ng x Ng x 2] float
    void *beam_output = nullptr;        // [Nf x n_beam] float
    float *baseline_uv_m = nullptr;     // [n_baselines x 2] float
    int  *beam_pixel_map = nullptr;     // [n_beam x 2] int (col, row)
    float *taper_weights = nullptr;     // [Ng x Ng] float (optional)
    double *freq_hz_d = nullptr;        // [n_channels] double on device

    // cuFFT plan handle (cufftHandle is an int)
    int fft_plan = 0;

    // cuBLAS path buffers (only allocated when engine != CUTLASS)
    float *promoted_data = nullptr;    // [batch x N x K x 2] float (FP32 interleaved)
    float *result_data = nullptr;      // [batch x N x N x 2] float (full Hermitian)

  protected:
    void init_memory();
    void init_fft_plan();
    void destroy_fft_plan();
    void compute_tile_size();

  public:
    VisibilityBeamformer() = default;
    VisibilityBeamformer(BeamformerParam &param);

    // Main pipeline: input = QC packets (host), output = beam intensities (host)
    void compute(void *output_data, void *input_data);

    // Configuration (call once per pointing / antenna config change)
    void set_baseline_uvw(const float *baseline_uv_metres_host, int n_baselines);
    void set_beam_pixels(const int *pixel_coords_host, int n_beams);   // [n_beam x 2] (col,row)
    void set_taper(const float *taper_host);                           // [Ng x Ng]
    void set_frequencies(const double *freq_hz_host);                  // [n_channels]

    void destroy_memory();
    ~VisibilityBeamformer();
  };

  
}
