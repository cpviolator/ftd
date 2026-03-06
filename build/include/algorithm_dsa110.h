#pragma once

#include <vector>
#include "memory"
#include "ggp.h"
#include "ggp_internal.h"
#include "data_structures.h"
#include "timer.h"

namespace ggp {

  class VoltageBeamformer110 {
    
  private:
    bool mem_init = false;
    bool blas_init = false;
    QudaBLASParam blas_param;
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

    // 110 specilities
    void *input_copy_110 = nullptr;
    void *weights_data_A_r = nullptr;
    void *weights_data_B_r = nullptr;  
    void *weights_data_A_i = nullptr;
    void *weights_data_B_i = nullptr;  
        
    uint64_t n_payload;
    uint64_t in_payload_size;
    uint64_t out_payload_size;
    uint64_t result_payload_size;

    uint64_t n_arm;
    uint64_t n_pol;
    uint64_t n_freq;
    uint64_t n_ant;
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
    
    // Load new voltage data
    void refresh_input(void *input_data_host, uint64_t stream_idx);
    
  public:

    VoltageBeamformer110() = default;
    
    // constructor
    VoltageBeamformer110(BeamformerParam &param);

    // Compute cycle
    //void compute(void *output_data_host, void *input_data_host);
    
    // Load new weights data from host
    void refresh_weights(void *weights_A, void *weights_B);

    // Load new flagants data from host
    void refresh_flagants(void *flagants_data);

    // DSA110 specialities
    void compute_by_arm(void *output_data_host, void *input_data_host);
    void refresh_input_by_arm(void *input_data_host, uint64_t stream_idx);
    void compute_weights_by_arm(void *input_data_host, void *flagants_host, int n_flags, float dec, float sfreq);

    void destroy_memory();
    ~VoltageBeamformer110();
  };

}
