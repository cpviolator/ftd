#include <ggp_internal.h>
#include <timer.h>
#include <algorithm.h>
#include <blas_lapack.h>
#include <dsa.h>

using namespace quda;
using namespace device;
using namespace blas_lapack;

namespace ggp {

  // Voltage Beamformer constructor
  VoltageBeamformer::VoltageBeamformer(BeamformerParam &param) {
    getProfile().TPSTART(QUDA_PROFILE_INIT);
    // Deduce all array sizes and compute parameters
    //---------------------------------------------------------
    // Sizes of arrays in bytes
    // Infer input vector length in char sized (4,4)bit complex
    compute_prec = param.compute_prec;

    // Hardcode to single prec for now
    compute_data_size = 4;
    n_payload = param.n_payload;
    packet_format = param.packet_format;

    // BLAS engine
    engine = param.engine;
    
    // Fill and deduce runtime parameters
    n_time_per_payload = param.n_time_per_payload;
    n_time_inner = param.n_time_inner;
    n_channels_per_payload = param.n_channels_per_payload;
    n_channels_inner = param.n_channels_inner;
    n_antennae_per_payload = param.n_antennae_per_payload;    
    n_beam = param.n_beam;
    n_time_power_sum = param.n_time_power_sum;
    n_pol = param.n_polarizations;
    n_arm = param.n_arm;
    sfreq = param.sfreq;
    wfreq = param.wfreq;
    declination = param.declination;
    
    n_freq = n_channels_per_payload/n_channels_inner;
    n_calib = 2 * n_antennae_per_payload * n_freq * n_pol;
    n_weights_per_pol = (n_freq * n_antennae_per_payload * n_beam)/n_arm;

    // Check parameters for sanity
    if(n_channels_inner < 1) errorQuda("n_channels_inner = %lu must be greater than 1", n_channels_inner);
    if(n_channels_per_payload < 1) errorQuda("n_channels_per_payload = %lu must be greater than n_channels_inner = %lu", n_channels_per_payload, n_channels_inner);
    if(n_time_inner < 2 || n_time_inner %2 != 0) errorQuda("n_time_inner = %lu must be a non-zero positive multiple of 2", n_time_inner);
    if(n_time_power_sum != 4) errorQuda("Unsupported n_time_power_sum = %lu (Must use 4 during development)", n_time_power_sum);
    
    // N complex elements in input block
    in_payload_size = (n_time_per_payload *
		       n_antennae_per_payload *
		       n_channels_per_payload *
		       n_pol);

    // This is always unsigned char
    output_data_size = sizeof(unsigned char);

    out_payload_size = output_data_size * ((n_time_per_payload/(n_time_power_sum * n_time_inner)) *
					   n_channels_per_payload/n_channels_inner *
					   n_beam);
    
    // Make device data arrays for CUBLAS result
    result_payload_size = 2 * compute_data_size * (param.n_channels_per_payload *
						   param.n_beam *
						   param.n_time_per_payload);
    
    logQuda(QUDA_VERBOSE, "in_payload_size = %lu\n", in_payload_size);
    logQuda(QUDA_VERBOSE, "out_payload_size = %lu\n", out_payload_size);
    logQuda(QUDA_VERBOSE, "result_payload_size = %lu\n", result_payload_size);
    //---------------------------------------------------------

    // Incoherent beam
    batches_ib_sum = n_channels_per_payload * n_time_per_payload;
    n_thread_ib_sum = (n_time_per_payload *
                       (n_antennae_per_payload / n_arm) *
                       n_channels_per_payload);

    // Power sum pareameters
    ps_data_elems = (n_time_per_payload/n_time_power_sum) * n_channels_per_payload * n_beam/n_arm;

    // Inner sum parameters
    n_thread_sict = ((n_time_per_payload/(n_time_inner*n_time_power_sum)) *
		     (n_channels_per_payload / n_channels_inner) * n_beam);
    
    // Allocate memory
    init_memory();

    // Populate BLAS param
    populate_blas_param(param);
    getProfile().TPSTOP(QUDA_PROFILE_INIT);
  }

  void VoltageBeamformer::refresh_weights(void *weights_data_A_host, void *weights_data_B_host) {
    qudaMemcpy(weights_data_A, weights_data_A_host, 2 * n_weights_per_pol * sizeof(float), qudaMemcpyHostToDevice);
    qudaMemcpy(weights_data_B, weights_data_B_host, 2 * n_weights_per_pol * sizeof(float), qudaMemcpyHostToDevice);
    if(getVerbosity() >= QUDA_SUMMARIZE) printfQuda("Refreshed weights.\n");
  }

  // FIX ME: Add compile time variable for default stream index
  void VoltageBeamformer::refresh_input(void *input_data_host, uint64_t stream_idx) {
    // Copy input data to device memory, promote to compute prec
    qudaMemcpyAsync(input_copy, input_data_host, in_payload_size, qudaMemcpyHostToDevice, get_stream(stream_idx));

    if (getVerbosity() == QUDA_DEBUG_VERBOSE) printfQuda("Populating two float arrays of length %lu from (4b,4b) char of length %lu\n", 2*in_payload_size/n_pol, in_payload_size);
    
    // Promote char data for CUBLAS
    switch(packet_format) {
    case QUDA_PACKET_FORMAT_DSA2K:
      reorderDataPolT(input_data_full_ptr_A.data(), input_data_full_ptr_B.data(), input_copy, n_time_per_payload, n_time_inner, n_channels_per_payload, n_pol, n_antennae_per_payload, compute_prec, stream_idx);
      break;
    case QUDA_PACKET_FORMAT_CASM:
      reorderDataPolT(input_data_full_ptr_A.data(), input_data_full_ptr_B.data(), input_copy, n_time_per_payload, n_time_inner, n_channels_per_payload, n_pol, n_antennae_per_payload, compute_prec, stream_idx);	    
      break;
    case QUDA_PACKET_FORMAT_DSA110: 
      // Promote char data for CUBLAS
      // initial data: [NPACKETS_PER_BLOCK, NANTS, NCHAN_PER_PACKET, 2 times, 2 pol, 4-bit complex]
      // final data: need to split by NANTS.      
      for (uint64_t iArm=0; iArm<n_arm; iArm++) {

	int inc_size = (n_antennae_per_payload/n_arm)*n_channels_per_payload*n_time_inner * n_pol;
	int cur_inc = 0;
	
	for (uint64_t i=0; i<n_time_per_payload/n_time_inner; i++) {
	  qudaMemcpyAsync((char*)input_copy + i * (n_antennae_per_payload/n_arm) * n_channels_per_payload * n_time_inner * n_pol,
			  (char*)input_copy + (i * n_antennae_per_payload * n_channels_per_payload * n_time_inner * n_pol +
					       iArm * (n_antennae_per_payload/n_arm) * n_channels_per_payload * n_time_inner * n_pol),
			  (n_antennae_per_payload/n_arm) * n_channels_per_payload * n_time_inner * n_pol,
			  qudaMemcpyDeviceToDevice, get_stream(stream_idx));
	  //printf("current increment: %d %d %d\n", iArm, i, cur_inc);
	  cur_inc += 2*inc_size + iArm*inc_size;
	}
	reorderDataPolT((float*)input_data_full_ptr_A.data() + iArm*(in_payload_size/n_arm),
			(float*)input_data_full_ptr_B.data() + iArm*(in_payload_size/n_arm),
			(char*)input_copy, n_time_per_payload, n_time_inner, n_channels_per_payload, n_pol, n_antennae_per_payload/n_arm, compute_prec, stream_idx);	  
	
      }      
      break;
      
    default: errorQuda("Unknown packet format %d\n", packet_format);
    }
  }
  
  void VoltageBeamformer::refresh_flagants(void *flagants_data_host) {
    qudaMemcpy(flagants_data, flagants_data_host, n_antennae_per_payload * sizeof(int), qudaMemcpyHostToDevice);
    if(getVerbosity() >= QUDA_SUMMARIZE) printfQuda("Refreshed flagants.\n");
  }
  
  void VoltageBeamformer::compute(void *output_data_host, void *input_data_host) {
    
    if(packet_format == QUDA_PACKET_FORMAT_DSA110) {
      blas_param.engine = engine;
      blas_param.data_order = QUDA_BLAS_DATAORDER_ROW;
      blas_param.blas_type = QUDA_BLAS_GEMM;
      blas_param.trans_a = QUDA_BLAS_OP_N;
      blas_param.trans_b = QUDA_BLAS_OP_T;
      blas_param.m = n_beam/n_arm;
      blas_param.n = n_channels_inner * n_time_per_payload;
      blas_param.k = n_antennae_per_payload/n_arm;
      blas_param.alpha = 1.0;
      blas_param.beta = 0.0;
      blas_param.lda = blas_param.k;
      blas_param.ldb = blas_param.k;
      blas_param.ldc = blas_param.n;    
        
      n_thread_ib_sum = (n_time_per_payload *
			 (n_antennae_per_payload/n_arm) *
			 n_channels_per_payload);
    
      batches_ib_sum = n_channels_per_payload * n_time_per_payload;

      // Inner sum parameters
      n_thread_sict = (((n_time_per_payload/(n_time_inner*n_time_power_sum)) *
			(n_channels_per_payload / n_channels_inner) * n_beam)/n_arm);
    
      // Compute pointer offsets (in bytes) for second arm.
      uint64_t input_offset = (in_payload_size * compute_data_size)/n_arm;
      uint64_t result_offset = result_payload_size/n_arm;
      uint64_t output_offset = out_payload_size/n_arm;
    
      uint64_t weights_offset = (2 * n_weights_per_pol * sizeof(float))/n_arm;
      uint64_t flagants_offset = (n_antennae_per_payload * sizeof(int))/n_arm;

      //qudaDeviceSynchronize();
    
      for (uint64_t stream_idx = 0; stream_idx<n_payload; stream_idx++) {
	logQuda(QUDA_VERBOSE, "Launching payload %lu of %lu\n", stream_idx, n_payload);
      
	// Load data from host. This will place arm0 data in the first half, and arm1 data
	// in the second half
	refresh_input(input_data_host, stream_idx);
      
	for(uint64_t iArm = 0; iArm<n_arm; iArm++) {
	  switch(compute_prec) {
	  case QUDA_DOUBLE_PRECISION:
	  case QUDA_SINGLE_PRECISION:
	    // Interleaved data
	    // (a - ib)(c + id)
	    blas_param.stream_idx = stream_idx;

	    logQuda(QUDA_DEBUG_VERBOSE, "Size of input data array in bytes %ld\n", in_payload_size * compute_data_size);
	    logQuda(QUDA_DEBUG_VERBOSE, "Size of input data array offset in bytes %ld\n", input_offset);
	    logQuda(QUDA_DEBUG_VERBOSE, "Size of input data array and offset ratio in bytes %ld\n", input_offset/in_payload_size);
	  
	    logQuda(QUDA_DEBUG_VERBOSE, "Size of weight data array in bytes %ld\n", 2 * n_weights_per_pol * sizeof(float));
	    logQuda(QUDA_DEBUG_VERBOSE, "Size of weight data array offset in bytes %ld\n", weights_offset);
	    logQuda(QUDA_DEBUG_VERBOSE, "Size of weight data array and offset ratio in bytes %ld\n", (2 * n_weights_per_pol * sizeof(float))/ weights_offset);

	    switch(engine) {
	    case QUDA_BLAS_ENGINE_CUBLAS:
	      blas_lapack::native::stridedBatchGEMM((char*)weights_data_A + weights_offset*iArm,
						    (char*)input_data_full_ptr_A.data() + input_offset*iArm,
						    (char*)result_data_A + result_offset*iArm,
						    blas_param, QUDA_CUDA_FIELD_LOCATION);
	      
	      blas_lapack::native::stridedBatchGEMM((char*)weights_data_B + weights_offset*iArm,
						    (char*)input_data_full_ptr_B.data() + input_offset*iArm,
						    (char*)result_data_B + result_offset*iArm,
						    blas_param, QUDA_CUDA_FIELD_LOCATION);
	      break;
	    case QUDA_BLAS_ENGINE_CUTLASS: 
	      stridedBatchGEMMCutlass((char*)weights_data_A + weights_offset*iArm,
				      (char*)input_data_full_ptr_A.data() + input_offset*iArm,
				      (char*)result_data_A + result_offset*iArm,
				      blas_param, QUDA_CUDA_FIELD_LOCATION);

	      stridedBatchGEMMCutlass((char*)weights_data_B + weights_offset*iArm,
				      (char*)input_data_full_ptr_B.data() + input_offset*iArm,
				      (char*)result_data_B + result_offset*iArm,
				      blas_param, QUDA_CUDA_FIELD_LOCATION);	      
	      break;
	    default:
	      errorQuda("Unknown BLAS engine %d", engine);
	    }
	    break;
	  default:
	    errorQuda("Unsupported Beamformer compute precision %d", compute_prec);
	  }

	  // DMH: Merge sumIncoherentBeam and powerSum kernels, also look at merging sumInnerChanTime
	  // DMH: fix me, improve reduction class to have option for device only array
	  qudaMemset(ib_sum_data, 0, batches_ib_sum*sizeof(float));
	  sumIncoherentBeam((char*)ib_sum_data,
			    (char*)flagants_data + flagants_offset*iArm,
			    (char*)input_data_full_ptr_A.data() + input_offset*iArm,
			    (char*)input_data_full_ptr_B.data() + input_offset*iArm,
			    n_thread_ib_sum, batches_ib_sum, compute_prec, stream_idx);

	  logQuda(QUDA_DEBUG_VERBOSE, "Size of result data array in bytes %ld\n", result_payload_size);
	  logQuda(QUDA_DEBUG_VERBOSE, "Size of result data array offset in bytes %ld\n", result_offset);
	  logQuda(QUDA_DEBUG_VERBOSE, "Size of result data array and offset ratio in bytes %ld\n", result_payload_size/result_offset);
	
	  //qudaMemset(ps_data, 0, ps_data_elems*sizeof(float));
	  powerSum((char*)ps_data,
		   (char*)result_data_A + result_offset*iArm,
		   (char*)result_data_B + result_offset*iArm,
		   (char*)ib_sum_data,
		   ps_data_elems, n_beam/n_arm, n_time_power_sum, compute_prec, stream_idx,
		   n_time_per_payload/n_time_inner, n_channels_per_payload, n_channels_inner, n_time_inner);
	
	  sumInnerChanTimeT((char*)output_copy + output_offset*iArm,
			    (char*)ps_data,
			    n_thread_sict, (n_time_per_payload/n_time_inner)/4, n_beam/n_arm, n_channels_per_payload, n_channels_inner, n_time_inner, compute_prec, stream_idx);    
	
	  qudaMemcpyAsync((char*)output_data_host + output_offset*iArm,
			  (char*)output_copy + output_offset*iArm, out_payload_size/n_arm, qudaMemcpyDeviceToHost, get_stream(stream_idx));
	}
	//qudaDeviceSynchronize();
      }
    }
    else {
      for(uint64_t stream_idx = 0; stream_idx < n_payload; stream_idx++) {
	if (getVerbosity() >= QUDA_SUMMARIZE) printfQuda("Launching with stream_idx = %lu, n_payload = %lu\n", stream_idx, n_payload);
	
	// Load data from host
	refresh_input(input_data_host, stream_idx);
	
	switch(compute_prec) {
	case QUDA_DOUBLE_PRECISION:
	case QUDA_SINGLE_PRECISION:
	  // Interleaved data
	  // (a - ib)(c + id)
	  blas_param.stream_idx = stream_idx;
	  blas_lapack::native::stridedBatchGEMM(weights_data_A, input_data_full_ptr_A.data(), result_data_A, blas_param, QUDA_CUDA_FIELD_LOCATION);
	  blas_lapack::native::stridedBatchGEMM(weights_data_B, input_data_full_ptr_B.data(), result_data_B, blas_param, QUDA_CUDA_FIELD_LOCATION);
	  break;
	default:
	  errorQuda("Unsupported Beamformer compute precision %d", compute_prec);
	}
	
	// DMH: Merge sumIncoherentBeam and powerSum kernels, also look at merging sumInnerChanTime
	// DMH: fix me, improve reduction class to have option for device only array
	sumIncoherentBeam(ib_sum_data, flagants_data, input_data_full_ptr_A.data(), input_data_full_ptr_B.data(), n_thread_ib_sum, batches_ib_sum, compute_prec, stream_idx);
	powerSum(ps_data, result_data_A, result_data_B, ib_sum_data, ps_data_elems, n_beam, n_time_power_sum, compute_prec, stream_idx,
		n_time_per_payload/n_time_inner, n_channels_per_payload, n_channels_inner, n_time_inner);
	sumInnerChanTimeT(output_copy, ps_data, n_thread_sict, (n_time_per_payload/n_time_inner)/n_time_power_sum, n_beam, n_channels_per_payload, n_channels_inner, n_time_inner, compute_prec, stream_idx);
	
	qudaMemcpyAsync(output_data_host, output_copy, out_payload_size, qudaMemcpyDeviceToHost, get_stream(stream_idx));
      }
      qudaDeviceSynchronize();
    }
  }

  void VoltageBeamformer::init_memory() {
    if(!mem_init) {
      logQuda(QUDA_VERBOSE, "Initialising memory in voltage_bf\n");
      blas_param = newQudaBLASParam();

      // DSA110 specialities
      if(packet_format == QUDA_PACKET_FORMAT_DSA110) input_copy_110 = device_pinned_malloc(in_payload_size/n_arm);
      
      // Input and output data from user
      input_copy = device_pinned_malloc(in_payload_size);
      output_copy = device_pinned_malloc(out_payload_size);
      weights_data_A = device_pinned_malloc(2 * n_weights_per_pol * sizeof(float));
      weights_data_B = device_pinned_malloc(2 * n_weights_per_pol * sizeof(float));
      flagants_data = device_pinned_malloc(n_antennae_per_payload * sizeof(int));
      
      // Internal memory for the class
      result_data_A = device_malloc(result_payload_size);
      result_data_B = device_malloc(result_payload_size);      
      ps_data = device_malloc(ps_data_elems * sizeof(float));
      ib_sum_data = device_malloc(batches_ib_sum * sizeof(float));
      
      input_data_full_ptr_A = quda::quda_ptr(QUDA_MEMORY_DEVICE, in_payload_size * compute_data_size, false);
      input_data_full_ptr_B = quda::quda_ptr(QUDA_MEMORY_DEVICE, in_payload_size * compute_data_size, false);

      mem_init = true;
      logQuda(QUDA_VERBOSE, "Memory init complete in voltage_bf\n");
    }
  }
  
  void VoltageBeamformer::destroy_memory() {
    if(mem_init) {
      if(input_copy) device_pinned_free(input_copy);
      if(input_copy_110) device_pinned_free(input_copy_110);
      if(output_copy) device_pinned_free(output_copy);
      if(result_data_A) device_free(result_data_A);
      if(result_data_B) device_free(result_data_B);
      if(weights_data_A) device_pinned_free(weights_data_A);
      if(weights_data_B) device_pinned_free(weights_data_B);
      if(flagants_data) device_pinned_free(flagants_data);
      if(ib_sum_data) device_pinned_free(ib_sum_data);
      if(ps_data) device_free(ps_data);
      mem_init = false;
    }
  }  

  void VoltageBeamformer::populate_blas_param(BeamformerParam &param) {

    // Set up for gemm
    //----------------
    getProfile().TPSTART(QUDA_PROFILE_INIT);
    blas_param.data_order = QUDA_BLAS_DATAORDER_ROW;
    blas_param.location = QUDA_CUDA_FIELD_LOCATION;
    switch(compute_prec) {
    case QUDA_DOUBLE_PRECISION: blas_param.data_type = QUDA_BLAS_DATATYPE_Z; break;
    case QUDA_SINGLE_PRECISION: blas_param.data_type = QUDA_BLAS_DATATYPE_C; break;
    case QUDA_HALF_PRECISION: blas_param.data_type = QUDA_BLAS_DATATYPE_H; break;
    default:
      errorQuda("Unsupported Beamformer compute precision %d", param.compute_prec);
    }
    
    blas_param.blas_type = QUDA_BLAS_GEMM;
    blas_param.trans_a = QUDA_BLAS_OP_N;
    blas_param.trans_b = QUDA_BLAS_OP_C;
    blas_param.m = param.n_beam/param.n_arm;
    blas_param.n = param.n_channels_inner * param.n_time_per_payload;
    blas_param.k = param.n_antennae_per_payload/param.n_arm;
    blas_param.alpha = 1.0;
    blas_param.beta = 0.0;
    blas_param.lda = blas_param.k;
    blas_param.ldb = blas_param.k;
    blas_param.ldc = blas_param.n;
    
    // NB: `stride` here refers to "the number of matrices
    // in arrays A, B, C, to stride over in the batches." In
    // CUBLAS (and pretty much all other BLAS interfaces) `stride`
    // would be the number of matrix elements. We take care of this
    // conversion in the interface.
    blas_param.a_stride = 1;
    blas_param.b_stride = 1;  
    blas_param.c_stride = 1;
    
    blas_param.batch_count = param.n_channels_per_payload / param.n_channels_inner;
    blas_param.a_offset = 0;
    blas_param.b_offset = 0;
    blas_param.c_offset = 0;
    blas_param.struct_size = sizeof(blas_param);
    if (getVerbosity() >= QUDA_SUMMARIZE) {
      printfQuda("Beamformer class instantiated with BLAS params:\n");
      printQudaBLASParam(&blas_param);      
    }
    if (getVerbosity() >= QUDA_DEBUG_VERBOSE) {
      printfQuda("Complex A = %d elems\n", blas_param.m * blas_param.k * blas_param.batch_count);
      printfQuda("Complex B = %d elems\n", blas_param.n * blas_param.k * blas_param.batch_count);
      printfQuda("Complex C = %d elems\n", blas_param.m * blas_param.n * blas_param.batch_count);
    }
    
    getProfile().TPSTOP(QUDA_PROFILE_INIT);
  }

  void VoltageBeamformer::compute_weights(void *input_data_host, void *flagants_host, int n_flags, float dec, float sfreq) {
    
    int n_calibs = (2 * n_antennae_per_payload * n_channels_per_payload)/n_channels_inner;
    int n_freq = n_channels_per_payload/n_channels_inner;
    
    // Device side data
    void *dev_ant_pos_E = device_pinned_malloc(n_antennae_per_payload * sizeof(float));
    void *dev_ant_pos_N = device_pinned_malloc(n_antennae_per_payload * sizeof(float));
    void *dev_freq = device_pinned_malloc(n_freq * sizeof(float));
    void *dev_calibrations = device_pinned_malloc(2*n_calibs * sizeof(float));
        
    // Host side data
    std::vector<float> ant_pos_E(n_antennae_per_payload, 0.0);
    std::vector<float> ant_pos_N(n_antennae_per_payload, 0.0);
    std::vector<float> calibrations(2*n_calibs, 0.0);
    std::vector<float> frequencies(n_freq, 0.0);    
    std::vector<int> flagants(n_antennae_per_payload, 0);

    // deal with antpos and calibs
    int iant, found = 0;
    float norm = 0;
    for (uint64_t i=0; i<n_antennae_per_payload; i++) {
      ant_pos_E[i] = ((float*)input_data_host)[i];
      ant_pos_N[i] = ((float*)input_data_host)[n_antennae_per_payload + i];
    }

    for (int i=0;i<n_calibs;i++) {
      
      iant = i/(n_calibs/n_antennae_per_payload);
      flagants[iant] = 0;
      
      found = 0;
      for (int j=0; j<n_flags; j++) {
	if ( ((int*)flagants_host)[j] == iant) {
	  found = 1;
	  flagants[iant] = 1;
	}
      }
      
      calibrations[2*i]   = ((float*)input_data_host)[2*n_antennae_per_payload + 2*i];
      calibrations[2*i+1] = ((float*)input_data_host)[2*n_antennae_per_payload + 2*i + 1];
      
      norm = sqrt(calibrations[2*i]*calibrations[2*i] + calibrations[2*i+1]*calibrations[2*i+1]);
      if (norm != 0.0) {
	calibrations[2*i]   /= norm;
	calibrations[2*i+1] /= norm;
      }
      
      if (found == 1) {
	calibrations[2*i]   = 0.;
	calibrations[2*i+1] = 0.;
      }
    }

    for (uint64_t i=0; i<n_channels_per_payload/n_channels_inner; i++) frequencies[i] = 1e6*(sfreq - i*250.0/1024.0);
    
    refresh_flagants(flagants.data());
    //for(int i=0; i<n_antennae_per_payload; i++) printf("FLAGVAL(%d) = %d\n", i, flagants[i]);
    //for(int i=0; i<n_calibs; i++) printf("CALIB(%d) = (%f,%f)\n", i, calibrations[2*i], calibrations[2*i+1]);
    //for(int i=0; i<n_channels_per_payload/n_channels_inner; i++) printf("FREQ(%d) = %f\n", i, frequencies[i]);
    
    
    // Move data to GPU, compute weights
    qudaMemcpy(dev_ant_pos_E, ant_pos_E.data(), n_antennae_per_payload * sizeof(float), qudaMemcpyHostToDevice);
    qudaMemcpy(dev_ant_pos_N, ant_pos_N.data(), n_antennae_per_payload * sizeof(float), qudaMemcpyHostToDevice);
    qudaMemcpy(dev_freq, frequencies.data(), n_freq * sizeof(float), qudaMemcpyHostToDevice);
    qudaMemcpy(dev_calibrations, calibrations.data(), 2 * n_calibs * sizeof(float), qudaMemcpyHostToDevice);

    int64_t n_threads = 2*(n_channels_per_payload/n_channels_inner)*(n_beam/n_arm)*(n_antennae_per_payload/n_arm);
    
    logQuda(QUDA_VERBOSE, "Launching computeWeights with %lu threads\n", n_threads); 
    computeWeights(weights_data_A, weights_data_B, dev_ant_pos_E, dev_ant_pos_N, dev_calibrations, flagants_data, dev_freq, 37.23 - dec, n_arm, n_threads, compute_prec);
    logQuda(QUDA_VERBOSE, "Completed computeWeights\n");
    
    //device_free(dev_ant_pos_E);
    //device_free(dev_ant_pos_N);
    //device_free(dev_freq);
    //device_free(dev_calibrations);
  }
  
  VoltageBeamformer::~VoltageBeamformer() {
    destroy_memory();
  }
}
