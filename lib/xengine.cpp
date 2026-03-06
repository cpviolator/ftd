#include <future>

#include "ggp_internal.h"
#include "timer.h"
#include "algorithm.h"
#include "blas_lapack.h"
#include "dsa.h"
#ifdef TCC_API
#include "tcc_interface.h"
#endif

using namespace quda;
using namespace device;
using namespace blas_lapack;

namespace ggp {

  // Forward declaration: QC sign-magnitude -> FTD/TCC two's complement nibble conversion
  void convertQcSmToFtd(void *data, unsigned long long int N, int stream_idx);

  // XEngine constructor
  XEngine::XEngine(XEngineParam &param) {

    // Deduce all array sizes and compute parameters
    //---------------------------------------------------------
    // Sizes of arrays in bytes
    // Infer input vector length in char sized (4,4)bit complex
    compute_prec = param.compute_prec;

    // Device/Host location of data
    in_location = param.in_location;
    out_location = param.out_location;    
    
    switch(compute_prec) {
    case QUDA_DOUBLE_PRECISION: compute_data_size = sizeof(double); break;
    case QUDA_SINGLE_PRECISION: compute_data_size = sizeof(float); break;
    default: errorQuda("Unsupported XEngine compute precision %d", output_prec);
    }

    packet_format = param.packet_format;

    // Store params for TCC path
    n_pols_stored = param.n_polarizations;
    n_channels_stored = param.n_channels_per_payload;
    n_antennae_stored = param.n_antennae_per_payload;
    n_time_stored = param.n_time_per_payload;
    n_time_inner_stored = param.n_time_inner;

    n_payload = param.n_payload;
    // N elements in input block
    in_payload_size = (param.n_time_per_payload *
		       param.n_antennae_per_payload *
		       param.n_channels_per_payload *
		       param.n_polarizations);

    output_prec = param.output_prec;
    if(output_prec != compute_prec) errorQuda("output and compute precisions must match until a copy engine is formulated");
    output_data_size = 0;
    switch(output_prec) {
    case QUDA_DOUBLE_PRECISION: output_data_size = sizeof(double); break;
    case QUDA_SINGLE_PRECISION: output_data_size = sizeof(float); break;
    default: errorQuda("Unsupported XEngine output precision %d", output_prec);
    }

    // Matrix output format
    mat_format = param.format;
    switch(mat_format) {
    case QUDA_XENGINE_MAT_HERM: {
      // Full Hermitian matrix payload output size
      out_payload_size = 2ULL * output_data_size * (param.n_antennae_per_payload *
						    param.n_antennae_per_payload *
						    param.n_channels_per_payload *
						    param.n_polarizations *
						    param.n_time_inner);
    } break;
    case QUDA_XENGINE_MAT_TRI: {
      // Packed tri matrix payload output size
      out_payload_size = 2ULL * output_data_size * ((((param.n_antennae_per_payload + 1) * (param.n_antennae_per_payload))/2ULL) *
						    param.n_channels_per_payload *
						    param.n_polarizations *
						    param.n_time_inner);
      
    } break;
    default: errorQuda("Unknown XEngine mat format %d", mat_format);
    }
    
    // Make device data arrays for CUBLAS result in half prec
    result_data_size = compute_data_size;
    
    result_payload_size = 2 * result_data_size * (param.n_antennae_per_payload *
						  param.n_antennae_per_payload *
						  param.n_channels_per_payload *
						  param.n_polarizations *
						  param.n_time_inner);
    
    logQuda(QUDA_VERBOSE, "result_payload_size = %lu\n", result_payload_size);
    logQuda(QUDA_VERBOSE, "in_payload_size = %lu\n", in_payload_size);
    logQuda(QUDA_VERBOSE, "out_payload_size = %lu\n", out_payload_size);
    //---------------------------------------------------------

    // Allocate memory
    init_memory();

    // Populate BLAS param
    populate_blas_param(param);
  }

  void XEngine::compute(void *output_data, void *input_data) {
    
    //qudaEvent_t event[n_payload];
    //for(uint64_t i=0; i<n_payload; i++) event[i] = qudaEventCreate();

    for(uint64_t stream_idx = 0; stream_idx<n_payload; stream_idx++) {
      printfQuda("Launching with stream_idx = %lu, n_payload = %lu\n", stream_idx, n_payload);

      // Refresh host pointers
      current_input_data_ptr = (char*)input_data + stream_idx * in_payload_size;
      if(in_location == QUDA_CPU_FIELD_LOCATION) {
	// Copy host input data to device memory
	qudaMemcpyAsync(input_copy, current_input_data_ptr, in_payload_size, qudaMemcpyHostToDevice, get_stream(stream_idx));
      } else {
	input_copy = current_input_data_ptr;
      }
      
#ifdef TCC_API
      if(blas_param.engine == QUDA_BLAS_ENGINE_TCC) {
        // TCC path: bypass promoteData and BLAS — pass raw QC bytes directly
        if(packet_format != QUDA_PACKET_FORMAT_DSA2K) {
          errorQuda("TCC engine only supports DSA2K packet format, got %d", packet_format);
        }
        if(mat_format != QUDA_XENGINE_MAT_TRI) {
          errorQuda("TCC engine only supports triangular output format (QUDA_XENGINE_MAT_TRI)");
        }

        // Convert sign-magnitude (high=Re, low=Im) -> two's complement (low=Re, high=Im)
        convertQcSmToFtd(input_copy, in_payload_size, stream_idx);

        // TCC writes directly to packed triangle output
        current_output_data_ptr = (char*)output_data + stream_idx * out_payload_size;
        correlateTCC(tri_output, input_copy,
                     n_antennae_stored, n_channels_stored,
                     n_time_stored, n_time_inner_stored,
                     n_pols_stored, stream_idx);

        if(out_location == QUDA_CPU_FIELD_LOCATION) {
          qudaMemcpyAsync(current_output_data_ptr, tri_output, out_payload_size, qudaMemcpyDeviceToHost, get_stream(stream_idx));
        }
      } else
#endif
      {
        // cuBLAS / CUTLASS path
        // Promote char data for CUBLAS
        switch(packet_format) {
        case QUDA_PACKET_FORMAT_DSA2K: promoteData(input_data_full_ptr.data(), input_copy, in_payload_size, compute_prec, stream_idx); break;
        case QUDA_PACKET_FORMAT_DSA110: reorderData(input_data_full_ptr.data(), input_copy, compute_prec, stream_idx); break;
        default: errorQuda("Unknown packet format %d\n", packet_format);
        }

        // Interleaved data
        // (a - ib)(c + id)
        blas_param.stream_idx = stream_idx;
        blas_lapack::native::stridedBatchGEMM(input_data_full_ptr.data(), input_data_full_ptr.data(), result_data, blas_param, QUDA_CUDA_FIELD_LOCATION); // GGP_DEVICE_FIELD_LOCATION

        // Reorder data for user
        current_output_data_ptr = (char*)output_data + stream_idx * out_payload_size;
        if(mat_format == QUDA_XENGINE_MAT_TRI) {
	  int n = result_payload_size/(2ULL*output_data_size); // Complex elems
	  int n_batch = blas_param.m * blas_param.n;   // Complex elems in one (full) matrix
	  triangulateFromHerm(tri_output, result_data, n, n_batch, output_prec, stream_idx);
	  if(out_location == QUDA_CPU_FIELD_LOCATION) qudaMemcpyAsync(current_output_data_ptr, tri_output, out_payload_size, qudaMemcpyDeviceToHost, get_stream(stream_idx));
        } else {
	  if(out_location == QUDA_CPU_FIELD_LOCATION) qudaMemcpyAsync(current_output_data_ptr, result_data, out_payload_size, qudaMemcpyDeviceToHost, get_stream(stream_idx));
        }
      }
    }
  
    qudaDeviceSynchronize();
  }

#if 0
  void XEngine::compute_half_prec() {
    // COMPUTE STREAM
    int stream_idx = 0;
    
    // Copy input data to device memory
    qudaMemcpyAsync(input_copy, input_data_host_ptr, in_payload_size, qudaMemcpyHostToDevice, get_stream(0));
    
    // Promote char data to half for CUBLAS
    getProfile().TPSTART(QUDA_PROFILE_COMPUTE);
    promoteDataPlanar(input_data_real_ptr.data(), input_data_imag_ptr.data(), input_copy, in_payload_size, compute_prec, stream_idx);
    
    // DO CUBLAS
    blas_param.trans_a = QUDA_BLAS_OP_T;
    // ac
    blas_param.alpha = 1.0;
    blas_param.beta = 0.0;
    blas_lapack::native::stridedBatchGEMM(input_data_real_ptr.data(), input_data_real_ptr.data(), result_real, blas_param, QUDA_CUDA_FIELD_LOCATION);
    // bd
    blas_param.alpha = 1.0;
    blas_param.beta = 1.0;
    blas_lapack::native::stridedBatchGEMM(input_data_imag_ptr.data(), input_data_imag_ptr.data(), result_real, blas_param, QUDA_CUDA_FIELD_LOCATION);
    //-bc
    blas_param.alpha = -1.0;
    blas_param.beta = 0.0;
    blas_lapack::native::stridedBatchGEMM(input_data_imag_ptr.data(), input_data_real_ptr.data(), result_imag, blas_param, QUDA_CUDA_FIELD_LOCATION);
    //ad
    blas_param.alpha = 1.0;
    blas_param.beta = 1.0;
    blas_lapack::native::stridedBatchGEMM(input_data_real_ptr.data(), input_data_imag_ptr.data(), result_imag, blas_param, QUDA_CUDA_FIELD_LOCATION);
    getProfile().TPSTOP(QUDA_PROFILE_COMPUTE);
    
    if(mat_format == QUDA_XENGINE_MAT_TRI) {
      promInterTri(output_copy, result_real,  result_imag, out_payload_size/(2 * output_data_size), output_prec);
      int n = out_payload_size/(2*output_data_size); // Complex elems
      logQuda(QUDA_VERBOSE, "result complex elems = %d\n", n);
      int n_batch = blas_param.m * blas_param.n;   // Complex elems in one (full) matrix
      triangulateFromHerm(tri_output, output_copy, n, n_batch, output_prec, stream_idx);
      qudaMemcpy(output_data_host_ptr, tri_output, tri_output_payload_size, qudaMemcpyDeviceToHost);
    } else {
      promInter(output_copy, result_real, result_imag, out_payload_size/(2 * output_data_size), output_prec);
      qudaMemcpy(output_data_host_ptr, output_copy, out_payload_size, qudaMemcpyDeviceToHost);
    }    
  }
#endif

  void XEngine::init_memory() {
    if(!mem_init) {
      blas_param = newQudaBLASParam();
      result_data = device_pinned_malloc(result_payload_size);
      
      if(in_location == QUDA_CPU_FIELD_LOCATION) input_copy = device_pinned_malloc(in_payload_size);
      if(out_location == QUDA_CPU_FIELD_LOCATION) output_copy = device_pinned_malloc(out_payload_size);
      
      if(mat_format == QUDA_XENGINE_MAT_TRI) tri_output = device_pinned_malloc(out_payload_size);
      input_data_full_ptr = quda::quda_ptr(QUDA_MEMORY_DEVICE, 2*in_payload_size * compute_data_size, false);
      
      //result_real = device_pinned_malloc(result_payload_size);
      //result_imag = device_pinned_malloc(result_payload_size);
      //input_data_real_ptr = quda::quda_ptr(QUDA_MEMORY_DEVICE, in_payload_size * compute_data_size, false);
      //input_data_imag_ptr = quda::quda_ptr(QUDA_MEMORY_DEVICE, in_payload_size * compute_data_size, false);

      mem_init = true;
    }
  }

  void XEngine::destroy_memory() {
    if(mem_init) {
      if(in_location == QUDA_CPU_FIELD_LOCATION) device_pinned_free(input_copy);
      if(out_location == QUDA_CPU_FIELD_LOCATION) device_pinned_free(output_copy);
      if(tri_output) device_pinned_free(tri_output);
      //if(result_real) device_free(result_real);
      //if(result_imag) device_free(result_imag);
      if(result_data) device_pinned_free(result_data);

      mem_init = false;
    }
  }  

  void XEngine::populate_blas_param(XEngineParam &param) {
    // Set up for gemm
    //----------------
    blas_param.engine = param.engine;
    blas_param.data_order = QUDA_BLAS_DATAORDER_ROW;
    blas_param.location = QUDA_CUDA_FIELD_LOCATION;
    switch(compute_prec) {
    case QUDA_DOUBLE_PRECISION: blas_param.data_type = QUDA_BLAS_DATATYPE_Z; break;
    case QUDA_SINGLE_PRECISION: blas_param.data_type = QUDA_BLAS_DATATYPE_C; break;
    case QUDA_HALF_PRECISION: blas_param.data_type = QUDA_BLAS_DATATYPE_H; break;
    default:
      errorQuda("Unsupported XEngine compute precision %d", compute_prec);
    }
    
    blas_param.blas_type = QUDA_BLAS_GEMM;
    blas_param.trans_a = QUDA_BLAS_OP_C;
    blas_param.trans_b = QUDA_BLAS_OP_N;
    blas_param.m = param.n_antennae_per_payload;
    blas_param.n = param.n_antennae_per_payload;
    blas_param.k = param.n_time_per_payload/param.n_time_inner;
    blas_param.alpha = 1.0;
    blas_param.beta = 0.0;
    blas_param.lda = blas_param.m;
    blas_param.ldb = blas_param.m;
    blas_param.ldc = blas_param.n;
    
    // NB: `stride` here refers to "the number of matrices
    // in arrays A, B, C, to stride over in the batches." In
    // CUBLAS (and pretty much all other BLAS interfaces) `stride`
    // would be the number of matrix elements. We take care of this
    // conversion in the interface.
    blas_param.a_stride = 1;
    blas_param.b_stride = 1;
    blas_param.c_stride = 1;
    
    blas_param.batch_count = param.n_channels_per_payload * param.n_polarizations * param.n_time_inner;
    blas_param.a_offset = 0;
    blas_param.b_offset = 0;
    blas_param.c_offset = 0;
    blas_param.struct_size = sizeof(blas_param);
    //if (getVerbosity() >= QUDA_VERBOSE) printQudaBLASParam(&blas_param); 
  }
  
  XEngine::~XEngine() {
    // Clean up memory
    destroy_memory();
  }  
}

