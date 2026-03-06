#include <tunable_nd.h>
#include <tunable_reduction.h>
#include <instantiate.h>
#include <device_vector.h>
#include <kernels/reorder_data.cuh>

namespace quda {

  // reorderAndPromote data from dsa110 order to dsa2k order
  template <typename PrecOut> class ReorderDataPolT : TunableKernel1D {
    
    // Copies of function arguments
  protected:
    
    const void *input;
    void *output_A;
    void *output_B;
    int n_time;
    int n_time_inner;
    int n_chan;
    int n_pol;
    int n_ant;
    int stream_idx;
    
    unsigned int sharedBytesPerBlock(const TuneParam &param) const {
      // Each thread will have
      size_t sz = ((param.block.x) * (param.block.x + 1) * 4 * sizeof(PrecOut));
      logQuda(QUDA_VERBOSE, "ReorderDataPol ShMemSz = (%d, %d) = %d chunks of size %lu bytes\n",
	      param.block.x,
	      param.block.x + 1,
	      param.block.x * (param.block.x + 1),
	      sizeof(PrecOut));
      return sz;
    }
    
    //unsigned int sharedBytesPerThread() const { return (5*sizeof(PrecOut)); }
    bool tuneSharedBytes() const { return false; }

  public:
    // We use a blank function signature on `const QudaPrecision` to supress `-Wunused-parameter` compiler warnings.
    ReorderDataPolT(const QudaPrecision, void *output_A, void *output_B, const void *input, const uint64_t n_time, const uint64_t n_time_inner, const uint64_t n_chan, const uint64_t n_pol, const uint64_t n_ant, int stream_idx) : 
      TunableKernel1D((n_time/n_time_inner  * n_chan * n_pol * n_ant)/2, QUDA_CUDA_FIELD_LOCATION),
      input(input),
      output_A(output_A),
      output_B(output_B),
      n_time(n_time),
      n_time_inner(n_time_inner),
      n_chan(n_chan),
      n_pol(n_pol),
      n_ant(n_ant),
      stream_idx(stream_idx)
    {
      switch (n_time_inner) {
      case 1: strcat(aux, "n-time-inner1,"); break;
      case 2: strcat(aux, "n-time-inner2,"); break;
      default: errorQuda("Unsupported number of inner time sums %d. Please use 1,2 or add template launch", n_time_inner);
      }
      
      switch (n_pol) {
      case 1: strcat(aux, "n-pol1,"); break;
      case 2: strcat(aux, "n-pol2,"); break;
      default: errorQuda("Unsupported number of polarisations %d. Please use 1,2 or add template launch", n_pol);
      }
      apply(device::get_stream(stream_idx));
    }

    // `apply` is a `TunableKernel1D` member function that runs both the tuning step and the kernel launch
    // with the tuned or untuned kernel bounds.
    void apply(const qudaStream_t &stream) {
      
      //printfQuda("Launching reorderDataPol with stream %d\n", stream_idx);
      TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
      switch (n_time_inner) {
      case 1: {
	switch(n_pol) {
	case 1: {
	  launch<DataReorderPolT>(tp, stream, ReorderPolArgT<PrecOut, 1, 1>((PrecOut*)output_A, (PrecOut*)output_B, (uint8_t*)input, n_time, n_chan, n_ant));
	} break;
	case 2: {
	  launch<DataReorderPolT>(tp, stream, ReorderPolArgT<PrecOut, 2, 1>((PrecOut*)output_A, (PrecOut*)output_B, (uint8_t*)input, n_time, n_chan, n_ant));
	} break;
	default: errorQuda("Unsupported number of polarisations %d. Please use 1,2 or add template launch", n_pol); 
	}
      } break;
      case 2: {
	switch(n_pol) {
	case 1: {
	  launch<DataReorderPolT>(tp, stream, ReorderPolArgT<PrecOut, 1, 2>((PrecOut*)output_A, (PrecOut*)output_B, (uint8_t*)input, n_time/2, n_chan, n_ant));
	} break;
	case 2: {
	  launch<DataReorderPolT>(tp, stream, ReorderPolArgT<PrecOut, 2, 2>((PrecOut*)output_A, (PrecOut*)output_B, (uint8_t*)input, n_time/2, n_chan, n_ant));
	} break;
	default: errorQuda("Unsupported number of polarisations %d. Please use 1,2 or add template launch", n_pol); 
	}	
      } break;
      default: errorQuda("Unsupported number of inner time sums %d. Please use 1,2 or add template launch", n_time_inner);
      }
    }
    
    unsigned long long int N = (n_time/n_time_inner * n_chan * n_pol * n_ant)/2;
    unsigned int minThreads() const { return N; }
    
    // `TunableKernel1D` member for FLOPS
    long long flops() const { return 0; }    
    
    // Amount of input data throughput in kernel.
    long long bytes() const { return N * sizeof(uint8_t); }
  };
  
  void reorderDataPolT(void *output_A, void *output_B, const void *input, const uint64_t n_time, const uint64_t n_time_inner, const uint64_t n_chan, const uint64_t n_pol, const uint64_t n_ant, const QudaPrecision prec_out, int stream_idx)
  {
    logQuda(QUDA_VERBOSE, "launch reorderDataPol with threads = %d\n", (n_time/n_time_inner * n_chan * n_pol * n_ant)/2);
    getProfile().TPSTART(QUDA_PROFILE_COMPUTE);
    instantiatePrecisionTEST<ReorderDataPolT>(prec_out, output_A, output_B, input, n_time, n_time_inner, n_chan, n_pol, n_ant, stream_idx);
    getProfile().TPSTOP(QUDA_PROFILE_COMPUTE);
  }
  
  // reorderAndPromote data from dsa110 order to dsa2k order
  template <typename PrecOut> class ReorderDataPol : TunableKernel1D {
    
    // Copies of function arguments
  protected:
    unsigned long long int N = (2048 * 384 * 2 * (96/4));
    const void *input;
    void *output_A;
    void *output_B;
    // Kernel thread volume (member of TunableKernel1D)
    unsigned int minThreads() const { return N; }
    int stream_idx;
    
    unsigned int sharedBytesPerBlock(const TuneParam &param) const {
      // Each thread will have
      size_t sz = ((param.block.x) * (param.block. z + 1) * 4 * sizeof(PrecOut));
      logQuda(QUDA_VERBOSE, "ReorderDataPol ShMemSz = (%d, %d) = %d chunks of size %lu bytes\n",
	      param.block.x,
	      param.block.z + 1,
	      param.block.x * (param.block.z + 1),
	      sizeof(PrecOut));
      return sz;
    }
    
    //unsigned int sharedBytesPerThread() const { return (5*sizeof(PrecOut)); }
    bool tuneSharedBytes() const { return false; }

  public:
    // We use a blank function signature on `const QudaPrecision` to supress `-Wunused-parameter` compiler warnings.
    ReorderDataPol(const QudaPrecision, void *output_A, void *output_B, const void *input, int stream_idx) : 
      TunableKernel1D(2048 * 384 * 2 * (96/4), QUDA_CUDA_FIELD_LOCATION),
      input(input),
      output_A(output_A),
      output_B(output_B),
      stream_idx(stream_idx)
    {
      apply(device::get_stream(stream_idx));
    }

    // `apply` is a `TunableKernel1D` member function that runs both the tuning step and the kernel launch
    // with the tuned or untuned kernel bounds.
    void apply(const qudaStream_t &stream) {
      //printfQuda("Launching reorderDataPol with stream %d\n", stream_idx);
      TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
      launch<DataReorderPol>(tp, stream, ReorderPolArg<PrecOut>((PrecOut*)output_A, (PrecOut*)output_B, (uint8_t*)input));
    }

    // `TunableKernel1D` member for FLOPS
    long long flops() const { return 0; }    
    
    // Amount of input data throughput in kernel.
    long long bytes() const { return N * sizeof(uint8_t); }
  };
  
  void reorderDataPol(void *output_A, void *output_B, const void *input, const QudaPrecision prec_out, int stream_idx)
  {
    logQuda(QUDA_VERBOSE, "launch reorderDataPol with threads = %d\n", (2048 * 384 * 2 * 96/4));
    getProfile().TPSTART(QUDA_PROFILE_COMPUTE);
    instantiatePrecisionTEST<ReorderDataPol>(prec_out, output_A, output_B, input, stream_idx);
    getProfile().TPSTOP(QUDA_PROFILE_COMPUTE);
  }












  
  // reorderAndPromote data from dsa110 order to dsa2k order
  template <typename PrecOut> class ReorderData : TunableKernel1D {
    
    // Copies of function arguments
  protected:
    unsigned long long int N = 384 * 2048 * 96;
    const void *input;
    void *output;
    // Kernel thread volume (member of TunableKernel1D)
    unsigned int minThreads() const { return N; }
    int stream_idx;
    unsigned int sharedBytesPerThread() const { return (4*sizeof(PrecOut)); }
    //unsigned int sharedBytesPerThread() const { return (4*sizeof(PrecOut)); }
    
  public:
    // We use a blank function signature on `const QudaPrecision` to supress `-Wunused-parameter` compiler warnings.
    ReorderData(const QudaPrecision, void *output, const void *input, int stream_idx) : 
      //TunableKernel3D((size_t)(2048), 96, 384, QUDA_CUDA_FIELD_LOCATION),
      TunableKernel1D(2048 * 96 * 384, QUDA_CUDA_FIELD_LOCATION),
      input(input),
      output(output),
      stream_idx(stream_idx)
    {
      apply(device::get_stream(stream_idx));
    }
    
    // `apply` is a `TunableKernel1D` member function that runs both the tuning step and the kernel launch
    // with the tuned or untuned kernel bounds.
    void apply(const qudaStream_t &stream) {
      //printfQuda("Launching reorderData with stream %d\n", stream_idx);
      TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
      launch<DataReorder>(tp, stream, ReorderArg<PrecOut>((PrecOut*)output, (uint8_t*)input));
    }

    // `TunableKernel1D` member for FLOPS
    long long flops() const { return 0; }    
    
    // Amount of input data throughput in kernel.
    long long bytes() const { return N * sizeof(uint8_t); }
  };  
  
  void reorderData(void *output, const void *input, const QudaPrecision prec_out, int stream_idx)
  {
    getProfile().TPSTART(QUDA_PROFILE_COMPUTE);
    instantiatePrecisionTEST<ReorderData>(prec_out, output, input, stream_idx);
    getProfile().TPSTOP(QUDA_PROFILE_COMPUTE);
  }
  
} // namespace quda
