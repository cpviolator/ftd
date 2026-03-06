#include <tunable_nd.h>
#include <tunable_reduction.h>
#include <instantiate.h>
#include <device_vector.h>
#include <kernels/sum_inner_chan_time.cuh>

namespace quda {
  
  template <typename Prec> class SumInnerChanTime : TunableKernel1D {
    
    // Copies of function arguments
  protected:
    unsigned int n_chan;
    unsigned int n_time;
    unsigned long long int N;
    void *output;
    const void *ps_data;
    // Kernel thread volume (member of TunableKernel1D)
    unsigned int minThreads() const { return N; }
    int stream_idx;    
  public:
    // We use a blank function signature on `const QudaPrecision` to supress `-Wunused-parameter` compiler warnings.
    SumInnerChanTime(const QudaPrecision, void *output, const void *ps_data, const unsigned int n_chan, const unsigned int n_time, const unsigned long long int N, int stream_idx) : 
      TunableKernel1D(N, QUDA_CUDA_FIELD_LOCATION),
      n_chan(n_chan),
      n_time(n_time),
      N(N),
      output(output),
      ps_data(ps_data),
      stream_idx(stream_idx)
    {
      switch (n_chan) {
      case 1: strcat(aux, "n-chan1,"); break;
      case 2: strcat(aux, "n-chan2,"); break;
      case 4: strcat(aux, "n-chan4,"); break;
      case 8: strcat(aux, "n-chan8,"); break;
      default: errorQuda("Unsupported number of inner channel sums %d. Please use 1,2,4,8 or add template launch", n_chan);
      }

      switch (n_time) {
      case 2: strcat(aux, "n-time2,"); break;
      default: errorQuda("Unsupported number of inner time sums %d. Please use 2 or add template launch", n_time);
	
      }
      apply(device::get_stream(stream_idx));
    }

    // `apply` is a `TunableKernel1D` member function that runs both the tuning step and the kernel launch
    // with the tuned or untuned kernel bounds.
    void apply(const qudaStream_t &stream) {
      if(n_time != 2) errorQuda("Unsupported number of inner time sums %d. Please use 2 or add template launch", n_time);
      TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
      switch (n_chan) {
      case 1: {
	SICTArg<Prec, 1, 2> arg((unsigned char*)output, (Prec*)ps_data, N);
	launch<sumChanTimeInner>(tp, stream, arg);
      } break;
      case 2: {
	SICTArg<Prec, 2, 2> arg((unsigned char*)output, (Prec*)ps_data, N);
	launch<sumChanTimeInner>(tp, stream, arg);
      } break;
      case 4: {
	SICTArg<Prec, 4, 2> arg((unsigned char*)output, (Prec*)ps_data, N);
	launch<sumChanTimeInner>(tp, stream, arg);
      } break;
      case 8: {
	SICTArg<Prec, 8, 2> arg((unsigned char*)output, (Prec*)ps_data, N);
	launch<sumChanTimeInner>(tp, stream, arg);
      } break;
      default: errorQuda("Unsupported number of inner channel sums %d. Please use 1,2,4,8 or add template launch", n_chan);
      }
    }
    
    // FLOPS in kernel: 
    long long flops() const { return N * n_chan * n_time; }    
    
    // Amount of data transfer in kernel.
    long long bytes() const { return N * sizeof(Prec) * (1 + 1.0/(n_chan*n_time)); }    
  };
  
  void sumInnerChanTime(void *output, const void *ps_data, const uint64_t N, const uint64_t n_channels_inner, const uint64_t n_time_inner, const QudaPrecision prec, int stream_idx)
  {
    logQuda(QUDA_VERBOSE, "Launching STS kernel with = %lu threads\n", N);    
    getProfile().TPSTART(QUDA_PROFILE_COMPUTE);
    instantiatePrecisionTEST<SumInnerChanTime>(prec, output, ps_data, n_channels_inner, n_time_inner, N, stream_idx);
    getProfile().TPSTOP(QUDA_PROFILE_COMPUTE);
  }

  template <typename Prec> class SumInnerChanTimeT : TunableKernel1D {
    
    // Copies of function arguments
  protected:
    unsigned int n_time;
    unsigned int n_beam;
    unsigned int n_chan;
    unsigned int n_chan_inner;
    unsigned int n_time_inner;
    unsigned long long int N;
    void *output;
    const void *ps_data;
    // Kernel thread volume (member of TunableKernel1D)
    unsigned int minThreads() const { return N; }
    int stream_idx;    
  public:
    // We use a blank function signature on `const QudaPrecision` to supress `-Wunused-parameter` compiler warnings.
    SumInnerChanTimeT(const QudaPrecision, void *output, const void *ps_data, const uint64_t n_time, const uint64_t n_beam, const uint64_t n_chan, const unsigned int n_chan_inner, const unsigned int n_time_inner, const unsigned long long int N, int stream_idx) : 
      TunableKernel1D(N, QUDA_CUDA_FIELD_LOCATION),
      n_time(n_time),
      n_beam(n_beam),
      n_chan(n_chan),      
      n_chan_inner(n_chan_inner),
      n_time_inner(n_time_inner),
      N(N),
      output(output),
      ps_data(ps_data),
      stream_idx(stream_idx)
    {
      switch (n_chan_inner) {
      case 1: strcat(aux, "n-chan1,"); break;
      case 2: strcat(aux, "n-chan2,"); break;
      case 4: strcat(aux, "n-chan4,"); break;
      case 8: strcat(aux, "n-chan8,"); break;
      default: errorQuda("Unsupported number of inner channel sums %d. Please use 1,2,4,8 or add template launch", n_chan);
      }

      switch (n_time_inner) {
      case 2: strcat(aux, "n-time2,"); break;
      default: errorQuda("Unsupported number of inner time sums %d. Please use 2 or add template launch", n_time);
	
      }
      apply(device::get_stream(stream_idx));
    }

    // `apply` is a `TunableKernel1D` member function that runs both the tuning step and the kernel launch
    // with the tuned or untuned kernel bounds.
    void apply(const qudaStream_t &stream) {
      if(n_time_inner != 2) errorQuda("Unsupported number of inner time sums %d. Please use 2 or add template launch", n_time_inner);
      TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
      switch (n_chan_inner) {
      case 1: {
	SICTArgT<Prec, 1, 2> arg((unsigned char*)output, (Prec*)ps_data, N, n_time, n_beam, n_chan);
	launch<sumChanTimeInnerT>(tp, stream, arg);
      } break;
      case 2: {
	SICTArgT<Prec, 2, 2> arg((unsigned char*)output, (Prec*)ps_data, N, n_time, n_beam, n_chan);
	launch<sumChanTimeInnerT>(tp, stream, arg);
      } break;
      case 4: {
	SICTArgT<Prec, 4, 2> arg((unsigned char*)output, (Prec*)ps_data, N, n_time, n_beam, n_chan);
	launch<sumChanTimeInnerT>(tp, stream, arg);
      } break;
      case 8: {
	SICTArgT<Prec, 8, 2> arg((unsigned char*)output, (Prec*)ps_data, N, n_time, n_beam, n_chan);
	launch<sumChanTimeInnerT>(tp, stream, arg);
      } break;
      default: errorQuda("Unsupported number of inner channel sums %d. Please use 1,2,4,8 or add template launch", n_chan_inner);
      }
    }
    
    // FLOPS in kernel: 
    long long flops() const { return N * n_chan * n_time; }    
    
    // Amount of data transfer in kernel.
    long long bytes() const { return N * sizeof(Prec) * (1 + 1.0/(n_chan*n_time)); }    
  };

  
  
  void sumInnerChanTimeT(void *output, const void *ps_data, const uint64_t N, const uint64_t n_time, const uint64_t n_beam, const uint64_t n_chan, const uint64_t n_channels_inner, const uint64_t n_time_inner, const QudaPrecision prec, int stream_idx)
  {
    logQuda(QUDA_VERBOSE, "Launching STS kernel with = %lu threads\n", N);
    getProfile().TPSTART(QUDA_PROFILE_COMPUTE);
    instantiatePrecisionTEST<SumInnerChanTimeT>(prec, output, ps_data, n_time, n_beam, n_chan, n_channels_inner, n_time_inner, N, stream_idx);
    getProfile().TPSTOP(QUDA_PROFILE_COMPUTE);
  }

  
} // namespace quda
