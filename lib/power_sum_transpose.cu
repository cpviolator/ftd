#include <tunable_nd.h>
#include <tunable_reduction.h>
#include <instantiate.h>
#include <device_vector.h>
#include <kernels/power_sum_transpose.cuh>

namespace quda {
  
  template <typename Prec> class PowerSum : TunableKernel1D {

    // Copies of function arguments
  protected:
    unsigned long long int N;
    unsigned int n_time_power_sum;
    unsigned int n_beam;
    unsigned int n_time;
    unsigned int n_chan;
    unsigned int n_chan_sum;
    unsigned int n_time_inner;
    const void *input_A;
    const void *input_B;
    const void *ib_sum_data;
    void *ps_data;
    // Kernel thread volume (member of TunableKernel1D)
    unsigned int minThreads() const { return N; }
    int stream_idx;

  public:
    // We use a blank function signature on `const QudaPrecision` to supress `-Wunused-parameter` compiler warnings.
    PowerSum(const QudaPrecision, void *ps_data, const void *input_A, const void *input_B, const void *ib_sum_data, const unsigned int n_time_power_sum, const unsigned int n_beam, const unsigned long long int N, int stream_idx,
             unsigned int n_time, unsigned int n_chan, unsigned int n_chan_sum, unsigned int n_time_inner) :
      TunableKernel1D(N, QUDA_CUDA_FIELD_LOCATION),
      N(N),
      n_time_power_sum(n_time_power_sum),
      n_beam(n_beam),
      n_time(n_time),
      n_chan(n_chan),
      n_chan_sum(n_chan_sum),
      n_time_inner(n_time_inner),
      input_A(input_A),
      input_B(input_B),
      ib_sum_data(ib_sum_data),
      ps_data(ps_data),
      stream_idx(stream_idx)
    {
      switch (n_beam) {
      case 256: strcat(aux, "n-beam256,"); break;
      case 512: strcat(aux, "n-beam512,"); break;
      case 1024: strcat(aux, "n-beam1024,"); break;
      case 2080: strcat(aux, "n-beam2080,"); break;
      case 4096: strcat(aux, "n-beam4096,"); break;
      case 4656: strcat(aux, "n-beam4656,"); break;
      case 16384: strcat(aux, "n-beam16384,"); break;
      case 262144: strcat(aux, "n-beam262144,"); break;
      case 524288: strcat(aux, "n-beam524288,"); break;
      default: errorQuda("Unsupported number of beams %d. Please use 256, 512, 1024, 2080, 4096, 4656, 16384, 262144, 524288, or add template launch", n_beam);
      }

      switch (n_time_power_sum) {
      case 4: strcat(aux, "n-time-power-sum4,"); break;
      default: errorQuda("Unsupported number of power time sums %d. Please use 4 or add template launch", n_time_power_sum);
      }
      apply(device::get_stream(stream_idx));
    }

    // `apply` is a `TunableKernel1D` member function that runs both the tuning step and the kernel launch
    // with the tuned or untuned kernel bounds.
    void apply(const qudaStream_t &stream) {
      TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
      if(n_time_power_sum != 4) errorQuda("Unsupported number of power time sums %d. Please use 4 or add template launch", n_time_power_sum);
      switch (n_beam) {
      case 256: {
	PSArg<Prec, 4, 256> arg((Prec*)ps_data, (Prec*)input_A, (Prec*)input_B, (Prec*)ib_sum_data, N, n_time, n_chan, n_chan_sum, n_time_inner);
	launch<powSum>(tp, stream, arg);
      } break;
      case 512: {
	PSArg<Prec, 4, 512> arg((Prec*)ps_data, (Prec*)input_A, (Prec*)input_B, (Prec*)ib_sum_data, N, n_time, n_chan, n_chan_sum, n_time_inner);
	launch<powSum>(tp, stream, arg);
      } break;
      case 1024: {
	PSArg<Prec, 4, 1024> arg((Prec*)ps_data, (Prec*)input_A, (Prec*)input_B, (Prec*)ib_sum_data, N, n_time, n_chan, n_chan_sum, n_time_inner);
	launch<powSum>(tp, stream, arg);
      } break;
      case 2080: {
	PSArg<Prec, 4, 2080> arg((Prec*)ps_data, (Prec*)input_A, (Prec*)input_B, (Prec*)ib_sum_data, N, n_time, n_chan, n_chan_sum, n_time_inner);
	launch<powSum>(tp, stream, arg);
      } break;
      case 4096: {
	PSArg<Prec, 4, 4096> arg((Prec*)ps_data, (Prec*)input_A, (Prec*)input_B, (Prec*)ib_sum_data, N, n_time, n_chan, n_chan_sum, n_time_inner);
	launch<powSum>(tp, stream, arg);
      } break;
      case 4656: {
	PSArg<Prec, 4, 4656> arg((Prec*)ps_data, (Prec*)input_A, (Prec*)input_B, (Prec*)ib_sum_data, N, n_time, n_chan, n_chan_sum, n_time_inner);
	launch<powSum>(tp, stream, arg);
      } break;
      case 16384: {
	PSArg<Prec, 4, 16384> arg((Prec*)ps_data, (Prec*)input_A, (Prec*)input_B, (Prec*)ib_sum_data, N, n_time, n_chan, n_chan_sum, n_time_inner);
	launch<powSum>(tp, stream, arg);
      } break;
      case 262144: {
	PSArg<Prec, 4, 262144> arg((Prec*)ps_data, (Prec*)input_A, (Prec*)input_B, (Prec*)ib_sum_data, N, n_time, n_chan, n_chan_sum, n_time_inner);
	launch<powSum>(tp, stream, arg);
      } break;
      case 524288: {
	PSArg<Prec, 4, 524288> arg((Prec*)ps_data, (Prec*)input_A, (Prec*)input_B, (Prec*)ib_sum_data, N, n_time, n_chan, n_chan_sum, n_time_inner);
	launch<powSum>(tp, stream, arg);
      } break;
      default: errorQuda("Unsupported number of beams %d. Please use 256, 512, 1024, 2080, 4096, 4656, 16384, 262144, 524288, or add template launch", n_beam);
      }
    }
      
    // FLOPS in kernel: 
    long long flops() const { return N * n_time_power_sum * 9; }    
    
    // Amount of data transfer in kernel.
    long long bytes() const { return N * (1 + 2*2*n_time_power_sum + (1.0*n_time_power_sum/n_beam)) * sizeof(Prec); }    
  };
  
  void powerSum(void *ps_data, const void *input_A, const void *input_B, const void *ib_sum_data, const uint64_t N, const uint64_t n_beam, const uint64_t n_time_power_sum, const QudaPrecision prec, int stream_idx,
                unsigned int n_time, unsigned int n_chan, unsigned int n_chan_sum, unsigned int n_time_inner)
  {
    getProfile().TPSTART(QUDA_PROFILE_COMPUTE);
    logQuda(QUDA_VERBOSE, "Launching PowerSum with %d threads\n", N);
    instantiatePrecisionTEST<PowerSum>(prec, ps_data, input_A, input_B, ib_sum_data, n_time_power_sum, n_beam, N, stream_idx, n_time, n_chan, n_chan_sum, n_time_inner);
    getProfile().TPSTOP(QUDA_PROFILE_COMPUTE);
  }
  
} // namespace quda
