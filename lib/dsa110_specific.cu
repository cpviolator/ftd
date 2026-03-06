#include <tunable_nd.h>
#include <tunable_reduction.h>
#include <instantiate.h>
#include <device_vector.h>
#include <kernels/compute_weights.cuh>

namespace quda {

  template <typename PrecOut> class ComputeWeights : TunableKernel1D {
    
    // Copies of function arguments
  protected:
    unsigned long long int N;
    void *weights_A;
    void *weights_B;
    const int n_arm;
    const void *ant_E;
    const void *ant_N;
    const void *calibs;
    const void *flagants;
    const void *freqs;

    PrecOut dec;
    // Kernel thread volume (member of TunableKernel1D)
    unsigned int minThreads() const { return N; }
    
  public:
    // We use a blank function signature on `const QudaPrecision` to supress `-Wunused-parameter` compiler warnings.
    ComputeWeights(const QudaPrecision, void *weights_A, void *weights_B, const void *ant_E, const void *ant_N, const void *calibs, const void *flagants, const void *freqs, float dec, const int n_arm, const unsigned long long int N) : 
      TunableKernel1D(N, QUDA_CUDA_FIELD_LOCATION),
      N(N),
      weights_A(weights_A),
      weights_B(weights_B),
      ant_E(ant_E),
      ant_N(ant_N),
      calibs(calibs),
      flagants(flagants),
      freqs(freqs),
      dec(dec),
      n_arm(n_arm)
    {
      switch (n_arm) {
      case 1: strcat(aux, "n-arm1,"); break;
      case 2: strcat(aux, "n-arm2,"); break;
      default: errorQuda("Unsupported number of arms %d. Please use 1, 2, or add template launch", n_arm);
      }
      apply(device::get_default_stream());
    }
    
    // `apply` is a `TunableKernel1D` member function that runs both the tuning step and the kernel launch
    // with the tuned or untuned kernel bounds.
    void apply(const qudaStream_t &stream) {
      TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());      
      logQuda(QUDA_VERBOSE, "Launch WeightsCompute with %llu threads\n", N);
      switch (n_arm) {
      case 1: launch<WeightsCompute>(tp, stream, WeightsComputeArg<PrecOut>((PrecOut*)weights_A, (PrecOut*)weights_B, (PrecOut*)ant_E, (PrecOut*)ant_N, (int*)flagants, (PrecOut*)calibs, (PrecOut*)freqs, dec, N));
	break;
      case 2: launch<WeightsCompute110>(tp, stream, WeightsComputeArg110<PrecOut>((PrecOut*)weights_A, (PrecOut*)weights_B, (PrecOut*)ant_E, (PrecOut*)ant_N, (int*)flagants, (PrecOut*)calibs, (PrecOut*)freqs, dec, N));
	break;
      default: errorQuda("Unsupported number of arms %d. Please use 1, 2, or add template launch", n_arm);
      }
      logQuda(QUDA_VERBOSE, "WeightsCompute kernel Launch complete\n");
    }
    
    // `TunableKernel1D` member for FLOPS
    long long flops() const { return 48*N; }    
    
    // Amount of data transfer in kernel.
    long long bytes() const { return N * (sizeof(short) + sizeof(PrecOut)); }    
  };
  
  void computeWeights(void *weights_A, void *weights_B, const void *ant_E, const void *ant_N, const void *calibs, const void *flagants, const void *freq, const float dec, const int n_arm, const unsigned long long int N, const QudaPrecision prec_out)
  {
    getProfile().TPSTART(QUDA_PROFILE_COMPUTE);
    printfQuda("Instantiate with %llu threads\n", N);
    instantiatePrecisionTEST<ComputeWeights>(prec_out, weights_A, weights_B, ant_E, ant_N, calibs, flagants, freq, dec, n_arm, N);
    getProfile().TPSTOP(QUDA_PROFILE_COMPUTE);
  }

  
} //namespace quda
