#include <tunable_nd.h>
#include <tunable_reduction.h>
#include <instantiate.h>
#include <device_vector.h>
#include <kernels/promote_data.cuh>

namespace quda {

  // Promote data from (4,4)b complex to PrecOut complex planar
  template <typename PrecOut> class PromoteDataPlanar : TunableKernel1D {
    
  protected:
    unsigned long long int N;
    const void *input_data;
    void *output_real;
    void *output_imag;
    unsigned int minThreads() const { return N; }
    int stream_idx;
    
  public:
    // We use a blank function signature on `const QudaPrecision` to supress `-Wunused-parameter` compiler warnings.
    PromoteDataPlanar(const QudaPrecision, void *output_real, void *output_imag, const void *input_data, const unsigned long long int N, int stream_idx) : 
      TunableKernel1D(N, QUDA_CUDA_FIELD_LOCATION),
      N(N),
      input_data(input_data),
      output_real(output_real),
      output_imag(output_imag),
      stream_idx(stream_idx)
    {
      apply(device::get_stream(stream_idx));
    }
    
    void apply(const qudaStream_t &stream)
    {
      TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
      launch<DataPlanarPromote>(tp, stream, PromotePlanarArg<PrecOut>((PrecOut*)output_real, (PrecOut*)output_imag, (uint8_t*)input_data, N));
    }
    
    long long flops() const { return 0; }    
    
    // Amount of H2D data transfer in kernel.
    long long bytes() const { return N * (sizeof(uint8_t) + sizeof(PrecOut)); }
    
  };
  
  void promoteDataPlanar(void *output_real, void *output_imag, const void *input_data, const unsigned long long int N, const QudaPrecision precOut, int stream_idx)
  {
    getProfile().TPSTART(QUDA_PROFILE_COMPUTE);
    instantiatePrecisionTEST<PromoteDataPlanar>(precOut, output_real, output_imag, input_data, N, stream_idx);
    getProfile().TPSTOP(QUDA_PROFILE_COMPUTE);
  }

  // Promote data from (4,4)b complex interleaved to PrecOut complex inreleaved
  template <typename PrecOut> class PromoteData : TunableKernel1D {
    
    // Copies of function arguments
  protected:
    unsigned long long int N;
    const void *input;
    void *output;
    // Kernel thread volume (member of TunableKernel1D)
    unsigned int minThreads() const { return N; }
    int stream_idx;
    
  public:
    // We use a blank function signature on `const QudaPrecision` to supress `-Wunused-parameter` compiler warnings.
    PromoteData(const QudaPrecision, void *output, const void *input, const unsigned long long int N, int stream_idx) : 
      TunableKernel1D(N, QUDA_CUDA_FIELD_LOCATION),
      N(N),
      input(input),
      output(output),
      stream_idx(stream_idx)
    {
      apply(device::get_stream(stream_idx));
    }

    // `apply` is a `TunableKernel1D` member function that runs both the tuning step and the kernel launch
    // with the tuned or untuned kernel bounds.
    void apply(const qudaStream_t &stream) {
      //printfQuda("Launching PromoteData with stream %d\n", stream_idx);
      TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
      launch<DataPromote>(tp, stream, PromoteArg<PrecOut>((PrecOut*)output, (uint8_t*)input, N));
    }

    // `TunableKernel1D` member for FLOPS
    long long flops() const { return 0; }    
    
    // Amount of data transfer in kernel.
    long long bytes() const { return N * (sizeof(uint8_t) + sizeof(PrecOut)); }    
  };
  
  void promoteData(void *output, const void *input, const unsigned long long int N, const QudaPrecision prec_out, int stream_idx)
  {
    getProfile().TPSTART(QUDA_PROFILE_COMPUTE);
    instantiatePrecisionTEST<PromoteData>(prec_out, output, input, N, stream_idx);
    getProfile().TPSTOP(QUDA_PROFILE_COMPUTE);
  }

  // Promote data from (4,4)b complex interleaved to PrecOut complex interleaved for two polarisations
  template <typename PrecOut> class PromoteDataPol : TunableKernel1D {
    
    // Copies of function arguments
  protected:
    unsigned long long int N;
    const void *input;
    void *output_A;
    void *output_B;
    // Kernel thread volume (member of TunableKernel1D)
    unsigned int minThreads() const { return N; }
    int stream_idx;
    
  public:
    // We use a blank function signature on `const QudaPrecision` to supress `-Wunused-parameter` compiler warnings.
    PromoteDataPol(const QudaPrecision, void *output_A, void *output_B, const void *input, const unsigned long long int N, int stream_idx) : 
      TunableKernel1D(N, QUDA_CUDA_FIELD_LOCATION),
      N(N),
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
      TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
      launch<DataPromotePol>(tp, stream, PromotePolArg<PrecOut>((PrecOut*)output_A, (PrecOut*)output_B, (uint8_t*)input, N));
    }

    // `TunableKernel1D` member for FLOPS
    long long flops() const { return 0; }    
    
    // Amount of data transfer in kernel.
    long long bytes() const { return N * (sizeof(uint8_t) + sizeof(PrecOut)); }    
  };
  
  void promoteDataPol(void *output_A, void *output_B, const void *input, const unsigned long long int N, const QudaPrecision prec_out, int stream_idx)
  {
    getProfile().TPSTART(QUDA_PROFILE_COMPUTE);
    instantiatePrecisionTEST<PromoteDataPol>(prec_out, output_A, output_B, input, N, stream_idx);
    getProfile().TPSTOP(QUDA_PROFILE_COMPUTE);
  }
  
} // namespace quda
