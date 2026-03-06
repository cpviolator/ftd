#include <tunable_nd.h>
#include <tunable_reduction.h>
#include <instantiate.h>
#include <device_vector.h>
#include <kernels/interleave_tri.cuh>

namespace quda {

  template <typename Prec> class InterleaveTri : TunableKernel1D {
    
  protected:
    unsigned long long int N;
    const void *output_real;
    const void *output_imag;
    void *output_full;
    QudaPrecision prec_out;
    unsigned int minThreads() const { return N; }

  public:
    InterleaveTri(const QudaPrecision prec_out, void *output_full, const void *output_real, const void *output_imag, const unsigned long long int N) : 
      TunableKernel1D(N, QUDA_CUDA_FIELD_LOCATION),
      N(N),
      output_real(output_real),
      output_imag(output_imag),
      output_full(output_full),
      prec_out(prec_out)
    {
      apply(device::get_default_stream());
    }
    
    void apply(const qudaStream_t &stream)
    {
      TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
      switch(prec_out) {
      case QUDA_DOUBLE_PRECISION: launch<PromTriInterFloat>(tp, stream, InterTriArg<Prec>((Prec*)output_full, (half*)output_real, (half*)output_imag, N)); break;
      case QUDA_SINGLE_PRECISION: launch<PromTriInterFloat>(tp, stream, InterTriArg<Prec>((Prec*)output_full, (half*)output_real, (half*)output_imag, N)); break;
      case QUDA_HALF_PRECISION: launch<PromTriInterFloat>(tp, stream, InterTriArg<Prec>((Prec*)output_full, (half*)output_real, (half*)output_imag, N)); break;
      default:
	errorQuda("Unknown Quda precision %d", prec_out);
      }
    }
    
    long long flops() const { return 0; }    
    
    // Amount of H2D data transfer in kernel.
    long long bytes() const { return N * sizeof(Prec); }    
  };
  
  void promInterTri(void *output_full, const void *input_real, const void *input_imag, const unsigned long long int N, const QudaPrecision prec)
  {
    getProfile().TPSTART(QUDA_PROFILE_COMPUTE);
    instantiatePrecisionTEST<InterleaveTri>(prec, output_full, input_real, input_imag, N);
    getProfile().TPSTOP(QUDA_PROFILE_COMPUTE);
  }

  template <typename Prec> class Interleave : TunableKernel1D {
    
  protected:
    unsigned long long int N;
    const void *output_real;
    const void *output_imag;
    void *output_full;
    QudaPrecision prec_out;
    unsigned int minThreads() const { return N; }

  public:
    Interleave(const QudaPrecision prec_out, void *output_full, const void *output_real, const void *output_imag, const unsigned long long int N) : 
      TunableKernel1D(N, QUDA_CUDA_FIELD_LOCATION),
      N(N),
      output_real(output_real),
      output_imag(output_imag),
      output_full(output_full),
      prec_out(prec_out)
    {
      apply(device::get_default_stream());
    }
    
    void apply(const qudaStream_t &stream)
    {
      TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
      switch(prec_out) {
      case QUDA_DOUBLE_PRECISION: launch<PromInterFloat>(tp, stream, InterArg<Prec>((Prec*)output_full, (half*)output_real, (half*)output_imag, N)); break;
      case QUDA_SINGLE_PRECISION: launch<PromInterFloat>(tp, stream, InterArg<Prec>((Prec*)output_full, (half*)output_real, (half*)output_imag, N)); break;
      case QUDA_HALF_PRECISION: launch<PromInterFloat>(tp, stream, InterArg<Prec>((Prec*)output_full, (half*)output_real, (half*)output_imag, N)); break;
      default:
	errorQuda("Unknown Quda precision %d", prec_out);
      }
    }
    
    long long flops() const { return 0; }    
    
    // Amount of H2D data transfer in kernel.
    long long bytes() const { return N * sizeof(Prec); }    
  };
  
  void promInter(void *output_full, const void *input_real, const void *input_imag, const unsigned long long int N, const QudaPrecision prec)
  {
    getProfile().TPSTART(QUDA_PROFILE_COMPUTE);
    instantiatePrecisionTEST<Interleave>(prec, output_full, input_real, input_imag, N);
    getProfile().TPSTOP(QUDA_PROFILE_COMPUTE);
  }

  
} // namespace quda
