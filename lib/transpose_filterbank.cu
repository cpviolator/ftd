#include <tunable_nd.h>
#include <tunable_reduction.h>
#include <instantiate.h>
#include <device_vector.h>
#include <kernels/transpose_filterbank.cuh>

namespace quda {
  
  template <typename Prec> class TransposeFilterbank : TunableKernel2D {
    
    // Copies of function arguments
  protected:
    unsigned int n_chan;
    unsigned int n_beam;
    void *output;
    const void *input;
    // Kernel thread volume (member of TunableKernel)
    unsigned int minThreads() const { return n_chan * n_beam; }
    int stream_idx;    
  public:
    // We use a blank function signature on `const QudaPrecision` to supress `-Wunused-parameter` compiler warnings.
    TransposeFilterbank(const QudaPrecision, void *output, const void *input, const unsigned int n_chan, const unsigned int n_beam, int stream_idx) : 
      TunableKernel2D(n_chan * n_beam, n_beam, QUDA_CUDA_FIELD_LOCATION),
      n_chan(n_chan),
      n_beam(n_beam),
      output(output),
      input(input),
      stream_idx(stream_idx)
    {
      apply(device::get_stream(stream_idx));
    }

    // `apply` is a `TunableKernel1D` member function that runs both the tuning step and the kernel launch
    // with the tuned or untuned kernel bounds.
    void apply(const qudaStream_t &stream) {
      TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
      FBTArg<Prec> arg((Prec*)output, (Prec*)input);
      launch<FilterbankTranspose>(tp, stream, arg);
    }
    
    // FLOPS in kernel: 
    long long flops() const { return 0; }    
    
    // Amount of data transfer in kernel.
    long long bytes() const { return n_chan * n_beam * sizeof(Prec); }    
  };
  
  void transposeFilterbank(void *output, void *input, const uint64_t n_chan, const uint64_t n_beam, const QudaPrecision prec, int stream_idx)
  {
    logQuda(QUDA_VERBOSE, "Launching TF kernel with = %lu threads\n", n_chan * n_beam);    
    getProfile().TPSTART(QUDA_PROFILE_COMPUTE);
    instantiatePrecisionTEST<TransposeFilterbank>(prec, output, input, n_chan, n_beam, stream_idx);
    getProfile().TPSTOP(QUDA_PROFILE_COMPUTE);
  }
  
} // namespace quda
