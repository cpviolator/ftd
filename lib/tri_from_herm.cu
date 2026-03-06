#include <tunable_nd.h>
#include <tunable_reduction.h>
#include <instantiate.h>
#include <device_vector.h>
#include <kernels/tri_from_herm.cuh>

namespace quda {
  
  // Triangulate data from Hermitian
  template <typename Prec> class TriFromHerm : TunableKernel1D {
    
  protected:
    unsigned long long int N;
    unsigned long long int N_batch;
    const void *input;
    void *output;
    unsigned int minThreads() const { return N; }
    int stream_idx;
    
  public:
    TriFromHerm(const QudaPrecision, void *output, const void *input, const unsigned long long int N, const unsigned long long int N_batch, int stream_idx) : 
      TunableKernel1D(N, QUDA_CUDA_FIELD_LOCATION),
      N(N),
      N_batch(N_batch),
      input(input),
      output(output),
      stream_idx(stream_idx)
    {
      apply(device::get_stream(stream_idx));
    }
    
    void apply(const qudaStream_t &stream)
    {
      TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
      launch<HermToTri>(tp, stream, HermToTriArg<Prec>((complex<Prec>*)output, (complex<Prec>*)input, N, N_batch));
    }
    
    long long flops() const { return 0; }    
    
    // Amount of data transfer in kernel.
    long long bytes() const { return ((N * 2 + N) * sizeof(Prec)); }
    
  };
  
  void triangulateFromHerm(void *tri_output, const void *input, const unsigned long long int N, const unsigned long long int N_batch, const QudaPrecision prec, int stream_idx)
  {
    if(N%N_batch != 0) {
      errorQuda("Total complex elements in array %llu not a multiple of the number of complex elements in a matrix %llu", N, N_batch);
    }
    
    getProfile().TPSTART(QUDA_PROFILE_COMPUTE);
    instantiatePrecisionTEST<TriFromHerm>(prec, tri_output, input, N, N_batch, stream_idx);
    getProfile().TPSTOP(QUDA_PROFILE_COMPUTE);
  }


  
} // namespace quda
