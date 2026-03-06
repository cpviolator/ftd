#include <tunable_nd.h>
#include <tunable_reduction.h>
#include <instantiate.h>
#include <device_vector.h>
#include <kernels/inspect_dev_elems.cuh>

namespace quda {
  
  // Promote data from (4,4)b complex to PrecOut complex planar
  template <typename Prec> class InspectDevElems : TunableKernel1D {
    
  protected:
    unsigned long long int N;
    unsigned long long int N_low;
    unsigned long long int N_high;
    int ID;
    const void *array;
    QudaPrecision prec;
    unsigned int minThreads() const { return N; }

  public:
    InspectDevElems(const QudaPrecision prec, const void *array, const unsigned long long int N, const unsigned long long int N_low, const unsigned long long int N_high, const int ID) : 
      TunableKernel1D(N, QUDA_CUDA_FIELD_LOCATION),
      N(N),
      N_low(N_low),
      N_high(N_high),
      ID(ID),
      array(array),
      prec(prec)
    {
      apply(device::get_default_stream());
    }
    
    void apply(const qudaStream_t &stream)
    {
      TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
      if(prec == QUDA_HALF_PRECISION) launch<ElemHalfInspect>(tp, stream, ElemInspArg<Prec>((Prec*)array, N, N_low, N_high, ID));
      else launch<ElemInspect>(tp, stream, ElemInspArg<Prec>((Prec*)array, N, N_low, N_high, ID));
    }
    
    long long flops() const { return 0; }    
    
    // Amount of H2D data transfer in kernel.
    long long bytes() const { return N * sizeof(Prec); }
    
  };
  
  void inspectDevElems(const void *array,  const unsigned long long int N, const unsigned long long int N_low, const unsigned long long int N_high, const QudaPrecision prec, const int ID)
  {
    getProfile().TPSTART(QUDA_PROFILE_COMPUTE);
    instantiatePrecisionTEST<InspectDevElems>(prec, array, N, N_low, N_high, ID);
    getProfile().TPSTOP(QUDA_PROFILE_COMPUTE);
  }


  
} // namespace quda
