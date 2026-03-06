#include <tunable_nd.h>
#include <tunable_reduction.h>
#include <instantiate.h>
#include <device_vector.h>
#include <kernels/sum_incoherent_beam.cuh>

#define LOCAL_DEBUG
//#define DEVICE_REDUCE

namespace quda {

  template <typename Prec> class SumIncoherentBeamSimple : TunableKernel1D {
    
    // Copies of function arguments
  protected:
    unsigned long long int N;
    int batches;
    const void *input_A;
    const void *input_B;
    const void *flagants;
    void *ib_sum;
    int stream_idx;
    
  public:
    // We use a blank function signature on `const QudaPrecision` to supress `-Wunused-parameter` compiler warnings.
    SumIncoherentBeamSimple(const QudaPrecision, void *ib_sum, const void *input_A, const void *input_B, const void *flagants, const unsigned long long int N, const int batches, int stream_idx) : 
      TunableKernel1D(N, QUDA_CUDA_FIELD_LOCATION),
      N(N),
      batches(batches),
      input_A(input_A),
      input_B(input_B),
      flagants(flagants),
      ib_sum(ib_sum),
      stream_idx(stream_idx)
    {
      apply(device::get_stream(stream_idx));
    }

    // `apply` is a `TunableKernel1D` member function that runs both the tuning step and the kernel launch
    // with the tuned or untuned kernel bounds.
    void apply(const qudaStream_t &stream) {
      TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());	  
      sumIBArgSimple<Prec> arg((Prec*)ib_sum, (Prec*)input_A, (Prec*)input_B, (int*)flagants, N, batches);
      launch<sumIBSimple>(tp, stream, arg);
#if 0
      // Inspect ib_sum to ensure proper transfer has occured
      if(!activeTuning()) {
	std::vector<Prec> result_local_prec(batches, 0.0);
	qudaMemcpy(result_local_prec.data(), ib_sum, batches*sizeof(Prec), qudaMemcpyDeviceToHost);
	//for(int j=0; j<batches; j++) {
	for(int j=0; j<16; j++) {
	  if((float)result_local_prec[j] > 0) printfQuda("SIB_simple bidx %d, ib_sum=%f\n", j, (float)result_local_prec[j]);	
	}
      }
#endif
      
    }
    bool tuneSharedBytes() const { return false; }    
    unsigned int sharedBytesPerBlock(const TuneParam &param) const {
      // Each thread will have
      size_t sz = 32*sizeof(Prec);
      return sz;
    }
    
    // Kernel thread volume (member of TunableKernel1D)
    unsigned int minThreads() const { return batches*32; }
    unsigned int maxBlockSize(const TuneParam &) const {
      //if((N/batches)%32 == 0) return N/batches;
      //else return N/batches + (N/batches)%32;
      return 32;
    }
    
    // FLOPS in kernel: compute(8) + reduction(1) 
    long long flops() const { return N * 8 + N; }    
    
    // Amount of data transfer in kernel.
    long long bytes() const { return N * (3 * sizeof(Prec) + sizeof(int)); }    
  };
  
  template <typename Prec> class SumIncoherentBeam : TunableMultiReduction {
    
    // Copies of function arguments
  protected:
    unsigned long long int N;
    int batches;
    const void *input_A;
    const void *input_B;
    const void *flagants;
    void *ib_sum;
    int stream_idx;

    // Kernel thread volume (member of TunableKernel1D)
    unsigned int minThreads() const { return N; }
    
  public:
    // We use a blank function signature on `const QudaPrecision` to supress `-Wunused-parameter` compiler warnings.
    SumIncoherentBeam(const QudaPrecision, void *ib_sum, const void *input_A, const void *input_B, const void *flagants, const unsigned long long int N, const int batches, int stream_idx) : 
      TunableMultiReduction(N, batches, 1, QUDA_CUDA_FIELD_LOCATION),
      N(N),
      batches(batches),
      input_A(input_A),
      input_B(input_B),
      flagants(flagants),
      ib_sum(ib_sum),
      stream_idx(stream_idx)
    {
      apply(device::get_stream(stream_idx));
    }

    // `apply` is a `TunableKernel1D` member function that runs both the tuning step and the kernel launch
    // with the tuned or untuned kernel bounds.
    void apply(const qudaStream_t &stream) {
      TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
      std::vector<double> result_local(batches, 0.0);
#ifdef DEVICE_REDUCE
      qudaDeviceSynchronize();
      sumIBArg<Prec> arg((Prec*)ib_sum, (Prec*)input_A, (Prec*)input_B, (int*)flagants, N, batches);
      commAsyncReductionSet(true);
      launch<sumIB>(result_local, tp, stream, arg);
      qudaMemcpy(ib_sum, reducer::get_device_buffer(), batches*sizeof(Prec), qudaMemcpyDeviceToDevice);
      commAsyncReductionSet(false);

#ifdef LOCAL_DEBUG
      // Inspect ib_sum to ensure proper transfer has occured
      if(!activeTuning()) {
	std::vector<Prec> result_local_prec(batches, 0.0);
	qudaMemcpy(result_local_prec.data(), ib_sum, batches*sizeof(Prec), qudaMemcpyDeviceToHost);
	for(int j=0; j<batches; j++) {
	  //for(int j=0; j<16; j++) {
	  printfQuda("SIB_device bidx %d, ib_sum=%f\n", j, (float)result_local_prec[j]);	
	}
      }
#endif
#else
      sumIBArg<Prec> arg((Prec*)ib_sum, (Prec*)input_A, (Prec*)input_B, (int*)flagants, N, batches);
      launch<sumIB>(result_local, tp, stream, arg);
#ifdef LOCAL_DEBUG
      // Inspect ib_sum to ensure proper transfer has occured
      std::vector<Prec> result_local_prec(batches, 0.0);
      if(!activeTuning()) {
	for(int j=0; j<batches; j++) {
	  //for(int j=0; j<16; j++) {
	  result_local_prec[j] = result_local[j];
	  //printfQuda("SIB_host bidx %d, ib_sum=%f\n", j, result_local_prec[j]);	
	}
      }
#endif
      qudaMemcpyAsync(ib_sum, result_local_prec.data(), batches*sizeof(Prec), qudaMemcpyHostToDevice, stream);
#endif
      
    }
    // FLOPS in kernel: compute(8) + reduction(1) 
    long long flops() const { return N * 8 + N; }    
    
    // Amount of data transfer in kernel.
    long long bytes() const { return N * (3 * sizeof(Prec) + sizeof(int)); }    
  };
  
  void sumIncoherentBeam(void *ib_sum, const void *flagants, const void *input_A, const void *input_B, const uint64_t N, const uint64_t batches, const QudaPrecision prec, int stream_idx)
  {
    if(N % batches != 0) errorQuda("batches %lu does not evenly divide N %lu (N mod batches = %lu\n", batches, N, N%batches);
    logQuda(QUDA_VERBOSE, "Launching Sum IB with %lu threads, %lu batches, %lu items per batch\n", N, batches, N/batches);
    getProfile().TPSTART(QUDA_PROFILE_COMPUTE);
    if(batches < 128) instantiatePrecisionTEST<SumIncoherentBeam>(prec, ib_sum, input_A, input_B, flagants, N, batches, stream_idx);
    else instantiatePrecisionTEST<SumIncoherentBeamSimple>(prec, ib_sum, input_A, input_B, flagants, N, batches, stream_idx);
    getProfile().TPSTOP(QUDA_PROFILE_COMPUTE);
  }
  
} // namespace quda
