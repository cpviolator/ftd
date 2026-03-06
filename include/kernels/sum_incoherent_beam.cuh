#pragma once

#include <comm_ggp.h>
#include <kernel.h>
#include <register_traits.h>
#include <shared_memory_cache_helper.h>

namespace quda {

  template <typename Float> struct sumIBArg : public ReduceArg<double> {
    using real = typename mapper<Float>::type;
    Float *ib_sum;
    const Float *input_A;
    const Float *input_B;
    const int *flagants;
    unsigned long long int N;
    const int batches;
    
    sumIBArg(Float *ib_sum, const Float *input_A, const Float *input_B, const int *flagants, unsigned long long int N, const int batches) :
      ReduceArg<double>(dim3(N/batches, 1, batches), batches),
      ib_sum(ib_sum),
      input_A(input_A),
      input_B(input_B),
      flagants(flagants),
      N(N),
      batches(batches)
    {
      //printf("(ggp) SIC info N %llu, batches %d\n", N, batches);
    }
  };

  template <typename Float> struct sumIBArgSimple : kernel_param<> {
    using real = typename mapper<Float>::type;
    Float *ib_sum;
    const Float *input_A;
    const Float *input_B;
    const int *flagants;
    unsigned long long int N;
    const int batches;
    
    // compute type
    typedef typename mapper<Float>::type cache_prec_t;
    
    sumIBArgSimple(Float *ib_sum, const Float *input_A, const Float *input_B, const int *flagants, unsigned long long int N, const int batches) :
      kernel_param(dim3(N, 1, 1)),
      ib_sum(ib_sum),
      input_A(input_A),
      input_B(input_B),
      flagants(flagants),
      N(N),
      batches(batches)
    {
      //printf("(ggp) SIC info N %llu, batches %d\n", N, batches);
    }
  };

  template <typename Arg> struct sumIBOps {
    using CacheT = SharedMemoryCache<typename Arg::cache_prec_t>;    
    using Ops = KernelOps<CacheT>;
  };
  
  template <typename Arg> struct sumIBSimple : sumIBOps<Arg>::Ops {
    using cache_prec_t = typename Arg::cache_prec_t;
    const Arg &arg;
    using typename sumIBOps<Arg>::Ops::KernelOpsT;
    
    template<typename... OpsArgs>
    constexpr sumIBSimple(const Arg &arg, const OpsArgs&...ops) :  KernelOpsT(ops...), arg(arg) {}
    
    static constexpr const char *filename() { return KERNEL_FILE; }
    
    using Float = typename Arg::real;
    
    __device__ __host__ inline void operator()(int idx)
    {
      typename sumIBOps<Arg>::CacheT cache {*this};
      
      Float local_result = 0;
      Float block_result = 0;
      int ants = arg.N/arg.batches;
      idx = target::thread_idx().x + ants*target::block_idx().x;
      //int ant = arg.N/ants;
      
      //int idx = ant + (arg.N/arg.batches) * chan_time;
      if(arg.flagants[target::thread_idx().x] == 0 || true) {
	
	// batches is the number of channel-time reduction batches
	// reduce data from all antennae into each batch	
	int ELEMS = 96;
	
	if(idx < ELEMS && false) {
	  printf("(ggp) SIC thread %d block %d thread_blk %d: Input A_r = %f, Input A_i = %f, Input B_r = %f, Input B_i = %f\n",
		 idx, target::block_idx().x, target::thread_idx().x,
		 arg.input_A[2*idx], arg.input_A[2*idx+1],
		 arg.input_B[2*idx], arg.input_B[2*idx+1]);
	}
	
	local_result = (arg.input_A[2*idx]*arg.input_A[2*idx] + arg.input_A[2*idx+1]*arg.input_A[2*idx+1] +
			arg.input_B[2*idx]*arg.input_B[2*idx] + arg.input_B[2*idx+1]*arg.input_B[2*idx+1]);


	int remainder = ants - target::block_dim().x;
	idx += remainder;
	if(target::thread_idx().x < remainder) {
	  local_result += (arg.input_A[2*idx]*arg.input_A[2*idx] + arg.input_A[2*idx+1]*arg.input_A[2*idx+1] +
			   arg.input_B[2*idx]*arg.input_B[2*idx] + arg.input_B[2*idx+1]*arg.input_B[2*idx+1]);
	}
      }
      cache.save(local_result, target::thread_idx().x);
      //sync(); ?
      
      if(target::thread_idx().x == 0) {
	for(int i=0; i<target::block_dim().x; i++) block_result += cache.load(i); 
      }
      
      arg.ib_sum[target::block_idx().x] = block_result;
    }
  };  
  
    
  template <typename Arg> struct sumIB : plus<double> {
    using reduce_t = double;
    using plus<reduce_t>::operator();
    static constexpr int reduce_block_dim = 1;
    const Arg &arg;
    constexpr sumIB(const Arg &arg) : arg(arg) {}
    static constexpr const char *filename() { return KERNEL_FILE; }
    
    __device__ __host__ inline reduce_t operator()(reduce_t &result, int ant, int, int chan_time)
    {
      reduce_t local_result = 0;

      int idx = ant + (arg.N/arg.batches) * chan_time;
      if(arg.flagants[ant] == 0) {
	// batches is the number of channel-time reduction batches
	// reduce data from all antennae into each batch

	int ELEMS = 1024;
	
	if(idx < ELEMS && false) {
	  printf("(ggp) SIC thread %d ant %d chan_time %d: Input A_r = %f, Input A_i = %f, Input B_r = %f, Input B_i = %f\n",
		 idx, ant, chan_time,
		 arg.input_A[2*idx], arg.input_A[2*idx+1],
		 arg.input_B[2*idx], arg.input_B[2*idx+1]);
	}
	
	local_result = (arg.input_A[2*idx]*arg.input_A[2*idx] + arg.input_A[2*idx+1]*arg.input_A[2*idx+1] +
			arg.input_B[2*idx]*arg.input_B[2*idx] + arg.input_B[2*idx+1]*arg.input_B[2*idx+1]);
	
      } else {
	printf("(ggp) SIC FLAGGED %d thread %d ant %d chan_time %d: Input A_r = %f, Input A_i = %f, Input B_r = %f, Input B_i = %f\n",
	       ant, idx, ant, chan_time,
	       arg.input_A[2*idx], arg.input_A[2*idx+1],
	       arg.input_B[2*idx], arg.input_B[2*idx+1]);
      }
      
      return plus::operator()(local_result, result);      
    }    
  };  
}
