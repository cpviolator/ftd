#pragma once

#include <comm_ggp.h>
#include <kernel.h>
#include <register_traits.h>
#include <shared_memory_cache_helper.h>

namespace quda {
  
#define MASK1 0x0F // (00001111)
#define MASK2 0xF0 // (11110000)

  template <typename cast_t> 
  __device__ __host__ inline cast_t complexCharCastReal(const unsigned char input) {

    return (cast_t)((char)((input & (unsigned char)(MASK1)) << 4) >> 4);
  }
  
  template <typename cast_t> 
  __device__ __host__ inline cast_t complexCharCastImag(const char input) {
    
    return  (cast_t)((char)((input & (unsigned char)(MASK2))) >> 4);    
  }

#undef MASK1
#undef MASK2

  template <typename Arg> struct computeReorderOps {
    using CacheT = SharedMemoryCache<typename Arg::cache_prec_t>;    
    using Ops = KernelOps<CacheT>;
  };
  
  template <typename storage_t, int n_pol_ = 2, int n_time_inner_ = 2> struct ReorderPolArgT : kernel_param<> {
    // Storage type
    using Prec = storage_t;

    // compute type
    typedef typename mapper<storage_t>::type cache_prec_t;
    
    Prec *output_A;
    Prec *output_B;
    const uint8_t *input;
    
    // shared dimensions
    int n_time;
    int n_chan;
    int n_ant;

    // inner-outer time dimension relationship
    static constexpr int n_time_inner = n_time_inner_;
    static constexpr int n_pol = n_pol_;
    
    ReorderPolArgT(Prec *output_A, Prec *output_B, const uint8_t *input, const int n_time, const int n_chan, const int n_ant) :
      kernel_param((n_time * n_chan * n_pol * n_ant)/2),
      output_A(output_A),
      output_B(output_B),
      input(input),
      n_time(n_time),
      n_chan(n_chan),
      n_ant(n_ant)      
    {
    }
  };
    
  template <typename Arg> struct DataReorderPolT : computeReorderOps<Arg>::Ops {
    using cache_prec_t = typename Arg::cache_prec_t;
    const Arg &arg;
    using typename computeReorderOps<Arg>::Ops::KernelOpsT;

    template<typename... OpsArgs>
    constexpr DataReorderPolT(const Arg &arg, const OpsArgs&...ops) :  KernelOpsT(ops...), arg(arg) {}
    
    static constexpr const char *filename() { return KERNEL_FILE; }

    // Get the storage type from the Arg, cast data to that type.
    typedef typename Arg::Prec Prec;
    
    __device__ __host__ inline void operator()(int idx)
    {
      typename computeReorderOps<Arg>::CacheT cache {*this};
      
      int time = idx / (arg.n_chan * arg.n_ant);
      int ant  = (idx / (arg.n_chan)) % arg.n_ant;
      int chan = idx % arg.n_chan;
      
      int ELEMS = 32;

      // DEBUG
      if(idx < ELEMS && false) {
	for(int t=0; t<Arg::n_time_inner; t++) {
	  int lin_idx = Arg::n_pol * (Arg::n_time_inner * idx + t);
	  printf("TFbf INPUT (linear): %d dra %e, dia %e, drb %e, dib %e -- %d %d %d %d\n", idx,
		 // Pol A
		 complexCharCastReal<Prec>(arg.input[lin_idx]),
		 complexCharCastImag<Prec>(arg.input[lin_idx]),
		 // Pol B
		 complexCharCastReal<Prec>(arg.input[lin_idx + 1]),
		 complexCharCastImag<Prec>(arg.input[lin_idx + 1]),
		 time, ant, chan, t);
	}
      }

      // linear index in out array
      int chan_out = chan * (Arg::n_time_inner * arg.n_time * arg.n_ant);      
      int time_outer_out = time * (arg.n_ant);
      int ant_out = ant;
      
      int t, inner, idx_in, idx_out;
#pragma unroll
      for(t=0; t<Arg::n_time_inner; t++) {
	
	idx_out = chan_out + t * (arg.n_time * arg.n_ant) + time_outer_out + ant_out;
	idx_in = Arg::n_time_inner * Arg::n_pol * idx;
	
	inner = t*Arg::n_pol;

	// Place promoted data directly in the cache
	cache.save(0.05*complexCharCastReal<Prec>(arg.input[idx_in + inner]), Arg::n_pol*Arg::n_time_inner*(t + 2*target::thread_idx().x)+0);
	cache.save(0.05*complexCharCastImag<Prec>(arg.input[idx_in + inner]), Arg::n_pol*Arg::n_time_inner*(t + 2*target::thread_idx().x)+1);

	inner = t*Arg::n_pol + 1;

	cache.save(0.05*complexCharCastReal<Prec>(arg.input[idx_in + inner]), Arg::n_pol*Arg::n_time_inner*(t + 2*target::thread_idx().x)+2);
	cache.save(0.05*complexCharCastImag<Prec>(arg.input[idx_in + inner]), Arg::n_pol*Arg::n_time_inner*(t + 2*target::thread_idx().x)+3);

	if(idx_out < ELEMS && false) {
	  printf("TFbf OUTPUT: thread %d in (%d and %d) out %d: dra %e, dia %e, drb %e, dib %e\n", idx, idx_in + inner - 1, idx_in + inner, idx_out,
		 arg.output_A[2*(idx_out)], arg.output_A[2*(idx_out)+1],
		 arg.output_B[2*(idx_out)], arg.output_B[2*(idx_out)+1]); 
	}
      }

      // Load from cache
      for(t=0; t<Arg::n_time_inner; t++) {
	
	idx_out = chan_out + t * (arg.n_time * arg.n_ant) + time_outer_out + ant_out;
      
	arg.output_A[2*(idx_out)]     = cache.load(Arg::n_pol*Arg::n_time_inner*(t + 2*target::thread_idx().x)+0);
	arg.output_A[2*(idx_out) + 1] = cache.load(Arg::n_pol*Arg::n_time_inner*(t + 2*target::thread_idx().x)+1);
	
	arg.output_B[2*(idx_out)]     = cache.load(Arg::n_pol*Arg::n_time_inner*(t + 2*target::thread_idx().x)+2);
	arg.output_B[2*(idx_out) + 1] = cache.load(Arg::n_pol*Arg::n_time_inner*(t + 2*target::thread_idx().x)+3);
      }
    }
  };

  
  
  template <typename storage_t> struct ReorderPolArg : kernel_param<> {
    // Storage type
    using Prec = storage_t;

    // compute type
    typedef typename mapper<storage_t>::type cache_prec_t;
    
    Prec *output_A;
    Prec *output_B;
    const uint8_t *input;

    // inner-outer time dimension relationship
    static constexpr int n_time_inner = 2;
    static constexpr int n_time_outer = 2048;
    
    // shared dimensions
    static constexpr int n_pol = 2;
    static constexpr int n_chan = 384;
    static constexpr int n_ant = 96 / 2;
    
    ReorderPolArg(Prec *output_A, Prec *output_B, const uint8_t *input) :
      kernel_param(2048 * 384 * 2 * (96/4)),
      output_A(output_A),
      output_B(output_B),
      input(input)
    {
    }
  };

  template <typename Arg> struct DataReorderPol : computeReorderOps<Arg>::Ops {
    using cache_prec_t = typename Arg::cache_prec_t;
    const Arg &arg;
    using typename computeReorderOps<Arg>::Ops::KernelOpsT;

    template<typename... OpsArgs>
    constexpr DataReorderPol(const Arg &arg, const OpsArgs&...ops) :  KernelOpsT(ops...), arg(arg) {}
    
    static constexpr const char *filename() { return KERNEL_FILE; }

    // Get the storage type from the Arg, cast data to that type.
    typedef typename Arg::Prec Prec;
    
    __device__ __host__ inline void operator()(int idx)
    {
      typename computeReorderOps<Arg>::CacheT cache {*this};
      
      int time = idx / (Arg::n_chan * Arg::n_ant);
      int ant  = (idx / (Arg::n_chan)) % Arg::n_ant;
      int chan = idx % Arg::n_chan;
      
      int ELEMS = 32;

      // DEBUG
      if(idx < ELEMS && false) {
	for(int t=0; t<Arg::n_time_inner; t++) {
	  int lin_idx = Arg::n_pol * (Arg::n_time_inner * idx + t);
	  printf("TFbf INPUT (linear): %d dra %e, dia %e, drb %e, dib %e -- %d %d %d %d\n", idx,
		 // Pol A
		 complexCharCastReal<Prec>(arg.input[lin_idx]),
		 complexCharCastImag<Prec>(arg.input[lin_idx]),
		 // Pol B
		 complexCharCastReal<Prec>(arg.input[lin_idx + 1]),
		 complexCharCastImag<Prec>(arg.input[lin_idx + 1]),
		 time, ant, chan, t);
	}
      }

      // linear index in out array
      int chan_out = chan * (Arg::n_time_inner * Arg::n_time_outer * Arg::n_ant);      
      int time_outer_out = time * (Arg::n_ant);
      int ant_out = ant;
      
      int t, inner, idx_in, idx_out;
#pragma unroll
      for(t=0; t<Arg::n_time_inner; t++) {
	
	idx_out = chan_out + t * (Arg::n_time_outer * Arg::n_ant) + time_outer_out + ant_out;
	idx_in = Arg::n_time_inner * Arg::n_pol * idx;
	
	inner = t*Arg::n_pol;

	// Place promoted data directly in the cache, check that save load is working correctly.
	//cache.save(0.05*complexCharCastReal<Prec>(arg.input[idx_in + inner]), 2*Arg::n_time_inner*target::thread_idx().x+0);
	//cache.save(0.05*complexCharCastImag<Prec>(arg.input[idx_in + inner]), 2*Arg::n_time_inner*target::thread_idx().x+1);
	//arg.output_A[2*(idx_out)]     = cache.load(2*Arg::n_time_inner*target::thread_idx().x+0);
	//arg.output_A[2*(idx_out) + 1] = cache.load(2*Arg::n_time_inner*target::thread_idx().x+1);
	
	arg.output_A[2*(idx_out)]     = 0.05*complexCharCastReal<Prec>(arg.input[idx_in + inner]);
	arg.output_A[2*(idx_out) + 1] = 0.05*complexCharCastImag<Prec>(arg.input[idx_in + inner]);	

	inner = t*Arg::n_pol + 1;

	arg.output_B[2*(idx_out)]     = 0.05*complexCharCastReal<Prec>(arg.input[idx_in + inner]);
	arg.output_B[2*(idx_out) + 1] = 0.05*complexCharCastImag<Prec>(arg.input[idx_in + inner]);
	
	if(idx_out < ELEMS && false) {
	  printf("TFbf OUTPUT: thread %d in (%d and %d) out %d: dra %e, dia %e, drb %e, dib %e\n", idx, idx_in + inner - 1, idx_in + inner, idx_out,
		 arg.output_A[2*(idx_out)], arg.output_A[2*(idx_out)+1],
		 arg.output_B[2*(idx_out)], arg.output_B[2*(idx_out)+1]); 
	}
      }
    }
  };
  
  template <typename storage_t> struct ReorderArg : kernel_param<> {
    
    // Storage type is the precision we want the
    // output to be in.
    using Prec = storage_t;

    // Cached data 
    typedef typename mapper<storage_t>::type cache_prec_t;  
    
    Prec *output;
    const uint8_t *input;

    // inner-outer time dimension relationship
    static constexpr int n_time_inner_110 = 2;
    static constexpr int n_time_inner_2k = 32;
    static constexpr int n_time_outer_2k = 128;

    static constexpr int n_time_inner = 2;
    static constexpr int n_time_outer = 2048;
    
    // shared dimensions
    static constexpr int n_pol = 2;
    static constexpr int n_chan = 384;
    static constexpr int n_ant = 96;
    
    ReorderArg(Prec *output, const uint8_t *input) :
      kernel_param(2048 * 96 * 384),
      output(output),
      input(input)
    {
    }
  };

  
  template <typename Arg> struct DataReorder : computeReorderOps<Arg>::Ops {
    using cache_prec_t = typename Arg::cache_prec_t;
    const Arg &arg;
    using typename computeReorderOps<Arg>::Ops::KernelOpsT;
    
    template<typename... OpsArgs>    
    constexpr DataReorder(const Arg &arg, const OpsArgs&...ops) : KernelOpsT(ops...), arg(arg) {}
    
    static constexpr const char *filename() { return KERNEL_FILE; }

    // Get the storage type from the Arg, cast data to that type.
    typedef typename Arg::Prec Prec;
    
    __device__ __host__ inline void operator()(int idx)
    {

      typename computeReorderOps<Arg>::CacheT cache {*this};

      
      // dsa110 input is [NPACKETS_PER_BLOCK, NANTS, NCHAN_PER_PACKET, 2 times, 2 pol, 4-bit complex]
      // or              [2048 n_time_outer, 96 ant, 384 channels, 2 time inner, 2 pol, 4-bit complex]
      // dsa2k input is  [384 channels, 128 time outer, 96 ant, 2 pol, 32 time inner, sizeof(Float)-bit complex]

      // We will place in cuBLAS consumption order:
      // double/float    [384 channels, 2 time inner, 2 pol, 16*128 time outer, 96 ant, sizeof(Float)-bit complex]
      // half/quarter    [Re/Im, 384 channels, 2 time inner, 2 pol, 16*128 time outer, 96 ant, sizeof(Float)-bit]

      // With respect to the input dsa110 data layout, we deduce the corresponding dsa2k index,
      // promote the data, then populate the output array.

      int chan = idx % Arg::n_chan;
      int ant = (idx / Arg::n_chan) % Arg::n_ant;
      int time = idx / (Arg::n_chan * Arg::n_ant); 

      int chan_2k = chan * (Arg::n_time_inner * Arg::n_pol * Arg::n_time_outer * Arg::n_ant);
      int ant_2k  = ant;
      int time_2k = time * Arg::n_ant;
      // This is the start of the 4 entries for time_inner and pol data for 2k
      int idx_2k = chan_2k + ant_2k + time_2k;
      // This is the start of the 4 entries for time_inner and pol data for 110
      int idx_110 = idx * Arg::n_pol * Arg::n_time_inner;

      //SharedMemoryCache<float> cache;
      
      int t, pol, inner;
      for(t=0; t<Arg::n_time_inner; t++) {
	for(pol=0; pol<Arg::n_pol; pol++) {
	  
	  inner = t*Arg::n_pol + pol;
	  
	  arg.output[2*(idx_2k + inner)]     = complexCharCastReal<Prec>(arg.input[idx_110 + inner]);
	  arg.output[2*(idx_2k + inner) + 1] = complexCharCastImag<Prec>(arg.input[idx_110 + inner]);

	  //arg.output[2*(idx_2k + inner)]     = (cache_prec_t)((char)(((unsigned char)(arg.input[idx_110 + inner]) & (unsigned char)(MASK1)) << 4) >> 4);
	  //arg.output[2*(idx_2k + inner) + 1] = (cache_prec_t)((char)(((unsigned char)(arg.input[idx_110 + inner]) & (unsigned char)(MASK2))) >> 4);
	}
      }
    }
  };
  
} // End namespace quda

#if 0
      //if(idx < 8 || (1024 <= idx && idx < 1032)) {
      if(time == 0 && chan_ant == 0) {
	printf("time %d ch_ant %d Block dims are %d %d %d\n", time, chan_ant, target::block_dim().x, target::block_dim().y, target::block_dim().z);
	printf("time %d ch_ant %d Block idxs are %d %d %d\n", time, chan_ant, target::block_idx().x, target::block_idx().y, target::block_idx().z);
	printf("time %d ch_ant %d Thread idxs are %d %d %d\n", time, chan_ant, target::thread_idx().x, target::thread_idx().y, target::thread_idx().z);
      }
      
      if(idx < 8 || (1024 <= idx && idx < 1032 && 0)) {	
	printf("idx %d Block dims are %d %d %d\n", idx, target::block_dim().x, target::block_dim().y, target::block_dim().z);
	printf("idx %d Block idxs are %d %d %d\n", idx, target::block_idx().x, target::block_idx().y, target::block_idx().z);
	printf("idx %d Thread idxs are %d %d %d\n", idx, target::thread_idx().x, target::thread_idx().y, target::thread_idx().z);
      }
#endif      
