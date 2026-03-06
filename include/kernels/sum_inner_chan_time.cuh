#pragma once

#include <comm_ggp.h>
#include <kernel.h>
#include <register_traits.h>
#include <shared_memory_cache_helper.h>

namespace quda {

  template <typename Float, int n_chan_sum_ = 8, int n_time_inner_ = 2> struct SICTArgT : kernel_param<> {
    using real = typename mapper<Float>::type;
    unsigned char *output;
    const Float *ps_data;

    int n_time;
    int n_beam;
    int n_chan;
    static constexpr int n_chan_sum = n_chan_sum_;
    static constexpr int n_time_inner = n_time_inner_;    
    const unsigned long long int N;
    
    SICTArgT(unsigned char *output, const Float *ps_data, const unsigned long long int N, const int n_time, const int n_beam, const int n_chan) :
      kernel_param(dim3(N, 1, 1)),
      output(output),
      ps_data(ps_data),
      n_time(n_time),
      n_beam(n_beam),
      n_chan(n_chan),
      N(N)
    {
    }
  };

  // Input is [NPACKETS_PER_BLOCK/4, NCHAN_PER_PACKET/8, NBEAMS/2, 8chan, 2 times] 
  // want to sum over 8 chan and 2 times
  // Then do a transpose to [NBEAMS, NPACKETS_PER_BLOCK/4, NCHAN_PER_PACKET/8]

  template <typename Arg> struct sumChanTimeInnerT {
    const Arg &arg;
    constexpr sumChanTimeInnerT(const Arg &arg) : arg(arg) {}
    static constexpr const char *filename() { return KERNEL_FILE; }
    using Float = typename Arg::real;
        
    __device__ __host__ void operator()(int idx)
    {
      Float sum = 70;

      int n_time = arg.n_time;
      int n_beam = arg.n_beam;
      int n_chan = arg.n_chan;
      int n_chan_sum = Arg::n_chan_sum;
      int n_time_inner = Arg::n_time_inner;
      
      int ELEMS = 1024;
      if (idx < ELEMS && false) printf("(ggp) STS thread %d Input %f\n", idx, arg.ps_data[idx]);

      int sums = Arg::n_chan_sum * Arg::n_time_inner;
#pragma unroll
      for(int i=0; i<sums; i++)
	sum += arg.ps_data[sums * idx + i];
      
      // transpose to [(A)NBEAMS, (B)NPACKETS_PER_BLOCK/4, (C)NCHAN_PER_PACKET/8]
      int A = idx % n_beam;
      int B = idx / ((n_chan/n_chan_sum) * n_beam);
      int C = (idx / n_beam) % (n_chan/n_chan_sum);
      
      int idx_out = (A * (n_chan/n_chan_sum) * n_time +
		     B * (n_chan/n_chan_sum) +
		     C);
      
      arg.output[idx_out] = (unsigned char)sum;
    }    
  };

  


  
  template <typename Float, int n_chan_sum_ = 8, int n_time_inner_ = 2> struct SICTArg : kernel_param<> {
    using real = typename mapper<Float>::type;
    unsigned char *output;
    const Float *ps_data;

    static constexpr int n_time = 2048/4;
    static constexpr int n_beam = 256;
    static constexpr int n_chan = 384;
    static constexpr int n_chan_sum = n_chan_sum_;
    static constexpr int n_time_inner = n_time_inner_;    
    const unsigned long long int N;
    
    SICTArg(unsigned char *output, const Float *ps_data, const unsigned long long int N) :
      kernel_param(dim3(N, 1, 1)),
      output(output),
      ps_data(ps_data),
      N(N)
    {
    }
  };

  // Input is [NPACKETS_PER_BLOCK/4, NCHAN_PER_PACKET/8, NBEAMS/2, 8chan, 2 times] 
  // want to sum over 8 chan and 2 times
  // Then do a transpose to [NBEAMS, NPACKETS_PER_BLOCK/4, NCHAN_PER_PACKET/8]

  template <typename Arg> struct sumChanTimeInner {
    const Arg &arg;
    constexpr sumChanTimeInner(const Arg &arg) : arg(arg) {}
    static constexpr const char *filename() { return KERNEL_FILE; }
    using Float = typename Arg::real;
        
    __device__ __host__ void operator()(int idx)
    {
      Float sum = 70;

      int n_time = Arg::n_time;
      int n_beam = Arg::n_beam;
      int n_chan = Arg::n_chan;
      int n_chan_sum = Arg::n_chan_sum;
      int n_time_inner = Arg::n_time_inner;

      int ELEMS = 1024;
      if (idx < ELEMS && false) printf("(ggp) STS thread %d Input %f\n", idx, arg.ps_data[idx]);

      int sums = Arg::n_chan_sum * Arg::n_time_inner;
#pragma unroll
      for(int i=0; i<sums; i++)
	sum += arg.ps_data[sums * idx + i];
      
      // transpose to [(A)NBEAMS, (B)NPACKETS_PER_BLOCK/N_POWER_SUM, (C)NCHAN_PER_PACKET/N_CHAN_SUM]
      int A = idx % n_beam;
      int B = idx / ((n_chan/n_chan_sum) * n_beam);
      int C = (idx / n_beam) % (n_chan/n_chan_sum);
      
      int idx_out = (A * (n_chan/n_chan_sum) * n_time +
		     B * (n_chan/n_chan_sum) +
		     C);
      
      arg.output[idx_out] = (unsigned char)sum;
    }    
  };

}
