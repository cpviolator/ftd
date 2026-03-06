#pragma once

#include <comm_ggp.h>
#include <kernel.h>
#include <register_traits.h>
#include <shared_memory_cache_helper.h>

namespace quda {

  template <typename Float, int n_time_power_sum_, int n_beam_> struct PSArg : kernel_param<> {
    using real = typename mapper<Float>::type;
    Float *ps_data;
    const Float *input_A;
    const Float *input_B;
    const Float *ib_sum_data;
    int n_time;
    int n_chan;
    static constexpr int n_beam = n_beam_;

    static constexpr int n_power_sum = n_time_power_sum_;
    int n_chan_sum;
    int n_time_inner;

    unsigned long long int N;

    PSArg(Float *ps_data, const Float *input_A, const Float *input_B, const Float *ib_sum_data, unsigned long long int N,
          int n_time_, int n_chan_, int n_chan_sum_, int n_time_inner_) :
      kernel_param(dim3(N, 1, 1)),
      ps_data(ps_data),
      input_A(input_A),
      input_B(input_B),
      ib_sum_data(ib_sum_data),
      n_time(n_time_),
      n_chan(n_chan_),
      n_chan_sum(n_chan_sum_),
      n_time_inner(n_time_inner_),
      N(N)
    {
    }
  };

  /* POWER SUM AND TRANSPOSE OUTPUT
     - Input for each pol and r/i is [NCHAN_PER_PACKET/8, NBEAMS/2, 8chan, 2 times, NPACKETS_PER_BLOCK]
     - want to form total power, and sum total powers over 4 PACKETS_PER_BLOCK
     - Then do a transpose to [NPACKETS_PER_BLOCK/4, NCHAN_PER_PACKET/8, NBEAMS/2, 8chan, 2 times]
     - if doing subtract_ib:
     + ibsum has shape [NCHAN_PER_PACKET/8, 8chan, 2 times, NPACKETS_PER_BLOCK]
     +
  */
  // assume breakdown into tiles of 32x32, and run with 32x8 threads per block
  // launch with dim3 dimBlock(32, 8) and dim3 dimGrid(Width/32, Height/32)
  // here, width=NPACKETS_PER_BLOCK/4, height=NCHAN_PER_PACKET/8 * NBEAMS/2 * 8chan * 2times
  
  template <typename Arg> struct powSum {
    const Arg &arg;
    constexpr powSum(const Arg &arg) : arg(arg) {}
    static constexpr const char *filename() { return KERNEL_FILE; }
    using Float = typename Arg::real;
        
    __device__ __host__ void operator()(int idx)
    {
      int n_time = arg.n_time;
      int n_beam = Arg::n_beam;
      int n_chan = arg.n_chan;

      const int n_power_sum = Arg::n_power_sum;
      const int n_chan_sum = arg.n_chan_sum;
      const int n_time_inner = arg.n_time_inner;

      int low = 8192*100;
      int high = 8192*100 + 64;

      // Input [(A)NCHAN_PER_PACKET/8, (B)NBEAMS/2, (C)8chan, (D)2 times, (E)NPACKETS_PER_BLOCK]
      int A = idx / (n_beam * n_chan_sum * n_time_inner * n_time/n_power_sum);
      int B = (idx / (n_chan_sum * (n_time/n_power_sum) * n_time_inner)) % n_beam;
      int C = (idx / ((n_time/n_power_sum) * n_time_inner)) % n_chan_sum;
      int D = (idx / (n_time/n_power_sum)) % n_time_inner;
      int E = idx % (n_time/n_power_sum);

      if(low <= idx && idx < high && true) {
	printf("(ggp) PST thread %d in %d(%d,%d,%d,%d,%d) Input A_r = %f, Input A_i = %f, Input B_r = %f, Input B_i = %f\n", idx, idx,
	       A, B, C, D, E,
	       arg.input_A[2*idx], arg.input_A[2*idx+1],
	       arg.input_B[2*idx], arg.input_B[2*idx+1]);
      }
      
      // Deduce correct indices from linear index
      int chan_outer = idx / (n_beam * (n_time/n_power_sum) * n_chan_sum * n_time_inner);
      int beam_outer = idx % (n_beam * (n_time/n_power_sum) * n_chan_sum * n_time_inner);
      int beam_inner = beam_outer % ((n_time/n_power_sum) * n_chan_sum * n_time_inner);
      
      int sum_ib_idx = chan_outer * n_chan_sum * n_time_inner * (n_time/n_power_sum) + beam_inner;

      Float sum = 0.0;
      for (int k=0; k<Arg::n_power_sum; k++) {
	sum += (arg.input_A[2*(n_power_sum*idx + k)  ] * arg.input_A[2*(n_power_sum*idx + k)  ] +
		arg.input_A[2*(n_power_sum*idx + k)+1] * arg.input_A[2*(n_power_sum*idx + k)+1] +
		arg.input_B[2*(n_power_sum*idx + k)  ] * arg.input_B[2*(n_power_sum*idx + k)  ] +
		arg.input_B[2*(n_power_sum*idx + k)+1] * arg.input_B[2*(n_power_sum*idx + k)+1]);
	
	sum -= arg.ib_sum_data[n_power_sum*sum_ib_idx + k];
      }
	
      //if(0 < chan_outer && chan_outer < 2 && beam_outer < 2 && 0 < n_time && n_time < 2) {
      if(low <= idx && idx < high && false) {
	printf("(ggp) PST thread %d chan %d beam_outer %d beam_inner %d sum_ib_idx %d sum_ib %f %f %f %f sum_val %f\n",
	       idx, chan_outer, beam_outer, beam_inner, sum_ib_idx,
	       arg.ib_sum_data[n_power_sum*sum_ib_idx],
	       arg.ib_sum_data[n_power_sum*sum_ib_idx+1],
	       arg.ib_sum_data[n_power_sum*sum_ib_idx+2],
	       arg.ib_sum_data[n_power_sum*sum_ib_idx+3],
	       sum);
      }  
            
      // Deduce linear output index (use ouput ordering)
      // Output, transpose to [(A)NPACKETS_PER_BLOCK/4, (B)NCHAN_PER_PACKET/8, (C)NBEAMS/2, (D)8chan, (E)2 times]
      int E_in = E;
      int Eo = D;
      int Do = C * n_time_inner;
      int Co = B * n_chan_sum * n_time_inner;
      int Bo = A * (n_beam * n_chan_sum * n_time_inner);
      int Ao = E_in * n_beam * n_chan * n_time_inner;
      
      int lin_idx_out = Ao + Bo + Co + Do + Eo;
      
      if(low <= idx && idx < high && false) {
	printf("(ggp) PST thread %d in %d out %d(%d,%d,%d,%d,%d) sum_val %f\n",
	       idx, idx, lin_idx_out, E_in, A, B, C, D, sum);
      }        
      
      arg.ps_data[lin_idx_out] = sum;
      
    }    
  };
}
