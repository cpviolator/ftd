#pragma once

#include <comm_ggp.h>
#include <kernel.h>
#include <register_traits.h>

namespace quda {
  
  template <typename storage_t> struct WeightsComputeArg : kernel_param<> {
    
    // The storage type is double, single, short, or char.
    using Prec = storage_t;
    Prec *weights_A;
    Prec *weights_B;
    Prec *ant_E;
    Prec *ant_N;
    int *flagants;
    Prec *calibs;
    Prec *freqs;

    Prec dec;
    unsigned long long int N;

    static constexpr uint n_ant = 64;
    static constexpr uint n_beam = 1024;
    static constexpr uint n_chan = 256;
    static constexpr uint n_chan_sum = 8;
    static constexpr float sep = 1.0;
    static constexpr float sep_ns = 0.75;
    static constexpr float PI = 3.14159265358979323846;
    static constexpr float CVAC = 299792458.0;
    
    WeightsComputeArg(Prec *weights_A, Prec *weights_B, Prec *ant_E, Prec *ant_N, int *flagants, Prec *calibs, Prec *freqs, Prec dec, unsigned long long int N) :
      kernel_param(dim3(N, 1, 1)),
      weights_A(weights_A),
      weights_B(weights_B),
      ant_E(ant_E),
      ant_N(ant_N),
      flagants(flagants),
      calibs(calibs),
      freqs(freqs),
      dec(dec),
      N(N)
    {
    }
  };
  
  template <typename Arg> struct WeightsCompute {
    const Arg &arg;
    constexpr WeightsCompute(const Arg &arg) : arg(arg) {}
    static constexpr const char *filename() { return KERNEL_FILE; }

    // Get the storage type from the Arg, cast data to that type.
    typedef typename Arg::Prec Prec;

    const uint n_ant = Arg::n_ant;
    const uint n_beam = Arg::n_beam;
    const uint n_chan = Arg::n_chan;
    const uint n_chan_sum = Arg::n_chan_sum;
    const Prec sep = Arg::sep;
    const Prec sep_ns = Arg::sep_ns;
    const Prec PI = Arg::PI;
    const float CVAC = Arg::CVAC;
    
    __device__ __host__ inline void operator()(int idx)
    {
      int iPol = (int)(idx / ((n_chan/n_chan_sum)*(n_beam)*(n_ant)));
      int j = (int)(idx % ((n_chan/n_chan_sum)*(n_beam)*(n_ant)));
      int fq = (int)(j / ((n_beam)*(n_ant)));
      int iidx = (int)(j % ((n_beam)*(n_ant)));
      int bm = (int)(iidx / (n_ant));
      int a = (int)(iidx % (n_ant));
      int widx = (a)*(n_chan/n_chan_sum)*2*2 + fq*2*2;      

      // calculate weights
      Prec theta, afac, twr, twi;
      if (iPol==0) {
	theta = sep*(127 - bm)*PI/10800; // radians
	afac = -2.0*PI*arg.freqs[fq]*theta/CVAC; // factor for rotate
	
	twr = cosf(afac*arg.ant_E[a]);
	twi = sinf(afac*arg.ant_E[a]);
	
	arg.weights_A[2*idx]   = (twr*arg.calibs[widx] - twi*arg.calibs[widx+1]);
	arg.weights_A[2*idx+1] = (twi*arg.calibs[widx] + twr*arg.calibs[widx+1]);
	arg.weights_B[2*idx]   = (twr*arg.calibs[widx+2] - twi*arg.calibs[widx+3]);
	arg.weights_B[2*idx+1] = (twi*arg.calibs[widx+2] + twr*arg.calibs[widx+3]);
      } else {
	theta = sep_ns*(127 - bm)*PI/10800 - (PI/180.)*arg.dec; // radians
	afac = -2.*PI*arg.freqs[fq]*sinf(theta)/CVAC; // factor for rotate
	
	twr = cosf(afac*arg.ant_N[a]);
	twi = sinf(afac*arg.ant_N[a]);
	
	arg.weights_A[2*idx]   = (twr*arg.calibs[widx] - twi*arg.calibs[widx+1]);
	arg.weights_A[2*idx+1] = (twi*arg.calibs[widx] + twr*arg.calibs[widx+1]);
	arg.weights_B[2*idx]   = (twr*arg.calibs[widx+2] - twi*arg.calibs[widx+3]);
	arg.weights_B[2*idx+1] = (twi*arg.calibs[widx+2] + twr*arg.calibs[widx+3]);
      }
    }
  };
  
  template <typename storage_t> struct WeightsComputeArg110 : kernel_param<> {
    
    // The storage type is double, single, short, or char.
    using Prec = storage_t;
    Prec *weights_A;
    Prec *weights_B;
    Prec *ant_E;
    Prec *ant_N;
    int *flagants;
    Prec *calibs;
    Prec *freqs;

    Prec dec;
    unsigned long long int N;

    static constexpr uint n_ant = 96;
    static constexpr uint n_beam = 512;
    static constexpr uint n_chan = 384;
    static constexpr uint n_chan_sum = 8;
    static constexpr float sep = 1.0;
    static constexpr float sep_ns = 0.75;
    static constexpr float PI = 3.14159265358979323846;
    static constexpr float CVAC = 299792458.0;
    
    WeightsComputeArg110(Prec *weights_A, Prec *weights_B, Prec *ant_E, Prec *ant_N, int *flagants, Prec *calibs, Prec *freqs, Prec dec, unsigned long long int N) :
      kernel_param(dim3(N, 1, 1)),
      weights_A(weights_A),
      weights_B(weights_B),
      ant_E(ant_E),
      ant_N(ant_N),
      flagants(flagants),
      calibs(calibs),
      freqs(freqs),
      dec(dec),
      N(N)
    {
    }
  };
  
  template <typename Arg> struct WeightsCompute110 {
    const Arg &arg;
    constexpr WeightsCompute110(const Arg &arg) : arg(arg) {}
    static constexpr const char *filename() { return KERNEL_FILE; }

    // Get the storage type from the Arg, cast data to that type.
    typedef typename Arg::Prec Prec;

    const uint n_ant = Arg::n_ant;
    const uint n_beam = Arg::n_beam;
    const uint n_chan = Arg::n_chan;
    const uint n_chan_sum = Arg::n_chan_sum;
    const Prec sep = Arg::sep;
    const Prec sep_ns = Arg::sep_ns;
    const Prec PI = Arg::PI;
    const float CVAC = Arg::CVAC;
    
    __device__ __host__ inline void operator()(int idx)
    {
      int iArm = (int)(idx / ((n_chan/n_chan_sum)*(n_beam/2)*(n_ant/2)));
      int j = (int)(idx % ((n_chan/n_chan_sum)*(n_beam/2)*(n_ant/2)));
      int fq = (int)(j / ((n_beam/2)*(n_ant/2)));
      int iidx = (int)(j % ((n_beam/2)*(n_ant/2)));
      int bm = (int)(iidx / (n_ant/2));
      int a = (int)(iidx % (n_ant/2));
      int widx = (a + (n_ant/2)*iArm)*(n_chan/n_chan_sum)*2*2 + fq*2*2;      

      // calculate weights
      Prec theta, afac, twr, twi;
      if (iArm==0) {
	theta = sep*(127 - bm)*PI/10800; // radians
	afac = -2.0*PI*arg.freqs[fq]*theta/CVAC; // factor for rotate
	
	twr = cosf(afac*arg.ant_E[a + (n_ant/2)*iArm]);
	twi = sinf(afac*arg.ant_E[a + (n_ant/2)*iArm]);
	
	arg.weights_A[2*idx]   = (twr*arg.calibs[widx] - twi*arg.calibs[widx+1]);
	arg.weights_A[2*idx+1] = (twi*arg.calibs[widx] + twr*arg.calibs[widx+1]);
	arg.weights_B[2*idx]   = (twr*arg.calibs[widx+2] - twi*arg.calibs[widx+3]);
	arg.weights_B[2*idx+1] = (twi*arg.calibs[widx+2] + twr*arg.calibs[widx+3]);
      } else {
	theta = sep_ns*(127 - bm)*PI/10800 - (PI/180.)*arg.dec; // radians
	afac = -2.*PI*arg.freqs[fq]*sinf(theta)/CVAC; // factor for rotate
	
	twr = cosf(afac*arg.ant_N[a + (n_ant/2)*iArm]);
	twi = sinf(afac*arg.ant_N[a + (n_ant/2)*iArm]);
	
	arg.weights_A[2*idx]   = (twr*arg.calibs[widx] - twi*arg.calibs[widx+1]);
	arg.weights_A[2*idx+1] = (twi*arg.calibs[widx] + twr*arg.calibs[widx+1]);
	arg.weights_B[2*idx]   = (twr*arg.calibs[widx+2] - twi*arg.calibs[widx+3]);
	arg.weights_B[2*idx+1] = (twi*arg.calibs[widx+2] + twr*arg.calibs[widx+3]);
      }
#if 1
      int low = ((n_chan/n_chan_sum)*(n_beam/2)*(n_ant/2)) - 512;
      int high= ((n_chan/n_chan_sum)*(n_beam/2)*(n_ant/2)) + 512;
      
      if(low <= idx && idx < high && false) {
	printf("kernel %d of %llu (ggp): weights A = (%f,%f) weights B = (%f,%f)\n", idx, arg.N, arg.weights_A[2*idx], arg.weights_A[2*idx+1], arg.weights_B[2*idx], arg.weights_B[2*idx+1]);
      }
#endif
    }
  };
  
} // namespace GGP
