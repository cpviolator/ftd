#pragma once

#include <comm_ggp.h>
#include <kernel.h>
#include <register_traits.h>
#include <ggp_fp16.cuh>

//#define LOCAL_DEBUG

namespace quda {
  
  template <typename Float> struct InterTriArg : kernel_param<> {
    using real = typename mapper<Float>::type;
    const half *input_real;
    const half *input_imag;
    Float *output_full;
    
    InterTriArg(Float *output_full, const half *input_real, const half *input_imag, unsigned long long int N) :
      kernel_param(dim3(N, 1, 1)),
      input_real(input_real),
      input_imag(input_imag),
      output_full(output_full)
    {
    }
  };
  
  template <typename Arg> struct PromTriInterFloat {
    const Arg &arg;
    constexpr PromTriInterFloat(const Arg &arg) : arg(arg) {}
    static constexpr const char *filename() { return KERNEL_FILE; }
    
    __device__ __host__ inline void operator()(int idx)
    {
      arg.output_full[2*idx  ] = __half2float(arg.input_real[idx]);
      arg.output_full[2*idx+1] = __half2float(arg.input_imag[idx]);
#ifdef LOCAL_DEBUG
      if(idx < 8) {
	printf("PromTriInter_Float: %d: (%f,%f) from %f, %f\n", idx, (Arg::real)(arg.output_full[2*idx]), (Arg::real)(arg.output_full[2*idx+1]), __half2float(arg.input_real[idx]), __half2float(arg.input_imag[idx]));
      }
#endif
    }
  };

  template <typename Float> struct InterArg : kernel_param<> {
    using real = typename mapper<Float>::type;
    const half *input_real;
    const half *input_imag;
    Float *output_full;
    
    InterArg(Float *output_full, const half *input_real, const half *input_imag, unsigned long long int N) :
      kernel_param(dim3(N, 1, 1)),
      input_real(input_real),
      input_imag(input_imag),
      output_full(output_full)
    {
    }
  };
  
  template <typename Arg> struct PromInterFloat {
    const Arg &arg;
    constexpr PromInterFloat(const Arg &arg) : arg(arg) {}
    static constexpr const char *filename() { return KERNEL_FILE; }
    
    __device__ __host__ inline void operator()(int idx)
    {
      arg.output_full[2*idx  ] = __half2float(arg.input_real[idx]);
      arg.output_full[2*idx+1] = __half2float(arg.input_imag[idx]);
      
#ifdef LOCAL_DEBUG
      if(idx < 8) {
	printf("PromInter_Float: %d: (%f,%f) from %f, %f\n", idx, (Arg::real)(arg.output_full[2*idx]), (Arg::real)(arg.output_full[2*idx+1]), __half2float(arg.input_real[idx]), __half2float(arg.input_imag[idx]));
      }
#endif
    }
  };

  
}

