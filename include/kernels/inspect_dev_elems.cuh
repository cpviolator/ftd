#pragma once

#include <comm_ggp.h>
#include <kernel.h>
#include <register_traits.h>
#include <ggp_fp16.cuh>

namespace quda {
  
  template <typename Float> struct ElemInspArg : kernel_param<> {
    using real = typename mapper<Float>::type;

    Float *array;
    unsigned long long int N;
    unsigned long long int N_low;
    unsigned long long int N_high;
    int ID;
    
    ElemInspArg(Float *array, const unsigned long long int N, const unsigned long long int N_low, const unsigned long long int N_high, int ID) :
      kernel_param(dim3(N, 1, 1)),
      array(array),
      N_low(N_low),
      N_high(N_high),
      ID(ID)
    {
    }
  };
  
  template <typename Arg> struct ElemInspect {
    const Arg &arg;
    constexpr ElemInspect(const Arg &arg) : arg(arg) {}
    static constexpr const char *filename() { return KERNEL_FILE; }
    
    __device__ __host__ inline void operator()(int idx)
    {
      if(arg.N_low <= idx && idx < arg.N_high) printf("Elem Inspection(%d): %d %f\n", arg.ID, idx, (Arg::real)(arg.array[idx]));
    }
  };

  template <typename Arg> struct ElemHalfInspect {
    const Arg &arg;
    constexpr ElemHalfInspect(const Arg &arg) : arg(arg) {}
    static constexpr const char *filename() { return KERNEL_FILE; }
    
    __device__ __host__ inline void operator()(int idx)
    {
      if(arg.N_low <= idx && idx < arg.N_high) printf("Elem Inspection(%d): %d %f\n", arg.ID, idx, __half2float(arg.array[idx]));
    }
  };
  
}
