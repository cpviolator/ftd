#pragma once

#include <comm_ggp.h>
#include <kernel.h>
#include <register_traits.h>

//#define LOCAL_DEBUG

namespace quda {
          
  template <typename Float> struct  FBTArg: kernel_param<> {
    using real = typename mapper<Float>::type;
    Float *output;
    const Float *input;
    
    FBTArg(Float *output, const Float *input) :
      kernel_param(dim3(32, 1, 1)),
      output(output),
      input(input)
    {
    }
  };
  
  template <typename Arg> struct FilterbankTranspose {
    const Arg &arg;
    constexpr FilterbankTranspose(const Arg &arg) : arg(arg) {}
    static constexpr const char *filename() { return KERNEL_FILE; }
    
    __device__ __host__ inline void operator()(int chan, int beam)
    {
    }
  };  
} // namsespace quda

#undef LOCAL_DEBUG
