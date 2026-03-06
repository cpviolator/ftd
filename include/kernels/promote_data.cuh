#pragma once

#include <comm_ggp.h>
#include <kernel.h>
#include <register_traits.h>

//#define LOCAL_DEBUG

namespace quda {

#define MASK1 0x0F // (00001111)
#define MASK2 0xF0 // (11110000)
  
  template <typename storage_t> struct PromotePlanarArg : kernel_param<> {

    // The storage type is double, single, short, or char.
    using Prec = storage_t;
    Prec *output_real;
    Prec *output_imag;    
    const uint8_t *input_data;
    unsigned long long int N;
    
    PromotePlanarArg(Prec *output_real, Prec *output_imag, const uint8_t *input_data, unsigned long long int N) :
      kernel_param(dim3(N, 1, 1)),
      output_real(output_real),
      output_imag(output_imag),
      input_data(input_data),
      N(N)
    {
    }
  };
  
  template <typename Arg> struct DataPlanarPromote {
    const Arg &arg;
    constexpr DataPlanarPromote(const Arg &arg) : arg(arg) {}
    static constexpr const char *filename() { return KERNEL_FILE; }
    
    // Get the storage type from the Arg, cast data to that type.
    typedef typename Arg::Prec Prec;
    
    __device__ __host__ inline void operator()(int idx)
    {
      arg.output_real[idx] = (Prec)((char)(((unsigned char)(arg.input_data[idx]) & (unsigned char)(15)) << 4) >> 4);
      arg.output_imag[idx] = (Prec)((char)(((unsigned char)(arg.input_data[idx]) & (unsigned char)(240))) >> 4);
#ifdef LOCAL_DEBUG
      if(idx < 8) {
	printf("PlanarDataPromote: %d: %f, %f\n", idx, (Prec)(arg.output_real[idx]), (Prec)(arg.output_imag[idx]));
      }
#endif
    }
  };
        
  template <typename storage_t> struct PromoteArg : kernel_param<> {

    // The storage type is double, single, short, or char.
    using Prec = storage_t;
    Prec *output;
    const uint8_t *input;
    unsigned long long int N;
    
    PromoteArg(Prec *output, const uint8_t *input, unsigned long long int N) :
      kernel_param(dim3(N, 1, 1)),
      output(output),
      input(input),
      N(N)
    {
    }
  };
  
  template <typename Arg> struct DataPromote {
    const Arg &arg;
    constexpr DataPromote(const Arg &arg) : arg(arg) {}    
    static constexpr const char *filename() { return KERNEL_FILE; }

    // Get the storage type from the Arg, cast data to that type.
    typedef typename Arg::Prec Prec;
    
    __device__ __host__ inline void operator()(int idx)
    {
      arg.output[2*idx]   = (Prec)((char)(((unsigned char)(arg.input[idx]) & (unsigned char)(15)) << 4) >> 4);
      arg.output[2*idx+1] = (Prec)((char)(((unsigned char)(arg.input[idx]) & (unsigned char)(240))) >> 4);
#ifdef LOCAL_DEBUG
      if(idx < 8) {
	printf("DataPromote: %d: %f, %f\n", idx, (Prec)(arg.output[2*idx]), (Prec)(arg.output[2*idx+1]));
      }
#endif
    }    
  };

  template <typename storage_t> struct PromotePolArg : kernel_param<> {
    
    // The storage type is double, single, short, or char.
    using Prec = storage_t;
    Prec *output_A;
    Prec *output_B;
    const uint8_t *input;
    unsigned long long int N;
    
    PromotePolArg(Prec *output_A, Prec *output_B, const uint8_t *input, unsigned long long int N) :
      kernel_param(dim3(N, 1, 1)),
      output_A(output_A),
      output_B(output_B),
      input(input),
      N(N)
    {
    }
  };
  
  template <typename Arg> struct DataPromotePol {
    const Arg &arg;
    constexpr DataPromotePol(const Arg &arg) : arg(arg) {}
    static constexpr const char *filename() { return KERNEL_FILE; }

    // Get the storage type from the Arg, cast data to that type.
    typedef typename Arg::Prec Prec;

    __device__ __host__ inline void operator()(int idx)
    {
      arg.output_A[2*idx]   = 0.05*(Prec)((char)(((unsigned char)(arg.input[2*idx]) & (unsigned char)(15)) << 4) >> 4);
      arg.output_A[2*idx+1] = 0.05*(Prec)((char)(((unsigned char)(arg.input[2*idx]) & (unsigned char)(240))) >> 4);
      
      arg.output_B[2*idx]   = 0.05*(Prec)((char)(((unsigned char)(arg.input[2*idx+1]) & (unsigned char)(15)) << 4) >> 4);
      arg.output_B[2*idx+1] = 0.05*(Prec)((char)(((unsigned char)(arg.input[2*idx+1]) & (unsigned char)(240))) >> 4);
#ifdef LOCAL_DEBUG
      if(idx < 64) {
	printf("DataPromote: %d: Pol A %f, %f\n", idx, (Prec)(arg.output_A[2*idx]), (Prec)(arg.output_A[2*idx+1]));
	printf("DataPromote: %d: Pol B %f, %f\n", idx, (Prec)(arg.output_B[2*idx]), (Prec)(arg.output_B[2*idx+1]));
      }
#endif
    }    
  };
  
} // namespace GGP

#undef LOCAL_DEBUG
