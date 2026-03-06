#pragma once

#include <comm_ggp.h>
#include <kernel.h>
#include <register_traits.h>

//#define LOCAL_DEBUG

namespace quda {
          
  template <typename Float> struct  HermToTriArg: kernel_param<> {
    using real = typename mapper<Float>::type;
    complex<Float> *output;
    const complex<Float> *input;
    unsigned long long int N;
    unsigned long long int N_batch;
    
    HermToTriArg(complex<Float> *output, const complex<Float> *input, const unsigned long long int N, const unsigned long long int N_batch) :
      kernel_param(dim3(N, 1, 1)),
      output(output),
      input(input),
      N(N),
      N_batch(N_batch)
    {
    }
  };
  
  template <typename Arg> struct HermToTri {
    const Arg &arg;
    constexpr HermToTri(const Arg &arg) : arg(arg) {}
    static constexpr const char *filename() { return KERNEL_FILE; }
    
    __device__ __host__ inline void operator()(int idx)
    {
      // Compute matrix index
      // DMH: Use block BLAS infrastructure here to accelerate,
      //      and optimise arithmetic. (TunableKernel2D)
      //      N_batch is the number of complex elements in one batch
      int mat_rank = sqrt(arg.N_batch);
      int mat_idx = idx % arg.N_batch;
      int batch_idx = idx / arg.N_batch;
      
      // Column index
      int j = mat_idx % mat_rank;
      // Row index
      int i = mat_idx / mat_rank;
      // Elements in one triangular matrix
      int tri_batch_size = arg.N_batch - ((mat_rank - 1)*(mat_rank))/2;

#ifdef LOCAL_DEBUG
      if(idx < 2*mat_rank) {
	printf("Full: %d %d (%d,%d): (%f,%f)\n", idx, mat_idx, i, j, (Arg::real)(arg.input[idx].real()), (Arg::real)(arg.input[idx].imag()));
      }
#endif
      
      if(j>=i && idx < arg.N) {
	int tri_idx = mat_idx - (i*(i+1))/2;
#ifdef LOCAL_DEBUG
	if(idx < 2*mat_rank) {
	  printf("Tri: %d %d %d, (%f,%f)\n", idx, mat_idx, tri_idx, (Arg::real)(arg.input[idx].real()), (Arg::real)(arg.input[idx].imag()));
	}
#endif
	// Construct tri diagonal matrix using real/imag
	//arg.output[batch_idx * tri_batch_size + tri_idx].real(arg.input[idx].real());
	//arg.output[batch_idx * tri_batch_size + tri_idx].imag(arg.input[idx].imag());
	
	// using complex data type
	arg.output[batch_idx * tri_batch_size + tri_idx] = arg.input[idx];
	//arg.output[batch_idx * tri_batch_size + tri_idx] = arg.input[(2*2*iidx + pol + 0)];
	//arg.output[batch_idx * tri_batch_size + tri_idx] += arg.input[(2*2*iidx + pol + 2)];
      }
    }    
  };  
} // namsespace quda

#undef LOCAL_DEBUG
