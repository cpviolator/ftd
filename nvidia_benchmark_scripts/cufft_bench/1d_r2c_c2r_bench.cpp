/*
 * Copyright 2020 NVIDIA Corporation.  All rights reserved.
 *
 * NOTICE TO LICENSEE:
 *
 * This source code and/or documentation ("Licensed Deliverables") are
 * subject to NVIDIA intellectual property rights under U.S. and
 * international Copyright laws.
 *
 * These Licensed Deliverables contained herein is PROPRIETARY and
 * CONFIDENTIAL to NVIDIA and is being provided under the terms and
 * conditions of a form of NVIDIA software license agreement by and
 * between NVIDIA and Licensee ("License Agreement") or electronically
 * accepted by Licensee.  Notwithstanding any terms or conditions to
 * the contrary in the License Agreement, reproduction or disclosure
 * of the Licensed Deliverables to any third party without the express
 * written consent of NVIDIA is prohibited.
 *
 * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
 * LICENSE AGREEMENT, NVIDIA MAKES NO REPRESENTATION ABOUT THE
 * SUITABILITY OF THESE LICENSED DELIVERABLES FOR ANY PURPOSE.  IT IS
 * PROVIDED "AS IS" WITHOUT EXPRESS OR IMPLIED WARRANTY OF ANY KIND.
 * NVIDIA DISCLAIMS ALL WARRANTIES WITH REGARD TO THESE LICENSED
 * DELIVERABLES, INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY,
 * NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
 * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
 * LICENSE AGREEMENT, IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY
 * SPECIAL, INDIRECT, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, OR ANY
 * DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS,
 * WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS
 * ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE
 * OF THESE LICENSED DELIVERABLES.
 *
 * U.S. Government End Users.  These Licensed Deliverables are a
 * "commercial item" as that term is defined at 48 C.F.R. 2.101 (OCT
 * 1995), consisting of "commercial computer software" and "commercial
 * computer software documentation" as such terms are used in 48
 * C.F.R. 12.212 (SEPT 1995) and is provided to the U.S. Government
 * only as a commercial end item.  Consistent with 48 C.F.R.12.212 and
 * 48 C.F.R. 227.7202-1 through 227.7202-4 (JUNE 1995), all
 * U.S. Government End Users acquire the Licensed Deliverables with
 * only those rights set forth herein.
 *
 * Any use of the Licensed Deliverables in individual and commercial
 * software must include, in the user documentation and internal
 * comments to the code, the above Disclaimer and U.S. Government End
 * Users Notice.
 */

#include <complex>
#include <iostream>
#include <vector>
#include <chrono>
#include <omp.h>
#include <cufft.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cufftXt.h>
#include "cufft_utils.h"

#define CHECK_CUDA_ERROR(call) \
do { \
    cudaError_t result = call; \
    if (result != cudaSuccess) { \
        fprintf(stderr, "%s:%d: CUDA error %d: %s\n", __FILE__, __LINE__, result, cudaGetErrorString(result)); \
        exit(1); \
    } \
} while (0)

__global__ void scaling_kernel_local(cufftComplex* data, int element_count, float scale) {
  
  const int tid = threadIdx.x + blockIdx.x * blockDim.x;
  const int stride = blockDim.x * gridDim.x;
  
  //printf("tid %d\n", tid);
  //printf("stride %d\n", stride);
  for (int i = tid; i<element_count; i+= stride) {    
    data[i].x *= scale;
    data[i].y *= scale;
    //printf("Scaled at %d\n", i);
  }
}

int main(int argc, char *argv[]) {
    cufftHandle planr2c, planc2r;
    //cudaStream_t stream = NULL;

    if (argc < 6) {
      std::cout<<"Usage: ./1d_r2c_c2r_example <N> <batch_size> <use_Xt> <verbose> <verify>\n"
	       <<"Arguments:\n"
	       <<"\t N: The size of the array in 2^N\n"
	       <<"\t batch_size: The number of batches\n"
	       <<"\t 0/1: Use cuFFT/cuFFTXt\n"
	       <<"\t 0/1: verbose output\n"
	       <<"\t 0/1: verify with inverse transform"<<std::endl;
      return EXIT_FAILURE;
    }
    
    int fft_size = std::pow(2, atoi(argv[1]));
    int p[] = {fft_size};
    long long int pll = fft_size;
    int batch_size = atoi(argv[2]);
    bool use_Xt = atoi(argv[3]) == 0 ? false : true;
    bool verbose = atoi(argv[4]) == 0 ? false : true;
    bool verify = atoi(argv[5]) == 0 ? false : true;

    int reps = 1;
    int DT_reps = 1;
    
    float n_fft = reps * batch_size;
    int element_count = n_fft * fft_size;
    int64_t fft_flop = 2.5 * fft_size * (atoi(argv[1]));
    
    std::printf("fft_flop = %lu\n", fft_flop);
    std::printf("element_count = %d\n", element_count);
    
    size_t workSize;
    
    using Time = std::chrono::system_clock;
    using ns = std::chrono::nanoseconds;
    using float_sec = std::chrono::duration<float>;
    using float_time_point = std::chrono::time_point<Time, float_sec>;
    
    using scalar_type = float;
    using input_type = scalar_type;
    using output_type = std::complex<scalar_type>;

    std::vector<input_type> input(element_count, 0);
    std::vector<input_type> input_orig(element_count, 0);
    std::vector<output_type> output((fft_size / 2 + 1) * batch_size * reps);
    
    for (auto i = 0; i < element_count; i++) {
      //input[i] = static_cast<input_type>((1.0 * rand())/RAND_MAX);
      input[i] = static_cast<input_type>(i);
      input_orig[i] = input[i];
    }

    if(verbose) {
      std::printf("Input array:\n");
      for(int b=0; b<batch_size * reps; b++) {
	for(int i = 0; i < fft_size; i++) {
	  std::printf("%d,%d: %f\n", b, i, input[i + b*fft_size]);
	}
      }
      std::printf("=====\n");
    }
    
    input_type *d_input = nullptr;
    cufftComplex *d_output = nullptr;

    //CUFFT_CALL(cufftPlan1d(&planr2c, fft_size, CUFFT_R2C, batch_size));
    //CUFFT_CALL(cufftPlan1d(&planc2r, fft_size, CUFFT_C2R, batch_size));
    
    if(use_Xt == 1) {
      cufftCreate(&planr2c);
      CUFFT_CALL(cufftXtMakePlanMany(planr2c, 1, &pll,
				     NULL, 1, 1, CUDA_R_16F, 
				     NULL, 1, 1, CUDA_C_16F, 
				     batch_size, &workSize, CUDA_C_16F));
      printf("Temporary buffer size %li bytes\n", workSize);

      cufftCreate(&planc2r);
      CUFFT_CALL(cufftXtMakePlanMany(planc2r, 1, &pll,
				     nullptr, 1, 0, CUDA_C_16F, 
				     nullptr, 1, 0, CUDA_R_16F, 
				     batch_size, &workSize, CUDA_C_16F));
      printf("Temporary buffer size %li bytes\n", workSize);
    } else {
      CUFFT_CALL(cufftPlanMany(&planr2c, 1, p,
			       nullptr, 1, 0,
			       nullptr, 1, 0,
			       CUFFT_R2C, batch_size));
      
      CUFFT_CALL(cufftPlanMany(&planc2r, 1, p,
			       nullptr, 1, 0,
			       nullptr, 1, 0,
			       CUFFT_C2R, batch_size));    
    }

    //CUDA_RT_CALL(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    //CUFFT_CALL(cufftSetStream(planr2c, stream));
    //CUFFT_CALL(cufftSetStream(planc2r, stream));

    // Create device arrays
    CUDA_RT_CALL(cudaMalloc(reinterpret_cast<void **>(&d_input), sizeof(input_type) * input.size()));
    CUDA_RT_CALL(cudaMalloc(reinterpret_cast<void **>(&d_output), sizeof(output_type) * output.size()));

    // Copy input
    auto startTime = Time::now();
    for(int k=0; k<DT_reps; k++) {
      CUDA_RT_CALL(cudaMemcpy(d_input, input.data(), sizeof(input_type) * input.size(), cudaMemcpyHostToDevice));
    }
    auto endTime = Time::now();
    auto duration = std::chrono::duration_cast<ns>(endTime - startTime);
    double time = (1.0*duration.count())/(1e3 * DT_reps);
    std::printf("Memcpy H2D time average (us) = %.12f\n", time);
    
    // out-of-place Forward transform
    // warm up.
    //CUFFT_CALL(cufftExecR2C(planr2c, d_input, d_output));    
    cudaEvent_t start, stop;
    float ctime = 0.0;
    for(int k=0; k<reps; k++) {

      CHECK_CUDA_ERROR(cudaEventCreate(&start));
      CHECK_CUDA_ERROR(cudaEventCreate(&stop));
      CHECK_CUDA_ERROR(cudaEventRecord(start, 0));
      
      //CUFFT_CALL(cufftExecR2C(planr2c, d_input + k*fft_size*batch_size, d_output + k*(fft_size / 2 + 1) * batch_size));
      if(use_Xt) {
	CUFFT_CALL(cufftXtExec(planr2c, (void*)d_input, (void*)d_output, CUFFT_FORWARD));
	//CUFFT_CALL(cufftXtExec(planr2c, (void*)d_input, (void*)d_input, CUFFT_FORWARD));
      }
      else {
	CUFFT_CALL(cufftExecR2C(planr2c, reinterpret_cast<cufftReal *>(d_input), reinterpret_cast<cufftComplex *>(d_output)));
	//CUFFT_CALL(cufftExecR2C(planr2c, reinterpret_cast<cufftReal *>(d_input), reinterpret_cast<cufftComplex *>(d_input)));
      }
      
      CHECK_CUDA_ERROR(cudaEventRecord(stop, 0));
      CHECK_CUDA_ERROR(cudaEventSynchronize(stop));
      CHECK_CUDA_ERROR(cudaEventElapsedTime(&ctime, start, stop));
      
    }
    // Scale from milliseconds
    std::printf("Execute time = %.6f ms, %.6e s\n", ctime, (ctime/(1e3)));
    std::printf("Execute GFLOP : batch_size * reps * 2.5 * NX * log_2(NX) = %.12f\n", fft_flop * batch_size * reps / (1e9));
    std::printf("Execute TFLOPS : batch_size * reps * 2.5 * NX * log_2(NX) / time = %.12f\n", (1e-12 * fft_flop * batch_size * reps)/(ctime/(1e3)));

    if(verify) {
    
      // Copy output
      startTime = Time::now();
      for(int k=0; k<DT_reps; k++) {
	CUDA_RT_CALL(cudaMemcpy(output.data(), d_output, sizeof(output_type) * output.size(), cudaMemcpyDeviceToHost));
      }
      endTime = Time::now();
      duration = std::chrono::duration_cast<ns>(endTime - startTime);
      time = duration.count();
      std::printf("Memcpy D2H time average (us) = %.12f\n", (1.0*time)/(1e3 * DT_reps));
    
      //CUDA_RT_CALL(cudaStreamSynchronize(stream));
    
      if(verbose) {
	std::printf("Output array after Forward FFT:\n");
	for(int b=0; b<batch_size; b++) {
	  for(int i = 0; i < fft_size; i++) {
	    std::printf("%d,%d: %f + %fj\n", b, i, output[i + b*fft_size].real(), output[i + b*fft_size].imag());
	  }
	}
	std::printf("=====\n");
      }
        
      // out-of-place Inverse transform
      for(int k=0; k<reps; k++) {

	// Normalize the data
	scaling_kernel_local<<<1, 128>>>(d_output, batch_size * (fft_size / 2 + 1), 1.0/(fft_size));

	// do FFT
	CUFFT_CALL(cufftExecC2R(planc2r, d_output + k*(fft_size / 2 + 1) * batch_size, d_input + k*fft_size * batch_size));
	//CUFFT_CALL(cufftExecC2R(planc2r, reinterpret_cast<cufftComplex *>(d_output), reinterpret_cast<cufftReal *>(d_input)));
      }
      // Copy to input array
      CUDA_RT_CALL(cudaMemcpy(input.data(), d_input, sizeof(input_type) * input.size(), cudaMemcpyDeviceToHost));
      
      if(verbose) {
	std::printf("Input array after Forward FFT, Normalization, and Inverse FFT:\n");
	for(int b=0; b<batch_size; b++) {
	  for(int i = 0; i < fft_size; i++) {
	    std::printf("%d,%d: %f\n", b, i, input[i + b*fft_size]);
	  }
	}
	std::printf("=====\n");
      }
    
      // Check FFT
      float error = 0.0;
      //#pragma omp parallel for collapse(2) reduction(+:error) 
      for(int b=0; b<batch_size * reps; b++) {
	for(int i = 0; i < fft_size; i++) {
	  error += (input[i + b*fft_size] - input_orig[i + b*fft_size]) * (input[i + b*fft_size] - input_orig[i + b*fft_size]);
	}
      }
      error = sqrt(error/element_count);    
      std::printf("RMS error = %f\n", error);
    }
    
    /* free resources */
    CUDA_RT_CALL(cudaFree(d_input));
    CUDA_RT_CALL(cudaFree(d_output));

    CUFFT_CALL(cufftDestroy(planr2c));
    CUFFT_CALL(cufftDestroy(planc2r));

    //CUDA_RT_CALL(cudaStreamDestroy(stream));

    CUDA_RT_CALL(cudaDeviceReset());

    return EXIT_SUCCESS;
}
