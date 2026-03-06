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

#include <array>
#include <complex>
#include <iostream>
#include <vector>
#include <chrono>
#include <cufft.h>
#include <cuda_runtime.h>
#include <cufftXt.h>
#include "cufft_utils.h"

using dim_t = std::array<int, 2>;

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
    cufftHandle planc2r, planr2c;
    //cudaStream_t stream = NULL;

    if(argc < 7) {
      std::cout<<"Usage: ./2d_c2r_r2c_example <N> <M> <batch_size> <use_Xt> <verbose> <verify>\n"
	       <<"Arguments:\n"
	       <<"\t N: The size of the NX array in 2^N\n"
	       <<"\t M: The size of the NY array in 2^M\n"
	       <<"\t batch_size: The number of batches\n"
	       <<"\t 0/1: Use cuFFT/cuFFTXt\n"
	       <<"\t 0/1: verbose output\n"
	       <<"\t 0/1: verify with inverse transform"<<std::endl;
      return EXIT_FAILURE;
    }
    
    int nx = std::pow(2,atoi(argv[1]));
    int ny = std::pow(2,atoi(argv[2]));
    dim_t fft_size = {nx, ny};
    int batch_size = atoi(argv[3]);
    bool use_Xt = atoi(argv[4]) == 0 ? false : true;
    bool verbose = atoi(argv[5]) == 0 ? false : true;
    bool verify = atoi(argv[6]) == 0 ? false : true;

    // Use microsecond timing
    //float time_norm = 1e3 * batch_size;
    int64_t fft_flop = 5 * nx*ny * std::log2(nx*ny);
    std::printf("fft_flop = %lu\n", fft_flop);
    
    long long int pll[2] = {nx, ny};
    size_t workSize;
    
    using Time = std::chrono::steady_clock;
    using ns = std::chrono::nanoseconds;
    using float_sec = std::chrono::duration<float>;
    using float_time_point = std::chrono::time_point<Time, float_sec>;
    
    using scalar_type = float;
    using input_type = std::complex<scalar_type>;
    using output_type = scalar_type;

    int input_size = nx * (ny/2 + 1);
    std::vector<input_type> input_complex(batch_size * input_size);
    std::vector<input_type> input_complex_copy(batch_size * input_size);
    std::vector<output_type> output_real(batch_size * nx * ny, 0);

    for(int b=0; b < batch_size; b++) {
      for (int i = 0; i < input_size; i++) {
	//input_complex[i + b*input_size] = input_type(i/(1.0*input_complex.size()), (2*i)/(1.0*input_complex.size())); // fairly constant, decent RMS
	//input_complex[i + b*input_size] = input_type((1.0*rand())/RAND_MAX, (1.0*rand())/RAND_MAX);
	input_complex[i + b*input_size] = input_type(1, 0);
	//if(i == input_size - 1) input_complex[i + b*input_size] = input_type(1, 0);
	input_complex_copy[i + b*input_size] = input_complex[i + b*input_size];      
      }
    }
    
    if(verbose) {
      std::printf("Input array:\n");
      for(int b=0; b < batch_size; b++) {
	for(int i = 0; i < input_size; i++) {
	  std::printf("%d,%d: %.12f + %.12fj\n", b, i, input_complex[i + b*(input_size)].real(), input_complex[i + b*(input_size)].imag());
	}
      }
      std::printf("=====\n");
    }
    
    cufftComplex *d_data = nullptr;
    cufftComplex *d_data_aux = nullptr;
    
    // inembed/onembed being nullptr indicates contiguous data for each batch, then the stride and dist settings are ignored
    if(use_Xt == 1) {
      cufftCreate(&planr2c);
      CUFFT_CALL(cufftXtMakePlanMany(planr2c, 2, pll,
				     NULL, 1, 0, CUDA_R_16F, 
				     NULL, 1, 0, CUDA_C_16F, 
				     batch_size, &workSize, CUDA_C_16F));
      printf("Temporary buffer size %li bytes\n", workSize);
      
      cufftCreate(&planc2r);
      CUFFT_CALL(cufftXtMakePlanMany(planc2r, 2, pll,
				     nullptr, 1, 0, CUDA_C_16F, 
				     nullptr, 1, 0, CUDA_R_16F, 
				     batch_size, &workSize, CUDA_C_16F));
      printf("Temporary buffer size %li bytes\n", workSize);
    } else {
      cufftCreate(&planc2r);
      CUFFT_CALL(cufftPlanMany(&planc2r, fft_size.size(), fft_size.data(),
			       nullptr, 1, 1, // *inembed, istride, idist
			       nullptr, 1, 1, // *onembed, ostride, odist
			       CUFFT_C2R, batch_size));
      cufftCreate(&planr2c);
      CUFFT_CALL(cufftPlanMany(&planr2c, fft_size.size(), fft_size.data(),
			       nullptr, 1, 1, // *inembed, istride, idist
			       nullptr, 1, 1, // *onembed, ostride, odist
			       CUFFT_R2C, batch_size));
    }
    
    //CUDA_RT_CALL(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    //CUFFT_CALL(cufftSetStream(planc2r, stream));
    //CUFFT_CALL(cufftSetStream(planr2c, stream));

    // Create device arrays
    // For in-place r2c/c2r transforms, make sure the device array is always allocated to the size of complex array
    CUDA_RT_CALL(cudaMalloc(reinterpret_cast<void **>(&d_data), sizeof(input_type) * input_complex.size()));
    CUDA_RT_CALL(cudaMalloc(reinterpret_cast<void **>(&d_data_aux), sizeof(input_type) * input_complex.size()));
    CUDA_RT_CALL(cudaMemcpy(d_data, (input_complex.data()), sizeof(input_type) * input_complex.size(),
			    cudaMemcpyHostToDevice));
    CUDA_RT_CALL(cudaMemcpy(d_data_aux, (input_complex.data()), sizeof(input_type) * input_complex.size(),
			    cudaMemcpyHostToDevice));
    
    // C2R
    cudaEvent_t start, stop;
    float time = 0.0;

    // Warm up
    //if(use_Xt) CUFFT_CALL(cufftXtExec(planc2r, (void*)d_data, (void*)d_data, CUFFT_INVERSE));
    
    CHECK_CUDA_ERROR(cudaEventCreate(&start));
    CHECK_CUDA_ERROR(cudaEventCreate(&stop));
    CHECK_CUDA_ERROR(cudaEventRecord(start, 0));
    
    if(use_Xt) {
      //ip
      CUFFT_CALL(cufftXtExec(planc2r, (void*)d_data, (void*)d_data, CUFFT_INVERSE));
      //ooplace
      //CUFFT_CALL(cufftXtExec(planc2r, (void*)d_data, (void*)d_data_aux, CUFFT_INVERSE));
    }
    else {
      //ip
      CUFFT_CALL(cufftExecC2R(planc2r, reinterpret_cast<cufftComplex*>(d_data), reinterpret_cast<scalar_type*>(d_data)));
      //ooplace
      //CUFFT_CALL(cufftExecC2R(planc2r, reinterpret_cast<cufftComplex*>(d_data), reinterpret_cast<scalar_type*>(d_data_aux)));
    }
    
    CHECK_CUDA_ERROR(cudaEventRecord(stop, 0));
    CHECK_CUDA_ERROR(cudaEventSynchronize(stop));
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&time, start, stop));
    
    std::printf("Execute time = %.6f ns, %.6e s\n", time, (time/(1e3)));
    std::printf("Execute GFLOP : batch_size * 5 * NX * NY * log_2(NX*NY) = %.12f\n", fft_flop * batch_size / (1e9));
    std::printf("Execute TFLOPS : batch_size * 5 * NX * NY * log_2(NX*NY) / time = %.12f\n", (1e-12 * fft_flop * batch_size)/(time/(1e3)));

    if(verify) {

      //ip
      CUDA_RT_CALL(cudaMemcpy(output_real.data(), reinterpret_cast<scalar_type*>(d_data), sizeof(output_type) * output_real.size(), cudaMemcpyDeviceToHost));
      // oop
      //CUDA_RT_CALL(cudaMemcpy(output_real.data(), reinterpret_cast<scalar_type*>(d_data_aux), sizeof(output_type) * output_real.size(), cudaMemcpyDeviceToHost));
      
      //CUDA_RT_CALL(cudaStreamSynchronize(stream));
      
      if(verbose) {
	std::printf("Output array after C2R:\n");
	for (int b = 0; b < batch_size; b++) {
	  for (int i = 0; i < nx * ny; i++) {
	    std::printf("%d,%d: %f\n", b, i, output_real[i + b * nx * ny]);
	  }
	  std::printf("=====\n");
	}
      }
      
      // R2C
      // Normalize the data
      //ip
      scaling_kernel<<<1, 128>>>(d_data, input_complex.size(), 1.f/(nx * ny));
      //oop
      //scaling_kernel<<<1, 128>>>(d_data_aux, input_complex.size(), 1.f/(nx * ny));
      
      // Execute FFT
      if(use_Xt) {
	//ip
	CUFFT_CALL(cufftXtExec(planr2c, d_data, d_data, CUFFT_FORWARD));
	//oop
	//CUFFT_CALL(cufftXtExec(planr2c, d_data_aux, d_data, CUFFT_FORWARD));
      }
      else {
	//ip
	CUFFT_CALL(cufftExecR2C(planr2c, reinterpret_cast<scalar_type*>(d_data), d_data));
	//oop
	//CUFFT_CALL(cufftExecR2C(planr2c, reinterpret_cast<scalar_type*>(d_data_aux), d_data));
      }

      //CUFFT_CALL(cufftExecR2C(planr2c, reinterpret_cast<scalar_type*>(d_data), reinterpret_cast<cufftComplex*>(output_real.data())));
      //CUDA_RT_CALL(cudaMemcpy(input_complex.data(), d_data, sizeof(input_type) * input_complex.size(),
      //cudaMemcpyDeviceToHost));

      //ip
      CUDA_RT_CALL(cudaMemcpy(input_complex.data(), d_data, sizeof(input_type) * input_complex.size(), cudaMemcpyDeviceToHost));
      //oop
      //CUDA_RT_CALL(cudaMemcpy(input_complex.data(), d_data, sizeof(input_type) * input_complex.size(), cudaMemcpyDeviceToHost));
      
      if(verbose) {
	std::printf("Input array after FFT and Inverse FFT with scaling:\n");
	for(int b=0; b<batch_size; b++) {
	  for(int i = 0; i < input_size; i++) {
	    if(abs(input_complex[i + b*(input_size)].real() - input_complex_copy[i + b*(input_size)].real()) > 0.1 ||
	       abs(input_complex[i + b*(input_size)].imag() - input_complex_copy[i + b*(input_size)].imag()) > 0.1) {	      
	      std::printf("%d,%d: %f + %fj\n", b, i, input_complex[i + b*(input_size)].real(), input_complex[i + b*(input_size)].imag());
	    }
	  }
	}
	std::printf("=====\n");
      }
      
      // Check FFT
      float error = 0.0;
      //#pragma omp parallel for collapse(2) reduction(+:error) 
      for(int b=0; b<batch_size; b++) {
	for(int i = 0; i < input_size; i++) {
	  error += pow(abs(input_complex[i + b*input_size] - input_complex_copy[i + b*input_size]), 2);
	}
      }    
      error = sqrt(error/input_size);    
      std::printf("RMS error = %f\n", error);
    }
    
    /* free resources */
    CUDA_RT_CALL(cudaFree(d_data));
    CUDA_RT_CALL(cudaFree(d_data_aux));

    CUFFT_CALL(cufftDestroy(planc2r));
    CUFFT_CALL(cufftDestroy(planr2c));

    //CUDA_RT_CALL(cudaStreamDestroy(stream));

    CUDA_RT_CALL(cudaDeviceReset());

    return EXIT_SUCCESS;
}
