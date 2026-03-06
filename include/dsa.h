#pragma once

#include <ggp_internal.h>

namespace quda
{
  /**
   * Function to inspect device data
   *
   * @param[in] array The data to inspect
   * @param[in] N Total elements in array
   * @param[in] N_low The starting index
   * @param[in] N_high The index limit
   * @param[in] prec The precision of the data
   * @param[in] int array ID
   */
  void inspectDevElems(const void *array, const unsigned long long int N_low, const unsigned long long int N, const unsigned long long int N_high, const QudaPrecision prec, const int ID);
  
  /**
   * Function promotes (4,4)b complex interleaved data to planar with a specified precision
   *
   * @param[in/out] output_real The real promoted data
   * @param[in/out] output_imag The imag promoted data
   * @param[in] input_data The data to be promoted
   * @param[in] N The number of data elements
   * @param[in] precOut The precision of the output_data
   * @param[in] stream_idx The nth stream in the algorithm from which this function is called
   */
  void promoteDataPlanar(void *output_real, void *output_imag, const void *input_data, const unsigned long long int N, const QudaPrecision precOut, int stream_idx);

  /**
   * Function promotes (4,4)b complex interleaved data to planar with a specified precision
   *
   * @param[out] output The promoted data
   * @param[in] input The data to be promoted
   * @param[in] N The number of complex data elements
   * @param[in] precOut The precision of the output_data
   * @param[in] stream_idx The nth stream in the algorithm from which this function is called
   */
  void promoteData(void *output, const void *input, const unsigned long long int N, const QudaPrecision precOut, int stream_idx);

  /**
   * Function promotes (4,4)b complex interleaved data to planar with a specified precision
   *
   * @param[out] output The promoted data for Pol A
   * @param[out] output The promoted data for Pol B
   * @param[in] input The data to be promoted
   * @param[in] N The number of complex data elements
   * @param[in] precOut The precision of the output_data
   * @param[in] stream_idx The nth stream in the algorithm from which this function is called
   */
  void promoteDataPol(void *output_A, void *output_B, const void *input, const unsigned long long int N, const QudaPrecision precOut, int stream_idx);

  /**
   * Function interleaves planar complex data to upper/lower triangular form.
   *
   * @param[out] output_full The full interleaved triangular data
   * @param[in] input_real The real planar data
   * @param[in] input_imag The imag planar data
   * @param[in] N The number of complex data elements
   * @param[in] prec The precision of the data
   */
  void promInterTri(void *output_full, const void *input_real, const void *input_imag, const unsigned long long int N, const QudaPrecision prec);

  /**
   * Function interleaves planar complex data to hermitian form.
   *
   * @param[out] output_full The full interleaved hermitian data
   * @param[in] input_real The real planar data
   * @param[in] input_imag The imag planar data
   * @param[in] N The number of complex data elements
   * @param[in] prec The precision of the data
   */
  void promInter(void *output_full, const void *input_real, const void *input_imag, const unsigned long long int N, const QudaPrecision prec);
  
  /**
   * Function that triangulates interleaved complex data to upper/lower triangular form.
   *
   * @param[out] tri_output The full interleaved triangular data
   * @param[in] input The complex interleaved data
   * @param[in] N The number of complex input data elements
   * @param[in] N_batch The number of complex input data elements in a matrix
   * @param[in] prec The precision of the data
   * @param[in] stream_idx The nth stream in the algorithm from which this function is called
   */
  void triangulateFromHerm(void *tri_output, const void *input, const unsigned long long int N, const unsigned long long int N_batch, const QudaPrecision prec, int stream_idx);
  

  /**
   * Function that sums over antannae to compute the incoherent beam sum
   *
   * @param[out] ib_sum The incoherent beam sum
   * @param[in] flagants The array of valid and invalid antennae
   * @param[in] input_A voltage data polarization A
   * @param[in] input_B voltage data polarization B
   * @param[in] N number of reduction items
   * @param[in] batches number of reduction batches
   * @param[in] prec the compute precision
   * @param[in] stream_idx The nth stream in the algorithm from which this function is called
   */  
  void sumIncoherentBeam(void *ib_sum, const void *flagants, const void *input_A, const void *input_B, const uint64_t N, const uint64_t batches, const QudaPrecision prec, int stream_idx);
  
  /**
   * Function that forms the power sum over antennae summed data and a fime time width
   *
   * @param[out] ps_sum The power sum (with incoherent beam subtraction)
   * @param[in] input_A voltage data polarization A
   * @param[in] input_B voltage data polarization B
   * @param[out] ib_sum The incoherent beam sum
   * @param[in] N number of threads
   * @param[in] n_beam number of beams
   * @param[in] n_time_power_sum number of time steps in power sum
   * @param[in] prec the compute precision
   * @param[in] stream_idx The nth stream in the algorithm from which this function is called
   */
  void powerSum(void *ps_sum, const void *input_A, const void *input_B, const void *ib_sum_data, const uint64_t N, const uint64_t n_beam, const uint64_t n_time_power_sum, const QudaPrecision prec, int stream_idx,
                unsigned int n_time, unsigned int n_chan, unsigned int n_chan_sum, unsigned int n_time_inner);

  /**
   * Function that performs a sum over inner channels and time steps to form the final beam
   * in a given channel window
   *
   * @param[out] output The beam for the given frequency window
   * @param[in] ps_sum the power summed data.
   * @param[in] N number of threads
   * @param[in] n_channels_inner number of inner_channels to sum
   * @param[in] n_time_inner number of inner time steps to sum
   * @param[in] prec the compute precision
   * @param[in] stream_idx The nth stream in the algorithm from which this function is called
   */  
  void sumInnerChanTime(void *output, const void *ps_sum, const uint64_t N, const uint64_t n_channels_inner, const uint64_t n_time_inner, const QudaPrecision prec, int stream_idx);
  void sumInnerChanTimeT(void *output, const void *ps_sum, const uint64_t N, const uint64_t n_time, const uint64_t n_beam, const uint64_t n_chan, const uint64_t n_channels_inner, const uint64_t n_time_inner, const QudaPrecision prec, int stream_idx);

  /**
   * Function that reorders a dsa110 data block to native dsa2000 data order
   *
   * @param[out] output the dsa2000 ordered raw voltage
   * @param[in] input the dsa110 ordered raw voltage
   * @param[in] prec the precision of the output
   * @param[in] stream_idx The nth stream in the algorithm from which this function is called
   */  
  void reorderData(void *output, const void *input, const QudaPrecision prec, int stream_idx);

  /**
   * Custom: Function that reorders a dsa110 data block to native dsa2000 data order
   * for each polarization
   *
   * @param[out] output_A the dsa2000 ordered raw voltage for polarizaion A
   * @param[out] output_B the dsa2000 ordered raw voltage for polarizaion B
   * @param[in] input the dsa110 ordered raw voltage
   * @param[in] prec the precision of the output
   * @param[in] stream_idx The nth stream in the algorithm from which this function is called
   */  
  void reorderDataPol(void *output_A, void *output_B, const void *input, const QudaPrecision prec, int stream_idx);
  void reorderDataPolT(void *output_A, void *output_B, const void *input, const uint64_t n_time, const uint64_t n_time_inner, const uint64_t n_pol, const uint64_t n_chan, const uint64_t n_ant, const QudaPrecision prec, int stream_idx);
  
  /**
   * Custom: Function that reorders and promotes dsa110 weights to interleaved float
   *
   * @param[out] output_A interleaved weights for pol A
   * @param[out] output_B interleaved weights for pol B
   * @param[in] ant_E The east/west antennae positions
   * @param[in] ant_N The north/south antennae positions
   * @param[in] calibs The calibration data
   * @param[in] flagants Flagged (bad) antennae
   * @param[in] freqs Frequency bands
   * @param[in] dec The declination
   * @param[in] n_arm The the number of arms
   * @param[in] N The number of elements in an input array.
   * @param[in] prec the precision of the output
   */
  void computeWeights(void *output_A, void *output_B, const void *ant_E, const void *ant_N, const void *calibs, const void *flagants, const void *freq, const float dec, const int n_arm, const unsigned long long int N, const QudaPrecision prec_out);

  /**
     @brief Strided Batch GEMM. This function performs N GEMM type operations in a
     strided batched fashion. If the user passes
     
     stride<A,B,C> = -1
       
     it deduces the strides for the A, B, and C arrays from the matrix dimensions,
     leading dims, etc, and will behave identically to the batched GEMM.
     If any of the stride<A,B,C> values passed in the parameter structure are
     greater than or equal to 0, the routine accepts the user's values instead.
     
     Example: If the user passes
     
     a_stride = 0
     
     the routine will use only the first matrix in the A array and compute
     
     C_{n} <- a * A_{0} * B_{n} + b * C_{n}
     
     where n is the batch index.
     
     @param[in] A Matrix field containing the A input matrices
     @param[in] B Matrix field containing the B input matrices
     @param[in/out] C Matrix field containing the result, and matrix to be added
     @param[in] blas_param Parameter structure defining the GEMM type
     @param[in] Location of the input/output data
     @return Number of flops done in this computation
  */  
  //long long stridedBatchGEMMCutlass(void *A, void *B, void *C, QudaBLASParam blas_param, QudaFieldLocation location);
  
}
