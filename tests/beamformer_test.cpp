#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <math.h>
#include <string.h>
#include <complex>
#include <random> // DMH: Move to utils for common inclusion

#include <inttypes.h>

// We include here GGP headers in native test routines for convenience.
#include <test.h>
#include <blas_reference.h>
#include <misc.h>

// In a typical application, ggp.h is the only GGP header required.
#include <ggp.h>

// if "--enable-testing true" is passed, we run the tests defined in here
#include <beamformer_test_gtest.hpp>
#include <eigen_helper.h>

namespace quda {
  extern void setTransferGPU(bool);
}

template <typename T> using complex = std::complex<T>;

//#define LOCAL_DEBUG
#define CVAC 299792458.0 // Speed of light `in vacuo`

// Beamformer options
//----------------------
int BF_n_polarizations = 2; // Very unlikely to change.

// Payload dims
int BF_n_payload = 1;
int BF_n_packets_per_payload = 1;
int BF_n_antennae_per_payload = 64;
int BF_n_time_per_payload = 1024;
int BF_n_channels_per_payload = 4;

// Computation (FRB, Pulsar) specific
int BF_n_channels_inner = 2;
int BF_n_time_inner = 2;
int BF_n_time_power_sum = 4;
int BF_n_beam = 512;
int BF_n_flags = 0;
double BF_beam_separation = 1.0;
double BF_beam_separation_ns = 0.75;

// Packet specific, hide these from the CLI...
int BF_n_antennae_per_packet = 1;
int BF_n_time_per_packet = 1024;
int BF_n_channels_per_packet = 4;
// ...keep this exposed
QudaPacketFormat BF_packet_format = QUDA_PACKET_FORMAT_DSA2K;

// Compute properties
QudaPrecision BF_compute_prec = QUDA_SINGLE_PRECISION;
QudaBeamformerType BF_type = QUDA_BEAMFORMER_VOLTAGE;
QudaBLASEngine BF_engine = QUDA_BLAS_ENGINE_CUBLAS;

double BF_sfreq = 1498.75;
double BF_wfreq = 0.244140625;
double BF_declination = 71.66;

// Populate a parameter struct that describes the Beamformer
// Parameter structs are defined in generic_GPU_project/include/ggp.h.
// The relevant enums are defined in generic_GPU_project/include/enums_ggp.h.
// The print/check/new methods are in generic_GPU_project/lib/check_params.h.
void setBeamformerParam(BeamformerParam &bf_param) {

  // Deduce packet properties
  if(BF_packet_format == QUDA_PACKET_FORMAT_DSA110) {
    BF_n_antennae_per_packet = 3;
    BF_n_time_per_packet = 2;
    BF_n_channels_per_packet = 384;
  } else if (BF_packet_format == QUDA_PACKET_FORMAT_DSA2K) {
    BF_n_antennae_per_packet = 1;
    BF_n_time_per_packet = 16;
    BF_n_channels_per_packet = 4;
  } else {
    errorQuda("Unknown packet format %d", BF_packet_format);
  }    
  printfQuda("Packet size is %d bytes\n", BF_packet_format);
  
  // Compute payload size for this computation
  //------------------------------------------
  // Check that we have correct number of antennae number for a payload
  if(BF_n_antennae_per_payload % BF_n_antennae_per_packet != 0) {
    errorQuda("BF_n_antennae_per_payload(%d) %% BF_n_antennae_per_packet(%d) = %d. Ensure this is zero.",
	      BF_n_antennae_per_payload, BF_n_antennae_per_packet, BF_n_antennae_per_payload % BF_n_antennae_per_packet);
  }

  // Check that we have correct number of channels for a payload
  if(BF_n_channels_per_payload % BF_n_channels_per_packet != 0) {
    errorQuda("BF_n_channels_per_payload(%d) %% BF_n_channels_per_packet(%d) = %d. Ensure this is zero.",
	      BF_n_channels_per_payload, BF_n_channels_per_packet, BF_n_channels_per_payload % BF_n_channels_per_packet);    
  }
  if(BF_n_channels_per_payload % BF_n_channels_inner != 0) {
    errorQuda("BF_n_channels_per_payload(%d) %% BF_n_channels_inner(%d) = %d. Ensure this is zero.",
	      BF_n_channels_per_payload, BF_n_channels_inner, BF_n_channels_per_payload % BF_n_channels_inner);    
  }    

  // Check that we have correct number of time steps for a payload
  if(BF_n_time_per_payload % BF_n_time_per_packet != 0) {
    errorQuda("BF_n_time_per_payload(%d) %% BF_n_time_per_packet(%d) = %d. Ensure this is zero.",
	      BF_n_time_per_payload, BF_n_time_per_packet, BF_n_time_per_payload % BF_n_time_per_packet);
  }
  if(BF_n_time_per_payload % BF_n_time_inner != 0) {
    errorQuda("BF_n_time_per_payload(%d) %% BF_n_time_inner(%d) = %d. Ensure this is zero.",
	      BF_n_time_per_payload, BF_n_time_inner, BF_n_time_per_payload % BF_n_time_inner);
  }

  // Compute payload size
  uint64_t payload_size = sizeof(char) * (BF_n_polarizations*
					  BF_n_antennae_per_payload*
					  BF_n_channels_per_payload*
					  BF_n_time_per_payload);

  printfQuda("payload size = %lu bytes\n", payload_size);
  if(payload_size % (int)BF_packet_format != 0) {
    errorQuda("Incorrect payload size %lu for given packet format %d", payload_size, BF_packet_format);    
  }
  printfQuda("Running Beamformer with %lu packets per %d payload(s)\n", payload_size/((int)BF_packet_format), BF_n_payload);

  bf_param.type = BF_type;
  bf_param.packet_format = BF_packet_format;
  bf_param.n_payload = BF_n_payload;
  bf_param.n_antennae_per_payload = BF_n_antennae_per_payload;
  bf_param.n_channels_per_payload = BF_n_channels_per_payload;
  bf_param.n_time_per_payload = BF_n_time_per_payload;
  bf_param.n_time_inner = BF_n_time_inner;
  bf_param.n_time_power_sum = BF_n_time_power_sum;
  bf_param.n_channels_inner = BF_n_channels_inner;
  bf_param.n_arm = 1;  // DSA-2000 = 1 arm (DSA-110 = 2)
  bf_param.n_beam = ((bf_param.n_antennae_per_payload + 1) * bf_param.n_antennae_per_payload)/2;
  bf_param.n_polarizations = BF_n_polarizations;
  bf_param.n_flags = BF_n_flags;
  bf_param.beam_separation = BF_beam_separation;
  bf_param.beam_separation_ns = BF_beam_separation_ns;
  bf_param.sfreq = BF_sfreq;
  bf_param.wfreq = BF_wfreq;
  bf_param.declination = BF_declination;
  bf_param.compute_prec = BF_compute_prec;
  bf_param.engine = BF_engine;
  bf_param.data_type = QUDA_BLAS_DATATYPE_QC;
  bf_param.data_order = QUDA_BLAS_DATAORDER_ROW;
  bf_param.verbosity = verbosity;
  bf_param.struct_size = sizeof(bf_param);
}

void display_test_info() {
  if (getVerbosity() > QUDA_SILENT) {  
    printfQuda("running the following test:\n");
    printfQuda("Beamformer test\n");
    printfQuda("Grid partition info:     X  Y  Z  T\n");
    printfQuda("                         %d  %d  %d  %d\n", dimPartitioned(0), dimPartitioned(1), dimPartitioned(2),
             dimPartitioned(3));
    
    printfQuda("Beamformer parameters\n");
    printfQuda("-- Packet format %s\n", get_packet_format_str(BF_packet_format));
    printfQuda("-- N payloads %d\n", BF_n_payload);
    printfQuda("-- N antennae per payload %d\n", BF_n_antennae_per_payload);
    printfQuda("-- N time per payload %d\n", BF_n_time_per_payload);
    printfQuda("-- N channels per payload %d\n", BF_n_channels_per_payload);
    printfQuda("-- N time power sum %d\n", BF_n_time_power_sum);
    printfQuda("-- N time inner %d\n", BF_n_time_inner);
    printfQuda("-- N channels inner %d\n", BF_n_channels_inner);
    printfQuda("-- N polarizations %d\n", BF_n_polarizations);
    printfQuda("-- N beams %d\n", ((BF_n_antennae_per_payload + 1)*BF_n_antennae_per_payload)/2);
    printfQuda("-- N beam separation (arcmin) %f\n", BF_beam_separation);
    printfQuda("-- beam separation (arcmin) %f\n", BF_beam_separation);
    printfQuda("-- beam separation_ns (arcmin) %f\n", BF_beam_separation_ns);
    printfQuda("-- sfreq (NHz) %f\n", BF_sfreq);
    printfQuda("-- wfreq (NHz) %f\n", BF_wfreq);
    printfQuda("-- declination %f\n", BF_declination);    
    printfQuda("-- Compute precision %s\n", get_prec_str(BF_compute_prec));
  }
}
//----------------------

// Verification routines
//----------------------
#define MASK1 0x0F // (00001111)
#define MASK2 0xF0 // (11110000)

template <typename Float> void promote_to_float_by_pol_complex(Float *output_A, Float *output_B, unsigned char *input_4c4, unsigned long elems) {
  for(unsigned long i=0; i<elems/2; i++) {
    output_A[2*i  ] = 0.05*(Float)((char)(((input_4c4[2*i]) & (unsigned char)(MASK1)) << 4) >> 4);
    output_A[2*i+1] = 0.05*(Float)((char)(((input_4c4[2*i]) & (unsigned char)(MASK2))) >> 4);
    
    output_B[2*i  ] = 0.05*(Float)((char)(((input_4c4[2*i+1]) & (unsigned char)(MASK1)) << 4) >> 4);
    output_B[2*i+1] = 0.05*(Float)((char)(((input_4c4[2*i+1]) & (unsigned char)(MASK2))) >> 4);
    if(i<64) {
      printfQuda("HOST PROMOTE: %lu data_A(%f, %f), data_B(%f,%f)\n", i, output_A[2*i], output_A[2*i+1], output_B[2*i], output_B[2*i+1]);
    }
  }
}

template <typename Float> void promote_to_float_complex(Float *output, unsigned char *input_4c4, unsigned long elems) {
  for(unsigned long i=0; i<elems; i++) {
    output[2*i  ] = (Float)((char)(((input_4c4[i]) & (unsigned char)(MASK1)) << 4) >> 4);
    output[2*i+1] = (Float)((char)(((input_4c4[i]) & (unsigned char)(MASK2))) >> 4);
    if(i<64) {
      printfQuda("HOST PROMOTE: %lu data(%f, %f)\n", i, output[2*i], output[2*i+1]);
    }
  }
}


template <typename FloatOut, typename FloatIn> void prec_switch(FloatOut *output, FloatIn *input, uint64_t elems) {  
  for(uint64_t i=0; i<elems; i++) {
    output[i] = input[i];
  }
}

// DMH: Move to common file
void fillToEigenArray(MatrixXcd &EigenArr, complex<double> *arr, int rows, int cols, int offset, bool tri = false) {
  int counter = offset;
  for (int i = 0; i < rows; i++) {
    for (int j = (tri ? i : 0); j < cols; j++) {
      EigenArr(i, j) = arr[counter];
      counter++;
    }
  }
}

// DMH: Move to common file
void fillFromEigenArray(complex<double> *arr, MatrixXcd &EigenArr, int rows, int cols, int offset, bool tri = false) {
  int counter = offset;
  for (int i = 0; i < rows; i++) {
    for (int j = (tri ? i : 0); j < cols; j++) {
      arr[counter] = EigenArr(i, j);
      counter++;
    }
  }
}

double verifyVoltageBeamformer(void *output_data, void *rand_data, void* weights_A, void* weights_B, std::vector<int> flagants, BeamformerParam &bf_param) {
  
  // Initialise host side data
  //--------------------------
  uint64_t in_block_size = (BF_n_antennae_per_payload *
			    BF_n_channels_per_payload *
			    BF_n_polarizations*
			    BF_n_time_per_payload);
  
  // Create host side input data
  long unsigned prom_input_size = 2 * in_block_size * sizeof(double);
  void *promoted_input_data_A = malloc(prom_input_size/2);
  void *promoted_input_data_B = malloc(prom_input_size/2);
  promote_to_float_by_pol_complex((double*)promoted_input_data_A, (double*)promoted_input_data_B, (unsigned char*)rand_data, in_block_size);

  // Create host side output data
  long unsigned prom_output_size = (BF_n_time_per_payload/(BF_n_time_power_sum*BF_n_time_inner)*(BF_n_channels_per_packet/BF_n_channels_inner)*BF_n_beam) * sizeof(double);
  void *output_data_host = malloc(prom_output_size);
  prec_switch((double*)output_data_host, (unsigned char*)output_data, prom_output_size/sizeof(double));
  
  double sfreq = 1498.75;
  //double declination = 71.66; // FIXME: DSA-110 only?
  
  uint64_t n_freq = bf_param.n_channels_per_payload/bf_param.n_channels_inner;  
  uint64_t n_pos = bf_param.n_antennae_per_payload;
  uint64_t n_calib = bf_param.n_antennae_per_payload * n_freq * bf_param.n_polarizations;
  uint64_t n_weights_per_pol = n_freq * bf_param.n_beam * bf_param.n_antennae_per_payload;
  uint64_t n_results_per_pol = 2 * bf_param.n_channels_per_payload * bf_param.n_beam * bf_param.n_time_per_payload;
  uint64_t ib_sum_elems = bf_param.n_channels_per_payload * bf_param.n_time_per_payload;
  
  //std::vector<int> flagants(bf_param.n_antennae_per_payload, 1);
  std::vector<double> calib_w(2 * n_calib);
  std::vector<double> position(2 * n_pos);
  std::vector<double> freqs_dbl(n_freq);
  std::vector<complex<double>> weights_a(n_weights_per_pol, 0);
  std::vector<complex<double>> weights_b(n_weights_per_pol, 0);
  std::vector<complex<double>> result_a(n_results_per_pol, 0);
  std::vector<complex<double>> result_b(n_results_per_pol, 0);
  std::vector<double> ib_sum_data(ib_sum_elems, 0);

  // Copy float data to double
  for (uint64_t i=0; i<n_weights_per_pol; i++) {
    // FIX ME for arbitraty prec
    weights_a[i].real(((float*)weights_A)[2*i  ]);
    weights_a[i].imag(((float*)weights_A)[2*i+1]);
    weights_b[i].real(((float*)weights_B)[2*i  ]);
    weights_b[i].imag(((float*)weights_B)[2*i+1]);
  }
  
  // DMH: Add CL params to govern this
  for (uint64_t i=0; i<n_freq; i++)
    freqs_dbl[i] = 1e6 * (sfreq - i*250.0/1024.0);
  
  printfQuda("n_freq = %ld\n", n_freq);
  printfQuda("n_pos = %ld\n", n_pos);
  printfQuda("n_calib = %ld\n", n_calib);
  printfQuda("n_weights = n_freq * bf_param.n_beam * bf_param.n_antennae_per_payload = %ld = %ld * %ld * %ld\n", n_weights_per_pol, n_freq, bf_param.n_beam, bf_param.n_antennae_per_payload);    
  //--------------------------
  
  // Problem parameters
  //-------------------
  int m = bf_param.n_beam;
  int n = bf_param.n_channels_inner * bf_param.n_time_per_payload;
  int k = bf_param.n_antennae_per_payload;
  int batches = bf_param.n_channels_per_payload / bf_param.n_channels_inner;

  // Eigen objects
  // W = weights, V = voltages, res = W * V^H
  MatrixXcd W_a = MatrixXd::Zero(m, k);
  MatrixXcd W_b = MatrixXd::Zero(m, k);
  MatrixXcd V_a = MatrixXd::Zero(n, k);
  MatrixXcd V_b = MatrixXd::Zero(n, k);
  MatrixXcd res_a_host = MatrixXd::Zero(m, n);
  MatrixXcd res_b_host = MatrixXd::Zero(m, n);

  int W_offset = 0;
  int V_offset = 0;
  int res_offset = 0;
  double dev = 0.0;

  complex<double> *V_a_ptr = (complex<double> *)(&promoted_input_data_A)[0];
  complex<double> *V_b_ptr = (complex<double> *)(&promoted_input_data_B)[0];

  // Emulate Batched cuBLAS
  printfQuda("W(%d, %d), V(%d, %d), B(%d, %d)\n", m, k, k, n, m, n);
  for (int batch = 0; batch < batches; batch++) {
    
    // Ensure all arrays are zeroed out.
    W_a.setZero();
    W_b.setZero();
    V_a.setZero();
    V_b.setZero();
    res_a_host.setZero();
    res_b_host.setZero();

    // Populate Eigen objects
    fillToEigenArray(W_a, weights_a.data(), m, k, W_offset);
    fillToEigenArray(W_b, weights_b.data(), m, k, W_offset);
    fillToEigenArray(V_a, V_a_ptr, n, k, V_offset);
    fillToEigenArray(V_b, V_b_ptr, n, k, V_offset);
    
    // Inspect weights (good)
    for(int i=0; i<8; i++) {
      //printfQuda("%d: W_A = (%f,%f), W_B = (%f,%f)\n", i, W_a(0,i).real(),  W_a(0,i).imag(), W_b(0,i).real(), W_b(0,i).imag());
    }
    
    // Polariasation A
    res_a_host = W_a * V_a.adjoint();

    // Inspect result (good)
    for(int i=0; i<8; i++) {
      //std::cout << B_host(0,i) << std::endl;
    }
    
    // Polariasation B
    res_b_host = W_b * V_b.adjoint();

    // Inspect result (good)
    for(int i=0; i<8; i++) {
      //std::cout << B_host(0,i) << std::endl;
    }

    // Copy result data to array
    fillFromEigenArray(result_a.data(), res_a_host, m, n, res_offset);
    fillFromEigenArray(result_b.data(), res_b_host, m, n, res_offset);
    
    W_offset += m * k;
    V_offset += k * n;
    res_offset += m * n;
  }

  // Emulate sumIncoherentBeam
  
  
  free(output_data_host);
  free(promoted_input_data_A);
  free(promoted_input_data_B);
  
  return dev;
}

#if 0
double verifyVisibilityBeamformer(void *output_data, void *rand_data, void* weights_A, void* weights_B, std::vector<int> flagants, BeamformerParam &bf_param) {

  // Initialise host side data
  //--------------------------
  uint64_t in_block_size = (BF_n_antennae_per_payload *
			    BF_n_channels_per_payload *
			    BF_n_polarizations*
			    BF_n_time_per_payload);
			    
  // Create host side input data
  long unsigned prom_input_size = 2 * in_block_size * sizeof(double);
  void *promoted_input_data = malloc(prom_input_size);
  promote_to_float_complex((double*)promoted_input_data, (unsigned char*)rand_data, in_block_size);

  // Calculate frequencies
  std::vector<double> freqs(NCHAN);
  for (int i = 0; i < NCHAN; ++i) {
    freqs[i] = 2e9 - static_cast<double>(i) * 2e7;
  }
  const double lambda_max = CVAC / freqs.back();  

  // Create host side output data
  long unsigned prom_output_size = (BF_n_time_per_payload/(BF_n_time_power_sum*BF_n_time_inner)*(BF_n_channels_per_packet/BF_n_channels_inner)*BF_n_beam) * sizeof(double);
  void *output_data_host = malloc(prom_output_size);
  prec_switch((double*)output_data_host, (unsigned char*)output_data, prom_output_size/sizeof(double));
  
  double sfreq = 1498.75;
  //double declination = 71.66; // FIXME: DSA-110 only?
  
  uint64_t n_freq = bf_param.n_channels_per_payload/bf_param.n_channels_inner;  
  uint64_t n_pos = bf_param.n_antennae_per_payload;
  uint64_t n_calib = bf_param.n_antennae_per_payload * n_freq * bf_param.n_polarizations;
  uint64_t n_weights_per_pol = n_freq * bf_param.n_beam * bf_param.n_antennae_per_payload;
  uint64_t n_results_per_pol = 2 * bf_param.n_channels_per_payload * bf_param.n_beam * bf_param.n_time_per_payload;
  uint64_t ib_sum_elems = bf_param.n_channels_per_payload * bf_param.n_time_per_payload;
  
  //std::vector<int> flagants(bf_param.n_antennae_per_payload, 1);
  std::vector<double> calib_w(2 * n_calib);
  std::vector<double> position(2 * n_pos);
  std::vector<double> freqs_dbl(n_freq);
  std::vector<complex<double>> weights_a(n_weights_per_pol, 0);
  std::vector<complex<double>> weights_b(n_weights_per_pol, 0);
  std::vector<complex<double>> result_a(n_results_per_pol, 0);
  std::vector<complex<double>> result_b(n_results_per_pol, 0);
  std::vector<double> ib_sum_data(ib_sum_elems, 0);

  // Copy float data to double
  for (uint64_t i=0; i<n_weights_per_pol; i++) {
    // FIX ME for arbitraty prec
    weights_a[i].real(((float*)weights_A)[2*i  ]);
    weights_a[i].imag(((float*)weights_A)[2*i+1]);
    weights_b[i].real(((float*)weights_B)[2*i  ]);
    weights_b[i].imag(((float*)weights_B)[2*i+1]);
  }
  
  // DMH: Add CL params to govern this
  for (uint64_t i=0; i<n_freq; i++)
    freqs_dbl[i] = 1e6 * (sfreq - i*250.0/1024.0);
  
  printfQuda("n_freq = %ld\n", n_freq);
  printfQuda("n_pos = %ld\n", n_pos);
  printfQuda("n_calib = %ld\n", n_calib);
  printfQuda("n_weights = n_freq * bf_param.n_beam * bf_param.n_antennae_per_payload = %ld = %ld * %ld * %ld\n", n_weights_per_pol, n_freq, bf_param.n_beam, bf_param.n_antennae_per_payload);    
  //--------------------------
  
  // Problem parameters
  //-------------------
  int m = bf_param.n_beam;
  int n = bf_param.n_channels_inner * bf_param.n_time_per_payload;
  int k = bf_param.n_antennae_per_payload;
  int batches = bf_param.n_channels_per_payload / bf_param.n_channels_inner;

  // Eigen objects
  // W = weights, V = voltages, res = W * V^H
  MatrixXcd W_a = MatrixXd::Zero(m, k);
  MatrixXcd W_b = MatrixXd::Zero(m, k);
  MatrixXcd V_a = MatrixXd::Zero(n, k);
  MatrixXcd V_b = MatrixXd::Zero(n, k);
  MatrixXcd res_a_host = MatrixXd::Zero(m, n);
  MatrixXcd res_b_host = MatrixXd::Zero(m, n);

  int W_offset = 0;
  int V_offset = 0;
  int res_offset = 0;
  double dev = 0.0;

  complex<double> *V_a_ptr = (complex<double> *)(&promoted_input_data_A)[0];
  complex<double> *V_b_ptr = (complex<double> *)(&promoted_input_data_B)[0];

  // Emulate Batched cuBLAS
  printfQuda("W(%d, %d), V(%d, %d), B(%d, %d)\n", m, k, k, n, m, n);
  for (int batch = 0; batch < batches; batch++) {
    
    // Ensure all arrays are zeroed out.
    W_a.setZero();
    W_b.setZero();
    V_a.setZero();
    V_b.setZero();
    res_a_host.setZero();
    res_b_host.setZero();

    // Populate Eigen objects
    fillToEigenArray(W_a, weights_a.data(), m, k, W_offset);
    fillToEigenArray(W_b, weights_b.data(), m, k, W_offset);
    fillToEigenArray(V_a, V_a_ptr, n, k, V_offset);
    fillToEigenArray(V_b, V_b_ptr, n, k, V_offset);
    
    // Inspect weights (good)
    for(int i=0; i<8; i++) {
      //printfQuda("%d: W_A = (%f,%f), W_B = (%f,%f)\n", i, W_a(0,i).real(),  W_a(0,i).imag(), W_b(0,i).real(), W_b(0,i).imag());
    }
    
    // Polariasation A
    res_a_host = W_a * V_a.adjoint();

    // Inspect result (good)
    for(int i=0; i<8; i++) {
      //std::cout << B_host(0,i) << std::endl;
    }
    
    // Polariasation B
    res_b_host = W_b * V_b.adjoint();

    // Inspect result (good)
    for(int i=0; i<8; i++) {
      //std::cout << B_host(0,i) << std::endl;
    }

    // Copy result data to array
    fillFromEigenArray(result_a.data(), res_a_host, m, n, res_offset);
    fillFromEigenArray(result_b.data(), res_b_host, m, n, res_offset);
    
    W_offset += m * k;
    V_offset += k * n;
    res_offset += m * n;
  }

  // Emulate sumIncoherentBeam
  
  
  free(output_data_host);
  free(promoted_input_data_A);
  free(promoted_input_data_B);
  
  return dev;
}
#endif

static const char* engine_name(QudaBLASEngine e) {
  switch (e) {
    case QUDA_BLAS_ENGINE_CUTLASS: return "CUTLASS";
    case QUDA_BLAS_ENGINE_TCC:     return "TCC";
    default:                       return "cuBLAS";
  }
}

double BeamformerTest(test_t test_param) {

  BeamformerParam bf_param = newBeamformerParam();
  BF_compute_prec = ::testing::get<0>(test_param);
  setBeamformerParam(bf_param);
  display_test_info();

  // Create a data array for a single call to the XEngine
  // one (4,4)b piece of data fits into one char.
  uint64_t in_payload_size;
  in_payload_size = (BF_n_time_per_payload *
		     BF_n_antennae_per_payload *
		     BF_n_channels_per_payload *
		     BF_n_polarizations);
  
  logQuda(QUDA_VERBOSE, "Creating input array of size %f MB\n", (1.0*sizeof(char)*BF_n_payload*in_payload_size)/pow(1000,2));
  logQuda(QUDA_VERBOSE, "Input (4bit,4bit) complex elements %lu\n", BF_n_payload * in_payload_size);

  // Populate input array with random bits. These 32 bit ints will
  // be interpreted as (4bit,4bit) signed complex pairs, eg
  // 01110100101110000101100001101001 = 1,958,238,313
  // becomes
  // (0111,0100) (1011,1000) (0101,1000) (0110,1001) = (7,4) (-3,-8) (5,-8) (6,1)
  void *rand_data = pinned_malloc(BF_n_payload * in_payload_size);  
  int n_rand = (BF_n_payload * in_payload_size)/sizeof(int);
  int *p = (int*)rand_data;
  for (int i = 0; i < n_rand; i++) p[i] = rand();
  //for (int i = 0; i < n_rand; i++) p[i] = i;

  // Create an output array for the Beamformer
  uint64_t out_payload_size = (BF_n_time_per_payload/(BF_n_time_power_sum * BF_n_time_inner)) * (BF_n_channels_per_payload/BF_n_channels_inner) * BF_n_beam;
  uint64_t output_size = out_payload_size * BF_n_payload;  
  logQuda(QUDA_VERBOSE, "Creating output_array of size %f MiB\n", (1.0*output_size)/pow(1000,2));
  logQuda(QUDA_VERBOSE, "Output (unsigned char) real elements %lu\n", BF_n_payload * out_payload_size);
  void *output_data = pinned_malloc(output_size);

  double sfreq = 1498.75;
  //double declination = 71.66; // FIXME: DSA-110 only? Position on Earth?
    
  uint64_t n_freq = bf_param.n_channels_per_payload/bf_param.n_channels_inner;
  uint64_t n_pos = bf_param.n_antennae_per_payload;
  uint64_t n_calib = bf_param.n_antennae_per_payload * n_freq * bf_param.n_polarizations;
  uint64_t n_weights_per_pol = n_freq * n_pos * bf_param.n_beam * bf_param.n_time_inner;
  
  std::vector<float> calib_w(2 * n_calib);
  std::vector<float> position(2 * n_pos);
  std::vector<float> freqs(n_freq);
  std::vector<int> flagants(n_pos, 1);

  // Emulate 20% offline antennae
  for(uint64_t i=0; i<n_pos; i++) {
    if(i%5 == 0) flagants[i] = 0;
  }

  // Calculate frequencies
  for (uint64_t i=0; i<n_freq; i++) {
    freqs[i] = 1e6 * (sfreq - i*250.0/1024.0);
  }
  const double lambda_max = CVAC / freqs.back();
  
  // DMH: Add CL params to govern this
  for (uint64_t i=0; i<n_freq; i++) freqs[i] = 1e6 * (sfreq - i*250.0/1024.0);

  void *weights_A = pinned_malloc(2 * n_weights_per_pol * sizeof(float));
  void *weights_B = pinned_malloc(2 * n_weights_per_pol * sizeof(float));

  // DMH: Add CL params to govern this
  float one_on_2pi = 1.0/(2*M_PI);
  float theta = 0;  
  //#pragma omp parallel for
  for(uint64_t i=0; i<n_weights_per_pol; i++) {
    theta = one_on_2pi * rand();
    ((float*)weights_A)[2*i]   = sin(theta);
    ((float*)weights_A)[2*i+1] = cos(theta);
    theta = one_on_2pi * rand();
    ((float*)weights_B)[2*i]   = sin(theta);
    ((float*)weights_B)[2*i+1] = cos(theta);
  }
  
  // Input array is interpreted as a row major array of matrices,
  double dev = 0.0;  
  switch(bf_param.type) {
  case QUDA_BEAMFORMER_VOLTAGE:
    beamformerVoltageCHPC(output_data, rand_data, freqs.data(), weights_A, weights_B, flagants.data(), &bf_param);
    if(verify_results) dev = verifyVoltageBeamformer(output_data, rand_data, weights_A, weights_B, flagants, bf_param);
    break;
  case QUDA_BEAMFORMER_VISIBILITY: {
    // Set up proper visibility beamformer test data
    // n_beam for visibility BF = n_baselines = N*(N+1)/2
    int Ng = (bf_param.n_grid > 0) ? static_cast<int>(bf_param.n_grid) : 1024;
    int n_ant = static_cast<int>(bf_param.n_antennae_per_payload);
    int vis_n_baselines = n_ant * (n_ant + 1) / 2;
    int vis_n_beam = static_cast<int>(bf_param.n_beam);

    // Baseline UV coordinates [n_baselines x 2] floats (metres)
    // Spread baselines across the UV plane so they hit real grid cells
    std::vector<float> baseline_uvs(vis_n_baselines * 2);
    {
      int idx = 0;
      for (int i = 0; i < n_ant; i++) {
        for (int j = 0; j <= i; j++) {
          // Place baselines with separations that map to grid cells
          baseline_uvs[idx * 2]     = static_cast<float>((i - j) * 10.0);   // u in metres
          baseline_uvs[idx * 2 + 1] = static_cast<float>((i + j) * 5.0);    // v in metres
          idx++;
        }
      }
    }

    // Beam pixel coordinates [n_beam x 2] ints (col, row) within [0, Ng)
    // Place beams at evenly-spaced grid points to get non-zero outputs
    std::vector<int> beam_pixels(vis_n_beam * 2);
    for (int b = 0; b < vis_n_beam; b++) {
      beam_pixels[b * 2]     = (b * 7) % Ng;           // col
      beam_pixels[b * 2 + 1] = (b * 13 + Ng/4) % Ng;   // row
    }

    // Frequencies: use realistic DSA-2000 channels
    uint64_t vis_n_channels = bf_param.n_channels_per_payload;
    std::vector<double> vis_freqs(vis_n_channels);
    double sfreq_hz = 1.498750e9;  // 1498.75 MHz
    for (uint64_t i = 0; i < vis_n_channels; i++) {
      vis_freqs[i] = sfreq_hz - i * 244140.625;  // 250 MHz / 1024 channels
    }

    // Run first engine
    const char *eng1 = engine_name(bf_param.engine);
    printfQuda("\n=== Running visibility beamformer with %s engine ===\n", eng1);
    beamformerVisibilityCHPC(output_data, rand_data, vis_freqs.data(),
                            baseline_uvs.data(), beam_pixels.data(), nullptr, &bf_param);

    // Cross-check: run with the other engine and compare outputs
    uint64_t beam_output_elems = vis_n_channels * vis_n_beam;
    uint64_t beam_output_bytes = beam_output_elems * sizeof(float);

    // Save first engine's output
    std::vector<float> output_ref(beam_output_elems);
    memcpy(output_ref.data(), output_data, beam_output_bytes);

    // Run with alternate engine
    BeamformerParam alt_param = bf_param;
    alt_param.engine = (bf_param.engine == QUDA_BLAS_ENGINE_CUTLASS)
                        ? QUDA_BLAS_ENGINE_CUBLAS : QUDA_BLAS_ENGINE_CUTLASS;
    const char *eng2 = engine_name(alt_param.engine);

    printfQuda("\n=== Running visibility beamformer with %s engine ===\n", eng2);
    beamformerVisibilityCHPC(output_data, rand_data, vis_freqs.data(),
                            baseline_uvs.data(), beam_pixels.data(), nullptr, &alt_param);

    // Compare
    const float *out_alt = static_cast<const float*>(output_data);
    double max_abs_err = 0.0, max_rel_err = 0.0, sum_sq_err = 0.0, sum_sq_ref = 0.0;
    uint64_t n_nonzero = 0, n_mismatch = 0;
    for (uint64_t i = 0; i < beam_output_elems; i++) {
      float ref = output_ref[i];
      float alt = out_alt[i];
      double abs_err = std::abs(static_cast<double>(ref) - static_cast<double>(alt));
      if (abs_err > max_abs_err) max_abs_err = abs_err;
      sum_sq_err += abs_err * abs_err;
      sum_sq_ref += static_cast<double>(ref) * static_cast<double>(ref);
      if (ref != 0.0f) {
        n_nonzero++;
        double rel = abs_err / std::abs(static_cast<double>(ref));
        if (rel > max_rel_err) max_rel_err = rel;
        if (rel > 0.01) n_mismatch++;
      }
    }
    double rms_err = (sum_sq_ref > 0) ? std::sqrt(sum_sq_err / sum_sq_ref) : 0.0;
    printfQuda("\n=== Cross-check: %s vs %s ===\n", eng1, eng2);
    printfQuda("  Total beam output elements: %lu\n", beam_output_elems);
    printfQuda("  Non-zero reference values:  %lu\n", n_nonzero);
    printfQuda("  Max absolute error:         %e\n", max_abs_err);
    printfQuda("  Max relative error:         %e\n", max_rel_err);
    printfQuda("  RMS relative error:         %e\n", rms_err);
    printfQuda("  Elements >1%% relative err: %lu\n", n_mismatch);
    printfQuda("=== Cross-check %s ===\n\n", (rms_err < 0.05) ? "PASSED" : "FAILED");
    dev = rms_err;
    break;
  }
  default: errorQuda("Unknown beamformer type %d", bf_param.type);
  }
  
  host_free(rand_data);
  host_free(output_data);
  host_free(weights_A);
  host_free(weights_B);    
  return dev;
}


struct beamformer_test : quda_test {
  
  void add_command_line_group(std::shared_ptr<GGPApp> app) const override
  {
    quda_test::add_command_line_group(app);

    CLI::TransformPairs<QudaPrecision> precision_map {{"double", QUDA_DOUBLE_PRECISION},
						      {"single", QUDA_SINGLE_PRECISION},
						      {"half", QUDA_HALF_PRECISION},
						      {"quarter", QUDA_QUARTER_PRECISION}};

    CLI::TransformPairs<QudaPacketFormat> packet_map {{"dsa110", QUDA_PACKET_FORMAT_DSA110},
						      {"dsa2k", QUDA_PACKET_FORMAT_DSA2K}};

    CLI::TransformPairs<QudaBeamformerType> bf_map {{"voltage", QUDA_BEAMFORMER_VOLTAGE},
						    {"visibility", QUDA_BEAMFORMER_VISIBILITY}};

    CLI::TransformPairs<QudaBLASEngine> engine_map {{"cublas", QUDA_BLAS_ENGINE_CUBLAS},
						    {"cutlass", QUDA_BLAS_ENGINE_CUTLASS},
						    {"tcc", QUDA_BLAS_ENGINE_TCC}};

    CLI::GGPCheckedTransformer prec_transform(precision_map);
    CLI::GGPCheckedTransformer packet_transform(packet_map);
    CLI::GGPCheckedTransformer bf_transform(bf_map);
    CLI::GGPCheckedTransformer engine_transform(engine_map);
    
    auto opgroup = app->add_option_group("Beamformer", "Options controling the Beamformer test parameters");
    opgroup->add_option("--BF-type", BF_type, "Type of beamformer to compute (default voltage)")->transform(bf_transform);
    opgroup->add_option("--BF-packet-format", BF_packet_format, "Number of bytes and order in a packet (default dsa2k)")->transform(packet_transform);
    opgroup->add_option("--BF-n-payload", BF_n_payload, "Number of payloads to move to device (default 1)");
    opgroup->add_option("--BF-compute-prec", BF_compute_prec, "Compute and output precision (default float)")->transform(prec_transform);
    opgroup->add_option("--BF-n-antennae-per-payload", BF_n_antennae_per_payload, "Number of antennae producing data per payload(default 96)");
    opgroup->add_option("--BF-n-channels-per-payload", BF_n_channels_per_payload, "Number of frequency channels per payload of data (default 384)");
    opgroup->add_option("--BF-n-time-per-payload", BF_n_time_per_payload, "Number of time stamps per payload (default 2048)");
    opgroup->add_option("--BF-n-channels-inner", BF_n_channels_inner, "Number of frequency channels over which to sum (default 8)");
    opgroup->add_option("--BF-n-time-inner", BF_n_time_inner, "Number of fine time steps to sum over in channel sum (default 2)");
    opgroup->add_option("--BF-n-time-power-sum", BF_n_time_power_sum, "Number of time steps to sum over power sum (default 4)");
    
    opgroup->add_option("--BF-engine", BF_engine, "BLAS engine: cublas, cutlass, or tcc (default cublas)")->transform(engine_transform);
    opgroup->add_option("--BF-n-beam", BF_n_beam, "Number of beams (default 512)");
    opgroup->add_option("--BF-beam-separation", BF_beam_separation, "Beams separation in arcmin (default 1.0)");
    opgroup->add_option("--BF-n-polarizations", BF_n_polarizations, "Number of polarizations (default 2)");
  }
  
  beamformer_test(int argc, char **argv) : quda_test("Beamformer Test", argc, argv) { }
};

int main(int argc, char **argv) {
  
  beamformer_test test(argc, argv);
  test.init();
  
  int result = 0;
  if (enable_testing) {
    // Perform the tests defined in beamformer_test_gtest.hpp.
    result = test.execute();
    if (result) warningQuda("Google tests for Beamformer failed.");
  } else {
    // Perform the test specified by the command line.
    BeamformerTest(test_t {BF_compute_prec});
  }
  
  return result;
}

// Defer development for now
#if 0
template <typename Float> void populateWeights(std::vector<complex<Float>> &weights_a,
					       std::vector<complex<Float>> &weights_b,
					       std::vector<Float> &position, std::vector<Float> &freqs,
					       std::vector<Float> &calibs, BeamformerParam &bf_param) {
  
  int n_freq = bf_param.n_channels_per_payload/bf_param.n_channels_inner;
  int n_beam = bf_param.n_beam;
  int n_antennae = bf_param.n_antennae_per_payload;
  int n_pol = bf_param.n_polarizations;
  Float sep = bf_param.beam_separation;

  double const_val = sqrt(2)/2;
  
  uint64_t widx, i_weight;
  Float theta, afac, twr, twi;
  for(int i_antenna = 0; i_antenna < n_antennae; i_antenna++) {
    for(int i_beam = 0; i_beam < n_beam; i_beam++) {
      for(int i_freq = 0; i_freq < n_freq; i_freq++) {

	i_weight = i_freq + i_beam * n_freq + i_antenna * n_freq * n_beam;
	// Defer more accurate calibration and weight computation.
	// For now, use simple constants for weights.
	
	/*
	widx = 2 * n_pol * (i_antenna * n_freq + i_freq);	
	//printfQuda("i_weight = %ld, widx = %ld\n", i_weight, widx);
	theta = (sep*(127 - i_beam)*M_PI)/10800.0; // radians (FIXME for dsa2K)
	afac = -2.0 * (M_PI * freqs[i_freq] * theta)/CVAC; // factor for rotate
	twr = cos(afac * position[i_antenna]);
	twi = sin(afac * position[i_antenna]);

	weights_a[i_weight].real(twr*calibs[widx + 0] - twi*calibs[widx + 1]);
	weights_a[i_weight].imag(twi*calibs[widx + 0] + twr*calibs[widx + 1]);
	weights_b[i_weight].real(twr*calibs[widx + 2] - twi*calibs[widx + 3]);
	weights_b[i_weight].imag(twi*calibs[widx + 2] + twr*calibs[widx + 3]);
	*/

	weights_a[i_weight].real(const_val);
	weights_a[i_weight].imag(const_val);
	weights_b[i_weight].real(const_val);
	weights_b[i_weight].imag(const_val);       
      }
    } 
  }  
}

template <typename Float> void computeWeights(std::vector<complex<Float>> &weights_a,
					      std::vector<complex<Float>> &weights_b,
					      std::vector<Float> &calib_w, std::vector<Float> &position,
					      std::vector<Float> &freqs, std::vector<int> &flagants,
					      BeamformerParam &bf_param) {
  
  std::vector<Float> calibs(calib_w.size(), 0);
  
  int i_antenna = 0;
  Float wnorm = 0.0;
  int n_freq = bf_param.n_channels_per_payload/bf_param.n_channels_inner;
  std::vector<bool> flagged(bf_param.n_antennae_per_payload, 0);
  
  for (int i=0; i<2*bf_param.n_antennae_per_payload * n_freq; i++) {
    // Modulo out frequency band to identify an antenna, zero
    // out its calibration data if flagged.
    i_antenna = i/(2*n_freq);
    for (uint64_t j=0; j<flagants.size(); j++)
      if (flagants[j] == i_antenna) flagged[i] = true;

    if (!flagged[i_antenna]) {
      wnorm = sqrt(calib_w[2*i]*calib_w[2*i] + calib_w[2*i+1]*calib_w[2*i+1]);
      if (wnorm != 0.0) {
	calibs[2*i] /= wnorm;
	calibs[2*i+1] /= wnorm;
      }
    } else {
      calibs[2*i] = 0.0;
      calibs[2*i+1] = 0.0;
    }
  }

  populateWeights(weights_a, weights_b, position, freqs, calibs, bf_param);  
}
#endif
