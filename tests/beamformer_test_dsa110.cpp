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
int BF_n_antennae_per_payload = 96;
int BF_n_time_per_payload = 4096;
int BF_n_channels_per_payload = 384;

// Computation (FRB, Pulsar) specific
int BF_n_channels_inner = 8;
int BF_n_time_inner = 2;
int BF_n_time_power_sum = 4;
int BF_n_beam = 512;
int BF_n_flags = 0;
double BF_beam_separation = 1.0;
double BF_beam_separation_ns = 0.75;

// Packet specific, hide these from the CLI...
int BF_n_antennae_per_packet = 3;
int BF_n_time_per_packet = 2;
int BF_n_channels_per_packet = 384;
// ...keep this exposed
QudaPacketFormat BF_packet_format = QUDA_PACKET_FORMAT_DSA110;

// Compute properties
QudaPrecision BF_compute_prec = QUDA_SINGLE_PRECISION;
QudaBeamformerType BF_type = QUDA_BEAMFORMER_VOLTAGE;

double BF_sfreq = 1498.75;
double BF_wfreq = 0.244140625;
double BF_declination = 71.66;

// Populate a parameter struct that describes the Beamformer
// Parameter structs are defined in generic_GPU_project/include/ggp.h.
// The relevant enums are defined in generic_GPU_project/include/enums_ggp.h.
// The print/check/new methods are in generic_GPU_project/lib/check_params.h.
void setBeamformerParam(BeamformerParam &bf_param) {

  // Deduce packet properties
  BF_n_antennae_per_packet = 3;
  BF_n_time_per_packet = 2;
  BF_n_channels_per_packet = 384;
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
  bf_param.n_beam = BF_n_beam;
  bf_param.n_arm = 2; // DSA110 has 2 arms (East-West and North-South)
  bf_param.n_polarizations = BF_n_polarizations;
  bf_param.n_flags = BF_n_flags;
  bf_param.beam_separation = BF_beam_separation;
  bf_param.beam_separation_ns = BF_beam_separation_ns;
  bf_param.sfreq = BF_sfreq;
  bf_param.wfreq = BF_wfreq;
  bf_param.declination = BF_declination;
  bf_param.compute_prec = BF_compute_prec;
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
    printfQuda("-- beam separation (arcmin) %f\n", BF_beam_separation);
    printfQuda("-- beam separation_ns (arcmin) %f\n", BF_beam_separation_ns);
    printfQuda("-- sfreq - upper frequency limit (MHz) %f\n", BF_sfreq);
    printfQuda("-- wfreq - frequency width (MHz) %f\n", BF_wfreq);
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

void computeWeightsByArm(void *weights_A, void *weights_B, std::vector<float> input_calibs, std::vector<int> flagants_in, BeamformerParam *bf_param) {

  // Extract variables.
  int n_payload = bf_param->n_payload;
  int n_ant = bf_param->n_antennae_per_payload;
  int n_chan = bf_param->n_channels_per_payload;
  int n_time = bf_param->n_time_per_payload;
  int n_time_inner = bf_param->n_time_inner;
  int n_time_power_sum = bf_param->n_time_power_sum;
  int n_chan_inner = bf_param->n_channels_inner;
  int n_beam = bf_param->n_beam;
  int n_pol = bf_param->n_polarizations;
  int n_flags = bf_param->n_flags;
  
  int n_calibs = (2 * 2 * n_ant * n_chan)/n_chan_inner + 2 * n_ant;
  int n_freq = n_chan/n_chan_inner;
  
  float sep = bf_param->beam_separation;
  float sep_ns = bf_param->beam_separation_ns;
  float dec = bf_param->declination;
  float sfreq = bf_param->sfreq;
  float wfreq = bf_param->wfreq;
  
  // Host side data
  std::vector<float> ant_pos_E(n_ant, 0.0);
  std::vector<float> ant_pos_N(n_ant, 0.0);
  std::vector<float> calibrations(n_calibs, 0.0);
  std::vector<float> frequencies(n_freq, 0.0);    
  std::vector<int> flagants(n_ant, 0);

  // deal with antpos and calibs
  int iant, found = 0;
  float norm = 0;
  for (int i=0; i<n_ant; i++) {
    ant_pos_E[i] = input_calibs[i];
    ant_pos_N[i] = input_calibs[n_ant + i];    
  }

  for (int i=0;i<2*n_ant*(n_chan/n_chan_inner);i++) {
      
    iant = i/(2*(n_chan/n_chan_inner));
    flagants[iant] = 0;
    
    found = 0;
    for (int j=0; j<n_flags; j++) {
      if (flagants_in[j] == iant) {
	found = 1;
	flagants[iant] = 1;
      }
    }
    
    calibrations[2*i]   = input_calibs[2*n_ant + 2*i];
    calibrations[2*i+1] = input_calibs[2*n_ant + 2*i + 1];
    
    norm = sqrt(calibrations[2*i]*calibrations[2*i] + calibrations[2*i+1]*calibrations[2*i+1]);
    if (norm != 0.0) {
      calibrations[2*i]   /= norm;
      calibrations[2*i+1] /= norm;
    }
      
    if (found == 1) {
      calibrations[2*i]   = 0.;
      calibrations[2*i+1] = 0.;
    }
  }
  
  for (int i=0; i<n_chan/n_chan_inner; i++) frequencies[i] = 1e6*(sfreq - i*wfreq);

  static const float PI = 3.14159265358979323846;

  int n_weights = 2 * (n_chan/n_chan_inner) * (n_beam/2) * (n_ant/2);

  // DMH: Loop this using explicit indicies to demonstrate
  // CPU correctness with OMP.
  for(int idx=0; idx<n_weights; idx++) {
    
    int iArm = (int)(idx / ((n_chan/n_chan_inner)*(n_beam/2)*(n_ant/2)));
    int j = (int)(idx % ((n_chan/n_chan_inner)*(n_beam/2)*(n_ant/2)));
    int fq = (int)(j / ((n_beam/2)*(n_ant/2)));
    int iidx = (int)(j % ((n_beam/2)*(n_ant/2)));
    int bm = (int)(iidx / (n_ant/2));
    int a = (int)(iidx % (n_ant/2));
    int widx = (a + (n_ant/2)*iArm)*(n_chan/n_chan_inner)*2*2 + fq*2*2;      
      
    // calculate weights
    float theta, afac, twr, twi;
    if (iArm==0) {
      theta = sep*(127.0 - bm*1.0)*PI/10800.0; // radians
      afac = -2.0*PI*frequencies[fq]*theta/CVAC; // factor for rotate
	
      twr = cosf(afac*ant_pos_E[a + (n_ant/2)*iArm]);
      twi = sinf(afac*ant_pos_E[a + (n_ant/2)*iArm]);
      
      ((float*)weights_A)[2*idx]   = (twr*calibrations[widx] - twi*calibrations[widx+1]);
      ((float*)weights_A)[2*idx+1] = (twi*calibrations[widx] + twr*calibrations[widx+1]);
      ((float*)weights_B)[2*idx]   = (twr*calibrations[widx+2] - twi*calibrations[widx+3]);
      ((float*)weights_B)[2*idx+1] = (twi*calibrations[widx+2] + twr*calibrations[widx+3]);
    } else {
      theta = sep_ns*(127.0 - bm*1.0)*PI/10800.0 - (PI/180.)*dec; // radians
      afac = -2.*PI*frequencies[fq]*sinf(theta)/CVAC; // factor for rotate
      
      twr = cosf(afac*ant_pos_N[a + (n_ant/2)*iArm]);
      twi = sinf(afac*ant_pos_N[a + (n_ant/2)*iArm]);
      
      ((float*)weights_A)[2*idx]   = (twr*calibrations[widx] - twi*calibrations[widx+1]);
      ((float*)weights_A)[2*idx+1] = (twi*calibrations[widx] + twr*calibrations[widx+1]);
      ((float*)weights_B)[2*idx]   = (twr*calibrations[widx+2] - twi*calibrations[widx+3]);
      ((float*)weights_B)[2*idx+1] = (twi*calibrations[widx+2] + twr*calibrations[widx+3]);
    }
    
    int low = ((n_chan/n_chan_inner)*(n_beam/2)*(n_ant/2))*0 - 512;
    int high= ((n_chan/n_chan_inner)*(n_beam/2)*(n_ant/2))*0 + 512;
    
    if(low <= idx && idx < high && true) {
      printf("CPU %d of %d (ggp): weights A = (%f,%f) weights B = (%f,%f)\n", idx, n_weights, ((float*)weights_A)[2*idx], ((float*)weights_A)[2*idx+1], ((float*)weights_B)[2*idx], ((float*)weights_B)[2*idx+1]);
    }
  }
}

double verifyVoltageBeamformer(void *output_data, void *input_data, std::vector<float> input_calibs, std::vector<int> flagants, BeamformerParam *bf_param) {

  double dev = 0;
  
  int n_payload = bf_param->n_payload;
  int n_ant = bf_param->n_antennae_per_payload;
  int n_chan = bf_param->n_channels_per_payload;
  int n_time = bf_param->n_time_per_payload;
  int n_time_inner = bf_param->n_time_inner;
  int n_time_power_sum = bf_param->n_time_power_sum;
  int n_chan_inner = bf_param->n_channels_inner;
  int n_beam = bf_param->n_beam;
  int n_pol = bf_param->n_polarizations;

  int sfreq = bf_param->sfreq;
  
  int n_calibs = (2 * n_ant * n_chan)/n_chan_inner;
  int n_freq = n_chan/n_chan_inner;
  int n_weights_per_pol = (n_freq * n_ant * n_beam)/2;
  int n_results_per_pol = (n_chan * n_beam * n_time);  
  int ib_sum_elems = n_time * n_chan;
  
  void *weights_A = malloc(2 * n_weights_per_pol * sizeof(float));
  void *weights_B = malloc(2 * n_weights_per_pol * sizeof(float));  

  // Reorder data by arm, emulate device to device shuffle.
  
  computeWeightsByArm(weights_A, weights_B, input_calibs, flagants, bf_param);

  //free(weights_A);
  //free(weights_B);
  
  // Initialise host side data
  //--------------------------
  uint64_t in_block_size = (n_ant * n_chan * n_pol * n_time);
  
  // Create host side input data
  long unsigned prom_input_size = 2 * in_block_size * sizeof(double);
  void *promoted_input_data_A = malloc(prom_input_size/2);
  void *promoted_input_data_B = malloc(prom_input_size/2);
  promote_to_float_by_pol_complex((double*)promoted_input_data_A, (double*)promoted_input_data_B, (unsigned char*)input_data, in_block_size);

  // Create host side output data
  long unsigned prom_output_size = (n_time/(n_time_power_sum*n_time_inner)*(n_chan/n_chan_inner)*n_beam) * sizeof(double);
  void *output_data_host = malloc(prom_output_size);
  prec_switch((double*)output_data_host, (unsigned char*)output_data, prom_output_size/sizeof(double));
  
  std::vector<complex<double>> weights_a(n_weights_per_pol, 0);
  std::vector<complex<double>> weights_b(n_weights_per_pol, 0);
  std::vector<complex<double>> result_a(n_results_per_pol, 0);
  std::vector<complex<double>> result_b(n_results_per_pol, 0);
  std::vector<double> ib_sum_data(ib_sum_elems, 0);

  // Copy float data to double
  for (int i=0; i<n_weights_per_pol; i++) {
    // DMH: FIX ME for arbitraty prec
    weights_a[i].real(((float*)weights_A)[2*i  ]);
    weights_a[i].imag(((float*)weights_A)[2*i+1]);
    weights_b[i].real(((float*)weights_B)[2*i  ]);
    weights_b[i].imag(((float*)weights_B)[2*i+1]);
  }
  
  // DMH: Add CL params to govern this
  //for (uint64_t i=0; i<n_chan/n_chan_inner; i++) frequencies[i] = 1e6*(sfreq - i*250.0/1024.0);
  //for (uint64_t i=0; i<n_freq; i++)
  //freqs_dbl[i] = 1e6 * (sfreq - i*250.0/1024.0);
  
  printfQuda("n_freq = %d\n", n_freq);
  printfQuda("n_ant = %d\n", n_ant);
  printfQuda("n_calibs = %d\n", n_calibs);
  printfQuda("n_weights_per_pol = (n_freq * n_ant * n_beam) = (%d * %d * %d)/2 = %d\n", n_freq, n_ant, n_beam, n_weights_per_pol);
  printfQuda("n_results_per_pol = n_chan * n_beam * n_time = %d * %d * %d = %d\n", n_results_per_pol, n_chan, n_beam, n_time);    
  //--------------------------
  
  // Problem parameters
  //-------------------
  int m = n_beam/2;
  int n = n_chan_inner * n_time;
  int k = n_ant/2;
  int batches = n_chan / n_chan_inner;

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
  //double dev = 0.0;

  complex<double> *V_a_ptr = (complex<double> *)(&promoted_input_data_A)[0];
  complex<double> *V_b_ptr = (complex<double> *)(&promoted_input_data_B)[0];

  /*
  
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
  //...
  
  free(output_data_host);
  free(promoted_input_data_A);
  free(promoted_input_data_B);

  */  
  return dev;
}

#if 0
double verifyVisibilityBeamformer(void *output_data, void *input_data, void* weights_A, void* weights_B, std::vector<int> flagants, BeamformerParam &bf_param) {

  // Initialise host side data
  //--------------------------
  uint64_t in_block_size = (BF_n_antennae_per_payload *
			    BF_n_channels_per_payload *
			    BF_n_polarizations*
			    BF_n_time_per_payload);
			    
  // Create host side input data
  long unsigned prom_input_size = 2 * in_block_size * sizeof(double);
  void *promoted_input_data = malloc(prom_input_size);
  promote_to_float_complex((double*)promoted_input_data, (unsigned char*)input_data, in_block_size);

  // Calculate frequencies                                                                                                                                                                                        
  std::vector<double> freqs(NCHAN);
  for (int i = 0; i < NCHAN; ++i) {
    freqs[i] = 2e9 - static_cast<double>(i) * 2e7;
  }
  //const double lambda_max = CVAC / freqs.back();  

  // Create host side output data
  long unsigned prom_output_size = (BF_n_time_per_payload/(BF_n_time_power_sum*BF_n_time_inner)*(BF_n_channels_per_packet/BF_n_channels_inner)*BF_n_beam) * sizeof(double);
  void *output_data_host = malloc(prom_output_size);
  prec_switch((double*)output_data_host, (unsigned char*)output_data, prom_output_size/sizeof(double));
    
  uint64_t n_freq = bf_param.n_channels_per_payload/bf_param.n_channels_inner;  
  uint64_t n_ant = bf_param.n_antennae_per_payload;
  uint64_t n_calib = bf_param.n_antennae_per_payload * n_freq * bf_param.n_polarizations;
  uint64_t n_weights_per_pol = n_freq * bf_param.n_beam * bf_param.n_antennae_per_payload;
  uint64_t n_results_per_pol = 2 * bf_param.n_channels_per_payload * bf_param.n_beam * bf_param.n_time_per_payload;
  uint64_t ib_sum_elems = bf_param.n_channels_per_payload * bf_param.n_time_per_payload;
  
  //std::vector<int> flagants(bf_param.n_antennae_per_payload, 1);
  std::vector<double> calib_w(2 * n_calib);
  std::vector<double> position(2 * n_ant);
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
  printfQuda("n_ant = %ld\n", n_ant);
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

double BeamformerTest(test_t test_param) {

  BeamformerParam bf_param = newBeamformerParam();
  BF_compute_prec = ::testing::get<0>(test_param);
  setBeamformerParam(bf_param);
  display_test_info();

  int n_payload = bf_param.n_payload;
  int n_ant = bf_param.n_antennae_per_payload;
  int n_chan = bf_param.n_channels_per_payload;
  int n_time = bf_param.n_time_per_payload;
  int n_time_inner = bf_param.n_time_inner;
  int n_time_power_sum = bf_param.n_time_power_sum;
  int n_chan_inner = bf_param.n_channels_inner;
  int n_beam = bf_param.n_beam;
  int n_pol = bf_param.n_polarizations;
  int n_flags = bf_param.n_flags;
  double sfreq = bf_param.sfreq;
  double wfreq = bf_param.wfreq;  
  double declination = bf_param.declination;

  // This is the number of calibrations (2 * 2 * n_ant * n_chan)/n_chan_inner)
  // plus 2 * n_ant positions
  int n_calibs = (2 * 2 * n_ant * n_chan)/n_chan_inner + 2 * n_ant;
  int n_freq = n_chan/n_chan_inner;
  uint64_t n_weights_per_pol = n_freq * n_ant * n_beam * n_time_inner;  
  
  // Create a raw voltage data array for a single call to the Beamformer.
  // One (4,4)b piece of data fits into one char.
  uint64_t in_payload_size;
  in_payload_size = (n_time * n_ant * n_chan * n_pol);
  
  logQuda(QUDA_VERBOSE, "Creating input array of size %f MB\n", (1.0*sizeof(char)*n_payload*in_payload_size)/pow(1000,2));
  logQuda(QUDA_VERBOSE, "Input (4bit,4bit) complex elements %lu\n", n_payload*in_payload_size);
  
  void *input_data = pinned_malloc(n_payload * in_payload_size);
  int n_ints = (n_payload * in_payload_size)/sizeof(int); 
  int *p = (int*)input_data;
  unsigned char *p_uc = (unsigned char*)input_data;
  for(int i = 0; i < n_ints; i++) p[i] = rand();

  // DMH: Remove when done
#if 0
  // Inspect incoming packet
  for(int pidx = 0; pidx<1024; pidx++) {
    printf("Raw Voltage %d: (%f,%f)\n", pidx,
	   (float)((char)(((p_uc[pidx]) & (unsigned char)(15)) << 4) >> 4),
	   (float)((char)(((p_uc[pidx]) & (unsigned char)(240))) >> 4));
  }
#endif
  
  // Create an output array for the Beamformer
  uint64_t out_payload_size = (n_time/(n_time_power_sum * n_time_inner)) * (n_chan/n_chan_inner) * n_beam;
  uint64_t output_size = out_payload_size * n_payload;  
  logQuda(QUDA_VERBOSE, "Creating output_array of size %f MiB\n", (1.0*output_size)/pow(1000,2));
  logQuda(QUDA_VERBOSE, "Output (unsigned char) real elements %lu\n", n_payload * out_payload_size);
  void *output_data = pinned_malloc(output_size);
  
  std::vector<float> calibrations(n_calibs);
  std::vector<int> flagants;

  // Emulate offline antennae
  for(int i=0; i<n_ant; i++) {
    if(i<n_flags) flagants.push_back(i);
  }

  // Generate calibrations
  // 2 * n_ant for antennae positions +
  // 2(complex) * n_ant * 2(arms) * channel bands
  for(int i=0; i<n_calibs; i++) {
    calibrations[i] = 1.0/(i+1);
  }
  
  // Input array is interpreted as a row major array of matrices,
  double dev = 0.0;  
  switch(bf_param.type) {
  case QUDA_BEAMFORMER_VOLTAGE:    
    beamformerVoltageDsa110CHPC(output_data, input_data, calibrations.data(), flagants.data(), &bf_param);
    if(verify_results) dev = verifyVoltageBeamformer(output_data, input_data, calibrations, flagants, &bf_param);
    break;
  case QUDA_BEAMFORMER_VISIBILITY:
    //beamformer_visibility(output_data, rand_data, freqs.data(), weights_A, weights_B, flagants.data(), &bf_param);
    //if(verify_results) dev = verifyVisibilityBeamformer(output_data, rand_data, freqs.data(), weights_A, weights_B, flagants.data(), bf_param);
    errorQuda("Unsupported beamformer type %d", bf_param.type);
    break;
  default: errorQuda("Unknown beamformer type %d", bf_param.type);
  }
  
  host_free(input_data);
  host_free(output_data);
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
    
    CLI::GGPCheckedTransformer prec_transform(precision_map);
    CLI::GGPCheckedTransformer packet_transform(packet_map);
    CLI::GGPCheckedTransformer bf_transform(bf_map);
    
    auto opgroup = app->add_option_group("Beamformer", "Options controling the Beamformer test parameters");
    opgroup->add_option("--BF-type", BF_type, "Type of beamformer to compute (default voltage)")->transform(bf_transform);
    opgroup->add_option("--BF-n-payload", BF_n_payload, "Number of payloads to move to device (default 1)");
    opgroup->add_option("--BF-compute-prec", BF_compute_prec, "Compute and output precision (default float)")->transform(prec_transform);
    opgroup->add_option("--BF-n-antennae-per-payload", BF_n_antennae_per_payload, "Number of antennae producing data per payload(default 96)");
    opgroup->add_option("--BF-n-channels-per-payload", BF_n_channels_per_payload, "Number of frequency channels per payload of data (default 384)");
    opgroup->add_option("--BF-n-time-per-payload", BF_n_time_per_payload, "Number of time stamps per payload (default 2048)");
    opgroup->add_option("--BF-n-channels-inner", BF_n_channels_inner, "Number of frequency channels over which to sum (default 8)");
    opgroup->add_option("--BF-n-time-inner", BF_n_time_inner, "Number of fine time steps to sum over in channel sum (default 2)");
    opgroup->add_option("--BF-n-time-power-sum", BF_n_time_power_sum, "Number of time steps to sum over power sum (default 4)");
    
    opgroup->add_option("--BF-n-beam", BF_n_beam, "Number of beams (default 512)");
    opgroup->add_option("--BF-beam-separation", BF_beam_separation, "Beams separation in arcmin (default 1.0)");
    opgroup->add_option("--BF-beam-separation-ns", BF_beam_separation_ns, "Beams separation_ns in arcmin (default 0.75)");
    opgroup->add_option("--BF-sfreq", BF_sfreq, "Highest frequency band upper limit in MHz (default 1498.75)");
    opgroup->add_option("--BF-wfreq", BF_wfreq, "Width of frequency bands beloe sfreq MHz (default 0.244140625)");  
    opgroup->add_option("--BF-declination", BF_declination, "The pointing declination (default 71.66)");
    opgroup->add_option("--BF-n-flags", BF_n_flags, "Number of flagged antennae (default 0)");  
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

