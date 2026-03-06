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
#include <xengine_test_gtest.hpp>
#include <eigen_helper.h>

namespace quda {
  extern void setTransferGPU(bool);
}

// Comment out the below for test routine debugging
//#define HOST_DEBUG

template <typename T> using complex = std::complex<T>;

// XEngine options
//----------------------
uint64_t XE_n_polarizations = 2; // Very unlikely to change.

// Payload dims
uint64_t XE_n_payload = 1;
uint64_t XE_n_packets_per_payload = 1;
uint64_t XE_n_antennae_per_payload = 2048;
uint64_t XE_n_time_per_payload = 1024;
uint64_t XE_n_channels_per_payload = 4;
uint64_t XE_n_base = ((XE_n_antennae_per_payload + 1) * XE_n_antennae_per_payload)/2;
// Computation (FRB, Pulsar) specific
uint64_t XE_n_time_inner = 2;

// Packet specific, hide these from the CLI...
uint64_t XE_n_antennae_per_packet = 1;
uint64_t XE_n_time_per_packet = 1024;
uint64_t XE_n_channels_per_packet = 4;
// ...keep this exposed
QudaPacketFormat XE_packet_format = QUDA_PACKET_FORMAT_DSA2K;

// Compute properties
QudaPrecision XE_compute_prec = QUDA_SINGLE_PRECISION;
QudaPrecision XE_output_prec = QUDA_SINGLE_PRECISION;
QudaXEngineMatFormat XE_mat_format = QUDA_XENGINE_MAT_HERM;
QudaBLASEngine XE_engine_type = QUDA_BLAS_ENGINE_CUBLAS;

// Populate a parameter struct that describes the XEngine
// Parameter structs are defined in generic_GPU_project/include/ggp.h.
// The relevant enums are defined in generic_GPU_project/include/enums_ggp.h.
// The print/check/new methods are in generic_GPU_project/lib/check_params.h.
void setXEngineParam(XEngineParam &xe_param) {

  // Deduce packet properties
  if(XE_packet_format == QUDA_PACKET_FORMAT_DSA110) {
    XE_n_antennae_per_packet = 3;
    XE_n_time_per_packet = 2;
    XE_n_channels_per_packet = 384;
  } else if (XE_packet_format == QUDA_PACKET_FORMAT_DSA2K) {
    XE_n_antennae_per_packet = 1;
    XE_n_time_per_packet = 16;
    XE_n_channels_per_packet = 4;
  } else {
    errorQuda("Unknown packet format %d", XE_packet_format);
  }    
  printfQuda("Packet size is %d bytes\n", XE_packet_format);
  
  // Compute payload size for this computation
  //------------------------------------------
  // Check that we have correct number of antennae number for a payload
  if(XE_n_antennae_per_payload % XE_n_antennae_per_packet != 0) {
    errorQuda("XE_n_antennae_per_payload(%lu) %% XE_n_antennae_per_packet(%lu) = %lu. Ensure this is zero.",
	      XE_n_antennae_per_payload, XE_n_antennae_per_packet, XE_n_antennae_per_payload % XE_n_antennae_per_packet);
  }

  // Input payload size (always char)
  uint64_t payload_size = (XE_n_polarizations*
			   XE_n_antennae_per_payload*
			   XE_n_channels_per_payload*
			   XE_n_time_per_payload);
  
  printfQuda("payload size = %lu bytes\n", payload_size);
  if(payload_size % (int)XE_packet_format != 0) {
    errorQuda("Incorrect payload size %lu for given packet format %d", payload_size, XE_packet_format);    
  }
  printfQuda("Running XEngine with %lu packets per %lu payload(s)\n", payload_size/((int)XE_packet_format), XE_n_payload);

  xe_param.packet_format = XE_packet_format;
  xe_param.n_payload = XE_n_payload;
  xe_param.n_antennae_per_payload = XE_n_antennae_per_payload;
  xe_param.n_channels_per_payload = XE_n_channels_per_payload;
  xe_param.n_time_per_payload = XE_n_time_per_payload;
  xe_param.n_time_inner = XE_n_time_inner;
  xe_param.n_polarizations = XE_n_polarizations;
  xe_param.compute_prec = XE_compute_prec;
  XE_output_prec = XE_compute_prec;
  xe_param.output_prec = XE_output_prec;
  xe_param.data_type = QUDA_BLAS_DATATYPE_QC;
  xe_param.data_order = QUDA_BLAS_DATAORDER_ROW;
  xe_param.in_location = QUDA_CPU_FIELD_LOCATION;
  xe_param.out_location = QUDA_CPU_FIELD_LOCATION;
  xe_param.verbosity = verbosity;
  xe_param.format = XE_mat_format;
  if(xe_param.format == QUDA_XENGINE_MAT_HERM) {
    XE_n_base = XE_n_antennae_per_payload * XE_n_antennae_per_payload;
  } else {
    XE_n_base = ((XE_n_antennae_per_payload + 1)*(XE_n_antennae_per_payload))/2;
  }
  xe_param.engine = XE_engine_type;
  xe_param.struct_size = sizeof(xe_param);
}

void display_test_info() {
  if (getVerbosity() > QUDA_SILENT) {  
    printfQuda("running the following test:\n");
    printfQuda("XEngine test\n");
    printfQuda("Grid partition info:     X  Y  Z  T\n");
    printfQuda("                         %d  %d  %d  %d\n", dimPartitioned(0), dimPartitioned(1), dimPartitioned(2),
             dimPartitioned(3));

    printfQuda("XEngine parameters\n");
    printfQuda("-- Packet format %s\n", get_packet_format_str(XE_packet_format));
    printfQuda("-- N payloads %lu\n", XE_n_payload);
    printfQuda("-- N antennae per payload %lu\n", XE_n_antennae_per_payload);
    printfQuda("-- N time per payload %lu\n", XE_n_time_per_payload);
    printfQuda("-- N channels per payload %lu\n", XE_n_channels_per_payload);
    printfQuda("-- N time inner %lu\n", XE_n_time_inner);
    printfQuda("-- N polarizations %lu\n", XE_n_polarizations);
    printfQuda("-- Compute precision %s\n", get_prec_str(XE_compute_prec));
    printfQuda("-- Output precision %s\n", get_prec_str(XE_output_prec));
    printfQuda("-- Output format %s\n", get_xengine_mat_format_str(XE_mat_format));
  }
}
//----------------------

// Verification routines
//----------------------
void fillEigenArray(MatrixXcd &EigenArr, complex<double> *arr, uint64_t rows, uint64_t cols, uint64_t offset, bool tri = false) {
  uint64_t counter = offset;
  for (uint64_t i = 0; i < rows; i++) {
    for (uint64_t j = (tri ? i : 0); j < cols; j++) {
      EigenArr(i, j) = arr[counter];
      counter++;
    }
  }
}

#define MASK1 0x0F // (00001111)
#define MASK2 0xF0 // (11110000)

template <typename Float> void promote_to_float(Float *output, unsigned char *input_4c4, uint64_t elems) {
  for(uint64_t i=0; i<elems; i++) {
    output[2*i  ] = (Float)((char)(((input_4c4[i]) & (unsigned char)(MASK1)) << 4) >> 4);
    output[2*i+1] = (Float)((char)(((input_4c4[i]) & (unsigned char)(MASK2))) >> 4);
  }
}

template <typename FloatOut, typename FloatIn> void prec_switch(FloatOut *output, FloatIn *input, uint64_t elems) {  
  for(uint64_t i=0; i<elems; i++) {
    //printfQuda("prec switch %lu of %lu\n", i, elems);
    output[i] = input[i];
  }
}

void reorder_data(double *out_data, unsigned char *in_data, uint64_t in_payload, XEngineParam &xe_param) {

  int time_inner = 2;
  int n_chan = 384;
  int n_ant = 96;
  int n_time = 2048;
  int n_pol = 2;
  
  for(uint64_t idx=0; idx<(xe_param.n_time_per_payload/time_inner) * xe_param.n_channels_per_payload * xe_param.n_antennae_per_payload; idx++) {
    // Get 110 indices;
    int chan = idx % n_chan;
    int ant = (idx / n_chan) % n_ant;
    int time = idx / (n_chan * n_ant); 

    int chan_2k = chan * (time_inner * n_pol * n_time * n_ant);
    int ant_2k  = ant;
    int time_2k = time * n_ant;
    // This is the start of the 4 entries for time_inner and pol data for 2k
    int idx_2k = chan_2k + ant_2k + time_2k;
    // This is the start of the 4 entries for time_inner and pol data for 110
    int idx_110 = idx * n_pol * time_inner;
      
    int t, pol, inner;
    
    for(t=0; t<time_inner; t++) {
      for(pol=0; pol<n_pol; pol++) {
	
	inner = t*n_pol + pol;
	
	out_data[2*(idx_2k + inner)]     = (double)((char)(((unsigned char)(in_data[idx_110 + inner]) & (unsigned char)(MASK1)) << 4) >> 4);
	out_data[2*(idx_2k + inner) + 1] = (double)((char)(((unsigned char)(in_data[idx_110 + inner]) & (unsigned char)(MASK2))) >> 4);
      }
    }
  }
  
}

double verifyXEngine(void *output_data, void *rand_data, XEngineParam &xe_param) {
  
  uint64_t in_payload_size = (XE_n_antennae_per_payload *
			      XE_n_channels_per_payload *
			      XE_n_polarizations *
			      XE_n_time_per_payload);

  // Create host side input data
  long unsigned prom_input_size = 2 * in_payload_size * sizeof(double);
  void *promoted_input_data = malloc(prom_input_size);
  // If launching with dsa110 ordered data, reorder to dsa2k
  if(xe_param.packet_format == QUDA_PACKET_FORMAT_DSA110)
    reorder_data((double*)promoted_input_data, (unsigned char*)rand_data, in_payload_size, xe_param);
  else 
    promote_to_float((double*)promoted_input_data, (unsigned char*)rand_data, in_payload_size);
  
  // Create host side output data
  long unsigned prom_output_size = 2 * XE_n_base * XE_n_channels_per_payload * XE_n_polarizations * XE_n_time_inner * sizeof(double);
  void *output_data_host = malloc(prom_output_size);
  memset(output_data_host, 0, prom_output_size);

  // Copy GPU computed data to host array
  void *output_data_copy = malloc(prom_output_size);
  if(XE_output_prec == QUDA_SINGLE_PRECISION) {
    prec_switch((double*)output_data_copy, (float*)output_data, 2 * XE_n_base * XE_n_channels_per_payload * XE_n_polarizations * XE_n_time_inner);
  } else if(XE_output_prec == QUDA_HALF_PRECISION) {
    prec_switch((double*)output_data_copy, (half*)output_data, 2 * XE_n_base * XE_n_channels_per_payload * XE_n_polarizations * XE_n_time_inner);
  } else {
    memcpy(output_data_copy, output_data, prom_output_size);
  }

  
  // Problem parameters
  int m = xe_param.n_antennae_per_payload;
  int n = xe_param.n_antennae_per_payload;
  int k = xe_param.n_time_per_payload/xe_param.n_time_inner;    
  int batches = xe_param.n_channels_per_payload * xe_param.n_polarizations * xe_param.n_time_inner;
  
  // Eigen objects
  MatrixXcd M = MatrixXd::Zero(k, m);
  MatrixXcd C_host = MatrixXd::Zero(m, n);
  MatrixXcd C_host_upper = MatrixXd::Zero(m, n);
  MatrixXcd C_device = MatrixXd::Zero(m, n);
  MatrixXcd C_resid = MatrixXd::Zero(m, n);

  // Pointers to data
  complex<double> *M_ptr = (complex<double> *)(&promoted_input_data)[0];
  complex<double> *C_dev_ptr = (complex<double> *)(&output_data_copy)[0];
    
  int M_offset = 0;
  int C_offset = 0;
  double dev = 0.0;
  for (int batch = 0; batch < batches; batch++) {

    // Ensure all arrays are zeroed out.
    M.setZero();
    C_host.setZero();
    C_host_upper.setZero();
    C_device.setZero();
    C_resid.setZero();
    
    // Populate Eigen objects
    fillEigenArray(M, M_ptr, k, m, M_offset);
    fillEigenArray(C_device, C_dev_ptr, m, n, C_offset, xe_param.format == QUDA_XENGINE_MAT_TRI ? true : false);
    
    // Perform GEMM using Eigen
    C_host = M.adjoint() * M;

#if (defined HOST_DEBUG)
    std::cout << "batch " << batch << std::endl << "M" << std::endl << M << std::endl << "C_host" << std::endl << C_host << std::endl << "C_device" << std::endl << C_device << std::endl;
#endif
    
    M_offset += m * k;

    // Check Eigen result against device result
    switch(xe_param.format) {
    case QUDA_XENGINE_MAT_TRI:
      // Simply test element by element.
      C_host_upper = C_host.template triangularView<Eigen::Upper>();
      C_resid = C_device - C_host_upper;
      C_offset += ((m+1) * m)/2;
      break;
    case QUDA_XENGINE_MAT_HERM:
      // Test for hermiticity, element by element.
      C_resid = C_device.adjoint() - C_host;
      C_offset += n * m;
      break;
    default:
      errorQuda("Unknown XEngine mat format %d", xe_param.format);
    }

    double deviation = C_resid.norm();
    double relative_deviation = deviation / C_host.norm();      
    logQuda(QUDA_VERBOSE, "batch %d: (C_device - C_host) Frobenius norm = %e. Relative deviation = %e", batch, deviation,
	    relative_deviation);
    if(deviation != 0.0) {
      logQuda(QUDA_VERBOSE, ": ERROR!\n");
      dev += deviation;
    } else {
      logQuda(QUDA_VERBOSE, ": PASS\n");
    }
  }
  
  free(output_data_copy);
  free(output_data_host);
  free(promoted_input_data);

  return dev;
}
//----------------------

double XEngineTest(test_t test_param) {

  XEngineParam xengine_param = newXEngineParam();
  XE_mat_format = ::testing::get<0>(test_param);
  XE_compute_prec = ::testing::get<1>(test_param);
  setXEngineParam(xengine_param);
  display_test_info();

  // Create a data array for a single call to the XEngine
  // one (4,4)b piece of data fits into one char.
  uint64_t in_payload_size;
  in_payload_size = (XE_n_antennae_per_payload *
		     XE_n_channels_per_payload *
		     XE_n_time_per_payload *
		     XE_n_polarizations);
  
  logQuda(QUDA_VERBOSE, "Creating input array of size %f MB\n", (1.0*sizeof(char)*XE_n_payload*in_payload_size)/pow(1000,2));
  logQuda(QUDA_VERBOSE, "Input (4bit,4bit) complex elements %lu\n", XE_n_payload * in_payload_size);

  // Populate input array with random bits. These 32 bit ints will
  // be interpreted as (4bit,4bit) signed complex pairs, eg
  // 01110100101110000101100001101001 = 1,958,238,313
  // becomes
  // (0111,0100) (1011,1000) (0101,1000) (0110,1001) = (7,4) (-3,-8) (5,-8) (6,1)
  void *rand_data = pinned_malloc(XE_n_payload * in_payload_size);  
  uint64_t n_rand = (XE_n_payload * in_payload_size)/sizeof(int);
  int *p = (int*)rand_data;
  for (uint64_t i = 0; i < n_rand; i++) p[i] = rand();
  
  // Create an output array for the XEngine
  uint64_t output_data_size = 0;
  switch(XE_output_prec) {
  case QUDA_DOUBLE_PRECISION: output_data_size = sizeof(double); break;
  case QUDA_SINGLE_PRECISION: output_data_size = sizeof(float); break;
    //case QUDA_HALF_PRECISION: output_data_size = sizeof(short); break;
  default:
    errorQuda("Unsupported XEngine output precision %d", XE_output_prec);
  }

  uint64_t out_payload_size = output_data_size * 2ULL * XE_n_base * XE_n_channels_per_payload * XE_n_polarizations * XE_n_time_inner;
  uint64_t output_size = XE_n_payload * out_payload_size;
  logQuda(QUDA_VERBOSE, "Creating output_array of size %f MB\n", (1.0*output_size)/pow(1000,2));
  logQuda(QUDA_VERBOSE, "Output (%s) elements %lu\n", get_prec_str(XE_output_prec), (XE_n_payload * out_payload_size)/output_data_size);
  void *output_data = pinned_malloc(output_size);
  
  // Input array is interpreted as a row major array of matrices,
  // (XE_n_antennae,XE_n_packets_per_block). The output is
  // an array of hermitian (XE_n_antennae * XE_n_antennae)
  // or triangular ((XE_n_antennae +1)*(XE_n_antennae))/2 complex matrices, 
  XEngineCHPC((void*)output_data, (void*)rand_data, &xengine_param);
  
  double dev = 0.0;
  // Ouput data is already in CPU processable format. Promote input data to
  // CPU processable format and compute GEMM on CPU, crosscheck with GPU result
  // element by element.
  if(verify_results) dev = verifyXEngine(output_data, rand_data, xengine_param);
  
  host_free(rand_data);
  host_free(output_data);    
  return dev;
}


struct xengine_test : quda_test {

  void add_command_line_group(std::shared_ptr<GGPApp> app) const override
  {
    quda_test::add_command_line_group(app);

    CLI::TransformPairs<QudaXEngineMatFormat> mat_type_map {{"herm", QUDA_XENGINE_MAT_HERM},
							    {"tri", QUDA_XENGINE_MAT_TRI}};

    CLI::TransformPairs<QudaPacketFormat> packet_map {{"dsa110", QUDA_PACKET_FORMAT_DSA110},
						      {"dsa2k", QUDA_PACKET_FORMAT_DSA2K}};
    
    CLI::TransformPairs<QudaPrecision> precision_map {{"double", QUDA_DOUBLE_PRECISION},
						      {"single", QUDA_SINGLE_PRECISION},
						      {"half", QUDA_HALF_PRECISION},
						      {"quarter", QUDA_QUARTER_PRECISION}};

    CLI::GGPCheckedTransformer mat_transform(mat_type_map);   
    CLI::GGPCheckedTransformer packet_transform(packet_map);   
    CLI::GGPCheckedTransformer prec_transform(precision_map);
    
    CLI::TransformPairs<QudaBLASEngine> engine_map {{"cublas", QUDA_BLAS_ENGINE_CUBLAS},
                                                    {"cutlass", QUDA_BLAS_ENGINE_CUTLASS},
                                                    {"tcc", QUDA_BLAS_ENGINE_TCC}};
    CLI::GGPCheckedTransformer engine_transform(engine_map);

    auto opgroup = app->add_option_group("XEngine", "Options controling the XEngine test parameters");
    opgroup->add_option("--XE-engine", XE_engine_type, "BLAS engine to use (cublas|cutlass|tcc, default cublas)")->transform(engine_transform);
    opgroup->add_option("--XE-packet-format", XE_packet_format, "Type of data packet (default dsa2k)")->transform(packet_transform);
    opgroup->add_option("--XE-n-payload", XE_n_payload, "Number of payloads to move to device (default 1)");    
    opgroup->add_option("--XE-compute-prec", XE_compute_prec, "Compute precision (default float)")->transform(prec_transform);
    opgroup->add_option("--XE-output-prec", XE_output_prec, "Output precision (default float)")->transform(prec_transform);
    opgroup->add_option("--XE-n-antennae-per-payload", XE_n_antennae_per_payload, "Number of antennae producing data per payload (default 96)");
    opgroup->add_option("--XE-n-channels-per-payload", XE_n_channels_per_payload, "Number of frequency channels per payload of data (default 384)");
    opgroup->add_option("--XE-n-polarizations", XE_n_polarizations, "Number of polarizations (default 2)");
    opgroup->add_option("--XE-n-time-per-payload", XE_n_time_per_payload, "Number of time steps per payload (default 1024)");
    opgroup->add_option("--XE-n-time-inner", XE_n_time_inner, "Number of time steps to sum over (default 2)");
    opgroup->add_option("--XE-mat-format", XE_mat_format, "Output format of XEngine (default full hermitian [herm])")->transform(mat_transform);
  }
  
  xengine_test(int argc, char **argv) : quda_test("XEngine Test", argc, argv) { }
};

int main(int argc, char **argv) {

  xengine_test test(argc, argv);
  test.init();
  
  int result = 0;
  if (enable_testing) {
    // Perform the tests defined in xengine_test_gtest.hpp.
    result = test.execute();
    if (result) warningQuda("Google tests for XEngine failed.");
  } else {
    // Perform the test specified by the command line.
    XEngineTest(test_t {XE_mat_format, XE_compute_prec});
  }
  
  return result;
}
