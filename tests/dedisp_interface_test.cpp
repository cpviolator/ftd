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
#include <dedisp_interface_gtest.hpp>
#include <eigen_helper.h>

namespace quda {
  extern void setTransferGPU(bool);
}

// Comment out the below for test routine debugging
//#define HOST_DEBUG

QudaPrecision DEDISP_compute_prec = QUDA_SINGLE_PRECISION;
QudaPrecision DEDISP_storage_prec = QUDA_SINGLE_PRECISION;

template <typename T> using complex = std::complex<T>;

void display_test_info() {
  if (getVerbosity() > QUDA_SILENT) {  
  }
}

// Assume input is a 0 mean float and quantize to an unsigned 8-bit quantity
char bytequant(float f) {
  
  float v = f + 127.5f;
  char r;
  if (v > 255.0) { 
    r = (char)255; 
  } else if (v<0.0f) {
    r = (char)0; 
  } else {
    r = (char)roundf(v);
  }
  //printf("ROUND %f, %u\n",f,r);
  return r;
}

// Compute mean and standard deviation of an unsigned 8-bit array
void calc_stats_8bit(char *a, uint64_t n, float *mean, float *sigma)
{
  // Use doubles to prevent rounding error
  double sum=0.0, sum2=0.0;
  double mtmp=0.0, vartmp;
  double v;
  uint64_t i;

  // Use corrected 2-pass algorithm from Numerical Recipes
  sum = 0.0;
  for (i=0;i<n;i++) {
    v = (double)a[i];
    sum += v;
  }
  mtmp = sum/n;

  sum = 0.0;
  sum2 = 0.0;
  for (i=0;i<n;i++) {
    v = (double)a[i];
    sum2 += (v-mtmp)*(v-mtmp);
    sum += v-mtmp;
  }
  vartmp = (sum2-(sum*sum)/n)/(n-1);
  *mean = mtmp;
  *sigma = sqrt(vartmp);

  return;
}

// Compute mean and standard deviation of a float array
void calc_stats_float(float *a, uint64_t n, float *mean, float *sigma) {
  
  // Use doubles to prevent rounding error
  double sum = 0.0, sum2 = 0.0;
  double mtmp=0.0, vartmp;
  double v;
  uint64_t i;

  // Use corrected 2-pass algorithm from Numerical Recipes
  sum = 0.0;
  for (i=0; i<n; i++) {
    sum += a[i];
  }
  mtmp = sum/n;

  sum = 0.0;
  sum2 = 0.0;
  for (i=0; i<n; i++) {
    v = a[i];
    sum2 += (v-mtmp)*(v-mtmp);
    sum += v-mtmp;
  }
  vartmp = (sum2-(sum*sum)/n)/(n-1);
  *mean = mtmp;
  *sigma = sqrt(vartmp);

  return;
}

void setDedispParam(DedispParam &param) {

  param.location = QUDA_CPU_FIELD_LOCATION;
  param.in_nbits = 8;
  param.out_nbits = 32;
  param.in_stride = 0;
  param.out_stride = 0;
  param.n_time = 0;
  param.n_time_dd = 0;
  param.n_time_out = 0;
  param.n_boxcar = 0;
  param.n_time_dedisp = 0;
  param.minDM = 0;
  param.maxDM = 0;
  param.max_width = 0;
  param.tolerance = 0;
  param.DMs = 0;
  param.ndms = 0;
  param.gulp = 0;
  param.verbosity = QUDA_VERBOSE;
  param.dedisp_lib = CHPC_DEDISP_DEDISPERSION_LIB;
  param.struct_size = sizeof(param);
}

double DedispTest(test_t test_param) {

  float sampletime_base = 250.0E-6; // Base is 250 microsecond time samples
  float downsamp    = 1.0;
  float Tobs        = 30.0;    // Observation duration in seconds
  float dt          = downsamp*sampletime_base;     // s (0.25 ms sampling)
  float f0          = 1581.0;    // MHz (highest channel!)
  float bw          = 100.0; // MHz
  uint64_t nchans   = 1024;
  float df          = -1.0*bw/nchans;   // MHz   (This must be negative!)

  uint64_t nsamps   = Tobs / dt;
  float datarms     = 25.0;
  float sigDM       = 41.159; 
  float sigT        = 3.14159; // seconds into time series (at f0)
  float sigamp      = 25.0; // amplitude of signal

  float dm_start    = 2.0;    // pc cm^-3
  float dm_end      = 100.0;    // pc cm^-3
  float pulse_width = 4.0;   // ms
  float dm_tol      = 1.25;
  size_t in_nbits   = 8;
  size_t out_nbits  = 32;  // DON'T CHANGE THIS FROM 32, since that signals it to use floats
  const float DM_factor = 4.15e3;
  
  size_t dm_count;
  size_t max_delay;
  size_t nsamps_computed;
  void *input = 0;
  float *output = 0;

  DedispParam dedisp_param = newDedispParam();
  setDedispParam(dedisp_param);
  display_test_info(); 
  
  printf("----------------------------- INPUT DATA ---------------------------------\n");
  printf("Frequency of highest chanel (MHz)            : %.4f\n",f0);
  printf("Bandwidth (MHz)                              : %.2f\n",bw);
  printf("NCHANS (Channel Width [MHz])                 : %lu (%f)\n",nchans,df);
  printf("Sample time (after downsampling by %.0f)        : %f\n",downsamp,dt);
  printf("Observation duration (s)                     : %f (%lu samples)\n",Tobs,nsamps);
  printf("Data RMS (%2lu bit input data)                 : %f\n",in_nbits,datarms);
  printf("Input data array size                        : %lu MB\n",(nsamps*nchans*sizeof(float))/(1<<20));
  printf("\n");

  printf("nsamps = %lu, nchans = %lu\n", nsamps, nchans);

  // create raw background data
  void *filterbank = malloc(nsamps * nchans * sizeof(float));
  float *fb_p = (float*)filterbank;
  for (uint64_t j=0; j<nsamps * nchans; j++) {
    int rand_int = rand();
    rand_int -= (INT_MAX/2);
    fb_p[j] = (datarms*rand_int)/INT_MAX;
  }

  // create dispersed pulse
  void *delay_s = malloc(nchans*sizeof(float));
  float *delay_p = (float*)delay_s;
  for (uint64_t nc=0; nc<nchans; nc++) {
    float a = 1.0/(f0+nc*df);
    float b = 1.0/f0;
    delay_p[nc] = sigDM*DM_factor * (a*a - b*b);
  }

  // embed signal in noise;
  for (uint64_t nc=0; nc<nchans; nc++) {
    uint64_t ns = (uint64_t)((sigT + delay_p[nc])/dt);
    if (ns > nsamps) errorQuda("ns too big %lu\n", ns);
    fb_p[ns*nchans + nc] += sigamp;
  }
  
  printf("----------------------------- INJECTED SIGNAL  ----------------------------\n");
  printf("Pulse time at f0 (s)                      : %.6f (sample %lu)\n",sigT,(uint64_t)(sigT/dt));
  printf("Pulse DM (pc/cm^3)                        : %f \n",sigDM);
  printf("Signal Delays : %f, %f, %f ... %f\n",delay_p[0], delay_p[1], delay_p[2], delay_p[nchans-1]);

  // input is a pointer to an array containing a time series of length
  // nsamps for each frequency channel in plan. The data must be in
  // time-major order, i.e., frequency is the fastest-changing
  // dimension, time the slowest. There must be no padding between
  // consecutive frequency channels. 

  float raw_mean, raw_sigma;
  calc_stats_float(fb_p, nsamps*nchans, &raw_mean, &raw_sigma);
  printfQuda("Rawdata Mean (includes signal)    : %f\n", raw_mean);
  printfQuda("Rawdata StdDev (includes signal)  : %f\n", raw_sigma);
  printfQuda("Pulse S/N (per frequency channel) : %f\n", sigamp/datarms);

  input = malloc(nsamps * nchans * (in_nbits/8));

  output = (float*)malloc(nsamps * nchans * (in_nbits/8));

  // quantize array
  char *input_p = (char*)input;
  for (uint64_t ns=0; ns<nsamps; ns++) {
    for (uint64_t nc=0; nc<nchans; nc++) {
      input_p[ns*nchans+nc] = bytequant(fb_p[ns*nchans + nc]);
    }
  }

  float in_mean, in_sigma;
  calc_stats_8bit(input_p, nsamps*nchans, &in_mean, &in_sigma);

  printfQuda("Quantized data Mean (includes signal)    : %f\n", in_mean);
  printfQuda("Quantized data StdDev (includes signal)  : %f\n", in_sigma);
  printf("\n");

  dedispersionCHPC((void*)output, (void*)input, &dedisp_param);
  
  return 0.0;
}


struct dedisp_test : quda_test {

  void add_command_line_group(std::shared_ptr<GGPApp> app) const override
  {
    //quda_test::add_command_line_group(app);
  }
  
  dedisp_test(int argc, char **argv) : quda_test("Dedisp Test", argc, argv) { }
};

int main(int argc, char **argv) {
  
  dedisp_test test(argc, argv);
  test.init();
  
  int result = 0;
  if (enable_testing) {
    // Perform the tests defined in dedisp_test_gtest.hpp.
    result = test.execute();
    if (result) warningQuda("Google tests for Dedisp Interafce failed.");
  } else {
    // Perform the test specified by the command line.
    DedispTest(test_t {DEDISP_compute_prec, DEDISP_storage_prec});
  }  
  return result;
}
