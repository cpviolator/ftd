// check_params.h

// This file defines functions to either initialize, check, or print
// the QUDA gauge and inverter parameters.  It gets included in
// interface_quda.cpp, after either INIT_PARAM, CHECK_PARAM, or
// PRINT_PARAM is defined.
//
// If you're reading this file because it was mentioned in a "QUDA
// error" message, it probably means that you forgot to set one of the
// gauge or inverter parameters in your application before calling
// loadGaugeQuda() or invertQuda().

#include <float.h>
#define INVALID_INT QUDA_INVALID_ENUM
#define INVALID_DOUBLE DBL_MIN

// define macro to carry out the appropriate action for a given parameter

#if defined INIT_PARAM
#define P(x, val) ret.x = val
#elif defined CHECK_PARAM
#define P(x, val) if (param->x == val) errorQuda("Parameter " #x " undefined")
#elif defined PRINT_PARAM
#define P(x, val)							\
  { if (val == INVALID_DOUBLE) printfQuda(#x " = %g\n", (double)param->x); \
    else printfQuda(#x " = %d\n", (int)param->x); }
#else
#error INIT_PARAM, CHECK_PARAM, and PRINT_PARAM all undefined in check_params.h
#endif
  
#if defined INIT_PARAM
QudaBLASParam newQudaBLASParam(void)
{
  QudaBLASParam ret;
#elif defined CHECK_PARAM
static void checkBLASParam(QudaBLASParam *param)
{
#else
void printQudaBLASParam(QudaBLASParam *param)
{
  printfQuda("QUDA blas parameters:\n");
#endif

#if defined CHECK_PARAM
  if (param->struct_size != (size_t)INVALID_INT && param->struct_size != sizeof(*param))
    errorQuda("Unexpected QudaBLASParam struct size %lu, expected %lu", param->struct_size, sizeof(*param));
#else
  P(struct_size, (size_t)INVALID_INT);
#endif

#ifdef INIT_PARAM
  P(trans_a, QUDA_BLAS_OP_N);
  P(trans_b, QUDA_BLAS_OP_N);
  P(m, INVALID_INT);
  P(n, INVALID_INT);
  P(k, INVALID_INT);
  P(lda, INVALID_INT);
  P(ldb, INVALID_INT);
  P(ldc, INVALID_INT);
  P(a_offset, 0);
  P(b_offset, 0);
  P(c_offset, 0);
  P(a_stride, 1);
  P(b_stride, 1);
  P(c_stride, 1);
  P(batch_count, 1);
  P(location, QUDA_CPU_FIELD_LOCATION);
  P(engine, QUDA_BLAS_ENGINE_CUBLAS);
  P(data_type, QUDA_BLAS_DATATYPE_S);
  P(data_order, QUDA_BLAS_DATAORDER_ROW);
  P(blas_type, QUDA_BLAS_INVALID);
  P(inv_mat_size, 0);
  P(stream_idx, 0);
#else
  P(trans_a, QUDA_BLAS_OP_INVALID);
  P(trans_b, QUDA_BLAS_OP_INVALID);
  P(m, INVALID_INT);
  P(n, INVALID_INT);
  P(k, INVALID_INT);
  P(lda, INVALID_INT);
  P(ldb, INVALID_INT);
  P(ldc, INVALID_INT);
  P(a_offset, INVALID_INT);
  P(b_offset, INVALID_INT);
  P(c_offset, INVALID_INT);
  P(a_stride, INVALID_INT);
  P(b_stride, INVALID_INT);
  P(c_stride, INVALID_INT);
  P(batch_count, INVALID_INT);
  P(location, QUDA_INVALID_FIELD_LOCATION);
  P(engine, QUDA_BLAS_ENGINE_INVALID);
  P(data_type, QUDA_BLAS_DATATYPE_INVALID);
  P(data_order, QUDA_BLAS_DATAORDER_INVALID);
  P(blas_type, QUDA_BLAS_INVALID);
  P(inv_mat_size, INVALID_INT);
  P(stream_idx, INVALID_INT);
#endif

#ifdef INIT_PARAM
  return ret;
#endif
}

#if defined INIT_PARAM
XEngineParam newXEngineParam(void)
 {
   XEngineParam ret;
#elif defined CHECK_PARAM
static void checkXEngineParam(XEngineParam *param)
{
#else
void printXEngineParam(XEngineParam *param)
{
  printfQuda("XEngine parameters:\n");
#endif

#if defined CHECK_PARAM
  if (param->struct_size != (size_t)INVALID_INT && param->struct_size != sizeof(*param))
    errorQuda("Unexpected XEngineParam struct size %lu, expected %lu", param->struct_size, sizeof(*param));
#else
  P(struct_size, (size_t)INVALID_INT);
#endif

#ifdef INIT_PARAM
  P(packet_format, QUDA_PACKET_FORMAT_DSA2K);
  P(n_payload, (size_t)INVALID_INT);
  P(n_antennae_per_payload, (size_t)INVALID_INT);
  P(n_channels_per_payload, (size_t)INVALID_INT);
  P(n_time_per_payload, (size_t)INVALID_INT);
  P(n_time_inner, (size_t)INVALID_INT);
  P(n_polarizations, (size_t)INVALID_INT);
  P(compute_prec, QUDA_HALF_PRECISION);
  P(output_prec, QUDA_SINGLE_PRECISION);
  P(engine, QUDA_BLAS_ENGINE_CUBLAS);
  P(data_type, QUDA_BLAS_DATATYPE_QC);
  P(data_order, QUDA_BLAS_DATAORDER_ROW);
  P(format, QUDA_XENGINE_MAT_HERM);
#else
  P(packet_format, QUDA_PACKET_FORMAT_INVALID);
  P(n_payload, (size_t)INVALID_INT);
  P(n_antennae_per_payload, (size_t)INVALID_INT);
  P(n_channels_per_payload, (size_t)INVALID_INT);
  P(n_time_per_payload, (size_t)INVALID_INT);
  P(n_time_inner, (size_t)INVALID_INT);
  P(n_polarizations, (size_t)INVALID_INT);
  P(compute_prec, QUDA_INVALID_PRECISION);
  P(output_prec, QUDA_INVALID_PRECISION);
  P(engine, QUDA_BLAS_ENGINE_INVALID);
  P(data_type, QUDA_BLAS_DATATYPE_INVALID);
  P(data_order, QUDA_BLAS_DATAORDER_INVALID);
  P(format, QUDA_XENGINE_MAT_INVALID);
#endif

#ifdef INIT_PARAM
  return ret;
#endif
}

#if defined INIT_PARAM
BeamformerParam newBeamformerParam(void)
{
   BeamformerParam ret;
#elif defined CHECK_PARAM
static void checkBeamformerParam(BeamformerParam *param)
{
#else
void printBeamformerParam(BeamformerParam *param)
{
  printfQuda("Beamformer parameters:\n");
#endif

#if defined CHECK_PARAM
  if (param->struct_size != (size_t)INVALID_INT && param->struct_size != sizeof(*param))
    errorQuda("Unexpected BeamformerParam struct size %lu, expected %lu", param->struct_size, sizeof(*param));
#else
  P(struct_size, (size_t)(size_t)INVALID_INT);
#endif

#ifdef INIT_PARAM
  P(type, QUDA_BEAMFORMER_VOLTAGE);
  P(packet_format, QUDA_PACKET_FORMAT_DSA2K);
  P(n_payload, (size_t)INVALID_INT);
  P(n_antennae_per_payload, (size_t)INVALID_INT);
  P(n_channels_per_payload, (size_t)INVALID_INT);
  P(n_time_per_payload, (size_t)INVALID_INT);
  P(n_channels_inner, (size_t)INVALID_INT);
  P(n_time_inner, (size_t)INVALID_INT);
  P(n_arm, (size_t)INVALID_INT);
  P(n_beam, (size_t)INVALID_INT);
  P(n_flags, (size_t)INVALID_INT);
  P(n_polarizations, (size_t)INVALID_INT);
  P(beam_separation, INVALID_DOUBLE);
  P(beam_separation_ns, INVALID_DOUBLE);
  P(sfreq, INVALID_DOUBLE);
  P(wfreq, INVALID_DOUBLE);
  P(declination, INVALID_DOUBLE);
  P(n_grid, (size_t)0);
  P(cell_size_rad, 0.0);
  P(freq_start_hz, 0.0);
  P(freq_step_hz, 0.0);
  P(n_channels_per_tile, (size_t)0);
  P(compute_prec, QUDA_HALF_PRECISION);
  P(output_prec, QUDA_SINGLE_PRECISION);
  P(engine, QUDA_BLAS_ENGINE_CUBLAS);
  P(data_type, QUDA_BLAS_DATATYPE_QC);
  P(data_order, QUDA_BLAS_DATAORDER_ROW);
#else
  P(type, QUDA_BEAMFORMER_INVALID);
  P(packet_format, QUDA_PACKET_FORMAT_INVALID);
  P(n_payload, (size_t)INVALID_INT);
  P(n_antennae_per_payload, (size_t)INVALID_INT);
  P(n_channels_per_payload, (size_t)INVALID_INT);
  P(n_time_per_payload, (size_t)INVALID_INT);
  P(n_channels_inner, (size_t)INVALID_INT);
  P(n_time_inner, (size_t)INVALID_INT);
  P(n_arm, (size_t)INVALID_INT);
  P(n_beam, (size_t)INVALID_INT);
  P(n_flags, (size_t)INVALID_INT);
  P(n_polarizations, (size_t)INVALID_INT);
  P(beam_separation, INVALID_DOUBLE);
  P(beam_separation_ns, INVALID_DOUBLE);
  P(sfreq, INVALID_DOUBLE);
  P(wfreq, INVALID_DOUBLE);
  P(declination, INVALID_DOUBLE);
  // n_grid, cell_size_rad, freq_start_hz, freq_step_hz, n_channels_per_tile
  // are optional (defaults handled in VisibilityBeamformer constructor)
  P(compute_prec, QUDA_INVALID_PRECISION);
  P(output_prec, QUDA_INVALID_PRECISION);
  P(engine, QUDA_BLAS_ENGINE_INVALID);
  P(data_type, QUDA_BLAS_DATATYPE_INVALID);
  P(data_order, QUDA_BLAS_DATAORDER_INVALID);
#endif

#ifdef INIT_PARAM
  return ret;
#endif
}

#if defined INIT_PARAM
DedispParam newDedispParam(void)
{
   DedispParam ret;
#elif defined CHECK_PARAM
static void checkDedispParam(DedispParam *param)
{
#else
void printDedispParam(DedispParam *param)
{
  printfQuda("Dedisp parameters:\n");
#endif

#if defined CHECK_PARAM
  if (param->struct_size != (size_t)INVALID_INT && param->struct_size != sizeof(*param))
    errorQuda("Unexpected DedispParam struct size %lu, expected %lu", param->struct_size, sizeof(*param));
#else
  P(struct_size, (size_t)(size_t)INVALID_INT);
#endif

#ifdef INIT_PARAM
  P(in_nbits, (size_t)INVALID_INT);
  P(out_nbits, (size_t)INVALID_INT);
  P(in_stride, (size_t)INVALID_INT);
  P(out_stride, (size_t)INVALID_INT);
  P(n_time, INVALID_INT);
  P(n_time_dd, INVALID_INT);
  P(n_time_out, INVALID_INT);
  P(n_boxcar, INVALID_INT);
  P(n_time_dedisp, INVALID_INT);
  P(minDM, INVALID_DOUBLE);
  P(maxDM, INVALID_DOUBLE);
  P(max_width, INVALID_DOUBLE);
  P(tolerance, INVALID_DOUBLE);
  P(DMs, INVALID_INT);
  P(ndms, INVALID_INT);
  P(gulp, INVALID_INT);
  P(dedisp_lib, CHPC_DEDISPERSION_LIB_INVALID);    
#else
  P(in_nbits, (size_t)INVALID_INT);
  P(out_nbits, (size_t)INVALID_INT);
  P(in_stride, (size_t)INVALID_INT);
  P(out_stride, (size_t)INVALID_INT);
  P(n_time, INVALID_INT);
  P(n_time_dd, INVALID_INT);
  P(n_time_out, INVALID_INT);
  P(n_boxcar, INVALID_INT);
  P(n_time_dedisp, INVALID_INT);
  P(minDM, INVALID_DOUBLE);
  P(maxDM, INVALID_DOUBLE);
  P(max_width, INVALID_DOUBLE);
  P(tolerance, INVALID_DOUBLE);
  P(DMs, INVALID_INT);
  P(ndms, INVALID_INT);
  P(gulp, INVALID_INT);
  P(dedisp_lib, CHPC_DEDISPERSION_LIB_INVALID);    
#endif

#ifdef INIT_PARAM
  return ret;
#endif
}
// clean up
#undef INVALID_INT
#undef INVALID_DOUBLE
#undef P
 
