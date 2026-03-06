#include <stdio.h>
#include <stdlib.h>
#include "misc.h"
#include <assert.h>
#include "util_ggp.h"

const char *get_verbosity_str(QudaVerbosity type) {
  const char *ret;

  switch (type) {
  case QUDA_SILENT: ret = "silent"; break;
  case QUDA_SUMMARIZE: ret = "summarize"; break;
  case QUDA_VERBOSE: ret = "verbose"; break;
  case QUDA_DEBUG_VERBOSE: ret = "debug"; break;
  default: fprintf(stderr, "Error: invalid verbosity type %d\n", type); exit(1);
  }

  return ret;
}

const char *get_prec_str(QudaPrecision prec) {
  const char *ret;

  switch (prec) {
  case QUDA_DOUBLE_PRECISION: ret = "double"; break;
  case QUDA_SINGLE_PRECISION: ret = "single"; break;
  case QUDA_HALF_PRECISION: ret = "half"; break;
  case QUDA_QUARTER_PRECISION: ret = "quarter"; break;
  default: ret = "unknown"; break;
  }

  return ret;
}

const char *get_quda_ver_str() {
  static char vstr[32];
  int major_num = QUDA_VERSION_MAJOR;
  int minor_num = QUDA_VERSION_MINOR;
  int ext_num = QUDA_VERSION_SUBMINOR;
  sprintf(vstr, "%1d.%1d.%1d", major_num, minor_num, ext_num);
  return vstr;
}

const char *get_memory_type_str(QudaMemoryType type) {
  const char *s;

  switch (type) {
  case QUDA_MEMORY_DEVICE: s = "device"; break;
  case QUDA_MEMORY_DEVICE_PINNED: s = "device_pinned"; break;
  case QUDA_MEMORY_HOST: s = "host"; break;
  case QUDA_MEMORY_HOST_PINNED: s = "host_pinned"; break;
  case QUDA_MEMORY_MAPPED: s = "mapped"; break;
  default: fprintf(stderr, "Error: invalid memory type\n"); exit(1);
  }

  return s;
}

const char *get_blas_datatype_str(QudaBLASDataType type) {
  const char *ret;

  switch (type) {
  case QUDA_BLAS_DATATYPE_S: ret = "real single"; break;
  case QUDA_BLAS_DATATYPE_D: ret = "real double"; break;
  case QUDA_BLAS_DATATYPE_C: ret = "complex single"; break;
  case QUDA_BLAS_DATATYPE_Z: ret = "complex double"; break;
  case QUDA_BLAS_DATATYPE_QC: ret = "complex (8b in, 32b out)"; break;
  default: fprintf(stderr, "Error: invalid BLAS datatype %d\n", type); exit(1);
  }
  
  return ret;
}

const char *get_blas_dataorder_str(QudaBLASDataOrder order) {
  const char *ret;

  switch (order) {
  case QUDA_BLAS_DATAORDER_ROW: ret = "ROW major"; break;
  case QUDA_BLAS_DATAORDER_COL: ret = "COL major"; break;
  default: fprintf(stderr, "Error: invalid BLAS dataorder %d\n", order); exit(1);
  }
  
  return ret;
}

const char *get_xengine_mat_format_str(QudaXEngineMatFormat format) {
  const char *ret;

  switch (format) {
  case QUDA_XENGINE_MAT_HERM: ret = "HERM"; break;
  case QUDA_XENGINE_MAT_TRI: ret = "TRI"; break;
  default: fprintf(stderr, "Error: invalid XEngine mat format %d\n", format); exit(1);
  }
  
  return ret;
}

const char *get_beamformer_type_str(QudaBeamformerType type) {
  const char *ret;

  switch (type) {
  case QUDA_BEAMFORMER_VOLTAGE: ret = "Voltage"; break;
  case QUDA_BEAMFORMER_VISIBILITY: ret = "Visibility"; break;
  default: fprintf(stderr, "Error: invalid Beamformer type %d\n", type); exit(1);
  }
  
  return ret;
}


const char *get_packet_format_str(QudaPacketFormat format) {
  const char *ret;

  switch (format) {
  case QUDA_PACKET_FORMAT_DSA110: ret = "dsa110"; break;
  case QUDA_PACKET_FORMAT_DSA2K: ret = "dsa2K"; break;
  default: fprintf(stderr, "Error: invalid packet format %d\n", format); exit(1);
  }
  
  return ret;
}



const char *get_blas_type_str(QudaBLASType type) {
  const char *s;
  
  switch (type) {
  case QUDA_BLAS_GEMM: s = "gemm"; break;
  case QUDA_BLAS_LU_INV: s = "lu-inv"; break;
  default: fprintf(stderr, "Error: invalid BLAS type\n"); exit(1);
  }
  return s;
}
