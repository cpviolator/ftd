#pragma once
#include <ggp.h>
#include <string>

const char *get_quda_ver_str();
const char *get_prec_str(QudaPrecision prec);
const char *get_test_type(int t);
const char *get_verbosity_str(QudaVerbosity);
const char *get_memory_type_str(QudaMemoryType type);
const char *get_blas_dataorder_str(QudaBLASDataOrder order);
const char *get_blas_datatype_str(QudaBLASDataType type);
const char *get_xengine_mat_format_str(QudaXEngineMatFormat format);
const char *get_packet_format_str(QudaPacketFormat format);
const char *get_beamformer_type_str(QudaBeamformerType type);

#define XUP 0
#define YUP 1
#define ZUP 2
#define TUP 3
#define TDOWN 4
#define ZDOWN 5
#define YDOWN 6
#define XDOWN 7
#define OPP_DIR(dir) (7 - (dir))
#define GOES_FORWARDS(dir) (dir <= 3)
#define GOES_BACKWARDS(dir) (dir > 3)
