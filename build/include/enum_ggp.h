#pragma once

#define QUDA_INVALID_ENUM (-0x7fffffff - 1)

#ifdef __cplusplus
extern "C" {
#endif

typedef enum qudaError_t { QUDA_SUCCESS = 0, QUDA_ERROR = 1, QUDA_ERROR_UNINITIALIZED = 2 } qudaError_t;

typedef enum QudaMemoryType_s {
  QUDA_MEMORY_DEVICE,
  QUDA_MEMORY_DEVICE_PINNED,
  QUDA_MEMORY_HOST,
  QUDA_MEMORY_HOST_PINNED,
  QUDA_MEMORY_MAPPED,
  QUDA_MEMORY_MANAGED,
  QUDA_MEMORY_INVALID = QUDA_INVALID_ENUM
} QudaMemoryType;

typedef enum QudaPrecision_s {
  QUDA_QUARTER_PRECISION = 1,
  QUDA_HALF_PRECISION = 2,
  QUDA_SINGLE_PRECISION = 4,
  QUDA_DOUBLE_PRECISION = 8,
  QUDA_INVALID_PRECISION = QUDA_INVALID_ENUM
} QudaPrecision;
  
typedef enum QudaVerbosity_s {
  QUDA_SILENT,
  QUDA_SUMMARIZE,
  QUDA_VERBOSE,
  QUDA_DEBUG_VERBOSE,
  QUDA_INVALID_VERBOSITY = QUDA_INVALID_ENUM
} QudaVerbosity;

typedef enum QudaTune_s { QUDA_TUNE_NO, QUDA_TUNE_YES, QUDA_TUNE_INVALID = QUDA_INVALID_ENUM } QudaTune;
  
// Where the field is stored
typedef enum QudaFieldLocation_s {
  QUDA_CPU_FIELD_LOCATION = 1,
  QUDA_CUDA_FIELD_LOCATION = 2,
  QUDA_INVALID_FIELD_LOCATION = QUDA_INVALID_ENUM
} QudaFieldLocation;

  
typedef enum QudaFieldCreate_s {
  QUDA_NULL_FIELD_CREATE,      // new field
  QUDA_ZERO_FIELD_CREATE,      // new field and zero it
  QUDA_COPY_FIELD_CREATE,      // copy to field
  QUDA_REFERENCE_FIELD_CREATE, // reference to field
  QUDA_GHOST_FIELD_CREATE,     // dummy field used only for ghost storage
  QUDA_INVALID_FIELD_CREATE = QUDA_INVALID_ENUM
} QudaFieldCreate;
  
typedef enum QudaNoiseType_s {
  QUDA_NOISE_GAUSS,
  QUDA_NOISE_UNIFORM,
  QUDA_NOISE_INVALID = QUDA_INVALID_ENUM
} QudaNoiseType;
  
typedef enum QudaUseInitGuess_s {
  QUDA_USE_INIT_GUESS_NO,
  QUDA_USE_INIT_GUESS_YES,
  QUDA_USE_INIT_GUESS_INVALID = QUDA_INVALID_ENUM
} QudaUseInitGuess;
  
typedef enum QudaBoolean_s {
  QUDA_BOOLEAN_FALSE = 0,
  QUDA_BOOLEAN_TRUE = 1,
  QUDA_BOOLEAN_INVALID = QUDA_INVALID_ENUM
} QudaBoolean;

typedef enum QudaBLASType_s {
  QUDA_BLAS_GEMM = 0,
  QUDA_BLAS_LU_INV = 1,
  QUDA_BLAS_INVALID = QUDA_INVALID_ENUM
} QudaBLASType;

typedef enum QudaBLASOperation_s {
  QUDA_BLAS_OP_N = 0, // No transpose
  QUDA_BLAS_OP_T = 1, // Transpose only
  QUDA_BLAS_OP_C = 2, // Conjugate transpose
  QUDA_BLAS_OP_INVALID = QUDA_INVALID_ENUM
} QudaBLASOperation;

typedef enum QudaBLASDataType_s {
  QUDA_BLAS_DATATYPE_S = 0, // Single
  QUDA_BLAS_DATATYPE_D = 1, // Double
  QUDA_BLAS_DATATYPE_C = 2, // Single Complex
  QUDA_BLAS_DATATYPE_Z = 3, // Double Complex
  QUDA_BLAS_DATATYPE_H = 4, // Half Real (16bit CUDA intrinsic)
  QUDA_BLAS_DATATYPE_HC = 5, // Half Complex ((16,16)bit CUDA intrinsic)
  QUDA_BLAS_DATATYPE_QC = 6, // Quarter Complex ((4,4)b Custom)
  QUDA_BLAS_DATATYPE_INVALID = QUDA_INVALID_ENUM
} QudaBLASDataType;

typedef enum QudaBLASDataOrder_s {
  QUDA_BLAS_DATAORDER_ROW = 0,
  QUDA_BLAS_DATAORDER_COL = 1,
  //QUDA_BLAS_DATAORDER_INTER_ROW = 0,
  //QUDA_BLAS_DATAORDER_INTER_COL = 1,
  //QUDA_BLAS_DATAORDER_PLANAR_ROW = 2,
  //QUDA_BLAS_DATAORDER_PLANAR_COL = 3,
  QUDA_BLAS_DATAORDER_INVALID = QUDA_INVALID_ENUM
} QudaBLASDataOrder;

typedef enum QudaBLASEngine_s {
  QUDA_BLAS_ENGINE_CUBLAS = 0,
  QUDA_BLAS_ENGINE_CUTLASS = 1,
  QUDA_BLAS_ENGINE_TCC = 2,
  QUDA_BLAS_ENGINE_INVALID = QUDA_INVALID_ENUM
} QudaBLASEngine;
  
typedef enum QudaBeamformerType_s {
  QUDA_BEAMFORMER_VOLTAGE = 0,
  QUDA_BEAMFORMER_VISIBILITY = 1,
  QUDA_BEAMFORMER_INVALID = QUDA_INVALID_ENUM
} QudaBeamformerType;
  
typedef enum QudaXEngineMatFormat_s {
  QUDA_XENGINE_MAT_HERM = 0,
  QUDA_XENGINE_MAT_TRI = 1,
  QUDA_XENGINE_MAT_INVALID = QUDA_INVALID_ENUM
} QudaXEngineMatFormat;

typedef enum QudaPacketFormat_s {
  //QUDA_PACKET_FORMAT_DSA110,
  //QUDA_PACKET_FORMAT_CASM ,
  //QUDA_PACKET_FORMAT_DSA2K,
  
  QUDA_PACKET_FORMAT_DSA110 = 4608,
  QUDA_PACKET_FORMAT_CASM = 46080,
  QUDA_PACKET_FORMAT_DSA2K = 8192,
  
  QUDA_PACKET_FORMAT_INVALID = QUDA_INVALID_ENUM
} QudaPacketFormat;
  
// Allows to choose an appropriate external library
typedef enum QudaExtLibType_s {
  QUDA_CUSOLVE_EXTLIB,
  QUDA_EIGEN_EXTLIB,
  QUDA_EXTLIB_INVALID = QUDA_INVALID_ENUM
} QudaExtLibType;

typedef enum ChpcDedispersionLib_s {
  CHPC_DEDISP_DEDISPERSION_LIB,
  CHPC_ASTROACCEL_DEDISPERSION_LIB,
  CHPC_DEDISPERSION_LIB_INVALID = QUDA_INVALID_ENUM
} ChpcDedispersionLib;

  
#ifdef __cplusplus
}
#endif

