#pragma once

/**
 * @file  quda.h
 * @brief Main header file for the QUDA library
 *
 * Note to QUDA developers: When adding new members to QudaGaugeParam
 * and QudaInvertParam, be sure to update lib/check_params.h as well
 * as the Fortran interface in lib/quda_fortran.F90.
 */

#include <enum_ggp.h>
#include <stdio.h> /* for FILE */
#include <ggp_define.h>
#include <ggp_constants.h>

#ifndef __CUDACC_RTC__
#define double_complex double _Complex
#else // keep NVRTC happy since it can't handle C types
#define double_complex double2
#endif

#ifdef __cplusplus
extern "C" {
#endif

  typedef struct QudaBLASParam_s {
    size_t struct_size; /**< Size of this struct in bytes.  Used to ensure that the host application and QUDA see the same struct*/
    
    QudaBLASType blas_type; /**< Type of BLAS computation to perform */
    QudaFieldLocation location; /**< Location of data arrays */
    
    // GEMM params
    QudaBLASOperation trans_a; /**< operation op(A) that is non- or (conj.) transpose. */
    QudaBLASOperation trans_b; /**< operation op(B) that is non- or (conj.) transpose. */
    int m;                     /**< number of rows of matrix op(A) and C. */
    int n;                     /**< number of columns of matrix op(B) and C. */
    int k;                     /**< number of columns of op(A) and rows of op(B). */
    int lda;                   /**< leading dimension of two-dimensional array used to store the matrix A. */
    int ldb;                   /**< leading dimension of two-dimensional array used to store matrix B. */
    int ldc;                   /**< leading dimension of two-dimensional array used to store matrix C. */
    int a_offset;              /**< position of the A array from which begin read/write. */
    int b_offset;              /**< position of the B array from which begin read/write. */
    int c_offset;              /**< position of the C array from which begin read/write. */
    int a_stride;              /**< stride of the A array in strided(batched) mode */
    int b_stride;              /**< stride of the B array in strided(batched) mode */
    int c_stride;              /**< stride of the C array in strided(batched) mode */
    double_complex alpha; /**< scalar used for multiplication. */
    double_complex beta;  /**< scalar used for multiplication. If beta==0, C does not have to be a valid input. */

    // LU inversion params
    int inv_mat_size; /**< The rank of the square matrix in the LU inversion */

    // Common params
    int batch_count;              /**< Number of pointers contained in arrayA, arrayB and arrayC. */
    QudaBLASEngine engine;        /**< Which BLAS engine to use (cuBLAS, CUTLAS,...) */
    QudaBLASDataType data_type;   /**< Specifies if using S(C) or D(Z) BLAS type */
    QudaBLASDataOrder data_order; /**< Specifies if using Row or Column major */
    int stream_idx;               /**< The stream on which to launch cuBLAS instance */

    
  } QudaBLASParam;

  /**
   * Parameters relating to the PipelineFTD
   */
  typedef struct PipelineFTDParam_s {
    size_t struct_size; /**< Size of this struct in bytes.  Used to ensure that the host application and QUDA see the same struct*/

    QudaFieldLocation in_location;  /** Location of input data arrays */
    QudaFieldLocation out_location; /** Location of output data arrays */    
    QudaPacketFormat packet_format; /** Specifies the packet data layout and size */
    
    size_t n_payload;                  /** Number of payload(s) in this instance */
    size_t n_base;                     /** Number of bases in the payload */
    size_t n_antennae_per_payload;     /** Number of antennae in the payload */
    size_t n_channels_per_payload;     /** Number of channels in the payload */
    size_t n_time_per_payload;         /** Number of time steps in the payload */
    size_t n_time_inner;               /** Number of fine time steps to sum over in beamformer */
    size_t n_polarizations;            /** Number of polarisations in the data */
    size_t n_beam;                     /** Number of beams to form */
    double beam_separation;            /** The beam separation */
    
    size_t n_channels_inner;        /** Number of channels to sum over in voltage beamformer */
    size_t n_time_power_sum;        /** Number of time steps to sum over in power sum for voltage beamformer */
    
    QudaPrecision compute_prec;   /**< Specifies the precision of the compute to send to the desired compute method. */
    QudaPrecision output_prec;    /**< Specifies the precision of the output */
    QudaBLASEngine engine;        /**< Which BLAS engine to use (cuBLAS, CUTLAS,...) */
    QudaBLASDataType data_type;   /**< Specifies if using S(C), D(Z), or QC, a specialised (4,4) bit complex input */
    QudaBLASDataOrder data_order; /**< Specifies if using Row or Column major */
    QudaXEngineMatFormat format;  /**< Specifies if the output is full hermitian (testing) or triangular (production) */
    QudaVerbosity verbosity;      /**< The verbosity setting to use in the PipelineFTD */
    
  } PipelineFTDParam;

  
  /**
   * Parameters relating to the XEngine
   */
  typedef struct XEngineParam_s {
    size_t struct_size; /**< Size of this struct in bytes.  Used to ensure that the host application and QUDA see the same struct*/

    QudaFieldLocation in_location;   /** Location of input data arrays */
    QudaFieldLocation out_location;  /** Location of output data arrays */    
    QudaPacketFormat packet_format; /** Specifies the packet data layout and size */
    
    size_t n_payload;                  /** Number of payload(s) in this instance */
    size_t n_antennae_per_payload;     /** Number of antennae in the payload */
    size_t n_channels_per_payload;     /** Number of channels in the payload */
    size_t n_time_per_payload;         /** Number of time steps in the payload */
    size_t n_time_inner;               /** Number of fine time steps to sum over in beamformer */
    size_t n_polarizations;            /** Number of polarisations in the data */

    QudaPrecision compute_prec;   /**< Specifies the precision of the compute to send to the desired compute method. */
    QudaPrecision output_prec;    /**< Specifies the precision of the output */
    QudaBLASEngine engine;        /**< Which BLAS engine to use (cuBLAS, CUTLAS,...) */
    QudaBLASDataType data_type;   /**< Specifies if using S(C), D(Z), or QC, a specialised (4,4) bit complex input */
    QudaBLASDataOrder data_order; /**< Specifies if using Row or Column major */
    QudaXEngineMatFormat format;  /**< Specifies if the output is full hermitian (testing) or triangular (production) */
    QudaVerbosity verbosity;      /**< The verbosity setting to use in the XEngine */
    
  } XEngineParam;
  
  /**
   * Parameters relating to the Beamformer
   * DMH: Combine this with the XEngine parameter structure 
   *      when appropriate?
   */
  typedef struct BeamformerParam_s {
    size_t struct_size; /**< Size of this struct in bytes.  Used to ensure that the host application and QUDA see the same struct*/

    QudaPacketFormat packet_format; /** Specifies the packet data layout and size */
    QudaBeamformerType type;        /** Specifies which beamformer algorithm to use */
    size_t n_payload;               /** Number of payloads in this instance */
    size_t n_antennae_per_payload;  /** Number of antennae in the payload */
    size_t n_channels_per_payload;  /** Number of channels in the payload */
    size_t n_time_per_payload;      /** Number of time steps in the payload */
    size_t n_channels_inner;        /** Number of channels to sum over in beamformer */
    size_t n_time_inner;            /** Number of fine time steps to sum over in beamformer */
    size_t n_time_power_sum;        /** Number of time steps to sum over in power sum */
    size_t n_beam;                  /** Number of beams to form */
    size_t n_arm;                   /** Number of arms (DSA110 = 2, others = 1) */
    size_t n_polarizations;         /** Number of polarisations in the data */
    double n_flags;                 /** The number of flagged antennae */
    double beam_separation;         /** The beam separation */
    double beam_separation_ns;      /** The beam separation in N/S */
    double sfreq;                   /** The upper limit of the highest frequency band (MHz) */
    double wfreq;                   /** The width of the frequency bands (MHz) */
    double declination;             /** The pointing declination */

    // Visibility beamformer imaging parameters
    size_t n_grid;                  /** UV grid dimension (Ng x Ng), default 1024 */
    double cell_size_rad;           /** UV cell size in radians */
    double freq_start_hz;           /** Starting frequency in Hz */
    double freq_step_hz;            /** Frequency step in Hz */
    size_t n_channels_per_tile;     /** Frequency tile size (0 = auto) */

    QudaPrecision compute_prec;   /**< Specifies the precision of the compute to send to the desired compute method. */
    QudaPrecision output_prec;    /**< Specifies the precision of the output */
    QudaBLASEngine engine;        /**< Which BLAS engine to use (cuBLAS, CUTLAS,...) */
    QudaBLASDataType data_type;   /**< Specifies if using S(C), D(Z), or a specialised (4,4) bit complex input QC */
    QudaBLASDataOrder data_order; /**< Specifies if using Row or Column major */
    QudaVerbosity verbosity;      /**< The verbosity setting to use in the Beamformer */
    
  } BeamformerParam;

  /**
   * Parameters relating to the VisArray
   */  
  typedef struct VisArrayParam_s {
    size_t struct_size; /**< Size of this struct in bytes.  Used to ensure that the host application and QUDA see the same struct*/

    QudaFieldLocation location;     /**< Location of data array */
    
    size_t n_payload;               /** Number of payloads in this instance */
    size_t n_base;                  /** Number of bases in the payload */
    size_t n_channels;              /** Number of channels in the payload */
    size_t n_time;                  /** Number of time steps in the payload */
    size_t n_time_inner;            /** Number of fine time steps to sum over in beamformer */
    size_t n_polarizations;         /** Number of polarisations in the data */

    QudaPrecision compute_prec;   /** Specifies the precision of the compute to send to the desired compute method. */
    QudaPrecision storage_prec;   /** Specifies the storage precision of the array */
    QudaFieldCreate create_type;  /** Specifies the memory creation/reference type */
    
  } VisArrayParam;

  /**
   * Parameters relating to the FilterBank
   */  
  typedef struct FilterBankParam_s {
    size_t struct_size; /**< Size of this struct in bytes.  Used to ensure that the host application and QUDA see the same struct*/

    QudaFieldLocation location;     /**< Location of data array */
    
    size_t n_payload;               /** Number of payloads in this instance */
    size_t n_beam;                  /** Number of means in the payload */
    size_t n_channels;              /** Number of channels in the filterbank */
    size_t n_time;                  /** Number of time steps in the filterbank */

    QudaPrecision compute_prec;   /** Specifies the precision of the compute to send to the desired compute method. */
    QudaPrecision storage_prec;   /** Specifies the storage precision of the array */
    QudaFieldCreate create_type;  /** Specifies the memory creation/reference type */
    
  } FilterBankParam;

  /**
   * Parameters relating to the FilterBank
   */  
  typedef struct VoltArrayParam_s {
    size_t struct_size; /**< Size of this struct in bytes.  Used to ensure that the host application and QUDA see the same struct*/

    QudaFieldLocation location;     /**< Location of data array */
    
    size_t n_payload;               /** Number of payloads in this instance */
    size_t n_antennae_per_payload;  /** Number of antennae in the payload */
    size_t n_channels_per_payload;  /** Number of channels in the payload */
    size_t n_time_per_payload;      /** Number of time steps in the payload */
    size_t n_time_inner;            /** Number of fine time steps to sum over in beamformer */
    size_t n_polarizations;         /** Number of polarisations in the data */

    QudaPrecision compute_prec;   /** Specifies the precision of the compute to send to the desired compute method. */
    QudaPrecision storage_prec;   /** Specifies the storage precision of the array */
    QudaFieldCreate create_type;  /** Specifies the memory creation/reference type */
    
  } VoltArrayParam;

  
  /**
   * Parameters relating to the dedisp interface
   */
  typedef struct DedispParam_s {
    size_t struct_size; /**< Size of this struct in bytes.  Used to ensure that the host application and QUDA see the same struct*/

    QudaFieldLocation location;    /**< Location of data array */
    
    size_t in_nbits;
    size_t out_nbits;
    size_t in_stride;
    size_t out_stride;
    int n_time;
    int n_time_dd;
    int n_time_out;
    int n_boxcar;
    int n_time_dedisp;
    double minDM;
    double maxDM;
    double max_width;
    double tolerance;
    int DMs;
    int ndms;
    int gulp;
    ChpcDedispersionLib dedisp_lib;
    
    QudaVerbosity verbosity;      /**< The verbosity setting to use in Dedisp */
    
  } DedispParam;

  
  // Interface functions, found in interface_quda.cpp
  //-------------------------------------------------  
  /**
   * Set parameters related to status reporting.
   *
   * In typical usage, this function will be called once (or not at
   * all) just before the call to initQuda(), but it's valid to call
   * it any number of times at any point during execution.  Prior to
   * the first time it's called, the parameters take default values
   * as indicated below.
   *
   * @param verbosity  Default verbosity, ranging from QUDA_SILENT to
   *                   QUDA_DEBUG_VERBOSE.  Within a solver, this
   *                   parameter is overridden by the "verbosity"
   *                   member of QudaInvertParam.  The default value
   *                   is QUDA_SUMMARIZE.
   *
   * @param prefix     String to prepend to all messages from QUDA.  This
   *                   defaults to the empty string (""), but you may
   *                   wish to specify something like "QUDA: " to
   *                   distinguish QUDA's output from that of your
   *                   application.
   *
   * @param outfile    File pointer (such as stdout, stderr, or a handle
   *                   returned by fopen()) where messages should be
   *                   printed.  The default is stdout.
   */
  void setVerbosityQuda(QudaVerbosity verbosity, const char prefix[],
                        FILE *outfile);

  /**
   * initCommsGridQuda() takes an optional "rank_from_coords" argument that
   * should be a pointer to a user-defined function with this prototype.
   *
   * @param coords  Node coordinates
   * @param fdata   Any auxiliary data needed by the function
   * @return        MPI rank or QMP node ID cooresponding to the node coordinates
   *
   * @see initCommsGridQuda
   */
  typedef int (*QudaCommsMap)(const int *coords, void *fdata);

  /**
   * @param mycomm User provided MPI communicator in place of MPI_COMM_WORLD
   */

  void qudaSetCommHandle(void *mycomm);

  /**
   * Declare the grid mapping ("logical topology" in QMP parlance)
   * used for communications in a multi-GPU grid.  This function
   * should be called prior to initQuda().  The only case in which
   * it's optional is when QMP is used for communication and the
   * logical topology has already been declared by the application.
   *
   * @param nDim   Number of grid dimensions.  "4" is the only supported
   *               value currently.
   *
   * @param dims   Array of grid dimensions.  dims[0]*dims[1]*dims[2]*dims[3]
   *               must equal the total number of MPI ranks or QMP nodes.
   *
   * @param func   Pointer to a user-supplied function that maps coordinates
   *               in the communication grid to MPI ranks (or QMP node IDs).
   *               If the pointer is NULL, the default mapping depends on
   *               whether QMP or MPI is being used for communication.  With
   *               QMP, the existing logical topology is used if it's been
   *               declared.  With MPI or as a fallback with QMP, the default
   *               ordering is lexicographical with the fourth ("t") index
   *               varying fastest.
   *
   * @param fdata  Pointer to any data required by "func" (may be NULL)
   *
   * @see QudaCommsMap
   */

  void initCommsGridQuda(int nDim, const int *dims, QudaCommsMap func, void *fdata);

  /**
   * Initialize the library.  This is a low-level interface that is
   * called by initQuda.  Calling initQudaDevice requires that the
   * user also call initQudaMemory before using QUDA.
   *
   * @param device CUDA device number to use.  In a multi-GPU build,
   *               this parameter may either be set explicitly on a
   *               per-process basis or set to -1 to enable a default
   *               allocation of devices to processes.
   */
  void initQudaDevice(int device);

  /**
   * Initialize the library persistant memory allocations (both host
   * and device).  This is a low-level interface that is called by
   * initQuda.  Calling initQudaMemory requires that the user has
   * previously called initQudaDevice.
   */
  void initQudaMemory();

  /**
   * Initialize the library.  This function is actually a wrapper
   * around calls to initQudaDevice() and initQudaMemory().
   *
   * @param device  CUDA device number to use.  In a multi-GPU build,
   *                this parameter may either be set explicitly on a
   *                per-process basis or set to -1 to enable a default
   *                allocation of devices to processes.
   */
  void initQuda(int device);

  /**
   * Finalize the library.
   */
  void endQuda(void);

  /**
   * A new VisArrayParam should always be initialized immediately
   * after it's defined (and prior to explicitly setting its members)
   * using this function.  Typical usage is as follows:
   *
   *   VisArrayParam param = newVisArrayParam();
   */
  VisArrayParam newVisArrayParam(void);
  
  /**
   * Print the members of VisArrayParam.
   * @param param The VisArrayParam whose elements we are to print.
   */
  void printVisArrayParam(VisArrayParam *param);

  /**
   * A new VoltArrayParam should always be initialized immediately
   * after it's defined (and prior to explicitly setting its members)
   * using this function.  Typical usage is as follows:
   *
   *   VoltArrayParam param = newVoltArrayParam();
   */
  VoltArrayParam newVoltArrayParam(void);
  
  /**
   * Print the members of VoltArrayParam.
   * @param param The VoltArrayParam whose elements we are to print.
   */
  void printVoltArrayParam(VoltArrayParam *param);

  /**
   * A new FilterBankParam should always be initialized immediately
   * after it's defined (and prior to explicitly setting its members)
   * using this function.  Typical usage is as follows:
   *
   *   FilterBankParam param = newFilterBankParam();
   */
  FilterBankParam newFilterBankParam(void);
  
  /**
   * Print the members of FilterBankParam.
   * @param param The FilterBankParam whose elements we are to print.
   */
  void printFilterBankParam(FilterBankParam *param);

  /**
   * A new PipelineFTDParam should always be initialized immediately
   * after it's defined (and prior to explicitly setting its members)
   * using this function.  Typical usage is as follows:
   *
   *   PipelineFTDParam param = newPipelineFTDParam();
   */
  PipelineFTDParam newPipelineFTDParam(void);
  
  /**
   * Print the members of PipelineFTDParam.
   * @param param The PipelineFTDParam whose elements we are to print.
   */
  void printPipelineFTDParam(PipelineFTDParam *param);
  
  /**
   * A new DedispParam should always be initialized immediately
   * after it's defined (and prior to explicitly setting its members)
   * using this function.  Typical usage is as follows:
   *
   *   DedispParam param = newDedispParam();
   */
  DedispParam newDedispParam(void);
  
  /**
   * Print the members of DedispParam.
   * @param param The dedispParam whose elements we are to print.
   */
  void printDedispParam(DedispParam *param);
  
  /**
   * A new BeamformerParam should always be initialized immediately
   * after it's defined (and prior to explicitly setting its members)
   * using this function.  Typical usage is as follows:
   *
   *   BeamformerParam param = newBeamformerParam();
   */
  BeamformerParam newBeamformerParam(void);
  
  /**
   * Print the members of BeamformerParam.
   * @param param The BeamformerParam whose elements we are to print.
   */
  void printBeamformerParam(BeamformerParam *param);

  /**
   * A new XEngineParam should always be initialized immediately
   * after it's defined (and prior to explicitly setting its members)
   * using this function.  Typical usage is as follows:
   *
   *   XEngineParam param = newXEngineParam();
   */
  XEngineParam newXEngineParam(void);
  
  /**
   * Print the members of XEngineParam.
   * @param param The XEngineParam whose elements we are to print.
   */
  void printXEngineParam(XEngineParam *param);
  
  /**
   * A new QudaBLASParam should always be initialized immediately
   * after it's defined (and prior to explicitly setting its members)
   * using this function.  Typical usage is as follows:
   *
   *   QudaBLASParam param = newQudaBLASParam();
   */
  QudaBLASParam newQudaBLASParam(void);
  
  /**
   * Print the members of QudaBLASParam.
   * @param param The QudaBLASParam whose elements we are to print.
   */
  void printQudaBLASParam(QudaBLASParam *param);

  /**
   * @brief Strided Batched GEMM
   * @param[in] arrayA The array containing the A matrix data
   * @param[in] arrayB The array containing the B matrix data
   * @param[in] arrayC The array containing the C matrix data
   * @param[in] native Boolean to use either the native or generic version
   * @param[in] param The data defining the problem execution.
   */
  void blasGEMMQuda(void *arrayA, void *arrayB, void *arrayC, QudaBoolean native, QudaBLASParam *param);

  /**
   * @brief Strided Batched in-place matrix inversion via LU
   * @param[in] Ainv The array containing the A inverse matrix data
   * @param[in] A The array containing the A matrix data
   * @param[in] use_native Boolean to use either the native or generic version
   * @param[in] param The data defining the problem execution.
   */
  void blasLUInvQuda(void *Ainv, void *A, QudaBoolean use_native, QudaBLASParam *param);
  
  void setMPICommHandleQuda(void *mycomm);

  /**
   * First test function in DSA
   * 
   * Perform a vector scaling
   * 
   * @param[in]  sf The scaling factor
   * @param[in]  data The data to be scaled
   * @param[in]  N The number of elements.
   * @param[in]  prec The precision of the input elements.
   */
  void vectorScaleCHPC(double sf, void *data, unsigned long long int N, QudaPrecision prec);

  /**
   * Second test function in DSA
   * 
   * Perform a vector reduction (FIXME: add transform options (max, min, etc)
   * 
   * @param[in]  data The data to be reduced
   * @param[in]  N The number of elements.
   * @param[in]  prec The precision of the input elements.
   * @param[ret] The reduction result of the elements in the array.
   */
  double vector_reduce(void *data, unsigned long long int N, QudaPrecision prec);

  /**
   * Perform the X_engine computation
   * 
   * @param[out]  output_data The data where the result is written
   * @param[in]  input_data The data to be correlated
   * @param[in]  param Param structure that describes the operation
   */
  void XEngineCHPC(void *output_data, void *input_data, XEngineParam *param);

  /**
   * Perform the voltage based beamformer computation
   * 
   * @param[out]  output_data The data where the result is written
   * @param[in]  input_data The data from which to beamform
   * @param[in]  freqs The frequency widths
   * @param[in]  weights_A The voltage weights to apply for pol A
   * @param[in]  weights_B The voltage weights to apply for pol B
   * @param[in]  flagants flag invalid antennae
   * @param[in]  param Param structure that describes the operation
   */
  void beamformerVoltageCHPC(void *output_data, void *input_data, void *freqs, void *weights_A, void *weights_B, void *flagants, BeamformerParam *param);
  
  /**
   * Perform the voltage based beamformer computation
   * 
   * @param[out]  output_data The data where the result is written
   * @param[in]  input_data The data from which to beamform
   * @param[in]  calibrations Positions of antennae and beam pointing data
   * @param[in]  flagants Flag invalid antennae
   * @param[in]  param Param structure that describes the operation
   */
  void beamformerVoltageDsa110CHPC(void *output_data, void *input_data, void *calibrations, void *flagants, BeamformerParam *param);

  /**
   * Perform the visibility based beamformer computation
   * 
   * @param[out]  output_data The data where the result is written
   * @param[in]  input_data The data from which to beamform
   * @param[in]  freqs The frequency widths
   * @param[in]  weights_A The voltage weights to apply for pol A
   * @param[in]  weights_B The voltage weights to apply for pol B
   * @param[in]  flagants flag valid antennae
   * @param[in]  param Param structure that describes the operation
   */
  void beamformerVisibilityCHPC(void *output_data, void *input_data, void *freqs, void *weights_A, void *weights_B, void *flagants, BeamformerParam *param);

  /**
   * Perform dedispersion on filterbanks
   *
   * @param[out]  output_data The data where the result is written
   * @param[in]  input_data The data from which to beamform
   * @param[in]  param Param structure that describes the dedispersion
   */
  void dedispersionCHPC(void *output_data, void *input_data, DedispParam *param);
  
  
#ifdef __cplusplus
}
#endif

// remove NVRTC WAR
#undef double_complex

/* #include <quda_new_interface.h> */
