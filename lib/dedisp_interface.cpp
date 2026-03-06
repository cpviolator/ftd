#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <iostream>
#include <vector>
#include <algorithm>

#include <ggp_internal.h>
#include <dedisp_interface.h>

namespace ggp {

#if defined(DEDISP_LIB)

  dedisp_plan dedispersion_plan;
  bool init_plan = false;
  const float *DMs;

  /* Error codes for the dedisp library
   * DEDISP_NO_ERROR 
   * DEDISP_MEM_ALLOC_FAILED 
   * DEDISP_MEM_COPY_FAILED 
   * DEDISP_NCHANS_EXCEEDS_LIMIT 
   * DEDISP_INVALID_PLAN 
   * DEDISP_INVALID_POINTER 
   * DEDISP_INVALID_STRIDE 
   * DEDISP_NO_DM_LIST_SET 
   * DEDISP_TOO_FEW_NSAMPS 
   * DEDISP_INVALID_FLAG_COMBINATION 
   * DEDISP_UNSUPPORTED_IN_NBITS 
   * DEDISP_UNSUPPORTED_OUT_NBITS 
   * DEDISP_INVALID_DEVICE_INDEX 
   * DEDISP_DEVICE_ALREADY_SET 
   * DEDISP_PRIOR_GPU_ERROR 
   * DEDISP_INTERNAL_GPU_ERROR 
   * DEDISP_UNKNOWN_ERROR 
   */
  
  void get_dedisp_error(dedisp_error error) {
    
    switch(error) {
    case DEDISP_NO_ERROR:
      printfQuda("Dedisp ERROR 0: No error occurred.\n");
      break;
    case DEDISP_MEM_ALLOC_FAILED:
      printfQuda("Dedisp ERROR 1: A memory allocation failed.\n");
      break;
    case DEDISP_MEM_COPY_FAILED:
      printfQuda("Dedisp ERROR 2: A memory copy failed. This is often due to one of the arrays passed to dedisp_execute being too small.\n");
      break;
    case DEDISP_NCHANS_EXCEEDS_LIMIT:
      printfQuda("Dedisp ERROR 3: The number of channels exceeds the internal limit. The current limit is 8192.\n");
      break;
    case DEDISP_INVALID_PLAN:
      printfQuda("Dedisp ERROR 4: The given plan is NULL.\n");
      break;
    case DEDISP_INVALID_POINTER:
      printfQuda("Dedisp ERROR 5: A pointer is invalid, possibly NULL.\n");
      break;      
    case DEDISP_INVALID_STRIDE:
      printfQuda("Dedisp ERROR 6: A stride value is less than the corresponding dimension's size.\n");
      break;
    case DEDISP_NO_DM_LIST_SET:
      printfQuda("Dedisp ERROR 7: No DM list has yet been set using either ref dedisp_set_dm_list or ref dedisp_generate_dm_list.\n");
      break;
    case DEDISP_TOO_FEW_NSAMPS:
      printfQuda("Dedisp ERROR 8: The number of time samples is less than the maximum dedispersion delay.\n");
      break;
    case DEDISP_INVALID_FLAG_COMBINATION:
      printfQuda("Dedisp ERROR 9: Some of the given flags are incompatible.\n");
      break;
    case DEDISP_UNSUPPORTED_IN_NBITS:
      printfQuda("Dedisp ERROR 10: The given in_nbits value is not supported. See ref dedisp_execute for supported values.\n");
      break;
    case DEDISP_UNSUPPORTED_OUT_NBITS:
      printfQuda("Dedisp ERROR 11: The given out_nbits value is not supported. See ref dedisp_execute for supported values.\n");
      break;
    case DEDISP_INVALID_DEVICE_INDEX:
      printfQuda("Dedisp ERROR 12: The given device index does not correspond to a device in the system.\n");
      break;
    case DEDISP_DEVICE_ALREADY_SET:
      printfQuda("Dedisp ERROR 13: The device has already been set and cannot be changed. See ref dedisp_set_device for more info.\n");
      break;
    case DEDISP_PRIOR_GPU_ERROR:
      printfQuda("Dedisp ERROR 14: There was an existing GPU error prior to calling the function.\n");
      break;
    case DEDISP_INTERNAL_GPU_ERROR:
      printfQuda("Dedisp ERROR 15: An unexpected GPU error has occurred within the library. Please contact the authors if you get this error.\n");
      break;
    case DEDISP_UNKNOWN_ERROR:
      printfQuda("Dedisp ERROR 16: An unexpected error has occurred. Please contact the authors if you get this error.\n");
      break;
    default: errorQuda("Unknown dedisp_error code %d", error);
    }
  }
  

   void dedisp_create_plan(DedispParam &param) {
    if(init_plan) {
      errorQuda("Dedisp plan already created");
    }

    dedisp_size in_stride = param.in_stride; //NCHAN;// p->d_datapreT_step; //NCHAN * in_nbits/8;
    double val_1 = 262.144e-6;
    double val_2 = 1498.75;
    double val_3 = 0.244140625;
    
    double minDM = param.minDM;
    double maxDM = param.maxDM;

    int val40 = 40;
    double tolerance = param.tolerance;
    
    // set up DM plan
    dedisp_create_plan(&dedispersion_plan, in_stride, val_1, val_2, val_3);
    // generate DM list  
    dedisp_generate_dm_list(dedispersion_plan, minDM, maxDM, val40, tolerance);

    // Collect data from dedisp and deduce parameters
    DMs = dedisp_get_dm_list(dedispersion_plan);
    param.ndms = dedisp_get_dm_count(dedispersion_plan);    
    param.n_time_dd = param.n_time - dedisp_get_max_delay(dedispersion_plan);
    param.n_time_out = param.n_time_dd - param.max_width;
    param.n_time_dedisp = param.n_time_dd;
    
    // modify NTIME and ntime_dd in case of non-text input
#if 0
    int oo;
    if (param.inp_format == 0 || param.inp_format == 2) {
      param.n_time = param.gulp + dedisp_get_max_delay(dedispersion_plan) + param.max_width;
      oo = 32*((int)(param.n_time/32)+1);
      param.n_time = oo;
      param.n_time_dedisp = oo - dedisp_get_max_delay(dedispersion_plan);
      param.n_time_dd = param.gulp + param.max_width;
      param.n_time_out = param.gulp;
    }
#endif
    
    init_plan = true;
  }
  
  void dedisp_interface_func(void *output, const void *input, const DedispParam &param) {

    
    //cudaMemcpy(p->d_inputPacked, p->d_data+beam*NCHAN*p->NTIME, NCHAN*p->NTIME, cudaMemcpyDeviceToDevice);
    //cudaMemcpy(p->indata,p->d_data+beam*NCHAN*p->NTIME,NCHAN*p->NTIME,cudaMemcpyDeviceToHost);
    
    dedisp_error       derror;
    //const dedisp_byte* in = &((unsigned char *)(p->indata))[0];
    //dedisp_byte*       out = &((unsigned char *)(p->h_dedisp))[0];
    const dedisp_byte* in = (dedisp_byte*)input;
    dedisp_byte*       out = (dedisp_byte*)output;
    
    dedisp_size        in_nbits = param.in_nbits; //8;
    dedisp_size        in_stride = param.in_stride; //NCHAN;// p->d_datapreT_step; //NCHAN * in_nbits/8;
    dedisp_size        out_nbits = param.out_nbits; //32;
    dedisp_size        out_stride = param.out_stride; //p->ntime_dedisp * out_nbits/8;
    int                n_time = param.n_time;
    
    unsigned           flags = 1 << 2;
    derror = dedisp_execute_adv(dedispersion_plan, n_time,
				in, in_nbits, in_stride,
				out, out_nbits, out_stride,
				flags);
    
    if (derror != 0) {
      get_dedisp_error(derror);
    } else {
      logQuda(QUDA_SUMMARIZE, "Dedisp SUCCESS.\n");
      //cudaMemcpy2D(p->d_dedisp,p->d_dedisp_step,p->h_dedisp,4*p->ntime_dd,4*p->ntime_dd,p->ndms,cudaMemcpyHostToDevice);
      //cudaMemcpy2D(p->d_dedisp,p->d_dedisp_step,p->d_dedispPacked,4*p->ntime_dedisp,4*p->ntime_dd,p->ndms,cudaMemcpyDeviceToDevice);
    }
  }
  
#else
  void dedispError() {
    errorQuda("Dedisp has not been enabled. Recompile with GGP_DEDISP=ON and link or GGP_DOWNLOAD_DEDISP=ON to build and link automatically.");
  }
  
  void dedisp_create_plan(DedispParam &) {
    dedispError();
  }
  
  void dedisp_interface_func(void *, const void *, const DedispParam &){
    dedispError();
  }
#endif
}
