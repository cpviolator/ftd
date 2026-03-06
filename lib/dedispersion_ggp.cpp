#include <ggp_internal.h>
#include <timer.h>
#include <algorithm.h>
#include <blas_lapack.h>
#include <dedisp_interface.h>
#include <dsa.h>

using namespace quda;
using namespace device;
using namespace blas_lapack;

namespace ggp {

  // Dedispersion class constructor
  Dedispersion::Dedispersion(DedispParam &param) {
    getProfile().TPSTART(QUDA_PROFILE_INIT);

    dedisp_lib = param.dedisp_lib;

    switch(dedisp_lib) {
    case CHPC_DEDISP_DEDISPERSION_LIB:
      dedisp_create_plan(param);
      break;
    case CHPC_ASTROACCEL_DEDISPERSION_LIB:
      break;
    default: errorQuda("Unknown dedispersion library %d", dedisp_lib);
    }
  }
  
  void Dedispersion::compute(void *output_data_host, void *input_data_host, DedispParam &param) {
    
    switch(dedisp_lib) {
    case CHPC_DEDISP_DEDISPERSION_LIB:
      dedisp_interface_func(output_data_host, input_data_host, param);
      break;
    case CHPC_ASTROACCEL_DEDISPERSION_LIB:
      break;
    default: errorQuda("Unknown dedispersion library %d", dedisp_lib);
    }
    
  }
  
  Dedispersion::~Dedispersion() {
  }
}
  
