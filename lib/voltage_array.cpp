#include <string.h>
#include <typeinfo>  

#include "data_structures.h"

namespace ggp
{

  VoltArray::VoltArray(const VoltArrayParam &param) {
    create(param);
  }
  
  void VoltArray::create(const VoltArrayParam &param)
  {
    if (param.create_type == QUDA_INVALID_FIELD_CREATE) errorQuda("Create type not set in VoltArrayParam");

    n_payload = param.n_payload;
    n_antennae_per_payload = param.n_antennae_per_payload;
    n_channels_per_payload = param.n_channels_per_payload;
    n_time_per_payload = param.n_time_per_payload;
    n_time_inner = param.n_time_inner;
    n_polarizations = param.n_polarizations;

    location = param.location;

    storage_prec = param.storage_prec;
    compute_prec = param.compute_prec;
    create_type = param.create_type;
    
    uint data_size = 0;
    switch(storage_prec) {
    case QUDA_DOUBLE_PRECISION: data_size = 8; break;
    case QUDA_SINGLE_PRECISION: data_size = 4; break;
    default: errorQuda("Unsupported data size %d in VoltArray", data_size);
    }
          
    uint64_t n_bytes = n_payload * n_antennae_per_payload * n_channels_per_payload * n_time_per_payload * n_polarizations * 2 * data_size;
    if(create_type == QUDA_ZERO_FIELD_CREATE || create_type == QUDA_NULL_FIELD_CREATE)
      data_ptr = quda::quda_ptr(location == QUDA_CUDA_FIELD_LOCATION ? QUDA_MEMORY_DEVICE : QUDA_MEMORY_HOST, n_bytes);
    else errorQuda("VoltArray create type not implemented yet");
  }
  
  void VoltArray::destroy() {
    
  }
  
  VoltArray::~VoltArray() { destroy(); }  
}
