#include <string.h>
#include <typeinfo>  

#include "data_structures.h"

namespace ggp
{

  VisArray::VisArray(const VisArrayParam &param) {
    create(param);
  }
  
  void VisArray::create(const VisArrayParam &param)
  {
    if (param.create_type == QUDA_INVALID_FIELD_CREATE) errorQuda("Create type not set in VisArrayParam");
    
    n_base = param.n_base;
    n_channels = param.n_channels;
    n_time = param.n_time;
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
    default: errorQuda("Unsupported data size %d in VisArray", data_size);
    }
          
    uint64_t n_bytes = n_base * n_channels * n_time * n_polarizations * 2 * data_size;
    if(create_type == QUDA_ZERO_FIELD_CREATE || create_type == QUDA_NULL_FIELD_CREATE)
      data_ptr = quda::quda_ptr(location == QUDA_CUDA_FIELD_LOCATION ? QUDA_MEMORY_DEVICE : QUDA_MEMORY_HOST, n_bytes);
    else errorQuda("VisArray create type not implemented yet");
  }
  
  void VisArray::destroy() {
    
  }
  
  VisArray::~VisArray() { destroy(); }  
  
}
