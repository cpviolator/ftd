#pragma once

#include <ggp.h>
#if (defined DEDISP_LIB)
#include "../dedisp/include/dedisp.h"
#endif

namespace ggp {

  // Make me a class...
  void dedisp_create_plan(DedispParam &param);  
  void dedisp_interface_func(void *output, const void *input, const DedispParam &param);

}
