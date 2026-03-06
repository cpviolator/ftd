#include <string.h>
#include <typeinfo>  

#include "data_structures.h"

namespace ggp
{
  void PipelineArrays::destroy() {
    if(init_volt_array) volt_array->destroy();
    if(init_vis_array) vis_array->destroy();
    if(init_filter_bank) filter_bank->destroy();
  }
  
  void PipelineArrays::create_volt_array(VoltArrayParam &param) {
    volt_array = VoltArray::Create(param);
  }
  void PipelineArrays::create_vis_array(VisArrayParam &param) {
    vis_array = VisArray::Create(param);
  }
  void PipelineArrays::create_filter_bank(FilterBankParam &param) {
    filter_bank = FilterBank::Create(param);
  }
  
  PipelineArrays::~PipelineArrays() { destroy(); }
}
