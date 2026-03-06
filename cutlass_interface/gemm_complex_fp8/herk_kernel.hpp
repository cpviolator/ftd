// ========================================================================================
// Direct HERK Kernel — SM90 Wrapper
// ========================================================================================
//
// Thin wrapper that includes the shared HERK kernel implementation.
// All kernel code is in shared/herk_kernel_common.hpp.
//
// This is a textual include (no #pragma once) — included inside namespace gemm_complex_fp8.
// ========================================================================================

#include "../shared/herk_kernel_common.hpp"
