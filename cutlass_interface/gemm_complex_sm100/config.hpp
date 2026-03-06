// ========================================================================================
// SM100/SM120 Configuration — Shared enums + SM100-specific extensions
// ========================================================================================

#include "../shared/config_common.hpp"

// ========================================================================================
// SM100-Specific Configuration
// ========================================================================================

/// Compute precision for sub-GEMMs.
/// FP8_E4M3 is always available. FP6 and FP4 require compile-time gates.
enum class ComputePrecision {
    FP8_E4M3,       // cutlass::float_e4m3_t — 1 byte/element, range ±448 (default)
#ifdef COMPLEX_SM100_ENABLE_FP6
    FP6_E3M2,       // 6-bit float — 0.75 bytes/element, range ±28
    FP6_E2M3,       // 6-bit float — 0.75 bytes/element, range ±7.5
#endif
#ifdef COMPLEX_SM100_ENABLE_FP4
    FP4_E2M1,       // 4-bit float — 0.5 bytes/element, range ±6
#endif
};

// GemmConfig enum, config_name(), select_gemm_config(), is_config_valid_for_arch(),
// and all_baseline_configs() are defined in shared/config_common.hpp (included above).

/// Extra CUTLASS/runtime parameters for BLAS3-style wrappers (SM100/SM120)
struct CutlassParams {
    cudaStream_t stream = nullptr;
    HerkStrategy herk_strategy = HerkStrategy::Baseline;
    ComputePrecision precision = ComputePrecision::FP8_E4M3;
    GemmConfig config = GemmConfig::Default;
    TriangleConfig triangle_config;
};
