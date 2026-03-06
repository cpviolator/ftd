// ========================================================================================
// SM90 Configuration — Shared enums + SM90-specific extensions
// ========================================================================================

#include "../shared/config_common.hpp"

// ========================================================================================
// FP6 Forward Compatibility Placeholder
// ========================================================================================

/*
 * FP6 is not yet a standard type in CUTLASS 3.x.
 *
 * Blackwell (SM100) will support MXFP6 (block-scaled E3M2 and E2M3) via new
 * tensor core instructions. When CUTLASS adds support, the integration path will be:
 *
 *   // Future CUTLASS type (speculative):
 *   using ElementA_FP6 = cutlass::float_e3m2_t;          // 6-bit: 3 exp, 2 mantissa
 *   // Or with MX block scaling:
 *   using ElementA_MX6 = cutlass::mx_float6_e3m2_t;      // MX-scaled FP6
 *
 *   // The collective mainloop would change to handle the 6-bit packing:
 *   using CollectiveMainloop = cutlass::gemm::collective::CollectiveMma<
 *       cutlass::gemm::MainloopSm100MxTmaWarpSpecialized,  // SM100 MX-aware mainloop
 *       TileShape, ElementA_MX6, LayoutA, ElementB_MX6, LayoutB,
 *       ...>;
 *
 * For now, we provide this type alias as a placeholder. Users can swap it in when
 * official support lands.
 */
namespace fp6_placeholder {
    // Placeholder — will be replaced by cutlass::float_e3m2_t or similar
    struct float_e3m2_t {
        uint8_t storage;  // 6 bits used, 2 bits padding
        // In reality, CUTLASS will provide proper conversion operators
    };

    struct float_e2m3_t {
        uint8_t storage;
        // E2M3: 2 exponent bits, 3 mantissa bits — more precision, less range
    };
}  // namespace fp6_placeholder


// ========================================================================================
// SM90-Specific Configuration
// ========================================================================================

/// Extra CUTLASS/runtime parameters for BLAS3-style wrappers (SM90)
struct CutlassParams {
    cudaStream_t stream = nullptr;
    HerkStrategy herk_strategy = HerkStrategy::Baseline;
    GemmConfig config = GemmConfig::Default;
    TriangleConfig triangle_config;
};
