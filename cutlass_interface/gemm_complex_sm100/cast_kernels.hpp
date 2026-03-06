// ========================================================================================
// SM100/SM120 Cast/Transpose Kernels — Shared kernels + SM100-specific aliases
// ========================================================================================

#include "../shared/cast_kernels_common.hpp"

// ========================================================================================
// SM100-specific wrapper aliases
// ========================================================================================
//
// SM100 dispatch/api/herk code references these functions with _sm100 suffix.
// They are identical to the shared unsuffixed versions.

inline void stack_fp8_sm100(
    const cutlass::float_e4m3_t* src1,
    const cutlass::float_e4m3_t* src2,
    cutlass::float_e4m3_t* dst,
    int M, int K, cudaStream_t stream = nullptr)
{
    stack_fp8(src1, src2, dst, M, K, stream);
}

inline void negate_and_stack_fp8_sm100(
    const cutlass::float_e4m3_t* src1,
    const cutlass::float_e4m3_t* src2_negated,
    cutlass::float_e4m3_t* dst,
    int M, int K, cudaStream_t stream = nullptr)
{
    negate_and_stack_fp8(src1, src2_negated, dst, M, K, stream);
}

inline void cast_fp16_to_fp8_e4m3_paired_sm100(
    const __half* input1, const __half* input2,
    __nv_fp8_e4m3* output1, __nv_fp8_e4m3* output2,
    int64_t num_elements,
    cudaStream_t stream = nullptr)
{
    cast_fp16_to_fp8_e4m3_paired(input1, input2, output1, output2, num_elements, stream);
}

inline void deinterleave_fp8_sm100(
    const __nv_fp8_e4m3* interleaved,
    __nv_fp8_e4m3* out_real, __nv_fp8_e4m3* out_imag,
    int64_t num_complex_elements,
    cudaStream_t stream = nullptr)
{
    deinterleave_fp8(interleaved, out_real, out_imag, num_complex_elements, stream);
}

inline void cast_fp16_to_fp8_e4m3_transposed_paired_sm100(
    const __half* input1, const __half* input2,
    cutlass::float_e4m3_t* output1, cutlass::float_e4m3_t* output2,
    int rows, int cols,
    cudaStream_t stream = nullptr,
    int batch_count = 1)
{
    cast_fp16_to_fp8_e4m3_transposed_paired(input1, input2, output1, output2,
                                             rows, cols, stream, batch_count);
}

inline void cast_fp16_planar_to_fp8_interleaved_sm100(
    const __half* re_in, const __half* im_in,
    __nv_fp8_e4m3* out,
    int64_t num_elements,
    cudaStream_t stream = nullptr)
{
    cast_fp16_planar_to_fp8_interleaved(re_in, im_in, out, num_elements, stream);
}

inline void cast_fp16_planar_to_fp8_interleaved_negate_im_sm100(
    const __half* re_in, const __half* im_in,
    __nv_fp8_e4m3* out,
    int64_t num_elements,
    cudaStream_t stream = nullptr)
{
    cast_fp16_planar_to_fp8_interleaved_negate_im(re_in, im_in, out, num_elements, stream);
}

inline void qc_to_fp8_interleaved_polsplit_sm100(
    const uint8_t* qc_data,
    uint8_t* fp8_pol0, uint8_t* fp8_pol1,
    int n_ant, int n_time, int n_ch,
    cudaStream_t stream = nullptr)
{
    qc_to_fp8_interleaved_polsplit(qc_data, fp8_pol0, fp8_pol1, n_ant, n_time, n_ch, stream);
}

inline void qc_to_fp8_interleaved_transpose_sm100(
    const uint8_t* qc_pol,
    uint8_t* fp8_out,
    int n_ant, int n_time, int n_ch,
    cudaStream_t stream = nullptr)
{
    qc_to_fp8_interleaved_transpose(qc_pol, fp8_out, n_ant, n_time, n_ch, stream);
}
