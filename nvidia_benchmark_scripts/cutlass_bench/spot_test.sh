#!/bin/bash

EXE=/data/dmhowart/cuda_bench_package/superchip_bench/cutlass/build/tools/profiler/cutlass_profiler

declare -a K_vals=("16" "32" "64" "128" "256" "512" "1024" "2048" "4096" "8192" "16384")

for K in ${K_vals[@]}; do
    
    ${EXE} --gemm_kind=grouped_gemm --A=fe5m2:row --B=fe5m2:column --C=f16:column --D=f16:column --alpha=1 --beta=0  --raster_order=along_n --swizzle_size=2  --use_pdl=false --op_class=tensorop --accum=f32 --cta_m=128 --cta_n=128 --cta_k=128 --cluster_m=2 --cluster_n=1 --cluster_k=1  --cluster_m_fallback=1 --cluster_n_fallback=1 --cluster_k_fallback=1 --stages=6 --warps_m=4 --warps_n=1 --warps_k=1 --inst_m=64 --inst_n=128 --inst_k=32 --min_cc=90 --max_cc=90 --kernels=cutlass3x_sm90_tensorop_gemm_grouped_e5m2_e5m2_f32_f16_f16_128x128x128_2x1x1_0_tnn_align16_warpspecialized_pingpong_fp8_fastaccum_epi_tma --verification-enabled=false --warmup-iterations=4 --enable-best-kernel-for-fixed-shape=true --enable_sm90_mixed_dtype_shuffle_test=true --num_groups=128 --m=3328 --n=3328 --k=${K} --num_batches=8
done
