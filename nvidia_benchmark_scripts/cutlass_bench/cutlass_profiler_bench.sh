#!/bin/bash

declare -a M_vals=("512" "1024" "2048" "3328" "4096" "8192")
declare -a K_vals=("32" "64" "128" "256" "512" "1024" "2048" "4096" "8192")
# Batches
declare -a B_vals=("8")
# Groups
declare -a G_vals=("1" "2" "4" "8" "16" "32" "64" "128" "256" "512")

EXE=./cutlass/build/tools/profiler/cutlass_profiler

for M in ${M_vals[@]}; do
    for K in ${K_vals[@]}; do
	for B in ${B_vals[@]}; do
	    for G in ${G_vals[@]}; do
		COMMAND="${EXE} --m=${M} --n=${M} --k=${K} --batch_count=${B} --num_groups=${G} --A=fe5m2:row --B=fe5m2:column --C=f16:column --D=f16:column --output=cutlass_gemm_m${M}_n${M}_k${K}_batch${B}_group${G} --verification-enabled=false --warmup-iterations=2 --enable-best-kernel-for-fixed-shape=true --enable_sm90_mixed_dtype_shuffle_test=true --kernels=tensorop_gemm_grouped_e5m2_e5m2_f32_f16_f16 --operation=grouped_gemm"
		echo ${COMMAND}
		${COMMAND} > cutlass_gemm_m${M}_n${M}_k${K}_batch${B}_group${G}.log 
	    done
	done
    done
done
