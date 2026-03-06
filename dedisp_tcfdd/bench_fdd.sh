#!/bin/bash

#rm -rf logs_temp
#mkdir logs_temp

EXE=gpu_fdd_improved_cuda13x

CONST="--algorithm fdd-gemm-batched --f-min-MHz 700 --f-max-MHz 1500 --batch-size 32 --num-candidates-to-find 3 --precision single --seed 1234 --total-obs-time-s 1.0 --min-amplitude 4.0 --max-amplitude 24.0 --noise-mean 2.0 --noise-stddev 4.0 --num-pipelines 31 --fdd-mode cublas_lt_fp8"

for T_SAMP in {64..1024..64}
do
    for N_FREQ in {512..4096..512}
    do  
        for N_DM in {512..4096..512}
	do  
	    COMMAND="./${EXE} ${CONST} --num-time-samples ${T_SAMP} --num-freq-channels ${N_FREQ} --num-dm-trials ${N_DM}"
	    echo ${COMMAND}
	    ${COMMAND} >> logs_temp/T${T_SAMP}_F${N_FREQ}_DM${N_DM}.log
	done
    done
done
    
