#!/bin/bash

make -j

EXE=1d_r2c_c2r_bench

rm -rf logs
mkdir logs

#     for B in 1 2 4 8 16 32 64 128 256 512 1024 2048 4096 8192 16384; do
for NX in 8 10 12 14 16 18 20 22 24 26 27; do
    for B in 1 2 4 8 16 31 63 127 255 511 1023 2047 4095 8191 16383 32767 65535 131071; do
	./${EXE} ${NX} ${B} 0 0 0 > logs/${EXE}_${NX}_${B}_cuFFT.log
	./${EXE} ${NX} ${B} 1 0 0 > logs/${EXE}_${NX}_${B}_cuFFTXt.log
	echo "Done ${NX} ${B}"
    done
done

EXE=2d_c2r_r2c_bench

for NX in 6 8 10 11 12 13 14; do
    for B in 1 2 4 8 15 31 63 127 255 511 1023 2047 4095 8191 16383; do
	./${EXE} ${NX} ${NX} ${B} 0 0 0 > logs/${EXE}_${NX}_${NX}_${B}_cuFFT.log
	./${EXE} ${NX} ${NX} ${B} 1 0 0 > logs/${EXE}_${NX}_${NX}_${B}_cuFFTXt.log
	echo "Done ${NX} ${NX} ${B}"
    done
done
