#!/bin/bash

EXE=1d_r2c_c2r_bench

rm -rf plots_1d
mkdir plots_1d

# 1D FFT
for NX in 8 10 12 14 16 18 20 22 24 26 27; do
    TITLE=${EXE}_${NX}
    FILE=data_1d/${EXE}_${NX}_cuFFT_data.dat
    if [ -f "$FILE" ]; then
	cp plot_template.p plots_1d/${TITLE}.p
	sed -i "s/__ND__/1/g" plots_1d/${TITLE}.p
	sed -i "s/__TITLE__/${TITLE}/g" plots_1d/${TITLE}.p
	sed -i "s/__USING__/1:2/g" plots_1d/${TITLE}.p
	sed -i "s/__XLABEL__/Batches/g" plots_1d/${TITLE}.p
	sed -i "s/__NX__/${NX}/g" plots_1d/${TITLE}.p
	sed -i "s/__NY__/0/g" plots_1d/${TITLE}.p
	(cd plots_1d; gnuplot ${TITLE}.p)
    fi
done

EXE=2d_c2r_r2c_bench

rm -rf plots_2d
mkdir plots_2d

# 2D FFT
for NX in 6 8 10 11 12 13 14; do
    TITLE=${EXE}_${NX}_${NX}
    FILE=data_2d/${EXE}_${NX}_${NX}_cuFFT_data.dat
    if [ -f "$FILE" ]; then
	cp plot_template.p plots_2d/${TITLE}.p
	sed -i "s/__ND__/2/g" plots_2d/${TITLE}.p
	sed -i "s/__TITLE__/${TITLE}/g" plots_2d/${TITLE}.p
	sed -i "s/__USING__/1:2/g" plots_2d/${TITLE}.p
	sed -i "s/__XLABEL__/Batches/g" plots_2d/${TITLE}.p
	sed -i "s/__NX__/${NX}/g" plots_2d/${TITLE}.p
	sed -i "s/__NY__/${NX}/g" plots_2d/${TITLE}.p
	(cd plots_2d; gnuplot ${TITLE}.p)
    fi
done

#------------------------
