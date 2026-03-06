#!/bin/bash -l 

export QUDA_ENABLE_TUNING=1
export QUDA_RESOURCE_PATH=.

SANITIZE="compute-sanitizer"
SANITIZE=""

NCPP=384
NPPB=4096
NP=2
NA=96

PREC="double"
${SANITIZE} ./FXcorrelator --compute-prec ${PREC} --output-prec ${PREC} --verbosity verbose --FX-n-channels-per-packet ${NCPP} --FX-n-packets-per-block ${NPPB} --FX-n-polarizations ${NP} --FX-n-antennae ${NA} --FX-mat-format herm

PREC="single"
${SANITIZE} ./FXcorrelator --compute-prec ${PREC} --output-prec ${PREC} --verbosity verbose --FX-n-channels-per-packet ${NCPP} --FX-n-packets-per-block ${NPPB} --FX-n-polarizations ${NP} --FX-n-antennae ${NA} --FX-mat-format herm

