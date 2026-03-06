# This is set up for the CORR containers
CC=gcc
#CC=gcc
CFLAGS1 = -g -O3 -Wall -pthread -march=native -I/usr/local/include -I/usr/local/include/src -I/usr/local/cfitsio-3.47/include/ -I/home/ubuntu/proj/dsa110-shell/dsa110-sigproc
CDEPS1=dsaX_def.h dsaX_capture_manythread.h
CDEPS2=dsaX_def.h dsaX_capture.h

GGP_LIB=/home/dmhowart/install/lib/libggp.so
GGP_INC=-I/home/dmhowart/install/include

DEDISP_LIB=/home/dmhowart/DSA110/generic-GPU-project_FXC_PR/build/dedisp/lib/libdedisp.a
DEDISP_INC=-I/home/dmhowart/DSA110/generic-GPU-project_FXC_PR/build/dedisp/include

#CUDA_HOME=/fastpool/dmhowart/spack/opt/spack/linux-rhel8-zen2/gcc-12.4.0/cuda-12.5.1-pxjfj4binedoohvy4meosck2i2qpfzrm/
#CUDA_HOME=/usr/local/cuda
NVCC=${CUDA_HOME}/bin/nvcc -D CUDA -ccbin=g++
#THRUST_INC=-I${CUDA_HOME}/include/thrust

LIBS = -L/usr/local/lib -lpsrdada -L/usr/lib/gcc/x86_64-linux-gnu/5 -lgfortran -L${CUDA_HOME}/lib64 -lcudart -lcublas -lm -L/usr/local/cfitsio-3.47/lib -lxgpu -lnppc -lnppial -lnppicc -lnppidei -lnppif -lnppig -lnppim -lnppist -lnppisu -lnppi\
tc -L/home/dmhowart/applications/psrdada_build/src/ -lpsrdada #${GGP_LIB} #${DEDISP_LIB} -lcfitsio -lsigproc

LIBS2 = -L/home/ubuntu/PF_RING/userland/libpcap-1.9.1 -lpcap
CDEPS3=dsaX_def.h dsaX_capture_pcap.h

CFLAGS2=-I/home/ubuntu/proj/dsa110-shell/dsa110-xengine/src -I/home/ubuntu/proj/dsa110-shell/dsa110-xGPU/src -I/usr/local/include -I/usr/local/include/src -I/usr/local/cfitsio-3.47/include -I/home/ubuntu/proj/dsa110-shell/dsa110-sigproc -arch\
=sm_75 -O3 -Xcompiler="-pthread" -DMATRIX_ORDER_TRIANGULAR -std=c++14 -I${CUDA_HOME}/include -I${CUDA_HOME}/samples/common/inc #${THRUST_INC}

#HELLA_FLAGS=${DEDISP_INC} ${THRUST_INC} #${GGP_INC}

BF_LIBS=-L${CUDA_HOME}/lib64 -lcudart -lcublas -L/data/dmhowart/install/psrdada/lib -lpsrdada ${GGP_LIB}
BF_FLAGS=-arch=sm_89 -O3 -std=c++14 -I${CUDA_HOME}/include -I/data/dmhowart/install/psrdada/include ${GGP_INC}

HELLA_LIBS=-L${CUDA_HOME}/lib64 -lcudart -lcublas -lnppc -lnppial -lnppicc -lnppidei -lnppif -lnppig -lnppim -lnppist -lnppisu -lnppitc -L/home/dmhowart/applications/psrdada_build/src/ -lpsrdada ${DEDISP_LIB}
HELLA_FLAGS=-arch=sm_80 -O3 -std=c++14 -I${CUDA_HOME}/include

.DEFAULT_GOAL := all

dsaX_bfCorr: dsaX_bfCorr.cu
	$(NVCC) -o $@ $^ $(BF_FLAGS) $(BF_LIBS)

dsaX_bfCorr_111: dsaX_bfCorr_111.cu
	$(NVCC) -o $@ $^ $(BF_FLAGS) $(BF_LIBS)

dsaX_bfCorr_reduced: dsaX_bfCorr_reduced.cu
	$(NVCC) -o $@ $^ $(BF_FLAGS) $(BF_LIBS)

dsaX_hella: dsaX_hella.cu
	$(NVCC) -o $@ $^ $(HELLA_FLAGS) $(HELLA_LIBS)

dsaX_beamformer_passon: dsaX_beamformer_passon.cu
	$(NVCC) -o $@ $^ $(CFLAGS2) $(LIBS)

dsaX_beamformer_offline: dsaX_beamformer_offline.cu
	$(NVCC) -o $@ $^ $(CFLAGS2) $(LIBS)

.PHONY: clean all

clean:
	rm -f *.o *~ dsaX_beamformer dsaX_beamformer_passon dsaX_xgpu dsaX_reorder_raw dsaX_writeFil dsaX_writevis dsaX_fake dsaX_capture dsaX_dbnic dsaX_nicdb dsaX_split dsaX_wrangle fil2dada gpu_flagger dumpfil dsaX_simplesplit dsaX_store d\
saX_trigger dsaX_beamformer_offline dsaX_splice dsaX_filTrigger cuda_correlator dsaX_copydb dsaX_bfCorr dsaX_merge dsaX_hella dsaX_injector dsaX_nsfrb dsaX_filTrigger_twoInput

all: dsaX_beamformer dsaX_beamformer_passon dsaX_xgpu dsaX_reorder_raw dsaX_writeFil dsaX_writevis dsaX_fake dsaX_capture dsaX_capture_thread dsaX_capture_manythread dsaX_dbnic dsaX_nicdb dsaX_split dsaX_wrangle fil2dada gpu_flagger dumpfil d\
saX_simplesplit dsaX_store dsaX_trigger dsaX_filTrigger dsaX_beamformer_offline dsaX_splice dsaX_splitup cuda_correlator dsaX_copydb dsaX_bfCorr dsaX_merge dsaX_hella dsaX_injector dsaX_nsfrb dsaX_filTrigger_twoInput
