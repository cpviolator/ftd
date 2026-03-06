#set terminal dumb 120 40
#set terminal postscript enhanced
set terminal pngcairo enhanced
set output '__TITLE__.png'

set fit quiet
set fit logfile '/dev/null'

set xlabel "__XLABEL__" font ",24"
set ylabel "TFLOPS" font ",24"
set logscale x 2

set title 'N_X=__NX__ N_Y=__NY__ (Array of 2^{N_{dim}})' 
set key off

M=1.0
B=1
C=10
func1(x)=(C*atan(x*B + M))
fit [0.1:16384] func1(x) "../data___ND__d/__TITLE___cuFFT_data.dat" using 1:2 via B, C, M
MAX = func1(1e12)
FIT_CHI = FIT_STDFIT*FIT_STDFIT

MXt=1.0
BXt=1
CXt=10
func1Xt(x)=(CXt*atan(x*BXt + MXt))
fit [3:131072] func1Xt(x) "../data___ND__d/__TITLE___cuFFTXt_data.dat" using 1:2 via BXt, CXt, MXt
MAXXt = func1Xt(1e12)
FIT_CHIXt = FIT_STDFIT*FIT_STDFIT

set label sprintf("  cuFFT: {/Symbol c}^2=%.6e MAX TFLOPS=%.6e\ncuFFTXt: {/Symbol c}^2=%.6e MAXXt TFLOPS=%.6e\n", FIT_CHI, MAX, FIT_CHIXt, MAXXt) at 0.25, 28

set key center right
   
plot [0.1:131072][0:30] "../data___ND__d/__TITLE___cuFFT_data.dat" using __USING__ t "", "../data___ND__d/__TITLE___cuFFTXt_data.dat" using __USING__ t"" , func1(x) t "fit" , func1Xt(x) t "fitXt"

#set label sprintf("fit = const0 * atan(batches*const1 + const2)\n{/Symbol c}^2=%.6e\nMAX TFLOPS=%.6e\nconst0=%.6e\nconst1=%.6e\nconst2=%.6e", FIT_STDFIT*FIT_STDFIT, MAX, C, B, M) at 0.25, 28
