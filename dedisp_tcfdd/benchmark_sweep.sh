#!/bin/bash
# Performance sweep: CUTLASS vs cuBLASLt across problem sizes and precisions
# Output: CSV-formatted results for analysis

BIN="/home/deanhowarth/spark_cutlass_bench/cutlass_sandbox/blackwell/dedisp/build/gpu_fdd_improved_cuda13x"
OUTDIR="/home/deanhowarth/spark_cutlass_bench/cutlass_sandbox/blackwell/dedisp/bench_results"
mkdir -p "$OUTDIR"

CSV="$OUTDIR/results.csv"
echo "mode,Nf,Ndm,Nt,batch,fft_fwd_ms,transpose1_ms,gemm_ms,gemm_gflops,transpose2_ms,ifft_ms,total_exec_ms,per_item_ms,precomp_ms" > "$CSV"

MODES="cublas_lt_fp16 cublas_lt_fp8 cutlass cutlass_fp6 cutlass_fp4"

# Problem size sweep (square: Nf=Ndm, Nt=Nf)
SIZES="256 512 1024 2048"
BATCH=32

# Also test batch variation at Nf=Ndm=Nt=512
BATCH_SIZES="32 64 128"

run_benchmark() {
    local mode=$1
    local nf=$2
    local ndm=$3
    local nt=$4
    local batch=$5
    local tag="${mode}_${nf}x${ndm}x${nt}_b${batch}"
    local logfile="$OUTDIR/${tag}.log"

    echo "  Running: mode=$mode Nf=$nf Ndm=$ndm Nt=$nt batch=$batch"

    # Run with timeout (5 min max per run)
    timeout 300 "$BIN" \
        --algorithm fdd-gemm-batched \
        --precision single \
        --num-freq-channels "$nf" \
        --num-time-samples "$nt" \
        --num-dm-trials "$ndm" \
        --batch-size "$batch" \
        --fdd-mode "$mode" \
        --seed 1234 \
        2>&1 > "$logfile"

    local rc=$?
    if [ $rc -ne 0 ]; then
        echo "    FAILED (rc=$rc)"
        echo "$mode,$nf,$ndm,$nt,$batch,FAIL,FAIL,FAIL,FAIL,FAIL,FAIL,FAIL,FAIL,FAIL" >> "$CSV"
        return
    fi

    # Parse metrics from log
    local fft_fwd=$(grep -A1 "Forward FFT" "$logfile" | grep "Time:" | awk '{print $2}')
    local tr1=$(grep -A1 "Transpose 1" "$logfile" | grep "Time:" | awk '{print $2}')
    local gemm_ms=$(grep "GEMM:" "$logfile" | awk '{print $2}')
    local gemm_gflops=$(grep -A1 "GEMM:" "$logfile" | grep "Compute:" | awk '{print $2}')
    local tr2=$(grep -A1 "Transpose 2" "$logfile" | grep "Time:" | head -1 | awk '{print $2}')
    local ifft=$(grep -A1 "Inverse FFT" "$logfile" | grep "Time:" | awk '{print $2}')
    local total=$(grep "Total Execution took" "$logfile" | awk '{print $4}')
    local per_item=$(grep "Per item took" "$logfile" | awk '{print $4}')
    local precomp=$(grep "Pre-computation took" "$logfile" | awk '{print $3}')

    # For CUTLASS modes, GEMM line may say different things
    # Check for "CUTLASS" path or "cuBLAS" path
    if [ -z "$gemm_ms" ]; then
        # Try CUTLASS-specific format
        gemm_ms=$(grep "Time:" "$logfile" | grep -v "Forward\|Inverse\|Transpose" | head -1 | awk '{print $2}')
    fi

    # Some modes show "Setup:" + "GEMM:" separately; some show combined
    # Also check for just a compute line after the GEMM section
    if [ -z "$gemm_gflops" ]; then
        gemm_gflops=$(grep "Compute:" "$logfile" | grep -v "Forward\|Inverse" | head -1 | awk '{print $2}')
    fi

    echo "    GEMM: ${gemm_ms:-N/A} ms, ${gemm_gflops:-N/A} GFLOPS, Total: ${total:-N/A} ms"
    echo "$mode,$nf,$ndm,$nt,$batch,${fft_fwd:-N/A},${tr1:-N/A},${gemm_ms:-N/A},${gemm_gflops:-N/A},${tr2:-N/A},${ifft:-N/A},${total:-N/A},${per_item:-N/A},${precomp:-N/A}" >> "$CSV"
}

echo "=== Problem Size Sweep (batch=$BATCH) ==="
for size in $SIZES; do
    echo "--- Size: ${size}x${size}x${size} ---"
    for mode in $MODES; do
        run_benchmark "$mode" "$size" "$size" "$size" "$BATCH"
    done
done

echo ""
echo "=== Batch Size Sweep (Nf=Ndm=Nt=512) ==="
for batch in $BATCH_SIZES; do
    echo "--- Batch: $batch ---"
    for mode in $MODES; do
        run_benchmark "$mode" 512 512 512 "$batch"
    done
done

echo ""
echo "=== Asymmetric Problem Sizes (batch=32) ==="
# Wide N (many DM trials): Nf=512, Ndm=2048, Nt=512
for mode in $MODES; do
    run_benchmark "$mode" 512 2048 512 32
done
# Tall K (many freq channels): Nf=2048, Ndm=512, Nt=512
for mode in $MODES; do
    run_benchmark "$mode" 2048 512 512 32
done

echo ""
echo "Results written to $CSV"
echo "Logs in $OUTDIR/"
