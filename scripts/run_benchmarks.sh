#!/usr/bin/env bash
# ============================================================================
# Production Beamformer Benchmark Suite
# ============================================================================
#
# Benchmarks the Voltage and Visibility beamformers at DSA-2000 production
# parameters using the bench_pipeline binary (preferred) or the test binaries
# as fallback.
#
# Usage:
#   bash run_benchmarks.sh [--build-dir DIR] [--suite voltbf|visbf|dedisp|all]
#                          [--runs N] [--warmup N] [--output-dir DIR]
#
# Output:
#   Raw logs in OUTPUT_DIR/raw_*.log
#   Parsed CSV via parse_benchmarks.sh
# ============================================================================

set -euo pipefail

# ---- Defaults ----
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FTD_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
BUILD_DIR="${BUILD_DIR:-/home/deanhowarth/spark_cutlass_bench/cutlass_sandbox/blackwell/ftd_build}"
IMAGING_BUILD_DIR="${IMAGING_BUILD_DIR:-$BUILD_DIR/tests}"
SUITE="all"
RUNS=10
WARMUP=3
OUTPUT_DIR="${OUTPUT_DIR:-$SCRIPT_DIR/bench_results}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# ---- Parse arguments ----
while [[ $# -gt 0 ]]; do
    case "$1" in
        --build-dir)  BUILD_DIR="$2"; shift 2 ;;
        --suite)      SUITE="$2"; shift 2 ;;
        --runs)       RUNS="$2"; shift 2 ;;
        --warmup)     WARMUP="$2"; shift 2 ;;
        --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
        --help|-h)
            echo "Usage: $0 [--build-dir DIR] [--suite voltbf|visbf|dedisp|all] [--runs N] [--warmup N] [--output-dir DIR]"
            exit 0 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

mkdir -p "$OUTPUT_DIR"

# ---- GPU info ----
echo "================================================================"
echo "  Production Beamformer Benchmark Suite"
echo "  $(date)"
echo "  Build: $BUILD_DIR"
echo "  Output: $OUTPUT_DIR"
echo "================================================================"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>/dev/null || true
echo ""

# ---- Locate binaries ----
BENCH_PIPELINE=""
VP_TEST=""
VBF_TEST=""

for d in "$IMAGING_BUILD_DIR" "$BUILD_DIR"; do
    if [[ -x "$d/bench_pipeline" ]]; then
        BENCH_PIPELINE="$d/bench_pipeline"
        break
    fi
done
if [[ -n "$BENCH_PIPELINE" ]]; then
    echo "Found bench_pipeline: $BENCH_PIPELINE"
fi
if [[ -x "$BUILD_DIR/tests/visibility_pipeline_test" ]]; then
    VP_TEST="$BUILD_DIR/tests/visibility_pipeline_test"
    echo "Found visibility_pipeline_test: $VP_TEST"
fi
if [[ -x "$BUILD_DIR/tests/voltage_pipeline_test" ]]; then
    VBF_TEST="$BUILD_DIR/tests/voltage_pipeline_test"
    echo "Found voltage_pipeline_test: $VBF_TEST"
fi
echo ""

# ============================================================================
# DSA Chronoscope Band 2 — Production Parameters
# ============================================================================
#
# Visibility beamformer (XEngine / correlator):
#   HERK: N=1664 (antennae), K=128 (n_time=256, n_time_inner=2)
#   Batch = n_ch * 2 * n_time_inner (freq channels × pols)
#   Followed by 2D FP16 IFFT on 4096×4096 UV grid
#   n_beam=6000000 (beam pixels extracted from image)
#
# Voltage beamformer (INT4 GEMM path):
#   QC INT4 transpose + pol-split -> gemm_prepared_power_int4() x2 (Stokes I)
#   GEMM: M=4000 (time), N=4000 (beams), K=1664 (antennae), batch=n_ch
#   n_tps=1 (no time power sum — raw FP32 power output)
#
# Dedispersion (Band 2, centered 1410 MHz, 122.8 kHz channels):
#   Production: n_beam=4000, n_ch=1600, n_dm=2000, n_time=9000
#   FDD GEMM: M=Nt_complex, N=n_dm=2000, K=n_ch=1600, batch=n_beam
#   n_time=9000 → Nt_padded=32768 → shift matrix B=210 GB (infeasible)
#   Benchmark uses n_time=256 (one pipeline chunk, B=6.6 GB)
#   f_min=1312 MHz, f_max=1508 MHz, max_dm=2600
#   dedisp_mode=both (CuBLAS_FP32 vs CUTLASS_FP8 comparison)
# ============================================================================

run_bench_pipeline() {
    local label="$1"; shift
    local logfile="$OUTPUT_DIR/raw_${label}_${TIMESTAMP}.log"
    echo "--- Running: $label ---"
    echo "  Command: $BENCH_PIPELINE $*"
    echo "  Log: $logfile"
    "$BENCH_PIPELINE" "$@" warmup=$WARMUP runs=$RUNS 2>&1 | tee "$logfile"
    echo ""
}

run_vp_test() {
    local label="$1"; shift
    local logfile="$OUTPUT_DIR/raw_${label}_${TIMESTAMP}.log"
    echo "--- Running: $label ---"
    echo "  Command: $VP_TEST $*"
    echo "  Log: $logfile"
    "$VP_TEST" "$@" 2>&1 | tee "$logfile"
    echo ""
}

run_vbf_test() {
    local label="$1"; shift
    local logfile="$OUTPUT_DIR/raw_${label}_${TIMESTAMP}.log"
    echo "--- Running: $label ---"
    echo "  Command: $VBF_TEST $*"
    echo "  Log: $logfile"
    "$VBF_TEST" "$@" 2>&1 | tee "$logfile"
    echo ""
}

# ============================================================================
# Path A: bench_pipeline (full per-stage timing)
# ============================================================================
if [[ -n "$BENCH_PIPELINE" ]]; then
    echo "========================================"
    echo " Using bench_pipeline (per-stage timing)"
    echo "========================================"
    echo ""

    # ---- Visibility Beamformer Sweep ----
    if [[ "$SUITE" == "all" || "$SUITE" == "visbf" ]]; then
        echo "===== Visibility Beamformer: N=1664 ====="

        # Production config: HERK N=1664, K=128 (single channel)
        run_bench_pipeline "visbf_1ch_K128" \
            visbf n_ant=1664 n_ch=1 n_time=256 n_time_inner=2 \
            n_beam=6000000 Ng=4096 tune=true tune_verb=1

        # K sweep (K=64,256 at single channel)
        for K in 64 256; do
            N_TIME=$((K * 2))  # n_time_inner=2
            run_bench_pipeline "visbf_1ch_K${K}" \
                visbf n_ant=1664 n_ch=1 n_time=$N_TIME n_time_inner=2 \
                n_beam=6000000 Ng=4096 tune=true tune_verb=1
        done

        # Batch size sweep (n_ch scales HERK batch and imaging channels)
        # herk_batch = n_ch * 2 * n_time_inner = n_ch * 4
        for NCH in 2 4 8 16 32; do
            run_bench_pipeline "visbf_${NCH}ch_K128" \
                visbf n_ant=1664 n_ch=$NCH n_time=256 n_time_inner=2 \
                n_beam=6000000 Ng=4096 tune=true tune_verb=1
        done
    fi

    # ---- Voltage Beamformer Sweep (INT4 GEMM) ----
    if [[ "$SUITE" == "all" || "$SUITE" == "voltbf" ]]; then
        echo "===== Voltage Beamformer (INT4 GEMM): N=4096 beams, K=1664 ====="

        # Batch size sweep (n_ch = batch dimension for GEMM)
        # INT4 path: qc_transpose_polsplit -> gemm_prepared_power_int4() x2 pols
        # GEMM: M=n_time, N=n_beam=4096, K=n_ant=1664
        for NCH in 1 2 4 8 16 32; do
            run_bench_pipeline "voltbf_${NCH}ch" \
                voltbf n_ant=1664 n_beam=4096 n_ch=$NCH n_time=4000 n_tps=1 \
                tune=true tune_verb=1
        done

        # M-dimension sweep: increase n_time to saturate GPU compute.
        # GEMM: M=n_time, N=4096, K=1664, batch=1 (single channel).
        # Steps from 4096 to 16384 in increments of 2048.
        echo ""
        echo "===== VoltBF M-Sweep (N=4096, K=1664, batch=1) ====="
        for MT in 4096 6144 8192 10240 12288 14336 16384; do
            run_bench_pipeline "voltbf_M${MT}" \
                voltbf n_ant=1664 n_beam=4096 n_ch=1 n_time=$MT n_tps=1 \
                tune=true tune_verb=1
        done

        # ---- Short-Integration Fused Sweep ----
        # n_time=8 is deeply memory-bound (M=8 → 25% of tile, 0.6% TC util).
        # Channel fusion (ch_fuse) and payload batching (n_batch) increase M to
        # improve tensor core utilisation, trading batch dimension for M dimension.
        # B weights must be identical across fused channels (true for DSA-2000).
        #
        # The unfused baseline uses n_ch=200 (batch=200) so it fits in GPU memory
        # and matches the batch count of ch_fuse=8 (1600/8=200) for fair comparison.
        # Only M differs: 8 (unfused) vs 64 (ch_fuse=8) vs 512 (ch_fuse=8 × nb=8).
        #
        # Memory budget per config (C_power buffer = batch × M × N × 4 bytes):
        #   unfused(200):   200 × 8   × 4096 × 4 =   26 MB
        #   ch_fuse=8:      200 × 64  × 4096 × 4 =  210 MB
        #   +n_batch=2:     200 × 128 × 4096 × 4 =  419 MB
        #   +n_batch=4:     200 × 256 × 4096 × 4 =  839 MB
        #   +n_batch=8:     200 × 512 × 4096 × 4 = 1678 MB
        #   +n_batch=16:    200 × 1024× 4096 × 4 = 3355 MB
        echo ""
        echo "===== VoltBF Short-Integration Fused (n_time=8, n_ant=1664, n_beam=4096, n_ch=1600) ====="

        # Baseline: unfused M=8, n_ch=200 (same batch count as ch_fuse=8)
        # The unfused path at n_ch=1600 needs ~128 GB for weights — OOM on most GPUs.
        # Using n_ch=200 gives batch=200 which matches ch_fuse=8 (1600/8=200),
        # so the GEMM has identical batch count — only M differs (8 vs 64+).
        run_bench_pipeline "voltbf_short_unfused" \
            voltbf n_ant=1664 n_beam=4096 n_ch=200 n_time=8 n_tps=1 \
            tune=true tune_verb=1

        # ch_fuse sweep (payload batching = 1)
        for CF in 2 4 8; do
            run_bench_pipeline "voltbf_short_cf${CF}" \
                voltbf n_ant=1664 n_beam=4096 n_ch=1600 n_time=8 n_tps=1 \
                ch_fuse=$CF n_batch=1 \
                tune=true tune_verb=1
        done

        # ch_fuse=8 + payload batching sweep (saturate GB10 memory)
        # M = 8 × 8 × n_batch
        for NB in 2 4 8 16; do
            run_bench_pipeline "voltbf_short_cf8_nb${NB}" \
                voltbf n_ant=1664 n_beam=4096 n_ch=1600 n_time=8 n_tps=1 \
                ch_fuse=8 n_batch=$NB \
                tune=true tune_verb=1
        done
    fi

    # ---- Dedispersion Sweep ----
    # Production: Band 2, n_ch=1600, n_dm=2000
    # FDD GEMM: M=Nt_complex, N=n_dm, K=n_ch, batch=n_beam
    # Memory bottleneck: B buffer (shift matrix) = n_ch * n_dm * Nt_complex * 4 bytes
    #   n_time=9000 → Nt_padded=32768 → B=210 GB (infeasible)
    #   n_time=256  → Nt_padded=1024  → B=6.6 GB (fits)
    # Real pipeline chunks time; n_time=256 is one processing chunk.
    if [[ "$SUITE" == "all" || "$SUITE" == "dedisp" ]]; then
        echo "===== Dedispersion: Band 2 (n_ch=1600, n_dm=2000, chunked) ====="

        # Production channel/DM dimensions, chunked time
        run_bench_pipeline "dedisp_1600ch_2000dm_b64" \
            dedisp n_beam=64 n_ch=1600 n_time=256 n_dm=2000 \
            f_min_mhz=1312 f_max_mhz=1508 max_dm=2600 dedisp_mode=both \
            tune=true tune_verb=1

        # Batch sweep (n_beam = batch dimension)
        for NB in 16 32 128 256; do
            run_bench_pipeline "dedisp_1600ch_2000dm_b${NB}" \
                dedisp n_beam=$NB n_ch=1600 n_time=256 n_dm=2000 \
                f_min_mhz=1312 f_max_mhz=1508 max_dm=2600 dedisp_mode=both \
                tune=true tune_verb=1
        done

        # Sweep n_dm
        for DM in 512 1000; do
            run_bench_pipeline "dedisp_1600ch_${DM}dm_b64" \
                dedisp n_beam=64 n_ch=1600 n_time=256 n_dm=$DM \
                f_min_mhz=1312 f_max_mhz=1508 max_dm=2600 dedisp_mode=both \
                tune=true tune_verb=1
        done

        # Pipelined variant (CUTLASS only, pipeline=2 — each slot ~13 GB)
        run_bench_pipeline "dedisp_1600ch_2000dm_b64_pipe" \
            dedisp n_beam=64 n_ch=1600 n_time=256 n_dm=2000 \
            f_min_mhz=1312 f_max_mhz=1508 max_dm=2600 \
            dedisp_mode=cutlass n_payloads=8 pipeline=2 \
            tune=true tune_verb=1
    fi

    echo ""
    echo "All bench_pipeline runs complete."
    echo "Raw logs in: $OUTPUT_DIR/"
    echo "Run: bash $SCRIPT_DIR/parse_benchmarks.sh $OUTPUT_DIR"
    exit 0
fi

# ============================================================================
# Path B: Fallback to test binaries
# ============================================================================
echo "========================================"
echo " Using test binaries (event-timed loops)"
echo "========================================"
echo ""

# ---- Visibility Beamformer via visibility_pipeline_test ----
if [[ -n "$VP_TEST" ]] && [[ "$SUITE" == "all" || "$SUITE" == "visbf" ]]; then
    echo "===== Visibility Beamformer: N=1664 ====="

    for K in 64 128 256; do
        N_TIME=$((K * 2))  # n_time_inner=2, K = n_time / n_time_inner

        # Single channel
        run_vp_test "visbf_1ch_K${K}" \
            --enable-testing false --profile true \
            --VP-n-antennae 1664 --VP-n-channels 1 --VP-n-time $N_TIME \
            --VP-n-beams 256 --n-iters $RUNS

        # 8 channels
        run_vp_test "visbf_8ch_K${K}" \
            --enable-testing false --profile true \
            --VP-n-antennae 1664 --VP-n-channels 8 --VP-n-time $N_TIME \
            --VP-n-beams 256 --n-iters $RUNS
    done

    # Pipelined benchmark (single-channel, K=128)
    run_vp_test "visbf_1ch_K128_pipe" \
        --enable-testing false --profile true --pipelined true \
        --VP-n-antennae 1664 --VP-n-channels 1 --VP-n-time 256 \
        --VP-n-beams 256 --n-iters $RUNS
fi

# ---- Voltage Beamformer via voltage_pipeline_test ----
if [[ -n "$VBF_TEST" ]] && [[ "$SUITE" == "all" || "$SUITE" == "voltbf" ]]; then
    echo "===== Voltage Beamformer: N_ant=1664, N_beam=4096 ====="

    # Single channel, n_time=8
    run_vbf_test "voltbf_1ch_t8" \
        --enable-testing false --profile true \
        --VBF-n-antennae 1664 --VBF-n-beams 4096 \
        --VBF-n-channels 1 --VBF-n-time 8 --n-iters $RUNS

    # 8 channels (batched bunches)
    run_vbf_test "voltbf_8ch_t8" \
        --enable-testing false --profile true \
        --VBF-n-antennae 1664 --VBF-n-beams 4096 \
        --VBF-n-channels 8 --VBF-n-time 8 --n-iters $RUNS
fi

# ---- Dedispersion via dedisp_interface_test ----
DD_TEST=""
if [[ -x "$BUILD_DIR/tests/dedisp_interface_test" ]]; then
    DD_TEST="$BUILD_DIR/tests/dedisp_interface_test"
fi
if [[ -n "$DD_TEST" ]] && [[ "$SUITE" == "all" || "$SUITE" == "dedisp" ]]; then
    echo "===== Dedispersion (test binary) ====="

    run_logfile="$OUTPUT_DIR/raw_dedisp_test_${TIMESTAMP}.log"
    echo "--- Running: dedisp_test ---"
    echo "  Command: $DD_TEST --enable-testing false --profile true"
    echo "  Log: $run_logfile"
    "$DD_TEST" --enable-testing false --profile true 2>&1 | tee "$run_logfile"
    echo ""
fi

echo ""
echo "All benchmark runs complete."
echo "Raw logs in: $OUTPUT_DIR/"
echo "Run: bash $SCRIPT_DIR/parse_benchmarks.sh $OUTPUT_DIR"
