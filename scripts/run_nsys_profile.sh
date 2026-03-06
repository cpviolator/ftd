#!/usr/bin/env bash
# ============================================================================
# Nsys GPU Profiling for FTD Pipelines
# ============================================================================
#
# Runs a minimal subset of pipeline configurations under `nsys profile` to
# capture definitive per-kernel GPU timing.  Exports kernel and memcpy stats
# to CSV via `nsys stats`.
#
# Usage:
#   bash run_nsys_profile.sh [--build-dir DIR] [--suite voltbf|visbf|all]
#                            [--output-dir DIR]
#
# Output:
#   nsys_<label>.nsys-rep           — Full Nsight Systems report
#   nsys_<label>_kernels.csv        — Per-kernel GPU time summary
#   nsys_<label>_memcpy.csv         — Per-operation memcpy time summary
#
# Requires: nsys (Nsight Systems CLI) on PATH.
# ============================================================================

set -euo pipefail

# ---- Defaults ----
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FTD_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
BUILD_DIR="${BUILD_DIR:-/home/deanhowarth/spark_cutlass_bench/cutlass_sandbox/blackwell/ftd_build}"
IMAGING_BUILD_DIR="${IMAGING_BUILD_DIR:-$BUILD_DIR/tests}"
SUITE="all"
OUTPUT_DIR="${OUTPUT_DIR:-$SCRIPT_DIR/bench_results}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# ---- Parse arguments ----
while [[ $# -gt 0 ]]; do
    case "$1" in
        --build-dir)  BUILD_DIR="$2"; shift 2 ;;
        --suite)      SUITE="$2"; shift 2 ;;
        --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
        --help|-h)
            echo "Usage: $0 [--build-dir DIR] [--suite voltbf|visbf|all] [--output-dir DIR]"
            exit 0 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

mkdir -p "$OUTPUT_DIR"

# ---- Check nsys ----
if ! command -v nsys &>/dev/null; then
    echo "Error: nsys not found on PATH. Install Nsight Systems or add it to PATH."
    echo "  e.g. export PATH=/opt/nvidia/nsight-systems/bin:\$PATH"
    exit 1
fi

echo "================================================================"
echo "  Nsys GPU Profiling — FTD Pipelines"
echo "  $(date)"
echo "  Build: $BUILD_DIR"
echo "  Output: $OUTPUT_DIR"
echo "  nsys version: $(nsys --version 2>&1 | head -1)"
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

if [[ -z "$BENCH_PIPELINE" && -z "$VP_TEST" && -z "$VBF_TEST" ]]; then
    echo "Error: No pipeline binaries found in $BUILD_DIR"
    exit 1
fi
echo ""

# ============================================================================
# Helper: run under nsys and export stats
# ============================================================================
nsys_run() {
    local label="$1"; shift
    local binary="$1"; shift

    local rep_file="$OUTPUT_DIR/nsys_${label}_${TIMESTAMP}"
    local kern_csv="$OUTPUT_DIR/nsys_${label}_${TIMESTAMP}_kernels.csv"
    local mem_csv="$OUTPUT_DIR/nsys_${label}_${TIMESTAMP}_memcpy.csv"

    echo "--- Profiling: $label ---"
    echo "  Binary: $binary"
    echo "  Args: $*"
    echo "  Report: ${rep_file}.nsys-rep"

    # Run under nsys profile
    nsys profile \
        --trace=cuda \
        --cuda-memory-usage=false \
        --force-overwrite=true \
        --output="$rep_file" \
        "$binary" "$@" \
        2>&1 | tail -5

    # Export kernel and memcpy stats
    # nsys stats --output DIR writes CSV to DIR/_<report_name>.csv (fixed name).
    # We rename immediately after each export to prevent the next run from
    # overwriting.  --force-export=true is needed when a stale .sqlite exists.
    if [[ -f "${rep_file}.nsys-rep" ]]; then
        local nsys_kern_default="$OUTPUT_DIR/_cuda_gpu_kern_sum.csv"
        local nsys_mem_default="$OUTPUT_DIR/_cuda_gpu_mem_time_sum.csv"
        local rep="${rep_file}.nsys-rep"

        # Remove stale default files to avoid picking up a previous run
        rm -f "$nsys_kern_default" "$nsys_mem_default"

        echo "  Exporting kernel stats -> $kern_csv"
        nsys stats --report cuda_gpu_kern_sum --format csv --force-export=true --output "$OUTPUT_DIR" "$rep" >/dev/null 2>&1 || true
        if [[ -f "$nsys_kern_default" ]]; then
            mv "$nsys_kern_default" "$kern_csv"
        else
            echo "  WARNING: $nsys_kern_default not created"
        fi

        echo "  Exporting memcpy stats -> $mem_csv"
        nsys stats --report cuda_gpu_mem_time_sum --format csv --force-export=true --output "$OUTPUT_DIR" "$rep" >/dev/null 2>&1 || true
        if [[ -f "$nsys_mem_default" ]]; then
            mv "$nsys_mem_default" "$mem_csv"
        else
            echo "  WARNING: $nsys_mem_default not created"
        fi
    else
        echo "  WARNING: nsys-rep not found, skipping stats export"
    fi

    echo ""
}

# ============================================================================
# Profile configurations
# ============================================================================
# Profile configurations
# ============================================================================
# Two tiers:
#   "quick" — small configs for fast iteration (default)
#   "prod"  — production-scale parameters from xengine_blackwell_strategy-4.md
#
# bench_pipeline: runs=1 warmup=1
# test binaries: --n-iters 1
#
# Production parameters (from strategy doc §§3.2–3.5, 28):
#   VisBF: M=1664(3328 w/2pol), K=64-512, Nf=2445, Ng=4096, Nbeams=6M
#   VoltBF (INT4 GEMM): NA=1651, NB=4000, NC=2500, NT=128-512, n_tps=12
#
# NOTE: tune=false is used for profiling to avoid autotuning overhead
# distorting kernel traces. Run run_benchmarks.sh first to populate the
# strategy/kernel tune caches, which will be reused here.
#
# NOTE: test binaries hardcode Ng=64 for imaging tests. Production-scale
# FFT profiling (Ng=4096, Nbeams=6M) requires bench_pipeline.
# ============================================================================

# ---- Visibility Beamformer ----
if [[ "$SUITE" == "all" || "$SUITE" == "visbf" ]]; then
    echo "===== Visibility Beamformer Profiling ====="

    if [[ -n "$BENCH_PIPELINE" ]]; then
        # -- Quick configs (small) --
        nsys_run "visbf_1ch_K128" "$BENCH_PIPELINE" \
            visbf n_ant=1664 n_ch=1 n_time=256 n_time_inner=2 \
            n_beam=256 Ng=4096 warmup=1 runs=1 tune=false

        nsys_run "visbf_8ch_K128" "$BENCH_PIPELINE" \
            visbf n_ant=1664 n_ch=8 n_time=256 n_time_inner=2 \
            n_beam=256 Ng=4096 warmup=1 runs=1 tune=false

        # -- Production-scale configs --
        # Nf_eff = n_ch * n_time_inner = 128 * 2 = 256 frequency channels
        # 256 batched 4096x4096 FFTs + 6M beam extractions
        # UV grid: 256 * 4096^2 * 8 bytes = 34 GB (fits in GH200 96 GB)
        nsys_run "visbf_prod_K128" "$BENCH_PIPELINE" \
            visbf n_ant=1664 n_ch=128 n_time=256 n_time_inner=2 \
            n_beam=6000000 Ng=4096 warmup=1 runs=1 tune=false

        # Full production K=64 (nominal operating point, 122.68 kHz FSM)
        nsys_run "visbf_prod_K64" "$BENCH_PIPELINE" \
            visbf n_ant=1664 n_ch=128 n_time=128 n_time_inner=2 \
            n_beam=6000000 Ng=4096 warmup=1 runs=1 tune=false

    elif [[ -n "$VP_TEST" ]]; then
        # Test binary: Ng hardcoded to 64, no --VP-Ng option
        # Only HERK + small imaging; FFT will be trivial
        nsys_run "visbf_1ch_K128" "$VP_TEST" \
            --enable-testing false --profile true \
            --VP-n-antennae 1664 --VP-n-channels 1 --VP-n-time 256 \
            --VP-n-beams 256 --n-iters 1

        nsys_run "visbf_8ch_K128" "$VP_TEST" \
            --enable-testing false --profile true \
            --VP-n-antennae 1664 --VP-n-channels 8 --VP-n-time 256 \
            --VP-n-beams 256 --n-iters 1

        # Production channel count (imaging still tiny due to Ng=64)
        nsys_run "visbf_128ch_K128" "$VP_TEST" \
            --enable-testing false --profile true \
            --VP-n-antennae 1664 --VP-n-channels 128 --VP-n-time 256 \
            --VP-n-beams 256 --n-iters 1
    else
        echo "  SKIP: No visibility binary found"
    fi
fi

# ---- Voltage Beamformer ----
if [[ "$SUITE" == "all" || "$SUITE" == "voltbf" ]]; then
    echo "===== Voltage Beamformer Profiling (INT4 GEMM) ====="

    if [[ -n "$BENCH_PIPELINE" ]]; then
        # -- Quick configs --
        # INT4 path: qc_transpose_polsplit -> gemm_prepared_power_int4() x2 pols
        nsys_run "voltbf_1ch_t8" "$BENCH_PIPELINE" \
            voltbf n_ant=1664 n_beam=4096 n_ch=1 n_time=8 n_tps=1 \
            warmup=1 runs=1 tune=false

        nsys_run "voltbf_8ch_t8" "$BENCH_PIPELINE" \
            voltbf n_ant=1664 n_beam=4096 n_ch=8 n_time=8 n_tps=1 \
            warmup=1 runs=1 tune=false

        # -- Production-scale configs --
        # Strategy doc §28: NA=1651, NB=4000, NC=2500, NT=128-512, n_tps=12
        # GEMM: M=n_time, N=n_beam, K=n_ant (transposed from doc convention)
        # Weight matrix: n_ch * n_beam * n_ant * 2 * 2 bytes = significant
        # Start with smaller n_ch to fit in GH200 memory
        nsys_run "voltbf_prod_t128" "$BENCH_PIPELINE" \
            voltbf n_ant=1664 n_beam=4000 n_ch=64 n_time=128 n_tps=1 \
            warmup=1 runs=1 tune=false

        nsys_run "voltbf_prod_t512" "$BENCH_PIPELINE" \
            voltbf n_ant=1664 n_beam=4000 n_ch=64 n_time=512 n_tps=1 \
            warmup=1 runs=1 tune=false

    elif [[ -n "$VBF_TEST" ]]; then
        # -- Quick configs --
        nsys_run "voltbf_1ch_t8" "$VBF_TEST" \
            --enable-testing false --profile true \
            --VBF-n-antennae 1664 --VBF-n-beams 4096 \
            --VBF-n-channels 1 --VBF-n-time 8 --n-iters 1

        nsys_run "voltbf_8ch_t8" "$VBF_TEST" \
            --enable-testing false --profile true \
            --VBF-n-antennae 1664 --VBF-n-beams 4096 \
            --VBF-n-channels 8 --VBF-n-time 8 --n-iters 1

        # -- Production-scale configs --
        # NB=4000, NA=1664(closest to 1651), NC=64, NT=128
        # Weight matrix: 64 * 4000 * 1664 * 2 * 2 = 1.6 GB — fits easily
        nsys_run "voltbf_prod_t128" "$VBF_TEST" \
            --enable-testing false --profile true \
            --VBF-n-antennae 1664 --VBF-n-beams 4000 \
            --VBF-n-channels 64 --VBF-n-time 128 --n-iters 1

        nsys_run "voltbf_prod_t512" "$VBF_TEST" \
            --enable-testing false --profile true \
            --VBF-n-antennae 1664 --VBF-n-beams 4000 \
            --VBF-n-channels 64 --VBF-n-time 512 --n-iters 1
    else
        echo "  SKIP: No voltage binary found"
    fi
fi

echo "================================================================"
echo "  Profiling complete."
echo "  Reports in: $OUTPUT_DIR/"
echo ""
echo "  Next step: parse results"
echo "    bash $SCRIPT_DIR/parse_nsys_profiles.sh $OUTPUT_DIR"
echo "================================================================"
