#!/usr/bin/env bash
# find_best_flags.sh — Find the best compile-time build config for a given problem
#
# Two-phase approach per build:
#   1. Run with tune=true to autotune strategy + kernel params (timing discarded)
#   2. Run with tune=0 mode=bench for clean timing measurement
#
# Ranks builds by wall-clock time to identify the best compile-time config.
#
# Prerequisites: run build_sweep.sh first to populate sweep_builds/
#
# Usage:
#   bash find_best_flags.sh                                    # default problem
#   bash find_best_flags.sh --N 4096 --K 64 --batch 128       # custom problem
#   bash find_best_flags.sh --build-dir /path/to/sweep_builds  # custom build dir
#   bash find_best_flags.sh --dry-run                          # list builds only
#   bash find_best_flags.sh --help

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ============================================================================
# Defaults
# ============================================================================

N=1664
K=16
BATCH=256
PRECISION="fp8"
BUILD_DIR="${SCRIPT_DIR}/sweep_builds"
DRY_RUN=false

# ============================================================================
# Argument parsing
# ============================================================================

print_help() {
    cat <<'EOF'
Usage: bash find_best_flags.sh [OPTIONS]

Find the best compile-time build configuration for a given HERK problem.

Two-phase per build:
  1. tune=true  — autotune strategy + kernel params (timing discarded)
  2. tune=0     — clean benchmark measurement

Ranks by wall-clock time. Prints ms/item, TFLOPS, and GB/s.

Prerequisites: run build_sweep.sh first to populate sweep_builds/

Problem specification:
  --N <int>            N dimension (default: 1664)
  --K <int>            K dimension (default: 16)
  --batch <int>        Batch count (default: 256)
  --precision <str>    Compute precision: fp8, fp6e3m2, fp6e2m3, fp4 (default: fp8)

Execution:
  --build-dir <dir>    Directory containing build subdirs (default: sweep_builds/)
  --dry-run            List builds found without running
  -h, --help           Show this help

Examples:
  # Default problem (N=1664 K=16 batch=256) across all builds
  bash find_best_flags.sh

  # Custom problem
  bash find_best_flags.sh --N 4096 --K 64 --batch 128

  # SM90 builds
  bash find_best_flags.sh --build-dir sweep_builds_sm90/
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --N)         N="$2"; shift 2 ;;
        --K)         K="$2"; shift 2 ;;
        --batch)     BATCH="$2"; shift 2 ;;
        --precision) PRECISION="$2"; shift 2 ;;
        --build-dir) BUILD_DIR="$2"; shift 2 ;;
        --dry-run)   DRY_RUN=true; shift ;;
        -h|--help)   print_help; exit 0 ;;
        *)           echo "Unknown option: $1"; echo "Run with --help for usage."; exit 1 ;;
    esac
done

# ============================================================================
# Discover builds
# ============================================================================

if [[ ! -d "$BUILD_DIR" ]]; then
    echo "ERROR: Build directory not found: $BUILD_DIR"
    echo "  Run build_sweep.sh first, or use --build-dir to specify."
    exit 1
fi

declare -a BUILD_NAMES=()
declare -a BUILD_BINARIES=()

for dir in "$BUILD_DIR"/*/; do
    [[ -d "$dir" ]] || continue
    # Detect SM100/SM120 or SM90 binary
    binary=""
    if [[ -x "${dir}example_complex_sm100" ]]; then
        binary="${dir}example_complex_sm100"
    elif [[ -x "${dir}example_complex_fp8" ]]; then
        binary="${dir}example_complex_fp8"
    fi
    if [[ -n "$binary" ]]; then
        name="$(basename "$dir")"
        BUILD_NAMES+=("$name")
        BUILD_BINARIES+=("$binary")
    fi
done

NUM_BUILDS=${#BUILD_NAMES[@]}

if [[ "$NUM_BUILDS" -eq 0 ]]; then
    echo "ERROR: No builds found in $BUILD_DIR"
    echo "  Looked for example_complex_sm100 or example_complex_fp8 in each subdirectory."
    echo "  Run build_sweep.sh first."
    exit 1
fi

# ============================================================================
# Display header
# ============================================================================

echo "=========================================="
echo " Find Best Build Config"
echo "=========================================="
echo "  Problem:    N=${N} K=${K} batch=${BATCH}"
echo "  Precision:  ${PRECISION}"
echo "  Builds:     ${NUM_BUILDS} found in ${BUILD_DIR}"
echo "  Method:     tune=true (discarded), then tune=0 mode=bench (timed)"
echo "=========================================="
echo ""

echo "  Builds found:"
for i in "${!BUILD_NAMES[@]}"; do
    printf "    [%d/%d] %s\n" $((i+1)) "$NUM_BUILDS" "${BUILD_NAMES[$i]}"
done
echo ""

if $DRY_RUN; then
    echo "(dry-run — exiting without running)"
    exit 0
fi

# ============================================================================
# Setup output
# ============================================================================

RESULTS_DIR="${SCRIPT_DIR}/sweep_results"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
CSV_FILE="${RESULTS_DIR}/best_config_N${N}_K${K}_b${BATCH}_${PRECISION}_${TIMESTAMP}.csv"

mkdir -p "$RESULTS_DIR"
echo "config,N,K,batch,precision,time_ms,ms_per_item,tflops,gbs" > "$CSV_FILE"

# ============================================================================
# Run each build (two-phase: tune then measure)
# ============================================================================

BEST_TIME=""
BEST_CONFIG=""
BEST_BINARY=""
BEST_TFLOPS=""
BEST_GBS=""
BEST_MS_ITEM=""
RUN_COUNT=0
FAIL_COUNT=0

# Common args
BASE_ARGS="$N $N $K $BATCH"
if [[ "$PRECISION" != "fp8" ]]; then
    BASE_ARGS="$BASE_ARGS precision=$PRECISION"
fi

for i in "${!BUILD_NAMES[@]}"; do
    name="${BUILD_NAMES[$i]}"
    binary="${BUILD_BINARIES[$i]}"

    printf "  [%d/%d] %-20s " $((i+1)) "$NUM_BUILDS" "$name"

    tune_log="${RESULTS_DIR}/${name}_tune_N${N}_K${K}_b${BATCH}.log"
    bench_log="${RESULTS_DIR}/${name}_bench_N${N}_K${K}_b${BATCH}.log"

    # --- Phase 1: Autotune (timing discarded) ---
    # shellcheck disable=SC2086
    if ! "$binary" $BASE_ARGS tune=true mode=bench > "$tune_log" 2>&1; then
        echo "TUNE FAILED (see ${tune_log})"
        ((FAIL_COUNT++)) || true
        continue
    fi

    # --- Phase 2: Clean measurement ---
    # shellcheck disable=SC2086
    if ! "$binary" $BASE_ARGS tune=0 mode=bench > "$bench_log" 2>&1; then
        echo "BENCH FAILED (see ${bench_log})"
        ((FAIL_COUNT++)) || true
        continue
    fi

    # Parse output: look for batched/HERK lines with TFLOPS and GB/s
    # Formats from example_usage_sm100.cu:
    #   "  Batched Baseline:     12.34 ms  ( 567.8 TFLOPS,  123.4 GB/s, IO=1234, 0.048 ms/item)"
    #   "  Batched Triangle:     12.34 ms  ( 567.8 TFLOPS,  123.4 GB/s, IO=1234, 0.048 ms/item, 1.23x)"
    #   "  HERK Baseline        12.34 ms   567.8 TFLOPS   123.4 GB/s  IO=1234"
    time_ms=""
    tflops=""
    gbs=""
    ms_item=""

    while IFS= read -r line; do
        t=""
        tf=""
        gb=""
        mi=""

        # Format 1: batched — "12.34 ms  ( 567.8 TFLOPS,  123.4 GB/s, IO=..., 0.048 ms/item"
        if [[ "$line" =~ ([0-9]+\.[0-9]+)[[:space:]]*ms[[:space:]]+\([[:space:]]*([0-9]+\.[0-9]+)[[:space:]]*TFLOPS,[[:space:]]*([0-9]+\.[0-9]+)[[:space:]]*GB/s.*[[:space:]]([0-9]+\.[0-9]+)[[:space:]]*ms/item ]]; then
            t="${BASH_REMATCH[1]}"
            tf="${BASH_REMATCH[2]}"
            gb="${BASH_REMATCH[3]}"
            mi="${BASH_REMATCH[4]}"
        # Format 2: batched without ms/item — "12.34 ms  ( 567.8 TFLOPS,  123.4 GB/s, ...)"
        elif [[ "$line" =~ ([0-9]+\.[0-9]+)[[:space:]]*ms[[:space:]]+\([[:space:]]*([0-9]+\.[0-9]+)[[:space:]]*TFLOPS,[[:space:]]*([0-9]+\.[0-9]+)[[:space:]]*GB/s ]]; then
            t="${BASH_REMATCH[1]}"
            tf="${BASH_REMATCH[2]}"
            gb="${BASH_REMATCH[3]}"
        # Format 3: flat — "12.34 ms  567.8 TFLOPS  123.4 GB/s"
        elif [[ "$line" =~ ([0-9]+\.[0-9]+)[[:space:]]*ms[[:space:]]+([0-9]+\.[0-9]+)[[:space:]]*TFLOPS[[:space:]]+([0-9]+\.[0-9]+)[[:space:]]*GB/s ]]; then
            t="${BASH_REMATCH[1]}"
            tf="${BASH_REMATCH[2]}"
            gb="${BASH_REMATCH[3]}"
        # Format 4: flat without GB/s — "12.34 ms  567.8 TFLOPS"
        elif [[ "$line" =~ ([0-9]+\.[0-9]+)[[:space:]]*ms[[:space:]]+([0-9]+\.[0-9]+)[[:space:]]*TFLOPS ]]; then
            t="${BASH_REMATCH[1]}"
            tf="${BASH_REMATCH[2]}"
        fi

        if [[ -n "$t" ]]; then
            # Keep the fastest (lowest time) across all output lines
            if [[ -z "$time_ms" ]] || \
               python3 -c "exit(0 if $t < $time_ms else 1)" 2>/dev/null; then
                time_ms="$t"
                tflops="$tf"
                gbs="${gb:-0}"
                ms_item="${mi:-}"
            fi
        fi
    done < <(grep -i "batched\|herk\|triangle\|baseline" "$bench_log" || true)

    if [[ -z "$time_ms" ]]; then
        echo "NO DATA (see ${bench_log})"
        ((FAIL_COUNT++)) || true
        continue
    fi

    # Compute ms/item if not parsed from output
    if [[ -z "$ms_item" ]]; then
        ms_item=$(python3 -c "print(f'{$time_ms / $BATCH:.3f}')" 2>/dev/null || echo "?")
    fi

    printf "%8s ms  %6s ms/item  %6s TFLOPS  %6s GB/s\n" "$time_ms" "$ms_item" "$tflops" "$gbs"
    echo "${name},${N},${K},${BATCH},${PRECISION},${time_ms},${ms_item},${tflops},${gbs}" >> "$CSV_FILE"
    ((RUN_COUNT++)) || true

    # Track overall best
    if [[ -z "$BEST_TIME" ]] || \
       python3 -c "exit(0 if $time_ms < $BEST_TIME else 1)" 2>/dev/null; then
        BEST_TIME="$time_ms"
        BEST_CONFIG="$name"
        BEST_BINARY="$binary"
        BEST_TFLOPS="$tflops"
        BEST_GBS="$gbs"
        BEST_MS_ITEM="$ms_item"
    fi
done

# ============================================================================
# Summary
# ============================================================================

echo ""
echo "=========================================="
echo " Results"
echo "=========================================="
echo ""
echo "  Problem: N=${N} K=${K} batch=${BATCH} precision=${PRECISION}"
echo "  Builds tested: ${RUN_COUNT} / ${NUM_BUILDS}"
if [[ "$FAIL_COUNT" -gt 0 ]]; then
    echo "  Failed: ${FAIL_COUNT}"
fi
echo ""

if [[ -z "$BEST_TIME" ]]; then
    echo "  No successful measurements."
    echo ""
    exit 1
fi

# Ranked table
echo "  Ranked by wall-clock time:"
echo ""
printf "  %-4s %-20s %9s %9s %8s %8s %8s\n" \
    "Rank" "Config" "Time(ms)" "ms/item" "TFLOPS" "GB/s" "vs Best"
printf "  %-4s %-20s %9s %9s %8s %8s %8s\n" \
    "----" "------" "--------" "-------" "------" "----" "-------"

tail -n +2 "$CSV_FILE" \
    | sort -t, -k6 -n \
    | { rank=0; while IFS=, read -r cfg cn ck cb cp tms mi tfl gb; do
        ((rank++)) || true
        ratio=$(python3 -c "print(f'{$tms / $BEST_TIME:.2f}')" 2>/dev/null || echo "?")
        marker=""
        if [[ "$rank" -eq 1 ]]; then marker=" <--"; fi
        printf "  %-4d %-20s %9s %9s %8s %8s %7sx%s\n" \
            "$rank" "$cfg" "$tms" "$mi" "$tfl" "$gb" "$ratio" "$marker"
    done; }
echo ""

echo "  Best config:  ${BEST_CONFIG}"
echo "  Time:         ${BEST_TIME} ms"
echo "  Per item:     ${BEST_MS_ITEM} ms/item"
echo "  TFLOPS:       ${BEST_TFLOPS}"
echo "  Bandwidth:    ${BEST_GBS} GB/s"
echo ""

# Reproduce command
echo "  Reproduce:"
echo "    $BEST_BINARY $BASE_ARGS tune=0 mode=bench"
echo ""
echo "  CSV: ${CSV_FILE}"
echo ""
