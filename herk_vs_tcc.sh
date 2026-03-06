#!/usr/bin/env bash
# herk_vs_tcc.sh — Comprehensive HERK performance sweep + TCC comparison
#
# Part 1: Square problems (N=K=256,512,1024,2048,4096) batch=32
# Part 2: Small-K problems (N=3328, K=16,32,64,128,256,512) batch=128
#
# For each problem size:
#   1. Cycle through all CUTLASS compile-time configurations
#   2. Autotune overhead kernels (tune=2)
#   3. Test direct modes (auto, baseline, force-direct)
#   4. Compare best CUTLASS results with TCC
#
# Also validates the unified PIMPL API (herk/gemm with runtime precision enums)
# by building and running test_herk_int4, test_gemm_api, and test_tune_api.
#
# Usage:
#   bash herk_vs_tcc.sh                 # full sweep (rebuild missing configs)
#   bash herk_vs_tcc.sh --skip-build    # use existing binaries only
#   bash herk_vs_tcc.sh --part 1        # Part 1 only (square problems)
#   bash herk_vs_tcc.sh --part 2        # Part 2 only (small-K)
#   bash herk_vs_tcc.sh --tcc-only      # TCC benchmarks only
#   bash herk_vs_tcc.sh --dry-run       # preview what would run
#   bash herk_vs_tcc.sh --tcc-format fp4  # TCC format (default: fp8)
#   bash herk_vs_tcc.sh --skip-api-tests # skip PIMPL API validation

set -euo pipefail

# =============================================================================
# Configuration
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CUTLASS_DIR="${CUTLASS_DIR:-/home/deanhowarth/applications/cutlass}"
ARCH="120a"

# Source directory for CUTLASS GEMM library (in monorepo, but never build in-tree)
CUTLASS_GEMM_SRC="${SCRIPT_DIR}/dsa-2000-monorepo/packages/ftd/cutlass_interface"

# TCC paths
TCC_EXAMPLE="${SCRIPT_DIR}/tensor-core-correlator/example/build/example"

# Build infrastructure (outside monorepo per user constraint)
BUILD_ROOT="${SCRIPT_DIR}/sweep_builds"
RESULTS_DIR="${SCRIPT_DIR}/herk_results"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
CSV_FILE="${RESULTS_DIR}/herk_vs_tcc_${TIMESTAMP}.csv"

# Part 1: Square problems (N=K, batch=32)
# NOTE: User specified 2408; changed to 2048 (TCC requires K % 16 == 0 for FP8).
#       Change back to 2408 if that was intentional (CUTLASS handles it fine,
#       but TCC comparison won't work for non-multiple-of-16 K values).
SQUARE_SIZES=(256 512 1024 2048 4096)
SQUARE_BATCH=32

# Part 2: Small-K problems (N=3328, variable K, batch=128)
SMALLK_N=3328
SMALLK_K_VALUES=(16 32 64 128 256 512)
SMALLK_BATCH=128

# Direct HERK modes to test
DIRECT_MODES=("auto" "0" "1")

# TCC format
TCC_FORMAT="fp8"

# Benchmark settings
WARMUP=5
ITERATIONS=20

# =============================================================================
# Compile-time configurations
# =============================================================================
# Format: "name|cmake_flags"
# FP6+FP4 always enabled (required for link even if only benchmarking FP8 HERK)

FP6FP4="-DCOMPLEX_SM100_ENABLE_FP6=ON -DCOMPLEX_SM100_ENABLE_FP4=ON"

declare -a CONFIGS=(
    "fp8_n128|${FP6FP4}"
    "fp8_novec|${FP6FP4} -DCOMPLEX_FP8_EXTRA_VECTORIZATION=OFF"
    "fp8_reg255|${FP6FP4} -DCOMPLEX_FP8_MAX_REGISTERS=255"
    "stg3|${FP6FP4} -DCOMPLEX_FP8_SM100_STAGES=3"
    "stg3_reg255|${FP6FP4} -DCOMPLEX_FP8_SM100_STAGES=3 -DCOMPLEX_FP8_MAX_REGISTERS=255"
    "stg3_novec|${FP6FP4} -DCOMPLEX_FP8_SM100_STAGES=3 -DCOMPLEX_FP8_EXTRA_VECTORIZATION=OFF"
)

# =============================================================================
# Argument parsing
# =============================================================================

SKIP_BUILD=false
DRY_RUN=false
TCC_ONLY=false
SKIP_API_TESTS=false
PART="both"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --skip-build)      SKIP_BUILD=true; shift ;;
        --dry-run)         DRY_RUN=true; shift ;;
        --tcc-only)        TCC_ONLY=true; shift ;;
        --skip-api-tests)  SKIP_API_TESTS=true; shift ;;
        --part)            PART="$2"; shift 2 ;;
        --tcc-format)      TCC_FORMAT="$2"; shift 2 ;;
        -h|--help)
            echo "Usage: bash herk_vs_tcc.sh [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --skip-build      Use existing binaries only (skip cmake+make)"
            echo "  --dry-run         Preview what would run, don't execute"
            echo "  --tcc-only        Run TCC benchmarks only"
            echo "  --skip-api-tests  Skip PIMPL API validation (test_herk_int4 etc.)"
            echo "  --part 1|2|both   Which part to run (default: both)"
            echo "  --tcc-format X    TCC format: fp8|fp4|fp16 (default: fp8)"
            echo ""
            echo "Part 1: Square N=K=(256,512,1024,2048,4096) batch=${SQUARE_BATCH}"
            echo "Part 2: Small-K N=3328 K=(16,32,64,128,256,512) batch=${SMALLK_BATCH}"
            exit 0
            ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# =============================================================================
# Helper functions
# =============================================================================

mkdir -p "$RESULTS_DIR"

log() {
    echo "[$(date +%H:%M:%S)] $*"
}

# Compute normalized TFLOPS: 8*N^2*K*batch / time_ms / 1e9
# (standard complex HERK work: 4 sub-GEMMs × 2MNK)
compute_effective_tflops() {
    local N=$1 K=$2 batch=$3 time_ms=$4
    python3 -c "
N, K, batch, t = $N, $K, $batch, $time_ms
if t > 0:
    flops = 8.0 * N * N * K * batch
    print(f'{flops / (t * 1e-3) / 1e12:.2f}')
else:
    print('0.00')
"
}

# Extract batched FP8 HERK results from CUTLASS output
# Usage: parse_cutlass_batched_herk <output_file> <variant>
# variant: "Baseline" | "Triangle" | "Tri+Graph"
# Returns: "time_ms tflops" or empty if not found
parse_cutlass_batched() {
    local file=$1 variant=$2
    # Pattern: "  Batched Baseline:        0.31 ms  (  10.3 TFLOPS, ..."
    # or:      "  Batched Triangle:        0.27 ms  (  12.1 TFLOPS, ..."
    local escaped_variant
    escaped_variant=$(echo "$variant" | sed 's/+/\\+/g')
    grep -P "Batched ${escaped_variant}:" "$file" 2>/dev/null | head -1 | \
        perl -ne 'if (/(\d+\.\d+)\s+ms\s+\(\s+(\d+\.\d+)\s+TFLOPS/) { print "$1 $2\n"; }'
}

# Extract single (non-batched) FP8 HERK results
# Usage: parse_cutlass_single_herk <output_file> <variant>
# variant: "Baseline" | "Triangle" | "Tri+Graph"
parse_cutlass_single() {
    local file=$1 variant=$2
    local escaped_variant
    escaped_variant=$(echo "$variant" | sed 's/+/\\+/g')
    # Pattern: "  HERK Baseline          0.03 ms     3.4 TFLOPS  ..."
    # Only match in the "--- FP8 E4M3 ---" section (first occurrence)
    sed -n '/--- FP8 E4M3 ---/,/^---/p' "$file" 2>/dev/null | \
        grep -P "HERK ${escaped_variant}" | head -1 | \
        perl -ne 'if (/(\d+\.\d+)\s+ms\s+(\d+\.\d+)\s+TFLOPS/) { print "$1 $2\n"; }'
}

# Parse TCC CSV output
# Returns: "time_ms tflops"
parse_tcc_csv() {
    local file=$1
    grep "^CSV:" "$file" 2>/dev/null | head -1 | \
        awk -F',' '{print $5, $6}'
}

# =============================================================================
# Phase 0: Print plan and sanity checks
# =============================================================================

echo "╔══════════════════════════════════════════════════════════════════════╗"
echo "║           HERK Performance Sweep + TCC Comparison                  ║"
echo "╚══════════════════════════════════════════════════════════════════════╝"
echo ""

if [[ "$PART" == "1" || "$PART" == "both" ]]; then
    echo "Part 1 — Square (N=K): ${SQUARE_SIZES[*]}  batch=${SQUARE_BATCH}"
fi
if [[ "$PART" == "2" || "$PART" == "both" ]]; then
    echo "Part 2 — Small-K:  N=${SMALLK_N}  K=${SMALLK_K_VALUES[*]}  batch=${SMALLK_BATCH}"
fi
echo ""
echo "Compile configs: ${#CONFIGS[@]}"
for cfg_entry in "${CONFIGS[@]}"; do
    name="${cfg_entry%%|*}"
    flags="${cfg_entry#*|}"
    binary="${BUILD_ROOT}/${name}/example_complex_sm100"
    status="[NEEDS BUILD]"
    if [[ -x "$binary" ]]; then status="[EXISTS]"; fi
    printf "  %-16s %s  %s\n" "$name" "$status" "$flags"
done
echo ""
echo "Direct modes: ${DIRECT_MODES[*]}"
echo "TCC format: ${TCC_FORMAT}"
echo "TCC binary: ${TCC_EXAMPLE}"
echo "Results: ${CSV_FILE}"
echo ""

if $DRY_RUN; then
    echo "(dry-run mode — exiting)"
    exit 0
fi

# Check TCC binary exists
if [[ ! -x "$TCC_EXAMPLE" ]]; then
    echo "WARNING: TCC binary not found at ${TCC_EXAMPLE}"
    echo "  TCC comparison will be skipped."
    echo "  Build TCC: cd tensor-core-correlator/example/build && cmake .. && make"
    TCC_AVAILABLE=false
else
    TCC_AVAILABLE=true
fi

# =============================================================================
# Phase 1: Build missing CUTLASS configurations
# =============================================================================

if ! $TCC_ONLY && ! $SKIP_BUILD; then
    log "Phase 1: Building CUTLASS configurations..."
    echo ""

    for cfg_entry in "${CONFIGS[@]}"; do
        name="${cfg_entry%%|*}"
        flags="${cfg_entry#*|}"
        build_dir="${BUILD_ROOT}/${name}"
        binary="${build_dir}/example_complex_sm100"
        log_file="${RESULTS_DIR}/${name}_build.log"

        if [[ -x "$binary" ]]; then
            log "  ${name}: binary exists, skipping build"
            continue
        fi

        log "  ${name}: configuring..."
        mkdir -p "$build_dir"

        # shellcheck disable=SC2086
        if ! cmake -S "$CUTLASS_GEMM_SRC" -B "$build_dir" \
            -DCUTLASS_DIR="$CUTLASS_DIR" \
            -DCOMPLEX_FP8_ARCH="$ARCH" \
            $flags \
            > "$log_file" 2>&1; then
            log "  ${name}: CONFIGURE FAILED — see ${log_file}"
            continue
        fi

        log "  ${name}: building ($(nproc) threads)..."
        if ! cmake --build "$build_dir" -j"$(nproc)" -- example_complex_sm100 cutlass_gemm_api test_herk_int4 test_gemm_api test_tune_api >> "$log_file" 2>&1; then
            log "  ${name}: BUILD FAILED — see ${log_file}"
            continue
        fi

        log "  ${name}: build OK"
    done
    echo ""
fi

# =============================================================================
# Phase 1b: PIMPL API validation (unified herk/gemm with runtime precision enums)
# =============================================================================
# Runs test_herk_int4, test_gemm_api, and test_tune_api from the first available
# build config to verify the unified PIMPL API (herk/gemm/prepare_b/gemm_prepared).

if ! $TCC_ONLY && ! $SKIP_API_TESTS; then
    echo ""
    log "Phase 1b: PIMPL API validation (unified herk/gemm)"
    echo ""

    API_TEST_DIR=""
    for cfg_entry in "${CONFIGS[@]}"; do
        name="${cfg_entry%%|*}"
        build_dir="${BUILD_ROOT}/${name}"
        if [[ -x "${build_dir}/test_herk_int4" && -x "${build_dir}/test_gemm_api" ]]; then
            API_TEST_DIR="$build_dir"
            log "  Using build: ${name}"
            break
        fi
    done

    if [[ -z "$API_TEST_DIR" ]]; then
        log "  WARNING: No build with PIMPL test binaries found, skipping API tests"
    else
        API_TESTS_PASSED=0
        API_TESTS_FAILED=0

        for test_bin in test_herk_int4 test_gemm_api test_tune_api; do
            test_path="${API_TEST_DIR}/${test_bin}"
            if [[ ! -x "$test_path" ]]; then
                log "    ${test_bin}: not found, skipping"
                continue
            fi

            out_file="${RESULTS_DIR}/api_${test_bin}.log"
            log "    ${test_bin}..."

            if "$test_path" > "$out_file" 2>&1; then
                # Count PASS/FAIL from output
                pass_count=$(grep -c "PASS" "$out_file" 2>/dev/null || echo 0)
                fail_count=$(grep -c "FAIL" "$out_file" 2>/dev/null || echo 0)
                log "      PASS (${pass_count} checks passed)"
                API_TESTS_PASSED=$((API_TESTS_PASSED + 1))
            else
                log "      FAILED — see ${out_file}"
                API_TESTS_FAILED=$((API_TESTS_FAILED + 1))
            fi
        done

        echo ""
        if [[ $API_TESTS_FAILED -gt 0 ]]; then
            log "  PIMPL API: ${API_TESTS_PASSED} passed, ${API_TESTS_FAILED} FAILED"
            log "  WARNING: API validation failures — benchmark results may not be meaningful"
        else
            log "  PIMPL API: ${API_TESTS_PASSED} passed, 0 failed"
        fi
        echo ""
    fi
fi

# =============================================================================
# Phase 2: CUTLASS HERK benchmarks (internal GemmComplexSm100 API)
# =============================================================================

# CSV header
echo "tool,config,direct,N,K,batch,variant,time_ms,reported_tflops,effective_tflops" > "$CSV_FILE"

run_cutlass_problem() {
    local N=$1 K=$2 batch=$3 label=$4

    for cfg_entry in "${CONFIGS[@]}"; do
        local name="${cfg_entry%%|*}"
        local binary="${BUILD_ROOT}/${name}/example_complex_sm100"

        if [[ ! -x "$binary" ]]; then
            log "    ${name}: binary missing, skipping"
            continue
        fi

        for direct in "${DIRECT_MODES[@]}"; do
            local out_file="${RESULTS_DIR}/cutlass_${name}_direct${direct}_${label}.log"

            log "    ${name} direct=${direct} ${label}..."

            # First run with tune=2 to autotune overhead kernels
            # (tuning results are cached per build in cutlass_kernel_cache_{fingerprint}.txt)
            if ! "$binary" "$N" "$N" "$K" "$batch" mode=bench \
                direct="$direct" tune=2 > "$out_file" 2>&1; then
                log "      FAILED — see ${out_file}"
                continue
            fi

            # Extract batched FP8 results (preferred for TCC comparison)
            for variant in "Baseline" "Triangle" "Tri+Graph"; do
                local result
                result=$(parse_cutlass_batched "$out_file" "$variant")

                # Fall back to single HERK if batched section not found
                if [[ -z "$result" ]]; then
                    result=$(parse_cutlass_single "$out_file" "$variant")
                fi

                if [[ -n "$result" ]]; then
                    local time_ms tflops eff_tflops
                    read -r time_ms tflops <<< "$result"
                    eff_tflops=$(compute_effective_tflops "$N" "$K" "$batch" "$time_ms")
                    echo "cutlass,${name},${direct},${N},${K},${batch},${variant},${time_ms},${tflops},${eff_tflops}" >> "$CSV_FILE"
                fi
            done
        done
    done
}

run_tcc_problem() {
    local N=$1 K=$2 batch=$3 label=$4

    if ! $TCC_AVAILABLE; then
        log "    TCC: not available, skipping"
        return
    fi

    local out_file="${RESULTS_DIR}/tcc_${label}.log"

    # TCC args: receivers samples channels format
    # K must be multiple of timesPerBlock (16 for fp8, 32 for fp4)
    local times_per_block=16
    if [[ "$TCC_FORMAT" == "fp4" ]]; then times_per_block=32; fi

    if (( K % times_per_block != 0 )); then
        log "    TCC: K=${K} not multiple of ${times_per_block} for ${TCC_FORMAT}, skipping"
        echo "tcc,tcc,-,${N},${K},${batch},-,-,-,-" >> "$CSV_FILE"
        return
    fi

    log "    TCC ${TCC_FORMAT} N=${N} K=${K} batch=${batch}..."

    if ! "$TCC_EXAMPLE" "$N" "$K" "$batch" "$TCC_FORMAT" > "$out_file" 2>&1; then
        log "      TCC FAILED — see ${out_file}"
        echo "tcc,tcc,-,${N},${K},${batch},-,FAIL,FAIL,FAIL" >> "$CSV_FILE"
        return
    fi

    local result
    result=$(parse_tcc_csv "$out_file")

    if [[ -n "$result" ]]; then
        local time_ms tcc_tflops eff_tflops
        read -r time_ms tcc_tflops <<< "$result"
        eff_tflops=$(compute_effective_tflops "$N" "$K" "$batch" "$time_ms")
        echo "tcc,tcc,-,${N},${K},${batch},TCC,${time_ms},${tcc_tflops},${eff_tflops}" >> "$CSV_FILE"
    fi
}

if ! $TCC_ONLY; then

    # --- Part 1: Square problems ---
    if [[ "$PART" == "1" || "$PART" == "both" ]]; then
        echo ""
        log "Phase 2a: CUTLASS HERK — Square problems (N=K, batch=${SQUARE_BATCH})"
        echo ""

        for size in "${SQUARE_SIZES[@]}"; do
            local_label="sq_${size}"
            log "  N=K=${size} batch=${SQUARE_BATCH}"
            run_cutlass_problem "$size" "$size" "$SQUARE_BATCH" "$local_label"
        done
    fi

    # --- Part 2: Small-K problems ---
    if [[ "$PART" == "2" || "$PART" == "both" ]]; then
        echo ""
        log "Phase 2b: CUTLASS HERK — Small-K (N=${SMALLK_N}, batch=${SMALLK_BATCH})"
        echo ""

        for K in "${SMALLK_K_VALUES[@]}"; do
            local_label="sk_${SMALLK_N}_K${K}"
            log "  N=${SMALLK_N} K=${K} batch=${SMALLK_BATCH}"
            run_cutlass_problem "$SMALLK_N" "$K" "$SMALLK_BATCH" "$local_label"
        done
    fi
fi

# =============================================================================
# Phase 3: TCC benchmarks
# =============================================================================

echo ""
log "Phase 3: TCC benchmarks (format=${TCC_FORMAT})"
echo ""

# --- Part 1: Square ---
if [[ "$PART" == "1" || "$PART" == "both" ]]; then
    log "  TCC Square problems"
    for size in "${SQUARE_SIZES[@]}"; do
        run_tcc_problem "$size" "$size" "$SQUARE_BATCH" "sq_${size}"
    done
fi

# --- Part 2: Small-K ---
if [[ "$PART" == "2" || "$PART" == "both" ]]; then
    log "  TCC Small-K problems"
    for K in "${SMALLK_K_VALUES[@]}"; do
        run_tcc_problem "$SMALLK_N" "$K" "$SMALLK_BATCH" "sk_${SMALLK_N}_K${K}"
    done
fi

# =============================================================================
# Phase 4: Generate comparison summary
# =============================================================================

echo ""
echo "╔══════════════════════════════════════════════════════════════════════╗"
echo "║                     RESULTS SUMMARY                                ║"
echo "╚══════════════════════════════════════════════════════════════════════╝"
echo ""

# Generate comparison using Python for clean formatting
python3 - "$CSV_FILE" << 'PYEOF'
import csv
import sys
from collections import defaultdict

csv_file = sys.argv[1]

# Read CSV
rows = []
try:
    with open(csv_file) as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
except FileNotFoundError:
    print("  No results file found.")
    sys.exit(0)

if not rows:
    print("  No results found.")
    sys.exit(0)

# Group by (N, K, batch)
problems = defaultdict(list)
for r in rows:
    try:
        key = (int(r['N']), int(r['K']), int(r['batch']))
        problems[key].append(r)
    except (ValueError, KeyError):
        continue

# Summary table: best CUTLASS vs TCC for each problem
print("=" * 110)
print(f"{'Problem':>20s}  {'Best CUTLASS':>45s}  {'TCC':>20s}  {'Speedup':>10s}")
print(f"{'(NxKxB)':>20s}  {'Config/Direct/Variant':>25s} {'ms':>8s} {'eff TFLOPS':>10s}  {'ms':>8s} {'eff TFLOPS':>10s}  {'CUT/TCC':>10s}")
print("-" * 110)

for key in sorted(problems.keys()):
    N, K, batch = key
    entries = problems[key]

    cutlass_entries = [e for e in entries
                       if e['tool'] == 'cutlass'
                       and e.get('time_ms','') not in ('FAIL', '-', '')]
    tcc_entries = [e for e in entries
                   if e['tool'] == 'tcc'
                   and e.get('time_ms','') not in ('FAIL', '-', '')]

    best_cutlass = None
    if cutlass_entries:
        try:
            best_cutlass = min(cutlass_entries, key=lambda x: float(x['time_ms']))
        except ValueError:
            pass

    best_tcc = None
    if tcc_entries:
        try:
            best_tcc = min(tcc_entries, key=lambda x: float(x['time_ms']))
        except ValueError:
            pass

    problem_str = f"{N}x{K}x{batch}"

    cut_desc = cut_ms = cut_tflops = "-"
    if best_cutlass:
        cut_desc = f"{best_cutlass['config']}/d={best_cutlass['direct']}/{best_cutlass['variant']}"
        cut_ms = f"{float(best_cutlass['time_ms']):.3f}"
        cut_tflops = f"{float(best_cutlass['effective_tflops']):.1f}"

    tcc_ms = tcc_tflops = "-"
    if best_tcc:
        tcc_ms = f"{float(best_tcc['time_ms']):.3f}"
        tcc_tflops = f"{float(best_tcc['effective_tflops']):.1f}"

    speedup = "-"
    if best_cutlass and best_tcc:
        try:
            s = float(best_tcc['time_ms']) / float(best_cutlass['time_ms'])
            speedup = f"{s:.2f}x"
        except (ValueError, ZeroDivisionError):
            pass

    print(f"{problem_str:>20s}  {cut_desc:>25s} {cut_ms:>8s} {cut_tflops:>10s}  {tcc_ms:>8s} {tcc_tflops:>10s}  {speedup:>10s}")

print("=" * 110)
print()

# Detailed per-problem breakdown
print()
print("DETAILED CUTLASS RESULTS (sorted by time, best first)")
print("=" * 130)

for key in sorted(problems.keys()):
    N, K, batch = key
    entries = problems[key]
    cutlass_entries = [e for e in entries
                       if e['tool'] == 'cutlass'
                       and e.get('time_ms','') not in ('FAIL', '-', '')]
    tcc_entries = [e for e in entries
                   if e['tool'] == 'tcc'
                   and e.get('time_ms','') not in ('FAIL', '-', '')]

    if not cutlass_entries:
        continue

    print(f"\n--- N={N}  K={K}  batch={batch} ---")

    tcc_ms = None
    if tcc_entries:
        try:
            tcc_ms = float(min(tcc_entries, key=lambda x: float(x['time_ms']))['time_ms'])
            print(f"    TCC reference: {tcc_ms:.3f} ms")
        except ValueError:
            pass

    try:
        sorted_entries = sorted(cutlass_entries, key=lambda x: float(x['time_ms']))
    except ValueError:
        continue

    print(f"    {'#':>3s}  {'Config':>16s}  {'Direct':>6s}  {'Variant':>10s}  {'Time(ms)':>10s}  {'TFLOPS':>8s}  {'Eff TFLOPS':>11s}  {'vs TCC':>8s}")
    print(f"    {'---':>3s}  {'------':>16s}  {'-----':>5s}  {'-------':>10s}  {'--------':>10s}  {'------':>8s}  {'----------':>11s}  {'------':>8s}")

    for i, e in enumerate(sorted_entries[:20]):
        vs_tcc = "-"
        if tcc_ms and tcc_ms > 0:
            try:
                vs_tcc = f"{tcc_ms / float(e['time_ms']):.2f}x"
            except (ValueError, ZeroDivisionError):
                pass

        try:
            print(f"    {i+1:>3d}  {e['config']:>16s}  {e['direct']:>6s}  {e['variant']:>10s}  "
                  f"{float(e['time_ms']):>10.3f}  {float(e['reported_tflops']):>8.1f}  "
                  f"{float(e['effective_tflops']):>11.1f}  {vs_tcc:>8s}")
        except ValueError:
            continue

print()
print("=" * 130)
print()
print("Speedup > 1.0 = CUTLASS faster than TCC")
print("Effective TFLOPS = 8*N^2*K*batch / time  (standard complex HERK FLOPs)")
print(f"CUTLASS reported TFLOPS uses 6*N^2*K*batch (3 sub-GEMMs vs TCC's 4)")
print()
print("Note: Benchmarks use the internal GemmComplexSm100::HERK_batched() API.")
print("The unified PIMPL API (herk/gemm with runtime precision enums) is validated")
print("separately via test_herk_int4, test_gemm_api, and test_tune_api in Phase 1b.")
print()
PYEOF

echo ""
log "Done. Full CSV: ${CSV_FILE}"
log "Individual logs: ${RESULTS_DIR}/"
echo ""
