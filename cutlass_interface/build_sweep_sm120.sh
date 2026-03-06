#!/usr/bin/env bash
# build_sweep.sh — Compile-time configuration sweep for SM121 (GB10 Spark)
#
# Cycles through meaningful compile-time combinations, doing:
#   1. cmake configure with unique build dir
#   2. build
#   3. run ctest to validate
#
# Usage:
#   bash build_sweep.sh                    # tier 1 (optimal guesses + baseline)
#   bash build_sweep.sh --tier 2           # + codegen flags
#   bash build_sweep.sh --tier all         # + block-scaled + MMA_N variations
#   bash build_sweep.sh --dry-run          # preview configs without building
#   bash build_sweep.sh --clean            # remove sweep_builds/ and sweep_results/

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CUTLASS_DIR="${CUTLASS_DIR:-/home/deanhowarth/applications/cutlass}"
SOURCE_DIR="$SCRIPT_DIR"
ARCH="${ARCH:-120a}"
TIER="1"
DRY_RUN=false
CLEAN=false

# ============================================================================
# Argument parsing
# ============================================================================

while [[ $# -gt 0 ]]; do
    case "$1" in
        --tier)    TIER="$2"; shift 2 ;;
        --source)  SOURCE_DIR="$2"; shift 2 ;;
        --dry-run) DRY_RUN=true; shift ;;
        --clean)   CLEAN=true; shift ;;
        --cutlass) CUTLASS_DIR="$2"; shift 2 ;;
        --arch)    ARCH="$2"; shift 2 ;;
        -h|--help)
            echo "Usage: bash build_sweep.sh [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --tier NUM     Tier level: 1, 2, or all"
            echo "  --source DIR   CMake source directory (default: script directory)"
            echo "  --dry-run      Preview configs without building"
            echo "  --clean        Remove sweep_builds/ and sweep_results/"
            echo "  --cutlass DIR  Path to CUTLASS (default: \$CUTLASS_DIR)"
            echo "  --arch ARCH    GPU architecture (default: 120a)"
            echo ""
            echo "Tier 1:    Optimal guesses + baseline (3 configs)"
            echo "Tier 2:    + codegen flags (4 more)"
            echo "Tier all:  + block-scaled + MMA_N variations (4 more)"
            echo ""
            echo "After each successful build, ctest is run to validate."
            echo "Use find_best_flags.sh to benchmark across builds."
            exit 0
            ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

if $CLEAN; then
    echo "Cleaning sweep_builds/ and sweep_results/..."
    rm -rf "${SCRIPT_DIR}/sweep_builds" "${SCRIPT_DIR}/sweep_results"
    echo "Done."
    exit 0
fi

# ============================================================================
# Configuration matrix
# ============================================================================

# Each config: NAME CMAKE_FLAGS...
# Common flags are added automatically (CUTLASS_DIR, ARCH)

declare -a CONFIGS=()

add_config() {
    CONFIGS+=("$*")
}

# Note: FP6+FP4 are always included because the library code has MXFP
# function references that aren't fully #ifdef-guarded when disabled.
# This doesn't affect FP8 kernel codegen.

FP6FP4="-DCOMPLEX_SM100_ENABLE_FP6=ON -DCOMPLEX_SM100_ENABLE_FP4=ON"

# --- Tier 1: Optimal guesses + baseline ---
# SM120 optimal (HERK) — matches README "Recommended Build for SM120":
#   - STAGES=3 explicit (auto-defaults on SM120 but explicit is clearer)
#   - FP8 tile auto-selects 128x64 on SM120 (fits 3-stage in 99 KB SMEM)
#   - Direct HERK kernel (24 KB SMEM) bypasses CUTLASS tiles entirely
#   - Secondary flags (LTO, reg255, novec) have <5% impact within stg3
#   - Pipeline stages dominate all other compile-time flags
# SM120 optimal (GEMM) — wider 128x128 FP8 tile for GEMM workloads:
#   - TILE_N=128 forces 128x128 FP8 tile (2-stage auto carveout)
#   - 5-28% faster for large square/rectangular GEMMs (N>=2048)
add_config "optimal_herk" "$FP6FP4 -DCOMPLEX_FP8_SM100_STAGES=3"
add_config "optimal_gemm" "$FP6FP4 -DCOMPLEX_FP8_SM100_STAGES=3 -DCOMPLEX_SM100_FP8_TILE_N=128"
add_config "baseline"     "$FP6FP4"

# --- Tier 2: Codegen flags (each <5% impact, but may compound) ---
if [[ "$TIER" == "2" || "$TIER" == "all" ]]; then
    add_config "stg3_lto"    "$FP6FP4 -DCOMPLEX_FP8_SM100_STAGES=3 -DCOMPLEX_FP8_DEVICE_LTO=ON"
    add_config "stg3_reg255" "$FP6FP4 -DCOMPLEX_FP8_SM100_STAGES=3 -DCOMPLEX_FP8_MAX_REGISTERS=255"
    add_config "stg3_novec"  "$FP6FP4 -DCOMPLEX_FP8_SM100_STAGES=3 -DCOMPLEX_FP8_EXTRA_VECTORIZATION=OFF"
    add_config "stg3_lto_reg255" "$FP6FP4 -DCOMPLEX_FP8_SM100_STAGES=3 -DCOMPLEX_FP8_DEVICE_LTO=ON -DCOMPLEX_FP8_MAX_REGISTERS=255"
fi

# --- Tier 3: Block-scaled tile + MMA_N variations ---
if [[ "$TIER" == "all" ]]; then
    add_config "stg3_bsn64"  "$FP6FP4 -DCOMPLEX_FP8_SM100_STAGES=3 -DCOMPLEX_SM100_BLKSCALED_MMA_N=64"
    add_config "stg3_n64"    "$FP6FP4 -DCOMPLEX_FP8_SM100_STAGES=3 -DCOMPLEX_FP8_SM100_MMA_N=64"
    add_config "gemm_lto"    "$FP6FP4 -DCOMPLEX_FP8_SM100_STAGES=3 -DCOMPLEX_SM100_FP8_TILE_N=128 -DCOMPLEX_FP8_DEVICE_LTO=ON"
    add_config "stg3_novec_lto" "$FP6FP4 -DCOMPLEX_FP8_SM100_STAGES=3 -DCOMPLEX_FP8_EXTRA_VECTORIZATION=OFF -DCOMPLEX_FP8_DEVICE_LTO=ON"
fi

# ============================================================================
# Setup
# ============================================================================

BUILD_ROOT="${SCRIPT_DIR}/sweep_builds"
RESULTS_DIR="${SCRIPT_DIR}/sweep_results"

mkdir -p "$BUILD_ROOT" "$RESULTS_DIR"

NUM_CONFIGS=${#CONFIGS[@]}
echo "=========================================="
echo " Build Sweep — ${NUM_CONFIGS} configs"
echo " Tier: ${TIER}"
echo " Source: ${SOURCE_DIR}"
echo " CUTLASS: ${CUTLASS_DIR}"
echo " Arch: ${ARCH}"
echo "=========================================="
echo ""

# Print config summary
echo "  Compile-time configs:"
for i in "${!CONFIGS[@]}"; do
    cfg="${CONFIGS[$i]}"
    name="${cfg%% *}"
    flags="${cfg#* }"
    if [[ "$name" == "$flags" ]]; then flags="(defaults)"; fi
    printf "  [%d/%d] %-16s %s\n" $((i+1)) "$NUM_CONFIGS" "$name" "$flags"
done
echo ""

if $DRY_RUN; then
    echo "(dry-run mode — exiting without building)"
    exit 0
fi

# ============================================================================
# Build + test loop
# ============================================================================

PASS=0
FAIL=0
declare -A CONFIG_STATUS

for i in "${!CONFIGS[@]}"; do
    cfg="${CONFIGS[$i]}"
    name="${cfg%% *}"
    flags="${cfg#* }"
    if [[ "$name" == "$flags" ]]; then flags=""; fi

    echo "=========================================="
    echo " [$((i+1))/${NUM_CONFIGS}] Config: ${name}"
    echo "=========================================="

    build_dir="${BUILD_ROOT}/${name}"
    log_file="${RESULTS_DIR}/${name}_build.log"

    # --- Configure ---
    echo "  Configuring..."
    # shellcheck disable=SC2086
    if ! cmake -S "$SOURCE_DIR" -B "$build_dir" \
        -DCUTLASS_DIR="$CUTLASS_DIR" \
        -DCOMPLEX_FP8_ARCH="$ARCH" \
        $flags \
        > "$log_file" 2>&1; then
        echo "  CONFIGURE FAILED — see ${log_file}"
        CONFIG_STATUS[$name]="CONFIGURE_FAIL"
        ((FAIL++)) || true
        continue
    fi

    # --- Build ---
    echo "  Building ($(nproc) threads)..."
    if ! cmake --build "$build_dir" -j"$(nproc)" >> "$log_file" 2>&1; then
        echo "  BUILD FAILED — see ${log_file}"
        CONFIG_STATUS[$name]="BUILD_FAIL"
        ((FAIL++)) || true
        continue
    fi

    echo "  Build OK"

    # --- Test ---
    echo "  Running ctest..."
    test_log="${RESULTS_DIR}/${name}_test.log"
    if ! ctest --test-dir "$build_dir" --output-on-failure --timeout 300 > "$test_log" 2>&1; then
        echo "  TEST FAILED — see ${test_log}"
        CONFIG_STATUS[$name]="TEST_FAIL"
        ((FAIL++)) || true
        continue
    fi

    echo "  Tests OK"
    CONFIG_STATUS[$name]="OK"
    ((PASS++)) || true
done

# ============================================================================
# Summary
# ============================================================================

echo ""
echo "=========================================="
echo " Sweep Complete"
echo "=========================================="
echo ""
printf "  %-16s %s\n" "CONFIG" "STATUS"
printf "  %-16s %s\n" "------" "------"
for i in "${!CONFIGS[@]}"; do
    cfg="${CONFIGS[$i]}"
    name="${cfg%% *}"
    status="${CONFIG_STATUS[$name]:-UNKNOWN}"
    printf "  %-16s %s\n" "$name" "$status"
done
echo ""
echo "  Passed: ${PASS} / ${NUM_CONFIGS}"
echo "  Failed: ${FAIL} / ${NUM_CONFIGS}"
echo ""
echo "  Build dirs: ${BUILD_ROOT}/"
echo "  Logs:       ${RESULTS_DIR}/"
echo ""
echo "  Next: bash find_best_flags.sh --build-dir ${BUILD_ROOT}"
echo ""
