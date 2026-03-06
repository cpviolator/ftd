#!/usr/bin/env bash
# build_sweep_sm90.sh — Compile-time configuration sweep for SM90a (Hopper)
#
# Cycles through meaningful compile-time combinations, doing:
#   1. cmake configure with unique build dir
#   2. build
#   3. run ctest to validate
#
# Usage:
#   bash build_sweep_sm90.sh                  # tier 1 (optimal + tile shapes)
#   bash build_sweep_sm90.sh --tier 2         # + schedule/cluster variants
#   bash build_sweep_sm90.sh --tier all       # + cross-product combinations
#   bash build_sweep_sm90.sh --dry-run        # preview configs without building
#   bash build_sweep_sm90.sh --clean          # remove sweep_builds_sm90/ and results

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CUTLASS_DIR="${CUTLASS_DIR:-/home/deanhowarth/applications/cutlass}"
SOURCE_DIR="$SCRIPT_DIR"

# ============================================================================
# Defaults
# ============================================================================

TIER="1"
DRY_RUN=false
CLEAN=false

# ============================================================================
# Argument parsing
# ============================================================================

while [[ $# -gt 0 ]]; do
    case "$1" in
        --tier)        TIER="$2"; shift 2 ;;
        --source)      SOURCE_DIR="$2"; shift 2 ;;
        --dry-run)     DRY_RUN=true; shift ;;
        --clean)       CLEAN=true; shift ;;
        --cutlass)     CUTLASS_DIR="$2"; shift 2 ;;
        -h|--help)
            cat <<'HELP'
Usage: bash build_sweep_sm90.sh [OPTIONS]

Compile-time configuration sweep for SM90a (Hopper GH200/H100).
Builds each configuration and runs ctest to validate.

Options:
  --tier NUM        Tier level: 1, 2, or all
  --source DIR      CMake source directory (default: script directory)
  --dry-run         Preview configs without building
  --clean           Remove sweep_builds_sm90/ and sweep_results_sm90/
  --cutlass DIR     Path to CUTLASS (default: $CUTLASS_DIR)

Tiers:
  1:    Optimal + tile shapes (4 configs)
  2:    + schedule, accum, cluster variants (5 more = 9 total)
  all:  + cross-product combinations (7 more = 16 total)

After each successful build, ctest is run to validate.
Use find_best_flags.sh --build-dir sweep_builds_sm90/ to benchmark across builds.
HELP
            exit 0
            ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

if $CLEAN; then
    echo "Cleaning sweep_builds_sm90/ and sweep_results_sm90/..."
    rm -rf "${SCRIPT_DIR}/sweep_builds_sm90" "${SCRIPT_DIR}/sweep_results_sm90"
    echo "Done."
    exit 0
fi

# ============================================================================
# Configuration matrix
# ============================================================================
# Each config: "NAME|CMAKE_FLAGS"
# Common flags (-DCUTLASS_DIR, -DCOMPLEX_FP8_ARCH=90a) added automatically.
#
# SM90 parameters:
#   COMPLEX_FP8_TILE_M/N/K        CTA tile shape (default: 128x256x128)
#   COMPLEX_FP8_CLUSTER_M/N       Cluster shape (default: 1x1)
#   COMPLEX_FP8_USE_PINGPONG      PingPong schedule (best at 8192^3+)
#   COMPLEX_FP8_USE_FAST_ACCUM    FastAccum (default ON; OFF for accuracy)
#   COMPLEX_FP8_EXTRA_VECTORIZATION  wider mem coalescing (default ON)
#   COMPLEX_FP8_DISABLE_GROUPED_GEMM  force per-slab triangle (for comparison)
#
# NOT swept (known issues on SM90):
#   COMPLEX_FP8_DEVICE_LTO       broken (nvcc strips 'a' suffix)
#   COMPLEX_FP8_MAX_REGISTERS    ptxas rejects 255 for large kernels

declare -a CONFIGS=()

add_config() {
    CONFIGS+=("$1|$2")
}

# --- Tier 1: Core tile shapes + optimal guess ---
# Optimal guess rationale (SM90):
#   - 128x256x128 tile is the benchmark winner on GH200 at 4096^3
#   - TILE_K must be >=128 for FP8 wgmma (64 is invalid despite CMake allowing it)
#   - FastAccum ON (default) — disable only if training accuracy needed
#   - Grouped GEMM ON (default) — single-launch triangle, better for batched HERK
#   - Cluster 1x1 (default) — safest, multicast helps only on multi-GPC configs
#   - Extra vectorization ON (default) — wider memory coalescing
#   - Device LTO broken on SM90 (nvcc strips 'a' suffix from sm_90a)
#   - MAX_REGISTERS rejected by ptxas for large CUTLASS kernels on SM90
#   - Result: defaults ARE the optimal guess for SM90
add_config "optimal"    ""
add_config "t256x128"   "-DCOMPLEX_FP8_TILE_M=256 -DCOMPLEX_FP8_TILE_N=128"
add_config "t128x128"   "-DCOMPLEX_FP8_TILE_M=128 -DCOMPLEX_FP8_TILE_N=128"
add_config "t128x256_novec" "-DCOMPLEX_FP8_EXTRA_VECTORIZATION=OFF"

# --- Tier 2: Schedule, accumulation, and cluster variants ---
if [[ "$TIER" == "2" || "$TIER" == "all" ]]; then
    add_config "t128x256_pp"       "-DCOMPLEX_FP8_USE_PINGPONG=ON"
    add_config "t128x256_nofacc"   "-DCOMPLEX_FP8_USE_FAST_ACCUM=OFF"
    add_config "t128x256_no_grp"   "-DCOMPLEX_FP8_DISABLE_GROUPED_GEMM=ON"
    add_config "t128x256_cl2x1"    "-DCOMPLEX_FP8_CLUSTER_M=2"
    add_config "t128x256_cl1x2"    "-DCOMPLEX_FP8_CLUSTER_N=2"
fi

# --- Tier 3 (all): Cross-product combinations ---
if [[ "$TIER" == "all" ]]; then
    add_config "t256x128_pp"       "-DCOMPLEX_FP8_TILE_M=256 -DCOMPLEX_FP8_TILE_N=128 -DCOMPLEX_FP8_USE_PINGPONG=ON"
    add_config "t256x128_cl2x1"    "-DCOMPLEX_FP8_TILE_M=256 -DCOMPLEX_FP8_TILE_N=128 -DCOMPLEX_FP8_CLUSTER_M=2"
    add_config "t128x128_pp"       "-DCOMPLEX_FP8_TILE_M=128 -DCOMPLEX_FP8_TILE_N=128 -DCOMPLEX_FP8_USE_PINGPONG=ON"
    add_config "t128x128_nofacc"   "-DCOMPLEX_FP8_TILE_M=128 -DCOMPLEX_FP8_TILE_N=128 -DCOMPLEX_FP8_USE_FAST_ACCUM=OFF"
    add_config "t128x256_cl2x1_pp" "-DCOMPLEX_FP8_CLUSTER_M=2 -DCOMPLEX_FP8_USE_PINGPONG=ON"
    add_config "t128x256_cl1x2_pp" "-DCOMPLEX_FP8_CLUSTER_N=2 -DCOMPLEX_FP8_USE_PINGPONG=ON"
    add_config "t256x128_nofacc"   "-DCOMPLEX_FP8_TILE_M=256 -DCOMPLEX_FP8_TILE_N=128 -DCOMPLEX_FP8_USE_FAST_ACCUM=OFF"
fi

# ============================================================================
# Setup
# ============================================================================

BUILD_ROOT="${SCRIPT_DIR}/sweep_builds_sm90"
RESULTS_DIR="${SCRIPT_DIR}/sweep_results_sm90"

mkdir -p "$BUILD_ROOT" "$RESULTS_DIR"

NUM_CONFIGS=${#CONFIGS[@]}

echo "=========================================="
echo " SM90a Build Sweep — ${NUM_CONFIGS} configs"
echo " Tier: ${TIER}"
echo " Source: ${SOURCE_DIR}"
echo " CUTLASS: ${CUTLASS_DIR}"
echo "=========================================="
echo ""

# Print config summary
echo "  Compile-time configs:"
for i in "${!CONFIGS[@]}"; do
    IFS='|' read -r name flags <<< "${CONFIGS[$i]}"
    if [[ -z "$flags" ]]; then flags="(defaults: 128x256x128, Coop+FastAccum, Cluster 1x1)"; fi
    printf "  [%d/%d] %-22s %s\n" $((i+1)) "$NUM_CONFIGS" "$name" "$flags"
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
    IFS='|' read -r name flags <<< "${CONFIGS[$i]}"

    echo "=========================================="
    echo " [$((i+1))/${NUM_CONFIGS}] Config: ${name}"
    echo "=========================================="

    build_dir="${BUILD_ROOT}/${name}"
    log_file="${RESULTS_DIR}/${name}_build.log"

    # --- Configure ---
    echo "  Configuring (SM90a)..."
    # shellcheck disable=SC2086
    if ! cmake -S "$SOURCE_DIR" -B "$build_dir" \
        -DCUTLASS_DIR="$CUTLASS_DIR" \
        -DCOMPLEX_FP8_ARCH=90a \
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
echo " SM90a Sweep Complete"
echo "=========================================="
echo ""
printf "  %-22s %s\n" "CONFIG" "STATUS"
printf "  %-22s %s\n" "------" "------"
for i in "${!CONFIGS[@]}"; do
    IFS='|' read -r name flags <<< "${CONFIGS[$i]}"
    status="${CONFIG_STATUS[$name]:-UNKNOWN}"
    printf "  %-22s %s\n" "$name" "$status"
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
