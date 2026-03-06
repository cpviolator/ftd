# CMakeLists.txt — Complex FP8 GEMM with CUTLASS 3.x/4.x
#
# Supports:
#   - Hopper SM90a  (CUTLASS 3.5+, CUDA 12.9+)
#   - Blackwell SM100a / GB200  (CUTLASS 4.0+, CUDA 13.x)
#   - Blackwell SM120a / GB10 Spark / RTX 50-series  (CUTLASS 4.0+, CUDA 13.x)
#
# Build:
#   mkdir build && cd build
#
#   # Build all (default — FP8 + FP6 + FP4, all tile configs):
#   cmake .. -DCUTLASS_DIR=/path/to/cutlass -DCOMPLEX_FP8_ARCH=120a
#
#   # Minimal build (FP8 only, faster compile):
#   cmake .. -DCUTLASS_DIR=/path/to/cutlass -DCOMPLEX_FP8_ARCH=120a \
#       -DCOMPLEX_FP8_BUILD_ALL=OFF -DCOMPLEX_SM100_ENABLE_FP6=OFF -DCOMPLEX_SM100_ENABLE_FP4=OFF
#
#   # Hopper (default arch):
#   cmake .. -DCUTLASS_DIR=/path/to/cutlass
#
#   # Blackwell GB200 target:
#   cmake .. -DCUTLASS_DIR=/path/to/cutlass -DCOMPLEX_FP8_ARCH=100a
#
#   make -j$(nproc)
#
# Or use cmake-gui / ccmake to configure kernel tuning options interactively.

cmake_minimum_required(VERSION 3.24)
project(ComplexFP8Gemm LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# =============================================================================
# Architecture Selection
# =============================================================================

set(COMPLEX_FP8_ARCH "90a" CACHE STRING
    "Target GPU architecture. 90a=Hopper, 100a=Blackwell GB200, 120a=Blackwell GB10/RTX50")
set_property(CACHE COMPLEX_FP8_ARCH PROPERTY STRINGS "90a" "100a" "120a")

# Derive architecture family for conditional logic
if(COMPLEX_FP8_ARCH STREQUAL "90a")
    set(ARCH_FAMILY "hopper")
    set(ARCH_SM_NUMBER 90)
elseif(COMPLEX_FP8_ARCH STREQUAL "100a")
    set(ARCH_FAMILY "blackwell")
    set(ARCH_SM_NUMBER 100)
elseif(COMPLEX_FP8_ARCH STREQUAL "120a")
    set(ARCH_FAMILY "blackwell")
    set(ARCH_SM_NUMBER 120)
else()
    message(FATAL_ERROR "Unknown architecture: ${COMPLEX_FP8_ARCH}. Use 90a, 100a, or 120a.")
endif()

# ARCHITECTURE FLAG HANDLING:
# CMake 3.24+ handles the 'a' suffix (arch-accelerated) in CUDA_ARCHITECTURES
# natively, generating correct -gencode flags for both compile and device link.
#
# SM120 is special: GB10 Spark is compute 12.1 (sm_121), not 12.0 (sm_120).
# Neither CMake 'native' nor nvcc '-arch=native' append the 'a' suffix.
# We detect the actual compute capability and pass -arch=sm_121a manually.
set(FORCED_CUDA_ARCH "")
if(COMPLEX_FP8_ARCH STREQUAL "120a")
    set(CMAKE_CUDA_ARCHITECTURES OFF)  # Can't express 121a via CUDA_ARCHITECTURES
    # Query the GPU's compute capability via nvidia-smi
    execute_process(
        COMMAND nvidia-smi --query-gpu=compute_cap --format=csv,noheader
        OUTPUT_VARIABLE GPU_COMPUTE_CAP
        OUTPUT_STRIP_TRAILING_WHITESPACE
        RESULT_VARIABLE NVSMI_RESULT
    )
    if(NOT NVSMI_RESULT EQUAL 0)
        message(FATAL_ERROR
            "Failed to query GPU compute capability via nvidia-smi. "
            "SM120 builds require compiling on the target machine.")
    endif()
    # Handle multiple GPUs — take the first line
    string(REGEX MATCH "^[0-9]+\\.[0-9]+" GPU_COMPUTE_CAP "${GPU_COMPUTE_CAP}")
    # Convert "12.1" → "121", then "sm_121a"
    string(REPLACE "." "" GPU_ARCH_NUM "${GPU_COMPUTE_CAP}")
    set(FORCED_CUDA_ARCH "sm_${GPU_ARCH_NUM}a")
    message(STATUS "SM120: detected GPU compute ${GPU_COMPUTE_CAP} → -arch=${FORCED_CUDA_ARCH}")
else()
    # SM90a/SM100a: CMake 3.24 handles 'a' suffix natively
    set(CMAKE_CUDA_ARCHITECTURES ${COMPLEX_FP8_ARCH})
    message(STATUS "CUDA_ARCHITECTURES=${COMPLEX_FP8_ARCH}")
endif()

# CUDA toolkit version validation
if(ARCH_FAMILY STREQUAL "hopper")
    if(CMAKE_CUDA_COMPILER_VERSION VERSION_LESS "12.9")
        message(WARNING
            "CUDA ${CMAKE_CUDA_COMPILER_VERSION} detected. CUDA 12.9+ is strongly "
            "recommended for SM90a real-arch codegen and C7510 mitigation.")
    endif()
elseif(ARCH_FAMILY STREQUAL "blackwell")
    if(CMAKE_CUDA_COMPILER_VERSION VERSION_LESS "13.0")
        message(FATAL_ERROR
            "CUDA ${CMAKE_CUDA_COMPILER_VERSION} detected. SM${ARCH_SM_NUMBER}a requires "
            "CUDA 13.x for real-architecture codegen. Blackwell build cannot proceed.")
    endif()
endif()

# =============================================================================
# Build-All Toggle
# =============================================================================
# When ON (default), all available precision and feature variants are enabled.
# Set to OFF for manual control over individual precision options (faster builds).

option(COMPLEX_FP8_BUILD_ALL
    "Build all available precision and feature variants. When ON, enables FP6, FP4, \
and all architecture-appropriate features. Set to OFF to manually control options."
    ON)

find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    message(STATUS "OpenMP: FOUND (${OpenMP_CXX_FLAGS})")
else()
    message(STATUS "OpenMP: not found (CPU cross-check will be single-threaded)")
endif()

set(CUTLASS_DIR "" CACHE PATH "Path to CUTLASS source tree (must contain include/ and tools/)")
if(CUTLASS_DIR STREQUAL "")
    message(FATAL_ERROR "Please set CUTLASS_DIR to the CUTLASS source tree, e.g. -DCUTLASS_DIR=/path/to/cutlass")
endif()

set(CUTLASS_INCLUDE_DIRS
    ${CUTLASS_DIR}/include
    ${CUTLASS_DIR}/tools/util/include
)

# Extract CUTLASS version from version.h for compile-time defines
set(CUTLASS_VERSION_HEADER "${CUTLASS_DIR}/include/cutlass/version.h")
if(EXISTS "${CUTLASS_VERSION_HEADER}")
    file(STRINGS "${CUTLASS_VERSION_HEADER}" _cv_major REGEX "#define CUTLASS_MAJOR")
    file(STRINGS "${CUTLASS_VERSION_HEADER}" _cv_minor REGEX "#define CUTLASS_MINOR")
    file(STRINGS "${CUTLASS_VERSION_HEADER}" _cv_patch REGEX "#define CUTLASS_PATCH")
    string(REGEX MATCH "[0-9]+" CUTLASS_VER_MAJOR "${_cv_major}")
    string(REGEX MATCH "[0-9]+" CUTLASS_VER_MINOR "${_cv_minor}")
    string(REGEX MATCH "[0-9]+" CUTLASS_VER_PATCH "${_cv_patch}")
    message(STATUS "CUTLASS version: ${CUTLASS_VER_MAJOR}.${CUTLASS_VER_MINOR}.${CUTLASS_VER_PATCH}")
else()
    message(WARNING "CUTLASS version.h not found at ${CUTLASS_VERSION_HEADER}")
    set(CUTLASS_VER_MAJOR 0)
    set(CUTLASS_VER_MINOR 0)
    set(CUTLASS_VER_PATCH 0)
endif()

# =============================================================================
# Hopper (SM90) Kernel Tuning Options
# =============================================================================
# Only relevant when COMPLEX_FP8_ARCH=90a. These control the CUTLASS template
# parameters compiled into gemm_complex_fp8.hpp.

set(COMPLEX_FP8_TILE_M 128 CACHE STRING
    "[SM90] Tile M dimension. (64, 128, 256)")
set_property(CACHE COMPLEX_FP8_TILE_M PROPERTY STRINGS "64" "128" "256")

set(COMPLEX_FP8_TILE_N 256 CACHE STRING
    "[SM90] Tile N dimension. 256 won benchmark at 4096³. (64, 128, 256)")
set_property(CACHE COMPLEX_FP8_TILE_N PROPERTY STRINGS "64" "128" "256")

set(COMPLEX_FP8_TILE_K 128 CACHE STRING
    "[SM90] Tile K dimension. Must be >=128 for FP8 wgmma. (128)")
set_property(CACHE COMPLEX_FP8_TILE_K PROPERTY STRINGS "64" "128" "256")

set(COMPLEX_FP8_CLUSTER_M 1 CACHE STRING
    "[SM90] Cluster M dim. >1 enables TMA multicast. (1, 2, 4)")
set_property(CACHE COMPLEX_FP8_CLUSTER_M PROPERTY STRINGS "1" "2" "4")

set(COMPLEX_FP8_CLUSTER_N 1 CACHE STRING
    "[SM90] Cluster N dim. >1 enables TMA multicast. (1, 2, 4)")
set_property(CACHE COMPLEX_FP8_CLUSTER_N PROPERTY STRINGS "1" "2" "4")

option(COMPLEX_FP8_USE_PINGPONG
    "[SM90] Use PingpongFP8FastAccum schedule. Best at 8192³+."
    OFF)

option(COMPLEX_FP8_USE_FAST_ACCUM
    "[SM90] Use FastAccum (skip FP32 promotion). Disable for training accuracy."
    ON)

option(COMPLEX_FP8_DISABLE_GROUPED_GEMM
    "[SM90] Disable grouped GEMM for triangle HERK. Forces per-slab fallback (for debugging/comparison)."
    OFF)

# =============================================================================
# Blackwell (SM100/SM120) Kernel Tuning Options
# =============================================================================
# Only relevant when COMPLEX_FP8_ARCH=100a or 120a. These control the CUTLASS 4.x
# template parameters in gemm_complex_sm100.hpp.
#
# MmaTileShape: the tile consumed by 1 or 2 SM tcgen05.mma instructions.
# This is NOT the CTA output tile (that is derived by the builder).
#
# Valid shapes for vanilla FP8 E4M3 (TN layout):
#   1SM: 128x64x128, 128x128x128, 128x256x128
#   2SM: 256x64x128, 256x128x128, 256x256x128

set(COMPLEX_FP8_SM100_MMA_M 128 CACHE STRING
    "[Advanced] Default MMA tile M. FP8 tiles are all compiled (see GemmConfig). \
This only affects block-scaled (FP6/FP4) tile defaults.")
set_property(CACHE COMPLEX_FP8_SM100_MMA_M PROPERTY STRINGS "128" "256")
mark_as_advanced(COMPLEX_FP8_SM100_MMA_M)

set(COMPLEX_FP8_SM100_MMA_N 128 CACHE STRING
    "[Advanced] Default MMA tile N. FP8 tiles are all compiled (see GemmConfig). \
This only affects block-scaled (FP6/FP4) tile defaults.")
set_property(CACHE COMPLEX_FP8_SM100_MMA_N PROPERTY STRINGS "64" "128" "256")
mark_as_advanced(COMPLEX_FP8_SM100_MMA_N)

set(COMPLEX_SM100_FP8_TILE_N 0 CACHE STRING
    "[Advanced, SM120] Override FP8 tile N dimension. 0=auto (64 for HERK, see docs). \
Set to 128 for GEMM-optimized builds (dedisp, beamformer). Only affects SM120.")
set_property(CACHE COMPLEX_SM100_FP8_TILE_N PROPERTY STRINGS "0" "64" "128")
mark_as_advanced(COMPLEX_SM100_FP8_TILE_N)

set(COMPLEX_FP8_SM100_MMA_K 128 CACHE STRING
    "[Advanced] MMA tile K dimension. (128)")
set_property(CACHE COMPLEX_FP8_SM100_MMA_K PROPERTY STRINGS "128")
mark_as_advanced(COMPLEX_FP8_SM100_MMA_K)

set(COMPLEX_FP8_SM100_CLUSTER_M 1 CACHE STRING
    "[Advanced] Cluster M dim. FP8 cluster variants are all compiled (see GemmConfig).")
set_property(CACHE COMPLEX_FP8_SM100_CLUSTER_M PROPERTY STRINGS "1" "2" "4")
mark_as_advanced(COMPLEX_FP8_SM100_CLUSTER_M)

set(COMPLEX_FP8_SM100_CLUSTER_N 1 CACHE STRING
    "[Advanced] Cluster N dim. FP8 cluster variants are all compiled (see GemmConfig).")
set_property(CACHE COMPLEX_FP8_SM100_CLUSTER_N PROPERTY STRINGS "1" "2" "4")
mark_as_advanced(COMPLEX_FP8_SM100_CLUSTER_N)

option(COMPLEX_FP8_SM100_USE_2SM
    "[Advanced] Use 2SM cooperative MMA. MMA_M must be 256, Cluster M must be multiple of 2."
    OFF)
mark_as_advanced(COMPLEX_FP8_SM100_USE_2SM)

set(COMPLEX_FP8_SM100_STAGES 0 CACHE STRING
    "[SM100/SM120] Pipeline stages. 0=auto. SM120: auto-defaults to 3 (3-9x faster than 2).")
set_property(CACHE COMPLEX_FP8_SM100_STAGES PROPERTY STRINGS "0" "1" "2" "3" "4")

# =============================================================================
# Blackwell Multi-Precision Options (FP6/FP4)
# =============================================================================
# Enable narrow precision kernel instantiations. OFF by default to avoid
# build time impact (~15 min per additional precision).
#
# Each adds one KernelTypeChain instantiation and associated cast kernels.
# Build time: FP8-only ~15 min, +FP4 ~30 min, +FP6 ~60 min, +all ~90 min.

# Default FP6/FP4 to ON for Blackwell (SM100/SM120), OFF for Hopper (SM90)
if(ARCH_FAMILY STREQUAL "blackwell")
    set(_FP6_DEFAULT ON)
    set(_FP4_DEFAULT ON)
else()
    set(_FP6_DEFAULT OFF)
    set(_FP4_DEFAULT OFF)
endif()

option(COMPLEX_SM100_ENABLE_FP6
    "[SM100/SM120] Enable FP6 (E3M2 + E2M3) compute precision. Adds ~30 min build time."
    ${_FP6_DEFAULT})

option(COMPLEX_SM100_ENABLE_FP4
    "[SM100/SM120] Enable FP4 (E2M1) compute precision. Adds ~15 min build time."
    ${_FP4_DEFAULT})

# BUILD_ALL overrides: force-enable features when BUILD_ALL=ON
if(COMPLEX_FP8_BUILD_ALL)
    if(ARCH_FAMILY STREQUAL "blackwell")
        if(DEFINED CACHE{COMPLEX_SM100_ENABLE_FP6} AND NOT COMPLEX_SM100_ENABLE_FP6)
            message(WARNING "BUILD_ALL=ON overrides COMPLEX_SM100_ENABLE_FP6=OFF → ON")
        endif()
        if(DEFINED CACHE{COMPLEX_SM100_ENABLE_FP4} AND NOT COMPLEX_SM100_ENABLE_FP4)
            message(WARNING "BUILD_ALL=ON overrides COMPLEX_SM100_ENABLE_FP4=OFF → ON")
        endif()
        set(COMPLEX_SM100_ENABLE_FP6 ON CACHE BOOL "" FORCE)
        set(COMPLEX_SM100_ENABLE_FP4 ON CACHE BOOL "" FORCE)
    endif()
endif()

# Per-precision MMA tile K dimensions (advanced tuning)
set(COMPLEX_SM100_FP6_MMA_K 128 CACHE STRING
    "[SM100] MMA tile K dimension for FP6 precision. (128)")
set_property(CACHE COMPLEX_SM100_FP6_MMA_K PROPERTY STRINGS "128" "256")

set(COMPLEX_SM100_FP4_MMA_K 256 CACHE STRING
    "[SM100] MMA tile K dimension for FP4 precision. (256)")
set_property(CACHE COMPLEX_SM100_FP4_MMA_K PROPERTY STRINGS "128" "256" "512")

# NOTE: COMPLEX_SM100_MULTI_CONFIG has been removed. All tile/cluster configs
# are now always compiled into every binary and selected at runtime via GemmConfig.
# The strategy autotuner (tune=true) benchmarks across all valid configs automatically.

# Block-scaled tile M/N (for FP6/FP4 MXFP kernels)
# These kernels need extra SMEM for scale factors. On SM120 (99 KB),
# the FP8 default 128×128 may not fit. Try 64×128 or 128×64 if SMEM overflows.
set(COMPLEX_SM100_BLKSCALED_MMA_M ${COMPLEX_FP8_SM100_MMA_M} CACHE STRING
    "[SM100/SM120] Block-scaled tile M. Reduce on SM120 if SMEM overflows.")
set_property(CACHE COMPLEX_SM100_BLKSCALED_MMA_M PROPERTY STRINGS "64" "128" "256")

set(COMPLEX_SM100_BLKSCALED_MMA_N ${COMPLEX_FP8_SM100_MMA_N} CACHE STRING
    "[SM100/SM120] Block-scaled tile N. Reduce on SM120 if SMEM overflows.")
set_property(CACHE COMPLEX_SM100_BLKSCALED_MMA_N PROPERTY STRINGS "32" "64" "128" "256")

# =============================================================================
# HERK Production vs Debug Mode (cross-architecture)
# =============================================================================

option(COMPLEX_FP8_HERK_FULL_MATRIX
    "HERK debug mode: 4 sub-GEMMs, full matrix. OFF=production: 3 sub-GEMMs, 25pct fewer FLOPs."
    OFF)

# =============================================================================
# Option Visibility in ccmake / cmake-gui
# =============================================================================
# Primary options (BUILD_ALL, ARCH, FP6, FP4, STAGES) are visible by default.
# Per-tile MMA/cluster options are marked advanced (FP8 tiles are all compiled
# and runtime-selectable via GemmConfig; these only affect block-scaled defaults).
# Use ccmake -LA or cmake-gui "Advanced" view to see all options.

# =============================================================================
# Collect Compile Definitions
# =============================================================================

if(ARCH_FAMILY STREQUAL "hopper")
    set(KERNEL_TUNING_DEFS
        COMPLEX_FP8_TARGET_SM90=1
        COMPLEX_FP8_TILE_M=${COMPLEX_FP8_TILE_M}
        COMPLEX_FP8_TILE_N=${COMPLEX_FP8_TILE_N}
        COMPLEX_FP8_TILE_K=${COMPLEX_FP8_TILE_K}
        COMPLEX_FP8_CLUSTER_M=${COMPLEX_FP8_CLUSTER_M}
        COMPLEX_FP8_CLUSTER_N=${COMPLEX_FP8_CLUSTER_N}
    )
    if(COMPLEX_FP8_USE_PINGPONG)
        list(APPEND KERNEL_TUNING_DEFS COMPLEX_FP8_USE_PINGPONG=1)
    endif()
    if(NOT COMPLEX_FP8_USE_FAST_ACCUM)
        list(APPEND KERNEL_TUNING_DEFS COMPLEX_FP8_USE_FAST_ACCUM=0)
    endif()
    if(COMPLEX_FP8_DISABLE_GROUPED_GEMM)
        list(APPEND KERNEL_TUNING_DEFS COMPLEX_FP8_DISABLE_GROUPED_GEMM=1)
        message(STATUS "Grouped GEMM: DISABLED (per-slab fallback)")
    endif()
else()
    # Blackwell SM100 / SM120
    set(KERNEL_TUNING_DEFS
        COMPLEX_FP8_SM100_MMA_M=${COMPLEX_FP8_SM100_MMA_M}
        COMPLEX_FP8_SM100_MMA_N=${COMPLEX_FP8_SM100_MMA_N}
        COMPLEX_FP8_SM100_MMA_K=${COMPLEX_FP8_SM100_MMA_K}
        COMPLEX_FP8_SM100_CLUSTER_M=${COMPLEX_FP8_SM100_CLUSTER_M}
        COMPLEX_FP8_SM100_CLUSTER_N=${COMPLEX_FP8_SM100_CLUSTER_N}
    )
    if(COMPLEX_FP8_SM100_USE_2SM)
        list(APPEND KERNEL_TUNING_DEFS COMPLEX_FP8_SM100_USE_2SM=1)
    endif()
    if(COMPLEX_FP8_ARCH STREQUAL "120a")
        list(APPEND KERNEL_TUNING_DEFS COMPLEX_FP8_SM100_TARGET_SM120=1)

        # SM120 FP8 tile override: 0=auto (128×64 for HERK), >0=explicit N dimension.
        # 128×64 tile fits 3-stage in 99 KB SMEM (HERK-optimized).
        # 128×128 tile needs auto carveout → 2-stage (GEMM-optimized, wider tile).
        if(NOT COMPLEX_SM100_FP8_TILE_N EQUAL 0)
            list(APPEND KERNEL_TUNING_DEFS COMPLEX_SM100_FP8_TILE_N=${COMPLEX_SM100_FP8_TILE_N})
            message(STATUS "SM120 FP8 tile N override: ${COMPLEX_SM100_FP8_TILE_N} (GEMM-optimized)")
        else()
            message(STATUS "SM120 FP8 tile: 128x64 (auto, HERK-optimized)")
        endif()

        # SM120 has only 99KB SMEM. Default 128×64 tile fits 3-stage.
        # With tile override to 128×128, auto carveout selects 2 stages.
        if(COMPLEX_FP8_SM100_STAGES EQUAL 0)
            set(COMPLEX_FP8_SM100_STAGES 3)
            message(STATUS "SM120: setting pipeline stages to 3 (FP8 chains use auto carveout)")
        endif()
    endif()
    if(NOT COMPLEX_FP8_SM100_STAGES EQUAL 0)
        list(APPEND KERNEL_TUNING_DEFS COMPLEX_FP8_SM100_STAGES=${COMPLEX_FP8_SM100_STAGES})
        message(STATUS "Pipeline stages: ${COMPLEX_FP8_SM100_STAGES} (fixed)")
    else()
        message(STATUS "Pipeline stages: auto (StageCountAutoCarveout)")
    endif()

    # SM120 validation: MMA_N=64 does not compile (TMA SfAtom constraint)
    if(COMPLEX_FP8_ARCH STREQUAL "120a")
        if(COMPLEX_FP8_SM100_MMA_N EQUAL 64)
            message(FATAL_ERROR
                "COMPLEX_FP8_SM100_MMA_N=64 is not supported on SM120: "
                "TMA SfAtom constraint requires M_tile >= 128. Use MMA_N=128 or 256.")
        endif()
        if(NOT (COMPLEX_FP8_SM100_CLUSTER_M EQUAL 1 AND COMPLEX_FP8_SM100_CLUSTER_N EQUAL 1))
            message(FATAL_ERROR
                "SM120 only supports cluster 1x1x1. "
                "Set COMPLEX_FP8_SM100_CLUSTER_M=1 and COMPLEX_FP8_SM100_CLUSTER_N=1.")
        endif()
    endif()

    # Multi-precision gates
    if(COMPLEX_SM100_ENABLE_FP6 OR COMPLEX_SM100_ENABLE_FP4)
        list(APPEND KERNEL_TUNING_DEFS
            COMPLEX_SM100_BLKSCALED_MMA_M=${COMPLEX_SM100_BLKSCALED_MMA_M}
            COMPLEX_SM100_BLKSCALED_MMA_N=${COMPLEX_SM100_BLKSCALED_MMA_N})
        message(STATUS "Block-scaled tile: ${COMPLEX_SM100_BLKSCALED_MMA_M}x${COMPLEX_SM100_BLKSCALED_MMA_N}")
    endif()

    if(COMPLEX_SM100_ENABLE_FP6)
        list(APPEND KERNEL_TUNING_DEFS
            COMPLEX_SM100_ENABLE_FP6=1
            COMPLEX_SM100_FP6_MMA_K=${COMPLEX_SM100_FP6_MMA_K})
        message(STATUS "FP6 (E3M2 + E2M3): ENABLED (MMA_K=${COMPLEX_SM100_FP6_MMA_K})")
    else()
        message(STATUS "FP6 (E3M2 + E2M3): disabled (enable with -DCOMPLEX_SM100_ENABLE_FP6=ON)")
    endif()

    if(COMPLEX_SM100_ENABLE_FP4)
        list(APPEND KERNEL_TUNING_DEFS
            COMPLEX_SM100_ENABLE_FP4=1
            COMPLEX_SM100_FP4_MMA_K=${COMPLEX_SM100_FP4_MMA_K})
        message(STATUS "FP4 (E2M1): ENABLED (MMA_K=${COMPLEX_SM100_FP4_MMA_K})")
    else()
        message(STATUS "FP4 (E2M1): disabled (enable with -DCOMPLEX_SM100_ENABLE_FP4=ON)")
    endif()

    # Small-M tile optimization — automatically enabled on SM100 when FP6 or FP4 is on.
    # SM120 does not support SmallM (SfAtom TMA requires M_tile >= 128).
    if((COMPLEX_SM100_ENABLE_FP6 OR COMPLEX_SM100_ENABLE_FP4) AND NOT COMPLEX_FP8_ARCH STREQUAL "120a")
        list(APPEND KERNEL_TUNING_DEFS COMPLEX_SM100_ENABLE_SMALL_M=1)
        message(STATUS "Small-M tiles: ENABLED (auto, 64×128 block-scaled for M≤64, SM100 only)")
    else()
        message(STATUS "Small-M tiles: disabled (requires FP6 or FP4 on SM100)")
    endif()
endif()

# HERK mode applies to both architectures
if(COMPLEX_FP8_HERK_FULL_MATRIX)
    list(APPEND KERNEL_TUNING_DEFS COMPLEX_FP8_HERK_FULL_MATRIX=1)
    message(STATUS "HERK mode: DEBUG (4 sub-GEMMs, full matrix, both triangles valid)")
else()
    message(STATUS "HERK mode: PRODUCTION (3 sub-GEMMs, triangle-only, 25% fewer FLOPs)")
endif()

# Expose codegen flags as compile definitions for build fingerprinting
if(COMPLEX_FP8_EXTRA_VECTORIZATION)
    list(APPEND KERNEL_TUNING_DEFS COMPLEX_FP8_EXTRA_VECTORIZATION_ENABLED=1)
endif()
if(COMPLEX_FP8_DEVICE_LTO)
    list(APPEND KERNEL_TUNING_DEFS COMPLEX_FP8_DEVICE_LTO_ENABLED=1)
endif()
if(COMPLEX_FP8_MAX_REGISTERS GREATER 0)
    list(APPEND KERNEL_TUNING_DEFS COMPLEX_FP8_MAX_REGISTERS_VALUE=${COMPLEX_FP8_MAX_REGISTERS})
endif()

# CUTLASS version (available in all binaries via system_info.hpp)
if(CUTLASS_VER_MAJOR GREATER 0)
    list(APPEND KERNEL_TUNING_DEFS
        CUTLASS_LINKED_MAJOR=${CUTLASS_VER_MAJOR}
        CUTLASS_LINKED_MINOR=${CUTLASS_VER_MINOR}
        CUTLASS_LINKED_PATCH=${CUTLASS_VER_PATCH})
endif()

# =============================================================================
# Status Messages
# =============================================================================

message(STATUS "CUTLASS: ${CUTLASS_DIR}")
message(STATUS "CUDA: ${CMAKE_CUDA_COMPILER_VERSION}")
message(STATUS "Arch: ${COMPLEX_FP8_ARCH} (${ARCH_FAMILY})")
if(COMPLEX_FP8_BUILD_ALL)
    message(STATUS "BUILD_ALL: ON (all precisions and features enabled)")
else()
    message(STATUS "BUILD_ALL: OFF (manual precision control)")
endif()

if(ARCH_FAMILY STREQUAL "hopper")
    message(STATUS "Tile: ${COMPLEX_FP8_TILE_M}x${COMPLEX_FP8_TILE_N}x${COMPLEX_FP8_TILE_K}")
    message(STATUS "Cluster: ${COMPLEX_FP8_CLUSTER_M}x${COMPLEX_FP8_CLUSTER_N}x1")
    if(COMPLEX_FP8_USE_PINGPONG)
        message(STATUS "Schedule: PingpongFP8FastAccum")
    elseif(NOT COMPLEX_FP8_USE_FAST_ACCUM)
        message(STATUS "Schedule: Cooperative (non-FastAccum)")
    else()
        message(STATUS "Schedule: CooperativeFP8FastAccum")
    endif()
else()
    if(COMPLEX_FP8_ARCH STREQUAL "120a" AND NOT COMPLEX_SM100_FP8_TILE_N EQUAL 0)
        message(STATUS "MmaTile (FP8): 128x${COMPLEX_SM100_FP8_TILE_N}x${COMPLEX_FP8_SM100_MMA_K} (SM120: explicit override, GEMM-optimized)")
    elseif(COMPLEX_FP8_ARCH STREQUAL "120a")
        message(STATUS "MmaTile (FP8): 128x64x${COMPLEX_FP8_SM100_MMA_K} (SM120: HERK-optimized, 3-stage)")
    else()
        message(STATUS "MmaTile: ${COMPLEX_FP8_SM100_MMA_M}x${COMPLEX_FP8_SM100_MMA_N}x${COMPLEX_FP8_SM100_MMA_K}")
    endif()
    message(STATUS "Cluster: ${COMPLEX_FP8_SM100_CLUSTER_M}x${COMPLEX_FP8_SM100_CLUSTER_N}x1")
    if(COMPLEX_FP8_SM100_USE_2SM)
        message(STATUS "Schedule: 2SM warp-specialized (tcgen05)")
    else()
        message(STATUS "Schedule: 1SM warp-specialized (tcgen05)")
    endif()
    if(COMPLEX_FP8_ARCH STREQUAL "120a")
        message(STATUS "Target: SM120 (GeForce/GB10 — no multicast, cluster 1x1x1 only)")
    else()
        message(STATUS "Target: SM100 (datacenter/GB200)")
    endif()
endif()

# FP8 tile config summary (all always compiled, runtime-selectable)
message(STATUS "")
message(STATUS "FP8 tile configs (all compiled, runtime-selectable via GemmConfig):")
if(ARCH_FAMILY STREQUAL "hopper")
    message(STATUS "  [10] FP8_T128x256_C1x1_Coop   (default, GEMM-optimal)")
    message(STATUS "  [11] FP8_T128x128_C1x1_Coop   (smaller tile)")
    message(STATUS "  [12] FP8_T128x256_C1x1_PP     (Pingpong schedule)")
    message(STATUS "  [13] FP8_T128x128_C1x1_PP     (Pingpong, smaller)")
    message(STATUS "  [14] FP8_T128x256_C1x2_Coop   (1x2 cluster)")
    message(STATUS "  [15] FP8_T64x128_C1x1_Coop    (small-M tile)")
elseif(COMPLEX_FP8_ARCH STREQUAL "120a")
    message(STATUS "  [0]  FP8_T128x64_C1x1_S3      (default, HERK-optimal)")
    message(STATUS "  [1]  FP8_T128x64_C1x1_SAuto   (auto stages)")
    message(STATUS "  [2]  FP8_T128x128_C1x1_SAuto  (GEMM-optimal)")
else()
    message(STATUS "  [0]  FP8_T128x64_C1x1_S3")
    message(STATUS "  [1]  FP8_T128x64_C1x1_SAuto")
    message(STATUS "  [2]  FP8_T128x128_C1x1_SAuto")
    message(STATUS "  [3]  FP8_T128x128_C1x1_S3     (default)")
    message(STATUS "  [4]  FP8_T128x256_C1x1_SAuto  (wide-N)")
    message(STATUS "  [5]  FP8_T128x128_C1x2        (1x2 cluster)")
    message(STATUS "  [6]  FP8_T128x128_C2x2        (2x2 cluster)")
    message(STATUS "  [7]  FP8_T128x256_C1x2        (wide-N + cluster)")
endif()

if(COMPLEX_SM100_ENABLE_FP6 OR COMPLEX_SM100_ENABLE_FP4)
    message(STATUS "Block-scaled tile: ${COMPLEX_SM100_BLKSCALED_MMA_M}x${COMPLEX_SM100_BLKSCALED_MMA_N}")
endif()

set(_PREC_LIST "FP8")
if(COMPLEX_SM100_ENABLE_FP6)
    set(_PREC_LIST "${_PREC_LIST} + FP6 (E3M2, E2M3)")
endif()
if(COMPLEX_SM100_ENABLE_FP4)
    set(_PREC_LIST "${_PREC_LIST} + FP4 (E2M1)")
endif()
message(STATUS "Precisions: ${_PREC_LIST}")
message(STATUS "")

# =============================================================================
# Compile Flags
# =============================================================================

set(CUDA_COMMON_OPTIONS
    --expt-relaxed-constexpr
    --expt-extended-lambda
    -Xcompiler=-Wno-psabi
    -O3
)

# --- Cross-architecture options ---

set(COMPLEX_FP8_MAX_REGISTERS 0 CACHE STRING
    "Max registers per thread (--maxrregcount). 0 = compiler default. SM90/100 max=255.")
set_property(CACHE COMPLEX_FP8_MAX_REGISTERS PROPERTY STRINGS "0" "128" "192" "224" "255")

option(COMPLEX_FP8_VERBOSE_PTXAS
    "Show detailed ptxas output: register usage, spills, SMEM."
    OFF)

option(COMPLEX_FP8_EXTRA_VECTORIZATION
    "Enable --extra-device-vectorization for wider memory coalescing."
    ON)

option(COMPLEX_FP8_DEVICE_LTO
    "Enable device link-time optimization (-dlto). Adds link time."
    OFF)

if(COMPLEX_FP8_EXTRA_VECTORIZATION)
    list(APPEND CUDA_COMMON_OPTIONS --extra-device-vectorization)
    message(STATUS "Extra device vectorization: ON")
endif()

# Device LTO
# NOTE: nvcc/nvlink bug — '-dlto' causes the device linker to strip the 'a'
# suffix from arch-accelerated targets (sm_90a → sm_90), making ptxas reject
# wgmma/setmaxnreg instructions. Confirmed on CUDA 13.0. Also does not compile
# on aarch64 (gcc debug output level error). Only enable on Blackwell x86_64.
if(COMPLEX_FP8_DEVICE_LTO)
    if(ARCH_FAMILY STREQUAL "hopper")
        message(WARNING
            "Device LTO is not supported on SM90a: nvcc/nvlink bug strips the 'a' "
            "suffix during device linking, causing ptxas to reject wgmma instructions. "
            "Disabling COMPLEX_FP8_DEVICE_LTO.")
        set(COMPLEX_FP8_DEVICE_LTO OFF)
    else()
        string(REGEX REPLACE "a$" "" ARCH_NUM_LTO "${COMPLEX_FP8_ARCH}")
        string(FIND "${COMPLEX_FP8_ARCH}" "a" HAS_A_SUFFIX)
        if(HAS_A_SUFFIX GREATER_EQUAL 0)
            set(LTO_ARCH "${ARCH_NUM_LTO}a")
        else()
            set(LTO_ARCH "${ARCH_NUM_LTO}")
        endif()
        set(LTO_COMPILE_FLAGS
            -gencode=arch=compute_${ARCH_NUM_LTO},code=lto_${LTO_ARCH})
        set(LTO_DLINK_ARCH sm_${LTO_ARCH})
        message(STATUS "Device LTO: ON (compile: lto_${LTO_ARCH}, link: sm_${LTO_ARCH})")
    endif()
endif()

# Max registers
# Note: SM90 (Hopper) ptxas rejects --maxrregcount=255 for large CUTLASS kernels
# ("Too big maxrregcount value specified, will be ignored"). The benchmark study
# shows <5% impact from this flag, so we skip it on Hopper to avoid warnings.
if(COMPLEX_FP8_MAX_REGISTERS GREATER 0)
    if(ARCH_FAMILY STREQUAL "hopper")
        message(STATUS "Max registers: SKIPPED on SM90 (ptxas rejects for large CUTLASS kernels)")
    else()
        if(COMPLEX_FP8_MAX_REGISTERS GREATER 255)
            message(WARNING "SM max is 255 registers per thread. Clamping to 255.")
            set(COMPLEX_FP8_MAX_REGISTERS 255)
        endif()
        list(APPEND CUDA_COMMON_OPTIONS --maxrregcount=${COMPLEX_FP8_MAX_REGISTERS})
        message(STATUS "Max registers: ${COMPLEX_FP8_MAX_REGISTERS}")
    endif()
endif()

# Verbose ptxas
if(COMPLEX_FP8_VERBOSE_PTXAS)
    list(APPEND CUDA_COMMON_OPTIONS -Xptxas=-v)
    message(STATUS "Verbose ptxas: ON")
endif()

# =============================================================================
# Helper Function
# =============================================================================

function(configure_cuda_target TARGET_NAME)
    target_include_directories(${TARGET_NAME} PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CUTLASS_INCLUDE_DIRS}
    )

    target_compile_options(${TARGET_NAME} PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:${CUDA_COMMON_OPTIONS}>
    )

    target_link_libraries(${TARGET_NAME} PRIVATE cuda)

    # OpenMP: pass through -Xcompiler for nvcc, link gomp
    if(OpenMP_CXX_FOUND)
        target_compile_options(${TARGET_NAME} PRIVATE
            $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=${OpenMP_CXX_FLAGS}>)
        target_link_libraries(${TARGET_NAME} PRIVATE OpenMP::OpenMP_CXX)
    endif()

    # SM120: manual arch handling (CUDA_ARCHITECTURES=OFF, detected minor version)
    if(FORCED_CUDA_ARCH)
        set_target_properties(${TARGET_NAME} PROPERTIES CUDA_ARCHITECTURES OFF)
        target_compile_options(${TARGET_NAME} PRIVATE
            $<$<COMPILE_LANGUAGE:CUDA>:-arch=${FORCED_CUDA_ARCH}>)
    endif()

    if(COMPLEX_FP8_DEVICE_LTO)
        # Separable compilation required for device LTO.
        # Keep CUDA_ARCHITECTURES from the global setting (e.g. 90a) so CMake
        # generates correct -gencode for the device link step.
        set_target_properties(${TARGET_NAME} PROPERTIES
            CUDA_SEPARABLE_COMPILATION ON)
        # Compile: generate LTO intermediate code (in addition to CMake's gencode)
        target_compile_options(${TARGET_NAME} PRIVATE
            $<$<COMPILE_LANGUAGE:CUDA>:${LTO_COMPILE_FLAGS}>)
        # Device link: -dlto triggers link-time optimization
        target_link_options(${TARGET_NAME} PRIVATE
            $<DEVICE_LINK:-dlto>)
    endif()
endfunction()

# =============================================================================
# Targets — architecture-conditional
# =============================================================================

if(ARCH_FAMILY STREQUAL "hopper")
    # ---- Hopper SM90 targets ----
    add_executable(example_complex_fp8 example_usage.cu)
    configure_cuda_target(example_complex_fp8)
    target_compile_definitions(example_complex_fp8 PRIVATE ${KERNEL_TUNING_DEFS})

    add_executable(benchmark_configs benchmark_configs.cu)
    configure_cuda_target(benchmark_configs)
    target_compile_definitions(benchmark_configs PRIVATE ${KERNEL_TUNING_DEFS})

    # ---- CUTLASS GEMM API library (SM90 — for external consumers like dedisp) ----
    add_library(cutlass_gemm_api STATIC cutlass_gemm_api.cu)
    set_target_properties(cutlass_gemm_api PROPERTIES POSITION_INDEPENDENT_CODE ON)
    configure_cuda_target(cutlass_gemm_api)
    target_compile_definitions(cutlass_gemm_api PRIVATE ${KERNEL_TUNING_DEFS})
    target_include_directories(cutlass_gemm_api PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
    # OpenMP must be PUBLIC so consumers linking this static library get -lgomp.
    # configure_cuda_target() already links PRIVATE OpenMP for compilation;
    # this adds the transitive link requirement for external consumers.
    if(OpenMP_CXX_FOUND)
        target_link_libraries(cutlass_gemm_api PUBLIC OpenMP::OpenMP_CXX)
    endif()

    # HERK INT4 API test — links only against cutlass_gemm_api (PIMPL)
    add_executable(test_herk_int4 test_herk_int4.cu)
    configure_cuda_target(test_herk_int4)
    target_link_libraries(test_herk_int4 PRIVATE cutlass_gemm_api)

    # GEMM API correctness test — links only against cutlass_gemm_api (PIMPL)
    add_executable(test_gemm_api test_gemm_api.cu)
    configure_cuda_target(test_gemm_api)
    target_link_libraries(test_gemm_api PRIVATE cutlass_gemm_api)

    # Strategy autotuning API test — links only against cutlass_gemm_api (PIMPL)
    add_executable(test_tune_api test_tune_api.cu)
    configure_cuda_target(test_tune_api)
    target_link_libraries(test_tune_api PRIVATE cutlass_gemm_api)

    # Multi-precision HERK benchmark — links only against cutlass_gemm_api (PIMPL)
    add_executable(bench_herk_precision bench_herk_precision.cu)
    configure_cuda_target(bench_herk_precision)
    target_link_libraries(bench_herk_precision PRIVATE cutlass_gemm_api)

    # Precision accuracy test vs FP64 CPU reference — links only against cutlass_gemm_api (PIMPL)
    add_executable(test_precision_accuracy test_precision_accuracy.cu)
    configure_cuda_target(test_precision_accuracy)
    target_link_libraries(test_precision_accuracy PRIVATE cutlass_gemm_api)

    # Production GEMM benchmark (beamformer + dedisp) — links only against cutlass_gemm_api (PIMPL)
    add_executable(bench_gemm_production bench_gemm_production.cu)
    configure_cuda_target(bench_gemm_production)
    target_link_libraries(bench_gemm_production PRIVATE cutlass_gemm_api)

else()
    # ---- Blackwell SM100/SM120 targets ----

    # OBJECT library for block-scaled dispatch (compile once, link many)
    if(COMPLEX_SM100_ENABLE_FP6 OR COMPLEX_SM100_ENABLE_FP4)
        add_library(blockscaled_dispatch_obj OBJECT gemm_blockscaled_dispatch.cu)
        set_target_properties(blockscaled_dispatch_obj PROPERTIES POSITION_INDEPENDENT_CODE ON)
        configure_cuda_target(blockscaled_dispatch_obj)
        target_compile_definitions(blockscaled_dispatch_obj PRIVATE ${KERNEL_TUNING_DEFS})
    endif()

    add_executable(example_complex_sm100 example_usage_sm100.cu)
    configure_cuda_target(example_complex_sm100)
    target_compile_definitions(example_complex_sm100 PRIVATE ${KERNEL_TUNING_DEFS})
    if(COMPLEX_SM100_ENABLE_FP6 OR COMPLEX_SM100_ENABLE_FP4)
        target_link_libraries(example_complex_sm100 PRIVATE blockscaled_dispatch_obj)
    endif()

    add_executable(sweep_triangle_sm100 sweep_triangle_sm100.cu)
    configure_cuda_target(sweep_triangle_sm100)
    target_compile_definitions(sweep_triangle_sm100 PRIVATE ${KERNEL_TUNING_DEFS})
    if(COMPLEX_SM100_ENABLE_FP6 OR COMPLEX_SM100_ENABLE_FP4)
        target_link_libraries(sweep_triangle_sm100 PRIVATE blockscaled_dispatch_obj)
    endif()

    add_executable(benchmark_configs_sm100 benchmark_configs_sm100.cu)
    configure_cuda_target(benchmark_configs_sm100)
    target_compile_definitions(benchmark_configs_sm100 PRIVATE ${KERNEL_TUNING_DEFS})
    if(COMPLEX_SM100_ENABLE_FP6 OR COMPLEX_SM100_ENABLE_FP4)
        target_link_libraries(benchmark_configs_sm100 PRIVATE blockscaled_dispatch_obj)
    endif()

    # ---- CUTLASS GEMM API library (for external consumers like dedisp) ----
    add_library(cutlass_gemm_api STATIC cutlass_gemm_api.cu)
    set_target_properties(cutlass_gemm_api PROPERTIES POSITION_INDEPENDENT_CODE ON)
    configure_cuda_target(cutlass_gemm_api)
    target_compile_definitions(cutlass_gemm_api PRIVATE ${KERNEL_TUNING_DEFS})
    target_include_directories(cutlass_gemm_api PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
    # OpenMP must be PUBLIC so consumers linking this static library get -lgomp.
    # configure_cuda_target() already links PRIVATE OpenMP for compilation;
    # this adds the transitive link requirement for external consumers.
    if(OpenMP_CXX_FOUND)
        target_link_libraries(cutlass_gemm_api PUBLIC OpenMP::OpenMP_CXX)
    endif()
    if(COMPLEX_SM100_ENABLE_FP6 OR COMPLEX_SM100_ENABLE_FP4)
        target_link_libraries(cutlass_gemm_api PRIVATE blockscaled_dispatch_obj)
    endif()

    # ---- FP6 diagnostic test ----
    if(COMPLEX_SM100_ENABLE_FP6)
        add_executable(test_fp6_batched test_fp6_batched.cu)
        configure_cuda_target(test_fp6_batched)
        target_compile_definitions(test_fp6_batched PRIVATE ${KERNEL_TUNING_DEFS})
        target_link_libraries(test_fp6_batched PRIVATE cutlass_gemm_api)
    endif()

    # HERK INT4 API test — links only against cutlass_gemm_api (PIMPL)
    add_executable(test_herk_int4 test_herk_int4.cu)
    configure_cuda_target(test_herk_int4)
    target_link_libraries(test_herk_int4 PRIVATE cutlass_gemm_api)

    # GEMM API correctness test — links only against cutlass_gemm_api (PIMPL)
    # Needs precision gate defines to enable FP6/FP4 test paths at compile time.
    add_executable(test_gemm_api test_gemm_api.cu)
    configure_cuda_target(test_gemm_api)
    target_compile_definitions(test_gemm_api PRIVATE ${KERNEL_TUNING_DEFS})
    target_link_libraries(test_gemm_api PRIVATE cutlass_gemm_api)

    # Strategy autotuning API test — links only against cutlass_gemm_api (PIMPL)
    add_executable(test_tune_api test_tune_api.cu)
    configure_cuda_target(test_tune_api)
    target_link_libraries(test_tune_api PRIVATE cutlass_gemm_api)

    # Multi-precision HERK benchmark — links only against cutlass_gemm_api (PIMPL)
    add_executable(bench_herk_precision bench_herk_precision.cu)
    configure_cuda_target(bench_herk_precision)
    target_compile_definitions(bench_herk_precision PRIVATE ${KERNEL_TUNING_DEFS})
    target_link_libraries(bench_herk_precision PRIVATE cutlass_gemm_api)

    # Precision accuracy test vs FP64 CPU reference — links only against cutlass_gemm_api (PIMPL)
    add_executable(test_precision_accuracy test_precision_accuracy.cu)
    configure_cuda_target(test_precision_accuracy)
    target_compile_definitions(test_precision_accuracy PRIVATE ${KERNEL_TUNING_DEFS})
    target_link_libraries(test_precision_accuracy PRIVATE cutlass_gemm_api)

    # Production GEMM benchmark (beamformer + dedisp) — links only against cutlass_gemm_api (PIMPL)
    add_executable(bench_gemm_production bench_gemm_production.cu)
    configure_cuda_target(bench_gemm_production)
    target_compile_definitions(bench_gemm_production PRIVATE ${KERNEL_TUNING_DEFS})
    target_link_libraries(bench_gemm_production PRIVATE cutlass_gemm_api)

endif()

# =============================================================================
# CTest integration
# =============================================================================
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    enable_testing()
endif()

# HERK INT4 correctness test (small: N=64 K=128 batch=2)
add_test(NAME herk_int4_small
    COMMAND $<TARGET_FILE:test_herk_int4> 64 128 2)
set_tests_properties(herk_int4_small PROPERTIES
    TIMEOUT 120
    LABELS "cutlass;herk;int4;gpu")

# HERK INT4 larger problem (N=128 K=256 batch=4)
add_test(NAME herk_int4_medium
    COMMAND $<TARGET_FILE:test_herk_int4> 128 256 4)
set_tests_properties(herk_int4_medium PROPERTIES
    TIMEOUT 120
    LABELS "cutlass;herk;int4;gpu")

# GEMM planar batched FP16 output
add_test(NAME gemm_planar_fp16
    COMMAND $<TARGET_FILE:test_gemm_api> gemm_fp16)
set_tests_properties(gemm_planar_fp16 PROPERTIES
    TIMEOUT 120
    LABELS "cutlass;gemm;fp16;gpu")

# GEMM planar batched FP32 output
add_test(NAME gemm_planar_fp32
    COMMAND $<TARGET_FILE:test_gemm_api> gemm_fp32)
set_tests_properties(gemm_planar_fp32 PROPERTIES
    TIMEOUT 120
    LABELS "cutlass;gemm;fp32;gpu")

# Prepare/Execute pattern (pre-convert B, reuse)
add_test(NAME gemm_prepare_execute
    COMMAND $<TARGET_FILE:test_gemm_api> prepare_execute)
set_tests_properties(gemm_prepare_execute PROPERTIES
    TIMEOUT 120
    LABELS "cutlass;gemm;prepare;gpu")

# HERK batched FP32 output (FP16 input)
add_test(NAME herk_fp32
    COMMAND $<TARGET_FILE:test_gemm_api> herk_fp32)
set_tests_properties(herk_fp32 PROPERTIES
    TIMEOUT 120
    LABELS "cutlass;herk;fp32;gpu")

# HERK beta accumulation test
add_test(NAME herk_beta
    COMMAND $<TARGET_FILE:test_gemm_api> herk_beta)
set_tests_properties(herk_beta PROPERTIES
    TIMEOUT 120
    LABELS "cutlass;herk;beta;gpu")

# HERK single batch edge case
add_test(NAME herk_single_batch
    COMMAND $<TARGET_FILE:test_gemm_api> herk_single_batch)
set_tests_properties(herk_single_batch PROPERTIES
    TIMEOUT 120
    LABELS "cutlass;herk;gpu")

# GEMM non-aligned dimensions
add_test(NAME gemm_non_aligned
    COMMAND $<TARGET_FILE:test_gemm_api> gemm_non_aligned)
set_tests_properties(gemm_non_aligned PROPERTIES
    TIMEOUT 120
    LABELS "cutlass;gemm;gpu")

# GEMM FP32 output (different seeds)
add_test(NAME gemm_fp32_output
    COMMAND $<TARGET_FILE:test_gemm_api> gemm_fp32_output)
set_tests_properties(gemm_fp32_output PROPERTIES
    TIMEOUT 120
    LABELS "cutlass;gemm;fp32;gpu")

# HERK large K (>K_CHUNK)
add_test(NAME herk_large_k
    COMMAND $<TARGET_FILE:test_gemm_api> herk_large_k)
set_tests_properties(herk_large_k PROPERTIES
    TIMEOUT 120
    LABELS "cutlass;herk;gpu")

# GEMM alpha/beta accumulation test
add_test(NAME gemm_alpha_beta
    COMMAND $<TARGET_FILE:test_gemm_api> gemm_alpha_beta)
set_tests_properties(gemm_alpha_beta PROPERTIES
    TIMEOUT 120
    LABELS "cutlass;gemm;gpu")

# HERK direct PTX kernel correctness
add_test(NAME herk_direct
    COMMAND $<TARGET_FILE:test_gemm_api> herk_direct)
set_tests_properties(herk_direct PROPERTIES
    TIMEOUT 120
    LABELS "cutlass;herk;direct;gpu")

# HERK baseline CUTLASS path correctness
add_test(NAME herk_baseline
    COMMAND $<TARGET_FILE:test_gemm_api> herk_baseline)
set_tests_properties(herk_baseline PROPERTIES
    TIMEOUT 120
    LABELS "cutlass;herk;baseline;gpu")

# HERK direct vs baseline self-consistency
add_test(NAME herk_direct_vs_baseline
    COMMAND $<TARGET_FILE:test_gemm_api> herk_direct_vs_baseline)
set_tests_properties(herk_direct_vs_baseline PROPERTIES
    TIMEOUT 120
    LABELS "cutlass;herk;gpu")

# HERK K not aligned to K_CHUNK
add_test(NAME herk_k_not_aligned
    COMMAND $<TARGET_FILE:test_gemm_api> herk_k_not_aligned)
set_tests_properties(herk_k_not_aligned PROPERTIES
    TIMEOUT 120
    LABELS "cutlass;herk;gpu")

# GEMM K not aligned to MMA_K
add_test(NAME gemm_k_not_aligned
    COMMAND $<TARGET_FILE:test_gemm_api> gemm_k_not_aligned)
set_tests_properties(gemm_k_not_aligned PROPERTIES
    TIMEOUT 120
    LABELS "cutlass;gemm;gpu")

# GEMM with continuous (non-integer) data
add_test(NAME gemm_continuous
    COMMAND $<TARGET_FILE:test_gemm_api> gemm_continuous)
set_tests_properties(gemm_continuous PROPERTIES
    TIMEOUT 120
    LABELS "cutlass;gemm;accuracy;gpu")

# HERK with continuous (non-integer) data
add_test(NAME herk_continuous
    COMMAND $<TARGET_FILE:test_gemm_api> herk_continuous)
set_tests_properties(herk_continuous PROPERTIES
    TIMEOUT 120
    LABELS "cutlass;herk;accuracy;gpu")

# FP6 complex GEMM (if compiled)
if(COMPLEX_SM100_ENABLE_FP6)
    add_test(NAME gemm_fp6
        COMMAND $<TARGET_FILE:test_gemm_api> gemm_fp6)
    set_tests_properties(gemm_fp6 PROPERTIES
        TIMEOUT 120
        LABELS "cutlass;gemm;fp6;gpu")

    add_test(NAME prepare_fp6
        COMMAND $<TARGET_FILE:test_gemm_api> prepare_fp6)
    set_tests_properties(prepare_fp6 PROPERTIES
        TIMEOUT 120
        LABELS "cutlass;gemm;fp6;prepare;gpu")
endif()

# FP4 complex GEMM (if compiled)
if(COMPLEX_SM100_ENABLE_FP4)
    add_test(NAME gemm_fp4
        COMMAND $<TARGET_FILE:test_gemm_api> gemm_fp4)
    set_tests_properties(gemm_fp4 PROPERTIES
        TIMEOUT 120
        LABELS "cutlass;gemm;fp4;gpu")

    add_test(NAME prepare_fp4
        COMMAND $<TARGET_FILE:test_gemm_api> prepare_fp4)
    set_tests_properties(prepare_fp4 PROPERTIES
        TIMEOUT 120
        LABELS "cutlass;gemm;fp4;prepare;gpu")
endif()

# FP6 HERK correctness (if compiled)
if(COMPLEX_SM100_ENABLE_FP6)
    add_test(NAME herk_fp6
        COMMAND $<TARGET_FILE:test_gemm_api> herk_fp6)
    set_tests_properties(herk_fp6 PROPERTIES
        TIMEOUT 120
        LABELS "cutlass;herk;fp6;gpu")
endif()

# FP4 HERK correctness (if compiled)
if(COMPLEX_SM100_ENABLE_FP4)
    add_test(NAME herk_fp4
        COMMAND $<TARGET_FILE:test_gemm_api> herk_fp4)
    set_tests_properties(herk_fp4 PROPERTIES
        TIMEOUT 120
        LABELS "cutlass;herk;fp4;gpu")

    add_test(NAME herk_fp4_lossy
        COMMAND $<TARGET_FILE:test_gemm_api> herk_fp4_lossy)
    set_tests_properties(herk_fp4_lossy PROPERTIES
        TIMEOUT 120
        LABELS "cutlass;herk;fp4;accuracy;gpu")
endif()

# Strategy autotuning API test
add_test(NAME tune_api
    COMMAND $<TARGET_FILE:test_tune_api>)
set_tests_properties(tune_api PROPERTIES
    TIMEOUT 300
    LABELS "cutlass;tune;herk;gpu")

# FP6 diagnostic test (if compiled)
if(COMPLEX_SM100_ENABLE_FP6)
    add_test(NAME fp6_batched_diagnostic
        COMMAND $<TARGET_FILE:test_fp6_batched>)
    set_tests_properties(fp6_batched_diagnostic PROPERTIES
        TIMEOUT 120
        LABELS "cutlass;fp6;gpu")
endif()

# Precision accuracy test (small: N=128, batch=2, FP16 output)
add_test(NAME precision_accuracy_fp16
    COMMAND $<TARGET_FILE:test_precision_accuracy> 128 2)
set_tests_properties(precision_accuracy_fp16 PROPERTIES
    TIMEOUT 300
    LABELS "cutlass;accuracy;herk;gpu")

# Precision accuracy test (small: N=128, batch=2, FP32 output)
add_test(NAME precision_accuracy_fp32
    COMMAND $<TARGET_FILE:test_precision_accuracy> 128 2 fp32out)
set_tests_properties(precision_accuracy_fp32 PROPERTIES
    TIMEOUT 300
    LABELS "cutlass;accuracy;herk;fp32;gpu")

