# Set the minimum required version of CMake
cmake_minimum_required(VERSION 3.24)

# Standalone-only: project declaration and language standards
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    project(DedispersionSim CXX CUDA)
    set(CMAKE_CXX_STANDARD 17)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)
    set(CMAKE_CUDA_STANDARD 17)
endif()

# Find the OpenMP package
find_package(OpenMP REQUIRED)

# Find the CUDA toolkit
find_package(CUDAToolkit REQUIRED)

set(GPU_ARCH
    ${DEFAULT_GPU_ARCH}
    CACHE STRING "set the GPU architecture (90a 100a 121)")
set_property(CACHE GPU_ARCH PROPERTY STRINGS 90a 100a 121)

# --- Option B: Build CUTLASS GEMM from source as a subdirectory ---
option(BUILD_CUTLASS_GEMM "Build CUTLASS GEMM API from cutlass_interface source tree" OFF)

if(BUILD_CUTLASS_GEMM)
    # CUTLASS source tree (required)
    set(CUTLASS_DIR "" CACHE PATH "Path to CUTLASS source tree (required for BUILD_CUTLASS_GEMM)")
    if(CUTLASS_DIR STREQUAL "")
        message(FATAL_ERROR "BUILD_CUTLASS_GEMM requires CUTLASS_DIR, e.g. -DCUTLASS_DIR=/path/to/cutlass")
    endif()

    # Map dedisp GPU_ARCH → cutlass_interface COMPLEX_FP8_ARCH
    if(GPU_ARCH STREQUAL "90a")
        set(_CI_ARCH "90a")
    elseif(GPU_ARCH STREQUAL "100a")
        set(_CI_ARCH "100a")
    elseif(GPU_ARCH MATCHES "^12")
        set(_CI_ARCH "120a")   # 121 → 120a
    else()
        message(FATAL_ERROR "Unknown GPU_ARCH '${GPU_ARCH}' for CUTLASS GEMM build")
    endif()

    # Forward configuration to cutlass_interface
    set(COMPLEX_FP8_ARCH "${_CI_ARCH}" CACHE STRING "" FORCE)

    # Build only the cutlass_gemm_api target (EXCLUDE_FROM_ALL skips examples/benchmarks/tests)
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../cutlass_interface
                     ${CMAKE_CURRENT_BINARY_DIR}/cutlass_interface
                     EXCLUDE_FROM_ALL)

    message(STATUS "CUTLASS GEMM API: BUILD FROM SOURCE (${_CI_ARCH})")
    message(STATUS "  cutlass_interface: ${CMAKE_CURRENT_SOURCE_DIR}/../cutlass_interface")
    message(STATUS "  CUTLASS: ${CUTLASS_DIR}")
endif()

# Unified flag for "CUTLASS GEMM is available" (either pre-built or from source)
set(HAS_CUTLASS_GEMM_AVAILABLE OFF)
if(BUILD_CUTLASS_GEMM)
    set(HAS_CUTLASS_GEMM_AVAILABLE ON)
endif()

# --- Dedisp PIMPL API library (always built) ---
add_library(dedisp_api STATIC dedisp_api.cu)
set_target_properties(dedisp_api PROPERTIES
    CUDA_ARCHITECTURES "${GPU_ARCH}"
    POSITION_INDEPENDENT_CODE ON)
target_include_directories(dedisp_api PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(dedisp_api PUBLIC
    CUDA::cufft CUDA::cublas CUDA::curand CUDA::cublasLt)

# OpenMP for dedisp_api:
#  - USE_CUTLASS_GEMM path links a raw .a file, so OpenMP must be added here
#    (no CMake target to propagate from).
#  - BUILD_CUTLASS_GEMM path gets OpenMP transitively from the cutlass_gemm_api
#    CMake target (which has PUBLIC OpenMP::OpenMP_CXX), but the duplicate is harmless.
if(OpenMP_CXX_FOUND)
    target_compile_options(dedisp_api PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=${OpenMP_CXX_FLAGS}>)
    target_link_libraries(dedisp_api PUBLIC OpenMP::OpenMP_CXX)
endif()

if(BUILD_CUTLASS_GEMM)
    target_link_libraries(dedisp_api PUBLIC cutlass_gemm_api)
    target_compile_definitions(dedisp_api PRIVATE HAS_CUTLASS_GEMM=1)
endif()

# ======================================================================
# Standalone-only: executables, USE_CUTLASS_GEMM, tests
# ======================================================================
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)

# --- FDMT prototype (CPU) ---
add_executable(fdmt_perfect fdmt_perfect.cpp)
target_compile_options(fdmt_perfect PRIVATE -Wno-psabi)

# --- Main FDD executable ---
add_executable(gpu_fdd_improved gpu_fdd_improved.cu)
set_target_properties(gpu_fdd_improved PROPERTIES CUDA_ARCHITECTURES "${GPU_ARCH}")
target_link_libraries(gpu_fdd_improved PUBLIC
    ${CUDA_cufft_LIBRARY} ${CUDA_cublas_LIBRARY} ${CUDA_curand_LIBRARY} ${CUDA_cublasLt_LIBRARY})

# OpenMP: pass through -Xcompiler for nvcc, link gomp (matches cutlass_interface pattern)
if(OpenMP_CXX_FOUND)
    target_compile_options(gpu_fdd_improved PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=${OpenMP_CXX_FLAGS}>)
    target_link_libraries(gpu_fdd_improved PRIVATE OpenMP::OpenMP_CXX)
endif()

# --- CUTLASS GEMM API integration (pre-built, standalone only) ---
option(USE_CUTLASS_GEMM "Link gpu_fdd_improved against pre-built CUTLASS GEMM API" OFF)

if(USE_CUTLASS_GEMM AND BUILD_CUTLASS_GEMM)
    message(FATAL_ERROR "Set only one of USE_CUTLASS_GEMM (pre-built) or BUILD_CUTLASS_GEMM (from source), not both.")
endif()

if(USE_CUTLASS_GEMM)
    set(CUTLASS_GEMM_BUILD_DIR "${CMAKE_CURRENT_SOURCE_DIR}/.." CACHE PATH
        "Path to build dir containing libcutlass_gemm_api.a")
    set(CUTLASS_GEMM_INCLUDE_DIR "${CUTLASS_GEMM_BUILD_DIR}" CACHE PATH
        "Path to dir containing cutlass_gemm_api.h (defaults to CUTLASS_GEMM_BUILD_DIR)")

    target_include_directories(gpu_fdd_improved PRIVATE
        "${CUTLASS_GEMM_INCLUDE_DIR}")
    target_link_libraries(gpu_fdd_improved PUBLIC
        "${CUTLASS_GEMM_BUILD_DIR}/libcutlass_gemm_api.a")
    target_compile_definitions(gpu_fdd_improved PRIVATE HAS_CUTLASS_GEMM=1)
    message(STATUS "CUTLASS GEMM API: ENABLED")
    message(STATUS "  library: ${CUTLASS_GEMM_BUILD_DIR}/libcutlass_gemm_api.a")
    message(STATUS "  header:  ${CUTLASS_GEMM_INCLUDE_DIR}/cutlass_gemm_api.h")

    # --- cuFFT LTO callback fatbin (fused FFT output + data prep) ---
    option(DISABLE_CUFFT_LTO_CALLBACK "Disable cuFFT LTO store callback (for A/B benchmarking)" OFF)
    if(NOT DISABLE_CUFFT_LTO_CALLBACK)
    # Extract compute capability from GPU_ARCH (e.g., "121" -> "120" for LTO arch)
    string(REGEX REPLACE "a$" "" GPU_ARCH_NUM "${GPU_ARCH}")
    math(EXPR GPU_COMPUTE_ARCH "${GPU_ARCH_NUM} / 10 * 10")  # 121->120, 100->100, 90->90

    set(FFT_CALLBACK_SRC
        "${CMAKE_CURRENT_SOURCE_DIR}/gpu_fdd_improved/fft_store_callback.cu")
    set(FFT_CALLBACK_FATBIN
        "${CMAKE_CURRENT_BINARY_DIR}/fft_store_callback.fatbin")
    set(FFT_CALLBACK_HEADER
        "${CMAKE_CURRENT_BINARY_DIR}/fft_store_callback_fatbin.h")

    add_custom_command(
        OUTPUT ${FFT_CALLBACK_FATBIN}
        COMMAND ${CMAKE_CUDA_COMPILER}
            -dc -fatbin
            --generate-code arch=compute_${GPU_COMPUTE_ARCH},code=lto_${GPU_COMPUTE_ARCH}
            -I${CUDAToolkit_INCLUDE_DIRS}
            ${FFT_CALLBACK_SRC}
            -o ${FFT_CALLBACK_FATBIN}
        DEPENDS ${FFT_CALLBACK_SRC}
        COMMENT "Compiling cuFFT LTO callback fatbin (compute_${GPU_COMPUTE_ARCH})")

    add_custom_command(
        OUTPUT ${FFT_CALLBACK_HEADER}
        COMMAND ${CUDAToolkit_BIN_DIR}/bin2c
            --name fft_callback_fatbin_data
            --type char --const
            ${FFT_CALLBACK_FATBIN}
            > ${FFT_CALLBACK_HEADER}
        DEPENDS ${FFT_CALLBACK_FATBIN}
        COMMENT "Embedding cuFFT LTO callback fatbin via bin2c")

    # Add generated header as a dependency and enable the callback path
    target_sources(gpu_fdd_improved PRIVATE ${FFT_CALLBACK_HEADER})
    target_include_directories(gpu_fdd_improved PRIVATE ${CMAKE_CURRENT_BINARY_DIR})
    target_compile_definitions(gpu_fdd_improved PRIVATE HAS_CUFFT_LTO_CALLBACK=1)
    message(STATUS "cuFFT LTO callback: ENABLED (compute_${GPU_COMPUTE_ARCH})")
    else()
        message(STATUS "cuFFT LTO callback: DISABLED (A/B benchmark mode)")
    endif()

    # dedisp_api also links against pre-built CUTLASS
    target_include_directories(dedisp_api PRIVATE "${CUTLASS_GEMM_INCLUDE_DIR}")
    # The static .a contains OpenMP symbols that need libgomp AFTER it in the
    # link line. CMake deduplicates OpenMP::OpenMP_CXX targets, so we add the
    # gomp library path directly after the .a (raw string bypasses dedup).
    target_link_libraries(dedisp_api PUBLIC
        "${CUTLASS_GEMM_BUILD_DIR}/libcutlass_gemm_api.a"
        "${OpenMP_gomp_LIBRARY}")
    target_compile_definitions(dedisp_api PRIVATE HAS_CUTLASS_GEMM=1)

    # cuFFT LTO callback for dedisp_api (same fatbin as gpu_fdd_improved)
    target_sources(dedisp_api PRIVATE ${FFT_CALLBACK_HEADER})
    target_include_directories(dedisp_api PRIVATE ${CMAKE_CURRENT_BINARY_DIR})
    target_compile_definitions(dedisp_api PRIVATE HAS_CUFFT_LTO_CALLBACK=1)

    set(HAS_CUTLASS_GEMM_AVAILABLE ON)
endif()

if(BUILD_CUTLASS_GEMM)
    # Link gpu_fdd_improved to the source-built CUTLASS target
    target_link_libraries(gpu_fdd_improved PUBLIC cutlass_gemm_api)
    target_compile_definitions(gpu_fdd_improved PRIVATE HAS_CUTLASS_GEMM=1)
endif()

# --- Transpose kernel benchmark ---
add_executable(bench_transpose_kernels bench_transpose_kernels.cu)
set_target_properties(bench_transpose_kernels PROPERTIES CUDA_ARCHITECTURES "${GPU_ARCH}")
target_include_directories(bench_transpose_kernels PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(bench_transpose_kernels PUBLIC ${CUDA_cufft_LIBRARY})

# --- Dedisp API test binary ---
add_executable(test_dedisp_api test_dedisp_api.cu)
set_target_properties(test_dedisp_api PROPERTIES CUDA_ARCHITECTURES "${GPU_ARCH}")
target_link_libraries(test_dedisp_api PRIVATE dedisp_api)
if(HAS_CUTLASS_GEMM_AVAILABLE)
    target_compile_definitions(test_dedisp_api PRIVATE HAS_CUTLASS_GEMM=1)
endif()

# --- CTest integration ---
enable_testing()

# cuBLAS FP32 tests
add_test(NAME dedisp_cublas_inject_search
    COMMAND $<TARGET_FILE:test_dedisp_api> cublas inject_search)
set_tests_properties(dedisp_cublas_inject_search PROPERTIES
    TIMEOUT 60 LABELS "dedisp;cublas;gpu")

add_test(NAME dedisp_cublas_dedisperse_only
    COMMAND $<TARGET_FILE:test_dedisp_api> cublas dedisperse_only)
set_tests_properties(dedisp_cublas_dedisperse_only PROPERTIES
    TIMEOUT 60 LABELS "dedisp;cublas;gpu")

add_test(NAME dedisp_cublas_high_dm
    COMMAND $<TARGET_FILE:test_dedisp_api> cublas high_dm)
set_tests_properties(dedisp_cublas_high_dm PROPERTIES
    TIMEOUT 60 LABELS "dedisp;cublas;gpu")

add_test(NAME dedisp_cublas_multi_batch
    COMMAND $<TARGET_FILE:test_dedisp_api> cublas multi_batch)
set_tests_properties(dedisp_cublas_multi_batch PROPERTIES
    TIMEOUT 60 LABELS "dedisp;cublas;gpu")

# cuBLASLt FP16 tests
add_test(NAME dedisp_cublaslt_fp16_inject_search
    COMMAND $<TARGET_FILE:test_dedisp_api> cublas_lt_fp16 inject_search)
set_tests_properties(dedisp_cublaslt_fp16_inject_search PROPERTIES
    TIMEOUT 60 LABELS "dedisp;cublaslt;fp16;gpu")

add_test(NAME dedisp_cublaslt_fp16_high_dm
    COMMAND $<TARGET_FILE:test_dedisp_api> cublas_lt_fp16 high_dm)
set_tests_properties(dedisp_cublaslt_fp16_high_dm PROPERTIES
    TIMEOUT 60 LABELS "dedisp;cublaslt;fp16;gpu")

# cuBLASLt FP8 tests
add_test(NAME dedisp_cublaslt_fp8_inject_search
    COMMAND $<TARGET_FILE:test_dedisp_api> cublas_lt_fp8 inject_search)
set_tests_properties(dedisp_cublaslt_fp8_inject_search PROPERTIES
    TIMEOUT 60 LABELS "dedisp;cublaslt;fp8;gpu")

# CUTLASS FP8/FP6/FP4 tests (only when CUTLASS GEMM linked)
if(HAS_CUTLASS_GEMM_AVAILABLE)
    add_test(NAME dedisp_cutlass_fp8_inject_search
        COMMAND $<TARGET_FILE:test_dedisp_api> cutlass_fp8 inject_search)
    set_tests_properties(dedisp_cutlass_fp8_inject_search PROPERTIES
        TIMEOUT 60 LABELS "dedisp;cutlass;fp8;gpu")

    add_test(NAME dedisp_cutlass_fp6_inject_search
        COMMAND $<TARGET_FILE:test_dedisp_api> cutlass_fp6 inject_search)
    set_tests_properties(dedisp_cutlass_fp6_inject_search PROPERTIES
        TIMEOUT 60 LABELS "dedisp;cutlass;fp6;gpu"
        WILL_FAIL TRUE)  # FP6 block-scaled needs M>=128; small test has M=1

    add_test(NAME dedisp_cutlass_fp6_high_dm
        COMMAND $<TARGET_FILE:test_dedisp_api> cutlass_fp6 high_dm)
    set_tests_properties(dedisp_cutlass_fp6_high_dm PROPERTIES
        TIMEOUT 60 LABELS "dedisp;cutlass;fp6;gpu")

    add_test(NAME dedisp_cutlass_fp4_inject_search
        COMMAND $<TARGET_FILE:test_dedisp_api> cutlass_fp4 inject_search)
    set_tests_properties(dedisp_cutlass_fp4_inject_search PROPERTIES
        TIMEOUT 60 LABELS "dedisp;cutlass;fp4;gpu")
endif()

# Link the OpenMP libraries to the executable
target_link_libraries(fdmt_perfect PUBLIC OpenMP::OpenMP_CXX)

# Set the output directory for the executable
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

endif() # standalone-only
