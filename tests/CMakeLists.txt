# enable tests build a common library for all test utilities
add_library(ggp_test googletest/src/gtest-all.cc)
target_include_directories(ggp_test SYSTEM PUBLIC  googletest/include googletest)
target_link_libraries(ggp_test PUBLIC Eigen)
target_link_libraries(ggp_test PUBLIC GGP::ggp)
set_target_properties(ggp_test PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_compile_options(
  ggp_test PUBLIC
  $<IF:$<CONFIG:RELEASE>,-w,-Wall -Wextra>
  $<$<CONFIG:STRICT>:-Werror>
)

# ignore any unkown pragmas if not using OpenMP
if(NOT ${GGP_OPENMP})
  target_compile_options(ggp_test PUBLIC $<$<COMPILE_LANGUAGE:CXX>:
    $<$<CXX_COMPILER_ID:Clang>:-Wno-unknown-pragmas>
    $<$<CXX_COMPILER_ID:GNU>:-Wno-unknown-pragmas>
    >)
endif()

if(BUILD_SHARED_LIBS)
  install(TARGETS ggp_test ${GGP_EXCLUDE_FROM_INSTALL} DESTINATION ${CMAKE_INSTALL_LIBDIR})
endif()

add_subdirectory(utils)
add_subdirectory(host_reference)
# enable_language(Fortran)

if(GGP_NVSHMEM AND GGP_DOWNLOAD_NVSHMEM)
  add_dependencies(ggp_test NVSHMEM)
endif()

set(TEST_LIBS ggp_test)

# custom target for building all tests (useful if GGP_BUILD_ALL_TESTS is OFF)
add_custom_target(all_tests)

macro(GGP_CHECKBUILDTEST mytarget ggpbuildtests)
  if(NOT ${ggpbuildtests})
    set_property(TARGET ${mytarget} PROPERTY EXCLUDE_FROM_ALL 1)
    set(GGP_EXCLUDE_FROM_INSTALL "EXCLUDE_FROM_ALL")
  endif()
  add_dependencies(all_tests ${mytarget})
endmacro()

if(NOT ${GGP_INSTALL_ALL_TESTS})
  set(GGP_EXCLUDE_FROM_INSTALL "EXCLUDE_FROM_ALL")
endif()

if(GGP_ARPACK)
  list(APPEND TEST_LIBS ${ARPACK})
  if(GGP_MPI OR GGP_QMP)
    list(APPEND TEST_LIBS ${PARPACK})
  endif()
endif()

if(GGP_DEDISP_TCFDD)
  if(GGP_BUILD_FROM_SOURCE)
    list(APPEND TEST_LIBS dedisp_api)
  else()
    list(APPEND TEST_LIBS ${DEDISP_API_LIB_PATH})
  endif()
endif()


# define tests
add_executable(c_interface_test c_interface_test.c)
target_link_libraries(c_interface_test ${TEST_LIBS} "-lstdc++")
ggp_checkbuildtest(c_interface_test GGP_BUILD_ALL_TESTS)

add_executable(scale_vector_example scale_vector_example.cpp)
target_link_libraries(scale_vector_example ${TEST_LIBS} "-lstdc++")
ggp_checkbuildtest(scale_vector_example GGP_BUILD_ALL_TESTS)

add_executable(xengine_test xengine_test.cpp)
target_link_libraries(xengine_test ${TEST_LIBS})
ggp_checkbuildtest(xengine_test GGP_BUILD_ALL_TESTS)
install(TARGETS xengine_test ${GGP_EXCLUDE_FROM_INSTALL} DESTINATION ${CMAKE_INSTALL_BINDIR})

add_executable(beamformer_test beamformer_test.cpp)
target_link_libraries(beamformer_test ${TEST_LIBS} "-lstdc++")
ggp_checkbuildtest(beamformer_test GGP_BUILD_ALL_TESTS)

add_executable(beamformer_test_dsa110 beamformer_test_dsa110.cpp)
target_link_libraries(beamformer_test_dsa110 ${TEST_LIBS} "-lstdc++")
ggp_checkbuildtest(beamformer_test_dsa110 GGP_BUILD_ALL_TESTS)

add_executable(dedisp_interface_test dedisp_interface_test.cpp)
target_link_libraries(dedisp_interface_test ${TEST_LIBS})
ggp_checkbuildtest(dedisp_interface_test GGP_BUILD_ALL_TESTS)
install(TARGETS dedisp_interface_test ${GGP_EXCLUDE_FROM_INSTALL} DESTINATION ${CMAKE_INSTALL_BINDIR})

add_executable(visibility_pipeline_test visibility_pipeline_test.cpp)
target_link_libraries(visibility_pipeline_test ${TEST_LIBS})
if(GGP_DEDISP_TCFDD)
  target_compile_definitions(visibility_pipeline_test PRIVATE DEDISP_API_LIB)
  if(NOT GGP_BUILD_FROM_SOURCE AND GGP_DEDISP_TCFDD_INCLUDE)
    target_include_directories(visibility_pipeline_test SYSTEM PRIVATE ${GGP_DEDISP_TCFDD_INCLUDE})
  endif()
endif()
if(GGP_TCC)
  target_compile_definitions(visibility_pipeline_test PRIVATE GGP_TCC_ENABLED)
endif()
ggp_checkbuildtest(visibility_pipeline_test GGP_BUILD_ALL_TESTS)

add_executable(voltage_pipeline_test voltage_pipeline_test.cpp)
target_link_libraries(voltage_pipeline_test ${TEST_LIBS})
if(GGP_DEDISP_TCFDD)
  target_compile_definitions(voltage_pipeline_test PRIVATE DEDISP_API_LIB)
  if(NOT GGP_BUILD_FROM_SOURCE AND GGP_DEDISP_TCFDD_INCLUDE)
    target_include_directories(voltage_pipeline_test SYSTEM PRIVATE ${GGP_DEDISP_TCFDD_INCLUDE})
  endif()
endif()
ggp_checkbuildtest(voltage_pipeline_test GGP_BUILD_ALL_TESTS)

add_executable(cutlass_cublas_benchmark cutlass_cublas_benchmark.cpp)
target_link_libraries(cutlass_cublas_benchmark ${TEST_LIBS})
if(GGP_TCC)
  target_compile_definitions(cutlass_cublas_benchmark PRIVATE GGP_TCC_ENABLED)
endif()
ggp_checkbuildtest(cutlass_cublas_benchmark GGP_BUILD_ALL_TESTS)

if(GGP_BUILD_NATIVE_LAPACK)
  add_executable(blas_interface_test blas_interface_test.cpp)
  target_link_libraries(blas_interface_test ${TEST_LIBS})
  ggp_checkbuildtest(blas_interface_test GGP_BUILD_ALL_TESTS)
  install(TARGETS blas_interface_test ${GGP_EXCLUDE_FROM_INSTALL} DESTINATION ${CMAKE_INSTALL_BINDIR})
endif()

if(GGP_QIO)
  add_executable(io_test io_test.cpp)
  target_link_libraries(io_test ${TEST_LIBS})
  ggp_checkbuildtest(io_test GGP_BUILD_ALL_TESTS)
  install(TARGETS io_test ${GGP_EXCLUDE_FROM_INSTALL} DESTINATION ${CMAKE_INSTALL_BINDIR})
endif()

add_executable(tune_test tune_test.cpp)
target_link_libraries(tune_test ${TEST_LIBS})
ggp_checkbuildtest(tune_test GGP_BUILD_ALL_TESTS)
install(TARGETS tune_test ${GGP_EXCLUDE_FROM_INSTALL} DESTINATION ${CMAKE_INSTALL_BINDIR})

if(GGP_MPI OR GGP_QMP)
  #helper function to extract required number of ranks from environment variable
  function(get_test_ranks TEST_GRID RANKS)
    separate_arguments(TEST_GRID)
    LIST(LENGTH TEST_GRID TEST_LENGTH)
    if(NOT ${TEST_LENGTH} EQUAL 4)
      message(SEND_ERROR "GGP_TEST_GRID_SIZE takes 4 INTEGERS. GOT: ${TEST_GRID_SIZE}")
    endif()
      LIST(JOIN TEST_GRID * TEST_GRID_EXPR)
    math(EXPR TEST_RANKS ${TEST_GRID_EXPR} OUTPUT_FORMAT DECIMAL)
    set(${RANKS} ${TEST_RANKS} PARENT_SCOPE )
  endfunction()

  set(GGP_TEST_NUM_PROCS 1)
  if(DEFINED ENV{GGP_TEST_GRID_SIZE})
    get_test_ranks($ENV{GGP_TEST_GRID_SIZE} GGP_TEST_NUM_PROCS)
  endif()
  message(STATUS "ctest will run on ${GGP_TEST_NUM_PROCS} processes")
  set(GGP_CTEST_LAUNCH ${MPIEXEC_EXECUTABLE};${MPIEXEC_NUMPROC_FLAG};${GGP_TEST_NUM_PROCS};${MPIEXEC_PREFLAGS}
    CACHE STRING "CTest Launcher command for GGP's tests")
endif()

add_test(NAME xengine_test
  COMMAND ${GGP_CTEST_LAUNCH} $<TARGET_FILE:xengine_test> ${MPIEXEC_POSTFLAGS}
  --enable-testing true
  --gtest_output=xml:xengine_test.xml)

add_test(NAME beamformer_test_dsa110
  COMMAND ${GGP_CTEST_LAUNCH} $<TARGET_FILE:beamformer_test_dsa110> ${MPIEXEC_POSTFLAGS}
  --enable-testing true
  --gtest_output=xml:beamformer_test_dsa110.xml)

#BLAS interface test
if(GGP_BUILD_NATIVE_LAPACK)
  add_test(NAME blas_interface_test
    COMMAND ${GGP_CTEST_LAUNCH} $<TARGET_FILE:blas_interface_test> ${MPIEXEC_POSTFLAGS}
    --blas-gemm-mnk 64 64 64
    --blas-gemm-leading-dims 128 128 128
    --blas-gemm-offsets 16 16 16
    --blas-gemm-alpha 1.0 2.0
    --blas-gemm-beta -3.0 1.5
    --blas-gemm-trans-a T
    --blas-gemm-trans-b C
    --blas-lu-inv-mat-size 96
    --blas-data-type Z
    --blas-data-order row
    --blas-batch 20
    --enable-testing true
    --gtest_output=xml:blas_interface_test.xml)
endif()

# enable the precisions that are compiled
math(EXPR double_prec  "${GGP_PRECISION} & 8")
math(EXPR single_prec  "${GGP_PRECISION} & 4")
math(EXPR half_prec    "${GGP_PRECISION} & 2")
math(EXPR quarter_prec "${GGP_PRECISION} & 1")

if(double_prec AND single_prec)
  set(TEST_PRECS single double)
elseif(double_prec)
  set(TEST_PRECS double)
elseif(single_prec)
  set(TEST_PRECS single)
endif()


if(GGP_QIO)
  add_test(NAME io_test
           COMMAND ${GGP_CTEST_LAUNCH} $<TARGET_FILE:io_test> ${MPIEXEC_POSTFLAGS}
                   --dim 4 6 8 10
                   --gtest_output=xml:io_test.xml)
endif()

add_test(NAME tune_test
         COMMAND  ${GGP_CTEST_LAUNCH} $<TARGET_FILE:tune_test> ${MPIEXEC_POSTFLAGS}
                   --gtest_output=xml:tune_test.xml)

add_test(NAME beamformer_test
  COMMAND ${GGP_CTEST_LAUNCH} $<TARGET_FILE:beamformer_test> ${MPIEXEC_POSTFLAGS}
  --enable-testing true
  --gtest_output=xml:beamformer_test.xml)

if(GGP_DEDISP_TCFDD)
  add_test(NAME dedisp_interface_test
    COMMAND ${GGP_CTEST_LAUNCH} $<TARGET_FILE:dedisp_interface_test> ${MPIEXEC_POSTFLAGS})
endif()

add_test(NAME visibility_pipeline_test
  COMMAND ${GGP_CTEST_LAUNCH} $<TARGET_FILE:visibility_pipeline_test> ${MPIEXEC_POSTFLAGS}
  --enable-testing true
  --gtest_output=xml:visibility_pipeline_test.xml)

add_test(NAME voltage_pipeline_test
  COMMAND ${GGP_CTEST_LAUNCH} $<TARGET_FILE:voltage_pipeline_test> ${MPIEXEC_POSTFLAGS}
  --enable-testing true
  --gtest_output=xml:voltage_pipeline_test.xml)

add_test(NAME cutlass_cublas_benchmark
  COMMAND ${GGP_CTEST_LAUNCH} $<TARGET_FILE:cutlass_cublas_benchmark> ${MPIEXEC_POSTFLAGS}
  --enable-testing true
  --gtest_output=xml:cutlass_cublas_benchmark.xml)

add_test(NAME c_interface_test
  COMMAND ${GGP_CTEST_LAUNCH} $<TARGET_FILE:c_interface_test> ${MPIEXEC_POSTFLAGS})

# ======================================================================
# bench_pipeline: production pipeline benchmark with per-stage timing
# Requires CUTLASS interface + dedisp_tcfdd for the full pipeline
# ======================================================================
if(GGP_BUILD_BENCH_PIPELINE AND GGP_CUTLASS_INTERFACE AND GGP_DEDISP_TCFDD)
  enable_language(CUDA)
  find_package(CUDAToolkit REQUIRED)
  add_executable(bench_pipeline ${CMAKE_SOURCE_DIR}/lib/bench_pipeline.cu)
  target_link_libraries(bench_pipeline PRIVATE ${TEST_LIBS} CUDA::curand)
  target_compile_definitions(bench_pipeline PRIVATE CUTLASS_GEMM_API DEDISP_API_LIB)

  # Set CUDA architecture matching the project's GPU target (GGP_GPU_ARCH is a
  # cache variable, e.g. "sm_90a").  This ensures SASS is embedded rather than
  # relying on PTX JIT which fails when driver < toolkit version.
  if(GGP_GPU_ARCH)
    string(REGEX REPLACE "^sm_" "" _BENCH_ARCH_TOKEN "${GGP_GPU_ARCH}")
    set_target_properties(bench_pipeline PROPERTIES CUDA_ARCHITECTURES "${_BENCH_ARCH_TOKEN}")
  endif()

  # bench_pipeline uses CUTLASS PIMPL + dedisp APIs directly — needs their headers
  if(GGP_BUILD_FROM_SOURCE)
    target_include_directories(bench_pipeline SYSTEM PRIVATE
        ${CMAKE_SOURCE_DIR}/cutlass_interface
        ${CMAKE_SOURCE_DIR}/dedisp_tcfdd)
  else()
    if(GGP_CUTLASS_INTERFACE_INCLUDE)
      target_include_directories(bench_pipeline SYSTEM PRIVATE ${GGP_CUTLASS_INTERFACE_INCLUDE})
    endif()
    if(GGP_DEDISP_TCFDD_INCLUDE)
      target_include_directories(bench_pipeline SYSTEM PRIVATE ${GGP_DEDISP_TCFDD_INCLUDE})
    endif()
  endif()

  ggp_checkbuildtest(bench_pipeline GGP_BUILD_ALL_TESTS)
  message(STATUS "bench_pipeline: enabled (CUTLASS + dedisp)")
elseif(GGP_BUILD_BENCH_PIPELINE)
  message(STATUS "bench_pipeline: disabled (requires GGP_CUTLASS_INTERFACE + GGP_DEDISP_TCFDD)")
endif()

