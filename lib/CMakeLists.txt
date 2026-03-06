# cmake-format: off

# this allows simplified running of clang-tidy
if(${CMAKE_BUILD_TYPE} STREQUAL "DEVEL")
  set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
endif()

# build up git version add -debug to GITVERSION if we build with debug options enabled
string(REGEX MATCH [Dd][Ee][Bb][Uu][Gg] DEBUG_BUILD ${CMAKE_BUILD_TYPE})
if(DEBUG_BUILD)
  if(GITVERSION)
    set(GITVERSION ${GITVERSION}-debug)
  else()
    set(GITVERSION debug)
  endif()
endif()

set (GGP_OBJS
  # cmake-format: sortable
  monitor.cpp 
  timer.cpp
  ggp_ptr.cpp
  util_ggp.cpp
  comm_common.cpp
  communicator_stack.cpp
  tune.cpp
  interface_ggp.cpp
  xengine.cpp
  voltage_bf.cpp
  voltage_bf_dsa110.cpp
  visibility_bf.cpp
  dedispersion_ggp.cpp
  filterbank.cpp
  visibility_array.cpp
  voltage_array.cpp
  pipeline_arrays.cpp
  
  transform_reduce.cu
  reorder_data.cu
  scale_vector.cu
  promote_data.cu
  interleave_tri.cu
  reduce_helper.cu
  inspect_dev_elems.cu
  tri_from_herm.cu
  sum_incoherent_beam.cu
  power_sum_transpose.cu
  sum_inner_chan_time.cu

  dsa110_specific.cu
  
  dedisp_interface.cpp

  cutlass_interface.cu
  tcc_interface.cu
  corner_turn.cu
  visibility_pipeline.cu
  voltage_pipeline.cu
  visibility_bf_kernels.cu
  imaging_pipeline.cu

  version.cpp)
# cmake-format: on

# split source into cu and cpp files
foreach(item ${GGP_OBJS})
  string(REGEX MATCH ".+\\.cu$" item_match ${item})
  if(item_match)
    list(APPEND GGP_CU_OBJS ${item})
  endif(item_match)
endforeach(item ${GGP_OBJS})

list(REMOVE_ITEM GGP_OBJS ${GGP_CU_OBJS})

if(BUILD_FORTRAN_INTERFACE)
  list(APPEND GGP_OBJS ggp_fortran.F90)

  # set_source_files_properties(ggp_fortran.F90 PROPERTIES OBJECT_OUTPUTS ${CMAKE_CURRENT_BINARY_DIR}/ggp_fortran.mod)
endif()

# GGP_CU_OBJS should contain all cuda files now GGP_OBJS all c, cpp, fortran sources
# if we have a git version make version.cpp depend on git head so that it is rebuild if the git sha changed
if(${CMAKE_BUILD_TYPE} STREQUAL "DEVEL")
  if(GITVERSION)
    find_path(
      GGP_GITDIR NAME HEAD
      PATHS ${CMAKE_SOURCE_DIR}/.git/logs
      NO_DEFAULT_PATH)
    include(AddFileDependencies)
    if(GGP_GITDIR)
      add_file_dependencies(version.cpp ${GGP_GITDIR}/HEAD)
    endif()
  endif()
  mark_as_advanced(GGP_GITDIR)
endif()

# generate a cmake object library for all cpp files first
add_library(ggp_cpp OBJECT ${GGP_OBJS})

# add comms and QIO
target_sources(
  ggp_cpp
  PRIVATE
    $<IF:$<BOOL:${GGP_MPI}>,communicator_mpi.cpp,$<IF:$<BOOL:${GGP_QMP}>,communicator_qmp.cpp,communicator_single.cpp>>
)

target_sources(ggp_cpp PRIVATE $<$<BOOL:${GGP_QIO}>:qio_field.cpp layout_hyper.cpp>)

if(GGP_BUILD_SHAREDLIB)
  set_target_properties(ggp_cpp PROPERTIES POSITION_INDEPENDENT_CODE TRUE)
  add_library(ggp SHARED)
else()
  add_library(ggp STATIC)
endif()
add_library(GGP::ggp ALIAS ggp)

# add target specific files
if(${GGP_TARGET_TYPE} STREQUAL "CUDA")
  include(targets/cuda/target_cuda.cmake)
endif()
if(${GGP_TARGET_TYPE} STREQUAL "HIP")
  include(targets/hip/target_hip.cmake)
endif()
if(${GGP_TARGET_TYPE} STREQUAL "SYCL")
  include(targets/sycl/target_sycl.cmake)
endif()

# ======================================================================
# Derive internal feature flags from user-provided paths
# ======================================================================
if(GGP_BUILD_FROM_SOURCE)
  set(GGP_CUTLASS_INTERFACE ON)
  set(GGP_DEDISP_TCFDD ON)
else()
  if(GGP_CUTLASS_INTERFACE_PATH)
    set(GGP_CUTLASS_INTERFACE ON)
  endif()
  if(GGP_DEDISP_TCFDD_PATH)
    set(GGP_DEDISP_TCFDD ON)
  endif()
endif()
# Propagate to sibling subdirectories (tests/) so they can conditionally
# link dedisp_api and define DEDISP_API_LIB for test executables.
if(GGP_DEDISP_TCFDD)
  set(GGP_DEDISP_TCFDD ON PARENT_SCOPE)
endif()
if(GGP_CUTLASS_INTERFACE)
  set(GGP_CUTLASS_INTERFACE ON PARENT_SCOPE)
endif()
# Propagate include paths for bench_pipeline (needs direct API headers)
if(GGP_CUTLASS_INTERFACE_INCLUDE)
  set(GGP_CUTLASS_INTERFACE_INCLUDE "${GGP_CUTLASS_INTERFACE_INCLUDE}" PARENT_SCOPE)
endif()
if(GGP_DEDISP_TCFDD_INCLUDE)
  set(GGP_DEDISP_TCFDD_INCLUDE "${GGP_DEDISP_TCFDD_INCLUDE}" PARENT_SCOPE)
endif()

# ======================================================================
# Source build: cutlass_interface + dedisp_tcfdd as subdirectories
# ======================================================================
if(GGP_BUILD_FROM_SOURCE)
  # Map GGP_GPU_ARCH → dedisp GPU_ARCH (strip sm_ prefix)
  string(REGEX REPLACE "^sm_" "" _DEDISP_GPU_ARCH "${GGP_GPU_ARCH}")

  # Configure dedisp_tcfdd's cache variables before add_subdirectory
  set(GPU_ARCH "${_DEDISP_GPU_ARCH}" CACHE STRING "" FORCE)
  set(CUTLASS_DIR "${GGP_CUTLASS_DIR}" CACHE PATH "" FORCE)
  set(BUILD_CUTLASS_GEMM ON CACHE BOOL "" FORCE)

  # Add dedisp_tcfdd (which in turn adds cutlass_interface via BUILD_CUTLASS_GEMM)
  add_subdirectory(${CMAKE_SOURCE_DIR}/dedisp_tcfdd
                   ${CMAKE_BINARY_DIR}/dedisp_tcfdd
                   EXCLUDE_FROM_ALL)

  message(STATUS "Source build: dedisp_api + cutlass_gemm_api targets available")
endif()

# Set the maximum multi-RHS per kernel if not already set by the target
if(NOT DEFINED GGP_MAX_MULTI_RHS)
  set(GGP_MAX_MULTI_RHS "16" CACHE STRING "maximum number of simultaneous RHS in a kernel")
  message(STATUS "Max number of rhs per kernel: ${GGP_MAX_MULTI_RHS}")
endif()

# make one library
target_sources(ggp PRIVATE $<TARGET_OBJECTS:ggp_cpp> $<$<TARGET_EXISTS:ggp_pack>:$<TARGET_OBJECTS:ggp_pack>>
                            ${GGP_CU_OBJS})

# for a develop build reduce the size by compressing the debug information
include(CheckLinkerFlag)
check_linker_flag(CXX "-Wl,--compress-debug-sections=zlib" GGP_LINKER_COMPRESS)
include(CheckCXXCompilerFlag)
check_cxx_compiler_flag(-gz GGP_COMPRESS_DEBUG)

if(NOT GGP_BACKWARDS)
  if(${GGP_LINKER_COMPRESS})
    target_link_options(ggp PRIVATE $<$<CONFIG:DEVEL>:$<${GGP_LINKER_COMPRESS}:-Wl,--compress-debug-sections=zlib>>)
  endif()
  if(${GGP_COMPRESS_DEBUG})
    target_compile_options(ggp_cpp PRIVATE $<$<CONFIG:DEVEL>:$<${GGP_COMPRESS_DEBUG}:-gz>>)
    target_compile_options(ggp PRIVATE $<$<CONFIG:DEVEL>:$<${GGP_COMPRESS_DEBUG}:-gz>>)
  endif()
endif()

# set up GGP compile options, put them before target specific options
target_compile_options(
  ggp BEFORE PRIVATE $<$<COMPILE_LANGUAGE:CXX,C,Fortran>:
               $<IF:$<CONFIG:RELEASE>,-w,-Wall -Wextra>
               $<$<CONFIG:STRICT>:-Werror>
  >)

target_compile_definitions(
  ggp PRIVATE $<$<CONFIG:DEVEL>:DEVEL> $<$<CONFIG:HOSTDEBUG>:HOST_DEBUG> $<$<CONFIG:DEBUG>:HOST_DEBUG>
               $<$<CONFIG:DEBUG>:DEVICE_DEBUG> $<$<CONFIG:SANITIZE>:HOST_DEBUG>)
target_include_directories(ggp PRIVATE .)
target_include_directories(ggp PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(ggp SYSTEM PRIVATE ../include/externals)

target_link_libraries(ggp PRIVATE $<BUILD_INTERFACE:Eigen>)
target_include_directories(ggp PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include/>
                                       $<INSTALL_INTERFACE:include/>)

target_include_directories(ggp PUBLIC $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/include> $<INSTALL_INTERFACE:include>)

target_include_directories(ggp_cpp SYSTEM PRIVATE ../include/externals)

target_link_libraries(ggp_cpp PRIVATE $<BUILD_INTERFACE:Eigen>)
target_compile_definitions(ggp_cpp PRIVATE $<TARGET_PROPERTY:ggp,COMPILE_DEFINITIONS>)
target_include_directories(ggp_cpp PRIVATE $<TARGET_PROPERTY:ggp,INCLUDE_DIRECTORIES>)
target_compile_options(ggp_cpp PRIVATE $<TARGET_PROPERTY:ggp,COMPILE_OPTIONS>)

add_subdirectory(targets/generic)
target_include_directories(ggp PRIVATE ../include/targets/generic)
add_subdirectory(interface)

# ignore any unknown pragmas
target_compile_options(ggp_cpp PUBLIC $<$<COMPILE_LANGUAGE:CXX>:
  $<$<CXX_COMPILER_ID:Clang>:-Wno-unknown-pragmas>
  $<$<CXX_COMPILER_ID:GNU>:-Wno-unknown-pragmas>
  >)

# add some definitions that cause issues with cmake 3.7 and nvcc only to cpp files
target_compile_definitions(ggp_cpp PUBLIC -DGGP_HASH="${HASH}")
if(GITVERSION)
  target_compile_definitions(ggp_cpp PUBLIC -DGITVERSION="${GITVERSION}")
endif()

# some clang warnings should be warning even when turning warnings into errors
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  target_compile_options(ggp_cpp PUBLIC $<$<COMPILE_LANGUAGE:CXX>: -Wno-error=unused-function>)
  # this is a hack to get colored diagnostics back when using Ninja and clang
  if(CMAKE_GENERATOR MATCHES "Ninja")
    target_compile_options(ggp PUBLIC $<$<COMPILE_LANGUAGE:CXX>:-fcolor-diagnostics>)
  endif()
endif()

target_link_libraries(ggp PUBLIC Threads::Threads)

if(${GGP_BUILD_NATIVE_LAPACK} STREQUAL "ON")
  target_compile_definitions(ggp PRIVATE NATIVE_LAPACK_LIB)
endif()

if(${GGP_BUILD_NATIVE_FFT} STREQUAL "ON")
  target_compile_definitions(ggp PRIVATE NATIVE_FFT_LIB)
endif()

# MULTI GPU AND USQCD
if(GGP_MPI OR GGP_QMP)
  target_compile_definitions(ggp PUBLIC MULTI_GPU)
endif()

if(GGP_MPI)
  target_compile_definitions(ggp PUBLIC MPI_COMMS)
  target_link_libraries(ggp PUBLIC MPI::MPI_CXX)
endif()

if(GGP_QMP)
  target_compile_definitions(ggp PUBLIC QMP_COMMS)
  target_link_libraries(ggp PUBLIC QMP::qmp)
  target_link_libraries(ggp PUBLIC MPI::MPI_CXX)
endif()

if(GGP_QIO)
  target_compile_definitions(ggp PUBLIC HAVE_QIO)
  target_link_libraries(ggp PUBLIC QIO::qio)
endif()

# ======================================================================
# CUTLASS Interface (cutlass_interface/)
# ======================================================================
if(GGP_CUTLASS_INTERFACE)
  target_compile_definitions(ggp PRIVATE CUTLASS_GEMM_API)
  if(GGP_BUILD_FROM_SOURCE)
    # Source build: PRIVATE because ggp is a shared lib (symbols embedded in .so).
    # PUBLIC would require adding these targets to the install export set.
    target_link_libraries(ggp PRIVATE cutlass_gemm_api)
    message(STATUS "CUTLASS interface: cutlass_gemm_api (from source)")
  else()
    # Pre-built: build dir for .a, source dir for headers
    if(NOT GGP_CUTLASS_INTERFACE_INCLUDE)
      message(FATAL_ERROR
        "GGP_CUTLASS_INTERFACE_PATH requires GGP_CUTLASS_INTERFACE_INCLUDE "
        "(source directory containing cutlass_gemm_api.h)")
    endif()
    target_include_directories(ggp SYSTEM PRIVATE "${GGP_CUTLASS_INTERFACE_INCLUDE}")
    find_library(CUTLASS_GEMM_API_LIB_PATH cutlass_gemm_api
                 PATHS "${GGP_CUTLASS_INTERFACE_PATH}" NO_DEFAULT_PATH)
    if(NOT CUTLASS_GEMM_API_LIB_PATH)
      message(FATAL_ERROR "Could not find libcutlass_gemm_api.a in ${GGP_CUTLASS_INTERFACE_PATH}")
    endif()
    target_link_libraries(ggp PUBLIC ${CUTLASS_GEMM_API_LIB_PATH})
    message(STATUS "CUTLASS interface: ${CUTLASS_GEMM_API_LIB_PATH}")
    message(STATUS "  includes: ${GGP_CUTLASS_INTERFACE_INCLUDE}")
  endif()
endif()

# ======================================================================
# TCC (Tensor-Core Correlator)
# ======================================================================
if(GGP_TCC)
  if(NOT GGP_TCC_PATH)
    message(FATAL_ERROR "GGP_TCC=ON requires GGP_TCC_PATH to be set")
  endif()
  target_compile_definitions(ggp PRIVATE TCC_API)
  target_compile_definitions(ggp_cpp PRIVATE TCC_API)
  target_include_directories(ggp SYSTEM PUBLIC ${GGP_TCC_PATH})
  target_include_directories(ggp SYSTEM PUBLIC ${GGP_TCC_PATH}/build/_deps/cudawrappers-src/include)
  find_library(TCC_LIB tcc PATHS ${GGP_TCC_PATH}/build/libtcc NO_DEFAULT_PATH)
  if(NOT TCC_LIB)
    message(FATAL_ERROR "Could not find libtcc.so in ${GGP_TCC_PATH}/build/libtcc. "
            "Build TCC first: cd tensor-core-correlator && cmake -S . -B build && make -C build")
  endif()
  target_link_libraries(ggp PUBLIC ${TCC_LIB})
  find_package(CUDAToolkit REQUIRED)
  target_link_libraries(ggp PUBLIC CUDA::nvrtc CUDA::cuda_driver)
  message(STATUS "TCC: ${TCC_LIB}")
endif()

# ======================================================================
# Imaging Pipeline (gridding + 2D FFT)
# ======================================================================
find_package(CUDAToolkit QUIET)
if(CUDAToolkit_FOUND)
  target_link_libraries(ggp PUBLIC CUDA::cufft)
endif()

# ======================================================================
# Dedisp TCFDD (dedisp_tcfdd/)
# ======================================================================
if(GGP_DEDISP_TCFDD)
  target_compile_definitions(ggp PRIVATE DEDISP_API_LIB)
  target_compile_definitions(ggp_cpp PRIVATE DEDISP_API_LIB)
  if(GGP_BUILD_FROM_SOURCE)
    # Source build: PRIVATE because ggp is a shared lib (symbols embedded in .so).
    # PUBLIC would require adding these targets to the install export set.
    target_link_libraries(ggp PRIVATE dedisp_api)
    message(STATUS "Dedisp TCFDD: dedisp_api (from source)")
  else()
    # Pre-built: build dir for .a, source dir for headers
    if(NOT GGP_DEDISP_TCFDD_INCLUDE)
      message(FATAL_ERROR
        "GGP_DEDISP_TCFDD_PATH requires GGP_DEDISP_TCFDD_INCLUDE "
        "(source directory containing dedisp_api.h)")
    endif()
    target_include_directories(ggp SYSTEM PRIVATE "${GGP_DEDISP_TCFDD_INCLUDE}")
    find_library(DEDISP_API_LIB_PATH dedisp_api PATHS ${GGP_DEDISP_TCFDD_PATH} NO_DEFAULT_PATH)
    if(NOT DEDISP_API_LIB_PATH)
      message(FATAL_ERROR "Could not find libdedisp_api.a in ${GGP_DEDISP_TCFDD_PATH}")
    endif()
    target_link_libraries(ggp PUBLIC ${DEDISP_API_LIB_PATH})
    find_package(CUDAToolkit REQUIRED)
    target_link_libraries(ggp PUBLIC CUDA::cufft CUDA::cublas CUDA::curand CUDA::cublasLt)
    # libdedisp_api.a is built with OpenMP — link it so symbols resolve in libggp.so
    find_package(OpenMP REQUIRED)
    target_link_libraries(ggp PUBLIC OpenMP::OpenMP_CXX)
    message(STATUS "Dedisp TCFDD: ${DEDISP_API_LIB_PATH}")
    message(STATUS "  includes: ${GGP_DEDISP_TCFDD_INCLUDE}")
  endif()
endif()

if(GGP_OPENBLAS)
  target_compile_definitions(ggp PRIVATE OPENBLAS_LIB)

  if(GGP_DOWNLOAD_OPENBLAS)
    target_link_libraries(ggp PUBLIC openblas)
    target_link_libraries(ggp_cpp PUBLIC openblas)
  else()
    target_link_libraries(ggp INTERFACE ${OPENBLAS})
  endif()
endif()

if(GGP_USE_EIGEN)
  target_compile_definitions(ggp PRIVATE EIGEN)
endif()

if(GGP_OPENMP)
  target_link_libraries(ggp PUBLIC OpenMP::OpenMP_CXX)
  target_compile_definitions(ggp PUBLIC GGP_OPENMP)
endif()

# set which precisions to enable
target_compile_definitions(ggp PUBLIC GGP_PRECISION=${GGP_PRECISION})
target_compile_definitions(ggp PUBLIC GGP_RECONSTRUCT=${GGP_RECONSTRUCT})

if(GGP_ALTERNATIVE_I_TO_F)
  target_compile_definitions(ggp PRIVATE GGP_ALTERNATIVE_I_TO_F)
endif()

if(GGP_BACKWARDS)
  target_include_directories(ggp_cpp SYSTEM PRIVATE ${backward-cpp_SOURCE_DIR})

  set_property(
    SOURCE comm_common.cpp
    APPEND
    PROPERTY COMPILE_DEFINITIONS ${BACKWARD_DEFINITIONS})
  set_property(
    SOURCE comm_common.cpp
    APPEND
    PROPERTY COMPILE_DEFINITIONS GGP_BACKWARDSCPP)
  target_link_libraries(ggp PUBLIC ${BACKWARD_LIBRARIES})
endif()

configure_file(../include/ggp_define.h.in ../include/ggp_define.h @ONLY)
install(FILES "${CMAKE_BINARY_DIR}/include/ggp_define.h" DESTINATION include/)

# until we define an install step copy the include directory to the build directory
add_custom_command(
  TARGET ggp
  POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_SOURCE_DIR}/include ${CMAKE_BINARY_DIR}/include)

# some hackery to prevent having old shared / static builds of ggp messing with the current build
add_custom_command(
  TARGET ggp
  PRE_LINK
  COMMAND ${CMAKE_COMMAND} -E remove ${CMAKE_CURRENT_BINARY_DIR}/libggp.a ${CMAKE_CURRENT_BINARY_DIR}/libggp.so)

install(
  TARGETS ggp
  EXPORT GGPTargets
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
  RUNTIME DESTINATION bin
  INCLUDES
  DESTINATION include)

install(
  EXPORT GGPTargets
  FILE GGPTargets.cmake
  NAMESPACE GGP::
  DESTINATION lib/cmake/GGP)

install(DIRECTORY ${CMAKE_SOURCE_DIR}/include/ DESTINATION include)

include(CMakePackageConfigHelpers)
write_basic_package_version_file(
  "${CMAKE_CURRENT_BINARY_DIR}/ggpConfigVersion.cmake"
  VERSION ${GGP_VERSION}
  COMPATIBILITY AnyNewerVersion)

include(CMakePackageConfigHelpers)

configure_package_config_file(${CMAKE_SOURCE_DIR}/GGPConfig.cmake.in GGPConfig.cmake INSTALL_DESTINATION lib/cmake/GGP)

write_basic_package_version_file(
  GGPConfigVersion.cmake
  VERSION ${PACKAGE_VERSION}
  COMPATIBILITY AnyNewerVersion)

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/GGPConfigVersion.cmake ${CMAKE_CURRENT_BINARY_DIR}/GGPConfig.cmake
	DESTINATION lib/cmake/GGP)
