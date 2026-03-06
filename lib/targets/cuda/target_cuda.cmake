# ######################################################################################################################
# CUDA specific part of CMakeLists
set(CMAKE_CUDA_EXTENSIONS OFF)

find_package(CUDAToolkit REQUIRED)

set(GGP_TARGET_CUDA ON)

if(DEFINED ENV{GGP_GPU_ARCH})
  set(GGP_DEFAULT_GPU_ARCH $ENV{GGP_GPU_ARCH})
else()
  # Auto-detect GPU architecture via nvidia-smi
  execute_process(
    COMMAND nvidia-smi --query-gpu=compute_cap --format=csv,noheader
    OUTPUT_VARIABLE _GGP_GPU_COMPUTE_CAP
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE _GGP_NVSMI_RESULT
    ERROR_QUIET)
  if(_GGP_NVSMI_RESULT EQUAL 0 AND _GGP_GPU_COMPUTE_CAP)
    # Take the first GPU if multiple are present
    string(REGEX MATCH "^[0-9]+\\.[0-9]+" _GGP_GPU_COMPUTE_CAP "${_GGP_GPU_COMPUTE_CAP}")
    # Convert "12.1" → "121" → "sm_121", "9.0" → "90" → "sm_90"
    string(REPLACE "." "" _GGP_GPU_ARCH_NUM "${_GGP_GPU_COMPUTE_CAP}")
    # Append 'a' suffix for arch-accelerated targets (SM90+)
    if(_GGP_GPU_ARCH_NUM MATCHES "^(90|100)")
      set(GGP_DEFAULT_GPU_ARCH "sm_${_GGP_GPU_ARCH_NUM}a")
    else()
      set(GGP_DEFAULT_GPU_ARCH "sm_${_GGP_GPU_ARCH_NUM}")
    endif()
    message(STATUS "Auto-detected GPU compute ${_GGP_GPU_COMPUTE_CAP} → ${GGP_DEFAULT_GPU_ARCH}")
  else()
    set(GGP_DEFAULT_GPU_ARCH sm_70)
    message(STATUS "nvidia-smi not available, defaulting to ${GGP_DEFAULT_GPU_ARCH}")
  endif()
endif()
if(NOT GGP_GPU_ARCH)
  message(STATUS "Building GGP for GPU ARCH " "${GGP_DEFAULT_GPU_ARCH}")
endif()

set(GGP_GPU_ARCH
    ${GGP_DEFAULT_GPU_ARCH}
    CACHE STRING "set the GPU architecture (e.g. sm_70, sm_80, sm_90a, sm_100a, sm_121)")
set_property(CACHE GGP_GPU_ARCH PROPERTY STRINGS sm_60 sm_70 sm_80 sm_90 sm_90a sm_100a sm_120 sm_121)
set(GGP_GPU_ARCH_SUFFIX
    ""
    CACHE STRING "set the GPU architecture suffix (virtual, real). Leave empty for no suffix.")
set_property(CACHE GGP_GPU_ARCH_SUFFIX PROPERTY STRINGS "real" "virtual" " ")
mark_as_advanced(GGP_GPU_ARCH_SUFFIX)

# we don't yet use CMAKE_CUDA_ARCHITECTURES as primary way to set GPU architecture so marking it advanced to avoid
# confusion
mark_as_advanced(CMAKE_CUDA_ARCHITECTURES)

# ######################################################################################################################
# define CUDA flags
set(CMAKE_CUDA_HOST_COMPILER
    "${CMAKE_CXX_COMPILER}"
    CACHE FILEPATH "Host compiler to be used by nvcc")
set(CMAKE_CUDA_STANDARD ${GGP_CXX_STANDARD})
set(CMAKE_CUDA_STANDARD_REQUIRED True)
mark_as_advanced(CMAKE_CUDA_HOST_COMPILER)

set(CMAKE_CUDA_FLAGS_DEVEL
    "-g -O3 "
    CACHE STRING "Flags used by the CUDA compiler during regular development builds.")
set(CMAKE_CUDA_FLAGS_STRICT
    "-O3"
    CACHE STRING "Flags used by the CUDA compiler during strict jenkins builds.")
set(CMAKE_CUDA_FLAGS_RELEASE
    "-O3 -Xcompiler \"${CXX_OPT}\""
    CACHE STRING "Flags used by the CUDA compiler during release builds.")
set(CMAKE_CUDA_FLAGS_HOSTDEBUG
    "-g"
    CACHE STRING "Flags used by the CUDA compiler during host-debug builds.")
set(CMAKE_CUDA_FLAGS_DEBUG
    "-G -g -fno-inline"
    CACHE STRING "Flags used by the CUDA compiler during full (host+device) debug builds.")
set(CMAKE_CUDA_FLAGS_SANITIZE
    "-g -fno-inline \"-fsanitize=address,undefined\" "
    CACHE STRING "Flags used by the CUDA compiler during sanitizer debug builds.")

mark_as_advanced(CMAKE_CUDA_FLAGS_DEVEL)
mark_as_advanced(CMAKE_CUDA_FLAGS_STRICT)
mark_as_advanced(CMAKE_CUDA_FLAGS_RELEASE)
mark_as_advanced(CMAKE_CUDA_FLAGS_DEBUG)
mark_as_advanced(CMAKE_CUDA_FLAGS_HOSTDEBUG)
mark_as_advanced(CMAKE_CUDA_FLAGS_SANITIZE)
enable_language(CUDA)
message(STATUS "CUDA Compiler is" ${CMAKE_CUDA_COMPILER})
message(STATUS "Compiler ID is " ${CMAKE_CUDA_COMPILER_ID})
# TODO: Do we stil use that?
if(${CMAKE_CUDA_COMPILER} MATCHES "nvcc")
  set(GGP_CUDA_BUILD_TYPE "NVCC")
  message(STATUS "CUDA Build Type: ${GGP_CUDA_BUILD_TYPE}")
elseif(${CMAKE_CUDA_COMPILER} MATCHES "clang")
  set(GGP_CUDA_BUILD_TYPE "Clang")
  message(STATUS "CUDA Build Type: ${GGP_CUDA_BUILD_TYPE}")
elseif(${CMAKE_CUDA_COMPILER_ID} MATCHES "NVHPC")
  set(GGP_CUDA_BUILD_TYPE "NVHPC")
  message(STATUS "CUDA Build Type: ${GGP_CUDA_BUILD_TYPE}")
endif()

# ######################################################################################################################
# CUDA specific GGP options
include(CMakeDependentOption)

# large arg support requires CUDA 12.1
cmake_dependent_option(GGP_LARGE_KERNEL_ARG "enable large kernel arg support" ON "${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER_EQUAL 12.1" OFF )
message(STATUS "Large kernel arguments supported: ${GGP_LARGE_KERNEL_ARG}")
mark_as_advanced(GGP_LARGE_KERNEL_ARG)

# Set the maximum multi-RHS per kernel
if(GGP_LARGE_KERNEL_ARG)
  set(GGP_MAX_MULTI_RHS "64" CACHE STRING "maximum number of simultaneous RHS in a kernel")
else()
  set(GGP_MAX_MULTI_RHS "16" CACHE STRING "maximum number of simultaneous RHS in a kernel")
endif()
message(STATUS "Max number of rhs per kernel: ${GGP_MAX_MULTI_RHS}")

option(GGP_VERBOSE_BUILD "display kernel register usage" OFF)
option(GGP_JITIFY "build GGP using Jitify" OFF)
option(GGP_DOWNLOAD_NVSHMEM "Download NVSHMEM" OFF)
set(GGP_NVSHMEM
    OFF
    CACHE BOOL "set to 'yes' to build the NVSHMEM multi-GPU code")
set(GGP_NVSHMEM_HOME
    $ENV{NVSHMEM_HOME}
    CACHE PATH "path to NVSHMEM")
set(GGP_GDRCOPY_HOME
    "/usr/local/gdrcopy"
    CACHE STRING "path to gdrcopy used when GGP_DOWNLOAD_NVSHMEM is enabled")
# NVTX options
option(GGP_INTERFACE_NVTX "add NVTX markup to interface calls" OFF)

if(CMAKE_CUDA_COMPILER_ID MATCHES "NVIDIA" OR CMAKE_CUDA_COMPILER_ID MATCHES "NVHPC")
  set(GGP_HETEROGENEOUS_ATOMIC_SUPPORT ON)
  message(STATUS "Heterogeneous atomics supported: ${GGP_HETEROGENEOUS_ATOMIC_SUPPORT}")
endif()
cmake_dependent_option(GGP_HETEROGENEOUS_ATOMIC "enable heterogeneous atomic support ?" ON
                       "GGP_HETEROGENEOUS_ATOMIC_SUPPORT" OFF)

mark_as_advanced(GGP_HETEROGENEOUS_ATOMIC)
mark_as_advanced(GGP_JITIFY)
mark_as_advanced(GGP_DOWNLOAD_NVSHMEM)
mark_as_advanced(GGP_DOWNLOAD_NVSHMEM_TAR)
mark_as_advanced(GGP_GDRCOPY_HOME)
mark_as_advanced(GGP_VERBOSE_BUILD)
mark_as_advanced(GGP_INTERFACE_NVTX)

# ######################################################################################################################
# CUDA specific variables
# Extract the full arch token (e.g. "100a") for CMAKE_CUDA_ARCHITECTURES,
# and a purely numeric version (e.g. "100") for the __COMPUTE_CAPABILITY__ define.
string(REGEX REPLACE "sm_" "" GGP_CUDA_ARCH_TOKEN ${GGP_GPU_ARCH})
string(REGEX REPLACE "[^0-9]" "" GGP_COMPUTE_CAPABILITY ${GGP_CUDA_ARCH_TOKEN})
if(${CMAKE_BUILD_TYPE} STREQUAL "RELEASE")
  set(GGP_GPU_ARCH_SUFFIX real)
endif()
if(GGP_GPU_ARCH_SUFFIX)
  set(CMAKE_CUDA_ARCHITECTURES "${GGP_CUDA_ARCH_TOKEN}-${GGP_GPU_ARCH_SUFFIX}")
else()
  set(CMAKE_CUDA_ARCHITECTURES ${GGP_CUDA_ARCH_TOKEN})
endif()

set_target_properties(ggp PROPERTIES CUDA_ARCHITECTURES ${CMAKE_CUDA_ARCHITECTURES})

# GGP_HASH for tunecache
set(HASH cpu_arch=${CPU_ARCH},gpu_arch=${GGP_GPU_ARCH},cuda_version=${CMAKE_CUDA_COMPILER_VERSION})
set(GITVERSION "${PROJECT_VERSION}-${GITVERSION}-${GGP_GPU_ARCH}")

# ######################################################################################################################
# cuda specific compile options
target_compile_options(
  ggp
  PRIVATE $<$<COMPILE_LANG_AND_ID:CUDA,NVIDIA>:
          -ftz=true
          -prec-div=false
          -prec-sqrt=false>
          $<$<COMPILE_LANG_AND_ID:CUDA,NVHPC>:
          -Mflushz
          -Mfpapprox=div
          -Mfpapprox=sqrt>
          $<$<COMPILE_LANG_AND_ID:CUDA,Clang>:
          -fcuda-flush-denormals-to-zero
          -fcuda-approx-transcendentals
          -Xclang
          -fcuda-allow-variadic-functions>)
target_compile_options(
  ggp PRIVATE $<$<COMPILE_LANG_AND_ID:CUDA,Clang>:-Wno-unknown-cuda-version> $<$<COMPILE_LANG_AND_ID:CUDA,NVIDIA>:
               -Wno-deprecated-gpu-targets --expt-relaxed-constexpr>)

target_compile_options(ggp PRIVATE $<$<COMPILE_LANG_AND_ID:CUDA,NVIDIA>: -ftz=true -prec-div=false -prec-sqrt=false>)
target_compile_options(ggp PRIVATE $<$<COMPILE_LANG_AND_ID:CUDA,NVIDIA>: -Wno-deprecated-gpu-targets
                                    --expt-relaxed-constexpr>)
target_compile_options(ggp PRIVATE $<$<COMPILE_LANG_AND_ID:CUDA,Clang>: --cuda-path=${CUDAToolkit_TARGET_DIR}>)
target_link_options(ggp PUBLIC $<$<CUDA_COMPILER_ID:Clang>: --cuda-path=${CUDAToolkit_TARGET_DIR}>)

if(GGP_VERBOSE_BUILD)
  target_compile_options(ggp PRIVATE $<$<COMPILE_LANG_AND_ID:CUDA,NVIDIA>:--ptxas-options=-v>)
endif(GGP_VERBOSE_BUILD)

if(${CMAKE_CUDA_COMPILER_ID} MATCHES "NVHPC" AND NOT ${CMAKE_BUILD_TYPE} MATCHES "DEBUG")
  target_compile_options(ggp PRIVATE "$<$<COMPILE_LANG_AND_ID:CUDA,NVHPC>:SHELL: -gpu=nodebug" >)
endif()

if((${GGP_TARGET_TYPE} STREQUAL "CUDA") AND (${GGP_COMPUTE_CAPABILITY} GREATER_EQUAL 70))
  set(MRHS_MMA_ENABLED 1)
else()
  set(MRHS_MMA_ENABLED 0)
endif()

if(${MRHS_MMA_ENABLED})
  set(GGP_MULTIGRID_MRHS_DEFAULT_LIST "16")
else()
  set(GGP_MULTIGRID_MRHS_DEFAULT_LIST "")
endif()
set(GGP_MULTIGRID_MRHS_LIST ${GGP_MULTIGRID_MRHS_DEFAULT_LIST} CACHE STRING "The list of multi-rhs sizes that get compiled")
mark_as_advanced(GGP_MULTIGRID_MRHS_LIST)
message(STATUS "GGP_MULTIGRID_MRHS_LIST=${GGP_MULTIGRID_MRHS_LIST}")

if(GGP_MULTIGRID)
  option(GGP_ENABLE_MMA "Enabling using tensor core" ON)
  mark_as_advanced(GGP_ENABLE_MMA)

  option(GGP_MULTIGRID_SETUP_USE_SMMA "Enabling using SMMA (3xTF32/3xBF16/3xFP16) for multigrid setup" ON)
  mark_as_advanced(GGP_MULTIGRID_SETUP_USE_SMMA)

  string(REPLACE "," ";" GGP_MULTIGRID_MRHS_LIST_SEMICOLON "${GGP_MULTIGRID_MRHS_LIST}")

  if(${GGP_COMPUTE_CAPABILITY} GREATER_EQUAL 70)

    if(${GGP_COMPUTE_CAPABILITY} EQUAL 70)
      set(MRHS_ATOM 16)
    else()
      set(MRHS_ATOM 8)
    endif()

    # add dslash_coarse last to the list so it is compiled first
    foreach(GGP_MULTIGRID_NVEC ${GGP_MULTIGRID_NVEC_LIST_SEMICOLON})
      foreach(GGP_MULTIGRID_MRHS ${GGP_MULTIGRID_MRHS_LIST_SEMICOLON})

        math(EXPR MRHS_MODULO "${GGP_MULTIGRID_MRHS} % ${MRHS_ATOM}")

        if((${GGP_MULTIGRID_MRHS} GREATER 0) AND (${GGP_MULTIGRID_MRHS} LESS_EQUAL 64) AND (${MRHS_MODULO} EQUAL 0))
          set(GGP_MULTIGRID_DAGGER "false")
          configure_file(dslash_coarse_mma.in.cu "dslash_coarse_mma_${GGP_MULTIGRID_NVEC}_${GGP_MULTIGRID_MRHS}.cu" @ONLY)
          list(PREPEND GGP_CU_OBJS "dslash_coarse_mma_${GGP_MULTIGRID_NVEC}_${GGP_MULTIGRID_MRHS}.cu")
          set(GGP_MULTIGRID_DAGGER "true")
          configure_file(dslash_coarse_mma.in.cu "dslash_coarse_mma_dagger_${GGP_MULTIGRID_NVEC}_${GGP_MULTIGRID_MRHS}.cu" @ONLY)
          list(PREPEND GGP_CU_OBJS "dslash_coarse_mma_dagger_${GGP_MULTIGRID_NVEC}_${GGP_MULTIGRID_MRHS}.cu")
        else()
          message(SEND_ERROR "MRHS not supported:" "${GGP_MULTIGRID_MRHS}")
        endif()

      endforeach()
    endforeach()
  endif()

endif()

set(GGP_MAX_SHARED_MEMORY "0" CACHE STRING "Max shared memory per block, 0 corresponds to architecture default")
mark_as_advanced(GGP_MAX_SHARED_MEMORY)
configure_file(${CMAKE_SOURCE_DIR}/include/targets/cuda/device.in.hpp
               ${CMAKE_BINARY_DIR}/include/targets/cuda/device.hpp @ONLY)
install(FILES "${CMAKE_BINARY_DIR}/include/targets/cuda/device.hpp" DESTINATION include/)

target_include_directories(ggp SYSTEM PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:${CUDAToolkit_INCLUDE_DIRS}>)
target_include_directories(ggp SYSTEM PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:${CUDAToolkit_MATH_INCLUDE_DIR}>)
target_include_directories(ggp_cpp SYSTEM PUBLIC ${CUDAToolkit_INCLUDE_DIRS} ${CUDAToolkit_MATH_INCLUDE_DIR})

target_compile_options(ggp PRIVATE $<$<COMPILE_LANG_AND_ID:CUDA,Clang>:--cuda-path=${CUDAToolkit_TARGET_DIR}>)
target_compile_options(ggp PRIVATE $<$<COMPILE_LANG_AND_ID:CUDA,NVIDIA>:-Xfatbin=-compress-all>)
target_include_directories(ggp PRIVATE ${CMAKE_SOURCE_DIR}/include/targets/cuda)
target_include_directories(ggp PUBLIC $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/include/targets/cuda>
                                       $<INSTALL_INTERFACE:include/targets/cuda>)
target_include_directories(ggp SYSTEM PRIVATE ${CMAKE_SOURCE_DIR}/include/targets/cuda/externals)
target_include_directories(ggp_cpp SYSTEM PRIVATE ${CMAKE_SOURCE_DIR}/include/targets/cuda/externals)

# Specific config dependent warning suppressions and lineinfo forwarding

target_compile_options(
  ggp 
  PRIVATE $<$<COMPILE_LANGUAGE:CUDA>: 
          $<IF:$<CONFIG:RELEASE>,-w,-Wall -Wextra>
          >)

target_compile_options(
  ggp
  PRIVATE $<$<COMPILE_LANG_AND_ID:CUDA,NVIDIA>:
          -Wreorder
          $<$<CXX_COMPILER_ID:Clang>:
          -Xcompiler=-Wno-unused-function
          -Xcompiler=-Wno-unknown-pragmas
          -Xcompiler=-mllvm\ -unroll-count=4
          >
          $<$<CXX_COMPILER_ID:GNU>:
          -Xcompiler=-Wno-unknown-pragmas>
          $<$<CONFIG:DEVEL>:-Xptxas
          -warn-lmem-usage,-warn-spills
          -lineinfo>
          $<$<CONFIG:STRICT>:
          -Werror=all-warnings
          -lineinfo>
          $<$<CONFIG:HOSTDEBUG>:-lineinfo>
          $<$<CONFIG:SANITIZE>:-lineinfo>
          >)

# older gcc throws false warnings so disable these
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU")
  if (CMAKE_CXX_COMPILER_VERSION VERSION_LESS 11.0)
    target_compile_options(ggp PUBLIC $<$<COMPILE_LANGUAGE:CUDA,NVIDIA>: -Wno-unused-but-set-parameter>)
  endif()

  if (CMAKE_CXX_COMPILER_VERSION VERSION_LESS 10.0)
    target_compile_options(ggp PUBLIC $<$<COMPILE_LANGUAGE:CUDA,NVIDIA>: -Wno-unused-but-set-variable>)
  endif()
endif()

# older nvcc throws false warnings with respect to constexpr if code removal
if(${CMAKE_CUDA_COMPILER_VERSION} VERSION_LESS "11.3")
  target_compile_options(
    ggp
    PRIVATE $<$<COMPILE_LANG_AND_ID:CUDA,NVIDIA>:
            "SHELL:-Xcudafe --diag_suppress=607" >)
endif()

if(${CMAKE_CUDA_COMPILER_VERSION} VERSION_LESS "11.5")
  target_compile_options(
    ggp
    PRIVATE $<$<COMPILE_LANG_AND_ID:CUDA,NVIDIA>:
            "SHELL: -Xcudafe --diag_suppress=177" >)
endif()

target_compile_options(
  ggp 
  PRIVATE $<$<COMPILE_LANG_AND_ID:CUDA,NVHPC>:
          -gpu=lineinfo
          $<$<CONFIG:STRICT>:-Werror>
          >)

target_compile_options(
  ggp
  PRIVATE $<$<COMPILE_LANG_AND_ID:CUDA,Clang>:
          -Wall
          -Wextra
          -Wno-unknown-pragmas
          $<$<CONFIG:STRICT>:-Werror
          -Wno-error=pass-failed>
          $<$<CONFIG:SANITIZE>:-fsanitize=address
          -fsanitize=undefined>
          >)

if(GGP_OPENMP)
  target_compile_options(
    ggp
    PRIVATE $<$<COMPILE_LANG_AND_ID:CUDA,NVIDIA>:
    "-Xcompiler=${OpenMP_CXX_FLAGS}"
    >)
endif()

# malloc.cpp uses both the driver and runtime api So we need to find the CUDA_CUDA_LIBRARY (driver api) or the stub
target_link_libraries(ggp PUBLIC CUDA::cuda_driver)
target_link_libraries(ggp PUBLIC CUDA::nvml)
if(CUDAToolkit_FOUND)
  target_link_libraries(ggp INTERFACE CUDA::cudart_static)
endif()

# nvshmem enabled parts need SEPARABLE_COMPILATION ...
if(GGP_NVSHMEM)
  list(APPEND GGP_DSLASH_OBJS dslash_constant_arg.cu)
  add_library(ggp_pack OBJECT ${GGP_DSLASH_OBJS})
  # ####################################################################################################################
  # NVSHMEM Download
  # ####################################################################################################################
  if(GGP_DOWNLOAD_NVSHMEM)
    # workaround potential UCX interaction issue with CUDA 11.3+ and UCX in NVSHMEM 2.1.2
    if(${CMAKE_CUDA_COMPILER_VERSION} VERSION_LESS "11.3")
      set(GGP_DOWNLOAD_NVSHMEM_TAR
          "https://developer.download.nvidia.com/compute/redist/nvshmem/2.1.2/source/nvshmem_src_2.1.2-0.txz"
          CACHE STRING "location of NVSHMEM tarball")
    else()
      set(GGP_DOWNLOAD_NVSHMEM_TAR
          "https://developer.download.nvidia.com/compute/redist/nvshmem/2.2.1/source/nvshmem_src_2.2.1-0.txz"
          CACHE STRING "location of NVSHMEM tarball")
    endif()
    get_filename_component(NVSHMEM_CUDA_HOME ${CUDAToolkit_INCLUDE_DIRS} DIRECTORY)
    find_path(
      GDRCOPY_HOME NAME gdrapi.h
      PATHS "/usr/local/gdrcopy" ${GGP_GDRCOPY_HOME}
      PATH_SUFFIXES "include")
    mark_as_advanced(GDRCOPY_HOME)
    if(NOT GDRCOPY_HOME)
      message(
        SEND_ERROR
          "GGP_DOWNLOAD_NVSHMEM requires gdrcopy to be installed. Please set GGP_GDRCOPY_HOME to the location of your gdrcopy installation."
      )
    endif()
    get_filename_component(NVSHMEM_GDRCOPY_HOME ${GDRCOPY_HOME} DIRECTORY)
    ExternalProject_Add(
      NVSHMEM
      URL ${GGP_DOWNLOAD_NVSHMEM_TAR}
      PREFIX nvshmem
      CONFIGURE_COMMAND ""
      BUILD_IN_SOURCE ON
      BUILD_COMMAND make -j8 MPICC=${MPI_C_COMPILER} CUDA_HOME=${NVSHMEM_CUDA_HOME} NVSHMEM_PREFIX=<INSTALL_DIR>
                    NVSHMEM_MPI_SUPPORT=1 GDRCOPY_HOME=${NVSHMEM_GDRCOPY_HOME} install
      INSTALL_COMMAND ""
      LOG_INSTALL ON
      LOG_BUILD ON
      LOG_DOWNLOAD ON)
    ExternalProject_Get_Property(NVSHMEM INSTALL_DIR)
    set(GGP_NVSHMEM_HOME
        ${INSTALL_DIR}
        CACHE PATH "path to NVSHMEM" FORCE)
    set(NVSHMEM_LIBS ${INSTALL_DIR}/lib/libnvshmem.a)
    set(NVSHMEM_INCLUDE ${INSTALL_DIR}/include/)
  else()
    if("${GGP_NVSHMEM_HOME}" STREQUAL "")
      message(FATAL_ERROR "GGP_NVSHMEM_HOME must be defined if GGP_NVSHMEM is set")
    endif()
    find_library(
      NVSHMEM_LIBS
      NAMES nvshmem
      PATHS "${GGP_NVSHMEM_HOME}/lib/")
    find_path(
      NVSHMEM_INCLUDE
      NAMES nvshmem.h
      PATHS "${GGP_NVSHMEM_HOME}/include/")
  endif()

  mark_as_advanced(NVSHMEM_LIBS)
  mark_as_advanced(NVSHMEM_INCLUDE)
  add_library(nvshmem_lib STATIC IMPORTED)
  set_target_properties(nvshmem_lib PROPERTIES IMPORTED_LOCATION ${NVSHMEM_LIBS})
  set_target_properties(nvshmem_lib PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
  set_target_properties(nvshmem_lib PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS OFF)
  set_target_properties(nvshmem_lib PROPERTIES IMPORTED_LINK_INTERFACE_LANGUAGES CUDA)

  # set_target_properties(ggp_pack PROPERTIES CUDA_ARCHITECTURES ${GGP_COMPUTE_CAPABILITY})
  target_include_directories(ggp_pack PRIVATE dslash_core)
  target_include_directories(ggp_pack SYSTEM PRIVATE ../include/externals)
  target_include_directories(ggp_pack PRIVATE .)
  set_target_properties(ggp_pack PROPERTIES POSITION_INDEPENDENT_CODE ${GGP_BUILD_SHAREDLIB})
  target_compile_definitions(ggp_pack PRIVATE $<TARGET_PROPERTY:ggp,COMPILE_DEFINITIONS>)
  target_include_directories(ggp_pack PRIVATE $<TARGET_PROPERTY:ggp,INCLUDE_DIRECTORIES>)
  target_compile_options(ggp_pack PRIVATE $<TARGET_PROPERTY:ggp,COMPILE_OPTIONS>)
  if((${GGP_COMPUTE_CAPABILITY} LESS "70"))
    message(SEND_ERROR "GGP_NVSHMEM=ON requires at least GGP_GPU_ARCH=sm_70")
  endif()
  set_target_properties(ggp_pack PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
  set_property(TARGET ggp PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)
  target_link_libraries(ggp PUBLIC MPI::MPI_C)
  target_compile_definitions(ggp PUBLIC NVSHMEM_COMMS)
  if(GGP_DOWNLOAD_NVSHMEM)
    add_dependencies(ggp NVSHMEM)
    add_dependencies(ggp_cpp NVSHMEM)
    add_dependencies(ggp_pack NVSHMEM)
  endif()
  get_filename_component(NVSHMEM_LIBPATH ${NVSHMEM_LIBS} DIRECTORY)
  target_link_libraries(ggp PUBLIC -L${NVSHMEM_LIBPATH} -lnvshmem)
  target_include_directories(ggp SYSTEM PUBLIC $<BUILD_INTERFACE:${NVSHMEM_INCLUDE}>)
endif()

if(${GGP_BUILD_NATIVE_LAPACK} STREQUAL "ON")
  target_link_libraries(ggp PUBLIC ${CUDA_cublas_LIBRARY})
endif()

target_link_libraries(ggp PUBLIC ${CUDA_cufft_LIBRARY})

if(GGP_JITIFY)
  target_compile_definitions(ggp PRIVATE JITIFY)
  find_package(LibDL)
  target_link_libraries(ggp PUBLIC ${CUDA_nvrtc_LIBRARY})
  target_link_libraries(ggp PUBLIC ${LIBDL_LIBRARIES})
  target_include_directories(ggp PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/include)

  configure_file(${CMAKE_SOURCE_DIR}/include/targets/cuda/jitify_options.hpp.in
                 ${CMAKE_BINARY_DIR}/include/targets/cuda/jitify_options.hpp)
  install(FILES "${CMAKE_BINARY_DIR}/include/targets/cuda/jitify_options.hpp" DESTINATION include/)
endif()

if(GGP_INTERFACE_NVTX)
  target_compile_definitions(ggp PRIVATE INTERFACE_NVTX)
  target_link_libraries(ggp PRIVATE CUDA::nvtx3)
endif(GGP_INTERFACE_NVTX)

add_subdirectory(targets/cuda)

install(FILES ${CMAKE_SOURCE_DIR}/cmake/find_target_cuda_dependencies.cmake DESTINATION lib/cmake/GGP)
