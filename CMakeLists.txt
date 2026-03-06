# This file is divided into 5 sections:
# 1. Basic setup for cmake 
# 2. GGP Science options 
# 3. GGP Dependency options
# 4. GGP Advanced options
# 5. GGP CMake configuration set up
#
# Each section is delineated by START and END. If you make additions to this file,
# please add an appropriate comment and, if necessary, document the addition
# in the GGP Wiki page: https://github.com/cpviolator/genertic-GPU-project/wiki/GGP-Build-With-CMake

####################################################################################
# START 1. Basic setup for cmake
####################################################################################
# basic setup for cmake
cmake_minimum_required(VERSION 3.24 FATAL_ERROR)

if(POLICY CMP0074)
  cmake_policy(SET CMP0074 NEW)
endif()

set(CMAKE_INCLUDE_CURRENT_DIR ON)
set(CMAKE_INCLUDE_DIRECTORIES_PROJECT_BEFORE ON)
set(CMAKE_COLOR_MAKEFILE ON)
set(CMAKE_CXX_STANDARD_REQUIRED True)
# Disable gnu exentions
set(CMAKE_CXX_EXTENSIONS OFF)

# Disable in source builds this is only a temporary fix, but for now we need it as
# cmake will otherwise overwrite the existing makefiles.
set(CMAKE_DISABLE_SOURCE_CHANGES ON)
set(CMAKE_DISABLE_IN_SOURCE_BUILD ON)
# add a directory for cmake modules
list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")
include(cmake/CPM.cmake)

find_package(Git)

# By default we will build DEVEL. The different build types will pass different
# flags to the compiler which may be strict or permissive on warnings, or
# very verbose at run time and/or compile time.
if(DEFINED ENV{GGP_BUILD_TYPE})
  set(DEFBUILD $ENV{GGP_BUILD_TYPE})
else()
  set(DEFBUILD "DEVEL")
endif()

set(VALID_BUILD_TYPES
    DEVEL
    RELEASE
    STRICT
    DEBUG
    HOSTDEBUG
    SANITIZE)
set(CMAKE_BUILD_TYPE
  "${DEFBUILD}"
  CACHE STRING "Choose the type of build, options are: ${VALID_BUILD_TYPES}")
set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS ${VALID_BUILD_TYPES})

string(TOUPPER ${CMAKE_BUILD_TYPE} CHECK_BUILD_TYPE)
list(FIND VALID_BUILD_TYPES ${CHECK_BUILD_TYPE} BUILD_TYPE_VALID)

if(BUILD_TYPE_VALID LESS 0)
  message(SEND_ERROR "Please specify a valid CMAKE_BUILD_TYPE type! Valid build types are:" "${VALID_BUILD_TYPES}")
endif()

# GGP may be built to run using CUDA, HIP or SYCL, which we call the
# Target type. By default, the target is CUDA.
if(DEFINED ENV{GGP_TARGET})
  set(DEFTARGET $ENV{GGP_TARGET})
else()
  set(DEFTARGET "CUDA")
endif()

set(VALID_TARGET_TYPES CUDA HIP SYCL)
set(GGP_TARGET_TYPE
  "${DEFTARGET}"
  CACHE STRING "Choose the type of target, options are: ${VALID_TARGET_TYPES}")
set_property(CACHE GGP_TARGET_TYPE PROPERTY STRINGS CUDA HIP SYCL)

string(TOUPPER ${GGP_TARGET_TYPE} CHECK_TARGET_TYPE)
list(FIND VALID_TARGET_TYPES ${CHECK_TARGET_TYPE} TARGET_TYPE_VALID)

if(TARGET_TYPE_VALID LESS 0)
  message(SEND_ERROR "Please specify a valid GGP_TARGET_TYPE type! Valid target types are:" "${VALID_TARGET_TYPES}")
endif()

if(GIT_FOUND)
  execute_process(
    COMMAND ${GIT_EXECUTABLE} show
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    RESULT_VARIABLE IS_GIT_REPOSIITORY
    OUTPUT_QUIET ERROR_QUIET)
  if(${IS_GIT_REPOSIITORY} EQUAL 0)
    execute_process(
      COMMAND ${GIT_EXECUTABLE} describe --abbrev=0
      WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
      OUTPUT_VARIABLE GITTAG
      OUTPUT_STRIP_TRAILING_WHITESPACE)
    # we use git rev-list and pipe that through wc here. Newer git versions support --count as option to rev-list but
    # that might not always be available
    execute_process(
      COMMAND ${GIT_EXECUTABLE} rev-list ${GITTAG}..HEAD
      WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
      COMMAND wc -l
      OUTPUT_VARIABLE GITCOUNT
      OUTPUT_STRIP_TRAILING_WHITESPACE)
    execute_process(
      COMMAND ${GIT_EXECUTABLE} describe --match 1 --always  --long --dirty
      WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
      OUTPUT_VARIABLE GITVERSION
      OUTPUT_STRIP_TRAILING_WHITESPACE)
  endif()
endif(GIT_FOUND)

# Define the project
project(
  "GGP"
  VERSION 1.1.0
  LANGUAGES)

# Include the FetchContent module
include(FetchContent)

# Print the configuration details to stdout
message(STATUS "")
message(STATUS "${PROJECT_NAME} ${PROJECT_VERSION} (${GITVERSION}) **")
message(STATUS "cmake version: ${CMAKE_VERSION}")
message(STATUS "Source location: ${CMAKE_SOURCE_DIR}")
message(STATUS "Build location: ${CMAKE_BINARY_DIR}")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "GGP target: ${GGP_TARGET_TYPE}")

# ######################################################################################################################
# GGP OPTIONS likely to be changed by users
# ######################################################################################################################

# Multi-GPU options
option(GGP_MPI "build the MPI multi-GPU code" OFF)

# CUTLASS Interface — pre-built cutlass_gemm_api (auto-enabled when PATH is set)
set(GGP_CUTLASS_INTERFACE_PATH "" CACHE PATH
    "Build directory containing libcutlass_gemm_api.a")
set(GGP_CUTLASS_INTERFACE_INCLUDE "" CACHE PATH
    "Source directory containing cutlass_gemm_api.h")

# TCC (Tensor-Core Correlator)
option(GGP_TCC "Enable TCC correlator engine" OFF)
set(GGP_TCC_PATH "" CACHE PATH
    "Path to TCC source tree (e.g. /path/to/tensor-core-correlator)")

# Dedisp TCFDD — pre-built dedisp_api (auto-enabled when PATH is set)
set(GGP_DEDISP_TCFDD_PATH "" CACHE PATH
    "Build directory containing libdedisp_api.a")
set(GGP_DEDISP_TCFDD_INCLUDE "" CACHE PATH
    "Source directory containing dedisp_api.h")

# Build bench_pipeline (production pipeline benchmark with per-stage timing)
# Requires CUTLASS interface + dedisp_tcfdd (auto-enabled when both are available)
option(GGP_BUILD_BENCH_PIPELINE "Build bench_pipeline benchmark executable" ON)

# Build cutlass_interface + dedisp_tcfdd from source as subdirectories
# (instead of linking against pre-built .a files)
option(GGP_BUILD_FROM_SOURCE "Build cutlass_interface + dedisp_tcfdd from source" OFF)
set(GGP_CUTLASS_DIR "" CACHE PATH
    "Path to CUTLASS source tree (required when GGP_BUILD_FROM_SOURCE=ON)")

# ######################################################################################################################
# GGP ADVANCED OPTIONS that usually should not be changed by users
# ######################################################################################################################
option(GGP_BUILD_ALL_TESTS "build tests by default" ON)
option(GGP_INSTALL_ALL_TESTS "install tests by default" ON)
option(GGP_BUILD_SHAREDLIB "build GGP as a shared lib" ON)

set(GGP_ORDER_FP "8" CACHE STRING "which data order to use for Wilson, gauge (recon-8/9) and clover fixed-point fields (4, 8)")
set(GGP_ORDER_SP_MG "2" CACHE STRING "which data order to use for fp32 MG fields (2, 4)")
set(GGP_ORDER_FP_MG "2" CACHE STRING "which data order to use for fixed-point MG fields (2, 4, 8)")

set_property(CACHE GGP_ORDER_FP PROPERTY STRINGS 4 8)
set_property(CACHE GGP_ORDER_SP_MG PROPERTY STRINGS 2 4)
set_property(CACHE GGP_ORDER_FP_MG PROPERTY STRINGS 2 4 8)

option(GGP_BUILD_NATIVE_LAPACK "build the native blas/lapack library according to GGP_TARGET" ON)
option(GGP_BUILD_NATIVE_FFT "build the native FFT library according to GGP_TARGET" ON)

set(GGP_PRECISION
  "14"
  CACHE STRING "which precisions to instantiate in GGP (4-bit number - double, single, half, quarter)")
set(GGP_RECONSTRUCT
  "7"
  CACHE STRING "which reconstructs to instantiate in GGP (3-bit number - 18, 13/12, 9/8)")

option(GGP_ALTERNATIVE_I_TO_F "enable using alternative integer-to-float conversion" OFF)

option(GGP_OPENMP "enable OpenMP" OFF)
set(GGP_CXX_STANDARD 17 CACHE STRING "set the CXX Standard (14 or 17)")
set_property(CACHE GGP_CXX_STANDARD PROPERTY STRINGS 14 17)

option(GGP_BACKWARDS "Enable stacktrace generation using backwards-cpp")

option(GGP_USE_EIGEN "use EIGEN library (where optional)" ON)
option(GGP_DOWNLOAD_EIGEN "Download Eigen" ON)


option(GGP_GENERATE_DOXYGEN "generate doxygen documentation")

# mark as advanced
mark_as_advanced(GGP_BUILD_ALL_TESTS)
mark_as_advanced(GGP_INSTALL_ALL_TESTS)

mark_as_advanced(GGP_ORDER_FP)
mark_as_advanced(GGP_ORDER_SP_MG)
mark_as_advanced(GGP_ORDER_FP_MG)
mark_as_advanced(GGP_FAST_COMPILE_REDUCE)
mark_as_advanced(GGP_FAST_COMPILE_DSLASH)

mark_as_advanced(GGP_ALTERNATIVE_I_TO_F)

mark_as_advanced(GGP_MAX_MULTI_RHS)
mark_as_advanced(GGP_PRECISION)
mark_as_advanced(GGP_BACKWARDS)

mark_as_advanced(GGP_USE_EIGEN)
mark_as_advanced(GGP_CXX_STANDARD)

####################################################################################
# END 4. GGP advanced options that usually should not be changed by users
####################################################################################

####################################################################################
# START 5. GGP CMake configuration set up
####################################################################################
# All the CMake code from here and below is not exposed to the user. Its purpose
# is to configure the build given the options above

if(GGP_NVSHMEM AND NOT GGP_MPI)
  message(SEND_ERROR "Specifying GGP_NVSHMEM requires GGP_MPI.")
endif()

# COMPILER FLAGS Linux: CMAKE_HOST_SYSTEM_PROCESSOR "x86_64" Mac: CMAKE_HOST_SYSTEM_PROCESSOR "x86_64" Power:
# CMAKE_HOST_SYSTEM_PROCESSOR "ppc64le"

# We need to use different optimization flags depending on whether we are on x86 or
# power Note: This only applies to the RELEASE build type this is just a quick fix
# and we should probably use
# https://cmake.org/cmake/help/latest/module/CheckCXXCompilerFlag.html

set(CPU_ARCH ${CMAKE_HOST_SYSTEM_PROCESSOR})
if(${CPU_ARCH} STREQUAL "x86_64")
  set(CXX_OPT "-mtune=native")
elseif(${CPU_ARCH} STREQUAL "ppc64le")
  set(CXX_OPT "-Ofast -mcpu=native -mtune=native")
endif()

set(CMAKE_CXX_STANDARD ${GGP_CXX_STANDARD})

set(CMAKE_CXX_FLAGS_DEVEL
  "-g -O3"
  CACHE STRING "Flags used by the C++ compiler during regular development builds.")
set(CMAKE_CXX_FLAGS_STRICT
  "-Os"
  CACHE STRING "Flags used by the C++ compiler during strict jenkins builds.")
set(CMAKE_CXX_FLAGS_RELEASE
  "-O3 ${CXX_OPT}"
  CACHE STRING "Flags used by the C++ compiler during release builds.")
set(CMAKE_CXX_FLAGS_HOSTDEBUG
    "-g"
    CACHE STRING "Flags used by the C++ compiler during host-debug builds.")
set(CMAKE_CXX_FLAGS_DEBUG
    "-g -fno-inline"
    CACHE STRING "Flags used by the C++ compiler during full (host+device) debug builds.")
set(CMAKE_CXX_FLAGS_SANITIZE
    "-g -fno-inline \"-fsanitize=address,undefined\""
    CACHE STRING "Flags used by the C++ compiler during sanitizer debug builds.")

enable_language(CXX)

# define C FLAGS
set(CMAKE_C_FLAGS_DEVEL
  "-g -O3"
  CACHE STRING "Flags used by the C compiler during regular development builds.")
set(CMAKE_C_FLAGS_STRICT
  "-Os"
  CACHE STRING "Flags used by the C compiler during strict jenkins builds.")
set(CMAKE_C_FLAGS_RELEASE
  "-O3"
  CACHE STRING "Flags used by the C compiler during release builds.")
set(CMAKE_C_FLAGS_HOSTDEBUG
    "-g"
    CACHE STRING "Flags used by the C compiler during host-debug builds.")
set(CMAKE_C_FLAGS_DEBUG
  "-g -fno-inline"
  CACHE STRING "Flags used by the C compiler during full (host+device) debug builds.")
set(CMAKE_C_FLAGS_SANITIZE
    "-g -fno-inline \"-fsanitize=address,undefined\""
    CACHE STRING "Flags used by the C compiler during sanitizer debug builds.")


enable_language(C)

# define LINKER FLAGS
set(CMAKE_EXE_LINKER_FLAGS_SANITIZE
    "-fsanitize=address,undefined"
    CACHE STRING "Flags used by the linker during sanitizer debug builds.")

find_package(Threads REQUIRED)
if(GGP_OPENMP)
  if(${CMAKE_CXX_COMPILER_ID} MATCHES "NVHPC")
    message(FATAL_ERROR "Host compiler (nvc++) not supported with GGP_OPENMP=ON")
  endif()
  find_package(OpenMP REQUIRED)
endif()

# ######################################################################################################################
# Handle dependencies
# ######################################################################################################################
if(GGP_BUILD_SHAREDLIB)
  set(BUILD_SHARED_LIBS ON)
endif()

# ######################################################################################################################
# Eigen
# ######################################################################################################################
add_library(Eigen INTERFACE IMPORTED)
# set(CPM_USE_LOCAL_PACKAGES TRUE)
if(GGP_DOWNLOAD_EIGEN)
  set(GGP_EIGEN_VERSION
      "3.4.0"
      CACHE STRING "Eigen use for GGP_DOWNLOAD_EIGEN")
  mark_as_advanced(GGP_EIGEN_VERSION)
  CPMAddPackage(
    NAME Eigen
    VERSION ${GGP_EIGEN_VERSION}
    URL https://gitlab.com/libeigen/eigen/-/archive/${GGP_EIGEN_VERSION}/eigen-${GGP_EIGEN_VERSION}.tar.bz2
    DOWNLOAD_ONLY YES
    SYSTEM YES)
  target_include_directories(Eigen SYSTEM INTERFACE ${Eigen_SOURCE_DIR})
  install(DIRECTORY ${Eigen_SOURCE_DIR}/Eigen TYPE INCLUDE)
else()
  # fall back to using find_package
  find_package(Eigen QUIET)
  if(NOT EIGEN_FOUND)
    message(
      FATAL_ERROR
      "GGP requires Eigen (http://eigen.tuxfamily.org). Please either set EIGEN_INCLUDE_DIRS to path to eigen3 include directory, e.g. /usr/local/include/eigen3 or set GGP_DOWNLOAD_EIGEN to ON to enable automatic download of the necessary components."
      )
  endif()
  target_include_directories(Eigen SYSTEM INTERFACE ${EIGEN_INCLUDE_DIRS})
endif()

# ######################################################################################################################
# MPI
# ####################################################################################################################

if(GGP_MPI)
  # if we are using MPI and no MPI_<LANG>_COMPILER was specified on the command line
  # check for MPICXX and MPICC environment variables
  if( NOT GGP_SPACK_BUILD )
    if((NOT MPI_CXX_COMPILER) AND DEFINED ENV{MPICXX})
      set(MPI_CXX_COMPILER $ENV{MPICXX})
      set(mpimessage True)
      message(STATUS "Found environment variable MPICXX. Using it for MPI detection: $ENV{MPICXX}")
    endif()
    if((NOT MPI_C_COMPILER) AND DEFINED ENV{MPICC})
      message(STATUS "Found environment variable MPICC. Using it for MPI detection: $ENV{MPICC}")
      set(MPI_C_COMPILER $ENV{MPICC})
      set(mpimessage True)
    endif()
    # I think we don't use that at all but
    if((NOT MPI_Fortran_COMPILER) AND DEFINED ENV{MPIFORT})
      message(STATUS "Found environment variable MPIFORT. Using it for MPI detection: $ENV{MPIFORT}")
      set(MPI_Fortran_COMPILER $ENV{MPIFORT})
      set(mpimessage True)
    endif()
    if(mpimessage)
      message(
        "Found MPIFORT/MPICC/MPICXX environment variables. If this is not what you want please use -DMPI_<LANG>_COMPILER and consult the cmake FindMPI documentation."
        )
    endif()
  endif()
  find_package(MPI REQUIRED)
endif()


if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
  set(CMAKE_INSTALL_PREFIX
      ${CMAKE_BINARY_DIR}
      CACHE PATH "..." FORCE)
endif()


# ######################################################################################################################
# rpath handling before adding dependencies
# ######################################################################################################################

set(CMAKE_SKIP_BUILD_RPATH FALSE)
set(CMAKE_BUILD_WITH_INSTALL_RPATH FALSE)
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
list(FIND CMAKE_PLATFORM_IMPLICIT_LINK_DIRECTORIES "${CMAKE_INSTALL_PREFIX}/lib" isSystemDir)
if("${isSystemDir}" STREQUAL "-1")
    set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")
endif("${isSystemDir}" STREQUAL "-1")


# ######################################################################################################################
# BACKWARDS
# ######################################################################################################################
if(GGP_BACKWARDS)
  include(FetchContent)
  FetchContent_Declare(
    backward-cpp
    GIT_REPOSITORY https://github.com/bombela/backward-cpp.git
    GIT_TAG v1.6
    GIT_SHALLOW ON)
  FetchContent_GetProperties(backward-cpp)
  if(NOT backward-cpp_POPULATED)
    FetchContent_Populate(backward-cpp)
  endif()
  include(${backward-cpp_SOURCE_DIR}/BackwardConfig.cmake)
endif()


# ######################################################################################################################
# ADVANCED setup 
# ######################################################################################################################

# this allows simplified running of clang-tidy
if(${CMAKE_BUILD_TYPE} STREQUAL "DEVEL")
  set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
endif()


# make the compiler flags an advanced option for all user defined build types (cmake defined build types are advanced by
# default )
mark_as_advanced(CMAKE_CXX_FLAGS_DEVEL)
mark_as_advanced(CMAKE_CXX_FLAGS_STRICT)
mark_as_advanced(CMAKE_CXX_FLAGS_RELEASE)
mark_as_advanced(CMAKE_CXX_FLAGS_DEBUG)
mark_as_advanced(CMAKE_CXX_FLAGS_HOSTDEBUG)
mark_as_advanced(CMAKE_CXX_FLAGS_SANITIZE)

mark_as_advanced(CMAKE_C_FLAGS_DEVEL)
mark_as_advanced(CMAKE_C_FLAGS_STRICT)
mark_as_advanced(CMAKE_C_FLAGS_RELEASE)
mark_as_advanced(CMAKE_C_FLAGS_DEBUG)
mark_as_advanced(CMAKE_C_FLAGS_HOSTDEBUG)
mark_as_advanced(CMAKE_C_FLAGS_SANITIZE)
mark_as_advanced(CMAKE_F_FLAGS)

mark_as_advanced(CMAKE_EXE_LINKER_FLAGS_SANITIZE)

# enable ctest
include(CTest)

# ======================================================================
# Backward compatibility: map old variable names to new ones
# ======================================================================
if(GGP_TCC AND GGP_TCC_DIR AND NOT GGP_TCC_PATH)
  set(GGP_TCC_PATH "${GGP_TCC_DIR}")
  message(STATUS "Compat: GGP_TCC_DIR mapped to GGP_TCC_PATH")
endif()

# ======================================================================
# Validate source-build options
# ======================================================================
if(GGP_BUILD_FROM_SOURCE)
  if(NOT GGP_CUTLASS_DIR)
    if(DEFINED ENV{CUTLASS_DIR})
      set(GGP_CUTLASS_DIR "$ENV{CUTLASS_DIR}" CACHE PATH "" FORCE)
      message(STATUS "GGP_CUTLASS_DIR from environment: ${GGP_CUTLASS_DIR}")
    else()
      message(FATAL_ERROR
        "GGP_BUILD_FROM_SOURCE=ON requires GGP_CUTLASS_DIR to be set "
        "(or CUTLASS_DIR environment variable)")
    endif()
  endif()
  message(STATUS "Source build: cutlass_interface + dedisp_tcfdd from ${CMAKE_SOURCE_DIR}/dedisp_tcfdd")
  message(STATUS "  CUTLASS: ${GGP_CUTLASS_DIR}")
endif()

# add tests, utils, reference, and GGP library
add_subdirectory(lib)
add_subdirectory(tests)
add_subdirectory(examples)
add_subdirectory(doc)
