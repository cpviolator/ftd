# Install script for directory: /home/deanhowarth/spark_cutlass_bench/cutlass_sandbox/blackwell/dsa-2000-monorepo/packages/ftd/tests

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "/home/deanhowarth/spark_cutlass_bench/cutlass_sandbox/blackwell/dsa-2000-monorepo/packages/ftd/build")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "DEVEL")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Install shared libraries without execute permission?
if(NOT DEFINED CMAKE_INSTALL_SO_NO_EXE)
  set(CMAKE_INSTALL_SO_NO_EXE "1")
endif()

# Is this installation the result of a crosscompile?
if(NOT DEFINED CMAKE_CROSSCOMPILING)
  set(CMAKE_CROSSCOMPILING "FALSE")
endif()

# Set default install directory permissions.
if(NOT DEFINED CMAKE_OBJDUMP)
  set(CMAKE_OBJDUMP "/usr/bin/objdump")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libggp_test.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libggp_test.so")
    file(RPATH_CHECK
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libggp_test.so"
         RPATH "/home/deanhowarth/spark_cutlass_bench/cutlass_sandbox/blackwell/dsa-2000-monorepo/packages/ftd/build/lib:/usr/local/cuda-13.0/targets/sbsa-linux/lib")
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE SHARED_LIBRARY FILES "/home/deanhowarth/spark_cutlass_bench/cutlass_sandbox/blackwell/dsa-2000-monorepo/packages/ftd/build/tests/libggp_test.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libggp_test.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libggp_test.so")
    file(RPATH_CHANGE
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libggp_test.so"
         OLD_RPATH "/home/deanhowarth/spark_cutlass_bench/cutlass_sandbox/blackwell/dsa-2000-monorepo/packages/ftd/build/lib:/usr/local/cuda-13.0/targets/sbsa-linux/lib:"
         NEW_RPATH "/home/deanhowarth/spark_cutlass_bench/cutlass_sandbox/blackwell/dsa-2000-monorepo/packages/ftd/build/lib:/usr/local/cuda-13.0/targets/sbsa-linux/lib")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libggp_test.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("/home/deanhowarth/spark_cutlass_bench/cutlass_sandbox/blackwell/dsa-2000-monorepo/packages/ftd/build/tests/utils/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("/home/deanhowarth/spark_cutlass_bench/cutlass_sandbox/blackwell/dsa-2000-monorepo/packages/ftd/build/tests/host_reference/cmake_install.cmake")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/xengine_test" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/xengine_test")
    file(RPATH_CHECK
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/xengine_test"
         RPATH "/home/deanhowarth/spark_cutlass_bench/cutlass_sandbox/blackwell/dsa-2000-monorepo/packages/ftd/build/lib:/usr/local/cuda-13.0/targets/sbsa-linux/lib")
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/bin" TYPE EXECUTABLE FILES "/home/deanhowarth/spark_cutlass_bench/cutlass_sandbox/blackwell/dsa-2000-monorepo/packages/ftd/build/tests/xengine_test")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/xengine_test" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/xengine_test")
    file(RPATH_CHANGE
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/xengine_test"
         OLD_RPATH "/home/deanhowarth/spark_cutlass_bench/cutlass_sandbox/blackwell/dsa-2000-monorepo/packages/ftd/build/tests:/home/deanhowarth/spark_cutlass_bench/cutlass_sandbox/blackwell/dsa-2000-monorepo/packages/ftd/build/lib:/usr/local/cuda-13.0/targets/sbsa-linux/lib:"
         NEW_RPATH "/home/deanhowarth/spark_cutlass_bench/cutlass_sandbox/blackwell/dsa-2000-monorepo/packages/ftd/build/lib:/usr/local/cuda-13.0/targets/sbsa-linux/lib")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/xengine_test")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/dedisp_interface_test" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/dedisp_interface_test")
    file(RPATH_CHECK
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/dedisp_interface_test"
         RPATH "/home/deanhowarth/spark_cutlass_bench/cutlass_sandbox/blackwell/dsa-2000-monorepo/packages/ftd/build/lib:/usr/local/cuda-13.0/targets/sbsa-linux/lib")
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/bin" TYPE EXECUTABLE FILES "/home/deanhowarth/spark_cutlass_bench/cutlass_sandbox/blackwell/dsa-2000-monorepo/packages/ftd/build/tests/dedisp_interface_test")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/dedisp_interface_test" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/dedisp_interface_test")
    file(RPATH_CHANGE
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/dedisp_interface_test"
         OLD_RPATH "/home/deanhowarth/spark_cutlass_bench/cutlass_sandbox/blackwell/dsa-2000-monorepo/packages/ftd/build/tests:/home/deanhowarth/spark_cutlass_bench/cutlass_sandbox/blackwell/dsa-2000-monorepo/packages/ftd/build/lib:/usr/local/cuda-13.0/targets/sbsa-linux/lib:"
         NEW_RPATH "/home/deanhowarth/spark_cutlass_bench/cutlass_sandbox/blackwell/dsa-2000-monorepo/packages/ftd/build/lib:/usr/local/cuda-13.0/targets/sbsa-linux/lib")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/dedisp_interface_test")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/blas_interface_test" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/blas_interface_test")
    file(RPATH_CHECK
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/blas_interface_test"
         RPATH "/home/deanhowarth/spark_cutlass_bench/cutlass_sandbox/blackwell/dsa-2000-monorepo/packages/ftd/build/lib:/usr/local/cuda-13.0/targets/sbsa-linux/lib")
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/bin" TYPE EXECUTABLE FILES "/home/deanhowarth/spark_cutlass_bench/cutlass_sandbox/blackwell/dsa-2000-monorepo/packages/ftd/build/tests/blas_interface_test")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/blas_interface_test" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/blas_interface_test")
    file(RPATH_CHANGE
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/blas_interface_test"
         OLD_RPATH "/home/deanhowarth/spark_cutlass_bench/cutlass_sandbox/blackwell/dsa-2000-monorepo/packages/ftd/build/tests:/home/deanhowarth/spark_cutlass_bench/cutlass_sandbox/blackwell/dsa-2000-monorepo/packages/ftd/build/lib:/usr/local/cuda-13.0/targets/sbsa-linux/lib:"
         NEW_RPATH "/home/deanhowarth/spark_cutlass_bench/cutlass_sandbox/blackwell/dsa-2000-monorepo/packages/ftd/build/lib:/usr/local/cuda-13.0/targets/sbsa-linux/lib")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/blas_interface_test")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/tune_test" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/tune_test")
    file(RPATH_CHECK
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/tune_test"
         RPATH "/home/deanhowarth/spark_cutlass_bench/cutlass_sandbox/blackwell/dsa-2000-monorepo/packages/ftd/build/lib:/usr/local/cuda-13.0/targets/sbsa-linux/lib")
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/bin" TYPE EXECUTABLE FILES "/home/deanhowarth/spark_cutlass_bench/cutlass_sandbox/blackwell/dsa-2000-monorepo/packages/ftd/build/tests/tune_test")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/tune_test" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/tune_test")
    file(RPATH_CHANGE
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/tune_test"
         OLD_RPATH "/home/deanhowarth/spark_cutlass_bench/cutlass_sandbox/blackwell/dsa-2000-monorepo/packages/ftd/build/tests:/home/deanhowarth/spark_cutlass_bench/cutlass_sandbox/blackwell/dsa-2000-monorepo/packages/ftd/build/lib:/usr/local/cuda-13.0/targets/sbsa-linux/lib:"
         NEW_RPATH "/home/deanhowarth/spark_cutlass_bench/cutlass_sandbox/blackwell/dsa-2000-monorepo/packages/ftd/build/lib:/usr/local/cuda-13.0/targets/sbsa-linux/lib")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/tune_test")
    endif()
  endif()
endif()

