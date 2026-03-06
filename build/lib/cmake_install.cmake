# Install script for directory: /home/deanhowarth/spark_cutlass_bench/cutlass_sandbox/blackwell/dsa-2000-monorepo/packages/ftd/lib

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
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include" TYPE FILE FILES "/home/deanhowarth/spark_cutlass_bench/cutlass_sandbox/blackwell/dsa-2000-monorepo/packages/ftd/build/include/targets/cuda/device.hpp")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("/home/deanhowarth/spark_cutlass_bench/cutlass_sandbox/blackwell/dsa-2000-monorepo/packages/ftd/build/lib/targets/cuda/cmake_install.cmake")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/cmake/GGP" TYPE FILE FILES "/home/deanhowarth/spark_cutlass_bench/cutlass_sandbox/blackwell/dsa-2000-monorepo/packages/ftd/cmake/find_target_cuda_dependencies.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("/home/deanhowarth/spark_cutlass_bench/cutlass_sandbox/blackwell/dsa-2000-monorepo/packages/ftd/build/lib/targets/generic/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("/home/deanhowarth/spark_cutlass_bench/cutlass_sandbox/blackwell/dsa-2000-monorepo/packages/ftd/build/lib/interface/cmake_install.cmake")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include" TYPE FILE FILES "/home/deanhowarth/spark_cutlass_bench/cutlass_sandbox/blackwell/dsa-2000-monorepo/packages/ftd/build/include/ggp_define.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libggp.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libggp.so")
    file(RPATH_CHECK
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libggp.so"
         RPATH "/home/deanhowarth/spark_cutlass_bench/cutlass_sandbox/blackwell/dsa-2000-monorepo/packages/ftd/build/lib:/usr/local/cuda-13.0/targets/sbsa-linux/lib")
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE SHARED_LIBRARY FILES "/home/deanhowarth/spark_cutlass_bench/cutlass_sandbox/blackwell/dsa-2000-monorepo/packages/ftd/build/lib/libggp.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libggp.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libggp.so")
    file(RPATH_CHANGE
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libggp.so"
         OLD_RPATH "/usr/local/cuda-13.0/targets/sbsa-linux/lib:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::"
         NEW_RPATH "/home/deanhowarth/spark_cutlass_bench/cutlass_sandbox/blackwell/dsa-2000-monorepo/packages/ftd/build/lib:/usr/local/cuda-13.0/targets/sbsa-linux/lib")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libggp.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/cmake/GGP/GGPTargets.cmake")
    file(DIFFERENT _cmake_export_file_changed FILES
         "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/cmake/GGP/GGPTargets.cmake"
         "/home/deanhowarth/spark_cutlass_bench/cutlass_sandbox/blackwell/dsa-2000-monorepo/packages/ftd/build/lib/CMakeFiles/Export/ea0ea5e791047a06687a151ed238b93d/GGPTargets.cmake")
    if(_cmake_export_file_changed)
      file(GLOB _cmake_old_config_files "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/cmake/GGP/GGPTargets-*.cmake")
      if(_cmake_old_config_files)
        string(REPLACE ";" ", " _cmake_old_config_files_text "${_cmake_old_config_files}")
        message(STATUS "Old export file \"$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/cmake/GGP/GGPTargets.cmake\" will be replaced.  Removing files [${_cmake_old_config_files_text}].")
        unset(_cmake_old_config_files_text)
        file(REMOVE ${_cmake_old_config_files})
      endif()
      unset(_cmake_old_config_files)
    endif()
    unset(_cmake_export_file_changed)
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/cmake/GGP" TYPE FILE FILES "/home/deanhowarth/spark_cutlass_bench/cutlass_sandbox/blackwell/dsa-2000-monorepo/packages/ftd/build/lib/CMakeFiles/Export/ea0ea5e791047a06687a151ed238b93d/GGPTargets.cmake")
  if(CMAKE_INSTALL_CONFIG_NAME MATCHES "^([Dd][Ee][Vv][Ee][Ll])$")
    file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/cmake/GGP" TYPE FILE FILES "/home/deanhowarth/spark_cutlass_bench/cutlass_sandbox/blackwell/dsa-2000-monorepo/packages/ftd/build/lib/CMakeFiles/Export/ea0ea5e791047a06687a151ed238b93d/GGPTargets-devel.cmake")
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include" TYPE DIRECTORY FILES "/home/deanhowarth/spark_cutlass_bench/cutlass_sandbox/blackwell/dsa-2000-monorepo/packages/ftd/include/")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/cmake/GGP" TYPE FILE FILES
    "/home/deanhowarth/spark_cutlass_bench/cutlass_sandbox/blackwell/dsa-2000-monorepo/packages/ftd/build/lib/GGPConfigVersion.cmake"
    "/home/deanhowarth/spark_cutlass_bench/cutlass_sandbox/blackwell/dsa-2000-monorepo/packages/ftd/build/lib/GGPConfig.cmake"
    )
endif()

