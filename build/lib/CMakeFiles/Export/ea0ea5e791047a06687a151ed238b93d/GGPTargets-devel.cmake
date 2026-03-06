#----------------------------------------------------------------
# Generated CMake target import file for configuration "DEVEL".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "GGP::ggp" for configuration "DEVEL"
set_property(TARGET GGP::ggp APPEND PROPERTY IMPORTED_CONFIGURATIONS DEVEL)
set_target_properties(GGP::ggp PROPERTIES
  IMPORTED_LOCATION_DEVEL "${_IMPORT_PREFIX}/lib/libggp.so"
  IMPORTED_SONAME_DEVEL "libggp.so"
  )

list(APPEND _cmake_import_check_targets GGP::ggp )
list(APPEND _cmake_import_check_files_for_GGP::ggp "${_IMPORT_PREFIX}/lib/libggp.so" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
