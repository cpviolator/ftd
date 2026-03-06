# enable examples, build a common library for all examples
add_library(ggp_examples pipeline_example.cpp)
target_include_directories(ggp_examples SYSTEM PUBLIC)
target_link_libraries(ggp_examples PUBLIC Eigen)
target_link_libraries(ggp_examples PUBLIC GGP::ggp)
set_target_properties(ggp_examples PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_compile_options(
  ggp_examples PUBLIC
  $<IF:$<CONFIG:RELEASE>,-w,-Wall -Wextra>
  $<$<CONFIG:STRICT>:-Werror>
)

# ignore any unkown pragmas if not using OpenMP
if(NOT ${GGP_OPENMP})
  target_compile_options(ggp_examples PUBLIC $<$<COMPILE_LANGUAGE:CXX>:
    $<$<CXX_COMPILER_ID:Clang>:-Wno-unknown-pragmas>
    $<$<CXX_COMPILER_ID:GNU>:-Wno-unknown-pragmas>
    >)
endif()

if(BUILD_SHARED_LIBS)
  install(TARGETS ggp_examples ${GGP_EXCLUDE_FROM_INSTALL} DESTINATION ${CMAKE_INSTALL_LIBDIR})
endif()

if(GGP_NVSHMEM AND GGP_DOWNLOAD_NVSHMEM)
  add_dependencies(ggp_examples NVSHMEM)
endif()

set(EXAMPLE_LIBS ggp_examples)

# custom target for building all examples (useful if GGP_BUILD_ALL_EXAMPLES is OFF)
add_custom_target(all_examples)

macro(GGP_CHECKBUILDEXAMPLES mytarget ggpbuildexamples)
  if(NOT ${ggpbuildexamples})
    set_property(TARGET ${mytarget} PROPERTY EXCLUDE_FROM_ALL 1)
    set(GGP_EXCLUDE_FROM_INSTALL "EXCLUDE_FROM_ALL")
  endif()
  add_dependencies(all_examples ${mytarget})
endmacro()

if(NOT ${GGP_INSTALL_ALL_EXAMPLES})
  set(GGP_EXCLUDE_FROM_INSTALL "EXCLUDE_FROM_ALL")
endif()

# define examples
add_executable(pipeline_example pipeline_example.cpp)
target_link_libraries(pipeline_example ${EXAMPLE_LIBS})
ggp_checkbuildexamples(pipeline_example GGP_BUILD_ALL_EXAMPLES)
install(TARGETS pipeline_example ${GGP_EXCLUDE_FROM_INSTALL} DESTINATION ${CMAKE_INSTALL_BINDIR})
