# add utils files to ggp_test
target_sources(
  ggp_test PRIVATE
  command_line_params.cpp
  host_utils.cpp
  misc.cpp
  )

target_include_directories(ggp_test PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(ggp_test SYSTEM PUBLIC ${CMAKE_SOURCE_DIR}/include/externals)
target_include_directories(ggp_test PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_include_directories(ggp_test PRIVATE ${CMAKE_BINARY_DIR}/include)
