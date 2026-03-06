# ######################################################################################################################
# additonal sources
target_sources(ggp_cpp PRIVATE ggp_api.cpp device.cpp malloc.cpp blas_lapack_cublas.cpp comm_target.cpp)

if(GGP_JITIFY)
  target_sources(ggp_cpp PRIVATE jitify_helper.cpp)
endif()

if(GGP_BACKWARDS)
  set_property(
    SOURCE malloc.cpp
    DIRECTORY ${CMAKE_SOURCE_DIR}/lib
    APPEND
    PROPERTY COMPILE_DEFINITIONS ${BACKWARD_DEFINITIONS})
  set_property(
    SOURCE malloc.cpp
    DIRECTORY ${CMAKE_SOURCE_DIR}/lib
    APPEND
    PROPERTY COMPILE_DEFINITIONS GGP_BACKWARDSCPP)
endif()
