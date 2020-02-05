cmake_minimum_required(VERSION 3.14)
project(CudaPP LANGUAGES CXX CUDA)

set(ignoreMe "${CMAKE_C_COMPILER}")

set(CMAKE_CXX_STANDARD 14)
if(NOT DEFINED CMAKE_CUDA_STANDARD)
  set(CMAKE_CUDA_STANDARD 14)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif()

if(CMAKE_BUILD_TYPE MATCHES Debug)
  set(CUDA_NVCC_FLAGS -O0 -G -g --use_fast_math --ptxas-options=-v --compiler-options -Wall)
else()
  set(CUDA_NVCC_FLAGS -O3 --use_fast_math --ptxas-options=-v --compiler-options -Wall)
endif()

set(CUDA_COMPILED_ARCH "sm_75")

add_library(cudapp INTERFACE)

#set_target_properties(cudapp PROPERTIES
#  CUDA_SEPERABLE_COMPILATION ON
#  CUDA_HOST_COMPILER ${CMAKE_CXX_COMPILER}
#)
target_include_directories(cudapp
  INTERFACE
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
)
target_compile_options(cudapp
  INTERFACE
    $<$<COMPILE_LANGUAGE:CUDA>:-arch=${CUDA_COMPILED_ARCH}>
)
target_link_libraries(cudapp
  INTERFACE
    ${CUDA_LIBRARIES}
)