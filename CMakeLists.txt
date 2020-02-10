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
target_sources(cudapp INTERFACE
  ${CMAKE_CURRENT_LIST_DIR}/allocators/device_allocator.h
  ${CMAKE_CURRENT_LIST_DIR}/allocators/managed_allocator.h
  ${CMAKE_CURRENT_LIST_DIR}/allocators/pinned_allocator.h
  ${CMAKE_CURRENT_LIST_DIR}/arrays/array3d.h
  ${CMAKE_CURRENT_LIST_DIR}/arrays/mipmapped_array.h
  ${CMAKE_CURRENT_LIST_DIR}/arrays/mipmapped_texture_object.h
  ${CMAKE_CURRENT_LIST_DIR}/arrays/surface_object.h
  ${CMAKE_CURRENT_LIST_DIR}/arrays/surface_view.h
  ${CMAKE_CURRENT_LIST_DIR}/arrays/texture_object.h
  ${CMAKE_CURRENT_LIST_DIR}/arrays/texture_view.h
  ${CMAKE_CURRENT_LIST_DIR}/functions/kernel_launchers.h
  ${CMAKE_CURRENT_LIST_DIR}/utilities/ide_helpers.h
  ${CMAKE_CURRENT_LIST_DIR}/utilities/memory_helpers.h
  ${CMAKE_CURRENT_LIST_DIR}/vector_types/vector_type_traits.h
)

target_include_directories(cudapp
  INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/>
    ${CUDA_INCLUDE_DIRS}
)
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

add_subdirectory(test_application)