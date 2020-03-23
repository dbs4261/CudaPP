cmake_minimum_required(VERSION 3.15)
project(CudaPP LANGUAGES CXX CUDA)

set(ignoreMe "${CMAKE_C_COMPILER}")

include(cmake/StandardProjectSettings.cmake)

# Not sure if this works correctly, still have the sets further on. Gotta make it work with cuda.
add_library(project_options INTERFACE)
target_compile_features(project_options INTERFACE cxx_std_14)
include(cmake/Sanitizers.cmake)
enable_sanitizers(project_options)
include(cmake/StaticAnalyzers.cmake)

add_library(project_warnings INTERFACE)
include(cmake/CompilerWarnings.cmake)
set_project_warnings(project_warnings)

include(cmake/Doxygen.cmake)
enable_doxygen()

option(BUILD_SHARED_LIBS "Enable compilation of shared libraries" OFF)
option(ENABLE_TESTING "Enable Test Builds" ON)

# Very basic PCH example
option(ENABLE_PCH "Enable Precompiled Headers" OFF)
if (ENABLE_PCH)
  # This sets a global PCH parameter, each project will build its own PCH, which
  # is a good idea if any #define's change
  #
  #
  target_precompile_headers(project_options INTERFACE <vector> <string> <map> <utility>)
endif()

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

if (ENABLE_TESTING)
  message("Testing enabled...")
  include(GoogleTest)
  add_subdirectory(test)
elseif()
  message("Skipping tests...")
endif()

add_subdirectory(test_application)
