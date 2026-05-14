cmake_minimum_required(VERSION 3.20)
project(DifferentiableGPUImageRegistration LANGUAGES CXX)

option(DGIR_ENABLE_CUDA "Build CUDA starter module" OFF)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_library(dgir_core
  src/affine.cpp
  src/loss.cpp
  src/optimizer.cpp
  src/warp_cpu.cpp
  src/alignment.cpp
)

target_include_directories(dgir_core PUBLIC include)

target_compile_options(dgir_core PRIVATE -Wall -Wextra -Wpedantic)

if(DGIR_ENABLE_CUDA)
  enable_language(CUDA)
  add_library(dgir_cuda src/warp_cuda.cu)
  target_include_directories(dgir_cuda PUBLIC include)
endif()

add_executable(dgir_cli tools/register_cli.cpp)
target_link_libraries(dgir_cli PRIVATE dgir_core)

include(CTest)
if(BUILD_TESTING)
  add_executable(dgir_tests tests/test_starter.cpp)
  target_link_libraries(dgir_tests PRIVATE dgir_core)
  add_test(NAME dgir_tests COMMAND dgir_tests)
endif()
