cmake_minimum_required(VERSION 3.24)
project(differentiable_gpu_image_registration LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_ARCHITECTURES "75" CACHE STRING "CUDA architectures to build")

find_package(CUDAToolkit REQUIRED)
find_package(PNG REQUIRED)

add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:--generate-line-info>)
add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)

add_library(registration_core
    src/affine.cpp
    src/cpu_registration.cpp
    src/image.cpp
    src/image_io.cpp
)
target_include_directories(registration_core PUBLIC include)
target_link_libraries(registration_core PUBLIC PNG::PNG)
target_compile_options(registration_core PRIVATE -Wall -Wextra -Wpedantic)

add_library(registration_cuda
    src/gpu_registration.cu
)
target_include_directories(registration_cuda PUBLIC include)
target_link_libraries(registration_cuda PUBLIC registration_core CUDA::cudart)
target_compile_options(registration_cuda PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>
)

add_library(registration_optimizer
    src/optimizer.cpp
)
target_link_libraries(registration_optimizer PUBLIC registration_core registration_cuda)

enable_testing()

function(add_registration_test name source)
    add_executable(${name} ${source})
    target_link_libraries(${name} PRIVATE registration_optimizer)
    target_include_directories(${name} PRIVATE tests)
    add_test(NAME ${name} COMMAND ${name})
endfunction()

add_registration_test(test_affine tests/test_affine.cpp)
add_registration_test(test_image tests/test_image.cpp)
add_registration_test(test_cpu_baseline tests/test_cpu_baseline.cpp)
add_registration_test(test_gpu_warp tests/test_gpu_warp.cpp)
add_registration_test(test_image_io tests/test_image_io.cpp)
add_registration_test(test_optimizer tests/test_optimizer.cpp)
add_registration_test(test_gpu_loss_gradient tests/test_gpu_loss_gradient.cpp)

add_executable(register_images tools/register_images.cpp)
target_link_libraries(register_images PRIVATE registration_optimizer)

add_executable(registration_bench tools/registration_bench.cpp)
target_link_libraries(registration_bench PRIVATE registration_optimizer)

add_executable(make_synthetic_pair tools/make_synthetic_pair.cpp)
target_link_libraries(make_synthetic_pair PRIVATE registration_optimizer)
