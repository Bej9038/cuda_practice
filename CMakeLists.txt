cmake_minimum_required(VERSION 3.25)
project(cuda CUDA)

set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_VERBOSE_MAKEFILE ON)
add_executable(cuda conv.cu kernel.cu kernel.h)

set_target_properties(cuda PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
