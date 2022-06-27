cmake_minimum_required(VERSION 3.20)
project(FPS_GPU CUDA)

set(CMAKE_CUDA_STANDARD 11)

add_executable(FPS_GPU main.cu)

set_target_properties(FPS_GPU PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
