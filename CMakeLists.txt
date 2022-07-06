cmake_minimum_required(VERSION 3.19)
project(FPS_GPU CUDA)

set(CMAKE_CUDA_STANDARD 11)

set(default_build_type "Release")
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    message(STATUS "Setting build type to '${default_build_type}' as none was specified.")
    set(CMAKE_BUILD_TYPE "${default_build_type}" CACHE
            STRING "Choose the type of build." FORCE)
    # Set the possible values of build type for cmake-gui
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
            "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

SET(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -O0 -Wall -g -ggdb")
SET(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -O3 -Wall")

add_executable(FPS_KDTree_GPU  kdtree_main.cu host_common.h device_common.cuh kdtree.cu kdtree.cuh)

set_target_properties(FPS_KDTree_GPU PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

add_executable(FPS_Baseline_GPU baseline_main.cu host_common.h device_common.cuh)

set_target_properties(FPS_Baseline_GPU PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)


