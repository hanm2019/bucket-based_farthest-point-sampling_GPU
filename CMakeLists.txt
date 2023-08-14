cmake_minimum_required(VERSION 3.19)
project(FPS_GPU CUDA)

set(CMAKE_CUDA_STANDARD 14)

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

set(CMAKE_CUDA_FLAGS_DEBUG "-g -G")
set(CMAKE_CUDA_FLAGS_RELEASE  "-O3")


if(CMAKE_BUILD_TYPE MATCHES Debug)
    message("add define: DCUDA_ERROR_CHECK")
    add_definitions(-DCUDA_ERROR_CHECK)
    add_definitions(-DCUDA_DEBUG_)
endif()

add_executable(FPS_KDTree_GPU  src/kdtree_main.cu src/host_common.h src/device_common.cu src/device_common.cuh src/kdtree.cu src/kdtree.cuh)

set_target_properties(FPS_KDTree_GPU PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

add_executable(FPS_Baseline_GPU src/baseline_main.cu src/host_common.h src/device_common.cu src/device_common.cuh src/baseline_sampling.cu src/baseline_sampling.cuh)

set_target_properties(FPS_Baseline_GPU PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)


if(DEFINED ENV{CUDA_PATH})
    link_directories($ENV{CUDA_PATH}/lib64)
else()
    link_directories(/usr/local/cuda/lib64)
endif()

if(DEFINED ENV{CUDNN_PATH})
    link_directories($ENV{CUDNN_PATH}/lib64)
endif()


