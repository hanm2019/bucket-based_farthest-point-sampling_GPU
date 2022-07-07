//
// Created by hanm on 22-7-5.
//

#ifndef FPS_GPU_DEVICE_COMMON_CUH
#define FPS_GPU_DEVICE_COMMON_CUH
#include "iostream"

__global__ void warmup();


void check_GPU();

__device__ void __update(float *__restrict__ dists, int *__restrict__ dists_i, int idx1, int idx2);

__device__ void merge(float *__restrict__ dists, int *__restrict__ dists_i,int tid, int block_size);

#endif //FPS_GPU_DEVICE_COMMON_CUH
