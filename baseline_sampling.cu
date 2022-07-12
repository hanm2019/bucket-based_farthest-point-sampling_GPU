//
// Created by hanm on 22-7-12.
//
#include "baseline_sampling.cuh"

void farthest_point_sampling(int point_data_size, int sample_number, const float *coordinates, float * result) {
    float * temp;
    cudaMalloc((void **) &temp, (point_data_size)*sizeof(float));
    dim3 grid(1);
    dim3 block(1024);
    farthest_point_sampling_kernel<1024><<<grid, block>>>(point_data_size,sample_number,coordinates,temp,result);
    cudaFree(temp);
}
