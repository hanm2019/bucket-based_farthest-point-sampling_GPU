//
// Created by hanm on 22-7-6.
//

#ifndef FPS_GPU_KDTREE_CUH
#define FPS_GPU_KDTREE_CUH
#include "iostream"

#define numOfCudaCores  512
void init(int * bucketLength, int * bucketIndex, int point_data_size);
__global__ void init_bucketLength(int * bucketLength, int * bucketIndex, int point_data_size);
void    buildKDTree(int * bucketIndex, int * bucketLength, float3 * ptr, int kd_high);
__global__ void devide(float3* dPoints, int * bucketIndex, int * bucketLength, int numPartition, int bufferLength);

#endif //FPS_GPU_KDTREE_CUH
