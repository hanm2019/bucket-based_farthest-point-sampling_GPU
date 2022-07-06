//
// Created by hanm on 22-7-6.
//

#ifndef FPS_GPU_KDTREE_CUH
#define FPS_GPU_KDTREE_CUH
#include "iostream"

#define numOfCudaCores  1024

void    buildKDTree(int * bucketIndex, int * bucketLength, float3 * ptr, int kd_high, float3 * up, float3 * down);

__global__ void devide(float3* dPoints, int * bucketIndex, int * bucketLength, int numPartition, int bufferLength);

__global__ void generateBoundbox(int * bucketIndex, int * bucketLength, float3 * dPoints, int numPartition, float3 * up, float3 * down);

void sample(int * bucketIndex,int * bucketLength,float3 * ptr, int bucketSize, float3 * up, float3 * down, float4 * bucketTable, int sample_number, float3* result);

#endif //FPS_GPU_KDTREE_CUH
