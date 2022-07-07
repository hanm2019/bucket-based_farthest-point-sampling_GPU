//
// Created by hanm on 22-7-6.
//

#ifndef FPS_GPU_KDTREE_CUH
#define FPS_GPU_KDTREE_CUH
#include "iostream"
#include <thrust/device_vector.h>
#include <thrust/fill.h>
#include "device_common.cuh"

#define numOfCudaCores  1024

void    buildKDTree(int * bucketIndex, int * bucketLength, float3 * ptr, int kd_high, float3 * up, float3 * down);

__global__ void devide(float3* dPoints, int * bucketIndex, int * bucketLength, int numPartition, int bufferLength);

__global__ void generateBoundbox(int * bucketIndex, int * bucketLength, float3 * dPoints, int numPartition, int bufferLength, float3 * up, float3 * down);

void sample(int * bucketIndex, int * bucketLength, float3 * ptr,int pointSize, int bucketSize, float3 * up, float3 * down, int sample_number, float3 * result);

/*
 * bucketIndex: 每个bucket的起始偏移
 * bucketLength: 每个bucket的长度
 * ptr: 点云数据
 * temp: 距离的中间变量
 * bucketSize: bucket的数量
 * up, down: 每个bucket的上下边界
 * sample_num: 采样数量
 * result: 采样结果
 *
 * Note: gridDim = 1
 * */

template <unsigned int block_size> __global__ void sample_kernel(int *bucketIndex, int *bucketLength, float3 *ptr,float* temp, int bucketSize,  float3 *up, float3 *down,
              int sample_number, float3 *result){
    extern __shared__ float pointsBuffer[];

    float4* bucketTable = (float4*)&pointsBuffer[0]; //len: bucketSize

    float* dists = (float*)&bucketTable[bucketSize]; //len : block_size

    int* dists_i = (int*)&dists[block_size]; //len: block_size

    int * needToDeal = (int*)&dists_i[block_size]; // len: bucketSize

    int * needToDealSize = (int*)&needToDeal[bucketSize];

    const int tid = threadIdx.x;

    //init bucketTable
    for(int bucketPtr = tid; bucketPtr < bucketSize; bucketPtr += block_size){
        bucketTable[bucketPtr] = float4({0,0,0, 1e20});
    }
    if(tid == 0){
        result[0] = ptr[0];
    }
    __syncthreads();

    for(int j = 1; j < sample_number; j++){
        const float origin_x = result[j-1].x;
        const float origin_y = result[j-1].y;
        const float origin_z = result[j-1].z;

        //check every bucket
        for(int bucketPtr = tid ; bucketPtr < bucketSize; bucketPtr += block_size){
            const float4 bucketMaxPoint = bucketTable[bucketPtr];
            const float3 bucketUp = up[bucketPtr];
            const float3 bucketDown = down[bucketPtr];

            const float last_dist = bucketMaxPoint.w;
            const float cur_dist = pow((origin_x - bucketMaxPoint.x) ,2) +
                             pow((origin_y - bucketMaxPoint.y),2)  +
                             pow((origin_z - bucketMaxPoint.z),2);

            const float bound_dist = pow(max(origin_x, bucketUp.x) - bucketUp.x,2) + pow(bucketDown.x - min(origin_x, bucketDown.x),2) +
                               pow(max(origin_y, bucketUp.y) - bucketUp.y,2) + pow(bucketDown.y - min(origin_y, bucketDown.y),2) +
                               pow(max(origin_z, bucketUp.z) - bucketUp.z,2) + pow(bucketDown.z - min(origin_z, bucketDown.z),2) ;
            needToDeal[bucketPtr] = (cur_dist <= last_dist || bound_dist < last_dist) ? bucketPtr : -1;
        }
        __syncthreads();
        if(tid == 0){
            //merge needToDeal
            int target = 0;
            for(int left = 0; left < bucketSize; left ++){
                if(needToDeal[left] != -1){
                    needToDeal[target] = needToDeal[left];
                    target ++;
                }
            }
            (*needToDealSize) = target;
        }
        __syncthreads();

        const int needToDealSize_n = (*needToDealSize);
        for(int dealPtr = 0; dealPtr < needToDealSize_n; dealPtr ++){
            const int bucketPtr = needToDeal[dealPtr];

            const int partitionLen = bucketLength[bucketPtr];

            float3* dataset = (float3 *) &ptr[bucketIndex[bucketPtr]];
            float * distTemp = (float *) &temp[bucketIndex[bucketPtr]];

            float best = -1;
            int besti = 0;
            for(int k = tid; k < partitionLen; k += block_size){
                const float3 point = dataset[k];
                const float d = pow((point.x - origin_x),2) + pow((point.y - origin_y),2) + pow((point.z - origin_z),2);
                const float d2= min(d, distTemp[k]);
                distTemp[k] = d2;
                besti = d2 > best ? k : besti;
                best = d2 > best ? d2 : best;
            }
            dists[tid] = best;
            dists_i[tid] = besti;
            __syncthreads();

            merge(dists, dists_i, tid, block_size);

            if(tid == 0){
                const float3 maxPoint_ = dataset[dists_i[0]];
                bucketTable[bucketPtr] = float4({maxPoint_.x, maxPoint_.y, maxPoint_.z, dists[0]});
            }
            __syncthreads();
        }
        //after deal with each bucket
        //select the farthest points
        float best = -1;
        int besti = 0;
        for(int bucketPtr = tid; bucketPtr < bucketSize; bucketPtr += block_size){
            const float d3 = bucketTable[bucketPtr].w;
            besti = d3 > best ? bucketPtr : besti;
            best = d3 > best ? d3 : best;
        }
        dists[tid] = best;
        dists_i[tid] = besti;
        __syncthreads();
        //merge
        merge(dists, dists_i, tid, block_size);

        if(tid == 0){
            float4 maxPoint = bucketTable[dists_i[0]];
            result[j] = float3({maxPoint.x, maxPoint.y, maxPoint.z});
        }
        __syncthreads();
    }
}

#endif //FPS_GPU_KDTREE_CUH
