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
#define MergeLen 2048
#define CudaCheckError()    __cudaCheckError( __FILE__, __LINE__ )

inline void __cudaCheckError( const char *file, const int line )
{
#ifdef CUDA_ERROR_CHECK
    cudaError err = cudaGetLastError();
    if ( cudaSuccess != err )
    {
        fprintf( stderr, "cudaCheckError() failed at %s:%i : %s\n",
                 file, line, cudaGetErrorString( err ) );
        exit( -1 );
    }

    // More careful checking. However, this will affect performance.
    // Comment away if needed.
    err = cudaDeviceSynchronize();
    if( cudaSuccess != err )
    {
        fprintf( stderr, "cudaCheckError() with sync failed at %s:%i : %s\n",
                 file, line, cudaGetErrorString( err ) );
        exit( -1 );
    }
#endif

    return;
}

void    buildKDTree(int * bucketIndex, int * bucketLength, float3 * ptr, int kd_high, float3 * up, float3 * down, int point_data_size);

__global__ void devide(float3* dPoints, float3 *dtemp, int * bucketIndex, int * bucketLength, int numPartition, int bufferLength, int offset);

__global__ void generateBoundbox(int * bucketIndex, int * bucketLength, float3 * dPoints, int numPartition, int bufferLength, float3 * up, float3 * down);

__global__ void checkBucket(float4* bucketTable,float3 *result,int i,float3 *up,float3 *down,bool *needToDeal);

void reduce(int bucketSize, float4* bucketTable, float3 * result, int offset);

__device__ float pow2(float a);

void sample(int * bucketIndex, int * bucketLength, float3 * ptr,int pointSize, int bucketSize, float3 * up, float3 * down, int sample_number, float3 * result);

/*
 * bucketIndex: 每个bucket的起始偏移
 * bucketLength: 每个bucket的长度
 * ptr: 点云数据
 * temp: 距离的中间变量
 * result: 采样结果
 * offset: 采样指针
 * needToDeal: 是否需要计算的标志
 * bucketTable: 若需要计算，则将更新后的最远点写入
 * */

template <unsigned int block_size> __global__
void sample_kernel(int *bucketIndex, int *bucketLength, float3 *ptr,float* temp, float3 *result, int offset, bool * needToDeal, float4 * bucketTable) {
    __shared__ float dists[block_size];
    __shared__ int dists_i[block_size];

    const int bucketPtr = blockIdx.x;
    if (needToDeal[bucketPtr]) {

        const int tid = threadIdx.x;

        const float origin_x = result[offset - 1].x;
        const float origin_y = result[offset - 1].y;
        const float origin_z = result[offset - 1].z;

        const int partitionLen = bucketLength[bucketPtr];
        const int partitionOffset = bucketIndex[bucketPtr];

        float3 *dataset = (float3 *) &ptr[partitionOffset];
        float *distTemp = (float *) &temp[partitionOffset];

        float best = -1;
        int besti = 0;
        for (int k = tid; k < partitionLen; k += block_size) {
            const float3 point = dataset[k];
            const float d = pow2((point.x - origin_x)) + pow2((point.y - origin_y)) + pow2((point.z - origin_z));
            const float d2 = min(d, distTemp[k]);
            distTemp[k] = d2;
            besti = d2 > best ? k : besti;
            best = d2 > best ? d2 : best;
        }
        dists[tid] = best;
        dists_i[tid] = besti;
        __syncthreads();

        merge(dists, dists_i, tid, block_size);

        if (tid == 0) {
            const float3 maxPoint_ = dataset[dists_i[0]];
            bucketTable[bucketPtr] = float4({maxPoint_.x, maxPoint_.y, maxPoint_.z, dists[0]});
        }
        __syncthreads();
    }
}

template <unsigned int block_size>  __global__
void reduce_kernel(float4* bucketTable, float3 * result, int offset){
    __shared__ float dists[block_size];
    __shared__ int dists_i[block_size];

    const int tid = threadIdx.x;

    dists[tid] = bucketTable[tid].w;
    dists_i[tid] = tid;

    merge(dists, dists_i, tid, block_size);

    if(tid == 0){
        const float4 maxPoint = bucketTable[dists_i[0]];
        result[offset] = float3({maxPoint.x, maxPoint.y, maxPoint.z});
    }
    __syncthreads();
}


#endif //FPS_GPU_KDTREE_CUH
