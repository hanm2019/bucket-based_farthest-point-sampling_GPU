//
// Created by hanm on 22-7-6.
//
#include "kdtree.cuh"
/*
 * dPoints: 点序列
 * bucketIndex, bucketLength: 第n个桶在dPoints的起始位置以及长度
 * numPartition: 目前已划分的数量
 * bufferLength: 对于sharememory的切分长度
 * */


__global__ void devide(float3* dPoints, int * bucketIndex, int * bucketLength, int numPartition, int bufferLength){
    extern __shared__ float3 buffer[];

    float3* up = buffer;
    float3* down = (float3*)&up[bufferLength];
    float3* sum = (float3*)&down[bufferLength];
    int* partitionDim = (int*)&sum[bufferLength];
    float* partitionValue = (float*)&partitionDim[numPartition];


    int partitionStride = gridDim.x;
    int threadStride = blockDim.x;

    for(int partitionId = blockIdx.x; partitionId < numPartition; partitionId += partitionStride){
        int shareMemoryIdx = threadIdx.x + blockIdx.x * blockDim.x;

        float3* threadUp = (float3*)&up[shareMemoryIdx];
        float3* threadDown = (float3*)&down[shareMemoryIdx];
        float3* threadSum = (float3*)&sum[shareMemoryIdx];

        float3 dimUp = {-1e10, -1e10, -1e10};
        float3 dimDown = {1e10, 1e10, 1e10};
        float3 dimSum = {0, 0, 0};

        int partitionOffset = bucketIndex[partitionId];
        int partitionLen = bucketLength[partitionId];

        float3* dataset = dPoints + partitionOffset;
        for(int i = threadIdx.x; i < partitionLen; i += threadStride){
            dimUp.x = max(dimUp.x, dataset[i].x);
            dimUp.y = max(dimUp.y, dataset[i].y);
            dimUp.z = max(dimUp.z, dataset[i].z);

            dimDown.x = min(dimDown.x, dataset[i].x);
            dimDown.y = min(dimDown.y, dataset[i].y);
            dimDown.z = min(dimDown.z, dataset[i].z);

            dimSum.x  += dataset[i].x;
            dimSum.y  += dataset[i].y;
            dimSum.z  += dataset[i].z;
        }
        threadUp[0] = dimUp;
        threadDown[0] = dimDown;
        threadSum[0] = dimSum;
        __syncthreads();
        //reduce
        for (int32_t active_thread_num = blockDim.x / 2; active_thread_num >= 1; active_thread_num /= 2) {
            if (threadIdx.x < active_thread_num) {
                threadUp[0].x = max(threadUp[0].x, threadUp[active_thread_num].x);
                threadUp[0].y = max(threadUp[0].y, threadUp[active_thread_num].y);
                threadUp[0].z = max(threadUp[0].z, threadUp[active_thread_num].z);

                threadDown[0].x = min(threadDown[0].x, threadDown[active_thread_num].x);
                threadDown[0].y = min(threadDown[0].y, threadDown[active_thread_num].y);
                threadDown[0].z = min(threadDown[0].z, threadDown[active_thread_num].z);

                threadSum[0].x += threadSum[active_thread_num].x;
                threadSum[0].y += threadSum[active_thread_num].y;
                threadSum[0].z += threadSum[active_thread_num].z;
            }
            __syncthreads();
        }
        if (threadIdx.x == 0) {
            //find out the split dim and middle value
            float3 range = {threadUp[0].x - threadDown[0].x,
                            threadUp[0].y - threadDown[0].y,
                            threadUp[0].z - threadDown[0].z };
            int dim = 0;
            float middleValue = threadSum[0].x / (partitionLen+0.0);

            if(range.x > range.y && range.x > range.z) dim = 0;
            if(range.y > range.x && range.y > range.z) {dim = 1;middleValue = threadSum[0].y / (partitionLen+0.0);}
            if(range.z > range.x && range.z > range.y) {dim = 2;middleValue = threadSum[0].z / (partitionLen+0.0);}

            partitionDim[blockIdx.x] = dim;
            partitionValue[blockIdx.x] = middleValue;
        }
        __syncthreads();
        //merge sort
        int divideDim = partitionDim[blockIdx.x];
        float divideValue = partitionValue[blockIdx.x];
 
        int mid = 0;
        for(int stride = 1; stride < partitionLen; stride *= 2){
            for(int threadStart = threadIdx.x * 2 * stride; threadStart < partitionLen; threadStart += threadStride * stride * 2){
                int left = threadStart;
                int endLeft = (left + stride)  < partitionLen ?  (left + stride) : partitionLen; //边界条件: left < endLeft

                int right = (left + 2 * stride) < partitionLen ? (left + 2 * stride - 1) : (partitionLen -1);
                int endRight = (left + stride)  < partitionLen ? (left + stride - 1) : (partitionLen - 1); //边界条件: right > endRight

                if(divideDim == 0) {
                    while (left < endLeft) {
                        if (dataset[left].x <= divideValue) left++;
                        else break;
                        
                    }
                    while (right > endRight) {
                        if (dataset[right].x >= divideValue) right--;
                        else break;
                    }
                } else{
                    if(divideDim == 1){
                        while (left < endLeft) {
                            if (dataset[left].y <= divideValue) left++;
                            else break;
                        }
                        while (right > endRight) {
                            if (dataset[right].y >= divideValue) right--;
                            else break;
                        }
                    } else{
                        while (left < endLeft) {
                            if (dataset[left].z <= divideValue) left++;
                            else break;
                        }
                        while (right > endRight) {
                            if (dataset[right].z >= divideValue) right--;
                            else break;
                        }
                    }
                }

                while((left < endLeft) && (right > endRight)){
                    //swap
                    float3 tmp = dataset[left];
                    dataset[left] = dataset[right];
                    dataset[right] = tmp;
                    left ++;
                    right --;
                }
                if(left < endLeft) mid = left;
                else mid = right + 1;
            }
            __syncthreads();
        }
        if(threadIdx.x == 0){
            //update bucketIndex and bucketLength
            bucketIndex[partitionId + numPartition] = partitionOffset + mid ;
            bucketLength[partitionId + numPartition] = partitionLen - mid;
            bucketLength[partitionId] = mid;
        }
        __syncthreads();
    }
}

void    buildKDTree(int * bucketIndex, int * bucketLength, float3 * ptr, int kd_high, float3 * up, float3 * down){
    int currentLevel=0;
    int nThreads, nBlocks;
    cudaError_t err;
    while(currentLevel<kd_high)
    {
        nBlocks =  ((int) pow(2.0f,currentLevel+0.0f));
        nThreads = numOfCudaCores/nBlocks;

        int ThreadSize = nBlocks*nThreads;

        devide<<<nBlocks, nThreads,ThreadSize*3*sizeof(float3) + nBlocks * (sizeof(int) + sizeof(float))>>>(ptr,bucketIndex, bucketLength, nBlocks, ThreadSize);

        err = cudaGetLastError();
        if (cudaSuccess != err) {
            fprintf(stderr, "CUDA kernel failed : %s\n", cudaGetErrorString(err));
            exit(-1);
        }
        currentLevel++;
    }
    cudaDeviceSynchronize();
    nBlocks = 1 << kd_high;
    nThreads = numOfCudaCores/nBlocks;
    int ThreadSize = nBlocks * nThreads;
    generateBoundbox<<<nBlocks, nThreads, ThreadSize * 2 * sizeof(float3) >>>(bucketIndex, bucketLength, ptr, nBlocks,ThreadSize, up, down);

}

void
sample(int *bucketIndex, int *bucketLength, float3 *ptr, int bucketSize, float3 *up, float3 *down, float4 *bucketTable,
       int sample_number, float3 *result) {

}

__global__ void generateBoundbox(int * bucketIndex, int * bucketLength, float3 * dPoints, int numPartition, int bufferLength, float3 * up, float3 * down){
    extern __shared__ float3 buffer[];

    float3* shareUp = buffer;
    float3* shareDown = (float3*)&up[bufferLength];

    int partitionStride = gridDim.x;
    int threadStride = blockDim.x;

    for(int partitionId = blockIdx.x; partitionId < numPartition; partitionId += partitionStride) {
        int shareMemoryIdx = threadIdx.x + blockIdx.x * blockDim.x;

        float3 *threadUp = (float3 *) &shareUp[shareMemoryIdx];
        float3 *threadDown = (float3 *) &shareDown[shareMemoryIdx];

        float3 dimUp = {-1e10, -1e10, -1e10};
        float3 dimDown = {1e10, 1e10, 1e10};

        int partitionOffset = bucketIndex[partitionId];
        int partitionLen = bucketLength[partitionId];

        float3 *dataset = dPoints + partitionOffset;
        for (int i = threadIdx.x; i < partitionLen; i += threadStride) {
            dimUp.x = max(dimUp.x, dataset[i].x);
            dimUp.y = max(dimUp.y, dataset[i].y);
            dimUp.z = max(dimUp.z, dataset[i].z);

            dimDown.x = min(dimDown.x, dataset[i].x);
            dimDown.y = min(dimDown.y, dataset[i].y);
            dimDown.z = min(dimDown.z, dataset[i].z);
        }
        threadUp[0] = dimUp;
        threadDown[0] = dimDown;
        __syncthreads();
        //reduce
        for (int32_t active_thread_num = blockDim.x / 2; active_thread_num >= 1; active_thread_num /= 2) {
            if (threadIdx.x < active_thread_num) {
                threadUp[0].x = max(threadUp[0].x, threadUp[active_thread_num].x);
                threadUp[0].y = max(threadUp[0].y, threadUp[active_thread_num].y);
                threadUp[0].z = max(threadUp[0].z, threadUp[active_thread_num].z);

                threadDown[0].x = min(threadDown[0].x, threadDown[active_thread_num].x);
                threadDown[0].y = min(threadDown[0].y, threadDown[active_thread_num].y);
                threadDown[0].z = min(threadDown[0].z, threadDown[active_thread_num].z);
            }
            __syncthreads();
        }
        if (threadIdx.x == 0) {
            up[partitionId] = threadUp[0];
            down[partitionId] = threadDown[0];
        }
        __syncthreads();
    }
}

