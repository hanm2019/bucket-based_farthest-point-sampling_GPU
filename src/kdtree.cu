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




__global__ void devide(float3* dPoints, float3 * dtemp, int * bucketIndex, int * bucketLength, int numPartition, int bufferLength, int offset){
    extern __shared__ float shareBuffer[];

    float3* up = (float3*)&shareBuffer[0];
    float3* down = (float3*)&up[bufferLength];
    float3* sum = (float3*)&down[bufferLength];
    int* partitionDim = (int*)&sum[bufferLength];
    float* partitionValue = (float*)&partitionDim[1];

    int* shareMid = (int*)&shareBuffer[offset];

    int* lessWriteBackPtr = (int*)&shareMid[1];
    int * greaterWriteBackPtr = (int*)&lessWriteBackPtr[1];

    float3* buffer = (float3*)&shareBuffer[0];


    const int blockId = blockIdx.x;
    const int partitionStride = gridDim.x;

    const int threadId = threadIdx.x;
    const int threadStride = blockDim.x;

    for(int partitionId = blockId ; partitionId < numPartition; partitionId += partitionStride){

        float3 dimUp = {-1e10, -1e10, -1e10};
        float3 dimDown = {1e10, 1e10, 1e10};
        float3 dimSum = {0, 0, 0};

        int partitionOffset = bucketIndex[partitionId];
        int partitionLen = bucketLength[partitionId];


        float3* dataset = dPoints + partitionOffset;
        float3* dataTemp = dtemp + partitionOffset;

        for(int i = threadId; i < partitionLen; i += threadStride){
            float3 data = dataset[i];

            dimUp.x = max(dimUp.x, data.x);
            dimUp.y = max(dimUp.y, data.y);
            dimUp.z = max(dimUp.z, data.z);

            dimDown.x = min(dimDown.x, data.x);
            dimDown.y = min(dimDown.y, data.y);
            dimDown.z = min(dimDown.z, data.z);

            dimSum.x  += data.x;
            dimSum.y  += data.y;
            dimSum.z  += data.z;
        }
        up[threadId] = dimUp;
        down[threadId] = dimDown;
        sum[threadId] = dimSum;
        __syncthreads();
        //reduce
        for (int32_t active_thread_num = threadStride / 2; active_thread_num >= 1; active_thread_num /= 2) {
            if (threadId < active_thread_num) {
                up[threadId].x = max(up[threadId].x, up[threadId + active_thread_num].x);
                up[threadId].y = max(up[threadId].y, up[threadId + active_thread_num].y);
                up[threadId].z = max(up[threadId].z, up[threadId + active_thread_num].z);

                down[threadId].x = min(down[threadId].x, down[threadId + active_thread_num].x);
                down[threadId].y = min(down[threadId].y, down[threadId + active_thread_num].y);
                down[threadId].z = min(down[threadId].z, down[threadId + active_thread_num].z);

                sum[threadId].x += sum[threadId + active_thread_num].x;
                sum[threadId].y += sum[threadId + active_thread_num].y;
                sum[threadId].z += sum[threadId + active_thread_num].z;
            }
            __syncthreads();
        }
        if (threadId == 0) {
            //find out the split dim and middle value
            float3 range = {up[0].x - down[0].x,
                            up[0].y - down[0].y,
                            up[0].z - down[0].z };
            int dim = 0;
            float middleValue = sum[0].x / (partitionLen+0.0);

            if(range.x > range.y && range.x > range.z) dim = 0;
            if(range.y > range.x && range.y > range.z) {dim = 1;middleValue = sum[0].y / (partitionLen+0.0);}
            if(range.z > range.x && range.z > range.y) {dim = 2;middleValue = sum[0].z / (partitionLen+0.0);}

            (*partitionDim) = dim;
            (*partitionValue) = middleValue;

            (* lessWriteBackPtr) = 0;
            (* greaterWriteBackPtr) = 0;
        }
        __syncthreads();
        //merge sort
        int divideDim = (*partitionDim);
        float divideValue = (*partitionValue);
        const int partMergeLen = MergeLen;


        for(int dataPtr = 0; dataPtr < partitionLen; dataPtr += partMergeLen){
            const int currentPartLen = min(partMergeLen, partitionLen - dataPtr);

            //copy global memory to share memory
            float3* partData = (float3*) &dataset[dataPtr];

            for(int i = threadId; i < currentPartLen; i += threadStride){
                buffer[i] = partData[i];
            }
            __syncthreads();

            //merge sort
            int mid = 0;
            for(int stride = 1; stride < currentPartLen; stride *= 2){
                for(int threadStart = threadId * 2 * stride; threadStart < currentPartLen; threadStart += threadStride * stride * 2){
                    int left = threadStart;
                    int endLeft = (left + stride)  < currentPartLen ?  (left + stride) : currentPartLen; //边界条件: left < endLeft

                    int right = (left + 2 * stride) < currentPartLen ? (left + 2 * stride - 1) : (currentPartLen -1);
                    int endRight = (left + stride)  < currentPartLen ? (left + stride - 1) : (currentPartLen - 1); //边界条件: right > endRight

                    if(divideDim == 0) {
                        while (left < endLeft) {
                            if (buffer[left].x <= divideValue) left++;
                            else break;

                        }
                        while (right > endRight) {
                            if (buffer[right].x >= divideValue) right--;
                            else break;
                        }
                    } else{
                        if(divideDim == 1){
                            while (left < endLeft) {
                                if (buffer[left].y <= divideValue) left++;
                                else break;
                            }
                            while (right > endRight) {
                                if (buffer[right].y >= divideValue) right--;
                                else break;
                            }
                        } else{
                            while (left < endLeft) {
                                if (buffer[left].z <= divideValue) left++;
                                else break;
                            }
                            while (right > endRight) {
                                if (buffer[right].z >= divideValue) right--;
                                else break;
                            }
                        }
                    }

                    while((left < endLeft) && (right > endRight)){
                        //swap
                        float3 tmp = buffer[left];
                        buffer[left] = buffer[right];
                        buffer[right] = tmp;
                        left ++;
                        right --;
                    }
                    if(left < endLeft) mid = left;
                    else mid = right + 1;
                }
                __syncthreads();
            }
            // sync mid
            if(threadId == 0){
                (*shareMid) = mid;
            }
            __syncthreads();

            mid = (*shareMid);

            int lessPtr = *lessWriteBackPtr;
            int greaterPtr = *greaterWriteBackPtr;

            //copy back to data
            float3 * lessGlobalData = (float3 * )&(dataset[lessPtr]);
            float3 * greaterGlobalData = (float3 * )&(dataTemp[greaterPtr]);

            for(int i = threadId; i < mid; i += threadStride){
                lessGlobalData[i] = buffer[i];
            }
            //copy to temp

            float3 * greaterBuffer = (float3 * )&buffer[mid];

            const int greaterLen = currentPartLen - mid;

            for(int i = threadId ; i < greaterLen; i += threadStride){
                greaterGlobalData[i] = greaterBuffer[i];
            }
            // update lessWriteBackPtr and greaterWriteBackPtr
            if(threadId == 0){
                (*lessWriteBackPtr) += mid;
                (*greaterWriteBackPtr) += greaterLen;
            }
            __syncthreads();
        }
        //copy tempdata to dataset
        const int greaterLen = (*greaterWriteBackPtr);
        const int lessLen = (*lessWriteBackPtr);
        float3* dataset_2 = (float3*)&dataset[lessLen];

        for(int i = threadId; i < greaterLen; i += threadStride){
            dataset_2[i] = dataTemp[i];
        }
        if(threadId == 0){
            //update bucketIndex and bucketLength
            bucketIndex[partitionId + numPartition] = partitionOffset + lessLen ;
            bucketLength[partitionId + numPartition] = partitionLen - lessLen;
            bucketLength[partitionId] = lessLen;
        }
        __syncthreads();
    }
}

void buildKDTree(int * bucketIndex, int * bucketLength, float3 * ptr, int kd_high, float3 * up, float3 * down, int point_data_size ){
    int currentLevel=0;
    int nThreads, nBlocks;
    cudaError_t err;
    float3 * dtemp;
    cudaMalloc((void **)&dtemp, point_data_size*sizeof(float3));

    while(currentLevel<kd_high)
    {
        nBlocks =  ((int) pow(2.0f,currentLevel+0.0f));
        nThreads = currentLevel > 2 ? std::max(32, 4096/nBlocks) : 1024;

        const int bytes = std::max( nThreads*3*sizeof(float3) + sizeof(int) + sizeof(float) , MergeLen * sizeof(float3)) + 3 * sizeof(int);
        const int offset = (bytes/sizeof(float)) - 3;
        devide<<<nBlocks, nThreads,bytes>>>
            (ptr,dtemp, bucketIndex, bucketLength, nBlocks, nThreads, offset);

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
    cudaFree(dtemp);
}

void sample(int * bucketIndex, int * bucketLength, float3 * ptr, int pointSize,  int bucketSize, float3 * up, float3 * down, int sample_number, float3 * result){

    thrust::device_vector<float> tempVector(pointSize);
    thrust::fill(tempVector.begin(), tempVector.end(), 1e20);
    float * temp = thrust::raw_pointer_cast(&tempVector[0]);

    thrust::device_vector<float4> bucketTableVector(bucketSize);
    thrust::fill(bucketTableVector.begin(), bucketTableVector.end(), float4({0,0,0, 1e20}));
    float4 * bucketTable = thrust::raw_pointer_cast(&bucketTableVector[0]);

    thrust::device_vector<bool> needToDealVector(bucketSize);
    bool * needToDeal = thrust::raw_pointer_cast(&needToDealVector[0]);

#ifdef DEBUG_GG
    printf("bytes:%d\n", bytes);
#endif
    cudaMemcpy(result, ptr, sizeof(float3),cudaMemcpyDeviceToDevice); //first point

    dim3 bucketDim(bucketSize);
    for(int i = 1; i < sample_number; i++){
        checkBucket<<<1,bucketDim>>>(bucketTable, result, i, up, down, needToDeal);
        CudaCheckError();
        sample_kernel<numOfCudaCores><<<bucketDim,numOfCudaCores >>>(bucketIndex, bucketLength, ptr, temp , result, i ,needToDeal, bucketTable);
        CudaCheckError();
        reduce(bucketSize, bucketTable,result,i);
        CudaCheckError();
    }

}

__device__ float pow2(float a){
    return a*a;
}

__global__ void checkBucket(float4* bucketTable ,float3 *result,int i,float3 *up,float3 *down,bool *needToDeal) {
    const int tid = threadIdx.x;

    const float3 origin_point = result[i-1];

    const float4 bucketMaxPoint = bucketTable[tid];
    const float3 bucketUp = up[tid];
    const float3 bucketDown = down[tid];

    const float last_dist = bucketMaxPoint.w;
    const float cur_dist = pow2((origin_point.x - bucketMaxPoint.x)) +
                               pow2((origin_point.y - bucketMaxPoint.y))  +
                               pow2((origin_point.z - bucketMaxPoint.z));

    const float bound_dist = pow2(max(origin_point.x, bucketUp.x) - bucketUp.x) + pow2(bucketDown.x - min(origin_point.x, bucketDown.x)) +
                                 pow2(max(origin_point.y, bucketUp.y) - bucketUp.y) + pow2(bucketDown.y - min(origin_point.y, bucketDown.y)) +
                                 pow2(max(origin_point.z, bucketUp.z) - bucketUp.z) + pow2(bucketDown.z - min(origin_point.z, bucketDown.z)) ;
    needToDeal[tid] = (cur_dist <= last_dist || bound_dist < last_dist);
}

__global__ void generateBoundbox(int * bucketIndex, int * bucketLength, float3 * dPoints, int numPartition, int bufferLength, float3 * up, float3 * down){
    extern __shared__ float3 buffer[];

    float3* shareUp = buffer;
    float3* shareDown = (float3*)&up[bufferLength];

    const int partitionStride = gridDim.x;
    const int threadStride = blockDim.x;

    for(int partitionId = blockIdx.x; partitionId < numPartition; partitionId += partitionStride) {
        const int shareMemoryIdx = threadIdx.x + blockIdx.x * blockDim.x;

        float3 *threadUp = (float3 *) &shareUp[shareMemoryIdx];
        float3 *threadDown = (float3 *) &shareDown[shareMemoryIdx];

        float3 dimUp = {-1e10, -1e10, -1e10};
        float3 dimDown = {1e10, 1e10, 1e10};

        const int partitionOffset = bucketIndex[partitionId];
        const int partitionLen = bucketLength[partitionId];

        float3 *dataset = dPoints + partitionOffset;
        for (int i = threadIdx.x; i < partitionLen; i += threadStride) {
            float3 data = dataset[i];
            dimUp.x = max(dimUp.x, data.x);
            dimUp.y = max(dimUp.y, data.y);
            dimUp.z = max(dimUp.z, data.z);

            dimDown.x = min(dimDown.x, data.x);
            dimDown.y = min(dimDown.y, data.y);
            dimDown.z = min(dimDown.z, data.z);
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

void reduce(int bucketSize,  float4* bucketTable, float3 * result, int offset){
    assert(bucketSize <=numOfCudaCores);
    dim3 BucketDim(bucketSize);
    switch (bucketSize) {
        case 1:reduce_kernel<1><<<1, BucketDim>>>(bucketTable, result, offset);break;
        case 2:reduce_kernel<2><<<1, BucketDim>>>(bucketTable, result, offset);break;
        case 4:reduce_kernel<4><<<1, BucketDim>>>(bucketTable, result, offset);break;
        case 8:reduce_kernel<8><<<1, BucketDim>>>(bucketTable, result, offset);break;
        case 16:reduce_kernel<16><<<1, BucketDim>>>(bucketTable, result, offset);break;
        case 32:reduce_kernel<32><<<1, BucketDim>>>(bucketTable, result, offset);break;
        case 64:reduce_kernel<64><<<1, BucketDim>>>(bucketTable, result, offset);break;
        case 128:reduce_kernel<128><<<1, BucketDim>>>(bucketTable, result, offset);break;
        case 256:reduce_kernel<256><<<1, BucketDim>>>(bucketTable, result, offset);break;
        case 512:reduce_kernel<512><<<1, BucketDim>>>(bucketTable, result, offset);break;
        case 1024:reduce_kernel<1024><<<1, BucketDim>>>(bucketTable, result, offset);break;
    }
}


