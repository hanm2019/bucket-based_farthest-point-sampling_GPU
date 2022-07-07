#include "device_common.cuh"

__global__ void warmup(){
    return;
}


void check_GPU(){
    cudaError_t cudaStatus;
    int num = 0;
    cudaStatus = cudaGetDeviceCount(&num);
    std::cout << "Number of GPU: " << num << std::endl;
    cudaDeviceProp prop;
    if (num > 0) {
        cudaGetDeviceProperties(&prop, 0);
        std::cout << "Device: " << prop.name << std::endl;
    }
}

__device__ void __update(float *__restrict__ dists, int *__restrict__ dists_i, int idx1, int idx2){
    const float v1 = dists[idx1], v2 = dists[idx2];
    const int i1 = dists_i[idx1], i2 = dists_i[idx2];
    dists[idx1] = max(v1, v2);
    dists_i[idx1] = v2 > v1 ? i2 : i1;
}

__device__ void merge(float *__restrict__ dists, int *__restrict__ dists_i,int tid, int block_size){
    if (block_size >= 4096) {
        if (tid < 2048) {
            __update(dists, dists_i, tid, tid + 2048);
        }
        __syncthreads();
    }
    if (block_size >= 2048) {
        if (tid < 1024) {
            __update(dists, dists_i, tid, tid + 1024);
        }
        __syncthreads();
    }

    if (block_size >= 1024) {
        if (tid < 512) {
            __update(dists, dists_i, tid, tid + 512);
        }
        __syncthreads();
    }

    if (block_size >= 512) {
        if (tid < 256) {
            __update(dists, dists_i, tid, tid + 256);
        }
        __syncthreads();
    }
    if (block_size >= 256) {
        if (tid < 128) {
            __update(dists, dists_i, tid, tid + 128);
        }
        __syncthreads();
    }
    if (block_size >= 128) {
        if (tid < 64) {
            __update(dists, dists_i, tid, tid + 64);
        }
        __syncthreads();
    }
    if (block_size >= 64) {
        if (tid < 32) {
            __update(dists, dists_i, tid, tid + 32);
        }
        __syncthreads();
    }
    if (block_size >= 32) {
        if (tid < 16) {
            __update(dists, dists_i, tid, tid + 16);
        }
        __syncthreads();
    }

    if (block_size >= 16) {
        if (tid < 8) {
            __update(dists, dists_i, tid, tid + 8);
        }
        __syncthreads();
    }

    if (block_size >= 8) {
        if (tid < 4) {
            __update(dists, dists_i, tid, tid + 4);
        }
        __syncthreads();
    }
    if (block_size >= 4) {
        if (tid < 2) {
            __update(dists, dists_i, tid, tid + 2);
        }
        __syncthreads();
    }
    if (block_size >= 2) {
        if (tid < 1) {
            __update(dists, dists_i, tid, tid + 1);
        }
        __syncthreads();
    }
}
