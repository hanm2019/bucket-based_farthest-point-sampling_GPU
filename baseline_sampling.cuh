//
// Created by hanm on 22-7-12.
//

#ifndef FPS_GPU_BASELINE_SAMPLING_CUH
#define FPS_GPU_BASELINE_SAMPLING_CUH
#include "device_common.cuh"


template <unsigned int block_size>
__global__ void farthest_point_sampling_kernel(int N, int npoint, const float *dataset, float *temp, float * result){
    __shared__ float dists[block_size];
    __shared__ int dists_i[block_size];

    int tid = threadIdx.x;

    const int stride = block_size;
    for(int j = tid; j < N; j += stride){
        temp[j] = 1e10;
    }
    int old = 0;
    if(tid == 0){
        result[0] = dataset[old * 3];
        result[1] = dataset[old * 3 + 1];
        result[2] = dataset[old * 3 + 2];
    }
    __syncthreads();
    for(int j = 1; j < npoint; j++){
        float best = -1;
        int besti = 0;
        float x1 = dataset[old * 3 + 0];
        float y1 = dataset[old * 3 + 1];
        float z1 = dataset[old * 3 + 2];
        for(int k = tid; k < N; k += stride){
            float x2, y2, z2;
            x2 = dataset[k * 3 + 0];
            y2 = dataset[k * 3 + 1];
            z2 = dataset[k * 3 + 2];
            float d = (x2 - x1) * (x2 - x1) +
                      (y2 - y1) * (y2 - y1) +
                      (z2 - z1) * (z2 - z1);
            float d2 = min(d, temp[k]);
            temp[k] = d2;
            besti = d2 > best ? k : besti;
            best = d2 > best ? d2 : best;

        }
        dists[tid] = best;
        dists_i[tid] = besti;
        __syncthreads();

        merge(dists, dists_i, tid, block_size);

        old = dists_i[0];
        if(tid == 0){
            result[j * 3] = dataset[old * 3];
            result[j * 3 + 1] = dataset[old * 3 + 1];
            result[j * 3 + 2] = dataset[old * 3 + 2];
        }
        __syncthreads();
    }

}
void farthest_point_sampling(int point_data_size, int sample_number, const float *coordinates, float * result);

#endif //FPS_GPU_BASELINE_SAMPLING_CUH
