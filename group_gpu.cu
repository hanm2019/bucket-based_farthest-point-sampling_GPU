#include "group_gpu.cuh"

__global__ void group_points_kernel_fast(int c, int n, int npoints, int nsample, 
    const float *__restrict__ points, const int *__restrict__ idx, float *__restrict__ out) {
    // points: (1, C, N)
    // idx: (1, npoints, nsample)
    // output:
    //      out: (1, C, npoints, nsample)
    int c_idx = blockIdx.y;
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int pt_idx = index / nsample;
    if (c_idx >= c || pt_idx >= npoints) return;

    int sample_idx = index % nsample;

    idx +=  pt_idx * nsample + sample_idx; 
    int in_idx = c_idx * n + idx[0];
    int out_idx =  c_idx * npoints * nsample + pt_idx * nsample + sample_idx;

    out[out_idx] = points[in_idx];
}

void group_points(int c, int n, int npoints, int nsample, 
    const float *points, const int *idx, float *out) {
    // points: (1, C, N)
    // idx: (1, npoints, nsample)
    // output:
    //      out: (1, C, npoints, nsample)
    cudaError_t err;
    dim3 blocks(DIVUP(npoints * nsample, 1024), c);  // blockIdx.x(col), blockIdx.y(row)
    dim3 threads(1024);

    group_points_kernel_fast<<<blocks, threads>>>(c, n, npoints, nsample, points, idx, out);
    // cudaDeviceSynchronize();  // for using printf in kernel function
    err = cudaGetLastError();
    if (cudaSuccess != err) {
        fprintf(stderr, "CUDA kernel failed : %s\n", cudaGetErrorString(err));
        exit(-1);
    }
}