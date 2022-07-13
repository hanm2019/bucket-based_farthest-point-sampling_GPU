//
// Created by hanm on 22-7-12.
//

#ifndef FPS_GPU_BALL_QUERY_GPU_CUH
#define FPS_GPU_BALL_QUERY_GPU_CUH


__global__ void ball_query_kernel_fast(int n, int m, float radius2, int nsample, const float *__restrict__ new_xyz, const float *__restrict__ xyz, int *__restrict__ idx);

void ball_query(int pointSize, int sampleSize, float radius, int k, const float *__restrict__ new_xyz, const float *__restrict__ xyz,int *__restrict__ idx);
#endif //FPS_GPU_BALL_QUERY_GPU_CUH
