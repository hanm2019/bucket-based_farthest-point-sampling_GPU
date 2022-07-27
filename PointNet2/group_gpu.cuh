#include <stdio.h>
#include <stdlib.h>

#define DIVUP(m,n) ((m) / (n) + ((m) % (n) > 0))
__global__ void group_points_kernel_fast(int c, int n, int npoints, int nsample, 
    const float *__restrict__ points, const int *__restrict__ idx, float *__restrict__ out);

void group_points(int c, int n, int npoints, int nsample, 
    const float *points, const int *idx, float *out) ;