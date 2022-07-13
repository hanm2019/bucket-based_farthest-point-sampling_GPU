__global__ void ball_query_kernel_fast(int n, int m, float radius2, int nsample, 
    const float *__restrict__ new_xyz, const float *__restrict__ xyz, int *__restrict__ idx) {
    // new_xyz: (1, M, 3)
    // xyz: (1, N, 3)
    // output:
    //      idx: (1, M, nsample)
    int pt_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (pt_idx >= m) return;

    new_xyz += pt_idx * 3;
    idx += pt_idx * nsample;

    float new_x = new_xyz[0];
    float new_y = new_xyz[1];
    float new_z = new_xyz[2];

    int cnt = 0;
    for (int k = 0; k < n; ++k) {
        float x = xyz[k * 3 + 0];
        float y = xyz[k * 3 + 1];
        float z = xyz[k * 3 + 2];
        float d2 = (new_x - x) * (new_x - x) + (new_y - y) * (new_y - y) + (new_z - z) * (new_z - z);
        if (d2 < radius2){
            if (cnt == 0){
                for (int l = 0; l < nsample; ++l) {
                    idx[l] = k;
                }
            }
            idx[cnt] = k;
            ++cnt;
            if (cnt >= nsample) break;
        }
    }
}


void ball_query(int pointSize, int sampleSize, float radius, int k, const float *__restrict__ new_xyz, const float *__restrict__ xyz,int *__restrict__ idx){
    const int blockSize = sampleSize%1024 == 0 ? sampleSize/1024 : (sampleSize/1024 + 1);
    ball_query_kernel_fast<<<blockSize ,1024>>>(pointSize, sampleSize, radius*radius, k, new_xyz, xyz, idx);

}