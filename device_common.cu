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
