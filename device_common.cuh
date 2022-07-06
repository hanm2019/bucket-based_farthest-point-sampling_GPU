//
// Created by hanm on 22-7-5.
//

#ifndef FPS_GPU_DEVICE_COMMON_CUH
#define FPS_GPU_DEVICE_COMMON_CUH

__global__ void warmup(){}


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



#endif //FPS_GPU_DEVICE_COMMON_CUH
