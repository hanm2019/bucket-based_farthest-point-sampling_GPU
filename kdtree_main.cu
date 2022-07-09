//
// Created by hanm on 22-7-5.
//
#include <iostream>
#include "ctime"
#include "fstream"
#include "device_common.cuh"
#include "algorithm"

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/fill.h>
#include "kdtree.cuh"

int main(int argc, char **argv) {
    if (argc != 4) {
        printf("please run this program by the following parameter: kdtree_high sample_number filePath\n");
        return 0;
    }

    //check gpu

    check_GPU();

    int kd_high = atoi(argv[1]);
    int sample_number = atoi(argv[2]);
    std::string filename =  argv[3];

    clock_t start_t, end_t;
    clock_t start_build_t, end_build_t;


    //read point
    std::ifstream fin(filename);
    if (!fin.is_open()) {
        std::cout << "file not exist" << std::endl;
        return 0;
    }
    thrust::host_vector<float3> point_data;
    int count = 0;
    if (fin.is_open()) {
        float xx, yy, zz;
        while (fin >> xx >> yy >> zz) {
            point_data.push_back({xx, yy, zz});
            count++;
        }
    }
    fin.close();
    const int point_data_size = point_data.size();

    //warmup
    warmup<<<1, 1>>>();
    cudaDeviceSynchronize();

    //build

    start_build_t = clock();
    int bucketSize = 1 << kd_high;

    thrust::device_vector<float3> dPoints=point_data;
    float3 * ptr = thrust::raw_pointer_cast(&dPoints[0]);

    float3 * up;
    float3 * down;
    float3 * result;


    thrust::device_vector<int> bucketIndexVector(bucketSize);
    thrust::device_vector<int> bucketLengthVector(bucketSize);


    thrust::fill(bucketIndexVector.begin(), bucketIndexVector.end(), 0);
    thrust::fill(bucketLengthVector.begin(), bucketLengthVector.end(), point_data_size);

    int * bucketIndex = thrust::raw_pointer_cast(&bucketIndexVector[0]);
    int * bucketLength = thrust::raw_pointer_cast(&bucketLengthVector[0]);


    cudaMalloc((void **)&up, bucketSize*sizeof(float3));
    cudaMalloc((void **)&down, bucketSize*sizeof(float3));
    cudaMalloc((void **)&result, sample_number*sizeof(float3));



    buildKDTree(bucketIndex, bucketLength, ptr, kd_high, up, down, point_data_size);

#ifdef  DEBUG_GG
    thrust::host_vector<int>cpu_bucketLength(bucketSize);
    thrust::copy(bucketLengthVector.begin(), bucketLengthVector.end(), cpu_bucketLength.begin());
    for(const auto & leng: cpu_bucketLength){
        printf("len: %d\n", leng);
    }
#endif

    end_build_t = clock();
    //fps
    sample(bucketIndex, bucketLength, ptr, point_data_size, bucketSize, up, down, sample_number, result);

    end_t = clock();
    start_t = start_build_t;

    float3 result_cpu[sample_number];

    cudaMemcpy((void *)result_cpu,(void *)result, sample_number*sizeof(float3), cudaMemcpyDeviceToHost);

    //read point
    std::ofstream fout("kdtree.txt");
    if (!fout.is_open()) {
        std::cout << "file failed to open" << std::endl;
        return 0;
    }
    for(const auto& point : result_cpu){
        fout << point.x << " " << point.y << " " << point.z << std::endl;
    }

    fout.close();


    std::cout << "Report:" << std::endl;
    std::cout << "    Type   :kdline(GPU) high:" << kd_high << std::endl;
    std::cout << "    Points :" << point_data_size<< std::endl;
    std::cout << "    NPoint :" << sample_number << std::endl;
    std::cout << "    RunTime:" << (double) (end_t - start_t) << "us" << std::endl;
    std::cout << "    BuildTime:" << (double) (end_build_t - start_build_t) << "us" << std::endl;
    std::cout << "    Param  :" << filename << std::endl;
    std::time_t time_result = std::time(NULL);
    std::cout << "  Timestamp:" << std::asctime(std::localtime(&time_result)) << std::endl;

    cudaFree(up);
    cudaFree(down);
    cudaFree(result);

    return 0;
}