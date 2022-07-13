//
// Created by hanm on 22-7-12.
//
#include "../device_common.cuh"
#include <iostream>
#include "ctime"
#include "fstream"
#include "vector"
#include "../host_common.h"
#include "../baseline_sampling.cuh"
#include "ball_query_gpu.cuh"
#include "group_gpu.cuh"
#include "mlp.cuh"

int main(int argc, char **argv) {
    if (argc != 3) {
        printf("please run this program by the following parameter: sample_number filePath\n");
        return 0;
    }
    //check gpu

    check_GPU();

    int sample_number = atoi(argv[1]);
    std::string filename = argv[2];

    //read point
    std::ifstream fin(filename);
    if (!fin.is_open()) {
        std::cout << "file not exist" << std::endl;
        return 0;
    }
    std::vector<Point> point_data;
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

    cudaEvent_t start, stop, fps_start, fps_end, group_end;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventCreate(&fps_start);
    cudaEventCreate(&fps_end);
    cudaEventCreate(&group_end);
    int k = 20; //每个点的最多邻居数量
    int channelNum = 3;
    float radius = 200; // 1 x 200
    float (*coordinates) = new float[point_data_size * 3];
    float *d_coord;
    float *result;
    int * idx;
    float * group_out;
    float * mlp_out1;
    float * mlp_out2;
    float * mlp_out3;
    float (* mlp_result) = new float[(sample_number)*32*k];
    int cov_high = 1, cov_width = 1;

    for (int i = 0; i < point_data_size; i++) {
        coordinates[i * 3] = point_data[i].pos[0];
        coordinates[i * 3 + 1] = point_data[i].pos[1];
        coordinates[i * 3 + 2] = point_data[i].pos[2];
    }
    //warmup
    warmup<<<1, 1>>>();
    cudaDeviceSynchronize();

    cudaEventRecord(start);

    cudaMalloc((void **) &d_coord, (point_data_size)*sizeof(float)*3);
    cudaMalloc((void **) &result, (sample_number)*sizeof(float)*3);
    cudaMalloc((void **) &idx, (sample_number)*sizeof(int)*k);

    cudaMalloc((void **) &group_out, (sample_number)*sizeof(float)*3*k);

    cudaMalloc((void **) &mlp_out1, (sample_number)*sizeof(float)*16*k);
    cudaMalloc((void **) &mlp_out2, (sample_number)*sizeof(float)*16*k);
    cudaMalloc((void **) &mlp_out3, (sample_number)*sizeof(float)*32*k);

    cudaEventRecord(fps_start);
    cudaMemcpy(d_coord,coordinates,point_data_size *sizeof(float )*3 ,cudaMemcpyHostToDevice);
    //sample
    farthest_point_sampling(point_data_size,sample_number,d_coord,result);
    cudaEventRecord(fps_end);
    //query
    ball_query(point_data_size, sample_number, radius, k, result, d_coord, idx);
    //group map
    group_points(channelNum, point_data_size, sample_number, k, d_coord, idx, group_out);
    cudaEventRecord(group_end);
    //mlp
    mlp(group_out, mlp_out1, cov_high, cov_width, 3, 16, sample_number, k); // 3-> 16
    mlp(mlp_out1, mlp_out2, cov_high, cov_width, 16, 16, sample_number, k); // 16->16
    mlp(mlp_out2, mlp_out3, cov_high, cov_width, 16, 32, sample_number, k); // 16->32
    cudaEventRecord(stop);

    cudaMemcpy(mlp_result, mlp_out3, (sample_number)*sizeof(float)*32*k, cudaMemcpyDeviceToHost);

    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    float milliseconds_fps,milliseconds_group,milliseconds_mlp;
    cudaEventElapsedTime(&milliseconds_fps, fps_start, fps_end);
    cudaEventElapsedTime(&milliseconds_group, fps_end, group_end);
    cudaEventElapsedTime(&milliseconds_mlp, group_end, stop);

    cudaError_t err;
    err = cudaGetLastError();
    if (cudaSuccess != err) {
        fprintf(stderr, "CUDA kernel failed : %s\n", cudaGetErrorString(err));
        exit(-1);
    }

    std::cout << "Report:" << std::endl;
    std::cout << "    Type   :PointNet++(GPU)" << std::endl;
    std::cout << "    Points :" << point_data_size<< std::endl;
    std::cout << "    NPoint :" << sample_number << std::endl;
    std::cout << "    RunTime:" << milliseconds << "ms" << std::endl;
    std::cout << "       Sample: " << milliseconds_fps << "ms(" << (milliseconds_fps*100.0/milliseconds) << "%)" << std::endl;
    std::cout << "       Group: " << milliseconds_group << "ms(" << (milliseconds_group*100.0/milliseconds) << "%)" << std::endl;
    std::cout << "       MLP: " << milliseconds_mlp << "ms(" << (milliseconds_mlp*100.0/milliseconds) << "%)" << std::endl;
    std::cout << "    Param  :" << filename << std::endl;
    std::time_t time_result = std::time(NULL);
    std::cout << "  Timestamp:" << std::asctime(std::localtime(&time_result)) << std::endl;

    cudaFree(d_coord);
    cudaFree(result);
    cudaFree(idx);
    cudaFree(group_out);
    cudaFree(mlp_out1);
    cudaFree(mlp_out2);
    cudaFree(mlp_out3);
    free(coordinates);
    free(mlp_result);
    return 0;
}