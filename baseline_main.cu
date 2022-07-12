#include <iostream>
#include "ctime"
#include "fstream"
#include "vector"
#include "host_common.h"
#include "device_common.cuh"
#include "baseline_sampling.cuh"

int main(int argc, char **argv) {
    if (argc != 3) {
        printf("please run this program by the following parameter: sample_number filePath\n");
        return 0;
    }
    //check gpu

    check_GPU();

    int sample_number = atoi(argv[1]);
    std::string filename =  argv[2];

    clock_t start_t, end_t;
    clock_t start_build_t;

    //read point
    std::ifstream fin(filename);
    if (!fin.is_open()) {
        std::cout << "file not exist" << std::endl;
        return 0;
    }
    std::vector <Point> point_data;
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


    float (*coordinates) = new float[point_data_size * 3];
    float (*result_cpu) = new float[sample_number * 3];
    float * d_coord;
    float * result;


    for(int i = 0 ;i < point_data_size ; i++){
        coordinates[i*3] = point_data[i].pos[0];
        coordinates[i*3+1] = point_data[i].pos[1];
        coordinates[i*3+2] = point_data[i].pos[2];
    }
    //warmup
    warmup<<<1, 1>>>();
    cudaDeviceSynchronize();


    start_build_t = clock();

    cudaMalloc((void **) &d_coord, (point_data_size)*sizeof(float)*3);
    cudaMalloc((void **) &result, (sample_number)*sizeof(float)*3);
    cudaMemcpy(d_coord,coordinates,point_data_size *sizeof(float )*3 ,cudaMemcpyHostToDevice);
    farthest_point_sampling(point_data_size,sample_number,d_coord,result);
    cudaMemcpy(result_cpu,result, sample_number * 3 * sizeof(float), cudaMemcpyDeviceToHost);

    end_t = clock();


    cudaError_t err;
    err = cudaGetLastError();
    if (cudaSuccess != err) {
        fprintf(stderr, "CUDA kernel failed : %s\n", cudaGetErrorString(err));
        exit(-1);
    }

    //read point
    std::ofstream fout("baseline.txt");
    if (!fout.is_open()) {
        std::cout << "file failed to open" << std::endl;
        return 0;
    }
    for(int i = 0 ;i < sample_number ;i ++){
        fout << result_cpu[i*3] << " " << result_cpu[i*3+1] << " " << result_cpu[i*3+2] << std::endl;
    }

    fout.close();


    start_t = start_build_t;
    std::cout << "Report:" << std::endl;
    std::cout << "    Type   :baseline(GPU)" << std::endl;
    std::cout << "    Points :" << point_data_size<< std::endl;
    std::cout << "    NPoint :" << sample_number << std::endl;
    std::cout << "    RunTime:" << (double) (end_t - start_t) << "us" << std::endl;
    std::cout << "    Param  :" << filename << std::endl;
    std::time_t time_result = std::time(NULL);
    std::cout << "  Timestamp:" << std::asctime(std::localtime(&time_result)) << std::endl;

    cudaFree(d_coord);
    cudaFree(result);
    free(coordinates);
    free(result_cpu);
    return 0;

}


