#include <iostream>
#include "ctime"
#include "fstream"
#include "vector"
#include "host_common.h"
#include "device_common.cuh"


template <unsigned int block_size>
__global__ void farthest_point_sampling_kernel(int N, int npoint, const float *dataset, float *temp, float * result){
    __shared__ float dists[block_size];
    __shared__ int dists_i[block_size];

    int tid = threadIdx.x + blockDim.x * blockIdx.x;
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

void farthest_point_sampling(int point_data_size, int sample_number, const float *coordinates, float * result) {
    float * temp;
    cudaMalloc((void **) &temp, (point_data_size)*sizeof(float));

    farthest_point_sampling_kernel<4096><<<32, 128>>>(point_data_size,sample_number,coordinates,temp,result);
    cudaFree(temp);
}



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


