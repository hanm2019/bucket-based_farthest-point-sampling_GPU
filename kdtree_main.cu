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

    thrust::device_vector<float4> bucketTableVector(bucketSize);
    thrust::device_vector<int> bucketIndexVector(bucketSize);
    thrust::device_vector<int> bucketLengthVector(bucketSize);

    thrust::fill(bucketTableVector.begin(), bucketTableVector.end(), float4{0,0,0,1e20});
    thrust::fill(bucketIndexVector.begin(), bucketIndexVector.end(), 0);
    thrust::fill(bucketLengthVector.begin(), bucketLengthVector.end(), point_data_size);

    int * bucketIndex = thrust::raw_pointer_cast(&bucketIndexVector[0]);
    int * bucketLength = thrust::raw_pointer_cast(&bucketLengthVector[0]);
    float4 * bucketTable = thrust::raw_pointer_cast(&bucketTableVector[0]);

    cudaMalloc((void **)&up, bucketSize*sizeof(float3));
    cudaMalloc((void **)&down, bucketSize*sizeof(float3));
    cudaMalloc((void **)&result, sample_number*sizeof(float3));


    buildKDTree(bucketIndex, bucketLength, ptr, kd_high, up, down);

    end_build_t = clock();
    //fps
    sample(bucketIndex, bucketLength, ptr, bucketSize, up, down, bucketTable, sample_number, result);
    end_t = clock();
    start_t = start_build_t;

    thrust::copy(dPoints.begin(), dPoints.end(), point_data.begin());

    //read point
    std::ofstream fout("kdtree.txt");
    if (!fout.is_open()) {
        std::cout << "file failed to open" << std::endl;
        return 0;
    }
    for(const auto& point : point_data){
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



////whichDim simply means which dimension we are sorting by, 0 = x, 1 = y, 2 = z
//int constructKD(thrust::device_vector<float3>& dPoints, int begin, int end,	compare_float3_x& comp_x, compare_float3_y& comp_y ,compare_float3_z& comp_z, int numLevels) {
//    int whichDim = 0;
//    thrust::detail::normal_iterator<thrust::device_ptr<float3>> maxx = thrust::max_element(dPoints.begin() + begin,
//                                                                                           dPoints.begin() + end,
//                                                                                           comp_x);
//    thrust::detail::normal_iterator<thrust::device_ptr<float3>> maxy = thrust::max_element(dPoints.begin() + begin,
//                                                                                           dPoints.begin() + end,
//                                                                                           comp_y);
//    thrust::detail::normal_iterator<thrust::device_ptr<float3>> maxz = thrust::max_element(dPoints.begin() + begin,
//                                                                                           dPoints.begin() + end,
//                                                                                           comp_z);
//    thrust::detail::normal_iterator<thrust::device_ptr<float3>> minx = thrust::min_element(dPoints.begin() + begin,
//                                                                                           dPoints.begin() + end,
//                                                                                           comp_x);
//    thrust::detail::normal_iterator<thrust::device_ptr<float3>> miny = thrust::min_element(dPoints.begin() + begin,
//                                                                                           dPoints.begin() + end,
//                                                                                           comp_y);
//    thrust::detail::normal_iterator<thrust::device_ptr<float3>> minz = thrust::min_element(dPoints.begin() + begin,
//                                                                                           dPoints.begin() + end,
//                                                                                           comp_z);
//
//    float rangeX = static_cast<float3>(*maxx).x - static_cast<float3>(*minx).x;
//    float rangeY = static_cast<float3>(*maxy).y - static_cast<float3>(*miny).y;
//    float rangeZ = static_cast<float3>(*maxz).z - static_cast<float3>(*minz).z;
//
//    if (rangeX > rangeY && rangeX > rangeZ) {
//        whichDim = 0;
//    } else {
//        if (rangeY > rangeX && rangeY > rangeZ) {
//            whichDim = 1;
//        } else {
//            if (rangeZ > rangeX && rangeZ > rangeY) {
//                whichDim = 2;
//            } else {
//                whichDim = 0;
//            }
//        }
//    }
//    switch(whichDim)
//    {
//        case 0:
//            thrust::sort(dPoints.begin()+begin, dPoints.begin()+end, comp_x);
//            break;
//        case 1:
//            thrust::sort(dPoints.begin()+begin, dPoints.begin()+end, comp_y);
//            break;
//        case 2:
//            thrust::sort(dPoints.begin()+begin, dPoints.begin()+end, comp_z);
//            break;
//        default:
//            printf("You shouldn't be here; i.e. wrong case number");
//            break;
//    }
//
//    numLevels--;
//    int numOfPoints = end-begin;
//    int lowerBound = ((int)numOfPoints/2)+begin;
//    int upperBound = ((int)numOfPoints/2)+1+begin;
//    int toReturn=0;
//    if(numLevels>0)
//    {
//        toReturn=constructKD(dPoints, begin, lowerBound, comp_x, comp_y, comp_z, numLevels);
//        toReturn=constructKD(dPoints, upperBound, end, comp_x, comp_y, comp_z, numLevels);
//    }
//    return toReturn;
//
//}
//int constructKD(thrust::device_vector<float3>& dPoints, int whichDim, int begin, int end,	compare_float3_x& comp_x, compare_float3_y& comp_y ,compare_float3_z& comp_z, int numLevels) {
//
//    switch(whichDim)
//    {
//        case 0:
//            thrust::sort(dPoints.begin()+begin, dPoints.begin()+end, comp_x);
//            break;
//        case 1:
//            thrust::sort(dPoints.begin()+begin, dPoints.begin()+end, comp_y);
//            break;
//        case 2:
//            thrust::sort(dPoints.begin()+begin, dPoints.begin()+end, comp_z);
//            break;
//        default:
//            printf("You shouldn't be here; i.e. wrong case number");
//            break;
//    }
//
//    numLevels--;
//    int numOfPoints = end-begin;
//    int lowerBound = ((int)numOfPoints/2)+begin;
//    int upperBound = ((int)numOfPoints/2)+1+begin;
//    int toReturn=0;
//    if(numLevels>0)
//    {
//        toReturn=constructKD(dPoints, (whichDim + 1) % 3 , begin, lowerBound, comp_x, comp_y, comp_z, numLevels);
//        toReturn=constructKD(dPoints, (whichDim + 1) % 3 ,upperBound, end, comp_x, comp_y, comp_z, numLevels);
//    }
//    return toReturn;
//}