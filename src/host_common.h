//
// Created by hanm on 22-7-5.
//

#ifndef FPS_GPU_HOST_COMMON_H
#define FPS_GPU_HOST_COMMON_H


struct Point{
    float pos[3];
    Point(float x,float y, float z){
        this->pos[0] = x;
        this->pos[1]= y;
        this->pos[2] = z;
    }
};


#endif //FPS_GPU_HOST_COMMON_H
