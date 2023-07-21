#include "process.h"

int PointCloud2Array(float *&points_array, PointICloud  & points){
    int size = points.size();
    points_array = new float[size * 4];
    size_t i ;
    for (i = 0; i< size; i++) {
        PointI pointi;
        pointi = points.at(i);
        points_array[4*i] = pointi.x;
        points_array[4*i + 1] = pointi.y;
        points_array[4*i + 2] = pointi.z;
        points_array[4*i + 3] = pointi.intensity;
    }
    return size;
}

Eigen::MatrixXf center2corner_3d(float l,float w,float h,float r)
{
    Eigen::MatrixXf matCorner(3,8);
    Eigen::MatrixXf matR(3,3);
    matCorner<< -l/2, -l/2, l/2, l/2, -l/2, -l/2, l/2, l/2,
                w/2, -w/2, -w/2, w/2, -w/2, -w/2, -w/2, w/2,
                -h/2, -h/2, -h/2, -h/2, h/2, h/2, h/2, h/2;
    float c = cos(r);
    float s = sin(r);
    matR<< c,-s,0,s,c,0,0,0,1;
    Eigen::MatrixXf matCorner_3d = matR*matCorner;
    return matCorner_3d;
}