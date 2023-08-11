#include <stdio.h>
#include "preprocess.h"
#include "common.h"

__device__ void warpReduce(volatile float* sdata, int ith_point, int axis) {
  sdata[ith_point * blockDim.y + axis] +=
      sdata[(ith_point + 8) * blockDim.y + axis];
  sdata[ith_point * blockDim.y + axis] +=
      sdata[(ith_point + 4) * blockDim.y + axis];
  sdata[ith_point * blockDim.y + axis] +=
      sdata[(ith_point + 2) * blockDim.y + axis];
  sdata[ith_point * blockDim.y + axis] +=
      sdata[(ith_point + 1) * blockDim.y + axis];
}

__global__ void bool_map_kernel(const float* dev_points,
                        const int in_num_points,
                        const int num_point_feature,
                        const int num_threads,
                        const int grid_x_size,
                        const int grid_y_size,
                        const int grid_z_size,
                        const float min_x_range,
                        const float min_y_range,
                        const float min_z_range,
                        const float max_x_range,
                        const float max_y_range,
                        const float max_z_range,
                        const float voxel_x_size,
                        const float voxel_y_size,
                        const float voxel_z_size,
                        float* feature_map)
{
    int th_i = blockIdx.x * blockDim.x + threadIdx.x;
    if (th_i >= in_num_points) return; 
    
    float x = dev_points[th_i * num_point_feature + 0];
    float y = dev_points[th_i * num_point_feature + 1];
    float z = dev_points[th_i * num_point_feature + 2] + 1.8;
    if ((x - min_x_range) < EPSINON || (x-max_x_range)>EPSINON || (y - min_y_range) < EPSINON ||
     (y-max_y_range)>EPSINON || (z - min_z_range) < EPSINON || (z - max_z_range) > EPSINON ) return;
    
    unsigned int x_coor = floor((x - min_x_range)/voxel_x_size);
    unsigned int y_coor = floor((y - min_y_range)/voxel_y_size);
    unsigned int z_coor = floor((z - min_z_range)/voxel_z_size);

    unsigned int voxel_idx = z_coor * grid_y_size * grid_x_size + y_coor * grid_x_size + x_coor;
    feature_map[voxel_idx] = 1.0;
}

PreprocessPointsCuda::PreprocessPointsCuda(const int num_threads,
        const int grid_x_size,
        const int grid_y_size,
        const int grid_z_size,
        const float min_x_range,
        const float min_y_range,
        const float min_z_range,
        const float max_x_range,
        const float max_y_range,
        const float max_z_range,
        const float voxel_x_size,
        const float voxel_y_size,
        const float voxel_z_size,
        const int num_point_feature)
        : num_threads_(num_threads),
          grid_x_size_(grid_x_size),
          grid_y_size_(grid_y_size),
          grid_z_size_(grid_z_size),
          min_x_range_(min_x_range),
          min_y_range_(min_y_range),
          min_z_range_(min_z_range),
          max_x_range_(max_x_range),
          max_y_range_(max_y_range),
          max_z_range_(max_z_range),
          voxel_x_size_(voxel_x_size),
          voxel_y_size_(voxel_y_size),
          voxel_z_size_(voxel_z_size),
          num_point_feature_(num_point_feature){
    GPU_CHECK(cudaMalloc(reinterpret_cast<void**>(&feature_map_),
                    grid_z_size_ * grid_y_size_ * grid_z_size_ * sizeof(float)));
}

PreprocessPointsCuda::~PreprocessPointsCuda() {
  GPU_CHECK(cudaFree(feature_map_));
}

void PreprocessPointsCuda::DoPreprocessPointsCuda(const float* dev_points, const int in_num_points,
                                float* feature_map)
{
    // initialize paraments
    GPU_CHECK(cudaMemset(feature_map_, 0,grid_z_size_ * grid_y_size_ * grid_z_size_ * sizeof(float)));
    int mum_block = DIVUP(in_num_points, num_threads_);
    bool_map_kernel<<< mum_block,num_threads_>>>(dev_points,in_num_points,num_point_feature_,num_threads_,grid_x_size_,
                grid_y_size_,grid_z_size_,min_x_range_,min_y_range_,min_z_range_,(max_x_range_-0.01),
                (max_y_range_ - 0.01),(max_z_range_-0.01),voxel_x_size_,voxel_y_size_,voxel_z_size_,feature_map);
}