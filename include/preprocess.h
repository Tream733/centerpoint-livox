#pragma once

/**
 * @author TXY
*/

const float EPSINON = 0.000001f;

class PreprocessPointsCuda
{
private:
    /* data */
    const int num_threads_;
    const int grid_x_size_;
    const int grid_y_size_;
    const int grid_z_size_;
    const int num_point_feature_;
    const float min_x_range_;
    const float min_y_range_;
    const float min_z_range_;
    const float max_x_range_;
    const float max_y_range_;
    const float max_z_range_;
    const float voxel_x_size_;
    const float voxel_y_size_;
    const float voxel_z_size_;
    float * feature_map_;
public:
    /**
     * @Constructor
     * @param[in] num_threads Number of threads when launching cuda kernel
     * 
     * @param[in] grid_x_size Number of voxel in x-coordinate
     * @param[in] grid_y_size Number of voxel in y-coordinate
     * @param[in] grid_z_size Number of voxel in z-coordinate
     * 
     * @param[in] min_x_range Minimum x value for point cloud
     * @param[in] min_y_range Minimum y value for point cloud
     * @param[in] min_z_range Minimum z value for point cloud
     * 
     * @param[in] max_x_range Maximum x value for point cloud
     * @param[in] max_y_range Maximum y value for point cloud
     * @param[in] max_z_range Maximum z value for point cloud
     * 
     * @param[in] voxel_x_size Size of x-dimension for a voxel
     * @param[in] voxel_y_size Size of y-dimension for a voxel
     * @param[in] voxel_z_size Size of z-dimension for a voxel
     * 
    */

    PreprocessPointsCuda(const int num_threads,
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
                        const int num_point_feature);

    ~PreprocessPointsCuda();



    void DoPreprocessPointsCuda(const float* dev_points, const int in_num_points,
                                float* feature_map);
};

