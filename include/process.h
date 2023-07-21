#pragma once
#include "centerpoint.h"
#include <pcl/point_cloud.h>
#include <pcl/io/ply_io.h>
#include <pcl/common/common.h>

typedef pcl::PointXYZ Point;
typedef pcl::PointXYZI PointI;
typedef pcl::PointCloud<PointI> PointICloud;
typedef PointICloud::Ptr PointICloudPtr;
typedef PointICloud::ConstPtr PointICloudConstPtr;

int PointCloud2Array(float *&points_array, PointICloud & points);

Eigen::MatrixXf center2corner_3d(float l,float w,float h,float r);