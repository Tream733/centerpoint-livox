/*
3D IoU Calculation and Rotated NMS(modified from 2D NMS written by others)
Written by Shaoshuai Shi
All Rights Reserved 2019-2022.
*/
#pragma once

#include <iostream>

// #define DEBUG
const int THREADS_PER_BLOCK = 16;
const int THREADS_PER_BLOCK_NMS = sizeof(unsigned long long) * 8;
const float EPS = 1e-8;



class Iou3dNmsCuda {
 public:
  Iou3dNmsCuda(
               const float nms_overlap_thresh);
  ~Iou3dNmsCuda() = default;

  int DoIou3dNms(const int box_num_pre,
                 const float* res_box,
                 const int* res_sorted_indices,
                 long* host_keep_data);

 private:
  const float nms_overlap_thresh_;
};
