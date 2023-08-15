/*
3D IoU Calculation and Rotated NMS(modified from 2D NMS written by others)
Written by Shaoshuai Shi
All Rights Reserved 2019-2022.
*/
#pragma once

#include <iostream>
#include "common.h"
#include <vector>

// #define DEBUG
const int THREADS_PER_BLOCK = 16;
const int THREADS_PER_BLOCK_NMS = sizeof(unsigned long long) * 8;
const float EPS = 1e-8;



class Iou3dNmsCuda {
 public:
  Iou3dNmsCuda(const float nms_overlap_thresh);
  ~Iou3dNmsCuda() = default;

  int DoIou3dNms(const int box_num_pre,
                 const float* res_box,
                 long* host_keep_data);
  
  void nmsNormalLauncher(const float *boxes, unsigned long long * mask, 
  int boxes_num, float nms_overlap_thresh);

 private:
  float nms_overlap_thresh_;
};
