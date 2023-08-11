#pragma once

#include <map>
#include <memory>
#include <vector>
#include <math.h>
#include <iostream>
#include <vector>
// #define BLOCK_SIZE 256
#include "iou3d_nms.h"
#include "common.h"
#include "preprocess.h"

static const int kBoxBlockSize = 7;

struct Box {
  float x;
  float y;
  float z;
  float l;
  float w;
  float h;
  float r;
  float vx = 0.0f;  // optional
  float vy = 0.0f;  // optional
  float score;
  int label;
  bool is_drop;  // for nms
};

class PostprocessCuda {
 public:
  PostprocessCuda(  const float min_x_range,
                    const float min_y_range,
                    const float min_z_range,
                    const float max_x_range,
                    const float max_y_range,
                    const float max_z_range,
                    int max_obj,
                    float nms_overlap_thresh);
  ~PostprocessCuda();

  void DoPostprocessCuda(
                          float* pred_scores,
                          int* pred_labels,
                          float* pred_bbox,
                          int max_obj,
                          std::vector<Box> &out_detections);
 private:
    const float min_x_range_;
    const float min_y_range_;
    const float min_z_range_;
    const float max_x_range_;
    const float max_y_range_;
    const float max_z_range_;
    int max_obj_;
    float* dev_res_box_ ;
    float* dev_res_score_ ;
    int* dev_res_cls_ ;
    int* dev_res_box_num_;
    int* dev_res_sorted_indices_ ;
    float* host_res_box_ ;
    float* host_res_score_;
    int* host_res_cls_ ;
    int* host_res_sorted_indices_ ;
    long* host_keep_data_ ;
    float nms_overlap_thresh_;
    std::vector<Box> output_;
    float score_thresh_[3] = {0.2f,0.5f,0.5f};
    std::unique_ptr<Iou3dNmsCuda> iou3d_nms_cuda_;
};