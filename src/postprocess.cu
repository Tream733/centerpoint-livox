#include <stdio.h>
#include "postprocess.h"
#include <thrust/count.h>
#include <thrust/device_vector.h>
#include <thrust/gather.h>
#include <thrust/host_vector.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/transform.h>

template <typename T>
void swap_warp(T& a, T& b, T& swp) {
  swp = a;
  a = b;
  b = swp;
}
void quicksort_warp(float* score, int* index, int start, int end) {
  if (start >= end) return;
  float pivot = score[end];
  float value_swp;
  int index_swp;
  // set a pointer to divide array into two parts
  // one part is smaller than pivot and another larger
  int pointer = start;
  for (int i = start; i < end; i++) {
    if (score[i] > pivot) {
      if (pointer != i) {
        // swap score[i] with score[pointer]
        // score[pointer] behind larger than pivot
        swap_warp<float>(score[i], score[pointer], value_swp);
        swap_warp<int>(index[i], index[pointer], index_swp);
      }
      pointer++;
    }
  }
  // swap back pivot to proper position
  swap_warp<float>(score[end], score[pointer], value_swp);
  swap_warp<int>(index[end], index[pointer], index_swp);
  quicksort_warp(score, index, start, pointer - 1);
  quicksort_warp(score, index, pointer + 1, end);
  return;
}

void quicksort_kernel(float* score, int* indexes, int len) {
  quicksort_warp(score, indexes, 0, len - 1);
}


__global__ void BoxFilterKernel(const float* score_thresh,
                                const float min_x_range,
                                const float min_y_range,
                                const float min_z_range,
                                const float max_x_range,
                                const float max_y_range,
                                const float max_z_range,
                                const int max_obj,
                                const float*pred_scores,
                                const int* pred_labels,
                                const float* pred_bbox,
                                float* pf_scores,
                                int* pf_labels,
                                float* pf_bbox,
                                int* res_box_num)
{
  int th_i = blockIdx.x * blockDim.x + threadIdx.x;
  if (th_i >= max_obj) return;
  float x = pred_bbox[th_i*7 + 0]; 
  float y = pred_bbox[th_i*7 + 1];
  float z = pred_bbox[th_i*7 + 2];
  int cur_label = pred_labels[th_i];
  float cur_score = pred_scores[th_i]; 
  if ((x > min_x_range) && (x < max_x_range) && (y > min_y_range) &&
     (y < max_y_range) && (z > min_z_range) && (z < max_z_range)) {
      int flag = 1;
      int out_flag = 1;
    for(int i = 0; i< 3; i++) {
      if((cur_label!= i) ||((cur_label == i)&&(cur_score > score_thresh[i])))
      {
        flag = 1;
      } else {
        flag = 0;
      }
      out_flag = out_flag * flag;
    }
    if (out_flag) {
      int cur_box_id = atomicAdd(res_box_num,1);
      pf_scores[cur_box_id] = cur_score;
      pf_labels[cur_box_id] = cur_label;
      pf_bbox[cur_box_id* 7 + 0] = pred_bbox[th_i * 7 +0];
      pf_bbox[cur_box_id* 7 + 1] = pred_bbox[th_i * 7 +1];
      pf_bbox[cur_box_id* 7 + 2] = pred_bbox[th_i * 7 +2];
      pf_bbox[cur_box_id* 7 + 3] = pred_bbox[th_i * 7 +3];
      pf_bbox[cur_box_id* 7 + 4] = pred_bbox[th_i * 7 +4];
      pf_bbox[cur_box_id* 7 + 5] = pred_bbox[th_i * 7 +5];
      pf_bbox[cur_box_id* 7 + 6] = pred_bbox[th_i * 7 +6];
    }
  }
}

PostprocessCuda::PostprocessCuda( const float min_x_range,
                                  const float min_y_range,
                                  const float min_z_range,
                                  const float max_x_range,
                                  const float max_y_range,
                                  const float max_z_range,
                                  int max_obj,
                                  float nms_overlap_thresh)
                                  :min_x_range_(min_x_range)
                                  ,min_y_range_(min_y_range)
                                  ,min_z_range_(min_z_range)
                                  ,max_x_range_(max_x_range)
                                  ,max_y_range_(max_y_range)
                                  ,max_z_range_(max_z_range)
                                  ,max_obj_(max_obj)
                                  ,nms_overlap_thresh_(nms_overlap_thresh)
{
  std::cout<<"--------------PostprocessCuda---------------------"<<std::endl;
  GPU_CHECK(cudaMalloc((void**)&dev_res_score_, sizeof(float) * max_obj_));
  GPU_CHECK(cudaMalloc((void**)&dev_res_cls_, sizeof(int) * max_obj_));
  GPU_CHECK(cudaMalloc((void**)&dev_res_box_, sizeof(float) * max_obj_ * kBoxBlockSize));
  GPU_CHECK(cudaMalloc((void**)&dev_res_sorted_indices_, sizeof(int) * max_obj_));
  GPU_CHECK(cudaMalloc((void**)&dev_res_box_num_, sizeof(int)));
  GPU_CHECK(cudaMallocHost((void**)&host_res_box_, sizeof(float) * max_obj_ * kBoxBlockSize));
  GPU_CHECK(cudaMallocHost((void**)&host_res_score_, sizeof(float) * max_obj_));
  GPU_CHECK(cudaMallocHost((void**)&host_res_cls_, sizeof(int) * max_obj_));
  GPU_CHECK(cudaMallocHost((void**)&host_res_sorted_indices_, sizeof(int) * max_obj_));
  GPU_CHECK(cudaMallocHost((void**)&host_keep_data_, sizeof(long) * max_obj_));

  GPU_CHECK(cudaMemset(dev_res_score_, 0.f, sizeof(float) * max_obj_));
  GPU_CHECK(cudaMemset(dev_res_cls_, 0, sizeof(int) * max_obj_));
  GPU_CHECK(cudaMemset(dev_res_box_, 0.f, sizeof(float) * max_obj_ * kBoxBlockSize));
  GPU_CHECK(cudaMemset(dev_res_box_num_, 0, sizeof(int)));
  GPU_CHECK(cudaMemset(dev_res_sorted_indices_, 0, sizeof(int) * max_obj_));
  
  GPU_CHECK(cudaMemset(host_res_box_, 0.f, sizeof(float) * max_obj_ * kBoxBlockSize));
  GPU_CHECK(cudaMemset(host_res_score_, 0.f, sizeof(float) * max_obj_));
  GPU_CHECK(cudaMemset(host_res_cls_, 0, sizeof(int) * max_obj_));

  GPU_CHECK(cudaMemset(host_res_sorted_indices_, 0, sizeof(int) * max_obj_));
  GPU_CHECK(cudaMemset(host_keep_data_, 0L, sizeof(long) * max_obj_));

  iou3d_nms_cuda_.reset(new Iou3dNmsCuda(nms_overlap_thresh_));
}

void PostprocessCuda::DoPostprocessCuda(
                                        float* pred_scores,
                                        int* pred_labels,
                                        float* pred_bbox,
                                        int max_obj,
                                        std::vector<Box> &out_detections)
{
  GPU_CHECK(cudaMemset(dev_res_box_, 0.f, sizeof(float) * max_obj_ * kBoxBlockSize));
  GPU_CHECK(cudaMemset(dev_res_score_, 0.f, sizeof(float) * max_obj_));
  GPU_CHECK(cudaMemset(dev_res_cls_, 0, sizeof(int) * max_obj_));
  GPU_CHECK(cudaMemset(dev_res_box_num_, 0, sizeof(int)));
  GPU_CHECK(cudaMemset(host_keep_data_, -1, sizeof(long) * max_obj_));
  float* score_thresh;
  GPU_CHECK(cudaMalloc((void**)&score_thresh, sizeof(float) * 3));
  GPU_CHECK(cudaMemcpy(score_thresh, score_thresh_, sizeof(int), cudaMemcpyHostToDevice));

  int mum_block = DIVUP(max_obj, NUM_THREADS);
  cudaDeviceSynchronize();
  BoxFilterKernel<<<mum_block,NUM_THREADS>>>(score_thresh,min_x_range_,min_y_range_,min_z_range_,
                                (max_x_range_-0.01),(max_y_range_ - 0.01),(max_z_range_-0.01),max_obj_,
                                pred_scores,pred_labels,pred_bbox,
                                dev_res_score_,dev_res_cls_,dev_res_box_,dev_res_box_num_);
  int box_num_pre = 0;
  GPU_CHECK(cudaMemcpy(&box_num_pre, dev_res_box_num_, sizeof(int), cudaMemcpyDeviceToHost));
  if(box_num_pre > 0) {
    thrust::sequence(thrust::device, dev_res_sorted_indices_, dev_res_sorted_indices_ + box_num_pre);
    thrust::sort_by_key(thrust::device,
                        dev_res_score_,
                        dev_res_score_ + box_num_pre,
                        dev_res_sorted_indices_,
                        thrust::greater<float>());

    int box_num_post = iou3d_nms_cuda_->DoIou3dNms(box_num_pre,
                                                   dev_res_box_, 
                                                   dev_res_sorted_indices_,
                                                   host_keep_data_);

    box_num_post = box_num_post > max_obj_ ? max_obj_ : box_num_post;
    std::cout << " gets " << box_num_pre 
              << " objects before nms, and " << box_num_post
              << " after nms" << std::endl;
    GPU_CHECK(cudaMemcpy(host_res_box_, dev_res_box_, sizeof(float) * box_num_pre * kBoxBlockSize, cudaMemcpyDeviceToHost));
    GPU_CHECK(cudaMemcpy(host_res_score_, dev_res_score_, sizeof(float) * box_num_pre, cudaMemcpyDeviceToHost));
    GPU_CHECK(cudaMemcpy(host_res_cls_, dev_res_cls_, sizeof(int) * box_num_pre, cudaMemcpyDeviceToHost));
    GPU_CHECK(cudaMemcpy(host_res_sorted_indices_, dev_res_sorted_indices_, sizeof(int) * box_num_pre, cudaMemcpyDeviceToHost));

    for (auto j = 0; j < box_num_post; j++) {
      int k = host_keep_data_[j];
      int idx = host_res_sorted_indices_[k];

      Box box;
      box.x = host_res_box_[idx * kBoxBlockSize + 0];
      box.y = host_res_box_[idx * kBoxBlockSize + 1];
      box.z = host_res_box_[idx * kBoxBlockSize + 2];
      box.l = host_res_box_[idx * kBoxBlockSize + 3];
      box.w = host_res_box_[idx * kBoxBlockSize + 4];
      box.h = host_res_box_[idx * kBoxBlockSize + 5];
      box.r = host_res_box_[idx * kBoxBlockSize + 6];
      box.label = host_res_cls_[idx];
      box.score = host_res_score_[k];
      box.z -= box.h * 0.5; // bottom height
      out_detections.push_back(box);
    }
  } else return;
}

PostprocessCuda::~PostprocessCuda()
{
  GPU_CHECK(cudaFree(dev_res_box_));
  GPU_CHECK(cudaFree(dev_res_score_));
  GPU_CHECK(cudaFree(dev_res_cls_));
  GPU_CHECK(cudaFree(dev_res_sorted_indices_));
  GPU_CHECK(cudaFree(dev_res_box_num_));
  GPU_CHECK(cudaFreeHost(host_res_box_));
  GPU_CHECK(cudaFreeHost(host_res_score_));
  GPU_CHECK(cudaFreeHost(host_res_cls_));
  GPU_CHECK(cudaFreeHost(host_res_sorted_indices_));
  GPU_CHECK(cudaFreeHost(host_keep_data_));
}
