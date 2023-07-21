#pragma once
#include <memory>
#include <assert.h>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <vector>
#include <yaml-cpp/yaml.h>

#include "common.h"
#include "NvInfer.h"
#include "NvInferPlugin.h"
#include "NvOnnxParser.h"
#include "preprocess.h"
#include "postprocess.h"
#include "logger.h"

// typedef struct float11 { float val[11]; } float11;

class CenterPoint {
  private:
    // initialize in initializer list
    const bool use_onnx_;
    bool enable_debug_ = false;
    std::string trt_mode_ = "fp16";
    const std::string rpn_file_;
    const std::string centerpoint_config_;
    // voxel size 
    float kVoxelXSize_;
    float kVoxelYSize_;
    float kVoxelZSize_;
    // Point cloud range
    float kMinXRange_;
    float kMinYRange_;
    float kMinZRange_;
    float kMaxXRange_;
    float kMaxYRange_;
    float kMaxZRange_;
    // hyper parameters
    int kNumClass_;
    int kNumPointFeature_ = 4; // [x,y,z,i]
    // perprocess
    int kGridXSize_;
    int kGridYSize_;
    int kGridZSize_;
    int kBatchSize_;
    int kNumThreads_ = 64;
    int kNumBoxCorners_ = 8;

    int rpn_input_size_;

    int box_range_;
    int score_range_;
    int label_range_;
    int max_obj_;

    int head_x_size_;
    int head_y_size_;
    int nms_pre_max_size_;
    int nms_post_max_size_;
    std::vector<float> score_thresh_;
    float nms_overlap_thresh_;

    float* bool_map_;
    void *rpn_buf_[4];

    int head_map_size_;
    
    std::map<std::string, int> head_map_;
   
    std::unique_ptr<PreprocessPointsCuda> preprocess_points_cuda_ptr_;
    std::unique_ptr<PostprocessCuda> postprocess_cuda_ptr_;

    Logger g_logger_;
    nvinfer1::ICudaEngine *rpn_engine_;
    nvinfer1::IExecutionContext *rpn_context_;

    void DeviceMemoryMalloc();

    void SetDeviceMemoryToZero();

    void InitParam();

    void InitTRT(const bool use_onnx);

    void OnnxToTRTModel(const std::string &model_file,
                        nvinfer1::ICudaEngine ** engine_ptr);

    void EngineToTRTModel(const std::string &engine_file,
                        nvinfer1::ICudaEngine **engine_ptr);
    
    void Preprocess(const float *in_points_array, const int in_num_points);
    
  public:
    CenterPoint(const bool use_onnx,const std::string rpn_file,
              const std::string centerpoint_config);
    ~CenterPoint();

    bool DoInference(const float *in_points_array, const int in_num_points,
                std::vector<Box> &out_detection);
};