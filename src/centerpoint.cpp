#include "centerpoint.h"

#include <chrono>
#include <cstring>
#include <iostream>

using std::chrono::duration;
using std::chrono::high_resolution_clock;

void CenterPoint::InitParam(){
    YAML::Node params = YAML::LoadFile(centerpoint_config_);
    // config
    trt_mode_ = params["TRT_MODE"].as<std::string>();
    enable_debug_ = params["EnableDebug"].as<bool>();
    // data config
    kBatchSize_ = params["DATA_CONFIG"]["LOAD_BATCH"].as<int>();
    kNumPointFeature_ = params["DATA_CONFIG"]["LOAD_DIM"].as<int>();
    kVoxelXSize_ =
      params["DATA_CONFIG"]["DATA_PROCESSOR"]["VOXEL_SIZE"][0].as<float>();
    kVoxelYSize_ =
      params["DATA_CONFIG"]["DATA_PROCESSOR"]["VOXEL_SIZE"][1].as<float>();
    kVoxelZSize_ =
      params["DATA_CONFIG"]["DATA_PROCESSOR"]["VOXEL_SIZE"][2].as<float>();
    kMinXRange_ = params["DATA_CONFIG"]["POINT_CLOUD_RANGE"][0].as<float>();
    kMinYRange_ = params["DATA_CONFIG"]["POINT_CLOUD_RANGE"][1].as<float>();
    kMinZRange_ = params["DATA_CONFIG"]["POINT_CLOUD_RANGE"][2].as<float>();
    kMaxXRange_ = params["DATA_CONFIG"]["POINT_CLOUD_RANGE"][3].as<float>();
    kMaxYRange_ = params["DATA_CONFIG"]["POINT_CLOUD_RANGE"][4].as<float>();
    kMaxZRange_ = params["DATA_CONFIG"]["POINT_CLOUD_RANGE"][5].as<float>();
    // model
    rpn_input_size_ = params["MODEL"]["BACKBONE_2D"]["INPUT_CHANNELS"].as<int>();
    nms_pre_max_size_ = params["MODEL"]["POST_PROCESSING"]["NMS_CONFIG"]["NMS_PRE_MAXSIZE"].as<int>();
    nms_post_max_size_ = params["MODEL"]["POST_PROCESSING"]["NMS_CONFIG"]["NMS_POST_MAXSIZE"].as<int>();
    kGridXSize_ =
      static_cast<int>((kMaxXRange_ - kMinXRange_) / kVoxelXSize_);  // 1120
    kGridYSize_ =
      static_cast<int>((kMaxYRange_ - kMinYRange_) / kVoxelYSize_);  // 448
    kGridZSize_ = static_cast<int>((kMaxZRange_ - kMinZRange_) / kVoxelZSize_);  // 30
    // label
    kNumClass_ = params["MODEL"]["DENSE_HEAD"]["NUM_CLASS"].as<int>();
    for(int i =0; i<3; i++) {
      score_thresh_.push_back(params["MODEL"]["POST_PROCESSING"]["SCORE_THRESH"][i].as<float>());
    }
    nms_overlap_thresh_ = params["MODEL"]["POST_PROCESSING"]["NMS_CONFIG"]["NMS_THRESH"].as<float>();   
    std::vector<std::string> head_names{"center", "center_z", "dim", "rot","hm"};
    for (auto name : head_names) {
      head_map_[name] = params["MODEL"]["DENSE_HEAD"]["SEPARATE_HEAD_CFG"]["HEAD_DICT"][name]["out_channels"].as<int>();
    }
    // output
    head_x_size_ = kGridXSize_;
    head_y_size_ = kGridYSize_;
    head_map_size_ = head_x_size_ * head_y_size_;
    box_range_ = params["MODEL"]["POST_PROCESSING"]["BOX_SIZE"].as<int>();
    score_range_ = params["MODEL"]["DENSE_HEAD"]["TARGET_ASSIGNER_CONFIG"]["SCORE_DIM"].as<int>();
    label_range_ = params["MODEL"]["DENSE_HEAD"]["TARGET_ASSIGNER_CONFIG"]["LABEL_DIM"].as<int>();
    max_obj_ = params["MODEL"]["DENSE_HEAD"]["TARGET_ASSIGNER_CONFIG"]["NUM_MAX_OBJS"].as<int>();
}

CenterPoint::CenterPoint(const bool use_onnx,const std::string rpn_file,
              const std::string centerpoint_config)
    : use_onnx_(use_onnx),
      rpn_file_(rpn_file),
      centerpoint_config_(centerpoint_config) {
  InitParam();
  InitTRT(use_onnx_);
  DeviceMemoryMalloc();
  cudaDeviceSynchronize();
  // init preprocess_points_cuda_ptr_
  preprocess_points_cuda_ptr_.reset(new PreprocessPointsCuda(
      kNumThreads_, kGridXSize_, kGridYSize_, kGridZSize_, kMinXRange_, kMinYRange_, kMinZRange_, 
      kMaxXRange_, kMaxYRange_, kMaxZRange_,kVoxelXSize_, kVoxelYSize_,kVoxelZSize_,kNumPointFeature_));

  const float float_min = std::numeric_limits<float>::lowest();
  const float float_max = std::numeric_limits<float>::max();
  // init postprocess_cuda_ptr_
  cudaDeviceSynchronize();
  postprocess_cuda_ptr_.reset(new PostprocessCuda(
      kMinXRange_, kMinYRange_, kMinZRange_, kMaxXRange_, kMaxYRange_, kMaxZRange_,max_obj_,nms_overlap_thresh_));
  cudaDeviceSynchronize();
}

void CenterPoint::DeviceMemoryMalloc() {
  // for boolmap 
  GPU_CHECK(cudaMalloc(reinterpret_cast<void**>(&bool_map_),
                       kGridXSize_ * kGridYSize_ * kGridZSize_ * sizeof(float)));
  // rpn
  GPU_CHECK(
      cudaMalloc(&rpn_buf_[0],  
                        kGridXSize_ * kGridYSize_ * kGridZSize_ * sizeof(float)));
  // scores
  GPU_CHECK(cudaMalloc(&rpn_buf_[1],
                      score_range_ * max_obj_*  sizeof(float)));
  // labels
  GPU_CHECK(cudaMalloc(&rpn_buf_[2],
                       label_range_ * max_obj_ * sizeof(int)));
  // bbox_preds
  GPU_CHECK(cudaMalloc(&rpn_buf_[3],
                       box_range_ * max_obj_ * sizeof(float)));
}

void CenterPoint::SetDeviceMemoryToZero(){
  GPU_CHECK(cudaMemset(bool_map_, 0, kGridXSize_ * kGridYSize_ * kGridZSize_ * sizeof(float)));
  GPU_CHECK(cudaMemset(rpn_buf_[0], 0, kGridXSize_* kGridYSize_ * kGridZSize_ * sizeof(float)));
  GPU_CHECK(cudaMemset(rpn_buf_[1], 0, score_range_* max_obj_ * sizeof(float)));
  GPU_CHECK(cudaMemset(rpn_buf_[2], 0, label_range_* max_obj_ * sizeof(int)));
  GPU_CHECK(cudaMemset(rpn_buf_[3], 0, box_range_* max_obj_ * sizeof(float)));
}

void CenterPoint::InitTRT(const bool use_onnx)
{
  if(use_onnx) {
    OnnxToTRTModel(rpn_file_,&rpn_engine_);
  } else {
    EngineToTRTModel(rpn_file_, &rpn_engine_);
  }
  if (rpn_engine_ == nullptr) {
    std::cerr<< "Failed to load ONNX file.";
  }

  // create execution context from the engine
  rpn_context_ = rpn_engine_->createExecutionContext();
  if (rpn_context_ == nullptr) {
    std::cerr<< " Failed to create TensorRT Exection Context.";
  }
}

void CenterPoint::OnnxToTRTModel(const std::string &model_file,nvinfer1::ICudaEngine **engine_ptr) 
{
  std::string model_cache = model_file + ".cache";
  std::fstream trt_cache(model_cache, std::ifstream::in);
  if (!trt_cache.is_open()) {
    std::cout << "Building TRT engine." << std::endl;
    // create the builder
    const auto explicit_batch =
        static_cast<uint32_t>(kBatchSize_) << static_cast<uint32_t>(
            nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    nvinfer1::IBuilder *builder = nvinfer1::createInferBuilder(g_logger_);
    nvinfer1::INetworkDefinition *network = builder->createNetworkV2(explicit_batch);

    // parse onnx model
    int verbosity = static_cast<int>(nvinfer1::ILogger::Severity::kWARNING);
    auto parser = nvonnxparser::createParser(*network, g_logger_);
    if (!parser->parseFromFile(model_file.c_str(), verbosity)) {
      std::string msg("failed to parse onnx file");
      g_logger_.log(nvinfer1::ILogger::Severity::kERROR, msg.c_str());
      exit(EXIT_FAILURE);
    }

    // Build the engine
    builder->setMaxBatchSize(kBatchSize_);
    // builder->setHalf2Mode(true);
    nvinfer1::IBuilderConfig *config = builder->createBuilderConfig();
    // config->setMaxWorkspaceSize(16*(1 << 30));
    bool has_fast_fp16 = builder->platformHasFastFp16();
    if (trt_mode_ == "fp16" && has_fast_fp16) {
      std::cout << "the platform supports Fp16, use Fp16." << std::endl;
      config->setFlag(nvinfer1::BuilderFlag::kFP16);
    }
    nvinfer1::ICudaEngine *engine = builder->buildEngineWithConfig(*network, *config);
    if (engine == nullptr) {
      std::cerr << ": engine init null!" << std::endl;
      exit(-1);
    }

    // serialize the engine, then close everything down
    auto model_stream = (engine->serialize());
    std::fstream trt_out(model_cache, std::ifstream::out);
    if (!trt_out.is_open()) {
      std::cout << "Can't store trt cache.\n";
      exit(-1);
    }
    trt_out.write((char *)model_stream->data(), model_stream->size());
    trt_out.close();
    model_stream->destroy();

    *engine_ptr = engine;
    parser->destroy();
    network->destroy();
    config->destroy();
    builder->destroy();
  } else {
    std::cout << "Load TRT cache." << std::endl;
    char *data;
    unsigned int length;

    // get length of file:
    trt_cache.seekg(0, trt_cache.end);
    length = trt_cache.tellg();
    trt_cache.seekg(0, trt_cache.beg);

    data = (char *)malloc(length);
    if (data == NULL) {
      std::cout << "Can't malloc data.\n";
      exit(-1);
    }

    trt_cache.read(data, length);
    // create context
    auto runtime = nvinfer1::createInferRuntime(g_logger_);
    if (runtime == nullptr) {
      std::cerr << ": runtime null!" << std::endl;
      exit(-1);
    }
    // plugin_ = nvonnxparser::createPluginFactory(g_logger_);
    nvinfer1::ICudaEngine *engine =
        (runtime->deserializeCudaEngine(data, length, 0));
    if (engine == nullptr) {
      std::cerr << ": engine null!" << std::endl;
      exit(-1);
    }
    *engine_ptr = engine;
    free(data);
    trt_cache.close();
  }
}


void CenterPoint::EngineToTRTModel(const std::string &engine_file,
                                   nvinfer1::ICudaEngine **engine_ptr) {
  int verbosity = static_cast<int>(nvinfer1::ILogger::Severity::kWARNING);
  std::stringstream gieModelStream;
  gieModelStream.seekg(0, gieModelStream.beg);
  std::ifstream cache(engine_file);
  gieModelStream << cache.rdbuf();
  cache.close();
  nvinfer1::IRuntime *runtime = nvinfer1::createInferRuntime(g_logger_);
  if (runtime == nullptr) {
    std::string msg("failed to build runtime parser");
    g_logger_.log(nvinfer1::ILogger::Severity::kERROR, msg.c_str());
    exit(EXIT_FAILURE);
  }
  gieModelStream.seekg(0, std::ios::end);
  const int modelSize = gieModelStream.tellg();
  gieModelStream.seekg(0, std::ios::beg);
  void *modelMem = malloc(modelSize);
  gieModelStream.read((char *)modelMem, modelSize);
  nvinfer1::ICudaEngine *engine = runtime->deserializeCudaEngine(modelMem, modelSize, NULL);
  if (engine == nullptr) {
    std::string msg("failed to build engine parser");
    g_logger_.log(nvinfer1::ILogger::Severity::kERROR, msg.c_str());
    exit(EXIT_FAILURE);
  }
  *engine_ptr = engine;
  for (int bi = 0; bi < engine->getNbBindings(); bi++) {
    if (engine->bindingIsInput(bi) == true)
      printf("Binding %d (%s): Input. \n", bi, engine->getBindingName(bi));
    else
      printf("Binding %d (%s): Output. \n", bi, engine->getBindingName(bi));
  }
}

bool CenterPoint::DoInference(const float *in_points_array, const int in_num_points,
                std::vector<Box> &out_detection) 
{
  SetDeviceMemoryToZero();
  cudaDeviceSynchronize();
  // [step 1] : load pointcloud
  auto load_start = high_resolution_clock::now();
  float *dev_points;
  GPU_CHECK(cudaMalloc(reinterpret_cast<void**>(&dev_points), in_num_points * kNumPointFeature_ * sizeof(float)));
  GPU_CHECK(cudaMemcpy(dev_points, in_points_array, in_num_points * kNumPointFeature_ * sizeof(float), cudaMemcpyHostToDevice));
  auto load_end = high_resolution_clock::now();

  // [step 2] : preprocess
  auto preprocess_start = high_resolution_clock::now();
  preprocess_points_cuda_ptr_->DoPreprocessPointsCuda(
                                  dev_points, in_num_points,bool_map_
                          );
  cudaDeviceSynchronize();
  auto preprocess_end = high_resolution_clock::now();
  // if (enable_debug_) {
  //   DEVICE_SAVE<float>(dev_points,  in_num_points , 4,"00_points.txt");
  //   DEVICE_SAVE<float>(bool_map_, 13440, 1120, "01_boolmap.txt");
  // }

  // [step 3] : rpn forward
  cudaStream_t stream;
  GPU_CHECK(cudaStreamCreate(&stream));
  auto rpn_start = high_resolution_clock::now();
  GPU_CHECK(cudaMemcpyAsync(rpn_buf_[0], bool_map_, kGridZSize_*kGridYSize_*kGridXSize_ * sizeof(float),cudaMemcpyDeviceToDevice, stream));
  rpn_context_->enqueueV2(rpn_buf_, stream, nullptr);
  cudaDeviceSynchronize();
  auto rpn_end = high_resolution_clock::now();
  // if (enable_debug_) {
  //   DEVICE_SAVE<float>((float *)rpn_buf_[1], max_obj_, score_range_, "003_output.txt");
  //   DEVICE_SAVE<int>((int *)rpn_buf_[2], max_obj_, label_range_, "004_output.txt");
  //   DEVICE_SAVE<float>((float *)rpn_buf_[3], max_obj_, box_range_, "002_output.txt");
  // }

  // [step 4 ]: postprocess
  auto postprocess_start = high_resolution_clock::now();
  postprocess_cuda_ptr_->DoPostprocessCuda(
      reinterpret_cast<float *>(rpn_buf_[1]),    // score
      reinterpret_cast<int *>(rpn_buf_[2]),    // label
      reinterpret_cast<float *>(rpn_buf_[3]),    // preds_bbox
      max_obj_,
      out_detection);
  cudaDeviceSynchronize();
  auto postprocess_end = high_resolution_clock::now();

  // release the stream and the buffers
  duration<double> preprocess_cost = preprocess_end - preprocess_start;
  duration<double> rpn_cost = rpn_end - rpn_start;
  duration<double> postprocess_cost = postprocess_end - postprocess_start;
  duration<double> centerpoint_cost = postprocess_end - preprocess_start;
  std::cout << "------------------------------------" << std::endl;
    std::cout << setiosflags(ios::left) << setw(14) << "Module" << setw(12)
            << "Time" << resetiosflags(ios::left) << std::endl;
  std::cout << "------------------------------------" << std::endl;
  std::string Modules[] = {"Preprocess", "Rpn", "Postprocess", "Summary"};
  double Times[] = {preprocess_cost.count(),  rpn_cost.count(),
                postprocess_cost.count(), centerpoint_cost.count()};

  for (int i = 0; i < 4; ++i) {
    std::cout << setiosflags(ios::left) << setw(14) << Modules[i] << setw(8)
              << Times[i] * 1000 << " ms" << resetiosflags(ios::left)
              << std::endl;
  }
  std::cout << "------------------------------------" << std::endl;
  cudaStreamDestroy(stream);
  GPU_CHECK(cudaFree(dev_points));
}

CenterPoint::~CenterPoint() {
  // boolmap
  GPU_CHECK(cudaFree(bool_map_));
  // rpn forward
  GPU_CHECK(cudaFree(rpn_buf_[0]));
  GPU_CHECK(cudaFree(rpn_buf_[1]));
  GPU_CHECK(cudaFree(rpn_buf_[2]));
  GPU_CHECK(cudaFree(rpn_buf_[3]));
  rpn_context_->destroy();
  rpn_engine_->destroy();
}
