#pragma once
#include "common.h"
#include <cuda_runtime_api.h>
namespace nvinfer1
{
namespace plugin
{
    int Atan2Inference(cudaStream_t stream, int n,const void* input1,const void* input2,void* output);
}
}