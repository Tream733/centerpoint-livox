#pragma once
#include "common.h"
#include <cuda_runtime_api.h>
namespace nvinfer1
{
namespace plugin
{
    int ModInference(cudaStream_t stream, int n,int b,const void* input,void* output);
}
}