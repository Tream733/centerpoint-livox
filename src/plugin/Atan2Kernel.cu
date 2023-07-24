#include "plugin/Atan2Kernel.h"

namespace nvinfer1
{
namespace plugin
{
    template <unsigned nthdsPerCTA>
    __launch_bounds__(nthdsPerCTA) __global__
        void FmodKernel(const int n, const float* input1, const float* input2, float* output)
    {
        int th_i = blockIdx.x * blockDim.x + threadIdx.x;
        if (th_i >= n) return; 
        output[th_i] = std::atan2(input1[th_i],input2[th_i]);
    }

    int FmodGPU(cudaStream_t stream, const int n, const void* input1, const void* input2, void* output)
    {
        const int BS = 512;
        const int GS = (n + BS -1) / BS;
        FmodKernel<BS><<<GS,BS,0,stream>>>(n,
                    (const float*)input1, (const float*)input2,(float*)output);
        return 0;
    }

    int Atan2Inference(cudaStream_t stream, int n, const void* input1, const void* input2,void* output)
    {
        return FmodGPU(stream,n,(const float*)input1, (const float*)input2,(float*)output);
    }
}
}

