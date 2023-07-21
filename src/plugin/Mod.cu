#include "plugin/Mod.h"

namespace nvinfer1
{
namespace plugin
{
    template <unsigned nthdsPerCTA>
    __launch_bounds__(nthdsPerCTA) __global__
        void FmodKernel(const int n,int b, const int* input, int* output)
    {
        int th_i = blockIdx.x * blockDim.x + threadIdx.x;
        if (th_i >= n) return; 
        output[th_i] = input[th_i] % b;
    }

    int FmodGPU(cudaStream_t stream, const int n, int b, const void* input, void* output)
    {
        const int BS = 512;
        const int GS = (n + BS -1) / BS;
        int* test = (int*)input;
        FmodKernel<BS><<<GS,BS,0,stream>>>(n,b,
                    (const int*)input,(int*)output);
        return 0;
    }

    int ModInference(cudaStream_t stream, int n, int b, const void* input,void* output)
    {
        return FmodGPU(stream,n, b,(const int*)input,(int*)output);
    }
}
}

