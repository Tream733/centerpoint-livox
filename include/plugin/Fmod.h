#pragma once

#include <cuda_runtime_api.h>
#include "NvInferPlugin.h"
#include <vector>
#include <iostream>
#include <string>
#include <cublas_v2.h>
#include <cuda.h>
// #include <cudnn.h>
#include "common.h"
#include <cassert>
#include "plugin/Mod.h"
namespace nvinfer1
{
namespace plugin
{

template <typename T>
T read(const char *&buffer)
{
    T val{};
    std::memcpy(&val, buffer, sizeof(T));
    buffer += sizeof(T);
    return val;
}

template <typename T>
void write(char*&buffer, const T&val)
{
    std::memcpy(buffer, &val, sizeof(T));
    buffer += sizeof(T);
}

class ModPlugin final : public nvinfer1::IPluginV2DynamicExt
{
public:
    /**
     * 构造：第一个用于在parse阶段，PluginCreator用于创建该插件时调用的构造函数，需要传递权重信息以及参数。
     *      第二个用于在clone阶段，复制这个plugin时会用到的构造函数。
     *      第三个用于在deserialize阶段，用于将序列化好的权重和参数传入该plugin并创建。
    */
    ModPlugin(int bias);
    ModPlugin(void const* serialData, size_t serialLength);

    ~ModPlugin() override = default;

    // 插件op返回多少个Tensor
    int getNbOutputs() const noexcept override;

    /*
    *TensorRT支持Dynamic-shape的时候，batch这一维度必须是explicit的
    *例如：TensorRT处理的维度从以往的三维[3,-1,-1]变成了[1,3,-1,-1]
    *最新的onnx-tensorrt也必须设置explicit的batchsize，而且这个batch维度在getOutputDimensions中是可以获取到的
    */
    nvinfer1::DimsExprs getOutputDimensions(int outputIndex, const nvinfer1::DimsExprs* input, int nbInputs,
                                            nvinfer1::IExprBuilder& exprBuilder) noexcept override;

    // 主要初始化一些提前开辟空间的参数
    int initialize() noexcept override;

    /**
     * terminate函数就是释放这个op之前开辟的一些显存空间:析构函数则需要执行terminate
    */
    void terminate() noexcept override;

    // 返回这个插件op需要中间显存变量的实际数据大小(bytesize)
    size_t getWorkspaceSize(const nvinfer1::PluginTensorDesc* inputs, int nbInputs,
        const nvinfer1::PluginTensorDesc* outputs, int nbOutputs) const noexcept override;

    // 实际插件op的执行函数，自己实现的cuda操作就放到这里，接受输入inputs产生输出outputs
    int enqueue(const nvinfer1::PluginTensorDesc* inputDesc, const nvinfer1::PluginTensorDesc* outputDesc,
        const void * const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept override;
    
    // 返回序列化时需要写多少字节到buffer中
    size_t getSerializationSize() const noexcept override;

    // 把需要用的数据按照顺序序列化到buffer里头
    void serialize(void* buffer) const noexcept override;

    // TensorRT调用此方法以判断pos索引的输入/输出是否支持inOut[pos].format和inOut[pos].type指定的格式/数据类型
    bool supportsFormatCombination(
        int pos, const nvinfer1::PluginTensorDesc* inOut,
        int nbInputs, int nbOutputs) noexcept override;

    // IPluginV2DynamicExt Methods
    const char* getPluginType() const noexcept override;
    const char* getPluginVersion() const noexcept override;

    void destroy() noexcept override;

    // clone成员函数主要用于传递不变的权重和参数，将plugin复制n多份，从而可以被不同engine或者builder或者network使用。
    nvinfer1::IPluginV2DynamicExt* clone() const noexcept override;

    // 为这个插件设置namespace名字，如果不设置则默认是""，需要注意的是同一个namespace下的plugin如果名字相同会冲突
    void setPluginNamespace(const char* pluginNamespace) noexcept override;

    const char* getPluginNamespace() const noexcept override;
    
    // 返回结果的类型
    nvinfer1::DataType getOutputDataType(int index, const nvinfer1::DataType* inputTypes, int nbInputs) const noexcept override;

    void attachToContext(cudnnContext *cudnn, cublasContext *cublas, nvinfer1::IGpuAllocator *allocator) noexcept override;
    void detachFromContext() noexcept override;

    // 配置这个插件op，判断输入和输出类型数量是否正确
    void configurePlugin(const nvinfer1::DynamicPluginTensorDesc* in, int nbInputs,
    const nvinfer1::DynamicPluginTensorDesc* out, int nbOutputs) noexcept override;

private:
    std::string mPluginNamespace;
    std::string mNamespace;
    int batch_ ;
    int hotmap_dim_ = 3; // 参数
    int nmo_ = 500;  //  NUM_MAX_OBJS  
    int bias_;
};


class ModPluginCreator : public IPluginCreator
{
public:
    // 创建一个空的mPluginAttributes初始化mFC
    ModPluginCreator();
    ~ModPluginCreator() override = default;
 
    const char* getPluginName() const noexcept override;

    const char* getPluginVersion() const noexcept override;

    // PluginFieldCollection的主要作用是传递这个插件op所需要的权重和参数
    const nvinfer1::PluginFieldCollection* getFieldNames() noexcept override;

    // 通过PluginFieldCollection去创建plugin，将op需要的权重和参数一个一个取出来
    nvinfer1::IPluginV2DynamicExt* createPlugin(const char* name, const nvinfer1::PluginFieldCollection* fc) noexcept override;

    // 这个函数会被onnx-tensorrt的一个叫做TRT_PluginV2的转换op调用，这个op会读取onnx模型的data数据将其反序列化到network中
    nvinfer1::IPluginV2DynamicExt* deserializePlugin(const char* name, const void* serialData, size_t serialLength) noexcept override;

    void setPluginNamespace(const char* pluginNamespace) noexcept override;

    const char* getPluginNamespace() const noexcept override;
    
private:
    static nvinfer1::PluginFieldCollection mFC;

    static std::vector<nvinfer1::PluginField> mPluginAttributes;

    std::string mNamespace;
};
        
} // namespace plugin    
} // namespace nvinfer1
