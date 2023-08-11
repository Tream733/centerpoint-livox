#include "plugin/Atan2.h"

using namespace nvinfer1;
using nvinfer1::plugin::Atan2;
using nvinfer1::plugin::Atan2PluginCreator;

namespace nvinfer1
{

namespace plugin
{
    static char const* const PLUGIN_VERSION{"1"};
    static char const* const PLUGIN_NAME{"atan2"};

    PluginFieldCollection Atan2PluginCreator::mFC{};
    std::vector<PluginField> Atan2PluginCreator::mPluginAttributes;

    Atan2::Atan2()
        : batch_(1)
        , nmo_(500)
    {
        //std::cout<<" ModPlugin::ModPlugin 1 "<<std::endl;
    }

    Atan2::Atan2(void const* serialData, size_t serialLength) 
    {
       // std::cout<<" Atan2::Atan2 2 "<<std::endl;
        //std::cout<<"---------------------"<<serialLength<<std::endl;
        char const* d = reinterpret_cast<char const*>(serialData);
        char const* a = d;
        batch_ = read<int>(d);
        assert(d == a+serialLength);
    }

    int Atan2::getNbOutputs() const noexcept
    {
        return 1;
    }

    nvinfer1::DimsExprs Atan2::getOutputDimensions(int outputIndex, const nvinfer1::DimsExprs* inputs, int nbInputs,
                                                nvinfer1::IExprBuilder& exprBuilder) noexcept 
    {
        assert(outputIndex == 0);
        //std::cout<<"nbInputs      "<<nbInputs<<std::endl;
        // for(int i = 0; i<nbInputs; i++){
        //     printf("input[%d]: ", i);
        //     for(int j = 0; j <inputs[i].nbDims; j++) {
        //         printf("%d ", inputs[i].d[j]->getConstantValue());
        //     }
        //     printf("\n");
        // }
        nvinfer1::DimsExprs output;
        output.nbDims = 3;
        output.d[0] = exprBuilder.constant(1);
        output.d[1] = exprBuilder.constant(nmo_);
        output.d[2] = exprBuilder.constant(1);
        return output;
    }

    int Atan2::enqueue(nvinfer1::PluginTensorDesc const* inputDesc, nvinfer1::PluginTensorDesc const* outputDesc,
             const void* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept
    {
        void const* inputData1 = inputs[0];
        void const* inputData2 = inputs[1];
        void* outputData = outputs[0];
        int status =  Atan2Inference(stream, batch_*nmo_,inputData1,inputData2, outputData);
        return status;
    }

    size_t Atan2::getSerializationSize() const noexcept
    {
        // bias, mBatch
        return sizeof(int);
    }

    void Atan2::serialize(void* buffer) const noexcept
    {
        char *d = reinterpret_cast<char *>(buffer), *a = d;
        write(d, batch_);
        assert(d == a+ getSerializationSize());
        //std::cout<<"ModPlugin::serialize"<<std::endl;
    }

    bool Atan2::supportsFormatCombination(int pos, const nvinfer1::PluginTensorDesc* inOut,
            int nbInputs, int nbOutputs) noexcept 
    {
        assert(nbInputs == 2);
        assert(nbOutputs == 1);
        const PluginTensorDesc& in = inOut[pos];
        if (pos == 0)
        {
            return (in.type == nvinfer1::DataType::kFLOAT) && (in.format == TensorFormat::kLINEAR);
        }
        if (pos == 1)
        {
            return (in.type == nvinfer1::DataType::kFLOAT) && (in.format == TensorFormat::kLINEAR);
        }
        if (pos == 2)
        {
            return (in.type == nvinfer1::DataType::kFLOAT) && (in.format == TensorFormat::kLINEAR);
        }
        return false;
    }

    int Atan2::initialize() noexcept
    {
        return 0;
    }

    void Atan2::terminate() noexcept {}

    size_t Atan2::getWorkspaceSize(const nvinfer1::PluginTensorDesc* inputs, int nbInputs,const nvinfer1::PluginTensorDesc* outputs, int nbOutputs) const noexcept
    {
        size_t HmSize = inputs[0].dims.d[1]*sizeof(float);
        return HmSize;
    }

    // IPluginV2 Methods
    const char* Atan2::getPluginType() const noexcept
    {
        return PLUGIN_NAME;
    }

    const char* Atan2::getPluginVersion() const noexcept
    {
        return PLUGIN_VERSION;
    }

    void Atan2::destroy() noexcept
    {
        delete this;
    }

    nvinfer1::IPluginV2DynamicExt* Atan2::clone() const noexcept
    {
        auto* plugin = new Atan2(*this);
        plugin->setPluginNamespace(mPluginNamespace.c_str());
        plugin->initialize();
        return plugin;
    }

    void Atan2::setPluginNamespace(const char* pluginNamespace) noexcept
    {
        mPluginNamespace = pluginNamespace;
    }

    const char* Atan2::getPluginNamespace() const noexcept
    {
        return mPluginNamespace.c_str();
    }

    // IPluginV2Ext Methods
    nvinfer1::DataType Atan2::getOutputDataType(
        int index, const nvinfer1::DataType* inputTypes, int nbInputs) const noexcept
    {
        assert(inputTypes && nbInputs > 0 && index == 0);
        return inputTypes[0];
    }

    // Attach the plugin object to an execution context and grant the plugin the access to some context resource.
    void Atan2::attachToContext(
        cudnnContext *cudnnContext, cublasContext *cublasContext, IGpuAllocator *gpuAllocator) noexcept
    {
    }

    // Detach the plugin object from its execution context.
    void Atan2::detachFromContext() noexcept {}


    void Atan2::configurePlugin(const nvinfer1::DynamicPluginTensorDesc* in, int nbInputs,
        const nvinfer1::DynamicPluginTensorDesc* out, int nbOutputs) noexcept
    {
        assert(nbInputs == 2 && in[0].desc.dims.d[1] != -1);
    }


    /*
    *------------------------------------ModPluginCreator-----------------------------------------------
    */

    Atan2PluginCreator::Atan2PluginCreator() {
        //std::cout<< "Atan2PluginCreator::Atan2PluginCreator"<<std::endl;
        mPluginAttributes.clear();
        mFC.nbFields = mPluginAttributes.size();
        mFC.fields = mPluginAttributes.data();
    }

    const char* Atan2PluginCreator::getPluginName() const noexcept
    {
        //std::cout<< "Atan2PluginCreator::getPluginName"<<std::endl;
        return PLUGIN_NAME;
    }

    const char* Atan2PluginCreator::getPluginVersion() const noexcept 
    {
        //std::cout<< "Atan2PluginCreator::getPluginVersion"<<std::endl;
        return PLUGIN_VERSION;
    }
 
    const PluginFieldCollection* Atan2PluginCreator::getFieldNames() noexcept
    {
        // std::cout<< "Atan2PluginCreator::getFieldNames"<<std::endl;
        return &mFC;
    }

    IPluginV2DynamicExt* Atan2PluginCreator::createPlugin(const char* name, const PluginFieldCollection* fc) noexcept
    {
        //std::cout<< "Atan2PluginCreator::createPlugin"<<std::endl;
        return new Atan2();
    }

    IPluginV2DynamicExt* Atan2PluginCreator::deserializePlugin(
        const char* name, const void* serialData, size_t serialLength) noexcept
    {
        //std::cout<< "Atan2PluginCreator::deserializePlugin"<<std::endl;
        return new Atan2(serialData,serialLength);
    }

    void Atan2PluginCreator::setPluginNamespace(const char* libNamespace) noexcept
    {
        //std::cout<< "Atan2PluginCreator::setPluginNamespace"<<std::endl;
        mNamespace = libNamespace;
    }

    const char* Atan2PluginCreator::getPluginNamespace() const noexcept
    {
        //std::cout<< "Atan2PluginCreator::getPluginNamespace"<<std::endl;
        return mNamespace.c_str();
    }

    REGISTER_TENSORRT_PLUGIN(Atan2PluginCreator);
}
}