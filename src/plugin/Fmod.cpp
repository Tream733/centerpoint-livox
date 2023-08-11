#include "plugin/Fmod.h"

using namespace nvinfer1;
using nvinfer1::plugin::ModPlugin;
using nvinfer1::plugin::ModPluginCreator;

namespace nvinfer1
{

namespace plugin
{
    static char const* const PLUGIN_VERSION{"1"};
    static char const* const PLUGIN_NAME{"Mod"};

    PluginFieldCollection ModPluginCreator::mFC{};
    std::vector<PluginField> ModPluginCreator::mPluginAttributes;

    ModPlugin::ModPlugin(int bias)
        : batch_(1)
        , hotmap_dim_(3)
        , nmo_(500)
        , bias_(bias)
    {
        //std::cout<<" ModPlugin::ModPlugin 1 "<<std::endl;
    }

    ModPlugin::ModPlugin(void const* serialData, size_t serialLength) 
    {
       // std::cout<<" ModPlugin::ModPlugin 2 "<<std::endl;
        //std::cout<<"---------------------"<<serialLength<<std::endl;
        char const* d = reinterpret_cast<char const*>(serialData);
        char const* a = d;
        bias_  = read<int>(d);
        batch_ = read<int>(d);
        assert(d == a+serialLength);
    }

    int ModPlugin::getNbOutputs() const noexcept
    {
        return 1;
    }

    nvinfer1::DimsExprs ModPlugin::getOutputDimensions(int outputIndex, const nvinfer1::DimsExprs* inputs, int nbInputs,
                                                nvinfer1::IExprBuilder& exprBuilder) noexcept 
    {
        assert(outputIndex == 0);
        nvinfer1::DimsExprs output;
        output.nbDims = 3;
        output.d[0] = exprBuilder.constant(1);
        output.d[1] = exprBuilder.constant(hotmap_dim_);
        output.d[2] = exprBuilder.constant(nmo_);
        return output;
    }

    int ModPlugin::enqueue(nvinfer1::PluginTensorDesc const* inputDesc, nvinfer1::PluginTensorDesc const* outputDesc,
             const void* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept
    {
        void const* inputData = inputs[0];
        void* outputData = outputs[0];
        int status =  ModInference(stream, batch_*hotmap_dim_*nmo_, bias_,inputData, outputData);
        return status;
    }

    size_t ModPlugin::getSerializationSize() const noexcept
    {
        // bias, mBatch
        return sizeof(int) + sizeof(int);
    }

    void ModPlugin::serialize(void* buffer) const noexcept
    {
        char *d = reinterpret_cast<char *>(buffer), *a = d;
        write(d, bias_);
        write(d, batch_);
        assert(d == a+ getSerializationSize());
        //std::cout<<"ModPlugin::serialize"<<std::endl;
    }

    bool ModPlugin::supportsFormatCombination(int pos, const nvinfer1::PluginTensorDesc* inOut,
            int nbInputs, int nbOutputs) noexcept 
    {
        assert(nbInputs == 2);
        assert(nbOutputs == 1);
        const PluginTensorDesc& in = inOut[pos];
        if (pos == 0)
        {
            return (in.type == nvinfer1::DataType::kINT32) && (in.format == TensorFormat::kLINEAR);
        }
        if (pos == 1)
        {
            return (in.type == nvinfer1::DataType::kINT32) && (in.format == TensorFormat::kLINEAR);
        }
        if (pos == 2)
        {
            return (in.type == nvinfer1::DataType::kINT32) && (in.format == TensorFormat::kLINEAR);
        }
        return false;
    }

    int ModPlugin::initialize() noexcept
    {
        return 0;
    }

    void ModPlugin::terminate() noexcept {}

    size_t ModPlugin::getWorkspaceSize(const nvinfer1::PluginTensorDesc* inputs, int nbInputs,const nvinfer1::PluginTensorDesc* outputs, int nbOutputs) const noexcept
    {
        size_t HmSize = inputs[0].dims.d[1]*inputs[0].dims.d[2]*sizeof(int);
        return HmSize;
    }

    // IPluginV2 Methods
    const char* ModPlugin::getPluginType() const noexcept
    {
        return PLUGIN_NAME;
    }

    const char* ModPlugin::getPluginVersion() const noexcept
    {
        return PLUGIN_VERSION;
    }

    void ModPlugin::destroy() noexcept
    {
        delete this;
    }

    nvinfer1::IPluginV2DynamicExt* ModPlugin::clone() const noexcept
    {
        auto* plugin = new ModPlugin(*this);
        plugin->setPluginNamespace(mPluginNamespace.c_str());
        plugin->initialize();
        return plugin;
    }

    void ModPlugin::setPluginNamespace(const char* pluginNamespace) noexcept
    {
        mPluginNamespace = pluginNamespace;
    }

    const char* ModPlugin::getPluginNamespace() const noexcept
    {
        return mPluginNamespace.c_str();
    }

    // IPluginV2Ext Methods
    nvinfer1::DataType ModPlugin::getOutputDataType(
        int index, const nvinfer1::DataType* inputTypes, int nbInputs) const noexcept
    {
        assert(inputTypes && nbInputs > 0 && index == 0);
        return inputTypes[0];
    }

    // Attach the plugin object to an execution context and grant the plugin the access to some context resource.
    void ModPlugin::attachToContext(
        cudnnContext *cudnnContext, cublasContext *cublasContext, IGpuAllocator *gpuAllocator) noexcept
    {
    }

    // Detach the plugin object from its execution context.
    void ModPlugin::detachFromContext() noexcept {}


    void ModPlugin::configurePlugin(const nvinfer1::DynamicPluginTensorDesc* in, int nbInputs,
        const nvinfer1::DynamicPluginTensorDesc* out, int nbOutputs) noexcept
    {
        assert(nbInputs == 2 && in[0].desc.dims.d[1] != -1);
    }


    /*
    *------------------------------------ModPluginCreator-----------------------------------------------
    */

    ModPluginCreator::ModPluginCreator() {
        //std::cout<< "ModPluginCreator::ModPluginCreator"<<std::endl;
        mPluginAttributes.clear();
        mPluginAttributes.emplace_back("fmod", nullptr, PluginFieldType::kINT32, 1);
        mFC.nbFields = mPluginAttributes.size();
        mFC.fields = mPluginAttributes.data();
    }

    const char* ModPluginCreator::getPluginName() const noexcept
    {
        //std::cout<< "ModPluginCreator::getPluginName"<<std::endl;
        return PLUGIN_NAME;
    }

    const char* ModPluginCreator::getPluginVersion() const noexcept 
    {
        //std::cout<< "ModPluginCreator::getPluginVersion"<<std::endl;
        return PLUGIN_VERSION;
    }
 
    const PluginFieldCollection* ModPluginCreator::getFieldNames() noexcept
    {
        //std::cout<< "ModPluginCreator::getFieldNames"<<std::endl;
        return &mFC;
    }

    IPluginV2DynamicExt* ModPluginCreator::createPlugin(const char* name, const PluginFieldCollection* fc) noexcept
    {
        //std::cout<< "ModPluginCreator::createPlugin"<<std::endl;
        const PluginField* fields = fc->fields;
       // std::cout<<"----------------------------------"<<std::endl;
        //std::cout << fc->nbFields << std::endl;
        assert(fc->nbFields == 1);
        assert(fields[0].type == PluginFieldType::kINT32);
        int bias = *(static_cast<const int*>(fields[0].data));
        //std::cout<<"----------------------------------"<<bias<<std::endl;
        ModPlugin* fmod = new ModPlugin(bias);
        fmod->setPluginNamespace(mNamespace.c_str());
        fmod->initialize();
        return fmod;
        // return new ModPlugin();
    }

    IPluginV2DynamicExt* ModPluginCreator::deserializePlugin(
        const char* name, const void* serialData, size_t serialLength) noexcept
    {
        //std::cout<< "ModPluginCreator::deserializePlugin"<<std::endl;
        ModPlugin* fmod = new ModPlugin(serialData,serialLength);
        fmod->setPluginNamespace(mNamespace.c_str());
        fmod->initialize();
        return fmod;
        // return new ModPlugin(serialData,serialLength);
    }

    void ModPluginCreator::setPluginNamespace(const char* libNamespace) noexcept
    {
        //std::cout<< "ModPluginCreator::setPluginNamespace"<<std::endl;
        mNamespace = libNamespace;
    }

    const char* ModPluginCreator::getPluginNamespace() const noexcept
    {
        //std::cout<< "ModPluginCreator::getPluginNamespace"<<std::endl;
        return mNamespace.c_str();
    }

    REGISTER_TENSORRT_PLUGIN(ModPluginCreator);
}
}