/* ============================================================================
# [2017] - Robik AI Ltd - Paul Balanca
# All Rights Reserved.

# NOTICE: All information contained herein is, and remains
# the property of Robik AI Ltd, and its suppliers
# if any.  The intellectual and technical concepts contained
# herein are proprietary to Robik AI Ltd
# and its suppliers and may be covered by U.S., European and Foreign Patents,
# patents in process, and are protected by trade secret or copyright law.
# Dissemination of this information or reproduction of this material
# is strictly forbidden unless prior written permission is obtained
# from Robik AI Ltd.
# =========================================================================== */
#include <fstream>
#include <iostream>
#include <sys/stat.h>
#include <time.h>
#include <chrono>

#include <glog/logging.h>
#include <gflags/gflags.h>

#include <cuda_runtime_api.h>
#include <NvInfer.h>

#include <tensorflowrt.h>
#include <tensorflowrt_nets.h>
#include <tensorflowrt_models.h>


using namespace nvinfer1;

#define CHECK_CUDA(status)									\
{														\
    if (status != 0)									\
    {													\
        std::cout << "Cuda failure: " << status;		\
        abort();										\
    }													\
}

// FLAGS...
DEFINE_string(network, "ssd_inception2_v0", "SSD network network to test.");
DEFINE_string(network_pb, "../data/networks/ssd_inception2_v0_orig.tfrt32",
    "Network protobuf parameter file.");
DEFINE_int32(batch_size, 2, "Batch size.");
DEFINE_int32(workspace, 16, "Workspace size in MB.");
DEFINE_int32(height, 224, "Input height.");
DEFINE_int32(width, 224, "Input height.");

// static const int BATCH_SIZE = 2;
static const int TIMING_ITERATIONS = 1000;
std::string  INPUT_BLOB_NAME;
std::string  OUTPUT_BLOB_NAME;


// Logger for GIE info/warning/errors
class Logger : public ILogger
{
    void log(Severity severity, const char* msg) override
    {
        if (severity!=Severity::kINFO)
            std::cout << msg << std::endl;
    }
} gLogger;

struct Profiler : public IProfiler
{
    typedef std::pair<std::string, float> Record;
    std::vector<Record> mProfile;

    virtual void reportLayerTime(const char* layerName, float ms)
    {
        auto record = std::find_if(mProfile.begin(), mProfile.end(), [&](const Record& r){ return r.first == layerName; });
        if (record == mProfile.end())
            mProfile.push_back(std::make_pair(layerName, ms));
        else
            record->second += ms;
    }

    void printLayerTimes()
    {
        float totalTime = 0;
        for (size_t i = 0; i < mProfile.size(); i++)
        {
            printf("%-100.100s %4.3fms\n", mProfile[i].first.c_str(), mProfile[i].second / TIMING_ITERATIONS);
            totalTime += mProfile[i].second;
        }
        printf("Time over all layers: %4.3f\n", totalTime / TIMING_ITERATIONS);
    }

} gProfiler;

/* ============================================================================
 * Static collection of nets.
 * ========================================================================== */
std::unique_ptr<tfrt::network>&& networks_map(const std::string& key)
{
    static std::map<std::string, std::unique_ptr<tfrt::network> > nets;
    // Fill the map at first call!
    if(nets.empty()) {
        nets["inception1"] = std::make_unique<inception1::net>();
        nets["inception2"] = std::make_unique<inception2::net>();
        nets["ssd_inception2_v0"] = std::make_unique<ssd_inception2_v0::net>();
        nets["seg_inception2_v1"] = std::make_unique<seg_inception2_v1::net>();
        nets["seg_inception2_v1_5x5"] = std::make_unique<seg_inception2_v1_5x5::net>();
        nets["seg_inception2_logits_v1"] = std::make_unique<seg_inception2_logits_v1::net>();
        nets["seg_inception2_2x2"] = std::make_unique<seg_inception2_2x2::net>();
    }
    return std::move(nets.at(key));
}

/* ============================================================================
 * Build + inference.
 * ========================================================================== */
ICudaEngine* tfrt_to_gie_model()
{
    // Builder + network.
    IBuilder* builder = createInferBuilder(gLogger);
    INetworkDefinition* network = builder->createNetwork();

    // Build TF-RT network.
    auto tf_network = networks_map(FLAGS_network);
    tf_network->create_missing_tensors(true);
    tf_network->load_weights(FLAGS_network_pb);
    tf_network->input_shape({3, FLAGS_height, FLAGS_width});
    tfrt::scope sc = tf_network->scope(network);
    tf_network->build(sc);
    // Input and output information.
    INPUT_BLOB_NAME = {tf_network->input_name(true)};
    OUTPUT_BLOB_NAME = tf_network->outputs_name(true, true).at(0);

    // Build the engine
    builder->setMaxBatchSize(FLAGS_batch_size);
    builder->setMaxWorkspaceSize(FLAGS_workspace << 20);
    // Set up the floating mode.
    bool compatibleType = (tf_network->datatype() == nvinfer1::DataType::kFLOAT ||
                            builder->platformHasFastFp16());
    CHECK(compatibleType) << "CAN NOT build network with FP16 data type. Platform not compatible.";
    bool useFP16 = (tf_network->datatype() == nvinfer1::DataType::kHALF &&
                    builder->platformHasFastFp16());
    LOG_IF(INFO, useFP16) << "BUILD network with FP16 data type.";
    LOG_IF(INFO, !useFP16) << "BUILD network with FP32 data type.";
    builder->setHalf2Mode(useFP16);

    ICudaEngine* engine = builder->buildCudaEngine(*network);
    if (engine == nullptr) {
        std::cout << "Could not build engine" << std::endl;
    }
    network->destroy();
    builder->destroy();
    tf_network->clear_weights();
    return engine;
}
void model_serialized(IHostMemory*& gieModelStream)
{
    auto engine = tfrt_to_gie_model();
    // serialize the engine.
    // TensorRT 1
    #ifndef NV_TENSORRT_MAJOR   
    engine->serialize(*gieModelStream);
    // TensorRT 2
    #else   
    gieModelStream = engine->serialize();
    #endif
    engine->destroy();
}

void timeInference(ICudaEngine* engine, int batchSize)
{
    // input and output buffer pointers that we pass to the engine - the engine requires exactly ICudaEngine::getNbBindings(),
    // of these, but in this case we know that there is exactly one input and one output.
    assert(engine->getNbBindings() == 2);
    void* buffers[2];

    // In order to bind the buffers, we need to know the names of the input and output tensors.
    // note that indices are guaranteed to be less than ICudaEngine::getNbBindings()
    int inputIndex = engine->getBindingIndex(INPUT_BLOB_NAME.c_str());
    int outputIndex = engine->getBindingIndex(OUTPUT_BLOB_NAME.c_str());

    // allocate GPU buffers
    DimsCHW inputDims = static_cast<DimsCHW&&>(engine->getBindingDimensions(inputIndex)), outputDims = static_cast<DimsCHW&&>(engine->getBindingDimensions(outputIndex));
    size_t inputSize = batchSize * inputDims.c() * inputDims.h() * inputDims.w() * sizeof(float);
    size_t outputSize = batchSize * outputDims.c() * outputDims.h() * outputDims.w() * sizeof(float);

    std::cout << "Input index: " << inputIndex << " of size: " << inputSize << std::endl;
    std::cout << "Output index: " << outputIndex << " of size: " << outputSize << std::endl;
    CHECK_CUDA(cudaMalloc(&buffers[inputIndex], inputSize));
    CHECK_CUDA(cudaMalloc(&buffers[outputIndex], outputSize));

    IExecutionContext* context = engine->createExecutionContext();
    context->setProfiler(&gProfiler);

    // zero the input buffer
    CHECK_CUDA(cudaMemset(buffers[inputIndex], 0, inputSize));

    for (int i = 0; i < TIMING_ITERATIONS;i++)
        context->execute(batchSize, buffers);

    // release the context and buffers
    context->destroy();
    CHECK_CUDA(cudaFree(buffers[inputIndex]));
    CHECK_CUDA(cudaFree(buffers[outputIndex]));
}


int main(int argc, char** argv)
{
    google::InitGoogleLogging(argv[0]);
    gflags::ParseCommandLineFlags(&argc, &argv, true);

    std::cout << "Building and running a GPU inference engine N=2..." << std::endl;

    // Parse the model file.
    #ifndef NV_TENSORRT_MAJOR   
    nvinfer1::IHostMemory  gieModelStream;
    gieModelStream.seekg(0, gieModelStream.beg);
    auto pgieModelStream = &gieModelStream;
    model_serialized(pgieModelStream);
    #else
    IHostMemory* gieModelStream{nullptr};
    model_serialized(gieModelStream);
    #endif

    // Create an engine
    IRuntime* infer = createInferRuntime(gLogger);
    #ifndef NV_TENSORRT_MAJOR   
    nvinfer1::ICudaEngine* engine = infer->deserializeCudaEngine(gieModelStream);
    #else
    ICudaEngine* engine = infer->deserializeCudaEngine(gieModelStream->data(),
        gieModelStream->size(), nullptr);
    #endif

    printf("Bindings after deserializing:\n");
    for (int bi = 0; bi < engine->getNbBindings(); bi++) {
        if (engine->bindingIsInput(bi) == true) {
            printf("Binding %d (%s): Input.\n",  bi, engine->getBindingName(bi));
        } else {
            printf("Binding %d (%s): Output.\n", bi, engine->getBindingName(bi));
        }
    }
    // run inference with null data to time network performance
    timeInference(engine, FLAGS_batch_size);
    engine->destroy();
    infer->destroy();
    gProfiler.printLayerTimes();
    std::cout << "Done." << std::endl;
    return 0;
}
