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

/* ============================================================================
 * Parameters + NV logger.
 * ========================================================================== */
struct Params
{
    std::string modelFile, modelName, engine;
    std::vector<std::string> outputs;
    int device{ 0 }, batchSize{ 1 }, workspaceSize{ 16 }, iterations{ 10 }, avgRuns{ 10 };
    int inwidth{ 224 }, inheight{ 224 };
    bool half2{ false }, verbose{ false }, hostTime{ false };
} gParams;
std::vector<std::string> gInputs;

// Logger for GIE info/warning/errors
class Logger : public ILogger
{
    void log(Severity severity, const char* msg) override
    {
        // suppress info-level messages
        if (severity != Severity::kINFO || gParams.verbose)
            std::cout << msg << std::endl;
    }
} gLogger;

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
ICudaEngine* tfrtToGIEModel()
{
    // Builder + network.
    IBuilder* builder = createInferBuilder(gLogger);
    INetworkDefinition* network = builder->createNetwork();

    // Build TF-RT network.
    auto tf_network = networks_map(gParams.modelName);
    tf_network->create_missing_tensors(true);
    tf_network->load_weights(gParams.modelFile.c_str());
    tf_network->input_shape({3, gParams.inheight, gParams.inwidth});
    tfrt::scope sc = tf_network->scope(network);
    tf_network->build(sc);
    // Input and output information.
    gInputs = {tf_network->input_name(true)};
    gParams.outputs = tf_network->outputs_name(true, true);

    // Build the engine
    builder->setMaxBatchSize(gParams.batchSize);
    builder->setMaxWorkspaceSize(gParams.workspaceSize << 20);
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
    if (engine == nullptr)
        std::cout << "could not build engine" << std::endl;

    network->destroy();
    builder->destroy();
    tf_network->clear_weights();
    return engine;
}


void createMemory(const ICudaEngine& engine, std::vector<void*>& buffers, const std::string& name)
{
    size_t bindingIndex = engine.getBindingIndex(name.c_str());
    assert(bindingIndex < buffers.size());
    DimsCHW dimensions =
        static_cast<nvinfer1::DimsCHW&&>(engine.getBindingDimensions((int)bindingIndex));
    size_t eltCount = dimensions.c()*dimensions.h()*dimensions.w()*gParams.batchSize, memSize = eltCount * sizeof(float);

    float* localMem = new float[eltCount];
    for (size_t i = 0; i < eltCount; i++) {
        localMem[i] = (float(rand()) / RAND_MAX) * 2 - 1;
    }
    void* deviceMem;
    CHECK_CUDA(cudaMalloc(&deviceMem, memSize));
    if (deviceMem == nullptr) {
        std::cerr << "Out of memory" << std::endl;
        exit(1);
    }
    CHECK_CUDA(cudaMemcpy(deviceMem, localMem, memSize, cudaMemcpyHostToDevice));

    delete[] localMem;
    buffers[bindingIndex] = deviceMem;
}

void doInference(ICudaEngine& engine)
{
    IExecutionContext *context = engine.createExecutionContext();
    // input and output buffer pointers that we pass to the engine - the engine requires exactly IEngine::getNbBindings(),
    // of these, but in this case we know that there is exactly one input and one output.

    std::vector<void*> buffers(gInputs.size() + gParams.outputs.size());
    std::cout << "Buffers size: " << buffers.size() << std::endl;
    for (size_t i = 0; i < gInputs.size(); i++) {
        std::cout << "Input: " << i << " | " << gInputs[i] << std::endl;
        createMemory(engine, buffers, gInputs[i]);
    }

    for (size_t i = 0; i < gParams.outputs.size(); i++) {
        std::cout << "Output: " << i << " | " << gParams.outputs[i] << std::endl;
        createMemory(engine, buffers, gParams.outputs[i]);
    }

    cudaStream_t stream;
    CHECK_CUDA(cudaStreamCreate(&stream));
    cudaEvent_t start, end;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&end));

    for (int j = 0; j < gParams.iterations; j++)
    {
        float total = 0, ms;
        for (int i = 0; i < gParams.avgRuns; i++)
        {
            if (gParams.hostTime)
            {
                auto t_start = std::chrono::high_resolution_clock::now();
                context->execute(gParams.batchSize, &buffers[0]);
                auto t_end = std::chrono::high_resolution_clock::now();
                ms = std::chrono::duration<float, std::milli>(t_end - t_start).count();
            }
            else
            {
                cudaEventRecord(start, stream);
                context->enqueue(gParams.batchSize, &buffers[0], stream, nullptr);
                cudaEventRecord(end, stream);
                cudaEventSynchronize(end);
                cudaEventElapsedTime(&ms, start, end);
            }
            total += ms;
        }
        total /= gParams.avgRuns;
        std::cout << "Average over " << gParams.avgRuns << " runs is " << total << " ms." << std::endl;
    }
    cudaStreamDestroy(stream);
    cudaEventDestroy(start);
    cudaEventDestroy(end);
}



static void printUsage()
{
    printf("\n");
    printf("Mandatory params:\n");
    printf("  --modelName=<name>       TF-RT model name\n");
    printf("  --modelFile=<file>     Weights file\n");
    printf("  --output=<name>      Output blob name (can be specified multiple times\n");

    printf("\nOptional params:\n");

    printf("  --height=N           Input height (default = %d)\n", gParams.inheight);
    printf("  --width=N            Input width (default = %d)\n", gParams.inwidth);
    printf("  --batch=N            Set batch size (default = %d)\n", gParams.batchSize);
    printf("  --device=N           Set cuda device to N (default = %d)\n", gParams.device);
    printf("  --iterations=N       Run N iterations (default = %d)\n", gParams.iterations);
    printf("  --avgRuns=N          Set avgRuns to N - perf is measured as an average of avgRuns (default=%d)\n", gParams.avgRuns);
    printf("  --workspace=N        Set workspace size in megabytes (default = %d)\n", gParams.workspaceSize);
    printf("  --half2              Run in paired fp16 mode - default = false\n");
    printf("  --verbose            Use verbose logging - default = false\n");
    printf("  --hostTime	       Measure host time rather than GPU time - default = false\n");
    printf("  --engine=<file>      Generate a serialized GIE engine\n");

    fflush(stdout);
}

bool parseString(const char* arg, const char* name, std::string& value)
{
    size_t n = strlen(name);
    bool match = arg[0] == '-' && arg[1] == '-' && !strncmp(arg + 2, name, n) && arg[n + 2] == '=';
    if (match)
    {
        value = arg + n + 3;
        std::cout << name << ": " << value << std::endl;
    }
    return match;
}

bool parseInt(const char* arg, const char* name, int& value)
{
    size_t n = strlen(name);
    bool match = arg[0] == '-' && arg[1] == '-' && !strncmp(arg + 2, name, n) && arg[n + 2] == '=';
    if (match)
    {
        value = atoi(arg + n + 3);
        std::cout << name << ": " << value << std::endl;
    }
    return match;
}

bool parseBool(const char* arg, const char* name, bool& value)
{
    size_t n = strlen(name);
    bool match = arg[0] == '-' && arg[1] == '-' && !strncmp(arg + 2, name, n);
    if (match)
    {
        std::cout << name << std::endl;
        value = true;
    }
    return match;

}


bool parseArgs(int argc, char* argv[])
{
    if (argc < 4)
    {
        printUsage();
        return false;
    }

    for (int j = 1; j < argc; j++)
    {
        if (parseString(argv[j], "modelName", gParams.modelName) || parseString(argv[j], "modelFile", gParams.modelFile) || parseString(argv[j], "engine", gParams.engine))
            continue;

        std::string output;
        if (parseString(argv[j], "output", output))
        {
            gParams.outputs.push_back(output);
            continue;
        }

        if (parseInt(argv[j], "height", gParams.inheight) ||
            parseInt(argv[j], "width", gParams.inwidth) ||
            parseInt(argv[j], "batch", gParams.batchSize) ||
            parseInt(argv[j], "iterations", gParams.iterations) ||
            parseInt(argv[j], "avgRuns", gParams.avgRuns) ||
            parseInt(argv[j], "device", gParams.device)	||
            parseInt(argv[j], "workspace", gParams.workspaceSize))
            continue;

        if (parseBool(argv[j], "half2", gParams.half2) || parseBool(argv[j], "verbose", gParams.verbose) || parseBool(argv[j], "hostTime", gParams.hostTime))
            continue;

        printf("Unknown argument: %s\n", argv[j]);
        return false;
    }
    return true;
}

int main(int argc, char** argv)
{
    google::InitGoogleLogging(argv[0]);
	// gflags::ParseCommandLineFlags(&argc, &argv, true);
    // create a GIE model from the caffe model and serialize it to a stream
    if(!parseArgs(argc, argv)) {
        return -1;
    }
    cudaSetDevice(gParams.device);
    if (gParams.modelName.empty() || gParams.modelFile.empty()) {
        std::cerr << "Model name or file not specified" << std::endl;
        return -1;
    }
    ICudaEngine* engine = tfrtToGIEModel();
    if (!engine) {
        std::cerr << "Engine could not be created" << std::endl;
        return -1;
    }
    // if (!gParams.engine.empty())
    // {
    //     std::ofstream p(gParams.engine);
    //     if (!p)
    //     {
    //         std::cerr << "could not open plan output file" << std::endl;
    //         return -1;
    //     }
    //     engine->serialize(p);
    // }

    doInference(*engine);
    engine->destroy();

    return 0;
}
