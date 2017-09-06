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
#include <iomanip>
#include <sys/stat.h>
#include <time.h>
#include <chrono>

#include <half.hpp>

#include <VX/vx.h>
#include <VX/vxu.h>
#include <NVX/nvx.h>
#include <OVX/UtilityOVX.hpp>

#include <tensorflowrt.h>
#include <tensorflowrt_nets.h>
#include <tensorflowrt_models.h>

#include <cuda/cudaCHWImage.h>

DEFINE_double(value, 1.0, "Float value.");

// CUDA methods...
void cuda_float2half_array(float* host_input, uint16_t* host_output, uint32_t size);
void cuda_half2float_array(uint16_t* host_input, float* host_output, uint32_t size);

void print_tensor_hw(const tfrt::nchw<float>::tensor& t)
{
    for (int i = 0 ; i < t.dimension(2) ; ++i){
        for (int j = 0 ; j < t.dimension(3) ; ++j){
            std::cout << t(0, 0, i, j) << " ";
        }
        std::cout << std::endl;
    }
}

/* ============================================================================
 * Test transpose convolution.
 * ========================================================================== */
typedef tfrt::convolution2d_transpose<tfrt::ActivationType::NONE, tfrt::PaddingType::CUSTOM, false>  conv2d_transpose;
class transpose_conv_net : public tfrt::network
{
public:
    /** Constructor with default name.  */
    transpose_conv_net(int width=2, int height=2) :
        tfrt::network("tranpose_net"), m_width{width}, m_height{height} {}
    /** Build network. */
    virtual nvinfer1::ITensor* build(tfrt::scope sc)
    {
        // Set basic parameters.
        // this->datatype(nvinfer1::DataType::kFLOAT);
        this->input("input", {1, m_height, m_width});
        this->outputs({"tconv"}, {{1, m_height*2, m_width*2}});

        // Construct simple network...
        auto net = tfrt::input(sc)();
        // Tranpose convolution. Weights in CKRS format (C: input)
        tfrt::nchw<float>::tensor  weights(1, 1, 2, 2);
        weights(0, 0, 0, 0) = 0;
        weights(0, 0, 0, 1) = 1;
        weights(0, 0, 1, 0) = 2;
        weights(0, 0, 1, 1) = 3;
        std::cout << "Weights:" << std::endl;
        print_tensor_hw(weights);

        this->create_tensor(sc.sub("tconv").sub("weights").name(), weights, this->datatype());
        net = conv2d_transpose(sc, "tconv")
            .noutputs(1).ksize({2, 2}).stride({2, 2}).padding({0, 0}).is_output(true)(net);
        return net;
    }
    tfrt::nchw<float>::tensor inference(const tfrt::nchw<float>::tensor& tensor)
    {
        this->network::inference(tensor);
        return m_cuda_outputs[0].tensor();
    }
private:
    int  m_width;
    int  m_height;
};

/* ============================================================================
 * Test average pooling
 * ========================================================================== */
class avg_pool_net : public tfrt::network
{
public:
    /** Constructor with default name.  */
    avg_pool_net(int width=4, int height=4) :
        tfrt::network("avg_pool_net"), m_width{width}, m_height{height} {}
    /** Build network. */
    virtual nvinfer1::ITensor* build(tfrt::scope sc)
    {
        // this->create_missing_tensors(true);
        // Set basic parameters.
        // this->datatype(nvinfer1::DataType::kFLOAT);
        this->input("input", {1, m_height, m_width});
        this->outputs({"avgpool"}, {{1, m_height, m_width}});

        // Construct simple network...
        auto net = tfrt::input(sc)();
        // net = tfrt::avg_pool2d(sc, "avgpool").ksize({3, 3}).is_output(true)(net);
        net = tfrt::bilinear2d(sc, "avgpool").is_output(true)(net);
        return net;
    }
    tfrt::nchw<float>::tensor inference(const tfrt::nchw<float>::tensor& tensor)
    {
        this->network::inference(tensor);
        return m_cuda_outputs[0].tensor();
    }
private:
    int  m_width;
    int  m_height;
};


void print_half(uint16_t* half)
{
    uint8_t* ptr = (uint8_t*) half;
    LOG(INFO) << "HALF: 0x" << std::hex << int(ptr[0]) << int(ptr[1]) << std::endl;
}

int main(int argc, char **argv)
{
    // google::InitGoogleLogging(argv[0]);
    gflags::ParseCommandLineFlags(&argc, &argv, true);

    // Half precision tests.
    LOG(INFO) << "Half size: " << sizeof(half_float::half);
    // Convert a vector of float to half.
    std::vector<float> vec_f = {FLAGS_value};
    std::vector<float> vec_f2 = {0.0};
    std::vector<uint16_t> vec_h = {0};

    LOG(INFO) << "Original data: " << std::setprecision(6) << vec_f[0] << " | " << vec_f2[0];
    cuda_float2half_array(vec_f.data(), vec_h.data(), vec_f.size());
    cuda_half2float_array(vec_h.data(), vec_f2.data(), vec_f.size());
    LOG(INFO) << "Half data: " << std::setprecision(6) << vec_f[0] << " | " << vec_f2[0];
    print_half(vec_h.data());

    // VX image testing...
    ovxio::ContextGuard context;
    vxRegisterLogCallback(context, &ovxio::stdoutLogCallback, vx_false_e);
    vxDirective(context, VX_DIRECTIVE_ENABLE_PERFORMANCE);

    vx_pixel_value_t initVal;
    initVal.RGBX[0] = 255;
    initVal.RGBX[1] = 128;
    initVal.RGBX[2] = 64;
    initVal.RGBX[3] = 32;
    vx_image frame = vxCreateUniformImage(context, 200, 100, VX_DF_IMAGE_RGBX, &initVal);

    vx_rectangle_t rect;
    vxGetValidRegionImage(frame, &rect);
    tfrt::nvx_image_inpatch img_patch{frame};


    // Interaction with cuda tensor?
    tfrt::cuda_tensor ctensor{"test", {1, 3, 100, 200}};
    ctensor.allocate();
    LOG(INFO) << "CUDA tensor: " << tfrt::dims_str(ctensor.shape);
    auto r = cuda_rgba_to_chw(img_patch.cuda, ctensor.cuda, 200, 100,
        img_patch.addr.stride_x, img_patch.addr.stride_y);

    CUDA(cudaDeviceSynchronize());
    int start = 20000-10;
    for (int i = start ; i < start+1 ; ++i) {
        LOG(INFO) << "CUDA tensor: " << i << " | " << ctensor.cpu[i] << " | " << ctensor.cpu[i];
    }

    // TEST TRANSPOSE CONVOLUTION.
    {
        int width = 2;
        int height = 2;
        transpose_conv_net net(width, height);
        // net.datatype(nvinfer1::DataType::kHALF);
        net.max_workspace_size(16 << 24);
        net.load("");

        tfrt::nchw<float>::tensor  inputs(1, 1, height, width);
        inputs.setConstant(0.0);
        int count = 1;
        for (int i = 0 ; i < inputs.dimension(2) ; ++i){
            for (int j = 0 ; j < inputs.dimension(3) ; ++j){
                inputs(0, 0, i, j) = count;
                count++;
            }
        }
        std::cout << "Input tensor: " << std::endl;
        print_tensor_hw(inputs);
        auto output = net.inference(inputs);
        std::cout << "Output tensor: " << std::endl;
        print_tensor_hw(output);
        std::cout << "Output dimensions: "
            << output.dimension(1) << " | " << output.dimension(2) << " | " << output.dimension(3) << std::endl;
    }
    // TEST AVG POOLING
    {
        int width = 4;
        int height = 4;
        avg_pool_net net(width, height);
        // net.datatype(nvinfer1::DataType::kHALF);
        net.max_workspace_size(16 << 24);
        net.load("");

        tfrt::nchw<float>::tensor  inputs(1, 1, height, width);
        inputs.setConstant(0.0);
        int count = 1;
        for (int i = 0 ; i < inputs.dimension(2) ; i+=2){
            for (int j = 0 ; j < inputs.dimension(3) ; j+=2){
                inputs(0, 0, i, j) = count;
                count++;
            }
        }
        std::cout << "Input tensor: " << std::endl;
        print_tensor_hw(inputs);
        auto output = net.inference(inputs);
        std::cout << "Output tensor: " << std::endl;
        CUDA(cudaDeviceSynchronize());
        print_tensor_hw(output);
    }

    return 0;
}
