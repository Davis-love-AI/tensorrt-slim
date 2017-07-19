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
#ifndef TFRT_SSD_INCEPTION2_V0
#define TFRT_SSD_INCEPTION2_V0

#include <NvInfer.h>

#include "../tensorflowrt.h"
#include "../nets/inception2.h"

namespace ssd_inception2_v0
{
/* ============================================================================
 * Inception2 network: functional declaration.
 * ========================================================================== */
inline nvinfer1::ITensor* base(nvinfer1::ITensor* input, tfrt::scope sc)
{
    nvinfer1::ITensor* net{input};
    // Main blocks 1 to 5.
    net = block1(net, sc);
    net = block2(net, sc);
    net = block3(net, sc);
    net = block4(net, sc);
    net = block5(net, sc);
    return net;
}
inline nvinfer1::ITensor* inception2(nvinfer1::ITensor* input,
                                     tfrt::scope sc,
                                     int num_classes=1001)
{
    nvinfer1::ITensor* net;
    // Construct backbone network.
    net = base(input, sc);
    // Logits end block.
    {
        typedef tfrt::avg_pooling2d<tfrt::PaddingType::VALID>  avg_pool2d;
        typedef tfrt::convolution2d<tfrt::ActivationType::NONE, tfrt::PaddingType::SAME, false>  conv2d;
        auto ssc = sc.sub("Logits");
        net = avg_pool2d(ssc, "AvgPool_1a_7x7").ksize({7, 7})(net);
        net = conv2d(ssc, "Conv2d_1c_1x1").noutputs(num_classes).ksize({1, 1})(net);
    }
    net = tfrt::softmax(sc, "Softmax")(net);
    return net;
}

/* ============================================================================
 * Inception2 class: as imagenet network.
 * ========================================================================== */
class net : public tfrt::imagenet_network
{
public:
    /** Constructor with default name.  */
    net() : tfrt::imagenet_network("InceptionV2", 1000, true) {}

    /** Inception2 building method. Take a network scope and do the work!
     */
    virtual nvinfer1::ITensor* build(tfrt::scope sc) {
        auto net = tfrt::input(sc)();
        // auto net = tfrt::input(sc).shape({64, 112, 112})();
        net = inception2(net, sc, 1001);
        return net;
    }
};

}

#endif
