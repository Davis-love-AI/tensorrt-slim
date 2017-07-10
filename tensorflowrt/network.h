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
#ifndef TFRT_NETWORK_H
#define TFRT_NETWORK_H

// #include <gflags/gflags.h>
// #include <glog/logging.h>

// #include <cmath>
// #include <string>
// #include <sstream>

// // #include <cuda_runtime_api.h>
#include <NvInfer.h>

#include "tfrt_jetson.h"
#include "network.pb.h"

namespace tfrt
{
class scope;

/** Generic network class, implementation the basic building, inference
 * profiling methods.
 */
class network
{
public:
    /** Create network, specifying the name and the datatype.
     */
    network(std::string name, nvinfer1::DataType datatype) {

    }
    /** Clear the network and its weights. */
    void clear();

    /** Load network weights. */
    bool load_weights(std::string filename);

    /** Get the default scope for this network. */
    tfrt::scope scope(nvinfer1::INetworkDefinition* nv_network) const;

public:
    /** Get a tensor by name. Return empty tensor if not found.
     */
    const tfrt_pb::tensor& tensor_by_name(std::string name) const;
    /** Get NV weights by name. Return empty weights if not found.
     */
    nvinfer1::Weights weights_by_name(std::string name) const;

    const std::string& name() const;
    network& name(const std::string& name);
    nvinfer1::DataType datatype() const;

    nvinfer1::DimsHW input_shape() const;
    const std::string& input_name() const;
    std::vector<nvinfer1::DimsHW> outputs_shape() const;
    std::vector<std::string> outputs_name() const;


public:
    /** Generate empty weights. */
    nvinfer1::Weights empty_weights() const {
        return nvinfer1::Weights{.type = this->datatype(), .values = nullptr, .count = 0};
    }
    /** Convert TF protobuf tensor to NV weights. */
    static nvinfer1::Weights tensor_to_weights(const tfrt_pb::tensor& tensor);

protected:
    // Protected setters...
    // void name(std::string name);


    /** Clear out the collections of network weights, to save memory.
     */
    void clear_weights();

protected:
    // Protobuf network object.
    tfrt_pb::network  m_pb_network;


    // Temporary collection of zero tensors.
    std::vector<tfrt_pb::tensor>  m_zero_tensors;
};

}

#endif