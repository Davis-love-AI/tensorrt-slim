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
#include <fcntl.h>
#include <unistd.h>

#include <iostream>
#include <fstream>

#include <gflags/gflags.h>
#include <glog/logging.h>

#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/text_format.h>

#include "scope.h"
#include "network.h"
#include "tensorflowrt.h"

const int kProtoReadBytesLimit = INT_MAX;

namespace tfrt {

using google::protobuf::io::FileInputStream;
using google::protobuf::io::FileOutputStream;
using google::protobuf::io::ZeroCopyInputStream;
using google::protobuf::io::CodedInputStream;
using google::protobuf::io::ZeroCopyOutputStream;
using google::protobuf::io::CodedOutputStream;
// using google::protobuf::Message;

/* ============================================================================
 * Nice utils
 * ========================================================================== */
inline nvinfer1::Dims tensor_shape(const tfrt_pb::tensor& t)
{
    nvinfer1::Dims dims;
    for(int i = 0; i < t.shape_size(); i++) {
        dims.d[i] = t.shape(i);
    }
    dims.nbDims = t.shape_size();
    return dims;
}

/* ============================================================================
 * tfrt::network methods.
 * ========================================================================== */
void network::clear()
{
    m_pb_network.Clear();
    m_zero_tensors.clear();
}
tfrt::scope network::scope(nvinfer1::INetworkDefinition* nv_network) const
{
    return tfrt::scope(nv_network, this, this->name());
}

bool network::load_weights(std::string filename)
{
    this->clear();
    // Highly inspired by Caffe source code!
    int fd = open(filename.c_str(), O_RDONLY);
    CHECK_NE(fd, -1) << "FILE not found: " << filename;
    ZeroCopyInputStream* raw_input = new FileInputStream(fd);
    CodedInputStream* coded_input = new CodedInputStream(raw_input);
    coded_input->SetTotalBytesLimit(kProtoReadBytesLimit, 536870912);
    bool success = m_pb_network.ParseFromCodedStream(coded_input);
    delete coded_input;
    delete raw_input;
    close(fd);
    return success;
}


/* ============================================================================
 * Getters / setters wrapping protobuf methods.
 * ========================================================================== */
const tfrt_pb::tensor& network::tensor_by_name(std::string name) const
{
    // Best search algorithm ever!
    for(int i = 0 ; i < m_pb_network.weights_size() ; ++i) {
        const tfrt_pb::tensor& tensor = m_pb_network.weights(i);
        if(tensor.name() == name) {
            LOG(INFO) << "FOUND tfrt_pb::tensor '" << name << "'. "
                << "SHAPE: " << dims_str(tensor_shape(tensor)) << " "
                << "SIZE: " << tensor.size();
            return tensor;
        }
    }
    // Default empty tensor.
    LOG(WARNING) << "FAILED to find the tfrt_pb::tensor '" << name
        << "'. Using default empty tensor." ;
    return tfrt_pb::tensor::default_instance();
}
nvinfer1::Weights network::weights_by_name(std::string name) const
{
    const tfrt_pb::tensor& tensor = tensor_by_name(name);
    return tensor_to_weights(tensor);
}

nvinfer1::Weights network::tensor_to_weights(const tfrt_pb::tensor& tensor)
{
    nvinfer1::Weights w{
        .type = nvinfer1::DataType(int(tensor.datatype())),
        .values = tensor.data().data(),
        .count = int(tensor.size())
    };
    if(w.count == 0) {
        w.values = nullptr;
    }
    return w;
}
const std::string& network::name() const {
    return m_pb_network.name();
}
network& network::name(const std::string& name) {
    m_pb_network.set_name(name);
    return *this;
}
nvinfer1::DataType network::datatype() const {
    // Datatype of the network. Hopefully consistent with weights...
    auto dt = m_pb_network.datatype();
    return nvinfer1::DataType(int(dt));
}

nvinfer1::DimsHW network::input_shape() const
{
    return nvinfer1::DimsHW{m_pb_network.input().height(), m_pb_network.input().width()};
}
const std::string& network::input_name() const
{
    return m_pb_network.input().name();
}
std::vector<nvinfer1::DimsHW> network::outputs_shape() const
{
    std::vector<nvinfer1::DimsHW> v;
    for(int i = 0 ; i < m_pb_network.outputs_size() ; ++i) {
        const tfrt_pb::output& output = m_pb_network.outputs(i);
        v.push_back(nvinfer1::DimsHW{output.height(), output.width()});
    }
    return v;
}
std::vector<std::string> network::outputs_name() const
{
    std::vector<std::string> v;
    for(int i = 0 ; i < m_pb_network.outputs_size() ; ++i) {
        const tfrt_pb::output& output = m_pb_network.outputs(i);
        v.push_back(output.name());
    }
    return v;
}

/* ============================================================================
 * Private tfrt::network methods... Tambouille interne.
 * ========================================================================== */
void network::clear_weights()
{
    m_pb_network.clear_weights();
}

}
