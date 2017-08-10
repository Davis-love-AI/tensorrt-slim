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
#ifndef TFRT_SCOPE_H
#define TFRT_SCOPE_H

#include <glog/logging.h>
#include <NvInfer.h>

#include "tfrt_jetson.h"
#include "network.h"

namespace tfrt
{
/**
 * \class scope
 * \brief structure mimicking TensorFlow scope, with a few additional roles:
 weights data, ?
 */
class scope
{
public:
    /** Create a scope with a default name.
     */
    scope(nvinfer1::INetworkDefinition* nv_network,
          const tfrt::network* tf_network,
          const std::string& name="") :
            m_nv_network(CHECK_NOTNULL(nv_network)),
            m_tf_network(CHECK_NOTNULL(tf_network)),
            m_name(name) {
    }
    /** Create a sub-scope from the current one.
     * Returns a copy of the scope with a new name.
     */
    scope sub(const std::string& subname="") const {
        scope subscope(*this);
        subscope.m_name = subscope.subname(subname);
        return subscope;
    }

public:
    /** Get the parent network objects. */
    nvinfer1::INetworkDefinition* network() const {  return m_nv_network;  }
    const tfrt::network* tfrt_network() const {  return m_tf_network;  }
    /** Get the scope name, as a std::string. */
    std::string name() const {  return m_name;  }
    /** Get the scope name, as a std::string. */
    const char* cname() const {  return m_name.c_str();  }

    /** Generate a sub-scope name. */
    std::string subname(std::string subname) const {
        if(subname.length() > 0) {
            if(m_name.length() > 0 and m_name[m_name.length()-1] != '/') {
                return m_name + "/" + subname;
            }
            else {
                return m_name + subname;    // Already '/' or empty name.
            }
        } else {
            return m_name;      // Empty subname: do nothing.
        }
    }

public:
    /** Get the weights from this scope (and with specific name).
     */
    nvinfer1::Weights weights(const std::string& wname, const nvinfer1::Dims& wshape) const {
        return m_tf_network->weights_by_name(subname(wname), wshape);
    }

protected:
    // Parent network.
    nvinfer1::INetworkDefinition*  m_nv_network;
    // Parent TFRT network.
    const tfrt::network*  m_tf_network;
    // Scope name.
    std::string  m_name;
};

}

#endif
