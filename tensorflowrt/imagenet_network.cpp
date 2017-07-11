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
#include <gflags/gflags.h>
#include <glog/logging.h>

#include "imagenet_network.h"

namespace tfrt
{

bool imagenet_network::load_info(const std::string& filename)
{
    // HIGHLY inspired by Jetson Inference!
    LOG(INFO) << "Loading ImageNet descriptions from: " << filename;
    FILE* f = fopen(filename.c_str(), "r");
    CHECK_NOTNULL(f);

    uint32_t mCustomClasses{0};
    char str[512];
    while( fgets(str, 512, f) != NULL ) {
        const int syn = 9;  // length of synset prefix (in characters)
        const int len = strlen(str);
        if( len > syn && str[0] == 'n' && str[syn] == ' ' ) {
            str[syn]   = 0;
            str[len-1] = 0;
            const std::string a = str;
            const std::string b = (str + syn + 1);
            //printf("a=%s b=%s\n", a.c_str(), b.c_str());

            m_synset_classes.push_back(a);
            m_desc_classes.push_back(b);
        }
        else if(len > 0) {  // no 9-character synset prefix (i.e. from DIGITS snapshot)
            char a[10];
            sprintf(a, "n%08u", mCustomClasses);
            //printf("a=%s b=%s (custom non-synset)\n", a, str);
            mCustomClasses++;
            if( str[len-1] == '\n' ) {
                str[len-1] = 0;
            }
            m_synset_classes.push_back(a);
            m_desc_classes.push_back(str);
        }
    }
    fclose(f);
    LOG(INFO) << "ImageNet loaded with #entries: " << m_synset_classes.size();
    if(m_synset_classes.size() == 0) {
        return false;
    }
    return true;
}



}