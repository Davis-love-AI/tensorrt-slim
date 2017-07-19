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
#include <glog/logging.h>
#include <gflags/gflags.h>

#include <map>
#include <string>
#include <iomanip>

// TensorFlowRT headers
#include <tensorflowrt.h>
#include <tensorflowrt_util.h>
#include <tensorflowrt_nets.h>

#define IMGNET "<imagenet-console> "
// FLAGS...
DEFINE_string(network, "inception2", "ImageNet network to test.");
DEFINE_string(network_pb, "../data/networks/inception_v2_fused.tfrt16",
    "Network protobuf parameter file.");
DEFINE_string(imagenet_info, "../data/networks/ilsvrc12_synset_words.txt",
    "ImageNet information (classes, ...).");
DEFINE_string(image, "../data/images/orange_0.jpg",
    "Image to classify.");
DEFINE_bool(image_save, false, "Save the result in some new image.");


/** Map-method containing the list of available ImageNet networks.
 */
std::unique_ptr<tfrt::imagenet_network>&& networks_map(const std::string& key)
{
    static std::map<std::string, std::unique_ptr<tfrt::imagenet_network> > nets;
    // Fill the map at first call!
    if(nets.empty()) {
        nets["inception1"] = std::make_unique<inception1::net>();
        nets["inception2"] = std::make_unique<inception2::net>();
    }
    return std::move(nets.at(key));
}


int main( int argc, char** argv )
{
    google::InitGoogleLogging(argv[0]);
	gflags::ParseCommandLineFlags(&argc, &argv, true);

    bool r;
    LOG(INFO) << IMGNET << "Loading network: " << FLAGS_network;
    // Get network and load parameters & weights.
    auto network = networks_map(FLAGS_network);
    // network->EnableProfiler();
    network->load(FLAGS_network_pb);
    network->load_info(FLAGS_imagenet_info);

    // Load image from file on disk.
    LOG(INFO) << IMGNET << "Opening image: " << FLAGS_image;
    tfrt::cuda_tensor img("image", {1, 1, 0, 0});
    r = loadImageRGBA(FLAGS_image.c_str(), (float4**)&img.cpu, (float4**)&img.cuda,
        &img.shape.w(), &img.shape.h());
    CHECK(r) << IMGNET << "Failed to read image file: " << FLAGS_image;

    // Classifying image.
    auto imgclass = network->classify(img.cuda, img.shape.h(), img.shape.w());

    if(imgclass.first >= 0) {
        LOG(INFO) << IMGNET
            << "Classification result: " << network->description(imgclass.first)
            << " with confidence " << std::setprecision(10) << imgclass.second;

        if(FLAGS_image_save) {
            std::string  output_filename = FLAGS_image + ".class";
            // Overlay the classification on the image
            cudaFont* font = cudaFont::Create();
            if(font != NULL) {
                char str[512];
                sprintf(str, "%2.3f%% %s",
                    imgclass.second * 100.0f,
                    network->description(imgclass.first).c_str());
                const int overlay_x = 10;
                const int overlay_y = 10;
                const int px_offset = overlay_y * img.shape.w() * 4 + overlay_x * 4;

                // If the image has a white background, use black text (otherwise, white)
                const float white_cutoff = 225.0f;
                bool white_background = false;
                if( img.cpu[px_offset] > white_cutoff && img.cpu[px_offset + 1] > white_cutoff && img.cpu[px_offset + 2] > white_cutoff ) {
                    white_background = true;
                }
                // overlay the text on the image
                font->RenderOverlay((float4*)img.cuda, (float4*)img.cuda,
                    img.shape.w(), img.shape.h(), (const char*)str, 10, 10,
                    white_background ? make_float4(0.0f, 0.0f, 0.0f, 255.0f) : make_float4(255.0f, 255.0f, 255.0f, 255.0f));
            }
            LOG(INFO) << IMGNET << "Saving the output image to: " << output_filename;
            r = saveImageRGBA(output_filename.c_str(), (float4*)img.cpu,
                img.shape.w(), img.shape.h());
            CHECK(r) << IMGNET << "Failed to save the output image to:" << output_filename;
        }
    }
    else {
        LOG(WARNING) << IMGNET << "Failed to classify the image.";
    }
    // Free network: avoid CUDA problems...
    // network.reset(nullptr);
    return 0;
}
