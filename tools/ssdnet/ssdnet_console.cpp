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
#include <sys/time.h>

// TensorFlowRT headers
#include <tensorflowrt.h>
#include <tensorflowrt_util.h>
#include <tensorflowrt_ssd_models.h>

#define SSDNET "<ssdnet-console> "
// FLAGS...
DEFINE_string(network, "ssd_inception2_v0", "SSD network network to test.");
DEFINE_string(network_pb, "../data/networks/ssd_inception2_v0_orig.tfrt32",
    "Network protobuf parameter file.");
DEFINE_string(image, "../data/images/peds-001.jpg",
    "Image to use for detection..");
DEFINE_bool(image_save, false, "Save the result in some new image.");


// uint64_t current_timestamp() {
//     struct timeval te;
//     gettimeofday(&te, NULL); // get current time
//     return te.tv_sec*1000LL + te.tv_usec/1000; // caculate milliseconds
// }

/** Map-method containing the list of available SSD networks.
 */
std::unique_ptr<tfrt::ssd_network>&& networks_map(const std::string& key)
{
    static std::map<std::string, std::unique_ptr<tfrt::ssd_network> > nets;
    // Fill the map at first call!
    if(nets.empty()) {
        nets["ssd_inception2_v0"] = std::make_unique<ssd_inception2_v0::net>();
    }
    return std::move(nets.at(key));
}

// main entry point
int main( int argc, char** argv )
{
    google::InitGoogleLogging(argv[0]);
    gflags::ParseCommandLineFlags(&argc, &argv, true);

    bool r;
    LOG(INFO) << SSDNET << "Loading network: " << FLAGS_network;
    auto network = networks_map(FLAGS_network);
    // network->EnableProfiler();
    network->load(FLAGS_network_pb);

    // Load image from file on disk.
    LOG(INFO) << SSDNET << "Opening image: " << FLAGS_image;
    tfrt::cuda_tensor img("image", {1, 1, 0, 0});
    r = loadImageRGBA(FLAGS_image.c_str(), (float4**)&img.cpu, (float4**)&img.cuda,
        &img.shape.w(), &img.shape.h());
    CHECK(r) << SSDNET << "Failed to read image file: " << FLAGS_image;

    // Raw 2D detections.
    tfrt::boxes2d::bboxes2d bboxes2d;
    size_t max_detections{200};
    float threshold = 0.5;
    LOG(INFO) << SSDNET << "Detecting object on image...";
    bboxes2d = network->raw_detect2d(
        img.cuda, img.shape.h(), img.shape.w(), threshold, max_detections);
    LOG(INFO) << SSDNET << "Raw 2D objects: " << bboxes2d;

    // Save images...
    if(FLAGS_image_save) {
        LOG(INFO) << SSDNET << "Print 2D bounding boxes on images.";


    }
    // else if( argc > 2 )		// if the user supplied an output filename
    // {
        // printf("%i bounding boxes detected\n", numBoundingBoxes);

        // int lastClass = 0;
        // int lastStart = 0;

        // for( int n=0; n < numBoundingBoxes; n++ )
        // {
        //     const int nc = confCPU[n*2+1];
        //     float* bb = bbCPU + (n * 4);

        //     printf("bounding box %i   (%f, %f)  (%f, %f)  w=%f  h=%f\n", n, bb[0], bb[1], bb[2], bb[3], bb[2] - bb[0], bb[3] - bb[1]);

        //     if( nc != lastClass || n == (numBoundingBoxes - 1) )
        //     {
        //         if( !net->DrawBoxes(imgCUDA, imgCUDA, imgWidth, imgHeight, bbCUDA + (lastStart * 4), (n - lastStart) + 1, lastClass) )
        //             printf("detectnet-console:  failed to draw boxes\n");

        //         lastClass = nc;
        //         lastStart = n;
        //     }
        // }

    //     CUDA(cudaThreadSynchronize());

    //     // save image to disk
    //     printf("detectnet-console:  writing %ix%i image to '%s'\n", imgWidth, imgHeight, argv[2]);

    //     if( !saveImageRGBA(argv[2], (float4*)imgCPU, imgWidth, imgHeight, 255.0f) )
    //         printf("detectnet-console:  failed saving %ix%i image to '%s'\n", imgWidth, imgHeight, argv[2]);
    //     else
    //         printf("detectnet-console:  successfully wrote %ix%i image to '%s'\n", imgWidth, imgHeight, argv[2]);

    // }
    // //printf("detectnet-console:  '%s' -> %2.5f%% class #%i (%s)\n", imgFilename, confidence * 100.0f, img_class, "pedestrian");

    return 0;
}
