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
#include <signal.h>
#include <unistd.h>

// TensorFlowRT headers
#include <tensorflowrt.h>
#include <tensorflowrt_util.h>
#include <nets/nets.h>

#include "cuda/cudaNormalize.h"
#include "cuda/cudaImageNet.h"

// #include "gstCamera.h"
// #include "glDisplay.h"
// #include "glTexture.h"

// #include <stdio.h>
// #include "cudaFont.h"

// FLAGS...
DEFINE_string(network, "inception2", "ImageNet network to test.");
DEFINE_string(network_pb, "../data/networks/inception_v2_fused.tfrt16",
    "Network protobuf parameter file.");
DEFINE_string(imagenet_info, "../data/networks/ilsvrc12_synset_words.txt",
    "ImageNet information (classes, ...).");


#define IMGNET "<imagenet-camera> "
#define DEFAULT_CAMERA -1	// -1 for onboard, or chg to idx of /dev/video V4L2 camera (>=0)

bool signal_recieved = false;
void sig_handler(int signo) {
    if( signo == SIGINT ) {
        printf("received SIGINT\n");
        signal_recieved = true;
    }
}

/** Map-method containing the list of available ImageNet networks.
 * Return a unique_ptr of a network, releasing the resource.
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

    LOG(INFO) << IMGNET << "Classification of camera input...";
    // Attach signal handler
    if( signal(SIGINT, sig_handler) == SIG_ERR ) {
        printf("\ncan't catch SIGINT\n");
    }

    // Create the camera device.
    gstCamera* camera = gstCamera::Create(DEFAULT_CAMERA);
    if( !camera ) {
        printf("\nimagenet-camera:  failed to initialize video device\n");
        return 0;
    }
    printf("\nimagenet-camera:  successfully initialized video device\n");
    printf("    width:  %u\n", camera->GetWidth());
    printf("   height:  %u\n", camera->GetHeight());
    printf("    depth:  %u (bpp)\n\n", camera->GetPixelDepth());


    LOG(INFO) << IMGNET << "Loading network: " << FLAGS_network;
    // Get network and load parameters & weights.
    auto network = networks_map(FLAGS_network);
    // network->EnableProfiler();
    network->load(FLAGS_network_pb);
    network->load_info(FLAGS_imagenet_info);

    // Create openGL window
    glDisplay* display = glDisplay::Create();
    glTexture* texture = NULL;
    if( !display ) {
        printf("\nimagenet-camera:  failed to create openGL display\n");
    }
    else {
        texture = glTexture::Create(camera->GetWidth(), camera->GetHeight(), GL_RGBA32F_ARB/*GL_RGBA8*/);
        if( !texture )
            printf("imagenet-camera:  failed to create openGL texture\n");
    }

    // CUDA font
    cudaFont* font = cudaFont::Create();
    // start streaming
    if( !camera->Open() ) {
        printf("\nimagenet-camera:  failed to open camera for streaming\n");
        return 0;
    }
    printf("\nimagenet-camera:  camera open for streaming\n");

    while( !signal_recieved )
    {
        void* imgCPU  = NULL;
        void* imgCUDA = NULL;

        // get the latest frame
        if( !camera->Capture(&imgCPU, &imgCUDA, 1000) )
            printf("\nimagenet-camera:  failed to capture frame\n");
        //else
        //	printf("imagenet-camera:  recieved new frame  CPU=0x%p  GPU=0x%p\n", imgCPU, imgCUDA);

        // convert from YUV to RGBA
        void* imgRGBA = NULL;
        if( !camera->ConvertRGBA(imgCUDA, &imgRGBA) )
            printf("imagenet-camera:  failed to convert from NV12 to RGBA\n");

        // classify image
        auto imgclass = network->classify((float*)imgRGBA, camera->GetHeight(), camera->GetWidth());
        int img_class = imgclass.first;
        float confidence = imgclass.second;

        if( img_class >= 0 )
        {
            printf("imagenet-camera:  %2.5f%% class #%i (%s)\n", confidence * 100.0f, img_class, network->description(img_class).c_str());

            if( font != NULL )
            {
                char str[256];
                sprintf(str, "%05.2f%% %s", confidence * 100.0f, network->description(img_class).c_str());

                font->RenderOverlay((float4*)imgRGBA, (float4*)imgRGBA, camera->GetWidth(), camera->GetHeight(), str, 0, 0, make_float4(255.0f, 255.0f, 255.0f, 255.0f));
            }
            if( display != NULL )
            {
                char str[256];
                sprintf(str, "TensorRT build %x | %s | %i | %04.1f FPS", NV_GIE_VERSION, network->name().c_str(), int(network->datatype()), display->GetFPS());
                //sprintf(str, "TensorRT build %x | %s | %04.1f FPS | %05.2f%% %s", NV_GIE_VERSION, net->GetNetworkName(), display->GetFPS(), confidence * 100.0f, net->GetClassDesc(img_class));
                display->SetTitle(str);
            }
        }
        // update display
        if( display != NULL )
        {
            display->UserEvents();
            display->BeginRender();

            if( texture != NULL )
            {
                // rescale image pixel intensities for display
                CUDA(cudaNormalizeRGBA((float4*)imgRGBA, make_float2(0.0f, 255.0f),
                                   (float4*)imgRGBA, make_float2(0.0f, 1.0f),
                                    camera->GetWidth(), camera->GetHeight()));
                // map from CUDA to openGL using GL interop
                void* tex_map = texture->MapCUDA();
                if( tex_map != NULL )
                {
                    cudaMemcpy(tex_map, imgRGBA, texture->GetSize(), cudaMemcpyDeviceToDevice);
                    texture->Unmap();
                }
                // draw the texture
                texture->Render(100,100);
            }
            display->EndRender();
        }
    }
    printf("\nimagenet-camera:  un-initializing video device\n");
    if( camera != NULL ) {
        delete camera;
        camera = NULL;
    }
    if( display != NULL ) {
        delete display;
        display = NULL;
    }
    printf("imagenet-camera:  video device has been un-initialized.\n");
    printf("imagenet-camera:  this concludes the test of the video device.\n");
    return 0;
}

