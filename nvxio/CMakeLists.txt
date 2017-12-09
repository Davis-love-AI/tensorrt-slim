# Find all NVXIO sources files!
FILE(GLOB NVXIO_SRCS src/NVX/Private/*.cpp src/NVX/FrameSource/OpenCV/*.cpp src/NVX/FrameSource/NvMedia/OV10640/*.cpp src/NVX/FrameSource/NvMedia/OV10635/*.cpp src/NVX/FrameSource/NvMedia/*.cpp src/NVX/FrameSource/GStreamer/*.cpp src/NVX/FrameSource/Wrappers/*.cpp src/NVX/FrameSource/*.cpp src/NVX/Render/OpenCV/*.cpp src/NVX/Render/GStreamer/*.cpp src/NVX/Render/CUDA-OpenGL/*.cpp src/NVX/Render/Wrappers/*.cpp src/NVX/Render/*.cpp src/NVX/*.cpp src/OVX/Private/*.cpp src/OVX/FrameSource/Wrappers/*.cpp src/OVX/FrameSource/*.cpp src/OVX/Render/Wrappers/*.cpp src/OVX/Render/*.cpp src/OVX/*.cpp src/*.cpp src/NVX/Private/*.cpp src/NVX/FrameSource/OpenCV/*.cpp src/NVX/FrameSource/NvMedia/OV10640/*.cpp src/NVX/FrameSource/NvMedia/OV10635/*.cpp src/NVX/FrameSource/NvMedia/*.cpp src/NVX/FrameSource/GStreamer/*.cpp src/NVX/FrameSource/Wrappers/*.cpp src/NVX/FrameSource/*.cpp src/NVX/Render/OpenCV/*.cpp src/NVX/Render/GStreamer/*.cpp src/NVX/Render/CUDA-OpenGL/*.cpp src/NVX/Render/Wrappers/*.cpp src/NVX/Render/*.cpp src/NVX/*.cpp src/NVX/FrameSource/NvMedia/OV10640/*.c)

FILE(GLOB NVXIO_HEADERS include/NVX/*.h include/OVX/*.h include/NVX/detail*.h)

# NVXIO flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DCUDA_API_PER_THREAD_DEFAULT_STREAM -DUSE_GUI=1 -DUSE_GLFW=1  -DUSE_GSTREAMER=1")
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DUSE_OPENCV=1")

# Build NVXIO
include_directories(include)
cuda_add_library(nvxio STATIC ${NVXIO_SRCS})
target_link_libraries(nvxio ${GSTREAMER_LIBRARIES} ${GSTREAMER_APP_LIBRARIES} ${GSTREAMER_PBUTILS_LIBRARIES} glib-2.0 gobject-2.0 freetype glfw3 GL Xrandr Xi Xxf86vm X11 visionworks)
# Copy NVXIO headers
add_subdirectory(include)

# GL Xi X11 x264 opencv_calib3d opencv_contrib opencv_core opencv_features2d opencv_flann opencv_gpu opencv_highgui opencv_imgproc opencv_legacy opencv_ml opencv_objdetect opencv_photo opencv_stitching opencv_superres opencv_ts opencv_video opencv_videostab opencv_facedetect opencv_imuvstab opencv_tegra opencv_vstab

# Installation targets...
install(TARGETS nvxio DESTINATION lib)
