# Find all NVXIO sources files!
FILE(GLOB NVXIO_SRCS src/NVX/Private/*.cpp src/NVX/FrameSource/OpenCV/*.cpp src/NVX/FrameSource/NvMedia/OV10640/*.cpp src/NVX/FrameSource/NvMedia/OV10635/*.cpp src/NVX/FrameSource/NvMedia/*.cpp src/NVX/FrameSource/GStreamer/*.cpp src/NVX/FrameSource/Wrappers/*.cpp src/NVX/FrameSource/*.cpp src/NVX/Render/OpenCV/*.cpp src/NVX/Render/GStreamer/*.cpp src/NVX/Render/CUDA-OpenGL/*.cpp src/NVX/Render/Wrappers/*.cpp src/NVX/Render/*.cpp src/NVX/*.cpp src/OVX/Private/*.cpp src/OVX/FrameSource/Wrappers/*.cpp src/OVX/FrameSource/*.cpp src/OVX/Render/Wrappers/*.cpp src/OVX/Render/*.cpp src/OVX src/*.cpp src/NVX/Private/*.cpp src/NVX/FrameSource/OpenCV/*.cpp src/NVX/FrameSource/NvMedia/OV10640/*.cpp src/NVX/FrameSource/NvMedia/OV10635/*.cpp src/NVX/FrameSource/NvMedia/*.cpp src/NVX/FrameSource/GStreamer/*.cpp src/NVX/FrameSource/Wrappers/*.cpp src/NVX/FrameSource/*.cpp src/NVX/Render/OpenCV/*.cpp src/NVX/Render/GStreamer/*.cpp src/NVX/Render/CUDA-OpenGL/*.cpp src/NVX/Render/Wrappers/*.cpp src/NVX/Render/*.cpp src/NVX/*.cpp src/NVX/FrameSource/NvMedia/OV10640/*.c)

FILE(GLOB NVXIO_HEADERS include/NVX/*.h include/OVX/*.h include/NVX/detail*.h)

# Build NVXIO
cuda_add_library(nvxio STATIC ${NVXIO_SRCS})
target_link_libraries(nvxio gstreamer-1.0 gstapp-1.0 glib-2.0 gobject-2.0 freetype2 GL GLEW)

# # Copy TF-RT headers
# set(HEADERS_DIRS . nets cuda misc models boxes2d boxes3d)
# BUILD_COPY_HEADERS(HEADERS_DIRS)

# foreach(HEADER ${PROTO_HEADERS})
# 	message("-- Copying ${HEADER}")
# 	configure_file(${HEADER} ${PROJECT_INCLUDE_DIR} COPYONLY)
# endforeach(HEADER)
