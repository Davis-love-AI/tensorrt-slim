# TF-RT util sources.
FILE(GLOB TFRT_UTIL_SRCS *.cpp cuda/*.cpp cuda/*.cu display/*.cpp camera/*.cpp)
FILE(GLOB TFRT_UTIL_HEADERS *.h)

# Build TensorFlowRT-util
cuda_add_library(tensorflowrt_util STATIC ${TFRT_UTIL_SRCS})
target_link_libraries(tensorflowrt_util nvinfer gstreamer-1.0 gstapp-1.0 glib-2.0 gobject-2.0 Qt4::QtGui GL GLEW glog gflags)

# Copy TF-RT headers
set(HEADERS_DIRS . cuda camera display)
BUILD_COPY_HEADERS(HEADERS_DIRS)
