# TensorRT samples.
cuda_add_executable(trt_giexec giexec.cpp)
target_link_libraries(trt_giexec nvinfer nvparsers nvinfer_plugin glog gflags)

cuda_add_executable(trt_googlenet sampleGoogleNet.cpp common.cpp)
target_link_libraries(trt_googlenet nvinfer nvparsers nvinfer_plugin glog gflags)

# Installation
install(TARGETS trt_giexec DESTINATION bin)
install(TARGETS trt_googlenet DESTINATION bin)
