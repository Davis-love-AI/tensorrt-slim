# Multiple small programs...

# TF-RT giexec
cuda_add_executable(tfrt_giexec tfrt_giexec.cpp)
target_link_libraries(tfrt_giexec nvinfer tensorflowrt glog gflags)
