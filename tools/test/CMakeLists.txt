# Multiple small programs...

# TF-RT giexec
cuda_add_executable(tfrt_giexec tfrt_giexec.cpp)
target_link_libraries(tfrt_giexec nvinfer tensorflowrt glog gflags)

# Testing some CUDA functions...
cuda_add_executable(cuda_tests cuda_tests.cpp cuda_tests.cu)
target_link_libraries(cuda_tests tensorflowrt glog gflags)

# Installation
install(TARGETS tfrt_giexec DESTINATION bin)
