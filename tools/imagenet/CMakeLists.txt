# ImageNet console and camera test programs
cuda_add_executable(imagenet_console imagenet_console.cpp)
target_link_libraries(imagenet_console nvinfer tensorflowrt tensorflowrt_util glog gflags)

# cuda_add_executable(imagenet_camera imagenet_camera.cpp)
# target_link_libraries(imagenet_camera nvinfer tensorflowrt glog gflags)
