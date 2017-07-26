# ImageNet console and camera test programs
cuda_add_executable(ssdnet_console ssdnet_console.cpp)
target_link_libraries(ssdnet_console tensorflowrt_util tensorflowrt)

cuda_add_executable(ssdnet_camera ssdnet_camera.cpp)
target_link_libraries(ssdnet_camera tensorflowrt_util tensorflowrt X11)
