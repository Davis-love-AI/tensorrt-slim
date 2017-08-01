# Demo of video stabilisation using NVX.
FILE(GLOB NVX_VIDEO_STAB_SRCS *.cpp)

cuda_add_executable(nvx_video_stabilizer ${NVX_VIDEO_STAB_SRCS})
target_link_libraries(nvx_video_stabilizer nvxio GL Xi X11 x264)
