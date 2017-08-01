# Demo of video stabilisation using NVX.
FILE(GLOB NVX_VIDEO_STAB_SRCS *.cpp)

cuda_add_executable(nvx_video_stabilizer ${NVX_VIDEO_STAB_SRCS})
target_link_libraries(nvx_video_stabilizer nvxio GL Xi X11 x264 opencv_calib3d opencv_contrib opencv_core opencv_features2d opencv_flann opencv_gpu opencv_highgui opencv_imgproc opencv_legacy opencv_ml opencv_objdetect opencv_photo opencv_stitching opencv_superres opencv_ts opencv_video opencv_videostab opencv_facedetect opencv_imuvstab opencv_tegra opencv_vstab )
