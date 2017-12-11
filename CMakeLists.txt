# Let's cmake it! Inspired by jetson-inference configuration.
cmake_minimum_required(VERSION 3.0)
project(ssd-tensorrt)

# gflags and glog libraries
set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
message("-- CMAKE modules path:  ${CMAKE_MODULE_PATH}")
find_package(glog REQUIRED)
find_package(gflags REQUIRED)

# Find freetype, X11
find_package(Freetype REQUIRED)
find_package(X11 REQUIRED)
# Qt is used to load images
find_package(Qt4 REQUIRED)
include(${QT_USE_FILE})
add_definitions(${QT_DEFINITIONS})
# Find GLIB and GStreamer
find_package(GLIB REQUIRED)
find_package(GStreamer REQUIRED)
# ROS packages.
# find_package(catkin REQUIRED)

# Set-up CUDA
find_package(CUDA)
set(
	CUDA_NVCC_FLAGS
	${CUDA_NVCC_FLAGS};
    -O3
    # -G -g
	-gencode arch=compute_50,code=sm_50
	-gencode arch=compute_52,code=sm_52
	-gencode arch=compute_53,code=sm_53
	-gencode arch=compute_62,code=sm_62
)

# Main CXX flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -fPIC -std=c++11")	# -std=gnu++11
# set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -w -Xcompiler -fPIC -std=c++11" )
set(BUILD_DEPS "YES" CACHE BOOL "If YES, will install dependencies into sandbox.  Automatically reset to NO after dependencies are installed.")

# Setup project output paths
set(PROJECT_OUTPUT_DIR  ${PROJECT_BINARY_DIR}/${CMAKE_SYSTEM_PROCESSOR})
set(PROJECT_INCLUDE_DIR ${PROJECT_OUTPUT_DIR}/include)
file(MAKE_DIRECTORY ${PROJECT_OUTPUT_DIR}/bin)
file(MAKE_DIRECTORY ${PROJECT_OUTPUT_DIR}/data)
file(MAKE_DIRECTORY ${PROJECT_INCLUDE_DIR})

message("-- SYSTEM arch:  ${CMAKE_SYSTEM_PROCESSOR}")
message("-- OUTPUT path:  ${PROJECT_OUTPUT_DIR}")

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_OUTPUT_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_OUTPUT_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_OUTPUT_DIR}/lib)

# C/C++ include paths.
# message("-- ROS:  ${CATKIN_INCLUDES}")
include_directories(${PROJECT_INCLUDE_DIR} ${GIE_PATH}/include)
include_directories(${GLIB_INCLUDE_DIRS})
include_directories(${GSTREAMER_INCLUDE_DIRS})
include_directories(${CATKIN_INCLUDE_DIRS})
include_directories(/usr/include/opencv)
include_directories(/usr/include/freetype2)
include_directories(/opt/ros/kinetic/include)

# include_directories(/usr/include/gstreamer-1.0 /usr/include/glib-2.0 /usr/include/libxml2)
# include_directories(/usr/lib/aarch64-linux-gnu/glib-2.0/include /usr/lib/aarch64-linux-gnu/gstreamer-1.0/include)
# include_directories(/usr/lib/x86_64-linux-gnu/glib-2.0/include /usr/lib/x86_64-linux-gnu/gstreamer-1.0/include)
# OpenCV and freetype2 include dir

# Third party libraries.
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/fmt)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/eigen)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/half/include)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/glfw3/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/opengl)
link_directories(${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/glfw3/lib)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/nvxio/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/nvxio/src)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/nvxio/src/NVX)

# Copy headers to include directory.
MACRO(BUILD_COPY_HEADERS DIRS_LIST)
    foreach(HEADER_DIR ${${DIRS_LIST}})
        message("-- Copying headers from ${HEADER_DIR}")
	    file(GLOB TMP_TFRT_HEADERS ${HEADER_DIR}/*.h ${HEADER_DIR}/*.hpp)
        foreach(HEADER ${TMP_TFRT_HEADERS})
            message("-- Copying ${HEADER}")
            set(HEADERS_DIR_BUILD ${PROJECT_INCLUDE_DIR}/${HEADER_DIR})
            file(MAKE_DIRECTORY ${HEADERS_DIR_BUILD})
            # Copy the file to build directory.
            configure_file(${HEADER} ${HEADERS_DIR_BUILD} COPYONLY)
            # Configure for installation as well...
            install(FILES ${HEADER} DESTINATION include/${HEADER_DIR})
        endforeach(HEADER)
    endforeach(HEADER_DIR)
ENDMACRO(BUILD_COPY_HEADERS)

# TF-RT include
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/tensorflowrt)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/util)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/modules)

# Build NVXIO dependency.
add_subdirectory(nvxio)
# 3rd party libraries rules (install, ...)
add_subdirectory(3rdparty)

# Build TensorFlowRT library and modules.
add_subdirectory(tensorflowrt)
add_subdirectory(modules)
add_subdirectory(util)

# Build tools and demos...
add_subdirectory(tools)
add_subdirectory(demos)

