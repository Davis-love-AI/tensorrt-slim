# Let's cmake it! Inspired by jetson-inference configuration.
cmake_minimum_required(VERSION 3.0)
project(ssd-tensorrt)

# TensorRT flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -std=c++11")	# -std=gnu++11
set(BUILD_DEPS "YES" CACHE BOOL "If YES, will install dependencies into sandbox.  Automatically reset to NO after dependencies are installed.")

# Qt is used to load images
find_package(Qt4 REQUIRED)
include(${QT_USE_FILE})
add_definitions(${QT_DEFINITIONS})

# Setup CUDA
find_package(CUDA)
set(
	CUDA_NVCC_FLAGS
	${CUDA_NVCC_FLAGS};
    -O3
	-gencode arch=compute_53,code=sm_53
	-gencode arch=compute_62,code=sm_62
)

# Setup project output paths
set(PROJECT_OUTPUT_DIR  ${PROJECT_BINARY_DIR}/${CMAKE_SYSTEM_PROCESSOR})
set(PROJECT_INCLUDE_DIR ${PROJECT_OUTPUT_DIR}/include)
file(MAKE_DIRECTORY ${PROJECT_OUTPUT_DIR}/bin)
file(MAKE_DIRECTORY ${PROJECT_INCLUDE_DIR})

message("-- SYSTEM arch:  ${CMAKE_SYSTEM_PROCESSOR}")
message("-- OUTPUT path:  ${PROJECT_OUTPUT_DIR}")

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_OUTPUT_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_OUTPUT_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_OUTPUT_DIR}/lib)

# C/C++ include paths.
include_directories(${PROJECT_INCLUDE_DIR} ${GIE_PATH}/include)
include_directories(/usr/include/gstreamer-1.0 /usr/lib/aarch64-linux-gnu/gstreamer-1.0/include /usr/include/glib-2.0 /usr/include/libxml2 /usr/lib/aarch64-linux-gnu/glib-2.0/include)
include_directories(/usr/lib/x86_64-linux-gnu/glib-2.0/include /usr/lib/x86_64-linux-gnu/gstreamer-1.0/include)

# Build TensorFlowRT library.
add_subdirectory(tensorflowrt)


# file(GLOB inferenceSources *.cpp *.cu util/*.cpp util/camera/*.cpp util/cuda/*.cu util/display/*.cpp)
# file(GLOB inferenceIncludes *.h util/*.h util/camera/*.h util/cuda/*.h util/display/*.h)

# cuda_add_library(ssd-tensorrt SHARED ${inferenceSources})
# target_link_libraries(ssd-tensorrt nvinfer Qt4::QtGui GL GLEW gstreamer-1.0 gstapp-1.0)


# # transfer all headers to the include directory
# foreach(include ${inferenceIncludes})
# 	message("-- Copying ${include}")
# 	configure_file(${include} ${PROJECT_INCLUDE_DIR} COPYONLY)
# endforeach()

