# Let's cmake it! Inspired by jetson-inference configuration.
cmake_minimum_required(VERSION 3.0)
project(ssd-tensorrt)

# gflags and glog libraries
set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
message("-- CMAKE modules path:  ${CMAKE_MODULE_PATH}")
find_package(glog REQUIRED)
find_package(gflags REQUIRED)

# TensorRT flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -fPIC -std=c++11")	# -std=gnu++11
# set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -w -Xcompiler -fPIC -std=c++11" )
set(BUILD_DEPS "YES" CACHE BOOL "If YES, will install dependencies into sandbox.  Automatically reset to NO after dependencies are installed.")

# Qt is used to load images
find_package(Qt4 REQUIRED)
include(${QT_USE_FILE})
add_definitions(${QT_DEFINITIONS})
# Find freetype
find_package(Freetype REQUIRED)

# Set-up CUDA
find_package(CUDA)
set(
	CUDA_NVCC_FLAGS
	${CUDA_NVCC_FLAGS};
    -O3
    # -G -g
	-gencode arch=compute_52,code=sm_52
	-gencode arch=compute_53,code=sm_53
	-gencode arch=compute_62,code=sm_62
)

# Copy headers to include directory.
MACRO(BUILD_COPY_HEADERS DIRS_LIST)
    foreach(HEADER_DIR ${${DIRS_LIST}})
        message("-- Copying headers from ${HEADER_DIR}")
	    file(GLOB TMP_TFRT_HEADERS ${HEADER_DIR}/*.h ${HEADER_DIR}/*.hpp)
        foreach(HEADER ${TMP_TFRT_HEADERS})
            message("-- Copying ${HEADER}")
            set(HEADERS_DIR_BUILD ${PROJECT_INCLUDE_DIR}/${HEADER_DIR})
            file(MAKE_DIRECTORY ${HEADERS_DIR_BUILD})
	        configure_file(${HEADER} ${HEADERS_DIR_BUILD} COPYONLY)
        endforeach(HEADER)
    endforeach(HEADER_DIR)
ENDMACRO(BUILD_COPY_HEADERS)

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
# Additional weird directories for Gstreamer and Glib
include_directories(/usr/include/gstreamer-1.0 /usr/include/glib-2.0 /usr/include/libxml2)
include_directories(/usr/lib/aarch64-linux-gnu/glib-2.0/include /usr/lib/aarch64-linux-gnu/gstreamer-1.0/include)
include_directories(/usr/lib/x86_64-linux-gnu/glib-2.0/include /usr/lib/x86_64-linux-gnu/gstreamer-1.0/include)

# Third party libraries.
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/eigen)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/half/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/glfw3/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/opengl)

# Build NVXIO dependency
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/nvxio/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/nvxio/src)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/nvxio/src/NVX)
add_subdirectory(nvxio)

# Build TensorFlowRT library.
add_subdirectory(tensorflowrt)
add_subdirectory(util)

# Build several small tools...
add_subdirectory(tools)


# install
# foreach(include ${inferenceIncludes})
#     install(FILES "${include}" DESTINATION include/jetson-inference)
# endforeach()

# # install the shared library
# install(TARGETS jetson-inference DESTINATION lib/jetson-inference EXPORT jetson-inferenceConfig)

# # install the cmake project, for importing
# install(EXPORT jetson-inferenceConfig DESTINATION share/jetson-inference/cmake)

