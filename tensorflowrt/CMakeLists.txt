# Find protobuf and generate files...
INCLUDE(FindProtobuf)
FIND_PACKAGE(Protobuf REQUIRED)

# Protobuf generate files...
message("-- Generating Protobuf source files...")
include_directories(${PROTOBUF_INCLUDE_DIR})
PROTOBUF_GENERATE_CPP(PROTO_SRCS PROTO_HEADERS network.proto ssd_network.proto)
# Ugly direct generation! Don't know how to copy headers otherwise
execute_process(COMMAND protoc --cpp_out=./ network.proto ssd_network.proto
                WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
# Generate python too...
execute_process(COMMAND protoc --python_out=../python/ network.proto ssd_network.proto
                WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})

# TF-RT sources.
FILE(GLOB TFRT_SRCS *.cpp *.cu cuda/*.cpp cuda/*.cu)
FILE(GLOB TFRT_HEADERS *.h)

# Some debugging info...
# MESSAGE("TF-RT sources: " ${TFRT_SRCS})
# MESSAGE("TF-RT headers: " ${TFRT_HEADERS})
# MESSAGE("TF-RT headers: " ${CMAKE_CURRENT_BINARY_DIR})
# MESSAGE("TF-RT headers: " ${CMAKE_CURRENT_SOURCE_DIR})
# MESSAGE("TF-RT protobuf sources: " ${PROTO_SRCS})

# Build TensorFlowRT
include_directories(${CMAKE_CURRENT_BINARY_DIR})
cuda_add_library(tensorflowrt STATIC ${TFRT_SRCS} ${PROTO_SRCS})
target_link_libraries(tensorflowrt nvinfer ${PROTOBUF_LIBRARY} glog gflags)

# Copy TF-RT headers
set(HEADERS_DIRS . nets cuda misc models boxes2d boxes3d)
BUILD_COPY_HEADERS(HEADERS_DIRS)

# foreach(HEADER ${PROTO_HEADERS})
# 	message("-- Copying ${HEADER}")
# 	configure_file(${HEADER} ${PROJECT_INCLUDE_DIR} COPYONLY)
# endforeach(HEADER)
