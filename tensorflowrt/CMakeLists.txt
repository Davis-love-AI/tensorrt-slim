# Find protobuf and generate files...
INCLUDE(FindProtobuf)
FIND_PACKAGE(Protobuf REQUIRED)

INCLUDE_DIRECTORIES(${PROTOBUF_INCLUDE_DIR})
PROTOBUF_GENERATE_CPP(PROTO_SRCS PROTO_HEADERS tensorflowrt.proto)
# Ugly direct generation!
# execute_process(COMMAND /usr/bin/protoc --cpp_out=./ tensorflowrt.proto
#                 WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})

file(GLOB TFRT_SOURCES *.cpp *.cu)
file(GLOB TFRT_INCLUDES *.h)
# MESSAGE("TF-RT sources: " ${TFRT_SOURCES})
MESSAGE("TF-RT protobuf sources: " ${PROTO_SRCS})

# Build TensorFlowRT
cuda_add_library(tensorflowrt STATIC ${TFRT_SOURCES} ${PROTO_SRCS})
target_link_libraries(tensorflowrt nvinfer ${PROTOBUF_LIBRARY})
