# NVX stabilization module.
FILE(GLOB NVX_STAB_MOD_SRCS *.cpp)

# Compile stabilization library.
cuda_add_library(tfrt_stabilization STATIC ${NVX_STAB_MOD_SRCS})
target_link_libraries(tfrt_stabilization nvxio)

# Install stuff!
install(TARGETS tfrt_stabilization DESTINATION lib)
