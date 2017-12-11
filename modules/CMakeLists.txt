# Multiple TFRT modules.
add_subdirectory(stabilization)

# Copy modules headers
install(DIRECTORY ./
    DESTINATION include
    FILES_MATCHING PATTERN "*.hpp")
# set(HEADERS_DIRS stabilization)
# BUILD_COPY_HEADERS(HEADERS_DIRS)

