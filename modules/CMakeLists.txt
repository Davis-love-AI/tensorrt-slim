# Multiple TFRT modules.
add_subdirectory(stabilization)

# Copy modules headers
set(HEADERS_DIRS stabilization)
BUILD_COPY_HEADERS(HEADERS_DIRS)

