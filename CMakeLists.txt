cmake_minimum_required(VERSION 2.8)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

find_package(CUDA REQUIRED)

# Pass options to NVCC
#set(
#        CUDA_NVCC_FLAGS
#        ${CUDA_NVCC_FLAGS};
#        -gencode arch=compute_22,code=sm_22
#)

# Specify include directories
include_directories(
        kernels
        utility
)

# For compilation ...
# Specify target & source files to compile it from
cuda_add_executable(
        main
        main.cu
)

# For linking ...
# Specify target & libraries to link it with
#target_link_libraries(
#        hellocuda
#        -lfoobar
#        -ljoestuff
#)