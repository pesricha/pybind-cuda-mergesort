cmake_minimum_required(VERSION 3.20)
project(GPU_MergeSort LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CUDA_ARCHITECTURES 75)

include_directories(include)

add_executable(merge_sort
    src/main.cu
    src/merge_sort.cu
)

set_target_properties(merge_sort PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# Optional flags
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -use_fast_math -lineinfo")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native")
