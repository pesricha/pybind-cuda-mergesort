#!/bin/bash
set -e  # Exit on any error

# ------------------------------
# Build directory
# ------------------------------
BUILD_DIR=build
mkdir -p $BUILD_DIR
cd $BUILD_DIR

# ------------------------------
# CMake configuration
# ------------------------------
# Replace 75 with your GPU compute capability
GPU_ARCH=75

cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_ARCHITECTURES=${GPU_ARCH} \
    -DCMAKE_CUDA_FLAGS="-use_fast_math -lineinfo" \
    -DCMAKE_CXX_FLAGS="-O3 -march=native"

# ------------------------------
# Build
# ------------------------------
make -j$(nproc)

# ------------------------------
# Done
# ------------------------------
echo "Build complete. Run the executable:"
echo "./merge_sort"
