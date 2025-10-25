#!/bin/bash
# Build script for CUDA merge sort Python bindings

echo "Building CUDA merge sort Python bindings..."

# Check if pybind11 is installed
if ! python3 -c "import pybind11" 2>/dev/null; then
    echo "pybind11 not found. Installing..."
    pip3 install pybind11
fi

# Create build directory
mkdir -p build
cd build

# Configure with CMake
echo "Running CMake..."
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_ARCHITECTURES=${GPU_ARCH} \
    -DCMAKE_CUDA_FLAGS="-use_fast_math -lineinfo" \
    -DCMAKE_CXX_FLAGS="-O3 -march=native"

# Build
echo "Building..."
make -j$(nproc)

if [ $? -eq 0 ]; then
    echo ""
    echo "Build successful!"
    echo ""
    echo "To use the Python module, run:"
    echo "  export PYTHONPATH=\$PYTHONPATH:$(pwd)"
    echo "  cd .."
    echo "  python3 test_python_bindings.py"
else
    echo ""
    echo "Build failed!"
    exit 1
fi
