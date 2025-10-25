#include "thrust_sort.h"
#include <thrust/sort.h>
#include <thrust/device_ptr.h>
#include <cuda_runtime.h>

// Thrust-based sorting function
template <typename T>
void thrustSortCUDA(T *arr, unsigned long long n) {
    // arr is expected to already be a device pointer
    // Wrap raw pointer in thrust device pointer
    thrust::device_ptr<T> dev_ptr(arr);
    
    // Sort using Thrust's highly optimized sort
    thrust::sort(dev_ptr, dev_ptr + n);
    
    // Synchronize to ensure sort is complete
    cudaDeviceSynchronize();
}

// Explicit instantiations
template void thrustSortCUDA<int>(int*, unsigned long long);
template void thrustSortCUDA<int64_t>(int64_t*, unsigned long long);
template void thrustSortCUDA<float>(float*, unsigned long long);
template void thrustSortCUDA<double>(double*, unsigned long long);
