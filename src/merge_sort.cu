#include "merge_sort.h"
#include <cstdint>
#include <cuda_runtime.h>
#include <algorithm>
#include <vector>
#include <climits>

#define THREADS_PER_BLOCK 256

// -----------------------------
// Kernel: Merge sort within a thread block using shared memory
// -----------------------------
template <typename T>
__global__ void mergeSortBlockKernel(T *d_arr, unsigned long long n) {
    __shared__ T shared_data[THREADS_PER_BLOCK];

    unsigned long long tid = threadIdx.x;
    unsigned long long gid = blockIdx.x * blockDim.x + threadIdx.x;

    if (gid < n) {
        shared_data[tid] = d_arr[gid];
    } else {
        if constexpr (std::is_same<T,int>::value) {
            shared_data[tid] = INT_MAX;
        } else if constexpr (std::is_same<T,int64_t>::value) {
            shared_data[tid] = INT64_MAX;
        } else if constexpr (std::is_same<T,float>::value) {
            shared_data[tid] = 3.4028235e38f;
        } else if constexpr (std::is_same<T,double>::value) {
            shared_data[tid] = 1e300;
        }
    }
    __syncthreads();

    // Iterative merge sort within shared memory
    for (unsigned long long size = 2; size <= THREADS_PER_BLOCK; size *= 2) {
        for (unsigned long long stride = size / 2; stride > 0; stride /= 2) {
            unsigned long long swap_idx = tid ^ stride;

            if (swap_idx > tid && swap_idx < THREADS_PER_BLOCK) {
                if ((tid & size) == 0) { // Ascending order
                    if (shared_data[tid] > shared_data[swap_idx]) {
                        T temp = shared_data[tid];
                        shared_data[tid] = shared_data[swap_idx];
                        shared_data[swap_idx] = temp;
                    }
                } else { // Descending order
                    if (shared_data[tid] < shared_data[swap_idx]) {
                        T temp = shared_data[tid];
                        shared_data[tid] = shared_data[swap_idx];
                        shared_data[swap_idx] = temp;
                    }
                }
            }
            __syncthreads();
        }
    }

    if (gid < n) {
        d_arr[gid] = shared_data[tid];
    }
}

// -----------------------------
// Kernel: Fast parallel merge using merge path with shared memory
// -----------------------------
template <typename T>
__device__ unsigned long long mergePath(const T* A, unsigned long long aCount,
                                        const T* B, unsigned long long bCount,
                                        unsigned long long diag) {
    unsigned long long begin = diag > bCount ? diag - bCount : 0;
    unsigned long long end = diag < aCount ? diag : aCount;
    
    while (begin < end) {
        unsigned long long mid = (begin + end) >> 1;
        T aVal = A[mid];
        T bVal = B[diag - 1 - mid];
        
        if (aVal <= bVal) {
            begin = mid + 1;
        } else {
            end = mid;
        }
    }
    return begin;
}

template <typename T>
__global__ void mergeSortedSubarraysFast(T *output, const T *input, 
                                         unsigned long long subarraySize,
                                         unsigned long long totalSize) {
    unsigned long long tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long subarrayIdx = tid / subarraySize;
    unsigned long long subarrayStart = subarrayIdx * subarraySize;
    
    if (subarrayStart >= totalSize) return;
    
    unsigned long long halfSize = subarraySize / 2;
    const T* A = input + subarrayStart;
    const T* B = A + halfSize;
    
    unsigned long long aCount = min(halfSize, totalSize - subarrayStart);
    unsigned long long bCount = min(halfSize, totalSize - (subarrayStart + halfSize));
    
    if (bCount == 0) {
        // No second half, just copy
        unsigned long long localIdx = tid % subarraySize;
        if (localIdx < aCount) {
            output[subarrayStart + localIdx] = A[localIdx];
        }
        return;
    }
    
    // Each thread handles one output element
    unsigned long long localIdx = tid % subarraySize;
    if (localIdx >= aCount + bCount) return;
    
    // Find merge path coordinates for this thread
    unsigned long long aIdx = mergePath(A, aCount, B, bCount, localIdx);
    unsigned long long bIdx = localIdx - aIdx;
    
    // Determine which value to write
    T value;
    if (aIdx >= aCount) {
        value = B[bIdx];
    } else if (bIdx >= bCount) {
        value = A[aIdx];
    } else {
        value = (A[aIdx] <= B[bIdx]) ? A[aIdx] : B[bIdx];
    }
    
    output[subarrayStart + localIdx] = value;
}

// -----------------------------
// Host function: MergeSortCUDA
// -----------------------------
template <typename T>
void mergeSortCUDA(T *arr, unsigned long long n) {
    // arr is expected to already be a device pointer
    unsigned long long nextPow2 = 1;
    while (nextPow2 < n) nextPow2 <<= 1;

    T *d_arr, *d_temp;
    
    // Allocate device memory for the padded array
    cudaMalloc(&d_arr, nextPow2 * sizeof(T));
    cudaMalloc(&d_temp, nextPow2 * sizeof(T));
    
    // Copy input data to d_arr
    cudaMemcpy(d_arr, arr, n * sizeof(T), cudaMemcpyDeviceToDevice);
    
    // Fill padding with sentinel values (only if needed)
    if (nextPow2 > n) {
        T sentinelValue;
        if constexpr (std::is_same<T,int>::value) sentinelValue = INT_MAX;
        else if constexpr (std::is_same<T,int64_t>::value) sentinelValue = INT64_MAX;
        else if constexpr (std::is_same<T,float>::value) sentinelValue = 3.4028235e38f;
        else if constexpr (std::is_same<T,double>::value) sentinelValue = 1e300;
        
        unsigned long long paddingSize = nextPow2 - n;
        std::vector<T> padding(paddingSize, sentinelValue);
        cudaMemcpy(d_arr + n, padding.data(), paddingSize * sizeof(T), cudaMemcpyHostToDevice);
    }

    // Sort within blocks using bitonic sort
    dim3 blockDim(THREADS_PER_BLOCK);
    dim3 gridDim((nextPow2 + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK);
    mergeSortBlockKernel<<<gridDim, blockDim>>>(d_arr, nextPow2);

    // Iterative merging with optimized parallel merge
    unsigned long long subarraySize = THREADS_PER_BLOCK;

    while (subarraySize < nextPow2) {
        subarraySize *= 2;
        
        // Launch enough threads - one per output element
        unsigned long long numBlocks = (nextPow2 + 255) / 256;
        
        mergeSortedSubarraysFast<<<numBlocks, 256>>>(d_temp, d_arr, subarraySize, nextPow2);
        std::swap(d_arr, d_temp);
    }

    // Copy result back to input array
    cudaMemcpy(arr, d_arr, n * sizeof(T), cudaMemcpyDeviceToDevice);
    
    // Only synchronize once at the very end
    cudaDeviceSynchronize();

    cudaFree(d_arr);
    cudaFree(d_temp);
}

// -----------------------------
// Explicit instantiation for supported types
// -----------------------------
template void mergeSortCUDA<int>(int*, unsigned long long);
template void mergeSortCUDA<long>(long*, unsigned long long);
template void mergeSortCUDA<float>(float*, unsigned long long);
template void mergeSortCUDA<double>(double*, unsigned long long);
