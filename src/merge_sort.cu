#include "merge_sort.h"
#include <cstdint>
#include <cuda_runtime.h>
#include <algorithm>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
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
// Kernel: Merge sorted subarrays
// -----------------------------
template <typename T>
__global__ void mergeSortedSubarrays(T *outputArray, T *inputArray, unsigned long long subarraySize) {
    unsigned long long subarrayStart = blockIdx.x * subarraySize;
    unsigned long long stride = blockDim.x;

    for (unsigned long long i = threadIdx.x; i < subarraySize; i += stride) {
        unsigned long long sizeA = subarraySize / 2;
        unsigned long long sizeB = subarraySize - sizeA;

        long long K[2], P[2], Q[2];
        if (i > sizeA) {
            K[0] = P[1] = i - sizeA;
            K[1] = P[0] = sizeA;
        } else {
            K[0] = P[1] = 0;
            K[1] = P[0] = i;
        }

        long long offset;
        while (true) {
            // offset = abs(K[1] - P[1]) / 2;
            offset = (K[1] - P[1] >= 0 ? K[1] - P[1] : -(K[1] - P[1])) / 2;
            Q[0] = K[0] + offset;
            Q[1] = K[1] - offset;

            if (Q[1] >= 0 && Q[0] <= sizeB &&
                (Q[1] == sizeA || Q[0] == 0 ||
                 inputArray[subarrayStart + Q[1]] > inputArray[subarrayStart + sizeA + Q[0] - 1])) {

                if (Q[0] == sizeB || Q[1] == 0 ||
                    inputArray[subarrayStart + Q[1] - 1] <= inputArray[subarrayStart + sizeA + Q[0]]) {

                    if (Q[1] < sizeA &&
                        (Q[0] == sizeB || inputArray[subarrayStart + Q[1]] <= inputArray[subarrayStart + sizeA + Q[0]])) {
                        outputArray[subarrayStart + i] = inputArray[subarrayStart + Q[1]];
                    } else {
                        outputArray[subarrayStart + i] = inputArray[subarrayStart + sizeA + Q[0]];
                    }
                    break;
                } else {
                    K[0] = Q[0] + 1;
                    K[1] = Q[1] - 1;
                }
            } else {
                P[0] = Q[0] - 1;
                P[1] = Q[1] + 1;
            }
        }
    }
    __syncthreads();
}

// -----------------------------
// Host function: MergeSortCUDA
// -----------------------------
template <typename T>
void mergeSortCUDA(T *arr, unsigned long long n) {
    T *d_arr, *d_temp;
    unsigned long long nextPow2 = 1;
    while (nextPow2 < n) nextPow2 <<= 1;

    T *paddedArray;
    cudaMallocManaged(&paddedArray, nextPow2 * sizeof(T));
    memcpy(paddedArray, arr, n * sizeof(T));

    // Pad with large values
    for (unsigned long long i = n; i < nextPow2; i++) {
        if constexpr (std::is_same<T,int>::value) paddedArray[i] = INT_MAX;
        else if constexpr (std::is_same<T,int64_t>::value) paddedArray[i] = INT64_MAX;
        else if constexpr (std::is_same<T,float>::value) paddedArray[i] = 3.4028235e38f;
        else if constexpr (std::is_same<T,double>::value) paddedArray[i] = 1e300;
    }

    cudaMalloc(&d_arr, nextPow2 * sizeof(T));
    cudaMalloc(&d_temp, nextPow2 * sizeof(T));
    cudaMemcpy(d_arr, paddedArray, nextPow2 * sizeof(T), cudaMemcpyHostToDevice);

    // Sort within blocks
    dim3 blockDim(THREADS_PER_BLOCK);
    dim3 gridDim((nextPow2 + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK);
    mergeSortBlockKernel<<<gridDim, blockDim>>>(d_arr, nextPow2);
    cudaDeviceSynchronize();

    // Iterative merging
    unsigned long long numSubarrays = nextPow2 / THREADS_PER_BLOCK;
    unsigned long long subarraySize = THREADS_PER_BLOCK;

    while (numSubarrays != 1) {
        numSubarrays /= 2;
        subarraySize *= 2;
        mergeSortedSubarrays<<<numSubarrays, min(subarraySize, 1024ull)>>>(d_temp, d_arr, subarraySize);
        cudaDeviceSynchronize();
        std::swap(d_arr, d_temp);
    }

    cudaMemcpy(paddedArray, d_arr, nextPow2 * sizeof(T), cudaMemcpyDeviceToHost);
    memcpy(arr, paddedArray, n * sizeof(T));

    cudaFree(d_arr);
    cudaFree(d_temp);
    cudaFree(paddedArray);
}

// -----------------------------
// Explicit instantiation for supported types
// -----------------------------
template void mergeSortCUDA<int>(int*, unsigned long long);
template void mergeSortCUDA<long>(long*, unsigned long long);
template void mergeSortCUDA<float>(float*, unsigned long long);
template void mergeSortCUDA<double>(double*, unsigned long long);
