#include "merge_sort.h"
#include <algorithm>
#include <climits>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define THREADS_PER_BLOCK 256

// Shared memory merge sort for sorting within thread blocks
__global__ void mergeSortBlockKernel(int *d_arr, unsigned long long n) {
    __shared__ int shared_data[THREADS_PER_BLOCK];
    unsigned long long tid = threadIdx.x;
    unsigned long long gid = blockIdx.x * blockDim.x + threadIdx.x;

    if (gid < n) shared_data[tid] = d_arr[gid];
    else shared_data[tid] = INT_MAX;
    __syncthreads();

    for (unsigned long long size = 2; size <= THREADS_PER_BLOCK; size *= 2) {
        for (unsigned long long stride = size / 2; stride > 0; stride /= 2) {
            unsigned long long swap_idx = tid ^ stride;
            if (swap_idx > tid && swap_idx < THREADS_PER_BLOCK) {
                if ((tid & size) == 0) {
                    if (shared_data[tid] > shared_data[swap_idx]) {
                        int temp = shared_data[tid];
                        shared_data[tid] = shared_data[swap_idx];
                        shared_data[swap_idx] = temp;
                    }
                } else {
                    if (shared_data[tid] < shared_data[swap_idx]) {
                        int temp = shared_data[tid];
                        shared_data[tid] = shared_data[swap_idx];
                        shared_data[swap_idx] = temp;
                    }
                }
            }
            __syncthreads();
        }
    }

    if (gid < n) d_arr[gid] = shared_data[tid];
}

// Kernel to merge sorted subarrays
__global__ void mergeSortedSubarrays(int *outputArray, int *inputArray, unsigned long long subarraySize) {
    unsigned long long subarrayStart = blockIdx.x * subarraySize;
    unsigned long long stride = blockDim.x;

    for (unsigned long long i = threadIdx.x; i < subarraySize; i += stride) {
        unsigned long long sizeA = subarraySize / 2;
        unsigned long long sizeB = subarraySize / 2;
        unsigned long long offset;

        long long K[2], P[2], Q[2];
        if (i > sizeA) {
            K[0] = P[1] = i - sizeA;
            K[1] = P[0] = sizeA;
        } else {
            K[0] = P[1] = 0;
            K[1] = P[0] = i;
        }

        while (true) {
            offset = abs(K[1] - P[1]) / 2;
            Q[0] = K[0] + offset;
            Q[1] = K[1] - offset;

            if (Q[1] >= 0 && Q[0] <= sizeB &&
                (Q[1] == sizeA || Q[0] == 0 || inputArray[subarrayStart + Q[1]] > inputArray[subarrayStart + sizeA + Q[0] - 1])) {
                if (Q[0] == sizeB || Q[1] == 0 || inputArray[subarrayStart + Q[1] - 1] <= inputArray[subarrayStart + sizeA + Q[0]]) {
                    if (Q[1] < sizeA && (Q[0] == sizeB || inputArray[subarrayStart + Q[1]] <= inputArray[subarrayStart + sizeA + Q[0]])) {
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

// Host function
void mergeSortCUDA(int *arr, unsigned long long n) {
    int *d_arr, *d_temp;
    unsigned long long nextPow2 = 1;
    while (nextPow2 < n) nextPow2 <<= 1;

    int *paddedArray;
    cudaMallocManaged(&paddedArray, nextPow2 * sizeof(int));
    memcpy(paddedArray, arr, n * sizeof(int));
    for (unsigned long long i = n; i < nextPow2; i++) paddedArray[i] = INT_MAX;

    cudaMalloc(&d_arr, nextPow2 * sizeof(int));
    cudaMalloc(&d_temp, nextPow2 * sizeof(int));
    cudaMemcpy(d_arr, paddedArray, nextPow2 * sizeof(int), cudaMemcpyHostToDevice);

    dim3 blockDim(THREADS_PER_BLOCK);
    dim3 gridDim((nextPow2 + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK);
    mergeSortBlockKernel<<<gridDim, blockDim>>>(d_arr, nextPow2);
    cudaDeviceSynchronize();

    unsigned long long numSubarrays = nextPow2 / THREADS_PER_BLOCK;
    unsigned long long subarraySize = THREADS_PER_BLOCK;

    while (numSubarrays != 1) {
        numSubarrays /= 2;
        subarraySize *= 2;
        mergeSortedSubarrays<<<numSubarrays, min(subarraySize, 1024ull)>>>(d_temp, d_arr, subarraySize);
        cudaDeviceSynchronize();
        std::swap(d_arr, d_temp);
    }

    cudaMemcpy(paddedArray, d_arr, nextPow2 * sizeof(int), cudaMemcpyDeviceToHost);
    memcpy(arr, paddedArray, n * sizeof(int));

    cudaFree(d_arr);
    cudaFree(d_temp);
    cudaFree(paddedArray);
}
