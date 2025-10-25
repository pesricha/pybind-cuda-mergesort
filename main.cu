#include <algorithm>
#include <cstddef>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <climits> // For INT_MAX
#include <iostream>

#define THREADS_PER_BLOCK 256
#define ARRAY_SIZE ((1ull << 16ull) + 1) 

// Shared memory merge sort for sorting within thread blocks
__global__ void mergeSortBlockKernel(int *d_arr, unsigned long long n) {
    __shared__ int shared_data[THREADS_PER_BLOCK];

    unsigned long long tid = threadIdx.x;
    unsigned long long gid = blockIdx.x * blockDim.x + threadIdx.x;

    if (gid < n) {
        shared_data[tid] = d_arr[gid]; // Load data into shared memory
    } else {
        shared_data[tid] = INT_MAX; // Pad with a large value
    }
    __syncthreads();

    // Iterative merge sort within the shared memory block
    for (unsigned long long size = 2; size <= THREADS_PER_BLOCK; size *= 2) {
        for (unsigned long long stride = size / 2; stride > 0; stride /= 2) {
            unsigned long long swap_idx = tid ^ stride;

            if (swap_idx > tid && swap_idx < THREADS_PER_BLOCK) {
                if ((tid & size) == 0) { // Ascending order
                    if (shared_data[tid] > shared_data[swap_idx]) {
                        int temp = shared_data[tid];
                        shared_data[tid] = shared_data[swap_idx];
                        shared_data[swap_idx] = temp;
                    }
                } else { // Descending order
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

    if (gid < n) {
        d_arr[gid] = shared_data[tid]; // Store sorted data back to global memory
    }
}

// Kernel to merge sorted subarrays
__global__ void mergeSortedSubarrays(int *outputArray, int *inputArray,
                                     unsigned long long subarraySize) {
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
                (Q[1] == sizeA || Q[0] == 0 ||
                    inputArray[subarrayStart + Q[1]] > inputArray[subarrayStart + sizeA + Q[0] - 1])) {
                if (Q[0] == sizeB || Q[1] == 0 ||
                    inputArray[subarrayStart + Q[1] - 1] <= inputArray[subarrayStart + sizeA + Q[0]]) {
                    if (Q[1] < sizeA &&
                        (Q[0] == sizeB ||
                            inputArray[subarrayStart + Q[1]] <= inputArray[subarrayStart + sizeA + Q[0]])) {
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

// Host function: Merge sort with memory optimizations
void mergeSortCUDA(int *arr, unsigned long long n) {
    int *d_arr, *d_temp;
    unsigned long long size = n * sizeof(int);

    // Pad the array to the next power of two
    unsigned long long nextPow2 = 1;
    while (nextPow2 < n) nextPow2 <<= 1;
    int *paddedArray;
    cudaMallocManaged(&paddedArray, nextPow2 * sizeof(int));
    memcpy(paddedArray, arr, n * sizeof(int));
    for (unsigned long long i = n; i < nextPow2; i++) paddedArray[i] = INT_MAX; // Pad with INT_MAX

    cudaMalloc(&d_arr, nextPow2 * sizeof(int));
    cudaMalloc(&d_temp, nextPow2 * sizeof(int));
    cudaMemcpy(d_arr, paddedArray, nextPow2 * sizeof(int), cudaMemcpyHostToDevice);

    // Sort within thread blocks using shared memory
    dim3 blockDim(THREADS_PER_BLOCK);
    dim3 gridDim((nextPow2 + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK);
    mergeSortBlockKernel<<<gridDim, blockDim>>>(d_arr, nextPow2);
    cudaDeviceSynchronize();

    unsigned long long numSubarrays = nextPow2 / THREADS_PER_BLOCK;
    unsigned long long subarraySize = THREADS_PER_BLOCK;
    // Iterative merging with reduced kernel launches
    // for (unsigned long long width = THREADS_PER_BLOCK, numSubArrays = nextPow2 / THREADS_PER_BLOCK; numSubArrays != 1; width *= 2, numSubArrays /= 2) {
    while (numSubarrays != 1) {
        numSubarrays /= 2; // Number of subarrays to merge in the next step
        subarraySize *= 2; // Size of subarrays in the next step
        mergeSortedSubarrays<<<numSubarrays, min(subarraySize, 1024ull)>>>(d_temp, d_arr, subarraySize);
        cudaDeviceSynchronize();

        // Swap pointers
        int *temp = d_arr;
        d_arr = d_temp;
        d_temp = temp;
    }

    // Copy the sorted data back to the host
    cudaMemcpy(paddedArray, d_arr, nextPow2 * sizeof(int), cudaMemcpyDeviceToHost);
    memcpy(arr, paddedArray, n * sizeof(int)); // Copy only the first n elements

    cudaFree(d_arr);
    cudaFree(d_temp);
    cudaFree(paddedArray);
}

int main() {
    unsigned long long n = ARRAY_SIZE; // Non-power-of-two size
    std::cout << "Array Size: " << n << std::endl;
    int *arr = (int*)malloc(n * sizeof(int));
    int *arr_cpu = (int*)malloc(n * sizeof(int)); // Copy for CPU sorting

    srand(time(NULL)); 
    // Generate random input
    for (unsigned long long i = 0; i < n; i++) { 
        arr[i] = rand() % 100000000;
        arr_cpu[i] = arr[i];  // Copy for CPU sorting
    }

    for (unsigned long long i = 0; i < 10; i++)
        printf("%d ", arr[i]);
    printf("\n");

    printf("Sorting on GPU using Optimized Merge Sort...\n");

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    mergeSortCUDA(arr, n);
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Kernel execution time: %f ms\n", milliseconds);

    // // Copy for CPU sorting
    std::sort(arr_cpu, arr_cpu + n);
    int isSorted = 1;
    for (unsigned long long i = 0; i < n; i++) {
        if (arr[i] != arr_cpu[i]) {
            isSorted = 0;
            printf("Mismatch at index %llu: GPU = %d, CPU = %d\n", i, arr[i], arr_cpu[i]);
            break;
        }
    }
    printf(isSorted ? "Array is sorted.\n" : "Array is NOT sorted.\n");
    // Copy for CPU sorting [END]

    for (unsigned long long i = 0; i < 10; i++)
        printf("%d ", arr[i]);
    printf("\n");

    free(arr);
    free(arr_cpu); // Copy for CPU sorting
    return 0;
}