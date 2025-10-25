#include "merge_sort.h"
#include <iostream>
#include <algorithm>
#include <cstdlib>
#include <ctime>
#include <stdio.h>
#include <cstdint> // for int64_t

#define ARRAY_SIZE ((1ull << 16ull) - 46)

// -----------------------------
// Select the data type here
// -----------------------------
using DataType = long;  // change to int, int64_t, float, or double

int main() {
    unsigned long long n = ARRAY_SIZE;
    std::cout << "Array Size: " << n << std::endl;

    DataType *arr = (DataType*)malloc(n * sizeof(DataType));
    DataType *arr_cpu = (DataType*)malloc(n * sizeof(DataType));

    srand(time(NULL));
    for (unsigned long long i = 0; i < n; i++) {
        if constexpr (std::is_integral<DataType>::value) {
            arr[i] = rand() % 100000000; // integer types
        } else {
            arr[i] = (DataType)(rand() % 100000000) / 10.0; // float/double
        }
        arr_cpu[i] = arr[i];
    }

    // Print first 10 elements before sort
    for (unsigned long long i = 0; i < 10; i++)
        printf("%f ", static_cast<double>(arr[i])); // print everything as double
    printf("\n");

    printf("Sorting on GPU using Optimized Merge Sort...\n");

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    mergeSortCUDA<DataType>(arr, n);  // templated call
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Kernel execution time: %f ms\n", milliseconds);

    // CPU sort for verification
    std::sort(arr_cpu, arr_cpu + n);
    int isSorted = 1;
    for (unsigned long long i = 0; i < n; i++) {
        if (arr[i] != arr_cpu[i]) {
            isSorted = 0;
            printf("Mismatch at index %llu: GPU = %f, CPU = %f\n",
                   i, static_cast<double>(arr[i]), static_cast<double>(arr_cpu[i]));
            break;
        }
    }
    printf(isSorted ? "Array is sorted.\n" : "Array is NOT sorted.\n");

    // Print first 10 elements after sort
    for (unsigned long long i = 0; i < 10; i++)
        printf("%f ", static_cast<double>(arr[i]));
    printf("\n");

    free(arr);
    free(arr_cpu);

    return 0;
}
