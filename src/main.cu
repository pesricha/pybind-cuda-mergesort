#include "merge_sort.h"
#include <iostream>
#include <algorithm>
#include <cstdlib>
#include <ctime>

#define ARRAY_SIZE (1ull << 20ull)

int main() {
    unsigned long long n = ARRAY_SIZE;
    std::cout << "Array Size: " << n << std::endl;

    int *arr = (int*)malloc(n * sizeof(int));
    int *arr_cpu = (int*)malloc(n * sizeof(int));

    srand(time(NULL));
    for (unsigned long long i = 0; i < n; i++) {
        arr[i] = rand() % 100000000;
        arr_cpu[i] = arr[i];
    }

    std::cout << "Sorting on GPU using Optimized Merge Sort...\n";

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    mergeSortCUDA(arr, n);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "Kernel execution time: " << milliseconds << " ms\n";

    std::sort(arr_cpu, arr_cpu + n);
    bool isSorted = true;
    for (unsigned long long i = 0; i < n; i++) {
        if (arr[i] != arr_cpu[i]) {
            isSorted = false;
            std::cout << "Mismatch at index " << i << ": GPU = " << arr[i] << ", CPU = " << arr_cpu[i] << "\n";
            break;
        }
    }

    std::cout << (isSorted ? "Array is sorted.\n" : "Array is NOT sorted.\n");

    free(arr);
    free(arr_cpu);
    return 0;
}
