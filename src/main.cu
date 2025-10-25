#include "merge_sort.h"
#include <iostream>
#include <algorithm>
#include <cstdlib>
#include <ctime>
#include <cstdint>
#include <type_traits>

#define ARRAY_SIZE ((1ull << 16ull) - 46)
#define PRINT_COUNT 10

// Helper to print first N elements
template <typename T>
void printArray(const T* arr, unsigned long long n, unsigned long long maxPrint = PRINT_COUNT) {
    for (unsigned long long i = 0; i < n && i < maxPrint; i++) {
        if constexpr (std::is_floating_point<T>::value)
            printf("%f ", static_cast<double>(arr[i]));
        else
            printf("%lld ", static_cast<long long>(arr[i]));
    }
    if (n > maxPrint) printf("...");
    printf("\n");
}

// Generic test function
template <typename T>
void testMergeSortType(const char* typeName) {
    std::cout << "==============================\n";
    std::cout << "Testing type: " << typeName << std::endl;
    unsigned long long n = ARRAY_SIZE;

    T* arr = (T*)malloc(n * sizeof(T));
    T* arr_cpu = (T*)malloc(n * sizeof(T));

    srand(time(NULL));
    for (unsigned long long i = 0; i < n; i++) {
        if constexpr (std::is_integral<T>::value) {
            arr[i] = rand() % 100000000;
        } else {
            arr[i] = static_cast<T>(rand() % 100000000) / 10.0;
        }
        arr_cpu[i] = arr[i];
    }

    std::cout << "First elements before sort: ";
    printArray(arr, n);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    mergeSortCUDA<T>(arr, n);
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("GPU kernel execution time: %f ms\n", milliseconds);

    std::sort(arr_cpu, arr_cpu + n);

    bool isSorted = true;
    for (unsigned long long i = 0; i < n; i++) {
        if (arr[i] != arr_cpu[i]) {
            isSorted = false;
            printf("Mismatch at index %llu: GPU = ", i);
            if constexpr (std::is_floating_point<T>::value)
                printf("%f, CPU = %f\n", static_cast<double>(arr[i]), static_cast<double>(arr_cpu[i]));
            else
                printf("%lld, CPU = %lld\n", static_cast<long long>(arr[i]), static_cast<long long>(arr_cpu[i]));
            break;
        }
    }

    std::cout << (isSorted ? "Array is sorted.\n" : "Array is NOT sorted.\n");
    std::cout << "First elements after sort: ";
    printArray(arr, n);

    free(arr);
    free(arr_cpu);
}

int main() {
    testMergeSortType<int>("int32");
    testMergeSortType<int64_t>("int64");
    testMergeSortType<float>("float32");
    testMergeSortType<double>("float64");
    return 0;
}
