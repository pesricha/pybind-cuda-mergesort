#pragma once
#include <cuda_runtime.h>

template <typename T>
void thrustSortCUDA(T *arr, unsigned long long n);
