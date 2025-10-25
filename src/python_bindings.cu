#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "merge_sort.h"
#include "thrust_sort.h"
#include <stdexcept>
#include <cuda_runtime.h>

namespace py = pybind11;

// Wrapper functions for numpy arrays - Merge Sort
template <typename T>
void mergeSortCUDA_numpy(py::array_t<T> arr) {
    py::buffer_info buf = arr.request();
    
    if (buf.ndim != 1) {
        throw std::runtime_error("Input array must be 1-dimensional");
    }
    
    T* host_ptr = static_cast<T*>(buf.ptr);
    unsigned long long n = buf.shape[0];
    
    // Allocate device memory
    T* d_arr;
    cudaMalloc(&d_arr, n * sizeof(T));
    
    // Copy data to device
    cudaMemcpy(d_arr, host_ptr, n * sizeof(T), cudaMemcpyHostToDevice);
    
    // Sort on GPU
    mergeSortCUDA<T>(d_arr, n);
    
    // Copy sorted data back to host
    cudaMemcpy(host_ptr, d_arr, n * sizeof(T), cudaMemcpyDeviceToHost);
    
    // Free device memory
    cudaFree(d_arr);
}

// Wrapper functions for numpy arrays - Thrust Sort
template <typename T>
void thrustSortCUDA_numpy(py::array_t<T> arr) {
    py::buffer_info buf = arr.request();
    
    if (buf.ndim != 1) {
        throw std::runtime_error("Input array must be 1-dimensional");
    }
    
    T* host_ptr = static_cast<T*>(buf.ptr);
    unsigned long long n = buf.shape[0];
    
    // Allocate device memory
    T* d_arr;
    cudaMalloc(&d_arr, n * sizeof(T));
    
    // Copy data to device
    cudaMemcpy(d_arr, host_ptr, n * sizeof(T), cudaMemcpyHostToDevice);
    
    // Sort on GPU using Thrust
    thrustSortCUDA<T>(d_arr, n);
    
    // Copy sorted data back to host
    cudaMemcpy(host_ptr, d_arr, n * sizeof(T), cudaMemcpyDeviceToHost);
    
    // Free device memory
    cudaFree(d_arr);
}

PYBIND11_MODULE(merge_sort_cuda, m) {
    m.doc() = "CUDA-accelerated sorting using Pybind11";
    
    // Merge Sort bindings
    m.def("merge_sort_int32", &mergeSortCUDA_numpy<int32_t>, 
          py::arg("arr"),
          "Sort a numpy int32 array in-place using CUDA merge sort");
    
    m.def("merge_sort_int64", &mergeSortCUDA_numpy<int64_t>, 
          py::arg("arr"),
          "Sort a numpy int64 array in-place using CUDA merge sort");
    
    m.def("merge_sort_float32", &mergeSortCUDA_numpy<float>, 
          py::arg("arr"),
          "Sort a numpy float32 array in-place using CUDA merge sort");
    
    m.def("merge_sort_float64", &mergeSortCUDA_numpy<double>, 
          py::arg("arr"),
          "Sort a numpy float64 array in-place using CUDA merge sort");
    
    // Thrust Sort bindings
    m.def("thrust_sort_int32", &thrustSortCUDA_numpy<int32_t>, 
          py::arg("arr"),
          "Sort a numpy int32 array in-place using Thrust sort");
    
    m.def("thrust_sort_int64", &thrustSortCUDA_numpy<int64_t>, 
          py::arg("arr"),
          "Sort a numpy int64 array in-place using Thrust sort");
    
    m.def("thrust_sort_float32", &thrustSortCUDA_numpy<float>, 
          py::arg("arr"),
          "Sort a numpy float32 array in-place using Thrust sort");
    
    m.def("thrust_sort_float64", &thrustSortCUDA_numpy<double>, 
          py::arg("arr"),
          "Sort a numpy float64 array in-place using Thrust sort");
}
