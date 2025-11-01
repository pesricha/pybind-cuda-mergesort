# PYBIND11 + Cuda Custom Merge Sort 

GPU-accelerated sorting with Python bindings. Compares custom CUDA merge sort vs Thrust vs NumPy.

## Build

```bash
./build_python.sh
export PYTHONPATH=$PYTHONPATH:./build
```

## Usage

```python
import numpy as np
import merge_sort_cuda

arr = np.random.randint(0, 1000000, 50_000_000, dtype=np.int32)

# Custom merge sort (6-10x faster than NumPy)
merge_sort_cuda.merge_sort_int32(arr)

# Nvidia Thrust sort (6-30x faster than NumPy)
merge_sort_cuda.thrust_sort_int32(arr)
```

## Available Sorting Functions

```python
merge_sort_int32(data: np.ndarray) -> None
merge_sort_int64(data: np.ndarray) -> None
merge_sort_float32(data: np.ndarray) -> None
merge_sort_float64(data: np.ndarray) -> None

thrust_sort_int32(data: np.ndarray) -> None
thrust_sort_int64(data: np.ndarray) -> None
thrust_sort_float32(data: np.ndarray) -> None
thrust_sort_float64(data: np.ndarray) -> None
```

# Benchmark Results RTX 4060

## Hardware Specs
```bash
$ neofetch
            .-/+oossssoo+/-.               nemon@naman 
        `:+ssssssssssssssssss+:`           ----------- 
      -+ssssssssssssssssssyyssss+-         OS: Ubuntu 24.04.3 LTS x86_64 
    .ossssssssssssssssssdMMMNysssso.       Host: 82Y9 Legion Slim 5 16APH8 
   /ssssssssssshdmmNNmmyNMMMMhssssss/      Kernel: 6.14.0-33-generic 
  +ssssssssshmydMMMMMMMNddddyssssssss+     Uptime: 3 hours, 38 mins 
 /sssssssshNMMMyhhyyyyhmNMMMNhssssssss/    Packages: 2681 (dpkg), 17 (snap) 
.ssssssssdMMMNhsssssssssshNMMMdssssssss.   Shell: bash 5.2.21 
+sssshhhyNMMNyssssssssssssyNMMMysssssss+   Resolution: 2560x1600 
ossyNMMMNyMMhsssssssssssssshmmmhssssssso   DE: GNOME 46.0 
ossyNMMMNyMMhsssssssssssssshmmmhssssssso   WM: Mutter 
+sssshhhyNMMNyssssssssssssyNMMMysssssss+   WM Theme: Adwaita 
.ssssssssdMMMNhsssssssssshNMMMdssssssss.   Theme: Yaru-magenta-dark [GTK2/3] 
 /sssssssshNMMMyhhyyyyhdNMMMNhssssssss/    Icons: Yaru-magenta [GTK2/3] 
  +sssssssssdmydMMMMMMMMddddyssssssss+     Terminal: gnome-terminal 
   /ssssssssssshdmNNNNmyNMMMMhssssss/      CPU: AMD Ryzen 7 7840HS w/ Radeon 78 
    .ossssssssssssssssssdMMMNysssso.       GPU: NVIDIA GeForce RTX 4060 Max-Q / 
      -+sssssssssssssssssyyyssss+-         GPU: AMD ATI 05:00.0 Phoenix1 
        `:+ssssssssssssssssss+:`           Memory: 7136MiB / 15167MiB 
            .-/+oossssoo+/-.

```

## Results (0.1M elements)

| Type    | NumPy (ms) | Merge Sort (ms) | Thrust (ms) | Merge Speedup | Thrust Speedup |
|---------|------------|-----------------|-------------|---------------|----------------|
| int32   | 1.64       | 0.41            | 0.27        | **4.06x**     | **6.05x**      |
| int64   | 3.05       | 0.67            | 0.45        | **4.52x**     | **6.80x**      |
| float32 | 1.80       | 0.57            | 0.29        | **3.14x**     | **6.23x**      |
| float64 | 3.10       | 0.85            | 0.49        | **3.65x**     | **6.33x**      |


## Results (1M elements)

| Type    | NumPy (ms) | Merge Sort (ms) | Thrust (ms) | Merge Speedup | Thrust Speedup |
|---------|------------|-----------------|-------------|---------------|----------------|
| int32   | 22.17      | 2.73            | 1.10        | **8.13x**     | **20.24x**     |
| int64   | 38.93      | 3.77            | 2.03        | **10.33x**    | **19.19x**     |
| float32 | 23.18      | 2.80            | 1.12        | **8.27x**     | **20.76x**     |
| float64 | 38.98      | 4.92            | 2.12        | **7.92x**     | **18.38x**     |


## Results (10M elements)

| Type    | NumPy (ms) | Merge Sort (ms) | Thrust (ms) | Merge Speedup | Thrust Speedup |
|---------|------------|-----------------|-------------|---------------|----------------|
| int32   | 289.92     | 46.88           | 9.17        | **6.18x**     | **31.61x**     |
| int64   | 499.83     | 79.59           | 22.00       | **6.28x**     | **22.72x**     |
| float32 | 298.62     | 45.98           | 9.12        | **6.49x**     | **32.74x**     |
| float64 | 504.76     | 95.43           | 22.46       | **5.29x**     | **22.47x**     |

## Results (100M elements)

| Type    | NumPy (ms) | Merge Sort (ms) | Thrust (ms) | Merge Speedup | Thrust Speedup |
|---------|------------|-----------------|-------------|---------------|----------------|
| int32   | 3603.28    | 517.20          | 99.89       | **6.97x**     | **36.07x**     |
| int64   | 5993.88    | 658.48          | 201.75      | **9.10x**     | **29.71x**     |
| float32 | 3736.80    | 501.75          | 91.64       | **7.45x**     | **40.78x**     |
| float64 | 6015.87    | 847.79          | 215.99      | **7.10x**     | **27.85x**     |



*Averaged over 10 runs (+ 1 Warm up). NumPy uses Timsort, Thrust uses radix sort, custom implementation uses bitonic sort + merge path.*

## Build & Test

```bash
# Python Bindings Test
./build_python.sh
export PYTHONPATH=$PYTHONPATH:./build
python3 test_python_bindings.py
```
