#!/usr/bin/env python3
"""
Test script for CUDA merge sort Python bindings with multiple runs
"""
import numpy as np
import sys
import time

# Import the CUDA merge sort module
try:
    import merge_sort_cuda
except ImportError as e:
    print(f"Error importing merge_sort_cuda: {e}")
    print("Make sure to build the module first:")
    print("  mkdir -p build && cd build")
    print("  cmake ..")
    print("  make")
    print("  export PYTHONPATH=$PYTHONPATH:$(pwd)")
    sys.exit(1)

NUM_RUNS = 10

def test_sort_comparison(data_generator, merge_func, thrust_func, dtype_name, array_size):
    """Compare merge sort, thrust sort, and NumPy with multiple runs"""
    print(f"\n{'='*70}")
    print(f"Testing {dtype_name} ({NUM_RUNS} runs + 1 warmup)")
    print(f"{'='*70}")
    print(f"Array size: {array_size:,}")
    
    # Warmup run to initialize GPU
    print("Running warmup...")
    warmup_data = data_generator()
    merge_func(warmup_data.copy())
    thrust_func(warmup_data.copy())
    
    numpy_times = []
    merge_times = []
    thrust_times = []
    
    all_correct = True
    
    for run in range(NUM_RUNS):
        # Generate fresh data for each run
        arr_original = data_generator()
        arr_merge = arr_original.copy()
        arr_thrust = arr_original.copy()
        arr_numpy = arr_original.copy()
        
        # Time NumPy sort
        start = time.time()
        arr_numpy.sort()
        numpy_times.append((time.time() - start) * 1000)
        
        # Time CUDA merge sort
        start = time.time()
        merge_func(arr_merge)
        merge_times.append((time.time() - start) * 1000)
        
        # Time Thrust sort
        start = time.time()
        thrust_func(arr_thrust)
        thrust_times.append((time.time() - start) * 1000)
        
        # Validate (only on first run to save time)
        if run == 0:
            merge_correct = np.allclose(arr_merge, arr_numpy)
            thrust_correct = np.allclose(arr_thrust, arr_numpy)
            all_correct = merge_correct and thrust_correct
            
            if not merge_correct:
                mismatch = np.where(arr_merge != arr_numpy)[0]
                if len(mismatch) > 0:
                    idx = mismatch[0]
                    print(f"⚠️  Merge sort FAILED at index {idx}: GPU={arr_merge[idx]}, Expected={arr_numpy[idx]}")
            
            if not thrust_correct:
                mismatch = np.where(arr_thrust != arr_numpy)[0]
                if len(mismatch) > 0:
                    idx = mismatch[0]
                    print(f"⚠️  Thrust sort FAILED at index {idx}: GPU={arr_thrust[idx]}, Expected={arr_numpy[idx]}")
    
    # Calculate statistics
    numpy_avg = np.mean(numpy_times)
    numpy_min = np.min(numpy_times)
    numpy_max = np.max(numpy_times)
    
    merge_avg = np.mean(merge_times)
    merge_min = np.min(merge_times)
    merge_max = np.max(merge_times)
    merge_speedup = numpy_avg / merge_avg
    
    thrust_avg = np.mean(thrust_times)
    thrust_min = np.min(thrust_times)
    thrust_max = np.max(thrust_times)
    thrust_speedup = numpy_avg / thrust_avg
    
    # Display results
    print(f"\nResults (average over {NUM_RUNS} runs):")
    print(f"  NumPy:       {numpy_avg:8.2f} ms  (min: {numpy_min:7.2f}, max: {numpy_max:7.2f})")
    print(f"  Merge sort:  {merge_avg:8.2f} ms  (min: {merge_min:7.2f}, max: {merge_max:7.2f})  [{merge_speedup:5.2f}x]")
    print(f"  Thrust:      {thrust_avg:8.2f} ms  (min: {thrust_min:7.2f}, max: {thrust_max:7.2f})  [{thrust_speedup:5.2f}x]")
    
    if all_correct:
        print(f"  Validation: ✓ All correct")
    
    return (merge_avg, merge_min, merge_max), (thrust_avg, thrust_min, thrust_max), (numpy_avg, numpy_min, numpy_max)

def main():
    # Test parameters
    array_size = 100_000_000
    
    print("="*70)
    print("CUDA Sort Comparison: Merge Sort vs Thrust vs NumPy")
    print("="*70)
    print(f"Array size: {array_size:,} elements")
    print(f"Running {NUM_RUNS} iterations per test\n")
    
    results = []
    
    # Test int32
    print("\n" + "▶" * 35)
    merge_stats, thrust_stats, numpy_stats = test_sort_comparison(
        lambda: np.random.randint(0, 100000000, array_size, dtype=np.int32),
        merge_sort_cuda.merge_sort_int32,
        merge_sort_cuda.thrust_sort_int32,
        "int32",
        array_size
    )
    results.append(("int32", merge_stats, thrust_stats, numpy_stats))
    
    # Test int64
    print("\n" + "▶" * 35)
    merge_stats, thrust_stats, numpy_stats = test_sort_comparison(
        lambda: np.random.randint(0, 100000000, array_size, dtype=np.int64),
        merge_sort_cuda.merge_sort_int64,
        merge_sort_cuda.thrust_sort_int64,
        "int64",
        array_size
    )
    results.append(("int64", merge_stats, thrust_stats, numpy_stats))
    
    # Test float32
    print("\n" + "▶" * 35)
    merge_stats, thrust_stats, numpy_stats = test_sort_comparison(
        lambda: np.random.uniform(0, 10000000, array_size).astype(np.float32),
        merge_sort_cuda.merge_sort_float32,
        merge_sort_cuda.thrust_sort_float32,
        "float32",
        array_size
    )
    results.append(("float32", merge_stats, thrust_stats, numpy_stats))
    
    # Test float64
    print("\n" + "▶" * 35)
    merge_stats, thrust_stats, numpy_stats = test_sort_comparison(
        lambda: np.random.uniform(0, 10000000, array_size).astype(np.float64),
        merge_sort_cuda.merge_sort_float64,
        merge_sort_cuda.thrust_sort_float64,
        "float64",
        array_size
    )
    results.append(("float64", merge_stats, thrust_stats, numpy_stats))
    
    # Print summary table
    print(f"\n{'='*100}")
    print(f"PERFORMANCE SUMMARY (Average of {NUM_RUNS} runs)")
    print(f"{'='*100}")
    print(f"{'Type':<10} {'NumPy':>12} {'Merge':>12} {'Thrust':>12} {'Merge':>12} {'Thrust':>12}")
    print(f"{'':10} {'(ms)':>12} {'(ms)':>12} {'(ms)':>12} {'Speedup':>12} {'Speedup':>12}")
    print(f"{'-'*100}")
    
    for dtype, merge_stats, thrust_stats, numpy_stats in results:
        merge_avg, merge_min, merge_max = merge_stats
        thrust_avg, thrust_min, thrust_max = thrust_stats
        numpy_avg, numpy_min, numpy_max = numpy_stats
        
        merge_speedup = numpy_avg / merge_avg if merge_avg > 0 else 0
        thrust_speedup = numpy_avg / thrust_avg if thrust_avg > 0 else 0
        
        print(f"{dtype:<10} {numpy_avg:12.2f} {merge_avg:12.2f} {thrust_avg:12.2f} "
              f"{merge_speedup:11.2f}x {thrust_speedup:11.2f}x")
    
    print(f"{'='*100}")
    print("\n✓ All tests completed!")
    print(f"{'='*100}")

if __name__ == "__main__":
    main()

