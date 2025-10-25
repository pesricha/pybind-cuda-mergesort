# Makefile for CUDA MergeSort

# Compiler and flags
NVCC    := nvcc
CFLAGS  := -O3 --use_fast_math -Xptxas -dlcm=ca -arch=sm_75  # Adjust compute capability (sm_52) as needed

# Target and source files
TARGET  := main.out

SRCS    := main.cu

# Default target
all: $(TARGET)

# Build target using NVCC
$(TARGET): $(SRCS)
	$(NVCC) $(CFLAGS) -o $@ $^

# Build and run
run: $(TARGET)
	./$(TARGET)

profile: $(TARGET)
	nsys profile ./$(TARGET)

# Clean build artifacts
clean:
	rm -f $(TARGET)