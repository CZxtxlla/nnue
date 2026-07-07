# Compilers
CC = gcc
NVCC = nvcc

# Flags
CFLAGS = -Wall -O3
NVCCFLAGS = -O3 -use_fast_math -lcublas

# Define target executable
TARGET = train_nnue

# Core Library Files (Everything EXCEPT files containing a main() function)
CORE_C_SOURCES = src/autograd.c src/ops.c src/nn.c
CORE_CU_SOURCES = src/ops_gpu.cu src/autograd_gpu.cu src/optim_gpu.cu src/tensor.cu src/optim.cu

# Object files for the core library
CORE_C_OBJECTS = $(CORE_C_SOURCES:.c=.o)
CORE_CU_OBJECTS = $(CORE_CU_SOURCES:.cu=.o)

# The default rule when you type `make` builds the MNIST training demo
all: $(TARGET)

# Rule to link the training demo (We still use NVCC here to pull in the CUDA libraries for the backend)
$(TARGET): $(CORE_C_OBJECTS) $(CORE_CU_OBJECTS) src/train_nnue.o
	$(NVCC) $(CORE_C_OBJECTS) $(CORE_CU_OBJECTS) src/train_nnue.o -o $(TARGET) -lcublas

# Rule to compile standard C files
%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

# Rule to compile CUDA files
%.o: %.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

# Bulletproof clean rule
.PHONY: all clean

clean:
	rm -f src/*.o
	rm -f *.o
	rm -f $(TARGET)