
# CUDA Kernel Launch Parameters

## Overview

CUDA kernels are launched with specific parameters that define the grid and block dimensions. These parameters determine how the threads are organized and executed on the GPU.

## Types of Launch Parameters

### `dim3` Type

- `dim3` is a 3D type used for specifying grid and thread block dimensions.
- Allows indexing of elements as vector, matrix, or volume (tensor).

Example:
```cpp
dim3 gridDim(4, 4, 1);  // 4 blocks in x, 4 blocks in y, 1 block in z
dim3 blockDim(4, 2, 2); // 4 threads in x, 2 threads in y, 2 threads in z
```

int Type

int can be used to specify a 1D vector of blocks or threads.

Example:

```cpp
int gridDim = 16;  // 16 blocks
int blockDim = 32; // 32 threads per block
<<<gridDim, blockDim>>>
```

Understanding Dimensions

gridDim: Total number of blocks = gridDim.x * gridDim.y * gridDim.z

blockDim: Threads per block = blockDim.x * blockDim.y * blockDim.z

Total threads = (threads per block) * (number of blocks)

Kernel Launch Syntax

The execution configuration is specified in the following format:

<<<gridDim, blockDim, Ns, S>>>

Where:

gridDim (dim3): Specifies the dimension and size of the grid.

blockDim (dim3): Specifies the dimension and size of each block.

Ns (size_t): Optional. Specifies the number of bytes in shared memory dynamically allocated per block.

S (cudaStream_t): Optional. Specifies the associated stream (defaults to 0).

Notes


The Ns and S parameters are often omitted in basic kernel launches.
When using int types for 1D configurations, they are still valid even though they're not dim3 types.