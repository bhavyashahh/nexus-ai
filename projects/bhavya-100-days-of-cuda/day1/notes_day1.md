# CUDA Basics

Compute Unified Device Architecture (CUDA) is a software programming model that enables users to harness the power of Nvidia GPUs for general-purpose computing.

## Key Terminologies

| Term | Description |
|------|-------------|
| Host | CPU |
| Device | GPU |
| Kernel | Function to be executed on a GPU |
| Stream Multiprocessors (SM) | GPU's computing unit |
| Stream Processors (SP) | Actual computation units within an SM (also called CUDA cores) |
| Thread | Single instance of execution on one SP |
| Block | Group of threads |
| Grid | Group of blocks |
| Warp | Number of threads in a block running simultaneously on an SM |

## Memory Allocation

| Function | Purpose | Location | Initialization |
|----------|---------|----------|----------------|
| malloc() | Allocates memory block | Process heap | Uninitialized |
| calloc() | Allocates and initializes memory | Heap | Initialized to zero |
| alloca() | Allocates memory on the stack | Stack frame | Uninitialized |

## Day 1 Key Learnings

1. **CUDA Basics**: Learned the fundamentals of CUDA programming, including kernel definition, thread/block organization, and memory management between CPU and GPU.

2. **VectorAddition Kernel**: Implemented vector addition using CUDA, demonstrating the ability to distribute computational tasks across multiple GPU threads.

3. **CUDA Syntax and Execution**: Understood CUDA-specific syntax for kernel launches, thread identification, and device synchronization, essential for effective GPU programming.

4. **Resource Management**: Learned the importance of proper GPU memory allocation, data transfer, and resource deallocation to optimize CUDA program efficiency.