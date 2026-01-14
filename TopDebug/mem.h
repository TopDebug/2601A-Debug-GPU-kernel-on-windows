
#pragma once
#include <stdio.h>
#include <string.h>
#include <cuda_runtime.h>

#ifdef _DEBUG
static unsigned char* CPU = NULL;
static __device__ unsigned char* GPU = NULL;

#define BUFFER_SIZE (sizeof(unsigned char) * 640 * 480)

/*/
template <typename T>
__device__ __forceinline__
static unsigned char* mem2(T* ptr, int size)
{
    unsigned char* data = (unsigned char*)ptr;

    for (int i = 0; i < size; i++) {
        GPU[i] = data[i];
    }

    __threadfence_system();
    return GPU;
}

/*/
#define mem2(ptr, size) { \
    unsigned char* data = (unsigned char*)ptr; \
    for (int i = 0; i < size; i++) { \
        GPU[i] = data[i]; \
    } \
    __threadfence_system();} \
//*/

static void mem2_init()
{
    if (CPU) {
        return;
    }

    unsigned char* GPU_ptr = NULL;
    cudaHostAlloc(&CPU, BUFFER_SIZE, cudaHostAllocMapped);
    cudaHostGetDevicePointer(&GPU_ptr, CPU, 0);
    cudaMemcpyToSymbol(GPU, &GPU_ptr, sizeof(unsigned char*));
}

static void mem2_free()
{
    if (CPU) {
        cudaFreeHost(CPU);
        CPU = NULL;
    }
}

// mem1(img, 640 * 480)
void mem1(unsigned char* ptr, int size)
{
    cudaMemcpy(CPU, ptr, size, cudaMemcpyDeviceToHost);
}

#define IF(idy, idx) \
    if ((blockIdx.x * blockDim.x + threadIdx.x) == (idx) && \
        (blockIdx.y * blockDim.y + threadIdx.y) == (idy))

#else

#define mem(ptr, size)  
#define mem_init()
#define mem_free()
#define IF(idy, idx)
#endif