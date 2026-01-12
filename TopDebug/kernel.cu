
#include "kernel.h"
#include "mem.h"

__global__ void TestKernel(unsigned char* img, int nx, int ny)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= nx || y >= ny) return;

	int idx = y * nx + x;
    if (img[idx] == 1)
        img[idx] = 50;

    if (img[idx] == 128)
        img[idx] = 100;

    IF(0, 1)
        mem2(img, nx * ny); // debug

    if (img[idx] == 255)
        img[idx] = 200;

    IF(0, 1)
        mem2(img, nx * ny); // debug
}

void testKernel(unsigned char* img, int nx, int ny)
{
    mem2_init();

    //dim3 block(160, 160);
    dim3 block(32, 32);
    dim3 grid(
        (640 + block.x - 1) / block.x, // 4
        (480 + block.y - 1) / block.y // 3
    );

    TestKernel <<<grid, block >>> (img, nx, ny);
    cudaDeviceSynchronize();

    mem2_free();
}
