#include <cstdio>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>
#include "kernel.h"

int main()
{
    unsigned char* inputImg = (unsigned char*)calloc(sizeof(unsigned char), 640 * 480);

    memset(inputImg, 1, sizeof(unsigned char) * 640 * 480);
    int x0 = 200, y0 = 100;
    for (int y = y0; y <= y0 + 300; y++) {
        float t = float(y - y0) / 300;
        int w = 80 * (1 - t) + 1 * t;
        memset(inputImg + y * 640 + x0 - w / 2, 128, sizeof(unsigned char) * w);
    }
    for (int dy = -25; dy <= 25; dy++) {
        memset(inputImg + (y0 + dy) * 640 + x0 - 150 - dy, 255, sizeof(unsigned char) * 300);
    }

    cudaError_t cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        return 0;
    }

    unsigned char* inputImg_d = NULL;

    cudaStatus = cudaMalloc((void**)&inputImg_d, sizeof(unsigned char) * 640 * 480);
    if (cudaStatus != cudaSuccess) {
        return 0;
    }

    cudaStatus = cudaMemcpy(inputImg_d, inputImg, sizeof(unsigned char) * 640 * 480, cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        return 0;
    }

    testKernel(inputImg_d, 640, 480);

    unsigned char* outputImg = (unsigned char*)calloc(sizeof(unsigned char), 640 * 480);

    cudaStatus = cudaMemcpy(outputImg, inputImg_d, sizeof(unsigned char) * 640 * 480, cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        return 0;
    }

    FILE* fp_output = fopen("result.img", "wb");
    fwrite(inputImg, sizeof(unsigned char), 640 * 480, fp_output);
    fwrite(outputImg, sizeof(unsigned char), 640 * 480, fp_output);
    fclose(fp_output);

    cudaFree(inputImg_d);
	free(inputImg);
    free(outputImg);
    
    return 1;
}