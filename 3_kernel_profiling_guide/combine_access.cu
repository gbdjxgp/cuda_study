#include "cuda.h"
#include "iostream"
#include "random"

__global__ void add1(float *d_input_x, float *d_input_y, float *d_output)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    d_output[idx] = d_input_x[idx] + d_input_y[idx];
}
int main()
{
    int N = 32 * 1024 * 1024;
    float *input_x = (float *)malloc(N * sizeof(float));
    float *input_y = (float *)malloc(N * sizeof(float));
    float *output_cpu = (float *)malloc(N * sizeof(float));
    float *output_gpu = (float *)malloc(N * sizeof(float));
    float *d_input_x, *d_input_y, *d_output;
    cudaMalloc(&d_input_x, N * sizeof(float));
    cudaMalloc(&d_input_y, N * sizeof(float));
    cudaMalloc(&d_output, N * sizeof(float));
    cudaMemcpy(d_input_x, input_x, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_input_y, input_y, N * sizeof(float), cudaMemcpyHostToDevice);
    for (int i = 0; i < 2; i++)
    {
        add1<<<dim3(N / 256), dim3(64)>>>(d_input_x, d_input_y, d_output);
        cudaDeviceSynchronize();
    }
    cudaMemcpy(output_gpu, d_output, N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_input_x);
    cudaFree(d_input_y);
    cudaFree(d_output);
    free(input_x);
    free(input_y);
    free(output_cpu);
    free(output_gpu);

    return 0;
}
