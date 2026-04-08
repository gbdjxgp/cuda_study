#include "cuda.h"
#include "iostream"
#include "random"

void __global__ add1(float *d_input_x, float *d_input_y, float *d_output)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    d_output[idx] = d_input_x[idx] + d_input_y[idx];
}
void __global__ add2(float *d_input_x, float *d_input_y, float *d_output)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x + 1;
    d_output[idx] = d_input_x[idx] + d_input_y[idx];
}
void __global__ add3(float *d_input_x, float *d_input_y, float *d_output)
{
    int tid_permuted = threadIdx.x ^ 0x1;
    int idx = blockDim.x * blockIdx.x + tid_permuted;
    d_output[idx] = d_input_x[idx] + d_input_y[idx];
}
void __global__ add4(float *d_input_x, float *d_input_y, float *d_output)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x + 1;
    int warp_idx = idx / 32;

    d_output[warp_idx] = d_input_x[warp_idx] + d_input_y[warp_idx];
}
void __global__ add5(float *d_input_x, float *d_input_y, float *d_output)
{
    int idx = (blockDim.x * blockIdx.x + threadIdx.x) * 4;
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
    for (int i = 0; i < 2; i++)
    {
        add2<<<dim3(N / 256), dim3(64)>>>(d_input_x, d_input_y, d_output);
        cudaDeviceSynchronize();
    }
    for (int i = 0; i < 2; i++)
    {
        add3<<<dim3(N / 256), dim3(64)>>>(d_input_x, d_input_y, d_output);
        cudaDeviceSynchronize();
    }
    for (int i = 0; i < 2; i++)
    {
        add4<<<dim3(N / 256), dim3(64)>>>(d_input_x, d_input_y, d_output);
        cudaDeviceSynchronize();
    }
    for (int i = 0; i < 2; i++)
    {
        add5<<<dim3(N / 256), dim3(64)>>>(d_input_x, d_input_y, d_output);
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
