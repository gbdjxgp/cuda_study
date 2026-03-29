// 用于格式化打印字符串
#include <cstdio>
// 用于cout
#include <iostream>
// 用于malloc，随机数生成
#include <cstdlib>
// 用于float绝对值计算
#include <cmath>
#include <cuda.h>
#define THREAD_PER_BLOCK 256 / 2

__global__ void all_reduce_v4_add_during_load_plan_b(float *d_input, float *d_output, int N)
{
    int idx = blockIdx.x * blockDim.x * 2 + threadIdx.x;
    __shared__ float sdata[THREAD_PER_BLOCK];
    sdata[threadIdx.x] = (idx < N) ? d_input[idx] : 0.0f;
    sdata[threadIdx.x] += (idx + blockDim.x < N) ? d_input[idx + blockDim.x] : 0.0f;
    __syncthreads();
    for (int i = blockDim.x / 2; i > 0; i /= 2)
    {
        if (threadIdx.x < i)
        {
            sdata[threadIdx.x] += sdata[threadIdx.x + i];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0 && blockIdx.x * blockDim.x * 2 < N)
        d_output[blockIdx.x] = sdata[0];
}

bool all_close(float *out, float *res, int n)
{
    for (int i = 0; i < n; i++)
    {
        // 这里1e-5会报错
        if (std::abs(out[i] - res[i]) > 1e-4)
        {
            return false;
        }
    }
    return true;
}
int main()
{
    const int N = 32 * 1024 * 1024;
    if (N == 0)
    {
        std::cout << "success!" << std::endl;
        return 0;
    }
    int block_num = (N + 2 * THREAD_PER_BLOCK - 1) / (2 * THREAD_PER_BLOCK);
    // 准备内存空间
    float *h_input = (float *)malloc(N * sizeof(float));
    // 随机数生成
    for (int i = 0; i < N; i++)
    {
        // drand48 返回一个 [0.0, 1.0) 区间内的 double 随机数，rand()返回 int，drand返回double
        h_input[i] = 2.0 * (float)drand48() - 1.0;
    }

    // h_output_cpu：cpu的计算结果
    float *h_output_cpu = (float *)malloc(block_num * sizeof(float));

    // cpu计算
    for (int i = 0; i < block_num; i++)
    {
        float cur = 0.0;
        for (int j = 0; j < THREAD_PER_BLOCK * 2; j++)
        {
            if (i * THREAD_PER_BLOCK * 2 + j < N)
                cur += h_input[i * THREAD_PER_BLOCK * 2 + j];
        }
        h_output_cpu[i] = cur;
    }

    // 准备GPU中的数据
    float *d_input;
    // 这里的&很关键！！要传递指针的引用，也就是二级指针！
    cudaMalloc((void **)&d_input, N * sizeof(float));
    cudaMemcpy(d_input, h_input, N * sizeof(float), cudaMemcpyHostToDevice);
    float *d_output;
    cudaMalloc((void **)&d_output, block_num * sizeof(float));
    dim3 Grid(block_num);
    dim3 Block(THREAD_PER_BLOCK);
    // 调用核函数
    // all_reduce_v0_global_memory<<<Grid, Block>>>(d_input, d_output, N);
    all_reduce_v4_add_during_load_plan_b<<<Grid, Block>>>(d_input, d_output, N);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        std::cout << "CUDA error: " << cudaGetErrorString(err) << std::endl;
    cudaError_t err_sync = cudaDeviceSynchronize();
    if (err_sync != cudaSuccess)
        std::cout << "CUDA error: " << cudaGetErrorString(err_sync) << std::endl;
    // GPU中的输出结果
    float *h_output_gpu = (float *)malloc(block_num * sizeof(float));
    cudaMemcpy(h_output_gpu, d_output, block_num * sizeof(float), cudaMemcpyDeviceToHost);
    // 判断计算是否正确
    if (all_close(h_output_cpu, h_output_gpu, block_num))
    {
        std::cout << "success!" << std::endl;
    }
    else
    {
        for (int i = 0; i < block_num; i++)
        {
            printf("cpu: %f, gpu: %f\n", h_output_cpu[i], h_output_gpu[i]);
        }
        std::cout << "failed!" << std::endl;
    }
    return 0;
}
