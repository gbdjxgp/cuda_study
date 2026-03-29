// 用于格式化打印字符串
#include <cstdio>
// 用于cout
#include <iostream>
// 用于malloc，随机数生成
#include <cstdlib>
// 用于float绝对值计算
#include <cmath>
#include <cuda.h>
#define THREAD_PER_BLOCK 256
__device__ void warpReduce(volatile float *sdata)
{
    sdata[threadIdx.x] += sdata[threadIdx.x + 32];
    sdata[threadIdx.x] += sdata[threadIdx.x + 16];
    sdata[threadIdx.x] += sdata[threadIdx.x + 8];
    sdata[threadIdx.x] += sdata[threadIdx.x + 4];
    sdata[threadIdx.x] += sdata[threadIdx.x + 2];
    sdata[threadIdx.x] += sdata[threadIdx.x + 1];
}
template <unsigned int NUM_PER_BLOCK>
__global__ void all_reduce_v7_multi_add(float *d_input, float *d_output, int N)
{
    int block_start = blockIdx.x * NUM_PER_BLOCK;
    float *input_start = d_input + block_start;
    __shared__ float sdata[THREAD_PER_BLOCK];
    sdata[threadIdx.x] = 0.0f;
    for (int i = 0; i < NUM_PER_BLOCK; i += blockDim.x)
        sdata[threadIdx.x] += ((block_start + i + threadIdx.x) < N) ? input_start[threadIdx.x + i] : 0.0f;
    __syncthreads();
    if (threadIdx.x < 128)
    {
        sdata[threadIdx.x] += sdata[threadIdx.x + 128];
    }
    __syncthreads();
    if (threadIdx.x < 64)
    {
        sdata[threadIdx.x] += sdata[threadIdx.x + 64];
    }
    __syncthreads();
    if (threadIdx.x < 32)
    {
        warpReduce(sdata);
    }
    if (threadIdx.x == 0 && block_start < N)
        d_output[blockIdx.x] = sdata[0];
}
template <unsigned int NUM_PER_BLOCK>
__global__ void all_reduce_v8_shuffle(float *d_input, float *d_output, int N)
{
    int block_start = blockIdx.x * NUM_PER_BLOCK;
    float *input_start = d_input + block_start;
    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;
    float data = 0.0f;
    // 首先从global_memory放到寄存器中，此时要求一个thread处理多个数据。
    for (int i = threadIdx.x; i < NUM_PER_BLOCK; i += blockDim.x)
        data += ((block_start + i) < N) ? input_start[i] : 0.0f;
    // 此时每个block中的所有数据已经在threads上了，因此可以做warp内的shuffle求和。
    data += __shfl_down_sync(0xffffffff, data, 16);
    data += __shfl_down_sync(0xffffffff, data, 8);
    data += __shfl_down_sync(0xffffffff, data, 4);
    data += __shfl_down_sync(0xffffffff, data, 2);
    data += __shfl_down_sync(0xffffffff, data, 1);
    __shared__ float sdata[32];
    if (lane_id == 0)
        sdata[warp_id] = data;
    __syncthreads();
    if (warp_id == 0)
    {
        data = (lane_id < (THREAD_PER_BLOCK / 32)) ? sdata[lane_id] : 0.0f;
        data += __shfl_down_sync(0xffffffff, data, 16);
        data += __shfl_down_sync(0xffffffff, data, 8);
        data += __shfl_down_sync(0xffffffff, data, 4);
        data += __shfl_down_sync(0xffffffff, data, 2);
        data += __shfl_down_sync(0xffffffff, data, 1);
    }
    if (lane_id == 0 && warp_id == 0)
        d_output[blockIdx.x] = data;
}
bool all_close(float *out, float *res, int n)
{
    for (int i = 0; i < n; i++)
    {
        // 这里1e-5会报错
        if (std::abs(out[i] - res[i]) > 5e-3)
        {
            return false;
        }
    }
    return true;
}
int main()
{
    constexpr int N = 32 * 1024 * 1024;
    if (N == 0)
    {
        std::cout << "success!" << std::endl;
        return 0;
    }
    constexpr unsigned int block_num = 1024;
    constexpr unsigned int number_per_block = (N + block_num - 1) / block_num;
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
        float cur = 0.0f;
        for (unsigned int j = 0; j < number_per_block; j++)
        {
            unsigned int idx = i * number_per_block + j;
            if (idx < N)
                cur += h_input[idx];
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
    all_reduce_v8_shuffle<number_per_block><<<Grid, Block>>>(d_input, d_output, N);
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
    free(h_input);
    free(h_output_cpu);
    free(h_output_gpu);
    cudaFree(d_input);
    cudaFree(d_output);
    return 0;
}
