#include <cstdio>
#include <stdlib.h>
#include <cstring>
#include <iostream>
#include <cmath>
#define A(i, j) a[i * n + j]
void random_matrix(float *a, int M, int N)
{
    for (int m = 0; m < M; m++)
    {
        for (int n = 0; n < N; n++)
        {
            a[m * N + n] = 2.0 * (float)drand48() - 1.0;
        }
    }
}
void sgemm_cpu(float *A_ptr, float *B_ptr, float *C_ptr, const int M, const int N, const int K)
{
    for (int m = 0; m < M; m++)
    {
        for (int n = 0; n < N; n++)
        {
            float res = 0;
            for (int k = 0; k < K; k++)
            {
                res += A_ptr[m * K + k] * B_ptr[k * N + n];
            }
            C_ptr[m * N + n] = res;
        }
    }
}

bool all_close(float *A_ptr, float *B_ptr, const int M, const int N)
{
    for (int m = 0; m < M; m++)
    {
        for (int n = 0; n < N; n++)
        {
            if (std::abs(A_ptr[m * N + n] - B_ptr[m * N + n]) > 1e-4)
            {
                return false;
            }
        }
    }
    return true;
}
__global__ void sgemm_v0_global_memory(float *A, float *B, float *C, int M, int N, int K)
{
    float *block_start_x = B + blockDim.x * blockIdx.x;
    float *block_start_y = A + blockDim.y * blockIdx.y * K;
    float temp = 0.f;
    for (int k = 0; k < K; k++)
    {
        temp += block_start_y[threadIdx.y * K + k] * block_start_x[k * N + threadIdx.x];
    }
    C[(blockDim.y * blockIdx.y + threadIdx.y) * N + blockDim.x * blockIdx.x + threadIdx.x] = temp;
}
template <unsigned int BLOCK_SIZE, unsigned int K_>
__global__ void sgemm_v1_shared_memory(float *A, float *B, float *C, int M, int N, int K)
{
    float *block_start_b = B + blockDim.x * blockIdx.x;
    float *block_start_a = A + blockDim.y * blockIdx.y * K;
    __shared__ float a_shared[BLOCK_SIZE][K_];
    __shared__ float b_shared[K_][BLOCK_SIZE];
    // // 这么做是错的!!
    // for (int k = 0; k < K_; k++)
    // {
    //     a_shared[threadIdx.y][k] = block_start_a[threadIdx.y * K + k];
    //     b_shared[k][threadIdx.x] = block_start_b[k * N + threadIdx.x];
    // }
    for (int s = 0; s < K; s += BLOCK_SIZE)
    {

        a_shared[threadIdx.y][threadIdx.x + s] = block_start_a[threadIdx.y * K + threadIdx.x + s];
        b_shared[threadIdx.y + s][threadIdx.x] = block_start_b[(threadIdx.y + s) * N + threadIdx.x];
    }
    __syncthreads();
    float temp = 0.f;
    for (int k = 0; k < K; k++)
    {
        temp += a_shared[threadIdx.y][k] * b_shared[k][threadIdx.x];
    }
    C[(blockDim.y * blockIdx.y + threadIdx.y) * N + blockDim.x * blockIdx.x + threadIdx.x] = temp;
}
template <unsigned int BLOCK_SIZE>
__global__ void sgemm_v2_shared_memory_sliding_window(float *A, float *B, float *C, int M, int N, int K)
{
    float *block_start_b = B + blockDim.x * blockIdx.x;
    float *block_start_a = A + blockDim.y * blockIdx.y * K;
    __shared__ float a_shared[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float b_shared[BLOCK_SIZE][BLOCK_SIZE];
    float temp = .0f;
    for (int s = 0; s < K; s += BLOCK_SIZE)
    {
        a_shared[threadIdx.y][threadIdx.x] = block_start_a[threadIdx.y * K + threadIdx.x + s];
        b_shared[threadIdx.y][threadIdx.x] = block_start_b[(threadIdx.y + s) * N + threadIdx.x];
        __syncthreads();
        for (int k = 0; k < BLOCK_SIZE; k++)
            temp += a_shared[threadIdx.y][k] * b_shared[k][threadIdx.x];
        __syncthreads();
    }
    C[(blockDim.y * blockIdx.y + threadIdx.y) * N + blockDim.x * blockIdx.x + threadIdx.x] = temp;
}
template <unsigned int BLOCK_SIZE, unsigned int STRIDE>
__global__ void sgemm_v3_increase_work_of_per_thread(float *A, float *B, float *C, int M, int N, int K)
{
    constexpr unsigned int STEP = BLOCK_SIZE * STRIDE;
    float *block_start_a = A + STEP * blockIdx.y * K;
    float *block_start_b = B + STEP * blockIdx.x;
    __shared__ float a_shared[STEP][STEP];
    __shared__ float b_shared[STEP][STEP];
    float temp[STRIDE][STRIDE] = {.0f};
    for (int s = 0; s < K; s += STEP)
    {
        for (int i = 0; i < STRIDE; i++)
        {
            for (int j = 0; j < STRIDE; j++)
            {
                a_shared[threadIdx.y + i * BLOCK_SIZE][threadIdx.x + j * BLOCK_SIZE] = block_start_a[(threadIdx.y + i * BLOCK_SIZE) * K + s + threadIdx.x + j * BLOCK_SIZE];
                b_shared[threadIdx.y + i * BLOCK_SIZE][threadIdx.x + j * BLOCK_SIZE] = block_start_b[(s + threadIdx.y + i * BLOCK_SIZE) * N + threadIdx.x + j * BLOCK_SIZE];
            }
        }
        __syncthreads();
        for (int i = 0; i < STRIDE; i++)
        {

            for (int j = 0; j < STRIDE; j++)
            {
                for (int k = 0; k < STEP; k++)
                {
                    temp[i][j] += a_shared[i * BLOCK_SIZE + threadIdx.y][k] * b_shared[k][j * BLOCK_SIZE + threadIdx.x];
                }
            }
        }
        __syncthreads();
    }
    float *block_start_c = C + STEP * blockIdx.y * N + STEP * blockIdx.x;
    for (int i = 0; i < STRIDE; i++)
    {
        for (int j = 0; j < STRIDE; j++)
        {
            block_start_c[(i * BLOCK_SIZE + threadIdx.y) * N + (j * BLOCK_SIZE + threadIdx.x)] = temp[i][j];
        }
    }
}
#define FETCH_FLOAT4(x) (reinterpret_cast<float4 *>(&(x))[0])
template <unsigned int M_NUM_PER_BLOCK, unsigned int N_NUM_PER_BLOCK, unsigned int K_NUM_PER_BLOCK, unsigned int NUM_PER_THREAD>
__global__ void sgemm_v4_using_float4(float *A, float *B, float *C, int M, int N, int K)
{
    float *block_start_a = A + blockIdx.y * M_NUM_PER_BLOCK * K;
    float *block_start_b = B + blockIdx.x * N_NUM_PER_BLOCK;
    __shared__ float a_shared[M_NUM_PER_BLOCK][K_NUM_PER_BLOCK];
    __shared__ float b_shared[K_NUM_PER_BLOCK][N_NUM_PER_BLOCK];
    float temp[NUM_PER_THREAD] = {.0f};
    for (int s = 0; s < K; s += K_NUM_PER_BLOCK)
    {
        FETCH_FLOAT4(a_shared[threadIdx.y][threadIdx.x * NUM_PER_THREAD]) = FETCH_FLOAT4(block_start_a[threadIdx.y * K + threadIdx.x * NUM_PER_THREAD + s]);
        FETCH_FLOAT4(b_shared[threadIdx.y][threadIdx.x * NUM_PER_THREAD]) = FETCH_FLOAT4(block_start_b[(threadIdx.y + s) * N + threadIdx.x * NUM_PER_THREAD]);
        // a_shared[threadIdx.y][threadIdx.x * NUM_PER_THREAD] = block_start_a[threadIdx.y * K + threadIdx.x * NUM_PER_THREAD + s];
        // a_shared[threadIdx.y][threadIdx.x * NUM_PER_THREAD + 1] = block_start_a[threadIdx.y * K + threadIdx.x * NUM_PER_THREAD + s + 1];
        // a_shared[threadIdx.y][threadIdx.x * NUM_PER_THREAD + 2] = block_start_a[threadIdx.y * K + threadIdx.x * NUM_PER_THREAD + s + 2];
        // a_shared[threadIdx.y][threadIdx.x * NUM_PER_THREAD + 3] = block_start_a[threadIdx.y * K + threadIdx.x * NUM_PER_THREAD + s + 3];
        // b_shared[threadIdx.y][threadIdx.x * NUM_PER_THREAD] = block_start_b[(threadIdx.y + s) * N + threadIdx.x * NUM_PER_THREAD];
        // b_shared[threadIdx.y][threadIdx.x * NUM_PER_THREAD + 1] = block_start_b[(threadIdx.y + s) * N + threadIdx.x * NUM_PER_THREAD + 1];
        // b_shared[threadIdx.y][threadIdx.x * NUM_PER_THREAD + 2] = block_start_b[(threadIdx.y + s) * N + threadIdx.x * NUM_PER_THREAD + 2];
        // b_shared[threadIdx.y][threadIdx.x * NUM_PER_THREAD + 3] = block_start_b[(threadIdx.y + s) * N + threadIdx.x * NUM_PER_THREAD + 3];
        __syncthreads();
        for (int i = 0; i < NUM_PER_THREAD; i++)
        {
            for (int k = 0; k < K_NUM_PER_BLOCK; k++)
            {
                temp[i] += a_shared[threadIdx.y][k] * b_shared[k][threadIdx.x * NUM_PER_THREAD + i];
            }
        }
        __syncthreads();
    }
    float *block_start_c = C + blockIdx.y * M_NUM_PER_BLOCK * N + blockIdx.x * N_NUM_PER_BLOCK;
    for (int i = 0; i < NUM_PER_THREAD; i++)
    {
        block_start_c[threadIdx.y * N + threadIdx.x * NUM_PER_THREAD + i] = temp[i];
    }
}
int main()
{
    int m = 128;
    int n = 128;
    constexpr int k = 128;
    const size_t mem_size_A = m * k * sizeof(float);
    const size_t mem_size_B = k * n * sizeof(float);
    const size_t mem_size_C = m * n * sizeof(float);
    float *matrix_A_host = (float *)malloc(mem_size_A);
    float *matrix_B_host = (float *)malloc(mem_size_B);
    float *matrix_C_host_gpu_calc = (float *)malloc(mem_size_C);
    float *matrix_C_host_cpu_calc = (float *)malloc(mem_size_C);
    random_matrix(matrix_A_host, m, k);
    random_matrix(matrix_B_host, k, n);
    memset(matrix_C_host_gpu_calc, 0, mem_size_C);
    memset(matrix_C_host_cpu_calc, 0, mem_size_C);
    sgemm_cpu(matrix_A_host, matrix_B_host, matrix_C_host_cpu_calc, m, n, k);
    float *matrix_A_device;
    float *matrix_B_device;
    float *matrix_C_device;
    cudaMalloc(&matrix_A_device, mem_size_A);
    cudaMalloc(&matrix_B_device, mem_size_B);
    cudaMalloc(&matrix_C_device, mem_size_C);
    cudaMemcpy(matrix_A_device, matrix_A_host, mem_size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(matrix_B_device, matrix_B_host, mem_size_B, cudaMemcpyHostToDevice);
    constexpr int BLOCK = 16;
    dim3 block(BLOCK, BLOCK);
    dim3 grid((n + BLOCK - 1) / BLOCK, (m + BLOCK - 1) / BLOCK);
    // sgemm_v0_global_memory<<<grid, block>>>(matrix_A_device, matrix_B_device, matrix_C_device, m, n, k);
    // sgemm_v1_shared_memory<BLOCK, k><<<grid, block>>>(matrix_A_device, matrix_B_device, matrix_C_device, m, n, k);
    // sgemm_v2_shared_memory_sliding_window<BLOCK><<<grid, block>>>(matrix_A_device, matrix_B_device, matrix_C_device, m, n, k);
    // constexpr int STRIDE = 2;
    // dim3 grid1((n + BLOCK - 1) / BLOCK / STRIDE, (m + BLOCK - 1) / BLOCK / STRIDE);
    // sgemm_v3_increase_work_of_per_thread<BLOCK, STRIDE><<<grid1, block>>>(matrix_A_device, matrix_B_device, matrix_C_device, m, n, k);

    constexpr int M_NUM_PER_BLOCK = 32;
    constexpr int N_NUM_PER_BLOCK = 32;
    constexpr int K_NUM_PER_BLOCK = 32;
    constexpr int NUM_PER_THREAD = 4;
    dim3 block_v4(8, 32);
    dim3 grid_v4(m / M_NUM_PER_BLOCK, n / N_NUM_PER_BLOCK);
    sgemm_v4_using_float4<M_NUM_PER_BLOCK, N_NUM_PER_BLOCK, K_NUM_PER_BLOCK, NUM_PER_THREAD><<<grid_v4, block_v4>>>(matrix_A_device, matrix_B_device, matrix_C_device, m, n, k);
    cudaMemcpy(matrix_C_host_gpu_calc, matrix_C_device, mem_size_C, cudaMemcpyDeviceToHost);
    if (all_close(matrix_C_host_gpu_calc, matrix_C_host_cpu_calc, m, n))
    {
        printf(" success!\n");
    }
    else
    {
        for (int i = 0; i < m * n; i++)
        {
            printf("cpu: %f, gpu: %f\n", matrix_C_host_cpu_calc[i], matrix_C_host_gpu_calc[i]);
        }
        std::cout << "failed!" << std::endl;
        printf(" failed!\n");
    }
    cudaFree(matrix_A_device);
    cudaFree(matrix_B_device);
    cudaFree(matrix_C_device);
    free(matrix_A_host);
    free(matrix_B_host);
    free(matrix_C_host_gpu_calc);
    free(matrix_C_host_cpu_calc);
    return 0;
}