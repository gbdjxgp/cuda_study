#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#define FETCH_FLOAT4(x) (reinterpret_cast<float4 *>(&(x))[0])
#define FETCH_FLOAT2(x) (reinterpret_cast<float2 *>(&(x)))[0]
static void check_cuda_at_end(const char *where)
{
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        std::fprintf(stderr, "[%s] CUDA launch error: %s\n", where, cudaGetErrorString(err));
        std::exit(EXIT_FAILURE);
    }

    err = cudaDeviceSynchronize();
    if (err != cudaSuccess)
    {
        std::fprintf(stderr, "[%s] CUDA runtime error: %s\n", where, cudaGetErrorString(err));
        std::exit(EXIT_FAILURE);
    }
}

class Perf
{
public:
    Perf(const std::string &name)
    {
        m_name = name;
        cudaEventCreate(&m_start);
        cudaEventCreate(&m_end);
        cudaEventRecord(m_start);
        cudaEventSynchronize(m_start);
    }
    ~Perf()
    {
        cudaEventRecord(m_end);
        cudaEventSynchronize(m_end);
        float elapsed_time = 0.0;
        cudaEventElapsedTime(&elapsed_time, m_start, m_end);
        std::cout << m_name << "elapsed: " << elapsed_time << " ms" << std::endl;
    }

private:
    std::string m_name;
    cudaEvent_t m_start, m_end;
};
bool check(float *cpu_result, float *gpu_result, const int M, const int N)
{
    const int size = M * N;
    for (int i = 0; i < size; i++)
    {
        if (cpu_result[i] != gpu_result[i])
        {
            return false;
        }
    }
    return true;
}
__global__ void transpose_v1_naive(float *input, float *output, const int M, const int N)
{
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;
    int idx = y * N + x;
    int trans_idx = x * M + y;
    output[trans_idx] = input[idx];
}
__global__ void transpose_v2_float4_inner_4x4(float *input, float *output, const int M, const int N)
{
    const int THREAD_SIZE_X = blockDim.x * 4;
    const int THREAD_SIZE_Y = blockDim.y * 4;
    float *block_start = input + blockIdx.y * THREAD_SIZE_Y * N + blockIdx.x * THREAD_SIZE_X;
    float src_transpose[4][4];
    float dst_transpose[4][4];
    for (int i = 0; i < 4; i++)
    {
        FETCH_FLOAT4(src_transpose[i]) = FETCH_FLOAT4(block_start[(threadIdx.y * 4 + i) * N + threadIdx.x * 4]);
    }
    for (int i = 0; i < 4; i++)
    {

        FETCH_FLOAT4(dst_transpose[i]) = make_float4(src_transpose[0][i], src_transpose[1][i], src_transpose[2][i], src_transpose[3][i]);
    }
    float *output_start = output + blockIdx.x * THREAD_SIZE_X * M + blockIdx.y * THREAD_SIZE_Y;
    for (int i = 0; i < 4; i++)
    {
        FETCH_FLOAT4(output_start[(threadIdx.x * 4 + i) * M + threadIdx.y * 4]) = FETCH_FLOAT4(dst_transpose[i]);
    }
}
__global__ void transpose_v3_float2_inner_2x2(float *input, float *output, const int M, const int N)
{
    const int THREAD_SIZE_X = blockDim.x * 2;
    const int THREAD_SIZE_Y = blockDim.y * 2;
    float *block_start = input + blockIdx.y * THREAD_SIZE_Y * N + blockIdx.x * THREAD_SIZE_X;
    float src_transpose[2][2];
    float dst_transpose[2][2];
    for (int i = 0; i < 2; i++)
    {
        FETCH_FLOAT2(src_transpose[i]) = FETCH_FLOAT2(block_start[(threadIdx.y * 2 + i) * N + threadIdx.x * 2]);
    }
    for (int i = 0; i < 2; i++)
    {
        FETCH_FLOAT2(dst_transpose[i]) = make_float2(src_transpose[0][i], src_transpose[1][i]);
    }
    float *output_start = output + blockIdx.x * THREAD_SIZE_X * M + blockIdx.y * THREAD_SIZE_Y;
    for (int i = 0; i < 2; i++)
    {
        FETCH_FLOAT2(output_start[(threadIdx.x * 2 + i) * M + threadIdx.y * 2]) = FETCH_FLOAT2(dst_transpose[i]);
    }
}
__global__ void transpose_v4_float2_inner_1x2(float *input, float *output, const int M, const int N)
{
    const int THREAD_SIZE_X = blockDim.x;
    const int THREAD_SIZE_Y = blockDim.y * 2;
    float *block_start = input + blockIdx.y * THREAD_SIZE_Y * N + blockIdx.x * THREAD_SIZE_X;
    float src_transpose[2];
    float dst_transpose[2];
    src_transpose[0] = block_start[(threadIdx.y * 2) * N + threadIdx.x];
    src_transpose[1] = block_start[(threadIdx.y * 2 + 1) * N + threadIdx.x];
    FETCH_FLOAT2(dst_transpose[0]) = make_float2(src_transpose[0], src_transpose[1]);
    float *output_start = output + blockIdx.x * THREAD_SIZE_X * M + blockIdx.y * THREAD_SIZE_Y;
    FETCH_FLOAT2(output_start[threadIdx.x * M + threadIdx.y * 2]) = FETCH_FLOAT2(dst_transpose[0]);
}
__global__ void transpose_v5_shared_memory(float *input, float *output, const int M, const int N)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // float *block_start = input + blockIdx.y * blockDim.y * N + blockDim.x * blockIdx.x;
    __shared__ float sdata[16][16];
    if (y < M && x < N)
    {
        sdata[threadIdx.x][threadIdx.y] = input[y * N + x];
    }
    __syncthreads();

    int y_out = blockIdx.x * blockDim.x + threadIdx.y;
    int x_out = blockIdx.y * blockDim.y + threadIdx.x;
    if (y_out < N && x_out < M)
    {
        output[y_out * M + x_out] = sdata[threadIdx.y][threadIdx.x];
    }
}
void transpose_cpu(float *input, float *output, const int M, const int N)
{
    for (int m = 0; m < M; m++)
    {
        for (int n = 0; n < N; n++)
        {
            const int input_index = m * N + n;
            const int output_index = n * M + m;
            output[output_index] = input[input_index];
        }
    }
}
int main()
{
    // const int MATRIX_M = 2048;
    // const int MATRIX_N = 512;
    const int MATRIX_M = 2300;
    const int MATRIX_N = 1500;
    const size_t size = MATRIX_M * MATRIX_N;
    float *input_host = (float *)malloc(size * sizeof(float));
    float *output_host_cpu_calc = (float *)malloc(size * sizeof(float));
    float *output_host_gpu_calc = (float *)malloc(size * sizeof(float));
    for (int i = 0; i < size; i++)
    {
        input_host[i] = 2.0 * (float)drand48() - 1.0;
    }
    transpose_cpu(input_host, output_host_cpu_calc, MATRIX_M, MATRIX_N);
    float *input_device, *output_device;
    cudaMalloc(&input_device, size * sizeof(float));
    cudaMalloc(&output_device, size * sizeof(float));
    cudaMemcpy(input_device, input_host, size * sizeof(float), cudaMemcpyHostToDevice);
    // cudaMemset(output_device, 0, size * sizeof(float));
    // for (int i = 0; i < 5; i++)
    // {
    //     Perf perf("transpose32_8");
    //     dim3 block_size(32, 8);
    //     dim3 grid_size((MATRIX_N - 1) / 32 + 1, (MATRIX_M - 1) / 8 + 1);
    //     transpose_v1_naive<<<grid_size, block_size>>>(input_device, output_device, MATRIX_M, MATRIX_N);
    //     cudaDeviceSynchronize();
    // }
    // cudaMemcpy(output_host_gpu_calc, output_device, size * sizeof(float), cudaMemcpyDeviceToHost);
    // if (check(output_host_cpu_calc, output_host_gpu_calc, MATRIX_M, MATRIX_N))
    // {
    //     std::cout << "right" << std::endl;
    // }

    // cudaMemset(output_device, 0, size * sizeof(float));
    // for (int i = 0; i < 5; i++)
    // {
    //     Perf perf("transpose16-16");
    //     dim3 block_size(16, 16);
    //     dim3 grid_size((MATRIX_N - 1) / 16 + 1, (MATRIX_M - 1) / 16 + 1);
    //     transpose_v1_naive<<<grid_size, block_size>>>(input_device, output_device, MATRIX_M, MATRIX_N);
    //     cudaDeviceSynchronize();
    // }
    // cudaMemcpy(output_host_gpu_calc, output_device, size * sizeof(float), cudaMemcpyDeviceToHost);
    // if (check(output_host_cpu_calc, output_host_gpu_calc, MATRIX_M, MATRIX_N))
    // {
    //     std::cout << "right" << std::endl;
    // }

    // cudaMemset(output_device, 0, size * sizeof(float));
    // for (int i = 0; i < 5; i++)
    // {
    //     Perf perf("transpose8-32");
    //     dim3 block_size(8, 32);
    //     dim3 grid_size((MATRIX_N - 1) / 8 + 1, (MATRIX_M - 1) / 32 + 1);
    //     transpose_v1_naive<<<grid_size, block_size>>>(input_device, output_device, MATRIX_M, MATRIX_N);
    //     cudaDeviceSynchronize();
    // }
    // cudaMemcpy(output_host_gpu_calc, output_device, size * sizeof(float), cudaMemcpyDeviceToHost);
    // if (check(output_host_cpu_calc, output_host_gpu_calc, MATRIX_M, MATRIX_N))
    // {
    //     std::cout << "right" << std::endl;
    // }

    // cudaMemset(output_device, 0, size * sizeof(float));
    // for (int i = 0; i < 5; i++)
    // {
    //     Perf perf("transpose_v2_32_8");
    //     dim3 block_size(32, 8);
    //     dim3 grid_size(((MATRIX_N >> 2) - 1) / block_size.x + 1, ((MATRIX_M >> 2) - 1) / block_size.y + 1);
    //     transpose_v2_float4_inner_4x4<<<grid_size, block_size>>>(input_device, output_device, MATRIX_M, MATRIX_N);
    //     cudaDeviceSynchronize();
    // }
    // cudaMemcpy(output_host_gpu_calc, output_device, size * sizeof(float), cudaMemcpyDeviceToHost);
    // if (check(output_host_cpu_calc, output_host_gpu_calc, MATRIX_M, MATRIX_N))
    // {
    //     std::cout << "right" << std::endl;
    // }

    // cudaMemset(output_device, 0, size * sizeof(float));
    // for (int i = 0; i < 5; i++)
    // {

    //     Perf perf("transpose_v2_16_16");
    //     dim3 block_size(16, 16);
    //     dim3 grid_size(((MATRIX_N >> 2) - 1) / block_size.x + 1, ((MATRIX_M >> 2) - 1) / block_size.y + 1);
    //     transpose_v2_float4_inner_4x4<<<grid_size, block_size>>>(input_device, output_device, MATRIX_M, MATRIX_N);
    //     cudaDeviceSynchronize();
    // }
    // cudaMemcpy(output_host_gpu_calc, output_device, size * sizeof(float), cudaMemcpyDeviceToHost);
    // if (check(output_host_cpu_calc, output_host_gpu_calc, MATRIX_M, MATRIX_N))
    // {
    //     std::cout << "right" << std::endl;
    // }

    // cudaMemset(output_device, 0, size * sizeof(float));
    // for (int i = 0; i < 5; i++)
    // {
    //     Perf perf("transpose_v2_8_32");
    //     dim3 block_size(8, 32);
    //     dim3 grid_size(((MATRIX_N >> 2) - 1) / block_size.x + 1, ((MATRIX_M >> 2) - 1) / block_size.y + 1);
    //     transpose_v2_float4_inner_4x4<<<grid_size, block_size>>>(input_device, output_device, MATRIX_M, MATRIX_N);
    //     cudaDeviceSynchronize();
    // }
    // cudaMemcpy(output_host_gpu_calc, output_device, size * sizeof(float), cudaMemcpyDeviceToHost);
    // if (check(output_host_cpu_calc, output_host_gpu_calc, MATRIX_M, MATRIX_N))
    // {
    //     std::cout << "right" << std::endl;
    // }

    // cudaMemset(output_device, 0, size * sizeof(float));
    // for (int i = 0; i < 5; i++)
    // {

    //     Perf perf("transpose_v3_8_32");
    //     dim3 block_size(8, 32);
    //     dim3 grid_size(((MATRIX_N >> 1) - 1) / block_size.x + 1, ((MATRIX_M >> 1) - 1) / block_size.y + 1);
    //     transpose_v3_float2_inner_2x2<<<grid_size, block_size>>>(input_device, output_device, MATRIX_M, MATRIX_N);
    //     cudaDeviceSynchronize();
    // }
    // cudaMemcpy(output_host_gpu_calc, output_device, size * sizeof(float), cudaMemcpyDeviceToHost);
    // if (check(output_host_cpu_calc, output_host_gpu_calc, MATRIX_M, MATRIX_N))
    // {
    //     std::cout << "right" << std::endl;
    // }

    // cudaMemset(output_device, 0, size * sizeof(float));
    // for (int i = 0; i < 5; i++)
    // {
    //     Perf perf("transpose_v4_8_32");
    //     dim3 block_size(8, 32);
    //     dim3 grid_size(((MATRIX_N)-1) / block_size.x + 1, ((MATRIX_M >> 1) - 1) / block_size.y + 1);
    //     transpose_v4_float2_inner_1x2<<<grid_size, block_size>>>(input_device, output_device, MATRIX_M, MATRIX_N);
    //     cudaDeviceSynchronize();
    // }
    // cudaMemcpy(output_host_gpu_calc, output_device, size * sizeof(float), cudaMemcpyDeviceToHost);
    // if (check(output_host_cpu_calc, output_host_gpu_calc, MATRIX_M, MATRIX_N))
    // {
    //     std::cout << "right" << std::endl;
    // }

    cudaMemset(output_device, 0, size * sizeof(float));
    for (int i = 0; i < 5; i++)
    {
        Perf perf("transpose_v5_16_16");
        dim3 block_size(16, 16);
        dim3 grid_size(((MATRIX_N)-1) / block_size.x + 1, ((MATRIX_M)-1) / block_size.y + 1);
        transpose_v5_shared_memory<<<grid_size, block_size>>>(input_device, output_device, MATRIX_M, MATRIX_N);
        cudaDeviceSynchronize();
    }
    cudaMemcpy(output_host_gpu_calc, output_device, size * sizeof(float), cudaMemcpyDeviceToHost);
    if (check(output_host_cpu_calc, output_host_gpu_calc, MATRIX_M, MATRIX_N))
    {
        std::cout << "right" << std::endl;
    }

    check_cuda_at_end("main");
    return 0;
}
