#include <stdio.h>
#include <stdlib.h>
#include <iostream>

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
__global__ void transpose_naive(float *input, float *output, const int M, const int N)
{
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;
    int idx = y * N + x;
    int trans_idx = x * M + y;
    output[trans_idx] = input[idx];
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
    const int MATRIX_M = 2048;
    const int MATRIX_N = 512;
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
    cudaMemset(output_device, 0, size * sizeof(float));
    for (int i = 0; i < 5; i++)
    {
        Perf perf("transpose32_8");
        dim3 block_size(32, 8);
        dim3 grid_size(MATRIX_N / 32, MATRIX_M / 8);
        transpose_naive<<<grid_size, block_size>>>(input_device, output_device, MATRIX_M, MATRIX_N);
        cudaDeviceSynchronize();
    }
    cudaMemcpy(output_host_gpu_calc, output_device, size * sizeof(float), cudaMemcpyDeviceToHost);
    if (check(output_host_cpu_calc, output_host_gpu_calc, MATRIX_M, MATRIX_N))
    {
        std::cout << "right" << std::endl;
    }

    return 0;
}