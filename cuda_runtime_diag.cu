#include <cstdio>
#include <cuda_runtime.h>

__global__ void noop_kernel(float *out)
{
    if (threadIdx.x == 0 && blockIdx.x == 0)
    {
        out[0] = 1.0f;
    }
}

int main()
{
    int device_count = -1;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    printf("cudaGetDeviceCount: %d (%s), count=%d\n", (int)err, cudaGetErrorString(err), device_count);

    float *device_ptr = nullptr;
    err = cudaMalloc(&device_ptr, sizeof(float));
    printf("cudaMalloc: %d (%s), ptr=%p\n", (int)err, cudaGetErrorString(err), (void *)device_ptr);

    if (err == cudaSuccess)
    {
        noop_kernel<<<1, 1>>>(device_ptr);
        err = cudaGetLastError();
        printf("kernel launch: %d (%s)\n", (int)err, cudaGetErrorString(err));

        err = cudaDeviceSynchronize();
        printf("cudaDeviceSynchronize: %d (%s)\n", (int)err, cudaGetErrorString(err));

        float host = 0.0f;
        err = cudaMemcpy(&host, device_ptr, sizeof(float), cudaMemcpyDeviceToHost);
        printf("cudaMemcpy D2H: %d (%s), host=%f\n", (int)err, cudaGetErrorString(err), host);
        cudaFree(device_ptr);
    }

    return 0;
}
