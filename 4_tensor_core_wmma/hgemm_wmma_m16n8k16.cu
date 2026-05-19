#include "common/tester.h"
using namespace nvcuda;
__device__ __forceinline__ void ld_st_128bit(void *dst, void *src)
{
    *reinterpret_cast<float4 *>(dst) = *reinterpret_cast<float4 *>(src);
}
__global__ void shared_memory_wmma_kernel(half *A, half *B, half *C)
{
    __shared__ half smem_a[16 * 16];
    __shared__ half smem_b[16 * 16];
    __shared__ half smem_c[16 * 16];
    int tx = threadIdx.x;
    // 128bit相当于16字节，8个half
    ld_st_128bit(smem_a + 8 * tx, A + 8 * tx);
    ld_st_128bit(smem_b + 8 * tx, B + 8 * tx);
    __syncthreads();
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, half> c_frag;

    wmma::load_matrix_sync(a_frag, smem_a, 16);
    wmma::load_matrix_sync(b_frag, smem_b, 16);
    wmma::fill_fragment(c_frag, 0.0f);
    wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    wmma::store_matrix_sync(smem_c, c_frag, 16, wmma::mem_row_major);
    __syncthreads();
    ld_st_128bit(C + 8 * tx, smem_c + 8 * tx);
}

__global__ void wmma_simple_kernel(half *A, half *B, half *C)
{

    wmma::fragment<wmma::accumulator, 16, 16, 16, half> C_frag;
    wmma::fill_fragment(C_frag, 0.0);
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> A_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> B_frag;
    wmma::load_matrix_sync(A_frag, A, 16);
    wmma::load_matrix_sync(B_frag, B, 16);
    wmma::mma_sync(C_frag, A_frag, B_frag, C_frag);
    wmma::store_matrix_sync(C, C_frag, 16, wmma::mem_row_major);
}
void wmma_simple(half *A, half *B, half *C, int M, int N, int K)
{
    dim3 block(32);
    dim3 grid(1);
    wmma_simple_kernel<<<grid, block>>>(A, B, C);
}
int main(int argc, char *argv[])
{
    // 前三个参数:mnk矩阵乘法,最后的true是做结果正确性对比
    Tester tester(16, 16, 16, 1, 10, 100, true);
    tester.evaluate(wmma_simple, "wmma_simple");
    return 0;
}
