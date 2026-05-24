#include "common/tester.h"
using namespace nvcuda;

template <auto Kernel>
void wrapper(half *A, half *B, half *C, int M, int N, int K)
{
    dim3 block(32);
    dim3 grid(1);
    Kernel<<<grid, block>>>(A, B, C);
    HGEMM_CHECK_CUDART_ERROR(cudaGetLastError());
}
template <uint32_t S, uint32_t B, uint32_t M>
__device__ __forceinline__ uint32_t swizzle(uint32_t addr)
{
    uint32_t BMask = (1 << B - 1) << M;
    return ((addr >> S) & BMask) ^ addr;
}
__device__ __forceinline__ void ld_st_128bit(void *dst, void *src)
{
    *reinterpret_cast<float4 *>(dst) = *reinterpret_cast<float4 *>(src);
}

__device__ __forceinline__ uint32_t ld_st_32bit(void *src)
{
    return *reinterpret_cast<uint32_t *>(src);
}

__device__ __forceinline__ void st_ld_32bit(void *dst, uint32_t value)
{
    *reinterpret_cast<uint32_t *>(dst) = value;
}

__global__ void v1_simple_wmma(half *A, half *B, half *C)
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
__global__ void v2_shared_memory_wmma(half *A, half *B, half *C)
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
    // B is stored/interpreted as col-major throughout this test path.
    // Copying it into shared memory preserves that layout, so the WMMA
    // fragment declaration must stay col_major here as well.
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, half> c_frag;

    wmma::load_matrix_sync(a_frag, smem_a, 16);
    wmma::load_matrix_sync(b_frag, smem_b, 16);
    wmma::fill_fragment(c_frag, 0.0f);
    wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    wmma::store_matrix_sync(smem_c, c_frag, 16, wmma::mem_row_major);
    __syncthreads();
    ld_st_128bit(C + 8 * tx, smem_c + 8 * tx);
}

__global__ void v3_shared_memory_wmma_padding(half *A, half *B, half *C)
{
    __shared__ half smem_a[16][16 + 8];
    __shared__ half smem_b[16][16 + 8];
    __shared__ half smem_c[16 * 16];
    int tx = threadIdx.x;
    int smem_thread_start = tx * 8;
    // 128bit相当于16字节，8个half
    ld_st_128bit(&smem_a[smem_thread_start / 16][smem_thread_start % 16], A + smem_thread_start);
    ld_st_128bit(&smem_b[smem_thread_start / 16][smem_thread_start % 16], B + smem_thread_start);
    __syncthreads();
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
    // B is stored/interpreted as col-major throughout this test path.
    // Copying it into shared memory preserves that layout, so the WMMA
    // fragment declaration must stay col_major here as well.
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, half> c_frag;

    wmma::load_matrix_sync(a_frag, smem_a[0], 16 + 8);
    wmma::load_matrix_sync(b_frag, smem_b[0], 16 + 8);
    wmma::fill_fragment(c_frag, 0.0f);
    wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    wmma::store_matrix_sync(smem_c, c_frag, 16, wmma::mem_row_major);
    __syncthreads();
    ld_st_128bit(C + 8 * tx, smem_c + 8 * tx);
}

__global__ void v4_shared_memory_mma(half *A, half *B, half *C)
{
    __shared__ half smem_a[16 * 16];
    __shared__ half smem_b[16 * 16];
    __shared__ half smem_c[16 * 16];
    int tx = threadIdx.x;
    // 128bit相当于16字节，8个half
    ld_st_128bit(smem_a + 8 * tx, A + 8 * tx);
    ld_st_128bit(smem_b + 8 * tx, B + 8 * tx);
    __syncthreads();
    uint32_t RA[4];
    uint32_t RB[4];
    uint32_t RC[4] = {0x0};
    // 一共32个线程，4个线程为1组，大概这样(group_id,thread_id_in_group)
    // group_id:    0 0 0 0 1 1 1 1 2 2 2 2 3 3 3 3 4 4 4 4 ... 7 7 7 7
    // in_group_id: 0 1 2 3 0 1 2 3 0 1 2 3 0 1 2 3 0 1 2 3 ... .......
    // 8行
    uint32_t group_id = tx / 4;
    // 每行4列
    uint32_t thread_id_in_group = tx % 4;

    uint32_t row = tx % 16;
    uint32_t col = tx / 16;
    uint32_t addr_a = __cvta_generic_to_shared(smem_a + row * 16 + col * 8);
    LDMATRIX_X4(RA[0], RA[1], RA[2], RA[3], addr_a);
    uint32_t addr_b = __cvta_generic_to_shared(smem_b + row * 16 + col * 8);
    // 注意：B矩阵左半边对应的是RB[0]和RB[2]
    LDMATRIX_X4(RB[0], RB[1], RB[2], RB[3], addr_b);

    HMMA16816(RC[0], RC[1], RA[0], RA[1], RA[2], RA[3], RB[0], RB[2], RC[0], RC[1]);
    HMMA16816(RC[2], RC[3], RA[0], RA[1], RA[2], RA[3], RB[1], RB[3], RC[2], RC[3]);
    // uint32_t addr_c = __cvta_generic_to_shared(smem_c + row * 16 + col * 8);
    // asm volatile("stmatrix.sync.aligned.x4.m8n8.shared.b16 [%0], {%1, %2, %3, %4};\n" ::"r"(addr_c), "r"(RC[0]), "r"(RC[1]), "r"(RC[2]), "r"(RC[3]));
    // 这里thread_id_in_group * 2表示每列2个数，对应32位的寄存器。
    // smem_c是half类型的，因此每+1相当于2字节，group_id+8则意味着C矩阵的下半部分。
    st_ld_32bit(smem_c + group_id * 16 + thread_id_in_group * 2, RC[0]);
    st_ld_32bit(smem_c + (group_id + 8) * 16 + thread_id_in_group * 2, RC[1]);
    st_ld_32bit(smem_c + group_id * 16 + thread_id_in_group * 2 + 8, RC[2]);
    st_ld_32bit(smem_c + (group_id + 8) * 16 + thread_id_in_group * 2 + 8, RC[3]);
    __syncthreads();
    ld_st_128bit(C + 8 * tx, smem_c + 8 * tx);
}

__global__ void v5_shared_memory_mma_swizzle(half *A, half *B, half *C)
{
    __shared__ half smem_a[16 * 16];
    __shared__ half smem_b[16 * 16];
    __shared__ half smem_c[16 * 16];
    int tx = threadIdx.x;
    // 128bit相当于16字节，8个half
    int gAddr = 8 * tx;
    int g2sAddr = swizzle<3, 1, 3>(gAddr);

    ld_st_128bit(smem_a + g2sAddr, A + gAddr);
    ld_st_128bit(smem_b + g2sAddr, B + gAddr);
    __syncthreads();
    uint32_t RA[4];
    uint32_t RB[4];
    uint32_t RC[4] = {0x0};
    uint32_t group_id = tx / 4;
    uint32_t thread_id_in_group = tx % 4;

    uint32_t row = tx % 16;
    uint32_t col = tx / 16;
    uint32_t sAddr = row * 16 + col * 8;
    uint32_t s2fAddr = swizzle<3, 1, 3>(sAddr);
    uint32_t addr_a = __cvta_generic_to_shared(smem_a + s2fAddr);
    LDMATRIX_X4(RA[0], RA[1], RA[2], RA[3], addr_a);
    uint32_t addr_b = __cvta_generic_to_shared(smem_b + s2fAddr);
    LDMATRIX_X4(RB[0], RB[1], RB[2], RB[3], addr_b);

    HMMA16816(RC[0], RC[1], RA[0], RA[1], RA[2], RA[3], RB[0], RB[2], RC[0], RC[1]);
    HMMA16816(RC[2], RC[3], RA[0], RA[1], RA[2], RA[3], RB[1], RB[3], RC[2], RC[3]);
    // uint32_t addr_c = __cvta_generic_to_shared(smem_c + row * 16 + col * 8);
    // asm volatile("stmatrix.sync.aligned.x4.m8n8.shared.b16 [%0], {%1, %2, %3, %4};\n" ::"r"(addr_c), "r"(RC[0]), "r"(RC[1]), "r"(RC[2]), "r"(RC[3]));
    // 这里thread_id_in_group * 2表示每列2个数，对应32位的寄存器。
    // smem_c是half类型的，因此每+1相当于2字节，group_id+8则意味着C矩阵的下半部分。
    uint32_t swizzle_c0 = swizzle<3, 1, 3>(group_id * 16 + thread_id_in_group * 2);
    uint32_t swizzle_c1 = swizzle<3, 1, 3>((group_id + 8) * 16 + thread_id_in_group * 2);
    uint32_t swizzle_c2 = swizzle<3, 1, 3>(group_id * 16 + thread_id_in_group * 2 + 8);
    uint32_t swizzle_c3 = swizzle<3, 1, 3>((group_id + 8) * 16 + thread_id_in_group * 2 + 8);
    st_ld_32bit(smem_c + swizzle_c0, RC[0]);
    st_ld_32bit(smem_c + swizzle_c1, RC[1]);
    st_ld_32bit(smem_c + swizzle_c2, RC[2]);
    st_ld_32bit(smem_c + swizzle_c3, RC[3]);
    __syncthreads();

    ld_st_128bit(C + gAddr, smem_c + g2sAddr);
}

int main(int argc, char *argv[])
{
    // 前三个参数:mnk矩阵乘法,最后的true是做结果正确性对比
    Tester tester(16, 16, 16, 1, 10, 100, true);
    // tester.evaluate(wrapper<v1_simple_wmma>, "v1_simple_wmma");
    // tester.evaluate(wrapper<v2_shared_memory_wmma>, "v2_shared_memory_wmma");
    // tester.evaluate(wrapper<v3_shared_memory_wmma_padding>, "v3_shared_memory_wmma_padding");
    tester.evaluate(wrapper<v4_shared_memory_mma>, "v4_shared_memory_mma");
    tester.evaluate(wrapper<v5_shared_memory_mma_swizzle>, "v5_shared_memory_mma_swizzle");
    return 0;
}
