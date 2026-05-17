#include "common/tester.h"
using namespace nvcuda;
template <const int WMMA_M = 16, const int WMMA_N = 16, const int WMMA_K = 16>
__global__ void hgemm_wmma_m16n16k16_naive_kernel(half *A, half *B, half *C, int M, int N, int K)
{
    const int NUM_K_TILES = div_ceil(K, WMMA_K);
    const int load_gmem_a_m = blockIdx.y * WMMA_M;
    const int load_gmem_b_n = blockIdx.x * WMMA_N;
    if (load_gmem_a_m >= M || load_gmem_b_n >= N)
    {
        return;
    }
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, half> C_frag;
    wmma::fill_fragment(C_frag, 0.0);
#pragma unroll
    for (int k = 0; k < NUM_K_TILES; ++k)
    {
        wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> A_frag;
        wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> B_frag;
        wmma::load_matrix_sync(A_frag, A + load_gmem_a_m * K + k * WMMA_K, K);
        wmma::load_matrix_sync(B_frag, B + load_gmem_b_n * K + k * WMMA_K, K);
        wmma::mma_sync(C_frag, A_frag, B_frag, C_frag);
        // __syncthreads();
    }
    wmma::store_matrix_sync(C + load_gmem_a_m * N + load_gmem_b_n, C_frag, N, wmma::mem_row_major);
}

void hgemm_wmma_m16n16k16_naive(half *A, half *B, half *C, int M, int N, int K)
{
    const int WMMA_M = 16;
    const int WMMA_N = 16;
    const int WMMA_K = 16;
    dim3 block(32);
    dim3 grid(div_ceil(N, WMMA_N), div_ceil(M, WMMA_M));
    hgemm_wmma_m16n16k16_naive_kernel<WMMA_M, WMMA_N, WMMA_K><<<grid, block>>>(A, B, C, M, N, K);
}
#define FETCH_HALF4 FETCH_FLOAT2
#define FETCH_HALF2 FETCH_FLOAT
#define FETCH_FLOAT2(x) (reinterpret_cast<float2 *>(&(x))[0])
#define FETCH_FLOAT(x) (reinterpret_cast<float *>(&(x))[0])
template <const int WMMA_M = 16, const int WMMA_N = 16, const int WMMA_K = 16, const int WMMA_TILE_M = 4, const int WMMA_TILE_N = 2>
__global__ void hgemm_wmma_m6n16k16_mma4x2_kernel(half *A, half *B, half *C, int M, int N, int K)
{
    // 每个block的起始索引,一个block处理的跨度是(64,32)个元素
    int block_start_x = WMMA_N * WMMA_TILE_N * blockIdx.x;
    int block_start_y = WMMA_M * WMMA_TILE_M * blockIdx.y;

    __shared__ half shared_M[WMMA_M * WMMA_TILE_M][WMMA_K];
    // 不要看shared_N的索引,否则你会晕头转向!一定手动计算偏移,然后你就知道为什么这么定义了
    // 这里重要的是shared_N的元素个数对上就没问题,甚至你手动计算一维数组的偏移都更清晰
    __shared__ half shared_N[WMMA_N * WMMA_TILE_N][WMMA_K];
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, half> C_frag;
    wmma::fill_fragment(C_frag, 0.);
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> A_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> B_frag;
    // int NUM_K_TILES = div_ceil(K, WMMA_K);
    // 计算&C矩阵写回要用到!
    int warp_id = threadIdx.x / warpSize;
    int tile_n = (warp_id % WMMA_TILE_N) * WMMA_N;
    int tile_m = (warp_id / WMMA_TILE_N) * WMMA_M;
    for (int k = 0; k < K; k += WMMA_K)
    {
        // smem中a是16*4行,16列,b是16行,2*16列,一个warp负责smem_a中16行以及smem_b中16列的数据运算,因此一个block需要4*2=8个warp
        // 但是搬运的时候需要注意:搬运的数据量A中是16*4行,16列,B中是16行,16*2列.因此一个block256线程需要重新排布
        // 我这里设计的是每个线程都干活,并且都负责搬运数据,每个线程负责搬运a中的4个数据以及b中的2个数据.
        // 重排后的thread_start_a_x/thread_start_a_y/thread_start_b_x/thread_start_b_y
        int offset_a_y = threadIdx.x / 4;
        int offset_a_x = (threadIdx.x % 4) * 4;
        // 由于b矩阵是col-major,因此,我们按照连续的列来取元素.
        // smem_b是16行32列,每次搬运2个数据,y(行)方向就是16%2=8个thread
        int offset_b_y = (threadIdx.x % 8) * 2;
        int offset_b_x = threadIdx.x / 8;
        // 计算整体的偏移
        int gmem_a_y = block_start_y + offset_a_y;
        int gmem_a_x = k + offset_a_x;
        int gmem_b_y = k + offset_b_y;
        int gmem_b_x = block_start_x + offset_b_x;
        // 开始搬运数据
        FETCH_HALF4(shared_M[offset_a_y][offset_a_x]) = FETCH_HALF4(A[gmem_a_y * K + gmem_a_x]);
        // 手动计算偏移:全局取得地址是B[gmem_b_x * K + gmem_b_y]
        // 偏移逻辑坐标(行-列)为[gmem_b_y,gmem_b_x]
        // shared_N也是col-major,本地偏移也就是:[offset_b_x*WMMA_K+offset_b_y]
        // 由于shared_N二维数组是row-major且列数是WMMA_K,等效写法是shared_N[offset_b_x][offset_b_y]
        FETCH_HALF2(shared_N[offset_b_x][offset_b_y]) = FETCH_HALF2(B[gmem_b_x * K + gmem_b_y]);
        __syncthreads();
        wmma::load_matrix_sync(A_frag, &(shared_M[tile_m][0]), WMMA_K);
        // 这里同样,shared_N的leading-dimension就是WMMA_K,针对col-major的矩阵,其leading-dimension为矩阵的行数(16)
        // 取元素应该是shared_N第tile_n列,第0行的地址,在当前row-major的数组上也正好就是shared_N[tile_n][0]
        wmma::load_matrix_sync(B_frag, &(shared_N[tile_n][0]), WMMA_K);
        wmma::mma_sync(C_frag, A_frag, B_frag, C_frag);
        __syncthreads();
    }
    // 存储的时候需要额外计算一下warp的偏移
    int gmem_c_y = block_start_y + tile_m;
    int gmem_c_x = block_start_x + tile_n;
    wmma::store_matrix_sync(C + gmem_c_y * N + gmem_c_x, C_frag, N, wmma::mem_row_major);
}
#define WARP_SIZE 32
#define LDST32BITS(value) (reinterpret_cast<half2 *>(&(value))[0])
#define LDST64BITS(value) (reinterpret_cast<float2 *>(&(value))[0])
template <const int WMMA_M = 16, const int WMMA_N = 16, const int WMMA_K = 16, const int WMMA_TILE_M = 4, const int WMMA_TILE_N = 2>
__global__ void hgemm_wmma_m16n16k16_mma4x2_kernel_orig(half *A, half *B, half *C, int M, int N, int K)
{
    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    const int NUM_K_TILES = div_ceil(K, WMMA_K);
    constexpr int BM = WMMA_M * WMMA_TILE_M;
    constexpr int BN = WMMA_N * WMMA_TILE_N;
    constexpr int BK = WMMA_K;
    __shared__ half s_a[BM][BK], s_b[WMMA_K][BN];
    const int tid = threadIdx.y * blockDim.x + threadIdx.x;
    const int warp_id = tid / WARP_SIZE;
    const int lane_id = tid % WARP_SIZE;
    const int warp_m = warp_id / 2;
    const int warp_n = warp_id % 2;

    const int load_smem_a_m = tid / 4;
    const int load_smem_a_k = (tid % 4) * 4;

    const int load_smem_b_k = tid / 16;
    const int load_smem_b_n = (tid % 16) * 2;
    const int load_gmem_a_m = by * BM + load_smem_a_m;
    const int load_gmem_b_n = bx * BN + load_smem_b_n;

    if (load_gmem_a_m >= M && load_gmem_b_n >= N)
    {
        return;
    }
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, half> C_frag;
    wmma::fill_fragment(C_frag, 0.0f);

    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> A_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> B_frag;
#pragma unroll
    for (int k = 0; k < NUM_K_TILES; ++k)
    {
        int load_gmem_a_k = k * WMMA_K + load_smem_a_k;
        int load_gmem_a_addr = load_gmem_a_m * K + load_gmem_a_k;
        // int load_gmem_b_k = k * WMMA_K + load_smem_b_k;
        // int load_gmem_b_addr = load_gmem_b_k * N + load_gmem_b_n;
        int load_gmem_b_k = k * WMMA_K + load_smem_b_k;
        int load_gmem_b_n0 = bx * BN + load_smem_b_n;
        int load_gmem_b_n1 = bx * BN + load_smem_b_n + 1;
        s_b[load_smem_b_k][load_smem_b_n] = B[load_gmem_b_n0 * K + load_gmem_b_k];
        s_b[load_smem_b_k][load_smem_b_n + 1] = B[load_gmem_b_n1 * K + load_gmem_b_k];
        LDST64BITS(s_a[load_smem_a_m][load_smem_a_k]) = LDST64BITS(A[load_gmem_a_addr]);
        // LDST32BITS(s_b[load_smem_b_k][load_smem_b_n]) = LDST32BITS(B[load_gmem_b_addr]);
        __syncthreads();
        wmma::load_matrix_sync(A_frag, &s_a[warp_m * WMMA_M][0], BK);
        wmma::load_matrix_sync(B_frag, &s_b[0][warp_n * WMMA_N], BN);
        wmma::mma_sync(C_frag, A_frag, B_frag, C_frag);
        __syncthreads();
    }
    const int store_gmem_a_m = by * BM + warp_m * WMMA_M;
    const int store_gmem_a_n = bx * BN + warp_n * WMMA_N;
    wmma::store_matrix_sync(C + store_gmem_a_m * N + store_gmem_a_n, C_frag, N,
                            wmma::mem_row_major);
}
void hgemm_wmma_m6n16k16_mma4x2(half *A, half *B, half *C, int M, int N, int K)
{
    const int WMMA_M = 16;
    const int WMMA_N = 16;
    const int WMMA_K = 16;
    const int WMMA_TILE_M = 4;
    const int WMMA_TILE_N = 2;
    dim3 block(256);
    dim3 grid(div_ceil(N, WMMA_N * WMMA_TILE_N), div_ceil(M, WMMA_M * WMMA_TILE_M));
    // hgemm_wmma_m6n16k16_mma4x2_kernel_orig<WMMA_M, WMMA_N, WMMA_K, WMMA_TILE_M, WMMA_TILE_N><<<grid, block>>>(A, B, C, M, N, K);
    hgemm_wmma_m6n16k16_mma4x2_kernel<WMMA_M, WMMA_N, WMMA_K, WMMA_TILE_M, WMMA_TILE_N><<<grid, block>>>(A, B, C, M, N, K);
}

#define FETCH_HALF8 FETCH_FLOAT4
#define FETCH_FLOAT4(x) (reinterpret_cast<float4 *>(&(x))[0])
template <
    const int WMMA_M = 16,
    const int WMMA_N = 16,
    const int WMMA_K = 16,
    const int WMMA_TILE_M = 4,
    const int WMMA_TILE_N = 2,
    const int WARP_TILE_M = 2,
    const int WARP_TILE_N = 4>
__global__ void hgemm_wmma_m16n16k16_mma4x2_warp2x4_kernel(half *A, half *B, half *C, int M, int N, int K)
{
    const int size_per_m = WMMA_M * WMMA_TILE_M * WARP_TILE_M;
    const int size_per_n = WMMA_N * WMMA_TILE_N * WARP_TILE_N;
    int block_offset_m = size_per_m * blockIdx.y;
    int block_offset_n = size_per_n * blockIdx.x;
    int warp_id = threadIdx.x / WARP_SIZE;
    // int lane_id = threadIdx.x % WARP_SIZE;
    int warp_offset_m = WMMA_M * WARP_TILE_M * (warp_id / 2);
    int warp_offset_n = WMMA_N * WARP_TILE_N * (warp_id % 2);
    __shared__ half shared_M[size_per_m][WMMA_K];
    __shared__ half shared_N[size_per_n][WMMA_K];
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, half> C_frag[WARP_TILE_M * WARP_TILE_N];
    for (int i = 0; i < WARP_TILE_M * WARP_TILE_N; i++)
    {
        wmma::fill_fragment(C_frag[i], 0.);
    }
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> A_frag[WARP_TILE_M];
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> B_frag[WARP_TILE_N];
    for (int k = 0; k < K; k += WMMA_K)
    {
        // 搬运，这次一个block要搬运的数据量就是128*16以及16*128，针对AB矩阵，正好一个线程搬运一行16个数，只需要搬运一次就全部搬运完毕
        // 首先计算偏移
        int smem_offset_m = threadIdx.x / 2;
        int smem_offset_k = (threadIdx.x % 2) * 8;
        int smem_offset_n = threadIdx.x / 2;
        // 根据偏移写数据
        FETCH_HALF8(shared_M[smem_offset_m][smem_offset_k]) = FETCH_HALF8(A[(block_offset_m + smem_offset_m) * K + k + smem_offset_k]);
        FETCH_HALF8(shared_N[smem_offset_n][smem_offset_k]) = FETCH_HALF8(B[(block_offset_n + smem_offset_n) * K + k + smem_offset_k]);
        // 内部调用循环进行WMMA
        __syncthreads();

        for (int i = 0; i < WARP_TILE_M; i++)
        {
            wmma::load_matrix_sync(A_frag[i], &(shared_M[warp_offset_m + i * WMMA_M][0]), WMMA_K);
        }
        for (int j = 0; j < WARP_TILE_N; j++)
        {

            wmma::load_matrix_sync(B_frag[j], &(shared_N[warp_offset_n + j * WMMA_N][0]), WMMA_K);
        }
        for (int i = 0; i < WARP_TILE_M; i++)
        {
            for (int j = 0; j < WARP_TILE_N; j++)
            {

                wmma::mma_sync(C_frag[i * WARP_TILE_N + j], A_frag[i], B_frag[j], C_frag[i * WARP_TILE_N + j]);
            }
        }
        __syncthreads();
    }
    // 循环调用保存
    for (int i = 0; i < WARP_TILE_M; i++)
    {
        for (int j = 0; j < WARP_TILE_N; j++)
        {
            int c_offset_m = block_offset_m + warp_offset_m + i * WMMA_M;
            int c_offset_n = block_offset_n + warp_offset_n + j * WMMA_N;
            wmma::store_matrix_sync(C + c_offset_m * N + c_offset_n, C_frag[i * WARP_TILE_N + j], N, wmma::mem_row_major);
        }
    }
}

void hgemm_wmma_m16n16k16_mma4x2_warp2x4(half *A, half *B, half *C, int M, int N, int K)
{
    const int WMMA_M = 16;
    const int WMMA_N = 16;
    const int WMMA_K = 16;
    const int WMMA_TILE_M = 4;
    const int WMMA_TILE_N = 2;
    const int WARP_TILE_M = 2;
    const int WARP_TILE_N = 4;
    dim3 block(256);
    dim3 grid(div_ceil(N, WMMA_N * WMMA_TILE_N * WARP_TILE_N), div_ceil(M, WMMA_M * WMMA_TILE_M * WARP_TILE_M));
    hgemm_wmma_m16n16k16_mma4x2_warp2x4_kernel<WMMA_M, WMMA_N, WMMA_K, WMMA_TILE_M, WMMA_TILE_N, WARP_TILE_M, WARP_TILE_N><<<grid, block>>>(A, B, C, M, N, K);
}

template <
    const int WMMA_M = 16,
    const int WMMA_N = 16,
    const int WMMA_K = 16,
    const int WMMA_TILE_M = 4,
    const int WMMA_TILE_N = 2,
    const int WARP_TILE_M = 2,
    const int WARP_TILE_N = 4>
__global__ void hgemm_v4_wmma_m16n16k16_mma4x2_Warp2x4_dbuf_async_kernel(half *A, half *B, half *C, int M, int N, int K)
{
    const int size_per_m = WMMA_M * WMMA_TILE_M * WARP_TILE_M;
    const int size_per_n = WMMA_N * WMMA_TILE_N * WARP_TILE_N;
    int block_offset_m = size_per_m * blockIdx.y;
    int block_offset_n = size_per_n * blockIdx.x;
    int warp_id = threadIdx.x / WARP_SIZE;
    // int lane_id = threadIdx.x % WARP_SIZE;
    int warp_offset_m = WMMA_M * WARP_TILE_M * (warp_id / 2);
    int warp_offset_n = WMMA_N * WARP_TILE_N * (warp_id % 2);
    __shared__ half shared_M[2][size_per_m][WMMA_K];
    __shared__ half shared_N[2][size_per_n][WMMA_K];
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, half> C_frag[WARP_TILE_M * WARP_TILE_N];
    for (int i = 0; i < WARP_TILE_M * WARP_TILE_N; i++)
    {
        wmma::fill_fragment(C_frag[i], 0.);
    }
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> A_frag[WARP_TILE_M];
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> B_frag[WARP_TILE_N];
    // 流水线的装载！
    // 搬运，这次一个block要搬运的数据量就是128*16以及16*128，针对AB矩阵，正好一个线程搬运一行16个数，只需要搬运一次就全部搬运完毕
    // 首先计算偏移
    int smem_offset_m = threadIdx.x / 2;
    int smem_offset_k = (threadIdx.x % 2) * 8;
    int smem_offset_n = threadIdx.x / 2;
    // 根据偏移写数据
    FETCH_HALF8(shared_M[0][smem_offset_m][smem_offset_k]) = FETCH_HALF8(A[(block_offset_m + smem_offset_m) * K + 0 + smem_offset_k]);
    FETCH_HALF8(shared_N[0][smem_offset_n][smem_offset_k]) = FETCH_HALF8(B[(block_offset_n + smem_offset_n) * K + 0 + smem_offset_k]);
    __syncthreads();
    // 流水线装载完毕
    int write_stage_idx = 1;
    for (int k = WMMA_K; k < K; k += WMMA_K)
    {
        // 搬运，这次一个block要搬运的数据量就是128*16以及16*128，针对AB矩阵，正好一个线程搬运一行16个数，只需要搬运一次就全部搬运完毕
        // 首先计算偏移
        int smem_offset_m = threadIdx.x / 2;
        int smem_offset_k = (threadIdx.x % 2) * 8;
        int smem_offset_n = threadIdx.x / 2;
        // 根据偏移写数据
        // FETCH_HALF8(shared_M[write_stage_idx][smem_offset_m][smem_offset_k]) = FETCH_HALF8(A[(block_offset_m + smem_offset_m) * K + k + smem_offset_k]);
        // FETCH_HALF8(shared_N[write_stage_idx][smem_offset_n][smem_offset_k]) = FETCH_HALF8(B[(block_offset_n + smem_offset_n) * K + k + smem_offset_k]);
        // 使用async写数据
        uint32_t load_shared_M_ptr = __cvta_generic_to_shared(&shared_M[write_stage_idx][smem_offset_m][smem_offset_k]);
        uint32_t load_shared_N_ptr = __cvta_generic_to_shared(&shared_N[write_stage_idx][smem_offset_n][smem_offset_k]);
        CP_ASYNC_CG(load_shared_M_ptr, &A[(block_offset_m + smem_offset_m) * K + k + smem_offset_k], 16);
        CP_ASYNC_CG(load_shared_N_ptr, &B[(block_offset_n + smem_offset_n) * K + k + smem_offset_k], 16);
        CP_ASYNC_COMMIT_GROUP();
        // 内部调用循环进行WMMA
        write_stage_idx ^= 1;
        for (int i = 0; i < WARP_TILE_M; i++)
        {
            wmma::load_matrix_sync(A_frag[i], &(shared_M[write_stage_idx][warp_offset_m + i * WMMA_M][0]), WMMA_K);
        }
        for (int j = 0; j < WARP_TILE_N; j++)
        {

            wmma::load_matrix_sync(B_frag[j], &(shared_N[write_stage_idx][warp_offset_n + j * WMMA_N][0]), WMMA_K);
        }
        for (int i = 0; i < WARP_TILE_M; i++)
        {
            for (int j = 0; j < WARP_TILE_N; j++)
            {

                wmma::mma_sync(C_frag[i * WARP_TILE_N + j], A_frag[i], B_frag[j], C_frag[i * WARP_TILE_N + j]);
            }
        }
        CP_ASYNC_WAIT_GROUP(0);
        __syncthreads();
    }
    // 流水线排空
    write_stage_idx ^= 1;

    for (int i = 0; i < WARP_TILE_M; i++)
    {
        wmma::load_matrix_sync(A_frag[i], &(shared_M[write_stage_idx][warp_offset_m + i * WMMA_M][0]), WMMA_K);
    }
    for (int j = 0; j < WARP_TILE_N; j++)
    {

        wmma::load_matrix_sync(B_frag[j], &(shared_N[write_stage_idx][warp_offset_n + j * WMMA_N][0]), WMMA_K);
    }
    for (int i = 0; i < WARP_TILE_M; i++)
    {
        for (int j = 0; j < WARP_TILE_N; j++)
        {

            wmma::mma_sync(C_frag[i * WARP_TILE_N + j], A_frag[i], B_frag[j], C_frag[i * WARP_TILE_N + j]);
        }
    }
    __syncthreads();
    // 流水线排空完毕
    // 循环调用保存
    for (int i = 0; i < WARP_TILE_M; i++)
    {
        for (int j = 0; j < WARP_TILE_N; j++)
        {
            int c_offset_m = block_offset_m + warp_offset_m + i * WMMA_M;
            int c_offset_n = block_offset_n + warp_offset_n + j * WMMA_N;
            wmma::store_matrix_sync(C + c_offset_m * N + c_offset_n, C_frag[i * WARP_TILE_N + j], N, wmma::mem_row_major);
        }
    }
}

void hgemm_v4_wmma_m16n16k16_mma4x2_Warp2x4_dbuf_async(half *A, half *B, half *C, int M, int N, int K)
{
    const int WMMA_M = 16;
    const int WMMA_N = 16;
    const int WMMA_K = 16;
    const int WMMA_TILE_M = 4;
    const int WMMA_TILE_N = 2;
    const int WARP_TILE_M = 2;
    const int WARP_TILE_N = 4;
    dim3 block(256);
    dim3 grid(div_ceil(N, WMMA_N * WMMA_TILE_N * WARP_TILE_N), div_ceil(M, WMMA_M * WMMA_TILE_M * WARP_TILE_M));
    hgemm_v4_wmma_m16n16k16_mma4x2_Warp2x4_dbuf_async_kernel<WMMA_M, WMMA_N, WMMA_K, WMMA_TILE_M, WMMA_TILE_N, WARP_TILE_M, WARP_TILE_N><<<grid, block>>>(A, B, C, M, N, K);
}
int main(int argc, char *argv[])
{
    // 前三个参数:mnk矩阵乘法,最后的true是做结果正确性对比
    Tester tester(512, 2048, 1024, 1, 10, 100, true);
    // tester.evaluate(hgemm_wmma_m16n16k16_naive, "hgemm_wmma_m16n16k16_naive");
    // tester.evaluate(hgemm_wmma_m16n16k16_mma4x2, "hgemm_wmma_m6n16k16_mma4x2");
    // tester.evaluate(hgemm_wmma_m16n16k16_mma4x2_warp2x4, "hgemm_wmma_m16n16k16_mma4x2_warp2x4");
    tester.evaluate(hgemm_v4_wmma_m16n16k16_mma4x2_Warp2x4_dbuf_async, "hgemm_v4_wmma_m16n16k16_mma4x2_Warp2x4_dbuf_async");
    return 0;
}
