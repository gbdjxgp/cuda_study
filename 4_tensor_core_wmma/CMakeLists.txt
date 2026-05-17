# 添加CUDA
add_executable(hgemm_v1_wmma_m16n16k16_naive_kernel hgemm_v1_wmma_m16n16k16_naive_kernel.cu)
# This translation unit contains cp.async PTX, which is only supported on sm_80+.
set_property(TARGET hgemm_v1_wmma_m16n16k16_naive_kernel PROPERTY CUDA_ARCHITECTURES 80;86;89;90)
# 可执行文件链接
target_link_libraries(hgemm_v1_wmma_m16n16k16_naive_kernel PRIVATE CUDA::cudart ${CUDA_cublas_LIBRARY})
# 下面的debug才会用到，release用不到
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(hgemm_v1_wmma_m16n16k16_naive_kernel PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-G>)
endif()
target_compile_options(hgemm_v1_wmma_m16n16k16_naive_kernel PRIVATE -lineinfo)
