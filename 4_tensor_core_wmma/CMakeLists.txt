# 添加CUDA
add_executable(hgemm_wmma_m16n16k16 hgemm_wmma_m16n16k16.cu)
# This translation unit contains cp.async PTX, which is only supported on sm_80+.
set_property(TARGET hgemm_wmma_m16n16k16 PROPERTY CUDA_ARCHITECTURES 80;86;89;90)
# 可执行文件链接
target_link_libraries(hgemm_wmma_m16n16k16 PRIVATE CUDA::cudart ${CUDA_cublas_LIBRARY})
# 下面的debug才会用到，release用不到
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(hgemm_wmma_m16n16k16 PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-G>)
endif()
target_compile_options(hgemm_wmma_m16n16k16 PRIVATE -lineinfo)


# 添加CUDA
add_executable(hgemm_wmma_m16n8k16 hgemm_wmma_m16n8k16.cu)
# This translation unit contains cp.async PTX, which is only supported on sm_80+.
set_property(TARGET hgemm_wmma_m16n8k16 PROPERTY CUDA_ARCHITECTURES 80;86;89;90)
# 可执行文件链接
target_link_libraries(hgemm_wmma_m16n8k16 PRIVATE CUDA::cudart ${CUDA_cublas_LIBRARY})
# 下面的debug才会用到，release用不到
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(hgemm_wmma_m16n8k16 PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-G>)
endif()
target_compile_options(hgemm_wmma_m16n8k16 PRIVATE -lineinfo)
