# 添加CUDA
add_executable(sgemm_v0_v8 sgemm_v0_v8.cu)
# 可执行文件链接
target_link_libraries(sgemm_v0_v8 PRIVATE CUDA::cudart ${CUDA_cublas_LIBRARY})
# 下面的debug才会用到，release用不到
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(sgemm_v0_v8 PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-G>)
endif()