# 添加CUDA
add_executable(all_reduce_v7_v8 all_reduce_v7_v8.cu)
# 可执行文件链接
target_link_libraries(all_reduce_v7_v8 PRIVATE CUDA::cudart ${CUDA_cublas_LIBRARY})
# 下面的debug才会用到，release用不到
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(all_reduce_v7_v8 PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-G>)
endif()