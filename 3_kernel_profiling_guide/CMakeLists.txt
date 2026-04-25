# 添加CUDA
add_executable(combine_access combine_access.cu)
# 可执行文件链接
target_link_libraries(combine_access PRIVATE CUDA::cudart ${CUDA_cublas_LIBRARY})
# 下面的debug才会用到，release用不到
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(combine_access PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-G>)
endif()
target_compile_options(combine_access PRIVATE -lineinfo)



# 添加CUDA
add_executable(my_transpose my_transpose.cu)
# 可执行文件链接
target_link_libraries(my_transpose PRIVATE CUDA::cudart ${CUDA_cublas_LIBRARY})
# 下面的debug才会用到，release用不到
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(my_transpose PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-G>)
endif()
target_compile_options(my_transpose PRIVATE -lineinfo)