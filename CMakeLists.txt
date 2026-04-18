cmake_minimum_required(VERSION 3.20.0)
project(cuda_learning VERSION 0.1.0 LANGUAGES CXX CUDA C)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES OR CMAKE_CUDA_ARCHITECTURES STREQUAL "52")
    # Avoid nvcc's legacy sm_52 default. On newer GPUs that forces PTX JIT,
    # which can fail on machines whose driver is older than the installed toolkit.
    set(CMAKE_CUDA_ARCHITECTURES "70;75;80;86;89;90" CACHE STRING "CUDA architectures to build for" FORCE)
endif()

find_package(CUDAToolkit REQUIRED)
add_executable(cuda_runtime_diag cuda_runtime_diag.cu)
target_link_libraries(cuda_runtime_diag PRIVATE CUDA::cudart)
add_subdirectory(1_cuda_reduce_study)
add_subdirectory(2_cuda_sgemm_study)
add_subdirectory(3_kernel_profiling_guide)

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(cuda_runtime_diag PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-G>)
endif()
