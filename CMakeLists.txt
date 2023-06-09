cmake_minimum_required(VERSION 3.25)
project(CudaRsa CUDA)

set(CMAKE_CUDA_STANDARD 20)

add_executable(CudaRsa main.cu)
set_target_properties(CudaRsa PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
