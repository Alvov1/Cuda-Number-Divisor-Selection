cmake_minimum_required(VERSION 3.25)
project(CudaRsa CUDA)

set(CMAKE_CUDA_STANDARD 20)

add_executable(CudaRsa main.cu)
add_subdirectory(mpz)
set_target_properties(CudaRsa PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
