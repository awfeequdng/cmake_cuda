cmake_minimum_required(VERSION 3.15)

project(cmake_cuda LANGUAGES CXX CUDA)


if (USE_CUDA)
    enable_language(CUDA)
endif(USE_CUDA)

add_library(matmul_cuda matmul_cuda.cu)
target_compile_features(matmul_cuda PUBLIC cxx_std_11)

add_executable(main main.cc)
target_link_libraries(main matmul_cuda)