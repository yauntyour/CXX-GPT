cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(CXX-GPT LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(TENSORN_BUILD_EXAMPLES OFF)
set(TENSORN_BUILD_BENCHMARKS OFF)
add_subdirectory(TensorN)

add_executable(gpt_train src/train.cpp src/cuda_kernels.cu)
target_include_directories(gpt_train PRIVATE src)
target_link_libraries(gpt_train PRIVATE TensorN)

add_executable(gpt_demo src/demo.cpp src/cuda_kernels.cu)
target_include_directories(gpt_demo PRIVATE src)
target_link_libraries(gpt_demo PRIVATE TensorN)

set_target_properties(gpt_train gpt_demo PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)
