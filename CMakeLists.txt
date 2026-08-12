cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(CXX-GPT LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Default to native code for both Turing (RTX 20x0) and Blackwell (RTX 50x0)
# so the training loops run on cubins instead of JIT-compiled PTX.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES OR CMAKE_CUDA_ARCHITECTURES STREQUAL "")
    set(CMAKE_CUDA_ARCHITECTURES "75;120" CACHE STRING "CUDA architectures" FORCE)
endif()

set(TENSORN_BUILD_EXAMPLES OFF)
set(TENSORN_BUILD_BENCHMARKS OFF)
add_subdirectory(TensorN)

# Dataset preprocessor: JSONL -> binary "big block" (BOS/EOS wrapped, flat).
# CPU-only; requires only nlohmann/json (provided by TensorN).
add_executable(preprocess_dataset src/tools/preprocess_dataset.cpp)
target_include_directories(preprocess_dataset PRIVATE src)
target_link_libraries(preprocess_dataset PRIVATE nlohmann_json::nlohmann_json)

# Original GPT model (softmax attention)
add_executable(gpt_train src/GPT/gpt_train.cpp src/GPT/gpt_cuda_kernels.cu)
target_include_directories(gpt_train PRIVATE src)
target_link_libraries(gpt_train PRIVATE TensorN)

add_executable(gpt_demo src/GPT/gpt_demo.cpp src/GPT/gpt_cuda_kernels.cu)
target_include_directories(gpt_demo PRIVATE src)
target_link_libraries(gpt_demo PRIVATE TensorN)

# LinearGPT model (Hadamard+Exp kernel, linear attention)
add_executable(linear_train
    src/LinearAttention/linear_train.cpp
    src/LinearAttention/linear_cuda_kernels.cu
    src/GPT/gpt_cuda_kernels.cu
)
target_include_directories(linear_train PRIVATE src)
target_link_libraries(linear_train PRIVATE TensorN)

add_executable(linear_demo
    src/LinearAttention/linear_demo.cpp
    src/LinearAttention/linear_cuda_kernels.cu
    src/GPT/gpt_cuda_kernels.cu
)
target_include_directories(linear_demo PRIVATE src)
target_link_libraries(linear_demo PRIVATE TensorN)

# TwoStageGPT model (expanded softmax MHA + ELU kernel attention)
add_executable(two_stage_train
    src/TwoStage/two_stage_train.cpp
    src/TwoStage/two_stage_cuda_kernels.cu
    src/GPT/gpt_cuda_kernels.cu
)
target_include_directories(two_stage_train PRIVATE src)
target_link_libraries(two_stage_train PRIVATE TensorN)

add_executable(two_stage_demo
    src/TwoStage/two_stage_demo.cpp
    src/TwoStage/two_stage_cuda_kernels.cu
    src/GPT/gpt_cuda_kernels.cu
)
target_include_directories(two_stage_demo PRIVATE src)
target_link_libraries(two_stage_demo PRIVATE TensorN)

# TwoStageGPT CUDA kernel unit test (CPU reference vs fused kernels)
add_executable(kernel_test
    src/TwoStage/kernel_test.cpp
    src/TwoStage/two_stage_cuda_kernels.cu
    src/GPT/gpt_cuda_kernels.cu
)
target_include_directories(kernel_test PRIVATE src)
target_link_libraries(kernel_test PRIVATE TensorN)

set_target_properties(gpt_train gpt_demo linear_train linear_demo two_stage_train two_stage_demo kernel_test preprocess_dataset PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)
