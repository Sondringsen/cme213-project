cmake_minimum_required(VERSION 3.18)
project(cme213_llm LANGUAGES CXX CUDA)

# ---------------------------------------------------------------------------
# Build configuration
# ---------------------------------------------------------------------------
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 70 80 86 90)
endif()

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS}  -Wall -Wextra")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr --expt-extended-lambda -lineinfo")

# ---------------------------------------------------------------------------
# Optional MPI support
# ---------------------------------------------------------------------------
find_package(MPI QUIET)
find_package(CUDAToolkit QUIET)
if(MPI_FOUND)
    message(STATUS "MPI found — building distributed training target")
    add_compile_definitions(MPI_ENABLED)
else()
    message(STATUS "MPI not found — skipping train_distributed target")
endif()

# ---------------------------------------------------------------------------
# Core library: all kernels + CPU oracles. Tests and training link against this.
# ---------------------------------------------------------------------------
add_library(llm STATIC
    # GEMM (forward + TN/NT transpose variants for backward)
    src/kernels/gemm.cu
    src/kernels/gemm_cpu.cpp
    # LayerNorm (forward + backward)
    src/kernels/layernorm.cu
    src/kernels/layernorm_cpu.cpp
    # Softmax
    src/kernels/softmax.cu
    src/kernels/softmax_cpu.cpp
    # GELU (forward + backward)
    src/kernels/gelu.cu
    src/kernels/gelu_cpu.cpp
    # Cross-entropy (forward + backward)
    src/kernels/cross_entropy.cu
    src/kernels/cross_entropy_cpu.cpp
    # Flash Attention (forward + naive backward)
    src/kernels/attention.cu
    src/kernels/attention_cpu.cpp
    # PagedAttention decode (inference)
    src/kernels/paged_attention.cu
    # Embedding (forward + backward)
    src/kernels/embedding.cu
    # Adam optimizer
    src/kernels/adam.cu
    # Pointwise helpers (add, scale, fill)
    src/kernels/pointwise.cu
    # Layout reshape helpers
    src/kernels/reshape.cu
)

target_include_directories(llm PUBLIC ${CMAKE_SOURCE_DIR}/src)
set_target_properties(llm PROPERTIES POSITION_INDEPENDENT_CODE ON)

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
enable_testing()

function(add_kernel_test name)
    add_executable(${name} tests/${name}.cu)
    target_link_libraries(${name} PRIVATE llm)
    target_include_directories(${name} PRIVATE ${CMAKE_SOURCE_DIR}/tests)
    add_test(NAME ${name} COMMAND ${name})
endfunction()

# Existing forward tests
add_kernel_test(test_gemm)
if(CUDAToolkit_FOUND AND TARGET CUDA::cublas)
    target_link_libraries(test_gemm PRIVATE CUDA::cublas)
    target_compile_definitions(test_gemm PRIVATE WITH_CUBLAS)
    message(STATUS "cuBLAS found — cuBLAS comparison enabled in test_gemm")
endif()
add_kernel_test(test_layernorm)
add_kernel_test(test_softmax)
add_kernel_test(test_gelu)
add_kernel_test(test_cross_entropy)
add_kernel_test(test_attention)

# New backward tests
add_kernel_test(test_backward_layernorm)
add_kernel_test(test_backward_gelu)
add_kernel_test(test_backward_cross_entropy)
add_kernel_test(test_backward_attention)
add_kernel_test(test_paged_attention)

# PyTorch comparison test
add_executable(test_vs_pytorch tests/test_vs_pytorch.cu)
target_link_libraries(test_vs_pytorch PRIVATE llm)
target_include_directories(test_vs_pytorch PRIVATE ${CMAKE_SOURCE_DIR}/tests)
target_compile_definitions(test_vs_pytorch PRIVATE
    REF_DATA_DIR="${CMAKE_SOURCE_DIR}/tests/ref_data")
add_test(NAME test_vs_pytorch COMMAND test_vs_pytorch)

# ---------------------------------------------------------------------------
# Distributed training (requires MPI)
# ---------------------------------------------------------------------------
if(MPI_FOUND)
    add_executable(train_distributed src/main_distributed.cu)
    target_link_libraries(train_distributed PRIVATE llm MPI::MPI_CXX)
    target_include_directories(train_distributed PRIVATE ${CMAKE_SOURCE_DIR}/src)
    # Link nvToolsExt when available so NVTX range markers appear in Nsight timelines.
    if(CUDAToolkit_FOUND AND TARGET CUDA::nvToolsExt)
        target_link_libraries(train_distributed PRIVATE CUDA::nvToolsExt)
        target_compile_definitions(train_distributed PRIVATE WITH_NVTX)
        message(STATUS "nvToolsExt found — NVTX annotations enabled for train_distributed")
    endif()
endif()

# ---------------------------------------------------------------------------
# MPI allreduce latency benchmark (requires MPI)
# ---------------------------------------------------------------------------
if(MPI_FOUND)
    add_executable(allreduce_bench benchmarks/allreduce_bench.cu)
    target_link_libraries(allreduce_bench PRIVATE MPI::MPI_CXX)
endif()

# ---------------------------------------------------------------------------
# Inference binary (no MPI needed — single GPU)
# ---------------------------------------------------------------------------
add_executable(run_inference src/main_inference.cu)
target_link_libraries(run_inference PRIVATE llm)
target_include_directories(run_inference PRIVATE ${CMAKE_SOURCE_DIR}/src)

# ---------------------------------------------------------------------------
# PagedAttention benchmark (single GPU, no MPI)
# ---------------------------------------------------------------------------
add_executable(paged_attention_bench benchmarks/paged_attention_bench.cu)
target_link_libraries(paged_attention_bench PRIVATE llm)
target_include_directories(paged_attention_bench PRIVATE ${CMAKE_SOURCE_DIR}/src)
