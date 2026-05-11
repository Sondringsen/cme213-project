#pragma once

// ---------------------------------------------------------------------------
// CUDA error-checking macros.
//
// Every CUDA runtime call returns a cudaError_t. If we ignore those, a kernel
// launch error or a bad cudaMemcpy silently corrupts later results and we
// chase ghosts. CUDA_CHECK wraps a call and aborts with a clear message on
// failure. CUDA_CHECK_LAST() queries the most recent kernel launch (kernels
// don't return errors directly; they go through cudaGetLastError).
//
// Use CUDA_CHECK around every runtime API call (cudaMalloc, cudaMemcpy, ...)
// and CUDA_CHECK_LAST() immediately after every kernel launch.
// ---------------------------------------------------------------------------

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

#define CUDA_CHECK(stmt)                                                       \
    do {                                                                       \
        cudaError_t err__ = (stmt);                                            \
        if (err__ != cudaSuccess) {                                            \
            std::fprintf(stderr,                                               \
                         "CUDA error %s at %s:%d: %s\n",                       \
                         cudaGetErrorName(err__),                              \
                         __FILE__, __LINE__,                                   \
                         cudaGetErrorString(err__));                           \
            std::abort();                                                      \
        }                                                                      \
    } while (0)

#define CUDA_CHECK_LAST() CUDA_CHECK(cudaGetLastError())
