#pragma once

// ---------------------------------------------------------------------------
// Shared test utilities.
//
// Every test follows the same pattern: fill host buffers with deterministic
// random values, copy to device, run the GPU kernel, copy back, compare to
// a CPU reference, and time the kernel. The helpers here factor out the
// boilerplate so each test_*.cu only has to spell out the kernel-specific
// bits (which CPU oracle to call, what shapes to sweep, how to interpret
// the result).
// ---------------------------------------------------------------------------

#include "utils/cuda_check.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

// Fill a host vector with deterministic pseudo-random FP32 values in [-1, 1].
// The seed is explicit so that reruns are bit-exact and bugs reproduce.
inline void fill_random(std::vector<float>& v, unsigned seed) {
    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (auto& x : v) x = dist(gen);
}

// Fill with random ints in [0, V). Used for token-id targets.
inline void fill_random_int(std::vector<int>& v, int V, unsigned seed) {
    std::mt19937 gen(seed);
    std::uniform_int_distribution<int> dist(0, V - 1);
    for (auto& x : v) x = dist(gen);
}

// Element-wise comparison. Reports max-abs and max-rel error.
// Relative error uses max(|a|, eps) so near-zero references don't blow up.
inline void compare(const std::vector<float>& a,
                    const std::vector<float>& b,
                    float& max_abs, float& max_rel) {
    max_abs = 0.0f;
    max_rel = 0.0f;
    for (size_t i = 0; i < a.size(); ++i) {
        float diff = std::fabs(a[i] - b[i]);
        max_abs = std::max(max_abs, diff);
        float denom = std::max(std::fabs(a[i]), 1e-6f);
        max_rel = std::max(max_rel, diff / denom);
    }
}

// Generic kernel timer. Takes any callable that issues a kernel launch
// (typically a lambda capturing the input/output pointers and shapes).
// Does one untimed warm-up call (pays JIT / first-touch costs), then n_iter
// timed launches inside a single cudaEvent pair.
//
// Returns the average runtime per launch in milliseconds.
template <typename LaunchFn>
inline float time_kernel(LaunchFn&& launch, int n_iter) {
    launch();
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < n_iter; ++i) launch();
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return ms / n_iter;
}
