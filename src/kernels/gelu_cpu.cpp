#include "kernels/gelu_cpu.hpp"

#include <cmath>

// Exact GELU using the standard library erff.
// Must match gelu_forward_kernel in gelu.cu exactly so the GPU/CPU
// comparison in test_gelu.cu is a meaningful correctness check.
void gelu_cpu(const float* x, float* y, int n) {
    constexpr float kInvSqrt2 = 0.7071067811865476f;
    for (int i = 0; i < n; ++i) {
        float xi = x[i];
        y[i] = xi * 0.5f * (1.0f + std::erff(xi * kInvSqrt2));
    }
}
