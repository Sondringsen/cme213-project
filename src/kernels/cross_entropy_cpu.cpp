#include "kernels/cross_entropy_cpu.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

// Per-token cross-entropy via the LSE identity (same form as the GPU kernel
// so the comparison is meaningful).
void cross_entropy_cpu(const float* logits, const int* targets, float* losses,
                       int N, int V) {
    const float neg_inf = -std::numeric_limits<float>::infinity();
    for (int n = 0; n < N; ++n) {
        const float* row = logits + n * V;

        float m = neg_inf;
        for (int i = 0; i < V; ++i) m = std::max(m, row[i]);

        float s = 0.0f;
        for (int i = 0; i < V; ++i) s += std::exp(row[i] - m);

        float lse = std::log(s) + m;
        losses[n] = lse - row[targets[n]];
    }
}
