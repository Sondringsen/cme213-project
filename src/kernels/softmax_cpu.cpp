#include "kernels/softmax_cpu.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

// Plain three-pass softmax: find max, sum exp(x-max), divide.
void softmax_cpu(const float* x, float* y, int N, int V) {
    const float neg_inf = -std::numeric_limits<float>::infinity();
    for (int n = 0; n < N; ++n) {
        const float* row_x = x + n * V;
        float*       row_y = y + n * V;

        float m = neg_inf;
        for (int i = 0; i < V; ++i) m = std::max(m, row_x[i]);

        float s = 0.0f;
        for (int i = 0; i < V; ++i) s += std::exp(row_x[i] - m);

        float inv_s = 1.0f / s;
        for (int i = 0; i < V; ++i) row_y[i] = std::exp(row_x[i] - m) * inv_s;
    }
}
