#include "kernels/layernorm_cpu.hpp"

#include <cmath>

// Straightforward two-pass-per-row implementation. Computes mean and
// variance separately, then normalizes. No fusion, no SIMD; this is a
// correctness oracle, not a baseline we care to beat.
void layernorm_cpu(const float* x, const float* gamma, const float* beta,
                   float* y, int N, int H, float eps) {
    for (int n = 0; n < N; ++n) {
        const float* row_x = x + n * H;
        float*       row_y = y + n * H;

        // Mean of the row.
        float mean = 0.0f;
        for (int i = 0; i < H; ++i) mean += row_x[i];
        mean /= H;

        // Variance of the row.
        float var = 0.0f;
        for (int i = 0; i < H; ++i) {
            float d = row_x[i] - mean;
            var += d * d;
        }
        var /= H;

        float rstd = 1.0f / std::sqrt(var + eps);

        // Affine-shifted output.
        for (int i = 0; i < H; ++i) {
            row_y[i] = gamma[i] * (row_x[i] - mean) * rstd + beta[i];
        }
    }
}
