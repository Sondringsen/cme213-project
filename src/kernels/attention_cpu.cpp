#include "kernels/attention_cpu.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

// Plain quadratic-in-S attention. For each (batch, head, query) we build
// the full row of scores, take a numerically stable softmax, and read out
// V weighted by the resulting probabilities. The math is exactly the
// definition of attention; this is the oracle we compare Flash Attention
// against.
void attention_cpu(const float* Q, const float* K, const float* V, float* O,
                   int B, int H, int S, int D, float scale, bool causal) {
    const float neg_inf = -std::numeric_limits<float>::infinity();
    std::vector<float> scores(S);

    for (int b = 0; b < B; ++b) {
        for (int h = 0; h < H; ++h) {
            const float* Qbh = Q + (b * H + h) * S * D;
            const float* Kbh = K + (b * H + h) * S * D;
            const float* Vbh = V + (b * H + h) * S * D;
            float*       Obh = O + (b * H + h) * S * D;

            for (int i = 0; i < S; ++i) {
                // 1. Compute raw scores and their max (for stability).
                float m = neg_inf;
                for (int j = 0; j < S; ++j) {
                    if (causal && j > i) {
                        scores[j] = neg_inf;
                    } else {
                        float dot = 0.0f;
                        for (int d = 0; d < D; ++d) {
                            dot += Qbh[i * D + d] * Kbh[j * D + d];
                        }
                        scores[j] = dot * scale;
                    }
                    if (scores[j] > m) m = scores[j];
                }

                // 2. Numerically stable softmax over the row.
                float sum = 0.0f;
                for (int j = 0; j < S; ++j) {
                    float e = (scores[j] == neg_inf) ? 0.0f
                                                     : std::exp(scores[j] - m);
                    scores[j] = e;
                    sum += e;
                }
                float inv_sum = 1.0f / sum;
                for (int j = 0; j < S; ++j) scores[j] *= inv_sum;

                // 3. Weighted sum O[i] = sum_j P[i,j] * V[j].
                for (int d = 0; d < D; ++d) {
                    float acc = 0.0f;
                    for (int j = 0; j < S; ++j) {
                        acc += scores[j] * Vbh[j * D + d];
                    }
                    Obh[i * D + d] = acc;
                }
            }
        }
    }
}
