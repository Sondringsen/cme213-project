#pragma once

// Reference CPU attention. Standard quadratic-in-S algorithm.
//   Q, K, V, O : (B, H, S, D) row-major FP32
//   scale      : usually 1/sqrt(D)
//   causal     : mask future keys if true
//
// Complexity: O(B * H * S^2 * D). Used as the correctness oracle for the
// Flash Attention kernel; only run on small S because of the quadratic.
void attention_cpu(const float* Q, const float* K, const float* V, float* O,
                   int B, int H, int S, int D, float scale, bool causal);
