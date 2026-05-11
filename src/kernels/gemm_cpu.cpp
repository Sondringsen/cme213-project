#include "kernels/gemm_cpu.hpp"

// Triple-loop GEMM. The middle ordering (i, j, k) is the most cache-
// unfriendly of the six permutations, but performance does not matter here:
// this is a correctness oracle, not a baseline we are trying to beat.
void gemm_cpu(const float* A, const float* B, float* C,
              int M, int N, int K) {
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float s = 0.0f;
            for (int k = 0; k < K; ++k) {
                s += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = s;
        }
    }
}
