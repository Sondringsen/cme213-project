#pragma once

// Reference CPU implementation of per-token cross-entropy loss.
//   logits  : (N, V)
//   targets : (N,)
//   losses  : (N,)
void cross_entropy_cpu(const float* logits, const int* targets, float* losses,
                       int N, int V);
