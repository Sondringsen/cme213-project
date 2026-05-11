#pragma once

// Reference CPU softmax. Row-wise, numerically stable.
//   x : (N, V)
//   y : (N, V)
void softmax_cpu(const float* x, float* y, int N, int V);
