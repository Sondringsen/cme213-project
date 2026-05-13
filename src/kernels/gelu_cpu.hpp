#pragma once

// CPU reference implementation of GELU forward.
// Used for correctness checking in tests.
void gelu_cpu(const float* x, float* y, int n);
