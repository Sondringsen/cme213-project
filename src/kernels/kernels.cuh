#pragma once

// ---------------------------------------------------------------------------
// Umbrella header: include all kernel APIs in one shot.
// ---------------------------------------------------------------------------

#include "kernels/gemm.cuh"
#include "kernels/layernorm.cuh"
#include "kernels/softmax.cuh"
#include "kernels/gelu.cuh"
#include "kernels/cross_entropy.cuh"
#include "kernels/attention.cuh"
#include "kernels/embedding.cuh"
#include "kernels/adam.cuh"
#include "kernels/reshape.cuh"
#include "kernels/pointwise.cuh"
