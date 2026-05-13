#pragma once

// ---------------------------------------------------------------------------
// Umbrella header: include all kernel APIs in one shot.
//
// Usage in downstream code (layers, training loop, tests):
//     #include "kernels/kernels.cuh"
//
// Note: this controls *compilation visibility* only. You still need to link
// against the compiled .cu objects (handled by CMakeLists.txt via the `llm`
// static library). Adding a new kernel means: create the .cuh/.cu pair, add
// it to the library in CMakeLists.txt, then add the include below.
// ---------------------------------------------------------------------------

#include "kernels/gemm.cuh"
#include "kernels/layernorm.cuh"
#include "kernels/softmax.cuh"
#include "kernels/gelu.cuh"
#include "kernels/cross_entropy.cuh"
#include "kernels/attention.cuh"
