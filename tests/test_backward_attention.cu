// ===========================================================================
// test_backward_attention.cu
//
// Correctness test for launch_attention_backward.
//
// The GPU backward materializes the S×S attention weight matrix P per
// (batch, head) and computes dQ, dK, dV via GEMM chains. The CPU reference
// replicates the same steps in plain C++ so that floating-point differences
// are only from the parallel reduction order (relative error ~1e-4).
//
// Steps (per bh):
//   1. Recompute P = softmax(Q K^T * scale, causal mask)
//   2. dV = P^T dO
//   3. dP = dO V^T
//   4. dS = P * (dP - rowsum(P*dP)) * scale   [softmax backward, scaled]
//   5. dQ = dS K
//   6. dK = dS^T Q
// ===========================================================================

#include "kernels/attention.cuh"
#include "utils/cuda_check.hpp"
#include "utils/tensor.hpp"
#include "test_utils.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

// ---------------------------------------------------------------------------
// CPU reference for naive attention backward.
// All tensors in (B, H, S, D) layout (flat array, row-major).
// ---------------------------------------------------------------------------
static void attention_backward_cpu(
        const float* Q, const float* K, const float* V, const float* dO,
        float* dQ, float* dK, float* dV,
        int B, int H, int S, int D, float scale, bool causal) {

    int BH  = B * H;
    int SD  = S * D;
    int SS  = S * S;

    std::fill(dQ, dQ + BH * SD, 0.0f);
    std::fill(dK, dK + BH * SD, 0.0f);
    std::fill(dV, dV + BH * SD, 0.0f);

    std::vector<float> P(SS), dP(SS);

    for (int bh = 0; bh < BH; ++bh) {
        const float* Qbh  = Q  + bh * SD;
        const float* Kbh  = K  + bh * SD;
        const float* Vbh  = V  + bh * SD;
        const float* dObh = dO + bh * SD;
        float*       dQbh = dQ + bh * SD;
        float*       dKbh = dK + bh * SD;
        float*       dVbh = dV + bh * SD;

        // Step 1: compute P = softmax(Q K^T * scale, optional causal mask)
        for (int i = 0; i < S; ++i) {
            float row_max = -INFINITY;
            for (int j = 0; j < S; ++j) {
                if (causal && j > i) { P[i*S+j] = -INFINITY; continue; }
                float dot = 0.0f;
                for (int d = 0; d < D; ++d) dot += Qbh[i*D+d] * Kbh[j*D+d];
                P[i*S+j] = dot * scale;
                row_max   = std::fmax(row_max, P[i*S+j]);
            }
            float sum = 0.0f;
            for (int j = 0; j < S; ++j) {
                float v = (P[i*S+j] == -INFINITY) ? 0.0f
                                                  : expf(P[i*S+j] - row_max);
                P[i*S+j] = v;
                sum      += v;
            }
            for (int j = 0; j < S; ++j) P[i*S+j] /= sum;
        }

        // Step 2: dV = P^T dO   (j,d) += sum_i P[i,j]*dO[i,d]
        for (int j = 0; j < S; ++j)
            for (int i = 0; i < S; ++i)
                for (int d = 0; d < D; ++d)
                    dVbh[j*D+d] += P[i*S+j] * dObh[i*D+d];

        // Step 3: dP = dO V^T   (i,j) = sum_d dO[i,d]*V[j,d]
        for (int i = 0; i < S; ++i)
            for (int j = 0; j < S; ++j) {
                float dot = 0.0f;
                for (int d = 0; d < D; ++d) dot += dObh[i*D+d] * Vbh[j*D+d];
                dP[i*S+j] = dot;
            }

        // Step 4: softmax backward (dS = P * (dP - rowsum(P*dP))) * scale
        for (int i = 0; i < S; ++i) {
            float rowdot = 0.0f;
            for (int j = 0; j < S; ++j) rowdot += P[i*S+j] * dP[i*S+j];
            for (int j = 0; j < S; ++j)
                dP[i*S+j] = scale * P[i*S+j] * (dP[i*S+j] - rowdot);
        }
        // dP now holds dS

        // Step 5: dQ = dS K   (i,d) += sum_j dS[i,j]*K[j,d]
        for (int i = 0; i < S; ++i)
            for (int j = 0; j < S; ++j)
                for (int d = 0; d < D; ++d)
                    dQbh[i*D+d] += dP[i*S+j] * Kbh[j*D+d];

        // Step 6: dK = dS^T Q   (j,d) += sum_i dS[i,j]*Q[i,d]
        for (int j = 0; j < S; ++j)
            for (int i = 0; i < S; ++i)
                for (int d = 0; d < D; ++d)
                    dKbh[j*D+d] += dP[i*S+j] * Qbh[i*D+d];
    }
}

// ---------------------------------------------------------------------------
// One test case.
// ---------------------------------------------------------------------------
static int run_case(int B, int H, int S, int D, bool causal) {
    float scale = 1.0f / sqrtf(static_cast<float>(D));
    int total   = B * H * S * D;

    std::vector<float> hQ(total), hK(total), hV(total), hdO(total);
    fill_random(hQ,  1);
    fill_random(hK,  2);
    fill_random(hV,  3);
    fill_random(hdO, 4);

    Tensor<float> Q({B,H,S,D}), K({B,H,S,D}), V({B,H,S,D}), dO({B,H,S,D});
    Tensor<float> dQ({B,H,S,D}), dK({B,H,S,D}), dV({B,H,S,D});
    Tensor<float> P_buf({B*H, S*S}), dP_buf({B*H, S*S});

    Q.copy_from_host(hQ.data());
    K.copy_from_host(hK.data());
    V.copy_from_host(hV.data());
    dO.copy_from_host(hdO.data());
    dQ.zero(); dK.zero(); dV.zero();

    launch_attention_backward(Q.data(), K.data(), V.data(), dO.data(),
                              dQ.data(), dK.data(), dV.data(),
                              B, H, S, D, scale, causal,
                              P_buf.data(), dP_buf.data());
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> gpu_dQ(total), gpu_dK(total), gpu_dV(total);
    dQ.copy_to_host(gpu_dQ.data());
    dK.copy_to_host(gpu_dK.data());
    dV.copy_to_host(gpu_dV.data());

    std::vector<float> cpu_dQ(total), cpu_dK(total), cpu_dV(total);
    attention_backward_cpu(hQ.data(), hK.data(), hV.data(), hdO.data(),
                           cpu_dQ.data(), cpu_dK.data(), cpu_dV.data(),
                           B, H, S, D, scale, causal);

    float dq_abs, dq_rel, dk_abs, dk_rel, dv_abs, dv_rel;
    compare(cpu_dQ, gpu_dQ, dq_abs, dq_rel);
    compare(cpu_dK, gpu_dK, dk_abs, dk_rel);
    compare(cpu_dV, gpu_dV, dv_abs, dv_rel);

    std::printf("  B=%d H=%d S=%3d D=%3d causal=%d | "
                "dQ rel=%.2e dK rel=%.2e dV rel=%.2e",
                B, H, S, D, (int)causal, dq_rel, dk_rel, dv_rel);

    constexpr float TOL = 2e-2f;
    if (dq_rel > TOL || dk_rel > TOL || dv_rel > TOL) {
        std::printf(" FAIL\n");
        return 1;
    }
    std::printf(" PASS\n");
    return 0;
}

int main() {
    int fails = 0;
    std::printf("=== Attention Backward ===\n");
    fails += run_case(1, 1, 16, 32,  false);
    fails += run_case(1, 1, 16, 32,  true);
    fails += run_case(2, 4, 32, 64,  true);
    fails += run_case(2, 4, 64, 64,  true);

    if (fails) {
        std::printf("\n%d case(s) FAILED\n", fails);
        return 1;
    }
    std::printf("\nAll cases PASSED\n");
    return 0;
}
