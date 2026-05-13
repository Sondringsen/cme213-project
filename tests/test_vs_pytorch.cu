// ===========================================================================
// test_vs_pytorch.cu
//
// Compares our GPU kernels against PyTorch-generated reference data.
// Before running this test, generate the reference files with:
//     python3 scripts/generate_ref_data.py
//
// The shapes below MUST match those in generate_ref_data.py.
//
// Build note: CMakeLists.txt defines REF_DATA_DIR as a compile-time string
// pointing to <project_root>/tests/ref_data. The test skips gracefully if
// that directory does not exist (i.e. the Python script hasn't been run).
// ===========================================================================

#include "kernels/kernels.cuh"
#include "utils/utils.hpp"
#include "test_utils.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

// REF_DATA_DIR is injected by CMakeLists.txt at compile time.
#ifndef REF_DATA_DIR
#  define REF_DATA_DIR "tests/ref_data"
#endif

// ---------------------------------------------------------------------------
// Binary file helpers
// ---------------------------------------------------------------------------

static std::string ref(const char* filename) {
    return std::string(REF_DATA_DIR) + "/" + filename;
}

// Load n float32 values from a raw binary file. Returns false on failure.
static bool load_floats(const std::string& path, std::vector<float>& v, size_t n) {
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) {
        std::fprintf(stderr, "  [skip] cannot open %s\n", path.c_str());
        return false;
    }
    v.resize(n);
    size_t got = std::fread(v.data(), sizeof(float), n, f);
    std::fclose(f);
    if (got != n) {
        std::fprintf(stderr, "  [fail] short read in %s: got %zu, want %zu\n",
                     path.c_str(), got, n);
        return false;
    }
    return true;
}

// Load n int32 values from a raw binary file. Returns false on failure.
static bool load_ints(const std::string& path, std::vector<int>& v, size_t n) {
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) {
        std::fprintf(stderr, "  [skip] cannot open %s\n", path.c_str());
        return false;
    }
    v.resize(n);
    size_t got = std::fread(v.data(), sizeof(int), n, f);
    std::fclose(f);
    if (got != n) {
        std::fprintf(stderr, "  [fail] short read in %s: got %zu, want %zu\n",
                     path.c_str(), got, n);
        return false;
    }
    return true;
}

// ---------------------------------------------------------------------------
// Per-kernel comparison helpers
// ---------------------------------------------------------------------------

// Returns 0 on pass, 1 on fail, -1 on skip (missing ref data).
static int check(const char* label,
                 const std::vector<float>& ref_out,
                 const std::vector<float>& gpu_out,
                 float tol) {
    float abs_err, rel_err;
    compare(ref_out, gpu_out, abs_err, rel_err);
    bool pass = (rel_err <= tol);
    std::printf("  %-18s  abs=%.3e  rel=%.3e  tol=%.0e  %s\n",
                label, abs_err, rel_err, tol, pass ? "PASS" : "FAIL");
    return pass ? 0 : 1;
}

// ---------------------------------------------------------------------------
// Individual kernel tests
// ---------------------------------------------------------------------------

static int test_gemm() {
    std::printf("\n[GEMM]  M=256 N=256 K=128\n");
    const int M = 256, N = 256, K = 128;
    std::vector<float> hA, hB, hC_ref;
    if (!load_floats(ref("gemm_A.bin"), hA, (size_t)M * K)) return -1;
    if (!load_floats(ref("gemm_B.bin"), hB, (size_t)K * N)) return -1;
    if (!load_floats(ref("gemm_C.bin"), hC_ref, (size_t)M * N)) return -1;

    Tensor<float> dA({M, K}), dB({K, N}), dC({M, N});
    dA.copy_from_host(hA.data());
    dB.copy_from_host(hB.data());
    launch_gemm_tiled(dA.data(), dB.data(), dC.data(), M, N, K);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> hC_gpu(M * N);
    dC.copy_to_host(hC_gpu.data());
    return check("gemm", hC_ref, hC_gpu, /*tol=*/1e-3f);
}

static int test_layernorm() {
    std::printf("\n[LayerNorm]  N=32 H=256\n");
    const int N = 32, H = 256;
    std::vector<float> hx, hgamma, hbeta, hy_ref;
    if (!load_floats(ref("layernorm_x.bin"),     hx,     (size_t)N * H)) return -1;
    if (!load_floats(ref("layernorm_gamma.bin"), hgamma, (size_t)H))     return -1;
    if (!load_floats(ref("layernorm_beta.bin"),  hbeta,  (size_t)H))     return -1;
    if (!load_floats(ref("layernorm_y.bin"),     hy_ref, (size_t)N * H)) return -1;

    Tensor<float> dx({N, H}), dgamma({H}), dbeta({H}), dy({N, H});
    dx.copy_from_host(hx.data());
    dgamma.copy_from_host(hgamma.data());
    dbeta.copy_from_host(hbeta.data());
    launch_layernorm_forward(dx.data(), dgamma.data(), dbeta.data(),
                             dy.data(), N, H, /*eps=*/1e-5f);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> hy_gpu(N * H);
    dy.copy_to_host(hy_gpu.data());
    return check("layernorm", hy_ref, hy_gpu, /*tol=*/1e-4f);
}

static int test_softmax() {
    std::printf("\n[Softmax]  N=32 V=512\n");
    const int N = 32, V = 512;
    std::vector<float> hx, hy_ref;
    if (!load_floats(ref("softmax_x.bin"), hx,     (size_t)N * V)) return -1;
    if (!load_floats(ref("softmax_y.bin"), hy_ref, (size_t)N * V)) return -1;

    Tensor<float> dx({N, V}), dy({N, V});
    dx.copy_from_host(hx.data());
    launch_softmax_forward(dx.data(), dy.data(), N, V);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> hy_gpu(N * V);
    dy.copy_to_host(hy_gpu.data());
    return check("softmax", hy_ref, hy_gpu, /*tol=*/1e-5f);
}

static int test_gelu() {
    std::printf("\n[GELU]  N=4096\n");
    const int N = 4096;
    std::vector<float> hx, hy_ref;
    if (!load_floats(ref("gelu_x.bin"), hx,     (size_t)N)) return -1;
    if (!load_floats(ref("gelu_y.bin"), hy_ref, (size_t)N)) return -1;

    Tensor<float> dx({N}), dy({N});
    dx.copy_from_host(hx.data());
    launch_gelu_forward(dx.data(), dy.data(), N);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> hy_gpu(N);
    dy.copy_to_host(hy_gpu.data());
    return check("gelu", hy_ref, hy_gpu, /*tol=*/1e-3f);
}

static int test_cross_entropy() {
    std::printf("\n[CrossEntropy]  N=32 V=512\n");
    const int N = 32, V = 512;
    std::vector<float> hlogits, hlosses_ref;
    std::vector<int>   htargets;
    if (!load_floats(ref("cross_entropy_logits.bin"),  hlogits,     (size_t)N * V)) return -1;
    if (!load_ints  (ref("cross_entropy_targets.bin"), htargets,    (size_t)N))     return -1;
    if (!load_floats(ref("cross_entropy_losses.bin"),  hlosses_ref, (size_t)N))     return -1;

    Tensor<float> dlogits({N, V}), dlosses({N});
    Tensor<int>   dtargets({N});
    dlogits.copy_from_host(hlogits.data());
    dtargets.copy_from_host(htargets.data());
    launch_cross_entropy_forward(dlogits.data(), dtargets.data(),
                                 dlosses.data(), N, V);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> hlosses_gpu(N);
    dlosses.copy_to_host(hlosses_gpu.data());
    return check("cross_entropy", hlosses_ref, hlosses_gpu, /*tol=*/1e-4f);
}

static int test_attention() {
    std::printf("\n[Attention]  B=2 H=4 S=64 D=64 causal=true\n");
    const int B = 2, H = 4, S = 64, D = 64;
    size_t numel = (size_t)B * H * S * D;
    float  scale = 1.0f / sqrtf((float)D);

    std::vector<float> hQ, hK, hV, hO_ref;
    if (!load_floats(ref("attention_Q.bin"), hQ,     numel)) return -1;
    if (!load_floats(ref("attention_K.bin"), hK,     numel)) return -1;
    if (!load_floats(ref("attention_V.bin"), hV,     numel)) return -1;
    if (!load_floats(ref("attention_O.bin"), hO_ref, numel)) return -1;

    Tensor<float> dQ({B,H,S,D}), dK({B,H,S,D}), dV({B,H,S,D}), dO({B,H,S,D});
    dQ.copy_from_host(hQ.data());
    dK.copy_from_host(hK.data());
    dV.copy_from_host(hV.data());
    launch_flash_attention_forward(dQ.data(), dK.data(), dV.data(), dO.data(),
                                   B, H, S, D, scale, /*causal=*/true);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> hO_gpu(numel);
    dO.copy_to_host(hO_gpu.data());
    // Flash Attention accumulates many expf() calls; 5e-3 matches the
    // threshold used in test_attention.cu against the CPU oracle.
    return check("attention", hO_ref, hO_gpu, /*tol=*/5e-3f);
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------
int main() {
    std::printf("Comparing GPU kernels against PyTorch reference data.\n");
    std::printf("  REF_DATA_DIR = %s\n", REF_DATA_DIR);

    int fails = 0, skips = 0;
    auto run = [&](int result, const char* name) {
        if (result == -1) { ++skips; }
        else if (result != 0) { ++fails; }
    };

    run(test_gemm(),          "gemm");
    run(test_layernorm(),     "layernorm");
    run(test_softmax(),       "softmax");
    run(test_gelu(),          "gelu");
    run(test_cross_entropy(), "cross_entropy");
    run(test_attention(),     "attention");

    std::printf("\n");
    if (skips > 0) {
        std::printf("%d test(s) skipped -- run 'python3 scripts/generate_ref_data.py' first.\n",
                    skips);
    }
    if (fails > 0) {
        std::printf("%d test(s) FAILED\n", fails);
        return 1;
    }
    if (skips == 0) {
        std::printf("All PyTorch comparison tests passed.\n");
    }
    return 0;
}
