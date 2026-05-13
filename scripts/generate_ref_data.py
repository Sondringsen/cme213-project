#!/usr/bin/env python3
"""
generate_ref_data.py

Generates reference inputs and outputs for each GPU kernel using PyTorch,
then saves them as raw float32 / int32 binary files that test_vs_pytorch.cu
loads and compares against our CUDA implementations.

Usage:
    python3 scripts/generate_ref_data.py

Output directory: tests/ref_data/
  Each kernel gets a set of .bin files:
    <kernel>_<tensor>.bin  -- raw float32 (or int32 for targets) in C order

The shapes below are shared with test_vs_pytorch.cu -- both files must agree.

Requirements:
    pip install torch numpy
"""

import os
import sys
import struct

import numpy as np

try:
    import torch
    import torch.nn.functional as F
except ImportError:
    sys.exit("PyTorch not found. Install with: pip install torch")

# ---------------------------------------------------------------------------
# Output directory -- relative to this script's location.
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(SCRIPT_DIR, "..", "tests", "ref_data")
os.makedirs(OUT_DIR, exist_ok=True)

# Use CPU + float32 throughout so results match what our FP32 kernels produce.
torch.manual_seed(0)
DEVICE = "cpu"
DTYPE  = torch.float32


def save(name: str, t):
    """Save a tensor (or numpy array) as a raw binary file."""
    path = os.path.join(OUT_DIR, name)
    if isinstance(t, torch.Tensor):
        t = t.detach().numpy()
    t = np.ascontiguousarray(t)
    t.tofile(path)
    print(f"  wrote {path}  shape={t.shape}  dtype={t.dtype}")


# ---------------------------------------------------------------------------
# GEMM: C = A @ B,  A:(M,K), B:(K,N), C:(M,N)
# ---------------------------------------------------------------------------
def gen_gemm():
    print("\n[GEMM]  M=256  N=256  K=128")
    M, N, K = 256, 256, 128
    A = torch.randn(M, K, dtype=DTYPE)
    B = torch.randn(K, N, dtype=DTYPE)
    C = torch.mm(A, B)
    save("gemm_A.bin", A)
    save("gemm_B.bin", B)
    save("gemm_C.bin", C)


# ---------------------------------------------------------------------------
# LayerNorm: y = gamma * (x - mean) / sqrt(var + eps) + beta
#   x:(N,H), gamma:(H,), beta:(H,), y:(N,H)
# ---------------------------------------------------------------------------
def gen_layernorm():
    print("\n[LayerNorm]  N=32  H=256")
    N, H = 32, 256
    x     = torch.randn(N, H, dtype=DTYPE)
    gamma = torch.randn(H,    dtype=DTYPE)
    beta  = torch.randn(H,    dtype=DTYPE)
    y = F.layer_norm(x, [H], gamma, beta, eps=1e-5)
    save("layernorm_x.bin",     x)
    save("layernorm_gamma.bin", gamma)
    save("layernorm_beta.bin",  beta)
    save("layernorm_y.bin",     y)


# ---------------------------------------------------------------------------
# Softmax: y = softmax(x, dim=-1),  x:(N,V), y:(N,V)
# ---------------------------------------------------------------------------
def gen_softmax():
    print("\n[Softmax]  N=32  V=512")
    N, V = 32, 512
    x = torch.randn(N, V, dtype=DTYPE)
    y = torch.softmax(x, dim=-1)
    save("softmax_x.bin", x)
    save("softmax_y.bin", y)


# ---------------------------------------------------------------------------
# GELU (exact erf formulation, matching PyTorch's default approximate='none')
# ---------------------------------------------------------------------------
def gen_gelu():
    print("\n[GELU]  N=4096")
    N = 4096
    x = torch.randn(N, dtype=DTYPE)
    y = F.gelu(x, approximate="none")
    save("gelu_x.bin", x)
    save("gelu_y.bin", y)


# ---------------------------------------------------------------------------
# Cross-entropy: loss[n] = -log(softmax(logits[n])[targets[n]])
#   logits:(N,V) float32,  targets:(N,) int32,  losses:(N,) float32
# ---------------------------------------------------------------------------
def gen_cross_entropy():
    print("\n[CrossEntropy]  N=32  V=512")
    N, V = 32, 512
    logits  = torch.randn(N, V, dtype=DTYPE)
    targets = torch.randint(0, V, (N,), dtype=torch.int32)
    losses  = F.cross_entropy(logits, targets.long(), reduction="none")
    save("cross_entropy_logits.bin",  logits)
    # Save targets as int32 so C++ can fread() them into int arrays directly.
    np.array(targets.numpy(), dtype=np.int32).tofile(
        os.path.join(OUT_DIR, "cross_entropy_targets.bin"))
    print(f"  wrote {os.path.join(OUT_DIR, 'cross_entropy_targets.bin')}"
          f"  shape={targets.shape}  dtype=int32")
    save("cross_entropy_losses.bin",  losses)


# ---------------------------------------------------------------------------
# Flash Attention: O = softmax(Q K^T / sqrt(D)) V,  causal mask applied.
#   Q,K,V,O:(B,H,S,D)
# ---------------------------------------------------------------------------
def gen_attention():
    print("\n[Attention]  B=2  H=4  S=64  D=64  causal=True")
    B, H, S, D = 2, 4, 64, 64
    scale = D ** -0.5

    Q = torch.randn(B, H, S, D, dtype=DTYPE)
    K = torch.randn(B, H, S, D, dtype=DTYPE)
    V = torch.randn(B, H, S, D, dtype=DTYPE)

    # scaled_dot_product_attention with is_causal=True matches our kernel's
    # causal masking (each query only attends to keys at positions <= itself).
    O = F.scaled_dot_product_attention(Q, K, V,
                                       scale=scale,
                                       is_causal=True)
    save("attention_Q.bin", Q)
    save("attention_K.bin", K)
    save("attention_V.bin", V)
    save("attention_O.bin", O)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print(f"Writing reference data to: {os.path.abspath(OUT_DIR)}")
    gen_gemm()
    gen_layernorm()
    gen_softmax()
    gen_gelu()
    gen_cross_entropy()
    gen_attention()
    print("\nDone. Run 'make test' (or ctest) to compare against GPU kernels.")
