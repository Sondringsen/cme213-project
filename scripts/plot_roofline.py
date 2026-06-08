#!/usr/bin/env python3
"""
Generate two roofline plots (before / after optimization) for the CME 213 final report.

Kernel performance values come from Table~\ref{tab:kernels} in the report,
which were read from the Nsight Compute GUI (ncu_kernels_{before,after}.ncu-rep).
Arithmetic intensities are derived from algorithm analysis; the GEMM and decode
values (8, 64, 0.5 FLOP/byte) are stated explicitly in the report text.

Outputs: plots/roofline_before.pdf  and  plots/roofline_after.pdf
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

# ---------------------------------------------------------------------------
# Hardware constants (Quadro RTX 6000, sm_75)
# ---------------------------------------------------------------------------
PEAK_TFLOPS = 16.31          # FP32
PEAK_GBPS   = 672.0          # HBM bandwidth
RIDGE = PEAK_TFLOPS * 1e12 / (PEAK_GBPS * 1e9)   # ≈ 24.27 FLOP/byte

def bw_tflops(gbps, ai):
    """Achieved TFLOPS given bandwidth (GB/s) and arithmetic intensity (FLOP/byte)."""
    return gbps * ai / 1e3

# ---------------------------------------------------------------------------
# Arithmetic intensities (FLOP / byte)
# GEMM values stated in the paper (T/4 for shared-memory tile, 64 for reg tile).
# Others from algorithm analysis (reads + writes to/from HBM, FLOPs per element).
# ---------------------------------------------------------------------------
AI = {
    "GEMM_before"  : 8.0,       # shared-memory tiling, T=32  → T/4 = 8
    "GEMM_after"   : 64.0,      # 8×8 register sub-tile       → 64
    "FlashAttn"    : 2000.0,    # S=2048, H=8, D=64: ~4S²HD FLOPs / 4SHD bytes = S
    "GELU"         : 1.0,       # ~8 FLOPs / 8 bytes (read+write)
    "LayerNorm"    : 0.58,      # ~7 FLOPs / 12 bytes (two-pass, reads element twice)
    "Softmax"      : 0.50,      # ~4 FLOPs / 8 bytes
    "CE"           : 0.63,      # ~5 FLOPs / 8 bytes
    "Decode"       : 0.50,      # stated in paper: ≈0.5 FLOP/byte
}

# ---------------------------------------------------------------------------
# Kernel data: (display_label, arithmetic_intensity, achieved_TFLOPS, color, marker)
# All bandwidth values (GB/s) taken from Table~\ref{tab:kernels} in the report.
# ---------------------------------------------------------------------------
BEFORE = [
    ("GEMM\n(shared-mem tile)",  AI["GEMM_before"], 2.20,                           "#2166ac", "s"),
    ("Flash Attn\n(B=1, S=128)", AI["FlashAttn"],   1.63,                           "#5aae61", "^"),
    ("GELU\n(scalar)",           AI["GELU"],        bw_tflops(114, AI["GELU"]),     "#d6604d", "o"),
    ("LayerNorm\n(scalar)",      AI["LayerNorm"],   bw_tflops(110, AI["LayerNorm"]),"#f4a582", "o"),
    ("Softmax\n(scalar)",        AI["Softmax"],     bw_tflops(121, AI["Softmax"]),  "#762a83", "o"),
]

AFTER = [
    ("GEMM\n(register tile)",    AI["GEMM_after"],  9.56,                           "#2166ac", "s"),
    ("Flash Attn\n(B=1, S=2048)",AI["FlashAttn"],   2.09,                           "#5aae61", "^"),
    ("GELU\n(float4)",           AI["GELU"],        bw_tflops(471, AI["GELU"]),     "#d6604d", "o"),
    ("LayerNorm\n(float4)",      AI["LayerNorm"],   bw_tflops(432, AI["LayerNorm"]),"#f4a582", "o"),
    ("Softmax\n(float4)",        AI["Softmax"],     bw_tflops(431, AI["Softmax"]),  "#762a83", "o"),
    ("Cross-Entropy\n(float4)",  AI["CE"],          bw_tflops(472, AI["CE"]),       "#9970ab", "o"),
    ("Decode\n(paged, B=64)",    AI["Decode"],      bw_tflops(435, AI["Decode"]),   "#1b7837", "D"),
]

# ---------------------------------------------------------------------------
# Per-kernel label offset (dx_factor, dy_factor) relative to the point,
# in log-space fractions.  Positive dx_factor moves the label right.
# ---------------------------------------------------------------------------
LABEL_OFFSETS_BEFORE = {
    "GEMM\n(shared-mem tile)":  (2.0, 0.6),
    "Flash Attn\n(B=1, S=128)": (0.35, 2.5),
    "GELU\n(scalar)":           (2.2, 1.5),
    "LayerNorm\n(scalar)":      (2.2, 0.35),
    "Softmax\n(scalar)":        (0.18, 2.8),
}

LABEL_OFFSETS_AFTER = {
    "GEMM\n(register tile)":     (2.0, 0.55),
    "Flash Attn\n(B=1, S=2048)": (0.35, 2.5),
    "GELU\n(float4)":            (2.5, 1.5),
    "LayerNorm\n(float4)":       (2.5, 0.55),
    "Softmax\n(float4)":         (0.18, 2.8),
    "Cross-Entropy\n(float4)":   (0.18, 0.25),
    "Decode\n(paged, B=64)":     (0.18, 0.2),
}

# ---------------------------------------------------------------------------
def make_plot(kernels, label_offsets, title, out_path, xmin=0.12, xmax=None):
    if xmax is None:
        xmax = max(ai for _, ai, *_ in kernels) * 4.0

    fig, ax = plt.subplots(figsize=(7, 5))

    # ── Roofline ──────────────────────────────────────────────────────────
    x = np.logspace(np.log10(xmin * 0.5), np.log10(xmax * 2), 3000)
    mem_roof  = PEAK_GBPS * x / 1e3   # TFLOPS
    comp_roof = np.full_like(x, PEAK_TFLOPS)
    roof      = np.minimum(mem_roof, comp_roof)

    ax.plot(x, roof, "k-", linewidth=2.0, zorder=8, label="Roofline")

    # Ridge and peak labels
    ax.axvline(x=RIDGE, color="0.65", linestyle=":", linewidth=0.8)
    ax.text(RIDGE * 0.78, ax.get_ylim()[0] if False else 0.014,
            f"ridge\n{RIDGE:.0f} F/B",
            fontsize=7, ha="right", va="bottom", color="0.50")

    # HBM slope label (along the sloped part)
    x_bw_label = xmin * 2.5
    y_bw_label = PEAK_GBPS * x_bw_label / 1e3
    angle_deg  = np.degrees(np.arctan2(
        np.log10(PEAK_GBPS * x_bw_label * 2 / 1e3) - np.log10(y_bw_label),
        np.log10(x_bw_label * 2)  - np.log10(x_bw_label)
    )) * 0.45   # rough visual angle on log-log
    ax.text(x_bw_label, y_bw_label * 0.38,
            f"{PEAK_GBPS:.0f} GB/s", fontsize=8,
            rotation=30, ha="left", va="top", color="black")

    # Compute ceiling label
    ax.text(xmax * 0.88, PEAK_TFLOPS * 1.08,
            f"{PEAK_TFLOPS} TFLOPS (FP32 peak)",
            fontsize=8, ha="right", va="bottom")

    # ── Kernel points and labels ───────────────────────────────────────────
    for (label, ai, perf, color, marker) in kernels:
        ax.scatter(ai, perf, color=color, s=90, marker=marker,
                   zorder=15, edgecolors="white", linewidths=0.6)

        dx_f, dy_f = label_offsets.get(label, (2.2, 1.0))
        xt = ai  * dx_f
        yt = perf * dy_f
        # clamp to plot bounds
        xt = min(max(xt, xmin * 1.2), xmax * 0.88)
        yt = min(max(yt, 0.011), PEAK_TFLOPS * 1.3)

        ax.annotate(
            label, xy=(ai, perf), xytext=(xt, yt),
            fontsize=7.5, color=color, va="center", ha="left",
            arrowprops=dict(arrowstyle="-", color=color, lw=0.8,
                            connectionstyle="arc3,rad=0.0"),
        )

    # ── Flash Attention parallelism note (after plot only) ────────────────
    for (label, ai, perf, color, _) in kernels:
        if "Attn" in label:
            ax.annotate(
                "parallelism-\nlimited\n(8 blocks / 72 SMs)",
                xy=(ai, perf), xytext=(ai * 0.3, perf * 0.3),
                fontsize=6.5, color="0.45", ha="right",
                arrowprops=dict(arrowstyle="->", color="0.65", lw=0.7),
            )

    # ── Axes formatting ───────────────────────────────────────────────────
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(0.012, 25)
    ax.set_xlabel("Arithmetic Intensity  (FLOP / byte)", fontsize=10)
    ax.set_ylabel("Performance  (TFLOPS)", fontsize=10)
    ax.set_title(title, fontsize=10.5, pad=8)

    # Nice tick formatting
    from matplotlib.ticker import LogLocator, NullFormatter
    ax.xaxis.set_major_locator(LogLocator(base=10, numticks=8))
    ax.yaxis.set_major_locator(LogLocator(base=10, numticks=8))
    ax.grid(True, which="major", linestyle="--", linewidth=0.45, alpha=0.6)
    ax.grid(True, which="minor", linestyle=":",  linewidth=0.25, alpha=0.4)

    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    os.makedirs("plots", exist_ok=True)

    make_plot(
        BEFORE, LABEL_OFFSETS_BEFORE,
        title="Roofline — Before Optimization  (RTX 6000, sm\\_75)",
        out_path="plots/roofline_before.png",
        xmin=0.12, xmax=120,
    )

    make_plot(
        AFTER, LABEL_OFFSETS_AFTER,
        title="Roofline — After Optimization  (RTX 6000, sm\\_75)",
        out_path="plots/roofline_after.png",
        xmin=0.12, xmax=8000,
    )
