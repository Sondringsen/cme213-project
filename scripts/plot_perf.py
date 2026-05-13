#!/usr/bin/env python3
"""
plot_perf.py

Parses PERF: lines from a SLURM output log and produces:
  1. A roofline plot showing every kernel against the hardware ceiling.
  2. A LaTeX performance table for the Milestone 3 report.

Usage:
    python3 scripts/plot_perf.py logs/run_<jobid>.out

Optional flags:
    --peak-gflops FLOAT   FP32 peak in GFLOPS   (default: 16310, Quadro RTX 6000)
    --peak-bw     FLOAT   Peak DRAM bandwidth GB/s (default: 672)
    --out-dir     PATH    Output directory         (default: reports/)

Outputs:
    plots/roofline.png             -- roofline plot  (copy to Overleaf as plots/roofline.png)
    results_tables/perf_table.tex  -- LaTeX table    (copy to Overleaf as results_tables/perf_table.tex)

Roofline model recap
--------------------
Every kernel has an arithmetic intensity (AI = FLOPs / bytes).  The
hardware has two ceilings:

    attainable_gflops = min(peak_gflops,  AI * peak_bw)

                         ^--- compute ceiling   ^--- bandwidth ceiling

If a kernel's AI is above the "ridge point" (peak_gflops / peak_bw),
compute is the bottleneck. Below it, bandwidth is the bottleneck. A dot
on the roofline plot shows where a kernel actually lands relative to the
theoretical ceiling.

Requirements:
    pip install matplotlib numpy
"""

import argparse
import os
import sys

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patheffects as pe
    import numpy as np
except ImportError:
    sys.exit("matplotlib/numpy not found.  pip install matplotlib numpy")


# ---------------------------------------------------------------------------
# Hardware defaults: Quadro RTX 6000 (Turing sm_75), matching run.sh
# ---------------------------------------------------------------------------
DEFAULT_PEAK_GFLOPS = 16310.0   # FP32 peak GFLOPS
DEFAULT_PEAK_BW_GBS =   672.0   # DRAM bandwidth GB/s


# ---------------------------------------------------------------------------
# Kernel metadata
# ---------------------------------------------------------------------------

# Visual style per kernel group
STYLE = {
    "gemm":          dict(color="#E05C3A", marker="D", label="GEMM (tiled)"),
    "gemm_naive":    dict(color="#F4A460", marker="D", label="GEMM (naive)"),
    "attention":     dict(color="#C0392B", marker="*", label="Flash Attention"),
    "layernorm":     dict(color="#2E86AB", marker="o", label="LayerNorm"),
    "softmax":       dict(color="#1A6B5A", marker="s", label="Softmax"),
    "gelu":          dict(color="#8E44AD", marker="^", label="GELU"),
    "cross_entropy": dict(color="#2980B9", marker="P", label="Cross-Entropy"),
}

# How to compute arithmetic intensity from the PERF parameters.
# AI = FLOPs / bytes_moved_to_from_DRAM (minimum, not counting cache effects).
def compute_ai(r: dict) -> float | None:
    k = r.get("kernel")
    if k in ("gemm", "gemm_naive"):
        M, N, K = r["M"], r["N"], r["K"]
        flops  = 2.0 * M * N * K
        bytes_ = (M*K + K*N + M*N) * 4.0   # read A, read B, write C
        return flops / bytes_
    if k == "attention":
        B, H, S, D = r["B"], r["H"], r["S"], r["D"]
        flops  = 4.0 * B * H * S * S * D   # QK^T + PV (each 2*S^2*D per head)
        bytes_ = 4.0 * B * H * S * D * 4.0  # Q, K, V, O
        return flops / bytes_
    if k == "layernorm":
        N, H = r["N"], r["H"]
        flops  = 8.0 * N * H               # mean, var, normalize, affine
        # Our kernel does 3 reads of x (passes 1, 2a, 2b) + gamma + beta + write y
        bytes_ = (3*N*H + H + H + N*H) * 4.0
        return flops / bytes_
    if k == "softmax":
        N, V = r["N"], r["V"]
        flops  = 3.0 * N * V               # max pass, exp+sum pass, divide pass
        bytes_ = (3*N*V + N*V) * 4.0       # 3 reads + 1 write
        return flops / bytes_
    if k == "gelu":
        N = r["N"]
        flops  = 6.0 * N                   # ~6 ops per element (mul, erff, add…)
        bytes_ = 2.0 * N * 4.0             # 1 read + 1 write
        return flops / bytes_
    if k == "cross_entropy":
        N, V = r["N"], r["V"]
        flops  = 3.0 * N * V
        bytes_ = (2*N*V + N) * 4.0         # 2 reads of logits + write losses
        return flops / bytes_
    return None


def achieved_gflops(r: dict) -> float | None:
    """Return achieved GFLOPS from either a direct measurement or bandwidth × AI."""
    if "gflops" in r:
        return float(r["gflops"])
    if "bandwidth_gbs" in r:
        ai = compute_ai(r)
        if ai:
            return float(r["bandwidth_gbs"]) * ai
    return None


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def parse_perf_line(line: str) -> dict | None:
    line = line.strip()
    if not line.startswith("PERF:"):
        return None
    record: dict = {}
    for token in line[len("PERF:"):].split():
        key, _, val = token.partition("=")
        try:
            record[key] = float(val)
        except ValueError:
            record[key] = val
    return record


def load_records(path: str) -> list[dict]:
    records = []
    try:
        with open(path) as f:
            for line in f:
                r = parse_perf_line(line)
                if r:
                    records.append(r)
    except FileNotFoundError:
        sys.exit(f"Log file not found: {path}")
    if not records:
        sys.exit(f"No PERF: lines found in {path}\n"
                 "Build the tests with the PERF: printf lines in place first.")
    return records


def best_per_kernel(records: list[dict]) -> dict[str, dict]:
    """Keep only the highest-throughput result per kernel name."""
    best: dict[str, dict] = {}
    for r in records:
        name = r.get("kernel")
        if not isinstance(name, str) or name not in STYLE:
            continue
        val = achieved_gflops(r) or 0.0
        if val > achieved_gflops(best.get(name, {})) if name in best else True:
            best[name] = r
    return best


# ---------------------------------------------------------------------------
# Roofline plot
# ---------------------------------------------------------------------------

def make_roofline(best: dict, peak_gflops: float, peak_bw: float, out_path: str):
    """
    Draw the roofline model and overlay a dot for each kernel.

    The roofline ceiling is:
        attainable(AI) = min(peak_gflops,  AI * peak_bw)

    which forms an upside-down L-shape on a log-log plot:
        - Left portion (AI < ridge):  a rising slope of gradient peak_bw
        - Right portion (AI > ridge): a flat line at peak_gflops

    Each kernel's dot lands at (AI, achieved_gflops). The gap between the
    dot and the ceiling shows how much performance is still on the table.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # ---- Build the roofline curve ----
    ridge = peak_gflops / peak_bw          # FLOP/byte where the two lines meet
    ai_min, ai_max = 0.05, 2000.0          # x-axis range

    ai_curve = np.logspace(np.log10(ai_min), np.log10(ai_max), 500)
    roof     = np.minimum(peak_gflops, ai_curve * peak_bw)

    ax.plot(ai_curve, roof, color="black", linewidth=2.0, zorder=3,
            label="Hardware roofline")

    # Annotate the two segments of the roof
    ax.text(ridge * 0.25, peak_bw * ridge * 0.25 * 0.6,
            f"BW-bound\nslope = {peak_bw:.0f} GB/s",
            rotation=np.degrees(np.arctan(np.log10(peak_bw))),
            fontsize=8, color="black", ha="center")
    ax.axhline(peak_gflops, color="black", linewidth=0.6, linestyle="--", alpha=0.4)
    ax.text(ai_max * 0.6, peak_gflops * 1.08,
            f"Compute peak  {peak_gflops/1000:.1f} TFLOPS (FP32)",
            fontsize=8, color="black")

    # Ridge point
    ax.axvline(ridge, color="grey", linewidth=0.8, linestyle=":", alpha=0.6)
    ax.text(ridge * 1.05, peak_gflops * 0.55,
            f"Ridge\n{ridge:.1f} FLOP/B", fontsize=7.5, color="grey")

    # ---- Plot each kernel ----
    for name, r in sorted(best.items()):
        ai  = compute_ai(r)
        gfl = achieved_gflops(r)
        if ai is None or gfl is None:
            continue
        style = STYLE.get(name, {})
        ax.scatter(ai, gfl,
                   color=style.get("color", "grey"),
                   marker=style.get("marker", "o"),
                   s=120, zorder=5,
                   label=f"{style.get('label', name)}  ({gfl:.0f} GFLOPS, {gfl/peak_gflops*100:.1f}% of peak)")

        # Vertical dashed line from dot up to the roofline ceiling
        ceil = min(peak_gflops, ai * peak_bw)
        ax.plot([ai, ai], [gfl, ceil], color=style.get("color", "grey"),
                linewidth=0.8, linestyle="--", alpha=0.5, zorder=4)

        # Label each dot with the kernel name
        ax.annotate(style.get("label", name).replace("\n", " "),
                    xy=(ai, gfl), xytext=(0, 8),
                    textcoords="offset points",
                    fontsize=7.5, ha="center",
                    color=style.get("color", "grey"),
                    path_effects=[pe.withStroke(linewidth=2, foreground="white")])

    # ---- Axes ----
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Arithmetic Intensity (FLOP / byte)", fontsize=11)
    ax.set_ylabel("Achieved Throughput (GFLOPS)", fontsize=11)
    ax.set_title("Roofline Model — CME 213 Final Project, Milestone 3\n"
                 "Quadro RTX 6000 (Turing sm_75)", fontsize=11)
    ax.set_xlim(ai_min, ai_max)
    ax.set_ylim(1, peak_gflops * 3)
    ax.grid(True, which="both", alpha=0.25)

    # Legend outside the plot so it doesn't obscure the dots
    ax.legend(loc="upper left", fontsize=8, framealpha=0.85)

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight", format="png")
    print(f"Roofline -> {out_path}")


# ---------------------------------------------------------------------------
# LaTeX table
# ---------------------------------------------------------------------------

def size_label(r: dict) -> str:
    k = r["kernel"]
    if k in ("gemm", "gemm_naive"):
        M, N, K = int(r["M"]), int(r["N"]), int(r["K"])
        return f"$M=N=K={M}$" if M == N == K else f"$M={M}, N={N}, K={K}$"
    if k == "layernorm":
        return f"$N={int(r['N'])},\\ H={int(r['H'])}$"
    if k == "softmax":
        return f"$N={int(r['N'])},\\ V={int(r['V'])}$"
    if k == "gelu":
        return f"$N={int(r['N']):,}$"
    if k == "cross_entropy":
        return f"$N={int(r['N'])},\\ V={int(r['V'])}$"
    if k == "attention":
        return f"$B={int(r['B'])},S={int(r['S'])},D={int(r['D'])}$"
    return ""


def make_latex_table(best: dict, peak_gflops: float, peak_bw: float, out_path: str):
    """Write a booktabs LaTeX table with one row per kernel."""
    order = ["gemm", "gemm_naive", "attention",
             "layernorm", "softmax", "gelu", "cross_entropy"]

    rows = []
    for k in order:
        if k not in best:
            continue
        r    = best[k]
        ms   = r["ms"]
        size = size_label(r)
        ai   = compute_ai(r) or 0.0
        gfl  = achieved_gflops(r) or 0.0
        ceil = min(peak_gflops, ai * peak_bw)
        pct  = gfl / ceil * 100 if ceil > 0 else 0.0

        if k in ("gemm", "gemm_naive", "attention"):
            metric_str = f"{gfl:.1f} GFLOPS"
            bound = "Compute"
        else:
            bw  = r.get("bandwidth_gbs", gfl / ai if ai else 0.0)
            metric_str = f"{bw:.1f} GB/s"
            bound = "Memory"

        label = STYLE.get(k, {}).get("label", k).replace("\n", " ")
        label_tex = label.replace("_", r"\_")
        rows.append(
            rf"  \texttt{{{label_tex}}} & {size} & {ms:.3f} & "
            rf"{metric_str} & {ai:.1f} & {pct:.1f}\% & {bound} \\"
        )

    lines = [
        r"\begin{table}[h]",
        r"  \centering",
        r"  \small",
        r"  \begin{tabular}{lllrrrr}",
        r"    \toprule",
        r"    \textbf{Kernel} & \textbf{Problem size} & \textbf{Time (ms)}"
        r" & \textbf{Achieved} & \textbf{AI (F/B)} & \textbf{\% of ceil.} & \textbf{Bound} \\",
        r"    \midrule",
    ] + rows + [
        r"    \bottomrule",
        r"  \end{tabular}",
        r"  \caption{Best achieved performance per kernel on the Quadro RTX 6000"
        r" (Turing sm\_75, FP32 peak 16.3\,TFLOPS, DRAM 672\,GB/s)."
        r" \emph{AI} = arithmetic intensity (FLOP/byte)."
        r" \emph{\% of ceil.} = achieved / roofline ceiling for that kernel's AI.}",
        r"  \label{tab:m3_perf}",
        r"\end{table}",
    ]

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Table   -> {out_path}")


# ---------------------------------------------------------------------------
# Console analysis
# ---------------------------------------------------------------------------

def print_analysis(best: dict, peak_gflops: float, peak_bw: float):
    ridge = peak_gflops / peak_bw
    print()
    print("=" * 72)
    print("  Roofline Analysis")
    print(f"  Peak FP32: {peak_gflops/1000:.2f} TFLOPS  |  Peak BW: {peak_bw:.0f} GB/s  "
          f"|  Ridge: {ridge:.1f} FLOP/byte")
    print("=" * 72)
    print(f"  {'Kernel':<18} {'AI (F/B)':<12} {'Achieved':<16} {'Ceiling':<16} {'% of ceil':<12} {'Bound'}")
    print("  " + "-" * 70)
    order = ["gemm", "gemm_naive", "attention", "layernorm", "softmax", "gelu", "cross_entropy"]
    for k in order:
        if k not in best:
            continue
        r   = best[k]
        ai  = compute_ai(r) or 0.0
        gfl = achieved_gflops(r) or 0.0
        ceil = min(peak_gflops, ai * peak_bw)
        pct  = gfl / ceil * 100 if ceil > 0 else 0.0
        bound = "Compute" if ai > ridge else "Memory BW"
        flag  = "  <-- far from ceiling" if pct < 30 else ""
        ach_str  = f"{gfl:.1f} GFLOPS"
        ceil_str = f"{ceil:.1f} GFLOPS"
        print(f"  {k:<18} {ai:<12.2f} {ach_str:<16} {ceil_str:<16} {pct:<12.1f} {bound}{flag}")
    print()
    print("  Notes:")
    print("  - % of ceiling = achieved / min(peak_compute, AI * peak_bw)")
    print("    This is a tighter measure than % of raw hardware peak.")
    print("  - For memory-bound kernels the ceiling is AI * 672 GB/s, not 16.3 TFLOPS.")
    print("  - Kernels far below their ceiling indicate room for further optimization")
    print("=" * 72)
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(
        description="Generate roofline plot + LaTeX table from SLURM log.")
    p.add_argument("log", help="SLURM .out log file containing PERF: lines")
    p.add_argument("--peak-gflops", type=float, default=DEFAULT_PEAK_GFLOPS,
                   metavar="F", help="FP32 peak in GFLOPS (default: Quadro RTX 6000)")
    p.add_argument("--peak-bw",     type=float, default=DEFAULT_PEAK_BW_GBS,
                   metavar="B", help="DRAM bandwidth in GB/s (default: Quadro RTX 6000)")
    p.add_argument("--plot-dir",    default="plots",          metavar="DIR",
                   help="Directory for PNG figures (default: plots/)")
    p.add_argument("--table-dir",   default="results_tables", metavar="DIR",
                   help="Directory for LaTeX table (default: results_tables/)")
    args = p.parse_args()

    records = load_records(args.log)
    print(f"Parsed {len(records)} PERF: records from {args.log}")

    best = best_per_kernel(records)
    print(f"Kernels found: {', '.join(sorted(best))}")

    make_roofline(best, args.peak_gflops, args.peak_bw,
                  os.path.join(args.plot_dir, "roofline.png"))
    make_latex_table(best, args.peak_gflops, args.peak_bw,
                     os.path.join(args.table_dir, "perf_table.tex"))
    print_analysis(best, args.peak_gflops, args.peak_bw)


if __name__ == "__main__":
    main()
