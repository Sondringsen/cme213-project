"""
Fit alpha+beta*n to MPI_Allreduce timing data and produce a log-log plot.

Usage:
    python scripts/plot_comm.py logs/allreduce_bench_<jobid>.out

Expected input lines (one per config):
    ALLREDUCE: ranks=4 n_bytes=4096 avg_us=12.34
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

GRADIENT_BYTES = 19 * 1024 * 1024  # ~19 MB actual training gradient size


def parse_log(filename):
    data = {}
    with open(filename) as f:
        for line in f:
            if not line.startswith("ALLREDUCE:"):
                continue
            parts = line.strip().split()
            ranks    = int(parts[1].split("=")[1])
            n_bytes  = int(parts[2].split("=")[1])
            avg_us   = float(parts[3].split("=")[1])
            data.setdefault(ranks, []).append((n_bytes, avg_us))
    return data


def alpha_beta(n, alpha, beta):
    return alpha + beta * n


def main():
    if len(sys.argv) < 2:
        print("Usage: python plot_comm.py <log_file>")
        sys.exit(1)

    data = parse_log(sys.argv[1])

    fig, ax = plt.subplots(figsize=(6, 4))
    colors = {1: "gray", 2: "steelblue", 4: "firebrick"}

    for ranks in sorted(data.keys()):
        pts = sorted(data[ranks])
        n_bytes = np.array([p[0] for p in pts])
        t_us    = np.array([p[1] for p in pts])

        if ranks > 1:
            popt, _ = curve_fit(alpha_beta, n_bytes, t_us, p0=[10.0, 1e-7])
            alpha_us, beta_us_per_byte = popt
            bw_GBps = 1.0 / beta_us_per_byte / 1e3  # convert us/byte -> GB/s
            print(f"P={ranks}: alpha={alpha_us:.1f} us, "
                  f"beta={beta_us_per_byte*1e9:.3f} us/GB = {bw_GBps:.2f} GB/s")

            # Predicted cost at actual training gradient size
            pred_us = alpha_beta(GRADIENT_BYTES, alpha_us, beta_us_per_byte)
            print(f"  Predicted allreduce at {GRADIENT_BYTES/1e6:.1f} MB: {pred_us/1e3:.2f} ms")

        ax.loglog(n_bytes, t_us, "o-", color=colors[ranks],
                  label=f"Measured P={ranks}")

    # Mark the actual training gradient size
    ax.axvline(GRADIENT_BYTES, color="k", linestyle=":", linewidth=1, alpha=0.6)
    ax.text(GRADIENT_BYTES * 1.05, ax.get_ylim()[0] * 1.5,
            "training\ngradients\n(~19 MB)", fontsize=7, va="bottom")

    ax.set_xlabel("Message size (bytes)")
    ax.set_ylabel("Latency (µs)")
    ax.set_title("MPI_Allreduce: α+βn fit vs Ring lower bound")
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.3)
    plt.tight_layout()

    out = "plots/comm_alpha_beta.png"
    plt.savefig(out, dpi=150)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
