#!/usr/bin/env python3
"""
plot_scaling.py

Parses PERF: lines from a run_distributed.sh log and produces:
  - plots/strong_scaling.png  (speedup + parallel efficiency vs P)
  - plots/weak_scaling.png    (normalized step time + efficiency vs P)
  - results_tables/scaling_table.tex  (LaTeX table)

Usage:
    python3 scripts/plot_scaling.py logs/distributed_<jobid>.out

Optional flags:
    --strong-total-b INT   total batch for strong scaling (default: 16)
    --weak-local-b   INT   per-rank batch for weak scaling (default: 4)
    --plot-dir       PATH  output directory for PNGs (default: plots/)
    --table-dir      PATH  output directory for .tex  (default: results_tables/)

PERF line format (emitted by train_distributed):
    PERF: ranks=N local_B=B S=S C=C layers=L avg_step_ms=T comm_fraction=F
"""

import argparse
import os
import sys

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
except ImportError:
    sys.exit("matplotlib/numpy not found.  pip install matplotlib numpy")


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
                if r and "ranks" in r and "avg_step_ms" in r:
                    records.append(r)
    except FileNotFoundError:
        sys.exit(f"Log file not found: {path}")
    if not records:
        sys.exit(f"No PERF: lines found in {path}")
    return records


def split_records(records: list[dict], strong_total_b: int, weak_local_b: int):
    """
    Separate records into strong-scaling and weak-scaling groups.

    Strong: ranks * local_B == strong_total_b
    Weak:   local_B == weak_local_b

    When the same config appears multiple times (PERF summary re-runs),
    keep the median avg_step_ms per rank count to reduce noise.
    """
    from statistics import median

    strong_raw: dict[int, list[float]] = {}
    weak_raw:   dict[int, list[float]] = {}

    for r in records:
        ranks   = int(r["ranks"])
        local_b = int(r["local_B"])
        total_b = ranks * local_b
        ms      = r["avg_step_ms"]

        if total_b == strong_total_b:
            strong_raw.setdefault(ranks, []).append(ms)
        if local_b == weak_local_b:
            weak_raw.setdefault(ranks, []).append(ms)

    strong = {p: median(v) for p, v in strong_raw.items()}
    weak   = {p: median(v) for p, v in weak_raw.items()}
    return strong, weak


# ---------------------------------------------------------------------------
# Compute speedup / efficiency
# ---------------------------------------------------------------------------

def scaling_metrics(times: dict[int, float]) -> tuple[list, list, list, list]:
    """Return (ranks, step_ms, speedup, efficiency) lists sorted by rank count."""
    ranks_sorted = sorted(times)
    t1 = times[1] if 1 in times else times[min(times)]
    ms  = [times[p] for p in ranks_sorted]
    su  = [t1 / times[p] for p in ranks_sorted]
    eff = [t1 / (p * times[p]) * 100 for p in ranks_sorted]
    return ranks_sorted, ms, su, eff


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

COLORS = {"data": "#2E86AB", "ideal": "#cccccc"}


def _ideal_line(ax, ranks, values, label="Ideal", linestyle="--", color=COLORS["ideal"]):
    """Draw the ideal (linear) reference from the first data point."""
    r0, v0 = ranks[0], values[0]
    ideal = [v0 * r / r0 for r in ranks]
    ax.plot(ranks, ideal, linestyle=linestyle, color=color, linewidth=1.2,
            label=label, zorder=1)
    return ideal


def _style_ax(ax, xlabel, ylabel, title):
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(title, fontsize=10)
    ax.set_xticks([1, 2, 4])
    ax.set_xticklabels(["1", "2", "4"])
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)


# ---------------------------------------------------------------------------
# Strong scaling figure
# ---------------------------------------------------------------------------

def plot_strong(strong: dict[int, float], out_path: str, total_b: int):
    if not strong or 1 not in strong:
        print("WARNING: strong scaling data missing P=1 baseline — skipping figure.")
        return

    ranks, ms, su, eff = scaling_metrics(strong)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    fig.suptitle(
        f"Strong Scaling  (total batch = {total_b}, S=128, C=256, 6 layers)\n"
        "Quadro RTX 6000 cluster — 1 GPU/rank",
        fontsize=10
    )

    # ---- Left: speedup ----
    _ideal_line(ax1, ranks, [1.0] * len(ranks), label="Ideal (linear)")
    ax1.plot(ranks, su, "o-", color=COLORS["data"], linewidth=2, markersize=7,
             label="Measured speedup", zorder=2)
    for p, s in zip(ranks, su):
        ax1.annotate(f"{s:.2f}×", xy=(p, s), xytext=(4, 5),
                     textcoords="offset points", fontsize=8)
    _style_ax(ax1, "Number of GPUs (P)", "Speedup  T(1) / T(P)",
              "Speedup vs P")
    ax1.set_ylim(0, max(ranks) * 1.15)

    # ---- Right: efficiency ----
    ax2.axhline(100, linestyle="--", color=COLORS["ideal"], linewidth=1.2,
                label="Ideal (100%)")
    ax2.plot(ranks, eff, "s-", color=COLORS["data"], linewidth=2, markersize=7,
             label="Parallel efficiency", zorder=2)
    for p, e in zip(ranks, eff):
        ax2.annotate(f"{e:.1f}%", xy=(p, e), xytext=(4, -12),
                     textcoords="offset points", fontsize=8)
    _style_ax(ax2, "Number of GPUs (P)",
              "Parallel Efficiency  T(1) / (P · T(P))  [%]",
              "Parallel Efficiency vs P")
    ax2.set_ylim(0, 115)

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Strong scaling -> {out_path}")
    plt.close()


# ---------------------------------------------------------------------------
# Weak scaling figure
# ---------------------------------------------------------------------------

def plot_weak(weak: dict[int, float], out_path: str, local_b: int):
    if not weak or 1 not in weak:
        print("WARNING: weak scaling data missing P=1 baseline — skipping figure.")
        return

    ranks, ms, _, _ = scaling_metrics(weak)
    t1 = weak[1]
    norm = [weak[p] / t1 for p in ranks]                  # normalized step time
    eff  = [t1 / weak[p] * 100 for p in ranks]            # efficiency = T(1)/T(P)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    fig.suptitle(
        f"Weak Scaling  (local batch = {local_b}/rank, S=128, C=256, 6 layers)\n"
        "Quadro RTX 6000 cluster — 1 GPU/rank",
        fontsize=10
    )

    # ---- Left: normalized step time (ideal = flat at 1.0) ----
    ax1.axhline(1.0, linestyle="--", color=COLORS["ideal"], linewidth=1.2,
                label="Ideal (no overhead)")
    ax1.plot(ranks, norm, "o-", color=COLORS["data"], linewidth=2, markersize=7,
             label="Measured (normalized)", zorder=2)
    for p, n in zip(ranks, norm):
        ax1.annotate(f"{n:.2f}×", xy=(p, n), xytext=(4, 5),
                     textcoords="offset points", fontsize=8)
    _style_ax(ax1, "Number of GPUs (P)", "Normalized step time  T(P) / T(1)",
              "Step Time vs P  (ideal = 1.0)")
    ymax = max(norm) * 1.25 if max(norm) > 1.0 else 1.5
    ax1.set_ylim(0, ymax)

    # ---- Right: weak-scaling efficiency ----
    ax2.axhline(100, linestyle="--", color=COLORS["ideal"], linewidth=1.2,
                label="Ideal (100%)")
    ax2.plot(ranks, eff, "s-", color=COLORS["data"], linewidth=2, markersize=7,
             label="Weak-scaling efficiency", zorder=2)
    for p, e in zip(ranks, eff):
        ax2.annotate(f"{e:.1f}%", xy=(p, e), xytext=(4, -12),
                     textcoords="offset points", fontsize=8)
    _style_ax(ax2, "Number of GPUs (P)",
              "Weak-scaling Efficiency  T(1) / T(P)  [%]",
              "Efficiency vs P")
    ax2.set_ylim(0, 115)

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Weak scaling   -> {out_path}")
    plt.close()


# ---------------------------------------------------------------------------
# LaTeX table
# ---------------------------------------------------------------------------

def make_latex_table(strong: dict, weak: dict, strong_total_b: int,
                     weak_local_b: int, out_path: str):
    lines = [
        r"\begin{table}[h]",
        r"  \centering",
        r"  \small",
        r"  \begin{tabular}{r r r r | r r r}",
        r"    \toprule",
        r"    \multicolumn{4}{c|}{\textbf{Strong Scaling} (total $B="
        + str(strong_total_b) + r"$)} &"
        r" \multicolumn{3}{c}{\textbf{Weak Scaling} (local $B="
        + str(weak_local_b) + r"$/rank)} \\",
        r"    $P$ & Step (ms) & Speedup & Efficiency &"
        r" Step (ms) & Norm.\ time & Efficiency \\",
        r"    \midrule",
    ]

    all_ranks = sorted(set(list(strong.keys()) + list(weak.keys())))
    t1_strong = strong.get(1, None)
    t1_weak   = weak.get(1, None)

    for p in all_ranks:
        s_ms  = strong.get(p)
        w_ms  = weak.get(p)

        if s_ms is not None and t1_strong:
            su  = t1_strong / s_ms
            eff = su / p * 100
            s_str = f"{s_ms:.1f} & {su:.2f}\\times & {eff:.1f}\\%"
        else:
            s_str = r"-- & -- & --"

        if w_ms is not None and t1_weak:
            norm = w_ms / t1_weak
            weff = t1_weak / w_ms * 100
            w_str = f"{w_ms:.1f} & {norm:.2f}\\times & {weff:.1f}\\%"
        else:
            w_str = r"-- & -- & --"

        lines.append(f"    {p} & {s_str} & {w_str} \\\\")

    lines += [
        r"    \bottomrule",
        r"  \end{tabular}",
        r"  \caption{Strong and weak scaling on up to 4 Quadro RTX 6000 GPUs."
        r" Strong scaling: total batch $B=" + str(strong_total_b) + r"$, $S=128$, $C=256$, 6 layers."
        r" Weak scaling: local batch $B_\text{local}=" + str(weak_local_b) + r"$ per rank."
        r" Efficiency $= T(1)/(P \cdot T(P))$ (strong) and $T(1)/T(P)$ (weak).}",
        r"  \label{tab:scaling}",
        r"\end{table}",
    ]

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"LaTeX table    -> {out_path}")


# ---------------------------------------------------------------------------
# Console summary
# ---------------------------------------------------------------------------

def print_summary(strong: dict, weak: dict, strong_total_b: int, weak_local_b: int):
    print()
    print("=" * 60)
    print(f"  STRONG SCALING  (total_B={strong_total_b}, S=128, C=256, 6 layers)")
    print(f"  {'P':<6} {'Step ms':<12} {'Speedup':<12} {'Efficiency'}")
    print("  " + "-" * 44)
    t1 = strong.get(1)
    for p in sorted(strong):
        ms  = strong[p]
        su  = (t1 / ms) if t1 else 0
        eff = su / p * 100 if t1 else 0
        print(f"  {p:<6} {ms:<12.2f} {su:<12.2f} {eff:.1f}%")

    print()
    print(f"  WEAK SCALING  (local_B={weak_local_b}/rank, S=128, C=256, 6 layers)")
    print(f"  {'P':<6} {'Step ms':<12} {'Norm. time':<14} {'Efficiency'}")
    print("  " + "-" * 44)
    t1w = weak.get(1)
    for p in sorted(weak):
        ms   = weak[p]
        norm = (ms / t1w) if t1w else 0
        eff  = (t1w / ms * 100) if t1w else 0
        print(f"  {p:<6} {ms:<12.2f} {norm:<14.2f} {eff:.1f}%")
    print("=" * 60)
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Generate scaling figures from run_distributed.sh log.")
    ap.add_argument("log", help="SLURM .out log file with PERF: lines")
    ap.add_argument("--strong-total-b", type=int, default=16, metavar="B",
                    help="Total batch for strong scaling (default: 16)")
    ap.add_argument("--weak-local-b",   type=int, default=4,  metavar="B",
                    help="Per-rank batch for weak scaling (default: 4)")
    ap.add_argument("--plot-dir",   default="plots",          metavar="DIR")
    ap.add_argument("--table-dir",  default="results_tables", metavar="DIR")
    args = ap.parse_args()

    records = load_records(args.log)
    print(f"Parsed {len(records)} PERF: records from {args.log}")

    strong, weak = split_records(records, args.strong_total_b, args.weak_local_b)
    print(f"Strong scaling: P = {sorted(strong.keys())}")
    print(f"Weak   scaling: P = {sorted(weak.keys())}")

    print_summary(strong, weak, args.strong_total_b, args.weak_local_b)

    plot_strong(strong, os.path.join(args.plot_dir, "strong_scaling.png"),
                args.strong_total_b)
    plot_weak(weak,     os.path.join(args.plot_dir, "weak_scaling.png"),
              args.weak_local_b)
    make_latex_table(strong, weak, args.strong_total_b, args.weak_local_b,
                     os.path.join(args.table_dir, "scaling_table.tex"))


if __name__ == "__main__":
    main()
