"""
Two-panel figure for the PagedAttention benchmark.

Left:  Memory utilization (useful KV bytes / allocated KV bytes) vs batch size.
Right: Decode throughput (tokens/sec) vs batch size.

Usage:
    python scripts/plot_paged.py logs/paged_bench_<jobid>.out

Expected PERF: lines (one per (kernel, B) pair):
    PERF: kernel=paged B=64 H=8 D=64 S_max=128 mean_len=64.0 \
          useful_bytes=... alloc_bytes=... util=0.96 step_ms=0.42 tps=152000
    PERF: kernel=dense ...

The log may contain multiple sweeps (different H/D/S_max/mean_len combinations);
the script picks the first sweep with the largest data unless filtered.
"""

import sys
import re
import os
import matplotlib.pyplot as plt
from collections import defaultdict


def parse_log(filename):
    """Return dict[(H, D, S_max, mean_len_int)][kernel] -> list of (B, util, tps, step_ms)."""
    pat = re.compile(
        r"PERF:\s+kernel=(\w+)\s+B=(\d+)\s+H=(\d+)\s+D=(\d+)\s+"
        r"S_max=(\d+)\s+mean_len=([\d.]+)\s+"
        r"useful_bytes=(\d+)\s+alloc_bytes=(\d+)\s+util=([\d.]+)\s+"
        r"step_ms=([\d.]+)\s+tps=([\d.]+)"
    )
    data = defaultdict(lambda: defaultdict(list))
    with open(filename) as f:
        for line in f:
            m = pat.search(line)
            if not m:
                continue
            kernel  = m.group(1)
            B       = int(m.group(2))
            H       = int(m.group(3))
            D       = int(m.group(4))
            S_max   = int(m.group(5))
            mean    = float(m.group(6))
            util    = float(m.group(9))
            step_ms = float(m.group(10))
            tps     = float(m.group(11))
            key = (H, D, S_max, round(mean))
            data[key][kernel].append((B, util, tps, step_ms))
    # Sort points by B.
    for k in data:
        for kernel in data[k]:
            data[k][kernel].sort()
    return data


def main():
    if len(sys.argv) < 2:
        print("Usage: python plot_paged.py <log_file> [out=plots/paged_attention.png]")
        sys.exit(1)

    log = sys.argv[1]
    out = sys.argv[2] if len(sys.argv) > 2 else "plots/paged_attention.png"
    os.makedirs(os.path.dirname(out), exist_ok=True)

    sweeps = parse_log(log)
    if not sweeps:
        print("No PERF: lines parsed. Did the benchmark run successfully?")
        sys.exit(1)

    # Pick the sweep with the most points (typically the default config).
    key = max(sweeps, key=lambda k: sum(len(v) for v in sweeps[k].values()))
    H, D, S_max, mean = key
    print(f"Plotting sweep: H={H} D={D} S_max={S_max} mean_len={mean}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

    colors = {"paged": "firebrick", "dense": "steelblue"}
    labels = {"paged": "PagedAttention", "dense": "Naive padded"}

    for kernel in ("dense", "paged"):
        if kernel not in sweeps[key]:
            continue
        pts = sweeps[key][kernel]
        Bs   = [p[0] for p in pts]
        util = [p[1] * 100 for p in pts]
        tps  = [p[2] for p in pts]
        ax1.plot(Bs, util, "o-", color=colors[kernel], label=labels[kernel])
        ax2.plot(Bs, tps,  "o-", color=colors[kernel], label=labels[kernel])

    ax1.set_xscale("log", base=2)
    ax1.set_xlabel("Concurrent requests (B)")
    ax1.set_ylabel("KV-cache memory utilization (%)")
    ax1.set_ylim(0, 105)
    ax1.set_title(f"Memory utilization  (H={H}, D={D}, $S_{{\\max}}$={S_max}, $\\bar{{L}}$={mean})")
    ax1.grid(True, which="both", alpha=0.3)
    ax1.legend()

    ax2.set_xscale("log", base=2)
    ax2.set_yscale("log")
    ax2.set_xlabel("Concurrent requests (B)")
    ax2.set_ylabel("Decode throughput (tokens / s)")
    ax2.set_title("Decode throughput vs batch size")
    ax2.grid(True, which="both", alpha=0.3)
    ax2.legend()

    plt.tight_layout()
    plt.savefig(out, dpi=150)
    print(f"Saved {out}")

    # Also dump a small summary table for the report writeup.
    print("\n  B | naive util |  paged util | dense tok/s | paged tok/s")
    print("----+------------+-------------+-------------+------------")
    paged = {p[0]: p for p in sweeps[key].get("paged", [])}
    dense = {p[0]: p for p in sweeps[key].get("dense", [])}
    for B in sorted(set(paged) | set(dense)):
        nd = dense.get(B, (B, 0, 0, 0))
        np_ = paged.get(B, (B, 0, 0, 0))
        print(f"{B:4d} | {nd[1]*100:9.1f}% | {np_[1]*100:10.1f}% | "
              f"{nd[2]:11.0f} | {np_[2]:10.0f}")


if __name__ == "__main__":
    main()
