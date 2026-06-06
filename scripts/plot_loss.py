#!/usr/bin/env python3
"""
Parse training log and plot loss curve.

Usage:
  python3 scripts/plot_loss.py logs/wikitext_training.log [plots/loss_curve.png]

Expected log format (from train_distributed stdout):
  step  N | loss X.XXXX | step Y.Y ms | comm Z.Z ms (W.W%)
"""

import sys
import os
import re

log_path = sys.argv[1] if len(sys.argv) > 1 else "logs/wikitext_training.log"
out_path = sys.argv[2] if len(sys.argv) > 2 else "plots/loss_curve.png"

steps, losses = [], []
pattern = re.compile(r"step\s+(\d+)\s*\|\s*loss\s+([\d.]+)")

with open(log_path) as f:
    for line in f:
        m = pattern.search(line)
        if m:
            steps.append(int(m.group(1)))
            losses.append(float(m.group(2)))

if not steps:
    print("No 'step N | loss X' lines found in log.")
    sys.exit(1)

print(f"Found {len(steps)} steps. Loss: {losses[0]:.4f} → {losses[-1]:.4f}")

try:
    import matplotlib.pyplot as plt
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plt.figure(figsize=(7, 3))
    plt.plot(steps, losses, linewidth=1.5)
    plt.xlabel("Training step")
    plt.ylabel("Cross-entropy loss")
    plt.title("WikiText-2 training loss")
    # random baseline: log(V) where V=10000
    import math
    baseline = math.log(10_000)
    plt.axhline(y=baseline, color="gray", linestyle="--",
                label=f"Random baseline (log V ≈ {baseline:.2f})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"Saved → {out_path}")
except ImportError:
    print("matplotlib not available — printing loss values instead:")
    for s, l in zip(steps, losses):
        print(f"  step {s:4d}: {l:.4f}")
