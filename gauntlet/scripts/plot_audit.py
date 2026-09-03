#!/usr/bin/env python3
"""Figure: per-study rank correlation of each proxy against measured outcomes.

One point per study, so the reader sees the spread rather than a single pooled
number that a large study could carry on its own.
"""

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

PROXIES = ["esm2_wtm", "blosum62", "hydropathy", "n_mut"]
LABEL = {
    "esm2_wtm": "ESM-2 650M\n(zero-shot)",
    "blosum62": "BLOSUM62",
    "hydropathy": "hydropathy",
    "n_mut": "mutation count",
}


def panel(ax, per_study, title):
    rng = np.random.default_rng(0)
    for i, col in enumerate(PROXIES):
        v = per_study[["n", col]].dropna()
        if v.empty:
            continue
        rho, n = v[col].to_numpy(), v["n"].to_numpy()
        ax.scatter(
            i + rng.uniform(-0.13, 0.13, len(rho)), rho,
            s=8 + 3 * n, alpha=0.55,
            color="#c1440e" if col == "esm2_wtm" else "#4a5568",
            edgecolor="white", linewidth=0.6, zorder=3,
        )
        wm = float(np.average(rho, weights=n))
        ax.plot([i - 0.28, i + 0.28], [wm, wm], color="black", lw=2.2, zorder=4)
        ax.text(i, 1.06, f"{wm:+.2f}", ha="center", fontsize=9, fontweight="bold")

    ax.axhline(0, color="#999", lw=1, ls="--", zorder=1)
    ax.set_xticks(range(len(PROXIES)))
    ax.set_xticklabels([LABEL[c] for c in PROXIES], fontsize=8)
    ax.set_ylim(-1.15, 1.2)
    ax.set_ylabel("Spearman $\\rho$ within study")
    ax.set_title(title, fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    args = ap.parse_args()

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.2))
    for ax, target, nice in zip(axes, ["logActivity", "Tm"],
                                ["PET-hydrolytic activity", "melting temperature ($T_m$)"]):
        path = os.path.join(args.results, f"per_study_{target}.csv")
        if not os.path.exists(path):
            continue
        per = pd.read_csv(path)
        panel(ax, per, f"{nice}\n{int(per.n.sum())} variants, {len(per)} studies")

    fig.suptitle("Do zero-shot proxies rank measured PET-hydrolase outcomes?",
                 fontsize=12, fontweight="bold")
    fig.text(0.5, 0.005, "point area ∝ variants in study; bar = sample-size-weighted mean",
             ha="center", fontsize=8, color="#555")
    fig.tight_layout(rect=[0, 0.03, 1, 0.98])
    out = os.path.join(args.results, "proxy_audit.png")
    fig.savefig(out, dpi=200)
    print("wrote", out)


if __name__ == "__main__":
    main()
