#!/usr/bin/env python3
"""Figure: selection-policy performance across ProteinGym DMS assays."""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

NAMES = ["random", "zero_shot_esm2", "supervised_greedy", "supervised_ucb"]
LABEL = {
    "random": "random",
    "zero_shot_esm2": "ESM-2\nzero-shot",
    "supervised_greedy": "supervised\ngreedy",
    "supervised_ucb": "supervised\n+ UCB",
}
COLOR = {"zero_shot_esm2": "#c1440e", "random": "#8a8f98",
         "supervised_greedy": "#1f6f4a", "supervised_ucb": "#2e8b57"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    args = ap.parse_args()
    res = pd.read_csv(os.path.join(args.results, "proteingym_backtest.csv"))

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6),
                             gridspec_kw={"width_ratios": [1, 1.55]})

    ax = axes[0]
    for i, n in enumerate(NAMES):
        v = res[n].to_numpy()
        ax.scatter(np.full(len(v), i) + np.random.default_rng(0).uniform(-.12, .12, len(v)),
                   v, s=26, alpha=.65, color=COLOR[n], edgecolor="white", lw=.6, zorder=3)
        ax.plot([i - .3, i + .3], [v.mean()] * 2, color="black", lw=2.2, zorder=4)
        ax.text(i, ax.get_ylim()[1] if False else v.max() + .02, f"{v.mean():.3f}",
                ha="center", fontsize=9, fontweight="bold")
    ax.set_xticks(range(len(NAMES)))
    ax.set_xticklabels([LABEL[n] for n in NAMES], fontsize=8)
    ax.set_ylabel("recall of the assay's true top 1%")
    ax.set_title(f"{len(res)} ProteinGym DMS assays", fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)

    ax = axes[1]
    order = res.sort_values("supervised_greedy").reset_index(drop=True)
    ypos = np.arange(len(order))
    for n in NAMES:
        ax.scatter(order[n], ypos, s=34, alpha=.85, color=COLOR[n],
                   label=LABEL[n].replace("\n", " "), edgecolor="white", lw=.6, zorder=3)
    for j in ypos:
        ax.plot([order.loc[j, "random"], order.loc[j, "supervised_greedy"]],
                [j, j], color="#ccc", lw=1, zorder=1)
    ax.set_yticks(ypos)
    ax.set_yticklabels([f"{a.split('_')[0]} ({n})" for a, n in
                        zip(order.assay, order.n)], fontsize=7)
    ax.set_xlabel("recall of true top 1%")
    ax.legend(fontsize=7, loc="lower right", frameon=False)
    ax.set_title("per assay", fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle("Does the selection policy generalise beyond PETase?",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = os.path.join(args.results, "proteingym_backtest.png")
    fig.savefig(out, dpi=200)
    print("wrote", out)


if __name__ == "__main__":
    main()
