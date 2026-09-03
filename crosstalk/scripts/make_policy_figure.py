"""Figure 2: what training buys, by which reward it optimizes."""
import csv, sys
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

R = Path(__file__).resolve().parents[1]
rows = [r for r in csv.DictReader(open(R / "results/sweep.csv")) if r["final"] == "1"]

def deltas(task, reward, key):
    cell = [r for r in rows if r["task"] == task and r["reward"] == reward]
    out = []
    for s in sorted({int(r["seed"]) for r in cell}):
        pre = [r for r in cell if int(r["seed"]) == s and int(r["batches"]) == 0]
        post = [r for r in cell if int(r["seed"]) == s and int(r["batches"]) != 0]
        if pre and post:
            out.append(float(post[0][key]) - float(pre[0][key]))
    return np.array(out)

fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True)
COL = {"affinity": "#c0392b", "margin": "#27ae60"}
for ax, task in zip(axes, ("cognate", "swap")):
    for i, reward in enumerate(("affinity", "margin")):
        dr, ds = deltas(task, reward, "reward_achieved"), deltas(task, reward, "success_rate")
        ax.scatter(dr, ds, s=64, alpha=.85, color=COL[reward], edgecolor="k", lw=.6,
                   label=f"{reward} reward" if task == "cognate" else None, zorder=3)
        ax.scatter([dr.mean()], [ds.mean()], marker="X", s=230, color=COL[reward],
                   edgecolor="k", lw=1.3, zorder=4)
    ax.axhline(0, color="k", lw=.9, ls="--", alpha=.6)
    ax.axvline(0, color="k", lw=.9, ls="--", alpha=.6)
    ax.set_xlabel("change in the reward the agent optimized")
    ax.set_title(f"task: {task}", fontweight="bold")
    ax.grid(alpha=.22)
axes[0].set_ylabel("change in ground-truth success")
fig.suptitle("Training reliably improves the reward. Whether that reaches ground truth\n"
             "depends entirely on which reward it is.", fontweight="bold")
fig.legend(loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.04))
fig.tight_layout(rect=[0, 0.06, 1, 0.93])
out = R / "results/fig2_policy_dissociation.png"
fig.savefig(out, dpi=160, bbox_inches="tight")
print("wrote", out)
