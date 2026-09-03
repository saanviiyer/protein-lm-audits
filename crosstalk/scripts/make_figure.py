"""Figure 1: budget does not substitute for the right objective."""
import csv, sys
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

rows = list(csv.DictReader(open(Path(__file__).resolve().parents[1] / "results/benchmark.csv")))
BUD = [50, 100, 200, 400, 800]
ARMS = [("affinity", "0", "affinity, screen only (2x throughput)", "#c0392b", "--"),
        ("affinity", "1", "affinity, counter-screened but ignored", "#e67e22", "-."),
        ("margin",   "1", "margin (uses off-target)", "#27ae60", "-"),
        ("gated",    "1", "gated", "#2980b9", "-"),
        ("lagrangian","1","Lagrangian", "#8e44ad", ":")]

def get(task, reward, cs, b, key):
    r = [x for x in rows if x["task"] == task and x["agent"] == "additive_model"
         and x["reward"] == reward and x["counter_screen"] == cs and int(x["budget"]) == b]
    return float(r[0][key]) if r else float("nan")

fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), sharex=True)
for col, task in enumerate(("cognate", "swap")):
    for row, key, lab in ((0, "success_rate", "ground-truth success rate"),
                          (1, "crosstalk_rate", "crosstalk rate (off-target > 0.5)")):
        ax = axes[row][col]
        for reward, cs, label, color, ls in ARMS:
            ys = [get(task, reward, cs, b, key) for b in BUD]
            ax.plot(BUD, ys, marker="o", ms=4, color=color, ls=ls,
                    label=label if (row == 0 and col == 0) else None)
        ax.set_xscale("log"); ax.set_xticks(BUD)
        ax.set_xticklabels([str(b) for b in BUD]); ax.set_ylim(-0.04, 1.04)
        ax.grid(alpha=.25)
        if col == 0: ax.set_ylabel(lab)
        if row == 0: ax.set_title(f"task: {task}", fontweight="bold")
        if row == 1: ax.set_xlabel("assay budget (log)")
fig.suptitle("Specificity is a reward-specification problem, not a data problem",
             fontweight="bold", y=0.98)
fig.legend(loc="lower center", ncol=3, frameon=False, fontsize=9, bbox_to_anchor=(0.5, -0.02))
fig.tight_layout(rect=[0, 0.06, 1, 0.96])
out = Path(__file__).resolve().parents[1] / "results/fig1_reward_specification.png"
fig.savefig(out, dpi=160, bbox_inches="tight")
print("wrote", out)
