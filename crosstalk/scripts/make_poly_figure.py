"""Figure 3: the affinity objective anti-scales in both budget and off-targets."""
import csv
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = Path(__file__).resolve().parents[1]
rows = list(csv.DictReader(open(R / "results/polyspecificity.csv")))
BUD = sorted({int(r["budget"]) for r in rows})
KS = sorted({int(r["K"]) for r in rows})
COL = {"affinity": "#c0392b", "margin": "#27ae60"}

def get(K, reward, b, key):
    m = [r for r in rows if int(r["K"]) == K and r["reward"] == reward and int(r["budget"]) == b]
    return float(m[0][key]) if m else float("nan")

fig, axes = plt.subplots(2, len(KS), figsize=(15, 6.6), sharey="row", sharex=True)
for c, K in enumerate(KS):
    for r_i, key, lab in ((0, "success_rate", "ground-truth success"),
                          (1, "crosstalk_rate", "crosstalk rate")):
        ax = axes[r_i][c]
        for reward in ("affinity", "margin"):
            ys = [get(K, reward, b, key) for b in BUD]
            ax.plot(BUD, ys, marker="o", ms=6, color=COL[reward],
                    label=reward if (r_i == 0 and c == 0) else None)
        ax.set_xscale("log"); ax.set_xticks(BUD); ax.set_xticklabels([str(b) for b in BUD])
        ax.set_ylim(-0.04, 1.04); ax.grid(alpha=.25)
        if r_i == 0: ax.set_title(f"K = {K} off-target{'s' if K > 1 else ''}", fontweight="bold")
        if c == 0: ax.set_ylabel(lab)
        if r_i == 1: ax.set_xlabel("assay budget")
fig.suptitle("More budget makes the affinity objective WORSE, and the effect grows with\n"
             "the number of off-targets it ignores. The specificity objective improves.",
             fontweight="bold")
fig.legend(loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.03))
fig.tight_layout(rect=[0, 0.05, 1, 0.92])
out = R / "results/fig3_polyspecificity.png"
fig.savefig(out, dpi=160, bbox_inches="tight"); print("wrote", out)
