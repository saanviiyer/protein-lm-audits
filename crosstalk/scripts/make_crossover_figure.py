"""Figure 4: where counter-screening starts paying for itself."""
import csv
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = Path(__file__).resolve().parents[1]
rows = list(csv.DictReader(open(R / "results/crossover.csv")))
BUD = sorted({int(r["budget"]) for r in rows})
KS = sorted({int(r["K"]) for r in rows})
BSTAR = {1: None, 2: 200, 3: 200, 4: 300}
COL = {"affinity": "#c0392b", "margin": "#27ae60"}

def get(K, reward, b):
    m = [r for r in rows if int(r["K"]) == K and r["reward"] == reward and int(r["budget"]) == b]
    return float(m[0]["success_rate"]) if m else float("nan")

fig, axes = plt.subplots(1, len(KS), figsize=(15, 3.9), sharey=True)
for ax, K in zip(axes, KS):
    for reward in ("affinity", "margin"):
        ax.plot(BUD, [get(K, reward, b) for b in BUD], marker="o", ms=5,
                color=COL[reward], label=reward if K == 1 else None)
    b = BSTAR[K]
    if b:
        ax.axvline(b, color="k", ls=":", lw=1.6)
        ax.annotate(f"B*={b}", xy=(b, 0.95), fontsize=9, ha="left",
                    xytext=(4, 0), textcoords="offset points")
    else:
        ax.annotate("no stable B*\nwithin budget", xy=(0.5, 0.88), xycoords="axes fraction",
                    ha="center", fontsize=9, color="#555")
    ax.set_xscale("log"); ax.set_xticks(BUD)
    ax.set_xticklabels([str(x) for x in BUD], rotation=45, fontsize=7)
    ax.set_ylim(-0.04, 1.04); ax.grid(alpha=.25)
    ax.set_title(f"K = {K}", fontweight="bold"); ax.set_xlabel("assay budget")
axes[0].set_ylabel("ground-truth success")
fig.suptitle("Counter-screening costs throughput, so it only pays above a budget B*.\n"
             "With one off-target it may never pay; with two or more it pays from ~200 assays.",
             fontweight="bold")
fig.legend(loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.13))
fig.tight_layout(rect=[0, 0.03, 1, 0.86])
out = R / "results/fig4_crossover.png"
fig.savefig(out, dpi=160, bbox_inches="tight"); print("wrote", out)
