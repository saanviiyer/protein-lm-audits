"""Figure 6: the same failure on two systems that disagree by eight orders of magnitude."""
import csv
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

R = Path(__file__).resolve().parents[1]
fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.8))

# left: correlation with each partner
ax = axes[0]
systems = [("ParD3\n(WT selectivity ~5x)", ["ParE3\n(cognate)", "ParE2"], [0.510, 0.540]),
           ("BPTI\n(WT selectivity 1e6-1e9)", ["trypsin\n(cognate)", "chymo-\ntrypsin", "meso-\ntrypsin"],
            [0.160, 0.332, 0.060])]
x = 0
ticks, labels = [], []
for name, partners, rhos in systems:
    for i, (p, r) in enumerate(zip(partners, rhos)):
        ax.bar(x, r, color="#27ae60" if i == 0 else "#c0392b", width=.72,
               edgecolor="k", linewidth=.6)
        ticks.append(x); labels.append(p); x += 1
    x += 0.8
ax.set_xticks(ticks); ax.set_xticklabels(labels, fontsize=8)
ax.set_ylabel("rho(ESM-2 likelihood, measured binding)")
ax.set_title("The proxy does not favour the cognate partner", fontweight="bold", fontsize=11)
ax.set_ylim(0, 0.63)
ax.text(0.5, 0.575, "ParD3: indifferent\n(spread 0.03)", fontsize=8, ha="center")
ax.text(4.0, 0.44, "BPTI: prefers an off-target\n(spread 0.27)", fontsize=8, ha="center")
ax.axhline(0, color="k", lw=.8); ax.grid(alpha=.22, axis="y")
ax.legend(handles=[plt.Rectangle((0,0),1,1,color="#27ae60",ec="k"),
                   plt.Rectangle((0,0),1,1,color="#c0392b",ec="k")],
          labels=["cognate target", "off-target"], frameon=False, fontsize=8, loc="upper right")

# right: AUC across thresholds
ax = axes[1]
bpti = [0.372, 0.321, 0.256, 0.390, 0.328, 0.249]
ax.scatter(np.full(len(bpti), 1) + np.linspace(-.13, .13, len(bpti)), bpti,
           s=52, color="#c0392b", edgecolor="k", zorder=3, label="BPTI (6 thresholds)")
ax.scatter([0], [0.151], s=120, marker="D", color="#c0392b", edgecolor="k", zorder=3,
           label="ParD3")
ax.axhline(0.5, color="k", ls="--", lw=1.3)
ax.text(1.72, 0.512, "chance", fontsize=9, ha="right")
ax.scatter([0], [0.664], s=90, marker="^", color="#7f8c8d", edgecolor="k", zorder=3)
ax.text(0.06, 0.664, " mutation count (ParD3)", fontsize=8, va="center", color="#555")
ax.scatter([1], [0.499], s=90, marker="^", color="#7f8c8d", edgecolor="k", zorder=3)
ax.text(1.06, 0.462, "mutation count (BPTI)", fontsize=8, va="center", color="#555")
ax.set_xticks([0, 1]); ax.set_xticklabels(["ParD3", "BPTI"])
ax.set_xlim(-0.45, 2.0); ax.set_ylim(0.1, 0.78)
ax.set_ylabel("AUC: specific vs promiscuous binder")
ax.set_title("Below chance on both systems", fontweight="bold", fontsize=11)
ax.grid(alpha=.22, axis="y"); ax.legend(frameon=False, fontsize=8, loc="lower right")

fig.suptitle("A single-sequence likelihood has no privileged relationship with the cognate partner",
             fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.93])
out = R / "results/fig6_two_systems.png"
fig.savefig(out, dpi=160); print("wrote", out)
