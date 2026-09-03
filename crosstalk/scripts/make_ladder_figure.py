"""Figure 5: scaling the proxy makes it worse at specificity."""
import csv
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

R = Path(__file__).resolve().parents[1]
rows = list(csv.DictReader(open(R / "results/proxy_ladder.csv")))
base = [r for r in rows if r["mode"] == "trivial"][0]
MODES = [("partner_blind", "partner-blind (ParD3 alone)", "#c0392b", "o"),
         ("partner_aware_ParE3", "partner-aware score", "#e67e22", "s"),
         ("partner_aware_margin", "partner-aware margin", "#2980b9", "^")]

fig, ax = plt.subplots(figsize=(8.4, 5.2))
for mode, label, color, mark in MODES:
    rs = sorted([r for r in rows if r["mode"] == mode], key=lambda r: float(r["params_M"]))
    x = [float(r["params_M"]) for r in rs]
    y = [float(r["auc_specific_vs_promiscuous"]) for r in rs]
    lo = [float(r["auc_lo"]) for r in rs]; hi = [float(r["auc_hi"]) for r in rs]
    ax.plot(x, y, marker=mark, ms=7, color=color, label=label)
    ax.fill_between(x, lo, hi, color=color, alpha=.13)

ax.axhline(0.5, color="k", ls="--", lw=1.2)
ax.annotate("chance", xy=(9, 0.51), fontsize=9)
b = float(base["auc_specific_vs_promiscuous"])
ax.axhline(b, color="#7f8c8d", ls=":", lw=1.8)
ax.annotate(f"trivial baseline: mutation count ({b:.2f})", xy=(9, b + 0.015),
            fontsize=9, color="#555")
ax.set_xscale("log"); ax.set_xticks([8, 35, 150, 650])
ax.set_xticklabels(["8M", "35M", "150M", "650M"])
ax.set_xlabel("ESM-2 parameters"); ax.set_ylabel("AUC: specific vs promiscuous binder")
ax.set_ylim(0.05, 0.78); ax.grid(alpha=.25)
ax.set_title("Protein-LM likelihood is a promiscuity detector.\n"
             "Below chance, and further below as the model grows.", fontweight="bold")
ax.legend(frameon=False, loc="upper right", fontsize=9)
fig.tight_layout()
out = R / "results/fig5_proxy_ladder.png"
fig.savefig(out, dpi=160); print("wrote", out)
