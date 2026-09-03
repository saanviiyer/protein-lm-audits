#!/usr/bin/env python3
"""A transfer scaling law for DNA-to-protein, in the shape of ATLAS.

Longpre et al. fit scaling laws for cross-lingual transfer and, importantly,
locate the compute crossover points at which one training choice overtakes
another. The genomic-to-protein question has the same shape: DNA and protein are
two encodings of one molecule, and the empirical question is whether a genomic
model's protein-fitness skill is a function of scale.

Five points are now available on identical assays: Nucleotide Transformer v2 at
50M, 100M, 250M and 500M, and Evo 2 at 7B, a 140x span. The reference lines that
make the numbers mean anything are BLOSUM62 at +0.228 and ESM-2 650M at +0.466,
both measured on the same assays.

The question the fit answers is not "does scale help" but "does the NT trend
EXTRAPOLATE to Evo 2". If it does, transfer is a scaling phenomenon. If Evo 2 sits
far off the NT curve, transfer is an architecture and corpus phenomenon that scale
alone does not buy, which is the genomic analogue of the point [Liu2024mobilellm]
makes for sub-billion language models.
"""
import csv, sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
from run_proxy_ladder import spearman

HANDOFF = Path("/Users/saanviiyer/Downloads/evo2_crosstalk_handoff/results/evo2_dms.csv")
PARAMS = {"50m": 50e6, "100m": 100e6, "250m": 250e6, "500m": 500e6}
BLOSUM62, ESM2 = 0.2282, 0.4655          # measured on the same 25 assays


def main():
    sweep = list(csv.DictReader((ROOT / "results/dms_scale_sweep.csv").open()))
    evo = {r["dms_id"]: float(r["rho_evo2"]) for r in csv.DictReader(HANDOFF.open())}
    mine = {r["dms_id"] for r in csv.DictReader((ROOT / "results/dms_transfer.csv").open())}
    common = sorted(mine & set(evo))

    pts = []
    for tag, p in PARAMS.items():
        rows = [r for r in sweep if tag in r["nt_model"] and r["dms_id"] in common]
        if rows:
            v = np.array([float(r["rho_nt"]) for r in rows])
            pts.append((f"NT-v2 {tag}", p, v.mean(), len(v)))
    e = np.array([evo[i] for i in common])
    pts.append(("Evo 2 7B", 7e9, e.mean(), len(e)))

    print(f"assays common to every model: {len(common)}\n")
    print(f"{'model':16s} {'params':>10s} {'mean rho':>9s}  n")
    for n, p, r, k in pts:
        print(f"  {n:14s} {p:10.3g} {r:+9.4f}  {k}")

    nt = [(p, r) for n, p, r, _ in pts if n.startswith("NT")]
    x = np.log10([p for p, _ in nt]); y = np.array([r for _, r in nt])
    slope, icept = np.polyfit(x, y, 1)
    print(f"\n--- NT-v2 trend, fitted on {len(nt)} points ---")
    print(f"  rho = {slope:+.4f} per decade of parameters {icept:+.4f}")
    print(f"  Spearman(params, rho) = {spearman([p for p, _ in nt], y):+.3f}")

    print("\n--- what that trend extrapolates to ---")
    for name, target in [("BLOSUM62 (+0.228)", BLOSUM62), ("ESM-2 650M (+0.466)", ESM2)]:
        need = (target - icept) / slope
        print(f"  to reach {name:22s}: 10^{need:.1f} = {10**need:.3g} parameters")
    evo_pred = slope * np.log10(7e9) + icept
    evo_act = e.mean()
    print(f"\n  NT trend PREDICTS Evo 2 at 7B: {evo_pred:+.4f}")
    print(f"  Evo 2 actually scores:         {evo_act:+.4f}")
    print(f"  the trend under-predicts it by {evo_act - evo_pred:+.4f}, "
          f"a factor of {evo_act/max(evo_pred, 1e-9):.1f}x" if evo_pred > 0 else
          f"  the trend predicts a NEGATIVE value; Evo 2 is off the curve entirely")

    print("\n--- reading ---")
    if evo_act > evo_pred + 0.05:
        print("  Evo 2 sits far above the Nucleotide Transformer scaling line, so its")
        print("  protein-fitness signal is NOT what scaling that family would buy.")
        print("  Transfer here is an architecture and corpus property, not a size one,")
        print("  which is the genomic analogue of the sub-billion result in MobileLLM.")
        print(f"  Scaling NT to Evo 2's size would predict {evo_pred:+.3f}, still below")
        print(f"  BLOSUM62 at +{BLOSUM62:.3f}.")

    out = ROOT / "results/transfer_scaling.csv"
    with out.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "params", "mean_rho", "n_assays", "nt_trend_prediction"])
        for n, p, r, k in pts:
            w.writerow([n, p, r, k, slope * np.log10(p) + icept])
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
