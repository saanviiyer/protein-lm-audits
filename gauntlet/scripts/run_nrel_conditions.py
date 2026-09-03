#!/usr/bin/env python3
"""Does pooling real assay conditions cost you? Measured on the NREL grid.

The condition-enforcement number quoted so far (+0.231 pooled to +0.808 within a
single condition) came from SYNTHETIC strata built by splitting one campaign and
adding an offset. It demonstrates the mechanism but is not evidence about real
data. The NREL release has 11 genuine conditions -- pH 4.5-8.5 x 40/60 C x
crystalline powder / amorphous film -- so the same question can be asked for real.

Design. For each condition c, fit two models and evaluate BOTH on c:
  within   trained only on measurements from c
  pooled   trained on every condition at once
Pooling buys more rows (up to 1,570 against at most 213) but mixes assays whose
activity numbers are not comparable. If the readiness claim is right, the extra
rows do not pay for the mixing.

Evaluation is leave-one-out within c, so the comparison is like-for-like and the
model is scored at its deployment training size.

    python scripts/run_nrel_conditions.py
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import Ridge

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from gauntlet.campaign import (elite_auroc, loo_ridge_predictions,  # noqa: E402
                               sequence_features)

ALPHA = 1.0


def rho(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 5 or np.std(a[ok]) < 1e-12 or np.std(b[ok]) < 1e-12:
        return np.nan
    return float(stats.spearmanr(a[ok], b[ok]).statistic)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--campaign", default="data/nrel/nrel_campaign.csv")
    ap.add_argument("--out", default="results")
    ap.add_argument("--min_per_condition", type=int, default=40)
    args = ap.parse_args()

    d = pd.read_csv(args.campaign)
    d["cond"] = d[["pH", "temperature_C", "substrate"]].astype(str).agg(" | ".join, axis=1)
    Xall, _ = sequence_features(d.sequence.tolist())
    yall = d.fitness.to_numpy(float)

    rows = []
    for c, g in d.groupby("cond"):
        idx = g.index.to_numpy()
        if len(idx) < args.min_per_condition or np.std(yall[idx]) < 1e-12:
            continue
        Xc, yc = Xall[idx], yall[idx]

        # within: trained only on this condition, scored leave-one-out
        within = loo_ridge_predictions(Xc, yc, ALPHA)

        # pooled: trained on everything, scored on this condition. Held out one
        # at a time so the point being scored is never in its own training set.
        pooled = np.empty(len(idx))
        for j in range(len(idx)):
            keep = np.ones(len(d), bool)
            keep[idx[j]] = False
            m = Ridge(alpha=ALPHA).fit(Xall[keep], yall[keep])
            pooled[j] = m.predict(Xall[idx[j]:idx[j] + 1])[0]

        rows.append({
            "condition": c, "n": len(idx), "n_pooled": len(d),
            "within_auroc": elite_auroc(within, yc),
            "pooled_auroc": elite_auroc(pooled, yc),
            "within_rho": rho(within, yc),
            "pooled_rho": rho(pooled, yc),
        })
        print(f"  {c:38s} n={len(idx):4d}", flush=True)

    t = pd.DataFrame(rows)
    os.makedirs(args.out, exist_ok=True)
    t.to_csv(os.path.join(args.out, "nrel_conditions.csv"), index=False)

    t["d_auroc"] = t.within_auroc - t.pooled_auroc
    t["d_rho"] = t.within_rho - t.pooled_rho
    print(f"\n{'='*74}\nPOOLING REAL ASSAY CONDITIONS — {len(t)} conditions, "
          f"{t.n.sum()} measurements\n{'='*74}")
    print(t[["condition", "n", "within_auroc", "pooled_auroc",
             "within_rho", "pooled_rho"]].round(3).to_string(index=False))

    print(f"\nmean AUROC   within {t.within_auroc.mean():.3f}   "
          f"pooled {t.pooled_auroc.mean():.3f}   diff {t.d_auroc.mean():+.3f}")
    print(f"mean Spearman within {t.within_rho.mean():+.3f}   "
          f"pooled {t.pooled_rho.mean():+.3f}   diff {t.d_rho.mean():+.3f}")
    for lab, col in [("AUROC", "d_auroc"), ("Spearman", "d_rho")]:
        v = t[col].dropna()
        if len(v) > 5:
            p = stats.wilcoxon(v).pvalue
            print(f"  within beats pooled on {lab}: {int((v > 0).sum())}/{len(v)} "
                  f"conditions, paired p={p:.4f}")

    print("\nPooling gives every model up to "
          f"{t.n_pooled.iloc[0] // t.n.median():.0f}x more rows. "
          "The question is whether that pays for\nmixing assays whose numbers are "
          "not comparable.")
    print(f"\nwrote {args.out}/nrel_conditions.csv")


if __name__ == "__main__":
    main()
