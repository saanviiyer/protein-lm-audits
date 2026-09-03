#!/usr/bin/env python3
"""Does pooling help because of MORE DATA, or despite MIXING assays?

`run_nrel_conditions.py` found that a model pooled across all 11 NREL assay
conditions predicts better within a condition (AUROC 0.716) than a model fit only
on that condition (0.645). That contradicts the readiness claim -- but it confounds
two things, because the pooled model also sees up to 9x more rows.

This separates them with a size-matched control: train on the SAME number of rows
drawn entirely from OTHER conditions. Now only the mixing differs.

  within          n rows, all from this condition
  mixed@equal-n   n rows, none from this condition
  pooled-full     all 1,570 rows

If within > mixed@equal-n, condition purity is worth something and pooling wins
only by brute force. If they match, the conditions are interchangeable for
ranking and the tool's refusal to pool is wrong.

    python scripts/run_nrel_condition_control.py
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--campaign", default="data/nrel/nrel_campaign.csv")
    ap.add_argument("--out", default="results")
    ap.add_argument("--min_per_condition", type=int, default=40)
    ap.add_argument("--reps", type=int, default=15)
    args = ap.parse_args()

    d = pd.read_csv(args.campaign)
    d["cond"] = d[["pH", "temperature_C", "substrate"]].astype(str).agg(" | ".join, axis=1)
    X = sequence_features(d.sequence.tolist())[0]
    y = d.fitness.to_numpy(float)
    rng = np.random.default_rng(0)

    rows = []
    for c, g in d.groupby("cond"):
        idx = g.index.to_numpy()
        n = len(idx)
        if n < args.min_per_condition or np.std(y[idx]) < 1e-12:
            continue

        within = elite_auroc(loo_ridge_predictions(X[idx], y[idx], 1.0), y[idx])

        other = np.setdiff1d(np.arange(len(d)), idx)
        vals = []
        for _ in range(args.reps):
            tr = rng.choice(other, min(n, len(other)), replace=False)
            m = Ridge(alpha=1.0).fit(X[tr], y[tr])
            vals.append(elite_auroc(m.predict(X[idx]), y[idx]))
        mixed_eq = float(np.nanmean(vals))

        vals = []
        for j in range(n):
            keep = np.ones(len(d), bool)
            keep[idx[j]] = False
            vals.append(Ridge(alpha=1.0).fit(X[keep], y[keep])
                        .predict(X[idx[j]:idx[j] + 1])[0])
        pooled_full = elite_auroc(np.array(vals), y[idx])

        rows.append({"condition": c, "n": n, "within": within,
                     "mixed_equal_n": mixed_eq, "pooled_full": pooled_full})
        print(f"  {c:38s} n={n:4d}", flush=True)

    t = pd.DataFrame(rows)
    os.makedirs(args.out, exist_ok=True)
    t.to_csv(os.path.join(args.out, "nrel_condition_control.csv"), index=False)

    print(f"\n{'=' * 74}\nSIZE-MATCHED CONTROL — {len(t)} conditions\n{'=' * 74}")
    print(t.round(3).to_string(index=False))
    print(f"\nmean   within {t.within.mean():.3f}   "
          f"mixed@equal-n {t.mixed_equal_n.mean():.3f}   "
          f"pooled-full {t.pooled_full.mean():.3f}")
    for lab, col in [("mixed@equal-n", "mixed_equal_n"), ("pooled-full", "pooled_full")]:
        dv = (t["within"] - t[col]).dropna()
        p = stats.wilcoxon(dv).pvalue if len(dv) > 5 else float("nan")
        print(f"  within vs {lab:14s} diff {dv.mean():+.3f}  "
              f"within better in {int((dv > 0).sum())}/{len(dv)}  p={p:.4f}")
    print(f"\nwrote {args.out}/nrel_condition_control.csv")


if __name__ == "__main__":
    main()
