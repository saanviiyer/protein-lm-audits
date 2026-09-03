#!/usr/bin/env python3
"""Bootstrap confidence intervals for the within-study correlations.

The audit reports sample-size-weighted means of per-study Spearman coefficients.
Those are point estimates over ~24 studies, many small: a per-study rho from
n ~ 15-20 carries a standard error near 0.25, so a bare mean is not enough to
support a claim that one scorer beats another. The paper already applies a sign
test to the learned proxy; this applies the same standard everywhere.

Method. Cluster bootstrap over STUDIES, not variants -- the study is the unit of
independence, since variants within a campaign share a scaffold and a lab. Each
replicate resamples studies with replacement and recomputes the weighted mean.
The paired comparison resamples the same study set for both scorers, so it tests
the difference rather than two independent means.

    python scripts/run_bootstrap_cis.py
"""

import argparse
import os

import numpy as np
import pandas as pd
from scipy import stats

PROXIES = ["esm2_wtm", "blosum62", "hydropathy", "n_mut"]
LABEL = {"esm2_wtm": "zero-shot pLM", "blosum62": "BLOSUM62",
         "hydropathy": "hydropathy", "n_mut": "mutation count"}
REPS = 4000


def wmean(rho, n):
    ok = np.isfinite(rho) & np.isfinite(n)
    return float(np.average(rho[ok], weights=n[ok])) if ok.sum() else np.nan


def boot(rho, n, reps=REPS, seed=0):
    """Cluster bootstrap over studies. Returns (lo, hi, p_two_sided_vs_zero)."""
    rng = np.random.default_rng(seed)
    m = len(rho)
    vals = np.empty(reps)
    for b in range(reps):
        idx = rng.integers(0, m, m)
        vals[b] = wmean(rho[idx], n[idx])
    vals = vals[np.isfinite(vals)]
    lo, hi = np.percentile(vals, [2.5, 97.5])
    # two-sided bootstrap p: how often the replicate crosses zero
    p = 2 * min((vals <= 0).mean(), (vals >= 0).mean())
    return float(lo), float(hi), float(min(1.0, p))


def paired(a, b, n, reps=REPS, seed=1):
    """Bootstrap the DIFFERENCE a-b, resampling the same studies for both."""
    rng = np.random.default_rng(seed)
    ok = np.isfinite(a) & np.isfinite(b)
    a, b, n = a[ok], b[ok], n[ok]
    m = len(a)
    d = np.empty(reps)
    for i in range(reps):
        idx = rng.integers(0, m, m)
        d[i] = wmean(a[idx], n[idx]) - wmean(b[idx], n[idx])
    lo, hi = np.percentile(d, [2.5, 97.5])
    w = stats.wilcoxon(a - b).pvalue if m > 5 else np.nan
    return wmean(a, n) - wmean(b, n), float(lo), float(hi), float(w), int((a > b).sum()), m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    rows, pairs = [], []
    for tag, model in [("", "ESM-2 650M"), ("_esmc", "ESM-C 300M")]:
        for target, nice in [("logActivity", "activity"), ("Tm", "stability")]:
            f = os.path.join(args.results, f"per_study_{target}{tag}.csv")
            if not os.path.exists(f):
                continue
            d = pd.read_csv(f)
            n = d["n"].to_numpy(float)
            for c in PROXIES:
                if c not in d:
                    continue
                rho = d[c].to_numpy(float)
                lo, hi, p = boot(rho, n)
                rows.append({"model": model, "target": nice, "scorer": LABEL[c],
                             "mean": wmean(rho, n), "lo": lo, "hi": hi,
                             "p_vs_zero": p, "studies": int(np.isfinite(rho).sum())})
            if "n_mut" in d and "esm2_wtm" in d:
                diff, lo, hi, w, wins, m = paired(
                    d["n_mut"].to_numpy(float), d["esm2_wtm"].to_numpy(float), n)
                pairs.append({"model": model, "target": nice, "diff": diff,
                              "lo": lo, "hi": hi, "wilcoxon_p": w,
                              "baseline_wins": wins, "studies": m})

    t = pd.DataFrame(rows)
    q = pd.DataFrame(pairs)
    t.to_csv(os.path.join(args.out, "bootstrap_cis.csv"), index=False)
    q.to_csv(os.path.join(args.out, "bootstrap_paired.csv"), index=False)

    print(f"\n{'=' * 78}\nCLUSTER BOOTSTRAP OVER STUDIES ({REPS} replicates)\n{'=' * 78}")
    for (model, target), g in t.groupby(["model", "target"], sort=False):
        print(f"\n{model}, {target}  ({int(g.studies.iloc[0])} studies)")
        print(f"  {'scorer':16s} {'mean':>7s}  {'95% CI':>18s}  {'p vs 0':>8s}")
        for r in g.itertuples():
            print(f"  {r.scorer:16s} {r.mean:>+7.3f}  [{r.lo:>+6.3f}, {r.hi:>+6.3f}]  "
                  f"{r.p_vs_zero:>8.3f}")

    print(f"\n{'=' * 78}\nPAIRED: mutation count MINUS zero-shot pLM, per study\n{'=' * 78}")
    print(f"  {'model':14s} {'target':10s} {'diff':>7s}  {'95% CI':>18s} "
          f"{'wilcoxon':>9s} {'baseline wins':>14s}")
    for r in q.itertuples():
        print(f"  {r.model:14s} {r.target:10s} {r.diff:>+7.3f}  "
              f"[{r.lo:>+6.3f}, {r.hi:>+6.3f}] {r.wilcoxon_p:>9.4f} "
              f"{str(r.baseline_wins) + '/' + str(r.studies):>14s}")

    print(f"\nwrote {args.out}/bootstrap_{{cis,paired}}.csv")


if __name__ == "__main__":
    main()
