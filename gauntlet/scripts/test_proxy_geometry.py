#!/usr/bin/env python3
"""Does the geometry of the PROXY's own score distribution explain the offset?

    python scripts/test_proxy_geometry.py --scorer esm2 --out results

Phases 26-28 established a corpus offset -- at equal bulk correlation an SSMuLA
landscape returns more elite utility than a ProteinGym assay -- that survives
four label-side explanations (Phase 27) and a change of scorer family (Phase 28).
Everything tested so far describes the LABELS. This tests the SCORES.

The idea being tested: elite utility is not about ranking everything well, it is
about the top of the ranking being trustworthy. That should depend on the shape
of the proxy's own score distribution -- whether its top-k is a well-separated
tail or the arbitrary top of a smooth blob, and how much resolution the score
has to separate near-ties with.

CANDIDATES, all fixed before looking at any of them, and all required to vary
WITHIN each corpus -- Phase 27's rule, learned when a degenerate covariate
identical to the corpus indicator closed 96% of the gap and explained nothing.

  top_separation  (mean of the proxy's top 1%  -  median) / IQR
                  how far the proxy's own elite sits above its bulk
  tie_frac        fraction of variants sharing a score value with another
                  the score's resolution for breaking near-ties
  score_skew      skew of the score distribution
  score_kurtosis  tailedness -- a heavy upper tail is a different selection
                  problem from a Gaussian one
  iqr_over_range  dispersion: how much of the score range the middle half spans

WHAT THIS CANNOT TEST, stated rather than quietly skipped. "Confidence
concentration" in the strict sense -- the entropy of the model's amino-acid
distribution at each position -- needs the per-position log-probability tables.
Those exist on disk for SSMuLA and rubisco but NOT for ProteinGym, whose cache
holds only final per-variant scores. Measuring entropy on two corpora and not
the third would compare exactly the two groups whose difference is the question.
Testing it properly means rescoring 41 ProteinGym assays to keep the tables,
which is a real GPU run, not a reanalysis.
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from gauntlet import proteingym as pg  # noqa: E402

CANDIDATES = ["top_separation", "tie_frac", "score_skew", "score_kurtosis",
              "iqr_over_range"]


def geometry(z):
    """Shape statistics of one unit's proxy-score vector."""
    z = np.asarray(z, float)
    z = z[np.isfinite(z)]
    if len(z) < 20:
        return None
    iqr = stats.iqr(z)
    rng = float(np.ptp(z)) or np.nan
    k = max(1, len(z) // 100)
    top = np.sort(z)[-k:].mean()
    _, counts = np.unique(np.round(z, 9), return_counts=True)
    return dict(
        top_separation=float((top - np.median(z)) / iqr) if iqr else np.nan,
        tie_frac=float((counts[counts > 1].sum()) / len(z)),
        score_skew=float(stats.skew(z)),
        score_kurtosis=float(stats.kurtosis(z)),
        iqr_over_range=float(iqr / rng) if rng == rng else np.nan,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pg_dir", default="data/proteingym")
    ap.add_argument("--out", default="results")
    ap.add_argument("--scorer", default="esm2", choices=["esm2", "esmc"])
    args = ap.parse_args()
    tag = "" if args.scorer == "esm2" else "_esmc"
    pg_cache = os.path.join(args.pg_dir,
                            "esm_cache" if args.scorer == "esm2" else "esmc_cache")

    rows = []
    ref = pg.select_assays(os.path.join(args.pg_dir, "reference.csv"))
    cached = {f[:-4] for f in os.listdir(pg_cache)}
    for r in ref[ref.DMS_id.isin(cached)].itertuples():
        g = geometry(np.load(os.path.join(pg_cache, f"{r.DMS_id}.npy")))
        if g:
            rows.append(dict(unit=r.DMS_id, corpus="ProteinGym", **g))

    s = pd.read_csv(f"results/ssmula_scored_variants{tag}.csv")
    for name, grp in s.groupby("landscape"):
        g = geometry(grp.esm2_wtm.to_numpy())
        if g:
            rows.append(dict(unit=name, corpus="SSMuLA", **g))

    d = pd.read_csv(f"results/rubisco_scored_variants{tag}.csv")
    d = d[(d.vmax_err <= 0.5) & (d.kc_err <= 0.5)]
    g = geometry(d.esm2_wtm.to_numpy())
    for label in ["rubisco Vmax", "rubisco K_C", "rubisco fitness"]:
        rows.append(dict(unit=label, corpus="rubisco", **g))

    feat = pd.DataFrame(rows)
    pts = pd.read_csv(f"results/bulk_vs_elite{tag}.csv")
    t = pts.merge(feat, on=["unit", "corpus"], how="inner")
    t.to_csv(os.path.join(args.out, f"proxy_geometry{tag}.csv"), index=False)
    print(f"scorer={args.scorer}   {len(t)} units\n")

    print("=" * 74)
    print("ADMISSIBILITY -- does each candidate vary WITHIN each corpus?")
    print("(Phase 27: a covariate constant within corpus can only restate the")
    print(" grouping, never explain it)")
    print("=" * 74)
    for c in CANDIDATES:
        g = t.groupby("corpus")[c]
        spreads = " ".join(f"{k}:{v:.3f}" for k, v in g.std().items())
        print(f"  {c:16s} within-corpus SD  {spreads}")

    for frac in ["0.01", "0.1"]:
        col = f"util_{frac}"
        sub = t[t[col].notna()].copy()
        fit = np.poly1d(np.polyfit(sub.bulk, sub[col], 1))
        sub["resid"] = sub[col] - fit(sub.bulk)
        pgm, ssm = sub[sub.corpus == "ProteinGym"], sub[sub.corpus == "SSMuLA"]
        gap0 = ssm.resid.mean() - pgm.resid.mean()

        print("\n" + "=" * 74)
        print(f"TOP {'1' if frac == '0.01' else '10'}%   "
              f"(corpus gap on bulk alone {gap0:+.3f}, "
              f"p={stats.mannwhitneyu(ssm.resid, pgm.resid).pvalue:.4f})")
        print("=" * 74)
        print(f"  {'candidate':16s} {'vs resid':>9s} {'p':>8s} | "
              f"{'gap after':>9s} {'p':>8s} {'closed':>8s} | "
              f"{'within-PGym':>11s} {'p':>7s}")
        for c in CANDIDATES:
            r = stats.spearmanr(sub[c], sub.resid)
            X = np.column_stack([sub.bulk, sub[c], np.ones(len(sub))])
            ok = np.isfinite(X).all(axis=1)
            beta, *_ = np.linalg.lstsq(X[ok], sub[col].to_numpy()[ok], rcond=None)
            res2 = sub[col].to_numpy()[ok] - X[ok] @ beta
            cor = sub.corpus.to_numpy()[ok]
            a, b = res2[cor == "SSMuLA"], res2[cor == "ProteinGym"]
            gap, gp = a.mean() - b.mean(), stats.mannwhitneyu(a, b).pvalue
            # The decisive one: does it act inside the corpus with 41 units?
            wr = stats.spearmanr(pgm[c], pgm.resid)
            print(f"  {c:16s} {r.statistic:+9.3f} {r.pvalue:8.4f} | "
                  f"{gap:+9.3f} {gp:8.4f} {100*(1-abs(gap)/abs(gap0)):+7.0f}% | "
                  f"{wr.statistic:+11.3f} {wr.pvalue:7.3f}")

    print(f"\nwrote {args.out}/proxy_geometry{tag}.csv")


if __name__ == "__main__":
    main()
