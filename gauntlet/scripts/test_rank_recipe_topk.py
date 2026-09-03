#!/usr/bin/env python3
"""Does a rank-oriented gain in Spearman carry over to top-k selection?

    python scripts/test_rank_recipe_topk.py --assays 12 --out results

MOTIVATION. AutoScientists (arXiv:2605.28655) reports raising ProteinGym's
average supervised Spearman from 0.657 to 0.700, a gain larger than the whole
gap between the current first and second place methods. It reports Spearman and
MSE and nothing else. A full-text search of the paper finds no top-k, recall,
NDCG, enrichment, or hit-rate metric, and no per-assay results. The paper's own
one caveat is that its rank-oriented changes "improve variant ordering but do not
necessarily improve calibrated regression", which is a ranking-versus-calibration
point and not a ranking-versus-selection one.

WHAT THIS IS NOT. It is not a reproduction of Kermut, of AutoScientists, or of
their headline number. Their code carries no licence, so it is not reusable, and
their per-variant predictions are not released, so their recipe cannot be scored
directly. Nothing here should be reported as their result.

WHAT THIS IS. A test of the mechanism the paper credits, on a supervised model we
control. Two predictors differ only in rank-oriented machinery:

  base    ridge on an additive position x residue decomposition plus the ESM-2
          zero-shot score
  recipe  an ensemble of three such ridges over different zero-shot feature sets,
          fit on QUANTILE-WARPED targets

Quantile warping and ensembling over expanded zero-shot features are two of the
four ingredients the paper credits (the others, greedy diversity-based feature
selection and Kermut's structure kernel, are not reproduced). Both predictors are
evaluated under ProteinGym's three cross-validation schemes on the same folds.

THE QUESTION, which is answerable even though the reproduction is not. For each
assay we get a change in bulk Spearman and a change in top-k utility from the
same intervention. If rank-oriented gains are worth what a leaderboard implies,
the two should move together. If a method can buy Spearman without buying
selection, that is visible here and is the thing the paper could not have seen,
because it never measured the second quantity.
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from gauntlet import proteingym as pg  # noqa: E402
from gauntlet.campaign import factorized_features  # noqa: E402

FRACS = [0.01, 0.10]
SCHEMES = ["random", "modulo", "contiguous"]
ALPHA = 1.0
N_FOLDS = 5


def folds(records, scheme, n, seed=0):
    """ProteinGym's three split styles, over mutated positions."""
    pos = np.array([r["muts"][0][1] for r in records])
    if scheme == "random":
        rng = np.random.default_rng(seed)
        return rng.integers(0, N_FOLDS, n)
    if scheme == "modulo":
        return pos % N_FOLDS
    uniq = np.sort(np.unique(pos))                    # contiguous blocks
    edges = np.array_split(uniq, N_FOLDS)
    m = {p: i for i, blk in enumerate(edges) for p in blk}
    return np.array([m[p] for p in pos])


def ridge(X, y, alpha=ALPHA):
    Xb = np.hstack([X, np.ones((len(X), 1), np.float32)])
    A = Xb.T @ Xb + alpha * np.eye(Xb.shape[1], dtype=np.float32)
    return np.linalg.solve(A, Xb.T @ y)


def predict(w, X):
    return np.hstack([X, np.ones((len(X), 1), np.float32)]) @ w


def quantile_warp(y):
    """Rank-transform targets to normal scores. Fit on training targets only."""
    r = stats.rankdata(y) / (len(y) + 1.0)
    return stats.norm.ppf(r).astype(np.float32)


def topk_utility(pred, y, frac):
    k = max(1, int(round(frac * len(y))))
    rand, best = y.mean(), np.sort(y)[-k:].mean()
    got = y[np.argsort(-pred)[:k]].mean()
    return np.nan if best == rand else float((got - rand) / (best - rand))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pg_dir", default="data/proteingym")
    ap.add_argument("--assays", type=int, default=12)
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    ref = pg.select_assays(os.path.join(args.pg_dir, "reference.csv"))
    cached = {f[:-4] for f in os.listdir(os.path.join(args.pg_dir, "esm_cache"))}
    ref = ref[ref.DMS_id.isin(cached)]
    # Pre-specified selection: the largest assays, so folds are well populated.
    ref = ref.sort_values("DMS_number_single_mutants", ascending=False).head(args.assays)
    print(f"{len(ref)} assays (largest by single-mutant count)\n")

    rows = []
    for r in ref.itertuples():
        records, y, _ = pg.load_assay(r.DMS_id, r.target_seq,
                                      os.path.join(args.pg_dir, "assays"))
        z2 = np.load(os.path.join(args.pg_dir, "esm_cache", f"{r.DMS_id}.npy"))
        zc_path = os.path.join(args.pg_dir, "esmc_cache", f"{r.DMS_id}.npy")
        zc = np.load(zc_path) if os.path.exists(zc_path) else z2
        if not (len(z2) == len(zc) == len(y)):
            print(f"  skip {r.DMS_id}: length mismatch")
            continue
        X0, _ = factorized_features(records)
        y = np.asarray(y, np.float32)

        # three feature sets: the ensemble members
        sets = [np.hstack([X0, z2[:, None]]).astype(np.float32),
                np.hstack([X0, zc[:, None]]).astype(np.float32),
                np.hstack([X0, z2[:, None], zc[:, None]]).astype(np.float32)]

        for scheme in SCHEMES:
            f = folds(records, scheme, len(y))
            pb = np.zeros(len(y)); pr = np.zeros(len(y))
            for k in range(N_FOLDS):
                te, tr = f == k, f != k
                if tr.sum() < 50 or te.sum() < 10:
                    continue
                # base: single model, raw targets
                pb[te] = predict(ridge(sets[0][tr], y[tr]), sets[0][te])
                # recipe: ensemble over expanded zero-shot sets, warped targets
                yw = quantile_warp(y[tr])
                pr[te] = np.mean([predict(ridge(S[tr], yw), S[te]) for S in sets], axis=0)

            ok = np.zeros(len(y), bool)
            for k in range(N_FOLDS):
                te, tr = f == k, f != k
                if tr.sum() >= 50 and te.sum() >= 10:
                    ok |= te
            if ok.sum() < 100:
                continue

            rec = dict(assay=r.DMS_id, scheme=scheme, n=int(ok.sum()),
                       bulk_base=stats.spearmanr(pb[ok], y[ok]).statistic,
                       bulk_recipe=stats.spearmanr(pr[ok], y[ok]).statistic)
            for fr in FRACS:
                rec[f"util_base_{fr}"] = topk_utility(pb[ok], y[ok], fr)
                rec[f"util_recipe_{fr}"] = topk_utility(pr[ok], y[ok], fr)
            rows.append(rec)
        print(f"  {r.DMS_id}", flush=True)

    t = pd.DataFrame(rows)
    t["d_bulk"] = t.bulk_recipe - t.bulk_base
    for fr in FRACS:
        t[f"d_util_{fr}"] = t[f"util_recipe_{fr}"] - t[f"util_base_{fr}"]
    t.to_csv(os.path.join(args.out, "rank_recipe_topk.csv"), index=False)

    print("\n" + "=" * 74)
    print("DOES THE RECIPE RAISE BULK SPEARMAN? (the metric the paper reports)")
    print("=" * 74)
    for scheme in SCHEMES + ["ALL"]:
        s = t if scheme == "ALL" else t[t.scheme == scheme]
        w = stats.wilcoxon(s.bulk_recipe, s.bulk_base).pvalue if len(s) > 5 else np.nan
        print(f"  {scheme:11s} n={len(s):2d}  base {s.bulk_base.mean():+.3f} -> "
              f"recipe {s.bulk_recipe.mean():+.3f}   mean gain {s.d_bulk.mean():+.4f}"
              f"   wins {int((s.d_bulk > 0).sum())}/{len(s)}  p={w:.4f}")

    print("\n" + "=" * 74)
    print("DOES THE SAME GAIN APPEAR IN TOP-K SELECTION?")
    print("=" * 74)
    for fr in FRACS:
        col = f"d_util_{fr}"
        for scheme in SCHEMES + ["ALL"]:
            s = (t if scheme == "ALL" else t[t.scheme == scheme]).dropna(subset=[col])
            if len(s) < 3:
                continue
            w = stats.wilcoxon(s[f"util_recipe_{fr}"], s[f"util_base_{fr}"]).pvalue \
                if len(s) > 5 else np.nan
            print(f"  top-{int(fr*100):<3d} {scheme:11s} n={len(s):2d}  "
                  f"base {s[f'util_base_{fr}'].mean():+.3f} -> "
                  f"recipe {s[f'util_recipe_{fr}'].mean():+.3f}   "
                  f"mean gain {s[col].mean():+.4f}   "
                  f"wins {int((s[col] > 0).sum())}/{len(s)}  p={w:.4f}")
        print()

    print("=" * 74)
    print("DO THE TWO GAINS TRACK EACH OTHER, PER ASSAY-SCHEME?")
    print("=" * 74)
    for fr in FRACS:
        s = t.dropna(subset=[f"d_util_{fr}"])
        r = stats.spearmanr(s.d_bulk, s[f"d_util_{fr}"])
        print(f"  top-{int(fr*100):<3d}  rho(d_bulk, d_util) = {r.statistic:+.3f}  "
              f"p={r.pvalue:.4f}  n={len(s)}")

    print(f"\nwrote {args.out}/rank_recipe_topk.csv")


if __name__ == "__main__":
    main()
