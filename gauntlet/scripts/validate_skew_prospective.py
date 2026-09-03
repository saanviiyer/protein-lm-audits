#!/usr/bin/env python3
"""Prospective test: can label-free score skew predict elite utility out of sample?

    python scripts/validate_skew_prospective.py --scorer esm2 --out results

Phase 29 found that the skew of a proxy's own score distribution explains the
corpus offset, and that it needs no measured outcomes to compute. That is a
correlation across 60 units. This asks the question a practitioner would:
**before I measure anything, does skew tell me whether my top-k is worth
trusting?**

THREE MODELS, one of which is the incumbent and one of which is free.

  intercept   predict the training mean -- the null any method must beat
  bulk        predict from bulk Spearman -- REQUIRES measured labels, so it is
              not available before a campaign; it is the expensive incumbent
  skew        predict from score skew -- needs only the scores, available on the
              candidate pool before anything is ordered
  both        skew + bulk

TWO SPLITS, in increasing order of difficulty.

  LOO           leave one unit out over all 60. The easy version.
  leave-corpus  train on one benchmark family, predict another. This is the
                honest test of the Phase 29 claim: if skew really explains the
                between-family offset, a model fit on ProteinGym must transfer
                to SSMuLA without ever having seen it. If skew were just a
                within-family regularity, this is where it breaks.

AND A DECISION TEST. Out-of-sample correlation is a statistician's metric. The
practitioner's question is a selection: given several candidate pools and budget
for some of them, does ranking by skew put you on the ones whose top-k actually
pays out? Reported as the same normalised utility the rest of the project uses,
now applied to selecting ASSAYS rather than variants -- 1 = picked the best
available, 0 = picked at random.
"""

import argparse
import os

import numpy as np
import pandas as pd
from scipy import stats

MODELS = {
    "intercept": [],
    "bulk (needs labels)": ["bulk"],
    "skew (label-free)": ["score_skew"],
    "skew + bulk": ["score_skew", "bulk"],
}


def fit_predict(train, test, feats, target):
    """Least-squares fit on train, prediction on test. Intercept always in."""
    def design(d):
        cols = [d[f].to_numpy(float) for f in feats] + [np.ones(len(d))]
        return np.column_stack(cols)
    beta, *_ = np.linalg.lstsq(design(train), train[target].to_numpy(float), rcond=None)
    return design(test) @ beta


def selection_utility(pred, actual, frac=0.25):
    """Normalised utility of picking the top-`frac` UNITS by predicted score."""
    k = max(1, int(round(frac * len(actual))))
    rand = actual.mean()
    best = np.sort(actual)[-k:].mean()
    got = actual[np.argsort(-pred)[:k]].mean()
    return np.nan if best == rand else float((got - rand) / (best - rand))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scorer", default="esm2", choices=["esm2", "esmc"])
    ap.add_argument("--out", default="results")
    args = ap.parse_args()
    tag = "" if args.scorer == "esm2" else "_esmc"

    t = pd.read_csv(f"results/proxy_geometry{tag}.csv")
    print(f"scorer={args.scorer}   {len(t)} units   "
          f"({', '.join(f'{c} {n}' for c, n in t.corpus.value_counts().items())})")

    rows = []
    for frac in ["0.01", "0.1"]:
        target = f"util_{frac}"
        d = t[t[target].notna()].reset_index(drop=True)
        klab = "top-1%" if frac == "0.01" else "top-10%"

        print("\n" + "=" * 76)
        print(f"{klab}   LEAVE-ONE-OUT over all {len(d)} units")
        print("=" * 76)
        print(f"  {'model':22s} {'rho(pred,actual)':>17s} {'p':>9s} "
              f"{'MAE':>7s} {'select@25%':>11s}")
        for name, feats in MODELS.items():
            pred = np.array([
                fit_predict(d.drop(index=i), d.iloc[[i]], feats, target)[0]
                for i in range(len(d))])
            act = d[target].to_numpy()
            if len(feats) == 0:
                r, p = np.nan, np.nan          # constant prediction: no ranking
                sel = np.nan
            else:
                r, p = stats.spearmanr(pred, act)
                sel = selection_utility(pred, act)
            rows.append(dict(scorer=args.scorer, k=klab, split="LOO", model=name,
                             rho=r, p=p, mae=np.abs(pred - act).mean(), sel=sel))
            print(f"  {name:22s} {r:+17.3f} {p:9.4f} "
                  f"{np.abs(pred - act).mean():7.3f} {sel:11.3f}")

        print(f"\n{klab}   LEAVE-ONE-CORPUS-OUT (the strict test)")
        print("-" * 76)
        for tr, te in [("ProteinGym", "SSMuLA"), ("SSMuLA", "ProteinGym")]:
            train, test = d[d.corpus == tr], d[d.corpus == te]
            if len(train) < 6 or len(test) < 6:
                continue
            print(f"  train {tr} (n={len(train)})  ->  predict {te} (n={len(test)})")
            for name, feats in MODELS.items():
                if not feats:
                    continue
                pred = fit_predict(train, test, feats, target)
                act = test[target].to_numpy()
                r, p = stats.spearmanr(pred, act)
                sel = selection_utility(pred, act)
                rows.append(dict(scorer=args.scorer, k=klab,
                                 split=f"{tr}->{te}", model=name, rho=r, p=p,
                                 mae=np.abs(pred - act).mean(), sel=sel))
                print(f"    {name:22s} rho {r:+.3f} (p={p:.4f})   "
                      f"MAE {np.abs(pred - act).mean():.3f}   select@25% {sel:+.3f}")

    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(args.out, f"skew_prospective{tag}.csv"), index=False)
    print(f"\nwrote {args.out}/skew_prospective{tag}.csv")


if __name__ == "__main__":
    main()
