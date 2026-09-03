#!/usr/bin/env python3
"""What does the audit sample actually buy, across its whole range?

The adaptive pool-side attacker is invisible to both gates, so the only mitigation
we have is to spend part of each batch on candidates chosen at random rather than
by the scorer. We previously reported that at a single operating point (25%),
which says nothing about whether that point is sensible.

This sweeps the audit fraction from 0 to 1 and measures, for each scorer, the good
candidates captured per batch relative to random selection. The honest scorer
loses yield linearly as the audit grows; the adversarial case gains. Where those
meet, and what insurance costs when nothing is wrong, is the decision a developer
actually has to make.

    python scripts/run_audit_sweep.py
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from gauntlet.campaign import factorized_features  # noqa: E402
from gauntlet import proteingym as pg  # noqa: E402

ALPHA = 1.0
FRACS = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.75, 1.0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pg_dir", default="data/proteingym")
    ap.add_argument("--out", default="results")
    ap.add_argument("--budget", type=int, default=48)
    ap.add_argument("--seeds", type=int, default=40)
    ap.add_argument("--max_assays", type=int, default=12)
    args = ap.parse_args()

    ref = pg.select_assays(os.path.join(args.pg_dir, "reference.csv"),
                           max_len=500, min_singles=1200)
    cached = {f[:-4] for f in os.listdir(os.path.join(args.pg_dir, "esm_cache"))}
    ref = ref[ref.DMS_id.isin(cached)].head(args.max_assays)

    rows = []
    for r in ref.itertuples():
        records, y, _ = pg.load_assay(r.DMS_id, r.target_seq,
                                      os.path.join(args.pg_dir, "assays"))
        zs = np.load(os.path.join(args.pg_dir, "esm_cache", f"{r.DMS_id}.npy"))
        if len(zs) != len(y):
            continue
        X, _ = factorized_features(records)
        k = max(1, int(round(0.01 * len(y))))
        elite = np.zeros(len(y), bool)
        elite[np.argsort(-y)[:k]] = True
        B = args.budget

        for seed in range(args.seeds):
            rng = np.random.default_rng(seed)
            obs = rng.choice(len(y), B, replace=False)
            pool = np.setdiff1d(np.arange(len(y)), obs)
            early = obs[:B // 2]
            mdl = Ridge(alpha=ALPHA).fit(X[early], y[early])
            pred_pool = mdl.predict(X[pool])
            base = B * elite[pool].sum() / len(pool)      # expected random yield

            scorers = {"honest": zs.copy()}
            a = np.empty(len(y))
            a[:] = np.nan
            a[early] = y[early]
            rest = np.setdiff1d(np.arange(len(y)), early)
            a[rest] = mdl.predict(X[rest])
            scorers["adaptive_leak"] = a
            b = zs.copy()
            b[pool[np.argsort(pred_pool)[:B]]] = np.max(zs) + 1.0
            scorers["pool_side"] = b

            for name, sc in scorers.items():
                order = pool[np.argsort(-sc[pool])]
                for f in FRACS:
                    n_rand = int(round(f * B))
                    greedy = order[:B - n_rand]
                    got = float(elite[greedy].sum())
                    if n_rand:
                        rest_pool = np.setdiff1d(pool, greedy, assume_unique=False)
                        extra = rng.choice(rest_pool, min(n_rand, len(rest_pool)),
                                           replace=False)
                        got += float(elite[extra].sum())
                    rows.append({"assay": r.DMS_id, "seed": seed, "scorer": name,
                                 "audit_frac": f, "yield": got, "random": base})
        print(f"  {r.DMS_id}", flush=True)

    t = pd.DataFrame(rows)
    t["adv"] = t["yield"] - t["random"]
    os.makedirs(args.out, exist_ok=True)
    t.to_csv(os.path.join(args.out, "audit_sweep.csv"), index=False)

    piv = t.pivot_table(index="audit_frac", columns="scorer", values="adv")
    print(f"\n{'=' * 74}\nAUDIT SWEEP — good candidates per batch vs random\n{'=' * 74}")
    print(piv.round(3).to_string())

    hon, ps = piv["honest"], piv["pool_side"]
    print(f"\nHonest scorer gives up {hon.iloc[0] - hon.loc[0.25]:.3f} of its "
          f"{hon.iloc[0]:.3f} advantage at a 25% audit "
          f"({100 * (hon.iloc[0] - hon.loc[0.25]) / hon.iloc[0]:.0f}%).")
    reach = ps[ps >= -0.05]
    if len(reach):
        f = reach.index[0]
        print(f"The adversarial case first reaches parity with random at an audit "
              f"of {f:.0%},\nwhere the honest scorer still keeps "
              f"{100 * hon.loc[f] / hon.iloc[0]:.0f}% of its advantage.")
    # cost of insurance per unit of protection
    prot = ps - ps.iloc[0]
    cost = hon.iloc[0] - hon
    ratio = (cost / prot.replace(0, np.nan)).dropna()
    if len(ratio):
        best = ratio.idxmin()
        print(f"\nCost per unit of protection is lowest at an audit of {best:.0%} "
              f"({ratio.loc[best]:.2f} honest\ncandidates forgone per adversarial "
              f"candidate recovered), and rises on both sides.")
    print(f"\nwrote {args.out}/audit_sweep.csv")


if __name__ == "__main__":
    main()
