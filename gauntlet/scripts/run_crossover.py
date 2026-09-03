#!/usr/bin/env python3
"""When does a model you fit yourself match the scorer someone hands you?

If the in-house model catches up, the cleanest defence against a compromised
supplied scorer is to stop using it -- no probe, no fallback, no attack surface.
Phase 23 saw the point estimates cross between 192 and 384 measurements but could
not resolve it on 12 tasks, because between-task heterogeneity swamped the
contrast. This runs the same comparison over every cached scan.

Both arms select the same budget from the same pool; only the ranking differs.

    python scripts/run_crossover.py --max_assays 41
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.linear_model import Ridge

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from gauntlet.campaign import factorized_features  # noqa: E402
from gauntlet import proteingym as pg  # noqa: E402

ALPHA = 1.0
OBS_SIZES = [24, 48, 96, 192, 384]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pg_dir", default="data/proteingym")
    ap.add_argument("--out", default="results")
    ap.add_argument("--budget", type=int, default=48)
    ap.add_argument("--seeds", type=int, default=40)
    ap.add_argument("--max_assays", type=int, default=41)
    args = ap.parse_args()

    ref = pg.select_assays(os.path.join(args.pg_dir, "reference.csv"),
                           max_len=500, min_singles=1200)
    cached = {f[:-4] for f in os.listdir(os.path.join(args.pg_dir, "esm_cache"))}
    ref = ref[ref.DMS_id.isin(cached)].head(args.max_assays)

    B, rows = args.budget, []
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
        for n_obs in OBS_SIZES:
            if n_obs > 0.25 * len(y):
                continue
            for seed in range(args.seeds):
                rng = np.random.default_rng(seed)
                obs = rng.choice(len(y), n_obs, replace=False)
                pool = np.setdiff1d(np.arange(len(y)), obs)
                base = B * elite[pool].sum() / len(pool)
                pred = Ridge(alpha=ALPHA).fit(X[obs], y[obs]).predict(X[pool])
                rows.append({
                    "assay": r.DMS_id, "n_obs": n_obs, "seed": seed,
                    "own": float(elite[pool[np.argsort(-pred)[:B]]].sum()) - base,
                    "supplied": float(elite[pool[np.argsort(-zs[pool])[:B]]].sum()) - base,
                })
        print(f"  {r.DMS_id}", flush=True)

    t = pd.DataFrame(rows)
    os.makedirs(args.out, exist_ok=True)
    t.to_csv(os.path.join(args.out, "crossover.csv"), index=False)

    rng = np.random.default_rng(0)
    print(f"\n{'=' * 92}\nSELECTING WITH YOUR OWN MODEL VS THE SUPPLIED SCORER\n{'=' * 92}")
    print(f"{'measured':>9}{'tasks':>7}{'own model':>12}{'supplied':>11}"
          f"{'difference':>13}{'95% CI (task bootstrap)':>27}{'wins':>9}{'p':>8}")
    for n in OBS_SIZES:
        d = t[t.n_obs == n]
        if not len(d):
            continue
        per = d.groupby("assay")[["own", "supplied"]].mean()
        diff = (per.own - per.supplied).values
        reps = [np.mean(rng.choice(diff, len(diff), replace=True)) for _ in range(4000)]
        lo, hi = np.percentile(reps, [2.5, 97.5])
        p = wilcoxon(diff).pvalue if len(diff) > 5 else np.nan
        star = "  *" if lo > 0 or hi < 0 else ""
        print(f"{n:>9}{len(diff):>7}{per.own.mean():>12.3f}{per.supplied.mean():>11.3f}"
              f"{diff.mean():>+13.3f}{f'[{lo:+.3f}, {hi:+.3f}]':>27}"
              f"{f'{int((diff > 0).sum())}/{len(diff)}':>9}{p:>8.3f}{star}")
    print(f"\nwrote {args.out}/crossover.csv")


if __name__ == "__main__":
    main()
