#!/usr/bin/env python3
"""Does auditing the scorer's own top picks catch what a random audit cannot?

The random audit of run_audit_sweep.py is insurance, not a detector: it buys
yield regardless of what the scorer says, and pays its premium unconditionally.
The obvious developer instinct is different and much cheaper -- measure a few of
the scorer's top recommendations first, check whether they came out good, and
fall back to random selection for the rest of the batch if they did not.

That costs nothing extra when the scorer is honest, because a greedy batch
measures those candidates anyway; you have only reordered the spend and kept the
option to bail. Its only cost is a false alarm. And it should in principle catch
the pool-side attacker, whose whole method is to promote the predicted-*worst*
candidates to the top of the ranking -- precisely where the probe looks.

Reference for the comparison is free: the campaign's already-measured variants
are an unbiased sample of the pool, so "are the probe's outcomes better than a
random draw?" costs no budget. We flag a scorer when a one-sided Mann-Whitney
test fails to show the probe beats that reference.

    python scripts/run_topk_audit.py
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from sklearn.linear_model import Ridge

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from gauntlet.campaign import factorized_features  # noqa: E402
from gauntlet import proteingym as pg  # noqa: E402

ALPHA = 1.0
FRACS = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]
SIG = 0.05


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
            base = B * elite[pool].sum() / len(pool)
            reference = y[obs]           # free: already measured, unbiased

            scorers = {"honest": zs.copy()}
            a = np.empty(len(y))
            a[:] = np.nan
            a[early] = y[early]
            rest_idx = np.setdiff1d(np.arange(len(y)), early)
            a[rest_idx] = mdl.predict(X[rest_idx])
            scorers["adaptive_leak"] = a
            b = zs.copy()
            b[pool[np.argsort(pred_pool)[:B]]] = np.max(zs) + 1.0
            scorers["pool_side"] = b

            for name, sc in scorers.items():
                order = pool[np.argsort(-sc[pool])]
                for f in FRACS:
                    n_probe = int(round(f * B))
                    if n_probe == 0:                      # pure greedy control
                        picked, flagged = order[:B], False
                    else:
                        probe = order[:n_probe]
                        # one-sided: is the probe better than a random draw?
                        try:
                            p = mannwhitneyu(y[probe], reference,
                                             alternative="greater").pvalue
                        except ValueError:                # all ties
                            p = 1.0
                        flagged = bool(p > SIG)
                        left = B - n_probe
                        if flagged:
                            avail = np.setdiff1d(pool, probe)
                            extra = rng.choice(avail, min(left, len(avail)),
                                               replace=False)
                        else:
                            extra = order[n_probe:n_probe + left]
                        picked = np.concatenate([probe, extra])
                    rows.append({"assay": r.DMS_id, "seed": seed, "scorer": name,
                                 "probe_frac": f, "flagged": flagged,
                                 "yield": float(elite[picked].sum()),
                                 "random": base})
        print(f"  {r.DMS_id}", flush=True)

    t = pd.DataFrame(rows)
    t["adv"] = t["yield"] - t["random"]
    os.makedirs(args.out, exist_ok=True)
    t.to_csv(os.path.join(args.out, "topk_audit.csv"), index=False)

    adv = t.pivot_table(index="probe_frac", columns="scorer", values="adv")
    flag = t.pivot_table(index="probe_frac", columns="scorer", values="flagged")
    print(f"\n{'=' * 78}\nTOP-PICK AUDIT — good candidates per batch vs random\n{'=' * 78}")
    print(adv.round(3).to_string())
    print(f"\n{'=' * 78}\nFLAG RATE — how often the probe distrusts the scorer\n{'=' * 78}")
    print(flag.round(3).to_string())

    hon, ps = adv["honest"], adv["pool_side"]
    dmg = hon.iloc[0] - ps.iloc[0]
    print(f"\nAttack costs {dmg:.3f} candidates per batch with no probe.")
    for f in FRACS[1:]:
        rec = 100 * (ps.loc[f] - ps.iloc[0]) / dmg
        cost = 100 * (hon.iloc[0] - hon.loc[f]) / hon.iloc[0]
        print(f"  probe {f:>5.0%}: recovers {rec:5.1f}% of damage, "
              f"costs {cost:5.1f}% of honest yield "
              f"(false alarms {flag['honest'].loc[f]:.1%}, "
              f"catches {flag['pool_side'].loc[f]:.1%})")
    print(f"\nwrote {args.out}/topk_audit.csv")


if __name__ == "__main__":
    main()
