#!/usr/bin/env python3
"""Does the fallback ceiling rise as the campaign accumulates measurements?

Phase 21's 23% ceiling is a statement about a model fit on 48 measurements, which
is where a campaign starts, not where it ends. In a real design loop the in-house
model is refit every round on everything measured so far, so the fallback should
strengthen as the campaign runs. If it does, the recommendation changes from
"never fall back to random" to "and it gets better the longer you go".

The attacker is refit on the same measured set, so a larger campaign sharpens the
poison as well as the defence. That keeps the test conservative: any improvement
is net of the attacker improving too.

Batch budget stays fixed at 48 so yields remain comparable across campaign sizes;
only the measured history grows.

    python scripts/run_fallback_scaling.py
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
OBS_SIZES = [24, 48, 96, 192, 384]
PROBES = [0.25, 0.50]
BAITS = [0.0, 0.10, 0.25, 0.40, 0.50]
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
        B, top = args.budget, float(np.max(zs))

        for n_obs in OBS_SIZES:
            if n_obs > 0.25 * len(y):
                continue
            for seed in range(args.seeds):
                rng = np.random.default_rng(seed)
                obs = rng.choice(len(y), n_obs, replace=False)
                pool = np.setdiff1d(np.arange(len(y)), obs)
                mdl = Ridge(alpha=ALPHA).fit(X[obs], y[obs])
                pred = np.full(len(y), -np.inf)
                pred[pool] = mdl.predict(X[pool])
                base = B * elite[pool].sum() / len(pool)
                reference = y[obs]
                rank = np.argsort(pred[pool])
                poison, best = pool[rank[:B]], pool[rank[::-1]]

                # the in-house model used alone, for the mechanism
                solo = float(elite[pool[np.argsort(-pred[pool])[:B]]].sum()) - base
                rows.append({"assay": r.DMS_id, "seed": seed, "n_obs": n_obs,
                             "arm": "ridge_only", "probe_frac": np.nan,
                             "bait_frac": np.nan, "fallback": "n/a",
                             "adv": solo, "flagged": False})

                def run(order, f, fb):
                    n = int(round(f * B))
                    probe = order[:n]
                    try:
                        p = mannwhitneyu(y[probe], reference,
                                         alternative="greater").pvalue
                    except ValueError:
                        p = 1.0
                    flagged = bool(p > SIG)
                    left = B - n
                    if not flagged:
                        extra = order[n:n + left]
                    else:
                        avail = np.setdiff1d(pool, probe)
                        extra = (rng.choice(avail, min(left, len(avail)), replace=False)
                                 if fb is None
                                 else avail[np.argsort(-pred[avail])[:left]])
                    return (float(elite[np.concatenate([probe, extra])].sum()) - base,
                            flagged)

                hon_order = pool[np.argsort(-zs[pool])]
                for fb_name, fb in (("random", None), ("own_ridge", pred)):
                    for f in PROBES:
                        adv, fl = run(hon_order, f, fb)
                        rows.append({"assay": r.DMS_id, "seed": seed, "n_obs": n_obs,
                                     "arm": "honest", "probe_frac": f,
                                     "bait_frac": np.nan, "fallback": fb_name,
                                     "adv": adv, "flagged": fl})
                    for bf in BAITS:
                        nb = int(round(bf * B))
                        sc = zs.copy()
                        sc[poison] = top + 1.0
                        if nb:
                            sc[best[:nb]] = top + 2.0
                        order = pool[np.argsort(-sc[pool])]
                        for f in PROBES:
                            adv, fl = run(order, f, fb)
                            rows.append({"assay": r.DMS_id, "seed": seed,
                                         "n_obs": n_obs, "arm": "attacked",
                                         "probe_frac": f, "bait_frac": bf,
                                         "fallback": fb_name, "adv": adv,
                                         "flagged": fl})
        print(f"  {r.DMS_id}", flush=True)

    t = pd.DataFrame(rows)
    os.makedirs(args.out, exist_ok=True)
    t.to_csv(os.path.join(args.out, "fallback_scaling.csv"), index=False)

    print(f"\n{'=' * 90}\nMECHANISM — the in-house model used alone, by campaign size\n{'=' * 90}")
    solo = t[t.arm == "ridge_only"].groupby("n_obs").adv.mean()
    for n, v in solo.items():
        print(f"  {n:>4} measured: {v:+.3f} good candidates per batch vs random")

    print(f"\n{'=' * 90}\nDOES THE CEILING RISE?  (attacker best-responds at each cell)\n{'=' * 90}")
    print(f"{'probe':>6}{'n_obs':>7}{'honest':>9}{'attacked (rand fb)':>20}"
          f"{'attacked (ridge fb)':>21}{'recovered':>11}")
    a = t[t.arm == "attacked"]
    h = t[t.arm == "honest"]
    for f in PROBES:
        for n in sorted(set(t.n_obs)):
            sub = a[(a.probe_frac == f) & (a.n_obs == n)]
            if not len(sub):
                continue
            r0 = sub[sub.fallback == "random"].groupby("bait_frac").adv.mean().min()
            r1 = sub[sub.fallback == "own_ridge"].groupby("bait_frac").adv.mean().min()
            hh = h[(h.probe_frac == f) & (h.n_obs == n) &
                   (h.fallback == "own_ridge")].adv.mean()
            rec = (r1 - r0) / (hh - r0) if hh > r0 else np.nan
            print(f"{f:>6.0%}{n:>7}{hh:>9.3f}{r0:>20.3f}{r1:>21.3f}{rec:>10.0%}")
    print(f"\nwrote {args.out}/fallback_scaling.csv")


if __name__ == "__main__":
    main()
