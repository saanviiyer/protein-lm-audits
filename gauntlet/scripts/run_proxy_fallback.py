#!/usr/bin/env python3
"""Does falling back to something better than random restore deterrence?

Phases 19-20 found a probe that detects a compromised scorer cheaply but cannot
deter one, because its only sanction -- abandon the scorer, select at random --
is precisely what the attacker wanted. That suggests the fix is not a better
detector but a better fallback.

The threat model makes some fallbacks available and not others. The attacker
supplies a scorer; it does not control the developer's own code. So a sequence
proxy computed in-house (BLOSUM62, hydropathy) and a ridge model fit on the
campaign's own measurements are both beyond its reach. It knows what they will
pick, but cannot change what they pick.

If the fallback carries real value, being caught costs the attacker something for
the first time, and it must pay bait to avoid it -- which hands value back to the
victim. Either branch is an improvement. This measures whether that happens.

Both sides fit ridge on the same measured set, so the attacker is not handicapped
relative to the defender.

    python scripts/run_proxy_fallback.py
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
from gauntlet.proxies import blosum_score, hydropathy_shift  # noqa: E402
from gauntlet import proteingym as pg  # noqa: E402

ALPHA = 1.0
PROBES = [0.10, 0.25, 0.50]
BAITS = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]
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
        top = float(np.max(zs))

        # in-house proxies: fixed priors, no access to outcomes
        bl = np.array([blosum_score(rec["muts"]) for rec in records])
        hy = -np.abs([hydropathy_shift(rec["muts"]) for rec in records])

        for seed in range(args.seeds):
            rng = np.random.default_rng(seed)
            obs = rng.choice(len(y), B, replace=False)
            pool = np.setdiff1d(np.arange(len(y)), obs)
            mdl = Ridge(alpha=ALPHA).fit(X[obs], y[obs])   # symmetric information
            pred = np.full(len(y), -np.inf)
            pred[pool] = mdl.predict(X[pool])
            base = B * elite[pool].sum() / len(pool)
            reference = y[obs]
            rank = np.argsort(pred[pool])
            poison = pool[rank[:B]]
            best = pool[rank[::-1]]

            fallbacks = {"random": None, "blosum": bl, "hydropathy": hy,
                         "own_ridge": pred}

            def run(order, f, fb_name, fb):
                n = int(round(f * B))
                probe = order[:n]
                try:
                    p = mannwhitneyu(y[probe], reference, alternative="greater").pvalue
                except ValueError:
                    p = 1.0
                flagged = bool(p > SIG)
                left = B - n
                if not flagged:
                    extra = order[n:n + left]
                elif fb is None:
                    avail = np.setdiff1d(pool, probe)
                    extra = rng.choice(avail, min(left, len(avail)), replace=False)
                else:
                    avail = np.setdiff1d(pool, probe)
                    extra = avail[np.argsort(-fb[avail])[:left]]
                picked = np.concatenate([probe, extra])
                return float(elite[picked].sum()), flagged

            hon_order = pool[np.argsort(-zs[pool])]
            for fb_name, fb in fallbacks.items():
                for f in PROBES:
                    got, fl = run(hon_order, f, fb_name, fb)
                    rows.append({"assay": r.DMS_id, "seed": seed, "attacker": "honest",
                                 "bait_frac": np.nan, "probe_frac": f,
                                 "fallback": fb_name, "yield": got, "flagged": fl,
                                 "random": base})
                for bf in BAITS:
                    nb = int(round(bf * B))
                    sc = zs.copy()
                    sc[poison] = top + 1.0
                    if nb:
                        sc[best[:nb]] = top + 2.0
                    order = pool[np.argsort(-sc[pool])]
                    for f in PROBES:
                        got, fl = run(order, f, fb_name, fb)
                        rows.append({"assay": r.DMS_id, "seed": seed,
                                     "attacker": "probe_aware", "bait_frac": bf,
                                     "probe_frac": f, "fallback": fb_name,
                                     "yield": got, "flagged": fl, "random": base})
        print(f"  {r.DMS_id}", flush=True)

    t = pd.DataFrame(rows)
    t["adv"] = t["yield"] - t["random"]
    os.makedirs(args.out, exist_ok=True)
    t.to_csv(os.path.join(args.out, "proxy_fallback.csv"), index=False)

    print(f"\n{'=' * 86}\nVICTIM'S YIELD UNDER THE ATTACKER'S BEST RESPONSE\n{'=' * 86}")
    print(f"{'fallback':<12}{'probe':>7}{'honest':>9}{'best bait':>11}"
          f"{'attacked':>10}{'caught':>9}{'floor lifted':>14}")
    atk = t[t.attacker == "probe_aware"]
    hon = t[t.attacker == "honest"]
    for fb in ["random", "blosum", "hydropathy", "own_ridge"]:
        for f in PROBES:
            a = atk[(atk.fallback == fb) & (atk.probe_frac == f)]
            h = hon[(hon.fallback == fb) & (hon.probe_frac == f)].adv.mean()
            g = a.groupby("bait_frac").adv.mean()
            b = g.idxmin()
            c = a[a.bait_frac == b].flagged.mean()
            ref0 = atk[(atk.fallback == "random") & (atk.probe_frac == f)]
            r0 = ref0.groupby("bait_frac").adv.mean().min()
            print(f"{fb:<12}{f:>7.0%}{h:>9.3f}{b:>11.0%}{g.loc[b]:>10.3f}"
                  f"{c:>9.1%}{g.loc[b] - r0:>+14.3f}")
    print(f"\nwrote {args.out}/proxy_fallback.csv")


if __name__ == "__main__":
    main()
