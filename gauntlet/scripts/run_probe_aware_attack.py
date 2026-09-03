#!/usr/bin/env python3
"""What does it cost an attacker to evade the top-pick probe?

The probe of run_topk_audit.py measures the scorer's highest-ranked candidates
and bails if they do not beat a random reference. An attacker who knows this is
coming has an obvious counter: bait the top of the ranking with genuinely
promising candidates so the probe passes, then fill everything below it with the
predicted-worst variants, which is where the rest of the batch will be spent.

The counter is not free. Bait is real value handed to the victim, and the
attacker must hand over at least as much as the probe will measure. If it baits
too little the probe fires; too much and it has simply helped. This sweeps bait
against probe size and reports the attacker's best response at each probe --- the
game-theoretic answer rather than one arbitrary pairing.

We also separate an attacker that knows the probe size exactly (an oracle, which
best-responds per column) from one that must commit to a single bait fraction
without knowing it.

    python scripts/run_probe_aware_attack.py
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
PROBES = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]
BAITS = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]
SIG = 0.05


def run_probe(order, pool, elite, y, reference, B, f, rng):
    """Probe the top f*B of `order`; fall back to random if it fails."""
    n = int(round(f * B))
    probe = order[:n]
    try:
        p = mannwhitneyu(y[probe], reference, alternative="greater").pvalue
    except ValueError:
        p = 1.0
    flagged = bool(p > SIG)
    left = B - n
    if flagged:
        avail = np.setdiff1d(pool, probe)
        extra = rng.choice(avail, min(left, len(avail)), replace=False)
    else:
        extra = order[n:n + left]
    picked = np.concatenate([probe, extra])
    return float(elite[picked].sum()), flagged


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

        for seed in range(args.seeds):
            rng = np.random.default_rng(seed)
            obs = rng.choice(len(y), B, replace=False)
            pool = np.setdiff1d(np.arange(len(y)), obs)
            early = obs[:B // 2]
            mdl = Ridge(alpha=ALPHA).fit(X[early], y[early])
            pred = mdl.predict(X[pool])
            base = B * elite[pool].sum() / len(pool)
            reference = y[obs]
            rank = np.argsort(pred)                 # ascending predicted fitness
            poison = pool[rank[:B]]                 # the B worst it can find
            best = pool[rank[::-1]]                 # descending, for bait

            hon_order = pool[np.argsort(-zs[pool])]
            for f in PROBES:
                got, fl = run_probe(hon_order, pool, elite, y, reference, B, f, rng)
                rows.append({"assay": r.DMS_id, "seed": seed, "attacker": "honest",
                             "bait_frac": np.nan, "probe_frac": f, "yield": got,
                             "flagged": fl, "random": base})

            for bf in BAITS:
                nb = int(round(bf * B))
                bait = best[:nb]
                sc = zs.copy()
                sc[poison] = top + 1.0
                if nb:
                    sc[bait] = top + 2.0             # bait outranks the poison
                order = pool[np.argsort(-sc[pool])]
                for f in PROBES:
                    got, fl = run_probe(order, pool, elite, y, reference, B, f, rng)
                    rows.append({"assay": r.DMS_id, "seed": seed,
                                 "attacker": "probe_aware", "bait_frac": bf,
                                 "probe_frac": f, "yield": got, "flagged": fl,
                                 "random": base})
        print(f"  {r.DMS_id}", flush=True)

    t = pd.DataFrame(rows)
    t["adv"] = t["yield"] - t["random"]
    os.makedirs(args.out, exist_ok=True)
    t.to_csv(os.path.join(args.out, "probe_aware_attack.csv"), index=False)

    atk = t[t.attacker == "probe_aware"]
    yld = atk.pivot_table(index="bait_frac", columns="probe_frac", values="adv")
    cau = atk.pivot_table(index="bait_frac", columns="probe_frac", values="flagged")
    hon = t[t.attacker == "honest"].groupby("probe_frac").adv.mean()

    print(f"\n{'=' * 78}\nATTACKER YIELD (good candidates vs random) — rows bait, cols probe\n{'=' * 78}")
    print(yld.round(3).to_string())
    print(f"\n{'=' * 78}\nCAUGHT RATE — rows bait, cols probe\n{'=' * 78}")
    print(cau.round(3).to_string())

    print(f"\n{'=' * 78}\nATTACKER'S BEST RESPONSE AT EACH PROBE\n{'=' * 78}")
    print(f"{'probe':>7}{'honest':>9}{'best bait':>11}{'attacked':>10}"
          f"{'caught':>9}{'damage kept':>13}")
    naive = yld.loc[0.0]
    for f in PROBES:
        b = yld[f].idxmin()
        dmg0 = hon.loc[f] - naive.loc[f]           # damage the naive attack does
        dmg = hon.loc[f] - yld.loc[b, f]
        print(f"{f:>7.0%}{hon.loc[f]:>9.3f}{b:>11.0%}{yld.loc[b, f]:>10.3f}"
              f"{cau.loc[b, f]:>9.1%}{dmg / dmg0 if dmg0 else np.nan:>12.0%}")

    print("\nBlind attacker — one bait fraction, averaged over probe sizes:")
    m = yld.mean(axis=1)
    b = m.idxmin()
    print(f"  best single bait is {b:.0%}, giving {m.loc[b]:.3f} "
          f"(caught {cau.loc[b].mean():.1%} on average)")
    print(f"\nwrote {args.out}/probe_aware_attack.csv")


if __name__ == "__main__":
    main()
