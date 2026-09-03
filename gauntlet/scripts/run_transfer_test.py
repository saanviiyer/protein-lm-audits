#!/usr/bin/env python3
"""Are the pathologies properties of the verification protocol, or of our agent?

Both drafts argue the failures we report are properties of verification protocols
rather than of any particular policy class, and both concede the argument is
structural rather than demonstrated. This demonstrates the part that can be
demonstrated here: hold the protocol and the attack fixed, vary the policy, and
see what changes.

The prediction the structural argument makes is specific. The gate reads a
supplied scorer against already-measured data, so its verdict cannot depend on
the policy at all -- including for a policy that consults the gate before acting.
What should vary is only the damage, and only in proportion to how much of its
batch a policy allocates through the supplied scorer. A policy that never
consults it should be untouched.

We instantiate the shipped policy objects rather than reimplementing them, so
this tests the agent as written.

    python scripts/run_transfer_test.py --max_assays 41
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from gauntlet.campaign import (Policy, PlannerPolicy, factorized_features,  # noqa: E402
                               elite_auroc)
from gauntlet import proteingym as pg  # noqa: E402

ALPHA = 1.0


def select(name, X, y_obs, observed, pool, B, prior, rng):
    """Return (picked, used_supplied_flag). Shipped policies where they exist."""
    if name == "zero_shot_greedy":
        return Policy("zero_shot").select(X, y_obs, observed, B, prior, rng), 1.0
    if name == "supervised_greedy":
        return Policy("greedy").select(X, y_obs, observed, B, prior, rng), 0.0
    if name == "ucb_supervised":
        return Policy("ucb").select(X, y_obs, observed, B, prior, rng), 0.0
    if name == "planner":
        p = PlannerPolicy()
        picked = p.select(X, y_obs, observed, B, {"esm2": prior}, rng)
        return picked, float(p.decisions[-1].startswith("proxy"))
    if name == "eps25_zero_shot":
        n_rand = B // 4
        top = pool[np.argsort(-prior[pool])[:B - n_rand]]
        rest = np.setdiff1d(pool, top)
        return np.concatenate([top, rng.choice(rest, n_rand, replace=False)]), 0.75
    if name == "half_supplied_half_own":
        h = B // 2
        top = pool[np.argsort(-prior[pool])[:h]]
        seen = np.flatnonzero(observed)
        mdl = Ridge(alpha=ALPHA).fit(X[seen], y_obs[seen])
        rest = np.setdiff1d(pool, top)
        own = rest[np.argsort(-mdl.predict(X[rest]))[:B - h]]
        return np.concatenate([top, own]), 0.5
    raise ValueError(name)


POLICIES = ["zero_shot_greedy", "eps25_zero_shot", "half_supplied_half_own",
            "planner", "supervised_greedy", "ucb_supervised"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pg_dir", default="data/proteingym")
    ap.add_argument("--out", default="results41")
    ap.add_argument("--budget", type=int, default=48)
    ap.add_argument("--seeds", type=int, default=40)
    ap.add_argument("--max_assays", type=int, default=41)
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

        for seed in range(args.seeds):
            rng = np.random.default_rng(seed)
            obs = rng.choice(len(y), B, replace=False)
            observed = np.zeros(len(y), bool)
            observed[obs] = True
            y_obs = np.where(observed, y, np.nan)
            pool = np.setdiff1d(np.arange(len(y)), obs)
            base = B * elite[pool].sum() / len(pool)

            mdl = Ridge(alpha=ALPHA).fit(X[obs], y[obs])
            pred = mdl.predict(X[pool])
            attacked = zs.copy()
            attacked[pool[np.argsort(pred)[:B]]] = top + 1.0

            # the gate reads the supplied scorer on measured data only
            gate_h = elite_auroc(zs[obs], y[obs])
            gate_a = elite_auroc(attacked[obs], y[obs])

            for pol in POLICIES:
                for cond, prior in (("honest", zs), ("attacked", attacked)):
                    picked, used = select(pol, X, y_obs.copy(), observed.copy(),
                                          pool, B, prior, np.random.default_rng(seed))
                    rows.append({
                        "assay": r.DMS_id, "seed": seed, "policy": pol,
                        "condition": cond, "adv": float(elite[picked].sum()) - base,
                        "used_supplied": used,
                        "gate": gate_h if cond == "honest" else gate_a})
        print(f"  {r.DMS_id}", flush=True)

    t = pd.DataFrame(rows)
    os.makedirs(args.out, exist_ok=True)
    t.to_csv(os.path.join(args.out, "transfer_test.csv"), index=False)

    g = t.groupby(["policy", "condition"]).adv.mean().unstack()
    used = t[t.condition == "attacked"].groupby("policy").used_supplied.mean()
    gates = t.groupby("condition").gate.mean()

    print(f"\n{'=' * 84}\nSAME ATTACK, SAME GATE, DIFFERENT POLICIES\n{'=' * 84}")
    print(f"{'policy':<26}{'batch via supplied':>19}{'honest':>9}{'attacked':>10}{'damage':>9}")
    for p in POLICIES:
        if p not in g.index:
            continue
        d = g.loc[p, "honest"] - g.loc[p, "attacked"]
        print(f"{p:<26}{used.get(p, float('nan')):>18.0%}"
              f"{g.loc[p, 'honest']:>10.3f}{g.loc[p, 'attacked']:>10.3f}{d:>9.3f}")
    print(f"\nGate reading on measured data: honest {gates['honest']:.4f}, "
          f"attacked {gates['attacked']:.4f}")

    pl = t[t.policy == "planner"]
    tr = pl.groupby("condition").used_supplied.mean()
    print(f"Planner trusts the supplied scorer in {tr['honest']:.1%} of trials when it is "
          f"honest\nand {tr['attacked']:.1%} when it is attacked --- the gate cannot tell "
          f"them apart.")
    print(f"\nwrote {args.out}/transfer_test.csv")


if __name__ == "__main__":
    main()
