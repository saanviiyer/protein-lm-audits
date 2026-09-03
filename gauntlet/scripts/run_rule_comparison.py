#!/usr/bin/env python3
"""Which statistic should the planner decide on?

Phase 8 found the planner picks the best available action 47.3% of the time, and
that BOTH failure modes matter: it explores when exploiting would pay (234 misses,
regret 2.23) and it exploits the wrong option (272 misses, regret 1.36). Softening
the threshold would fix under half of that, so the statistic itself is the
suspect: top-decile enrichment is a k-way set intersection that takes four
distinct values on 48 observations.

This compares candidate statistics at identical decision points. The trajectory
is driven by the incumbent rule, and at every round each rule's choice is scored
against an oracle that sees the held-out fitnesses.

Two fixes to the Phase 8 oracle:
  * explore is scored by its EXPECTED yield, budget * elites_in_pool / pool,
    not one random draw -- so it can no longer win by luck.
  * every action is therefore scored deterministically, making regret comparable.

    python scripts/run_rule_comparison.py
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats as sps
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from gauntlet.campaign import (elite_auroc, factorized_features,  # noqa: E402
                               top_decile_enrichment)
from gauntlet import proteingym as pg  # noqa: E402

ASSAYS = ["ESTA_BACSU_Nutschel_2020", "A4GRB6_PSEAI_Chen_2020",
          "AMIE_PSEAE_Wrenbeck_2017", "DYR_ECOLI_Thompson_plusLon_2019",
          "KKA2_KLEPN_Melnikov_2014", "NUD15_HUMAN_Suiter_2020"]

MIN_FOR_TRUST, FOLDS, ALPHA = 12, 5, 1.0

# (name, statistic, threshold, tie-margin on that statistic's scale)
RULES = [
    ("enrich@1.5", "enrich", 1.50, 0.05),      # incumbent
    ("enrich@1.0", "enrich", 1.00, 0.05),      # softer threshold, same statistic
    ("auroc@0.55", "auroc", 0.55, 0.01),
    ("auroc@0.60", "auroc", 0.60, 0.01),
    ("auroc@0.65", "auroc", 0.65, 0.01),
    ("spearman@0.20", "rho", 0.20, 0.05),
]


def oof_predictions(X, y):
    if len(y) < MIN_FOR_TRUST or np.std(y) < 1e-12:
        return None
    pred = np.full(len(y), np.nan)
    for tr, te in KFold(min(FOLDS, len(y)), shuffle=True, random_state=0).split(X):
        if np.std(y[tr]) < 1e-12:
            continue
        pred[te] = Ridge(alpha=ALPHA).fit(X[tr], y[tr]).predict(X[te])
    return pred


def stat(kind, scores, fitness):
    if kind == "enrich":
        return top_decile_enrichment(scores, fitness)
    if kind == "auroc":
        return elite_auroc(scores, fitness)
    a, b = np.asarray(scores, float), np.asarray(fitness, float)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 5 or np.std(a[ok]) < 1e-12 or np.std(b[ok]) < 1e-12:
        return np.nan
    return float(sps.spearmanr(a[ok], b[ok]).statistic)


def choose(est, thr, tie):
    sup = est.get("supervised", np.nan)
    prox = {k: v for k, v in est.items() if k != "supervised"}
    bp = max(prox, key=lambda k: (prox[k] if np.isfinite(prox[k]) else -np.inf),
             default=None)
    bpv = prox.get(bp, np.nan)
    sup_ok = np.isfinite(sup) and sup >= thr
    prox_ok = np.isfinite(bpv) and bpv >= thr
    if sup_ok and (not prox_ok or sup + tie >= bpv):
        return "supervised"
    if prox_ok:
        return bp
    return "explore"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pg_dir", default="data/proteingym")
    ap.add_argument("--out", default="results")
    ap.add_argument("--budget", type=int, default=48)
    ap.add_argument("--rounds", type=int, default=4)
    ap.add_argument("--seeds", type=int, default=40)
    ap.add_argument("--assays", help="comma-separated DMS ids (default: the tuning set)")
    ap.add_argument("--tag", default="", help="suffix for the output file")
    args = ap.parse_args()

    ids = [a.strip() for a in args.assays.split(",")] if args.assays else ASSAYS
    ref = pg.select_assays(os.path.join(args.pg_dir, "reference.csv"), ids=ids)
    rows = []

    for r in ref.itertuples():
        records, y, _ = pg.load_assay(r.DMS_id, r.target_seq,
                                      os.path.join(args.pg_dir, "assays"))
        cache = os.path.join(args.pg_dir, "esm_cache", f"{r.DMS_id}.npy")
        if not os.path.exists(cache):
            continue
        X, _ = factorized_features(records)
        priors = {"esm2_wtm": np.load(cache)}
        k = max(1, int(round(0.01 * len(y))))
        elite_mask = np.zeros(len(y), bool)
        elite_mask[np.argsort(-y)[:k]] = True

        for seed in range(args.seeds):
            rng = np.random.default_rng(seed)
            observed = np.zeros(len(y), bool)
            y_obs = np.full(len(y), np.nan)

            for rnd in range(args.rounds):
                pool = np.flatnonzero(~observed)
                seen = np.flatnonzero(observed)
                if len(pool) <= args.budget:
                    break

                # --- estimates every rule shares -----------------------------
                est = {k2: {} for k2 in ("enrich", "auroc", "rho")}
                have = len(seen) >= MIN_FOR_TRUST and np.std(y_obs[seen]) > 1e-12
                if have:
                    pred = oof_predictions(X[seen], y_obs[seen])
                    for kind in est:
                        if pred is not None:
                            est[kind]["supervised"] = stat(kind, pred, y_obs[seen])
                        for nm, arr in priors.items():
                            est[kind][nm] = stat(kind, arr[seen], y_obs[seen])

                # --- oracle: deterministic yield of every action --------------
                def det_pick(a):
                    if a == "supervised":
                        m = Ridge(alpha=ALPHA).fit(X[seen], y_obs[seen])
                        return pool[np.argsort(-m.predict(X[pool]))[:args.budget]]
                    return pool[np.argsort(-priors[a][pool])[:args.budget]]

                yields = {"explore": args.budget * elite_mask[pool].sum() / len(pool)}
                if have:
                    for a in ["supervised"] + list(priors):
                        yields[a] = float(elite_mask[det_pick(a)].sum())
                best_action = max(yields, key=lambda a: yields[a])

                for name, kind, thr, tie in RULES:
                    c = choose(est[kind], thr, tie) if have else "explore"
                    rows.append({
                        "assay": r.DMS_id, "seed": seed, "round": rnd,
                        "n_seen": len(seen), "rule": name, "chosen": c,
                        "best_action": best_action,
                        "got": yields.get(c, 0.0), "best": yields[best_action],
                        "regret": yields[best_action] - yields.get(c, 0.0),
                    })

                # advance under the incumbent rule
                adv = choose(est["enrich"], 1.50, 0.05) if have else "explore"
                take = (rng.choice(pool, args.budget, replace=False)
                        if adv == "explore" else det_pick(adv))
                observed[take] = True
                y_obs[take] = y[take]
        print(f"  traced {r.DMS_id}", flush=True)

    t = pd.DataFrame(rows)
    os.makedirs(args.out, exist_ok=True)
    t.to_csv(os.path.join(args.out, f"rule_comparison{args.tag}.csv"), index=False)

    print(f"\n{'='*74}\nDECISION RULES COMPARED — {t.rule.nunique()} rules, "
          f"{len(t)//t.rule.nunique()} decision points each\n{'='*74}")
    g = t.groupby("rule").apply(lambda d: pd.Series({
        "picked_best": (d.chosen == d.best_action).mean(),
        "mean_regret": d.regret.mean(),
        "explored": (d.chosen == "explore").mean(),
        "threshold_miss": ((d.chosen == "explore") & (d.best_action != "explore")).mean(),
        "misrank_miss": ((d.chosen != "explore") & (d.chosen != d.best_action)).mean(),
    }), include_groups=False).sort_values("mean_regret")
    print(g.round(3).to_string())

    print("\nRegret by round (lower is better):")
    print(t.pivot_table(index="round", columns="rule", values="regret",
                        aggfunc="mean").round(3).to_string())

    best = g.index[0]
    inc = "enrich@1.5"
    print(f"\nBest by regret: {best}   (incumbent {inc} at "
          f"{g.loc[inc, 'mean_regret']:.3f})")
    print(f"  regret {g.loc[inc,'mean_regret']:.3f} -> {g.loc[best,'mean_regret']:.3f} "
          f"({100*(g.loc[inc,'mean_regret']-g.loc[best,'mean_regret'])/g.loc[inc,'mean_regret']:+.0f}%)")
    print(f"  picked-best {g.loc[inc,'picked_best']:.1%} -> {g.loc[best,'picked_best']:.1%}")
    print(f"\nwrote {args.out}/rule_comparison{args.tag}.csv")


if __name__ == "__main__":
    main()
