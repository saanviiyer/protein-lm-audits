#!/usr/bin/env python3
"""Why does the planner under-exploit? Log every decision against the oracle.

Phase 7 found the planner beats random in 12/13 ProteinGym assays but leaves
substantial value unclaimed in about a third of them (DYR 0.078 where zero-shot
reached 0.208). Two explanations were possible and the aggregate could not tell
them apart:

  MIS-RANKING     the enrichment estimates ordered the options wrongly, so it
                  exploited the worse scorer.
  THRESHOLD       the estimates were fine but nothing cleared the 1.5x bar, so
                  it explored when exploiting would have paid.

This replays with tracing. At every round it records what the planner chose and
what it estimated, then evaluates what EACH available action would actually have
captured from the pool -- using the held-out fitnesses the planner cannot see.
Regret is elites-missed against the best action available at that moment.

    python scripts/run_planner_trace.py
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from gauntlet import proteingym as pg  # noqa: E402
from gauntlet.campaign import (MIN_ELITE, TOP_FRAC, factorized_features,  # noqa: E402
                               top_decile_enrichment)

# Three the planner won on, three it under-exploited.
ASSAYS = ["ESTA_BACSU_Nutschel_2020", "A4GRB6_PSEAI_Chen_2020",
          "AMIE_PSEAE_Wrenbeck_2017", "DYR_ECOLI_Thompson_plusLon_2019",
          "KKA2_KLEPN_Melnikov_2014", "NUD15_HUMAN_Suiter_2020"]

MIN_FOR_TRUST, TRUST, TIE, FOLDS, ALPHA = 12, 1.5, 0.05, 5, 1.0


def cv_enrichment(X, y):
    from sklearn.model_selection import KFold
    if len(y) < MIN_FOR_TRUST or np.std(y) < 1e-12:
        return np.nan
    pred = np.full(len(y), np.nan)
    for tr, te in KFold(min(FOLDS, len(y)), shuffle=True, random_state=0).split(X):
        if np.std(y[tr]) < 1e-12:
            continue
        pred[te] = Ridge(alpha=ALPHA).fit(X[tr], y[tr]).predict(X[te])
    return top_decile_enrichment(pred, y)


def trace_one(X, y, priors, elites, budget, rounds, rng):
    """One campaign, recording the planner's call and the oracle's at each round."""
    observed = np.zeros(len(y), dtype=bool)
    y_obs = np.full(len(y), np.nan)
    out = []

    for rnd in range(rounds):
        pool = np.flatnonzero(~observed)
        seen = np.flatnonzero(observed)
        if len(pool) <= budget:
            break

        # ---- what the planner knows and decides -------------------------------
        est = {}
        if len(seen) >= MIN_FOR_TRUST and np.std(y_obs[seen]) > 1e-12:
            est["supervised"] = cv_enrichment(X[seen], y_obs[seen])
            for name, arr in priors.items():
                est[name] = top_decile_enrichment(arr[seen], y_obs[seen])
            sup_e = est.get("supervised", np.nan)
            prox = {k: v for k, v in est.items() if k != "supervised"}
            best_prox = max(prox, key=lambda k: (prox[k] if np.isfinite(prox[k]) else -np.inf),
                            default=None)
            best_pe = prox.get(best_prox, np.nan)
            if np.isfinite(sup_e) and sup_e >= TRUST and (
                    not (np.isfinite(best_pe) and best_pe >= TRUST) or sup_e + TIE >= best_pe):
                chosen = "supervised"
            elif np.isfinite(best_pe) and best_pe >= TRUST:
                chosen = best_prox
            else:
                chosen = "explore"
        else:
            chosen = "explore"

        # ---- what every action would actually have captured -------------------
        def pick(action):
            if action == "explore":
                return rng.choice(pool, budget, replace=False)
            if action == "supervised":
                if len(seen) < 3 or np.std(y_obs[seen]) < 1e-12:
                    return rng.choice(pool, budget, replace=False)
                m = Ridge(alpha=ALPHA).fit(X[seen], y_obs[seen])
                return pool[np.argsort(-m.predict(X[pool]))[:budget]]
            return pool[np.argsort(-priors[action][pool])[:budget]]

        actions = ["explore", "supervised"] + list(priors)
        yield_ = {}
        for a in actions:
            idx = pick(a)
            yield_[a] = len(elites & set(idx.tolist()))
        best_action = max(yield_, key=lambda a: yield_[a])

        out.append({
            "round": rnd, "n_seen": len(seen), "chosen": chosen,
            "best_action": best_action,
            "got": yield_[chosen], "best_possible": yield_[best_action],
            "regret": yield_[best_action] - yield_[chosen],
            "explore_would_get": yield_["explore"],
            **{f"est_{k}": v for k, v in est.items()},
        })

        take = pick(chosen)
        observed[take] = True
        y_obs[take] = y[take]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pg_dir", default="data/proteingym")
    ap.add_argument("--out", default="results")
    ap.add_argument("--budget", type=int, default=48)
    ap.add_argument("--rounds", type=int, default=4)
    ap.add_argument("--seeds", type=int, default=40)
    ap.add_argument("--all_cached", action="store_true",
                    help="use every cached scan instead of the diagnostic set")
    args = ap.parse_args()

    if args.all_cached:
        ref = pg.select_assays(os.path.join(args.pg_dir, "reference.csv"))
        cached = {f[:-4] for f in os.listdir(os.path.join(args.pg_dir, "esm_cache"))}
        ref = ref[ref.DMS_id.isin(cached)].reset_index(drop=True)
    else:
        ref = pg.select_assays(os.path.join(args.pg_dir, "reference.csv"), ids=ASSAYS)
    rows = []
    for r in ref.itertuples():
        records, y, _ = pg.load_assay(r.DMS_id, r.target_seq,
                                      os.path.join(args.pg_dir, "assays"))
        cache = os.path.join(args.pg_dir, "esm_cache", f"{r.DMS_id}.npy")
        if not os.path.exists(cache):
            continue
        X, _ = factorized_features(records)
        priors = {"esm2_wtm": np.load(cache)}
        k = max(1, int(round(0.01 * len(y))))          # the backtest's objective
        elites = set(np.argsort(-y)[:k].tolist())

        for s in range(args.seeds):
            for rec in trace_one(X, y, priors, elites, args.budget, args.rounds,
                                 np.random.default_rng(s)):
                rec["assay"] = r.DMS_id
                rows.append(rec)
        print(f"  traced {r.DMS_id}", flush=True)

    t = pd.DataFrame(rows)
    os.makedirs(args.out, exist_ok=True)
    t.to_csv(os.path.join(args.out, "planner_trace.csv"), index=False)

    print(f"\n{'='*74}\nPLANNER DECISION TRACE — {len(t)} decisions\n{'='*74}")
    print(f"chose the best available action: {(t.chosen == t.best_action).mean():.1%}")
    print(f"mean regret (elites missed per round): {t.regret.mean():.2f}")

    print("\nWhat it chose, vs what was best:")
    print(pd.crosstab(t.chosen, t.best_action).to_string())

    miss = t[t.chosen != t.best_action]
    if len(miss):
        explore_when_exploit = miss[(miss.chosen == "explore")]
        wrong_exploit = miss[(miss.chosen != "explore")]
        print(f"\nMisses: {len(miss)} of {len(t)}")
        print(f"  THRESHOLD (explored, exploiting was better): {len(explore_when_exploit)}"
              f"  — mean regret {explore_when_exploit.regret.mean():.2f}")
        print(f"  MIS-RANKING (exploited the wrong option):    {len(wrong_exploit)}"
              f"  — mean regret {wrong_exploit.regret.mean():.2f}")

    print("\nBy round (n_seen is what the estimate was computed on):")
    g = t.groupby("round").agg(n_seen=("n_seen", "median"),
                               correct=("chosen", "size"),
                               acc=("chosen", lambda s: np.nan),
                               regret=("regret", "mean"))
    g["acc"] = t.groupby("round").apply(
        lambda d: (d.chosen == d.best_action).mean(), include_groups=False)
    g["explored"] = t.groupby("round").apply(
        lambda d: (d.chosen == "explore").mean(), include_groups=False)
    print(g[["n_seen", "acc", "explored", "regret"]].round(3).to_string())

    for c in [c for c in t.columns if c.startswith("est_")]:
        v = t[c].dropna()
        if len(v):
            print(f"\nDistinct values of {c} (quantisation of the estimate):")
            print("  " + ", ".join(f"{x:.2f}" for x in sorted(v.unique())[:12])
                  + (" ..." if v.nunique() > 12 else ""))
            print(f"  fraction at exactly 0.00 (below the {TRUST}x bar): "
                  f"{(v == 0).mean():.1%}")

    print(f"\nwrote {args.out}/planner_trace.csv")


if __name__ == "__main__":
    main()
