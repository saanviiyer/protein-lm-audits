#!/usr/bin/env python3
"""Backtest selection policies on published PET-hydrolase campaigns.

Asks the question a developer actually faces: with a budget of B variants per
round, does my selection policy reach a good enzyme in fewer rounds than
ordering at random?

    python scripts/run_backtest.py --budget 8 --rounds 5
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from gauntlet.campaign import (Policy, PlannerPolicy, evaluate,
                               mutation_features)  # noqa: E402
from gauntlet.petase_data import load_corpus  # noqa: E402

# (name, policy, which prior the policy ranks by)
#
# "per_mut" divides the masked-marginals score by the number of mutations.
# The raw score is a SUM over mutated positions of mostly-negative terms, so it
# falls mechanically as variants accumulate edits; in an engineering campaign,
# where the best variants are also the most-mutated, that ranks the winners
# last. Averaging instead of summing removes the confound without touching the
# model. This is the correction the Phase 1 audit implies.
POLICIES = [
    ("random", Policy("random"), "raw"),
    ("zero_shot_esm2", Policy("zero_shot"), "raw"),
    ("zero_shot_per_mut", Policy("zero_shot"), "per_mut"),
    ("supervised_greedy", Policy("greedy"), "raw"),
    ("supervised_ucb", Policy("ucb", beta=1.0), "raw"),
    # The tool's own decision rule, re-judged every round on what it would have
    # known at that point. "__all__" hands it the whole proxy menu.
    ("planner", PlannerPolicy(), "__all__"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--petml_dir", default="data/petml")
    ap.add_argument("--scored", default="results/scored_variants.csv")
    ap.add_argument("--out", default="results")
    ap.add_argument("--budget", type=int, default=8)
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--min_variants", type=int, default=30)
    ap.add_argument("--seeds", type=int, default=200)
    args = ap.parse_args()

    df = load_corpus(args.petml_dir)
    esm = pd.read_csv(args.scored)[["study", "protein", "esm2_wtm"]]
    df = df.merge(esm, on=["study", "protein"], how="left")

    rows = []
    for study, grp in df.groupby("study"):
        g = grp[grp.logActivity.notna() & grp.muts.notna()]
        g = g[[isinstance(m, list) and len(m) > 0 for m in g.muts]]  # list-comp: .apply on an empty Series returns a DataFrame
        if len(g) < args.min_variants:
            continue
        records = [{"muts": m} for m in g.muts]
        y = g.logActivity.to_numpy(dtype=float)
        raw = np.nan_to_num(g.esm2_wtm.to_numpy(dtype=float))
        nmut = np.array([len(m) for m in g.muts], dtype=float)
        priors = {"raw": raw, "per_mut": raw / np.maximum(nmut, 1.0)}
        # Give the planner the same menu `gauntlet plan` would see.
        from gauntlet.proxies import blosum_score, hydropathy_shift
        priors["__all__"] = {
            "esm2_wtm": raw,
            "esm2_per_mut": priors["per_mut"],
            "blosum62": np.array([blosum_score(m) for m in g.muts], float),
            "hydropathy": np.array([hydropathy_shift(m) for m in g.muts], float),
        }

        # Density: how much of the reachable design space was actually measured.
        vocab = {(p, m) for r in records for _, p, m in r["muts"]}
        rec = {
            "study": study, "n": len(g), "n_sites": len(vocab),
            "density": len(g) / max(len(vocab), 1),
        }
        rec["frac_multi"] = float((nmut > 1).mean())
        X, _ = mutation_features(records)
        for name, pol, pk in POLICIES:
            m = evaluate(X, y, priors[pk], pol, args.budget, args.rounds, seeds=args.seeds)
            rec[name] = m["best_norm"]
        rows.append(rec)

    res = pd.DataFrame(rows)
    if res.empty:
        print("no campaigns met the size threshold")
        return
    names = [n for n, _, _ in POLICIES]

    print(f"\nBudget {args.budget}/round, {args.rounds} rounds "
          f"({args.budget * args.rounds} variants ordered), {args.seeds} seeds")
    print("Value = fraction of the campaign's true best found. 1.0 = found the winner.\n")
    show = res[["study", "n", "frac_multi"] + names].copy()
    show["frac_multi"] = show["frac_multi"].round(2)
    for n in names:
        show[n] = show[n].round(3)
    print(show.to_string(index=False))

    print("\nMean across campaigns, and margin over random:")
    for n in names:
        print(f"  {n:20s} {res[n].mean():.3f}   {res[n].mean() - res['random'].mean():+.3f}")

    print("\nBeats random in:")
    for n in names[1:]:
        print(f"  {n:20s} {(res[n] > res.random).sum()}/{len(res)}")

    os.makedirs(args.out, exist_ok=True)
    res.to_csv(os.path.join(args.out, "backtest.csv"), index=False)
    print(f"\nwrote {args.out}/backtest.csv")


if __name__ == "__main__":
    main()
