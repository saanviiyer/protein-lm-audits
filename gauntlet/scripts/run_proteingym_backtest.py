#!/usr/bin/env python3
"""Validate selection policies on ProteinGym deep-mutational-scanning assays.

DMS scans are near-complete, so replaying them is sound counterfactual
evaluation rather than a re-ordering of somebody else's prior selection. This is
the check that decides whether the PETase campaign result generalises.

    python scripts/run_proteingym_backtest.py --budget 48 --rounds 4
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from gauntlet import proteingym as pg  # noqa: E402
from gauntlet.campaign import (EpsilonSupervisedPolicy, Policy,  # noqa: E402
                               PlannerPolicy, evaluate, factorized_features)

# Enzyme-weighted, length-bounded, large-pool. ESTA_BACSU is an esterase
# thermostability scan -- the closest assay in ProteinGym to a PET hydrolase.
ASSAYS = [
    "B3VI55_LIPST_Klesmith_2015", "AMIE_PSEAE_Wrenbeck_2017",
    "A4GRB6_PSEAI_Chen_2020", "BLAT_ECOLX_Firnberg_2014",
    "KKA2_KLEPN_Melnikov_2014", "TPMT_HUMAN_Matreyek_2018",
    "TPK1_HUMAN_Weile_2017", "DYR_ECOLI_Thompson_plusLon_2019",
    "ESTA_BACSU_Nutschel_2020", "UBC9_HUMAN_Weile_2017",
    "RASH_HUMAN_Bandaru_2017", "NUD15_HUMAN_Suiter_2020",
    "CALM1_HUMAN_Weile_2017",
]

POLICIES = [
    ("random", Policy("random")),
    ("zero_shot_esm2", Policy("zero_shot")),
    ("supervised_greedy", Policy("greedy")),
    ("supervised_ucb", Policy("ucb", beta=1.0)),
    ("planner", PlannerPolicy()),
    # The ablation: no scorer switching at all, just a fixed exploration slice.
    ("sup_eps10", EpsilonSupervisedPolicy(eps=0.10)),
    ("sup_eps25", EpsilonSupervisedPolicy(eps=0.25)),
    ("sup_eps50", EpsilonSupervisedPolicy(eps=0.50)),
]


def esm_scores(dms_id, target_seq, records, cache_dir, scorer):
    """Wild-type-marginals score per variant, cached to disk."""
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, f"{dms_id}.npy")
    if os.path.exists(path):
        arr = np.load(path)
        if len(arr) == len(records):
            return arr
    positions = sorted({p - 1 for r in records for _, p, _ in r["muts"]})
    scorer.logprobs_at(target_seq, positions)
    arr = np.array([scorer.score(target_seq, r["muts"], offset=0) for r in records])
    np.save(path, arr)
    return arr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pg_dir", default="data/proteingym")
    ap.add_argument("--out", default="results")
    ap.add_argument("--budget", type=int, default=48)
    ap.add_argument("--rounds", type=int, default=4)
    ap.add_argument("--seeds", type=int, default=100)
    ap.add_argument("--model", default="facebook/esm2_t33_650M_UR50D")
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--all_cached", action="store_true",
                    help="use every assay with a cached score, not the fixed 13")
    ap.add_argument("--max_len", type=int, default=500)
    ap.add_argument("--min_singles", type=int, default=1200)
    args = ap.parse_args()

    if args.all_cached:
        cached = {f[:-4] for f in os.listdir(os.path.join(args.pg_dir, "esm_cache"))}
        ref = pg.select_assays(os.path.join(args.pg_dir, "reference.csv"),
                               max_len=args.max_len, min_singles=args.min_singles)
        ref = ref[ref.DMS_id.isin(cached)].reset_index(drop=True)
    else:
        ref = pg.select_assays(os.path.join(args.pg_dir, "reference.csv"), ids=ASSAYS)
    print(f"{len(ref)} assays selected\n")

    from gauntlet.proxies import ESM2Marginals
    scorer = ESM2Marginals(args.model, batch_size=args.batch_size)
    print(f"ESM-2 on {scorer.device}\n")

    rows = []
    for r in ref.itertuples():
        records, y, dropped = pg.load_assay(
            r.DMS_id, r.target_seq, os.path.join(args.pg_dir, "assays"))
        if len(records) < 500:
            print(f"  skip {r.DMS_id}: only {len(records)} usable")
            continue
        prior = esm_scores(r.DMS_id, r.target_seq, records,
                           os.path.join(args.pg_dir, "esm_cache"), scorer)
        X, _ = factorized_features(records)

        rec = {"assay": r.DMS_id, "molecule": r.molecule_name,
               "selection": r.selection_type, "n": len(records), "dropped": dropped}
        menu = {"esm2_wtm": prior}
        for name, pol in POLICIES:
            arg = menu if isinstance(pol, PlannerPolicy) else prior
            m = evaluate(X, y, arg, pol, args.budget, args.rounds, seeds=args.seeds)
            rec[name] = m["top_recall"]
            rec[f"{name}_best"] = m["best_norm"]
        rows.append(rec)
        print(f"  {r.DMS_id:38s} n={len(records):5d}  "
              + "  ".join(f"{n}={rec[n]:.3f}" for n, _ in POLICIES), flush=True)

    res = pd.DataFrame(rows)
    if res.empty:
        print("no assays usable")
        return
    names = [n for n, _ in POLICIES]

    print(f"\n{'='*78}")
    print(f"Budget {args.budget}/round x {args.rounds} rounds = {args.budget*args.rounds} "
          f"variants ordered, {args.seeds} seeds")
    print("Primary metric: recall of the assay's true top 1%\n")
    print(res[["assay", "n"] + names].round(3).to_string(index=False))

    print("\nMean across assays (top-1% recall), and margin over random:")
    for n in names:
        print(f"  {n:20s} {res[n].mean():.3f}   {res[n].mean()-res['random'].mean():+.3f}"
              f"   beats random in {(res[n] > res['random']).sum()}/{len(res)}")

    print("\nSecondary metric (best variant found, normalised):")
    for n in names:
        c = f"{n}_best"
        print(f"  {n:20s} {res[c].mean():.3f}   {res[c].mean()-res['random_best'].mean():+.3f}"
              f"   beats random in {(res[c] > res['random_best']).sum()}/{len(res)}")

    os.makedirs(args.out, exist_ok=True)
    res.to_csv(os.path.join(args.out, "proteingym_backtest.csv"), index=False)
    print(f"\nwrote {args.out}/proteingym_backtest.csv")


if __name__ == "__main__":
    main()
