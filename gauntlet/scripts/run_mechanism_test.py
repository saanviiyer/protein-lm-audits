#!/usr/bin/env python3
"""Controlled test of the mutation-count confound.

Phase 2 inferred, from five PETase campaigns, that a zero-shot language-model
score fails on multi-mutant pools because it is a SUM over mutated positions of
mostly-negative terms, so it falls as edits accumulate while campaign fitness
rises. That inference rested on one catastrophic case. This tests it directly.

Construction. Take a DMS assay's single mutants, whose fitness and per-mutation
ESM delta are both known. Compose synthetic k-mutants by combining single
mutants at distinct positions. Ground truth is the SUM of the component single
fitnesses -- an explicit additivity assumption, stated below -- and the ESM
score is the sum of the component deltas, exactly as masked marginals would
compute it.

Two conditions isolate the mechanism:

- ``beneficial``: components drawn from the top of the fitness distribution,
  mimicking an engineering campaign, where more edits means better.
- ``random``: components drawn uniformly, where more edits means worse.

And one control: within a FIXED k, mutation count is constant, so the confound
cannot operate. If mixed-k pools break while fixed-k pools do not, the failure
is pool composition rather than the model.

ASSUMPTION. Additive ground truth. Real multi-mutants have epistasis, so this
does not predict actual fitness -- it isolates whether the summation in the
scoring protocol produces the observed anti-correlation. Epistasis would add
noise, not remove the arithmetic.

    python scripts/run_mechanism_test.py
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from gauntlet import proteingym as pg  # noqa: E402

ASSAYS = [
    "ESTA_BACSU_Nutschel_2020", "A4GRB6_PSEAI_Chen_2020",
    "AMIE_PSEAE_Wrenbeck_2017", "BLAT_ECOLX_Firnberg_2014",
    "DYR_ECOLI_Thompson_plusLon_2019", "KKA2_KLEPN_Melnikov_2014",
    "TPMT_HUMAN_Matreyek_2018", "UBC9_HUMAN_Weile_2017",
]


def compose(singles, k_values, n_per_k, rng, top_frac=None):
    """Sample synthetic multi-mutants at distinct positions.

    singles: DataFrame with columns pos, fitness, delta.
    top_frac: if set, draw components only from that top fraction by fitness.
    """
    pool = singles
    if top_frac is not None:
        cut = pool.fitness.quantile(1 - top_frac)
        pool = pool[pool.fitness >= cut]
    by_pos = {p: g.index.to_numpy() for p, g in pool.groupby("pos")}
    positions = np.array(list(by_pos))
    if len(positions) < max(k_values):
        return None

    fit = pool.fitness.to_dict()
    dlt = pool.delta.to_dict()
    rows = []
    for k in k_values:
        for _ in range(n_per_k):
            ps = rng.choice(positions, k, replace=False)
            idx = [rng.choice(by_pos[p]) for p in ps]
            rows.append({
                "k": k,
                "truth": float(sum(fit[i] for i in idx)),
                "esm_sum": float(sum(dlt[i] for i in idx)),
            })
    df = pd.DataFrame(rows)
    df["esm_per_mut"] = df.esm_sum / df.k
    return df


def spearman(a, b):
    if len(a) < 5 or np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return np.nan
    return float(stats.spearmanr(a, b).statistic)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pg_dir", default="data/proteingym")
    ap.add_argument("--out", default="results")
    ap.add_argument("--max_k", type=int, default=6)
    ap.add_argument("--n_per_k", type=int, default=400)
    ap.add_argument("--top_frac", type=float, default=0.30)
    ap.add_argument("--all_cached", action="store_true",
                    help="use every cached scan instead of the diagnostic set")
    args = ap.parse_args()

    if args.all_cached:
        ref = pg.select_assays(os.path.join(args.pg_dir, "reference.csv"))
        cached = {f[:-4] for f in os.listdir(os.path.join(args.pg_dir, "esm_cache"))}
        ref = ref[ref.DMS_id.isin(cached)].reset_index(drop=True)
    else:
        ref = pg.select_assays(os.path.join(args.pg_dir, "reference.csv"), ids=ASSAYS)
    ks = list(range(1, args.max_k + 1))
    mixed_rows, fixed_rows = [], []

    for r in ref.itertuples():
        records, y, _ = pg.load_assay(
            r.DMS_id, r.target_seq, os.path.join(args.pg_dir, "assays"))
        cache = os.path.join(args.pg_dir, "esm_cache", f"{r.DMS_id}.npy")
        if not os.path.exists(cache):
            continue
        delta = np.load(cache)
        singles = pd.DataFrame({
            "pos": [m["muts"][0][1] for m in records],
            "fitness": y,
            "delta": delta,
        })

        for cond, tf in [("beneficial", args.top_frac), ("random", None)]:
            rng = np.random.default_rng(0)
            df = compose(singles, ks, args.n_per_k, rng, top_frac=tf)
            if df is None:
                continue

            # Mixed-k pools, truncated at each K: the campaign-like regime.
            for K in ks:
                sub = df[df.k <= K]
                mixed_rows.append({
                    "assay": r.DMS_id, "condition": cond, "max_k": K,
                    "rho_sum": spearman(sub.esm_sum, sub.truth),
                    "rho_per_mut": spearman(sub.esm_per_mut, sub.truth),
                    "rho_k_vs_truth": spearman(sub.k, sub.truth),
                    "rho_k_vs_esm": spearman(sub.k, sub.esm_sum),
                })
            # Fixed-k control: mutation count cannot confound.
            for k in ks:
                sub = df[df.k == k]
                fixed_rows.append({
                    "assay": r.DMS_id, "condition": cond, "k": k,
                    "rho_sum": spearman(sub.esm_sum, sub.truth),
                })

    mixed = pd.DataFrame(mixed_rows)
    fixed = pd.DataFrame(fixed_rows)
    os.makedirs(args.out, exist_ok=True)
    mixed.to_csv(os.path.join(args.out, "mechanism_mixed.csv"), index=False)
    fixed.to_csv(os.path.join(args.out, "mechanism_fixed.csv"), index=False)

    print(f"{ref.shape[0]} assays, {args.n_per_k} composites per k, k=1..{args.max_k}")
    print(f"Additive ground truth. 'beneficial' draws components from the top "
          f"{args.top_frac:.0%} by fitness.\n")

    for cond in ["beneficial", "random"]:
        m = mixed[mixed.condition == cond].groupby("max_k").mean(numeric_only=True)
        print(f"=== MIXED-k pool, {cond} components "
              f"(mean Spearman across assays) ===")
        print(m[["rho_sum", "rho_per_mut", "rho_k_vs_truth", "rho_k_vs_esm"]]
              .round(3).to_string())
        print()

    print("=== FIXED-k control: rho(ESM sum, truth) within constant k ===")
    f = fixed.pivot_table(index="k", columns="condition", values="rho_sum")
    print(f.round(3).to_string())

    print("\nInterpretation check:")
    for cond in ["beneficial", "random"]:
        m = mixed[(mixed.condition == cond)]
        k1 = m[m.max_k == 1].rho_sum.mean()
        kmax = m[m.max_k == args.max_k].rho_sum.mean()
        pm = m[m.max_k == args.max_k].rho_per_mut.mean()
        fx = fixed[(fixed.condition == cond) & (fixed.k == args.max_k)].rho_sum.mean()
        print(f"  {cond:11s} single-only rho={k1:+.3f} -> mixed k<={args.max_k} "
              f"rho={kmax:+.3f} | per-mut {pm:+.3f} | fixed-k={args.max_k} {fx:+.3f}")

    print(f"\nwrote {args.out}/mechanism_{{mixed,fixed}}.csv")


if __name__ == "__main__":
    main()
