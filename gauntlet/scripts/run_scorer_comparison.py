#!/usr/bin/env python3
"""Does relaxing additivity rescue zero-shot scoring on multi-mutant campaigns?

Phase 3 showed the failure is not the model's competence at multi-mutants but
the fact that a SUM over mutated positions falls as edits accumulate. Three
scorers, increasingly free of that structure:

  masked_wt     mask position i in the WILD TYPE, sum over mutations.
                Each term is blind to the variant's other substitutions.
  mutant_marg   mask position i in the MUTANT, sum over mutations.
                Context-aware per term, but still a sum of k terms.
  seq_loglik    whole-sequence log-likelihood of mutant minus wild type.
                Not a sum over mutated positions at all.

If context-awareness is what matters, mutant_marg recovers. If the summation is
what matters, only seq_loglik does. If neither recovers, the problem is the
model's grasp of these enzymes, not the scoring protocol.

    python scripts/run_scorer_comparison.py
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from gauntlet.campaign import Policy, evaluate, mutation_features  # noqa: E402
from gauntlet.petase_data import load_corpus, load_sequences, resolve_wildtypes  # noqa: E402

SCORERS = ["masked_wt", "mutant_marg", "seq_loglik"]


def rho(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 5 or np.std(a[ok]) < 1e-12 or np.std(b[ok]) < 1e-12:
        return np.nan
    return float(stats.spearmanr(a[ok], b[ok]).statistic)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--petml_dir", default="data/petml")
    ap.add_argument("--out", default="results")
    ap.add_argument("--min_variants", type=int, default=20)
    ap.add_argument("--budget", type=int, default=3)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--seeds", type=int, default=400)
    ap.add_argument("--batch_size", type=int, default=4)
    args = ap.parse_args()

    df = load_corpus(args.petml_dir)
    seqs = load_sequences(os.path.join(args.petml_dir, "sequences.fasta"))
    wts = resolve_wildtypes(df, seqs)

    from gauntlet.proxies import ESM2Marginals
    sc = ESM2Marginals(batch_size=args.batch_size)
    print(f"ESM-2 on {sc.device}\n")

    rows = []
    for study, grp in df.groupby("study"):
        g = grp[grp.logActivity.notna()]
        if g.empty:
            continue
        g = g[[isinstance(m, list) and len(m) > 0 for m in g.muts]]  # list-comp: .apply on an empty Series returns a DataFrame
        g = g[g.scaffold.isin(wts)]
        if len(g) < args.min_variants:
            continue
        nmut = np.array([len(m) for m in g.muts], float)

        vals = {k: [] for k in SCORERS}
        for r in g.itertuples():
            info = wts[r.scaffold]
            wt, off = info["wt"], info["offset"]
            vals["masked_wt"].append(sc.score(wt, r.muts, off))
            vals["mutant_marg"].append(sc.score_mutant_marginals(wt, r.muts, off))
            vals["seq_loglik"].append(sc.score_sequence_loglik(wt, r.muts, off))

        y = g.logActivity.to_numpy(float)
        rec = {"study": study, "n": len(g), "frac_multi": float((nmut > 1).mean()),
               "max_k": int(nmut.max())}
        for k in SCORERS:
            v = np.array(vals[k], float)
            rec[f"rho_{k}"] = rho(v, y)
            rec[f"conf_{k}"] = rho(v, nmut)
            rec[f"rho_{k}_per_mut"] = rho(v / np.maximum(nmut, 1), y)

        # Does it change the decision? Replay selection with each scorer.
        X, _ = mutation_features([{"muts": m} for m in g.muts])
        rec["recall_random"] = evaluate(
            X, y, np.zeros(len(y)), Policy("random"),
            args.budget, args.rounds, seeds=args.seeds, top_frac=0.10)["top_recall"]
        for k in SCORERS:
            rec[f"recall_{k}"] = evaluate(
                X, y, np.nan_to_num(np.array(vals[k], float)), Policy("zero_shot"),
                args.budget, args.rounds, seeds=args.seeds, top_frac=0.10)["top_recall"]
        rows.append(rec)
        print(f"  {study:32s} n={len(g):3d} multi={rec['frac_multi']:.2f} "
              + " ".join(f"{k}={rec['rho_' + k]:+.3f}" for k in SCORERS), flush=True)

    res = pd.DataFrame(rows)
    os.makedirs(args.out, exist_ok=True)
    res.to_csv(os.path.join(args.out, "scorer_comparison.csv"), index=False)

    print("\n" + "=" * 78)
    print("Spearman vs measured activity, within campaign")
    print("=" * 78)
    print(res[["study", "n", "frac_multi", "max_k"]
              + [f"rho_{k}" for k in SCORERS]].round(3).to_string(index=False))

    print("\nCorrelation with mutation count (the confound):")
    print(res[["study", "frac_multi"] + [f"conf_{k}" for k in SCORERS]]
          .round(3).to_string(index=False))

    multi = res[res.frac_multi >= 0.5]
    single = res[res.frac_multi < 0.5]
    print("\nMean Spearman vs activity:")
    for label, sub in [("multi-mutant campaigns", multi), ("mostly-single", single)]:
        if sub.empty:
            continue
        print(f"  {label:24s} " + "  ".join(
            f"{k}={sub[f'rho_{k}'].mean():+.3f}" for k in SCORERS))

    print(f"\nSelection replay, top-10% recall (budget {args.budget} x {args.rounds}):")
    cols = ["recall_random"] + [f"recall_{k}" for k in SCORERS]
    print(res[["study", "frac_multi"] + cols].round(3).to_string(index=False))
    print("\n  mean  " + "  ".join(f"{c.replace('recall_','')}={res[c].mean():.3f}"
                                   for c in cols))
    print(f"\nwrote {args.out}/scorer_comparison.csv")


if __name__ == "__main__":
    main()
