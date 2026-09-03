#!/usr/bin/env python3
"""Is a mutation count a better fallback than the developer's own model?

Phase 21 found that falling back to an in-house ridge, rather than to random,
recovers up to 23% of what a compromised scorer costs. It could not test the one
proxy that beats both language models on measured campaigns -- a bare count of
mutations -- because every assay in the scan testbed is single-mutant, so the
count is constant there.

The PETML corpus is multi-mutant and does vary in edit count within a study, so
the comparison is available. The question is narrow and does not need the
attacker simulated: once you have decided to abandon a supplied scorer, you must
rank the pool by something, and this measures how much each candidate ranker
captures.

IMPORTANT CAVEAT, which the paper must carry. A mutation count ranks well on this
corpus partly because published campaigns accumulate beneficial edits: later
variants carry more mutations AND are better, so the count is reading the
campaign's own optimisation history. A prospective design pool that you generate
yourself has no such structure. This measures a retrospective ceiling, not a
deployable defence, and we report it as such.

    python scripts/run_petml_fallback.py
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

ALPHA = 1.0
ELITE_Q = 0.20
MIN_N = 15
SEEDS = 200


def factorize(seqs):
    """One-hot the positions that vary within a study."""
    L = {len(s) for s in seqs}
    if len(L) != 1:
        return None
    arr = np.array([list(s) for s in seqs])
    var = [j for j in range(arr.shape[1]) if len(set(arr[:, j])) > 1]
    if not var:
        return None
    cols = []
    for j in var:
        for aa in sorted(set(arr[:, j])):
            cols.append((arr[:, j] == aa).astype(float))
    return np.array(cols).T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", default="results/scored_variants.csv")
    ap.add_argument("--out", default="results")
    ap.add_argument("--tag", default="esm2")
    args = ap.parse_args()

    d = pd.read_csv(args.scores)
    rows = []
    for tgt in ["logActivity", "Tm"]:
        sub = d.dropna(subset=[tgt, "sequence"])
        for study, g in sub.groupby("study"):
            g = g.reset_index(drop=True)
            if len(g) < MIN_N or g[tgt].nunique() < 3:
                continue
            X = factorize(list(g.sequence))
            if X is None:
                continue
            y = g[tgt].to_numpy(float)
            n = len(g)
            k = max(1, int(round(ELITE_Q * n)))
            elite = np.zeros(n, bool)
            elite[np.argsort(-y)[:k]] = True

            rankers = {
                "n_mut": g.n_mut.to_numpy(float),
                "blosum": g.blosum62.to_numpy(float),
                "hydropathy": -np.abs(g.hydropathy.to_numpy(float)),
                "supplied_scorer": g.esm2_wtm.to_numpy(float),
            }
            for seed in range(SEEDS):
                rng = np.random.default_rng(seed)
                obs = rng.choice(n, n // 2, replace=False)
                pool = np.setdiff1d(np.arange(n), obs)
                if len(pool) < 4 or elite[pool].sum() == 0:
                    continue
                B = max(2, int(round(0.25 * len(pool))))
                base = B * elite[pool].sum() / len(pool)

                mdl = Ridge(alpha=ALPHA).fit(X[obs], y[obs])
                r = dict(rankers)
                r["own_ridge"] = np.full(n, -np.inf)
                r["own_ridge"][pool] = mdl.predict(X[pool])

                got = float(elite[rng.choice(pool, B, replace=False)].sum())
                rows.append({"target": tgt, "study": study, "seed": seed,
                             "ranker": "random", "adv": got - base, "n": n})
                for name, sc in r.items():
                    pick = pool[np.argsort(-sc[pool])[:B]]
                    rows.append({"target": tgt, "study": study, "seed": seed,
                                 "ranker": name,
                                 "adv": float(elite[pick].sum()) - base, "n": n})

    t = pd.DataFrame(rows)
    os.makedirs(args.out, exist_ok=True)
    t.to_csv(os.path.join(args.out, f"petml_fallback_{args.tag}.csv"), index=False)

    rng = np.random.default_rng(0)
    order = ["random", "n_mut", "blosum", "hydropathy", "own_ridge", "supplied_scorer"]
    for tgt in ["logActivity", "Tm"]:
        s = t[t.target == tgt]
        studies = sorted(set(s.study))
        print(f"\n{'=' * 76}\n{tgt} — {len(studies)} studies, elite = top {ELITE_Q:.0%} "
              f"within study\n{'=' * 76}")
        print(f"{'ranker':<18}{'elite captured vs random':>26}{'95% CI (study bootstrap)':>28}")
        per = {r: s[s.ranker == r].groupby("study").adv.mean().reindex(studies).values
               for r in order}
        for r in order:
            v = per[r] - per["random"]
            reps = [np.mean(rng.choice(v, len(v), replace=True)) for _ in range(4000)]
            lo, hi = np.percentile(reps, [2.5, 97.5])
            star = "  *" if lo > 0 else ""
            print(f"{r:<18}{v.mean():>26.3f}{f'[{lo:+.3f}, {hi:+.3f}]':>28}{star}")
        nm, rg = per["n_mut"] - per["random"], per["own_ridge"] - per["random"]
        dv = nm - rg
        reps = [np.mean(rng.choice(dv, len(dv), replace=True)) for _ in range(4000)]
        lo, hi = np.percentile(reps, [2.5, 97.5])
        print(f"\n  n_mut minus own_ridge: {dv.mean():+.3f}  95% CI [{lo:+.3f}, {hi:+.3f}]"
              f"   (n_mut wins in {int((dv > 0).sum())}/{len(dv)} studies)")
    print(f"\nwrote {args.out}/petml_fallback_{args.tag}.csv")


if __name__ == "__main__":
    main()
