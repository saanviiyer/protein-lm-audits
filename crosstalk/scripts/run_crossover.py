"""Locate the budget at which counter-screening starts paying for itself.

At K=1 on a tight budget the affinity-only agent wins, because counter-screening
K off-targets costs 1+K assays and so cuts throughput by that factor. Somewhere
above that the specificity-aware agent overtakes it. That crossover budget B* is
the practically useful output: it converts "your objective is wrong" into "below
this many assays per off-target, screening-only is the better spend".

B* is estimated per K on a fine budget grid, with a bootstrap CI over seeds.
"""
import argparse, csv, pickle, sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from crosstalk import agents as A
from crosstalk import objectives as O
from crosstalk.landscape import Landscape
from crosstalk.objectives import is_specific

ON_MIN, TAU = 0.74, 0.65


def slice_partners(L, keep):
    return Landscape(seqs=L.seqs, partners=[L.partners[i] for i in keep],
                     F=L.F[:, keep], noise_sd=L.noise_sd[:, keep],
                     name=f"{L.name}_K{len(keep)-1}", wt=L.wt)


def crossover(budgets, succ_aff, succ_mar):
    """Smallest budget from which margin is ahead and stays ahead."""
    for i in range(len(budgets)):
        if all(succ_mar[j] >= succ_aff[j] for j in range(i, len(budgets))):
            return budgets[i]
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--landscape", default="data/absolut/landscape_5ag.pkl")
    ap.add_argument("--seeds", type=int, default=40)
    ap.add_argument("--budgets", type=int, nargs="+",
                    default=[50, 75, 100, 150, 200, 300, 450, 600, 900])
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--out", default="results/crossover.csv")
    args = ap.parse_args()

    full = pickle.load(open(args.landscape, "rb"))
    rows, per_seed = [], {}

    for K in (1, 2, 3, 4):
        L = slice_partners(full, [0] + list(range(1, K + 1)))
        off = tuple(range(1, K + 1))
        print(f"\nK={K} (counter-screen costs {1+K} assays/variant)", flush=True)
        for reward, counter in (("affinity", False), ("margin", True)):
            obj_r = O.make(reward, target=0, off=off)
            for budget in args.budgets:
                hits = []
                for s in range(args.seeds):
                    rng = np.random.default_rng(1000 * s + budget + 7 * K)
                    i = A.additive_model(L, obj_r, budget=budget,
                                         counter_screen=counter, rng=rng)
                    hits.append(bool(is_specific(L.F[i], target=0, off=off,
                                                 tau=TAU, on_min=ON_MIN)))
                per_seed[(K, reward, budget)] = np.array(hits)
                sr = float(np.mean(hits))
                rows.append(dict(K=K, reward=reward, budget=budget,
                                 variants_seen=budget // ((1 + K) if counter else 1),
                                 success_rate=sr, n_seeds=args.seeds))
                print(f"   {reward:9s} B={budget:4d} success={sr:.3f}", flush=True)

    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

    print(f"\n{'='*70}\nCROSSOVER BUDGET B* (bootstrap over seeds, {args.boot} resamples)\n{'='*70}")
    rng = np.random.default_rng(0)
    for K in (1, 2, 3, 4):
        a = [per_seed[(K, 'affinity', b)] for b in args.budgets]
        m = [per_seed[(K, 'margin', b)] for b in args.budgets]
        point = crossover(args.budgets, [x.mean() for x in a], [x.mean() for x in m])
        boots = []
        for _ in range(args.boot):
            idx = rng.integers(0, args.seeds, args.seeds)
            c = crossover(args.budgets, [x[idx].mean() for x in a], [x[idx].mean() for x in m])
            if c is not None:
                boots.append(c)
        if boots:
            lo, hi = np.percentile(boots, [2.5, 97.5])
            frac = len(boots) / args.boot
            print(f"  K={K}: B* = {point}  95% CI [{lo:.0f}, {hi:.0f}]  "
                  f"(a crossover existed in {100*frac:.0f}% of resamples)")
            print(f"        = {point/(1+K):.0f} assays per counter-screened variant")
        else:
            print(f"  K={K}: B* = {point} (bootstrap found no stable crossover)")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
