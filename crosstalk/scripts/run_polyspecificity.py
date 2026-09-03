"""Does specificity-aware optimization survive more off-targets?

ParD3 caps at one off-target, so `margin`'s max-over-off-targets was untested
beyond K=1. Absolut! gives ground-truth energies for the same sequences against
many antigens, so K can be varied directly.

The cost structure is what makes this a real trade rather than a free lunch:
screening the target costs 1 assay, counter-screening K off-targets costs 1+K.
As K grows, the affinity-only agent's throughput advantage grows too -- it sees
(1+K)x more variants than the agent that counter-screens each one.
"""
import argparse, csv, pickle, sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from crosstalk import agents as A
from crosstalk import objectives as O
from crosstalk.landscape import Landscape
from crosstalk.metrics import evaluate

ON_MIN, TAU = 0.74, 0.65


def slice_partners(L: Landscape, keep: list[int]) -> Landscape:
    return Landscape(seqs=L.seqs, partners=[L.partners[i] for i in keep],
                     F=L.F[:, keep], noise_sd=L.noise_sd[:, keep],
                     name=f"{L.name}_K{len(keep)-1}", wt=L.wt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--landscape", default="data/absolut/landscape_5ag.pkl")
    ap.add_argument("--seeds", type=int, default=30)
    ap.add_argument("--budgets", type=int, nargs="+", default=[100, 300, 900])
    ap.add_argument("--out", default="results/polyspecificity.csv")
    args = ap.parse_args()

    full = pickle.load(open(args.landscape, "rb"))
    print(f"{full.n_seqs} seqs, partners {full.partners}")
    print(f"target = {full.partners[0]}; success = on>={ON_MIN} and every off<={TAU}\n")

    rows = []
    for K in (1, 2, 3, 4):
        L = slice_partners(full, [0] + list(range(1, K + 1)))
        off = tuple(range(1, K + 1))
        obj_eval = O.make("margin", target=0, off=off)
        n_spec = int(((L.F[:, 0] >= ON_MIN) & (L.F[:, 1:].max(axis=1) <= TAU)).sum())
        print(f"K={K} off-targets {L.partners[1:]}  ({n_spec} specific sequences exist)")

        for reward, counter in (("affinity", False), ("margin", True)):
            obj_r = O.make(reward, target=0, off=off)
            for budget in args.budgets:
                noms = []
                for s in range(args.seeds):
                    rng = np.random.default_rng(1000 * s + budget + 7 * K)
                    i = A.additive_model(L, obj_r, budget=budget,
                                         counter_screen=counter, rng=rng)
                    noms.append(L.seqs[i])
                m = evaluate(L, noms, obj_eval, tau=TAU, on_min=ON_MIN)
                cost = (1 + K) if counter else 1
                rows.append(dict(K=K, reward=reward, counter_screen=int(counter),
                                 budget=budget, assay_cost=cost,
                                 variants_seen=budget // cost, **m))
                print(f"   {reward:9s} B={budget:4d} ({budget//cost:4d} variants) "
                      f"success={m['success_rate']:.2f} crosstalk={m['crosstalk_rate']:.2f}",
                      flush=True)
        print()

    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
