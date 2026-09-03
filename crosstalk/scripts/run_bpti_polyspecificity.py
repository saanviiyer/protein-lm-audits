"""Polyspecificity on measured data.

Section 7 showed the affinity objective anti-scales in both budget and number of
off-targets, but on the Absolut lattice, which is simulated. BPTI has three
measured partners, so K can be varied from 1 to 2 on real wet-lab numbers. The
question is whether the direction survives leaving simulation.

Cost model is unchanged: screening the target costs 1 assay, counter-screening K
off-targets costs 1+K, so the affinity-only agent sees (1+K)x more variants.
"""
import argparse, csv, sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from crosstalk import agents as A
from crosstalk import objectives as O
from crosstalk.bpti import load_bpti
from crosstalk.landscape import Landscape
from crosstalk.objectives import is_specific


def slice_partners(L, keep):
    return Landscape(seqs=L.seqs, partners=[L.partners[i] for i in keep],
                     F=L.F[:, keep], noise_sd=L.noise_sd[:, keep],
                     name=f"{L.name}_K{len(keep)-1}", wt=L.wt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=30)
    ap.add_argument("--budgets", type=int, nargs="+", default=[100, 300, 900])
    ap.add_argument("--on-pct", type=float, default=90)
    ap.add_argument("--off-pct", type=float, default=75)
    ap.add_argument("--out", default="results/bpti_polyspecificity.csv")
    args = ap.parse_args()

    full = load_bpti()
    ON_MIN = float(np.percentile(full.F[:, 0], args.on_pct))
    TAU = float(np.percentile(full.F[:, 1:], args.off_pct))
    print(f"{full.n_seqs} variants; target={full.partners[0]}; "
          f"success = on>={ON_MIN:.3f} and every off<={TAU:.3f}\n")

    # K=1 twice (each off-target alone) then K=2 (both), all real measurements
    configs = [("chymotrypsin", [0, 1]), ("mesotrypsin", [0, 2]), ("both", [0, 1, 2])]
    rows = []
    for label, keep in configs:
        L = slice_partners(full, keep)
        K = len(keep) - 1
        off = tuple(range(1, K + 1))
        obj_eval = O.make("margin", target=0, off=off)
        n_spec = int(((L.F[:, 0] >= ON_MIN) & (L.F[:, 1:].max(axis=1) <= TAU)).sum())
        print(f"K={K} off-targets={label}  ({n_spec} specific variants exist)")

        for reward, counter in (("affinity", False), ("margin", True)):
            obj_r = O.make(reward, target=0, off=off)
            for budget in args.budgets:
                hits = []
                for s in range(args.seeds):
                    rng = np.random.default_rng(1000 * s + budget + 31 * K)
                    i = A.additive_model(L, obj_r, budget=budget,
                                         counter_screen=counter, rng=rng)
                    hits.append(bool(is_specific(L.F[i], target=0, off=off,
                                                 tau=TAU, on_min=ON_MIN)))
                    if s == 0:
                        pass
                sr = float(np.mean(hits))
                offmax = np.array([L.F[i, 1:].max() for i in [0]])
                cost = (1 + K) if counter else 1
                rows.append(dict(off_targets=label, K=K, reward=reward,
                                 counter_screen=int(counter), budget=budget,
                                 variants_seen=budget // cost, success_rate=sr,
                                 n_seeds=args.seeds))
                print(f"   {reward:9s} B={budget:4d} ({budget//cost:4d} variants) "
                      f"success={sr:.2f}", flush=True)
        print()

    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
