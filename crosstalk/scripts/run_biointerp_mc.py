#!/usr/bin/env python3
"""How much of a stochastic intervention's delta is the draw rather than the model?

Section 26 reported a synonymous-recode delta of +0.2334 from ONE draw of
codons, with no error bar for the draw itself. The biointerp reproduction, which
keys its generator differently, got +0.3300 on the identical genes and the
identical scorer. Either that gap is Monte-Carlo noise over which synonymous
codon was picked, or one of the two numbers is not a property of the model.

This measures it: the same intervention at R independent seeds, all else fixed.
The between-seed spread of the mean delta is the error bar section 26 omitted.
"""
import argparse, csv, sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))

from crosstalk.biointerp import interventions as ivm
from run_biointerp_battery import MODELS, load_sequences


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="nt50m", choices=sorted(MODELS))
    ap.add_argument("--interventions", nargs="+",
                    default=["synonymous recode", "codon-order shuffle"])
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--max-nt", type=int, default=2400)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    seqs = load_sequences(args.max_nt)
    keys = sorted(seqs)
    model = MODELS[args.model]()
    real = np.asarray(model.score([seqs[k] for k in keys]))
    print(f"{len(keys)} genes, {args.repeats} seeds, {model.name}\n", flush=True)

    rows = []
    for name in args.interventions:
        iv = ivm.get(name)
        means, highers = [], []
        for seed in range(args.seed0, args.seed0 + args.repeats):
            pert = [iv.apply(seqs[k], iv.rng_for(k, seed)) for k in keys]
            d = real - np.asarray(model.score(pert))
            means.append(float(d.mean()))
            highers.append(int((d > 0).sum()))
            print(f"  {name:22s} seed {seed}: {d.mean():+.4f}  "
                  f"real higher {highers[-1]}/{len(keys)}", flush=True)
            rows.append(dict(intervention=name, seed=seed, mean_delta=means[-1],
                             real_higher=highers[-1], n=len(keys),
                             **{f"d:{k}": float(v) for k, v in zip(keys, d)}))
        m = np.array(means)
        print(f"  {name:22s} across seeds: mean {m.mean():+.4f}, "
              f"sd {m.std(ddof=1):+.4f}, range [{m.min():+.4f}, {m.max():+.4f}], "
              f"real higher {min(highers)}-{max(highers)}/{len(keys)}\n", flush=True)

    out = ROOT / (args.out or f"results/biointerp_mc_{args.model}.csv")
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
