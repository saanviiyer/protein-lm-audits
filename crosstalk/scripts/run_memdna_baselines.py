#!/usr/bin/env python3
"""Trivial baselines for the DNA memorisation spans.

Per-nucleotide accuracy on DNA is dominated by base composition: on a 63% GC
genome, always writing G scores 0.315 without looking at anything. Any genomic-LM
reconstruction number is uninterpretable without that floor beside it, and
without a local-context model that costs nothing to fit.

  composition   always emit the commonest base in this window
  markov5       order-5 chain fitted to the SOURCE GENOME, greedy left-to-right
                fill of the masked span from its true left context

Windows and span placements are regenerated from the same seeds the model run
used, so rows join on (wid, L, span_start) with no extra state.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import memdna_data as D
from run_memorisation_dna import build_windows, plan_spans

BASES = "ACGT"
BIDX = {b: i for i, b in enumerate(BASES)}


def markov_logp(chain, left5: str, span: str) -> float:
    """Total log-prob the order-5 chain assigns to the span, given true context.

    Used as the reference model in a likelihood-ratio membership attack: the
    quantity that separates a memorised sample from a merely predictable one is
    the model's loss ABOVE what a cheap local model already achieves.
    """
    ctx = 0
    for c in left5:
        ctx = ctx * 4 + BIDX[c]
    mod = 4 ** chain.k
    tot = 0.0
    for c in span:
        j = BIDX[c]
        tot += float(np.log(chain.p[ctx, j]))
        ctx = (ctx * 4 + j) % mod
    return tot


def markov_fill(chain, left5: str, n: int) -> str:
    """Greedy order-5 continuation of length n from a true 5-nt left context."""
    ctx = 0
    for c in left5:
        ctx = ctx * 4 + BIDX[c]
    mod = 4 ** chain.k
    out = []
    for _ in range(n):
        j = int(chain.p[ctx].argmax())
        out.append(BASES[j])
        ctx = (ctx * 4 + j) % mod
    return "".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--window", type=int, default=3000)
    ap.add_argument("--lengths", type=int, nargs="+", default=[1, 2, 5, 10, 20, 40])
    ap.add_argument("--n-spans", type=int, default=6)
    ap.add_argument("--ann-lengths", type=int, nargs="+", default=[1, 2, 5])
    ap.add_argument("--n-ann", type=int, default=4)
    ap.add_argument("--n-pool", type=int, default=24)
    ap.add_argument("--n-focus", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/memorisation_dna_baselines.csv")
    args = ap.parse_args()

    print("rebuilding the identical window set ...", flush=True)
    wins = build_windows(args.window, args.seed, args.n_pool, args.n_focus)

    gen = ROOT / "data" / "genomes"
    src = {}                       # genome label -> fasta dict, for chain fitting
    for name, fn in D.MEMBERS.items():
        src[name] = gen / fn
    for name, fn in D.NONMEMBERS.items():
        src[name] = gen / fn
    chains: dict[str, D.Markov5] = {}

    def chain_for(g):
        if g not in chains:
            if g == "CP002279":
                chains[g] = D.Markov5(D.load_opportunistum())
            elif g in src:
                chains[g] = D.Markov5(D.load_fasta(src[g]))
            else:
                chains[g] = None
        return chains[g]

    real = {"member", "nonmember", "member_offframe", "meso_member", "meso_non",
            "opportunistum"}
    ntok = args.window // 6
    out = (ROOT / args.out).open("w", newline="")
    w = csv.DictWriter(out, fieldnames=["wid", "arm", "genome", "L", "span_start",
                                        "floor_nt", "mk5_nt", "mk5_tok", "mk5_exact",
                                        "mk5_logp_tok"])
    w.writeheader()
    for n, rec in enumerate(wins, 1):
        seq = rec["seq"]
        cnt = np.array([seq.count(b) for b in BASES], float)
        floor = cnt.max() / len(seq)
        ch = chain_for(rec["genome"]) if rec["arm"] in real else None
        rng = np.random.default_rng(args.seed * 100003 + rec["wid"])
        for L, st, tag in plan_spans(rec, ntok, args.lengths, args.n_spans,
                                     set(args.ann_lengths), args.n_ann, rng):
            row = dict(wid=rec["wid"], arm=rec["arm"], genome=rec["genome"],
                       L=L, span_start=st, floor_nt=round(floor, 5),
                       mk5_nt="", mk5_tok="", mk5_exact="", mk5_logp_tok="")
            if ch is not None and st * 6 >= 5:
                truth = seq[st * 6:(st + L) * 6]
                pred = markov_fill(ch, seq[st * 6 - 5:st * 6], 6 * L)
                hits = sum(a == b for a, b in zip(pred, truth))
                tk = sum(pred[i:i + 6] == truth[i:i + 6] for i in range(0, 6 * L, 6))
                row.update(mk5_nt=round(hits / (6 * L), 5), mk5_tok=round(tk / L, 5),
                           mk5_exact=int(tk == L),
                           mk5_logp_tok=round(
                               markov_logp(ch, seq[st * 6 - 5:st * 6], truth) / L, 4))
            w.writerow(row)
        if n % 100 == 0:
            print(f"  {n}/{len(wins)}", flush=True)
    out.close()
    print(f"wrote {ROOT / args.out}")


if __name__ == "__main__":
    main()
