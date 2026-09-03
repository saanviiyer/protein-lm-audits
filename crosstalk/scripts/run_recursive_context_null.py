#!/usr/bin/env python3
"""The shuffled twin of the context dose-response, with seeds, in BOTH modes.

The main run (scripts/run_recursive_context.py) shows the recursive arm's AUC
decaying with chunk count on real chromosome AND on shuffled flanks, but with one
shuffle seed there is no error bar on that control. This adds seeds, and extends
the same control to the DIRECT arm on section 19's own flank grid -- which asks a
question section 19 never asked: is its -0.771 trend a property of REAL genomic
context, or of any composition-matched sequence of that length?

Nothing here is a new model or a new metric. Same scorer, same discrimination
set, same AUC, same chunking, same pre-registered mean aggregation.
"""
import argparse, csv, sys, time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))

from crosstalk.landscape import load_pard3
from crosstalk import glm
from run_proxy_ladder import auc, boot_ci, spearman
from run_genomic_rung import discrimination_set
from recursive_context import CHUNK, MarkovScorer, aggregate, chunk_inputs, \
    dinuc_shuffle, mono_shuffle

COMP = str.maketrans("ACGT", "TGCA")
S19_FLANKS = [0, 60, 300, 1200, 3000, 5400]
REC_FLANKS = [1440, 2880, 4320, 5760, 8640, 11520, 14400, 17280, 20160]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--chunk", type=int, default=CHUNK)
    ap.add_argument("--out", default="results/recursive_context_null.csv")
    args = ap.parse_args()
    import torch; torch.set_num_threads(2)

    L = load_pard3(); variants = L.seqs
    mask, lab, w3, w2 = discrimination_set(L)
    wt_cds = glm.load_cds()["ParD3"]["cds"]
    g = (ROOT / "data" / "cds" / "CP002279.txt").read_text().strip()
    oriented = g if wt_cds in g else g.translate(COMP)[::-1]
    start = oriented.find(wt_cds); end = start + len(wt_cds)
    sc = glm.NTScorer()

    def metrics(s):
        a = auc(s[mask], lab); lo, hi = boot_ci(s[mask], lab)
        return dict(auc=a, auc_lo=lo, auc_hi=hi,
                    rho_on=spearman(s, w3), rho_off=spearman(s, w2))

    rows = []
    # ------- direct arm on section 19's own grid, real vs shuffled, n seeds ----
    for kind in ("shuffle_mono", "shuffle_dinuc"):
        fn = mono_shuffle if kind == "shuffle_mono" else dinuc_shuffle
        for seed in range(args.seeds):
            for F in S19_FLANKS:
                if F == 0:
                    continue
                u = fn(oriented[start - F:start], seed * 7919 + F)
                d = fn(oriented[end:end + F], seed * 7919 + F + 1)
                s = sc.score_variants_masked_marginal(variants, u + wt_cds + d, cds_offset=F)
                m = metrics(s)
                rows.append(dict(mode="direct", transform=kind, seed=seed, flank=F,
                                 n_chunks=1, **m))
                print(f"direct    {kind:14s} s{seed} F={F:6d} AUC {m['auc']:.3f}", flush=True)
    # ------- recursive arm, shuffled flanks, n seeds ---------------------------
    kmax = max(REC_FLANKS) // args.chunk
    for kind in ("shuffle_mono", "shuffle_dinuc"):
        fn = mono_shuffle if kind == "shuffle_mono" else dinuc_shuffle
        for seed in range(args.seeds):
            def tr(seq, tag, _s=seed, _f=fn):
                return _f(seq, (abs(hash(tag)) + _s * 104729) % (2 ** 31))
            per_chunk = np.zeros((kmax, len(variants)))
            for i in range(kmax):
                ins = chunk_inputs(oriented, start, end, (i + 1) * args.chunk,
                                   chunk=args.chunk, transform=tr)
                seq, off = ins[-1]
                per_chunk[i] = sc.score_variants_masked_marginal(variants, seq, cds_offset=off)
            for F in REC_FLANKS:
                k = F // args.chunk
                m = metrics(aggregate(per_chunk[:k], "mean"))
                rows.append(dict(mode="recursive", transform=kind, seed=seed,
                                 flank=F, n_chunks=k, **m))
            a = [r for r in rows if r["mode"] == "recursive" and r["transform"] == kind
                 and r["seed"] == seed]
            print(f"recursive {kind:14s} s{seed} trend "
                  f"{spearman([r['flank'] for r in a], [r['auc'] for r in a]):+.3f}", flush=True)

    p = ROOT / args.out
    keys = sorted({k for r in rows for k in r})
    with p.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys); w.writeheader(); w.writerows(rows)
    print(f"wrote {p}")


if __name__ == "__main__":
    main()
