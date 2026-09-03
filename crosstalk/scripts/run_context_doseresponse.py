#!/usr/bin/env python3
"""Does a genomic model need more genomic context? A dose-response on ParD3.

Section 18's standing objection is that a genomic LM handed an isolated coding
sequence was given the wrong input. The DMS assays can only partly answer it:
most EMBL records are gene-only entries with a few dozen bases of flank.

ParD3 can answer it properly, because the whole 6.9 Mb M. opportunistum
chromosome is on disk and the locus's true coordinates are known. So the amount
of real genomic context can be dialled up -- the actual upstream and downstream
sequence, not padding -- and the specificity signal measured at each step.

If context is what was missing, AUC should climb toward the mutation-count
baseline of 0.664 as flanks grow. If the signal is simply absent, it stays at
chance no matter how much of the genome is supplied. The flank is snapped to a
multiple of 6 so codons never straddle 6-mer token boundaries, which would
otherwise confound a context effect with a tokenisation artifact.
"""
import argparse, csv, json, sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))

from crosstalk.boltz import MUT_POSITIONS, PARD3
from crosstalk.landscape import load_pard3
from crosstalk import glm
from run_proxy_ladder import auc, boot_ci, spearman
from run_genomic_rung import discrimination_set

COMP = str.maketrans("ACGT", "TGCA")
GENOME = "CP002279"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="InstaDeepAI/nucleotide-transformer-v2-50m-multi-species")
    ap.add_argument("--flanks", type=int, nargs="+",
                    default=[0, 60, 300, 1200, 3000, 5400])
    ap.add_argument("--out", default="results/context_doseresponse.csv")
    args = ap.parse_args()

    L = load_pard3()
    variants = L.seqs
    mask, lab, w3, w2 = discrimination_set(L)
    cds = glm.load_cds()
    wt_cds = cds["ParD3"]["cds"]

    genome = (ROOT / "data" / "cds" / f"{GENOME}.txt").read_text().strip()
    oriented = genome if wt_cds in genome else genome.translate(COMP)[::-1]
    start = oriented.find(wt_cds)
    assert start >= 0, "ParD3 CDS not found in the genome"
    end = start + len(wt_cds)
    print(f"genome {len(genome):,} nt; ParD3 at {start:,}-{end:,} "
          f"(oriented so the CDS reads forward)\n", flush=True)

    sc = glm.NTScorer(args.model)
    wt_code = "".join(PARD3[p - 1] for p in MUT_POSITIONS)
    nmut = np.array([sum(a != b for a, b in zip(v, wt_code)) for v in variants], float)
    base_auc = auc(-nmut[mask], lab)
    print(f"{'flank (nt each side)':22s} {'total nt':>9s} {'tokens':>7s} "
          f"{'AUC':>7s} {'95% CI':>18s} {'rho_on':>8s} {'rho_off':>8s}")
    print(f"{'trivial: mut count':22s} {'':>9s} {'':>7s} {base_auc:7.3f}")

    rows = []
    for f in args.flanks:
        lo, hi = max(start - f, 0), min(end + f, len(oriented))
        lo += (start - lo) % 6                       # keep codons inside single tokens
        seg = oriented[lo:hi]
        off = start - lo
        assert seg[off:off + len(wt_cds)] == wt_cds
        n_tok = len(seg) // 6
        if n_tok > 2040:
            print(f"  flank {f}: {n_tok} tokens exceeds context, skipped")
            continue
        s = sc.score_variants_masked_marginal(variants, seg, cds_offset=off)
        a = auc(s[mask], lab)
        cl, ch = boot_ci(s[mask], lab)
        r_on, r_off = spearman(s, w3), spearman(s, w2)
        print(f"{f:22d} {len(seg):9,d} {n_tok:7d} {a:7.3f} "
              f"[{cl:6.3f},{ch:6.3f}] {r_on:+8.3f} {r_off:+8.3f}", flush=True)
        rows.append(dict(flank=f, total_nt=len(seg), tokens=n_tok, upstream=off,
                         auc=a, auc_lo=cl, auc_hi=ch, rho_on=r_on, rho_off=r_off,
                         baseline_auc=base_auc, model=args.model))

    if rows:
        a = np.array([r["auc"] for r in rows])
        print(f"\nAUC range across {len(rows)} context sizes: "
              f"{a.min():.3f} to {a.max():.3f}; baseline {base_auc:.3f}")
        print("trend (Spearman AUC vs flank): "
              f"{spearman([r['flank'] for r in rows], a):+.3f}")
        out = ROOT / args.out
        keys = sorted({k for r in rows for k in r})
        with out.open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=keys); w.writeheader(); w.writerows(rows)
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
