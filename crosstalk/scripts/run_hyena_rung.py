#!/usr/bin/env python3
"""HyenaDNA arm: autoregressive, single-nucleotide, same audit.

Split out from run_genomic_rung.py only because it costs one forward pass per
sequence rather than three for the whole landscape, so it runs on its own clock.

Its role in the argument is to remove two explanations at once. If the Nucleotide
Transformer reproduces the protein rung's anti-correlation, someone can still say
it is an artifact of 6-mer tokenization straddling codons, or of masked-language
training. HyenaDNA has neither: it reads single nucleotides and it is trained to
predict the next one. Raw scores are cached so the analysis can be redone without
re-running the model.
"""
import argparse, csv, sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))

from crosstalk.boltz import MUT_POSITIONS, PARD3
from crosstalk.landscape import load_pard3
from crosstalk import glm
from run_proxy_ladder import auc, boot_ci, spearman
from run_genomic_rung import discrimination_set, report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="LongSafari/hyenadna-small-32k-seqlen-hf")
    ap.add_argument("--out", default="results/hyena_rung.csv")
    ap.add_argument("--cache", default="results/hyena_scores.npz")
    args = ap.parse_args()

    L = load_pard3()
    variants = L.seqs
    mask, lab, w3, w2 = discrimination_set(L)
    print(f"{len(variants)} variants; discrimination set = {int(mask.sum())}", flush=True)

    cds = glm.load_cds(); ctx = glm.load_context()
    wt_cds = cds["ParD3"]["cds"]
    operon = ctx["ParD3_ParE3"]["segment"]
    d3_off = ctx["ParD3_ParE3"]["offset_in_segment"]["ParD3"]
    noncog = ctx["ParD3_ParE2_synthetic"]["segment"]

    cache = ROOT / args.cache
    hs = glm.HyenaScorer(args.model)
    print(f"loaded {args.model} on {hs.device}", flush=True)

    def scores_for(tag, build):
        seqs = [build(v) for v in variants]
        print(f"  scoring {tag} ({len(seqs[0])} nt)...", flush=True)
        return hs.log_likelihood(seqs, progress=1000)

    blind = scores_for("CDS alone", lambda v: glm.variant_cds(v, wt_cds))
    aware3 = scores_for("real operon", lambda v: operon[:d3_off]
                        + glm.variant_cds(v, wt_cds) + operon[d3_off + len(wt_cds):])
    aware2 = scores_for("synthetic ParD3:ParE2", lambda v: glm.variant_cds(v, wt_cds)
                        + noncog[len(wt_cds):])
    np.savez(cache, variants=np.array(variants), blind=blind, aware3=aware3, aware2=aware2)
    print(f"cached raw scores -> {cache}")

    rows = []
    wt_code = "".join(PARD3[p - 1] for p in MUT_POSITIONS)
    nmut = np.array([sum(a != b for a, b in zip(v, wt_code)) for v in variants], float)
    report(rows, "baseline:mutation_count", -nmut, mask, lab, w3, w2,
           extra=dict(model="trivial", modality="none"))
    m = dict(model=args.model, modality="dna")
    report(rows, "HyenaDNA AR-LL partner-blind (CDS alone)", blind, mask, lab, w3, w2, extra=m)
    report(rows, "HyenaDNA AR-LL (REAL ParD3:ParE3 operon)", aware3, mask, lab, w3, w2, extra=m)
    report(rows, "HyenaDNA AR-LL (ParD3:ParE2, synthetic)", aware2, mask, lab, w3, w2, extra=m)
    report(rows, "HyenaDNA AR-LL MARGIN (E3 - E2)", aware3 - aware2, mask, lab, w3, w2, extra=m)

    out = ROOT / args.out
    keys = sorted({k for r in rows for k in r})
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
