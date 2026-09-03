#!/usr/bin/env python3
"""The genomic rung: does the anti-correlation survive crossing to DNA?

Protocol is deliberately identical to scripts/run_proxy_ladder.py -- same
discrimination set, same AUC, same bootstrap, same trivial baseline -- so the
only thing that changes between the protein rungs and this one is the model and
the modality. Anything that differs in the result cannot be attributed to a
differently-scored task.

Three arms that the protein rung could not run:

  real operon      partner-aware context taken from the genome rather than built
                   with an invented linker.
  synonymous floor within-variant score spread across encodings that translate
                   identically, and therefore share a label exactly. This upper-
                   bounds how much of any genomic score could possibly be signal.
  autoregressive   HyenaDNA, single-nucleotide and causal, so neither 6-mer
                   tokenization nor masked training can explain a shared result.
"""
import argparse, csv, json, sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from crosstalk.boltz import MUT_POSITIONS, PARD3
from crosstalk.landscape import load_pard3
from crosstalk import glm
from run_proxy_ladder import auc, boot_ci, spearman

NT_MODELS = ["InstaDeepAI/nucleotide-transformer-v2-50m-multi-species"]


def discrimination_set(L):
    w3, w2 = L.F[:, 0], L.F[:, 1]
    specific = (w3 >= 0.8) & (w2 <= 0.2)
    promisc = (w3 >= 0.8) & (w2 >= 0.6)
    mask = specific | promisc
    return mask, specific[mask], w3, w2


def report(rows, name, scores, mask, lab, w3, w2, extra=None):
    a = auc(scores[mask], lab)
    lo, hi = boot_ci(scores[mask], lab)
    row = dict(arm=name, auc=a, auc_lo=lo, auc_hi=hi,
               rho_on_target=spearman(scores, w3), rho_off_target=spearman(scores, w2),
               rho_margin_full=spearman(scores, w3 - w2),
               rho_margin_disc=spearman(scores[mask], (w3 - w2)[mask]))
    row.update(extra or {})
    rows.append(row)
    print(f"{name:44s} AUC {a:.3f} [{lo:.3f}, {hi:.3f}]   "
          f"rho_on {row['rho_on_target']:+.3f}  rho_off {row['rho_off_target']:+.3f}",
          flush=True)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=NT_MODELS)
    ap.add_argument("--hyena", default="LongSafari/hyenadna-small-32k-seqlen-hf")
    ap.add_argument("--skip-hyena", action="store_true")
    ap.add_argument("--syn-variants", type=int, default=40,
                    help="variants sampled for the synonymous-floor estimate")
    ap.add_argument("--syn-n", type=int, default=12,
                    help="synonymous encodings per sampled variant")
    ap.add_argument("--syn-policy", default="uniform", choices=["uniform", "usage"],
                    help="uniform = widest synonymous ensemble (loosest floor); "
                         "usage = weighted by genome codon usage (what a real "
                         "pipeline would pick, so a tighter and fairer floor)")
    ap.add_argument("--out", default="results/genomic_rung.csv")
    args = ap.parse_args()

    L = load_pard3()
    variants = L.seqs
    mask, lab, w3, w2 = discrimination_set(L)
    print(f"{len(variants)} variants; discrimination set = {int(mask.sum())} "
          f"({int(lab.sum())} specific vs {int((~lab).sum())} promiscuous)\n")

    cds = glm.load_cds()
    ctx = glm.load_context()
    wt_cds = cds["ParD3"]["cds"]
    assert glm.translate(wt_cds).rstrip("*") == PARD3

    operon = ctx["ParD3_ParE3"]["segment"]
    d3_off = ctx["ParD3_ParE3"]["offset_in_segment"]["ParD3"]
    assert operon[d3_off:d3_off + len(wt_cds)] == wt_cds, "ParD3 not where the context says"
    noncog = ctx["ParD3_ParE2_synthetic"]["segment"]
    assert noncog[:len(wt_cds)] == wt_cds

    rows = []
    wt_code = "".join(PARD3[p - 1] for p in MUT_POSITIONS)
    nmut = np.array([sum(a != b for a, b in zip(v, wt_code)) for v in variants], float)
    report(rows, "baseline:mutation_count", -nmut, mask, lab, w3, w2,
           extra=dict(model="trivial", modality="none"))

    for name in args.models:
        print(f"\n=== {name} ===", flush=True)
        sc = glm.NTScorer(name)
        blind = sc.score_variants_masked_marginal(variants, wt_cds, cds_offset=0)
        report(rows, f"NT partner-blind (CDS alone)", blind, mask, lab, w3, w2,
               extra=dict(model=name, modality="dna"))
        aware3 = sc.score_variants_masked_marginal(variants, operon, cds_offset=d3_off)
        report(rows, f"NT partner-aware (REAL ParD3:ParE3 operon)", aware3, mask, lab, w3, w2,
               extra=dict(model=name, modality="dna"))
        aware2 = sc.score_variants_masked_marginal(variants, noncog, cds_offset=0)
        report(rows, f"NT partner-aware (ParD3:ParE2, synthetic)", aware2, mask, lab, w3, w2,
               extra=dict(model=name, modality="dna"))
        report(rows, f"NT partner-aware MARGIN (E3 - E2)", aware3 - aware2, mask, lab, w3, w2,
               extra=dict(model=name, modality="dna"))

        # ---- synonymous floor -------------------------------------------------
        idx = np.where(mask)[0]
        rng = np.random.default_rng(0)
        pick = rng.choice(idx, size=min(args.syn_variants, len(idx)), replace=False)
        within, sizes = [], []
        for vi in pick:
            ens = glm.synonymous_ensemble(variants[vi], wt_cds, args.syn_n, seed=int(vi),
                                          policy=args.syn_policy)
            if len(ens) < 2:
                continue
            s = sc.pseudo_likelihood(ens)
            within.append(np.std(s, ddof=1))
            sizes.append(len(ens))
        canon = [glm.variant_cds(variants[vi], wt_cds) for vi in pick]
        between = sc.pseudo_likelihood(canon)
        w_sd, b_sd = float(np.mean(within)), float(np.std(between, ddof=1))
        print(f"\nsynonymous floor ({len(within)} variants, mean ensemble "
              f"{np.mean(sizes):.1f} encodings):")
        print(f"    within-variant SD (label-irrelevant) : {w_sd:.3f}")
        print(f"    between-variant SD (canonical)       : {b_sd:.3f}")
        print(f"    ratio within/between                 : {w_sd/b_sd:.3f}")
        signal_sd = float(np.sqrt(max(b_sd**2 - w_sd**2, 0.0)))
        atten = signal_sd / b_sd if b_sd > 0 else float("nan")
        print(f"    implied signal SD (noise-corrected)  : {signal_sd:.3f}")
        print(f"    attenuation factor on any rho        : {atten:.3f}")
        rows.append(dict(arm="synonymous_floor", model=name, modality="dna",
                         syn_policy=args.syn_policy, syn_signal_sd=signal_sd,
                         syn_attenuation=atten,
                         syn_within_sd=w_sd, syn_between_sd=b_sd,
                         syn_ratio=w_sd / b_sd, syn_n_variants=len(within),
                         syn_mean_ensemble=float(np.mean(sizes))))
        del sc

    if not args.skip_hyena:
        print(f"\n=== {args.hyena} ===", flush=True)
        hs = glm.HyenaScorer(args.hyena)
        seqs = [glm.variant_cds(v, wt_cds) for v in variants]
        ll = hs.log_likelihood(seqs, progress=1000)
        report(rows, "HyenaDNA autoregressive LL (CDS alone)", ll, mask, lab, w3, w2,
               extra=dict(model=args.hyena, modality="dna"))

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({k for r in rows for k in r})
    with out.open("w", newline="") as f:
        wtr = csv.DictWriter(f, fieldnames=keys)
        wtr.writeheader()
        wtr.writerows(rows)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
