#!/usr/bin/env python3
"""Audit zero-shot fitness proxies against measured PET-hydrolase outcomes.

Runs the Gauntlet battery over the PETML corpus (33 papers, ~500 engineered
variants of 66 scaffolds, with measured activity and melting temperature) and
asks whether an ESM-2 zero-shot score ranks real enzymes better than a
substitution matrix or a bare mutation count does.

    python scripts/run_petase_audit.py --petml_dir data/petml --out results
"""

import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from gauntlet import controls, proxies  # noqa: E402
from gauntlet.petase_data import load_corpus, load_sequences, resolve_wildtypes  # noqa: E402

PROXIES = ["esm2_wtm", "blosum62", "hydropathy", "n_mut"]
BASELINES = ["blosum62", "hydropathy", "n_mut"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--petml_dir", default="data/petml")
    ap.add_argument("--out", default="results")
    ap.add_argument("--model", default=None)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--scorer", default="esm2", choices=["esm2", "esmc"],
                    help="which pLM family to audit; ESM-C is a different "
                         "architecture and lab, so agreement between them is "
                         "evidence about the class of scorer, not one checkpoint")
    ap.add_argument("--tag", default="", help="suffix for output files")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    df = load_corpus(args.petml_dir)
    seqs = load_sequences(os.path.join(args.petml_dir, "sequences.fasta"))
    wts = resolve_wildtypes(df, seqs)
    print(f"corpus: {len(df)} rows, {df.study.nunique()} studies, "
          f"{df.logActivity.notna().sum()} activity, {df.Tm.notna().sum()} Tm")

    df["blosum62"] = df["muts"].apply(proxies.blosum_score)
    df["hydropathy"] = df["muts"].apply(proxies.hydropathy_shift)

    print(f"loading {args.model} ...")
    if args.scorer == "esm2":
        scorer = proxies.ESM2Marginals(args.model or "facebook/esm2_t33_650M_UR50D",
                                       batch_size=args.batch_size)
    else:
        scorer = proxies.ESMCMarginals(args.model or "esmc_300m",
                                       batch_size=args.batch_size)
    print(f"  device={scorer.device}")

    # Mask only the residues the corpus actually mutates.
    needed = {}
    for r in df.itertuples():
        info = wts.get(r.scaffold)
        if r.muts and info:
            needed.setdefault(r.scaffold, set()).update(
                p - 1 + info["offset"] for _, p, _ in r.muts)
    total_pos = sum(len(v) for v in needed.values())
    print(f"masking {total_pos} positions across {len(needed)} scaffolds")
    for i, (sc, pos) in enumerate(sorted(needed.items(), key=lambda kv: -len(kv[1])), 1):
        scorer.logprobs_at(wts[sc]["wt"], pos)
        print(f"  [{i}/{len(needed)}] {sc}: {len(pos)} positions", flush=True)

    def esm(row):
        info = wts.get(row.scaffold)
        if info is None or not row.muts:
            return 0.0 if row.muts == [] else None
        return scorer.score(info["wt"], row.muts, info["offset"])

    df["esm2_wtm"] = [esm(r) for r in df.itertuples()]
    df.drop(columns=["muts"]).to_csv(os.path.join(args.out, f"scored_variants{args.tag}.csv"), index=False)

    report = {}
    for target in ["logActivity", "Tm"]:
        per_study = controls.within_study_rank(df, PROXIES, target)
        summary = controls.pooled_summary(per_study, PROXIES)
        per_study.to_csv(os.path.join(args.out, f"per_study_{target}{args.tag}.csv"), index=False)
        summary.to_csv(os.path.join(args.out, f"summary_{target}{args.tag}.csv"), index=False)

        print(f"\n{'='*66}\nTARGET: {target}   "
              f"({per_study.n.sum()} variants across {len(per_study)} studies)\n{'='*66}")
        print(summary.to_string(index=False))

        margin = controls.trivial_baseline_margin(summary, "esm2_wtm", BASELINES)
        if margin:
            print(f"\n  Control 1 (trivial-baseline dominance): ESM-2 rho={margin['learned_rho']:+.3f} "
                  f"vs best baseline {margin['best_baseline']} rho={margin['best_baseline_rho']:+.3f}"
                  f"  ->  margin {margin['margin']:+.3f}")

        part = controls.partial_out(df, "esm2_wtm", target, "n_mut")
        if not part.empty:
            # Renormalise the weights over the studies that actually produced a
            # value. Weighting over all rows while the sum silently skips NaNs
            # under-weights by the missing study's share -- worth 0.002-0.003
            # here, but it is a bias, not noise.
            def _wmean(col):
                d = part[[col, "n"]].dropna()
                return float((d[col] * d.n / d.n.sum()).sum()) if len(d) else float("nan")
            raw, par = _wmean("raw_rho"), _wmean("partial_rho")
            print(f"  Control 4 (mutation-count confound): raw rho={raw:+.3f} -> "
                  f"partial rho={par:+.3f}  (proxy vs n_mut itself: "
                  f"{_wmean('proxy_vs_nuisance'):+.3f})")
            part.to_csv(os.path.join(args.out, f"partial_{target}{args.tag}.csv"), index=False)
            report[target] = {"margin": margin, "raw": raw, "partial": par}

    print(f"\nwrote {args.out}/")
    return report


if __name__ == "__main__":
    main()
