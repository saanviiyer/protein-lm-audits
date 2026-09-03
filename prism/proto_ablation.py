#!/usr/bin/env python3
"""
proto_ablation.py

Does the physics-grounded generator actually help? Runs the PRISM/Proto
directed-design loop toward a target Pfam centroid twice from the same seed --
once with the COSMIC-signature generator (physics-grounded proposals) and once
with a random-mutation generator (matched baseline) -- and compares how close
each drives the design and how fast.

Outputs a comparison figure (best cosine distance to target vs. MCMC step for
both generators) and a summary (start distance, best distance, improvement,
steps-to-threshold, acceptance rate).

USAGE
-----
python proto_ablation.py \
    --wt_fasta all_runs/CL0072/pairs_cds.fasta \
    --target_pfam PF17041 --centroid_chunks_dir pfam_centroid_approach/seed_alignments/centroid_chunks \
    --signature_profile profiles/SBS-MS/SBS17a_PROFILE.txt \
    --steps 250 --output_dir results/proto_ablation
"""
import argparse
import os
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--wt_fasta", required=True)
    ap.add_argument("--target_pfam", default="PF17041")
    ap.add_argument("--centroid_chunks_dir", required=True)
    ap.add_argument("--signature_profile", required=True)
    ap.add_argument("--steps", type=int, default=250)
    ap.add_argument("--temperature", type=float, default=0.02)
    ap.add_argument("--n_mut", type=int, default=3, help="Random-generator mutations per step.")
    ap.add_argument("--output_dir", default="results/proto_ablation")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    os.makedirs(args.output_dir, exist_ok=True)

    from Bio import SeqIO
    from prism_proto_design import CentroidScorer, load_target_centroid, run_standalone
    from prism_utils import load_mutation_profile

    wt_cds = str(next(SeqIO.parse(args.wt_fasta, "fasta")).seq).upper().replace(" ", "")
    profile = load_mutation_profile(args.signature_profile)
    target = load_target_centroid(SimpleNamespace(
        target_centroid_npy=None, target_pfam=args.target_pfam,
        centroid_chunks_dir=args.centroid_chunks_dir))
    scorer = CentroidScorer(target)   # ESM-2 loaded once, shared across both runs

    curves = {}
    summary = []
    for gen in ("signature", "random"):
        out = os.path.join(args.output_dir, gen)
        run_args = SimpleNamespace(
            generator=gen, reset_each=False, n_mut=args.n_mut, steps=args.steps,
            temperature=args.temperature, plddt_min=0.0, seed=args.seed, output_dir=out,
            save_embeddings=True)
        print(f"\n=== generator: {gen} ===")
        run_standalone(run_args, scorer, wt_cds, profile, folder=None)
        tr = pd.read_csv(os.path.join(out, "trajectory.csv"))
        curves[gen] = tr
        start = float(tr["best_distance"].iloc[0])
        best = float(tr["best_distance"].min())
        acc = float(tr["accepted"].mean())
        # steps to reach 90% of the signature run's improvement (set after loop)
        summary.append({"generator": gen, "start": round(start, 4), "best": round(best, 4),
                        "improvement": round(start - best, 4), "accept_rate": round(acc, 3)})

    # steps-to-threshold: first step each run reaches the WORSE of the two final bests
    thresh = max(s["best"] for s in summary)
    for s in summary:
        tr = curves[s["generator"]]
        hit = tr.index[tr["best_distance"] <= thresh]
        s["steps_to_common_threshold"] = int(hit[0]) if len(hit) else None

    sm = pd.DataFrame(summary)
    sm.to_csv(os.path.join(args.output_dir, "proto_ablation_summary.csv"), index=False)
    print("\n=== Proto generator ablation (toward %s) ===" % args.target_pfam)
    print(sm.to_string(index=False))

    _plot(curves, args.target_pfam, os.path.join(args.output_dir, "proto_ablation.png"))
    print(f"\n[proto-abl] wrote {args.output_dir}/proto_ablation.png + summary.csv")


def _plot(curves, target, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = {"signature": "#c0392b", "random": "#7f8c8d"}
    labels = {"signature": "COSMIC-signature generator (physics-grounded)",
              "random": "random-mutation generator (baseline)"}
    for gen, tr in curves.items():
        ax.plot(tr["step"], tr["best_distance"], color=colors[gen], lw=2, label=labels[gen])
    ax.set_xlabel("MCMC step")
    ax.set_ylabel(f"best cosine distance to {target} centroid (ESM-2)")
    ax.set_title("Does the mutational-process generator help? (Proto directed design)")
    ax.legend()
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=200)


if __name__ == "__main__":
    main()
