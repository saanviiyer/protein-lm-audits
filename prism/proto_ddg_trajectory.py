#!/usr/bin/env python3
"""
proto_ddg_trajectory.py

Tracks the design's predicted STABILITY along the Proto directed-design walk, not
just its distance to the target. At each accepted MCMC step we record both
(a) the ESM-2 cosine distance to the target Pfam centroid (the optimized
objective) and (b) the ESM-2 masked-LM Delta-Delta-G proxy of the design relative
to the wild-type (its predicted stability change). This exposes the
energetic cost of signature-driven functional repositioning -- a physics-aligned
readout of the design loop, and the motivation for the ESMFold foldability gate.

USAGE (needs torch+transformers)
-----
python proto_ddg_trajectory.py --wt_fasta all_runs/CL0072/pairs_cds.fasta \
    --target_pfam PF17041 --centroid_chunks_dir pfam_centroid_approach/seed_alignments/centroid_chunks \
    --signature_profile profiles/SBS-MS/SBS17a_PROFILE.txt \
    --steps 200 --n_mut 65 --output_dir results/proto_ddg
"""
import argparse
import math
import os
import random
import sys
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
AA = "LAGVSERTIDPKQNFYMHWC"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--wt_fasta", required=True)
    ap.add_argument("--target_pfam", default="PF17041")
    ap.add_argument("--centroid_chunks_dir", required=True)
    ap.add_argument("--signature_profile", required=True)
    ap.add_argument("--generator", choices=["signature", "random"], default="signature")
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--n_mut", type=int, default=65)
    ap.add_argument("--temperature", type=float, default=0.02)
    ap.add_argument("--esm_model", default="facebook/esm2_t6_8M_UR50D")
    ap.add_argument("--max_aa", type=int, default=1022)
    ap.add_argument("--output_dir", default="results/proto_ddg")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    os.makedirs(args.output_dir, exist_ok=True)

    import torch
    from Bio import SeqIO
    from transformers import AutoTokenizer, EsmForMaskedLM
    from prism_proto_design import CentroidScorer, load_target_centroid, signature_proposal, random_aa_proposal_cds
    from prism_utils import translate_cds, load_mutation_profile

    wt_cds = str(next(SeqIO.parse(args.wt_fasta, "fasta")).seq).upper().replace(" ", "")
    profile = load_mutation_profile(args.signature_profile)
    target = load_target_centroid(SimpleNamespace(
        target_centroid_npy=None, target_pfam=args.target_pfam,
        centroid_chunks_dir=args.centroid_chunks_dir))
    scorer = CentroidScorer(target)          # ESM-2 embeddings for distance
    wt_aa = str(translate_cds(wt_cds))

    # ESM-2 masked-LM head for the ddG proxy (wt-marginals, computed once).
    tok = AutoTokenizer.from_pretrained(args.esm_model)
    mlm = EsmForMaskedLM.from_pretrained(args.esm_model).eval()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    mlm = mlm.to(dev)

    @torch.no_grad()
    def wt_logprobs(aa):
        enc = tok(aa[:args.max_aa], return_tensors="pt").to(dev)
        return torch.log_softmax(mlm(**enc).logits[0], dim=-1)

    lp = wt_logprobs(wt_aa)
    aid = {a: tok.convert_tokens_to_ids(a) for a in AA}
    L = min(len(wt_aa), args.max_aa)

    def ddg(design_aa):
        n = min(L, len(design_aa))
        s = 0.0
        for i in range(n):
            a, b = wt_aa[i], design_aa[i]
            if a != b and a in aid and b in aid:
                s += float(lp[i + 1, aid[b]] - lp[i + 1, aid[a]])
        return s

    rng = random.Random(args.seed)
    cur_cds, cur_aa = wt_cds, wt_aa
    cur_d = float(scorer.distances([cur_aa])[0])
    rows = [[0, cur_d, 0.0]]
    print(f"[ddg-traj] start distance={cur_d:.4f}  ddG=0.0")
    for step in range(1, args.steps + 1):
        if args.generator == "signature":
            cand_cds = signature_proposal(cur_cds, profile, rng)
        else:
            cand_cds = random_aa_proposal_cds(cur_cds, rng, n_mut=args.n_mut)
        cand_aa = str(translate_cds(cand_cds))
        if len(cand_aa) < 10:
            rows.append([step, cur_d, ddg(cur_aa)]); continue
        cand_d = float(scorer.distances([cand_aa])[0])
        delta = cand_d - cur_d
        if delta < 0 or rng.random() < math.exp(-delta / max(args.temperature, 1e-9)):
            cur_cds, cur_aa, cur_d = cand_cds, cand_aa, cand_d
        rows.append([step, cur_d, ddg(cur_aa)])
        if step % max(1, args.steps // 10) == 0:
            print(f"[ddg-traj] step {step:4d}  distance={cur_d:.4f}  ddG={rows[-1][2]:+.1f}")

    import csv
    with open(os.path.join(args.output_dir, "ddg_trajectory.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["step", "cosine_distance", "esm_ddg_vs_wt"]); w.writerows(rows)
    _plot(rows, args.target_pfam, os.path.join(args.output_dir, "proto_ddg_trajectory.png"))
    print(f"[ddg-traj] final distance={rows[-1][1]:.4f}  final ddG={rows[-1][2]:+.1f}  "
          f"-> {args.output_dir}/")


def _plot(rows, target, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    r = np.array(rows, dtype=float)
    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(r[:, 0], r[:, 1], color="#c0392b", lw=2, label="distance to target")
    ax1.set_xlabel("MCMC step"); ax1.set_ylabel(f"cosine distance to {target}", color="#c0392b")
    ax1.tick_params(axis="y", labelcolor="#c0392b")
    ax2 = ax1.twinx()
    ax2.plot(r[:, 0], r[:, 2], color="#2980b9", lw=2, label="ESM-2 ddG vs WT")
    ax2.set_ylabel("ESM-2 $\\Delta\\Delta$G vs WT (more negative = less stable)", color="#2980b9")
    ax2.tick_params(axis="y", labelcolor="#2980b9")
    ax1.set_title("Design approaches the target at a predicted-stability cost")
    for s in ("top",):
        ax1.spines[s].set_visible(False); ax2.spines[s].set_visible(False)
    fig.tight_layout(); fig.savefig(path, dpi=200)
    print(f"[ddg-traj] wrote {path}")


if __name__ == "__main__":
    main()
