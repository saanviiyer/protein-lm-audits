#!/usr/bin/env python3
"""Coarse competence, fine incompetence: reconciling a null with generative success.

Merchant et al. (Nature 2025) design functional de novo genes with a genomic
language model, and King et al. (Science 2026) design whole bacteriophages the
same way. Sections 14 to 20 of this file find that genomic-LM likelihood carries
no protein-fitness signal at all, mean Spearman -0.013 across 25 assays. Placed
side by side these look contradictory.

They are not, if the models are competent at a coarser grain than fitness
scoring. Generating a plausible gene requires knowing what genes look like.
Ranking two point mutants requires knowing which protein works better. This
measures both on the same sequences and the same models, and separates the
ingredients with a factorial of corruptions:

  synonymous recode   protein preserved exactly, codons changed
  codon-order shuffle codon usage preserved exactly, protein destroyed
  frameshift +1       nucleotide composition preserved, reading frame destroyed
  reverse complement  composition preserved, not a coding sequence at all

A model that represents the protein should rate the synonymous recode near the
real gene and the other three far below it. A model reading nucleotide statistics
should be largely indifferent to the reading frame, which is what section 17
already found for point mutants.

Scores are mean per-token log-likelihood so sequences of different length are
comparable, and every comparison is paired within gene.
"""
import argparse, csv, json, sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
from crosstalk import glm

COMP = str.maketrans("ACGT", "TGCA")


def synonymous_recode(cds, rng):
    """Same protein, every codon swapped for a synonymous alternative where one exists."""
    out = []
    for i in range(0, len(cds) - 2, 3):
        c = cds[i:i + 3]
        alts = [x for x in glm.SYNONYMOUS.get(glm.CODON_TABLE.get(c), [c]) if x != c]
        out.append(alts[rng.integers(len(alts))] if alts else c)
    return "".join(out)


def codon_shuffle(cds, rng):
    """Identical codon usage, protein destroyed. Isolates codon statistics."""
    cods = [cds[i:i + 3] for i in range(0, len(cds) - 2, 3)]
    order = rng.permutation(len(cods))
    return "".join(cods[i] for i in order)


def frameshift(cds, k=1):
    return cds[k:] + cds[:k]


def revcomp(cds):
    return cds.translate(COMP)[::-1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="InstaDeepAI/nucleotide-transformer-v2-50m-multi-species")
    ap.add_argument("--max-nt", type=int, default=2400)
    ap.add_argument("--out", default="results/granularity_ladder.csv")
    args = ap.parse_args()

    cds = {}
    for k, v in json.loads((ROOT / "data/cds/dms_cds.json").read_text()).items():
        seq = v["cds"]
        if len(seq) % 3 == 0 and 200 <= len(seq) <= args.max_nt and set(seq) <= set("ACGT"):
            cds[k] = seq
    p3 = glm.load_cds()["ParD3"]["cds"]
    cds["ParD3_Lite_2020"] = p3
    print(f"{len(cds)} real coding sequences, "
          f"{min(map(len, cds.values()))}-{max(map(len, cds.values()))} nt\n", flush=True)

    sc = glm.NTScorer(args.model)
    rng = np.random.default_rng(0)
    conditions = {
        "real": lambda s: s,
        "synonymous recode": lambda s: synonymous_recode(s, rng),
        "codon-order shuffle": lambda s: codon_shuffle(s, rng),
        "frameshift +1": lambda s: frameshift(s, 1),
        "reverse complement": lambda s: revcomp(s),
    }

    scores = {c: {} for c in conditions}
    for n, (name, seq) in enumerate(sorted(cds.items()), 1):
        for cond, fn in conditions.items():
            s = fn(seq)
            ll = float(sc.pseudo_likelihood([s])[0])
            scores[cond][name] = ll / max(len(sc.tok(s)["input_ids"]) - 1, 1)
        print(f"[{n:2d}/{len(cds)}] {name[:40]:40s} "
              + "  ".join(f"{c[:4]} {scores[c][name]:+.3f}" for c in conditions),
              flush=True)

    names = sorted(cds)
    real = np.array([scores["real"][k] for k in names])
    print(f"\n--- paired against the real gene, {len(names)} genes ---")
    print(f"{'condition':22s} {'mean delta':>11s} {'95% CI':>20s} "
          f"{'real higher':>12s}")
    rows = [dict(condition="real", mean_score=float(real.mean()))]
    for cond in list(conditions)[1:]:
        x = np.array([scores[cond][k] for k in names])
        d = real - x
        ci = 1.96 * d.std(ddof=1) / np.sqrt(len(d))
        print(f"{cond:22s} {d.mean():+11.4f} [{d.mean()-ci:+.4f}, {d.mean()+ci:+.4f}]"
              f" {int((d > 0).sum()):8d}/{len(d)}")
        rows.append(dict(condition=cond, mean_score=float(x.mean()),
                         mean_delta_vs_real=float(d.mean()), ci=float(ci),
                         real_higher=int((d > 0).sum()), n=len(d)))

    print("\n--- what the pattern implies ---")
    dsyn = real - np.array([scores["synonymous recode"][k] for k in names])
    dfs = real - np.array([scores["frameshift +1"][k] for k in names])
    print(f"  cost of destroying the protein but keeping the frame "
          f"(codon shuffle): {(real - np.array([scores['codon-order shuffle'][k] for k in names])).mean():+.4f}")
    print(f"  cost of destroying the reading frame (frameshift): {dfs.mean():+.4f}")
    print(f"  cost of keeping the protein and changing codons "
          f"(synonymous):  {dsyn.mean():+.4f}")
    print(f"  ratio synonymous / frameshift: "
          f"{dsyn.mean()/dfs.mean():.2f}" if dfs.mean() != 0 else "")

    out = ROOT / args.out
    keys = sorted({k for r in rows for k in r})
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)
    per = ROOT / "results/granularity_per_gene.csv"
    with per.open("w", newline="") as f:
        w = csv.writer(f); w.writerow(["gene"] + list(conditions))
        for k in names:
            w.writerow([k] + [f"{scores[c][k]:.5f}" for c in conditions])
    print(f"\nwrote {out} and {per}")


if __name__ == "__main__":
    main()
