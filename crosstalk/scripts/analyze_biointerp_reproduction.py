#!/usr/bin/env python3
"""Does the biointerp library reproduce FINDINGS section 26?

Section 26 was produced by scripts/run_granularity_ladder.py with a single
generator threaded through every (gene, condition) pair. biointerp keys its
generator on (seed, intervention, gene) instead, so the two DETERMINISTIC
interventions -- frameshift +1 and reverse complement -- must agree to the last
digit, while the two STOCHASTIC ones -- synonymous recode and codon-order
shuffle -- can only agree within Monte-Carlo noise over the choice of codons.

Splitting the comparison that way is the point: an exact match on the
deterministic pair proves the scoring path, the normalisation and the pairing are
identical, so any disagreement on the stochastic pair is about codon sampling and
nothing else.
"""
import csv, sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]

REFERENCE = {                    # FINDINGS section 26, NT-v2 50M, 29 genes
    "synonymous recode":   (+0.2334, 20, False),
    "codon-order shuffle": (+0.3218, 24, False),
    "frameshift +1":       (+0.0264, 14, True),
    "reverse complement":  (-0.0643, 11, True),
}


def load(path, prefix=""):
    rows = list(csv.DictReader(open(path)))
    return rows


def main(per_seq="results/biointerp_nt50m_per_sequence.csv"):
    rows = load(ROOT / per_seq)
    cols = [c for c in rows[0] if c.startswith("delta:")]
    old = {r["gene"]: r for r in load(ROOT / "results/granularity_per_gene.csv")}

    print(f"{len(rows)} genes; section-26 file has {len(old)}")
    shared = sorted({r["sequence"] for r in rows} & set(old))
    print(f"{len(shared)} genes in common\n")

    real_new = np.array([float(next(r for r in rows if r["sequence"] == g)["real_score"])
                         for g in shared])
    real_old = np.array([float(old[g]["real"]) for g in shared])
    print(f"real-sequence scores: max |new - old| = {np.abs(real_new - real_old).max():.2e}")
    print()

    print(f"{'intervention':22s} {'biointerp':>10s} {'section 26':>11s} {'diff':>9s} "
          f"{'real> new':>10s} {'ref':>5s}  status")
    print("-" * 84)
    for name, (ref_delta, ref_higher, deterministic) in REFERENCE.items():
        col = f"delta:{name}"
        if col not in cols:
            print(f"{name:22s} not in this battery")
            continue
        d = np.array([float(next(r for r in rows if r["sequence"] == g)[col]) for g in shared])
        m, k = float(d.mean()), int((d > 0).sum())
        diff = m - ref_delta
        if deterministic:
            ok = abs(diff) < 5e-4 and k == ref_higher
            status = "EXACT MATCH" if ok else "*** MISMATCH ***"
        else:
            # The right test is PAIRED on per-gene deltas against section 26's own
            # draw, not a between-seed spread: the two runs share the same 29
            # genes, so gene-to-gene variance -- which is an order of magnitude
            # larger than the Monte-Carlo term -- cancels.
            oldd = np.array([float(old[g]["real"]) - float(old[g][name]) for g in shared])
            pair = d - oldd
            se = pair.std(ddof=1) / np.sqrt(len(pair))
            z = pair.mean() / se
            ok = abs(z) < 1.96
            status = (f"consistent, paired z={z:+.2f}" if ok
                      else f"*** DISAGREES, paired z={z:+.2f} ***")
        print(f"{name:22s} {m:+10.4f} {ref_delta:+11.4f} {diff:+9.4f} "
              f"{k:6d}/{len(d):<3d} {ref_higher:5d}  {status}")

    print()
    fs = np.array([float(next(r for r in rows if r["sequence"] == g)["delta:frameshift +1"])
                   for g in shared]).mean()
    cs = np.array([float(next(r for r in rows if r["sequence"] == g)["delta:codon-order shuffle"])
                   for g in shared]).mean()
    print(f"codon-context / reading-frame sensitivity ratio: {cs / fs:.1f}x "
          f"(section 26 reported 12x)")


if __name__ == "__main__":
    main(*sys.argv[1:])
