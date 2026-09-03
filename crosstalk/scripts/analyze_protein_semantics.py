#!/usr/bin/env python3
"""Side-by-side of the DNA reading-frame battery (section 26) and its protein analogue.

NT-v2 tokenises 6-mers and ESM-2 tokenises single residues, so nats per token are
not comparable across the two. Every delta is therefore also expressed as a
fraction of that model's OWN dynamic range, where the range is the model's
composition-preserving total-destruction anchor: codon-order shuffle for NT,
residue shuffle for ESM. That ratio asks the same question of both models --
"how much of the distance to a scrambled sequence does this corruption travel?"
"""
import csv, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]

PAIRING = [
    ("read backwards",      "reverse complement", "reverse",
     "composition preserved, reading direction destroyed"),
    ("rotate the string",   "frameshift +1",      "rotate +1",
     "composition and all local k-mers preserved, chain ends moved"),
    ("total destruction",   "codon-order shuffle", "shuffle",
     "composition preserved exactly, everything else destroyed"),
]


def main():
    dna = {r["condition"]: r for r in csv.DictReader(
        (ROOT / "results/granularity_ladder.csv").open())}
    prot = {}
    for r in csv.DictReader((ROOT / "results/protein_semantics.csv").open()):
        if r["metric"] == "ESM-2 650M PLL":
            prot[r["condition"]] = r

    dna_anchor = float(dna["codon-order shuffle"]["mean_delta_vs_real"])
    prot_anchor = float(prot["shuffle"]["mean_delta"])
    print(f"dynamic-range anchor   NT-v2 50M {dna_anchor:+.3f} nats/6mer-token   "
          f"ESM-2 650M {prot_anchor:+.3f} nats/residue\n")

    print(f"{'operation':20s} {'NT-v2 (DNA)':>26s} {'ESM-2 (protein)':>26s}")
    print(f"{'':20s} {'delta':>10s}{'frac':>8s}{'higher':>8s} "
          f"{'delta':>10s}{'frac':>8s}{'higher':>8s}")
    for label, dc, pc, _ in PAIRING:
        d = float(dna[dc]["mean_delta_vs_real"]); dh = f"{dna[dc]['real_higher']}/{dna[dc]['n']}"
        p = float(prot[pc]["mean_delta"]);        ph = f"{prot[pc]['real_higher']}/{prot[pc]['n']}"
        print(f"{label:20s} {d:+10.3f}{d/dna_anchor:8.2f}{dh:>8s} "
              f"{p:+10.3f}{p/prot_anchor:8.2f}{ph:>8s}")

    print("\nper-protein heterogeneity (ESM-2)")
    rows = list(csv.DictReader((ROOT / "results/protein_semantics_per_protein.csv").open()))
    for c in ["rotate +1", "rotate mid", "reverse", "shuffle"]:
        d = np.array([float(r["esm:real"]) - float(r[f"esm:{c}"]) for r in rows])
        L = np.array([int(r["len"]) for r in rows])
        rho = np.corrcoef(np.argsort(np.argsort(d)), np.argsort(np.argsort(L)))[0, 1]
        lo = sorted(zip(d, [r["protein"] for r in rows]))[:2]
        print(f"  {c:12s} min {d.min():+.3f} ({lo[0][1][:22]})  max {d.max():+.3f}  "
              f"spearman with length {rho:+.2f}")

    print("\ntrivial-baseline check (how much is visible from dipeptides + chain ends)")
    big = {r["condition"]: r for r in csv.DictReader(
        (ROOT / "results/protein_semantics.csv").open()) if r["metric"] == "bigram baseline"}
    for c in ["rotate +1", "rotate mid", "reverse", "shuffle",
              "conservative 10%", "radical 10%"]:
        e, b = float(prot[c]["mean_delta"]), float(big[c]["mean_delta"])
        print(f"  {c:18s} ESM {e:+7.3f}   bigram {b:+7.3f}   "
              f"ratio {e/b if abs(b)>1e-6 else float('nan'):6.1f}x")


if __name__ == "__main__":
    main()
