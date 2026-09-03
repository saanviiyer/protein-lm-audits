"""Does the anti-correlation survive non-additive scoring?

The masked-marginal score is additive over the three mutated sites, so it cannot
represent epistasis between them -- and epistasis is the dominant error term on
this landscape. Full pseudo-likelihood masks every position in the variant's own
context and has no such restriction. If the proxy is still anti-predictive here,
additivity was not the explanation.
"""
import argparse, csv, sys, time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from crosstalk.boltz import MUT_POSITIONS, PARD3
from crosstalk.landscape import load_pard3
from crosstalk.plm import ESMScorer, pseudo_likelihood, variant_full
from run_proxy_ladder import auc, boot_ci, spearman


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="facebook/esm2_t33_650M_UR50D")
    ap.add_argument("--batch-size", type=int, default=24)
    ap.add_argument("--out", default="results/pseudolikelihood.csv")
    args = ap.parse_args()

    L = load_pard3()
    w3, w2 = L.F[:, 0], L.F[:, 1]
    specific = (w3 >= 0.8) & (w2 <= 0.2)
    promisc = (w3 >= 0.8) & (w2 >= 0.6)
    sel = np.where(specific | promisc)[0]
    variants = [L.seqs[i] for i in sel]
    lab = specific[sel]
    print(f"discrimination set: {len(sel)} variants "
          f"({int(lab.sum())} specific, {int((~lab).sum())} promiscuous)")
    print(f"{len(PARD3)} residues x {len(sel)} variants = "
          f"{len(PARD3)*len(sel)} masked forward passes\n", flush=True)

    sc = ESMScorer(args.model)
    t = time.time()
    seqs = [variant_full(v) for v in variants]
    pll = pseudo_likelihood(sc, seqs, batch_size=args.batch_size, progress=50)
    print(f"\ndone in {time.time()-t:.0f}s")

    a = auc(pll, lab); lo, hi = boot_ci(pll, lab)
    print(f"\nFULL PSEUDO-LIKELIHOOD, {args.model}")
    print(f"  AUC specific vs promiscuous = {a:.3f} [{lo:.3f}, {hi:.3f}]")
    print(f"  (masked-marginal on the same task was 0.151; chance 0.5;"
          f" mutation-count baseline 0.664)")
    print(f"  rho(PLL, on-target)  = {spearman(pll, w3[sel]):+.3f}")
    print(f"  rho(PLL, off-target) = {spearman(pll, w2[sel]):+.3f}")
    print(f"  rho(PLL, margin)     = {spearman(pll, (w3-w2)[sel]):+.3f}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["variant", "pll", "W_ParE3", "W_ParE2", "is_specific"])
        for v, p, i in zip(variants, pll, sel):
            w.writerow([v, p, w3[i], w2[i], int(specific[i])])
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
