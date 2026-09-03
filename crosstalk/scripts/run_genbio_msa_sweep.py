"""Does retrieval help as MSA depth grows?

The first RAG attempt used 20 rows taken from the top of the alignment, which
are the closest homologs and carry the least information, and capped context at
2048 tokens from a vestigial config field. GenBio's own tutorial uses 12.8K
context and a diversity-greedy subset. This redoes the test properly and sweeps
depth, so "retrieval does not help" is a claim about retrieval rather than about
a bad MSA.
"""
import argparse, csv, glob, sys, time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from crosstalk.genbio import GenBioScorer, load_msa
from crosstalk.landscape import load_pard3
from run_proxy_ladder import auc, boot_ci, spearman


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="genbio-ai/AIDO.Protein-RAG-3B")
    ap.add_argument("--depths", type=int, nargs="+", default=[0, 20, 64, 128])
    ap.add_argument("--out", default="results/genbio_msa_sweep.csv")
    args = ap.parse_args()

    L = load_pard3()
    w3, w2 = L.F[:, 0], L.F[:, 1]
    spec = (w3 >= 0.8) & (w2 <= 0.2)
    prom = (w3 >= 0.8) & (w2 >= 0.6)
    mask = spec | prom
    lab = spec[mask]

    msa_csv = sorted(glob.glob("results/boltz/*/boltz_results_*/msa/*_0.csv"))[0]
    print(f"MSA source: {Path(msa_csv).name} (chain 0 = ParD3)")
    print(f"discrimination set: {int(mask.sum())} variants\n")

    sc = GenBioScorer(args.repo, device="cpu")
    rows = []
    for depth in args.depths:
        msa = load_msa(msa_csv, max_rows=depth, diverse=True) if depth else []
        t = time.time()
        s = sc.score_variants(L.seqs, msa=msa or None)
        dt = time.time() - t
        a = auc(s[mask], lab); lo, hi = boot_ci(s[mask], lab)
        ntok = 93 * (len(msa) + 1)
        rows.append(dict(depth=len(msa), approx_tokens=ntok,
                         auc=a, auc_lo=lo, auc_hi=hi,
                         rho_on=spearman(s, w3), rho_off=spearman(s, w2),
                         rho_margin=spearman(s, w3 - w2), seconds=round(dt)))
        print(f"  MSA depth {len(msa):4d} (~{ntok:6d} tok)  AUC {a:.3f} [{lo:.3f}, {hi:.3f}]"
              f"  rho_on={spearman(s,w3):+.3f} rho_off={spearman(s,w2):+.3f}  ({dt:.0f}s)",
              flush=True)

    print("\n  reference: ESM-2 650M 0.151 | mutation-count baseline 0.664 | chance 0.5")
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
