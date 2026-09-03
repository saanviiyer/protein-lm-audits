"""Does 'scale makes it worse' hold on the second system?

On ParD3 the partner-blind AUC fell monotonically with model size: 0.296 (8M)
to 0.151 (650M). If BPTI reproduces that, the scaling claim is a two-system
result rather than a property of one landscape.
"""
import argparse, csv, sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from crosstalk.bpti import BPTI, MUT_POSITIONS, load_bpti, variant_positions
from run_bpti_test import plm_scores
from run_proxy_ladder import auc, boot_ci, spearman

LADDER = ["facebook/esm2_t6_8M_UR50D", "facebook/esm2_t12_35M_UR50D",
          "facebook/esm2_t30_150M_UR50D", "facebook/esm2_t33_650M_UR50D"]
PARAMS = dict(zip(LADDER, [8, 35, 150, 650]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=LADDER)
    ap.add_argument("--on-pct", type=float, default=90)
    ap.add_argument("--off-pct", type=float, default=75)
    ap.add_argument("--out", default="results/bpti_ladder.csv")
    args = ap.parse_args()

    L = load_bpti()
    on = L.F[:, 0]
    off = L.F[:, 1:].max(axis=1)
    on_min = float(np.percentile(on, args.on_pct))
    tau = float(np.percentile(L.F[:, 1:], args.off_pct))
    spec = (on >= on_min) & (off <= tau)
    prom = (on >= on_min) & (off > tau)
    mask = spec | prom
    lab = spec[mask]
    print(f"BPTI: {L.n_seqs} variants; decision set {int(mask.sum())} "
          f"({int(spec.sum())} specific vs {int(prom.sum())} promiscuous)\n")

    wt_at = {p: BPTI[p - 1] for p in MUT_POSITIONS}
    nmut = np.array([sum(1 for pos, aa in variant_positions(c) if wt_at.get(pos) != aa)
                     for c in L.codes], float)
    a = auc(-nmut[mask], lab); lo, hi = boot_ci(-nmut[mask], lab)
    rows = [dict(model="baseline:mutation_count", params_M=0, auc=a, auc_lo=lo, auc_hi=hi,
                 rho_cognate=spearman(-nmut, L.F[:, 0]),
                 rho_chymotrypsin=spearman(-nmut, L.F[:, 1]),
                 rho_mesotrypsin=spearman(-nmut, L.F[:, 2]))]
    print(f"  {'mutation count':22s} AUC {a:.3f} [{lo:.3f}, {hi:.3f}]")

    for m in args.models:
        s = plm_scores(L.codes, m)
        a = auc(s[mask], lab); lo, hi = boot_ci(s[mask], lab)
        rhos = [spearman(s, L.F[:, i]) for i in range(3)]
        rows.append(dict(model=m, params_M=PARAMS.get(m, np.nan), auc=a, auc_lo=lo, auc_hi=hi,
                         rho_cognate=rhos[0], rho_chymotrypsin=rhos[1], rho_mesotrypsin=rhos[2]))
        print(f"  {m.split('_')[1]:22s} AUC {a:.3f} [{lo:.3f}, {hi:.3f}]  "
              f"rho: trypsin {rhos[0]:+.3f} chymo {rhos[1]:+.3f} meso {rhos[2]:+.3f}",
              flush=True)

    print("\n  ParD3 for comparison: 0.296 (8M) -> 0.273 -> 0.199 -> 0.151 (650M)")
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
