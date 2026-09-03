"""GenBio AIDO.Protein rung, with and without MSA retrieval.

Same masked-marginal protocol as the ESM-2 rungs, so the comparison isolates the
model rather than the method. The RAG arm reuses the ColabFold MSAs that the
Boltz runs already produced, so it costs nothing extra to obtain.
"""
import argparse, csv, glob, sys, time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from crosstalk.genbio import GenBioScorer, load_msa
from crosstalk.landscape import load_pard3
from crosstalk.plm import complex_context
from run_proxy_ladder import auc, boot_ci, spearman


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="genbio-ai/AIDO.Protein-RAG-3B")
    ap.add_argument("--msa-rows", type=int, default=20)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="results/proxy_ladder_genbio.csv")
    args = ap.parse_args()

    L = load_pard3()
    w3, w2 = L.F[:, 0], L.F[:, 1]
    specific = (w3 >= 0.8) & (w2 <= 0.2)
    promisc = (w3 >= 0.8) & (w2 >= 0.6)
    mask = specific | promisc
    lab = specific[mask]
    print(f"discrimination set: {int(mask.sum())} "
          f"({int(specific.sum())} specific vs {int(promisc.sum())} promiscuous)\n")

    msa_csv = sorted(glob.glob("results/boltz/*/boltz_results_*/msa/*_0.csv"))
    msa = load_msa(msa_csv[0], max_rows=args.msa_rows) if msa_csv else []
    print(f"MSA: {len(msa)} homolog rows from {Path(msa_csv[0]).name if msa_csv else 'none'}"
          f" (chain 0 = ParD3)\n")

    print(f"loading {args.repo} ...", flush=True)
    t = time.time()
    sc = GenBioScorer(args.repo, device=args.device)
    print(f"loaded in {time.time()-t:.0f}s on {sc.device}\n", flush=True)

    arms = [("partner_blind", None, None),
            ("partner_aware_ParE3", complex_context("ParE3"), None),
            ("partner_aware_ParE2", complex_context("ParE2"), None)]
    if msa:
        arms.append(("partner_blind_RAG", None, msa))

    rows, scores = [], {}
    for mode, ctx, m in arms:
        t = time.time()
        s = sc.score_variants(L.seqs, context=ctx, msa=m)
        scores[mode] = s
        a = auc(s[mask], lab); lo, hi = boot_ci(s[mask], lab)
        rows.append(dict(model=args.repo, params_M=3000, mode=mode,
                         auc_specific_vs_promiscuous=a, auc_lo=lo, auc_hi=hi,
                         rho_on_target=spearman(s, w3), rho_off_target=spearman(s, w2),
                         rho_margin=spearman(s, w3 - w2), msa_rows=len(m) if m else 0))
        print(f"  {mode:22s} AUC {a:.3f} [{lo:.3f}, {hi:.3f}]  "
              f"rho_on={spearman(s,w3):+.3f} rho_off={spearman(s,w2):+.3f} "
              f"rho_margin={spearman(s,w3-w2):+.3f}   ({time.time()-t:.0f}s)", flush=True)

    if "partner_aware_ParE3" in scores and "partner_aware_ParE2" in scores:
        s = scores["partner_aware_ParE3"] - scores["partner_aware_ParE2"]
        a = auc(s[mask], lab); lo, hi = boot_ci(s[mask], lab)
        rows.append(dict(model=args.repo, params_M=3000, mode="partner_aware_margin",
                         auc_specific_vs_promiscuous=a, auc_lo=lo, auc_hi=hi,
                         rho_on_target=spearman(s, w3), rho_off_target=spearman(s, w2),
                         rho_margin=spearman(s, w3 - w2), msa_rows=0))
        print(f"  {'partner_aware_margin':22s} AUC {a:.3f} [{lo:.3f}, {hi:.3f}]")

    print("\n  reference: ESM-2 650M partner-blind AUC 0.151 (rho_off +0.540 > rho_on +0.510)")
    print("             trivial mutation-count baseline AUC 0.664")

    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
