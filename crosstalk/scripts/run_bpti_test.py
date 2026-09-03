"""Is the proxy partner-agnostic on a system with a hugely selective wild type?

Section 14 argues a single-sequence likelihood carries almost no information
about the *difference* between partners, so the sign of its specificity AUC is
decided by a marginal asymmetry in how it correlates with each one. That is a
claim about single-sequence scoring in general, not about ParD3.

BPTI is the adversarial case. Wild-type selectivity is 6-9 orders of magnitude
(K_D 1e-14 trypsin, 1e-8 chymotrypsin, 1e-5 mesotrypsin) where ParD3's is ~5x,
in a different fold with a different assay and three partners rather than two.

  If the proxy correlates similarly with all three -> partner-agnostic, mechanism
  generalises.
  If it tracks trypsin specifically -> mechanism is wrong, ParD3 was a special
  case, and that needs saying.
"""
import argparse, csv, sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from crosstalk.bpti import BPTI, MUT_POSITIONS, load_bpti, variant_positions
from run_proxy_ladder import auc, boot_ci, spearman


def plm_scores(variants, model_name):
    from transformers import AutoModelForMaskedLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_name)
    mdl = AutoModelForMaskedLM.from_pretrained(model_name).eval()

    lp = {}
    with torch.no_grad():
        for pos in MUT_POSITIONS:
            masked = BPTI[: pos - 1] + tok.mask_token + BPTI[pos:]
            enc = tok(masked, return_tensors="pt")
            logits = mdl(**enc).logits[0]
            at = (enc["input_ids"][0] == tok.mask_token_id).nonzero()[0, 0]
            lp[pos] = torch.log_softmax(logits[at].float(), -1).numpy()

    out = np.zeros(len(variants))
    for i, code in enumerate(variants):
        s = 0.0
        for pos, aa in variant_positions(code):
            if pos not in lp:
                continue
            s += (lp[pos][tok.convert_tokens_to_ids(aa)]
                  - lp[pos][tok.convert_tokens_to_ids(BPTI[pos - 1])])
        out[i] = s
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="facebook/esm2_t33_650M_UR50D")
    ap.add_argument("--on-pct", type=float, default=90)
    ap.add_argument("--off-pct", type=float, default=75)
    ap.add_argument("--out", default="results/bpti_test.csv")
    args = ap.parse_args()

    L = load_bpti()
    on = L.F[:, 0]                                  # trypsin, the cognate target
    off = L.F[:, 1:].max(axis=1)                    # worst of the two off-targets
    print(f"{L.n_seqs} variants x {L.n_partners} partners: {L.partners}\n")

    print(f"loading {args.model} ...", flush=True)
    # L.seqs are position strings for the environment; PLM scoring needs the
    # mutation codes, which the landscape keeps alongside them.
    plm = plm_scores(L.codes, args.model)

    print("\nTHE TEST: does the proxy prefer one partner, or treat them alike?")
    rhos = [spearman(plm, L.F[:, i]) for i in range(L.n_partners)]
    for p, r in zip(L.partners, rhos):
        print(f"   rho(PLM, {p:14s}) = {r:+.3f}")
    spread = max(rhos) - min(rhos)
    print(f"\n   spread across partners = {spread:.3f}")
    print(f"   (ParD3 was +0.510 / +0.540, a spread of 0.030 -- partner-agnostic)")

    on_min = float(np.percentile(on, args.on_pct))
    tau = float(np.percentile(L.F[:, 1:], args.off_pct))
    spec = (on >= on_min) & (off <= tau)
    prom = (on >= on_min) & (off > tau)
    mask = spec | prom
    print(f"\nDISCRIMINATION: binds trypsin (>= p{args.on_pct:.0f}), "
          f"avoids both others (<= p{args.off_pct:.0f})")
    print(f"   {int(spec.sum())} specific vs {int(prom.sum())} promiscuous (n={int(mask.sum())})")

    wt_at = {p: BPTI[p - 1] for p in MUT_POSITIONS}
    nmut = np.array([sum(1 for pos, aa in variant_positions(c) if wt_at.get(pos) != aa)
                     for c in L.codes], float)

    rows = []
    for name, s in (("PLM likelihood", plm), ("mutation count", -nmut)):
        a = auc(s[mask], spec[mask]); lo, hi = boot_ci(s[mask], spec[mask])
        rows.append(dict(system="bpti", model=args.model, score=name, auc=a,
                         auc_lo=lo, auc_hi=hi,
                         rho_trypsin=rhos[0], rho_chymotrypsin=rhos[1],
                         rho_mesotrypsin=rhos[2], rho_spread=spread,
                         n=int(mask.sum())))
        print(f"   AUC {name:16s} = {a:.3f} [{lo:.3f}, {hi:.3f}]")
    print("\n   ParD3 reference: PLM 0.151, mutation count 0.664, chance 0.5")

    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
