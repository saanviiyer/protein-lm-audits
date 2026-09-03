#!/usr/bin/env python3
"""How much of a protein does ESM-2 recall, and is that why it scores well?

Morris et al. (2025) separate what a language model memorises about specific
samples from what it generalises about the distribution. The protein analogue
matters here because section 22 found that a model's ability to reconstruct the
wild-type residue predicts its skill at ranking that protein's mutants. If that
ability is recall of a training sequence rather than knowledge of a family, then
skill on a novel sequence would not follow, which is exactly the regime a
rapid-response pipeline operates in.

Measurement: mask a contiguous span of length L, reconstruct by argmax at each
masked position in one pass, and record the fraction of spans recovered EXACTLY.
Exact recovery of a long span is hard to reach by generalisation alone, so the
decay of exact recall with L is informative about how sequence-specific the
knowledge is.

Two controls, because a high recall number alone means little:

  shuffled protein   the same residues in a random order. Real-protein-like
                     composition with no real context, so it bounds how much
                     recall comes from composition alone.
  homolog depth      ProteinGym's MSA Neff category. If recall is family
                     generalisation it should track homolog depth, and the part
                     of recall NOT explained by depth is the part that looks
                     sequence-specific.

The question that connects to section 22 is whether recall, and the depth-residual
of recall, predict per-assay DMS skill.
"""
import argparse, csv, sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
from crosstalk.glm import _device
from run_proxy_ladder import spearman
from analyze_reliability_full import clusters, NEFF

AA = "ACDEFGHIKLMNPQRSTVWY"


@torch.no_grad()
def span_recall(mdl, tok, dev, seq, lengths, n_spans, rng):
    """Exact-recovery rate for masked contiguous spans, per span length."""
    ids = [tok.convert_tokens_to_ids(a) for a in AA]
    out = {}
    for Lspan in lengths:
        if len(seq) <= Lspan + 2:
            out[Lspan] = np.nan
            continue
        starts = rng.choice(len(seq) - Lspan, size=min(n_spans, len(seq) - Lspan),
                            replace=False)
        exact, per_res = 0, []
        for st in starts:
            masked = seq[:st] + tok.mask_token * Lspan + seq[st + Lspan:]
            enc = tok(masked, return_tensors="pt").to(dev)
            logits = mdl(**enc).logits[0]
            pos = (enc["input_ids"][0] == tok.mask_token_id).nonzero().flatten()
            pred = "".join(AA[int(logits[p, ids].argmax())] for p in pos)
            true = seq[st:st + Lspan]
            n = min(len(pred), len(true))
            hits = sum(pred[i] == true[i] for i in range(n))
            per_res.append(hits / max(n, 1))
            exact += int(n == Lspan and hits == Lspan)
        out[Lspan] = exact / len(starts)
        out[f"res{Lspan}"] = float(np.mean(per_res))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="facebook/esm2_t33_650M_UR50D")
    ap.add_argument("--lengths", type=int, nargs="+", default=[1, 5, 10, 20, 40])
    ap.add_argument("--n-spans", type=int, default=12)
    ap.add_argument("--n-proteins", type=int, default=60)
    ap.add_argument("--max-len", type=int, default=500)
    ap.add_argument("--skill", default="results/reliability_forecast_full.csv")
    ap.add_argument("--out", default="results/memorization.csv")
    args = ap.parse_args()

    skill = {r["dms_id"]: float(r["rho_esm"]) for r in
             csv.DictReader((ROOT / args.skill).open())}
    ref = {r["DMS_id"]: r for r in
           csv.DictReader((ROOT / "data/proteingym_reference_v1.csv").open())}
    ids = [i for i in skill if i in ref and len(ref[i]["target_seq"]) <= args.max_len]
    rng0 = np.random.default_rng(0)
    ids = sorted(rng0.choice(ids, size=min(args.n_proteins, len(ids)), replace=False))
    print(f"{len(ids)} proteins, spans {args.lengths}, {args.n_spans} spans each\n",
          flush=True)

    from transformers import AutoModelForMaskedLM, AutoTokenizer
    dev = _device()
    tok = AutoTokenizer.from_pretrained(args.model)
    mdl = AutoModelForMaskedLM.from_pretrained(args.model).to(dev).eval()

    rows = []
    hdr = "  ".join(f"L{L}" for L in args.lengths)
    print(f"{'protein':34s} {hdr}   shuffled")
    for n, i in enumerate(ids, 1):
        seq = ref[i]["target_seq"]
        rng = np.random.default_rng(hash(i) % 10_000)
        real = span_recall(mdl, tok, dev, seq, args.lengths, args.n_spans, rng)
        sh = "".join(rng.permutation(list(seq)))
        shuf = span_recall(mdl, tok, dev, sh, args.lengths, args.n_spans, rng)
        row = dict(dms_id=i, skill=skill[i], length=len(seq),
                   neff=NEFF.get(ref[i].get("MSA_Neff_L_category", ""), np.nan),
                   taxon=ref[i]["taxon"])
        for L in args.lengths:
            row[f"exact_{L}"] = real.get(L, np.nan)
            row[f"res_{L}"] = real.get(f"res{L}", np.nan)
            row[f"shuf_exact_{L}"] = shuf.get(L, np.nan)
            row[f"shuf_res_{L}"] = shuf.get(f"res{L}", np.nan)
        rows.append(row)
        print(f"[{n:2d}] {i[:30]:30s} " +
              "  ".join(f"{real.get(L, np.nan):.2f}" for L in args.lengths) +
              f"   {shuf.get(args.lengths[1], np.nan):.2f}", flush=True)

    out = ROOT / args.out
    keys = sorted({k for r in rows for k in r})
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)

    Y = np.array([r["skill"] for r in rows])
    print("\n--- exact-span recall, real protein vs composition-matched shuffle ---")
    for L in args.lengths:
        a = np.array([r[f"exact_{L}"] for r in rows], float)
        b = np.array([r[f"shuf_exact_{L}"] for r in rows], float)
        ra = np.array([r[f"res_{L}"] for r in rows], float)
        rb = np.array([r[f"shuf_res_{L}"] for r in rows], float)
        print(f"  span {L:2d}: exact real {np.nanmean(a):.3f} vs shuffled "
              f"{np.nanmean(b):.3f}   per-residue real {np.nanmean(ra):.3f} vs "
              f"{np.nanmean(rb):.3f}")

    print("\n--- does recall track homolog depth (generalisation) ? ---")
    neff = np.array([r["neff"] for r in rows], float)
    for L in args.lengths:
        a = np.array([r[f"res_{L}"] for r in rows], float)
        ok = np.isfinite(a) & np.isfinite(neff)
        print(f"  span {L:2d}: rho(per-residue recall, Neff) = {spearman(a[ok], neff[ok]):+.3f}")

    print("\n--- does recall predict DMS skill, and does it survive removing depth? ---")
    g = clusters([r["dms_id"] for r in rows],
                 {r["dms_id"]: ref[r["dms_id"]]["target_seq"] for r in rows})
    print(f"  ({len(set(g.values()))} independent protein clusters)")
    for L in args.lengths:
        a = np.array([r[f"res_{L}"] for r in rows], float)
        ok = np.isfinite(a) & np.isfinite(neff) & np.isfinite(Y)
        raw = spearman(a[ok], Y[ok])
        # residualise recall on homolog depth, then re-correlate
        A = np.column_stack([neff[ok], np.ones(ok.sum())])
        beta, *_ = np.linalg.lstsq(A, a[ok], rcond=None)
        resid = a[ok] - A @ beta
        print(f"  span {L:2d}: rho(recall, skill) {raw:+.3f}   "
              f"depth-residualised {spearman(resid, Y[ok]):+.3f}")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
