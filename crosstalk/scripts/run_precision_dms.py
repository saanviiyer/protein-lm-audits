#!/usr/bin/env python3
"""Does serving precision change skill where the model actually has skill?

The ParD3 arm of the precision audit found quantisation almost harmless, but the
proxy there is near its floor already (coverage 0.077 at fp32), so there was
little left to lose. This repeats the audit on the DMS assays, where ESM-2 has
genuine skill and a loss would be visible.

Per-assay Spearman is recomputed at each precision on identical variants, and the
comparison is paired within assay, since assays differ far more from each other
than any precision differs from fp32.
"""
import argparse, csv, sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
from run_proxy_ladder import spearman
from run_reliability_forecast_full import load_assays, AA


def build(model_name, precision):
    from transformers import AutoModelForMaskedLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_name)
    mdl = AutoModelForMaskedLM.from_pretrained(model_name)
    if precision == "int8-dynamic":
        from torch.ao.quantization import quantize_dynamic
        if "qnnpack" in torch.backends.quantized.supported_engines:
            torch.backends.quantized.engine = "qnnpack"
        return tok, quantize_dynamic(mdl.eval(), {torch.nn.Linear}, dtype=torch.qint8), "cpu"
    dtype = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[precision]
    dev = "mps" if (torch.backends.mps.is_available() and precision != "bf16") else "cpu"
    return tok, mdl.to(dtype).to(dev).eval(), dev


@torch.no_grad()
def assay_scores(tok, mdl, dev, tgt, muts, batch_tokens=6000):
    positions = sorted({m[0] for m in muts})
    B = max(1, min(24, batch_tokens // max(len(tgt), 1)))
    lp = {}
    for i in range(0, len(positions), B):
        chunk = positions[i:i + B]
        seqs = [tgt[:p - 1] + tok.mask_token + tgt[p:] for p in chunk]
        enc = tok(seqs, return_tensors="pt", padding=True).to(dev)
        logits = mdl(**enc).logits
        for r, p in enumerate(chunk):
            lp[p] = torch.log_softmax(logits[r, p].float(), -1).cpu().numpy()
    s = np.array([lp[p][tok.convert_tokens_to_ids(mu)] - lp[p][tok.convert_tokens_to_ids(wt)]
                  for p, wt, mu, _ in muts])
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="facebook/esm2_t30_150M_UR50D")
    ap.add_argument("--precisions", nargs="+",
                    default=["fp32", "fp16", "int8-dynamic"])
    ap.add_argument("--n-assays", type=int, default=25)
    ap.add_argument("--max-len", type=int, default=400)
    ap.add_argument("--out", default="results/precision_dms.csv")
    args = ap.parse_args()

    assays = load_assays(200, args.max_len)
    rng = np.random.default_rng(0)
    pick = rng.choice(len(assays), size=min(args.n_assays, len(assays)), replace=False)
    assays = [assays[i] for i in sorted(pick)]
    print(f"{len(assays)} assays, {args.model}\n", flush=True)

    rho = {p: [] for p in args.precisions}
    names = []
    for prec in args.precisions:
        tok, mdl, dev = build(args.model, prec)
        for dms, tgt, muts in assays:
            s = assay_scores(tok, mdl, dev, tgt, muts)
            y = np.array([m[3] for m in muts])
            rho[prec].append(spearman(s, y))
            if prec == args.precisions[0]:
                names.append(dms)
        print(f"{prec:14s} mean rho {np.mean(rho[prec]):+.4f}", flush=True)
        del mdl

    ref = args.precisions[0]
    print(f"\n--- paired within assay, against {ref} ---")
    print(f"{'precision':14s} {'mean d':>9s} {'95% CI':>20s} {'max |d|':>9s} "
          f"{'assays worse':>13s}")
    rows = []
    for prec in args.precisions[1:]:
        d = np.array(rho[prec]) - np.array(rho[ref])
        ci = 1.96 * d.std(ddof=1) / np.sqrt(len(d))
        print(f"{prec:14s} {d.mean():+9.4f} [{d.mean()-ci:+.4f}, {d.mean()+ci:+.4f}]"
              f" {np.abs(d).max():9.4f} {int((d < 0).sum()):8d}/{len(d)}")
        rows.append(dict(precision=prec, mean_rho=float(np.mean(rho[prec])),
                         mean_delta=float(d.mean()), ci=float(ci),
                         max_abs_delta=float(np.abs(d).max()),
                         n_worse=int((d < 0).sum()), n_assays=len(d)))
    rows.append(dict(precision=ref, mean_rho=float(np.mean(rho[ref])),
                     n_assays=len(rho[ref])))

    out = ROOT / args.out
    keys = sorted({k for r in rows for k in r})
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)
    per = ROOT / "results/precision_dms_per_assay.csv"
    with per.open("w", newline="") as f:
        w = csv.writer(f); w.writerow(["dms_id"] + args.precisions)
        for i, n in enumerate(names):
            w.writerow([n] + [f"{rho[p][i]:.4f}" for p in args.precisions])
    print(f"\nwrote {out} and {per}")


if __name__ == "__main__":
    main()
