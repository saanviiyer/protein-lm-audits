#!/usr/bin/env python3
"""A proxy validated at fp32 and served quantized has not been validated.

Latency pressure means deployed models are rarely the ones that were evaluated.
Weights get cast to fp16 or bf16, or dynamically quantized to int8. The usual
reassurance is that rank correlation with the original stays near 1.0, which is
true and is the wrong quantity: a pipeline does not act on the ranking, it acts on
a decision taken at a threshold, and a threshold sits wherever the scores are
densest.

This scores the same landscape at each precision and asks three questions in
increasing order of what a deployment actually cares about. How well do the raw
scores agree? Does per-instance accuracy move? Does the decision move, both in
the count of variants that change side of the threshold and in the coverage
quantities from section 23?

The whole landscape costs three forward passes per precision because every
variant differs only at three positions, so the audit is close to free and there
is no throughput reason to skip it.
"""
import argparse, csv, sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))

from crosstalk.boltz import MUT_POSITIONS, PARD3
from crosstalk.landscape import load_pard3
from crosstalk import coverage as cov
from run_proxy_ladder import auc, spearman


def score_at(precision, model_name, variants, device):
    """Masked-marginal scores for the whole landscape at one numeric precision."""
    from transformers import AutoModelForMaskedLM, AutoTokenizer
    from crosstalk.plm import variant_full
    tok = AutoTokenizer.from_pretrained(model_name)
    mdl = AutoModelForMaskedLM.from_pretrained(model_name)

    if precision == "int8-dynamic":
        # dynamic int8 quantises the linear layers only, and runs on CPU. On Apple
        # silicon the backend must be named explicitly; without it the conversion
        # fails with "NoQEngine", which reads like a missing feature rather than an
        # unset flag.
        from torch.ao.quantization import quantize_dynamic
        if "qnnpack" in torch.backends.quantized.supported_engines:
            torch.backends.quantized.engine = "qnnpack"
        mdl = quantize_dynamic(mdl.eval(), {torch.nn.Linear}, dtype=torch.qint8)
        dev = "cpu"
    else:
        dtype = {"fp32": torch.float32, "fp16": torch.float16,
                 "bf16": torch.bfloat16}[precision]
        dev = device if precision != "bf16" else "cpu"   # bf16 on MPS is unreliable
        mdl = mdl.to(dtype).to(dev).eval()

    seq = PARD3
    lp = []
    with torch.no_grad():
        for p in MUT_POSITIONS:
            masked = seq[:p - 1] + tok.mask_token + seq[p:]
            enc = tok(masked, return_tensors="pt").to(dev)
            logits = mdl(**enc).logits[0]
            at = (enc["input_ids"][0] == tok.mask_token_id).nonzero()[0, 0]
            lp.append(torch.log_softmax(logits[at].float(), -1).cpu().numpy())
    lp = np.stack(lp)
    wt = [PARD3[p - 1] for p in MUT_POSITIONS]
    tid = {a: tok.convert_tokens_to_ids(a) for a in set("".join(variants) + "".join(wt))}
    out = np.zeros(len(variants))
    for i, v in enumerate(variants):
        out[i] = sum(lp[k, tid[a]] - lp[k, tid[wt[k]]] for k, a in enumerate(v))
    del mdl
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="facebook/esm2_t33_650M_UR50D")
    ap.add_argument("--precisions", nargs="+",
                    default=["fp32", "fp16", "bf16", "int8-dynamic"])
    ap.add_argument("--fpr", type=float, default=0.01)
    ap.add_argument("--out", default="results/precision_audit.csv")
    args = ap.parse_args()

    L = load_pard3()
    variants = L.seqs
    w3 = L.F[:, 0]
    functional = w3 >= 0.8
    negative = w3 < 0.5
    nb = cov.build_neighbours(L)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"{len(variants)} variants, {int(functional.sum())} viable; "
          f"budget FPR={args.fpr:.0%}\n", flush=True)

    scores, rows = {}, []
    for prec in args.precisions:
        s = score_at(prec, args.model, variants, device)
        scores[prec] = s
        a = auc(s[functional | negative], functional[functional | negative])
        c = cov.coverage(s, functional, negative, args.fpr)
        topk = cov.top_k_coverage(s, w3, functional, c["threshold"])
        loc = cov.local_search_escape_rate(w3, c["escape"], nb, 2000)
        rows.append(dict(precision=prec, auc=a, coverage=c["coverage"],
                         n_escape=c["n_escape"], threshold=c["threshold"],
                         local_escape_rate=loc["escape_rate"], **topk))
        print(f"{prec:14s} AUC {a:.4f}  coverage {c['coverage']:.3f}  "
              f"missed {c['n_escape']:3d}  top10 {topk['coverage_top10']:.2f}  "
              f"hill-climb miss {loc['escape_rate']:.3f}", flush=True)

    ref = args.precisions[0]
    print(f"\n--- agreement with {ref} ---")
    print(f"{'precision':14s} {'pearson':>9s} {'spearman':>9s} {'max|d|':>8s} "
          f"{'flips':>7s} {'flip%viable':>12s}")
    thr_ref = [r for r in rows if r["precision"] == ref][0]["threshold"]
    for prec in args.precisions[1:]:
        a, b = scores[ref], scores[prec]
        pear = float(np.corrcoef(a, b)[0, 1])
        sp = spearman(a, b)
        # decisions are taken at each build's own operating point, which is how a
        # deployment would set it: threshold calibrated on the served model
        thr_b = [r for r in rows if r["precision"] == prec][0]["threshold"]
        flips = int(((a >= thr_ref) != (b >= thr_b)).sum())
        fv = int((((a >= thr_ref) != (b >= thr_b)) & functional).sum())
        print(f"{prec:14s} {pear:9.5f} {sp:9.5f} {np.abs(a-b).max():8.3f} "
              f"{flips:7d} {fv:12d}")
        for r in rows:
            if r["precision"] == prec:
                r.update(pearson_vs_ref=pear, spearman_vs_ref=sp,
                         max_abs_diff=float(np.abs(a - b).max()),
                         decision_flips=flips, viable_flips=fv)

    out = ROOT / args.out
    keys = sorted({k for r in rows for k in r})
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
