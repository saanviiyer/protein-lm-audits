#!/usr/bin/env python3
"""Can this measurement see verbatim reproduction at all?

The memorisation sweep recovers no span longer than one token, and the one
30-nt span it did recover turned out to occur TWICE inside the 3000-nt window
the model was reading -- an in-context copy, not recall. A null that rests on
"exact recovery is zero" is worthless unless exact recovery is reachable, so
this builds a window where the answer is verbatim present in the context and
asks whether the same argmax decoder finds it.

Construction: take a real 3000-nt genomic window, choose a token-aligned donor
block of `--dup` nucleotides, and paste it over a second token-aligned block
further along the window. The window now contains that block twice, with
identical flanks. Mask a span inside the SECOND copy. Exact recovery now
requires only copying, which is the easiest possible form of the task the
memorisation sweep sets. Two controls share the arithmetic:

  copy      the duplicated window, masking inside the second copy
  nocopy    the same window and the same mask, donor block left untouched
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import memdna_data as D
from crosstalk.glm import _device, _nt_config, _patch_transformers_for_nt_v2
from run_memorisation_dna import FIELDS, run_window


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model",
                    default="InstaDeepAI/nucleotide-transformer-v2-50m-multi-species")
    ap.add_argument("--window", type=int, default=3000)
    ap.add_argument("--dup", type=int, default=600, help="duplicated block, nt")
    ap.add_argument("--src-tok", type=int, default=20, help="donor block, token idx")
    ap.add_argument("--dst-tok", type=int, default=300, help="paste block, token idx")
    ap.add_argument("--lengths", type=int, nargs="+", default=[1, 2, 5, 10, 20])
    ap.add_argument("--n-spans", type=int, default=6)
    ap.add_argument("--n-windows", type=int, default=40)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--device", default="")
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--out", default="results/memorisation_dna_poscontrol.csv")
    args = ap.parse_args()

    W, k = args.window, args.dup // 6
    rng = np.random.default_rng(args.seed)
    gen = ROOT / "data" / "genomes"
    src = D.load_fasta(gen / D.MEMBERS["M.albiziae"])
    src.update(D.load_opportunistum())
    base = D.sample_windows(src, args.n_windows, W, rng, frame=0)
    assert args.dst_tok + k <= W // 6 - 10 and args.src_tok + k <= args.dst_tok

    recs = []
    for i, b in enumerate(base):
        s = b["seq"]
        donor = s[args.src_tok * 6:(args.src_tok + k) * 6]
        dup = (s[:args.dst_tok * 6] + donor + s[(args.dst_tok + k) * 6:])
        assert len(dup) == W and dup.count(donor) >= 2
        recs.append(dict(wid=2 * i, arm="copy", genome=b["contig"], contig=b["contig"],
                         start=b["start"], gc=b["gc"], seq=dup))
        recs.append(dict(wid=2 * i + 1, arm="nocopy", genome=b["contig"],
                         contig=b["contig"], start=b["start"], gc=b["gc"], seq=s))

    _patch_transformers_for_nt_v2()
    from transformers import AutoModelForMaskedLM, AutoTokenizer
    dev = args.device or _device()
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    mdl = AutoModelForMaskedLM.from_pretrained(
        args.model, config=_nt_config(args.model), trust_remote_code=True).to(dev).eval()
    vocab = tok.get_vocab()
    six = sorted(t for t in vocab if len(t) == 6 and not set(t) - set("ACGT"))
    six_ids = [vocab[t] for t in six]
    six_tok = {vocab[t]: t for t in six}
    print(f"{args.model} on {dev}: {len(recs)} windows, donor {args.dup} nt "
          f"at token {args.src_tok}, copy at token {args.dst_tok}\n", flush=True)

    out = (ROOT / args.out).open("w", newline="")
    w = csv.DictWriter(out, fieldnames=FIELDS, extrasaction="ignore")
    w.writeheader()
    t0 = time.time()
    for n, rec in enumerate(recs, 1):
        r2 = np.random.default_rng(args.seed * 7919 + rec["wid"] // 2)
        jobs = []
        for L in args.lengths:
            hi = args.dst_tok + k - L - 2
            lo = args.dst_tok + 2
            if hi <= lo:
                continue
            jobs += [(L, int(s), "incopy") for s in r2.integers(lo, hi, size=args.n_spans)]
        rows = run_window(mdl, tok, dev, six_ids, six_tok, rec, jobs, args.batch)
        w.writerows(rows)
        out.flush()
        if n % 10 == 0:
            print(f"[{n}/{len(recs)}] {(time.time()-t0)/60:.1f} min", flush=True)
    out.close()

    rows = list(csv.DictReader((ROOT / args.out).open()))
    print("\n--- can the decoder reproduce a span that is verbatim in context?")
    print(f"{'arm':8s} {'L':>3s} {'exact':>8s} {'tok_acc':>8s} {'nt_acc':>8s} "
          f"{'max_run':>8s} {'n':>5s}")
    for arm in ("copy", "nocopy"):
        for L in args.lengths:
            x = [r for r in rows if r["arm"] == arm and int(r["L"]) == L]
            if not x:
                continue
            print(f"{arm:8s} {L:3d} "
                  f"{np.mean([int(r['exact']) for r in x]):8.4f} "
                  f"{np.mean([float(r['tok_acc']) for r in x]):8.4f} "
                  f"{np.mean([float(r['nt_acc']) for r in x]):8.4f} "
                  f"{np.mean([int(r['max_run_nt']) for r in x]):8.2f} {len(x):5d}")


if __name__ == "__main__":
    main()
