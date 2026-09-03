#!/usr/bin/env python3
"""The same memorisation question for an AUTOREGRESSIVE, single-nucleotide model.

NT-v2 is a masked LM over 6-mers, and its 6-mer head is so weak that exact
recovery has almost no dynamic range. HyenaDNA changes both axes at once: single
nucleotides (4-way decisions, not 4096-way) and next-token prediction, which
makes the canonical extraction attack -- feed a real prefix, decode greedily,
count how many nucleotides come back verbatim -- available for the first time
here. It also matters for section 30: Evo 2 is autoregressive, so a masked-LM
null need not carry over.

Membership here is coarser than the NT test and is stated as such. HyenaDNA was
pretrained on the human reference genome and nothing else, so human sequence is
training data (the published hg38 interval split is not in a repository this run
could reach, so this is assembly-level membership, not chunk-level) and bacterial
sequence was certainly never seen. Composition is the obvious confound -- human
is 41% GC and repeat-rich -- so the bacterial arm is drawn from genomes near 41%
GC, and the decisive within-human controls are a dinucleotide shuffle and an
order-5 chain fitted to the same human sequence.

Two decoders:
  teacher-forced   one pass; greedy argmax at every position given the TRUE prefix.
                   The direct analogue of the single-pass argmax used for NT and
                   in section 28.
  free-running     greedy generation from a true prefix, each nucleotide fed back.
                   This is the extraction attack proper, and the only decoder that
                   can show a model writing out training data unaided.
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
from crosstalk.glm import _device

BASES = "ACGT"
LOWGC = ["Tolypothrix", "Salibacter", "Agitococcus", "Abiotrophia"]   # GC 0.40-0.47


def build(W: int, n: int, seed: int):
    rng = np.random.default_rng(seed)
    gen = ROOT / "data" / "genomes"
    hum = D.load_fasta(gen / "H_chr21.fa.gz")
    hum = {k: v for k, v in hum.items()}
    recs = []
    hw = D.sample_windows(hum, n, W, rng, frame=None)
    for r in hw:
        recs.append(dict(arm="human", genome="chr21", **r))
    bac = []
    for name in LOWGC:
        bac += [dict(genome=name, **r) for r in
                D.sample_windows(D.load_fasta(gen / D.MEMBERS[name]),
                                 max(2, n // len(LOWGC)), W, rng, frame=None)]
    hb, bb = D.gc_match(hw, bac, rng, binw=0.02, cap=4)
    recs = [dict(arm="human", genome="chr21", **r) for r in hb]
    recs += [dict(arm="bacteria", **r) for r in bb]
    mk = D.Markov5(hum)
    for r in hb:
        d = D.dinuc_shuffle(r["seq"], rng)
        recs.append(dict(arm="dinuc_human", genome="chr21", contig=r["contig"],
                         start=r["start"], seq=d,
                         gc=(d.count("G") + d.count("C")) / len(d)))
        s = mk.emit(W, rng)
        recs.append(dict(arm="markov5_human", genome="chr21", contig="-", start=-1,
                         seq=s, gc=(s.count("G") + s.count("C")) / len(s)))
    # positive control: the answer is verbatim in context, 1700 nt back
    for r in hb[:max(4, len(hb) // 3)]:
        s = r["seq"]
        dup = s[:1800] + s[120:720] + s[2400:]
        recs.append(dict(arm="copy_control", genome="chr21", contig=r["contig"],
                         start=r["start"], seq=dup[:W], gc=r["gc"]))
    for i, r in enumerate(recs):
        r["wid"] = i
    return recs


@torch.no_grad()
def teacher_forced(mdl, tok, dev, seq, lengths, n_spans, rng, pad=600,
                   span_range=None):
    span_range = span_range or (lambda lo, hi: (lo, hi))
    ids = tok(seq, return_tensors="pt")["input_ids"].to(dev)
    logits = mdl(input_ids=ids).logits[0].float()
    bid = [tok.get_vocab()[b] for b in BASES]
    pred_idx = logits[:, bid].argmax(-1).cpu().numpy()
    true = np.array([BASES.index(c) if c in BASES else -1 for c in seq])
    # HyenaDNA appends [SEP] and prepends nothing, so ids == seq + [SEP] and
    # logits[i] predicts seq[i+1]. Getting this one index wrong turns any result
    # into a near-chance null that looks exactly like the finding, so it is
    # asserted against a poly-A probe in main() rather than inferred here.
    n = len(true) - 1
    correct = pred_idx[:n] == true[1:1 + n]
    rows = []
    for L in lengths:
        lo, hi = span_range(pad, n - L)
        if hi <= lo:
            continue
        for st in rng.integers(lo, hi, size=n_spans):
            c = correct[st:st + L]
            run = best = 0
            for v in c:
                run = run + 1 if v else 0
                best = max(best, run)
            rows.append(dict(decoder="teacher_forced", L=int(L), span_start=int(st),
                             exact=int(c.all()), nt_acc=float(c.mean()),
                             max_run_nt=int(best)))
    return rows


COPY_LO, COPY_HI = 1800, 2400          # where build() pastes the duplicated block


@torch.no_grad()
def free_running(mdl, tok, dev, seq, L, starts, prefix=1500):
    bid = [tok.get_vocab()[b] for b in BASES]
    rows = []
    for st in starts:
        ctx = seq[max(0, st - prefix):st]
        ids = tok(ctx, return_tensors="pt")["input_ids"].to(dev)
        gen = []
        for _ in range(L):
            j = int(mdl(input_ids=ids).logits[0, -1, bid].argmax())
            gen.append(BASES[j])
            ids = torch.cat([ids, torch.tensor([[bid[j]]], device=dev)], 1)
        truth = seq[st:st + L]
        c = np.array([a == b for a, b in zip(gen, truth)])
        lead = 0
        while lead < len(c) and c[lead]:
            lead += 1
        run = best = 0
        for v in c:
            run = run + 1 if v else 0
            best = max(best, run)
        rows.append(dict(decoder="free_running", L=L, span_start=int(st),
                         exact=int(c.all()), nt_acc=float(c.mean()),
                         max_run_nt=int(best), lead_nt=lead))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="LongSafari/hyenadna-small-32k-seqlen-hf")
    ap.add_argument("--window", type=int, default=3000)
    ap.add_argument("--lengths", type=int, nargs="+", default=[6, 12, 30, 60, 120, 240])
    ap.add_argument("--n-spans", type=int, default=8)
    ap.add_argument("--n-windows", type=int, default=30)
    ap.add_argument("--free-L", type=int, default=60)
    ap.add_argument("--free-spans", type=int, default=2)
    ap.add_argument("--device", default="")
    ap.add_argument("--seed", type=int, default=5)
    ap.add_argument("--out", default="results/memorisation_dna_hyena.csv")
    args = ap.parse_args()

    recs = build(args.window, args.n_windows, args.seed)
    from collections import Counter
    print("windows per arm:", dict(Counter(r["arm"] for r in recs)), flush=True)
    print("mean GC per arm:", {a: round(float(np.mean([r["gc"] for r in recs if r["arm"] == a])), 4)
                               for a in sorted({r["arm"] for r in recs})}, flush=True)

    from transformers import AutoModelForCausalLM, AutoTokenizer
    dev = args.device or _device()
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    mdl = AutoModelForCausalLM.from_pretrained(args.model, trust_remote_code=True).to(dev).eval()
    # alignment probe: after 400 A's the only sane next nucleotide is A.
    probe = teacher_forced(mdl, tok, dev, "A" * 600, [100], 1,
                           np.random.default_rng(0), pad=200)
    print(f"{args.model} on {dev}; poly-A alignment probe nt_acc "
          f"{probe[0]['nt_acc']:.3f} (must be ~1.0)", flush=True)
    assert probe[0]["nt_acc"] > 0.9, "next-nucleotide indexing is off by one"

    out = (ROOT / args.out).open("w", newline="")
    w = csv.DictWriter(out, fieldnames=["wid", "arm", "genome", "gc", "decoder", "L",
                                        "span_start", "exact", "nt_acc", "max_run_nt",
                                        "lead_nt"], extrasaction="ignore")
    w.writeheader()
    t0 = time.time()
    for n, rec in enumerate(recs, 1):
        rng = np.random.default_rng(args.seed * 31 + rec["wid"])
        # a copy_control span is only a control if it lands inside the pasted block
        sr = ((lambda lo, hi: (COPY_LO + 20, min(hi, COPY_HI - 20)))
              if rec["arm"] == "copy_control" else None)
        rows = teacher_forced(mdl, tok, dev, rec["seq"], args.lengths,
                              args.n_spans, rng, span_range=sr)
        if rec["arm"] in ("human", "bacteria", "copy_control"):
            lo = COPY_LO + 20 if rec["arm"] == "copy_control" else 1500
            hi = (COPY_HI - args.free_L - 10 if rec["arm"] == "copy_control"
                  else args.window - args.free_L - 10)
            starts = rng.integers(lo, hi, size=args.free_spans)
            rows += free_running(mdl, tok, dev, rec["seq"], args.free_L, starts)
        for r in rows:
            r.update(wid=rec["wid"], arm=rec["arm"], genome=rec["genome"],
                     gc=round(rec["gc"], 4))
        w.writerows(rows)
        out.flush()
        if n % 10 == 0:
            print(f"[{n}/{len(recs)}] {(time.time()-t0)/60:.1f} min", flush=True)
    out.close()

    rows = list(csv.DictReader((ROOT / args.out).open()))
    for dec in ("teacher_forced", "free_running"):
        x0 = [r for r in rows if r["decoder"] == dec]
        if not x0:
            continue
        print(f"\n--- {dec}")
        print(f"{'arm':15s} {'L(nt)':>6s} {'exact':>8s} {'nt_acc':>8s} "
              f"{'max_run':>8s} {'lead':>6s} {'n':>5s}")
        for arm in ("human", "bacteria", "markov5_human", "dinuc_human", "copy_control"):
            for L in sorted({int(r["L"]) for r in x0}):
                x = [r for r in x0 if r["arm"] == arm and int(r["L"]) == L]
                if not x:
                    continue
                lead = ([float(r["lead_nt"]) for r in x if r["lead_nt"] != ""])
                print(f"{arm:15s} {L:6d} {np.mean([int(r['exact']) for r in x]):8.4f} "
                      f"{np.mean([float(r['nt_acc']) for r in x]):8.4f} "
                      f"{np.mean([int(r['max_run_nt']) for r in x]):8.2f} "
                      f"{(np.mean(lead) if lead else float('nan')):6.2f} {len(x):5d}")


if __name__ == "__main__":
    main()
