#!/usr/bin/env python3
"""Recursive Language Models, for a genomic model: does section 19's negative
context dose-response survive when the model is no longer capped by its window?

Section 19 dialled real M. opportunistum chromosome up around the ParD3 locus and
found NT-v2 50M got monotonically WORSE (trend rho -0.771, AUC 0.505 -> 0.365).
The sweep stopped at 5,400 nt of flank because 11 kb is NT's whole 2,048-token
context. So the result cannot distinguish "genomic context does not help NT" from
"NT cannot use context past its window, and the sweep measured the window".

Zhang, Kraska & Khattab (arXiv:2512.24601) separate exactly this for text: treat
the long input as an environment, decompose it, call the model on pieces, combine
the returned answers, and reach inputs two orders of magnitude past the window.
`scripts/recursive_context.py` states the genomic decomposition in full and fixes
it before any run: 1,440 nt chunks per side walking outward from the CDS, each
spliced as U_i ++ CDS ++ D_i, aggregated by UNWEIGHTED MEAN. Five other
aggregation rules are computed and written out, and none of them is allowed to
be the headline (section 34: the readout is a degree of freedom big enough to
move splice-site accuracy from 0.756 to 0.938).

Three things make the result readable:

  faithfulness   at flank 1,440 the recursion has exactly one chunk and that
                 chunk IS the contiguous locus, so it must reproduce the direct
                 scorer bit for bit; at 2,880 / 4,320 / 5,760 both scorers run
                 and can be compared. If they disagree there, nothing past the
                 window is worth reading.
  shuffled flank same lengths, same composition, real order destroyed --
                 mononucleotide and dinucleotide-preserving. A trend on the real
                 locus but not the shuffled one is context; a trend on both is
                 an aggregation artifact.
  Markov order-5 the identical pipeline driven by a chain that beats nothing.
                 Section 32 put NT-v2 only ~2 points of nucleotide accuracy above
                 it. Without this reference a null here is uninterpretable.
"""
import argparse, csv, json, sys, time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))

from crosstalk.landscape import load_pard3
from crosstalk import glm
from run_proxy_ladder import auc, boot_ci, spearman
from run_genomic_rung import discrimination_set
from recursive_context import (CHUNK, AGGREGATIONS, MarkovScorer, aggregate,
                               chunk_inputs, dinuc_shuffle, mono_shuffle)

COMP = str.maketrans("ACGT", "TGCA")
GENOME = "CP002279"
NT_MAX_TOKENS = 2040

DIRECT_FLANKS = [0, 60, 300, 1200, 1440, 2880, 3000, 4320, 5400, 5760]
MATCHED = [1440, 2880, 4320, 5760]          # both scorers possible
REC_FLANKS = [1440, 2880, 4320, 5760, 8640, 11520, 14400, 17280, 20160,
               25920, 31680, 40320]
TRANSFORMS = ("real", "shuffle_mono", "shuffle_dinuc")


def make_transform(kind):
    if kind == "real":
        return None
    fn = mono_shuffle if kind == "shuffle_mono" else dinuc_shuffle
    def t(seq, tag):
        return fn(seq, abs(hash((kind, tag))) % (2 ** 31))
    return t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="InstaDeepAI/nucleotide-transformer-v2-50m-multi-species")
    ap.add_argument("--chunk", type=int, default=CHUNK)
    ap.add_argument("--max-flank", type=int, default=40320)
    ap.add_argument("--skip-nt", action="store_true")
    ap.add_argument("--out-prefix", default="results/recursive_context")
    args = ap.parse_args()

    import torch
    torch.set_num_threads(2)

    L = load_pard3()
    variants = L.seqs
    mask, lab, w3, w2 = discrimination_set(L)
    wt_cds = glm.load_cds()["ParD3"]["cds"]

    genome = (ROOT / "data" / "cds" / f"{GENOME}.txt").read_text().strip()
    oriented = genome if wt_cds in genome else genome.translate(COMP)[::-1]
    start = oriented.find(wt_cds); end = start + len(wt_cds)
    assert start >= 0
    print(f"{len(variants)} variants, discrimination set {int(mask.sum())}; "
          f"ParD3 at {start:,}-{end:,} on a {len(oriented):,} nt contig", flush=True)
    print(f"chunk {args.chunk} nt/side, aggregation headline = MEAN "
          f"(pre-registered); max flank {args.max_flank:,} nt/side\n", flush=True)

    from crosstalk.boltz import MUT_POSITIONS, PARD3
    wt_code = "".join(PARD3[p - 1] for p in MUT_POSITIONS)
    nmut = np.array([sum(a != b for a, b in zip(v, wt_code)) for v in variants], float)
    base_auc = auc(-nmut[mask], lab)
    print(f"trivial baseline (mutation count): AUC {base_auc:.3f}\n", flush=True)

    scorers = {}
    scorers["markov5"] = MarkovScorer(oriented, 5, exclude=(start - 25000, end + 25000))
    if not args.skip_nt:
        scorers["nt_v2_50m"] = glm.NTScorer(args.model)

    kmax = args.max_flank // args.chunk
    rows, chunk_rows, faith_rows = [], [], []
    direct_scores, rec_scores = {}, {}

    def metrics(s):
        a = auc(s[mask], lab); lo, hi = boot_ci(s[mask], lab)
        return dict(auc=a, auc_lo=lo, auc_hi=hi,
                    rho_on=spearman(s, w3), rho_off=spearman(s, w2))

    for sname, sc in scorers.items():
        # ---------------------------------------------------------- direct arm
        for tf in TRANSFORMS:
            flanks = DIRECT_FLANKS if tf == "real" else MATCHED
            trans = make_transform(tf)
            for F in flanks:
                lo = start - F; hi = end + F
                assert F % 6 == 0
                n_tok = (hi - lo) // 6
                if sname.startswith("nt") and n_tok > NT_MAX_TOKENS:
                    continue
                u, d = oriented[lo:start], oriented[end:hi]
                if trans is not None and F:
                    u, d = trans(u, "U"), trans(d, "D")
                seg = u + wt_cds + d
                t0 = time.time()
                s = sc.score_variants_masked_marginal(variants, seg, cds_offset=F)
                m = metrics(s)
                direct_scores[(sname, tf, F)] = s
                rows.append(dict(scorer=sname, mode="direct", transform=tf, flank=F,
                                 n_chunks=1, aggregation="none", total_nt=len(seg),
                                 tokens=n_tok, baseline_auc=base_auc, **m))
                print(f"{sname:10s} direct   {tf:14s} F={F:6d} tok={n_tok:5d} "
                      f"AUC {m['auc']:.3f} [{m['auc_lo']:.3f},{m['auc_hi']:.3f}] "
                      f"rho_on {m['rho_on']:+.3f} ({time.time()-t0:.1f}s)", flush=True)

        # ------------------------------------------------------- recursive arm
        for tf in TRANSFORMS:
            trans = make_transform(tf)
            per_chunk = np.zeros((kmax, len(variants)))
            for i in range(kmax):
                ins = chunk_inputs(oriented, start, end, (i + 1) * args.chunk,
                                   chunk=args.chunk, transform=trans)
                seq, off = ins[-1]                       # chunk i is the outermost
                t0 = time.time()
                per_chunk[i] = sc.score_variants_masked_marginal(variants, seq, cds_offset=off)
                m = metrics(per_chunk[i])
                chunk_rows.append(dict(scorer=sname, transform=tf, chunk_index=i,
                                       distance_nt=i * args.chunk, **m))
                print(f"{sname:10s} chunk {i:2d} {tf:14s} d={i*args.chunk:6d} "
                      f"AUC {m['auc']:.3f} rho_on {m['rho_on']:+.3f} "
                      f"({time.time()-t0:.1f}s)", flush=True)
            for F in REC_FLANKS:
                k = F // args.chunk
                if k > kmax:
                    continue
                for rule in AGGREGATIONS:
                    s = aggregate(per_chunk[:k], rule)
                    m = metrics(s)
                    if rule == "mean":
                        rec_scores[(sname, tf, F)] = s
                    rows.append(dict(scorer=sname, mode="recursive", transform=tf,
                                     flank=F, n_chunks=k, aggregation=rule,
                                     total_nt=2 * F + len(wt_cds),
                                     tokens=(2 * args.chunk + len(wt_cds)) // 6,
                                     baseline_auc=base_auc, **m))
                mm = metrics(aggregate(per_chunk[:k], "mean"))
                print(f"{sname:10s} recurse  {tf:14s} F={F:6d} k={k:2d} "
                      f"AUC {mm['auc']:.3f} [{mm['auc_lo']:.3f},{mm['auc_hi']:.3f}] "
                      f"rho_on {mm['rho_on']:+.3f}", flush=True)

    # ------------------------------------------------------------ faithfulness
    for (sname, tf, F), rs in rec_scores.items():
        ds = direct_scores.get((sname, tf, F))
        if ds is None:
            continue
        faith_rows.append(dict(scorer=sname, transform=tf, flank=F,
                               n_chunks=F // args.chunk,
                               max_abs_diff=float(np.max(np.abs(rs - ds))),
                               rho_rec_vs_direct=spearman(rs, ds),
                               pearson=float(np.corrcoef(rs, ds)[0, 1]),
                               auc_recursive=auc(rs[mask], lab),
                               auc_direct=auc(ds[mask], lab)))

    pre = ROOT / args.out_prefix
    for name, data in (("", rows), ("_chunks", chunk_rows), ("_faithfulness", faith_rows)):
        if not data:
            continue
        keys = sorted({k for r in data for k in r})
        p = Path(str(pre) + name + ".csv")
        with p.open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=keys); w.writeheader(); w.writerows(data)
        print(f"wrote {p}")

    # ------------------------------------------------------------------ trends
    print("\n=== trends (Spearman AUC vs flank), mean aggregation ===")
    trend = []
    for sname in scorers:
        for tf in TRANSFORMS:
            for mode, grid in (("direct", DIRECT_FLANKS if tf == "real" else MATCHED),
                               ("recursive", REC_FLANKS)):
                sel = [r for r in rows if r["scorer"] == sname and r["mode"] == mode
                       and r["transform"] == tf
                       and r["aggregation"] in ("none", "mean")]
                if len(sel) < 3:
                    continue
                sel.sort(key=lambda r: r["flank"])
                f = [r["flank"] for r in sel]; a = [r["auc"] for r in sel]
                rho = spearman(f, a)
                sub = [r for r in sel if r["flank"] in MATCHED]
                rho_m = spearman([r["flank"] for r in sub], [r["auc"] for r in sub]) \
                    if len(sub) >= 3 else np.nan
                trend.append(dict(scorer=sname, mode=mode, transform=tf, n_points=len(sel),
                                  flank_min=min(f), flank_max=max(f),
                                  auc_min=min(a), auc_max=max(a),
                                  trend_rho=rho, trend_rho_matched_flanks=rho_m))
                print(f"{sname:10s} {mode:9s} {tf:14s} n={len(sel):2d} "
                      f"F {min(f)}-{max(f)}  AUC {min(a):.3f}-{max(a):.3f}  "
                      f"rho {rho:+.3f}   matched-only rho {rho_m:+.3f}")
    p = Path(str(pre) + "_trends.csv")
    keys = sorted({k for r in trend for k in r})
    with p.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys); w.writeheader(); w.writerows(trend)
    print(f"wrote {p}")


if __name__ == "__main__":
    main()
