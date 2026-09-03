#!/usr/bin/env python3
"""Does a genomic LM write down the genomes it was trained on? (Morris et al. 2025, DNA)

Section 28 ran this for ESM-2 and could only proxy training-set membership. Here
membership is a fact on disk: the NT-v2 multi-species corpus manifest lists all
850 genomes and splits by line number, so a genome is either in pretraining or it
is not. `scripts/memdna_data.py` records what that manifest says, including the
correction that CP002279 -- the chromosome this project has -- is NOT in it.

Measurement, exactly parallel to section 28. Mask a contiguous span of L TOKENS
(NT-v2 tokens are non-overlapping 6-mers, so L tokens = 6L nucleotides),
reconstruct every masked position by argmax over the 4096 6-mer tokens in ONE
forward pass, and record whether the whole span came back exactly. L = 10 tokens
= 60 nt = 20 codons is the direct analogue of section 28's 20-residue span.

Arms:
  member       11 bacteria verified in the NT-v2 multi-species TRAIN split
  nonmember    13 bacteria first released to ENA in 2025-2026, after the model
  meso_member  M. albiziae alone      -- genus-matched membership contrast
  meso_non     2 Mesorhizobium, 2026  -- genus-matched, unseen
  opportunistum CP002279, this project's chromosome; unseen, and annotated
  mono         mononucleotide shuffle of the real windows  (GC matched exactly)
  dinuc        Altschul-Erikson dinucleotide shuffle       (dinucs matched exactly)
  markov5      order-5 chain fitted to the same genome     (6-mer/token matched, novel)

member and nonmember pools are GC-matched window by window in 1% bins.
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
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import memdna_data as D
from crosstalk.glm import _device, _nt_config, _patch_transformers_for_nt_v2

FIELDS = ["wid", "arm", "genome", "contig", "start", "gc", "L", "span_start",
          "placement", "cds_frac", "exact", "tok_acc", "nt_acc", "max_run_nt",
          "logp_true"]


# ------------------------------------------------------------------ windows
def build_windows(W: int, seed: int, n_pool: int, n_focus: int) -> list[dict]:
    rng = np.random.default_rng(seed)
    gen = ROOT / "data" / "genomes"
    out: list[dict] = []

    def add(arm, genome, recs):
        for r in recs:
            out.append(dict(arm=arm, genome=genome, **r))

    # --- member / nonmember pools, oversampled then GC-matched -------------
    memb, nonm, offf = [], [], []
    for name, fn in D.MEMBERS.items():
        seqs = D.load_fasta(gen / fn)
        for r in D.sample_windows(seqs, n_pool, W, rng, frame=0):
            memb.append(dict(genome=name, **r))
        for r in D.sample_windows(seqs, max(2, n_pool // 3), W, rng, frame=3):
            offf.append(dict(genome=name, **r))
    for name, fn in D.NONMEMBERS.items():
        for r in D.sample_windows(D.load_fasta(gen / fn), n_pool, W, rng, frame=0):
            nonm.append(dict(genome=name, **r))
    for r in offf:
        out.append(dict(arm="member_offframe", **r))
    ma, mb = D.gc_match(memb, nonm, rng, binw=0.01, cap=6)
    for r in ma:
        out.append(dict(arm="member", **r))
    for r in mb:
        out.append(dict(arm="nonmember", **r))
    print(f"  member pool {len(memb)} -> {len(ma)} GC-matched; "
          f"nonmember {len(nonm)} -> {len(mb)}", flush=True)

    # --- genus-matched membership contrast ---------------------------------
    alb = D.load_fasta(gen / D.MEMBERS[D.MESO_MEMBER])
    alb_cds = D.cds_intervals_gff(gen / "M_329.gff.gz")
    alb_mask = {c: D.coding_mask(len(s), alb_cds.get(c, [])) for c, s in alb.items()}
    meso_m = D.sample_windows(alb, n_focus, W, rng, masks=alb_mask, frame=0)
    add("meso_member", D.MESO_MEMBER, meso_m)

    meso_n = []
    for name in D.MESO_NONMEMBER:
        meso_n += [dict(genome=name, **r) for r in
                   D.sample_windows(D.load_fasta(gen / D.NONMEMBERS[name]),
                                    n_focus // 2, W, rng, frame=0)]
    for r in meso_n:
        out.append(dict(arm="meso_non", **r))

    # --- CP002279: unseen, annotated, and the genome this project uses -----
    opp = D.load_opportunistum()
    opp_mask = {"CP002279": D.coding_mask(len(opp["CP002279"]), D.cds_intervals_embl())}
    opp_w = D.sample_windows(opp, n_focus, W, rng, masks=opp_mask, frame=0)
    add("opportunistum", "CP002279", opp_w)

    # --- composition controls, derived from the two annotated Mesorhizobium -
    base = ([dict(genome=D.MESO_MEMBER, **r) for r in meso_m] +
            [dict(genome="CP002279", **r) for r in opp_w])
    for r in base:
        out.append(dict(arm="mono", genome=r["genome"], contig=r["contig"],
                        start=r["start"], seq=D.mono_shuffle(r["seq"], rng),
                        gc=r["gc"]))
        d = D.dinuc_shuffle(r["seq"], rng)
        out.append(dict(arm="dinuc", genome=r["genome"], contig=r["contig"],
                        start=r["start"], seq=d,
                        gc=(d.count("G") + d.count("C")) / len(d)))
    for label, seqs in (("M.albiziae", alb), ("CP002279", opp)):
        mk = D.Markov5(seqs)
        for _ in range(n_focus):
            s = mk.emit(W, rng)
            out.append(dict(arm="markov5", genome=label, contig="-", start=-1,
                            seq=s, gc=(s.count("G") + s.count("C")) / W))

    for i, r in enumerate(out):
        r["wid"] = i
    return out


# ------------------------------------------------------------------- spans
def plan_spans(rec, ntok: int, lengths, n_spans: int, ann_lengths, n_ann: int, rng):
    """(L, token_start, placement) jobs for one window."""
    pad = 8
    jobs = []
    cod = rec.get("coding")
    for L in lengths:
        hi = ntok - L - pad
        if hi <= pad:
            continue
        jobs += [(L, int(s), "random") for s in
                 rng.integers(pad, hi, size=n_spans)]
        if cod is None or L not in ann_lengths:
            continue
        # deliberately place some spans fully inside / fully outside CDS
        starts = np.arange(pad, hi)
        frac = np.array([cod[s * 6:(s + L) * 6].mean() for s in starts])
        for want, tag in ((1.0, "coding"), (0.0, "noncoding")):
            ok = starts[frac == want]
            if len(ok):
                jobs += [(L, int(s), tag) for s in
                         rng.choice(ok, size=min(n_ann, len(ok)), replace=False)]
    return jobs


@torch.no_grad()
def run_window(mdl, tok, dev, six_ids, six_tok, rec, jobs, batch: int):
    vocab_t = tok.get_vocab()
    seq = rec["seq"]
    base = tok(seq, return_tensors="pt")["input_ids"]
    ntok = base.shape[1] - 1
    # MPS: never search for the mask with nonzero(). Token t sits at index t+1.
    assert six_tok[int(base[0, 1])] == seq[:6], "token 1 is not seq[0:6]"
    true_tok = [seq[t * 6:(t + 1) * 6] for t in range(ntok)]
    cod = rec.get("coding")
    rows = []
    for s in range(0, len(jobs), batch):
        chunk = jobs[s:s + batch]
        ids = base.repeat(len(chunk), 1)
        for r, (L, st, _) in enumerate(chunk):
            ids[r, st + 1:st + L + 1] = tok.mask_token_id
        logits = mdl(input_ids=ids.to(dev)).logits.float()
        for r, (L, st, tag) in enumerate(chunk):
            sub = logits[r, st + 1:st + L + 1][:, six_ids]      # (L, 4096)
            pred = sub.argmax(-1).cpu().numpy()
            lp = torch.log_softmax(logits[r, st + 1:st + L + 1], -1).cpu().numpy()
            truth = true_tok[st:st + L]
            hits = 0
            nt_hits = 0
            logp = 0.0
            run = best = 0
            for j, t in enumerate(truth):
                p = six_tok[six_ids[int(pred[j])]]
                hits += int(p == t)
                nt_hits += sum(p[b] == t[b] for b in range(6))
                for b in range(6):
                    run = run + 1 if p[b] == t[b] else 0
                    best = max(best, run)
                logp += float(lp[j, vocab_t.get(t, tok.unk_token_id)])
            rows.append(dict(
                wid=rec["wid"], arm=rec["arm"], genome=rec["genome"],
                contig=rec["contig"], start=rec["start"], gc=round(rec["gc"], 4),
                L=L, span_start=st, placement=tag,
                cds_frac=(round(float(cod[st * 6:(st + L) * 6].mean()), 4)
                          if cod is not None else ""),
                exact=int(hits == L), tok_acc=hits / L, nt_acc=nt_hits / (6 * L),
                max_run_nt=best,
                logp_true=round(logp / L, 4)))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model",
                    default="InstaDeepAI/nucleotide-transformer-v2-50m-multi-species")
    ap.add_argument("--window", type=int, default=3000, help="nt, multiple of 6")
    ap.add_argument("--lengths", type=int, nargs="+", default=[1, 2, 5, 10, 20, 40],
                    help="span length in 6-mer TOKENS")
    ap.add_argument("--n-spans", type=int, default=6)
    ap.add_argument("--ann-lengths", type=int, nargs="+", default=[1, 2, 5])
    ap.add_argument("--n-ann", type=int, default=4)
    ap.add_argument("--n-pool", type=int, default=14, help="candidate windows/genome")
    ap.add_argument("--n-focus", type=int, default=40)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--arms", nargs="*", default=None, help="restrict to these arms")
    ap.add_argument("--per-arm", type=int, default=0, help="cap windows per arm")
    ap.add_argument("--out", default="results/memorisation_dna.csv")
    args = ap.parse_args()
    assert args.window % 6 == 0

    print("building windows ...", flush=True)
    wins = build_windows(args.window, args.seed, args.n_pool, args.n_focus)
    if args.arms:
        wins = [x for x in wins if x["arm"] in args.arms]
    if args.per_arm:
        rngA = np.random.default_rng(2)
        keep, seen = [], {}
        for i in rngA.permutation(len(wins)):
            x = wins[i]
            if seen.get(x["arm"], 0) < args.per_arm:
                seen[x["arm"]] = seen.get(x["arm"], 0) + 1
                keep.append(x)
        wins = sorted(keep, key=lambda x: x["wid"])
    if args.limit:
        rng0 = np.random.default_rng(1)
        wins = [wins[i] for i in sorted(rng0.permutation(len(wins))[:args.limit])]
    from collections import Counter
    print("  windows per arm:", dict(Counter(w["arm"] for w in wins)), flush=True)

    out = ROOT / args.out
    out.parent.mkdir(exist_ok=True)
    done = set()
    if out.exists():
        done = {int(r["wid"]) for r in csv.DictReader(out.open())}
        print(f"  resuming: {len(done)} windows already done", flush=True)
    f = out.open("a", newline="")
    w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
    if not done:
        w.writeheader()

    _patch_transformers_for_nt_v2()
    from transformers import AutoModelForMaskedLM, AutoTokenizer
    dev = _device()
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    mdl = AutoModelForMaskedLM.from_pretrained(
        args.model, config=_nt_config(args.model), trust_remote_code=True).to(dev).eval()
    vocab = tok.get_vocab()
    six = sorted(t for t in vocab if len(t) == 6 and not set(t) - set("ACGT"))
    assert len(six) == 4096, len(six)
    six_ids = [vocab[t] for t in six]
    six_tok = {vocab[t]: t for t in six}
    print(f"  {args.model} on {dev}, {len(six_ids)} 6-mer tokens\n", flush=True)

    ntok = args.window // 6
    t0 = time.time()
    todo = [x for x in wins if x["wid"] not in done]
    for n, rec in enumerate(todo, 1):
        rng = np.random.default_rng(args.seed * 100003 + rec["wid"])
        jobs = plan_spans(rec, ntok, args.lengths, args.n_spans,
                          set(args.ann_lengths), args.n_ann, rng)
        rows = run_window(mdl, tok, dev, six_ids, six_tok, rec, jobs, args.batch)
        w.writerows(rows)
        f.flush()
        if n % 10 == 0 or n == len(todo):
            el = time.time() - t0
            print(f"[{n:4d}/{len(todo)}] {rec['arm']:14s} {rec['genome'][:22]:22s} "
                  f"{el/60:5.1f} min, eta {(el/n)*(len(todo)-n)/60:5.1f} min",
                  flush=True)
    f.close()
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
