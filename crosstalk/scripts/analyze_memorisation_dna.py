#!/usr/bin/env python3
"""Recall curves, the membership contrast and the independence null for a genomic LM.

Windows inside one genome are not independent, so every interval and every test
resamples or permutes GENOMES, not rows -- the rule section 18 uses for
ProteinGym clusters.

Two things this reports that section 28 did not need. First a composition floor:
on a 63% GC genome, always emitting G scores 0.315 per nucleotide without looking
at anything, so raw per-nucleotide accuracy near 0.33 means almost nothing on its
own. Second a sensitivity bound: when per-token accuracy is a fraction of a
percent, zero exact spans is a weak observation unless it is turned into an upper
bound on how much verbatim recall could be hiding.
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ARMS = ["member", "nonmember", "meso_member", "meso_non", "opportunistum",
        "member_offframe", "markov5", "dinuc", "mono"]
NICE = {"member": "member (train)", "nonmember": "nonmember (2025-26)",
        "meso_member": "M.albiziae (train)", "meso_non": "Mesorhizobium 2026",
        "opportunistum": "CP002279 (unseen)", "member_offframe": "member off-frame",
        "markov5": "markov5 (matched)", "dinuc": "dinuc shuffle",
        "mono": "mono shuffle"}


def load(path, base=None):
    bl = {}
    if base and Path(base).exists():
        for r in csv.DictReader(Path(base).open()):
            bl.setdefault((int(r["wid"]), int(r["L"]), int(r["span_start"])), r)
    rows = []
    for r in csv.DictReader(Path(path).open()):
        r["L"] = int(r["L"]); r["exact"] = int(r["exact"])
        r["span_start"] = int(r["span_start"]); r["wid"] = int(r["wid"])
        for k in ("tok_acc", "nt_acc", "gc", "logp_true"):
            r[k] = float(r[k])
        r["max_run_nt"] = int(r["max_run_nt"])
        r["cds_frac"] = float(r["cds_frac"]) if r["cds_frac"] != "" else np.nan
        b = bl.get((r["wid"], r["L"], r["span_start"]))
        for k in ("floor_nt", "mk5_nt", "mk5_logp_tok"):
            r[k] = float(b[k]) if b and b.get(k) not in (None, "") else np.nan
        rows.append(r)
    return rows


def by(rows, **kw):
    return [r for r in rows if all(r[k] == v for k, v in kw.items())]


def gmeans(rows, field):
    g = defaultdict(list)
    for r in rows:
        v = r[field]
        if isinstance(v, float) and np.isnan(v):
            continue
        g[r["genome"]].append(v)
    return np.array([np.mean(v) for v in g.values()]), list(g)


def cluster_boot(rows, field, n=4000, seed=0):
    m, k = gmeans(rows, field)
    if len(m) < 2:
        return (float(m.mean()) if len(m) else np.nan), np.nan, np.nan, len(m)
    rng = np.random.default_rng(seed)
    bs = m[rng.integers(0, len(m), size=(n, len(m)))].mean(1)
    return float(m.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)), len(m)


def perm_genomes(a, b, field, n=20000, seed=0):
    ma, _ = gmeans(a, field); mb, _ = gmeans(b, field)
    if len(ma) < 2 or len(mb) < 2:
        return (ma.mean() - mb.mean()), np.nan, len(ma), len(mb)
    obs = ma.mean() - mb.mean()
    pool = np.concatenate([ma, mb]); na = len(ma)
    rng = np.random.default_rng(seed); cnt = 0
    for _ in range(n):
        p = pool[rng.permutation(len(pool))]
        cnt += abs(p[:na].mean() - p[na:].mean()) >= abs(obs) - 1e-12
    return obs, (cnt + 1) / (n + 1), len(ma), len(mb)


def auc(pos, neg):
    x = np.concatenate([pos, neg])
    r = np.empty(len(x)); o = np.argsort(x)
    rk = np.arange(1, len(x) + 1, dtype=float)
    i = 0
    while i < len(x):                      # average ranks over ties
        j = i
        while j + 1 < len(x) and x[o[j + 1]] == x[o[i]]:
            j += 1
        rk[i:j + 1] = (i + j + 2) / 2
        i = j + 1
    r[o] = rk
    n1 = len(pos)
    return (r[:n1].sum() - n1 * (n1 + 1) / 2) / (n1 * len(neg))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="results/memorisation_dna_nt50m.csv")
    ap.add_argument("--baselines", default="results/memorisation_dna_baselines.csv")
    ap.add_argument("--model", default="NT-v2 50M multi-species")
    a = ap.parse_args()
    rows = load(ROOT / a.csv, ROOT / a.baselines)
    rnd = [r for r in rows if r["placement"] == "random"]
    Ls = sorted({r["L"] for r in rnd})
    arms = [x for x in ARMS if by(rnd, arm=x)]

    print(f"=== {a.model}   {a.csv}")
    print(f"{len(rows)} span reconstructions over {len({r['wid'] for r in rows})} "
          f"windows of 3000 nt from {len({r['genome'] for r in rows})} sources.")
    print("L is the span length in NT 6-mer TOKENS; 6L nucleotides. L=10 tokens = "
          "60 nt = 20 codons,\nthe direct analogue of section 28's 20-residue span.\n")

    print("--- 1. RECALL CURVES (random placement)")
    print(f"{'arm':20s} {'g':>2s} " + " ".join(f"{'L=%-2d'%L:>24s}" for L in Ls))
    print(f"{'':20s} {'':>2s} " + " ".join(f"{'exact | tok | nt':>24s}" for L in Ls))
    for arm in arms:
        s = by(rnd, arm=arm); cells = []
        for L in Ls:
            x = by(s, L=L)
            if not x:
                cells.append(f"{'-':>24s}"); continue
            e = gmeans(x, "exact")[0].mean(); t = gmeans(x, "tok_acc")[0].mean()
            nt = gmeans(x, "nt_acc")[0].mean()
            cells.append(f"{e:8.5f} |{t:7.4f} |{nt:6.4f}")
        print(f"{NICE[arm]:20s} {len({r['genome'] for r in s}):2d} " + " ".join(cells))

    print("\n--- 2. TRIVIAL BASELINES on the identical spans")
    print("    composition = always emit this window's commonest base.")
    print("    markov5 = order-5 chain fitted to the SOURCE GENOME, greedy fill.")
    print(f"{'arm':20s} {'L':>3s} {'model nt':>9s} {'markov5':>9s} {'composition':>12s} "
          f"{'model-mk5':>10s} {'model-comp':>11s}")
    for arm in arms:
        for L in Ls:
            x = by(rnd, arm=arm, L=L)
            if not x:
                continue
            m = np.mean([r["nt_acc"] for r in x])
            mk = np.nanmean([r["mk5_nt"] for r in x])
            fl = np.nanmean([r["floor_nt"] for r in x])
            print(f"{NICE[arm]:20s} {L:3d} {m:9.4f} {mk:9.4f} {fl:12.4f} "
                  f"{m-mk:+10.4f} {m-fl:+11.4f}")

    print("\n--- 3. MEMBERSHIP: verified in the NT-v2 corpus vs released after it")
    print("    window pools GC-matched in 1% bins; permutation over GENOMES.")
    for label, A, B in (("all bacteria, GC-matched", "member", "nonmember"),
                        ("genus-matched: M.albiziae vs Mesorhizobium 2026",
                         "meso_member", "meso_non"),
                        ("genus-matched: M.albiziae vs CP002279",
                         "meso_member", "opportunistum")):
        x0, y0 = by(rnd, arm=A), by(rnd, arm=B)
        if not x0 or not y0:
            continue
        print(f"  [{label}]  GC {np.mean([r['gc'] for r in x0]):.4f} vs "
              f"{np.mean([r['gc'] for r in y0]):.4f}")
        for fld, tag in (("nt_acc", "per-nt accuracy"),
                         ("logp_true", "log p(true), nats/token")):
            for L in (1, 10):
                x, y = by(x0, L=L), by(y0, L=L)
                if not x or not y:
                    continue
                d, p, na, nb = perm_genomes(x, y, fld)
                mx, lo, hi, _ = cluster_boot(x, fld)
                my, lo2, hi2, _ = cluster_boot(y, fld)
                print(f"    L={L:2d} {tag:24s} member {mx:8.4f} [{lo:7.4f},{hi:7.4f}] "
                      f"({na}g)  non {my:8.4f} [{lo2:7.4f},{hi2:7.4f}] ({nb}g)  "
                      f"diff {d:+.4f} p={p if np.isnan(p) else round(p,3)}")

    print("\n--- 4. LOSS-BASED MEMBERSHIP INFERENCE (the sensitive attack)")
    print("    AUC of separating member from non-member WINDOWS. 0.5 = no signal.")
    print("    'ratio' subtracts the order-5 reference model, the standard fix for")
    print("    samples that are intrinsically easy rather than memorised.")
    for label, A, B in (("all bacteria, GC-matched", "member", "nonmember"),
                        ("genus-matched", "meso_member", "meso_non")):
        for L in Ls:
            xs, ys = by(rnd, arm=A, L=L), by(rnd, arm=B, L=L)
            if not xs or not ys:
                continue
            def win(rs, key):
                d = defaultdict(list)
                for r in rs:
                    v = r["logp_true"] - (r["mk5_logp_tok"] if key == "ratio" else 0.0)
                    if not np.isnan(v):
                        d[r["wid"]].append(v)
                return np.array([np.mean(v) for v in d.values()])
            out = []
            for key in ("raw", "ratio"):
                p_, n_ = win(xs, key), win(ys, key)
                if len(p_) < 5 or len(n_) < 5:
                    out.append("n/a"); continue
                out.append(f"{auc(p_, n_):.3f}")
            print(f"    [{label:24s}] L={L:2d}  AUC raw {out[0]}  AUC ratio {out[1]}"
                  f"   ({len(win(xs,'raw'))} vs {len(win(ys,'raw'))} windows)")

    print("\n--- 5. INDEPENDENCE NULL and the SENSITIVITY of the recall test")
    print("    p is per-TOKEN accuracy at that L. p^(6L) at nucleotide level is not")
    print("    a valid null: one token emits 6 nucleotides in a single decision.")
    print("    'ceiling' is the rule-of-three 95% upper bound on the exact-recovery")
    print("    rate given 0 successes -- what the null can and cannot exclude.")
    print(f"{'arm':20s} {'L':>3s} {'p_tok':>7s} {'p^L':>10s} {'observed':>9s} "
          f"{'excess':>10s} {'hits':>5s} {'n':>5s} {'ceiling':>8s}")
    for arm in arms:
        for L in Ls:
            x = by(rnd, arm=arm, L=L)
            if not x:
                continue
            p = np.mean([r["tok_acc"] for r in x]); obs = np.mean([r["exact"] for r in x])
            null = p ** L; ne = sum(r["exact"] for r in x); n = len(x)
            ex = "n/a" if null == 0 else (f"{obs/null:.1f}x" if obs > 0 else "0x")
            ceil = f"{3.0/n:.4f}" if ne == 0 else "-"
            print(f"{NICE[arm]:20s} {L:3d} {p:7.4f} {null:10.2e} {obs:9.5f} "
                  f"{ex:>10s} {ne:5d} {n:5d} {ceil:>8s}")

    print("\n--- 6. LONGEST EXACT RUN inside the span (nucleotides): mean / max / span")
    print(f"{'arm':20s} " + " ".join(f"{'L=%-2d'%L:>14s}" for L in Ls))
    for arm in arms:
        cells = []
        for L in Ls:
            x = by(rnd, arm=arm, L=L)
            cells.append(f"{'-':>14s}" if not x else
                         f"{np.mean([r['max_run_nt'] for r in x]):5.2f}/"
                         f"{max(r['max_run_nt'] for r in x):3d}/{6*L:3d}")
        print(f"{NICE[arm]:20s} " + " ".join(cells))

    if by(rnd, arm="member_offframe"):
        print("\n--- 7. READING FRAME: token grid on vs off the one training used")
        for L in Ls:
            x, y = by(rnd, arm="member", L=L), by(rnd, arm="member_offframe", L=L)
            if not x or not y:
                continue
            d, p, _, _ = perm_genomes(x, y, "nt_acc")
            print(f"    L={L:2d} on-grid {np.mean([r['nt_acc'] for r in x]):.4f}  "
                  f"off-grid {np.mean([r['nt_acc'] for r in y]):.4f}  "
                  f"diff {d:+.4f} p={p if np.isnan(p) else round(p,3)}")

    ann = [r for r in rows if r["placement"] in ("coding", "noncoding")]
    if ann:
        print("\n--- 8. CODING vs NON-CODING (spans placed wholly inside / outside CDS)")
        print("    CP002279 is 86.2% coding, M.albiziae 86.3%, so non-coding spans")
        print("    longer than a few tokens barely exist in a bacterial chromosome.")
        print(f"{'genome':14s} {'L':>3s} {'coding nt':>10s} {'noncoding':>10s} "
              f"{'diff':>8s} {'cod mk5':>8s} {'non mk5':>8s} {'n_c':>5s} {'n_n':>5s}")
        for g in sorted({r["genome"] for r in ann}):
            for L in sorted({r["L"] for r in ann}):
                c = [r for r in ann if r["genome"] == g and r["L"] == L and r["placement"] == "coding"]
                n = [r for r in ann if r["genome"] == g and r["L"] == L and r["placement"] == "noncoding"]
                if not c or not n:
                    continue
                mc = np.mean([r["nt_acc"] for r in c]); mn = np.mean([r["nt_acc"] for r in n])
                print(f"{g:14s} {L:3d} {mc:10.4f} {mn:10.4f} {mc-mn:+8.4f} "
                      f"{np.nanmean([r['mk5_nt'] for r in c]):8.4f} "
                      f"{np.nanmean([r['mk5_nt'] for r in n]):8.4f} {len(c):5d} {len(n):5d}")

    print("\n--- 9. WHAT THE MODEL DOES KNOW: real DNA against matched synthetic")
    for L in Ls:
        line = f"    L={L:2d} "
        for arm in ("meso_member", "opportunistum", "markov5", "dinuc", "mono"):
            x = by(rnd, arm=arm, L=L)
            if x:
                line += f"{arm}={np.mean([r['nt_acc'] for r in x]):.4f}  "
        print(line)


if __name__ == "__main__":
    main()
