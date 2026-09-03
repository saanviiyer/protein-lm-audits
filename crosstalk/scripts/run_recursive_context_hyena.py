#!/usr/bin/env python3
"""The long-window contrast: HyenaDNA-32k reading up to 30 kb of real upstream
chromosome DIRECTLY, no recursion needed.

NT-v2's 2,048 6-mer tokens cap it at ~12 kb, which is why section 19's sweep
stopped at 5,400 nt of flank and why the recursive decomposition exists at all.
HyenaDNA-small-32k has a 32,768 nt window, so it can be handed 30 kb of the same
locus in one pass. If long-range genomic context carries ParD3 specificity
signal, a model that can actually hold it should show the rise that NT's
recursion does not.

Two honest differences from the NT arms, stated rather than hidden:

  causal      HyenaDNA is autoregressive, so DOWNSTREAM flank cannot influence
              the CDS predictions at all. This arm therefore varies UPSTREAM
              context only, and is not a one-to-one substitute for the symmetric
              flanks of section 19.
  wt-marginal the variant score is sum over the three mutated codons of
              log p(codon | wild-type prefix) - log p(wt codon | wild-type
              prefix), the standard autoregressive wild-type-marginal. Only the
              codons actually reachable by the landscape are scored, so a whole
              context costs 3 sites x ceil(21/batch) passes rather than 7,882.

Same discrimination set, same AUC, same shuffled composition-matched controls.
"""
import argparse, csv, sys, time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))

from crosstalk.boltz import MUT_POSITIONS, PARD3
from crosstalk.landscape import load_pard3
from crosstalk import glm
from run_proxy_ladder import auc, boot_ci, spearman
from run_genomic_rung import discrimination_set
from recursive_context import dinuc_shuffle, mono_shuffle

COMP = str.maketrans("ACGT", "TGCA")
FLANKS = [0, 1440, 5760, 11520, 17280, 23040, 30000]


@torch.no_grad()
def codon_logprobs(sc, seq: str, nt_pos: int, codons: list[str], batch: int = 7):
    """log p(codon at nt_pos | prefix) for each candidate, autoregressive."""
    base_ids = sc.tok(seq, return_tensors="pt")["input_ids"][0]
    out = {}
    for s in range(0, len(codons), batch):
        grp = codons[s:s + batch]
        ids = base_ids.repeat(len(grp), 1)
        for r, c in enumerate(grp):
            for j, ch in enumerate(c):
                ids[r, nt_pos + j] = sc.tok.get_vocab()[ch]
        logits = sc.model(input_ids=ids.to(sc.device)).logits.float()
        lp = torch.log_softmax(logits, -1).cpu()
        for r, c in enumerate(grp):
            tot = 0.0
            for j, ch in enumerate(c):
                tot += float(lp[r, nt_pos + j - 1, sc.tok.get_vocab()[ch]])
            out[c] = tot
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--out", default="results/recursive_context_hyena.csv")
    args = ap.parse_args()
    torch.set_num_threads(2)

    L = load_pard3(); variants = L.seqs
    mask, lab, w3, w2 = discrimination_set(L)
    wt_cds = glm.load_cds()["ParD3"]["cds"]
    g = (ROOT / "data" / "cds" / "CP002279.txt").read_text().strip()
    oriented = g if wt_cds in g else g.translate(COMP)[::-1]
    start = oriented.find(wt_cds)
    sc = glm.HyenaScorer()
    pref = glm.preferred_codons()

    cands = sorted({pref[a] for a in "ACDEFGHIKLMNPQRSTVWY"} |
                   {wt_cds[(p - 1) * 3:(p - 1) * 3 + 3] for p in MUT_POSITIONS})

    def score(seq, off):
        tabs = []
        for p in MUT_POSITIONS:
            tabs.append(codon_logprobs(sc, seq, off + (p - 1) * 3, cands))
        wt = [seq[off + (p - 1) * 3: off + (p - 1) * 3 + 3] for p in MUT_POSITIONS]
        out = np.zeros(len(variants))
        for i, v in enumerate(variants):
            t = 0.0
            for k, (aa, p) in enumerate(zip(v, MUT_POSITIONS)):
                c = wt[k] if PARD3[p - 1] == aa else pref[aa]
                t += tabs[k][c] - tabs[k][wt[k]]
            out[i] = t
        return out

    rows = []
    for kind in ["real"] + [f"{t}_s{s}" for t in ("shuffle_mono", "shuffle_dinuc")
                            for s in range(args.seeds)]:
        for F in FLANKS:
            up = oriented[start - F:start] if F else ""
            if kind != "real" and F:
                fn = mono_shuffle if kind.startswith("shuffle_mono") else dinuc_shuffle
                up = fn(up, int(kind.split("_s")[1]) * 991 + F)
            t0 = time.time()
            s = score(up + wt_cds, len(up))
            a = auc(s[mask], lab); lo, hi = boot_ci(s[mask], lab)
            rows.append(dict(model="hyenadna_small_32k", transform=kind, flank_upstream=F,
                             total_nt=F + len(wt_cds), auc=a, auc_lo=lo, auc_hi=hi,
                             rho_on=spearman(s, w3), rho_off=spearman(s, w2)))
            print(f"{kind:18s} up={F:6d} nt={F+282:6d} AUC {a:.3f} [{lo:.3f},{hi:.3f}] "
                  f"rho_on {spearman(s, w3):+.3f} ({time.time()-t0:.1f}s)", flush=True)
        sel = [r for r in rows if r["transform"] == kind]
        print(f"  trend {kind}: "
              f"{spearman([r['flank_upstream'] for r in sel], [r['auc'] for r in sel]):+.3f}",
              flush=True)

    p = ROOT / args.out
    keys = sorted({k for r in rows for k in r})
    with p.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys); w.writeheader(); w.writerows(rows)
    print(f"wrote {p}")


if __name__ == "__main__":
    main()
