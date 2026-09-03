#!/usr/bin/env python3
"""Is an ortholog the protein-side analogue of synonymous recoding? No.

Synonymous recoding is the DNA battery's preserve-side intervention and it works
because a recoded gene preserves the protein exactly while being a sequence the
model has never seen. An ortholog is the usual protein-side proposal: different
sequence, same function. It fails for a reason that is easy to state and easy to
measure -- an ortholog is itself a real natural protein, so it sits inside the
model's training distribution, and a high likelihood for it is explained by
naturalness without any appeal to preserved function.

The control that decides this is an UNRELATED natural protein matched on length.
If ESM-2 scores the ortholog no higher than an unrelated natural protein, the
ortholog's score carries no information about the function it shares with the
target, and the preserve-side arm is dead as a semantic test.

  real          the target
  ortholog      different sequence, same function, natural
  unrelated     different sequence, different function, natural, length-matched
  matched scramble   the target mutated at the same number of positions as the
                     ortholog differs by; not natural, no function

Scores are mean per-token pseudo-log-likelihood, the same as run_protein_semantics.
"""
import csv, json, sys, urllib.request
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from crosstalk import glm

AA = "ACDEFGHIKLMNPQRSTVWY"
CACHE = ROOT / "data/orthologs.json"

# target DMS id -> (UniProt accession of a functional ortholog, label)
PAIRS = {
    "BLAT_ECOLX_Deng_2012":            ("P28585", "CTX-M-1 beta-lactamase, E. coli"),
    "DYR_ECOLI_Thompson_plusLon_2019": ("P00374", "DHFR, human"),
    "SUMO1_HUMAN_Weile_2017":          ("Q12306", "SMT3, S. cerevisiae"),
    "CALM1_HUMAN_Weile_2017":          ("P06787", "calmodulin, S. cerevisiae"),
    "MK01_HUMAN_Brenan_2016":          ("P16892", "FUS3 MAP kinase, S. cerevisiae"),
    "TRPC_SACS2_Chan_2017":            ("P00909", "TrpC, E. coli"),
    "GFP_AEQVI_Sarkisyan_2016":        ("LOCAL:D7PM05_CLYGR_Somermeyer_2022",
                                        "GFP, Clytia gregaria"),
}


def fetch(acc):
    url = f"https://rest.uniprot.org/uniprotkb/{acc}.fasta"
    with urllib.request.urlopen(url, timeout=30) as r:
        txt = r.read().decode()
    return "".join(l.strip() for l in txt.splitlines() if not l.startswith(">"))


def identity(a, b):
    from Bio import Align
    from Bio.Align import substitution_matrices
    al = Align.PairwiseAligner()
    al.substitution_matrix = substitution_matrices.load("BLOSUM62")
    al.open_gap_score, al.extend_gap_score = -11, -1
    aln = al.align(a, b)[0]
    x, y = aln[0], aln[1]
    same = sum(1 for i, j in zip(x, y) if i == j and i != "-")
    cols = sum(1 for i, j in zip(x, y) if i != "-" and j != "-")
    return same / max(cols, 1)


def targets():
    tab = glm.CODON_TABLE
    cds = json.loads((ROOT / "data/cds/dms_cds.json").read_text())
    out = {}
    for k, v in cds.items():
        s = v["cds"]
        if len(s) % 3 == 0 and set(s) <= set("ACGT"):
            out[k] = "".join(tab.get(s[i:i+3], "X")
                             for i in range(0, len(s)-2, 3)).rstrip("*")
    ref = {r["DMS_id"]: r["target_seq"] for r in csv.DictReader(
        (ROOT / "data/proteingym_reference_v1.csv").open())}
    out.update({k: v for k, v in ref.items() if set(v) <= set(AA)})
    return out


def main():
    tg = targets()
    cache = json.loads(CACHE.read_text()) if CACHE.exists() else {}
    rows = []
    for t, (acc, label) in PAIRS.items():
        if acc.startswith("LOCAL:"):
            orth = tg[acc[6:]]
        else:
            if acc not in cache:
                cache[acc] = fetch(acc)
                print(f"fetched {acc} ({len(cache[acc])} aa)", flush=True)
            orth = cache[acc]
        rows.append((t, tg[t], orth, label))
    CACHE.write_text(json.dumps(cache))

    # unrelated natural control: another target of the nearest length, different family
    pool = [(k, v) for k, v in tg.items() if 60 <= len(v) <= 700 and set(v) <= set(AA)]
    rng = np.random.default_rng(0)

    conds, meta = {}, []
    for t, seq, orth, label in rows:
        fam = t.split("_")[0]
        cand = [(abs(len(v) - len(seq)), k, v) for k, v in pool
                if not k.startswith(fam) and k != t and v != seq
                and identity(seq, v) < 0.25]
        cand.sort()
        _, uk, useq = cand[0]
        f = identity(seq, orth)
        k = int(round((1 - f) * len(seq)))
        s = list(seq)
        for i in rng.choice(len(seq), size=min(k, len(seq)), replace=False):
            s[i] = rng.choice([a for a in AA if a != seq[i]])
        conds[t] = {"real": seq, "ortholog": orth, "unrelated natural": useq,
                    "matched scramble": "".join(s)}
        meta.append(dict(target=t, ortholog=label, identity=round(f, 3),
                         hamming_k=k, unrelated=uk,
                         lens=[len(seq), len(orth), len(useq)]))
        print(f"{t[:34]:34s} L={len(seq):3d}  ortholog {label[:34]:34s} "
              f"id={f:.2f}  k={k:3d}  unrelated={uk[:26]} ({len(useq)} aa)", flush=True)

    from crosstalk.plm import ESMScorer, pseudo_likelihood_positions
    sc = ESMScorer()
    order = ["real", "ortholog", "unrelated natural", "matched scramble"]
    sco = {c: {} for c in order}
    for n, (t, _, _, _) in enumerate(rows, 1):
        for c in order:
            sco[c][t] = float(pseudo_likelihood_positions(
                sc, conds[t][c], batch_size=32).mean())
        print(f"[{n}/{len(rows)}] {t[:32]:32s} "
              + "  ".join(f"{c[:4]} {sco[c][t]:+.3f}" for c in order), flush=True)

    names = [r[0] for r in rows]
    real = np.array([sco["real"][t] for t in names])
    print(f"\n--- paired against the real protein, {len(names)} proteins ---")
    out = []
    for c in order[1:]:
        x = np.array([sco[c][t] for t in names])
        d = real - x
        ci = 1.96 * d.std(ddof=1) / np.sqrt(len(d))
        print(f"{c:20s} {d.mean():+8.4f} [{d.mean()-ci:+.4f}, {d.mean()+ci:+.4f}] "
              f"real higher {int((d>0).sum())}/{len(d)}")
        out.append(dict(condition=c, mean_delta=float(d.mean()), ci=float(ci),
                        real_higher=int((d > 0).sum()), n=len(d)))
    o = np.array([sco["ortholog"][t] for t in names])
    u = np.array([sco["unrelated natural"][t] for t in names])
    d = o - u
    ci = 1.96 * d.std(ddof=1) / np.sqrt(len(d))
    print(f"\northolog minus unrelated natural: {d.mean():+.4f} "
          f"[{d.mean()-ci:+.4f}, {d.mean()+ci:+.4f}], ortholog higher "
          f"{int((d>0).sum())}/{len(d)}")
    print("  a value near zero means the ortholog's score is naturalness, not "
          "shared function, and the arm cannot serve as a synonymy analogue.")
    out.append(dict(condition="ortholog - unrelated natural",
                    mean_delta=float(d.mean()), ci=float(ci),
                    real_higher=int((d > 0).sum()), n=len(d)))

    p = ROOT / "results/protein_ortholog_arm.csv"
    keys = sorted({k for r in out for k in r})
    with p.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(out)
    (ROOT / "results/protein_ortholog_meta.json").write_text(json.dumps(meta, indent=1))
    print(f"\nwrote {p}")


if __name__ == "__main__":
    main()
