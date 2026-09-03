#!/usr/bin/env python3
"""The protein-side analogue of the reading-frame test (section 26).

Section 26 is decisive because DNA supplies two interventions with exactly known
semantics: synonymous recoding preserves the protein exactly while replacing the
surface form, and a one-nucleotide rotation preserves nucleotide composition
exactly while destroying the reading frame. Protein has no genetic code, so the
equivalent pair has to be constructed and each half labelled for exactness.

What transfers exactly is the DESTROY side. On DNA, frameshift is a rotation of
the string. The identical operation on a protein is a circular permutation, and
it preserves the residue multiset exactly and every forward-read dipeptide except
the one spanning the new join -- exactly the invariance profile the frameshift
has. So `rotate +1` here is not an analogue of frameshift, it is the same
operation applied to the other alphabet, and its number is directly comparable to
the +0.0264 nats/token in the section-26 table.

conditions
  real
  rotate +1        composition exact, all dipeptides but the join exact; moves
                   the termini by one. Direct counterpart of frameshift +1.
  rotate mid       same invariances, cut at L//2; the circular-permutant regime
                   that some real proteins tolerate.
  reverse          composition exact; dipeptide matrix exactly transposed.
                   Destroys N->C directionality.
  shuffle          composition exact, everything else destroyed. Floor anchor,
                   counterpart of the codon-order shuffle.
  conservative k   k=10% of positions replaced by the highest-BLOSUM62 non-self
  radical k        THE SAME k positions replaced by the lowest-BLOSUM62 residue
                   Approximate semantics, and partly circular with BLOSUM. Graded
                   control only.

Scores are mean per-token pseudo-log-likelihood under ESM-2 650M, paired within
protein, using the scorer in crosstalk/plm.py. A leave-one-out bigram model over
the same real sequences is scored on every condition as the trivial baseline, so
each corruption's difficulty is calibrated by how much of it is visible from
dipeptide statistics and chain ends alone.
"""
import argparse, csv, json, sys, time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from crosstalk import glm

AA = "ACDEFGHIKLMNPQRSTVWY"

TOPUP = ["F7YBW8_MESOW_Ding_2023", "PHOT_CHLRE_Chen_2023",
         "AACC1_PSEAI_Dandage_2018", "GRB2_HUMAN_Faure_2021",
         "RNC_ECOLI_Weeks_2023", "GFP_AEQVI_Sarkisyan_2016"]


# ---------------------------------------------------------------- sequences
def load_proteins(max_len):
    """The 28 section-26 genes translated, deduplicated, plus six top-ups."""
    tab = glm.CODON_TABLE
    cds = json.loads((ROOT / "data/cds/dms_cds.json").read_text())
    prots, source = {}, {}
    for k, v in sorted(cds.items()):
        s = v["cds"]
        if len(s) % 3 or not (200 <= len(s) <= 2400) or set(s) - set("ACGT"):
            continue
        p = "".join(tab.get(s[i:i + 3], "X") for i in range(0, len(s) - 2, 3)).rstrip("*")
        if set(p) - set(AA) or len(p) > max_len:
            continue
        if p in prots.values():
            continue
        prots[k] = p
        source[k] = "sec26"
    ref = {r["DMS_id"]: r for r in csv.DictReader(
        (ROOT / "data/proteingym_reference_v1.csv").open())}
    for k in TOPUP:
        p = ref[k]["target_seq"]
        if set(p) - set(AA) or len(p) > max_len or p in prots.values():
            continue
        prots[k], source[k] = p, "topup"
    return prots, source


# ------------------------------------------------------------ interventions
def blosum_partners():
    """For each residue, the highest- and lowest-scoring non-self BLOSUM62 partner."""
    from Bio.Align import substitution_matrices
    m = substitution_matrices.load("BLOSUM62")
    hi, lo = {}, {}
    for a in AA:
        alts = sorted(((float(m[a, b]), b) for b in AA if b != a), reverse=True)
        hi[a], lo[a] = alts[0][1], alts[-1][1]
    return hi, lo


def build_conditions(seq, rng, hi, lo):
    L = len(seq)
    k = max(1, round(0.10 * L))
    pos = rng.choice(L, size=k, replace=False)
    cons, rad = list(seq), list(seq)
    for i in pos:
        cons[i], rad[i] = hi[seq[i]], lo[seq[i]]
    return {
        "real": seq,
        "rotate +1": seq[1:] + seq[:1],
        "rotate mid": seq[L // 2:] + seq[:L // 2],
        "reverse": seq[::-1],
        "shuffle": "".join(np.array(list(seq))[rng.permutation(L)]),
        "conservative 10%": "".join(cons),
        "radical 10%": "".join(rad),
    }


# ------------------------------------------------------- trivial baselines
def bigram_score(seq, counts, alpha=0.5):
    """Mean per-token log p under an add-alpha bigram model with START/END."""
    idx = {a: i for i, a in enumerate(AA)}
    S, E = 20, 21                                   # START, END rows/cols
    tot = 0.0
    prev = S
    for ch in seq + "$":
        cur = E if ch == "$" else idx[ch]
        row = counts[prev] + alpha
        tot += float(np.log(row[cur] / row.sum()))
        prev = cur
    return tot / (len(seq) + 1)


def fit_bigram(seqs):
    idx = {a: i for i, a in enumerate(AA)}
    c = np.zeros((22, 22))
    for s in seqs:
        prev = 20
        for ch in s + "$":
            cur = 21 if ch == "$" else idx[ch]
            c[prev, cur] += 1
            prev = cur
    return c


# --------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="facebook/esm2_t33_650M_UR50D")
    ap.add_argument("--max-len", type=int, default=500)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--out", default="results/protein_semantics.csv")
    args = ap.parse_args()

    prots, source = load_proteins(args.max_len)
    names = sorted(prots, key=lambda n: len(prots[n]))   # short first: bank results early
    print(f"{len(names)} unique proteins, "
          f"{min(map(len, prots.values()))}-{max(map(len, prots.values()))} aa "
          f"({sum(v=='sec26' for v in source.values())} from the section-26 gene set)\n",
          flush=True)

    hi, lo = blosum_partners()
    rng = np.random.default_rng(0)
    seqs = {n: build_conditions(prots[n], rng, hi, lo) for n in names}
    conds = list(seqs[names[0]])

    # sanity: the exact tier must preserve composition exactly
    for n in names:
        base = sorted(prots[n])
        for c in ["rotate +1", "rotate mid", "reverse", "shuffle"]:
            assert sorted(seqs[n][c]) == base, (n, c)
    print("composition preserved exactly for rotate/reverse/shuffle: OK\n", flush=True)

    from crosstalk.plm import ESMScorer, pseudo_likelihood_positions
    sc = ESMScorer(args.model)
    print(f"loaded {args.model} on {sc.device}\n", flush=True)

    cnt = fit_bigram([prots[n] for n in names])
    cachef = ROOT / "results/protein_semantics_cache.json"
    cache = json.loads(cachef.read_text()) if cachef.exists() else {}
    esm, big, perpos = {c: {} for c in conds}, {c: {} for c in conds}, {}
    t0 = time.time()
    done = []
    for i, n in enumerate(names, 1):
        loo = cnt - fit_bigram([prots[n]])          # leave-one-out bigram
        try:
            for c in conds:
                s = seqs[n][c]
                key = f"{n}|{c}"
                if key in cache:
                    lp = np.array(cache[key])
                else:
                    lp = pseudo_likelihood_positions(sc, s, batch_size=args.batch_size)
                    cache[key] = lp.tolist()
                esm[c][n] = float(lp.mean())
                big[c][n] = bigram_score(s, loo)
                if c in ("real", "rotate mid", "rotate +1"):
                    perpos[(n, c)] = lp.tolist()
        except KeyboardInterrupt:
            print(f"\ninterrupted at {n}; {len(done)} proteins banked", flush=True)
            break
        cachef.write_text(json.dumps(cache))
        done.append(n)
        print(f"[{i:2d}/{len(names)}] {n[:34]:34s} L={len(prots[n]):3d} "
              + " ".join(f"{c.split()[0][:4]}{esm[c][n]:+.2f}" for c in conds)
              + f"  ({time.time()-t0:.0f}s)", flush=True)
    names = done
    if not names:
        print("nothing completed"); return

    # ------------------------------------------------------------- reporting
    def table(d, label):
        from scipy import stats
        real = np.array([d["real"][n] for n in names])
        print(f"\n--- {label}: paired against the real protein, {len(names)} proteins ---")
        print(f"{'condition':18s} {'mean delta':>11s} {'95% CI (t)':>21s} {'real higher':>12s}"
              f" {'p (paired t)':>13s}")
        rows = []
        for c in conds[1:]:
            x = np.array([d[c][n] for n in names])
            dd = real - x
            tcrit = float(stats.t.ppf(0.975, len(dd) - 1))
            ci = tcrit * dd.std(ddof=1) / np.sqrt(len(dd))
            pv = float(stats.ttest_rel(real, x).pvalue)
            print(f"{c:18s} {dd.mean():+11.4f} [{dd.mean()-ci:+.4f}, {dd.mean()+ci:+.4f}]"
                  f" {int((dd>0).sum()):8d}/{len(dd)} {pv:13.2e}")
            rows.append(dict(metric=label, condition=c, mean_delta=float(dd.mean()),
                             ci=float(ci), p_paired_t=pv,
                             real_higher=int((dd > 0).sum()), n=len(dd)))
        return rows

    rows = table(esm, "ESM-2 650M PLL") + table(big, "bigram baseline")

    # where each rotation's cost falls along the chain
    for cond, kf in (("rotate +1", lambda L: 1), ("rotate mid", lambda L: L // 2)):
        print(f"\n--- where the {cond} cost falls (ESM-2, per residue) ---")
        bins = {"<=5 from a moved end/seam": [], "6-20": [], ">20 (chain interior)": []}
        for n in names:
            L = len(prots[n]); k = kf(L)
            r = np.array(perpos[(n, "real")]); q = np.array(perpos[(n, cond)])
            for i in range(L):
                j = (i - k) % L                        # rotated index of residue i
                d = min(j, L - 1 - j, abs(j - (L - k)))  # new N-term, C-term, seam
                b = ("<=5 from a moved end/seam" if d <= 5
                     else ("6-20" if d <= 20 else ">20 (chain interior)"))
                bins[b].append(r[i] - q[j])
        for b, v in bins.items():
            v = np.array(v)
            from scipy import stats as _st
            _ci = float(_st.t.ppf(0.975, len(v) - 1)) * v.std(ddof=1) / np.sqrt(len(v))
            print(f"  {b:26s} n={len(v):6d}  mean delta {v.mean():+.4f} +/- {_ci:.4f}")
            rows.append(dict(metric=f"{cond} localisation", condition=b,
                             mean_delta=float(v.mean()), n=len(v)))

    out = ROOT / args.out
    keys = sorted({k for r in rows for k in r})
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)
    per = ROOT / "results/protein_semantics_per_protein.csv"
    with per.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["protein", "source", "len"] + [f"esm:{c}" for c in conds]
                   + [f"bigram:{c}" for c in conds])
        for n in names:
            w.writerow([n, source[n], len(prots[n])]
                       + [f"{esm[c][n]:.5f}" for c in conds]
                       + [f"{big[c][n]:.5f}" for c in conds])
    print(f"\nwrote {out} and {per}")


if __name__ == "__main__":
    main()
