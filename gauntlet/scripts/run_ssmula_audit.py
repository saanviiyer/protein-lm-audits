#!/usr/bin/env python3
"""Audit ESM-2 zero-shot against the SSMuLA combinatorial landscapes.

    python scripts/run_ssmula_audit.py --out results

Third arm of the cross-corpus audit, after PETML (``run_petase_audit.py``) and
the rubisco DMS (``run_rubisco_audit.py``). Same scorer, same protocol.

Source: Li, Yang, Johnston, Gursoy, Yue, Arnold, "Evaluation of machine
learning-assisted directed evolution across diverse combinatorial landscapes",
Cell Systems 16(9):101387 (2025). Data from Zenodo 10.5281/zenodo.13910506,
``data.zip``, CC BY 4.0. The SSMuLA source itself is GPL-3.0 and is NOT vendored
here; the per-landscape mutated positions and wild-type residues below were read
off ``SSMuLA/landscape_global.py`` as facts and are restated, not copied.

WHY THIS IS THE TIEBREAK. PETML said the proxy loses to a mutation count, but
its corpus is aggregated literature and the count is really campaign
progression. The rubisco scan said the proxy works, but about half of that came
from telling dead enzymes from live ones and much of the rest from positional
conservation. These landscapes have neither defect: each is a near-complete
site-saturation library at 3-4 positions from ONE lab, so there is no
publication history to read and no cross-assay pooling.

They also restore the baseline rubisco could not test. Every rubisco variant was
a single mutant, so mutation count was constant; here it is the Hamming distance
from the wild type and varies from 0 to 4. Its meaning is different from PETML's
-- distance from a functional starting point rather than campaign progression --
so the sign is expected to run the other way, and that contrast is the point.

Sixteen landscapes, thirteen enzyme-activity and three binding, are reported
separately as well as pooled. The enzyme/binding split is the comparison the
project's central claim is about; pooling them would hide it.

Formats differ per landscape and are normalised here: GB1 uses ``Variants`` /
``Fitness``, T7 and TEV use ``Sequences`` / ``Fitness Mean``, TrpB and ParD use
``AAs`` / ``fitness``, and DHFR is CODON-level (9 nucleotides for its 3 sites),
so it is translated and collapsed to amino-acid combinations by median. Stop
codons are dropped everywhere.
"""

import argparse
import io
import json
import os
import sys
import zipfile

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from gauntlet import proxies  # noqa: E402

# Read off SSMuLA/landscape_global.py (GPL-3.0) as facts; positions are 1-based.
LANDSCAPES = {
    "DHFR":   ([26, 27, 28], "ADL", "Enzyme activity"),
    "GB1":    ([39, 40, 41, 54], "VDGV", "Binding"),
    "ParD2":  ([61, 64, 80], "ILK", "Binding"),
    "ParD3":  ([61, 64, 80], "DKE", "Binding"),
    "T7":     ([748, 756, 758], "NRQ", "Enzyme activity"),
    "TEV":    ([146, 148, 167, 170], "TDHS", "Enzyme activity"),
    "TrpB3A": ([104, 105, 106], "AET", "Enzyme activity"),
    "TrpB3B": ([105, 106, 107], "ETG", "Enzyme activity"),
    "TrpB3C": ([106, 107, 108], "TGA", "Enzyme activity"),
    "TrpB3D": ([117, 118, 119], "TAA", "Enzyme activity"),
    "TrpB3E": ([184, 185, 186], "FGS", "Enzyme activity"),
    "TrpB3F": ([162, 166, 301], "LIY", "Enzyme activity"),
    "TrpB3G": ([227, 228, 301], "VSY", "Enzyme activity"),
    "TrpB3H": ([228, 230, 231], "SGS", "Enzyme activity"),
    "TrpB3I": ([182, 183, 184], "YVF", "Enzyme activity"),
    "TrpB4":  ([183, 184, 227, 228], "VFVS", "Enzyme activity"),
}

# DHFR ships a partial nucleotide CDS rather than a protein FASTA, so the
# wild-type protein comes from UniProt. Verified: P0ABQ4 residues 26-28 are ADL,
# and the archive's nucleotide fragment translates to this sequence's prefix.
DHFR_UNIPROT = "P0ABQ4"

COMBO_COL = {"GB1": ("Variants", "Fitness"), "T7": ("Sequences", "Fitness Mean"),
             "TEV": ("Sequences", "Fitness Mean"), "DHFR": ("seq", "fitness")}
PROXIES = ["esm2_wtm", "wt_logp", "blosum62", "hydropathy", "n_mut"]
TOPK_FRACS = [0.01, 0.05, 0.10]
CODON_TABLE = None


def translate(nt):
    global CODON_TABLE
    if CODON_TABLE is None:
        from Bio.Data.CodonTable import standard_dna_table
        CODON_TABLE = dict(standard_dna_table.forward_table)
        for s in standard_dna_table.stop_codons:
            CODON_TABLE[s] = "*"
    return "".join(CODON_TABLE.get(nt[i:i + 3], "*") for i in range(0, len(nt) - 2, 3))


def system_of(name):
    return "TrpB" if name.startswith("TrpB") else name


def load_landscape(z, name, uniprot_seqs):
    positions, wt_combo, kind = LANDSCAPES[name]
    combo_col, fit_col = COMBO_COL.get(name, ("AAs", "fitness"))
    sysname = system_of(name)
    d = pd.read_csv(io.BytesIO(
        z.read(f"data/{sysname}/fitness_landscape/{name}.csv")))
    d = d[[combo_col, fit_col]].rename(columns={combo_col: "combo", fit_col: "fitness"})
    d["combo"] = d["combo"].astype(str)

    if name == "DHFR":
        d["combo"] = d["combo"].map(translate)
        # codon-level: collapse synonymous codons to one amino-acid variant
        d = d.groupby("combo", as_index=False).fitness.median()

    d = d[~d.combo.str.contains(r"\*", regex=True)]
    d = d[d.combo.str.len() == len(positions)]
    d = d[d.fitness.notna()]

    if name == "DHFR":
        wt_seq = uniprot_seqs[DHFR_UNIPROT]
    else:
        fa = z.read(f"data/{sysname}/{sysname}.fasta").decode()
        wt_seq = "".join(l.strip() for l in fa.split("\n") if not l.startswith(">"))

    got = "".join(wt_seq[p - 1] for p in positions)
    if got != wt_combo:
        raise ValueError(f"{name}: wild type at {positions} is {got}, expected {wt_combo}")
    return d.reset_index(drop=True), wt_seq, positions, wt_combo, kind


def topk_utility(proxy, y, frac):
    k = max(1, int(round(frac * len(y))))
    rand, best = y.mean(), np.sort(y)[-k:].mean()
    got = y[np.argsort(-proxy)[:k]].mean()
    return np.nan if best == rand else float((got - rand) / (best - rand))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", default="data/ssmula/data.zip")
    ap.add_argument("--out", default="results")
    ap.add_argument("--model", default=None)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--cache", default=None)
    ap.add_argument("--scorer", default="esm2", choices=["esm2", "esmc"],
                    help="ESM-C is a different architecture from a different lab, "
                         "so agreement between them is evidence about the class "
                         "of scorer rather than about one checkpoint")
    ap.add_argument("--tag", default=None, help="suffix for output files")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    if args.model is None:
        args.model = ("facebook/esm2_t33_650M_UR50D" if args.scorer == "esm2"
                      else "esmc_300m")
    if args.cache is None:
        args.cache = (f"data/ssmula/{'esm2_650M' if args.scorer == 'esm2' else 'esmc_300m'}"
                      "_logprobs.json")
    if args.tag is None:
        args.tag = "" if args.scorer == "esm2" else "_esmc"

    z = zipfile.ZipFile(args.zip)

    uniprot = {}
    if os.path.exists("data/ssmula/dhfr_P0ABQ4.fasta"):
        txt = open("data/ssmula/dhfr_P0ABQ4.fasta").read()
    else:
        import urllib.request
        txt = urllib.request.urlopen(
            f"https://rest.uniprot.org/uniprotkb/{DHFR_UNIPROT}.fasta").read().decode()
        with open("data/ssmula/dhfr_P0ABQ4.fasta", "w") as fh:
            fh.write(txt)
    uniprot[DHFR_UNIPROT] = "".join(
        l.strip() for l in txt.split("\n") if l and not l.startswith(">"))

    loaded = {}
    for name in LANDSCAPES:
        d, wt_seq, positions, wt_combo, kind = load_landscape(z, name, uniprot)
        loaded[name] = (d, wt_seq, positions, wt_combo, kind)
        print(f"  {name:8s} {kind:15s} n={len(d):7d}  sites={positions}  wt={wt_combo}")

    cache = json.load(open(args.cache)) if os.path.exists(args.cache) else {}
    need = [n for n in loaded if n not in cache]
    if need:
        print(f"\nloading {args.model} ...")
        cls = proxies.ESM2Marginals if args.scorer == "esm2" else proxies.ESMCMarginals
        scorer = cls(args.model, batch_size=args.batch_size)
        print(f"  device={scorer.device}")
        for name in need:
            _, wt_seq, positions, _, _ = loaded[name]
            tbl = scorer.logprobs_at(wt_seq, [p - 1 for p in positions])
            cache[name] = {str(p): [float(x) for x in tbl[p - 1]] for p in positions}
            print(f"  scored {name} ({len(positions)} positions)", flush=True)
        json.dump(cache, open(args.cache, "w"))
        print(f"  cached to {args.cache}")

    aa_index = {a: i for i, a in enumerate(proxies.ESM2Marginals.AA_ORDER)}
    rows, util_rows, frames = [], [], []

    for name, (d, wt_seq, positions, wt_combo, kind) in loaded.items():
        tbl = {int(p): np.asarray(v) for p, v in cache[name].items()}
        combos = d.combo.to_numpy()
        chars = np.array([list(c) for c in combos])

        esm = np.zeros(len(d))
        wt_lp = sum(float(tbl[p][aa_index[w]]) for p, w in zip(positions, wt_combo))
        nmut = np.zeros(len(d), int)
        muts_per_row = [[] for _ in range(len(d))]
        for j, (p, w) in enumerate(zip(positions, wt_combo)):
            col = chars[:, j]
            lp = tbl[p]
            base = float(lp[aa_index[w]])
            vals = np.array([float(lp[aa_index[a]]) - base if a != w else 0.0 for a in col])
            esm += vals
            diff = col != w
            nmut += diff
            for i in np.nonzero(diff)[0]:
                muts_per_row[i].append((w, p, col[i]))

        d = d.assign(
            landscape=name, kind=kind, n_mut=nmut, esm2_wtm=esm,
            wt_logp=wt_lp,  # constant within a landscape: see note below
            blosum62=[proxies.blosum_score(m) for m in muts_per_row],
            hydropathy=[proxies.hydropathy_shift(m) for m in muts_per_row],
        )
        frames.append(d)

        y = d.fitness.to_numpy()
        for p in PROXIES:
            x = d[p].to_numpy(float)
            if len(np.unique(x)) < 2:
                continue  # wt_logp is constant here -- the sites never change
            rows.append({"landscape": name, "kind": kind, "n": len(d), "proxy": p,
                         "rho": stats.spearmanr(x, y).statistic})
            for f in TOPK_FRACS:
                util_rows.append({"landscape": name, "kind": kind, "proxy": p,
                                  "frac": f, "utility": topk_utility(x, y, f)})

    all_df = pd.concat(frames, ignore_index=True)
    all_df.to_csv(os.path.join(args.out, f"ssmula_scored_variants{args.tag}.csv"), index=False)
    res, util = pd.DataFrame(rows), pd.DataFrame(util_rows)
    res.to_csv(os.path.join(args.out, f"ssmula_audit{args.tag}.csv"), index=False)
    util.to_csv(os.path.join(args.out, f"ssmula_topk{args.tag}.csv"), index=False)

    # wt_logp cannot vary inside a landscape -- the mutated sites are fixed, so
    # positional conservation is a constant. The rubisco confound is structurally
    # untestable here, which is worth stating rather than silently omitting.
    live = [p for p in PROXIES if p in set(res.proxy)]

    print("\n" + "=" * 78)
    print("CONTROL 3 -- Spearman against measured fitness, per landscape")
    print("=" * 78)
    piv = res.pivot_table(index=["kind", "landscape"], columns="proxy", values="rho")
    print(piv.reindex(columns=live).round(3).to_string())

    print("\n" + "=" * 78)
    print("POOLED (mean over landscapes, weighted by n)")
    print("=" * 78)
    for kind in ["Enzyme activity", "Binding"]:
        blk = res[res.kind == kind]
        if blk.empty:
            continue
        w = blk.pivot_table(index="landscape", columns="proxy", values="rho")
        n = blk.groupby("landscape").n.first()
        wm = (w.mul(n, axis=0).sum() / n.sum()).reindex(live)
        print(f"\n{kind}  ({blk.landscape.nunique()} landscapes, {n.sum()} variants)")
        print(wm.round(3).to_string())

    print("\n" + "=" * 78)
    print("ELITE REGIME -- normalised top-k utility (1 = optimal, 0 = random)")
    print("=" * 78)
    for kind in ["Enzyme activity", "Binding"]:
        blk = util[util.kind == kind]
        if blk.empty:
            continue
        print(f"\n{kind}")
        print(blk.pivot_table(index="proxy", columns="frac",
                              values="utility").reindex(live).round(3).to_string())

    print(f"\nwrote {args.out}/ssmula_audit{args.tag}.csv and {args.out}/ssmula_topk{args.tag}.csv")


if __name__ == "__main__":
    main()
