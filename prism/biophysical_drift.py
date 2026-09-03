#!/usr/bin/env python3
"""
biophysical_drift.py

Folding-free structural/energetic consequence of mutational signatures, computed
directly from sequence. This is the compute-light companion to structural_drift.py
(ESMFold), for when full structure prediction is not available (e.g. ESMFold
OOMs a commodity GPU). It measures, per (gene, signature), how biophysically
disruptive the signature's mutations are:

  * truncation_rate     fraction of draws introducing a premature stop (nonsense)
  * mean_missense       mean number of amino-acid substitutions per draw
  * mean_blosum         mean BLOSUM62 score of those substitutions (LOWER = more
                        chemically disruptive; a proxy for destabilization)
  * mean_abs_dhydro     mean |Delta Kyte-Doolittle hydrophobicity| at changed
                        residues (buried-core disruption proxy)

None of these require a 3D structure, so they run in seconds on CPU. They give a
signature-resolved, physically interpretable readout of how each mutational
process perturbs the protein, complementing PRISM's embedding-space drift.

USAGE
-----
python biophysical_drift.py \
    --cds_fasta all_runs/CL0072/pairs_cds.fasta \
    --profile_dir profiles \
    --signatures SBS17a,SBS1,SBS7a,SBS13,SBS2 \
    --n_draws 50 --max_genes 12 \
    --output_dir results/biophysical_drift
"""
import argparse
import csv
import os
import random
import sys

import numpy as np
from Bio import SeqIO
from Bio.Seq import Seq

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Kyte-Doolittle hydrophobicity
KD = {'A': 1.8, 'R': -4.5, 'N': -3.5, 'D': -3.5, 'C': 2.5, 'Q': -3.5, 'E': -3.5,
      'G': -0.4, 'H': -3.2, 'I': 4.5, 'L': 3.8, 'K': -3.9, 'M': 1.9, 'F': 2.8,
      'P': -1.6, 'S': -0.8, 'T': -0.7, 'W': -0.9, 'Y': -1.3, 'V': 4.2}


def load_blosum():
    from Bio.Align import substitution_matrices
    return substitution_matrices.load("BLOSUM62")


def translate_cds(nt):
    t = nt[:len(nt) - (len(nt) % 3)]
    return str(Seq(t).translate(to_stop=True))


def mutate_cds(nt, profile):
    s = list(nt)
    for i in range(len(s) - 2):
        ctx = "".join(s[i:i + 3])
        for sig, p in profile.items():
            if len(sig) >= 7 and ctx == sig[0] + sig[2] + sig[6] and random.random() < p:
                s[i + 1] = sig[4]
                break
    return "".join(s)


def load_profile(path):
    d = {}
    with open(path) as f:
        next(f, None)
        for ln in f:
            q = ln.split()
            if len(q) >= 2 and len(q[0]) >= 7 and q[0][1] == '[' and q[0][5] == ']':
                try:
                    d[q[0]] = float(q[1])
                except ValueError:
                    pass
    return d


def load_profiles(profile_dir, wanted):
    wanted = {w.strip() for w in wanted if w.strip()}
    take_all = "ALL" in wanted
    found = {}
    for root, _, files in os.walk(profile_dir):
        for fn in files:
            if not fn.upper().endswith("_PROFILE.TXT"):
                continue
            stem = fn.split("_")[0].split(".")[0]
            if take_all or stem in wanted:
                p = load_profile(os.path.join(root, fn))
                if p:
                    found[stem] = p
    return found


def compare(wt_aa, mut_aa, blosum):
    """Return (n_missense, mean_blosum_of_subs, mean_abs_dhydro) over aligned
    positions up to the common length (signatures are substitution-only, so
    positions correspond 1:1 until a truncation)."""
    n = min(len(wt_aa), len(mut_aa))
    subs = 0
    bl, dh = [], []
    for a, b in zip(wt_aa[:n], mut_aa[:n]):
        if a == b or a == 'X' or b == 'X' or a == '*' or b == '*':
            continue
        subs += 1
        try:
            bl.append(float(blosum[a, b]))
        except (KeyError, IndexError):
            pass
        if a in KD and b in KD:
            dh.append(abs(KD[a] - KD[b]))
    return subs, (np.mean(bl) if bl else np.nan), (np.mean(dh) if dh else np.nan)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cds_fasta", required=True, help="FASTA of coding sequences (DNA).")
    ap.add_argument("--profile_dir", default="profiles")
    ap.add_argument("--signatures", default="SBS17a,SBS1,SBS7a,SBS13,SBS2")
    ap.add_argument("--n_draws", type=int, default=50)
    ap.add_argument("--max_genes", type=int, default=12)
    ap.add_argument("--output_dir", default="results/biophysical_drift")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    os.makedirs(args.output_dir, exist_ok=True)

    blosum = load_blosum()
    profiles = load_profiles(args.profile_dir, args.signatures.split(","))
    if not profiles:
        raise SystemExit("No signature profiles loaded; check --profile_dir/--signatures.")
    genes = [(r.id, str(r.seq).upper().replace(" ", ""))
             for r in SeqIO.parse(args.cds_fasta, "fasta")]
    # keep unique, ACGT, sensible length
    seen, clean = set(), []
    for gid, s in genes:
        if gid in seen or not s or set(s) - set("ACGTN"):
            continue
        seen.add(gid); clean.append((gid, s))
    genes = clean[:args.max_genes]
    print(f"[biophys] {len(genes)} genes x {len(profiles)} signatures x {args.n_draws} draws")

    rng = random.Random(args.seed)
    rows = []
    for gid, cds in genes:
        wt = translate_cds(cds)
        if len(wt) < 10:
            continue
        for sig, prof in profiles.items():
            truncs, miss, bls, dhs = [], [], [], []
            for _ in range(args.n_draws):
                random.seed(rng.randrange(1 << 30))
                mc = mutate_cds(cds, prof)
                ma = translate_cds(mc)
                truncs.append(int(len(ma) < (len(mc) // 3) - 1))
                s, b, h = compare(wt, ma, blosum)
                miss.append(s)
                if not np.isnan(b): bls.append(b)
                if not np.isnan(h): dhs.append(h)
            rows.append([gid, sig, args.n_draws,
                         round(np.mean(truncs), 3), round(np.mean(miss), 2),
                         round(np.mean(bls), 3) if bls else "",
                         round(np.mean(dhs), 3) if dhs else ""])
    with open(os.path.join(args.output_dir, "biophysical_drift_summary.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["gene", "signature", "n_draws", "truncation_rate",
                    "mean_missense", "mean_blosum", "mean_abs_dhydro"])
        w.writerows(rows)

    # per-signature aggregate (the headline)
    import collections
    agg = collections.defaultdict(lambda: collections.defaultdict(list))
    for gid, sig, nd, tr, ms, bl, dh in rows:
        agg[sig]["trunc"].append(tr); agg[sig]["miss"].append(ms)
        if bl != "": agg[sig]["bl"].append(bl)
        if dh != "": agg[sig]["dh"].append(dh)
    print(f"\n{'signature':10}{'trunc':>8}{'missense':>10}{'BLOSUM':>9}{'|dHydro|':>10}")
    summary = []
    for sig in sorted(agg):
        a = agg[sig]
        t, m = np.mean(a["trunc"]), np.mean(a["miss"])
        b = np.mean(a["bl"]) if a["bl"] else float("nan")
        h = np.mean(a["dh"]) if a["dh"] else float("nan")
        print(f"{sig:10}{t:>8.1%}{m:>10.1f}{b:>9.2f}{h:>10.2f}")
        summary.append([sig, round(t, 3), round(m, 2), round(b, 3), round(h, 3)])
    with open(os.path.join(args.output_dir, "biophysical_by_signature.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["signature", "truncation_rate", "mean_missense", "mean_blosum", "mean_abs_dhydro"])
        w.writerows(summary)
    _plot(summary, args.output_dir)
    print(f"\n[biophys] wrote {args.output_dir}/biophysical_by_signature.csv")


def _plot(summary, out_dir):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    sigs = [r[0] for r in summary]
    fig, ax = plt.subplots(1, 3, figsize=(14, 4.2))
    for k, (idx, lbl, col) in enumerate([(1, "truncation rate", "#c0392b"),
                                          (3, "mean BLOSUM62 of substitutions\n(lower = more disruptive)", "#2980b9"),
                                          (4, "mean |Delta hydrophobicity|", "#27ae60")]):
        ax[k].bar(range(len(sigs)), [r[idx] for r in summary], color=col)
        ax[k].set_xticks(range(len(sigs))); ax[k].set_xticklabels(sigs, rotation=30)
        ax[k].set_title(lbl, fontsize=9); ax[k].spines[["top", "right"]].set_visible(False)
    fig.suptitle("Biophysical disruptiveness by mutational signature (folding-free)")
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "biophysical_drift.png"), dpi=200)
    print(f"[biophys] wrote {out_dir}/biophysical_drift.png")


if __name__ == "__main__":
    main()
