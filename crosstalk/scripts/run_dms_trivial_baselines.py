#!/usr/bin/env python3
"""Trivial chemistry baselines for the genomic-to-protein transfer result (section 18).

Section 18 reports ESM-2 650M at Spearman +0.466 and Nucleotide Transformer v2 at
-0.013 on 25 DMS assays, and reads the gap as the distance a genomic model has to
travel. That reading is only valid if +0.466 is a hard number to reach. Three
separate results in this project would have read as model successes without a
trivial baseline next to them -- mutation count beating ESM-2 on coverage, one-hot
beating learned embeddings under a hold-out-residue split, BLOSUM62 plus two
scalars beating ESM-2 on SKEMPI -- so the baseline is not optional.

The features are the ones scripts/run_skempi_transfer.py already uses, imported
from there rather than reimplemented: BLOSUM62, the Kyte-Doolittle hydropathy
change and the residue-volume change. Each is scored on exactly the mutations the
transfer run scored, so the comparison is like-for-like.

Two ways of using them are reported, because they answer different questions.
Zero-shot single features are the honest comparison to ESM-2 likelihood, which is
also unsupervised. The leave-one-CLUSTER-out ridge on all five features asks the
different question of how far a fitted lookup table gets; the fold is a cluster
rather than an assay because ProteinGym repeats proteins (BLAT_ECOLX three times,
PTEN and RL401 twice each here), and leaving out one BLAT assay while training on
the other two is not a transfer test.

  ./.venv-glm/bin/python scripts/run_dms_trivial_baselines.py
"""
import argparse, collections, csv, sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))

from run_proxy_ladder import spearman
from run_dms_transfer import load_singles, GAUNTLET
from run_skempi_transfer import trivial_features, zscore
from run_readout_probe import ridge_fit, ridge_pred
from crosstalk.biointerp.battery import _tcrit


class _Muts:
    """Minimal duck type for run_skempi_transfer.trivial_features."""
    def __init__(self, mutations):
        self.mutations = mutations


def cluster_of(dms_id: str) -> str:
    """Protein-level cluster key: everything before the first author-year field.

    ProteinGym ids are PROTEIN_ORGANISM_Author_year, sometimes with an extra
    assay qualifier. The protein and organism prefix is the cluster.
    """
    return "_".join(dms_id.split("_")[:2])


# each feature and the sign that makes "higher = more tolerated" a priori
FEATURES = [
    ("blosum62",            0, +1, "BLOSUM62 substitution score"),
    ("d_hydropathy",        1, +1, "signed Kyte-Doolittle change"),
    ("d_volume",            2, +1, "signed residue-volume change"),
    ("abs_d_hydropathy",    3, -1, "-|hydropathy change|"),
    ("abs_d_volume",        4, -1, "-|volume change|"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--transfer", default="results/dms_transfer.csv")
    ap.add_argument("--lam", type=float, default=300.0)
    ap.add_argument("--out", default="results/dms_trivial_baselines.csv")
    args = ap.parse_args()

    tr = list(csv.DictReader((ROOT / args.transfer).open()))
    ref = {r["DMS_id"]: r for r in csv.DictReader((GAUNTLET / "reference.csv").open())}
    print(f"{len(tr)} assays from {args.transfer}\n")

    assays, X, Y = [], [], []
    for r in tr:
        dms = r["dms_id"]
        tgt = ref[dms]["target_seq"]
        muts = load_singles(dms, tgt)
        feats = trivial_features([_Muts([f"{wt}{p}{mu}" for p, wt, mu, _ in muts])])[0]
        y = np.array([s for _, _, _, s in muts])
        assays.append(dict(dms_id=dms, cluster=cluster_of(dms),
                           n_transfer=int(r["n"]), n_here=len(muts),
                           rho_esm=float(r["rho_esm"]), rho_nt=float(r["rho_nt"]),
                           rho_nt_codon_marginalised=float(r["rho_nt_codon_marginalised"])))
        X.append(feats); Y.append(y)

    dn = [a for a in assays if a["n_here"] != a["n_transfer"]]
    if dn:
        print("mutation counts differ from the transfer run (NT token coverage), "
              "max |diff| = "
              f"{max(abs(a['n_here'] - a['n_transfer']) for a in dn)} of "
              f"{min(a['n_transfer'] for a in dn)}:")
        for a in dn:
            print(f"    {a['dms_id']:40s} here {a['n_here']:5d}  transfer {a['n_transfer']:5d}")
        print()

    # -------------------------------------------------------- zero-shot features
    for name, col, sign, _ in FEATURES:
        for a, x, y in zip(assays, X, Y):
            a[f"rho_{name}"] = spearman(sign * x[:, col], y)

    # ------------------------------------------- leave-one-cluster-out ridge fit
    clusters = sorted({a["cluster"] for a in assays})
    print(f"{len(assays)} assays -> {len(clusters)} protein clusters "
          f"(folds for the fitted baseline)")
    for c in clusters:
        idx = [i for i, a in enumerate(assays) if a["cluster"] == c]
        tr_i = [i for i in range(len(assays)) if i not in idx]
        Xtr = np.vstack([X[i] for i in tr_i])
        ytr = np.concatenate([zscore(Y[i]) for i in tr_i])
        mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-8
        w = ridge_fit((Xtr - mu) / sd, ytr, args.lam)
        for i in idx:
            assays[i]["rho_chem_locco"] = spearman(ridge_pred((X[i] - mu) / sd, w), Y[i])

    # ------------------------------------------------------------------ summary
    cols = ([("rho_esm", "ESM-2 650M likelihood"),
             ("rho_nt", "NT-v2 50M likelihood"),
             ("rho_nt_codon_marginalised", "NT-v2 codon-marginalised")]
            + [(f"rho_{n}", d) for n, _, _, d in FEATURES]
            + [("rho_chem_locco", "chem ridge, leave-one-cluster-out")])

    def summarise(key, per_cluster):
        if per_cluster:
            g = collections.defaultdict(list)
            for a in assays:
                g[a["cluster"]].append(a[key])
            v = np.array([np.mean(x) for x in g.values()])
        else:
            v = np.array([a[key] for a in assays])
        n = len(v); m = v.mean(); se = v.std(ddof=1) / np.sqrt(n)
        t = _tcrit(n - 1)
        return n, m, m - t * se, m + t * se, int((v > 0).sum())

    lines = []
    A = lines.append
    A("=" * 104)
    A("TRIVIAL CHEMISTRY BASELINES ON THE 25 DMS ASSAYS OF FINDINGS SECTION 18")
    A("  mean Spearman across assays, and across protein clusters, with t intervals")
    A("=" * 104)
    A(f"{'scorer':38s} {'per assay (n=%d)' % len(assays):>32s} "
      f"{'per cluster (n=%d)' % len(clusters):>32s}")
    A("-" * 104)
    rows_sum = []
    for key, desc in cols:
        na, ma, la, ha, pa = summarise(key, False)
        nc, mc, lc, hc, pc = summarise(key, True)
        A(f"{desc[:38]:38s} {ma:+8.3f} [{la:+.3f},{ha:+.3f}] {pa:2d}/{na:<2d} "
          f"{mc:+8.3f} [{lc:+.3f},{hc:+.3f}] {pc:2d}/{nc:<2d}")
        rows_sum.append(dict(scorer=desc, key=key, n_assays=na, mean_assay=ma,
                             lo_assay=la, hi_assay=ha, positive_assays=pa,
                             n_clusters=nc, mean_cluster=mc, lo_cluster=lc,
                             hi_cluster=hc, positive_clusters=pc))
    A("-" * 104)

    esm = np.array([a["rho_esm"] for a in assays])
    nt = np.array([a["rho_nt"] for a in assays])
    n = len(esm); t = _tcrit(n - 1)

    def paired(a, b):
        d = a - b
        se = d.std(ddof=1) / np.sqrt(len(d))
        return (f"{d.mean():+.3f} [{d.mean()-t*se:+.3f}, {d.mean()+t*se:+.3f}], "
                f"higher in {int((d>0).sum())}/{len(d)}")

    A("")
    A("Two baselines, because they answer different questions.")
    A("")
    b0 = np.array([a["rho_blosum62"] for a in assays])
    A(f"  ZERO-SHOT, the like-for-like comparison to a likelihood. BLOSUM62 alone,")
    A(f"  no fitting and no model of any kind, reaches {b0.mean():+.3f}, which is "
      f"{100*b0.mean()/esm.mean():.0f}% of ESM-2's")
    A(f"  {esm.mean():+.3f} and positive on {int((b0>0).sum())}/{n} assays. ESM-2 minus "
      f"BLOSUM62, paired: {paired(esm, b0)}.")
    A("")
    b1 = np.array([a["rho_chem_locco"] for a in assays])
    A(f"  FITTED, five chemistry scalars with weights learned on other protein")
    A(f"  clusters and never on the test protein: {b1.mean():+.3f}, "
      f"{100*b1.mean()/esm.mean():.0f}% of ESM-2. ESM-2 minus")
    A(f"  that, paired: {paired(esm, b1)}.")
    A("")
    A(f"So roughly half to two thirds of the DMS signal usually credited to a protein")
    A(f"language model is generic amino-acid chemistry that a lookup table already")
    A(f"has. ESM-2's genuinely learned margin over the fitted chemistry baseline is")
    A(f"{(esm - b1).mean():+.3f}, not {esm.mean():+.3f}.")
    A("")
    A(f"The bar a genomic model has to clear is therefore {b0.mean():+.3f} zero-shot, "
      f"not {esm.mean():+.3f}.")
    A(f"NT-v2 is at {nt.mean():+.3f} and does not clear it: BLOSUM62 minus NT-v2, "
      f"paired, {paired(b0, nt)}.")
    A(f"Moving the bar does not rescue the genomic rung; it relocates the headline. The")
    A(f"honest statement of section 18 is that a genomic LM scores below a substitution")
    A(f"matrix, which is a stronger and cheaper claim than scoring below ESM-2.")
    A("=" * 104)
    text = "\n".join(lines)
    print("\n" + text)

    dest = ROOT / args.out
    keys = sorted({k for a in assays for k in a})
    with dest.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(assays)
    with (ROOT / args.out.replace(".csv", "_summary.csv")).open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows_sum[0])); w.writeheader(); w.writerows(rows_sum)
    (ROOT / args.out.replace(".csv", ".txt")).write_text(text + "\n")
    print(f"\nwrote {dest}, its _summary.csv and its .txt")


if __name__ == "__main__":
    main()
