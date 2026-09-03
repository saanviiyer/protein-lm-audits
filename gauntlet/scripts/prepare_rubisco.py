#!/usr/bin/env python3
"""Convert the RuBisCO deep mutational scan into a campaign file plus a WT FASTA.

Source: Prywes, Michaels, Payne, Niyogi, Savage et al., "A map of the rubisco
biochemical landscape", Nature 638:823-828 (2025), doi:10.1038/s41586-024-08455-0.
Supplementary Data 2 (CC BY 4.0), 2.4 MB CSV. Author Correction:
doi:10.1038/s41586-025-08707-7.

    curl -L -o data/rubisco/suppdata2_dms.csv \
      'https://static-content.springer.com/esm/art%3A10.1038%2Fs41586-024-08455-0/MediaObjects/41586_2024_8455_MOESM4_ESM.csv'

    python scripts/prepare_rubisco.py

Why this dataset is in the project: it fixes the three defects of the PETML
corpus that Phase 1 had to work around.

1. Every variant is a SINGLE mutant, so mutation count is constant and the
   campaign-progression confound that dominated the PETase result (rho +0.007
   raw, +0.196 partialled) cannot operate here at all.
2. One assay, one lab, one protocol -- no within-study pooling.
3. Two orthogonal kinetic axes (Vmax and K_C) rather than one, so a proxy that
   tracks one and not the other is distinguishable from a proxy that works.

Form II rubisco from Rhodospirillum rubrum, assayed by enrichment in a
rubisco-dependent E. coli growth selection titrated across CO2 concentrations.

Columns in the source, and what they mean:

    mutant, position, WTresidue, residue   the substitution, 1-based
    rep1..rep9                             per-replicate enrichment
    Fitness                                growth-selection fitness
    Vmax_median, Vmax_qbcov                turnover, and its spread
    Km_median, Km_qbcov                    K_C (CO2 half-saturation), and spread

``qbcov`` is a quantile-based coefficient of variation -- a per-variant error
bar, not a p-value. It is heavy-tailed (Km_qbcov reaches 24.9 against a median
of 0.29), so ANY ranking analysis has to either filter on it or weight by it.
Reporting a correlation over all 8,760 rows unfiltered mixes tightly measured
variants with ones that carry no information, which is its own version of the
mistake this project is about. ``--max-qbcov`` is the knob; it defaults to off
so that the choice is always explicit in the command line rather than hidden in
a default.

WT SEQUENCE, and the one trap in this dataset: the assayed construct is NOT
byte-identical to UniProt P04718. At position 91 the scan's wild type is D and
UniProt has H (and H appears as one of the 19 substitutions at that position, so
this is a real construct difference, not a bookkeeping slip). Score against the
sequence this script emits -- reconstructed from the scan's own WTresidue column
-- not against a fresh UniProt pull, or every score at position 91 is wrong.

Positions 2, 465 and 466 were not assayed and are filled from P04718 so the
FASTA is a complete 466-mer; they are recorded in the header. Coverage is 8,760
of the 465 x 19 = 8,835 possible single substitutions (99.2%), with 447 of 462
assayed positions carrying all 19.
"""

import argparse
import os
import re

import pandas as pd

MUT_RE = re.compile(r"^([A-Z])(\d+)([A-Z])$")

# UniProt P04718 (RBL2_RHORU), used ONLY to fill the three unassayed positions.
P04718_FILL = {1: "M", 2: "D", 465: "P", 466: "A"}
SEQ_LEN = 466


def build_wt_sequence(d):
    """Rebuild the assayed construct from the scan's own WTresidue column."""
    wt = d.drop_duplicates("position").set_index("position").WTresidue
    conflicts = d.groupby("position").WTresidue.nunique()
    if (conflicts > 1).any():
        bad = conflicts[conflicts > 1].index.tolist()
        raise ValueError(f"positions disagree about their wild-type residue: {bad}")

    seq, filled = [], []
    for i in range(1, SEQ_LEN + 1):
        if i in wt.index:
            seq.append(wt[i])
        elif i in P04718_FILL:
            seq.append(P04718_FILL[i])
            filled.append(i)
        else:
            raise ValueError(f"position {i} is neither assayed nor fillable")
    return "".join(seq), filled


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/rubisco/suppdata2_dms.csv")
    ap.add_argument("--out", default="data/rubisco/rubisco_campaign.csv")
    ap.add_argument("--fasta", default="data/rubisco/rubisco_wt.fasta")
    ap.add_argument(
        "--max-qbcov",
        type=float,
        default=None,
        help="drop variants whose Vmax_qbcov or Km_qbcov exceeds this; "
             "off by default so the filter is always stated explicitly",
    )
    args = ap.parse_args()

    d = pd.read_csv(args.src, index_col=0)

    # The source is self-describing twice over -- as a mutant string and as three
    # columns. Check they agree before trusting either.
    parsed = d.mutant.str.extract(MUT_RE)
    if parsed.isna().any().any():
        bad = d.mutant[parsed.isna().any(axis=1)].tolist()[:5]
        raise ValueError(f"mutant strings not of the form A123C: {bad}")
    parsed.columns = ["wt", "pos", "mut"]
    if not (
        (parsed.wt == d.WTresidue.values).all()
        and (parsed.pos.astype(int) == d.position.values).all()
        and (parsed.mut == d.residue.values).all()
    ):
        raise ValueError("mutant strings disagree with the position/residue columns")
    if (d.residue == d.WTresidue).any():
        raise ValueError("synonymous rows present; expected substitutions only")
    if d.mutant.duplicated().any():
        raise ValueError("duplicate mutant rows")

    seq, filled = build_wt_sequence(d)

    n_before = len(d)
    if args.max_qbcov is not None:
        d = d[(d.Vmax_qbcov <= args.max_qbcov) & (d.Km_qbcov <= args.max_qbcov)]

    out = pd.DataFrame({
        "variant": d.mutant,
        "position": d.position,
        "wt_residue": d.WTresidue,
        "mut_residue": d.residue,
        "n_mut": 1,  # constant by construction -- the point of this dataset
        "fitness": d.Fitness,
        "vmax": d.Vmax_median,
        "vmax_err": d.Vmax_qbcov,
        "kc": d.Km_median,
        "kc_err": d.Km_qbcov,
        "study": "Prywes2025",
        "assay": "rubisco_dependent_ecoli_selection",
    }).sort_values(["position", "mut_residue"])

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    out.to_csv(args.out, index=False)

    header = (
        f"rubisco_R_rubrum_formII assayed_construct len={len(seq)} "
        f"source=Prywes2025_SupplementaryData2 "
        f"unassayed_filled_from_P04718={','.join(map(str, filled))} "
        f"note=position_91_is_D_here_and_H_in_P04718"
    )
    with open(args.fasta, "w") as fh:
        fh.write(f">{header}\n")
        for i in range(0, len(seq), 60):
            fh.write(seq[i:i + 60] + "\n")

    kept = f"{len(out)} of {n_before}" if args.max_qbcov is not None else str(len(out))
    print(f"wrote {args.out}: {kept} variants, {out.position.nunique()} positions")
    print(f"wrote {args.fasta}: {len(seq)} aa, filled {filled} from P04718")


if __name__ == "__main__":
    main()
