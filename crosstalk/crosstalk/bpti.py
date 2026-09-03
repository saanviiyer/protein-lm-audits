"""BPTI against three proteases: the second real landscape.

Beer et al., JACS 2021 (doi:10.1021/jacs.1c08707). 228 single mutants at 12
interface positions of bovine pancreatic trypsin inhibitor, measured as
ddG_bind against three proteases, plus ~13,000 double mutants per protease.

Why this system specifically. Section 14 argues that a single-sequence
likelihood is partner-agnostic, so the sign of its specificity AUC is set by a
marginal asymmetry in how it correlates with each partner -- a property of the
system rather than the model. ParD3's wild type prefers its cognate partner by
a factor of about 5. BPTI's prefers by six to nine orders of magnitude:

    bovine trypsin (cognate)   K_D 1e-14 M
    alpha-chymotrypsin         K_D 1e-8  M
    human mesotrypsin          K_D 1e-5  M

Different fold, different assay, three partners rather than two, and a wildly
different selectivity balance. If the proxy is partner-agnostic here too the
mechanism generalises; if it tracks trypsin specifically it does not.

Getting the data: the source is supplementary file `ja1c08707_si_001.xlsx`.
PMC serves it behind a JavaScript interstitial and the publisher behind a bot
check, so it needs a manual download to `data/raw/`:

    https://pmc.ncbi.nlm.nih.gov/articles/PMC8532158/   (Supplementary material)

Everything below runs as soon as that file exists.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from .landscape import Landscape

# 1-based positions randomised in the study
MUT_POSITIONS = (11, 12, 13, 15, 16, 17, 18, 34, 35, 36, 37, 39)

# UniProt P00974 (BPTI_BOVIN), mature chain 36-93.
# Verified the same way ParD3 was: the paper names the randomised positions as
# T11 G12 P13 K15 A16 R17 I18 V34 Y35 G36 G37 R39, and this sequence gives
# exactly TGPKARIVYGGR at those indices. All twelve match, so the sequence and
# the numbering convention are both right.
BPTI = "RPDFCLEPPYTGPCKARIIRYFYNAKAGLCQTFVYGGCRAKRNNFKSAEDCMRTCGGA"

PARTNERS = ("trypsin", "chymotrypsin", "mesotrypsin")
DEFAULT_XLSX = "data/raw/ja1c08707_si_001.xlsx"


def available(path: str | Path = DEFAULT_XLSX) -> bool:
    return Path(path).exists()


# ddG_bind in kcal/mol; everything real in this file lies within -5.2 to +13.
# The doubles sheet contains exactly one row with ~13,000 in all three columns,
# which is a data-entry artifact rather than a measurement.
MAX_ABS_DDG = 20.0

# The sheets end with aggregate rows -- MAX, MIN, AVG, "MAX DDG" and so on --
# which look like data to a naive reader and would be scored as variants.
_VARIANT_RE = re.compile(r"^[A-Z]\d+[A-Z]$")


def is_variant(code: str) -> bool:
    """True only for well-formed mutation codes such as T11A or T11A_G12S."""
    parts = code.replace(",", "_").replace("/", "_").split("_")
    return bool(parts) and all(_VARIANT_RE.fullmatch(p.strip()) for p in parts)


def _parse_sheet(df):
    """Rows are (mutation, ChT ddG, ChT SD, _, BT ddG, BT SD, _, MT ddG, MT SD).

    Row 0 of each sheet is a sub-header and row 1 is blank, so data starts at 2.
    Mutation labels are quote-wrapped in the doubles sheet ("'T11A_G12A'") and
    use an underscore separator.
    """
    import pandas as pd

    out, dropped = [], 0
    for _, r in df.iloc[2:].iterrows():
        mut = r.iloc[0]
        if not isinstance(mut, str):
            continue
        mut = mut.strip().strip("'\"")
        if not mut or not is_variant(mut):
            continue
        vals = [pd.to_numeric(r.iloc[c], errors="coerce") for c in (4, 1, 7)]  # BT, ChT, MT
        sds = [pd.to_numeric(r.iloc[c], errors="coerce") for c in (5, 2, 8)]
        if any(v != v for v in vals):          # any NaN
            continue
        if any(abs(v) > MAX_ABS_DDG for v in vals):
            dropped += 1
            continue
        out.append((mut, vals, sds))
    return out, dropped


def load_bpti(path: str | Path = DEFAULT_XLSX, include_doubles: bool = True,
              as_positions: bool = True) -> Landscape:
    """BPTI variants against trypsin (target), chymotrypsin and mesotrypsin.

    ddG_bind is reported so that a positive value is destabilising, i.e. weaker
    binding. It is negated here so that larger is better, matching every other
    landscape in this package, then scaled per partner -- the three proteases
    span very different ranges and a shared scale would let one dominate
    max-over-off-targets (the mistake made and fixed on the Absolut landscape).
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Beer et al. JACS 2021 supplementary file; needs a "
            "manual download from https://pmc.ncbi.nlm.nih.gov/articles/PMC8532158/ "
            "because both PMC and the publisher gate it behind bot checks."
        )
    import pandas as pd

    sheets = pd.read_excel(path, sheet_name=None)
    keys = list(sheets)
    recs, dropped = _parse_sheet(sheets[keys[0]])
    n_single = len(recs)
    if include_doubles and len(keys) > 1:
        more, d2 = _parse_sheet(sheets[keys[1]])
        recs += more
        dropped += d2
    if dropped:
        print(f"dropped {dropped} row(s) with |ddG_bind| > {MAX_ABS_DDG} kcal/mol")

    codes = [r[0] for r in recs]
    seqs = [position_string(c) for c in codes] if as_positions else codes
    ddg = np.array([r[1] for r in recs], dtype=float)
    sd = np.array([r[2] for r in recs], dtype=float)

    F = -ddg                                   # larger is better
    lo, hi = F.min(axis=0, keepdims=True), F.max(axis=0, keepdims=True)
    scale = hi - lo
    F = (F - lo) / scale
    noise = np.abs(sd) / scale                 # SDs carried into the same units

    # collapse duplicates: distinct codes can map to the same position string
    seen = {}
    keep = []
    for i, s_ in enumerate(seqs):
        if s_ not in seen:
            seen[s_] = i
            keep.append(i)
    if len(keep) < len(seqs):
        print(f"collapsed {len(seqs) - len(keep)} duplicate position strings")
    seqs = [seqs[i] for i in keep]
    F, noise = F[keep], noise[keep]

    L = Landscape(seqs=seqs, partners=list(PARTNERS), F=F,
                  noise_sd=np.clip(noise, 1e-3, None), name="bpti",
                  wt=WT_POSITIONS if as_positions else None)
    L.n_single = sum(1 for i in keep if i < n_single)
    L.codes = [codes[i] for i in keep]
    return L


def variant_positions(code: str) -> list[tuple[int, str]]:
    """Parse 'T11A' or 'T11A/G12S' into [(position, new_residue), ...]."""
    out = []
    for part in code.replace(",", "/").replace("_", "/").split("/"):
        part = part.strip()
        if len(part) < 3:
            continue
        pos = int("".join(c for c in part[1:-1] if c.isdigit()))
        out.append((pos, part[-1]))
    return out


def variant_sequence(code: str, scaffold: str = BPTI) -> str:
    """Full 58-residue BPTI carrying the mutations in `code`."""
    s = list(scaffold)
    for pos, aa in variant_positions(code):
        s[pos - 1] = aa
    return "".join(s)


def position_string(code: str, scaffold: str = BPTI) -> str:
    """The residues at the 12 randomised positions, as one fixed-length string.

    ParD3 variants are already written this way -- "DKE" is the residues at 61,
    64 and 80 -- which is what lets the environment, the belief model and the
    agents treat a landscape as a fixed-length alphabet problem. Doing the same
    here makes BPTI a drop-in Landscape rather than a special case.
    """
    full = variant_sequence(code, scaffold)
    return "".join(full[p - 1] for p in MUT_POSITIONS)


WT_POSITIONS = "".join(BPTI[p - 1] for p in MUT_POSITIONS)
