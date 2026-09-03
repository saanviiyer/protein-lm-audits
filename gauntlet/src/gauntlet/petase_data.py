"""Load the PETML activity/Tm corpus into one table joined to sequences.

The corpus is 33 per-paper spreadsheets from github.com/jafetgado/PETML. Each
sheet is one publication, so assay conditions (substrate crystallinity, solids
loading, temperature, pH) are constant *within* a sheet and incomparable
*across* sheets -- see Wei et al., Nat Commun 16:4684 (2025). Every downstream
correlation is therefore computed within study, never on the pooled table.
"""

import glob
import os
import re

import pandas as pd

VARIANT_RE = re.compile(r"^([A-Z])(\d+)([A-Z])$")


def load_sequences(fasta_path):
    seqs, name, buf = {}, None, []
    for line in open(fasta_path):
        line = line.strip()
        if line.startswith(">"):
            if name:
                seqs[name] = "".join(buf)
            name, buf = line[1:].strip(), []
        elif line:
            buf.append(line)
    if name:
        seqs[name] = "".join(buf)
    return seqs


def parse_variant(protein_name):
    """'IsPETase_S214H/I168R' -> ('IsPETase', [('S',214,'H'), ('I',168,'R')])."""
    if "_" not in protein_name:
        return protein_name, []
    scaffold, _, mutfield = protein_name.partition("_")
    muts = []
    for tok in mutfield.split("/"):
        m = VARIANT_RE.match(tok.strip())
        if m:
            muts.append((m.group(1), int(m.group(2)), m.group(3)))
        else:
            return protein_name, None  # unparseable label, e.g. a named chimera
    return scaffold, muts


def load_corpus(petml_dir):
    """Return one row per (study, protein) with sequence, activity and Tm."""
    seqs = load_sequences(os.path.join(petml_dir, "sequences.fasta"))
    rows = []
    for path in sorted(glob.glob(os.path.join(petml_dir, "*.xlsx"))):
        study = os.path.splitext(os.path.basename(path))[0]
        df = pd.read_excel(path)
        if "Protein" not in df.columns:
            continue
        for _, r in df.iterrows():
            name = str(r["Protein"]).strip()
            scaffold, muts = parse_variant(name)
            rows.append(
                {
                    "study": study,
                    "protein": name,
                    "scaffold": scaffold,
                    "n_mut": len(muts) if muts is not None else None,
                    "muts": muts,
                    "sequence": seqs.get(name),
                    "logActivity": r.get("logActivity"),
                    "Tm": r.get("Tm"),
                    "dTm": r.get("dTm"),
                }
            )
    return pd.DataFrame(rows)


def resolve_wildtypes(df, seqs):
    """Map each scaffold to its unmutated sequence and the numbering offset.

    PETase mutation numbering is conventionally relative to the mature protein
    while the deposited sequences carry the signal peptide, so the offset is a
    per-scaffold constant we recover by maximising agreement between the
    reported wild-type letters and the sequence.
    """
    out = {}
    for scaffold, grp in df.groupby("scaffold"):
        wt = seqs.get(scaffold)
        if wt is None:
            cand = grp[grp["sequence"].notna() & grp["n_mut"].notna()]
            if cand.empty:
                continue
            wt = cand.sort_values("n_mut").iloc[0]["sequence"]
        claims = [m for ms in grp["muts"] if ms for m in ms]
        if not claims:
            continue
        best_off, best_hits = None, -1
        for off in range(-60, 61):
            hits = sum(
                1
                for (aa, pos, _) in claims
                if 0 <= pos - 1 + off < len(wt) and wt[pos - 1 + off] == aa
            )
            if hits > best_hits:
                best_off, best_hits = off, hits
        out[scaffold] = {
            "wt": wt,
            "offset": best_off,
            "match_rate": best_hits / len(claims),
            "n_claims": len(claims),
        }
    return out
