"""Reading a developer's own campaign, not just published corpora.

A campaign file is a CSV with at least a ``variant`` column. Rows carrying a
``fitness`` value are what you have measured; rows without one are candidates
you could order. Everything else is preserved as metadata -- assay conditions
especially, because activity measured under different conditions is not
comparable and the tool refuses to pool across them silently.
"""

import os
import re

import pandas as pd

MUT_RE = re.compile(r"^([A-Za-z])(\d+)([A-Za-z*])$")
WT_TOKENS = {"wt", "wild-type", "wildtype", "none", "", "nan"}
#: Columns with a reserved meaning; anything else is treated as a condition.
RESERVED = {"variant", "fitness", "round", "sequence", "n_mut", "seq_len",
            "id", "mean_hydropathy",
            "esm2_wtm", "esm2_per_mut", "blosum62", "hydropathy"}


def parse_variant(field):
    """'S121E/D186H', 'S121E:D186H' or 'WT' -> [(wt, pos, mut), ...] or None."""
    s = str(field).strip()
    if s.lower() in WT_TOKENS:
        return []
    muts = []
    for tok in re.split(r"[/:,;+\s]+", s):
        if not tok:
            continue
        m = MUT_RE.match(tok)
        if not m:
            return None
        muts.append((m.group(1).upper(), int(m.group(2)), m.group(3).upper()))
    return muts


def read_fasta(path):
    seq = []
    for line in open(path):
        if not line.startswith(">"):
            seq.append(line.strip())
    return "".join(seq)


def load_campaign(path, scaffold=None, offset=None):
    """Load a campaign CSV. Returns (DataFrame, dict of notes).

    The frame gains ``muts`` and ``n_mut``. If a scaffold is supplied, the
    numbering offset is recovered by maximising agreement between the stated
    wild-type residues and the sequence, and any variant that still disagrees is
    flagged rather than silently accepted.
    """
    df = pd.read_csv(path)
    if "fitness" not in df.columns:
        df["fitness"] = float("nan")

    if "variant" in df.columns:
        # Variant mode: everything is an edit of one scaffold.
        mode = "variant"
        df["muts"] = df["variant"].apply(parse_variant)
        notes = {"mode": mode, "unparseable": int(df.muts.isna().sum())}
        df = df[df.muts.notna()].copy()
        df["n_mut"] = df.muts.apply(len)
    elif "sequence" in df.columns:
        # Sequence mode: a mining campaign over proteins with no common scaffold,
        # so there are no mutations and the mutation-count machinery does not apply.
        mode = "sequence"
        df = df[df.sequence.notna()].copy()
        if "variant" not in df.columns:
            df["variant"] = df["id"] if "id" in df.columns else [
                f"seq{i}" for i in range(len(df))]
        df["muts"] = None
        df["n_mut"] = float("nan")
        df["seq_len"] = df.sequence.str.len()
        notes = {"mode": mode, "unparseable": 0}
    else:
        raise ValueError(
            f"{path}: need a 'variant' column (edits of one scaffold) or a "
            "'sequence' column (a campaign of distinct proteins)")

    notes["conditions"] = sorted(set(df.columns) - RESERVED - {"muts"})
    notes["n_measured"] = int(df.fitness.notna().sum())
    notes["n_candidates"] = int(df.fitness.isna().sum())

    if scaffold and mode == "variant":
        seq = read_fasta(scaffold) if os.path.exists(scaffold) else scaffold
        claims = [m for ms in df.muts for m in ms]
        if claims:
            if offset is None:
                offset = max(
                    range(-60, 61),
                    key=lambda o: sum(
                        1 for (aa, p, _) in claims
                        if 0 <= p - 1 + o < len(seq) and seq[p - 1 + o] == aa),
                )
            bad = [
                f"{aa}{p}{mt}" for (aa, p, mt) in claims
                if not (0 <= p - 1 + offset < len(seq) and seq[p - 1 + offset] == aa)
            ]
            notes.update(scaffold_len=len(seq), offset=offset,
                         mismatched=sorted(set(bad)))
        notes["sequence"] = seq
    return df.reset_index(drop=True), notes


def enumerate_single_mutants(seq, positions=None, exclude=()):
    """All single substitutions of a sequence, as campaign rows."""
    AA = "ACDEFGHIKLMNPQRSTVWY"
    seen = set(exclude)
    rows = []
    for i, wt in enumerate(seq):
        if positions is not None and (i + 1) not in positions:
            continue
        for mt in AA:
            if mt == wt:
                continue
            name = f"{wt}{i + 1}{mt}"
            if name not in seen:
                rows.append({"variant": name, "fitness": float("nan")})
    return pd.DataFrame(rows)


def infer_condition_columns(df, notes, max_levels=12, max_frac_unique=0.5):
    """Which carried-through columns actually behave like assay conditions.

    A column only stratifies if it varies across measurements but takes few
    enough distinct values to be a condition rather than per-row metadata. A
    free-text notes or date column would otherwise put every variant in its own
    stratum and make the campaign unanalysable.

    Returns (condition_columns, ignored) where ignored is [(column, n_levels)].
    """
    measured = df[df.fitness.notna()]
    cols, ignored = [], []
    for c in notes.get("conditions", []):
        if c not in df.columns:
            continue
        n_levels = measured[c].dropna().nunique()
        if n_levels <= 1:
            continue                       # constant: nothing to conflict with
        if n_levels > max_levels or (len(measured) and
                                     n_levels > max_frac_unique * len(measured)):
            ignored.append((c, n_levels))  # too granular to be a condition
            continue
        cols.append(c)
    return cols, ignored


def condition_key(df, cols):
    """A single label per row identifying its assay condition."""
    if not cols:
        return pd.Series(["(single condition)"] * len(df), index=df.index)
    return df[cols].astype(str).agg(" | ".join, axis=1)


def select_stratum(df, cols, wanted=None):
    """Restrict to one assay condition. Returns (subset, label, dropped_counts).

    Measured rows must match the chosen condition. Candidates match if they
    declare the same condition or declare none at all -- an unassayed variant
    has no condition yet, so it is eligible whichever stratum you pick.
    """
    key = condition_key(df, cols)
    measured = df.fitness.notna()
    counts = key[measured].value_counts()
    if counts.empty:
        return df, None, {}

    if wanted is not None:
        matches = [k for k in counts.index if wanted.lower() in k.lower()]
        if not matches:
            raise ValueError(
                f"no measured variants match condition {wanted!r}. "
                f"Available: {'; '.join(counts.index)}")
        label = matches[0]
    else:
        label = counts.index[0]

    undeclared = df[cols].isna().all(axis=1) if cols else pd.Series(False, index=df.index)
    keep = (key == label) | (~measured & undeclared)
    return df[keep], label, counts.to_dict()
