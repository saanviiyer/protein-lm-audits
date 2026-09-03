"""ProteinGym deep-mutational-scanning assays as replayable campaigns.

DMS assays are where offline replay is actually sound: a near-complete single
mutant scan means almost every action a selection policy could take has a
recorded outcome, so the counterfactual is real rather than an artefact of
somebody else's earlier selection. That is the property published engineering
campaigns lack, and the reason this validation exists.
"""

import os
import re

import pandas as pd

MUT_RE = re.compile(r"^([A-Z])(\d+)([A-Z])$")
BASE = "https://huggingface.co/datasets/OATML-Markslab/ProteinGym_v0.1/resolve/main"


def parse_mutant(field):
    """'M1H' or 'M1H:A2G' -> [('M',1,'H'), ...]; None if unparseable."""
    muts = []
    for tok in str(field).split(":"):
        m = MUT_RE.match(tok.strip())
        if not m:
            return None
        muts.append((m.group(1), int(m.group(2)), m.group(3)))
    return muts


def fetch_assay(dms_id, out_dir):
    """Download one assay CSV if not already present. Returns its path."""
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{dms_id}.csv")
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        url = f"{BASE}/ProteinGym_substitutions/{dms_id}.csv"
        if os.system(f'curl -sL --max-time 120 -o "{path}" "{url}"') != 0:
            raise RuntimeError(f"download failed: {dms_id}")
    return path


def load_assay(dms_id, target_seq, assay_dir, singles_only=True):
    """Return (records, y) with mutations validated against the target sequence.

    Any mutant whose stated wild-type residue disagrees with ``target_seq`` is
    dropped rather than silently coerced -- a numbering mismatch that is quietly
    absorbed would corrupt every score downstream.
    """
    df = pd.read_csv(fetch_assay(dms_id, assay_dir))
    records, y, dropped = [], [], 0
    for mut_field, score in zip(df["mutant"], df["DMS_score"]):
        muts = parse_mutant(mut_field)
        if muts is None or (singles_only and len(muts) != 1):
            dropped += 1
            continue
        if any(p - 1 >= len(target_seq) or target_seq[p - 1] != wt for wt, p, _ in muts):
            dropped += 1
            continue
        records.append({"muts": muts})
        y.append(float(score))
    return records, pd.Series(y).to_numpy(), dropped


def select_assays(reference_csv, max_len=500, min_singles=1200, ids=None):
    ref = pd.read_csv(reference_csv)
    if ids is not None:
        ref = ref[ref.DMS_id.isin(ids)]
    else:
        ref = ref[(ref.seq_len <= max_len) & (ref.DMS_number_single_mutants >= min_singles)]
    return ref[["DMS_id", "target_seq", "seq_len", "DMS_number_single_mutants",
                "selection_type", "molecule_name"]].reset_index(drop=True)
