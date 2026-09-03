"""Regulatory benchmark loading, splits, clusters, and metrics.

The data are the *revised* Nucleotide Transformer downstream tasks
(InstaDeepAI/nucleotide_transformer_downstream_tasks_revised), which differ from
the original in two ways that matter here: negatives are real genomic chunks
rather than shuffled sequence, and the test set is a held-out chromosome pair
(chr20, chr21) rather than a random draw. Every record carries its coordinates in
`name` as `chrN:start-end|strand`, so a locus is recoverable and sequences from
one locus can be counted once.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).resolve().parents[1] / "data" / "regulatory"

TASKS = {
    # task            length  n_classes  strand-oriented element?
    "promoter_all":      (300, 2, True),
    "promoter_tata":     (300, 2, True),
    "enhancers":         (400, 2, False),
    "splice_sites_all":  (600, 3, True),
    "H3K4me3":          (1000, 2, False),
}

_COORD = re.compile(r"^(chr[^:]+):(\d+)-(\d+)")


def load_task(task: str) -> pd.DataFrame:
    frames = []
    for split in ("train", "test"):
        d = pd.read_parquet(DATA / f"{task}_{split}.parquet")
        d = d[["sequence", "name", "label"]].copy()
        d["split"] = split
        frames.append(d)
    d = pd.concat(frames, ignore_index=True)
    m = d["name"].str.extract(_COORD)
    d["chrom"] = m[0]
    d["start"] = m[1].astype(int)
    d["end"] = m[2].astype(int)
    d["sequence"] = d["sequence"].str.upper()
    # drop records whose sequence is not pure ACGT-N and exact duplicates of
    # (sequence, label); a duplicated sequence with two labels is dropped whole.
    dup = d.groupby("sequence")["label"].nunique()
    bad = set(dup[dup > 1].index)
    d = d[~d["sequence"].isin(bad)]
    d = d.drop_duplicates(subset=["sequence"]).reset_index(drop=True)
    return d


def locus_clusters(d: pd.DataFrame, gap: int = 20_000) -> np.ndarray:
    """Single-linkage clusters of genomic intervals, merged across gaps < `gap`.

    Two windows from the same promoter, or two ChIP peaks in one gene body, are
    not independent draws. Bootstraps and permutations resample these, not rows.
    """
    cid = np.full(len(d), -1, dtype=int)
    nxt = 0
    for chrom, idx in d.groupby("chrom").groups.items():
        idx = np.asarray(idx)
        order = idx[np.argsort(d["start"].values[idx])]
        last_end = -np.inf
        cur = -1
        for i in order:
            s, e = d["start"].values[i], d["end"].values[i]
            if s - last_end > gap:
                cur = nxt
                nxt += 1
            cid[i] = cur
            last_end = max(last_end, e)
    return cid


_COMP = str.maketrans("ACGTN", "TGCAN")


def revcomp(s: str) -> str:
    return s.translate(_COMP)[::-1]


# ------------------------------------------------------------------ splits

def shipped_split(d: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """The benchmark's own split: chr20 + chr21 held out."""
    return np.where(d["split"].values == "train")[0], np.where(d["split"].values == "test")[0]


def random_split(d: pd.DataFrame, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Pool everything and draw a test set of the shipped size at random.

    Deliberately leaky: same-locus windows land on both sides. Reported next to
    the shipped split so the gap is visible rather than assumed.
    """
    rng = np.random.default_rng(seed)
    n_test = int((d["split"].values == "test").sum())
    perm = rng.permutation(len(d))
    return np.sort(perm[n_test:]), np.sort(perm[:n_test])


def cluster_split(d: pd.DataFrame, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Random split at the level of locus clusters rather than rows.

    Stricter than `random`, looser than `shipped`: a locus is wholly on one side,
    but paralogues and repeats elsewhere in the genome are not controlled.
    """
    cid = locus_clusters(d)
    rng = np.random.default_rng(seed)
    clusters = np.unique(cid)
    rng.shuffle(clusters)
    n_test = int((d["split"].values == "test").sum())
    test_clusters, n = set(), 0
    for c in clusters:
        if n >= n_test:
            break
        test_clusters.add(c)
        n += int((cid == c).sum())
    te = np.where(np.isin(cid, list(test_clusters)))[0]
    tr = np.where(~np.isin(cid, list(test_clusters)))[0]
    return tr, te


def subsample(idx: np.ndarray, labels: np.ndarray, n: int, seed: int = 0) -> np.ndarray:
    """Stratified subsample of a training index, so every feature set sees the
    same rows and a model is never given more data than the baseline."""
    if n >= len(idx):
        return idx
    rng = np.random.default_rng(seed)
    out = []
    lab = labels[idx]
    for c in np.unique(lab):
        pool = idx[lab == c]
        k = int(round(n * len(pool) / len(idx)))
        out.append(rng.choice(pool, size=min(k, len(pool)), replace=False))
    return np.sort(np.concatenate(out))
