"""Trivial feature sets for the regulatory arm: the bar a genomic LM has to clear.

Three families, deliberately ordered by how little they know:

  composition   GC fraction alone (one number), then k-mer frequencies. These are
                position-blind: a shuffled sequence has identical features.
  position      a full-length one-hot, and a position weight matrix. These are
                composition-blind only in the sense that they also see order.
  PWM           per-class positional nucleotide frequencies with a pseudocount.
                No fitting beyond counting, no hyperparameter, 4*L*K numbers.

The PWM is the genomic BLOSUM62. Section 18's whole point is that a model number
without a trivial number beside it cannot be read.
"""
from __future__ import annotations

import numpy as np

BASES = "ACGT"
_CODE = np.full(256, 4, dtype=np.int8)
for i, b in enumerate(BASES):
    _CODE[ord(b)] = i


def encode(seqs) -> np.ndarray:
    """(n, L) int8 with 4 = N/other. Assumes equal length, which these tasks are."""
    L = len(seqs[0])
    arr = np.frombuffer("".join(seqs).encode(), dtype=np.uint8).reshape(len(seqs), L)
    return _CODE[arr]


def gc_features(seqs) -> np.ndarray:
    X = encode(seqs)
    gc = ((X == 1) | (X == 2)).mean(1)
    cpg = np.zeros(len(seqs))
    c, g = (X[:, :-1] == 1), (X[:, 1:] == 2)
    cpg = (c & g).mean(1)
    return np.stack([gc, cpg], 1)


def kmer_features(seqs, k: int) -> np.ndarray:
    X = encode(seqs).astype(np.int32)
    n, L = X.shape
    valid = X < 4
    idx = np.zeros((n, L - k + 1), dtype=np.int64)
    ok = np.ones((n, L - k + 1), dtype=bool)
    for j in range(k):
        idx = idx * 4 + np.where(valid[:, j:L - k + 1 + j], X[:, j:L - k + 1 + j], 0)
        ok &= valid[:, j:L - k + 1 + j]
    out = np.zeros((n, 4 ** k), dtype=np.float32)
    for i in range(n):
        cnt = np.bincount(idx[i][ok[i]], minlength=4 ** k)
        out[i] = cnt / max(cnt.sum(), 1)
    return out


def onehot_features(seqs) -> np.ndarray:
    X = encode(seqs)
    n, L = X.shape
    out = np.zeros((n, L * 4), dtype=np.float32)
    rows = np.repeat(np.arange(n), L)
    cols = np.tile(np.arange(L), n) * 4 + X.ravel()
    keep = X.ravel() < 4
    out[rows[keep], cols[keep]] = 1.0
    return out


class PWM:
    """Per-class positional nucleotide frequencies; scores are log-likelihoods.

    Generative and hyperparameter-free. For two classes the decision value is the
    familiar position weight matrix log-odds; for three it is a naive Bayes over
    positions with the class prior included.
    """

    def __init__(self, pseudocount: float = 1.0):
        self.a = pseudocount

    def fit(self, seqs, y):
        X = encode(seqs)
        self.classes_ = np.unique(y)
        L = X.shape[1]
        self.logf = np.zeros((len(self.classes_), L, 4))
        self.logprior = np.zeros(len(self.classes_))
        for ci, c in enumerate(self.classes_):
            sub = X[y == c]
            cnt = np.zeros((L, 4)) + self.a
            for b in range(4):
                cnt[:, b] += (sub == b).sum(0)
            self.logf[ci] = np.log(cnt / cnt.sum(1, keepdims=True))
            self.logprior[ci] = np.log(len(sub) / len(X))
        return self

    def decision(self, seqs) -> np.ndarray:
        X = encode(seqs)
        n, L = X.shape
        pos = np.arange(L)
        S = np.zeros((n, len(self.classes_)))
        for ci in range(len(self.classes_)):
            f = self.logf[ci]
            for i in range(n):
                row = X[i]
                m = row < 4
                S[i, ci] = f[pos[m], row[m]].sum() + self.logprior[ci]
        return S

    def score(self, seqs):
        S = self.decision(seqs)
        if S.shape[1] == 2:
            return S[:, 1] - S[:, 0]
        return S - S.max(1, keepdims=True)


FEATURE_SETS = {
    "gc_cpg":  lambda s: gc_features(s),
    "kmer3":   lambda s: kmer_features(s, 3),
    "kmer4":   lambda s: kmer_features(s, 4),
    "kmer5":   lambda s: kmer_features(s, 5),
    "kmer6":   lambda s: kmer_features(s, 6),
    "onehot":  lambda s: onehot_features(s),
}
