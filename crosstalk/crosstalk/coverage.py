"""Coverage metrics for a detector, where the space is small enough to enumerate.

Screening and detection systems are evaluated per instance: AUC, or a true
positive rate at a fixed false positive budget. Those answer "on a variant drawn
at random, how often is the detector right". Neither side of a real contest asks
that question.

Someone probing a detector needs ONE thing that works and is not flagged. Their
quantity is a maximum over a set they can search. The system needs the whole
viable space handled, and is undone by any region it misses. Its quantity is a
property of a set, not an average over instances.

Two detectors with the same true positive rate can therefore be very differently
useful, and per-instance metrics cannot tell them apart:

  a detector whose misses are scattered singletons forces a search to stumble on
  each one independently.

  a detector whose misses form one large mutually-reachable region can be
  defeated once and then exploited, because ordinary local optimisation walks
  into the region and stays there.

The distinction is invisible to AUC and visible in the mutational graph. It is
computable exactly here, and only here, because the ParD3 landscape is
combinatorially complete: every variant at the three mutated positions was
measured, so the viable set and the missed set are enumerated rather than
estimated.

Nothing in this module is specific to ParD3 beyond needing an exhaustively
measured landscape and a neighbour relation.
"""
from __future__ import annotations

import numpy as np


def operating_threshold(scores: np.ndarray, negative: np.ndarray, fpr: float) -> float:
    """Score threshold admitting at most `fpr` of the negatives.

    Defined on the negatives alone, so the threshold does not depend on how many
    positives exist. That keeps the operating point comparable across detectors
    and across landscapes with different base rates.
    """
    neg = np.sort(scores[negative])[::-1]
    k = int(np.floor(fpr * len(neg)))
    if k <= 0:
        return float(neg[0]) + 1e-12          # admit nothing: budget buys no calls
    return float(neg[k - 1])


def coverage(scores, functional, negative, fpr):
    """Fraction of the viable set flagged at a false-positive budget.

    Numerically this is the familiar TPR at a fixed FPR. It is reported here as
    the first of several coverage quantities, because on its own it says nothing
    about whether the misses are scattered or contiguous.
    """
    thr = operating_threshold(scores, negative, fpr)
    flagged = scores >= thr
    cov = float((flagged & functional).sum() / max(functional.sum(), 1))
    escape = functional & ~flagged
    return dict(threshold=thr, coverage=cov, n_escape=int(escape.sum()),
                escape=escape)


def components(mask: np.ndarray, neighbours: list[np.ndarray]) -> dict:
    """Connected components of the masked set under the mutational graph.

    A missed set that is one connected component behaves very differently from
    the same number of isolated points: within a component, single substitutions
    move between missed variants without ever being flagged.
    """
    idx = np.where(mask)[0]
    if len(idx) == 0:
        return dict(n_components=0, largest=0, largest_frac=0.0, singletons=0)
    pos = {int(i): k for k, i in enumerate(idx)}
    parent = list(range(len(idx)))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for i in idx:
        for j in neighbours[i]:
            if mask[j]:
                ra, rb = find(pos[int(i)]), find(pos[int(j)])
                if ra != rb:
                    parent[ra] = rb
    sizes = {}
    for k in range(len(idx)):
        r = find(k)
        sizes[r] = sizes.get(r, 0) + 1
    s = sorted(sizes.values(), reverse=True)
    return dict(n_components=len(s), largest=s[0],
                largest_frac=s[0] / len(idx), singletons=sum(1 for x in s if x == 1))


def top_k_coverage(scores, fitness, functional, threshold, ks=(10, 50, 100)):
    """Coverage restricted to the best viable variants.

    Connectivity saturates on a dense graph and hill-climbing endpoints collapse
    to a handful of optima, so both structural statistics lose resolution on a
    small landscape. This one does not. It asks whether the detector flags the
    variants an optimiser is actually trying to reach, which is the operational
    question, and it has as many distinct values as there are variants.

    Being right about the bulk while blind at the top is the failure mode that a
    per-instance average is least able to show.
    """
    idx = np.where(functional)[0]
    order = idx[np.argsort(fitness[idx])[::-1]]
    out = {}
    for k in ks:
        top = order[:k]
        out[f"coverage_top{k}"] = float((scores[top] >= threshold).mean()) if len(top) else float("nan")
    return out


def expected_draws_to_escape(escape: np.ndarray) -> float:
    """Uniform sampling: expected draws before one viable, unflagged variant."""
    p = escape.sum() / len(escape)
    return float("inf") if p == 0 else float(1.0 / p)


def local_search_escape_rate(fitness: np.ndarray, escape: np.ndarray,
                             neighbours: list[np.ndarray], n_starts: int = 2000,
                             seed: int = 0) -> dict:
    """Fraction of ordinary hill-climbs that finish on an unflagged viable variant.

    This is the adversary model that matters, and it is not adversarial. It is
    plain directed evolution: start somewhere, accept the best single
    substitution, stop at a local optimum. The climb optimises measured function
    and never consults the detector. If those endpoints are disproportionately
    unflagged, the detector is blind exactly where ordinary optimisation lands,
    which no per-instance metric would reveal.
    """
    rng = np.random.default_rng(seed)
    starts = rng.choice(len(fitness), size=min(n_starts, len(fitness)), replace=False)
    peak_of = {}

    def climb(i):
        path = []
        while True:
            if i in peak_of:
                p = peak_of[i]
                break
            path.append(i)
            nb = neighbours[i]
            if len(nb) == 0:
                p = i
                break
            best = nb[int(np.argmax(fitness[nb]))]
            if fitness[best] <= fitness[i]:
                p = i
                break
            i = int(best)
        for q in path:
            peak_of[q] = p
        return p

    peaks = np.array([climb(int(s)) for s in starts])
    return dict(n_starts=len(starts),
                escape_rate=float(escape[peaks].mean()),
                distinct_peaks=int(len(set(peaks.tolist()))),
                escape_peaks=int(len({int(p) for p in peaks if escape[p]})))


def build_neighbours(L) -> list[np.ndarray]:
    return [np.array([L.index[t] for t in L.neighbors(s)], dtype=int) for s in L.seqs]
