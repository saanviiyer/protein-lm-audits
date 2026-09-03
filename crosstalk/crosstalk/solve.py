"""Exact solutions. Available because every landscape here is enumerable.

This is the property that makes the suite a benchmark rather than a demo: the
optimal design and the optimal budgeted-walk policy are both computable, so
regret is exact rather than relative to the best method tried.
"""
from __future__ import annotations

import numpy as np

from .landscape import Landscape
from .objectives import Objective


def best_design(L: Landscape, obj: Objective) -> tuple[str, float]:
    """The globally optimal design under `obj`, by enumeration."""
    scores = obj(L.F)
    i = int(np.argmax(scores))
    return L.seqs[i], float(scores[i])


def optimal_walk_value(L: Landscape, obj: Objective, start: str, horizon: int) -> float:
    """Value of the optimal noiseless walk: best score within `horizon` mutations.

    Transitions are deterministic and single-substitution, so the reachable set
    is the Hamming ball of radius `horizon` and the optimal value is its max.
    """
    scores = obj(L.F)
    start_arr = np.frombuffer(start.encode(), dtype="S1")
    seq_arr = np.array([np.frombuffer(s.encode(), dtype="S1") for s in L.seqs])
    dist = (seq_arr != start_arr).sum(axis=1)
    reachable = dist <= horizon
    return float(scores[reachable].max())


def ruggedness(L: Landscape, obj: Objective) -> dict:
    """How hard is this objective to hill-climb?

    Reports the number of local optima under single substitutions and the
    fraction of exhaustive greedy ascents that reach the global optimum. A more
    rugged objective is a harder search problem even on identical data.
    """
    scores = obj(L.F)
    n = L.n_seqs
    nbr = [np.array([L.index[t] for t in L.neighbors(s)], dtype=int) for s in L.seqs]

    is_local_opt = np.array([bool(len(nb) == 0 or scores[i] >= scores[nb].max())
                             for i, nb in enumerate(nbr)])

    # greedy ascent from every state, memoised
    peak = np.full(n, -1, dtype=int)

    def ascend(i: int) -> int:
        path = []
        while peak[i] < 0:
            if is_local_opt[i]:
                peak[i] = i
                break
            path.append(i)
            j = int(nbr[i][np.argmax(scores[nbr[i]])])
            if scores[j] <= scores[i]:
                peak[i] = i
                break
            i = j
        top = peak[i]
        for k in path:
            peak[k] = top
        return top

    peaks = np.array([ascend(i) for i in range(n)])
    gopt = int(np.argmax(scores))
    return {
        "n_local_optima": int(is_local_opt.sum()),
        "frac_ascents_reaching_global": float((peaks == gopt).mean()),
        "global_opt_seq": L.seqs[gopt],
        "global_opt_score": float(scores[gopt]),
        "mean_peak_score": float(scores[peaks].mean()),
    }
