"""Evaluation. Agents are always scored on ground truth, never on their reward."""
from __future__ import annotations

import numpy as np

from .landscape import Landscape
from .objectives import Objective, is_specific


def evaluate(L: Landscape, nominations: list[str], obj: Objective,
             tau: float = 0.5, on_min: float = 0.8) -> dict:
    """Score a set of nominated designs against ground truth.

    specificity_regret  gap to the globally optimal design under `obj`
    crosstalk_rate      fraction of designs that bind an off-target above tau
    success_rate        fraction that both bind the target and stay off others
    """
    idx = [L.index[s] for s in nominations]
    F = L.F[idx]
    scores = obj(L.F)
    best = float(scores.max())
    got = obj(F)
    offmax = np.max(F[:, list(obj.off)], axis=1)
    ok = is_specific(F, target=obj.target, off=obj.off, tau=tau, on_min=on_min)
    return {
        "n": len(nominations),
        "mean_score": float(got.mean()),
        "specificity_regret": float(best - got.mean()),
        "crosstalk_rate": float((offmax > tau).mean()),
        "success_rate": float(ok.mean()),
        "mean_on_target": float(F[:, obj.target].mean()),
        "mean_off_target": float(offmax.mean()),
    }
