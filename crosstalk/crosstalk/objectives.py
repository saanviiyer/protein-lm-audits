"""The reward zoo.

Every objective maps a fitness vector over partners to a scalar. They differ
only in how they price off-target binding, which is the whole point: the
benchmark asks which of these an optimizer can actually be trusted to follow.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Objective:
    name: str
    target: int = 0
    off: tuple[int, ...] = (1,)
    tau: float = 0.5          # off-target fitness above this counts as crosstalk
    lam: float = 5.0          # Lagrangian penalty weight
    floor: float = -1.0       # score assigned to a gated design that fails

    def __call__(self, f: np.ndarray) -> np.ndarray:
        """f is (..., n_partners); returns (...,)."""
        f = np.asarray(f, dtype=float)
        on = f[..., self.target]
        offmax = np.max(f[..., list(self.off)], axis=-1)
        if self.name == "affinity":
            return on
        if self.name == "margin":
            return on - offmax
        if self.name == "gated":
            return np.where(offmax <= self.tau, on, self.floor)
        if self.name == "lagrangian":
            return on - self.lam * np.maximum(0.0, offmax - self.tau)
        if self.name == "logratio":
            eps = 1e-3
            return np.log(np.maximum(on, eps) + eps) - np.log(np.maximum(offmax, eps) + eps)
        raise ValueError(f"unknown objective {self.name!r}")


OBJECTIVES = ("affinity", "margin", "gated", "lagrangian", "logratio")


def make(name: str, target: int = 0, off: tuple[int, ...] = (1,), **kw) -> Objective:
    return Objective(name=name, target=target, off=off, **kw)


def is_specific(f: np.ndarray, target: int = 0, off=(1,), tau: float = 0.5,
                on_min: float = 0.8) -> np.ndarray:
    """Ground-truth success: binds the target well AND stays off every off-target.

    This is the evaluation criterion. It is deliberately not any of the rewards
    above -- an agent optimizes a reward, and we score it on this.
    """
    f = np.asarray(f, dtype=float)
    on = f[..., target]
    offmax = np.max(f[..., list(off)], axis=-1)
    return (on >= on_min) & (offmax <= tau)
