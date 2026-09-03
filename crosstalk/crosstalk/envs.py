"""Gymnasium-style environments over a specificity landscape.

Two settings share one landscape:

BudgetEnv  -- the benchmark. The agent has a fixed budget of noisy oracle
              queries (a wet-lab measurement campaign), then nominates one
              design. Reward is the ground-truth objective of the nomination.
              Replication versus exploration is a real decision here.

WalkEnv    -- calibration. Local search with a mutation budget and dense
              reward. Small enough that the optimal policy is exactly solvable,
              which is what makes it useful as a unit test rather than a
              benchmark.

Gymnasium is optional: the reset/step API is implemented natively and the envs
register with gymnasium only if it is installed.
"""
from __future__ import annotations

import numpy as np

from .landscape import AA, Landscape
from .objectives import Objective


class BudgetEnv:
    """Best-design identification under a noisy oracle budget.

    Observation: the agent's own measurement history, as a (n_seqs, n_partners)
    running mean plus a (n_seqs,) query count. Action: index of the sequence to
    measure next. Calling `nominate` ends the episode.
    """

    def __init__(self, landscape: Landscape, objective: Objective,
                 budget: int = 100, seed: int = 0):
        self.L = landscape
        self.obj = objective
        self.budget = budget
        self.rng = np.random.default_rng(seed)
        self.n = landscape.n_seqs
        self.p = landscape.n_partners

    def reset(self, seed: int | None = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.sum = np.zeros((self.n, self.p))
        self.cnt = np.zeros(self.n)
        self.spent = 0
        self.history: list[tuple[int, np.ndarray]] = []
        return self._obs(), {}

    def _obs(self):
        with np.errstate(invalid="ignore", divide="ignore"):
            mean = np.where(self.cnt[:, None] > 0, self.sum / np.maximum(self.cnt, 1)[:, None], 0.0)
        return {"mean": mean, "count": self.cnt.copy(), "remaining": self.budget - self.spent}

    def step(self, action: int):
        """Spend one query measuring sequence `action`."""
        if self.spent >= self.budget:
            return self._obs(), 0.0, True, False, {"reason": "budget exhausted"}
        y = self.L.measure(self.L.seqs[action], self.rng)
        self.sum[action] += y
        self.cnt[action] += 1
        self.spent += 1
        self.history.append((action, y))
        terminated = self.spent >= self.budget
        return self._obs(), 0.0, terminated, False, {"measurement": y}

    def nominate(self, action: int) -> dict:
        """End the episode by proposing a final design. Scored on ground truth."""
        f = self.L.truth(self.L.seqs[action])
        return {
            "seq": self.L.seqs[action],
            "truth": f,
            "score": float(self.obj(f)),
            "queries_used": self.spent,
        }

    # convenience for agents that reason in sequence space
    def seq_index(self, seq: str) -> int:
        return self.L.index[seq]


class WalkEnv:
    """Local search with a mutation budget and dense ground-truth reward."""

    def __init__(self, landscape: Landscape, objective: Objective,
                 horizon: int = 6, start: str | None = None,
                 noisy: bool = False, seed: int = 0):
        self.L = landscape
        self.obj = objective
        self.horizon = horizon
        self.start = start or landscape.wt or landscape.seqs[0]
        self.noisy = noisy
        self.rng = np.random.default_rng(seed)

    def reset(self, seed: int | None = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.seq = self.start
        self.t = 0
        return self.seq, {}

    def actions(self) -> list[str]:
        return self.L.neighbors(self.seq)

    def step(self, action: str):
        self.seq = action
        self.t += 1
        f = self.L.measure(action, self.rng) if self.noisy else self.L.truth(action)
        reward = float(self.obj(f))
        return self.seq, reward, self.t >= self.horizon, False, {"truth": self.L.truth(action)}


def register_gymnasium() -> bool:
    """Register the envs with gymnasium if it is installed. Optional."""
    try:
        import gymnasium as gym
        from gymnasium.envs.registration import register
    except ImportError:
        return False
    for name, cls in (("Crosstalk-Budget-v0", BudgetEnv), ("Crosstalk-Walk-v0", WalkEnv)):
        if name not in gym.registry:
            register(id=name, entry_point=f"crosstalk.envs:{cls.__name__}")
    return True
