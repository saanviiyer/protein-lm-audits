"""Baseline agents for the budgeted specificity campaign.

Budget is denominated in ASSAYS, not variants. Measuring the on-target costs
one assay; counter-screening against an off-target costs another. An agent that
optimizes raw affinity therefore screens twice as many variants as one that
counter-screens every candidate -- which is exactly the trade real campaigns
make, and the reason "just screen for binding first" is tempting.
"""
from __future__ import annotations

import numpy as np

from .landscape import AA, Landscape
from .objectives import Objective


class Campaign:
    """Bookkeeping for one budgeted campaign against a landscape."""

    def __init__(self, L: Landscape, budget: int, counter_screen: bool, rng,
                 target: int = 0):
        self.L, self.budget, self.counter_screen, self.rng = L, budget, counter_screen, rng
        self.target = target
        self.spent = 0
        self.sum = np.zeros((L.n_seqs, L.n_partners))
        self.cnt = np.zeros((L.n_seqs, L.n_partners))

    @property
    def cost(self) -> int:
        return self.L.n_partners if self.counter_screen else 1

    def can_afford(self) -> bool:
        return self.spent + self.cost <= self.budget

    def measure(self, i: int) -> np.ndarray:
        y = self.L.measure(self.L.seqs[i], self.rng)
        if self.counter_screen:
            self.sum[i] += y
            self.cnt[i] += 1
        else:
            # only the on-target assay is run; the off-target is never observed
            self.sum[i, self.target] += y[self.target]
            self.cnt[i, self.target] += 1
        self.spent += self.cost
        return y

    def mean(self) -> np.ndarray:
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(self.cnt > 0, self.sum / np.maximum(self.cnt, 1), np.nan)

    def tested(self) -> np.ndarray:
        return np.where(self.cnt[:, self.target] > 0)[0]


def _nominate(c: Campaign, obj: Objective, believed: np.ndarray) -> int:
    """Pick the best design the agent believes in, among those it actually tested."""
    tested = c.tested()
    if len(tested) == 0:
        return 0
    scores = obj(believed[tested])
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    return int(tested[int(np.argmax(scores))])


def random_search(L, obj, budget, counter_screen, rng, **kw) -> int:
    c = Campaign(L, budget, counter_screen, rng, target=obj.target)
    while c.can_afford():
        c.measure(int(rng.integers(L.n_seqs)))
    m = c.mean()
    if not counter_screen:
        m = np.nan_to_num(m, nan=0.0)  # unmeasured off-target assumed benign
    return _nominate(c, obj, m)


def local_search(L, obj, budget, counter_screen, rng, start=None, restarts=3, **kw) -> int:
    """Greedy ascent on measured values, with restarts."""
    c = Campaign(L, budget, counter_screen, rng, target=obj.target)
    cur = L.index[start or L.wt or L.seqs[0]]
    if c.can_afford():
        c.measure(cur)
    while c.can_afford():
        nbrs = [L.index[t] for t in L.neighbors(L.seqs[cur])]
        rng.shuffle(nbrs)
        best, best_s = cur, -np.inf
        moved = False
        for j in nbrs:
            if not c.can_afford():
                break
            c.measure(j)
            m = c.mean()[j]
            if not counter_screen:
                m = np.nan_to_num(m, nan=0.0)
            s = float(obj(m))
            if s > best_s:
                best, best_s = j, s
        m_cur = c.mean()[cur]
        if not counter_screen:
            m_cur = np.nan_to_num(m_cur, nan=0.0)
        if best_s > float(obj(m_cur)):
            cur, moved = best, True
        if not moved:
            cur = int(rng.integers(L.n_seqs))  # restart
            if c.can_afford():
                c.measure(cur)
    m = c.mean()
    if not counter_screen:
        m = np.nan_to_num(m, nan=0.0)
    return _nominate(c, obj, m)


def additive_model(L, obj, budget, counter_screen, rng, explore_frac=0.6, **kw) -> int:
    """Screen a random subset, fit a position-wise additive model, then exploit.

    This is what an ML-guided campaign actually does, and it is the agent that
    most cleanly exposes reward misspecification: it extrapolates its objective
    to all 7,882 variants from a few hundred measurements.
    """
    c = Campaign(L, budget, counter_screen, rng, target=obj.target)
    n_explore = int(explore_frac * budget / c.cost)
    for _ in range(n_explore):
        if not c.can_afford():
            break
        c.measure(int(rng.integers(L.n_seqs)))

    tested = c.tested()
    m = c.mean()
    pred = _fit_additive(L, tested, m, counter_screen, target=obj.target)

    # exploit: measure the model's top candidates, nominate the best confirmed one
    order = np.argsort(-obj(pred))
    for j in order:
        if not c.can_afford():
            break
        if c.cnt[j, c.target] == 0:
            c.measure(int(j))
    mm = c.mean()
    if not counter_screen:
        mm = np.nan_to_num(mm, nan=0.0)
    return _nominate(c, obj, mm)


def _fit_additive(L: Landscape, tested: np.ndarray, m: np.ndarray, counter_screen: bool,
                  target: int = 0) -> np.ndarray:
    """One-hot ridge regression per partner: fitness ~ sum of per-position effects."""
    ncol = L.seq_len * len(AA)
    aa_idx = {a: k for k, a in enumerate(AA)}

    def encode(idxs):
        X = np.zeros((len(idxs), ncol))
        for r, i in enumerate(idxs):
            for p, ch in enumerate(L.seqs[i]):
                X[r, p * len(AA) + aa_idx[ch]] = 1.0
        return X

    Xt = encode(tested)
    Xall = encode(range(L.n_seqs))
    pred = np.zeros((L.n_seqs, L.n_partners))
    partners = range(L.n_partners) if counter_screen else [target]
    for p in partners:
        y = m[tested, p]
        ok = np.isfinite(y)
        if ok.sum() < 5:
            continue
        A = Xt[ok]
        w = np.linalg.solve(A.T @ A + 1.0 * np.eye(ncol), A.T @ y[ok])
        pred[:, p] = Xall @ w
    return pred


AGENTS = {
    "random": random_search,
    "local_search": local_search,
    "additive_model": additive_model,
}
