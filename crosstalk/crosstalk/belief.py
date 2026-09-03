"""Conjugate Bayesian belief over a landscape, used as policy input.

A policy that decides what to assay next needs uncertainty, not just a point
estimate. Fitness is modelled as a position-wise additive function of the
sequence (one-hot, d = seq_len * 20) under a Bayesian linear model, so the
posterior mean and predictive variance are exact and update in closed form
after every measurement.

The observation noise is not a free parameter: it is the per-variant assay SD
calibrated from the biological replicates in the landscape.
"""
from __future__ import annotations

import numpy as np

from .landscape import AA, Landscape


class AdditiveBelief:
    """Independent Bayesian linear model per partner, over one-hot sequences."""

    def __init__(self, L: Landscape, tau: float = 1.0, model_sd: float = 0.11):
        """model_sd is the epistatic misspecification scale.

        An additive model cannot represent epistasis, and on this landscape that
        residual (SD ~0.105 ParE3, ~0.115 ParE2) is 7-12x the assay noise. Left
        out, the posterior is overconfident by ~6x and the policy stops
        exploring. It is a variance scale, not a location, so it carries no
        information about where the optimum is.
        """
        self.L = L
        self.model_var = float(model_sd) ** 2
        self.d = L.seq_len * len(AA)
        self.p = L.n_partners
        self.X = self._encode_all(L)                       # (n_seqs, d)
        self.tau2 = tau ** 2
        self.reset()

    @staticmethod
    def _encode_all(L: Landscape) -> np.ndarray:
        aa_idx = {a: k for k, a in enumerate(AA)}
        X = np.zeros((L.n_seqs, L.seq_len * len(AA)))
        for i, s in enumerate(L.seqs):
            for pos, ch in enumerate(s):
                X[i, pos * len(AA) + aa_idx[ch]] = 1.0
        return X

    def reset(self):
        # posterior precision inverse, starting from the prior tau^2 * I
        self.A_inv = np.stack([np.eye(self.d) * self.tau2 for _ in range(self.p)])
        self.b = np.zeros((self.p, self.d))                # X^T y / sigma^2
        self.n_obs = np.zeros(self.p, dtype=int)

    def update(self, seq_idx: int, y: np.ndarray, partners: list[int]):
        """Absorb one measurement. Rank-1 Sherman-Morrison per partner."""
        x = self.X[seq_idx]
        for p in partners:
            sigma2 = float(self.L.noise_sd[seq_idx, p] ** 2) + self.model_var
            Ainv = self.A_inv[p]
            Ax = Ainv @ x
            denom = sigma2 + x @ Ax
            self.A_inv[p] = Ainv - np.outer(Ax, Ax) / denom
            self.b[p] += x * (y[p] / sigma2)
            self.n_obs[p] += 1

    def posterior_mean(self, idx: np.ndarray | None = None) -> np.ndarray:
        """Predictive mean only. Ranking candidates does not need the variance,
        and skipping it is what keeps a full-landscape scan cheap."""
        Xs = self.X if idx is None else self.X[idx]
        W = np.stack([self.A_inv[p] @ self.b[p] for p in range(self.p)], axis=1)
        return Xs @ W

    def posterior(self, idx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Predictive mean and SD for candidate indices. Both (len(idx), p)."""
        Xs = self.X[idx]
        mean = np.empty((len(idx), self.p))
        sd = np.empty((len(idx), self.p))
        for p in range(self.p):
            w = self.A_inv[p] @ self.b[p]
            mean[:, p] = Xs @ w
            var = np.sum((Xs @ self.A_inv[p]) * Xs, axis=1)
            sd[:, p] = np.sqrt(np.maximum(var + self.model_var, 1e-12))
        return mean, sd
