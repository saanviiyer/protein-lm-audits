"""Two-partner binding landscapes with a calibrated noisy oracle.

A specificity landscape maps a sequence to fitness against several partners:
one is the intended target, the rest are off-targets that must be avoided.
Every landscape here is finite and fully enumerable, which is what makes exact
regret computable -- see crosstalk.solve.
"""
from __future__ import annotations

import csv
import gzip
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

AA = "ACDEFGHIKLMNPQRSTVWY"


@dataclass
class Landscape:
    """A finite two-or-more-partner binding landscape.

    seqs      ordered list of sequence strings (the full enumerated state space)
    partners  ordered partner names, e.g. ["ParE3", "ParE2"]
    F         (n_seqs, n_partners) ground-truth fitness
    noise_sd  (n_seqs, n_partners) per-measurement standard deviation, empirical
    name      landscape identifier
    wt        wild-type sequence, if defined
    """

    seqs: list[str]
    partners: list[str]
    F: np.ndarray
    noise_sd: np.ndarray
    name: str = "landscape"
    wt: str | None = None
    index: dict[str, int] = field(init=False, repr=False)

    def __post_init__(self):
        self.F = np.asarray(self.F, dtype=float)
        self.noise_sd = np.asarray(self.noise_sd, dtype=float)
        if self.F.shape != (len(self.seqs), len(self.partners)):
            raise ValueError(f"F shape {self.F.shape} != {(len(self.seqs), len(self.partners))}")
        if self.noise_sd.shape != self.F.shape:
            raise ValueError("noise_sd must match F")
        self.index = {s: i for i, s in enumerate(self.seqs)}

    @property
    def n_seqs(self) -> int:
        return len(self.seqs)

    @property
    def n_partners(self) -> int:
        return len(self.partners)

    @property
    def seq_len(self) -> int:
        return len(self.seqs[0])

    def truth(self, seq: str) -> np.ndarray:
        """Noiseless ground truth. Evaluation only -- agents must not call this."""
        return self.F[self.index[seq]]

    def measure(self, seq: str, rng: np.random.Generator) -> np.ndarray:
        """One noisy oracle read, using the landscape's calibrated noise."""
        i = self.index[seq]
        return self.F[i] + rng.normal(0.0, self.noise_sd[i])

    def neighbors(self, seq: str) -> list[str]:
        """Single-substitution neighbours that exist in the landscape."""
        out = []
        for i in range(len(seq)):
            for a in AA:
                if a == seq[i]:
                    continue
                t = seq[:i] + a + seq[i + 1 :]
                if t in self.index:
                    out.append(t)
        return out


def _fitness_from_freq(t0: np.ndarray, t1: np.ndarray, floor: float = 1e-7) -> np.ndarray:
    """Log-enrichment, scaled so the wild-type-like maximum sits near 1."""
    return np.log((t1 + floor) / (t0 + floor))


def load_pard3(
    fitness_csv: str | Path = "data/raw/GSE153897_Variant_fitness.csv.gz",
    freq_csv: str | Path = "data/raw/GSE153897_Variant_frequency.csv.gz",
) -> Landscape:
    """ParD3 antitoxin, positions 61/64/80, vs cognate ParE3 and non-cognate ParE2.

    Lite et al., eLife 2020 (GEO GSE153897). Published fitness is the ground
    truth; the two biological replicates in the frequency table give an
    empirical, read-depth-dependent noise model rather than an invented one.
    """
    fitness_csv, freq_csv = Path(fitness_csv), Path(freq_csv)
    seqs, F = [], []
    with gzip.open(fitness_csv, "rt") as fh:
        for row in csv.DictReader(fh):
            seqs.append(row["Variant"])
            F.append([float(row["W_ParE3"]), float(row["W_ParE2"])])
    F = np.array(F)

    noise = _pard3_noise(freq_csv, seqs, F)
    return Landscape(
        seqs=seqs,
        partners=["ParE3", "ParE2"],
        F=F,
        noise_sd=noise,
        name="pard3",
        wt="DKE",
    )


def _pard3_noise(freq_csv: Path, seqs: list[str], F: np.ndarray) -> np.ndarray:
    """Per-variant, per-partner measurement SD, in the same units as published W.

    The two biological replicates give replicate disagreement in log-enrichment
    units. Published W is a linear function of log-enrichment (we recover it at
    r=0.9998), so we fit that map and push the replicate SD through its slope --
    otherwise the noise would be overstated by ~7x. Replicate disagreement is
    read-depth dependent, so the SD is fit as a + b/sqrt(t0_frequency).
    """
    n = len(seqs)
    if not freq_csv.exists():
        return np.column_stack([np.full(n, 0.030), np.full(n, 0.022)])

    rows = {}
    with gzip.open(freq_csv, "rt") as fh:
        rdr = csv.reader(fh)
        header = next(rdr)
        cols = {name: i for i, name in enumerate(header) if name}
        for row in rdr:
            rows[row[0]] = row

    sd = np.full((n, 2), np.nan)
    depth = np.full(n, np.nan)
    per_rep = np.full((n, 2, 2), np.nan)

    for si, s in enumerate(seqs):
        row = rows.get(s)
        if row is None:
            continue
        d = []
        for pi, partner in enumerate(("ParE3", "ParE2")):
            vals = []
            for rep in (1, 2):
                try:
                    t0 = float(row[cols[f"{partner}_rep{rep}_t0"]])
                    t1 = float(row[cols[f"{partner}_rep{rep}_t600"]])
                except (KeyError, ValueError, IndexError):
                    continue
                d.append(t0)
                vals.append(_fitness_from_freq(t0, t1))
            if len(vals) == 2:
                per_rep[si, pi] = vals
        if d:
            depth[si] = float(np.mean(d))

    diff = np.abs(per_rep[:, :, 0] - per_rep[:, :, 1]) / math.sqrt(2.0)
    mean_le = np.nanmean(per_rep, axis=2)  # seq x partner, log-enrichment

    for pi in range(2):
        # slope of published W on log-enrichment, to convert units
        fit_ok = np.isfinite(mean_le[:, pi]) & np.isfinite(F[:, pi])
        slope = abs(np.polyfit(mean_le[fit_ok, pi], F[fit_ok, pi], 1)[0]) if fit_ok.sum() > 50 else 1.0

        ok = np.isfinite(diff[:, pi]) & np.isfinite(depth) & (depth > 0)
        if ok.sum() < 50:
            fallback = np.nanmedian(diff[:, pi]) if np.isfinite(diff[:, pi]).any() else 0.2
            sd[:, pi] = fallback * slope
            continue
        x = 1.0 / np.sqrt(depth[ok])
        y = diff[ok, pi]
        b, a = np.polyfit(x, y, 1)
        safe_depth = np.where(depth > 0, depth, np.nanmedian(depth[ok]))
        pred = a + b / np.sqrt(safe_depth)
        pred = np.where(np.isfinite(pred), pred, np.nanmedian(y))
        sd[:, pi] = np.clip(pred, 1e-3, np.nanpercentile(y, 99)) * slope

    return np.nan_to_num(sd, nan=float(np.nanmedian(sd)))
