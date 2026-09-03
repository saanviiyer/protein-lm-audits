"""Polyspecificity landscapes from the Absolut! lattice database.

ParD3 has only two partners measured, which caps it at one off-target. Real
specificity is against many competitors at once, and the margin objective's
max-over-off-targets is untested beyond K=1. Absolut! provides ground-truth
binding energies for the same ~6.9M murine CDRH3 sequences against 159 antigens
(Robert et al. 2021), which is the multi-partner ground truth this needs.

Two honest differences from the ParD3 landscape:
  1. These energies are simulated (lattice + Miyazawa-Jernigan), not measured.
  2. They are deterministic, so there is no replicate noise to calibrate. The
     assay noise here is a stated modelling choice, not an estimate from data.

The sequence unit is the 11-mer slide, not the full CDRH3. CDRH3s vary in
length, which breaks the fixed-length one-hot belief model, and Absolut's own
manuscript datasets are 11-mer-based for the same reason. Slides are subsampled
by a deterministic hash, so every antigen contributes the same sequence set
without needing a shared index.
"""
from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

import numpy as np

from .landscape import Landscape


def _keep(seq: str, keep_one_in: int) -> bool:
    h = hashlib.blake2b(seq.encode(), digest_size=8).digest()
    return int.from_bytes(h, "big") % keep_one_in == 0


def best_energies(zip_path: str | Path, keep_one_in: int = 256) -> dict[str, float]:
    """Best (most negative) binding energy per 11-mer slide for one antigen.

    Streams the archive: a single antigen is ~2.5 GB uncompressed and there are
    several, so nothing is written to disk.
    """
    zip_path = Path(zip_path)
    out: dict[str, float] = {}
    with zipfile.ZipFile(zip_path) as z:
        for member in z.namelist():
            if not member.endswith(".txt"):
                continue
            with z.open(member) as fh:
                for raw in fh:
                    line = raw.decode("utf-8", errors="replace")
                    if line.startswith("#") or line.startswith("ID_slide"):
                        continue
                    parts = line.rstrip("\n").split("\t")
                    if len(parts) < 5:
                        continue
                    slide, energy = parts[3], parts[4]
                    if len(slide) != 11 or not _keep(slide, keep_one_in):
                        continue
                    try:
                        e = float(energy)
                    except ValueError:
                        continue
                    if slide not in out or e < out[slide]:
                        out[slide] = e
    return out


def build_landscape(zip_paths: dict[str, str | Path], keep_one_in: int = 256,
                    noise_sd: float = 0.03, max_seqs: int | None = None,
                    cache_dir: str | Path | None = None) -> Landscape:
    """Assemble a multi-partner landscape from several antigen archives.

    Fitness is the negated binding energy, so larger is better and the
    objectives carry over from ParD3 unchanged.

    Each partner is normalised INDEPENDENTLY. The antigens have very different
    raw energy ranges, so a single global scale makes one partner dominate
    max-over-off-targets while others can never cross any shared threshold --
    which silently makes K=3 and K=4 identical to K=2. Per-antigen scaling also
    matches how counter-screens actually read out, one assay at a time.
    """
    per_antigen = {}
    for name, zp in zip_paths.items():
        cached = Path(cache_dir) / f"{name}_energies.json" if cache_dir else None
        if cached is not None and cached.exists():
            per_antigen[name] = json.loads(cached.read_text())
            continue
        per_antigen[name] = best_energies(zp, keep_one_in)
        if cached is not None:
            cached.parent.mkdir(parents=True, exist_ok=True)
            cached.write_text(json.dumps(per_antigen[name]))
    names = list(per_antigen)
    common = set.intersection(*(set(d) for d in per_antigen.values()))
    seqs = sorted(common)
    if max_seqs is not None and len(seqs) > max_seqs:
        rng = np.random.default_rng(0)
        seqs = [seqs[i] for i in sorted(rng.choice(len(seqs), max_seqs, replace=False))]

    E = np.array([[per_antigen[n][s] for n in names] for s in seqs])
    F = -E                                    # larger is better
    lo, hi = F.min(axis=0, keepdims=True), F.max(axis=0, keepdims=True)
    F = (F - lo) / (hi - lo)                  # per-partner 0..1

    return Landscape(
        seqs=seqs,
        partners=names,
        F=F,
        noise_sd=np.full_like(F, noise_sd),
        name="absolut",
        wt=None,
    )
