#!/usr/bin/env python3
"""Replay the exact generator stream of scripts/run_granularity_ladder.py.

The biointerp reproduction matched section 26 to the last digit on both
deterministic interventions and missed synonymous recode by +0.10 -- eight
between-seed standard deviations, far outside what the codon draw can explain.
Either the original number depended on its generator stream in a way an
independent draw does not reproduce, or the original number is not a property of
the model at all.

This decides it by reconstructing the original stream exactly: one generator,
seeded 0, consumed in the original order (genes in sorted order, and within each
gene synonymous recode before codon-order shuffle), and scoring the sequences
that stream produces.
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))

from crosstalk.biointerp import scorers
from crosstalk.biointerp.interventions import _synonymous_recode, _codon_shuffle
from run_biointerp_battery import load_sequences


def main():
    seqs = load_sequences(2400)
    keys = sorted(seqs)
    rng = np.random.default_rng(0)                 # the original single stream
    recoded, shuffled = {}, {}
    for k in keys:                                 # the original iteration order
        s = seqs[k]
        recoded[k] = _synonymous_recode(s, rng)    # consumed first, as in the original
        shuffled[k] = _codon_shuffle(s, rng)

    model = scorers.NTScorer()
    real = np.asarray(model.score([seqs[k] for k in keys]))
    for label, d in [("synonymous recode", recoded), ("codon-order shuffle", shuffled)]:
        x = np.asarray(model.score([d[k] for k in keys]))
        delta = real - x
        se = delta.std(ddof=1) / np.sqrt(len(delta))
        print(f"{label:22s} {delta.mean():+.4f} "
              f"[{delta.mean()-1.96*se:+.4f}, {delta.mean()+1.96*se:+.4f}] "
              f"real higher {int((delta>0).sum())}/{len(delta)}", flush=True)

    frac = np.mean([np.mean([recoded[k][i:i+3] != seqs[k][i:i+3]
                             for i in range(0, len(seqs[k]), 3)]) for k in keys])
    print(f"\nfraction of codons changed by the recode: {frac:.3f}")


if __name__ == "__main__":
    main()
