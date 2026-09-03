"""A one-method scorer interface, so the battery is model-agnostic.

The contract is deliberately narrow: `score(sequences) -> mean per-token
log-likelihood`. Per-token, because the battery compares sequences of equal
length but the same interface must work across genes of different length and
across tokenizers with different compression (NT packs six nucleotides into a
token, HyenaDNA one). Log-likelihood rather than a fitness head, because the
whole point is to interrogate the pretraining objective itself.

`alphabet` says what the model eats. Interventions are defined on nucleotides,
so a protein-alphabet scorer declares `alphabet="protein"` and the battery
translates the intervened DNA before scoring. That is what lets ESM-2 run the
identical battery as a positive control: a model that genuinely represents the
protein must score the real gene's translation above the frameshift's.
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np
import torch

from .. import glm


@runtime_checkable
class Scorer(Protocol):
    name: str
    alphabet: str            # "dna" or "protein"

    def score(self, sequences: list[str], progress: int | None = None) -> np.ndarray:
        """Mean per-token log-likelihood, one value per sequence."""


def to_protein(dna: str) -> str:
    """Translate for a protein-alphabet scorer, mapping stops to the unknown token.

    A frameshifted gene really does contain internal stops, so this is not a
    distortion of the intervention -- it is part of what the intervention did.
    It does hand a protein model an easy cue, which is why a protein model is
    used here only as a positive control and never as a comparison partner.
    """
    return glm.translate(dna).replace("*", "X")


class NTScorer:
    """Nucleotide Transformer v2: masked pseudo-log-likelihood over 6-mer tokens.

    Wraps crosstalk.glm.NTScorer and divides by the token count excluding <cls>,
    which is exactly the normalisation FINDINGS section 26 used.
    """

    alphabet = "dna"

    def __init__(self, model_name="InstaDeepAI/nucleotide-transformer-v2-50m-multi-species",
                 device=None):
        self._s = glm.NTScorer(model_name, device=device)
        self.name = model_name
        self.tokenization = "non-overlapping 6-mer"
        self.objective = "masked LM"

    def n_tokens(self, seq: str) -> int:
        return max(len(self._s.tok(seq)["input_ids"]) - 1, 1)

    def score(self, sequences, progress=None):
        ll = self._s.pseudo_likelihood(list(sequences), progress=progress)
        return np.array([l / self.n_tokens(s) for l, s in zip(ll, sequences)])


class HyenaScorer:
    """HyenaDNA: autoregressive log-likelihood over single nucleotides.

    Differs from NT on both axes that could otherwise explain a shared result:
    single-nucleotide tokens, so a codon is never straddled by a token boundary,
    and next-token prediction rather than masked reconstruction. One forward pass
    per sequence.

    Normalised by the number of *predicted* tokens (all but the first), which is
    the count that actually contributes to the sum. The tokenizer appends [SEP]
    and no BOS, so the first nucleotide is never scored and the terminal [SEP]
    is; both are constant across conditions of equal length, and every comparison
    is paired within sequence, so neither can move a delta.
    """

    alphabet = "dna"

    def __init__(self, model_name="LongSafari/hyenadna-small-32k-seqlen-hf", device=None):
        self._s = glm.HyenaScorer(model_name, device=device)
        self.name = model_name
        self.tokenization = "single nucleotide"
        self.objective = "autoregressive"

    def n_tokens(self, seq: str) -> int:
        return max(self._s.tok(seq, return_tensors="pt")["input_ids"].shape[1] - 1, 1)

    def score(self, sequences, progress=None):
        ll = self._s.log_likelihood(list(sequences), progress=progress)
        return np.array([l / self.n_tokens(s) for l, s in zip(ll, sequences)])


class ESMScorer:
    """ESM-2: masked pseudo-log-likelihood over amino acids. The positive control.

    Declared `alphabet="protein"`, so the battery translates each intervened DNA
    before scoring. Expensive: one forward pass per residue per sequence, so use
    a small checkpoint and a subset of genes.
    """

    alphabet = "protein"

    def __init__(self, model_name="facebook/esm2_t12_35M_UR50D", device=None):
        from .. import plm
        self._plm = plm
        self._s = plm.ESMScorer(model_name, device=device)
        self.name = model_name
        self.tokenization = "amino acid"
        self.objective = "masked LM"

    def score(self, sequences, progress=None):
        ll = self._plm.pseudo_likelihood(self._s, list(sequences), progress=progress)
        return np.array([l / max(len(s), 1) for l, s in zip(ll, sequences)])


class CountScorer:
    """A trivial baseline: log-likelihood under a k-th order Markov model of the input.

    Not a language model. It exists because every model result in this project is
    required to be paired with a trivial baseline, and because it makes the
    battery's own logic auditable: an order-1 Markov chain fitted on the real
    sequence MUST come out invariant to the dinucleotide shuffle and sensitive to
    nothing else, so if the battery reports anything else the battery is wrong.
    """

    alphabet = "dna"

    def __init__(self, order: int = 1, reference: str | None = None, pseudocount=1.0):
        self.order, self.reference, self.pseudocount = order, reference, pseudocount
        self.name = f"markov-order-{order}"
        self.tokenization = "single nucleotide"
        self.objective = f"order-{order} Markov MLE"

    def _fit(self, seq):
        k = self.order
        tab = {}
        for i in range(k, len(seq)):
            tab.setdefault(seq[i - k:i], {}).setdefault(seq[i], 0)
            tab[seq[i - k:i]][seq[i]] += 1
        return tab

    def score(self, sequences, progress=None):
        out = []
        for s in sequences:
            ref = self.reference or s
            tab, k = self._fit(ref), self.order
            tot = 0.0
            for i in range(k, len(s)):
                row = tab.get(s[i - k:i], {})
                denom = sum(row.values()) + 4 * self.pseudocount
                tot += float(np.log((row.get(s[i], 0) + self.pseudocount) / denom))
            out.append(tot / max(len(s) - k, 1))
        return np.array(out)


class PeriodicMarkovScorer:
    """A GeneMark-style 3-periodic Markov chain. The DNA-alphabet POSITIVE control.

    `CountScorer` is a negative control and can only ever be one: an order-k
    Markov chain with a single transition table is, by construction, unable to
    represent the reading frame, because it has no state that depends on
    position modulo three. Its null on the frameshift is therefore uninformative
    about whether the frameshift test *works*.

    This is the missing counterpart. It estimates a separate transition table
    for each codon position, exactly the device that made GeneMark (Borodovsky &
    McIninch 1993) a gene finder: coding DNA has period-three statistics, so
    P(base | context) differs at codon positions 1, 2 and 3. A model with three
    phase-indexed tables demonstrably represents the reading frame -- the phase
    is literally an index into its parameters -- so if the battery cannot detect
    frame representation in THIS model, the battery is the problem and no null it
    reports about a genomic LM can be interpreted.

    Two properties make it a fair instrument rather than a rigged one:

      * It is fitted on a corpus and scored leave-one-sequence-out, so it is a
        generative model of coding DNA rather than a fit to the string in front
        of it. `CountScorer` refits on its own input, which is why it comes out
        invariant to everything.
      * Phase is taken from the start of the sequence being scored and is never
        re-estimated. A gene finder would search over phases; doing that here
        would build in exactly the invariance under test. The convention is the
        same one every scorer in this file uses implicitly, namely that position
        zero of the input is position zero of the gene.

    `bind(keys)` fixes the key order the battery will call `score` in, which is
    what lets the held-out table be selected per sequence. The battery calls
    `score` once per condition with sequences in `sorted(sequences)` order.
    """

    alphabet = "dna"
    _IDX = {b: i for i, b in enumerate("ACGT")}

    def __init__(self, order: int = 2, period: int = 3, corpus: dict | None = None,
                 pseudocount: float = 1.0, holdout: bool = True):
        self.order, self.period, self.pseudocount = order, period, pseudocount
        self.holdout = holdout
        self.name = f"markov-order-{order}-periodic-{period}"
        self.tokenization = "single nucleotide"
        self.objective = (f"{period}-periodic order-{order} Markov MLE, "
                          f"{'leave-one-sequence-out' if holdout else 'pooled'} fit")
        self._keys: list | None = None
        self._counts: dict = {}
        self._total = None
        if corpus:
            self.fit(corpus)

    # ------------------------------------------------------------------ fitting
    def _count(self, seq: str) -> np.ndarray:
        """counts[phase, context, base]; phase is the position of the PREDICTED base."""
        k, P = self.order, self.period
        c = np.zeros((P, 4 ** k, 4), dtype=np.float64)
        for i in range(k, len(seq)):
            ctx = 0
            ok = True
            for j in range(i - k, i):
                v = self._IDX.get(seq[j])
                if v is None:
                    ok = False
                    break
                ctx = ctx * 4 + v
            b = self._IDX.get(seq[i])
            if ok and b is not None:
                c[i % P, ctx, b] += 1
        return c

    def fit(self, corpus: dict):
        self._counts = {k: self._count(v) for k, v in corpus.items()}
        self._total = sum(self._counts.values())
        return self

    def bind(self, keys):
        """Declare the order `score` will be called in, so hold-out can be applied."""
        self._keys = list(keys)
        missing = [k for k in self._keys if k not in self._counts]
        if self.holdout and missing:
            raise ValueError(f"no fitted counts for held-out keys {missing[:3]}")
        return self

    def _logtab(self, key):
        c = self._total
        if self.holdout and key is not None:
            c = c - self._counts[key]
        c = c + self.pseudocount
        return np.log(c / c.sum(-1, keepdims=True))

    # ------------------------------------------------------------------ scoring
    def score(self, sequences, progress=None):
        seqs = list(sequences)
        if self._total is None:
            raise RuntimeError("fit(corpus) before scoring")
        keys = self._keys if self._keys is not None else [None] * len(seqs)
        if len(keys) != len(seqs):
            raise ValueError(f"bound to {len(keys)} keys but got {len(seqs)} sequences")
        k, P = self.order, self.period
        out = []
        cache = {}
        for key, s in zip(keys, seqs):
            if key not in cache:
                cache[key] = self._logtab(key)
            lt = cache[key]
            tot = 0.0
            for i in range(k, len(s)):
                ctx = 0
                for j in range(i - k, i):
                    ctx = ctx * 4 + self._IDX[s[j]]
                tot += lt[i % P, ctx, self._IDX[s[i]]]
            out.append(tot / max(len(s) - k, 1))
        return np.array(out)
