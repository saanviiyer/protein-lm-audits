"""Protein language models as specificity proxies, and why most of them cannot be.

Every variant in the ParD3 landscape differs only at positions 61/64/80, so a
masked-marginal score for the whole landscape costs three forward passes: mask
those positions once, read the log-probabilities, and every variant's score is a
sum over the three columns. That makes a full proxy ladder cheap.

The ladder has two rungs, and the distinction is the point:

  PARTNER-BLIND    score the ParD3 variant on its own. One sequence gets one
                   number, so the same variant scores identically whether the
                   question is "binds ParE3" or "avoids ParE2". Such a proxy
                   cannot represent specificity at all -- not "does so poorly",
                   but has no channel to express it. Its expected AUC on
                   specific-versus-promiscuous is 0.5 by construction.

  PARTNER-AWARE    mask the same positions inside a ParD3:ParE concatenate. The
                   context now contains the partner, so the score depends on it
                   and specificity becomes expressible. Still three passes per
                   partner.

Whether the partner-aware rung actually beats chance is the empirical question.
"""
from __future__ import annotations

import numpy as np
import torch

from .boltz import MUT_POSITIONS, PARD3, PARTNERS

LINKER = "G" * 25  # flexible spacer so the two chains are not read as one domain


def variant_full(variant: str) -> str:
    """The complete ParD3 sequence carrying a 3-letter variant code."""
    s = list(PARD3)
    for aa, pos in zip(variant, MUT_POSITIONS):
        s[pos - 1] = aa
    return "".join(s)


def _device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    return "cuda" if torch.cuda.is_available() else "cpu"


class ESMScorer:
    """Masked-marginal scoring at the three mutated positions."""

    def __init__(self, model_name: str = "facebook/esm2_t33_650M_UR50D",
                 device: str | None = None):
        from transformers import AutoModelForMaskedLM, AutoTokenizer
        self.name = model_name
        self.device = device or _device()
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForMaskedLM.from_pretrained(model_name).to(self.device).eval()

    @torch.no_grad()
    def _logprobs_at(self, sequence: str, positions: list[int]) -> np.ndarray:
        """Mask each position in turn; return (len(positions), 20) log-probs.

        `positions` are 1-based indices into `sequence`.
        """
        out = []
        for pos in positions:
            masked = sequence[: pos - 1] + self.tok.mask_token + sequence[pos:]
            enc = self.tok(masked, return_tensors="pt").to(self.device)
            logits = self.model(**enc).logits[0]
            mask_at = (enc["input_ids"][0] == self.tok.mask_token_id).nonzero()[0, 0]
            lp = torch.log_softmax(logits[mask_at].float(), dim=-1)
            out.append(lp.cpu().numpy())
        return np.stack(out)

    def score_variants(self, variants: list[str], context: str | None = None) -> np.ndarray:
        """Masked-marginal score for each variant.

        context=None scores ParD3 alone (partner-blind). Otherwise `context` is
        the full concatenate, whose first len(PARD3) residues are the ParD3
        chain, and scoring is partner-aware.
        """
        seq = context if context is not None else PARD3
        lp = self._logprobs_at(seq, list(MUT_POSITIONS))
        wt = [PARD3[p - 1] for p in MUT_POSITIONS]
        tok_id = {}
        for aa in set("".join(variants) + "".join(wt)):
            tok_id[aa] = self.tok.convert_tokens_to_ids(aa)
        scores = np.zeros(len(variants))
        for i, v in enumerate(variants):
            s = 0.0
            for k, aa in enumerate(v):
                s += lp[k, tok_id[aa]] - lp[k, tok_id[wt[k]]]
            scores[i] = s
        return scores


def complex_context(partner: str) -> str:
    """ParD3 (wild type at the masked sites) joined to a partner chain."""
    return PARD3 + LINKER + PARTNERS[partner]


@torch.no_grad()
def pseudo_likelihood(scorer: "ESMScorer", sequences: list[str], batch_size: int = 24,
                      progress=None) -> np.ndarray:
    """Full pseudo-log-likelihood: sum_i log p(x_i | x_without_i).

    The masked-marginal score used elsewhere is additive over the mutated sites
    by construction, so it cannot express epistasis between them. This does not:
    every position is masked in turn in the variant's own context, at a cost of
    len(sequence) forward passes per variant instead of three for the whole
    landscape. The masked copies of one sequence are batched together.
    """
    out = np.zeros(len(sequences))
    for n, seq in enumerate(sequences):
        total = 0.0
        for start in range(0, len(seq), batch_size):
            chunk = list(range(start, min(start + batch_size, len(seq))))
            masked = [seq[:i] + scorer.tok.mask_token + seq[i + 1:] for i in chunk]
            enc = scorer.tok(masked, return_tensors="pt", padding=True).to(scorer.device)
            logits = scorer.model(**enc).logits
            ids = enc["input_ids"]
            for row, i in enumerate(chunk):
                at = (ids[row] == scorer.tok.mask_token_id).nonzero()[0, 0]
                lp = torch.log_softmax(logits[row, at].float(), dim=-1)
                total += float(lp[scorer.tok.convert_tokens_to_ids(seq[i])])
        out[n] = total
        if progress and (n + 1) % progress == 0:
            print(f"    {n+1}/{len(sequences)} sequences", flush=True)
    return out


@torch.no_grad()
def pseudo_likelihood_positions(scorer: "ESMScorer", seq: str,
                                batch_size: int = 24) -> np.ndarray:
    """Per-position log p(x_i | x_without_i) for one sequence.

    Identical masking to `pseudo_likelihood`; `.sum()` of the return value equals
    `pseudo_likelihood(scorer, [seq])[0]`. Separated out so a corruption's cost
    can be localised along the chain rather than only totalled.
    """
    out = np.zeros(len(seq))
    for start in range(0, len(seq), batch_size):
        chunk = list(range(start, min(start + batch_size, len(seq))))
        masked = [seq[:i] + scorer.tok.mask_token + seq[i + 1:] for i in chunk]
        enc = scorer.tok(masked, return_tensors="pt", padding=True).to(scorer.device)
        logits = scorer.model(**enc).logits
        ids = enc["input_ids"]
        for row, i in enumerate(chunk):
            at = (ids[row] == scorer.tok.mask_token_id).nonzero()[0, 0]
            lp = torch.log_softmax(logits[row, at].float(), dim=-1)
            out[i] = float(lp[scorer.tok.convert_tokens_to_ids(seq[i])])
    return out
