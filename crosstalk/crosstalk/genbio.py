"""GenBio AIDO.Protein as a rung on the proxy ladder, with and without RAG.

Why the 3B and not GB.Protein-16B: the 16B weights are 13 shards of ~4.9 GB
(64 GB fp32, ~32 GB bf16) against 26 GB of unified memory. The 3B is 15 GB.

This model is retrieval-augmented and uses 2D rotary embeddings, so it needs
`position_ids` of shape (B, 2, L) even for a single sequence -- channel 0 is the
residue/column index, channel 1 is which row of the MSA a token came from.
Omitting them does not fall back gracefully; it raises a permute error deep in
the model. With one sequence the encoding is simply (arange(L), zeros(L)).

Enabling RAG is then the same mechanism with more rows: append MSA sequences
aligned to the query, drop gap columns from both the tokens and the position
ids, and let channel 1 count the rows. The MSAs come free from the Boltz runs,
which already searched ColabFold for every folded complex.

Four setup obstacles are documented in FINDINGS section 12; the important ones
are that `docstring-inheritance` must be pinned to 2.2.2 and that the weight
repos ship no tokenizer, whose vocab is bundled inside the package.
"""
from __future__ import annotations

import csv
import os
from pathlib import Path

import numpy as np
import torch

from .boltz import MUT_POSITIONS, PARD3

# The config reports max_position_embeddings=2048, but that field is vestigial
# under rope_2d: the rotary table is built for whatever length is passed, and
# both the model card and GenBio's own tutorial use 12.8K. Capping at 2048
# silently limits a 93-residue query to ~21 MSA rows instead of ~137.
MAX_TOKENS = 12800


def greedy_select(msa: list[str], num_seqs: int | None = None,
                  num_tokens: int | None = None, seed: int = 0) -> list[str]:
    """Pick a maximally diverse MSA subset by Hamming distance.

    Ported from GenBio's own tutorial utilities
    (github.com/genbio-ai/GB-Foundations-Tutorials, Notebooks/utils/misc.py).
    Taking the first N rows of an MSA takes the *closest* homologs, which are
    the least informative; this repeatedly adds whichever sequence is furthest
    from those already chosen.
    """
    import random

    from scipy.spatial.distance import cdist

    msa = list(msa)
    if seed is not None:
        random.Random(seed).shuffle(msa)
    if num_seqs is not None and len(msa) <= num_seqs:
        return msa

    arr = np.array([list(s) for s in msa], dtype=np.bytes_).view(np.uint8)
    all_idx = np.arange(len(msa))
    indices = [0]
    selected = [msa[0]]
    pair = np.zeros((0, len(msa)))
    for _ in range(len(msa) - 1):
        d = cdist(arr[indices[-1:]], arr, "hamming")
        pair = np.concatenate([pair, d])
        shifted = np.delete(pair, indices, axis=1).mean(0)
        idx = np.delete(all_idx, indices)[int(np.argmax(shifted))]
        indices.append(idx)
        selected.append(msa[idx])
        if num_seqs is not None and len(selected) >= num_seqs:
            break
        if num_tokens is not None:
            used = sum(len(s) - s.count("-") for s in selected)
            if used >= num_tokens:
                break
    return selected


def load_msa(csv_path: str | Path, max_rows: int = 20, query_len: int | None = None,
             diverse: bool = True, pool: int = 400) -> list[str]:
    """Read a Boltz MSA CSV as rows aligned to the query.

    Boltz writes one CSV per chain, row 0 being the query. Lowercase letters are
    a3m insertions relative to the query and are dropped, which is what makes
    every row exactly query length -- the alignment the model requires.
    """
    rows = list(csv.DictReader(open(csv_path)))
    if not rows:
        return []
    q = rows[0]["sequence"]
    n = query_len or len(q)
    out = []
    for r in rows[1:]:
        s = "".join(c for c in r["sequence"] if not c.islower())
        if len(s) == n:
            out.append(s)
        if len(out) >= (pool if diverse else max_rows):
            break
    if diverse and len(out) > max_rows:
        out = greedy_select(out, num_seqs=max_rows)
    return out[:max_rows]


class GenBioScorer:
    """Masked-marginal scoring with AIDO.Protein, optionally MSA-augmented."""

    def __init__(self, repo: str = "genbio-ai/AIDO.Protein-RAG-3B",
                 device: str | None = None, dtype=torch.float32):
        import modelgenerator
        from modelgenerator.huggingface_models.fm4bio import FM4BioForMaskedLM
        from modelgenerator.huggingface_models.fm4bio.tokenization_fm4bio import (
            FM4BioTokenizer,
        )

        self.name = repo
        vocab = os.path.join(os.path.dirname(modelgenerator.__file__),
                             "huggingface_models", "fm4bio", "vocab_protein.txt")
        self.tokenizer = FM4BioTokenizer(vocab_file=vocab)
        self.model = FM4BioForMaskedLM.from_pretrained(repo, torch_dtype=dtype)
        self.device = device or ("mps" if torch.backends.mps.is_available() else "cpu")
        self.model = self.model.to(self.device).eval()

    @staticmethod
    def _flatten(query: str, msa: list[str]) -> tuple[str, np.ndarray]:
        """Query plus MSA rows as one token stream, with 2D position ids.

        Mirrors modelgenerator's own construction: column index tiled across
        rows, row index repeated across columns, then both the tokens and the
        positions masked by where the alignment has no gap.
        """
        n = len(query)
        rows = [query] + list(msa)
        flat = np.array(list("".join(rows)))
        pos = np.stack([np.tile(np.arange(n), len(rows)),
                        np.repeat(np.arange(len(rows)), n)])
        keep = flat != "-"
        flat, pos = flat[keep], pos[:, keep]
        return "".join(flat[:MAX_TOKENS]), pos[:, :MAX_TOKENS]

    @torch.no_grad()
    def _logprobs_at(self, sequence: str, positions: list[int],
                     msa: list[str] | None = None) -> np.ndarray:
        msa = msa or []
        out = []
        for pos in positions:
            masked_q = sequence[: pos - 1] + self.tokenizer.mask_token + sequence[pos:]
            # build the stream with a placeholder so gap logic sees one character
            placeholder = sequence[: pos - 1] + "X" + sequence[pos:]
            flat, pos_ids = self._flatten(placeholder, msa)
            idx = pos - 1  # query is first and never gapped
            tokens = list(flat)
            tokens[idx] = self.tokenizer.mask_token
            enc = self.tokenizer("".join(tokens), return_tensors="pt",
                                 add_special_tokens=False)
            ids = enc["input_ids"].to(self.device)
            pid = torch.as_tensor(pos_ids, dtype=torch.long).unsqueeze(0).to(self.device)
            logits = self.model(input_ids=ids, position_ids=pid).logits
            at = (ids[0] == self.tokenizer.mask_token_id).nonzero()[0, 0]
            out.append(torch.log_softmax(logits[0, at].float(), dim=-1).cpu().numpy())
        return np.stack(out)

    def score_variants(self, variants: list[str], context: str | None = None,
                       msa: list[str] | None = None) -> np.ndarray:
        seq = context if context is not None else PARD3
        lp = self._logprobs_at(seq, list(MUT_POSITIONS), msa=msa)
        wt = [PARD3[p - 1] for p in MUT_POSITIONS]
        tid = {aa: self.tokenizer.convert_tokens_to_ids(aa)
               for aa in set("".join(variants) + "".join(wt))}
        scores = np.zeros(len(variants))
        for i, v in enumerate(variants):
            scores[i] = sum(lp[k, tid[aa]] - lp[k, tid[wt[k]]] for k, aa in enumerate(v))
        return scores
