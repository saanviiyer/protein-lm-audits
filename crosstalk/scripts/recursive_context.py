#!/usr/bin/env python3
"""Recursive-LM decomposition for a short-context genomic model (library).

Zhang, Kraska & Khattab (arXiv:2512.24601) treat a long prompt as an external
environment: the model never sees it whole, it examines pieces, calls itself on
each piece, and combines the returned answers. This module is the genomic
analogue for the ParD3 specificity task of FINDINGS section 19.

THE DECOMPOSITION, FIXED BEFORE ANY RUN

  query        the 282 nt ParD3 CDS carrying the variant codons. Every recursive
               call must contain it, because the answer being asked for is a
               masked-marginal log-ratio at three codons inside it.
  environment  the flanking chromosome, F nt upstream and F nt downstream of the
               CDS on the oriented CP002279 contig.
  chunking     C = 1440 nt per side, non-overlapping, walking outward from the
               CDS. Chunk i (0-based) is upstream[-(i+1)C : -iC] paired with
               downstream[iC : (i+1)C]. F = kC gives exactly k chunks.
  call i       input_i = U_i ++ CDS ++ D_i, scored by the model's ordinary
               masked-marginal scorer with cds_offset = C. C is a multiple of 6,
               so codon-to-6mer-token alignment is identical in every call and
               identical to the direct scorer.
  aggregation  UNWEIGHTED MEAN over the k per-chunk score vectors. This is the
               pre-registered headline rule. Median, max, min, nearest-chunk
               (i=0 only) and farthest-chunk are also computed and written to the
               CSV, and are never allowed to become the headline.

Chunk 0 is the true contiguous locus at flank C, so at F = C the recursive
pipeline is exactly the direct scorer -- that is the exact faithfulness test.
For i > 0 the chunk is real chromosome spliced against the CDS at a junction
that does not occur in nature; that is the price of making a 12 kb model consume
a 40 kb locus, and it is what the shuffled control is there to price.
"""
from __future__ import annotations

import numpy as np

COMP = str.maketrans("ACGT", "TGCA")
CHUNK = 1440          # nt per side per recursive call; multiple of 6
AGGREGATIONS = ("mean", "median", "max", "min", "nearest", "farthest")


def chunk_inputs(oriented: str, start: int, end: int, flank: int,
                 chunk: int = CHUNK, transform=None):
    """[(input_seq, cds_offset), ...] for one recursive sweep at this flank.

    `transform(seq, tag)` is applied to each flank chunk before splicing; used
    for the shuffled controls. The CDS is never transformed.
    """
    cds = oriented[start:end]
    if flank == 0:
        return [(cds, 0)]
    k, rem = divmod(flank, chunk)
    assert rem == 0, "flank must be a whole number of chunks"
    out = []
    for i in range(k):
        u = oriented[start - (i + 1) * chunk: start - i * chunk]
        d = oriented[end + i * chunk: end + (i + 1) * chunk]
        assert len(u) == chunk and len(d) == chunk
        if transform is not None:
            u = transform(u, f"u{i}")
            d = transform(d, f"d{i}")
        out.append((u + cds + d, chunk))
    return out


def aggregate(per_chunk: np.ndarray, rule: str) -> np.ndarray:
    """per_chunk is (n_chunks, n_variants). Chunk 0 is the innermost."""
    if rule == "mean":
        return per_chunk.mean(0)
    if rule == "median":
        return np.median(per_chunk, 0)
    if rule == "max":
        return per_chunk.max(0)
    if rule == "min":
        return per_chunk.min(0)
    if rule == "nearest":
        return per_chunk[0]
    if rule == "farthest":
        return per_chunk[-1]
    raise ValueError(rule)


def mono_shuffle(seq: str, seed: int) -> str:
    rng = np.random.default_rng(seed)
    a = np.array(list(seq))
    rng.shuffle(a)
    return "".join(a)


def dinuc_shuffle(seq: str, seed: int) -> str:
    """Altschul-Erikson: preserves exact dinucleotide counts (hence mononucleotide
    composition too). Euler-path construction with a random last-edge tree."""
    rng = np.random.default_rng(seed)
    letters = sorted(set(seq))
    if len(seq) < 3 or len(letters) < 2:
        return seq
    edges = {c: [] for c in letters}
    for a, b in zip(seq, seq[1:]):
        edges[a].append(b)
    first, last = seq[0], seq[-1]
    for _ in range(200):
        # random last edge per vertex, must form a tree rooted at `last`
        le = {}
        for c in letters:
            if c == last:
                continue
            le[c] = edges[c][rng.integers(len(edges[c]))]
        ok = True
        for c in le:
            seen, v = set(), c
            while v != last:
                if v in seen:
                    ok = False
                    break
                seen.add(v)
                v = le.get(v)
                if v is None:
                    ok = False
                    break
            if not ok:
                break
        if ok:
            break
    else:
        return mono_shuffle(seq, seed)
    rest = {}
    for c in letters:
        e = list(edges[c])
        if c in le:
            e.remove(le[c])
        rng.shuffle(e)
        rest[c] = e + ([le[c]] if c in le else [])
    out, v, idx = [first], first, {c: 0 for c in letters}
    for _ in range(len(seq) - 1):
        nxt = rest[v][idx[v]]
        idx[v] += 1
        out.append(nxt)
        v = nxt
    return "".join(out)


# ------------------------------------------------------------------ Markov ref

BASE = {"A": 0, "C": 1, "G": 2, "T": 3}
INV = "ACGT"


class MarkovScorer:
    """Order-n Markov chain with the exact masked-marginal analogue of NTScorer.

    Section 32 found NT-v2 50M beats an order-5 chain by about two points of
    per-nucleotide accuracy on the pretraining objective itself. Running that
    chain through the identical recursive pipeline says whether any ceiling seen
    here belongs to the recursion or simply to the model.

    log p(6mer token | rest of the sequence) is exact for a Markov chain: the
    token's own six emissions plus the n following emissions that depend on it,
    normalised over all 4096 tokens. Everything outside that span cancels.
    """

    def __init__(self, genome: str, order: int = 5, exclude: tuple[int, int] | None = None,
                 pseudocount: float = 1.0):
        self.order = order
        self.k = 6
        self.name = f"markov-order{order}"
        counts = np.full((4 ** order, 4), pseudocount)
        segs = [genome]
        if exclude is not None:
            lo, hi = exclude
            segs = [genome[:lo], genome[hi:]]
        for seg in segs:
            arr = np.array([BASE.get(c, -1) for c in seg], np.int64)
            ok = arr >= 0
            if len(arr) <= order:
                continue
            ctx = np.zeros(len(arr) - order, np.int64)
            valid = np.ones(len(arr) - order, bool)
            for j in range(order):
                ctx = ctx * 4 + np.maximum(arr[j:len(arr) - order + j], 0)
                valid &= ok[j:len(arr) - order + j]
            nxt = arr[order:]
            valid &= ok[order:]
            np.add.at(counts, (ctx[valid], nxt[valid]), 1.0)
        self.logp = np.log(counts / counts.sum(1, keepdims=True))
        # marginal back-off tables for contexts shorter than `order`
        self.backoff = []
        c = counts
        for m in range(order):
            blocks = c.reshape(4 ** (order - m - 1), 4 ** (m + 1), 4).sum(0) if m < order else c
            self.backoff.append(np.log(blocks / blocks.sum(1, keepdims=True)))
        # backoff[m] is indexed by the base-4 code of the LAST (m+1) context chars
        self._codes6 = np.arange(4096)
        d = np.zeros((4096, 6), np.int64)
        t = self._codes6.copy()
        for j in range(5, -1, -1):
            d[:, j] = t % 4
            t //= 4
        self._digits6 = d

    def _cond(self, ctx_codes: np.ndarray, m: int) -> np.ndarray:
        """rows of log p(next | context) for contexts of length m (m<=order)."""
        if m >= self.order:
            return self.logp[ctx_codes]
        return self.backoff[m - 1][ctx_codes] if m >= 1 else \
            np.broadcast_to(self.backoff[0].mean(0), (len(ctx_codes), 4))

    def token_index(self, cds_offset: int, aa_pos: int) -> tuple[int, int]:
        nt = cds_offset + (aa_pos - 1) * 3
        return nt // self.k, (nt % self.k) // 3

    def _masked_logprobs(self, seq: str, token_positions: list[int]) -> np.ndarray:
        n, o = self.order, self.order
        arr = np.array([BASE.get(c, 0) for c in seq], np.int64)
        out = []
        for tp in token_positions:
            j = tp * self.k
            left = arr[max(0, j - o):j]
            right = arr[j + self.k: j + self.k + o]
            # build the running context code for all 4096 candidates
            ll = np.zeros(4096)
            hist = np.tile(left, (4096, 1)) if len(left) else np.zeros((4096, 0), np.int64)
            full = np.concatenate([hist, self._digits6,
                                   np.tile(right, (4096, 1)) if len(right) else
                                   np.zeros((4096, 0), np.int64)], 1)
            s = len(left)
            for e in range(s, s + self.k + len(right)):
                m = min(o, e)
                ctx = np.zeros(4096, np.int64)
                for c in range(e - m, e):
                    ctx = ctx * 4 + full[:, c]
                ll += self._cond(ctx, m)[np.arange(4096), full[:, e]]
            out.append(ll - _logsumexp(ll))
        return np.stack(out)

    def score_variants_masked_marginal(self, variants, seq: str, cds_offset: int = 0):
        from crosstalk.boltz import MUT_POSITIONS, PARD3
        from crosstalk.glm import preferred_codons
        tps, slots = zip(*[self.token_index(cds_offset, p) for p in MUT_POSITIONS])
        lp = self._masked_logprobs(seq, list(tps))
        wt_tokens = [seq[tp * self.k:(tp + 1) * self.k] for tp in tps]
        pref = preferred_codons()

        def code(s):
            v = 0
            for ch in s:
                v = v * 4 + BASE.get(ch, 0)
            return v

        wt_lp = [lp[i, code(wt_tokens[i])] for i in range(len(tps))]
        scores = np.zeros(len(variants))
        for i, v in enumerate(variants):
            s = 0.0
            for kk, (aa, pos) in enumerate(zip(v, MUT_POSITIONS)):
                codon = seq[cds_offset + (pos - 1) * 3: cds_offset + (pos - 1) * 3 + 3] \
                    if PARD3[pos - 1] == aa else pref[aa]
                t = list(wt_tokens[kk])
                t[slots[kk] * 3:slots[kk] * 3 + 3] = list(codon)
                s += lp[kk, code("".join(t))] - wt_lp[kk]
            scores[i] = s
        return scores


def _logsumexp(x):
    m = x.max()
    return m + np.log(np.exp(x - m).sum())
