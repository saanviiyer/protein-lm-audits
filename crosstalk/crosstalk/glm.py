"""Genomic language models as specificity proxies: the same audit, one modality down.

Sections 9 and 12 established that protein-LM likelihood is anti-predictive of
binding specificity on this landscape -- for ESM-2 at four scales and for a 3B
model from a different lab, with and without retrieval. Every one of those rungs
is a protein model reading a protein sequence, so the obvious residual
explanation is that something about protein-LM pretraining is at fault.

This module tests the claim one modality down, on the DNA that encodes the very
same variants. A genomic LM shares almost nothing with ESM-2: a different corpus
(whole genomes, not UniRef), a different alphabet (4 letters, not 20), a
different tokenization (6-mers or single nucleotides, not residues), and in the
HyenaDNA case a different training objective (autoregressive, not masked). If
likelihood is still anti-predictive there, the finding is about evolutionary
likelihood as such rather than about protein language models.

Two things are possible here that are not possible on the protein rung:

  THE REAL OPERON      ParD3 and ParE3 are adjacent genes -- in fact they overlap
                       by 11 nt -- so a partner-aware genomic context is the
                       locus the organism actually carries and the model actually
                       saw in training. The protein rung had to invent a 25-glycine
                       linker to put two chains in one context; nothing is invented
                       here except the deliberately non-natural ParD3:ParE2 control.

  THE SYNONYMOUS FLOOR A protein variant does not determine its DNA. Many
                       synonymous encodings translate to exactly the same protein
                       and therefore have exactly the same measured binding
                       fitness -- the assay reads protein binding, so synonymous
                       choice cannot change the label. Any variance a genomic LM
                       assigns across synonymous encodings of one variant is
                       therefore provably specificity-irrelevant. Comparing that
                       within-variant spread to the between-variant spread gives a
                       noise ceiling on *any* genomic-LM protein-fitness proxy,
                       which the protein rung has no way to measure.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from .boltz import MUT_POSITIONS, PARD3

DATA = Path(__file__).resolve().parents[1] / "data" / "cds"

BASES = "TCAG"
AAS = "FFLLSSSSYY**CC*WLLLLPPPPHHQQRRRRIIIMTTTTNNKKSSRRVVVVAAAADDEEGGGG"
CODON_TABLE = {b1 + b2 + b3: AAS[i] for i, (b1, b2, b3) in enumerate(
    (a, b, c) for a in BASES for b in BASES for c in BASES)}
SYNONYMOUS: dict[str, list[str]] = defaultdict(list)
for _c, _a in CODON_TABLE.items():
    SYNONYMOUS[_a].append(_c)
SYNONYMOUS = dict(SYNONYMOUS)


def translate(dna: str) -> str:
    dna = dna.upper()
    return "".join(CODON_TABLE.get(dna[i:i + 3], "X") for i in range(0, len(dna) - 2, 3))


def load_cds() -> dict:
    return json.loads((DATA / "cds.json").read_text())


def load_context() -> dict:
    return json.loads((DATA / "genomic_context.json").read_text())


def codon_usage() -> dict[str, int]:
    """Genome-wide codon counts for M. opportunistum WSM2075 (6,508 CDS)."""
    return json.loads((DATA / "codon_usage_CP002279.json").read_text())


def preferred_codons() -> dict[str, str]:
    """The most-used codon per amino acid in this organism's own genome."""
    use = codon_usage()
    return {aa: max(cods, key=lambda c: use.get(c, 0)) for aa, cods in SYNONYMOUS.items()}


def usage_weights(aa: str) -> np.ndarray:
    use = codon_usage()
    w = np.array([use.get(c, 0) for c in SYNONYMOUS[aa]], float)
    return w / w.sum() if w.sum() > 0 else np.ones(len(w)) / len(w)


# ---------------------------------------------------------------- variant CDS

def _codon_slice(cds: str, aa_pos: int) -> slice:
    """1-based amino-acid position -> slice of its codon in the CDS."""
    return slice((aa_pos - 1) * 3, (aa_pos - 1) * 3 + 3)


def variant_cds(variant: str, wt_cds: str, policy: str = "preferred",
                rng: np.random.Generator | None = None) -> str:
    """The ParD3 coding sequence carrying a 3-letter variant code.

    policy:
      "preferred"  the organism's most-used codon for each substituted residue.
                   Deterministic: one CDS, and therefore one score, per variant.
      "usage"      sample synonymous codons in proportion to genome usage.
      "uniform"    sample synonymous codons uniformly -- the widest synonymous
                   ensemble, used to bound the specificity-irrelevant variance.

    Positions that are not substituted keep the wild-type codon the organism
    actually uses, so the only DNA that ever changes is the codon under test.
    """
    if len(variant) != len(MUT_POSITIONS):
        raise ValueError(f"variant must be {len(MUT_POSITIONS)} residues, got {variant!r}")
    pref = preferred_codons()
    out = list(wt_cds)
    for aa, pos in zip(variant, MUT_POSITIONS):
        if PARD3[pos - 1] == aa:
            continue                      # unchanged residue keeps its native codon
        if policy == "preferred":
            codon = pref[aa]
        else:
            if rng is None:
                raise ValueError("sampling policies need an rng")
            cands = SYNONYMOUS[aa]
            p = usage_weights(aa) if policy == "usage" else None
            codon = cands[rng.choice(len(cands), p=p)]
        out[_codon_slice(wt_cds, pos)] = list(codon)
    seq = "".join(out)
    assert translate(seq)[:len(PARD3)] == "".join(
        variant[MUT_POSITIONS.index(i + 1)] if (i + 1) in MUT_POSITIONS else PARD3[i]
        for i in range(len(PARD3))), "variant CDS does not translate to its variant"
    return seq


def synonymous_ensemble(variant: str, wt_cds: str, n: int, seed: int = 0,
                        policy: str = "uniform") -> list[str]:
    """n distinct synonymous encodings of the same protein variant.

    Every member translates identically, so every member carries the identical
    measured fitness. Score variance across this set is pure specificity-
    irrelevant variance.
    """
    rng = np.random.default_rng(seed)
    seen, out = set(), []
    for _ in range(n * 40):
        s = variant_cds(variant, wt_cds, policy=policy, rng=rng)
        if s not in seen:
            seen.add(s)
            out.append(s)
        if len(out) == n:
            break
    return out


def _device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    return "cuda" if torch.cuda.is_available() else "cpu"


# ------------------------------------------------------------------- scorers

def _patch_transformers_for_nt_v2() -> None:
    """Nucleotide Transformer v2 ships remote code written against transformers <5.

    Its bundled `modeling_esm.py` imports `find_pruneable_heads_and_indices` from
    `transformers.pytorch_utils`, which transformers 5.x removed (it kept the
    sibling `prune_linear_layer`). The import fails at load time, so the model
    cannot be constructed at all -- and the failure looks like a missing model
    rather than a version skew. The function is only reachable from head pruning,
    which nothing here calls, but it must exist for the module to import.

    Recorded because nothing documents it: NT-v2 is uninstallable on current
    transformers without this, on any host.
    """
    import transformers.pytorch_utils as pu
    if hasattr(pu, "find_pruneable_heads_and_indices"):
        return

    def find_pruneable_heads_and_indices(heads, n_heads, head_size, already_pruned_heads):
        mask = torch.ones(n_heads, head_size)
        heads = set(heads) - already_pruned_heads
        for head in heads:
            head = head - sum(1 if h < head else 0 for h in already_pruned_heads)
            mask[head] = 0
        mask = mask.view(-1).contiguous().eq(1)
        index = torch.arange(len(mask))[mask].long()
        return heads, index

    pu.find_pruneable_heads_and_indices = find_pruneable_heads_and_indices


# transformers 5.x dropped several attributes that older remote code reads off the
# config object directly. They are all defaults for an encoder-only masked LM, so
# restoring them is a compatibility fix and not a modelling choice.
_NT_CONFIG_DEFAULTS = dict(is_decoder=False, add_cross_attention=False,
                           chunk_size_feed_forward=0, tie_word_embeddings=True,
                           use_cache=False, pruned_heads={})


def _nt_config(model_name: str):
    from transformers import AutoConfig
    cfg = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    for k, v in _NT_CONFIG_DEFAULTS.items():
        if not hasattr(cfg, k):
            setattr(cfg, k, v)
    return cfg


class NTScorer:
    """Nucleotide Transformer: masked scoring over non-overlapping 6-mer tokens.

    The ParD3 CDS is 282 nt = 47 tokens with no offset, so each token is exactly
    two codons and the three mutated codons fall in three distinct tokens. That
    makes the masked-marginal shortcut from the protein rung transfer exactly:
    mask those three tokens once, read the log-probabilities over the 4,096
    6-mers, and every variant's score is a sum over three columns -- three
    forward passes for a whole landscape, the same as ESM-2.

    The one thing that does not transfer is that a token is not a codon. Each
    masked token also contains one unsubstituted neighbouring codon, so the score
    is conditioned on a specific synonymous encoding of that neighbour as well.
    That is a genuine difference from the protein rung, and it is the reason the
    synonymous ensemble is not optional here.
    """

    def __init__(self, model_name: str = "InstaDeepAI/nucleotide-transformer-v2-50m-multi-species",
                 device: str | None = None):
        from transformers import AutoModelForMaskedLM, AutoTokenizer
        _patch_transformers_for_nt_v2()
        self.name = model_name
        self.device = device or _device()
        self.tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForMaskedLM.from_pretrained(
            model_name, config=_nt_config(model_name),
            trust_remote_code=True).to(self.device).eval()
        self.k = 6

    def token_index(self, cds_offset: int, aa_pos: int) -> tuple[int, int]:
        """(token position in the input, codon's slot inside that token)."""
        nt = cds_offset + (aa_pos - 1) * 3
        return nt // self.k, (nt % self.k) // 3

    @torch.no_grad()
    def _masked_logprobs(self, seq: str, token_positions: list[int]) -> np.ndarray:
        """Mask each listed token in turn; return (n_positions, vocab) log-probs."""
        out = []
        ids0 = self.tok(seq, return_tensors="pt")["input_ids"]
        for tp in token_positions:
            ids = ids0.clone()
            ids[0, tp + 1] = self.tok.mask_token_id      # +1 for <cls>
            logits = self.model(input_ids=ids.to(self.device)).logits[0]
            out.append(torch.log_softmax(logits[tp + 1].float(), -1).cpu().numpy())
        return np.stack(out)

    def score_variants_masked_marginal(self, variants: list[str], seq: str,
                                       cds_offset: int = 0) -> np.ndarray:
        """Masked-marginal score, relative to wild type, for each variant.

        `seq` is the full context (the ParD3 CDS alone, or the operon), and
        `cds_offset` says where the ParD3 CDS starts inside it.
        """
        tps, slots = zip(*[self.token_index(cds_offset, p) for p in MUT_POSITIONS])
        lp = self._masked_logprobs(seq, list(tps))
        wt_tokens = [seq[tp * self.k:(tp + 1) * self.k] for tp in tps]
        vocab = self.tok.get_vocab()

        def tok_id(s):
            return vocab.get(s, self.tok.unk_token_id)

        wt_lp = [lp[i, tok_id(wt_tokens[i])] for i in range(len(tps))]
        pref = preferred_codons()
        scores = np.zeros(len(variants))
        for i, v in enumerate(variants):
            s = 0.0
            for k, (aa, pos) in enumerate(zip(v, MUT_POSITIONS)):
                codon = seq[cds_offset + (pos - 1) * 3:cds_offset + (pos - 1) * 3 + 3] \
                    if PARD3[pos - 1] == aa else pref[aa]
                t = list(wt_tokens[k])
                t[slots[k] * 3:slots[k] * 3 + 3] = list(codon)
                s += lp[k, tok_id("".join(t))] - wt_lp[k]
            scores[i] = s
        return scores

    @torch.no_grad()
    def pseudo_likelihood(self, sequences: list[str], progress: int | None = None) -> np.ndarray:
        """Sum over tokens of log p(token | all other tokens): one pass per token.

        The masked-marginal score is additive over the three mutated tokens and
        cannot see interactions between them. This can, at 47 passes per sequence
        instead of three for the whole landscape -- the same trade, and the same
        control, as the protein rung's full pseudo-likelihood.
        """
        out = np.zeros(len(sequences))
        for n, seq in enumerate(sequences):
            ids0 = self.tok(seq, return_tensors="pt")["input_ids"]
            n_tok = ids0.shape[1] - 1                    # exclude <cls>
            batch = ids0.repeat(n_tok, 1)
            for j in range(n_tok):
                batch[j, j + 1] = self.tok.mask_token_id
            total = 0.0
            for s in range(0, n_tok, 32):
                chunk = batch[s:s + 32].to(self.device)
                logits = self.model(input_ids=chunk).logits
                for r in range(chunk.shape[0]):
                    j = s + r
                    lp = torch.log_softmax(logits[r, j + 1].float(), -1)
                    total += float(lp[ids0[0, j + 1]])
            out[n] = total
            if progress and (n + 1) % progress == 0:
                print(f"    {n+1}/{len(sequences)} sequences", flush=True)
        return out


class HyenaScorer:
    """HyenaDNA: autoregressive log-likelihood over single nucleotides.

    Included because it differs from both the protein rung and the Nucleotide
    Transformer on the two axes that could otherwise explain a shared result:
    tokenization (single nucleotide, so codons are never straddled) and training
    objective (next-token prediction rather than masked reconstruction). One
    forward pass scores a whole sequence.
    """

    def __init__(self, model_name: str = "LongSafari/hyenadna-small-32k-seqlen-hf",
                 device: str | None = None):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.name = model_name
        self.device = device or _device()
        self.tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, trust_remote_code=True).to(self.device).eval()

    @torch.no_grad()
    def log_likelihood(self, sequences: list[str], progress: int | None = None) -> np.ndarray:
        out = np.zeros(len(sequences))
        for n, seq in enumerate(sequences):
            ids = self.tok(seq, return_tensors="pt")["input_ids"].to(self.device)
            logits = self.model(input_ids=ids).logits.float()
            lp = torch.log_softmax(logits[0, :-1], -1)
            tgt = ids[0, 1:]
            out[n] = float(lp.gather(-1, tgt[:, None]).sum())
            if progress and (n + 1) % progress == 0:
                print(f"    {n+1}/{len(sequences)} sequences", flush=True)
        return out
