"""Frozen-embedding extraction for the regulatory arm: NT-v2 50M and HyenaDNA.

Protocol notes that are load-bearing, because layer and pooling are a degree of
freedom the trivial baselines never got:

  * The candidate grid is fixed BEFORE any test number exists: three depths
    (half, three-quarters, final) crossed with three poolings. Nine configs.
  * Every config is scored on the test set and written to the CSV, and the one
    the headline uses is chosen by grouped cross-validation on TRAINING rows
    only, with locus clusters as groups -- the same mechanism, and the same
    groups, that pick the probe's C for a k-mer baseline.
  * A pre-registered default (final layer, mean pooling) is reported alongside,
    so a reader who distrusts any selection at all still has a number.

  * Every sequence in a task has the same length and contains no N, so token
    counts are constant and no padding occurs. The pooling masks are still built
    arithmetically from known lengths rather than with nonzero(), because MPS
    returns empty index tensors under memory pressure and the failure is silent.
"""
from __future__ import annotations

import numpy as np
import torch

LAYER_FRACS = (0.5, 0.75, 1.0)


def _device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    return "cuda" if torch.cuda.is_available() else "cpu"


def _pool(h: torch.Tensor, first: int, last_len: torch.Tensor, kind: str) -> torch.Tensor:
    """h: (B, T, D). Real content occupies [first, first+last_len).

    `first` is 1 for a model with a leading <cls> and 0 otherwise; `last_len` is
    the per-row count of real tokens, known analytically from the sequence length.
    """
    B, T, D = h.shape
    ar = torch.arange(T, device=h.device)[None, :]
    m = (ar >= first) & (ar < (first + last_len[:, None]))
    if kind == "mean":
        return (h * m[:, :, None]).sum(1) / m.sum(1, keepdim=True).clamp(min=1)
    if kind == "max":
        return h.masked_fill(~m[:, :, None], -1e4).max(1).values
    if kind == "cls":
        return h[:, 0]
    if kind == "last":
        idx = (first + last_len - 1).clamp(min=0)
        return h[torch.arange(B, device=h.device), idx]
    if kind.startswith("win"):
        # Coarse positional readout: mean over `nb` equal blocks of real tokens,
        # concatenated. Plain mean pooling is position-blind, which hands the
        # positional one-hot baseline an unearned advantage on splice sites and
        # TATA boxes; this keeps position at a feature count comparable to the
        # one-hot's 4L. Block edges come from the known token count, not nonzero().
        nb = int(kind[3:])
        assert int(last_len.min().item()) == int(last_len.max().item()), \
            "windowed pooling assumes a constant token count, which these tasks have"
        n = int(last_len[0].item())
        edges = [first + (n * j) // nb for j in range(nb + 1)]
        parts = [h[:, edges[j]:edges[j + 1]].mean(1) for j in range(nb)]
        return torch.cat(parts, dim=1)
    raise ValueError(kind)


class NTEmbedder:
    """Nucleotide Transformer v2 50M multi-species, frozen, hidden states only.

    Tokenisation is non-overlapping 6-mers with a leading <cls>; the leftover
    1-5 nt at the end become single-nucleotide tokens. Poolings are mean and max
    over the 6-mer tokens (the <cls> excluded, so the two pooled statistics are
    genuinely over sequence content) and the <cls> vector itself.
    """
    name = "NT-v2-50M"
    hf = "InstaDeepAI/nucleotide-transformer-v2-50m-multi-species"
    poolings = ("mean", "max", "cls")
    first = 1

    def __init__(self, device=None):
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from crosstalk.glm import _patch_transformers_for_nt_v2, _nt_config
        from transformers import AutoModelForMaskedLM, AutoTokenizer
        _patch_transformers_for_nt_v2()
        self.device = device or _device()
        self.tok = AutoTokenizer.from_pretrained(self.hf, trust_remote_code=True)
        cfg = _nt_config(self.hf)
        self.model = AutoModelForMaskedLM.from_pretrained(
            self.hf, config=cfg, trust_remote_code=True).to(self.device).eval()
        self.n_layers = cfg.num_hidden_layers
        self.dim = cfg.hidden_size
        self.layers = sorted({max(1, int(round(f * self.n_layers))) for f in LAYER_FRACS})

    @torch.no_grad()
    def batch(self, seqs):
        enc = self.tok(seqs, return_tensors="pt")
        ids = enc["input_ids"].to(self.device)
        T = ids.shape[1]
        # constant-length task: every row has T-1 real tokens after <cls>.
        lens = torch.full((ids.shape[0],), T - 1, device=self.device, dtype=torch.long)
        out = self.model(input_ids=ids, output_hidden_states=True)
        hs = out.hidden_states                      # (n_layers+1) x (B, T, D)
        return {(L, p): _pool(hs[L].float(), self.first, lens, p).cpu().numpy()
                for L in self.layers for p in self.poolings}


class HyenaEmbedder:
    """HyenaDNA small-32k, frozen, hidden states from forward hooks.

    Single-nucleotide tokens, a trailing [SEP], no <cls>. It is causal, so the
    final real position is the natural sequence-level readout and is included as
    a pooling alongside mean and max. Only four blocks exist, so the depth grid
    is layers 2, 3, 4; layer 4 is post-final-layernorm, i.e. last_hidden_state.
    """
    name = "HyenaDNA-small-32k"
    hf = "LongSafari/hyenadna-small-32k-seqlen-hf"
    poolings = ("mean", "max", "last")
    first = 0

    def __init__(self, device=None):
        from transformers import AutoModel, AutoTokenizer
        self.device = device or _device()
        self.tok = AutoTokenizer.from_pretrained(self.hf, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(self.hf, trust_remote_code=True).to(self.device).eval()
        self.blocks = self.model.backbone.layers
        self.n_layers = len(self.blocks)
        self.dim = self.model.backbone.ln_f.normalized_shape[0]
        self.layers = sorted({max(1, int(round(f * self.n_layers))) for f in LAYER_FRACS})
        self._buf = {}
        for i, b in enumerate(self.blocks):
            b.register_forward_hook(self._mk(i + 1))

    def _mk(self, idx):
        def hook(_m, _i, o):
            self._buf[idx] = o[0] if isinstance(o, tuple) else o
        return hook

    @torch.no_grad()
    def batch(self, seqs):
        ids = self.tok(seqs, return_tensors="pt")["input_ids"].to(self.device)
        T = ids.shape[1]
        lens = torch.full((ids.shape[0],), T - 1, device=self.device, dtype=torch.long)  # drop [SEP]
        self._buf.clear()
        out = self.model(input_ids=ids)
        hs = dict(self._buf)
        hs[self.n_layers] = out.last_hidden_state       # post ln_f
        return {(L, p): _pool(hs[L].float(), self.first, lens, p).cpu().numpy()
                for L in self.layers for p in self.poolings}


EMBEDDERS = {"nt": NTEmbedder, "hyena": HyenaEmbedder}


def embed_all(emb, seqs, batch_size, progress=None, tag=""):
    """(config -> (n, D) float16) for a whole list of equal-length sequences."""
    acc = {}
    for s in range(0, len(seqs), batch_size):
        out = emb.batch(seqs[s:s + batch_size])
        for k, v in out.items():
            acc.setdefault(k, []).append(v.astype(np.float16))
        if progress and (s // batch_size) % progress == 0:
            print(f"    [{tag}] {min(s + batch_size, len(seqs))}/{len(seqs)}", flush=True)
    return {f"L{k[0]}_{k[1]}": np.concatenate(v) for k, v in acc.items()}
