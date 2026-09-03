"""The proxies under audit, plus the trivial baselines they must beat.

A proxy scores a variant without measuring it. The learned proxy here is the
standard zero-shot protein-language-model fitness score (masked marginals in
wild-type context, Meier et al. 2021) as used throughout ProteinGym and in the
Adaptyv design competitions, plus two variants that relax its additivity. The
baselines are deliberately dumb: a 1992 substitution matrix, a hydropathy scale,
and the mutation count. A learned proxy that cannot beat the mutation count is
not measuring biology.
"""

import numpy as np


def apply_muts(seq, muts, offset=0):
    """Return the mutant sequence, or None if any mutation disagrees with seq."""
    s = list(seq)
    for wt, pos, mt in muts:
        i = pos - 1 + offset
        if not (0 <= i < len(s)) or s[i] != wt:
            return None
        s[i] = mt
    return "".join(s)

KD_HYDROPATHY = {
    "A": 1.8, "R": -4.5, "N": -3.5, "D": -3.5, "C": 2.5, "Q": -3.5, "E": -3.5,
    "G": -0.4, "H": -3.2, "I": 4.5, "L": 3.8, "K": -3.9, "M": 1.9, "F": 2.8,
    "P": -1.6, "S": -0.8, "T": -0.7, "W": -0.9, "Y": -1.3, "V": 4.2,
}


def _blosum62():
    from Bio.Align import substitution_matrices

    return substitution_matrices.load("BLOSUM62")


def blosum_score(muts):
    """Sum of BLOSUM62 exchange scores. More negative = more disruptive."""
    if not muts:
        return 0.0
    m = _blosum62()
    tot = 0.0
    for wt, _, mt in muts:
        try:
            tot += float(m[wt, mt])
        except KeyError:
            pass
    return tot


def hydropathy_shift(muts):
    """Summed Kyte-Doolittle hydropathy change across mutated positions."""
    return float(sum(KD_HYDROPATHY.get(mt, 0.0) - KD_HYDROPATHY.get(wt, 0.0)
                     for wt, _, mt in (muts or [])))


class ESM2Marginals:
    """Zero-shot scorers over ESM-2, in three flavours.

    ``score`` (masked marginals, wild-type context) masks position i in the WILD
    TYPE and reads off log p(mut_i) - log p(wt_i). A variant is the sum of those
    per-position terms. This is the protocol behind the ESM numbers in
    ProteinGym, and the one a design pipeline reaches for -- but each term is
    blind to the variant's other mutations, so it is strictly additive.

    ``score_mutant_marginals`` masks position i in the MUTANT sequence instead,
    so every term sees the other substitutions. Still a sum over mutations, but
    each term is context-aware.

    ``score_sequence_loglik`` takes the whole-sequence log-likelihood of the
    mutant minus that of the wild type in one unmasked pass each. It is not a
    sum over mutated positions at all, so it is the only one of the three that
    is genuinely non-additive.
    """

    def __init__(self, model_name="facebook/esm2_t33_650M_UR50D", device=None, batch_size=8):
        import torch
        from transformers import AutoModelForMaskedLM, AutoTokenizer

        self.torch = torch
        if device is None:
            device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.device = device
        self.batch_size = batch_size
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForMaskedLM.from_pretrained(model_name).to(device).eval()
        self._cache = {}

    AA_ORDER = list(KD_HYDROPATHY.keys())

    def logprobs_at(self, seq, positions):
        """{position: 20-vector of log p(aa | seq with that position masked)}.

        Only the requested positions are masked. Across this corpus fewer than
        5% of residues are ever mutated, so scanning the full sequence would be
        twenty times the compute for identical scores.
        """
        torch = self.torch
        cache = self._cache.setdefault(seq, {})
        todo = sorted({p for p in positions if 0 <= p < len(seq) and p not in cache})
        if todo:
            ids = self.tok(seq, return_tensors="pt")["input_ids"][0]
            aa_ids = [self.tok.convert_tokens_to_ids(a) for a in self.AA_ORDER]
            for start in range(0, len(todo), self.batch_size):
                chunk = todo[start:start + self.batch_size]
                batch = ids.unsqueeze(0).repeat(len(chunk), 1).clone()
                for row, pos in enumerate(chunk):
                    batch[row, pos + 1] = self.tok.mask_token_id  # +1 for the CLS token
                with torch.no_grad():
                    logits = self.model(input_ids=batch.to(self.device)).logits
                    lp = torch.log_softmax(logits.float(), dim=-1)
                for row, pos in enumerate(chunk):
                    cache[pos] = lp[row, pos + 1, aa_ids].cpu().numpy()
        return cache

    def score(self, wt_seq, muts, offset=0):
        if not muts:
            return 0.0
        idx = {a: i for i, a in enumerate(self.AA_ORDER)}
        wanted = [pos - 1 + offset for _, pos, _ in muts]
        cache = self.logprobs_at(wt_seq, wanted)
        total = 0.0
        for (wt, pos, mt), i in zip(muts, wanted):
            if i not in cache or wt not in idx or mt not in idx:
                continue
            total += float(cache[i][idx[mt]] - cache[i][idx[wt]])
        return total

    def score_mutant_marginals(self, wt_seq, muts, offset=0):
        """Sum of per-position terms scored in the MUTANT sequence's context.

        Each mutated position is masked in the full mutant background, so every
        term sees the variant's other substitutions. Costs one forward pass per
        mutation rather than reusing a wild-type table, which is why it is not
        the default -- but it is the cheapest way to give an additive score
        access to epistasis.
        """
        if not muts:
            return 0.0
        mutant = apply_muts(wt_seq, muts, offset)
        if mutant is None:
            return float("nan")
        idx = {a: i for i, a in enumerate(self.AA_ORDER)}
        wanted = [pos - 1 + offset for _, pos, _ in muts]
        cache = self.logprobs_at(mutant, wanted)
        total = 0.0
        for (wt, pos, mt), i in zip(muts, wanted):
            if i not in cache or wt not in idx or mt not in idx:
                continue
            total += float(cache[i][idx[mt]] - cache[i][idx[wt]])
        return total

    def sequence_loglik(self, seq):
        """Mean log p(residue | full sequence) over one unmasked forward pass.

        Not a true pseudo-log-likelihood -- that needs one masked pass per
        residue, roughly 300x the compute here -- but it shares the property
        that matters: it is a property of the whole sequence rather than a sum
        of per-mutation terms, so it cannot inherit the mutation-count confound
        by construction.
        """
        torch = self.torch
        key = ("seqll", seq)
        if key in self._cache:
            return self._cache[key]
        ids = self.tok(seq, return_tensors="pt")["input_ids"]
        with torch.no_grad():
            logits = self.model(input_ids=ids.to(self.device)).logits
            lp = torch.log_softmax(logits.float(), dim=-1)[0]
        tok_ids = ids[0].to(self.device)
        per_pos = lp[torch.arange(len(tok_ids)), tok_ids][1:-1]  # drop CLS/EOS
        val = float(per_pos.mean())
        self._cache[key] = val
        return val

    def score_sequence_loglik(self, wt_seq, muts, offset=0):
        """Whole-sequence log-likelihood of the mutant minus the wild type."""
        if not muts:
            return 0.0
        mutant = apply_muts(wt_seq, muts, offset)
        if mutant is None:
            return float("nan")
        return self.sequence_loglik(mutant) - self.sequence_loglik(wt_seq)


class ESMCMarginals:
    """Masked marginals over ESM-C, a second scorer family.

    Every zero-shot result in this work used ESM-2 650M, which is the obvious
    objection to any claim about "protein language models" in general. ESM-C is
    a different architecture and training run from a different lab
    (EvolutionaryScale), so agreement between them is evidence about the class of
    scorer rather than about one checkpoint.

    Interface deliberately mirrors ESM2Marginals so the two are drop-in
    interchangeable everywhere downstream.
    """

    AA_ORDER = list(KD_HYDROPATHY)

    def __init__(self, model_name="esmc_300m", device=None, batch_size=4):
        import torch
        from esm.models.esmc import ESMC

        self.torch = torch
        if device is None:
            device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.device = device
        self.batch_size = batch_size
        self.model = ESMC.from_pretrained(model_name).to(device).eval()
        self.tok = self.model.tokenizer
        self._aa_ids = [self.tok.convert_tokens_to_ids(a) for a in self.AA_ORDER]
        self._mask_id = self.tok.mask_token_id
        self._cache = {}

    def logprobs_at(self, seq, positions):
        """{position: 20-vector of log p(aa | seq with that position masked)}."""
        torch = self.torch
        cache = self._cache.setdefault(seq, {})
        todo = sorted({p for p in positions if 0 <= p < len(seq) and p not in cache})
        if todo:
            ids = self.model._tokenize([seq])[0]      # includes BOS/EOS
            for start in range(0, len(todo), self.batch_size):
                chunk = todo[start:start + self.batch_size]
                batch = ids.unsqueeze(0).repeat(len(chunk), 1).clone()
                for row, pos in enumerate(chunk):
                    batch[row, pos + 1] = self._mask_id   # +1 for BOS
                with torch.no_grad():
                    out = self.model(batch.to(self.device))
                    lp = torch.log_softmax(out.sequence_logits.float(), dim=-1)
                for row, pos in enumerate(chunk):
                    cache[pos] = lp[row, pos + 1, self._aa_ids].cpu().numpy()
        return cache

    def score(self, wt_seq, muts, offset=0):
        if not muts:
            return 0.0
        idx = {a: i for i, a in enumerate(self.AA_ORDER)}
        wanted = [pos - 1 + offset for _, pos, _ in muts]
        cache = self.logprobs_at(wt_seq, wanted)
        total = 0.0
        for (wt, pos, mt), i in zip(muts, wanted):
            if i not in cache or wt not in idx or mt not in idx:
                continue
            total += float(cache[i][idx[mt]] - cache[i][idx[wt]])
        return total
