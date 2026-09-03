"""Run a set of typed interventions against a scorer and adjudicate the result.

Every comparison is paired within sequence: the same gene under the real and the
intervened condition, so gene-to-gene variation in likelihood -- which is large,
and has nothing to do with any intervention -- cancels.

Three rules this module enforces, all of them paid for elsewhere in this project:

  A NULL NEEDS A CEILING. "No significant difference" is not a finding on its
  own. The battery therefore always carries a reference intervention (by default
  the mononucleotide shuffle) whose delta is what the scorer demonstrably can
  resolve on these sequences at this n. A null is reported as NULL only when its
  confidence interval also excludes an effect as large as the reference; below
  that it is reported UNDERPOWERED and must not be quoted as evidence of absence.

  A PAIRED SIGN COUNT ACCOMPANIES EVERY MEAN. A mean delta driven by three genes
  out of twenty-nine is a different object from one where the real gene wins in
  twenty-four; the sign count and its exact binomial p are reported next to it.

  THE EXPECTATION IS DECLARED BEFORE THE NUMBER. Each intervention says in
  advance what a model with the property would have to do. PASS/FAIL is that
  comparison, not a post-hoc reading.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict

import numpy as np

from . import interventions as iv_mod
from .interventions import (Intervention, Contrast, CONTRASTS,
                            REAL_HIGHER, NO_PREFERENCE)
from .scorers import to_protein

REPRESENTS = "REPRESENTS"        # CI excludes zero in the direction expected
INVERTED = "INVERTED"            # CI excludes zero the wrong way
NULL = "NULL"                    # CI includes zero and is bounded below the reference
UNDERPOWERED = "UNDERPOWERED"    # CI includes zero but could hide a reference-sized effect
INVARIANT = "INVARIANT"          # NO_PREFERENCE interventions: bounded near zero
SENSITIVE = "SENSITIVE"          # NO_PREFERENCE interventions: a real preference exists


def _binom_p(k: int, n: int) -> float:
    """Two-sided exact binomial p against 1/2, without scipy."""
    from math import comb
    if n == 0:
        return float("nan")
    pmf = [comb(n, i) / 2 ** n for i in range(n + 1)]
    thr = pmf[k] * (1 + 1e-9)
    return float(min(1.0, sum(p for p in pmf if p <= thr)))


@dataclass
class InterventionResult:
    intervention: str
    probes: str
    expect: str
    preserves: tuple
    destroys: tuple
    n: int
    mean_delta: float
    ci_lo: float
    ci_hi: float
    se: float
    boot_lo: float
    boot_hi: float
    real_higher: int
    ties: int
    binom_p: float
    ratio_to_reference: float
    verdict: str
    passed: bool
    per_sequence: dict = field(default_factory=dict, repr=False)

    def row(self):
        d = asdict(self)
        d.pop("per_sequence")
        d["preserves"] = ";".join(self.preserves)
        d["destroys"] = ";".join(self.destroys)
        return d


@dataclass
class ContrastResult:
    """A matched difference of two deltas: (score of b) - (score of a), paired."""
    contrast: str
    a: str
    b: str
    probes: str
    n: int
    mean_diff: float
    ci_lo: float
    ci_hi: float
    boot_lo: float
    boot_hi: float
    a_lower: int
    ties: int
    binom_p: float
    ratio_to_reference: float
    verdict: str
    passed: bool

    def row(self):
        return asdict(self)


@dataclass
class BatteryReport:
    model: str
    tokenization: str
    objective: str
    alphabet: str
    n_sequences: int
    sequence_keys: list
    reference_intervention: str
    reference_delta: float
    results: list
    ci_dist: str = "t"
    contrasts: list = field(default_factory=list)
    real_scores: dict = field(default_factory=dict, repr=False)

    def by_name(self, name):
        for r in self.results:
            if r.intervention == name:
                return r
        raise KeyError(name)


def _tcrit(df: int) -> float:
    """Two-sided 97.5th percentile of Student t with `df` degrees of freedom.

    Local, so the battery keeps no scipy dependency. Hill's (1970) asymptotic
    inversion; agrees with scipy.stats.t.ppf to better than 1e-4 for df >= 3,
    which is far finer than any interval reported here.
    """
    if df <= 0:
        return float("nan")
    if df == 1:
        return 12.7062047
    if df == 2:
        return 4.30265273
    x = 1.959963985
    g1 = (x ** 3 + x) / 4
    g2 = (5 * x ** 5 + 16 * x ** 3 + 3 * x) / 96
    g3 = (3 * x ** 7 + 19 * x ** 5 + 17 * x ** 3 - 15 * x) / 384
    g4 = (79 * x ** 9 + 776 * x ** 7 + 1482 * x ** 5 - 1920 * x ** 3 - 945 * x) / 92160
    return x + g1 / df + g2 / df ** 2 + g3 / df ** 3 + g4 / df ** 4


def _paired_stats(delta: np.ndarray, seed=0, n_boot=10000, ci_dist: str = "t"):
    """Paired mean, SE, parametric CI and percentile-bootstrap CI.

    `ci_dist` is "t" (default) or "normal". The batteries reported in FINDINGS
    up to 2026-08-31 used "normal", i.e. 1.96 * SE. At n = 24 that understates
    the interval by 8.7% and at n = 9 by 25%, which is not negligible when the
    question is whether an interval excludes zero, so "t" is now the default and
    every affected number has been recomputed.
    """
    n = len(delta)
    m = float(delta.mean())
    se = float(delta.std(ddof=1) / np.sqrt(n)) if n > 1 else float("nan")
    crit = _tcrit(n - 1) if ci_dist == "t" else 1.959963985
    ci = crit * se
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    bs = delta[idx].mean(1)
    return m, se, m - ci, m + ci, float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def run_battery(model, sequences: dict, interventions=None, *, seed: int = 0,
                reference: str = "mononucleotide shuffle", progress: bool = True,
                n_boot: int = 10000, ci_dist: str = "t") -> BatteryReport:
    """Score every sequence under every intervention and adjudicate.

    Parameters
    ----------
    model        anything with `.name`, `.alphabet` and `.score(list) -> array`
                 of mean per-token log-likelihood. See scorers.py.
    sequences    {key: nucleotide sequence}. In-frame coding sequences whose
                 length is a multiple of three, unless every intervention used
                 has requires_cds=False.
    interventions list of Intervention or of registry names. The reference
                 intervention is appended if absent, because a null without a
                 ceiling is uninterpretable.
    """
    ivs = interventions or iv_mod.battery()
    ivs = [iv_mod.get(i) if isinstance(i, str) else i for i in ivs]
    if reference and reference not in [i.name for i in ivs]:
        ivs = ivs + [iv_mod.get(reference)]

    keys = sorted(sequences)
    bad = [k for k in keys if any(i.requires_cds for i in ivs)
           and (len(sequences[k]) % 3 or set(sequences[k]) - set("ACGT"))]
    if bad:
        raise ValueError(f"not in-frame ACGT coding sequences: {bad[:5]}")

    prep = (lambda s: s) if model.alphabet == "dna" else to_protein

    real_raw = [sequences[k] for k in keys]
    if progress:
        print(f"scoring {len(keys)} real sequences with {model.name}", flush=True)
    real = np.asarray(model.score([prep(s) for s in real_raw]))
    real_by_key = dict(zip(keys, map(float, real)))

    results = []
    deltas = {}
    for iv in ivs:
        pert = [iv.apply(sequences[k], iv.rng_for(k, seed)) for k in keys]
        if progress:
            print(f"scoring {len(keys)} sequences under '{iv.name}'", flush=True)
        sc = np.asarray(model.score([prep(s) for s in pert]))
        deltas[iv.name] = real - sc
        results.append((iv, sc))

    ref_delta = float(np.mean(deltas[reference])) if reference in deltas else float("nan")

    out = []
    for iv, sc in results:
        d = deltas[iv.name]
        m, se, lo, hi, blo, bhi = _paired_stats(d, seed=seed, n_boot=n_boot, ci_dist=ci_dist)
        # Exact ties carry no sign information and must not be fed to a sign
        # test. They are not hypothetical: a protein-alphabet scorer is exactly
        # tied under synonymous recode, because it is handed the same input twice.
        k, ties = int((d > 0).sum()), int((d == 0).sum())
        ratio = float(m / ref_delta) if ref_delta and np.isfinite(ref_delta) else float("nan")

        if iv.expect == REAL_HIGHER:
            if lo > 0:
                verdict = REPRESENTS
            elif hi < 0:
                verdict = INVERTED
            elif np.isfinite(ref_delta) and ref_delta > 0 and hi < ref_delta:
                verdict = NULL
            else:
                verdict = UNDERPOWERED
            passed = verdict == REPRESENTS
        else:                                   # NO_PREFERENCE
            if lo > 0 or hi < 0:
                verdict = SENSITIVE
            elif np.isfinite(ref_delta) and ref_delta > 0 and max(abs(lo), abs(hi)) < ref_delta:
                verdict = INVARIANT
            else:
                verdict = UNDERPOWERED
            passed = verdict == INVARIANT

        out.append(InterventionResult(
            intervention=iv.name, probes=iv.probes, expect=iv.expect,
            preserves=iv.preserves, destroys=iv.destroys, n=len(d),
            mean_delta=m, ci_lo=lo, ci_hi=hi, se=se, boot_lo=blo, boot_hi=bhi,
            real_higher=k, ties=ties, binom_p=_binom_p(k, len(d) - ties),
            ratio_to_reference=ratio,
            verdict=verdict, passed=passed,
            per_sequence={kk: float(v) for kk, v in zip(keys, d)}))

    # ------------------------------------------------------- matched contrasts
    con_out = []
    have = {r.intervention for r in out}
    for c in CONTRASTS.values():
        if not {c.a, c.b} <= have:
            continue
        da = np.array([dict(zip(keys, deltas[c.a]))[k] for k in keys])
        db = np.array([dict(zip(keys, deltas[c.b]))[k] for k in keys])
        diff = da - db                      # = score(b) - score(a), paired
        m, se, lo, hi, blo, bhi = _paired_stats(diff, seed=seed, n_boot=n_boot, ci_dist=ci_dist)
        kk, cties = int((diff > 0).sum()), int((diff == 0).sum())
        if lo > 0:
            verdict, passed = REPRESENTS, True
        elif hi < 0:
            verdict, passed = INVERTED, False
        elif np.isfinite(ref_delta) and ref_delta > 0 and hi < ref_delta:
            verdict, passed = NULL, False
        else:
            verdict, passed = UNDERPOWERED, False
        con_out.append(ContrastResult(
            contrast=c.name, a=c.a, b=c.b, probes=c.probes, n=len(diff),
            mean_diff=m, ci_lo=lo, ci_hi=hi, boot_lo=blo, boot_hi=bhi,
            a_lower=kk, ties=cties, binom_p=_binom_p(kk, len(diff) - cties),
            ratio_to_reference=float(m / ref_delta) if ref_delta else float("nan"),
            verdict=verdict, passed=passed))

    return BatteryReport(
        model=model.name, tokenization=getattr(model, "tokenization", "?"),
        objective=getattr(model, "objective", "?"), alphabet=model.alphabet,
        n_sequences=len(keys), sequence_keys=keys,
        reference_intervention=reference, reference_delta=ref_delta,
        ci_dist=ci_dist, results=out, contrasts=con_out, real_scores=real_by_key)


# --------------------------------------------------- re-adjudicate without rescoring

def rebuild_from_csv(per_sequence_csv, model: str, *, alphabet="dna",
                     tokenization="?", objective="?",
                     reference="mononucleotide shuffle", seed=0, n_boot=10000,
                     ci_dist="t", keep=None):
    """Rebuild and re-adjudicate a report from a saved per-sequence delta file.

    Scoring is by far the expensive part and the adjudication rules are the part
    most likely to be revised, so they are separable. Any change to a verdict
    rule can be applied to every model already measured without touching a GPU.
    """
    import csv as _csv
    rows = list(_csv.DictReader(open(per_sequence_csv)))
    if keep is not None:
        keep = set(keep)
        rows = [r for r in rows if r["sequence"] in keep]
        if not rows:
            raise ValueError("keep= removed every row")
    names = [c[len("delta:"):] for c in rows[0] if c.startswith("delta:")]
    keys = [r["sequence"] for r in rows]
    fake = {k: "ACG" for k in keys}

    class _Replay:
        name, tokenization_, objective_ = model, tokenization, objective
        def __init__(self):
            self.alphabet = alphabet
            self.tokenization, self.objective = tokenization, objective
            self._real = np.array([float(r["real_score"]) for r in rows])
            self._i = -1
        def score(self, seqs, progress=None):
            self._i += 1
            if self._i == 0:
                return self._real
            col = f"delta:{names[self._i - 1]}"
            return self._real - np.array([float(r[col]) for r in rows])

    return run_battery(_Replay(), fake, interventions=names, seed=seed,
                       reference=reference, progress=False, n_boot=n_boot,
                       ci_dist=ci_dist)
