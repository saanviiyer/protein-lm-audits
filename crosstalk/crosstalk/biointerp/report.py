"""Render a battery into a statement of what a model does and does not represent.

The renderer is not decoration. The value of a typed intervention is that its
result maps onto a sentence about semantics, and the sentence is fixed in
advance by the intervention's metadata rather than chosen after seeing the
number. This module does nothing but that mapping, plus the arithmetic that
turns a null into a bounded claim.
"""
from __future__ import annotations

import csv

from .battery import (REPRESENTS, INVERTED, NULL, UNDERPOWERED, INVARIANT,
                      SENSITIVE, BatteryReport)
from .interventions import REAL_HIGHER, NO_PREFERENCE

_MARK = {REPRESENTS: "PASS", INVARIANT: "PASS", NULL: "FAIL",
         INVERTED: "FAIL", SENSITIVE: "FAIL", UNDERPOWERED: "INCONCLUSIVE"}


def render(rep: BatteryReport, width: int = 96) -> str:
    L = []
    A = L.append
    A("=" * width)
    A(f"BIOINTERP BATTERY  |  {rep.model}")
    A(f"  {rep.objective}, {rep.tokenization} tokens, {rep.alphabet} alphabet"
      f"  |  {rep.n_sequences} sequences, paired within sequence")
    A("  scores are mean per-token log-likelihood; delta = real - intervened, nats/token")
    _ci = getattr(rep, "ci_dist", "normal")
    A(f"  intervals are m +/- {'t(n-1)' if _ci == 't' else '1.96'} * SE"
      f"{'' if _ci == 't' else '   [pre-2026-09-01 convention]'}")
    A("=" * width)
    A("")
    A(f"{'intervention':22s} {'probes':28s} {'delta':>9s} {'95% CI':>19s} "
      f"{'real>':>7s}  {'expected':>13s}  {'verdict':<26s}")
    A("-" * width)
    for r in rep.results:
        exp = "real higher" if r.expect == REAL_HIGHER else "indifferent"
        A(f"{r.intervention:22s} {r.probes[:28]:28s} {r.mean_delta:+9.4f} "
          f"[{r.ci_lo:+.4f},{r.ci_hi:+.4f}] {r.real_higher:3d}/{r.n:<3d}  "
          f"{exp:>13s}  {_MARK[r.verdict]:<12s} {r.verdict:<13s}")
    A("")
    A(f"reference effect ('{rep.reference_intervention}') = {rep.reference_delta:+.4f} "
      f"nats/token. That is what this")
    A(f"scorer demonstrably resolves on these {rep.n_sequences} sequences, and every null "
      f"below is bounded against it.")
    if rep.contrasts:
        A("")
        A("-" * width)
        A("MATCHED CONTRASTS -- two edits of equal size differing only in one property")
        A("-" * width)
        for c in rep.contrasts:
            A(f"  {c.contrast}: '{c.a}' minus '{c.b}'")
            A(f"      {c.mean_diff:+.4f} nats/token [{c.ci_lo:+.4f}, {c.ci_hi:+.4f}], "
              f"'{c.a}' scored lower in {c.a_lower}/{c.n}{_ties(c)} "
              f"(binomial p={c.binom_p:.3g})   {_MARK[c.verdict]}")
            A(f"      -> it {_contrast_claim(c)}")
    A("")
    A("-" * width)
    A("SEMANTIC PROPERTIES THIS MODEL DEMONSTRABLY REPRESENTS")
    A("-" * width)
    yes = [r for r in rep.results if r.verdict in (REPRESENTS, SENSITIVE)]
    if not yes:
        A("  none: no intervention in this battery produced a significant paired difference.")
    for r in sorted(yes, key=lambda r: -abs(r.mean_delta)):
        A(f"  + it {_claim(r, True)}")
        A(f"      via '{r.intervention}': {r.mean_delta:+.4f} nats/token "
          f"[{r.ci_lo:+.4f}, {r.ci_hi:+.4f}], real higher in "
          f"{r.real_higher}/{r.n}{_ties(r)} (binomial p={r.binom_p:.3g})")
    A("")
    A("-" * width)
    A("BOUNDED NULLS AND INVARIANCES -- effects ruled out, not merely unobserved")
    A("-" * width)
    no = [r for r in rep.results if r.verdict in (NULL, INVERTED, INVARIANT)]
    if not no:
        A("  none: no null in this battery is bounded tightly enough to report.")
    for r in no:
        A(f"  {'+' if r.passed else '-'} [{_MARK[r.verdict]}] it {_claim(r, False)}")
        A(f"      via '{r.intervention}': {r.mean_delta:+.4f} nats/token "
          f"[{r.ci_lo:+.4f}, {r.ci_hi:+.4f}], real higher in "
          f"{r.real_higher}/{r.n} (binomial p={r.binom_p:.3g})")
        if rep.reference_delta > 0:
            A(f"      the interval rules out any effect above {r.ci_hi:+.4f}, at most "
              f"{100 * r.ci_hi / rep.reference_delta:.0f}% of the reference effect")
    inc = [r for r in rep.results if r.verdict == UNDERPOWERED]
    if inc:
        A("")
        A("-" * width)
        A("INCONCLUSIVE -- these are not evidence of absence and must not be quoted as such")
        A("-" * width)
        for r in inc:
            A(f"  ? {r.intervention}: {r.mean_delta:+.4f} [{r.ci_lo:+.4f}, {r.ci_hi:+.4f}]; "
              f"the interval still admits a reference-sized effect.")
    A("")
    A("-" * width)
    A("RELATIVE SENSITIVITY, each delta as a multiple of the reading-frame delta")
    A("-" * width)
    try:
        fs = rep.by_name("frameshift +1").mean_delta
    except KeyError:
        fs = None
    if fs:
        for r in rep.results:
            A(f"  {r.intervention:24s} {r.mean_delta / fs:+8.2f} x")
        A("  A ratio far above 1 means the model reads that property more strongly than it")
        A("  reads the reading frame, which is the property that determines the protein.")
    A("=" * width)
    return "\n".join(L)


def _ties(r):
    return f" ({r.ties} exact ties, excluded from the sign test)" if r.ties else ""


def _contrast_claim(c):
    from .interventions import CONTRASTS
    spec = CONTRASTS[c.contrast]
    if c.verdict == REPRESENTS:
        return spec.claim_if_significant
    if c.verdict == INVERTED:
        return (f"separates the two, but the WRONG WAY: it prefers '{c.a}'. "
                f"That is a signal about the edit, not about the frame.")
    if c.verdict == UNDERPOWERED:
        return ("gives no resolvable answer here; the interval still admits a "
                "reference-sized effect.")
    return spec.claim_if_null


def _claim(r, significant):
    """The sentence the intervention declared in advance, not one chosen afterwards."""
    from .interventions import get
    iv = get(r.intervention)
    if r.verdict == INVERTED:
        return (f"{iv.claim_if_null}, and the point estimate actually favours the "
                f"'{r.intervention}' sequence")
    if r.verdict == SENSITIVE:
        return f"{iv.claim_if_significant} (so it is NOT invariant to it)"
    return iv.claim_if_significant if significant else iv.claim_if_null


def write_contrast_csv(rep: BatteryReport, path):
    rows = [dict(model=rep.model, n_sequences=rep.n_sequences,
                 reference_delta=rep.reference_delta, **c.row())
            for c in rep.contrasts]
    if not rows:
        return None
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    return path


def write_csv(rep: BatteryReport, path):
    rows = [dict(model=rep.model, alphabet=rep.alphabet,
                 n_sequences=rep.n_sequences,
                 reference=rep.reference_intervention,
                 reference_delta=rep.reference_delta, **r.row())
            for r in rep.results]
    keys = list(rows[0])
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    return path


def write_per_sequence_csv(rep: BatteryReport, path):
    names = [r.intervention for r in rep.results]
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sequence", "real_score"] + [f"delta:{n}" for n in names])
        for k in rep.sequence_keys:
            w.writerow([k, f"{rep.real_scores[k]:.6f}"] +
                       [f"{r.per_sequence[k]:.6f}" for r in rep.results])
    return path
