# Findings

## Phase 1 — Arm A first result (2026-08-07, `scripts/run_screening_operating_point.py`)

Cached ProteinGym scores, 41 assays, re-scored at screening operating points instead of
design ones. No new compute, no downloads.

| | ESM-2 650M | ESM-C 300M | permuted null (ESM-2) |
|---|---|---|---|
| median TPR @ FPR=1% | **0.0314** | **0.0323** | 0.0088 |
| median TPR @ FPR=0.1% (25 assays estimable) | **0.0019** | **0.0022** | 0.0005 |
| bulk Spearman → TPR@1%, rho | +0.683 (p=8.6e-7) | +0.513 (p=6.0e-4) | +0.013 (p=0.93) |
| bulk Spearman → TPR@0.1%, rho | +0.389 (p=0.054) | +0.613 (p=0.0011) | −0.340 (p=0.096) |

**G3 passes.** Under label permutation the median TPR lands at the nominal FPR (0.0088
at a 1% budget, 0.0005 at a 0.1% budget) and the bulk→TPR relationship collapses to
rho=+0.013, p=0.93. The metric implementation is measuring what it claims to.

**The headline.** At a 1% false-positive budget a protein language model function proxy
recovers about **3% of functional variants**. That is roughly 3.6× the permuted null, so
it is not nothing — but the screening proficiency bar is **sensitivity ≥0.95** (NIST,
median 0.9675 as of July 2026). The gap is two orders of magnitude, and it replicates
across two architectures from different labs at half the parameter count.

**Second result.** Bulk correlation is a decent guide to TPR@1% (rho +0.683) and a much
weaker one at 0.1% on ESM-2 (rho +0.389, p=0.054). This is directionally consistent with
the Gauntlet Phase-26 decay (rho² 0.50 at top-10% → 0.16 at top-1%). But **ESM-C runs
the other way** (+0.513 → +0.613), so the decay is **not established** — treat it as
suggestive at n=41/25 and do not put it in an abstract yet. The Phase-25 retraction is
the precedent: a headline that flips on a scorer swap was never load-bearing.

**Third result, structural.** TPR@FPR=0.01% is **not estimable on any assay** — no
ProteinGym assay has 10,000+ negatives. So the operating point a real screening
deployment would care about most cannot be measured on the field's standard function
benchmark at all. That is a benchmark-adequacy finding, and it stands independent of the
numbers above.

### What this does and does not claim

It does **not** claim a measured screening error rate. DMS positives are "functional
variant," not "sequence of concern." The claim is about the **reliability of the proxy
class at screening operating points**, and about whether standard validation would
surface it. Guard this boundary in writing; it is the obvious way for a reader to
overreach.

### Caveats carried forward

- 46 of 87 reference rows dropped: cache misses, not selection. Reported, not hidden.
- `DMS_score_bin` is ProteinGym's own binarization. Phase 27 established that "dead" is
  not commensurable across corpora, so do not compare these TPRs to an SSMuLA number
  computed from `active` without re-deriving both under one definition.

### Next on this arm

1. Bootstrap CIs on the per-assay TPRs — n_pos is small in the tail and the medians
   above have no interval yet.
2. Add SSMuLA and the RuBisCO axes for the corpus-offset test at screening operating
   points (Phase-26 analysis, new outcome variable).
3. Trivial baselines at the same operating points — hydropathy, BLOSUM62, mutation
   count. On GB1 hydropathy already beats ESM-2 in bulk; if it also beats it at FPR=1%,
   that is the trivial-baseline-dominance result in its screening form.
4. `wt_logp` partialling at the operating point, on the corpora where it is computable.
