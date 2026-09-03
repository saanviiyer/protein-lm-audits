# Findings

All numbers from `results/benchmark.csv` (30 seeds per cell) and
`scripts/run_benchmark.py`, on the ParD3 landscape (GEO GSE153897).

## 0. Data validation

Recomputing fitness from raw replicate frequencies reproduces the published
`W` at r=0.9998 (both partners), via a linear map. This is the check that the
landscape is being read correctly, and it also gives the noise model: a single
assay has SD 0.039 (ParE3) / 0.031 (ParE2) in W units, about 3% of the dynamic
range.

A unit trap worth recording: replicate disagreement is naturally expressed in
log-enrichment units, and pushing it through the linear map to W units matters.
Skipping that step overstates the oracle noise by about 7x.

## 1. Trap density: cross-reactivity is common among strong binders

| | strong binders | of those, cross-reactive |
|---|---|---|
| ParE3 (W > 0.8) | 714 (9.1% of landscape) | 210 = **29.4%** |
| ParE2 (W > 0.8) | 604 | 253 = **41.9%** |

Roughly a third of strong binders are promiscuous. An objective blind to the
off-target has no way to avoid them except by luck.

## 2. Specificity is a harder search problem than affinity

Exhaustive greedy ascent from all 7,882 states, single substitutions:

| task | objective | local optima | ascents reaching global optimum |
|---|---|---|---|
| cognate | affinity | 4 | 68.9% |
| cognate | **margin** | **11** | **48.2%** |
| swap | affinity | 5 | 99.5% |
| swap | **margin** | **10** | **95.4%** |

Pricing off-target binding roughly doubles the number of local optima on
identical data. This is the result that justifies an RL environment: specificity
is not just harder to *measure*, it is harder to *optimize*.

## 3. Budget does not substitute for the right objective

ML-guided campaign (`additive_model`), ground-truth success rate. Success
requires on-target W >= 0.8 **and** off-target W <= 0.5.

| task | reward | B=50 | B=100 | B=200 | B=400 | B=800 |
|---|---|---|---|---|---|---|
| cognate | affinity, screen only | 0.67 | 0.77 | 0.73 | 0.83 | 0.73 |
| cognate | affinity, counter-screened but ignored | 0.63 | 0.80 | 0.63 | 0.83 | 0.90 |
| cognate | **margin** | 0.77 | 0.93 | **1.00** | **1.00** | **1.00** |
| swap | affinity, screen only | 0.57 | 0.60 | 0.63 | 0.47 | 0.73 |
| swap | affinity, counter-screened but ignored | 0.53 | 0.57 | 0.60 | 0.73 | 0.70 |
| swap | **margin** | 0.73 | 0.93 | **1.00** | **1.00** | **1.00** |

Crosstalk rate tells the same story more sharply: specificity-aware rewards sit
at 0.00 from B=200 onward on both tasks, while affinity rewards never fall below
0.10 (cognate) or 0.26 (swap) at any budget.

**The isolating control.** The `affinity, counter-screened but ignored` arm pays
the full two-assay cost, so it *has* the off-target measurement, and merely
leaves it out of its reward. It fails about as badly as the arm that never
measured off-target at all. On the swap task, having the data is worth -0.03 in
success rate while *using* it is worth +0.30. The failure is reward
specification, not information.

Significance (Fisher exact, two-sided, affinity vs margin, both counter-screened):

- swap, B=800: 21/30 vs 30/30, p = 0.0019
- cognate, B=800: 27/30 vs 30/30, p = 0.24 (not significant at n=30)
- pooled over B >= 200 (n=90/arm): p < 1e-5 on both tasks

The effect is large and clearly significant on the swap task and when pooling
budgets. At a single budget on the cognate task, 30 seeds is underpowered.

## 4. Hard constraints are not free

`gated` and `lagrangian` match `margin` at low budget but plateau slightly below
it at high budget (0.90 to 0.97 vs 1.00 on swap). Both impose a discontinuity at
the threshold that the additive surrogate extrapolates across poorly. The
smooth margin objective is the better default here.

## Scope and limits

- One landscape, three mutated positions, two partners. ParD3 is the
  exactly-solvable calibration case, chosen because both sides are measured for
  every variant. It is not a hard search problem in absolute terms: plain hill
  climbing solves the walk task, which is why the benchmark is the budgeted
  campaign and not the walk.
- Specificity here is defined against one off-target. Real polyspecificity is
  against many, and the margin objective's max over off-targets is untested at
  scale.
- The agents are search baselines, not trained RL policies. The environment API
  is in place for policy learning; no policy has been trained yet.

## Bug found and fixed

The first benchmark run reported affinity-only on the swap task as exactly
0.00 success at every budget. That was an artifact: `Campaign.measure`
hardcoded assay channel 0, so swap-task agents (target = channel 1) measured
the wrong assay, learned nothing, and nominated a constant sequence. Fixed, and
covered by `test_campaign_measures_its_own_target_channel` and
`test_swap_agents_are_seed_sensitive`. The corrected affinity failure is real
but partial, and for a biological reason: adding ParE2 binding does not remove
ParE3 binding, so affinity-only swap designs retain the original partner.

## 5. Boltz as a structural proxy oracle (staged, needs a GPU)

**Boltz-2 affinity cannot be used here.** Its affinity head only accepts a
small-molecule binder: the docs require the binder to "be a ligand chain (not a
protein, DNA or RNA)" and at most 128 heavy atoms. ParD3 is a 93-residue
protein. A 2026 study (arXiv:2512.06592) fine-tuned Boltz-2 for protein-protein
affinity and reported it underperforms sequence-based baselines. Boltz 2.1 is
also closed-source and hosted-API-only as of June 2026.

What *is* usable is co-folding confidence. ipTM and interface PAE are the
proxies binder designers actually optimize, and this landscape can audit them,
because every variant has a measured on-target and off-target fitness.

**Sequences, each verified independently** (UniProt, *M. opportunistum*
LMG 24607 = WSM2075, the strain Lite et al. used):

| protein | accession | locus | length | verification |
|---|---|---|---|---|
| ParD3 | F7YBW8 | Mesop_5599 | 93 | D61/K64/E80 = `DKE`, matching the landscape wild type |
| ParE3 | F7YBW7 | Mesop_5598 | 103 | co-operonic with ParD3 (cognate) |
| ParD2 | F7Y4V9 | Mesop_5170 | 91 | 39.6% identity to ParD3 (paper reports 41%) |
| ParE2 | F7Y4W0 | Mesop_5171 | 94 | co-operonic with ParD2 (non-cognate) |

The wild-type residue check is what picked F7YBW8 out of many ParD paralogs: the
landscape's wild-type variant is `DKE`, so the true ParD3 must carry D, K, E at
61, 64, 80. It does.

**Status: attempted, and measured infeasible on this hardware.**

Boltz 2.2.1 installs cleanly in a Python 3.11 venv (`.venv-boltz`). Python 3.13
does *not* work: boltz pins a scipy with no 3.13 wheel, so pip falls back to a
source build and fails for want of a Fortran compiler.

The model then downloads ~7 GB of weights and CCD data and runs. MSA generation
via the ColabFold server succeeded. Inference did not finish:

| attempt | settings | outcome |
|---|---|---|
| 1 | defaults | no output after ~50 min, killed |
| 2 | `--recycling_steps 0 --diffusion_samples 1 --sampling_steps 25` | **completed in 39m45s** |

That is a single 196-residue complex (ParD3 93 aa + ParE3 103 aa) on MPS with no
CUDA device. The wild-type ParD3:ParE3 prediction is confident and sensible:

    ipTM 0.961   pTM 0.893   complex pLDDT 0.955

So the pipeline works end to end without a GPU; it is only the throughput that
is the problem. At ~40 min per fold the full 120-fold audit is ~80 hours
sequential. The process averages ~0.7 cores, so it parallelises: at 6 concurrent
folds a reduced 48-fold audit is ~5 hours, which is what is running now. A CUDA
GPU remains the right way to run the full version.

Everything upstream of the model call is verified: `--dry-run` builds and
validates all 120 inputs (60 variants stratified across the four specificity
quadrants x 2 partners), and `scripts/analyze_boltz_proxy.py` scores the result
the moment it exists. It follows the Gauntlet pattern of reporting a trivial
baseline (mutation count) alongside the proxy: a structural score that cannot
beat counting mutations is not measuring specificity.

**No claim is made about how Boltz performs on this landscape.** It was run, it
did not produce a prediction here, and that is all this section reports.

## 6. Training a policy: the reward improves either way, ground truth does not

A learned acquisition policy (REINFORCE, permutation-equivariant scorer over
candidates) trained at a 50-assay budget. Both arms are counter-screened, so
both pay for the off-target measurement and see identical information; they
differ only in whether the off-target enters the reward. **16 seeds per cell**,
run as 64 containers via Dagger, analysed paired within seed.

| task | reward | d(its own reward) | d(ground-truth success) | d(crosstalk) |
|---|---|---|---|---|
| cognate | affinity | **+0.0134** (p=2e-5, 15/16 up) | -0.011 (p=0.51, ns) | +0.023 (p=0.18, ns) |
| cognate | margin | **+0.0728** (p<1e-5, 16/16) | **+0.095** (p<1e-5, 16/16) | -0.001 (p=0.10) |
| swap | affinity | **+0.0244** (p<1e-5, 15/16 up) | **-0.044** (p=0.016, 4/16 up) | **+0.063** (p=0.0025) |
| swap | margin | **+0.0889** (p<1e-5, 16/16) | **+0.147** (p<1e-5, 16/16) | **-0.024** (p<1e-5) |

Three things, and the third is task-dependent:

1. **The optimizer works everywhere.** Every cell improves its own reward in 15
   or 16 of 16 seeds.
2. **The specificity-aware reward carries through to ground truth.** +0.095 and
   +0.147 success, 16/16 seeds up on both tasks, and it drives crosstalk down.
3. **The affinity reward does not, and on the harder task it actively hurts.**
   On `cognate` it is a pure dissociation: reward up reliably, ground truth
   flat (p=0.51). On `swap` it is worse than flat: ground-truth success *falls*
   (-0.044, p=0.016, up in only 4 of 16 seeds) and crosstalk *rises* (+0.063,
   p=0.0025). Optimizing the objective harder made the designs worse.

That asymmetry is biologically sensible. On `swap` the agent must reprogram ParD3
onto ParE2, and adding new binding does not remove the old: an affinity-only
objective has no term that pushes ParE3 binding away, so the better it gets at
its reward the more cross-reactive its designs become.

The clean test is the interaction, since the arms differ only in reward:

| contrast | task | affinity | margin | difference | p |
|---|---|---|---|---|---|
| d success | cognate | -0.011 | +0.095 | +0.105 | 1e-5 |
| d success | swap | -0.044 | +0.147 | +0.191 | <1e-5 |
| d crosstalk | cognate | +0.023 | -0.001 | -0.024 | 0.16 (ns) |
| d crosstalk | swap | +0.063 | -0.024 | -0.087 | 1.4e-4 |

**Revision from the 6-seed run.** At 6 seeds the within-arm decline for
affinity was not significant on either task (p=0.44, 0.49) and this section
concluded "dissociation, not demonstrated harm". At 16 seeds that holds for
`cognate` but not for `swap`, where both the success decline and the crosstalk
rise are significant. The earlier caution was a power limitation, not a ceiling
on the effect.

### Two bugs found here, both of which faked a result

- **The entropy bonus was not entropy.** It was the summed log-prob of taken
  actions, which at 25-100 decisions per episode swamped the policy gradient and
  flatlined training entirely (reward moved +0.0003 over 25 batches).
- **`train()` rebuilt Adam on every call.** Evaluating mid-training by chunking
  training into repeated calls therefore reset optimizer state at each
  checkpoint, and produced one cell where the agent got *worse at its own
  reward* -- an artifact that would have read as a finding. Fixed with an
  `eval_fn` callback so training runs continuously.

Both are covered by tests (`test_entropy_bonus_is_real_entropy`,
`test_training_improves_its_own_reward`).

## 7. Polyspecificity: the affinity objective anti-scales in budget AND off-targets

ParD3 has only two partners measured, capping it at one off-target, so
`margin`'s max-over-off-targets was untested beyond K=1. The Absolut! lattice
database (Robert et al. 2021) gives ground-truth energies for the same sequences
against many antigens, so K can be varied directly.

**Landscape.** 20,000 11-mer CDRH3 slides x 5 antigens (`1ADQ_A` target;
`1FBI_X`, `1FNS_A`, `1FSK_A`, `1H0D_C` off-targets), streamed from the raw
binding archives. Cross-partner correlations are 0.29-0.58: binders to one
antigen genuinely tend to bind others, which is what makes specificity hard.
Success = on-target >= 0.74 (top ~2%) and *every* off-target <= 0.65.

**Cost structure.** Screening the target costs 1 assay; counter-screening K
off-targets costs 1+K. So the affinity-only agent sees (1+K)x more variants for
the same budget. This is a real trade, not a strawman.

Ground-truth success rate, 30 seeds per cell:

| K | reward | B=100 | B=300 | B=900 |
|---|---|---|---|---|
| 1 | affinity | 0.87 | 0.63 | 0.67 |
| 1 | margin | 0.23 | 0.70 | **0.73** |
| 2 | affinity | 0.50 | 0.20 | **0.07** |
| 2 | margin | 0.33 | 0.27 | **0.63** |
| 3 | affinity | 0.57 | 0.13 | **0.10** |
| 3 | margin | 0.13 | 0.27 | **0.57** |
| 4 | affinity | 0.50 | 0.20 | **0.03** |
| 4 | margin | 0.17 | 0.30 | **0.50** |

**More budget makes the affinity agent strictly worse.** From B=100 to B=900:

| K | affinity success | Fisher p |
|---|---|---|
| 1 | 0.87 -> 0.67 | 0.13 (ns) |
| 2 | 0.50 -> 0.07 | 0.0004 |
| 3 | 0.57 -> 0.10 | 0.0003 |
| 4 | 0.50 -> 0.03 | 0.0001 |

At K=4 and B=900 the affinity agent succeeds 3% of the time with 97% crosstalk.
This is the mechanism stated plainly: because binding is positively correlated
across antigens, the *best* target binders are disproportionately the
promiscuous ones. Searching harder for affinity finds them more reliably. The
objective is not merely uninformative about specificity, it is actively
anti-correlated with it, and optimization pressure converts that into failure.

The margin objective moves the opposite way, improving with budget at every K
and holding crosstalk at 0.00-0.03 throughout.

**At B=900, affinity vs margin:** K=1 p=0.78 (ns), K=2 p=1e-5, K=3 p=0.0003,
K=4 p=6e-5.

### The honest caveat: counter-screening is not free, and at K=1 it can lose

At K=1 and B=100 the affinity-only agent *beats* the specificity-aware one,
0.87 to 0.23, because it sees 100 variants where the counter-screening agent
sees 50. The crossover is real and it moves left as K grows: with one
off-target, screening-only wins on a tight budget; by K=2 it is behind
everywhere and falling. A blanket "always counter-screen" recommendation would
be wrong at K=1 on a small budget, which is exactly the regime most single-target
campaigns operate in.

### Scope

These energies are simulated (lattice + Miyazawa-Jernigan), not measured, and
they are deterministic, so the assay noise here is a stated modelling choice
rather than something calibrated from replicates as it was for ParD3. What
Absolut! provides that no measured dataset does is ground truth against many
partners for the same sequences, which is the only way to vary K at all.

## 8. Where counter-screening starts paying: the crossover budget B*

Section 7 left a practical question open. Counter-screening K off-targets costs
1+K assays per variant, so it buys specificity at the price of throughput, and
at K=1 on a small budget the affinity-only agent won outright. The useful output
is therefore not "affinity is wrong" but the budget above which paying for
counter-screening is the better spend.

B* is the smallest budget from which the specificity-aware agent is ahead *and
stays ahead*, on a 9-point budget grid, 40 seeds, bootstrapped over seeds.

| K | B* | 95% CI | crossover found in | per counter-screened variant |
|---|---|---|---|---|
| 1 | 900 | [450, 900] | **54% of resamples** | 450 assays |
| 2 | 200 | [200, 450] | 100% | 67 assays |
| 3 | 200 | [150, 450] | 100% | 50 assays |
| 4 | 300 | [150, 450] | 100% | 60 assays |

**The rule.** With two or more off-targets, counter-screening pays from roughly
200-300 assays, about 50-70 assays per counter-screened variant. With a single
off-target it may not pay at all inside a realistic budget: a crossover appeared
in only 54% of bootstrap resamples, so B* is not identified there.

That K=1 instability is the honest headline of this section. The entire
polyspecificity effect in section 7 is driven by having more than one competitor
to avoid. A campaign against one known off-target, on a budget under a few
hundred assays, is genuinely better off screening for binding and
counter-screening the survivors afterwards. The case for building specificity
into the objective is a case about *polyspecificity*, and it should be argued
that way rather than as a general claim.

Note also that B* is roughly flat in K (200-300) even though the per-variant
cost rises with K. The benefit grows about as fast as the cost, which is why the
per-variant figure falls from 450 (K=1) to ~50-70 (K>=2).

## 9. The proxy ladder: protein-LM likelihood is a promiscuity detector

Boltz costs ~40 min per complex. A protein language model costs milliseconds
here, because every variant differs only at positions 61/64/80: mask those three
sites once and every variant's masked-marginal score falls out of three forward
passes. Masking the same sites inside a ParD3:ParE concatenate makes the score
partner-aware for three more passes per partner.

The task is the one that matters: separate a *specific* binder from a
*promiscuous* one, where both bind ParE3 well (n=399; 235 specific, 164
promiscuous). AUC, with bootstrap CI:

| proxy | 8M | 35M | 150M | 650M |
|---|---|---|---|---|
| partner-blind | 0.296 | 0.273 | 0.199 | **0.151** |
| partner-aware score | 0.288 | 0.292 | 0.276 | 0.170 |
| partner-aware margin | 0.649 | 0.407 | 0.537 | 0.545 |
| **trivial baseline: mutation count** | | | | **0.664** [0.625, 0.702] |

**No configuration of any model beats counting mutations.** The best PLM number
in the table (0.649, the 8M partner-aware margin) is below the trivial baseline,
and the larger models' margin scores have CIs straddling 0.5.

**I predicted chance and was wrong, in the more interesting direction.** The
argument was that a partner-blind score has no channel to express specificity,
so its AUC should be 0.5. It is not 0.5, it is 0.151. The proxy is not
uninformative; it is informative with the wrong sign.

The mechanism, measured directly on ESM-2 650M:

    rho(PLM, on-target ParE3)   = +0.510
    rho(PLM, off-target ParE2)  = +0.540      <- predicts the WRONG partner better
    rho(PLM, specificity margin)= -0.068

    within the 714 strong ParE3 binders:
      rho(PLM, off-target)                = +0.412
      mean off-target, bottom-quartile PLM = 0.270
      mean off-target, top-quartile PLM    = 0.617

PLM likelihood scores how natural or wild-type-like a sequence is. Wild-type
ParD3 binds its cognate partner *and* retains measurable affinity for the
paralog, so naturalness tracks promiscuity. Among strong binders, the
higher-likelihood ones are the more cross-reactive ones. Selecting designs by
PLM score therefore selects for exactly the failure mode the design is meant to
avoid.

**Scale makes it worse.** Partner-blind AUC falls monotonically from 0.296 at 8M
to 0.151 at 650M. A bigger model captures naturalness better, and naturalness is
the thing pointing the wrong way. This is the section-7 result restated at the
level of the proxy rather than the optimizer: more of the resource that helps on
the stated objective buys more harm on the real one.

**Partner-awareness is real but does not rescue it.** The scores do respond to
the partner (mean |score given ParE3 - score given ParE2| is 0.17-0.56, not
zero), yet partner-aware AUC tracks partner-blind AUC almost exactly. Having a
channel to express specificity is not the same as using it.

### Additivity was not the explanation

The masked-marginal score is additive over the three mutated sites, so it cannot
represent epistasis between them, and section 6 measured epistasis as the
dominant error term on this landscape. That was the one serious alternative
explanation for the anti-correlation. Full pseudo-likelihood removes the
restriction: every one of the 93 positions is masked in turn in the variant's
own context, at 93 forward passes per variant instead of 3 for the whole
landscape (37,107 passes, 26 min on ESM-2 650M).

Matched on the same 399-variant discrimination set:

| method | AUC | rho on-target | rho off-target | rho margin |
|---|---|---|---|---|
| masked-marginal | 0.151 [0.115, 0.191] | +0.276 | +0.541 | -0.494 |
| full pseudo-likelihood | 0.145 [0.110, 0.183] | +0.366 | **+0.568** | **-0.499** |

The two scores agree almost exactly (rank correlation +0.913) and are equally
anti-predictive. Dropping the additivity restriction changes nothing, so
additivity was not the cause.

Both methods correlate *more* strongly with off-target binding than with
on-target binding (+0.568 vs +0.366 for pseudo-likelihood). That is the finding
in one line: on this landscape ESM-2 likelihood is a better predictor of the
partner you are trying to avoid than of the one you are trying to bind.

(Note on units: the rho margin of -0.068 reported earlier for masked-marginal
was computed over the full 7,882-variant landscape, which is ~70% variants that
bind nothing. Restricted to the discrimination set it is -0.494. Both are
correct for what they measure; only the matched row above is a fair comparison.)

### Scope

One landscape and one model family, and the discrimination set is defined by
thresholds on the same measurements used to score it. The result establishes
that the standard zero-shot PLM score -- in both its cheap and its expensive
form -- is anti-predictive of specificity here, not that it must be everywhere.

## 10. Boltz co-folding: right sign, wrong sample size

32 folds completed (16 variants x 2 partners) at ~13 min each once weights and
CCD data were cached, 3 at a time. Matched head-to-head against ESM-2 650M on
the *same* 16 variants:

| proxy | rho vs measured specificity margin |
|---|---|
| Boltz ipTM margin | **+0.268** |
| ESM-2 650M likelihood | **-0.294** |

and the ordering that matters:

| proxy | rho vs on-target | rho vs off-target |
|---|---|---|
| Boltz ipTM (vs ParE3) | **+0.471** | +0.335 |
| ESM-2 650M likelihood | +0.338 | **+0.632** |

Boltz ranks the target above the off-target. ESM-2 does the reverse. That is the
qualitative difference predicted by the ladder: a proxy that actually models the
complex has a channel for specificity, and a single-sequence likelihood does not.

**This is not yet evidence.** n=16, and Spearman rho at n=16 needs roughly |rho|
> 0.50 to clear p=0.05, so neither correlation is individually significant. The
discrimination set here is 4 specific versus 4 promiscuous, which makes the AUC
numbers from `analyze_boltz_proxy.py` (ipTM margin 0.688, mutation count 0.906)
pure noise. They are reported only for completeness and should not be cited.

**The dynamic range is the real concern.** ipTM spans 0.942-0.968 against ParE3
and 0.908-0.945 against ParE2, with a mean margin of +0.028. Boltz is confident
that *every* variant forms a good complex with *both* partners, and separates
cognate from non-cognate by about three ipTM points. Even with the sign correct,
that is a compressed signal against a measured fitness range of ~1.2.

**What would settle it:** 20 variants per quadrant, 160 folds, about 11 hours at
the observed throughput. Interface PAE rather than the scalar ipTM may also
discriminate better, and the `pae_*.npz` files are already saved for every fold.

## 11. Interface PAE: more dynamic range, no more specificity signal

ipTM compressed everything into three points (0.942-0.968 against ParE3), so the
natural hypothesis was that the scalar was the bottleneck rather than the model.
Interface PAE is per-residue-pair and should have room to separate complexes
that ipTM calls equally good. Computed from the saved `pae_*.npz` over the
cross-chain blocks, as a mean over all cross pairs and as a mean over the most
confident 10% (closer to "how well is the actual interface resolved").

Dynamic range, as hoped:

| proxy | span vs ParE3 | relative to its own values |
|---|---|---|
| ipTM | 0.026 | ~2.7% |
| mean interface PAE | 1.40 (4.23-5.62) | ~28% |
| min-k interface PAE | 0.31 (0.71-1.02) | ~36% |

Correlation with measured fitness (PAE negated so larger is better):

| proxy | vs on-target | vs off-target | vs specificity margin |
|---|---|---|---|
| mean interface PAE | +0.450 | -0.347 | +0.088 |
| min-k interface PAE | +0.429 | +0.035 | **+0.279** |
| ipTM (reference) | +0.471 | +0.335 | **+0.268** |

**The hypothesis was wrong.** Interface PAE has roughly ten times the relative
dynamic range of ipTM and produces essentially the same specificity correlation
(+0.279 versus +0.268, indistinguishable at n=16). Compression was not what
limited ipTM. Boltz's confidence, however it is read out, simply does not track
this specificity margin strongly.

This is worth stating plainly because it removes the cheapest remaining
explanation. The next real test is sample size, not a different readout: at
n=16 nothing here clears significance, and the honest position remains that
Boltz gets the *sign* right where ESM-2 gets it backwards, on too few points to
call it.

## 12. Adding a GenBio rung (in progress)

`GB.Protein-16B` cannot run here: 13 shards of ~4.9 GB is 64 GB fp32 and ~32 GB
bf16, against 26 GB of unified memory and 23 GB of free disk. `AIDO.Protein-RAG-3B`
is 15 GB fp32 and fits. It is retrieval-augmented but `enable_rag` defaults to
False, so it runs single-sequence -- which is the point, because it makes the
comparison to ESM-2 a like-for-like swap of the model rather than a different
experiment.

Four obstacles, recorded because none are documented together anywhere:

1. `modelgenerator` breaks on import with `docstring-inheritance` 3.0.0 *and*
   2.3.0. Pin **2.2.2**.
2. The backbone wrapper refuses to build a `MASKED_LM` head without explicit
   `config_overwrites` and `model_init_args`, and fails with a bare
   `AttributeError: no attribute 'decoder'`.
3. The HF config has no `auto_map`, so `trust_remote_code=True` cannot load it
   either. Import `FM4BioForMaskedLM` from
   `modelgenerator.huggingface_models.fm4bio` directly instead; it has a
   standard HF masked-LM interface and returns logits.
4. The weight repos ship **no tokenizer files**. The vocab the models were
   trained with is bundled inside the package at
   `modelgenerator/huggingface_models/fm4bio/vocab_protein.txt`; construct
   `FM4BioTokenizer(vocab_file=...)` from it.

A fifth, found once it loaded: the model uses `rope_2d`, so it requires
`position_ids` of shape (B, 2, L) -- channel 0 the residue/column index, channel
1 which MSA row a token came from. Omitting them raises a permute error deep
inside the model rather than falling back. For a single sequence the encoding is
(arange(L), zeros(L)). The rotary table is also built on CPU, so the model must
run on CPU unless that is patched.

### Results

Identical masked-marginal protocol to the ESM-2 rungs. RAG reuses the ColabFold
MSAs the Boltz runs already produced (20 homolog rows, chain 0 = ParD3).

| arm | AUC | rho on-target | rho off-target |
|---|---|---|---|
| partner-blind | 0.212 [0.169, 0.256] | +0.588 | +0.540 |
| partner-blind **+ RAG** (20 MSA rows) | 0.200 [0.157, 0.244] | +0.566 | +0.557 |
| partner-aware (ParE3) | 0.216 [0.172, 0.261] | +0.591 | +0.534 |
| **partner-aware margin** | **0.698 [0.646, 0.747]** | | |
| ESM-2 650M partner-blind (reference) | 0.151 | +0.510 | +0.540 |
| trivial mutation-count baseline | 0.664 [0.625, 0.702] | | |

Three things:

1. **RAG does not fix it, tested properly.** The first attempt was weak in two
   ways, both found by reading GenBio's own tutorial code
   (github.com/genbio-ai/GB-Foundations-Tutorials): it capped context at 2048
   tokens from a vestigial config field when the model card and the tutorial
   both use 12.8K, and it took the first 20 MSA rows, which are the *closest*
   homologs and carry the least information. Their `greedy_select` picks a
   maximally diverse subset by Hamming distance instead (mean pairwise distance
   0.767 versus 0.633 for first-20).

   Redone with diverse selection and up to the model's full context:

   | MSA depth | ~tokens | AUC |
   |---|---|---|
   | 0 | 93 | 0.212 [0.169, 0.256] |
   | 20 | 1,953 | 0.196 [0.155, 0.238] |
   | 64 | 6,045 | 0.228 [0.184, 0.275] |
   | 128 | 11,997 | 0.230 [0.185, 0.276] |

   Flat across a 6x range of MSA depth, every interval overlapping, all far
   below chance. The caveat that AIDO is trained with MSA context and might be
   handicapped without one is now properly closed: give it a deep, diverse
   alignment filling its context and nothing changes.

2. **The anti-correlation is not an ESM-2 quirk.** A different lab, a different
   architecture, a different training corpus, 3B parameters, and partner-blind
   AUC is still 0.21, far below chance. This is the finding that makes section 9
   general rather than a property of one model family.

3. **One configuration finally reaches the trivial baseline.** AIDO's
   partner-aware margin scores 0.698 against the mutation-count baseline's
   0.664, the first PLM arm to exceed it on the point estimate -- though its
   CI [0.646, 0.747] contains 0.664, so the honest reading is that it *matches*
   counting mutations rather than beating it. Notably AIDO also ranks on-target
   above off-target (+0.588 vs +0.540) where ESM-2 650M inverts them (+0.510 vs
   +0.540).

## 13. Structure embeddings: blocked, and the check that showed it

`GB_Structure_Tokenizer` in GenBio's tutorial utilities turns a PDB into structure
embeddings the model accepts alongside sequence and MSA. We have 32 Boltz-predicted
structures on disk, so the input would be free, and it is the one modality the
anti-correlation has not been tested against.

It cannot run here:

| model | `str_embedding_in` | size |
|---|---|---|
| AIDO.Protein-RAG-3B | **None** | 15 GB fp32 |
| GB.Protein-RAG-16B | 384 | 64 GB fp32, ~32 GB bf16 |

Only the 16B accepts structure at all, and it fits neither the 26 GB of unified
memory nor the 14 GB of remaining disk. The generalisable lesson is to check
`str_embedding_in` in the config before assuming a modality is supported across a
model family: the 3B and 16B share a name, an architecture family and a tutorial,
and differ on whether structure input exists.

This is the same hardware gate as the full Boltz audit, so both clear together on
a rented GPU.

### An aside worth recording

GenBio's own perturbation benchmark
(github.com/genbio-ai/foundation-models-perturbation) evaluates against trivial
baselines by design: `emb_name: random`, and estimators `'no change'` and
`'context mean'`. The discipline this project applies to protein design
objectives is already standard practice in their cell-perturbation work. That is
useful precedent -- the argument is not that the field lacks the idea, but that
it is not applied to binding specificity.

## 14. The genomic rung: crossing to DNA, where the anti-correlation does not follow

Sections 9 and 12 are all protein models reading protein sequences, so the
standing alternative explanation was that something about protein-LM pretraining
is at fault. This tests the claim one modality down, on the DNA encoding the very
same variants.

**Data, verified rather than assumed.** All four CDS were fetched from ENA via
UniProt cross-references and each was accepted only after translating to an exact
match against the protein sequence already pinned in `crosstalk.boltz`
(`scripts/fetch_cds.py`): ParD3 `AEH90010.1` 282 nt, ParE3 `AEH90009.1` 312 nt,
ParD2 `AEH89587.1`, ParE2 `AEH89588.1`. Codon usage is the organism's own,
computed from all 6,508 CDS of the WSM2075 chromosome (CP002279, 1,980,535
codons, GC 63.6% with the expected GC3 bias).

Two things are possible here that the protein rung could not do.

**The partner-aware context is the real operon.** Locating the genes in the
genome (`scripts/build_genomic_context.py`) shows ParD3 and ParE3 do not merely
sit adjacent, they **overlap by 11 nt** -- the translationally-coupled
arrangement typical of toxin-antitoxin pairs. So where the protein rung had to
invent a 25-glycine linker to get two chains into one context, the genomic rung
scores the locus the organism carries and the model was trained on. Only the
deliberately non-natural ParD3:ParE2 control is constructed.

**Tokenization transfers exactly.** The ParD3 CDS is 282 nt = 47 non-overlapping
6-mers with no offset, so a token is exactly two codons and the three mutated
codons fall in three distinct tokens (30, 31, 39). The masked-marginal shortcut
therefore costs three forward passes for the whole landscape, the same as ESM-2.
Wild type scores exactly 0.0 by construction, which is the check that the token
arithmetic is right.

### Result: non-predictive, not anti-predictive

Partner-blind AUC on the same 399-variant discrimination set:

| model | AUC | |
|---|---|---|
| trivial baseline: mutation count | **0.664** [0.625, 0.702] | |
| ESM-2 650M likelihood (section 9) | **0.151** | anti-predictive |
| AIDO.Protein-RAG-3B (section 12) | 0.212 | anti-predictive |
| NT-v2 50M multi-species | 0.505 [0.445, 0.561] | chance |
| NT-v2 100M multi-species | 0.460 [0.401, 0.518] | chance |
| NT-v2 250M multi-species | 0.599 [0.543, 0.652] | chance |
| NT-v2 500M multi-species | 0.393 [0.334, 0.451] | chance |
| HyenaDNA (autoregressive, single-nt) | 0.483 [0.424, 0.541] | chance |

**This is a dissociation, not a replication.** Protein-LM likelihood points the
wrong way; genomic-LM likelihood points nowhere. Where ESM-2 650M had rho +0.510
on-target and +0.540 off-target, NT sits at -0.10 / -0.08 -- it is not tracking
protein binding at all, in either direction.

The scale ladder makes the point sharper by *failing* to have a trend. The
protein ladder fell monotonically (0.296 -> 0.151 from 8M to 650M): more capacity,
more naturalness, more harm. The genomic ladder wanders -- 0.505, 0.460, 0.599,
0.393 -- with every CI straddling or near chance and no monotone direction. That
is the signature of noise, not of a signal pointing anywhere.

HyenaDNA closes the two remaining loopholes at once: single-nucleotide
tokenization (no codon is ever straddled) and an autoregressive rather than
masked objective. It sits at 0.483. Neither tokenization nor training objective
explains the result.

### The synonymous floor: how much of a genomic score could possibly be signal

A protein variant does not determine its DNA. Synonymous encodings translate
identically and therefore carry an *identical* measured fitness -- the assay reads
protein binding, so codon choice cannot move the label. Any score variance across
them is provably specificity-irrelevant. No protein rung can measure this.

Sampling synonymous encodings weighted by genome codon usage (the choice a real
pipeline would make), full pseudo-likelihood, ~8 encodings per variant:

| model | within-variant SD | between-variant SD | ratio | attenuation ceiling |
|---|---|---|---|---|
| NT-v2 50M | 3.46 | 6.26 | 0.552 | 0.834 |
| NT-v2 100M | 2.66 | 5.71 | 0.465 | 0.885 |
| NT-v2 250M | 2.25 | 4.46 | 0.505 | 0.863 |
| NT-v2 500M | 3.26 | 6.91 | 0.471 | 0.882 |

Roughly **half the score spread between different variants is reproduced by
changing codons that cannot change the answer**, i.e. about a quarter of the
variance is label-irrelevant by construction. That is a real and previously
unquantified cost of using a genomic LM as a protein-fitness proxy.

It is not, however, the explanation. The implied attenuation on any correlation
is only 0.83-0.89, so noise-correcting the observed rho of about -0.10 still
leaves about -0.11. **The genomic model is not a good proxy that codon noise
spoiled; it carries essentially no protein-binding signal to spoil.**

### Scope

One landscape, one organism, one gene pair. NT-v2 multi-species includes bacterial
genomes but this specific locus's representation in pretraining is unknown, and
Evo 2 -- the model most likely to do better -- needs a GPU and has not been run.

## 15. Readout versus representation, and how much the split decides

Every rung so far reads one number out of a model: a likelihood. That is a claim
about a *readout*, not about a *representation*, and the two can come apart. A
linear probe on frozen embeddings tests the second directly
(`scripts/run_readout_probe.py`): no fine-tuning, ridge regression, same
discrimination set, same metrics.

**The split turned out to be the entire experiment**, and getting it wrong the
first time is the most useful thing in this section.

| features | random split | residue split | within-background |
|---|---|---|---|
| ESM-2 650M embedding | rho 0.923 / AUC 0.997 | 0.638 / 0.862 | **+0.155** |
| NT-v2 50M embedding (DNA) | 0.899 / 0.995 | 0.614 / 0.852 | **+0.088** |
| **trivial: one-hot over 3 sites** | 0.885 / 0.974 | **0.849 / 0.952** | **0.000** |

Read the columns right to left.

**Random splits are meaningless here.** Variants differ at three positions only,
so a random split leaves most (position, residue) combinations present in
training. Everything scores ~0.99, including a 60-parameter lookup table.

**Holding out amino acids is not enough either.** Hold out a variant because it
contains an unseen residue and it still carries two *seen* residues, and an
additive lookup predicts it from those. That is why one-hot scores 0.952 and
**beats both learned representations** -- not because it generalised, but because
two thirds of every test variant was familiar. Any paper reporting an embedding
win on this kind of split against a weaker baseline is reporting the baseline's
absence.

**The clean test** groups test variants by (mutated position, the two residues at
the other positions), so within a group every variant shares an identical
background and differs only at a position holding a residue never seen in
training. One-hot is uninformative *by construction* there: those weights were
never updated, so every member of a group gets the same prediction. Anything
above zero is transferred chemistry.

Under that test the ordering finally means something: **ESM-2 +0.155, the DNA
model +0.088, the lookup table 0.000** (3,814 background groups, 15,087 variants).

Two things follow, and the second is the one worth the section:

1. **The dissociation is real but modest.** The same ESM-2 650M whose likelihood
   scores AUC 0.151 -- actively anti-predictive -- has a representation that is
   positively predictive of the same quantity. The information is in the model;
   the likelihood inverts it. But the margin over a lookup table is +0.155, not
   the +0.86 the leaky split advertised.
2. **The genomic model's representation knows something its likelihood does not.**
   NT's likelihood is at chance (section 14) while its embeddings reach +0.088 on
   unseen residues. Even a DNA model carries some protein-level signal that
   scoring the sequence discards entirely.

**Split choice moved the same features, on the same task, from rho 0.155 to
0.997 -- a factor of six.** That is the practical warning: on a combinatorially
complete landscape, the evaluation protocol determines the conclusion more than
the model does.

## 16. The generality test on measured data, and it does not reproduce

Sections 9-15 are one protein. The obvious test is whether any of it holds
elsewhere, on measured rather than simulated data.

**The dataset.** SKEMPI 2.0 curates measured binding affinities for mutations in
solved complexes. Buried in it is exactly the needed structure: proteins whose
*same* mutations were measured against *two different partners*. Extracting it
(`crosstalk/skempi.py`) gives **15 system pairs, 249 verified two-sided single
mutations**, across about ten independent biological systems -- protease/inhibitor
(BPTI vs trypsin and chymotrypsin), hormone/receptor (hGH vs its own receptor and
the prolactin receptor), enzyme/inhibitor (BLIP vs TEM-1 and SHV-1), TCR/pMHC,
antibody/antigen, and RNase inhibitor vs angiogenin and RNase A.

**A numbering bug that quietly destroyed most of the data.** SKEMPI carries two
mutation numberings and they disagree on 63% of rows: `Mutation(s)_cleaned` is
renumbered to be comparable across entries, `Mutation(s)_PDB` is author numbering
in that row's own structure. Matching structures with the cleaned numbering --
the natural choice, since it is the one that joins across entries -- silently
dropped BPTI from 31 mutations to 0, MT-SP1 from 27 to 1, and the whole TCR set to
0, leaving 6 systems and 103 mutations. Using PDB numbering for the structural
lookup and cleaned numbering for the join recovers 15 systems and 249 mutations.
Every mutation is still accepted only if the residue actually present at that
position in that chain matches the stated wild type.

**Result: nothing works, and the trivial baseline wins.** Leave-one-system-out,
so the probe never sees the system it is scored on, target = ddG_B - ddG_A:

| method | pooled rho (within-system z, n=249) | systems positive |
|---|---|---|
| ESM-2 650M likelihood | **+0.027** | 8/15 |
| ESM-2 650M embedding, LOSO probe | **+0.020** | 9/15 |
| trivial: BLOSUM62 + hydropathy + volume, LOSO | **+0.159** | -- |

Per-system correlations range from -0.435 to +0.936, which at n=10-31 is noise:
n=20 needs |rho| > 0.44 to clear p=0.05.

**The null is real, not underpowered.** The obvious defence is that SKEMPI pools
assays from many labs and the margin is a difference of two noisy numbers, so
nothing could correlate. That is testable the same way section 0 calibrated the
ParD3 oracle. 365 mutations in SKEMPI are measured by more than one independent
reference; the cross-reference spread gives a single-measurement SD of **0.459
kcal/mol**, so margin noise is **0.649**. Against a two-sided margin SD of
**2.490**, the noise-to-signal ratio is 0.261 and the **attenuation ceiling on any
correlation is 0.965**. A true correlation of 0.5 would have shown up as 0.48.
It did not show up.

> **Superseded in part by section 26.** The SKEMPI null here is real for SKEMPI's
> *mutation sampling*, but it is not a refutation. Section 26 shows the two
> sources agree at rho +0.919 on the 27 mutations they share, and that the
> effect is absent from those 27 and present in the 201 dense-DMS mutations
> SKEMPI lacks. SKEMPI's BPTI entry is 48% alanine substitutions with 17 of 31
> at a single position. Read the paragraph below as "curated affinity data
> cannot see this", not as "the effect is not there".

**What this costs the argument.** The ParD3 findings do not currently generalise.
On measured data from ten other systems, ESM-2 likelihood is not anti-predictive
(it is ~0), a frozen-embedding probe is not predictive, and a substitution matrix
plus two physicochemical scalars beats both. The honest position is that sections
9-15 are established *for the ParD3 landscape* and are contradicted, not merely
unconfirmed, as a general claim about protein language models and specificity.

Two differences that may matter and are not yet separated: SKEMPI is single point
mutations at diverse interfaces with a modest dynamic range, whereas ParD3 is a
combinatorially complete three-site landscape scored on a sharper task
(specific-versus-promiscuous among *strong* binders). Whether the ParD3 effect
needs that task structure, or is simply particular to ParD3, is unresolved and is
the thing to resolve before any of it is written up as general.

## 17. The transfer claim, falsified: what looked protein-level is codon leakage

Section 15 reported that the genomic model's *representation* reaches rho +0.088
on unseen residues while its likelihood sits at chance, and read that as weak
genomic-to-protein transfer. Three tests were built to try to break that reading.
All three broke it (`scripts/run_transfer_analysis.py`).

### The causal controls, which are the decisive part

Two manipulations move in opposite directions, and a genuinely protein-level
representation makes an unambiguous prediction about each: destroying the reading
frame must destroy the signal, and changing the DNA while holding the protein
fixed must not.

Within-background rho for a probe trained on canonical encodings (mean-pooled
NT-v2 50M embeddings, 3,814 background groups):

| condition | protein | DNA | rho |
|---|---|---|---|
| canonical | reference | reference | **+0.141** |
| random mutant codon | identical | codon at the varying site randomised | **+0.143** |
| frameshift +1 | **destroyed** | nearly unchanged | **+0.143** |
| reverse complement | **destroyed** | wrong strand | **+0.129** |
| synonymous scramble | identical | all 94 codons replaced | **+0.064** |

**The signal is completely insensitive to whether the sequence is even in frame,
or on the coding strand.** Frameshifting by one nucleotide destroys every codon
downstream and changes nothing (+0.143 against +0.141). Reading the opposite
strand, where the sequence is not a coding sequence at all, costs almost nothing.
No protein-level representation can be invariant to that. This is the finding,
and it does not depend on any of the other rows.

The mechanism is mundane once seen, and it is simpler than it first appears.
Under *any* encoding rule the codon determines the amino acid, so the nucleotides
at the varying site already *are* residue identity. Nothing protein-level has to
be represented for a linear probe to recover them, and re-framing or
complementing the sequence leaves that information fully intact.

**A correction, since the obvious fix does not work.** The first version of this
section attributed the leak to the preferred-codon rule making the codon a
deterministic function of the amino acid, and predicted that randomising the
substituted codon would remove it. It does not: `random mutant codon` scores
+0.143, indistinguishable from canonical. The direction that matters is
codon-to-amino-acid, which is deterministic in every encoding, so randomising
which synonymous codon is used makes the map one-to-many without hiding anything.

**And the one row that does drop should not be read as leak removal.** Synonymous
scramble replaces all 94 codons rather than the one under test, so it is a global
perturbation far off the model's training distribution as well as a
protein-preserving one. Its fall to +0.064 is therefore confounded and is
reported only for completeness. The frameshift and reverse-complement rows carry
the argument; the scramble does not.

### The variance decomposition agrees

Synonymous encodings hold the protein fixed, so the fraction of variance
explained by amino-acid identity measures how protein-level a quantity is. A
protein model scores 1.000 by construction, since it never sees DNA.

| quantity | eta^2 from amino-acid identity |
|---|---|
| NT likelihood | 0.643 |
| NT representation | 0.672 |
| any protein model | 1.000 |

About a third of the genomic representation's variance is synonymous codon
choice, which cannot move a measured label. And the representation is barely more
protein-level than the likelihood (0.672 vs 0.643), so the readout-versus-
representation gap that motivated section 15 is not there on the genomic side.

### The alignment is weaker than a lookup table's

If the two modalities converged on a shared account of these molecules, the DNA
representation should align with the protein representation better than a trivial
encoding does. Linear CKA and a cross-modal ridge map fitted with whole amino
acids held out:

| pair | CKA | held-out R^2 |
|---|---|---|
| NT -> ESM-2 | 0.663 | +0.412 |
| **one-hot -> ESM-2** | **0.819** | **+0.553** |
| one-hot -> NT | 0.713 | +0.533 |

**A 60-parameter one-hot vector aligns with ESM-2 better than the genomic model
does.** Whatever NT shares with ESM-2 is less than what residue identity alone
shares with it, so there is nothing here that deserves the name cross-modal
transfer. NT is best described as a noisy re-encoding of which residue sits
where.

### Scope and one honest wrinkle

The canonical rho here is +0.141 against +0.088 in section 15 because the feature
set differs: section 15 concatenated pooled and per-site embeddings (2,048 dims),
this uses mean-pooled only (512). The comparison across conditions is internally
matched, which is what the controls require.

The controls establish that the apparent transfer on *this* landscape is codon
leakage. They do not establish that no genomic model could transfer: a
codon-randomised scoring rule would remove the leak by construction, and Evo 2 --
the model most likely to behave differently -- still has not been run.

## 18. Genomic-to-protein transfer across 25 DMS assays: no transfer, and the null is real

Sections 14 and 17 are one landscape. This is the generality test
(`scripts/run_dms_transfer.py`), on measured deep mutational scans.

**Setup.** Every ProteinGym assay for which a native coding sequence could be
verified (`scripts/fetch_dms_cds.py`: 29 of 41 resolved -- an assay is kept only
if some ENA CDS for its UniProt accession translates to a protein *containing*
the assayed target exactly, with the offset recorded). After dropping assays
longer than 420 aa or with fewer than 200 usable single mutants, **25 assays
across 9 organisms** remain: human, yeast, *E. coli*, *B. subtilis*,
*P. aeruginosa*, *K. pneumoniae*, *S. acidocaldarius*, SARS-CoV-2 and HIV-1.

The comparison is like-for-like: identical masked-marginal protocol, same region,
the assayed target sequence and the CDS encoding exactly it. Only the modality
differs.

| proxy | mean rho over 25 assays | 95% CI | assays positive |
|---|---|---|---|
| ESM-2 650M on the protein | **+0.466** | [+0.395, +0.536] | **25/25** |
| NT-v2 50M on the CDS | **-0.013** | [-0.033, +0.007] | 10/25 |
| NT-v2 50M, codon-marginalised | **-0.019** | [-0.041, +0.004] | -- |

Paired difference **+0.479** [+0.404, +0.555], t=12.3. ESM-2 has the larger |rho|
on **24 of 25** assays. The genomic model is positive on 10 of 25, a coin flip
(sign test p=0.42), its confidence interval contains zero, and its largest |rho|
on any assay is 0.146 against ESM-2's 0.737.

**The null is not a noise artifact.** Scoring a variant with a genomic LM requires
choosing a codon, and the same masked pass prices every synonymous alternative,
so the spread across encodings that translate identically -- and therefore share
a DMS score exactly -- comes free. Averaged over assays that spread implies a
**mean attenuation ceiling of 0.900**: a true correlation of 0.5 would still have
appeared as 0.45. Nothing appeared.

**Marginalising over codons does not rescue it.** Section 17 showed that
committing to one codon lets a proxy read residue identity off the nucleotides.
Averaging the score over all synonymous encodings removes that channel by
construction and is the genomic score actually entitled to be called a
protein-level proxy. It scores -0.019. The leak was never doing useful work; it
was only ever an alternative route to residue identity.

### The trivial baselines, added 1 Sept 2026, which change how to state this

Section 18 originally reported only ESM-2 and NT-v2, with no trivial baseline. That
was the project's own standing rule broken. Scoring the identical mutation sets
with the chemistry features already implemented in `run_skempi_transfer.py`
(`scripts/run_dms_trivial_baselines.py`, mutation counts matched 25/25):

| scorer | per assay (n=25) | per cluster (n=21) |
|---|---|---|
| ESM-2 650M likelihood | +0.465 [+0.391, +0.540] | +0.442 |
| **BLOSUM62 alone** | **+0.228** [+0.193, +0.263], 24/25 | +0.218 |
| signed Kyte-Doolittle change | +0.175 [+0.142, +0.209] | +0.173 |
| -\|volume change\| | +0.107 [+0.086, +0.127] | +0.100 |
| **chem ridge, leave-one-cluster-out** | **+0.286** [+0.245, +0.326] | +0.276 |
| NT-v2 50M likelihood | -0.013 [-0.034, +0.008] | -0.012 |
| NT-v2 codon-marginalised | -0.018 [-0.042, +0.006] | -0.017 |

A substitution matrix alone reaches 49% of ESM-2, and five fitted chemistry
features reach 61%. ESM-2's margin over BLOSUM62, paired within assay, is
**+0.237** [+0.178, +0.296], higher in 23 of 25.

**This makes the claim cheaper and stronger rather than weaker.** The bar a
genomic model has to clear is **+0.228, not +0.466**, and it does not clear it:
NT-v2 loses to BLOSUM62 by **+0.241** [+0.200, +0.283] in 24 of 25 assays. The
headline is therefore best stated as *a genomic language model scores below a
substitution matrix on protein fitness*, which needs no appeal to a large protein
model at all.

It also sets the honest bar for any future protein-model claim here: ESM-2's
genuinely learned margin over fitted chemistry is +0.179, not +0.465.

### Where this leaves the transfer claim

Stated plainly, and it is a negative result: **on 25 measured DMS assays spanning
9 organisms, a genomic language model scoring the native coding sequence carries
no protein-fitness signal, while a protein language model on the identical region
reaches rho +0.47.** On ParD3 the same holds for binding specificity across four
NT scales and a second architecture with a different tokenization and training
objective (section 14), and what looked like representation-level transfer there
survives frameshifting and reverse-complementing the sequence, so it was never
protein-level to begin with (section 17).

Three caveats, none of which the evidence currently supports leaning on:

- **Scale.** Only NT-v2 50M was run across all 25 assays. The ParD3 ladder covers
  50M-500M and shows no trend, but the DMS sweep does not.
- **Evo 2 is untested** and is the model most likely to differ; it needs a GPU.
- **Coding sequences are short.** Genomic LMs are built for long-range genomic
  context, and a 300-1,200 nt CDS in isolation may simply be the wrong input for
  them, even though it is the input any protein-fitness application would supply.

What the result does establish is that the obvious way to use a genomic model as
a protein-fitness or specificity proxy -- score the coding sequence -- does not
work, is not rescued by marginalising codons, and is not limited by measurement
noise.

## 19. The context objection, tested directly: more genome makes it worse

Section 18's standing objection is that genomic language models are built for
long-range context and were handed an isolated 300-1,200 nt coding sequence -- the
wrong input, so the null is uninformative. Two tests, one weak and one decisive.

**The weak one.** Refetching each DMS assay's parent nucleotide record
(`scripts/fetch_dms_context.py`) resolves 21 of 29 assays, only 10 of them
genuine `Genomic_DNA`. Most EMBL cross-references are gene-only entries, so the
available flanks are tiny -- IF1_ECOLI's whole record is 328 nt, RASH_HUMAN's is
an mRNA with zero upstream. Only a handful (BLAT 204/3000, ESTA_BACSU 3000/3000,
TRPC_SACS2 2232/4) carry real neighbourhood. This arm cannot settle the question
and is not asked to.

**The decisive one.** ParD3 can settle it, because the entire 6.9 Mb
*M. opportunistum* chromosome is on disk and the locus's coordinates are known, so
the real upstream and downstream sequence can be dialled up to the model's full
context (`scripts/run_context_doseresponse.py`). Flanks are snapped to multiples
of 6 so codons never straddle 6-mer token boundaries, which would otherwise
confound a context effect with a tokenisation artifact.

| real flank each side | total nt | tokens | AUC | 95% CI |
|---|---|---|---|---|
| 0 | 282 | 47 | 0.505 | [0.445, 0.561] |
| 60 | 402 | 67 | 0.511 | [0.453, 0.569] |
| 300 | 882 | 147 | 0.492 | [0.434, 0.552] |
| 1,200 | 2,682 | 447 | **0.379** | [0.325, 0.433] |
| 3,000 | 6,282 | 1,047 | **0.365** | [0.311, 0.421] |
| 5,400 | 11,082 | 1,847 | 0.410 | [0.354, 0.468] |
| *trivial baseline: mutation count* | | | **0.664** | |

**MECHANISM CORRECTED BY SECTION 35.** The trend below is real but it is NOT
evidence that *genomic* context hurts. Composition-matched shuffled flanks, which
carry no genomic information at all, reproduce it: mononucleotide-shuffled trend
**-0.633** across three seeds against real's -0.771. It is a length and composition
effect on the score scale. The trend is also grid-dependent, -0.430 on a denser
grid, with the curve turning back up past 3 kb. The conclusion that context does
not rescue NT survives; the sentence below does not, and should not be quoted.

**More real genomic context monotonically hurts** (Spearman AUC against flank
size **-0.771**). At 11 kb the model is being given 1,847 tokens -- close to its
full 2,048-token context, and all of it the organism's actual chromosome -- and
the specificity signal is still absent, in fact further below chance than with no
context at all. Nothing approaches the mutation-count baseline at any context
size.

The objection is therefore answered on this landscape: the failure in sections 14
and 18 is not an artifact of withholding genomic context. Supplying the real thing,
up to the model's capacity, does not produce a signal. Together with the real
ParD3:ParE3 operon arm of section 14 -- the natural locus, 11 nt gene overlap
included, also at chance -- there is no version of "give it proper genomic input"
left that has not been tried at this scale.

What remains untested is a model built for much longer range and much larger
capacity. That is Evo 2, which needs a GPU;
`notebooks/Evo2_crosstalk_GPU.ipynb` runs every arm above under the identical
protocol and is the one outstanding arm.

## 20. The DMS null holds at every genomic model scale

Section 18 ran one genomic model across the 25 assays, leaving open whether a
larger one would recover the signal. The ParD3 ladder showed no trend, but ParD3
is one protein. Repeating the full sweep at four scales
(`scripts/run_dms_transfer.py --nt ...`, `results/dms_scale_sweep.csv`):

| genomic model | mean rho | 95% CI | codon-marginalised | attenuation ceiling | ESM-2 larger \|rho\| |
|---|---|---|---|---|---|
| NT-v2 50M | -0.013 | [-0.033, +0.007] | -0.018 | 0.900 | 24/25 |
| NT-v2 100M | -0.013 | [-0.037, +0.011] | -0.025 | 0.899 | 24/25 |
| NT-v2 250M | -0.007 | [-0.025, +0.012] | -0.017 | 0.880 | 24/25 |
| NT-v2 500M | +0.008 | [-0.016, +0.032] | +0.010 | 0.880 | 24/25 |
| *ESM-2 650M on the protein* | **+0.465** | [+0.395, +0.536] | | | |

Every interval contains zero. A tenfold increase in parameters moves the mean
correlation by 0.021, from -0.013 to +0.008, which is inside the noise of a single
assay. The protein model has the larger absolute correlation on 24 of 25 assays at
every scale, and the same assay is the exception each time.

Two readings are now excluded together. Section 19 excluded missing genomic
context on ParD3; this excludes insufficient capacity across 25 assays and nine
organisms. What remains is a model class built for much longer range than any
tested here, which is Evo 2.

## 21. Forecasting model unreliability before any labels exist

A rapid-response pipeline scores a sequence nobody has characterised and must
decide without a wet-lab check. Accuracy is unavailable at that moment, because
accuracy needs labels. The operationally useful quantity is a warning computed
from the model and the sequence alone.

Shibboleth Arm A established that bulk correlation predicts tail performance
(rho +0.683 to TPR at a 1% false-positive budget). That predictor needs labels and
so cannot run at decision time. This asks whether anything label-free can do the
job (`scripts/run_reliability_forecast.py`, `scripts/analyze_reliability_forecast.py`).

**Setup.** 41 ProteinGym assays scored with ESM-2 650M. The outcome is per-assay
Spearman skill, which needs labels and is used only to grade the forecast
afterwards. Every predictor is computed with the DMS scores held back.

ProteinGym contains several assays per protein (three for BLAT_ECOLX, two each for
CP2C9, PTEN, RL401, P53) and two influenza nucleoprotein strains 94% identical to
each other, so 41 assays are not 41 independent observations. Assays are clustered
by protein and by sequence identity, giving **33 independent clusters**.
Permutations shuffle whole clusters and the forecast is scored
leave-one-protein-out.

Observed skill spans +0.017 to +0.738 (SD 0.193), so there is a real range to
predict.

| label-free signal | rho with skill | cluster-permuted p |
|---|---|---|
| **score dispersion** | **+0.458** | **0.0165** |
| wt_agreement (model's argmax equals nature's residue) | +0.370 | 0.0478 |
| mean wild-type log-probability | +0.362 | 0.0565 |
| mean entropy of masked distributions | -0.365 | 0.0565 |
| fraction of confident positions | +0.362 | 0.0625 |
| sequence length | +0.050 | 0.805 |
| number of variants | -0.150 | 0.463 |

**One signal survives correction, and it is the least model-mystical one.** Score
dispersion is the spread of the proxy's own scores across the variant library. A
model that assigns nearly the same score to every variant cannot rank them, and
that failure is visible without knowing the ranking. Assays in the bottom quartile
of dispersion average skill +0.306 against +0.516 for the rest.

The confidence-style signals all sit at p=0.05-0.06 and point the same way, but
none clears correction on its own. They are also mutually redundant, being four
summaries of the same masked distributions.

**The combined forecast does not work at this sample size.** A ridge model over
all seven signals, scored leave-one-protein-out, reaches rho +0.332 against its
cluster-permuted null at p=0.2975. Mean absolute error is 0.126 against a skill SD
of 0.193, so it beats predicting the mean, but not reliably enough to claim.
Combining redundant signals across 33 clusters overfits.

**Where it does work is the operational end.** Ranking assays by forecast and
taking the same number as there are failures:

| threshold | failing assays | caught by the forecast's lowest | chance |
|---|---|---|---|
| skill < 0.10 | 3 | **2** | 0.2 |
| skill < 0.20 | 6 | **4** | 0.9 |
| skill < 0.30 | 9 | 5 | 2.0 |

The two clearest failures are the influenza nucleoprotein assays, where ESM-2
reaches skill +0.020 and +0.028. Both carry the most extreme label-free values in
the set: wild-type agreement 0.104 and 0.114 against 0.35-0.81 elsewhere, and
dispersion 1.20 and 1.17 against 3.0-4.6. The model announces its own uselessness
on those inputs loudly, before any label is consulted.

That the two clearest cases are viral proteins is the relevant part for
rapid-response work, and also the reason not to over-read it: two strains of one
nucleoprotein from one study are a single observation, which is why they were
clustered.

### What this supports and what it does not

Supported: a single label-free statistic, computable at decision time, carries a
real signal about whether a protein language model is about to be useful
(rho +0.458, p=0.017 with correlated assays counted once), and it identifies the
extreme failures better than chance.

Not supported: a calibrated forecast of how well the model will do. The
multivariate predictor does not beat its null, and 33 clusters is too few to fit
one. A deployable version needs the assay count in the hundreds, which
ProteinGym's full release could supply.

Also untested: whether any of this transfers to the regime that matters, where
the input is a genuinely novel agent rather than a studied protein with a deep
homolog record. Every assay here has a parent.

## 22. What the proxy actually measures: a failed prediction, and a better mechanism

Section 9 explained the anti-correlation as "PLM likelihood scores naturalness, wild-type
ParD3 is itself somewhat promiscuous, so naturalness tracks promiscuity." That story makes a
testable within-data prediction. On the **swap** task the target is ParE2 and the wild type's
own natural partner, ParE3, becomes the *off-target* -- so naturalness should point at the
off-target even more directly, and the proxy should be *more* anti-predictive.

It is not. ESM-2 650M, partner-blind:

| task | AUC | rho vs target | rho vs off-target |
|---|---|---|---|
| cognate (target ParE3) | 0.151 [0.115, 0.191] | +0.510 | **+0.540** |
| swap (target ParE2) | 0.401 [0.323, 0.480] | **+0.540** | +0.510 |

The prediction failed: swap is *less* anti-predictive, not more.

The reason is visible in the table, and it is a cleaner mechanism than the original story.
Those are the same two correlations in both rows -- +0.510 with ParE3 and +0.540 with ParE2 --
because the score does not depend on which partner we have designated the target. **PLM
likelihood correlates positively and almost equally with binding to both partners.** It is a
general "is this a functional interface" detector with no channel for partner identity, so it
carries almost no information about the *difference* between them. Whichever partner it
happens to correlate with marginally more decides the sign of the specificity AUC, which is
why the result lands near or below chance rather than reliably above it.

The corrected claim is therefore narrower and stronger. Not "naturalness implies
promiscuity", which is a story about this particular wild type, but: **a single-sequence
likelihood is partner-agnostic by construction, and on a landscape where one sequence binds
two partners it cannot express the difference.** The 0.151 is not the proxy being wrong so
much as the proxy answering a different question and the sign falling where it may.

### The test that would settle it

The mechanism predicts the sign is set by which partner the score correlates with more, which
is a property of the *system*, not of the model. A second landscape with a very differently
balanced wild type would decide it.

BPTI is the natural candidate (Beer et al., JACS 2021, doi:10.1021/jacs.1c08707): 228 single
mutants at 12 interface positions, measured against three proteases, where wild-type
selectivity is enormous rather than marginal:

| partner | wild-type K_D |
|---|---|
| bovine trypsin (cognate) | 10^-14 M |
| alpha-chymotrypsin | 10^-8 M |
| human mesotrypsin | 10^-5 M |

ParD3's wild type prefers its cognate partner by a factor of ~5. BPTI's prefers by 6 to 9
orders of magnitude, across three partners rather than two, in a completely different fold
and assay. If the proxy is partner-agnostic there too -- correlating similarly with all three
proteases -- the mechanism holds and the result generalises beyond one system. If instead it
tracks trypsin specifically, the mechanism is wrong and the ParD3 finding is a special case.

**Blocked on data access, not analysis.** The dataset is a supplementary Excel file; PMC
serves it behind a JavaScript interstitial and the publisher behind a bot check, neither of
which should be worked around. `crosstalk/bpti.py` implements the loader and the test, so it
runs as soon as the file is in `data/raw/`.

## 23. A second real system: the anti-prediction generalises, the mechanism does not

The BPTI landscape is now loaded (Beer et al., JACS 2021): 14,454 variants (228 singles --
matching the paper's stated count exactly -- and 14,226 doubles) at 12 interface positions,
against three proteases. Two parsing hazards worth recording: the doubles sheet uses
quote-wrapped underscore labels (`'T11A_G12A'`), and both sheets end with aggregate rows
(`MAX`, `MIN`, `AVG`, `MAX DDG`, ...) which look like variants and would otherwise be scored
as data. One row carries ~13,000 kcal/mol in all three columns and is dropped.

This is the adversarial case for section 22. Wild-type BPTI prefers its cognate partner by
six to nine orders of magnitude, where wild-type ParD3 prefers by about five-fold.

### The anti-prediction generalises

ESM-2 650M, masked-marginal, separating specific from promiscuous trypsin binders:

| on-target cut | off-target cut | n | AUC |
|---|---|---|---|
| p80 | p60 | 2891 | 0.372 [0.346, 0.398] |
| p80 | p75 | 2891 | 0.321 [0.299, 0.343] |
| p80 | p90 | 2891 | 0.256 [0.236, 0.277] |
| p90 | p60 | 1677 | 0.390 [0.355, 0.424] |
| p90 | p75 | 1677 | 0.328 [0.300, 0.356] |
| p90 | p90 | 1677 | 0.249 [0.221, 0.278] |

Below chance at every threshold, on a different fold, a different assay, three partners
instead of two, and a wild type that is enormously more selective. The ParD3 result was not
a property of that system.

Note also that the trivial baseline behaves differently: mutation count scores 0.499
[0.497, 0.500] here, exactly chance, against 0.664 on ParD3. On BPTI *nothing* works, so
the comparison that matters is against chance rather than against counting mutations.

### But the mechanism from section 22 is wrong as stated

Section 22 claimed the proxy is *partner-agnostic* -- correlating near-equally with each
partner, so that a marginal asymmetry decides the sign. On ParD3 the two correlations were
+0.510 and +0.540, a spread of 0.030, which supported that reading. On BPTI:

| partner | rho with PLM |
|---|---|
| trypsin (**cognate**, K_D 1e-14) | +0.160 |
| chymotrypsin (off-target, 1e-8) | **+0.332** |
| mesotrypsin (off-target, 1e-5) | +0.060 |

The spread is 0.271, nearly ten times ParD3's. The proxy is emphatically *not* indifferent
here. It has a clear preference, and it prefers an **off-target** over the cognate partner.

So neither "naturalness implies promiscuity" (section 9) nor "partner-agnostic by
construction" (section 22) survives contact with the second system. What survives both is
weaker in form and stronger in scope:

> **A single-sequence likelihood has no privileged relationship with the cognate partner,
> even when the wild type is optimised for that partner by six to nine orders of magnitude.**
> On ParD3 it is indifferent between partners; on BPTI it actively prefers the wrong one.
> Either way it cannot be used to select for specificity.

That is the claim two independent systems support, and it is falsifiable in the same way:
find a landscape where PLM likelihood correlates most strongly with the cognate partner, and
it breaks.

### Why this matters more than the original framing

BPTI-trypsin is a textbook case of an evolutionarily optimised interface, exactly the kind of
signal a model trained on natural sequences should have absorbed. That ESM-2's likelihood
tracks chymotrypsin binding better than trypsin binding is a sharper indictment than anything
in the ParD3 data, and it is not explained by the wild type being promiscuous, because this
one is not.

## 24. Scale makes it worse on both systems, and BPTI shows how

The ParD3 ladder fell monotonically with model size (0.296 at 8M to 0.151 at 650M). BPTI,
same protocol, decision set of 1,677 variants:

| model | AUC | rho trypsin (**cognate**) | rho chymotrypsin | rho mesotrypsin |
|---|---|---|---|---|
| mutation count | 0.499 [0.497, 0.500] | | | |
| ESM-2 8M | **0.507** [0.475, 0.540] | -0.102 | -0.089 | -0.170 |
| ESM-2 35M | 0.391 [0.361, 0.422] | -0.020 | +0.154 | -0.071 |
| ESM-2 150M | 0.369 [0.341, 0.398] | +0.166 | +0.269 | +0.008 |
| ESM-2 650M | 0.328 [0.300, 0.356] | +0.160 | **+0.332** | +0.060 |

The same monotone decline, on a second system. "Scale makes it worse" is now a two-system
result rather than a property of one landscape.

BPTI also shows the mechanism more cleanly than ParD3 could, because its smallest model is
**unbiased**. At 8M the three correlations are all slightly negative and close together
(-0.10, -0.09, -0.17), there is no preference between partners, and the AUC is 0.507 -- exactly
chance. Scaling does not degrade a working signal; it *acquires* a preference, and the
preference is for the wrong partner:

    chymotrypsin (off-target)  -0.089 -> +0.154 -> +0.269 -> +0.332    (fastest growth)
    trypsin      (cognate)     -0.102 -> -0.020 -> +0.166 -> +0.160
    mesotrypsin  (off-target)  -0.170 -> -0.071 -> +0.008 -> +0.060

Every partner correlation increases with scale -- the models are learning something real
about this interface -- but the off-target correlation grows fastest, so the *difference*
that specificity depends on moves the wrong way. That is why the AUC falls while the raw
correlations improve, and it is the sharpest statement of the problem in this project:

> **Scaling a protein language model improves what it knows about binding while making it
> worse at choosing between partners.**

On ParD3 even the 8M model was already below chance, so this could not be seen. It took a
system with a genuinely uninformative small model to separate "the proxy is bad" from
"scaling makes it bad".

## 25. Polyspecificity on measured data: underpowered, and what survives anyway

BPTI has three measured partners, so K can be varied on real wet-lab numbers rather than the
Absolut lattice. Same cost model: screening the target costs 1 assay, counter-screening K
off-targets costs 1+K.

**The first run was underpowered and I nearly over-read it.** At 30 seeds a Wilson interval
on a success rate near 0.5 is 0.34 wide, so almost every comparison overlaps. Margin appeared
to *decline* with budget in two configurations (0.67 -> 0.47 and 0.57 -> 0.33), which would
have been a striking anomaly worth a mechanism. Both gaps are about two standard errors. I
tested one candidate explanation -- winner's curse in the nomination rule, since more
candidates means a larger maximum of the noise -- and it was wrong: nominating by the fitted
model rather than the observed mean made things worse, not better. The pattern is noise.

Recording this because the failure mode is the one this whole project is about. A metric that
moves the wrong way invites a mechanism, and the first question has to be whether it moved at
all.

### What survives at n=30

| off-targets | reward | success (B=300) | 95% CI |
|---|---|---|---|
| chymotrypsin | affinity | 0.20 | [0.10, 0.37] |
| chymotrypsin | **margin** | **0.67** | [0.49, 0.81] |
| mesotrypsin | affinity | 0.53 | [0.36, 0.70] |
| mesotrypsin | margin | 0.43 | [0.27, 0.61] |
| both (K=2) | affinity | 0.07 | [0.02, 0.21] |
| both (K=2) | **margin** | **0.57** | [0.39, 0.73] |

Two non-overlapping comparisons, and they agree with the simulated result: the
specificity-aware objective wins where the off-target is a genuine threat -- chymotrypsin,
which the wild type binds at 1e-8 M, and the K=2 case that must avoid both.

It does **not** win against mesotrypsin, where the intervals overlap completely. That is the
weakest off-target (wild-type K_D 1e-5 M, six orders below the cognate) and the one with the
noisiest measurements (median SD 0.115 versus 0.055 for the others). When the off-target is
already barely bound, an affinity-only objective is adequate -- which is the same lesson as
the K=1 crossover in section 8, now visible on measured data.

### At 150 seeds: the Absolut direction replicates on measured data

The apparent "margin declines with budget" pattern disappears, confirming it was noise.

| off-targets | reward | B=100 | B=300 | B=900 | change |
|---|---|---|---|---|---|
| chymotrypsin | affinity | 0.25 | 0.23 | 0.17 | **-0.07** |
| chymotrypsin | **margin** | 0.47 | 0.57 | 0.55 | **+0.07** |
| mesotrypsin | affinity | 0.47 | 0.47 | 0.47 | 0.00 |
| mesotrypsin | **margin** | 0.38 | 0.48 | 0.49 | **+0.11** |
| both (K=2) | affinity | 0.16 | 0.13 | 0.11 | **-0.05** |
| both (K=2) | **margin** | 0.33 | 0.50 | 0.45 | **+0.12** |

Margin improves with budget in all three configurations; affinity declines or is flat in all
three. No single delta is individually significant (a difference of two 150-seed proportions
has SE about 0.056, so +0.12 is roughly two standard errors and +0.07 is not), but the
direction is consistent 6 times out of 6, which is p = 0.016 on a sign test of the pattern.

The level differences are unambiguous. At B=900, margin versus affinity is 0.55 [0.47, 0.62]
versus 0.17 [0.12, 0.24] against chymotrypsin, and 0.45 [0.37, 0.53] versus 0.11 [0.07, 0.17]
at K=2 -- non-overlapping in both.

**The mesotrypsin exception survives.** 0.49 [0.41, 0.57] versus 0.47 [0.40, 0.55] at B=900,
fully overlapping. Against the weakest off-target -- six orders below the cognate at wild type
-- paying to counter-screen buys nothing at any budget. This is the section-8 crossover on
measured data: the case for specificity-aware objectives is a case about off-targets that are
genuine threats.

So the Absolut result does transfer off simulation, with the qualification that it needs the
off-target to matter.

## 26. Reconciling section 16 with section 23: curated affinity data cannot see this effect

Section 16 concluded from SKEMPI that the ParD3 findings are "contradicted, not merely
unconfirmed" as a general claim. Section 23 concluded from the BPTI DMS that they generalise.
Both used **the same system**: SKEMPI entry `2FTL` is BPTI against trypsin and
alpha-chymotrypsin, which is exactly the pair section 23 tested. They disagreed in sign.

Three candidate explanations were tested and two died.

**Not the metric or the task structure.** Scoring the BPTI DMS with SKEMPI's own metric --
rank correlation against the two-sided margin over all variants, no discrimination set --
still gives -0.148, and restricted to the 228 single mutants, **-0.394**. Section 16's
suggestion that the effect might need ParD3's sharper task does not survive: the effect is
present in plain margin correlation on singles.

**Not pooling across systems.** SKEMPI's own BPTI row is +0.027, not a negative value
averaged away by the pool.

**Not data quality.** 27 mutations are measured in both sources. On those, the two agree
strongly:

    rho(JACS DMS trypsin fitness, SKEMPI ddG) = +0.919   (n=27)

Both datasets are right about the biology.

**It is which mutations were sampled.** On those same 27 shared mutations, PLM against the
measured margin gives **-0.007** -- essentially SKEMPI's +0.027, not the DMS's -0.394. The
effect is absent from the mutations SKEMPI contains and present in the ones it does not.

The reason is visible immediately:

| | SKEMPI 2FTL | JACS DMS |
|---|---|---|
| mutations | 31 | 228 |
| substitutions to alanine | **48%** | 5% |
| concentrated at one position | **17 of 31 at residue 15** (P1) | ~19 per position, full scan |

SKEMPI's BPTI entry is an alanine scan plus a saturation series at the P1 residue -- the
classic shape of curated literature data, since those are the experiments people publish. It
is not a representative sample of the interface's mutational space, and the anti-correlation
lives in the part it does not sample.

### What this changes

Section 16's null is correct *for SKEMPI's mutation sampling* and should not be read as a
refutation. The corrected statement is methodological, and it is more useful than either
original claim:

> **Detecting this effect requires densely sampled mutational data. Curated affinity
> databases cannot see it, not because they are noisy or small, but because the mutations in
> them are selected -- heavily alanine, heavily hot-spot -- and the effect does not live
> there.**

That also explains section 16's per-system spread of -0.435 to +0.936, which was attributed
to noise at n=10-31. Some of it is noise, but each system is also a differently-biased
subsample, so the spread is not purely sampling error around a common null.

The practical consequence for the project is that generality must be argued from dense DMS
landscapes -- ParD3, BPTI, Absolut -- and that SKEMPI can only ever provide a weak negative.
It also predicts that the ~9 other SKEMPI systems would show the effect if a dense DMS
existed for them, which is falsifiable as those datasets appear.

## 27. Synthesis: where the specificity information is not

Twenty-six sections were written across several sessions and two threads that did not know
about each other -- a protein/proxy thread and a genomic/representation thread. They
constrain each other, and together they say something narrower and more defensible than any
of them alone.

### The grid

Every cell asks the same question: can this thing tell a specific binder from a promiscuous
one, on data where the answer is measured for both partners?

| level | protein models | genomic models |
|---|---|---|
| **likelihood readout** | **anti-predictive.** AUC 0.151 (ParD3), 0.328 (BPTI), both below chance | **nothing.** 0.393-0.599 across four NT-v2 sizes plus HyenaDNA, no trend |
| **frozen representation** | weak. +0.155 within-background, and a 60-parameter one-hot beats it on a residue split (0.849 vs 0.638) | **artefactual.** +0.088 apparent transfer, falsified in section 17 as codon composition |
| **effect of scale** | **makes it worse**, monotonically, on both systems | no trend to make worse |
| **trivial baseline** | mutation count 0.664 (ParD3); BLOSUM+hydropathy+volume 0.159 (SKEMPI) -- beats every model | one-hot over three sites beats both embeddings |

### What the two threads each contribute

The protein thread establishes that the likelihood readout is not merely uninformative but
*anti-correlated*, and that the anti-correlation grows with capacity: 0.296 to 0.151 on
ParD3, 0.507 to 0.328 on BPTI. BPTI adds the piece ParD3 could not, because its 8M model sits
exactly at chance with no partner preference -- so scaling does not degrade a working signal,
it acquires a wrong one, with the off-target correlation rising fastest.

The genomic thread rules out two explanations that would otherwise stand. It is not that
protein pretraining is uniquely broken: a genomic model reading the DNA for the same variants
is not anti-predictive, it is *absent*, so the failure does not transfer down a modality.
And it is not that the information is present in the representation and merely mis-read by
the likelihood: section 15 shows a frozen probe barely beating a lookup table, and section 17
shows the part that looked like genuine transfer was nucleotide composition surviving a
frameshift.

Section 26 supplies the boundary condition for all of it: none of this is visible in curated
affinity data, because those mutations are selected -- alanine scans and hot spots -- and the
effect does not live there.

### The claim that survives all of it

> Across two modalities, four model scales, two labs, three dense landscapes and both a
> likelihood readout and a frozen representation, **no configuration encodes
> partner-discriminating information that beats a trivial baseline.** Where a signal exists
> at all it is about binding in general rather than about *which* partner, and in the protein
> likelihood it points against specificity and does so more strongly as the model grows.

### What would break it

Stated so it can be attacked:

1. A dense two-sided DMS on a third system where PLM likelihood correlates *most* with the
   cognate partner. Section 26 predicts the ~9 other SKEMPI systems would behave like BPTI if
   dense data existed for them; a counterexample kills the generalisation.
2. A co-folding model clearing the trivial baseline at adequate n. Boltz has the right sign
   (+0.268 versus ESM-2's -0.294) on 16 variants, which is the one live possibility in the
   whole grid, and the A100 notebook exists to settle it at n=80.
3. A supervised probe trained across systems that beats BLOSUM plus two scalars. Section 16's
   LOSO probe did not (+0.020 versus +0.159), but it was trained on the sparse data section 26
   shows is the wrong substrate.

Item 2 is the one that matters. Everything cheap has failed in the same direction; the
untested case is the expensive one that actually models the complex.

## 28. Section 26 was wrong: the sampling explanation does not survive a causal test

Section 26 explained the SKEMPI/BPTI disagreement as mutation sampling -- SKEMPI's entry is 48%
alanine substitutions with 17 of 31 at one position, so the effect supposedly "does not live
there". That is a causal claim and it is testable directly: impose SKEMPI's sampling structure on
the dense scan and the effect should disappear.

It does not.

| subsample of the 228-mutation dense scan | n | rho(PLM, measured margin) |
|---|---|---|
| full dense scan | 228 | **-0.394** |
| alanine substitutions only (SKEMPI's shape) | 11 | **-0.409** |
| one *random* substitution per position | 12 | -0.410 [-0.804, +0.036] |
| size-matched random subsample | 11 | -0.357 [-0.800, +0.310] |

Every structural reduction preserves the effect. Restricting to alanine does not weaken it;
restricting to one substitution per position does not weaken it. **Alanine bias and positional
concentration are not the explanation.**

The composition analysis across all 15 two-sided systems (`scripts/run_sampling_bias.py`) says
the same thing from the other side. There is almost no composition to correlate with: 13 of the
15 systems are 88-100% alanine with exactly one substitution per position and a positional Gini
of 0.00. Composition barely varies, so it cannot explain a spread of -0.435 to +0.936, and the
measured correlations between composition and |rho| are all within noise (-0.17 to +0.05).

### But small n does not fully explain it either

The obvious fallback is that n=31 is simply too small. Sampling the dense scan at SKEMPI scale:

| n | mean rho | 95% interval | P(rho >= +0.027) |
|---|---|---|---|
| 12 | -0.371 | [-0.783, +0.175] | 6.8% |
| 27 | -0.381 | [-0.663, -0.055] | **0.8%** |
| 31 | -0.387 | [-0.635, -0.077] | **0.6%** |

At SKEMPI's own n=31, a random draw from the dense scan would land at or above +0.027 about 0.6%
of the time. So SKEMPI's BPTI value is not a typical small-sample draw either -- it sits in the
upper 1% tail. With 15 systems examined, seeing one such tail value is unsurprising
(1 - 0.99^15 is about 14%), which is the most likely reading, but it is a coincidence rather than
a mechanism.

### The corrected position

What is actually established:

- The two data sources **agree about the biology**: rho +0.919 between them on the 27 mutations
  both contain.
- They **disagree about the proxy**: -0.394 on the dense scan, +0.027 on SKEMPI's subset.
- That disagreement is **not** explained by alanine bias, positional concentration, or sample size
  alone, all three of which were tested.
- What distinguishes those particular 27 mutations is **unidentified**.

Section 26's headline -- "detecting this effect requires densely sampled mutational data, and
curated affinity databases cannot see it because their mutations are selected" -- is not
supported and should be withdrawn. The defensible version is weaker: two well-measured sources
disagree about this proxy on one system, the disagreement is not explained by the obvious
candidates, and resolving it needs a second system with both a dense scan and curated entries.

Section 16's null therefore stands on its own terms again, and the pointer added to it should be
read as "these two results conflict and the conflict is unresolved" rather than as a correction.

### Why this is recorded rather than quietly fixed

Section 26 was committed, and it was the top-ranked idea in the project docket. It survived one
round of checks -- the metric was matched, pooling was ruled out, the sources were shown to agree
-- and still failed the first genuinely causal test. The pattern across this whole project is
that a clean mechanism is the least reliable part of any result, and that a story consistent with
the data is not the same as a story the data forces.

## 29. The effect is position-dependent, which mostly resolves the SKEMPI conflict

Section 28 refuted the *alanine* explanation for the SKEMPI/BPTI disagreement and left the
conflict unexplained. One candidate had not been tested: SKEMPI's BPTI entry concentrates 17 of
31 mutations at a single position, and the "one substitution per position" control in section 28
spread draws across all positions, which is the opposite of concentration.

Per-position correlation on the dense scan, 19 substitutions at each of 12 positions:

| position | wild type | rho(PLM, measured margin) |
|---|---|---|
| 11 | T | +0.146 |
| 12 | G | +0.135 |
| 13 | P | +0.233 |
| **15** | **K** | **+0.300**  ← P1; SKEMPI puts 17 of 31 here |
| 16 | A | +0.049 |
| 17 | R | -0.204 |
| 18 | I | **-0.482** |
| 34 | V | -0.221 |
| 35 | Y | **-0.493** |
| 36 | G | +0.000 |
| 37 | G | +0.225 |
| 39 | R | **+0.639** |

The whole-scan value of -0.394 is an average over positions that **disagree in sign**, spanning
-0.493 to +0.639.

**The variation is real, not noise at n=19.** Permuting position labels while holding group sizes
fixed gives an observed across-position SD of 0.316 against a null of 0.189 [0.116, 0.276], and
an observed range of 1.132 against a null of 0.655 [0.384, 0.993]. Both give **p = 0.0035**.

### This resolves most of the conflict

Drawing 27 mutations with SKEMPI's actual shape -- 15 of them from position 15 -- rather than
uniformly:

| draw | rho | P(rho >= -0.007) |
|---|---|---|
| uniform 27 | -0.381 [-0.663, -0.055] | 1.4% |
| **SKEMPI-shaped 27** | **-0.159 [-0.488, +0.206]** | **18.9%** |

Concentration at one position raises the probability of SKEMPI's observed value roughly
thirteenfold. Section 26's instinct that sampling was responsible was right; its stated mechanism
was wrong. It is not alanine bias, which section 28 refuted directly. It is *positional*
concentration on a position where the effect reverses sign.

### What is and is not established

Established: positions genuinely differ (p=0.0035), the aggregate is an average over disagreeing
signs, and SKEMPI's concentration largely accounts for its value.

Not established: that position 15 specifically is positive. At n=19 a correlation needs |rho| >
0.46 to clear p=0.05, so +0.300 is not individually significant. Only positions 18, 35 and 39
clear that bar individually. The significant claim is about the *spread*, not any single cell.

Also tempting and not established: that the proxy works at P1 because P1 is the canonical
specificity determinant, the residue that inserts into the protease S1 pocket, so evolution has
demonstrably selected for specificity exactly there while elsewhere naturalness and specificity
decouple. That is a plausible reading of an ordering the data do not yet pin down, and it is
exactly the kind of story that has failed five times in this project. It should be treated as a
hypothesis to test on a third system, not as a result.

**ParD3 cannot test it.** Its landscape has three mutated positions, giving three groups of 19.
The permutation null SD spans [0.031, 0.354], so the observed 0.134 and p=0.65 are uninformative
rather than a negative. All three positions are negative (-0.105, -0.267, -0.433) with no sign
reversal, which is consistent with either reading.

### Consequence for the docket

This raises the value of a third dense two-sided landscape with many mutated positions, and it
sharpens what to look for: not just whether the aggregate anti-correlation reproduces, but
whether it reverses at the system's canonical specificity-determining residue.

### Addendum to 29: the P1 story fails its first test

Section 29 flagged a tempting reading -- that the proxy tracks specificity at the canonical
specificity-determining residue because evolution selected for specificity exactly there. That
makes a measurable prediction without any hand-labelling: per-position rho should rise with how
much the measured specificity margin actually varies at that position.

| relationship | measured |
|---|---|
| rho(per-position rho, SD of specificity margin) | **-0.168** |
| rho(per-position rho, SD of on-target fitness) | +0.252 |
| rho(per-position rho, specificity share of variation) | -0.161 |

| split at the median | mean per-position rho |
|---|---|
| positions where specificity varies **most** | **-0.034** |
| positions where specificity varies **least** | **+0.088** |

The relationship is slightly the *wrong* way round, and nothing here is significant -- 12
positions needs |rho| > 0.58 for p=0.05. Position 15 does have both the largest margin variation
(SD 0.268) and a positive correlation, which fits, but position 39 has among the smallest
variation (SD 0.091) and the largest positive correlation (+0.639), which does not. The aggregate
is driven against the story by that one position.

So the story gets no support and slight evidence against. **The position-dependence in section 29
is real and remains unexplained.** That is the honest state: a measured phenomenon (p=0.0035)
with no identified cause, and one candidate cause now tested and not supported.

This is the sixth mechanism in this project to fail on first contact with a direct test. The
observations have all held; it is the explanations that keep dying. Worth carrying into the
write-up: the position-dependence should be reported as a finding in its own right, not attached
to a mechanism it cannot support.

## 22. The label-free forecast at full scale: 194 assays, and it works

Section 21 measured 41 assays in 33 protein clusters, found one signal that
survived correction, and could not fit a calibrated forecast. This repeats the
measurement on ProteinGym v1: **194 assays in 165 independent protein clusters**,
scored with ESM-2 150M.

Getting the data required locating it. The HuggingFace `ProteinGym` repository
hosts only the older 87-assay v0.1 release and the Marks lab archive returns 404.
The current set lives in `OATML-Markslab/ProteinGym_v1` as five parquet shards
(2.47M rows, 217 assays), with the v1 reference file on GitHub. After filtering to
sequences ESM-2 can encode (<=1022 residues) and assays with at least 200 usable
single mutants, 194 remain.

Observed skill spans -0.018 to +0.762 with mean +0.402 and SD 0.199.

### The baseline that a model-internal signal has to beat

Homolog depth is the obvious predictor of whether an evolutionary model will work,
and ProteinGym ships it as `MSA_Neff_L_category`, computed from an alignment with
no DMS labels involved. It is therefore a legitimate decision-time predictor and
the right comparison.

Skill does rise with homolog depth (Low +0.300, Medium +0.373, High +0.482), and
the category correlates with skill at rho +0.331. As a *forecast*, however, it
fails completely: leave-one-protein-out rho **-0.050**, p=1.0. Three ordered
categories cannot rank 194 assays.

### Model-internal signals, cluster-permuted

| label-free signal | rho with skill | p |
|---|---|---|
| mean entropy of masked distributions | **-0.597** | <0.001 |
| fraction of confident positions | +0.595 | <0.001 |
| mean wild-type log-probability | +0.591 | <0.001 |
| wt_agreement (argmax equals nature's residue) | +0.582 | <0.001 |
| score dispersion | +0.489 | <0.001 |
| MSA Neff category | +0.331 | <0.001 |
| sequence length | -0.320 | <0.001 |
| number of variants | -0.199 | 0.0097 |

At 165 clusters every signal clears correction, and the ordering has changed from
the pilot. Score dispersion won at n=33 because it was the only signal with enough
effect to clear a weak-powered test; with power, the confidence-family signals
overtake it. The strongest single predictor is the mean entropy of the model's own
masked distributions, and its sign is the intuitive one: a model that is uncertain
everywhere is a model about to be useless.

### The forecast

Leave-one-protein-out, so no protein appears in both training and test:

| predictors | LOPO rho | p | MAE |
|---|---|---|---|
| homolog depth only | -0.050 | 1.000 | 0.159 |
| model internals only | +0.547 | 0.0067 | 0.120 |
| internals + depth | +0.579 | 0.0067 | 0.119 |
| **internals + depth + assay size** | **+0.617** | **<0.001** | **0.111** |

Mean absolute error 0.111 against a skill SD of 0.199, so the forecast explains
roughly a third of the variance in how well the model will do, on proteins it was
not fitted to. Section 21's conclusion that a calibrated forecast could not be fit
was a sample-size limitation, not a ceiling.

Calibration is close to the diagonal, with compression at the top:

| forecast quartile | predicted | actual | n |
|---|---|---|---|
| 1 | +0.196 | +0.172 | 49 |
| 2 | +0.395 | +0.419 | 48 |
| 3 | +0.469 | +0.503 | 48 |
| 4 | +0.549 | +0.515 | 49 |

### The operational number

Ranking assays by forecast and taking as many as there are failures:

| threshold | failing assays | caught | chance | precision |
|---|---|---|---|---|
| skill < 0.10 | 24 | **19** | 3.0 | 0.79 |
| skill < 0.20 | 40 | 28 | 8.2 | 0.70 |
| skill < 0.30 | 54 | 37 | 15.0 | 0.69 |

Nineteen of the twenty-four assays where ESM-2 150M is useless are identified
before any label is consulted, against a chance rate of three.

### Viral proteins, and the control that matters

Skill by taxon: Human +0.452, Eukaryote +0.438, Prokaryote +0.410, **Virus
+0.172** with a median of +0.068 and 19 of 26 assays below 0.20. The model is
close to useless on viral proteins, which is the category a rapid-response
pipeline most needs.

That raises the obvious objection: viral proteins are also the least familiar to
the model, so the forecast might be a virus detector wearing a forecast's clothes.
It is not.

| restriction | n | clusters | LOPO rho |
|---|---|---|---|
| all assays | 194 | 165 | +0.617 |
| **non-viral only** | 168 | 142 | **+0.490** |
| Human only | 85 | 72 | +0.398 |
| Prokaryote only | 46 | 41 | +0.579 |
| Eukaryote only | 37 | 30 | +0.657 |
| *taxon one-hot alone* | 194 | 165 | **-0.028** |
| taxon + internals + depth + size | 194 | 165 | +0.621 |

The forecast works inside every taxon, including the 85 human assays where no
viral contrast exists. Taxon by itself predicts nothing out of fold (-0.028) and
adds nothing to the model-internal signals (+0.621 against +0.617). Whatever the
signals are reading, it is not species.

### Scope

This sweep is ESM-2 **150M**, chosen because the machine was contended and 650M
would have taken the night. The 41-assay pilot in section 21 used 650M and gave
weaker univariate correlations with a non-significant forecast, which is
consistent with its 33 clusters but does not exclude a scale effect. The full
650M sweep is the outstanding check, and the run is resumable.

A forecast is fitted to *assay* skill. It predicts whether a proxy will rank a
mutational library well, which is the quantity a design pipeline depends on, and
it is not the same as predicting the error on any single variant.

Every protein here has homologs. The regime where a sequence has no relatives at
all remains untested, and the viral results are the closest available proxy for it.

### One bug worth recording

The sweep died twice at assay 23 with an IndexError. The cause was
`(input_ids == mask_token_id).nonzero()[0, 0]`, which intermittently returned an
empty tensor on MPS under memory pressure even though the mask was present. The
mask index is deterministic (`<cls>` occupies position 0, so a mask at
1-based residue p lands at token index p), so the search was replaced by the
analytic index with an assertion. A search for something guaranteed to be at a
known index is a failure mode waiting to happen.

## 23. A coverage metric for detectors, and what current proxies score on it

Detection and screening systems are evaluated per instance: AUC, or a true
positive rate at a fixed false-positive budget. Those answer "on a variant drawn
at random, how often is the detector right". Neither side of a real contest asks
that.

Someone probing a detector needs one thing that works and is not flagged, so
their quantity is a maximum over a set they can search. A system needs the whole
viable space handled and is undone by any region it misses, so its quantity is a
property of a set rather than an average over instances. Two detectors with
identical true positive rates can differ enormously in how easy their misses are
to reach.

`crosstalk/coverage.py` defines the metric family and
`scripts/run_coverage_metric.py` scores detectors on it. This is computable
exactly on ParD3, and essentially only there, because the landscape is
combinatorially complete: the viable set and the missed set are enumerated rather
than estimated, and single substitutions connect them into a graph.

Viable set: 714 variants with ParE3 W >= 0.8. Negatives: 6,035 with W < 0.5. The
threshold is set on the negatives alone at each false-positive budget, so the
operating point does not depend on the base rate.

### What the detectors score, at a 1% false-positive budget

| detector | per-instance AUC | coverage | missed | top-10 coverage | hill-climbs ending on a miss |
|---|---|---|---|---|---|
| supervised one-hot (held out) | 0.987 | 0.961 | 28 | 1.00 | 0.000 |
| **mutation count (trivial)** | **0.939** | **0.915** | 61 | **1.00** | **0.000** |
| **ESM-2 650M likelihood** | **0.804** | **0.077** | 659 | **0.10** | **0.995** |
| NT-v2 50M on the CDS | 0.480 | 0.000 | 714 | 0.00 | 0.995 |
| random (null) | 0.488 | 0.004 | 711 | 0.10 | 0.995 |

**The dissociation is the finding.** ESM-2 650M has a per-instance AUC of 0.804,
which reads as a working detector and is far above the random null at 0.488. On
coverage at a 1% budget it is nearly indistinguishable from that null: it flags
7.7% of viable variants against the null's 0.4%, catches one of the ten best
variants against the null's one, and misses the endpoint of 99.5% of ordinary
hill-climbs, exactly as the null does.

Meanwhile counting mutations, which is 0.135 lower in AUC, covers 91.5% of the
viable set, flags every one of the ten best variants, and misses no hill-climb
endpoint at all. A 0.135 gap in per-instance accuracy corresponds to a 0.84 gap
in coverage and a 0.90 gap at the top.

Relaxing the budget to 10% does not rescue ESM-2. Coverage rises to 0.419 and
top-10 coverage to 0.70, still below the trivial baseline's 1.00 at the tighter
budget.

### The statistic that carries the most weight

`hill-climbs ending on a miss` involves no adversarial search. It is plain
directed evolution: start at a random variant, accept the best single
substitution, stop at a local optimum. The climb optimises measured function and
never consults the detector. That 99.5% of those endpoints are unflagged by ESM-2
says the detector is blind precisely where ordinary optimisation lands, which no
per-instance metric reveals.

Top-k coverage says the same thing with more resolution. Being right about the
bulk while blind at the top is the failure an average is least able to show.

### Three honest limits on the structural half

**Connectivity saturates here.** Each variant has 57 neighbours in a 7,882-variant
space, so any large subset is connected. Every detector with a big missed set
reports one component containing 100% of it. The statistic only discriminated for
the supervised detector, whose 28 misses split into 4 components. Connectivity is
well defined but uninformative on a graph this dense.

**Hill-climbing has low resolution.** Greedy ascent on measured function from
2,000 starts reaches only 4 distinct local optima, consistent with section 2. The
escape rate is therefore close to a binary readout of whether those few peaks are
flagged. It is meaningful, and its effective sample size is 4.

**No second landscape supports the graph metrics.** The Absolut! landscape has
20,000 sequences, which should be better, but it is a 1-in-256 random subsample of
an enormous space: mean single-substitution neighbour degree is 0.0 and 19,809 of
20,000 sequences have no neighbour at all. Coverage and top-k transfer to it;
connectivity and local search cannot be computed on it.

That last point is part of the answer to why nobody reports a coverage metric.
Computing the structural half needs an exhaustively measured, connected landscape,
and the public ones are either too small and too smooth (ParD3) or too sparsely
sampled to have a graph (Absolut!). Coverage and top-k coverage need only an
exhaustive viable set and can be reported today.

### Smaller artifacts worth noting

Mutation count reaches coverage 1.000 at a 10% budget partly because it takes
four distinct values, so thresholds are coarsely quantised. The supervised
detector returns identical numbers at both budgets because no viable variant
falls between the two thresholds. Neither affects the comparison, and both are
reasons to read a single operating point alongside the curve.

## 24. Serving precision: the ranking survives, individual decisions do not

A proxy evaluated at fp32 and served quantised has not been evaluated. The usual
reassurance is that rank correlation with the original stays near 1.0. That is
true and it is the wrong quantity, because a pipeline acts on a decision taken at
a threshold rather than on a ranking, and a threshold sits wherever the scores are
densest.

`scripts/run_precision_audit.py` scores the whole ParD3 landscape with ESM-2 650M
at each precision. The three-forward-pass trick makes a full audit close to free,
so there is no throughput argument for skipping it.

| precision | AUC | coverage @ FPR 1% | missed | top-10 coverage |
|---|---|---|---|---|
| fp32 | 0.8038 | 0.077 | 659 | 0.10 |
| fp16 | 0.8044 | 0.076 | 660 | 0.10 |
| bf16 | 0.8031 | 0.071 | 663 | 0.10 |
| int8 dynamic | 0.8092 | 0.083 | 655 | 0.20 |

Agreement with fp32, where each build is thresholded at its own operating point,
as a deployment calibrating on the served model would do:

| precision | Pearson | Spearman | max abs difference | decisions flipped | of those, viable |
|---|---|---|---|---|---|
| fp16 | 1.00000 | 1.00000 | 0.039 | 3 | 1 |
| bf16 | 0.99975 | 0.99974 | 0.281 | 11 | 6 |
| int8 dynamic | 0.99601 | 0.99589 | 1.197 | 30 | 10 |

**Both halves of the standard story are visible at once.** Rank correlation is
0.996 even at int8, which is the number usually quoted to justify shipping a
quantised model, and aggregate accuracy genuinely does not move: AUC varies by
0.005 across every precision, and coverage by 0.012. On this task the aggregate
reassurance is earned.

At the same time 30 individual variants change side of the threshold at int8, 10
of them viable, on a single score with a maximum absolute shift of 1.197 log
units. A per-variant decision is not reproducible across builds even though every
summary statistic is.

**An important limit on how much this shows.** ESM-2's coverage here is 0.077 at
fp32, close to the floor established in section 23, so there was little left for
quantisation to destroy. A precision audit is most informative on a proxy that
works, and this one is being audited where it does not. The matching audit on the
DMS assays, where ESM-2 reaches mean rho +0.39 at 150M and has real skill to lose,
is written (`scripts/run_precision_dms.py`, fp32 arm complete at +0.3855) and is
queued behind the 650M reliability sweep rather than competing with it for the
GPU.

**A portability note worth recording.** Dynamic int8 quantisation fails on Apple
silicon with `RuntimeError: Didn't find engine for operation
quantized::linear_prepack NoQEngine` unless `torch.backends.quantized.engine` is
set to `qnnpack` first. The default is `none`, and the error reads like a missing
feature rather than an unset flag.

## 25. The forecast weakens as the model improves

Section 21 fitted the forecast on ESM-2 650M over 41 assays and found nothing
significant. Section 22 fitted it on 150M over 194 and found it worked. Two
explanations were confounded: the pilot was underpowered, and the effect might
genuinely differ with model scale.

Both sweeps are now complete at 194 assays, so they can be compared paired within
assay (`scripts/compare_reliability_scales.py`), which removes the between-assay
variance that dominates either sweep alone. **194 assays, 165 independent protein
clusters.**

### The larger model is better, and most so where it was worst

| | 150M | 650M | paired difference |
|---|---|---|---|
| all 194 assays | +0.4018 | +0.4377 | **+0.0360** [+0.0230, +0.0489] |
| Human (n=85) | +0.452 | +0.463 | +0.012 |
| Eukaryote (n=37) | +0.438 | +0.451 | +0.013 |
| Prokaryote (n=46) | +0.410 | +0.469 | +0.059 |
| **Virus (n=26)** | +0.172 | +0.280 | **+0.108** |

650M wins on 134 of 194 assays. The gain concentrates in viral proteins, where the
smaller model was nearly useless, and is small on human proteins, where it already
worked. Scale buys competence on unfamiliar sequence and little where the model is
already at home. Even at 650M, viral proteins remain the worst category by a wide
margin (+0.280 against +0.46 elsewhere, with 13 of 26 below skill 0.20).

### The forecast survives at 650M and loses much of its strength

| model | strongest single signal | LOPO forecast | p | catches skill < 0.20 |
|---|---|---|---|---|
| 150M | mean entropy -0.597 | **+0.617** | <0.001 | 28/40 (chance 8.2) |
| 650M | frac_confident +0.342 | **+0.416** | 0.0133 | 20/28 (chance 4.0) |

Both clear their cluster-permuted nulls, so section 21's null was partly a power
problem. The effect is nonetheless much weaker at the larger scale, and not
because there is less to predict: skill SD is 0.199 at 150M against 0.191 at 650M.
Every internal signal loses roughly 40% of its correlation, mean entropy from
-0.597 to -0.334.

The nested models show where the loss falls:

| predictors | 150M | 650M |
|---|---|---|
| homolog depth only | -0.050 (p=1.0) | -0.088 (p=1.0) |
| **model internals only** | **+0.547 (p=0.0067)** | **+0.286 (p=0.157)** |
| internals + depth | +0.579 | +0.364 (p=0.033) |
| internals + depth + assay size | +0.617 | +0.416 (p=0.0067) |

**At 650M the model-internal signals alone no longer clear significance.** They
need sequence length, library size and homolog depth alongside them to produce a
working forecast. What was a statement about reading a model's internals becomes
a statement about combining weak internals with metadata.

The taxon control degrades in the same direction. At 150M the forecast worked
inside every taxon; at 650M it barely works on the largest one:

| restriction | 150M | 650M |
|---|---|---|
| all assays | +0.617 | +0.416 |
| non-viral only | +0.490 | +0.319 |
| **Human only** | **+0.398** | **+0.114** |
| Prokaryote only | +0.579 | +0.484 |
| Eukaryote only | +0.657 | +0.457 |
| taxon one-hot alone | -0.028 | -0.265 |

Taxon alone still predicts nothing, so this is not a virus detector at either
scale. But at 650M the forecast has little to say about human proteins, which are
44% of the benchmark.

**This is the uncomfortable direction.** A model's confidence becomes a worse
guide to its own failures as the model improves, on every measure available here,
while the quantity being predicted stays equally variable. The forecast is
sharpest on the model one would least want to deploy. Two scales do not establish
a trend and nothing here shows it continues at served scales, but both points
agree and the mechanism is not mysterious: a more uniformly competent model has
less internal variation to read.

The operational form holds up better than the correlation. At 650M the forecast
still identifies 20 of the 28 assays where the model is unreliable against a
chance rate of 4, and 14 of the 17 worst against a chance rate of 1.5. Calibration
remains close to the diagonal (quartile predictions +0.271, +0.439, +0.493, +0.548
against actuals +0.274, +0.477, +0.500, +0.503), with the same compression at the
top seen at 150M. A ranking that flags where not to trust the model is the form
the answer would take at decision time, and that ranking survives scale better
than any single correlation does.

## 26. RETRACTED IN ITS CENTRAL CLAIM. See the correction below before reading

**Do not cite this section's diagnostic.** An adversarial audit on 2026-08-31 found
a confound that reproduces the entire result and a logical error in its headline
claim. The numbers below stand as measurements; the interpretation does not.

**The confound.** `frameshift(cds, k=1)` is a rotation, `cds[1:] + cds[:1]`. Apart
from one wrap junction that string is real genomic sequence read one base later:
`cds[1:]` is verifiably a substring of the chromosome. The reverse complement is
also real genomic sequence, on the other strand. The two conditions that were NOT
penalised are exactly the two that yield genuine genomic sequence, and the two
that were penalised, codon-order shuffle and synonymous recode, are exactly the two
that yield synthetic sequence. Rank the four conditions by whether they produce a
real genomic string and the table is reproduced without any appeal to reading
frame. NT-v2 behaves as a correct density model over genomes.

**The logical error.** "Reading-frame discrimination is a necessary condition for
ranking protein variants" is false. Ranking point mutants of one gene compares
variants that all share the same frame, and a discriminative ranker never needs to
represent a quantity that is constant across the comparison. Nucleotide
conservation scores with no frame variable at all (phyloP, GERP, CADD) predict
variant effect. Frame is also not identifiable from the string, since a genomic LM
sees all six phases at arbitrary chunk offsets during training.

**Pseudoreplication.** The 29 rows are **24 unique coding sequences**. BLAT_ECOLX
appears three times and CP2C9, PTEN and RL401 twice each, with scores identical to
five decimals. Recomputed on unique sequences with t-based intervals: synonymous
+0.2595 [+0.077, +0.442], codon shuffle +0.3587 [+0.211, +0.506], frameshift
+0.0505 [-0.078, +0.179], reverse complement -0.0375 [-0.169, +0.094]. **The
"twelve times more strongly" ratio becomes 7.1 and its bootstrap interval spans
zero, so that sentence is withdrawn.**

**Power.** At n=24 the minimum detectable effect is about 0.181 nats per token, so
a real frame effect of 0.15 would be missed more often than not. This is a
non-detection at a resolution too coarse to be informative, not a null. About 283
genes would be needed for 80% power at the observed effect size.

**Two further defects.** The synonymous recode excludes the native codon and draws
uniformly over alternatives, so it over-represents rare codons and conflates
protein preservation with wrecked codon usage. And pseudo-likelihood is a sum of
local conditionals, so rotation invariance may be a property of the scoring rule
rather than of the model; the matched protein-side control has not been run.

**The frame question is OPEN, not answered negatively (added after section 29).**
The confound-free replacement probe has now been built and calibrated against a
3-periodic Markov positive control with a matched aperiodic twin. Normalised by
each model's own reference effect, the positive control shows **+3.42%**
[+2.63, +4.20] on the matched contrast and the aperiodic twin shows -0.54%.
Against that yardstick:

- **NT-v2 50M is UNDERPOWERED, not null.** Its interval reaches **+11.59%**, well
  above the 3.42% a frame-representing model shows, and its point estimate
  (+4.38%) is larger than the positive control's. A frame effect of the size a
  frame-aware model exhibits would not have been detected here. Bounding it would
  take roughly **214 unique coding sequences**, against the 24 available.
- **HyenaDNA is bounded, but only just.** Its interval tops out at +3.35% against
  the control's +3.42%, a margin of 1.0x. That is a tie at the boundary, not a
  clean exclusion.
- ESM-2 35M detects the contrast at +4.88%, so the probe works.

**So no claim that a genomic language model lacks reading-frame representation is
supported by anything in this file.** The rotation probe was confounded and the
confound-free probe lacks the power to replace it at this sample size. Earlier
statements in this project that HyenaDNA "fails the frame test", and that its
single-nucleotide tokenisation closes the tokenisation loophole, overstated what
the data support and are withdrawn.

A further narrowing from the order sweep: every 3-periodic Markov order from 0 to
5 passes the rotation test, but orders 1 and 5 fail the stop contrast. A null on
the contrast therefore means "has not learned in-frame stop depletion", which is
narrower than "does not represent the reading frame".

**What survives.** Sections 18, 19 and 20, the DMS null, do not depend on this
section and are unaffected. Section 17 also stands, since its mechanism (codon to
amino acid is deterministic under every encoding, so a probe reads residue
identity off the nucleotides) does not rest on the frameshift row.

**Citation corrections.** Merchant et al. (Nature 2025) used Evo 1, not Evo 2. King
et al. (Science 2026) used Evo 1 and Evo 2 with fine-tuning, prompt engineering and
inference-time guidance from separate predictive models, so their viable phages did
not come from bare likelihood ranking, and the tension this section claimed is
correspondingly smaller.

**The replacement experiment** must match realness across arms: a 1-nucleotide
insertion against a 3-nucleotide insertion at the same site, both synthetic with
only one breaking frame; and an in-frame nonsense SNV against a synonymous SNV in
the same codon, both single-base edits to a real gene with only one destroying the
protein. A GeneMark-style 3-periodic Markov model is the missing positive control
and the missing trivial baseline.

---

## 26. The reading-frame test: a genomic LM that cannot tell a gene from its frameshift

Merchant et al. (Nature 2025) design functional de novo genes with a genomic
language model and King et al. (Science 2026) design whole bacteriophages the same
way, while sections 14 to 20 find that genomic-LM likelihood carries no
protein-fitness signal. The obvious reconciliation is a difference of grain:
generating a plausible gene needs coarse competence, ranking two point mutants
needs fine competence, and a model could have the first without the second.

`scripts/run_granularity_ladder.py` tests that on 29 verified coding sequences
with a factorial of corruptions, each isolating one ingredient. Scores are mean
per-token log-likelihood so lengths are comparable, and every comparison is paired
within gene.

| corruption | what it preserves | what it destroys | mean delta from the real gene | real higher |
|---|---|---|---|---|
| synonymous recode | the protein, exactly | native codon choice | **+0.2334** [+0.0849, +0.3820] | 20/29 |
| codon-order shuffle | codon usage, exactly | the protein | **+0.3218** [+0.1954, +0.4482] | 24/29 |
| frameshift +1 | nucleotide composition | the reading frame | **+0.0264** [-0.0778, +0.1307] | 14/29 |
| reverse complement | composition | coding status entirely | **-0.0643** [-0.1742, +0.0455] | 11/29 |

**The coarse-competence hypothesis is wrong, and the result is worse than that.**
The model cannot distinguish a real gene from the same gene read out of frame: the
difference is +0.026 nats per token with a confidence interval spanning zero, and
the real gene scores higher in 14 of 29 cases, which is a coin flip. It also
cannot distinguish a gene from its reverse complement, where the point estimate
actually favours the complement.

What the model is sensitive to is local sequence statistics. Shuffling codon order
costs 0.322 nats per token even though codon usage is preserved exactly, so the
penalty is for disrupted local context rather than for composition. Synonymous
recoding costs 0.233. The model discriminates codon context **twelve times more
strongly than it discriminates reading frame**.

That is a coherent account of everything in sections 14 to 20. A likelihood that
does not represent the reading frame cannot rank protein variants, because which
protein a sequence encodes is exactly the thing the reading frame determines. The
apparent representation-level signal in section 17 survived frameshifting for the
same reason: there was never a frame-dependent representation for frameshifting to
destroy.

### The diagnostic this yields

Reading-frame discrimination is a **necessary condition** for a genomic model's
likelihood to be usable as a protein-level proxy, and it is cheap to test. Score a
set of real coding sequences and their one-nucleotide rotations, and compare
paired. A model that fails has no gene-level representation, whatever its
performance on any benchmark, and its likelihood cannot be scoring protein
function.

The test needs no labels, no assay, and no fitness data. It costs two forward
passes per gene. Nothing in the genomic-LM literature reports it.

### Scope, and what it does not say about Evo 2

This is the Nucleotide Transformer v2 family at 50M parameters, on 29 genes. It
does not test Evo 2, which is far larger, trained on far more sequence, and built
for much longer context, and which is the model behind both the Nature and the
Science results. The honest reconciliation is therefore not settled here: it is
possible that Evo 2 passes the reading-frame test and NT does not, in which case
frame representation is a capability that appears with scale and the two
literatures are measuring different models rather than different grains.

That makes the test the sharpest available prediction to run on a GPU, and it has
been added to `notebooks/Evo2_crosstalk_GPU.ipynb`. If Evo 2 fails it too, the
generative results rest on something other than the likelihood this literature
reports.

## 27. Multi-sequence prompting: real conditioning, no in-context learning of fitness

Almonte et al. (2026) report that presenting several peptides in one context
enables in-context learning in protein language models. If that extended to
fitness it would address section 15 directly: the specificity information a probe
recovers from the representation might be reachable through the input, with no
probe and no labels.

`scripts/run_icl_prompting.py` tests it on ParD3, where one context yields a
scoring function for the whole landscape from three masked passes, so many
contexts can be compared with error bars. ESM-2 650M, k=8 context sequences,
12 repeats per condition.

| context | rho with fitness | AUC specific vs promiscuous |
|---|---|---|
| **none** | **+0.510** | 0.151 |
| high-fitness examples | +0.466 (+/-0.051) | 0.481 (+/-0.068) |
| low-fitness examples | **-0.197** (+/-0.073) | 0.484 (+/-0.066) |
| random examples | +0.213 (+/-0.070) | 0.421 (+/-0.077) |

**The conditioning is real and large.** Swapping a high-fitness context for a
low-fitness one moves the correlation by **+0.664 (+/-0.089)**, enough to flip its
sign. The model is unambiguously reading the context.

**It does not help.** No context is better than every context condition. Adding
context of any kind costs 0.350 in correlation. A context of the best variants in
the landscape is worse than no context at all.

### Is it learning or copying?

The context sequences carry residues at the three masked positions, so preferring
residues seen in a good context would raise the correlation without any learning.
The control counts residues at those positions in the context and does nothing
else.

| context | PLM rho | bag-of-residues rho | PLM AUC | bag AUC |
|---|---|---|---|---|
| high | +0.466 | +0.365 | 0.481 | **0.558** |
| low | -0.197 | -0.362 | 0.484 | 0.523 |
| random | +0.213 | -0.020 | 0.421 | 0.499 |

The model beats the lookup baseline on the fitness correlation, clearly so under a
random context where counting residues carries no information at all (+0.213
against -0.020). Something more than copying is happening. On the discrimination
task the ordering reverses and the lookup baseline is better, and the model sits
at chance.

**So the answer to section 15's question is no.** Multi-sequence prompting produces
genuine conditioning that exceeds copying, and it does not recover the specificity
information a linear probe reaches from the same model's representation. The
information stays inaccessible through the input.

One result here favours prompting, and it is worth stating precisely because it is
easy to overclaim. Without context the model is *anti*-predictive of specificity
at AUC 0.151. With any context it moves to 0.42-0.48, near chance. Context removes
a harmful signal without creating a useful one, which is an improvement only
against a baseline that was pointing the wrong way.

### Scope

One landscape, one model, k=8, and a crude prompt format: sequences concatenated
with no separator, which is what ESM-2's vocabulary allows. ESM-2 is a masked
model not trained for in-context use, so this tests whether the simplest form of
multi-sequence prompting recovers fitness information in this setting. It does not
test the prompt formats or the models Almonte et al. use, and it is not evidence
against their result.

## 28. What ESM-2 recalls: predictability, not memorised sequence

Morris et al. (2025) separate what a language model memorises about specific
samples from what it generalises about the distribution. That matters for section
22, which found a model's ability to reconstruct wild-type residues predicts its
skill at ranking that protein's mutants. If the reconstruction is recall of a
training sequence, the forecast would not transfer to a sequence nobody has seen,
which is the only regime a rapid-response pipeline cares about.

`scripts/run_memorization.py` masks contiguous spans of length L, reconstructs by
argmax in one pass, and records exact recovery. 55 ProteinGym targets, 10 spans
per length, ESM-2 650M, with each protein's composition-matched shuffle as a
control.

| span L | per-residue, real | per-residue, shuffled | exact recovery, real |
|---|---|---|---|
| 1 | 0.495 | 0.082 | 0.4945 |
| 5 | 0.485 | 0.092 | 0.0709 |
| 10 | 0.434 | 0.085 | 0.0127 |
| 20 | 0.357 | 0.087 | 0.0000 |
| 40 | 0.200 | 0.074 | 0.0000 |

**There is no verbatim recall.** Not one span of 20 residues was recovered exactly
out of 550 attempts, and none of 40. Real proteins are far more predictable than
composition-matched shuffles at every length, roughly 0.20 to 0.50 against 0.07 to
0.09, so the model clearly knows a great deal about proteins. It does not know
these particular sequences well enough to write them down.

**A weak positional-correlation signal does exist.** Comparing exact recovery to
what independent per-position prediction would give at the same accuracy:

| span | per-residue p | p^L if independent | observed | excess | spans recovered |
|---|---|---|---|---|---|
| 5 | 0.485 | 2.7e-2 | 0.0709 | 2.6x | 39 / 550 |
| 10 | 0.434 | 2.4e-4 | 0.0127 | **53.7x** | 7 / 550 |

At 10 residues the model recovers spans about fifty times more often than
independence predicts. Seven spans out of 550 is a small count and the estimate is
correspondingly loose, but the direction is clear: positions are not predicted
independently, and occasionally a whole stretch comes back. That is consistent
with memorisation of specific segments and equally consistent with regions whose
family conservation leaves only one plausible residue. This measurement cannot
separate those two.

### Recall predicts skill, and homolog depth does not explain it

| span | rho(recall, DMS skill) | after removing homolog depth |
|---|---|---|
| 1 | +0.309 | +0.232 |
| 5 | +0.463 | +0.392 |
| 10 | +0.465 | +0.348 |
| 20 | +0.449 | +0.382 |
| 40 | +0.289 | +0.395 |

52 independent protein clusters. Reconstruction ability predicts ranking ability
at rho about +0.46, and residualising on MSA Neff category barely dents it, from
+0.465 to +0.348 at L=10. Recall itself correlates with homolog depth only weakly
(+0.16 to +0.35), so the two are measuring different things.

### What this settles for section 22

The reliability forecast is reading **predictability, not training-set
membership.** A model that had memorised its training proteins would reconstruct
long spans exactly, and this one cannot reconstruct twenty residues even once in
550 tries, while its per-residue predictability still tracks its downstream skill
at +0.46 independent of homolog depth.

That is the better of the two outcomes for the forecast's intended use. A signal
driven by recall of training data would have no reason to work on a sequence
nobody has seen. A signal driven by how predictable the model finds a sequence has
a reason to. That is an inference about why the forecast should transfer, not a
demonstration that it does, and the no-parent regime remains untested for the same
reason it was untested in sections 21, 22 and 25: every assay in ProteinGym has
homologs.

### Scope

Argmax reconstruction in a single pass predicts masked positions independently, so
these numbers underestimate what iterative decoding would recover, and the
positional-correlation excess is a lower bound. 55 proteins under 500 residues,
one model. No claim is made about models trained with different objectives or at
different scales, and no attempt was made to determine which sequences were in
ESM-2's training set, which would make the memorisation question sharper than any
proxy used here.

## 29. The frame test has a working positive control, and it changes what the nulls mean

Section 26 was retracted because its frameshift diagnostic was confounded: a
one-nucleotide rotation of a gene is still real genomic sequence, so a correct
density model over genomes has no reason to penalise it. The replacement probe
is the matched in-frame / out-of-frame stop contrast — write TAA either on a
codon boundary or one base past it, two three-base edits of the same
trinucleotide at the same site, only one of which can terminate the protein.
ESM-2 passes it. But ESM-2 reads the translated protein, so its pass shows the
contrast works on a **protein** model and says nothing about whether it works on
a model that eats nucleotides. The order-1 Markov chain already in the repository
cannot settle it either: with one transition table it has no parameter that
depends on position modulo three, so it is a negative control by construction.

**A 3-periodic Markov chain is the missing case.** Separate transition tables for
codon positions 1, 2 and 3, the GeneMark device (Borodovsky & McIninch 1993),
represent the reading frame explicitly, in the DNA alphabet, with no protein
anywhere in the model. `scripts/run_biointerp_periodic.py` fits it on the 24
unique coding sequences leave-one-sequence-out, so it is a generative model of
coding DNA rather than a fit to the string in front of it, and runs it against a
**non-periodic twin at the same order, on the same corpus, by the same
leave-one-out procedure, through the same scoring code**, so periodicity is the
single difference between the two arms. Order 2 was chosen by held-out
log-likelihood on the real sequences, before any intervention was scored.

| probe | 3-periodic (frame-aware) | aperiodic twin (frame-blind) |
|---|---|---|
| frameshift +1 (rotation) | **+0.0861** [+0.0701, +0.1022], 23/24 | -0.0001 [-0.0002, -0.0000], 10/24 |
| matched stop contrast | **+0.0035** [+0.0027, +0.0043], 24/24, p=1.2e-07 | -0.0002 [-0.0005, +0.0001], 11/24, p=0.84 |

**The contrast has power against a DNA-alphabet model.** A model whose only frame
machinery is three phase-indexed tables passes it decisively; the identical model
without them fails. So a null on this contrast is evidence about the model rather
than about the instrument, and the genomic frame material can support a
conclusion. The edit-size asymmetry between the two arms (2.25 versus 2.12 bases
actually changed) is not what produces the pass, because the frame-blind twin
sees the same asymmetry and returns a slightly negative contrast.

### But the contrast is low-gain, and that is the finding

Nats per token are not comparable across a 6-mer tokenizer, a single-nucleotide
one and an amino-acid one, so every effect below is divided by that model's own
reference effect, the mononucleotide shuffle. All batteries are re-adjudicated on
the 24 unique coding sequences with t intervals.

| model | matched stop contrast, as % of its own reference | verdict |
|---|---|---|
| Markov order-2 3-periodic (positive control) | **+3.42%** [+2.63, +4.20] | REPRESENTS |
| Markov order-2 aperiodic (negative control) | -0.54% [-1.28, +0.20] | NULL |
| ESM-2 35M | +4.88% [+1.47, +8.29] | REPRESENTS |
| NT-v2 50M | +4.38% [-2.83, **+11.59**] | **underpowered** |
| HyenaDNA 32k | +1.21% [-0.93, **+3.35**] | at the boundary |

A frame-representing DNA model shows only 3.4% of its own reference effect on
this contrast, against 83.6% on the rotation. The contrast is roughly
twenty-five times weaker, because it edits three bases of a thousand while the
score is averaged over the whole sequence. At that gain and n=24:

- **NT-v2's contrast null is underpowered, not bounded.** Its interval reaches
  11.6% of its own reference, above the 3.4% a frame-representing model shows,
  and its point estimate (+4.38%) is in fact larger than the positive control's.
  It would take about **214 unique coding sequences** to bound it at the positive
  control's effect size. *NT-v2 has not been shown to lack frame representation.*
- **HyenaDNA's interval tops out at 3.35% against the control's 3.42%.** That is
  a tie, not an exclusion, and must not be quoted as a bounded null.
- On the rotation probe both are bounded (NT-v2 +11.98% [-18.40, +42.36],
  HyenaDNA -0.48% [-1.66, +0.71], against the control's +83.56%) — but that is
  the probe section 26 retracted, so the bound is on a gene-model quantity a
  genome model has no obligation to have.

**The contrast is also not a complete test of frame representation.** Sweeping
Markov order (`results/frame_power_markov_orders_full.csv`), every 3-periodic
model from order 0 to order 5 passes the rotation test at 71–134% of its
reference, but orders 1 and 5 **fail the stop contrast**. Order 1 has no context
for the third base of a stop codon; order 5 has 4096 contexts per phase against
17.8 kb of training sequence. So failing the contrast means "has not learned
in-frame stop-codon depletion", which is narrower than "does not represent the
frame".

### Corrections carried by this section

The 29 rows of every earlier battery are 24 unique coding sequences
(BLAT_ECOLX three times, CP2C9, PTEN and RL401 twice each) and the ESM-2 control's
10 are 9 (RL401 twice). All intervals were m ± 1.96·SE. Both are fixed here by
re-adjudication from the saved per-sequence deltas, with no rescoring; the full
before/after table is `results/biointerp_dedup_corrections.txt`. Four numbers
quoted elsewhere move:

| number | before | after |
|---|---|---|
| ESM-2 matched contrast | +0.0511 [+0.0243, +0.0779], 9/10, p=0.021 | **+0.0504 [+0.0151, +0.0856], 8/9, p=0.039** |
| NT-v2 matched contrast | +0.0191 [-0.0047, +0.0430], 18/29 | **+0.0185 [-0.0119, +0.0488], 14/24** |
| NT-v2 frameshift +1 | +0.0264 [-0.0778, +0.1307], 14/29 | **+0.0505 [-0.0775, +0.1784], 13/24** |
| HyenaDNA 'stop out of frame' | +0.0009 [+0.0000, +0.0018] REPRESENTS | **+0.0011 [-0.0000, +0.0022] NULL** |

One verdict changes (HyenaDNA's unmatched stop control) and no headline claim in
sections 18 to 20 is affected, since none depends on this material.

### What to say and what not to say

Say: the frame diagnostic is a validated instrument on DNA models, and it is
cheap — two forward passes per gene, no labels. Say: at the sample size the
literature would plausibly use, the confound-free version of it is too coarse to
convict a genomic LM, and the version that is not too coarse is confounded.

Do not say that NT-v2 or HyenaDNA fails to represent the reading frame. The
evidence does not reach that.

`notebooks/Evo2_crosstalk_GPU.ipynb` cell 7b is not yet usable as written: it
runs the four retracted section-26 conditions, has no matched stop contrast, and
does not deduplicate its genes. Re-cut it against
`scripts/run_biointerp_periodic.py` — the matched contrast, unique coding
sequences, t intervals, and the 3-periodic Markov control fitted on the same
genes as the yardstick — and give it at least 214 unique coding sequences before
any Evo 2 result is read as a null. On fewer than that, an Evo 2 null on this
contrast would mean nothing, exactly as NT-v2's does not.

## 29. What the forecast buys at decision time: abstention and routing

Sections 22 and 25 grade the label-free forecast by rank correlation. A rank
correlation is not what a pipeline consumes. The two decisions it could actually
drive are whether to use the model on this target at all, and whether to pay for a
larger model on it. Both are measured here on the same 194 assays and 165 clusters,
with leave-one-cluster-out predictions and a chance control for each.

### Abstention (`scripts/analyze_selective_forecast.py`)

Withholding the model from the assays with the lowest forecast, then measuring skill
on the assays it is still used on:

| coverage | 150M mean skill | random control | frac. skill < 0.20 |
|---|---|---|---|
| 100% | +0.4018 | +0.4018 | 0.206 |
| 90% | +0.4397 | +0.4016 | 0.126 |
| 80% | +0.4677 | +0.4019 | 0.077 |
| **70%** | **+0.4824** | +0.4015 | **0.059** |
| 60% | +0.4951 | +0.4018 | 0.052 |
| 50% | +0.5091 | +0.4013 | 0.052 |

The random control at matched coverage moves by less than 0.001 over 2000 draws, so
essentially the whole effect is the forecast. At 650M the same procedure moves
+0.4377 to +0.4941 at 70% coverage and the unreliable share from 0.144 to 0.059.

### Routing between scales (`scripts/run_route_pilot.py`, `results/route_pilot.log`)

The router sees only 150M's own internals and the sequence, never 650M and never a
label. Cost is charged as the 650/150 parameter ratio, which is a stand-in for a
hardware measurement and not one.

| sent to 650M | mean skill | of full-650M | random | oracle | rel. cost |
|---|---|---|---|---|---|
| 0% | +0.4018 | 0.0% | 0.0% | -- | 0.23 |
| 10% | +0.4111 | 25.9% | 7.0% | 58.3% | 0.31 |
| **20%** | **+0.4209** | **53.3%** | 25.8% | 89.8% | **0.38** |
| 30% | +0.4247 | 63.8% | 31.6% | 109.8% | 0.46 |
| 50% | +0.4353 | 93.2% | 50.5% | -- | 0.62 |
| 75% | +0.4388 | **102.9%** | 75.4% | -- | 0.81 |
| 100% | +0.4377 | 100.0% | 100.0% | -- | 1.00 |

Routing a fifth of the assays captures 53.3% of the full-650M gain at 38% of its
cost, against 25.8% for random routing at the same budget. Routing 75% beats routing
100%, because on 60 of the 194 assays the smaller model is genuinely better, by 0.048
on average, so a policy that always escalates pays more and gets less.

Routing on a predicted *gain* instead of predicted small-model weakness is worse at
small budgets (42.0% at the 20% budget). The gain is the harder quantity to predict,
at rho +0.336 against +0.617 for skill itself.

### Scope

Both procedures inherit every limit of the forecast they are built on. The 650M
column of the abstention table is the one to watch, because the forecast driving it
is the weaker one (section 25), and the routing frontier is charged in parameters
rather than wall-clock. The taxon-restriction table in section 22 now has a committed
script, `scripts/analyze_forecast_by_taxon.py`.

This material, sections 22, 25 and 28, and nothing else, is drafted as
`papers/ai4dd_2026/main.tex` for the AI4DD workshop at NeurIPS 2026.

## 30. Evo 2 refutes the general form of the genomic null, and the trivial baseline says by how much

Run on Dartmouth `discovery` (NVIDIA L40S, evo2 0.6.0) from
`notebooks/Evo2_crosstalk_GPU.ipynb`, with four repairs recorded in the handoff
README, the load-bearing one being that `evo2_1b_base` needs Transformer Engine and
was replaced by **`evo2_7b`**. The numbers therefore describe the 7B model. The
collaborator also found a real bug in my scorer: `Evo2.__call__` nests two deep, so
the single unwrap raised a TypeError.

### The DMS panel, matched assay-for-assay

Their run scored 32 assays against my 25. All 25 of mine are inside their 32, so
the comparison can be made exactly rather than approximately. On those 25 assays
(21 protein clusters):

| scorer | mean rho | 95% CI |
|---|---|---|
| ESM-2 650M (protein LM) | +0.4655 | [+0.3950, +0.5359] |
| chemistry ridge, leave-one-cluster-out | +0.2860 | [+0.2475, +0.3244] |
| **Evo 2 7B (genomic LM)** | **+0.2658** | [+0.2022, +0.3295] |
| BLOSUM62 alone | +0.2282 | [+0.1950, +0.2614] |
| NT-v2 50M (genomic LM) | -0.0132 | [-0.0333, +0.0069] |

**The blanket claim in sections 18 to 20 is refuted.** "A genomic language model
scoring a native coding sequence carries no protein-fitness signal" is false at Evo
2 scale: +0.266 with an interval excluding zero and positive on 23 of 25 assays.
That claim now holds only for the Nucleotide Transformer family at the scales
tested, and the original framing was too general.

**But the trivial baseline decides how much was gained**, and this is why it was
worth adding:

| paired contrast | difference | higher in | verdict |
|---|---|---|---|
| Evo 2 minus BLOSUM62 | +0.0376 [-0.0118, +0.0870] | 14/25 | **contains zero** |
| Evo 2 minus chemistry ridge | **-0.0202** [-0.0755, +0.0352] | 9/25 | contains zero, and negative |
| ESM-2 minus Evo 2 | +0.1997 [+0.1532, +0.2461] | 24/25 | excludes zero |

Per cluster the picture is the same: Evo 2 +0.2581 against BLOSUM62 +0.2183, a
difference of +0.0398 [-0.0219, +0.1014], higher in 11 of 21.

**So the corrected claim is narrower and more useful than either the original null
or its refutation.** At Evo 2 scale a genomic model does acquire protein-fitness
signal, it is **statistically indistinguishable from a substitution matrix**, it is
not better than five fitted chemistry features, and it remains far below a protein
language model. Without BLOSUM62 in the table, +0.266 would read as a substantial
positive result.

### ParD3 specificity: the null survives, and the noise floor is now much tighter

Every single-sequence arm sits at or below the 0.664 mutation-count baseline
(partner-blind 0.401, real operon 0.495, synthetic ParD3:ParE2 0.390), while
correlating +0.47 to +0.49 with on-target binding. Evo 2 tracks whether a protein
works, not which partner it chooses.

The margin arm (E3 minus E2) was the only arm above baseline at 0.689 and was
reported without an error bar. Bootstrapped here over 4,000 resamples:
**0.689, 95% CI [0.634, 0.740]**. It excludes chance but **contains the 0.664
baseline**, and beats that baseline in 82.4% of resamples. It is therefore not
shown to beat counting mutations.

The synonymous floor improves markedly: within-variant SD 0.983 against
between-variant 3.579, a ratio of 0.275 and an **attenuation ceiling of 0.962**,
where NT sat at 0.83 to 0.89. A true correlation could be attenuated by only about
4%, so codon noise cannot explain the specificity null. **This strengthens the
ParD3 negative result** rather than weakening it.

### The context dose-response reverses

Section 19 found that supplying real genomic context made NT-v2 monotonically
worse, trend rho -0.77. Evo 2 goes the other way: **+0.70**, with AUC rising from
0.401 at no flank to 0.543 at 3,000 nt, saturating by about 1,200 nt, and the gain
coming from off-target correlation falling (+0.363 to +0.219) while on-target stays
flat. On five points p = 0.19 and every interval overlaps, so **the sign reversal is
the finding and the magnitude is not resolved.**

**WITHDRAWN PENDING A CONTROL (section 35).** This arm has **no shuffled-flank
control**, and one is now known to be necessary. HyenaDNA, which holds 30 kb
natively, shows a real-flank trend of +0.536 that its shuffled twins fully account
for (+0.893, +0.750, +0.750, +0.750; real minus shuffled -0.008, real higher in 1
of 6). A positive context dose-response is therefore reproducible from flanks
containing no information. **Evo 2's +0.70 must not be read as long-range genomic
context until the shuffled twin is run.** The control ports directly from
`scripts/run_recursive_context_hyena.py` and is the single highest-value item
outstanding on the Evo 2 arm.

### What still stands from sections 18 to 20

The NT-v2 results are unaffected: it remains at -0.013, losing to BLOSUM62 by
+0.241 in 24 of 25 assays. The correct statement of the whole arm is now that
**genomic-to-protein transfer is a function of scale and architecture, absent in the
Nucleotide Transformer family and present but sub-baseline in Evo 2.**

## 31. Transfer is not a scaling phenomenon: the NT trend does not reach Evo 2

[Longpre2025atlas] fits scaling laws for cross-lingual transfer and locates the
compute crossovers at which one choice overtakes another. DNA and protein are two
encodings of one molecule, so the same question applies: is a genomic model's
protein-fitness skill a function of scale?

Five points are now available on the identical 25 assays, a 140x parameter span
(`scripts/run_transfer_scaling.py`):

| model | parameters | mean rho |
|---|---|---|
| NT-v2 50M | 5.0e7 | -0.0099 |
| NT-v2 100M | 1.0e8 | -0.0129 |
| NT-v2 250M | 2.5e8 | -0.0065 |
| NT-v2 500M | 5.0e8 | +0.0083 |
| **Evo 2 7B** | **7.0e9** | **+0.2658** |
| *BLOSUM62 reference* | | *+0.2282* |
| *ESM-2 650M reference* | | *+0.4655* |

Within the Nucleotide Transformer family the trend is **+0.0179 per decade of
parameters**, Spearman(params, rho) = +0.80 across four points.

**The trend does not explain Evo 2.** Extrapolated to 7B it predicts **+0.024**
against the **+0.266** actually measured, under-predicting by a factor of 11.

**And the extrapolation is absurd, which is the point.** Reaching BLOSUM62 at
+0.228 along the NT line would take about **1.9e21 parameters**. That number is not
an estimate of anything; it is what happens when a flat trend is extended thirteen
orders of magnitude, and it should be quoted only to make that point. The NT values
span -0.013 to +0.008, all within noise of zero, so the honest statement is that
the NT trend is flat and any crossover it implies is meaningless.

**The reading.** Evo 2's protein-fitness signal is not what scaling the Nucleotide
Transformer family would buy. It comes from architecture, corpus and context
length, which is the genomic analogue of the sub-billion result in
[Liu2024mobilellm], where architecture dominated parameter count. Section 30's
context dose-response points the same way: real genomic context helps Evo 2
(trend +0.70) and hurts NT (-0.77), and long-range context is exactly what Evo 2
is built for.

**Caveat that limits this to a demonstration rather than a law.** Four NT points,
one Evo 2 point, and the two families differ in architecture, tokenisation, corpus
and context window simultaneously, so nothing here isolates which of those is
responsible. A genuine scaling law in the sense of [Longpre2025atlas] would need a
single family swept across scales with everything else held fixed, which no public
genomic model family currently provides at this range.

## 32. A genomic model memorises nothing, and barely beats an order-5 Markov chain

Section 28 measured memorisation in a protein model and found no verbatim recall.
The genomic case admits a sharper test, because Nucleotide Transformer v2 was
trained on a known finite genome set, so membership can be checked rather than
assumed (`scripts/run_memorisation_dna.py`, `analyze_memorisation_dna.py`).

**Setup.** 27,787 span reconstructions over 720 windows of 3,000 nt from 25
sources, NT-v2 50M. L is the span in 6-mer tokens, so L=10 is 60 nt, the direct
analogue of section 28's 20-residue span. Arms: 11 genomes verified present in the
NT-v2 corpus, 13 released in 2025-26 and therefore after its training, plus
genus-matched pairs, off-frame controls, and matched Markov, dinucleotide and
mononucleotide shuffles. Window pools are GC-matched in 1% bins and permutations
are over genomes rather than windows.

### There is no membership signal at all

| comparison | member | non-member | difference | p |
|---|---|---|---|---|
| per-nt accuracy, L=1 | 0.3244 | 0.3321 | **-0.0077** | 0.647 |
| per-nt accuracy, L=10 | 0.3256 | 0.3296 | -0.0040 | 0.747 |
| log p(true), nats/token, L=1 | -8.2017 | -8.2278 | +0.0261 | 0.885 |
| log p(true), nats/token, L=10 | -7.8332 | -7.8181 | -0.0151 | 0.912 |

Genomes the model was trained on are reconstructed no better than genomes
published after its training finished, and the point estimates run slightly the
wrong way. The sensitive test agrees: loss-based membership inference, including
the standard order-5 reference-model ratio correction for samples that are
intrinsically easy rather than memorised, gives **AUC 0.49 to 0.55** across every
span length. That is chance.

Exact span recovery is approximately zero at every length beyond a single token, in
every arm, so there is no verbatim recall either, matching section 28's protein
result.

### The number that reframes sections 30 and 31

| scorer | per-nt accuracy |
|---|---|
| NT-v2 50M | ~0.330 |
| order-5 Markov chain fitted to the source genome | ~0.310 |
| always emit the window's commonest base | ~0.320 |
| chance for four bases | 0.250 |

**A 50M-parameter genomic language model beats an order-5 Markov chain by about
two percentage points of nucleotide accuracy on its own pretraining objective**,
and beats "emit the commonest base" by between 0.6 and 2.7 points.

That reframes the protein-fitness nulls in sections 18 to 20 and 31. NT-v2's
failure to carry protein-fitness signal is not a surprising fact about
cross-modality transfer. It is what one should expect from a model that is barely
distinguishable from a Markov chain at modelling DNA in the first place. The
interesting question was never why NT-v2 fails on proteins; it is why anyone
expected a model with this much margin over a 5-mer lookup to succeed at anything.
Section 31's finding that Evo 2 sits 11x above the NT scaling line reads the same
way from this side.

### A confound the audit raised is now closed

The adversarial audit of section 26 argued that the DMS coding sequences were
"well-curated, database-canonical genes of model organisms in NT-v2's 850-genome
training corpus", so both a gene and its rotations would be verbatim training
substrings, which alone could predict the observed null. Two results here retire
that. Membership confers no reconstruction advantage anywhere, and the specific
chromosome used throughout this project, CP002279 (*M. opportunistum*), is **not**
a member of the NT-v2 corpus. The memorisation confound is not what produced those
results.

### Scope

One model family at 50M, with a 500M arm and a HyenaDNA smoke test also on disk and
not yet analysed. Membership is established by corpus documentation and release
date, which is strong evidence but not the same as access to the training set.
Nothing here speaks to Evo 2, whose corpus and scale are both different and which
section 30 shows behaves differently.

## 33. Entropy is a direction, not just a statistic, and steering it does not help

[Rahn2024controlling] builds a steering vector as an entropy-weighted average of
activations taken just before a decision, shows it controls an agent's exploration
far beyond sampling temperature, and concludes that language-model agents
"explicitly encode uncertainty over their actions in their representation space".

Sections 22 and 25 found that the mean entropy of a protein model's masked
distributions is the single best predictor of where it will be unreliable, at
rho -0.597 with skill. That is a fact about a scalar. This tests the sharper
claim: is that uncertainty a **direction** in activation space?
(`scripts/run_east_steering.py`, `analyze_east_steering.py`, ESM-2 150M, one assay
per UniProt protein so the units are independent.)

### The direction exists

Entropy change under a matched-norm intervention at layer 29, paired within assay:

| alpha | steering vector | random | orthogonal | unweighted mean |
|---|---|---|---|---|
| **+1** | **+0.796** | +0.147 | +0.159 | +0.613 |
| **-1** | **-0.284** | +0.251 | +0.233 | -0.091 |

The steering direction moves entropy about five times as far as matched-norm
random and orthogonal directions, and it moves it **bidirectionally**: negative
alpha lowers entropy while both nulls raise it. The unweighted mean activation
direction is the harder control, since adding any large mean shift changes
entropy, and the entropy-weighted vector beats it in both directions.

**It localises late.** At layer 15 the effect is not cleanly bidirectional (both
signs of alpha raise entropy, +0.058 at alpha=-2 and +0.438 at alpha=+1), while at
layers 27 to 29 it separates properly. Per-assay monotonicity of entropy in alpha
at layer 15 is weak, mean rho +0.262, positive in 7 of 8 proteins. So the
representation of uncertainty is a late-layer property, which matches where
[Rahn2024controlling] found theirs.

### Steering it does not buy anything

| alpha | change in skill (rho) | random | orthogonal |
|---|---|---|---|
| -1 at L15 | -0.078 | -0.092 | -0.091 |
| -1 at L17 | -0.124 | -0.130 | -0.130 |
| -1 at L19 | -0.123 | -0.167 | -0.167 |

Skill degrades monotonically with the size of the intervention, from -0.002 at
|alpha|=0.25 to -0.493 at alpha=+2, and **it degrades by about as much as a random
perturbation of the same norm does**. The direction that controls entropy is not a
lever on usefulness. Whatever the model encodes about its own uncertainty, moving
along it does not make the model better at ranking variants, it just damages it at
the ordinary rate.

That is the same shape as section 27, where multi-sequence prompting produced large
genuine conditioning (a rho swing of 0.664) that nonetheless hurt. Two independent
control channels into a protein model now show the same pattern: real, measurable
influence over the model's behaviour, and no route from that influence to better
predictions.

### One thing that did survive, and is a useful robustness check

The entropy-to-skill relationship from section 22 holds **after** the intervention:
rho(entropy, steered skill) stays between -0.64 and -0.93 across every alpha
tested, against -0.64 to -0.86 at baseline. Steering moves entropy without breaking
what entropy predicts, so the forecast in sections 22 and 25 is not an artifact of
some particular activation scale.

### Scope

ESM-2 150M, 8 to 12 evaluation proteins depending on the sweep, one assay per
protein. That is a small panel and the intervals on the alpha sweep are wide. The
existence claim rests on the layer sweep, where the steering vector separates from
three distinct nulls at once; the no-benefit claim rests on steering tracking
random perturbation across layers and alphas, which is a weaker form of evidence
than an interval excluding a difference. A larger panel would tighten the second
claim and is the obvious next step if it matters.

## 34. Genomic models on their own home task: mostly beaten by a lookup table

Sections 14 to 32 all grade genomic models on coding sequence and protein fitness,
a task they were not built for. This measures them where they are meant to work
(`scripts/run_regulatory_models.py`, `run_regulatory_headtohead.py`).

**Setup.** Five Nucleotide Transformer benchmark tasks, three splits (shipped,
homology-clustered, random), frozen embeddings with a linear probe, identical
pipeline to the baselines so only the features change. The layer and pooling grid
was **fixed before any test number existed**, the headline configuration is chosen
by grouped cross-validation on training rows only, and a best-of-nine-on-test row
is reported separately purely to bound the sweep's headroom. The trivial
competitor is chosen **on test**, so the baseline gets an oracle the model does not.

### The answer, on the clustered split, paired cluster bootstrap

| task | best trivial | AUC | NT-v2 50M | difference | HyenaDNA | difference |
|---|---|---|---|---|---|---|
| splice_sites_all | one-hot | 0.959 | 0.756 | **-0.203** [-0.216, -0.189] | 0.704 | **-0.254** [-0.270, -0.240] |
| promoter_tata | one-hot | 0.941 | 0.930 | -0.010 [-0.041, +0.022] | 0.915 | -0.025 [-0.063, +0.011] |
| promoter_all | k-mer 5 | 0.925 | 0.934 | **+0.010** [+0.001, +0.018] | 0.938 | **+0.013** [+0.004, +0.022] |
| H3K4me3 | k-mer 3 | 0.875 | 0.887 | +0.012 [-0.001, +0.025] | 0.873 | -0.002 [-0.018, +0.013] |
| enhancers | k-mer 5 | 0.779 | 0.808 | **+0.029** [+0.017, +0.043] | 0.787 | +0.009 [-0.005, +0.022] |

**Two clear wins in ten model-task cells, both under 0.03 AUC, against one loss of
0.20.** A positional one-hot reaches 127% of NT-v2 and 136% of HyenaDNA on splice
sites; a 3-mer count reaches 99% of NT on H3K4me3. This is the genomic counterpart
of BLOSUM62 reaching 49% of ESM-2 in section 18, and it is worse, because here the
lookup table wins outright on two tasks.

### The splice-site loss is the readout, not the representation

Mean pooling is position-blind, which hands a positional one-hot an unearned
advantage on a task defined by a positional consensus. A **post-hoc** windowed
readout, mean over eight equal token blocks, was added after seeing the loss and is
labelled as post-hoc everywhere:

| task, clustered | one-hot | NT mean-pool | NT windowed | Hyena windowed |
|---|---|---|---|---|
| splice_sites_all | 0.959 | 0.756 | **0.938** | 0.869 |
| promoter_tata | 0.941 | 0.930 | **0.954** | 0.936 |

The 0.203 gap collapses to 0.021 and NT overtakes one-hot on promoter_tata. So the
information **is** in the representation and the standard frozen-embedding protocol
discards it. The headline loss stands for the standard protocol, and it must not be
restated as "the model does not represent splice sites".

That is the same lesson as section 15 in a new place: the protocol, not the model,
decided the number. Layer choice alone moves results by up to 0.028 within a model,
against 0.010 to 0.052 between models on four of five tasks.

### The models are nearly strand-blind where the biology is not

AUC(forward) minus AUC(reverse complement), probe trained forward, clustered split:

| task | best trivial | NT | HyenaDNA |
|---|---|---|---|
| **splice_sites_all** | **+0.248** | +0.037 | +0.011 |
| promoter_tata | +0.198 | +0.097 | +0.033 |
| promoter_all | +0.008 | +0.031 | +0.021 |
| H3K4me3 | +0.002 | +0.021 | +0.022 |
| enhancers | +0.027 | +0.033 | +0.016 |

A splice site is strictly strand-oriented. One-hot correctly collapses by 0.248
when the strand is flipped; NT loses 0.037 and HyenaDNA 0.011. On the
reverse-complement task NT **beats** one-hot by +0.047, and that is a failure
rather than a win: it scores the reverse complement of a splice donor almost as
highly as the donor. The control holds, because on the two non-oriented tasks
(enhancers, H3K4me3) trivial and model asymmetries agree at about 0.02, so this is
not an artifact of the protocol.

This independently reproduces the "strand-blind" finding of the Mechanistic
Invariance Test (arXiv:2604.06549), on different tasks and a different protocol.

### Coverage reproduces section 23, on both sides

At a 1% false-positive budget, true positive rate and locus coverage with the
longest run of consecutive missed positives:

| task | best trivial | NT | HyenaDNA |
|---|---|---|---|
| enhancers | 0.057 / 0.062 / 76 | 0.089 / **0.101** / 58 | 0.081 / 0.095 / 78 |
| H3K4me3 | 0.217 / 0.261 / 20 | 0.232 / 0.277 / 20 | 0.214 / 0.254 / 21 |
| splice_sites_all | 0.446 / 0.508 / 7 | 0.064 / 0.079 / 90 | 0.016 / **0.020** / 256 |

**NT's +0.029 AUC win on enhancers buys four points of locus coverage and leaves a
screen that finds 10% of positive-bearing loci while missing 58 in a row.**
HyenaDNA on splice sites is the extreme: AUC 0.704 reads as weak but real, and
locus coverage is 0.020 with a contiguous hole 256 positives long.

### What is reassuring

Split choice barely matters on these benchmarks. Shipped against clustered differs
by at most 0.012 on four of five tasks for both models, with promoter_tata the
outlier for everyone (+0.018 to +0.033). No model win depends on the split. Unlike
ParD3 in section 15, these benchmarks are not badly leaky, and that is worth saying
plainly.

### Scope

Two small models, 50M and roughly 1.6M. Nothing here licenses a claim about
NT-500M or Evo 2. The windowed readout is post-hoc and was run on two of five
tasks; it should be pre-registered and run on all five before being quoted as a
headline number.

## 35. The context dose-response reproduces on shuffled flanks, in both directions

Section 19 could not separate two readings: genomic context genuinely fails to help
Nucleotide Transformer, or NT cannot use context past its roughly 12 kb window and
the sweep was measuring the window. Recursive decomposition in the manner of
[Zhang2025recursive] was built to separate them
(`scripts/recursive_context.py`, `run_recursive_context*.py`).

It did not separate them. It found something more useful: **the measurement the
question was posed against is not stable.**

### The control that decides it

Direct scorer, section 19's own flank grid, real chromosome against
composition-matched shuffles, three seeds each:

| flank content | trend rho | per-seed |
|---|---|---|
| **real chromosome** | **-0.771** | |
| mononucleotide-shuffled | **-0.633** | -0.700, -0.500, -0.700 |
| dinucleotide-shuffled | -0.333 | -0.100, -0.300, -0.600 |

**Section 19's negative dose-response is substantially reproduced by flanks that
carry no genomic information at all.** It is a length and composition effect on the
score scale, not evidence that real genomic context specifically hurts. Section
19's conclusion survives, since context does not rescue NT either way, but **its
stated mechanism does not** and the paragraph asserting that supplying real genomic
context degrades the result should not be quoted.

The trend is also grid-dependent: -0.771 on section 19's grid, **-0.430** on a
denser one, with the curve turning back up past 3 kb.

### The recursion is faithful at one chunk and not past it

| flank | chunks | rho(recursive, direct) | AUC direct | AUC recursive |
|---|---|---|---|---|
| 1,440 | 1 | **1.000** | 0.373 | 0.373 |
| 2,880 | 2 | 0.859 | 0.360 | 0.348 |
| 4,320 | 3 | 0.861 | 0.383 | 0.329 |
| 5,760 | 4 | 0.840 | 0.409 | 0.321 |

Exact at one chunk, so the plumbing is right. Past that the recursive score is only
a rho 0.85 stand-in and the AUC gap widens monotonically. **Everything the
recursion reports beyond NT's window is therefore suspect**, and the sweep out to
80.9 kb (13,487 tokens, 6.6x the window) should not be read as a measurement of
long-range context. The agent reported this rather than proceeding, which is the
right call.

### Two references that set the level

An **order-5 Markov chain** through the identical pipeline returns AUC 0.3845 at
every flank, every mode and every transform, to machine precision. It provably
cannot see the flank, so the pipeline injects no flank dependence, which validates
the harness. It also sets the ceiling: NT's recursive AUC of 0.299 to 0.373 sits
**at or below a model that sees five nucleotides** at every flank past 2,880. The
limit here is the model, not the context mechanism, which is what section 32 would
predict.

**HyenaDNA** holds 30 kb natively and needs no recursion. Its real trend is
**+0.536**, which looks like the Evo 2 direction. Its shuffled twins trend +0.893,
+0.750, +0.750 and +0.750, reaching the same or higher AUC; real minus shuffled is
**-0.008**, real higher in 1 of 6. **A model that genuinely holds 30 kb produces a
clean positive dose-response from flanks containing no information whatsoever.**

### What this establishes

Context dose-response trends on this landscape are reproducible in both directions
by composition-matched noise: negative at 12 kb through NT, positive at 30 kb
through HyenaDNA. The whole comparison also sits in a no-signal regime, every arm
between 0.27 and 0.51 against a 0.664 mutation-count baseline.

Two honest limits on the recursion itself. Chunk i>0 splices distal chromosome
directly against the CDS at a junction that does not occur in nature, so what is
excluded is "NT can use distal context delivered this way", not the general claim.
And the analogy to [Zhang2025recursive] is partial: their root model adaptively
chooses what to examine, whereas a masked LM offers no interface for that, so this
is the map-reduce skeleton without the agent.

## 36. The 500M memorisation arm: much better at DNA, and an apparent membership signal that does not survive matching

Section 32 measured NT-v2 50M. The 500M run was collected at the same time and is
analysed here (`results/memorisation_dna_nt500m.csv`, 11,947 reconstructions over
280 windows). Two things change with scale and they point in opposite directions.

### Scale buys a great deal at the pretraining objective

| model | per-nt accuracy, L=1 | margin over an order-5 Markov chain |
|---|---|---|
| NT-v2 50M | 0.3244 | **+0.018** |
| NT-v2 500M | 0.4372 | **+0.131** |

The 500M model beats a 5-mer lookup by **seven times** the margin the 50M model
manages. Section 32's line that a genomic model "barely beats an order-5 Markov
chain" is a statement about 50M, not about the family, and should be qualified
wherever it is quoted. Scale does buy real competence at modelling DNA. What
section 31 showed is separate: that competence does not convert into
protein-fitness signal along the NT scaling line.

### The apparent membership signal

Unlike at 50M, the all-bacteria comparison shows something:

| measure | member | non-member | difference | p |
|---|---|---|---|---|
| per-nt accuracy, L=1 | 0.4372 | 0.4039 | +0.0333 | 0.22 |
| log p(true), L=1 | -6.5684 | -7.0755 | **+0.5071** | **0.07** |
| loss-based membership inference, L=1 | | | **AUC 0.664** (0.676 ratio-corrected) | |

At 50M the same attack returned AUC 0.522. So on its face, membership becomes
detectable at 500M.

### It does not survive composition matching, and the matching failed

**The GC pools are not matched in this run.** The member pool averages GC 0.6030
against the non-member pool's 0.5531, a five-point gap. The 50M run matched to
0.5763 against 0.5768, four thousandths. Genomic models are strongly
composition-sensitive, so a five-point GC gap is a live alternative explanation for
every number in the table above.

The genus-matched comparisons control composition properly, and they show nothing:

| comparison | per-nt difference, L=1 | membership inference AUC |
|---|---|---|
| M. albiziae (member) vs Mesorhizobium 2026 (non) | **-0.0097** | |
| M. albiziae (member) vs CP002279 (non) | +0.0028 | |
| genus-matched, pooled | | **0.462 raw, 0.509 corrected** |

The member genome is *worse* than its non-member relative on one comparison and
indistinguishable on the other, and the membership attack falls to chance. **The
signal appears only in the comparison whose composition matching failed, and
disappears in the comparison where it holds.**

Power also fell: the 500M membership test uses 40 windows per arm against 136 at
50M, so its intervals are much wider even before the confound is considered.

### What to say and not say

Say: at 500M there is no confound-free evidence of training-set memorisation, and
the apparent signal in the unmatched comparison is most plausibly composition.
There is still no verbatim recall, with exact span recovery at approximately zero
for every arm beyond a single token.

Do not say: that memorisation has been excluded at 500M. The genus-matched
comparison rests on a **single** member genome and 40 windows per arm, which is too
thin to exclude a moderate effect. The honest position is that the question is open
at 500M and answered only at 50M, and closing it needs a properly GC-matched pool
at the larger scale rather than a larger model.

This is the same failure mode as sections 26 and 35 in a third place: a signal that
looks real until the matched control is applied, and this time the control was
present in the design but degraded in execution.
