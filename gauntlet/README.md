# Gauntlet

**A command-line tool for wet-lab-in-the-loop protein engineering.** You tell it
what you have measured and how many variants you can afford to order; it tells
you which ones to order next, and whether any model has earned that decision.

It is built for the loop, not the model. Choosing the next batch is where the
money goes — a round of PET-hydrolase variants is thousands of dollars and
several weeks — and the scoring function you would normally rank by turns out to
be unreliable in ways that are measurable in advance.

```bash
pip install -e .

gauntlet audit    --campaign my_rounds.csv                  # is my scorer measuring anything?
gauntlet plan     --campaign my_rounds.csv --budget 96      # what do I order next?
gauntlet backtest --campaign my_rounds.csv --budget 8 --rounds 3   # would a strategy have worked?
```

## Why it exists

Measured across 13 ProteinGym DMS assays and 5 published PET-hydrolase
campaigns, **no single scoring strategy is safe**:

- A zero-shot protein language model produced both the best result observed
  (top-1% recall **0.409** against random 0.095, on an esterase thermostability
  scan) and the worst (**0.000**; and 0.109 against random 0.851 on a
  multi-mutant PETase campaign).
- A ridge regression on the developer's own measurements — no pretraining, no
  structure — was never the best and **never catastrophic**, beating random in
  12/13 assays and 5/5 campaigns.
- Across 504 measured PETase variants, the zero-shot score's within-study rank
  correlation with activity is **+0.007**. A bare mutation count reaches +0.25.

You cannot tell in advance which regime you are in. Gauntlet works it out from
your own data and says so plainly, including when the answer is "nothing has
earned trust yet, go explore."

## Input

A CSV with one required column, `variant`. Rows with a `fitness` value are what
you have measured; rows without one are candidates you could order.

```csv
variant,fitness,substrate,temperature_C
WT,0.31,PET film,40
S214H,0.72,PET film,40
S214H/I168R,1.04,PET film,40
S214H/I168R/W159H,,PET film,40
```

Any other column is preserved as an assay condition. That is deliberate:
activity measured under different substrate crystallinity, loading or
temperature is not comparable, so conditions have to survive into the data
rather than living in a figure caption — and `audit` and `plan` **refuse to pool
across them**:

```
CONDITION ENFORCEMENT — measurements span 2 assay conditions
  40 (n=25); 70 (n=22)
  Using: 40  (25 measured, 22 excluded)
  Measurements from different conditions are not comparable, so they are not pooled.
  Override with --pool-conditions, or choose another with --condition.
```

A column only counts as a condition if it varies but takes few enough distinct
values to be one; a free-text `notes` or `date` column is reported and ignored
rather than putting every variant in its own stratum. Override the inference
with `--conditions substrate,temperature_C`.

**The tool measures this rather than assuming it.** An earlier version refused to
pool on principle. On the NREL release — the only real multi-condition dataset
here — pooling turned out to be *better* (AUROC 0.716 vs 0.645), and a
size-matched control showed condition purity was worth nothing. So `audit` and
`plan` now run the comparison on your own campaign and let the number decide:

```
POOLING CHECK — measured on your data, 11 conditions with enough measurements
  within condition       AUROC 0.645   (trained on that condition only)
  same size, other conds AUROC 0.639   (isolates mixing from sample size)
  pooled                 AUROC 0.712   (trained on everything)
  Rows from a different condition are as useful as rows from the same one,
  so condition purity is buying you nothing here.
  -> POOL: pooling scores higher on your data (0.712 vs 0.645 AUROC)
```

The middle row is the one that matters: it trains on the *same number* of rows
drawn from *other* conditions, so it separates the cost of mixing from the
benefit of more data. `--condition` still forces a single stratum when you want
one. Activity *values* are never comparable across assays; what this measures is
whether they can be pooled for *ranking*. See FINDINGS.md Phase 13.

Pass `--scaffold wt.fasta` to validate mutation numbering against the sequence
(the offset is recovered automatically), to enumerate single mutants as
candidates, or to score with ESM-2 via `--esm`.

## What each command does

**`audit`** — checks whether your scoring function is measuring biology. Reports
each proxy's correlation with your measurements, its correlation with mutation
count, and the partial correlation with mutation count removed. Flags two
failure modes: a learned score that a substitution matrix matches, and the
mutation-count confound below.

Like `plan`, it audits a single assay condition rather than pooling — a
correlation computed across two assays is corrupted the same way a model fit is.

It then prints the **corrected view** — the same correlations computed inside a
fixed mutation count, which is the only intervention measured to remove the
confound without also removing the signal. On the bundled example this changes
the verdict completely: pooled `esm2_wtm` reads −0.728, but within a fixed
mutation count the mean is **+0.155**, and all four proxies flip sign.

**`plan`** — cross-validates a model on your own measurements, compares it
against any precomputed proxies by **top-decile enrichment**, and returns a
batch plus a verdict.

Enrichment, not rank correlation, because they are not the same question and can
disagree sharply. A batch needs the good ones near the top; it does not need the
whole ordering to be right. On the NREL release, mean hydropathy correlates
**+0.07** with activity — useless as a ranking — while enriching the top decile
**2.5x**, tying a model fit on 160 measurements. Selecting proxies by Spearman
passes over exactly that scorer:

```
  scorer                enrichment   spearman
  supervised (CV)            2.50x     +0.319
  mean_hydropathy            2.50x     +0.066
  seq_len                    0.00x     +0.054  (not selectable)
```

Verdicts:

- `SUPERVISED` — a model on your data cleared the 1.5x bar; exploit it.
- `PROXY` — a precomputed score enriches the top decile better than any model
  fit on your data does.
- `DIVERSITY` — nothing has earned trust. Order a spread-out batch and buy the
  data that makes the next round decidable.

Ties go to the model on your own data, since that was the policy that was never
catastrophic across 13 DMS assays, 5 campaigns and 11 assay conditions — but a
tied proxy is called out as worth ordering alongside, because it is finding the
top decile by a different route. Enrichment is quantised by the elite-set size,
which `plan` reports so you can see its resolution.

**`backtest`** — replays your campaign as if you had chosen in a different
order, and reports which strategies would have beaten random selection. Use it
before trusting any ranking.

By default it replays the campaign as recorded and warns if that pools assays.
`--split-conditions` replays each condition separately and compares the
conclusions, which is the question that actually matters — did a policy beat
random *consistently*, or win once by luck:

```
CROSS-CONDITION — recall of the true top 10%, per condition
                         40     70 beats_random
random                0.473  0.555          0/2
rank_by_esm2_wtm      0.000  0.000          0/2
supervised_greedy     0.858  0.980          2/2

Beat random in EVERY condition: supervised_greedy, supervised_ucb, ...
  Consistency across independent assays is the claim worth acting on.
Never beat random: rank_by_blosum62, rank_by_esm2_per_mut, rank_by_esm2_wtm
```

Strata with fewer than 20 measured variants are skipped and reported, never
dropped silently. `rank_by_n_mut` appears as a diagnostic and is labelled as
one: if it wins, your data carries campaign progression, and `plan` will still
refuse to order by it.

## The mutation-count confound

The standard zero-shot score (masked marginals in wild-type context) is a **sum** over mutated
positions of mostly-negative terms, so it falls as edits accumulate. In an
engineering campaign the best variants are also the most-mutated, so the score
ranks the winners last.

This was tested directly, not inferred. Composing synthetic multi-mutants from
DMS single mutants, whose fitness and per-mutation score are both known:

| Pool | k=1 | mixed k ≤ 6 |
|---|---|---|
| beneficial components | +0.077 | **−0.563** |
| random components | +0.546 | +0.004 |

But **within a fixed mutation count the correlation does not degrade at all** —
0.546, 0.548, 0.530, 0.537, 0.529, 0.538 for k = 1…6. The model is not worse at
multi-mutants. The failure is entirely pool composition.

Rescoring does not fix it. Context-aware marginals change nothing (−0.400 vs
−0.348 on multi-mutant campaigns) and a non-additive whole-sequence score halves
the anti-correlation but destroys the signal where the additive score worked
(+0.288 → +0.116); in selection replay no scorer beats random. The fix is a
better *comparison*, not a better score — which is what `audit`'s corrected view
and `plan`'s own-data model implement.

## Layout

```
src/gauntlet/
  cli.py           audit / plan / backtest
  io.py            campaign files, variant parsing, numbering recovery
  plan.py          regime detection, trust assessment, batch selection
  campaign.py      replay engine, selection policies, featurisers
  controls.py      the audit battery
  proxies.py       ESM-2 marginals + trivial baselines
  petase_data.py   PETML corpus loader
  proteingym.py    ProteinGym DMS loader
scripts/           the validation runs behind the numbers above
examples/          a runnable PETase campaign
```

Findings and full methodology: [FINDINGS.md](FINDINGS.md). Project rationale:
[PROPOSAL.md](PROPOSAL.md).

## Data

PETML activity corpus redistributed from
[jafetgado/PETML](https://github.com/jafetgado/PETML) (MIT); cite Norton-Baker,
Komp, Gado et al., *ACS Catalysis* (2025), doi:10.1021/acscatal.5c03460.
ProteinGym v0.1 via the OATML-Markslab HuggingFace mirror.
