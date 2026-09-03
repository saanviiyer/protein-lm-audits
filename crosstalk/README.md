# crosstalk

RL environments for protein binding **specificity**.

Binding specificity is a two-sided objective. A design must bind its target and
must *not* bind everything else. Almost every published in-silico design
objective optimizes only the first half. This package turns exhaustively
measured two-partner binding landscapes into budgeted decision problems, so a
specificity objective can be optimized and, more importantly, audited.

## Why this landscape

`crosstalk` is built on the ParD3 antitoxin landscape of Lite et al., eLife 2020
(GEO `GSE153897`): all 7,882 variants at positions 61/64/80, each measured
against the cognate toxin ParE3 **and** the non-cognate ParE2.

Two properties make it a benchmark rather than a demo:

1. **Both sides are measured for every variant.** Off-target binding is ground
   truth, not a prediction, so crosstalk is observed rather than modelled.
2. **The space is exhaustively enumerated.** The optimal design and the optimal
   budgeted-walk value are computable exactly, so regret is exact rather than
   relative to the best method anyone happened to try.

The oracle noise is calibrated from the two biological replicates rather than
invented: a single assay has SD 0.039 (ParE3) and 0.031 (ParE2), about 3% of the
dynamic range. Recovering published fitness from raw frequencies gives r=0.9998,
which is the check that the data is being read correctly.

## The environments

Budget is denominated in **assays**, not variants. Measuring the on-target costs
one assay; counter-screening against the off-target costs another. An
affinity-only campaign therefore screens twice as many variants as one that
counter-screens. That is the trade real campaigns make, and the reason "screen
for binding first, worry about selectivity later" is tempting.

- **`BudgetEnv`** (the benchmark). A fixed budget of noisy oracle queries, then
  one nomination. Scored on ground truth, never on the reward the agent
  optimized.
- **`WalkEnv`** (calibration). Local search with a mutation budget and dense
  reward. Exactly solvable, so it works as a unit test.

Two tasks: `cognate` (bind ParE3, avoid ParE2) and `swap` (reprogram to bind
ParE2, avoid ParE3).

Five rewards: `affinity`, `margin`, `gated`, `lagrangian`, `logratio`.

## Quick start

```bash
python3 scripts/fetch_data.py
python3 -m pytest tests/ -q
python3 scripts/run_benchmark.py --seeds 30
python3 scripts/make_figure.py
```

```python
from crosstalk import load_pard3, make, BudgetEnv, ruggedness

L = load_pard3()
env = BudgetEnv(L, make("margin"), budget=200, seed=0)
obs, _ = env.reset(0)
obs, r, done, _, info = env.step(L.index["DKE"])   # one noisy assay
result = env.nominate(L.index["DYE"])              # scored on ground truth
```

## Teaching notebook

`notebooks/auditing_ai_protein_tools.ipynb` is a hands-on tour of the tools this
project audits: DMS landscapes, protein language models, co-folding confidence,
GenBio AIDO with retrieval, and the RL environment. It runs live (ESM-2 35M
scores the whole landscape in three forward passes) with interactive widgets,
and falls back to stored results for anything that needs a GPU or a 15 GB
download.

```bash
/opt/anaconda3/bin/jupyter lab notebooks/auditing_ai_protein_tools.ipynb
```

Use the Anaconda kernel; it has numpy, matplotlib, torch, transformers and
ipywidgets. Rebuild the notebook from source with
`python3 scripts/build_notebook.py`.

## Findings

See [FINDINGS.md](FINDINGS.md). Two results so far:

1. **Pricing off-target binding roughly doubles the number of local optima**
   (11 vs 4 on the cognate task, 10 vs 5 on the swap task). Specificity is a
   harder *search* problem than affinity on identical data, which is what makes
   it worth an RL environment at all.
2. **Budget does not substitute for the right objective.** A specificity-aware
   reward reaches 100% success and 0% crosstalk by 200 assays. An
   affinity reward plateaus below that and keeps producing cross-reactive
   designs at every budget tested, including one that pays for the
   counter-screen data and then ignores it.

## Status

Working and tested (16 tests). The ParD3 landscape is deliberately small: it is
the exactly-solvable calibration case, not the hard one. Scaling to larger
specificity landscapes is the open next step.

## Reproducible pipeline (Dagger)

Training a policy needs multiple seeds: a single REINFORCE run is not separable
from its own variance. The seeds are independent, so the sweep runs each
`(task, reward, seed)` cell in its own container, in parallel, from a pinned
base image.

```bash
dagger call test --source=.
dagger call benchmark --source=. export --path=./results
dagger call sweep --source=. --seeds=6 export --path=./results
```

`sweep` fans out every cell, then pools them with a paired per-seed analysis:
within a single training run, does the reward and the ground-truth specificity
move in the same direction? Pairing matters, because pooling seeds without it
hides exactly the divergence the benchmark is built to detect.

Requires a container runtime. The base image installs CPU-only torch and fetches
the landscape from GEO in layers separate from the source mount, so editing code
does not re-download the data.

## The policy

The raw action space is one action per sequence (7,882), too large to
parameterize directly and not transferable between landscapes. The policy is
instead permutation-equivariant: a shared MLP scores each candidate from
belief-derived features and the next assay is sampled from a softmax over those
scores, which is a learned acquisition function. The policy and the search
baselines nominate identically, so they differ only in how they choose the next
measurement.

Belief is a conjugate Bayesian linear model over one-hot sequences, updated in
closed form. Its observation noise is the calibrated assay SD, plus an epistatic
misspecification term: an additive model cannot represent epistasis, and on this
landscape that residual is 7-12x the assay noise. Without it the posterior is
overconfident by about 6x and the policy stops exploring.
