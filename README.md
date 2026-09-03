# Protein LM Audits

**Four projects asking one question: has this proxy earned the decision it is being used for?**

Protein language models are rarely used to answer the question they were trained on.
They are used as *stand-ins* — for activity, for function, for hazard — and the substitution
usually goes unexamined. These four projects examine it in the four places the substitution
matters most: discovery, design, safety, and specificity.

| | project | the proxy under audit | the decision it is standing in for |
|---|---|---|---|
| **Discovery** | [`prism/`](prism/) | ESM-2 embedding drift toward Pfam centroids | "this mutational process drives this functional shift" |
| **Design** | [`gauntlet/`](gauntlet/) | zero-shot PLM scores, structure confidence | "order these variants next" |
| **Safety** | [`shibboleth/`](shibboleth/) | the same scores at screening thresholds | "flag this sequence as hazardous" |
| **Specificity** | [`crosstalk/`](crosstalk/) | PLM likelihood, co-folding confidence | "this binds my target and not the others" |

They are sequential rather than parallel. PRISM is where the discipline came from — a pipeline
that ended up auditing its own metric. Gauntlet turned that into a control battery for design
objectives; Shibboleth points the same battery at biosecurity screening and reads Gauntlet's
caches directly. Crosstalk is the most recent, and the only one with ground truth on both sides
of the question it asks.

---

## The findings, in one line each

**Gauntlet — a scoring function can be worse than counting mutations.**
Across 13 ProteinGym DMS assays and 5 published PET-hydrolase campaigns, no single scoring
strategy is safe. On measured PETase activity, ESM-2 zero-shot reaches Spearman **+0.007**;
a bare mutation count reaches **+0.25**. The same model produced both the best result observed
(top-1% recall 0.409 against random 0.095) and the worst (0.000). Which one you get is not
knowable from the model — but it is measurable in advance, which is what the tool does.

**Shibboleth — safeguards are proxies too, and the operating point is where they fail.**
A screener is not judged by bulk correlation but by recall at a low false-positive rate.
Re-scoring 41 cached ProteinGym assays at screening operating points gives a median
**TPR of 3.1% at 1% FPR**, and **0.19% at 0.1% FPR**, against the ≥95% sensitivity that
screening frameworks assume. Bulk Spearman predicts operating-point recall only loosely
(rho +0.683), so a proxy validated in bulk can still be useless where it is deployed.

**Crosstalk — the proxy gets worse at specificity as the model gets bigger.**
Binding specificity is two-sided: bind this, not that. On two dense landscapes where *both*
sides are measured for every variant — 7,882 ParD3 variants against two toxins, 14,454 BPTI
variants against three proteases — ESM-2 likelihood separates specific from promiscuous binders
*below chance*, and monotonically worse with scale: AUC **0.296 → 0.151** from 8M to 650M on one
system, **0.507 → 0.328** on the other. A mutation count reaches 0.664. Retrieval does not fix it
(flat across MSA depths 0–128), and the effect is position-dependent in a way no tested mechanism
explains (p=0.0035). Includes RL environments that turn these landscapes into budgeted design
problems.

**PRISM — an interpretability pipeline that had to audit itself.**
Measuring how mutational processes push proteins through function space, without a homology
anchor, produces artifacts that look exactly like signal. The self-audit
(`audit_prism_results.py`) is as much of the contribution as the pipeline: the embedding-proximity
metric is composition-confounded and adversarially gameable, so a signature that appears to
drive a functional transition is not thereby a validated driver.

---

## Layout

```
prism/         mutational signatures as directional drivers in protein function space
gauntlet/      CLI tool: audit a scorer, plan the next batch, backtest a strategy
shibboleth/    biosecurity screening validity; reads gauntlet's caches, no hazardous material
crosstalk/     two-sided specificity landscapes, the proxy ladder, and RL design environments
```

Each directory keeps its own `README.md`, and `FINDINGS.md` where one exists — those hold the
detail, the caveats, and the numbers that did not survive checking.

Manuscripts and submission planning are deliberately not here. This repository is the code and
the measurements; the write-ups live elsewhere.

## Running any of it

```bash
# Gauntlet — the entry point most people want
cd gauntlet && pip install -e .
gauntlet audit --campaign my_rounds.csv                # is my scorer measuring anything?
gauntlet plan  --campaign my_rounds.csv --budget 96    # what do I order next?

# Shibboleth — reuses gauntlet's caches, CPU only.
# It locates gauntlet as a sibling directory, so keep this layout intact;
# it needs gauntlet/data/proteingym/, which is excluded here (see below).
cd shibboleth
python scripts/run_screening_operating_point.py --scorer esm2

# PRISM — scripts import each other as siblings, keep them together
python audit_prism_results.py --runs_dir all_runs --target_domain PF17041 --focus_signature SBS17a

# Crosstalk — fetches its landscapes from GEO, then runs on CPU
cd crosstalk
python scripts/fetch_data.py && python -m pytest tests/ -q
python scripts/run_benchmark.py --seeds 30
```

`crosstalk/notebooks/auditing_ai_protein_tools.ipynb` is the guided tour: it scores the whole
landscape with three forward passes and is the fastest way to see what these projects do.

## What is not in this repository

Large inputs are excluded and are all re-obtainable:

- **PRISM** — the Pfam-A HMM database (~4 GB), downloadable from InterPro
- **Gauntlet** — cached ProteinGym assays and model scores (~350 MB) under
  `gauntlet/data/proteingym/`, and intermediate result tables above 2 MB; the small result
  files that back the figures are kept. Shibboleth reads this cache, so its screening
  script needs the cache restored before it will run
- **Crosstalk** — the ParD3 and BPTI landscapes (`scripts/fetch_data.py` re-downloads ParD3
  from GEO; BPTI is a supplementary file from the cited paper), Boltz structure predictions,
  and result tables above 2 MB
- Model weights throughout — fetched from HuggingFace on first run

This is the shareable subset: code, documentation, and the results small enough to read.
The full working trees live separately.

## A note on how these were built

A recurring pattern across all four, worth stating because it shaped the method: the
*observations* held up under attack, and the *explanations* mostly did not. Metrics that looked
predictive turned out to be composition-confounded; nulls that looked like noise turned out to
be real; mechanisms that fit the data cleanly failed the first causal test built to break them. In
crosstalk that happened six times in a row, and each failure is still written down in its
`FINDINGS.md` next to the observation that survived it.

So each project reports its trivial baselines alongside its models, states what a result would
look like if it were wrong, and keeps the failures in `FINDINGS.md` rather than deleting them.
A proxy that cannot beat counting mutations is not measuring what it claims, however good its
correlation looks in bulk — and that is a check worth running before the money is spent, not
after.
