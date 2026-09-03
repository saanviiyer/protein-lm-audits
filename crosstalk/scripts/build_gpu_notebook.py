"""Generate the A100 notebook: the two arms that this laptop cannot run."""
import json
from pathlib import Path

C = []
def _lines(s):
    b = s.strip("\n").split("\n")
    return [l + "\n" for l in b[:-1]] + [b[-1]]
def md(s): C.append({"cell_type": "markdown", "metadata": {}, "source": _lines(s)})
def code(s): C.append({"cell_type": "code", "execution_count": None, "metadata": {},
                       "outputs": [], "source": _lines(s)})

md(r"""
# crosstalk on an A100

### The two arms that a 26 GB laptop cannot run

Everything else in this project runs on CPU. Two things do not, and they are gated on the
same hardware, so one rented session clears both:

| arm | why it is blocked | what it settles |
|---|---|---|
| **Boltz co-folding audit at scale** | ~160 folds; 13-40 min each on CPU | whether ipTM's +0.268 specificity correlation is real (currently n=16, needs \|rho\|>0.50) |
| **AIDO 16B + structure embeddings** | 64 GB weights, ~32 GB resident | the last untested input modality; only the 16B has `str_embedding_in` |

**Requirements**: one A100 (40 GB is enough for both; 80 GB is more comfortable),
~120 GB disk, and about 6-8 hours. Costs roughly \$10-25 on current spot pricing.

**Before you start**, check you actually got the GPU you are paying for — the first cell
does this. Everything writes to `results/` incrementally, so an interrupted session loses
one fold, not the run.
""")

code(r"""
import subprocess, sys, os, shutil
print(subprocess.run(["nvidia-smi","--query-gpu=name,memory.total,driver_version",
                      "--format=csv"], capture_output=True, text=True).stdout)
free_gb = shutil.disk_usage(".").free / 1e9
print(f"free disk: {free_gb:.0f} GB   (need ~120 GB for both arms)")
if free_gb < 120:
    print("  -> not enough for the 16B arm (64 GB weights). The Boltz arm alone needs ~20 GB.")
""")

md(r"""
## Setup

Two environments, because they conflict: `boltz` pins `einops==0.8.0` while `modelgenerator`
wants a newer one, and `modelgenerator` breaks on `docstring-inheritance>=2.3.0`.

Keeping them separate is less painful than resolving it. Note the two pins below — both were
found the hard way.
""")

code(r"""
%%bash
set -e
git clone https://github.com/saanviiyer/crosstalk.git 2>/dev/null || echo "(using local checkout)"
cd crosstalk 2>/dev/null || true

python -m venv .venv-boltz
.venv-boltz/bin/pip install -q --upgrade pip
.venv-boltz/bin/pip install -q boltz modelgenerator
# modelgenerator breaks on 3.0.0 AND 2.3.0; 2.2.2 is the working pin
.venv-boltz/bin/pip install -q "docstring-inheritance==2.2.2"
.venv-boltz/bin/pip install -q scipy biotite

pip install -q numpy matplotlib torch transformers
python scripts/fetch_data.py
echo "setup done"
""")

md(r"""
---
# Arm 1: the Boltz audit at scale

## Why n=16 was not enough

The completed 32-fold run gave, matched on the same variants:

| proxy | rho vs measured specificity margin |
|---|---|
| Boltz ipTM margin | **+0.268** |
| ESM-2 650M likelihood | **-0.294** |

Boltz ranks the target above the off-target; ESM-2 inverts it. That is the qualitative split
you would predict from modelling the complex versus scoring one sequence — but at n=16
Spearman needs |rho| > 0.50 for p=0.05, so it is a direction, not a result.

We also ruled out the cheapest explanation for the weak signal: interface PAE has ~10x the
dynamic range of the saturated ipTM (0.942-0.968) and gives the *same* correlation (+0.279).
Compression was not the bottleneck, so the fix is sample size.

**This run**: 20 variants per specificity quadrant, 80 variants, 160 folds. On an A100 a fold
is ~30-60 s rather than 13-40 min, so this is roughly 2-3 hours.
""")

code(r"""
%%bash
cd crosstalk 2>/dev/null || true
# --workers 8 saturates an A100 without exhausting VRAM; each fold is independent
# and results are written incrementally, so this is safely resumable.
.venv-boltz/bin/python scripts/run_boltz_parallel.py \
    --n-per-cell 20 --workers 8 --timeout 3600 \
    --out results/boltz_proxy_a100.csv
""")

md(r"""
Boltz defaults never completed in 50 minutes on CPU, so the local runs used
`--recycling_steps 0 --diffusion_samples 1 --sampling_steps 25`. On an A100 you can afford
the defaults, which is a *better* test of the proxy: if ipTM only works at full sampling,
that matters for anyone deciding whether to use it.

Run this second, and compare. If the two agree, the cheap settings are validated for
everyone without a GPU.
""")

code(r"""
%%bash
cd crosstalk 2>/dev/null || true
# Full-quality re-run of the same variants. Edit run_boltz_parallel.py's one_fold()
# to drop the --recycling_steps/--diffusion_samples/--sampling_steps flags first,
# or pass a --full-quality flag if you have added one.
.venv-boltz/bin/python scripts/run_boltz_parallel.py \
    --n-per-cell 20 --workers 6 --timeout 7200 \
    --workdir results/boltz_full --out results/boltz_proxy_full.csv
""")

code(r"""
import sys; sys.path.insert(0, "crosstalk" if os.path.exists("crosstalk") else ".")
%run scripts/analyze_boltz_proxy.py --inp results/boltz_proxy_a100.csv
%run scripts/analyze_boltz_pae.py
""")

md(r"""
### What to conclude

The number that decides it is **AUC for separating specific from promiscuous binders**,
against the mutation-count baseline of **0.664**.

| outcome | what it means |
|---|---|
| Boltz AUC clearly **> 0.664** | specificity needs the complex modelled, and nothing cheaper works. A *constructive* result: the ladder has a top rung that works. |
| Boltz AUC ≈ 0.664 | co-folding confidence matches counting mutations. Expensive and unnecessary. |
| Boltz AUC **< 0.664** | the negative result is complete across cheap and expensive proxies. Stronger, and more uncomfortable for the field. |

All three are publishable. Only the first changes the recommendation to practitioners.
""")

md(r"""
---
# Arm 2: AIDO 16B with structure embeddings

## The modality that has not been tested

Sequence alone did not predict specificity. Retrieval did not rescue it — AUC was flat at
0.212 / 0.196 / 0.228 / 0.230 across MSA depths of 0, 20, 64 and 128 rows, using
diversity-greedy selection out to the model's real 12.8K context.

Structure is what is left, and it is the interesting one: it gives the language model the
same information Boltz has, which makes the PLM-versus-cofolding comparison like-for-like
rather than a comparison of input modalities.

**Check the config before planning around this.** Within one model family:

```
AIDO.Protein-RAG-3B    str_embedding_in = None    <- cannot accept structure at all
GB.Protein-RAG-16B     str_embedding_in = 384     <- can
```

The 3B and 16B share a name, an architecture family and a tutorial, and differ on whether
this modality exists.
""")

code(r"""
%%bash
cd crosstalk 2>/dev/null || true
# GenBio's own tutorial utilities: GB_Structure_Tokenizer, greedy_select, protein.from_pdb_string
git clone -q https://github.com/genbio-ai/GB-Foundations-Tutorials.git /tmp/gbtut || true
cp -r /tmp/gbtut/Notebooks/utils crosstalk_gbutils 2>/dev/null || cp -r /tmp/gbtut/Notebooks/utils ./
ls utils/ 2>/dev/null || ls crosstalk_gbutils/
""")

code(r"""
# Structure embeddings from the Boltz-predicted complexes we already have.
import glob, numpy as np, torch
sys.path.insert(0, ".")
from utils import misc, protein

str_tok = misc.GB_Structure_Tokenizer(device="cuda:0")

pdbs = sorted(glob.glob("results/boltz/*/boltz_results_*/predictions/*/*_model_0.pdb"))
print(f"{len(pdbs)} Boltz structures available")

def embed(pdb_path):
    prot = protein.from_pdb_string(open(pdb_path).read(), molecular_type="protein")
    emb, toks = str_tok.encode(prot.aatype, prot.atom_positions, prot.atom_mask,
                               get_embedding=True)
    return emb, toks

if pdbs:
    e, t = embed(pdbs[0])
    print(f"embedding {tuple(e.shape)}  tokens {tuple(t.shape)}")
    print("note: this is the WHOLE complex (ParD3 + partner);")
    print("slice the first 93 rows for the ParD3 chain the model scores.")
""")

code(r"""
# Score with the 16B, sequence + MSA + structure.
# ~64 GB download; load in bf16 (~32 GB) to fit a 40 GB card.
from modelgenerator.huggingface_models.fm4bio import FM4BioForMaskedLM, FM4BioTokenizer
import modelgenerator, os

REPO = "genbio-ai/GB.Protein-RAG-16B"
vocab = os.path.join(os.path.dirname(modelgenerator.__file__),
                     "huggingface_models", "fm4bio", "vocab_protein.txt")
tokenizer = FM4BioTokenizer(vocab_file=vocab)
model = FM4BioForMaskedLM.from_pretrained(REPO, torch_dtype=torch.bfloat16).cuda().eval()
print("loaded", REPO)
""")

code(r"""
# Masked-marginal with all three modalities. Mirrors crosstalk/genbio.py, plus str_embs.
from crosstalk.landscape import load_pard3
from crosstalk.genbio import load_msa
from crosstalk.boltz import PARD3, MUT_POSITIONS
sys.path.insert(0, "scripts")
from run_proxy_ladder import auc, boot_ci, spearman

L = load_pard3()
w3, w2 = L.F[:, 0], L.F[:, 1]
spec = (w3 >= 0.8) & (w2 <= 0.2); prom = (w3 >= 0.8) & (w2 >= 0.6)
mask = spec | prom; lab = spec[mask]

msa_csv = sorted(glob.glob("results/boltz/*/boltz_results_*/msa/*_0.csv"))[0]
msa = load_msa(msa_csv, max_rows=128, diverse=True)
print(f"{len(msa)} diverse MSA rows")

@torch.no_grad()
def score_all(msa_rows, str_embs=None):
    rows = [PARD3] + list(msa_rows)
    n = len(PARD3)
    flat = np.array(list("".join(rows)))
    pos = np.stack([np.tile(np.arange(n), len(rows)),
                    np.repeat(np.arange(len(rows)), n)])
    keep = flat != "-"
    flat, pos = flat[keep], pos[:, keep]

    lp = []
    for p in MUT_POSITIONS:
        toks = list(flat); toks[p-1] = tokenizer.mask_token
        enc = tokenizer("".join(toks), return_tensors="pt", add_special_tokens=False)
        kw = dict(input_ids=enc["input_ids"].cuda(),
                  position_ids=torch.as_tensor(pos).long().unsqueeze(0).cuda())
        if str_embs is not None:
            kw["inputs_str_embeds"] = str_embs
        out = model(**kw).logits
        at = (kw["input_ids"][0] == tokenizer.mask_token_id).nonzero()[0, 0]
        lp.append(torch.log_softmax(out[0, at].float(), -1).cpu().numpy())
    lp = np.stack(lp)
    wt = [PARD3[p-1] for p in MUT_POSITIONS]
    tid = {a: tokenizer.convert_tokens_to_ids(a) for a in set("".join(L.seqs) + "".join(wt))}
    return np.array([sum(lp[k, tid[a]] - lp[k, tid[wt[k]]] for k, a in enumerate(v))
                     for v in L.seqs])

for name, m, se in (("16B sequence only", [], None),
                    ("16B + MSA", msa, None)):
    s = score_all(m, se)
    a = auc(s[mask], lab); lo, hi = boot_ci(s[mask], lab)
    print(f"  {name:22s} AUC {a:.3f} [{lo:.3f}, {hi:.3f}]  "
          f"rho_on={spearman(s,w3):+.3f} rho_off={spearman(s,w2):+.3f}")
print("\n  3B reference: 0.212 (no MSA) / 0.230 (128 rows) | baseline 0.664 | chance 0.5")
print("  Add the structure arm once the chain slicing above is confirmed.")
""")

md(r"""
### What to conclude

| outcome | what it means |
|---|---|
| structure arm AUC **> 0.664** | the anti-correlation is a *sequence-representation* problem, and structure fixes it. Constructive, and a clear recommendation. |
| structure arm ≈ the others (~0.2) | the negative result holds across **sequence, retrieval and structure** — three modalities, two labs, 8M to 16B parameters. That is close to airtight. |

Note the second outcome is the stronger paper, and it is also the one the 3B results predict.
""")

md(r"""
---
# After the run

```bash
# pull results back before the instance dies
tar czf crosstalk_a100_results.tar.gz results/
```

Then locally:

```bash
python3 scripts/analyze_boltz_proxy.py --inp results/boltz_proxy_a100.csv
python3 scripts/analyze_boltz_pae.py
python3 scripts/make_ladder_figure.py
```

### Budget

| arm | A100 time | disk |
|---|---|---|
| Boltz, 160 folds, fast settings | ~2-3 h | ~20 GB |
| Boltz, 160 folds, full defaults | ~3-4 h | ~20 GB |
| AIDO 16B download + scoring | ~1 h | ~64 GB |

One 6-8 hour session covers everything. Run the Boltz arm first: it settles a number that is
already in the draft, whereas the 16B arm adds a new one.
""")

nb = {"cells": C, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
      "name": "python3"}, "language_info": {"name": "python", "version": "3.11"}},
      "nbformat": 4, "nbformat_minor": 5}
out = Path(__file__).resolve().parents[1] / "notebooks" / "run_on_a100.ipynb"
out.parent.mkdir(exist_ok=True)
out.write_text(json.dumps(nb, indent=1))
print(f"wrote {out} ({len(C)} cells)")
