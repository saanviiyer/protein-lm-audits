# =====================================================================
#  RAM-FRUGAL Evo 2 generation cell (paste into Colab).
#
#  Evo 2 is heavy. Use a Colab runtime = A100 (40GB VRAM) + High-RAM.
#  Do NOT run on a T4/L4 free tier -- the model OOMs.
#  Do NOT interrupt the one-time ~14GB weight download.
#
#  This produces evo2_output/<gene>_generated.fasta (+ _expected_continuation),
#  the drop-in third model for stress_test_generation.py (§5) and the ICBINB
#  collapse/artifact figure. It is self-contained (no zip dependency).
# =====================================================================
import os
# reduce CUDA fragmentation BEFORE importing torch (helps on tight VRAM)
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# --- install (light install is enough for the 7B model) ---
# !pip install -q flash-attn==2.8.0.post2 --no-build-isolation
# !pip install -q evo2

import re, gc, torch
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from evo2 import Evo2

BASE   = "/content/drive/MyDrive/PRISM/genmodel_bias"
PROMPTS = f"{BASE}/prompts.fasta"
OUT     = f"{BASE}/evo2_output"
os.makedirs(OUT, exist_ok=True)

# RAM-frugal knobs: small batch, modest N, per-gene write, aggressive cleanup.
N, BATCH, MAXTOK, PREFIX, TEMP, TOPK = 30, 2, 400, 0.2, 1.0, 4

print("Loading evo2_7b (~14GB, one-time; bf16 fits an A100). DO NOT interrupt...")
model = Evo2("evo2_7b")            # bf16, no Transformer Engine needed on A100
print("loaded.")
clean = lambda s: re.sub(r"[^ACGTN]", "", re.sub(r"<[^>]+>", "", s.upper()))

for rec in SeqIO.parse(PROMPTS, "fasta"):
    gid, seq = rec.id, str(rec.seq).upper().replace(" ", "")
    cut = int(len(seq) * PREFIX); prefix, suffix = seq[:cut], seq[cut:]
    outp = os.path.join(OUT, f"{gid}_generated.fasta")
    if not prefix or os.path.exists(outp):        # idempotent: skip done genes
        print("skip", gid); continue
    recs, n = [], 0
    while n < N:
        b = min(BATCH, N - n)
        with torch.inference_mode():
            out = model.generate(prompt_seqs=[prefix] * b, n_tokens=MAXTOK,
                                 temperature=TEMP, top_k=TOPK, verbose=0)
        seqs = out.sequences if hasattr(out, "sequences") else out[0]
        for s in seqs:
            cont = s[len(prefix):] if s.upper().startswith(prefix.upper()) else s
            recs.append(SeqRecord(Seq(clean(cont)), id=f"{gid}|sample_{n+1:03d}", description="evo2"))
            n += 1
        del out, seqs; gc.collect(); torch.cuda.empty_cache()     # <-- keep VRAM flat
    SeqIO.write(recs, outp, "fasta")
    if suffix:
        SeqIO.write([SeqRecord(Seq(suffix), id=gid, description="true continuation")],
                    os.path.join(OUT, f"{gid}_expected_continuation.fasta"), "fasta")
    print(f"{gid}: wrote {len(recs)} continuations")
    del recs; gc.collect(); torch.cuda.empty_cache()
print("DONE ->", OUT)

# ---------------------------------------------------------------------
# After this finishes, run (from the unzipped repo, on any runtime):
#   python stress_test_generation.py \
#       --model_dir carbon:{BASE}/carbon_output \
#       --model_dir generator:{BASE}/generator_output \
#       --model_dir evo2:{OUT} \
#       --reference_fasta {PROMPTS} --prefix_frac 0.2 \
#       --cosmic_dir <COSMIC> --output_dir {BASE}/results/stress_test
#   python fill_paper_numbers.py --stress_dir {BASE}/results/stress_test \
#       --sweep_dir {BASE}/results/sweep_analysis --paper_dir <icbinb tex dir>
# -> fills ICBINB's Evo 2 markers (the single-nucleotide third violin).
# ---------------------------------------------------------------------
