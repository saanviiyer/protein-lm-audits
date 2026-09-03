#!/usr/bin/env python3
"""
esm_ddg_drift.py

A LEARNED, context-aware stability readout for mutational signatures, moving the
SIMBIOCHEM "physical" axis from a static substitution matrix (BLOSUM) toward an
energy-like estimate. For each signature we score its mutations with the ESM-2
masked-language-model head using the standard "wild-type marginals" zero-shot
Delta-Delta-G proxy (Meier et al., 2021):

    ddG_proxy(mut) = sum over mutated positions i of
                     [ log p(mut_aa_i | WT context) - log p(wt_aa_i | WT context) ]

Lower (more negative) = the mutation moves residues toward amino acids the
language model finds less probable in context, i.e. predicted destabilization.
This needs a single ESM-2 forward pass per gene (WT), then cheap look-ups per
draw, so it runs on CPU in minutes -- no structure prediction, no MD.

USAGE (run with a python that has torch+transformers, e.g. the anaconda one)
-----
python esm_ddg_drift.py --cds_fasta all_runs/CL0023/pairs_cds.fasta \
    --profile_dir profiles --signatures SBS17a,SBS1,SBS7a,SBS13,SBS2 \
    --n_draws 40 --max_genes 12 --output_dir results/esm_ddg
"""
import argparse
import csv
import os
import random
import sys

import numpy as np
from Bio import SeqIO
from Bio.Seq import Seq

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

AA = "LAGVSERTIDPKQNFYMHWC"


def translate_cds(nt):
    t = nt[:len(nt) - (len(nt) % 3)]
    return str(Seq(t).translate(to_stop=True))


def mutate_cds(nt, profile):
    s = list(nt)
    for i in range(len(s) - 2):
        ctx = "".join(s[i:i + 3])
        for sig, p in profile.items():
            if len(sig) >= 7 and ctx == sig[0] + sig[2] + sig[6] and random.random() < p:
                s[i + 1] = sig[4]
                break
    return "".join(s)


def load_profile(path):
    d = {}
    for ln in open(path).readlines()[1:]:
        q = ln.split()
        if len(q) >= 2 and len(q[0]) >= 7 and q[0][1] == "[" and q[0][5] == "]":
            try:
                d[q[0]] = float(q[1])
            except ValueError:
                pass
    return d


def load_profiles(profile_dir, wanted):
    wanted = {w.strip() for w in wanted if w.strip()}
    found = {}
    for root, _, files in os.walk(profile_dir):
        for fn in files:
            if not fn.upper().endswith("_PROFILE.TXT"):
                continue
            stem = fn.split("_")[0].split(".")[0]
            if stem in wanted:
                p = load_profile(os.path.join(root, fn))
                if p:
                    found[stem] = p
    return found


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cds_fasta", required=True)
    ap.add_argument("--profile_dir", default="profiles")
    ap.add_argument("--signatures", default="SBS17a,SBS1,SBS7a,SBS13,SBS2")
    ap.add_argument("--n_draws", type=int, default=40)
    ap.add_argument("--max_genes", type=int, default=12)
    ap.add_argument("--max_aa", type=int, default=1022, help="ESM-2 context cap.")
    ap.add_argument("--esm_model", default="facebook/esm2_t6_8M_UR50D")
    ap.add_argument("--output_dir", default="results/esm_ddg")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    os.makedirs(args.output_dir, exist_ok=True)

    import torch
    from transformers import AutoTokenizer, EsmForMaskedLM
    tok = AutoTokenizer.from_pretrained(args.esm_model)
    model = EsmForMaskedLM.from_pretrained(args.esm_model).eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    @torch.no_grad()
    def wt_logprobs(aa):
        aa = aa[:args.max_aa]
        enc = tok(aa, return_tensors="pt").to(device)
        logits = model(**enc).logits[0]                 # (L+2, vocab)
        lp = torch.log_softmax(logits, dim=-1)
        return lp, enc["input_ids"][0]

    def aa_id(a):
        return tok.convert_tokens_to_ids(a)

    profiles = load_profiles(args.profile_dir, args.signatures.split(","))
    if not profiles:
        raise SystemExit("No signature profiles loaded.")
    genes = []
    for r in SeqIO.parse(args.cds_fasta, "fasta"):
        s = str(r.seq).upper().replace(" ", "")
        if s and set(s) <= set("ACGTN"):
            genes.append((r.id, s))
    genes = genes[:args.max_genes]
    print(f"[ddg] {len(genes)} genes x {len(profiles)} signatures x {args.n_draws} draws "
          f"(ESM-2 {args.esm_model.split('/')[-1]}, device={device})")

    rng = random.Random(args.seed)
    rows = []
    for gid, cds in genes:
        wt = translate_cds(cds)
        if len(wt) < 10:
            continue
        lp, ids = wt_logprobs(wt)                        # ids: [CLS] aa... [EOS]
        L = min(len(wt), args.max_aa)
        # per-position log-prob of the WT residue (offset by 1 for CLS)
        for sig, prof in profiles.items():
            ddgs = []
            for _ in range(args.n_draws):
                random.seed(rng.randrange(1 << 30))
                mut = translate_cds(mutate_cds(cds, prof))
                n = min(L, len(mut))
                score = 0.0
                for i in range(n):
                    a, b = wt[i], mut[i]
                    if a == b or a not in AA or b not in AA:
                        continue
                    pos = i + 1  # +1 for CLS token
                    score += float(lp[pos, aa_id(b)] - lp[pos, aa_id(a)])
                ddgs.append(score)
            rows.append([gid, sig, args.n_draws, round(float(np.mean(ddgs)), 3)])
        print(f"[ddg] {gid} ({len(wt)}aa) done")

    with open(os.path.join(args.output_dir, "esm_ddg_per_gene.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["gene", "signature", "n_draws", "mean_esm_ddg_proxy"])
        w.writerows(rows)

    import collections
    agg = collections.defaultdict(list)
    for gid, sig, nd, v in rows:
        agg[sig].append(v)
    summary = [[sig, round(float(np.mean(vs)), 3), round(float(np.std(vs)), 3), len(vs)]
               for sig, vs in agg.items()]
    summary.sort(key=lambda r: r[1])  # most destabilizing (most negative) first
    with open(os.path.join(args.output_dir, "esm_ddg_by_signature.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["signature", "mean_esm_ddg_proxy", "std", "n_genes"])
        w.writerows(summary)
    print(f"\n{'signature':10}{'ESM-2 ddG proxy':>18}{'std':>8}")
    for sig, m, sd, n in summary:
        print(f"{sig:10}{m:>18.2f}{sd:>8.2f}")
    print(f"\n[ddg] wrote {args.output_dir}/esm_ddg_by_signature.csv "
          "(lower = more destabilizing)")


if __name__ == "__main__":
    main()
