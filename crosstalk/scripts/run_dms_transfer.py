#!/usr/bin/env python3
"""Genomic->protein transfer across 29 DMS assays, with a noise floor per assay.

Section 14 showed a genomic LM carries no binding-specificity signal on ParD3.
One landscape cannot separate "genomic LMs do not transfer to protein function"
from "ParD3 is peculiar". This runs the matched comparison across every
ProteinGym assay for which a native coding sequence could be verified: human,
yeast, viral and bacterial proteins, 29 assays.

The comparison is deliberately like-for-like. Both models are scored by the same
masked-marginal protocol on the same region -- the assayed target sequence and
the CDS that encodes exactly it -- so the only difference is the modality.

The synonymous floor comes free. Scoring a variant with a genomic LM requires
choosing a codon, and the masked token distribution already assigns a probability
to every synonymous alternative. Reading them off the same masked pass gives, at
no extra cost, the spread of scores across encodings that translate identically
and therefore share a DMS score exactly. That spread is a per-assay ceiling on
how well any genomic-LM proxy could correlate, and it is not something the
protein side can be charged with.
"""
import argparse, csv, json, sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))

from crosstalk import glm
from run_proxy_ladder import spearman

GAUNTLET = Path("/Users/saanviiyer/Downloads/CALTECH/RESEARCH/gauntlet/data/proteingym")


def load_singles(dms_id, target_seq):
    """Single mutants only: (position 1-based, wt, mut, score)."""
    out = []
    with (GAUNTLET / "assays" / f"{dms_id}.csv").open() as f:
        for row in csv.DictReader(f):
            m = row.get("mutant", "")
            if not m or ":" in m:
                continue
            wt, mu = m[0], m[-1]
            try:
                p = int(m[1:-1])
            except ValueError:
                continue
            if not (1 <= p <= len(target_seq)) or target_seq[p - 1] != wt:
                continue
            try:
                s = float(row["DMS_score"])
            except (KeyError, ValueError):
                continue
            out.append((p, wt, mu, s))
    return out


@torch.no_grad()
def esm_scores(model, tok, dev, seq, positions):
    """Masked-marginal log-prob table at each position: {pos: (20,) logprobs}."""
    lp = {}
    for p in positions:
        masked = seq[:p - 1] + tok.mask_token + seq[p:]
        enc = tok(masked, return_tensors="pt").to(dev)
        logits = model(**enc).logits[0]
        at = (enc["input_ids"][0] == tok.mask_token_id).nonzero()[0, 0]
        lp[p] = torch.log_softmax(logits[at].float(), -1).cpu().numpy()
    return lp


@torch.no_grad()
def nt_token_logprobs(sc, cds, positions, cds_offset=0):
    """For each mutated aa position, the masked log-prob over all 4096 6-mers."""
    ids0 = sc.tok(cds, return_tensors="pt")["input_ids"]
    n_tok = ids0.shape[1] - 1
    need = {}
    for p in positions:
        t, slot = sc.token_index(cds_offset, p)
        if t < n_tok:
            need.setdefault(t, []).append((p, slot))
    lp = {}
    toks = list(need)
    for i in range(0, len(toks), 16):
        chunk = toks[i:i + 16]
        batch = ids0.repeat(len(chunk), 1)
        for j, t in enumerate(chunk):
            batch[j, t + 1] = sc.tok.mask_token_id
        logits = sc.model(input_ids=batch.to(sc.device)).logits
        for j, t in enumerate(chunk):
            lp[t] = torch.log_softmax(logits[j, t + 1].float(), -1).cpu().numpy()
    return lp, need


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--esm", default="facebook/esm2_t33_650M_UR50D")
    ap.add_argument("--nt", nargs="+",
                    default=["InstaDeepAI/nucleotide-transformer-v2-50m-multi-species"])
    ap.add_argument("--context", default="cds", choices=["cds", "flanked"],
                    help="cds = the coding sequence alone (matched to the protein "
                         "rung); flanked = the CDS in its real genomic neighbourhood, "
                         "which is the input a long-range genomic model expects")
    ap.add_argument("--flank", type=int, default=3000)
    ap.add_argument("--max-len", type=int, default=420)
    ap.add_argument("--out", default="results/dms_transfer.csv")
    args = ap.parse_args()

    cdsmap = json.loads((ROOT / "data" / "cds" / "dms_cds.json").read_text())
    ref = {r["DMS_id"]: r for r in csv.DictReader((GAUNTLET / "reference.csv").open())}

    from transformers import AutoModelForMaskedLM, AutoTokenizer
    dev = glm._device()
    etok = AutoTokenizer.from_pretrained(args.esm)
    emdl = AutoModelForMaskedLM.from_pretrained(args.esm).to(dev).eval()
    scorers = [(name, glm.NTScorer(name)) for name in args.nt]
    pref = glm.preferred_codons()
    print(f"loaded ESM + {len(scorers)} genomic model(s) on {dev}", flush=True)

    ctxmap = {}
    if args.context == "flanked":
        cf = ROOT / "data" / "cds" / "dms_context.json"
        ctxmap = json.loads(cf.read_text())
        print(f"flanked context available for {len(ctxmap)} assays", flush=True)
    print(flush=True)

    rows = []
    hdr = (f"{'assay':30s} {'ntmodel':9s} {'n':>5s} {'ESM':>7s} {'NT':>7s} "
           f"{'NTmarg':>7s} {'floor':>7s} {'ceil':>6s}")
    print(hdr); print("-" * len(hdr))
    for dms, rec in sorted(cdsmap.items()):
        tgt = ref[dms]["target_seq"]
        if len(tgt) > args.max_len:
            print(f"{dms[:40]:40s} skip (len {len(tgt)} > {args.max_len})")
            continue
        muts = load_singles(dms, tgt)
        if len(muts) < 200:
            print(f"{dms[:40]:40s} skip ({len(muts)} usable singles)")
            continue
        off = rec["aa_offset"]
        cds = rec["cds"][off * 3: (off + len(tgt)) * 3]
        if glm.translate(cds) != tgt:
            print(f"{dms[:40]:40s} skip (CDS slice does not translate to target)")
            continue

        positions = sorted({p for p, _, _, _ in muts})
        elp = esm_scores(emdl, etok, dev, tgt, positions)

        for nt_name, sc in scorers:
            vocab = sc.tok.get_vocab()
            if args.context == "flanked":
                c = ctxmap.get(dms)
                if c is None:
                    continue
                seq_for_nt, cds_off = c["segment"], c["cds_offset"]
            else:
                seq_for_nt, cds_off = cds, 0
            nlp, _ = nt_token_logprobs(sc, seq_for_nt, positions, cds_off)

            e_s, n_s, m_s, y, floors = [], [], [], [], []
            for p, wt, mu, s_ in muts:
                t, slot = sc.token_index(cds_off, p)
                if t not in nlp:
                    continue
                e_s.append(float(elp[p][etok.convert_tokens_to_ids(mu)]
                                 - elp[p][etok.convert_tokens_to_ids(wt)]))
                base = list(seq_for_nt[t * 6:(t + 1) * 6])
                wt_lp = nlp[t][vocab.get("".join(base), sc.tok.unk_token_id)]
                syn = []
                for codon in glm.SYNONYMOUS[mu]:
                    tk = list(base); tk[slot * 3:slot * 3 + 3] = list(codon)
                    syn.append(nlp[t][vocab.get("".join(tk), sc.tok.unk_token_id)] - wt_lp)
                tk = list(base); tk[slot * 3:slot * 3 + 3] = list(pref[mu])
                n_s.append(nlp[t][vocab.get("".join(tk), sc.tok.unk_token_id)] - wt_lp)
                m_s.append(float(np.mean(syn)))
                floors.append(np.std(syn, ddof=1) if len(syn) > 1 else 0.0)
                y.append(s_)

            if len(y) < 200:
                continue
            e_a, n_a, m_a, y_a = (np.array(e_s), np.array(n_s),
                                  np.array(m_s), np.array(y))
            r_e, r_n, r_m = spearman(e_a, y_a), spearman(n_a, y_a), spearman(m_a, y_a)
            w = float(np.mean(floors)); b = float(np.std(n_a, ddof=1))
            ceil = float(np.sqrt(max(b ** 2 - w ** 2, 0.0)) / b) if b > 0 else np.nan
            rows.append(dict(dms_id=dms, nt_model=nt_name, context=args.context,
                             organism=ref[dms].get("source_organism", ""),
                             n=len(y_a), target_len=len(tgt), rho_esm=r_e,
                             rho_nt=r_n, rho_nt_codon_marginalised=r_m,
                             syn_within_sd=w, nt_between_sd=b,
                             attenuation_ceiling=ceil))
            tag = nt_name.split("-v2-")[-1].replace("-multi-species", "")
            print(f"{dms[:30]:30s} {tag:9s} {len(y_a):5d} {r_e:+7.3f} {r_n:+7.3f} "
                  f"{r_m:+7.3f} {w:7.2f} {ceil:6.3f}", flush=True)

    if rows:
        print("-" * len(hdr))
        for nt_name, _ in scorers:
            g = [r for r in rows if r["nt_model"] == nt_name]
            if not g:
                continue
            e = np.array([r["rho_esm"] for r in g]); n = np.array([r["rho_nt"] for r in g])
            m = np.array([r["rho_nt_codon_marginalised"] for r in g])
            ci = 1.96 * n.std(ddof=1) / np.sqrt(len(n))
            tag = nt_name.split("-v2-")[-1].replace("-multi-species", "")
            print(f"{'MEAN ' + tag:30s} {'':9s} {len(g):5d} {e.mean():+7.3f} "
                  f"{n.mean():+7.3f} {m.mean():+7.3f}   CI [{n.mean()-ci:+.3f}, "
                  f"{n.mean()+ci:+.3f}]  ceil {np.nanmean([r['attenuation_ceiling'] for r in g]):.3f}"
                  f"  ESM larger |rho| {int((np.abs(e) > np.abs(n)).sum())}/{len(g)}")
        out = ROOT / args.out
        keys = sorted({k for r in rows for k in r})
        with out.open("w", newline="") as f:
            w_ = csv.DictWriter(f, fieldnames=keys); w_.writeheader(); w_.writerows(rows)
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
