#!/usr/bin/env python3
"""Does the readout/representation split hold beyond ParD3?

On ParD3 the same model gives opposite answers depending on what is read out of
it: ESM-2 650M likelihood scores AUC 0.151 on specific-versus-promiscuous, while
a linear probe on its frozen embeddings scores far above chance. That is one
protein, so it is a hypothesis, not a result.

This runs the same contrast across the SKEMPI two-sided set: about ten
independent biological systems -- protease/inhibitor, hormone/receptor, TCR/pMHC,
antibody/antigen, enzyme/inhibitor -- where the same mutations were measured
against two different partners.

The generalisation test is leave-one-system-out. The probe never sees the system
it is scored on, so it cannot fit that system's idiosyncratic scale, and a
correlation that survives is a statement about transfer rather than about fit.
Trivial baselines (BLOSUM62, hydrophobicity change, volume change) are reported
alongside, because a probe that cannot beat a substitution matrix has not learned
anything a lookup table does not already contain.
"""
import argparse, csv, sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))

from crosstalk import skempi
from crosstalk.glm import _device
from run_proxy_ladder import spearman
from run_readout_probe import ridge_fit, ridge_pred

# Kyte-Doolittle hydropathy and residue volume (A^3): the trivial chemistry baseline
KD = dict(A=1.8, R=-4.5, N=-3.5, D=-3.5, C=2.5, Q=-3.5, E=-3.5, G=-0.4, H=-3.2,
          I=4.5, L=3.8, K=-3.9, M=1.9, F=2.8, P=-1.6, S=-0.8, T=-0.7, W=-0.9,
          Y=-1.3, V=4.2)
VOL = dict(A=88.6, R=173.4, N=114.1, D=111.1, C=108.5, Q=143.8, E=138.4, G=60.1,
           H=153.2, I=166.7, L=166.7, K=168.6, M=162.9, F=189.9, P=112.7,
           S=89.0, T=116.1, W=227.8, Y=193.6, V=140.0)


def blosum62():
    from Bio.Align import substitution_matrices
    return substitution_matrices.load("BLOSUM62")


@torch.no_grad()
def esm_features(systems, model_name, batch=16):
    """Per-mutation likelihood score and embedding delta from one frozen model."""
    from transformers import AutoModelForMaskedLM, AutoTokenizer
    dev = _device()
    tok = AutoTokenizer.from_pretrained(model_name)
    mdl = AutoModelForMaskedLM.from_pretrained(model_name).to(dev).eval()

    out = []
    for s in systems:
        seq = s.sequence
        # masked-marginal likelihood at each mutated position, exactly as the ladder
        positions = sorted({int(m[1:-1]) for m in s.mutations})
        lp = {}
        for p in positions:
            masked = seq[:p - 1] + tok.mask_token + seq[p:]
            enc = tok(masked, return_tensors="pt").to(dev)
            logits = mdl(**enc).logits[0]
            at = (enc["input_ids"][0] == tok.mask_token_id).nonzero()[0, 0]
            lp[p] = torch.log_softmax(logits[at].float(), -1).cpu().numpy()

        # embeddings: wild type once, each mutant once; feature = delta at the site
        def embed(seqs):
            reps = []
            for i in range(0, len(seqs), batch):
                enc = tok(seqs[i:i + batch], return_tensors="pt", padding=True).to(dev)
                h = mdl(**enc, output_hidden_states=True).hidden_states[-1]
                reps.append(h.float().cpu())
            return reps

        wt_h = embed([seq])[0][0]
        mut_seqs = [seq[:int(m[1:-1]) - 1] + m[-1] + seq[int(m[1:-1]):] for m in s.mutations]
        mut_h = torch.cat(embed(mut_seqs), 0)

        lik, feats = [], []
        for i, m in enumerate(s.mutations):
            wt, p, mu = m[0], int(m[1:-1]), m[-1]
            lik.append(float(lp[p][tok.convert_tokens_to_ids(mu)]
                             - lp[p][tok.convert_tokens_to_ids(wt)]))
            d_site = (mut_h[i, p] - wt_h[p]).numpy()          # +1 for <cls>
            d_pool = (mut_h[i].mean(0) - wt_h.mean(0)).numpy()
            feats.append(np.concatenate([d_site, d_pool]))
        out.append((np.array(lik), np.stack(feats)))
        print(f"  {s.protein[:28]:28s} {s.partner_a[:18]:18s} vs {s.partner_b[:18]:18s} "
              f"n={len(s):3d}", flush=True)
    return out


def trivial_features(systems):
    B = blosum62()
    out = []
    for s in systems:
        rows = []
        for m in s.mutations:
            wt, mu = m[0], m[-1]
            b = float(B[wt, mu]) if (wt in B.alphabet and mu in B.alphabet) else 0.0
            rows.append([b, KD[mu] - KD[wt], VOL[mu] - VOL[wt],
                         abs(KD[mu] - KD[wt]), abs(VOL[mu] - VOL[wt])])
        out.append(np.array(rows))
    return out


def zscore(x):
    x = np.asarray(x, float)
    sd = x.std()
    return (x - x.mean()) / sd if sd > 1e-12 else x - x.mean()


def loso(features, targets, lam):
    """Leave-one-system-out predictions, each system z-scored within itself."""
    n = len(features)
    preds = []
    for k in range(n):
        Xtr = np.vstack([features[i] for i in range(n) if i != k])
        ytr = np.concatenate([zscore(targets[i]) for i in range(n) if i != k])
        mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-8
        w = ridge_fit((Xtr - mu) / sd, ytr, lam)
        preds.append(ridge_pred((features[k] - mu) / sd, w))
    return preds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="facebook/esm2_t33_650M_UR50D")
    ap.add_argument("--lam", type=float, default=300.0)
    ap.add_argument("--min-shared", type=int, default=10)
    ap.add_argument("--out", default="results/skempi_transfer.csv")
    args = ap.parse_args()

    print("building the two-sided set from SKEMPI 2.0 ...")
    systems = skempi.build(min_shared=args.min_shared)
    print(f"\n{len(systems)} systems, {sum(len(s) for s in systems)} verified "
          f"two-sided mutations\n")

    print(f"scoring with {args.model} ...")
    esm = esm_features(systems, args.model)
    lik = [e[0] for e in esm]
    emb = [e[1] for e in esm]
    triv = trivial_features(systems)
    targets = [s.margin for s in systems]

    emb_pred = loso(emb, targets, args.lam)
    triv_pred = loso(triv, targets, args.lam)

    rows = []
    print(f"\n{'system':46s} {'n':>4s} {'likelihood':>11s} {'chem LOSO':>10s} {'ESM LOSO':>9s}")
    for k, s in enumerate(systems):
        y = s.margin
        r_lik = spearman(lik[k], y)
        r_tri = spearman(triv_pred[k], y)
        r_emb = spearman(emb_pred[k], y)
        rows.append(dict(protein=s.protein, partner_a=s.partner_a, partner_b=s.partner_b,
                         pdb=s.pdb_a, chain=s.chain, n=len(s),
                         rho_likelihood=r_lik, rho_trivial_loso=r_tri,
                         rho_esm_loso=r_emb))
        print(f"{s.protein[:24]:24s} {s.partner_a[:10]:10s}/{s.partner_b[:9]:9s} "
              f"{len(s):4d} {r_lik:+11.3f} {r_tri:+10.3f} {r_emb:+9.3f}")

    # pooled, on within-system z-scores so systems with wide ddG ranges do not dominate
    zy = np.concatenate([zscore(t) for t in targets])
    pool = dict(protein="POOLED (within-system z)", n=len(zy),
                rho_likelihood=spearman(np.concatenate([zscore(l) for l in lik]), zy),
                rho_trivial_loso=spearman(np.concatenate(triv_pred), zy),
                rho_esm_loso=spearman(np.concatenate(emb_pred), zy))
    rows.append(pool)
    print(f"\n{'POOLED (within-system z)':46s} {pool['n']:4d} "
          f"{pool['rho_likelihood']:+11.3f} {pool['rho_trivial_loso']:+10.3f} "
          f"{pool['rho_esm_loso']:+9.3f}")

    n_lik = sum(1 for r in rows[:-1] if r["rho_likelihood"] > 0)
    n_emb = sum(1 for r in rows[:-1] if r["rho_esm_loso"] > 0)
    print(f"\nsystems with positive rho: likelihood {n_lik}/{len(systems)}, "
          f"ESM probe {n_emb}/{len(systems)}")

    out = ROOT / args.out
    keys = sorted({k for r in rows for k in r})
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
