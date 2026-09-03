#!/usr/bin/env python3
"""What, exactly, transfers from a genomic model to protein-level function?

Section 14 found that a genomic LM's *likelihood* of a coding sequence is at
chance for binding specificity, while section 15 found its *representation*
reaches +0.088 on unseen residues. That pair is suggestive but thin: one
landscape, one number, and no mechanism.

Three things are measured here.

  ALIGNMENT      How much of the protein model's geometry is recoverable from the
                 DNA model's, for the identical molecules? Linear CKA plus an
                 explicit cross-modal ridge map, against a one-hot reference so
                 "they agree" cannot just mean "both encode which residue is
                 where".

  CODON INVARIANCE  A variance decomposition. Synonymous encodings hold the
                 protein fixed and change only the DNA, so the fraction of
                 variance explained by amino-acid identity is a direct measure of
                 how protein-level a quantity is. Computed identically for the
                 scalar likelihood and for the representation, which makes them
                 comparable on one axis. A protein model is invariant here by
                 construction; the question is how close the DNA model gets.

  CAUSAL CONTROLS  The falsification. Two manipulations move in opposite
                 directions:
                   synonymous scramble -- protein IDENTICAL, DNA maximally changed
                   frameshift          -- DNA nearly identical, protein DESTROYED
                 If the DNA model's representation is genuinely about the protein,
                 a probe trained on canonical encodings must survive the scramble
                 and fail under the frameshift. If it tracks nucleotide statistics
                 instead, the pattern reverses. Nothing about "it correlates"
                 distinguishes these; the manipulations do.
"""
import argparse, csv, sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))

from crosstalk.boltz import MUT_POSITIONS, PARD3
from crosstalk.landscape import load_pard3
from crosstalk import glm
from run_proxy_ladder import auc, spearman
from run_readout_probe import (ALPHABET, onehot, ridge_fit, ridge_pred,
                               within_background_eval)

COMP = str.maketrans("ACGT", "TGCA")


# ------------------------------------------------------------------ alignment

def linear_cka(X, Y):
    """Linear CKA between two centred feature matrices over the same rows."""
    X = X - X.mean(0); Y = Y - Y.mean(0)
    xty = X.T @ Y
    num = float((xty ** 2).sum())
    den = float(np.sqrt(((X.T @ X) ** 2).sum()) * np.sqrt(((Y.T @ Y) ** 2).sum()))
    return num / den if den > 0 else np.nan


def cross_modal_map(X, Y, variants, lam, n_folds=5, seed=0):
    """Ridge from X-space to Y-space, scored by held-out R^2 on unseen residues.

    The split holds out whole amino acids, so the map is never fitted on any
    variant containing a residue it is tested on. A high R^2 then means the two
    representations agree about residues neither pairing was fitted for.
    """
    rng = np.random.default_rng(seed)
    perm = list(ALPHABET); rng.shuffle(perm)
    groups = [set(perm[i::n_folds]) for i in range(n_folds)]
    num, den = 0.0, 0.0
    for g in groups:
        test = np.array([any(a in g for a in v) for v in variants])
        tr = ~test
        if tr.sum() < 50 or test.sum() == 0:
            continue
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-8
        ym = Y[tr].mean(0)
        W = ridge_fit((X[tr] - mu) / sd, Y[tr] - ym, lam)
        pred = ridge_pred((X[test] - mu) / sd, W) + ym
        num += float(((Y[test] - pred) ** 2).sum())
        den += float(((Y[test] - ym) ** 2).sum())
    return 1.0 - num / den if den > 0 else np.nan


# --------------------------------------------------------- codon invariance

def variance_decomposition(values):
    """values: list over variants of (S_v, d) arrays. Returns eta^2.

    eta^2 = fraction of total variance explained by which variant it is, i.e. by
    the amino acid sequence. The remainder is synonymous codon choice, which
    cannot move the measured label.
    """
    allv = np.concatenate(values, 0)
    grand = allv.mean(0)
    tot = float(((allv - grand) ** 2).sum())
    within = float(sum(((v - v.mean(0)) ** 2).sum() for v in values))
    return 1.0 - within / tot if tot > 0 else np.nan


@torch.no_grad()
def nt_embed(scorer, seqs, batch=32):
    reps = []
    for i in range(0, len(seqs), batch):
        enc = scorer.tok(seqs[i:i + batch], return_tensors="pt",
                         padding=True).to(scorer.device)
        h = scorer.model(**enc, output_hidden_states=True).hidden_states[-1]
        m = enc["attention_mask"].unsqueeze(-1).float()
        reps.append(((h * m).sum(1) / m.sum(1)).float().cpu().numpy())
    return np.concatenate(reps)


# ------------------------------------------------------- sequence manipulations

def synonymous_scramble(variant, wt_cds, rng):
    """Same protein, every codon replaced by a synonymous alternative where one exists."""
    seq = glm.variant_cds(variant, wt_cds)
    out = []
    for i in range(0, len(seq) - 2, 3):
        c = seq[i:i + 3]
        aa = glm.CODON_TABLE.get(c)
        alts = [x for x in glm.SYNONYMOUS.get(aa, [c]) if x != c]
        out.append(alts[rng.integers(len(alts))] if alts else c)
    return "".join(out)


def random_mutant_codon(variant, wt_cds, rng):
    """Wild-type background kept; the SUBSTITUTED codon chosen at random.

    synonymous_scramble replaces every codon, which removes the leak but also
    adds background noise, so a drop there is ambiguous. This changes only the
    codon the leak runs through: the amino acid no longer determines the
    nucleotides, and nothing else moves. Whatever survives here is transfer that
    does not depend on the encoding rule.
    """
    return glm.variant_cds(variant, wt_cds, policy="uniform", rng=rng)


def frameshift(variant, wt_cds, k=1):
    """Nucleotide content almost unchanged; every downstream codon destroyed."""
    seq = glm.variant_cds(variant, wt_cds)
    return seq[k:] + seq[:k]


def revcomp(variant, wt_cds):
    return glm.variant_cds(variant, wt_cds).translate(COMP)[::-1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nt", default="InstaDeepAI/nucleotide-transformer-v2-50m-multi-species")
    ap.add_argument("--lam", type=float, default=100.0)
    ap.add_argument("--syn-variants", type=int, default=80)
    ap.add_argument("--syn-n", type=int, default=8)
    ap.add_argument("--features", default="results/probe_features.npz")
    ap.add_argument("--out", default="results/transfer_analysis.csv")
    args = ap.parse_args()

    L = load_pard3()
    variants = L.seqs
    w3, w2 = L.F[:, 0], L.F[:, 1]
    y = w3 - w2
    rows = []

    z = np.load(ROOT / args.features)
    assert list(z["variants"]) == list(variants), "cached features are for another landscape"
    esm, nt = z["esm"], z["nt"]
    oh = onehot(variants)

    # ---------------------------------------------------------- 1. alignment
    print("=== 1. representational alignment (same molecules, two modalities) ===")
    pairs = [("nt", nt, "esm", esm), ("onehot", oh, "esm", esm),
             ("onehot", oh, "nt", nt)]
    for an, A, bn, B in pairs:
        c = linear_cka(A, B)
        r2 = cross_modal_map(A, B, variants, args.lam)
        print(f"  {an:7s} -> {bn:5s}   CKA {c:.3f}   held-out R^2 {r2:+.3f}")
        rows.append(dict(stage="alignment", a=an, b=bn, cka=c, r2=r2))

    # ------------------------------------------------- 2. codon invariance
    print("\n=== 2. codon invariance: how protein-level is each quantity? ===")
    wt_cds = glm.load_cds()["ParD3"]["cds"]
    rng = np.random.default_rng(0)
    disc = np.where((w3 >= 0.8) & ((w2 <= 0.2) | (w2 >= 0.6)))[0]
    pick = rng.choice(disc, size=min(args.syn_variants, len(disc)), replace=False)
    sc = glm.NTScorer(args.nt)
    ens_emb, ens_lik, used = [], [], []
    for vi in pick:
        ens = glm.synonymous_ensemble(variants[vi], wt_cds, args.syn_n,
                                      seed=int(vi), policy="usage")
        if len(ens) < 2:
            continue
        ens_emb.append(nt_embed(sc, ens))
        ens_lik.append(sc.pseudo_likelihood(ens).reshape(-1, 1))
        used.append(vi)
    eta_emb = variance_decomposition(ens_emb)
    eta_lik = variance_decomposition(ens_lik)
    print(f"  variants {len(used)}, mean ensemble {np.mean([len(e) for e in ens_emb]):.1f}")
    print(f"  eta^2 explained by amino-acid identity:")
    print(f"    DNA model LIKELIHOOD    : {eta_lik:.3f}")
    print(f"    DNA model REPRESENTATION: {eta_emb:.3f}")
    print(f"    protein model (any)     : 1.000  (invariant by construction)")
    rows.append(dict(stage="codon_invariance", a="nt_likelihood", eta2=eta_lik,
                     n_variants=len(used)))
    rows.append(dict(stage="codon_invariance", a="nt_representation", eta2=eta_emb,
                     n_variants=len(used)))

    # --------------------------------------------- 3. causal controls
    print("\n=== 3. causal controls: does the representation track the protein? ===")
    rng = np.random.default_rng(1)
    conditions = {
        "canonical": lambda v: glm.variant_cds(v, wt_cds),
        "synonymous_scramble": lambda v: synonymous_scramble(v, wt_cds, rng),
        "random_mutant_codon": lambda v: random_mutant_codon(v, wt_cds, rng),
        "frameshift+1": lambda v: frameshift(v, wt_cds, 1),
        "reverse_complement": lambda v: revcomp(v, wt_cds),
    }
    for name, build in conditions.items():
        seqs = [build(v) for v in variants]
        X = nt_embed(sc, seqs)
        r = within_background_eval(X, y, variants, args.lam)
        print(f"  {name:20s} rho_within_bg {r['rho_within_bg']:+.3f}  "
              f"({r['n_groups']} groups)", flush=True)
        rows.append(dict(stage="causal_control", a=name,
                         rho_within_bg=r["rho_within_bg"], n_groups=r["n_groups"]))

    out = ROOT / args.out
    keys = sorted({k for r in rows for k in r})
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
