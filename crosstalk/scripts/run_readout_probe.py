#!/usr/bin/env python3
"""Is specificity absent from the model, or only from the likelihood?

Every rung of the proxy ladder so far reads one number out of a model: a
likelihood. On this landscape that number is anti-predictive for protein LMs
(AUC 0.151 at ESM-2 650M) and uninformative for genomic LMs. Both are statements
about a *readout*, not about a *representation*, and the two can come apart: a
model can encode something it does not score.

This probes the representation directly. Frozen embeddings, no fine-tuning, a
linear head, against the same measured specificity margin.

The split is the whole experiment. Variants differ at three positions only, so a
random split lets a probe memorise which residue is good where, and any
representation -- including a one-hot vector -- will appear to work. Instead the
amino acid alphabet is partitioned, and a variant is held out if it contains ANY
residue from the held-out group. Every test variant therefore carries a residue
the probe never saw at any position.

That makes the trivial baseline informative rather than merely conventional:
one-hot encoding is at chance by construction on unseen residues, because the
feature was never active in training. A representation that beats it has
transferred something about amino acid chemistry. A representation that does not
is contributing nothing a lookup table would not.
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
from crosstalk.plm import variant_full
from run_proxy_ladder import auc, spearman

ALPHABET = "ACDEFGHIKLMNPQRSTVWY"


# ------------------------------------------------------------------ features

@torch.no_grad()
def esm_embeddings(variants, model_name, batch=32, progress=2000):
    from transformers import AutoModel, AutoTokenizer
    dev = glm._device()
    tok = AutoTokenizer.from_pretrained(model_name)
    mdl = AutoModel.from_pretrained(model_name).to(dev).eval()
    out = []
    for s in range(0, len(variants), batch):
        seqs = [variant_full(v) for v in variants[s:s + batch]]
        enc = tok(seqs, return_tensors="pt", padding=True).to(dev)
        h = mdl(**enc).last_hidden_state
        # mean over real residues, plus the three mutated positions themselves:
        # pooling alone washes out a 3-residue change in a 93-residue protein.
        m = enc["attention_mask"].unsqueeze(-1).float()
        pooled = (h * m).sum(1) / m.sum(1)
        sites = torch.cat([h[:, p] for p in MUT_POSITIONS], -1)   # +1 for <cls>, -1 for 1-based
        out.append(torch.cat([pooled, sites], -1).float().cpu().numpy())
        if progress and (s + batch) % progress < batch:
            print(f"    {min(s+batch, len(variants))}/{len(variants)}", flush=True)
    return np.concatenate(out)


@torch.no_grad()
def nt_embeddings(variants, model_name, wt_cds, batch=32, progress=2000):
    sc = glm.NTScorer(model_name)
    dev = sc.device
    tps = [sc.token_index(0, p)[0] for p in MUT_POSITIONS]
    out = []
    for s in range(0, len(variants), batch):
        seqs = [glm.variant_cds(v, wt_cds) for v in variants[s:s + batch]]
        enc = sc.tok(seqs, return_tensors="pt", padding=True).to(dev)
        h = sc.model(**enc, output_hidden_states=True).hidden_states[-1]
        m = enc["attention_mask"].unsqueeze(-1).float()
        pooled = (h * m).sum(1) / m.sum(1)
        sites = torch.cat([h[:, t + 1] for t in tps], -1)
        out.append(torch.cat([pooled, sites], -1).float().cpu().numpy())
        if progress and (s + batch) % progress < batch:
            print(f"    {min(s+batch, len(variants))}/{len(variants)}", flush=True)
    return np.concatenate(out)


def onehot(variants):
    idx = {a: i for i, a in enumerate(ALPHABET)}
    X = np.zeros((len(variants), 3 * 20))
    for i, v in enumerate(variants):
        for k, aa in enumerate(v):
            if aa in idx:
                X[i, k * 20 + idx[aa]] = 1.0
    return X


# --------------------------------------------------------------------- probe

def ridge_fit(X, y, lam):
    Xb = np.hstack([X, np.ones((len(X), 1))])
    A = Xb.T @ Xb + lam * np.eye(Xb.shape[1])
    A[-1, -1] -= lam                       # do not penalise the intercept
    return np.linalg.solve(A, Xb.T @ y)


def ridge_pred(X, w):
    return np.hstack([X, np.ones((len(X), 1))]) @ w


def residue_folds(variants, n_folds=5, seed=0):
    """Partition the alphabet; hold out any variant containing a held-out residue."""
    rng = np.random.default_rng(seed)
    perm = list(ALPHABET)
    rng.shuffle(perm)
    groups = [set(perm[i::n_folds]) for i in range(n_folds)]
    for g in groups:
        test = np.array([any(a in g for a in v) for v in variants])
        yield ~test, test


def random_folds(variants, n_folds=5, seed=0):
    rng = np.random.default_rng(seed)
    f = rng.integers(0, n_folds, len(variants))
    for k in range(n_folds):
        yield f != k, f == k


def evaluate(X, y, specific, mask, variants, split, lam, n_folds=5, seed=0):
    """Out-of-fold predictions, then the same metrics the ladder reports.

    Under the residue split a variant can be held out in more than one fold (it
    has three residues, each in some group), so predictions are averaged over the
    folds that held it out rather than overwritten by the last one.
    """
    tot = np.zeros(len(y))
    cnt = np.zeros(len(y))
    folds = residue_folds if split == "residue" else random_folds
    for tr, te in folds(variants, n_folds, seed):
        if tr.sum() < 50 or te.sum() == 0:
            continue
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-8
        w = ridge_fit((X[tr] - mu) / sd, y[tr], lam)
        tot[te] += ridge_pred((X[te] - mu) / sd, w)
        cnt[te] += 1
    pred = np.where(cnt > 0, tot / np.maximum(cnt, 1), np.nan)
    ok = np.isfinite(pred)
    sel = ok & mask
    return dict(rho_margin=spearman(pred[ok], y[ok]),
                auc=auc(pred[sel], specific[sel]),
                coverage=float(ok.mean()), n_disc=int(sel.sum()),
                mean_train_frac=float(np.mean([tr.mean() for tr, _ in
                                               folds(variants, n_folds, seed)])))


def within_background_eval(X, y, variants, lam, n_folds=5, seed=0):
    """The split that actually isolates unseen chemistry.

    Holding out a variant because it contains an unseen residue is not enough:
    it still carries two SEEN residues, and an additive lookup table predicts it
    from those. That is why one-hot scored 0.952 on the residue split -- not
    because it generalised, but because two thirds of each test variant was
    familiar.

    This removes that path. Test variants are grouped by (mutated position, the
    two residues at the other positions), so within a group every variant shares
    an identical background and differs ONLY at the held-out position, where the
    residue was never seen in training. Ranking within a group therefore requires
    knowing something about the unseen amino acid itself.

    One-hot is exactly uninformative here by construction: the held-out residues'
    weights were never updated, so every member of a group receives the same
    prediction. Any method that beats zero has transferred chemistry rather than
    memorised identity.
    """
    rng = np.random.default_rng(seed)
    perm = list(ALPHABET); rng.shuffle(perm)
    groups = [set(perm[i::n_folds]) for i in range(n_folds)]

    rhos, weights, n_groups = [], [], 0
    for g in groups:
        contains = np.array([[a in g for a in v] for v in variants])
        train = ~contains.any(1)
        if train.sum() < 50:
            continue
        mu, sd = X[train].mean(0), X[train].std(0) + 1e-8
        w = ridge_fit((X[train] - mu) / sd, y[train], lam)
        for k in range(3):
            # exactly one unseen residue, and it is at position k
            sel = contains[:, k] & (contains.sum(1) == 1)
            if sel.sum() < 3:
                continue
            idx = np.where(sel)[0]
            bg = {}
            for i in idx:
                key = (k, variants[i][:k] + "_" + variants[i][k + 1:])
                bg.setdefault(key, []).append(i)
            for key, members in bg.items():
                if len(members) < 3:
                    continue
                m = np.array(members)
                pred = ridge_pred((X[m] - mu) / sd, w)
                if np.std(pred) < 1e-9 or np.std(y[m]) < 1e-9:
                    rho = 0.0            # constant prediction carries no ranking
                else:
                    rho = spearman(pred, y[m])
                if np.isfinite(rho):
                    rhos.append(rho); weights.append(len(m)); n_groups += 1
    if not rhos:
        return dict(rho_within_bg=np.nan, n_groups=0, n_pairs=0)
    rhos, weights = np.array(rhos), np.array(weights, float)
    return dict(rho_within_bg=float(np.average(rhos, weights=weights)),
                rho_within_bg_unweighted=float(np.mean(rhos)),
                n_groups=int(n_groups), n_pairs=int(weights.sum()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--esm", default="facebook/esm2_t33_650M_UR50D")
    ap.add_argument("--nt", default="InstaDeepAI/nucleotide-transformer-v2-50m-multi-species")
    ap.add_argument("--lam", type=float, default=100.0)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--out", default="results/readout_probe.csv")
    ap.add_argument("--cache", default="results/probe_features.npz")
    args = ap.parse_args()

    L = load_pard3()
    variants = L.seqs
    w3, w2 = L.F[:, 0], L.F[:, 1]
    y = w3 - w2
    specific = (w3 >= 0.8) & (w2 <= 0.2)
    promisc = (w3 >= 0.8) & (w2 >= 0.6)
    mask = specific | promisc
    lab = specific[mask]
    print(f"{len(variants)} variants; discrimination set {int(mask.sum())}", flush=True)

    cache = ROOT / args.cache
    if cache.exists():
        z = np.load(cache)
        feats = {k: z[k] for k in z.files if k != "variants"}
        print(f"loaded cached features: {list(feats)}")
    else:
        wt_cds = glm.load_cds()["ParD3"]["cds"]
        feats = {}
        print(f"embedding with {args.esm} ...", flush=True)
        feats["esm"] = esm_embeddings(variants, args.esm)
        print(f"embedding with {args.nt} ...", flush=True)
        feats["nt"] = nt_embeddings(variants, args.nt, wt_cds)
        np.savez_compressed(cache, variants=np.array(variants), **feats)
        print(f"cached -> {cache}")
    feats["onehot"] = onehot(variants)

    rows = []
    print("\n--- within-background ranking of UNSEEN residues "
          "(one-hot is uninformative by construction) ---")
    for name, X in feats.items():
        r = within_background_eval(X, y, variants, args.lam, args.folds)
        r.update(features=name, split="within_background", dim=X.shape[1])
        rows.append(r)
        print(f"{'within_bg':16s} {name:8s} dim {X.shape[1]:5d}  "
              f"rho {r['rho_within_bg']:+.3f}  "
              f"({r['n_groups']} groups, {r['n_pairs']} variants)", flush=True)

    print("\n--- whole-variant splits (note: residue split leaks via the two "
          "seen positions) ---")
    for split in ("residue", "random"):
        for name, X in feats.items():
            r = evaluate(X, y, specific, mask, variants, split, args.lam, args.folds)
            r.update(features=name, split=split, dim=X.shape[1])
            rows.append(r)
            print(f"{split:8s} {name:8s} dim {X.shape[1]:5d}  "
                  f"rho_margin {r['rho_margin']:+.3f}  AUC {r['auc']:.3f}  "
                  f"coverage {r['coverage']:.2f}", flush=True)

    out = ROOT / args.out
    keys = sorted({k for r in rows for k in r})
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
