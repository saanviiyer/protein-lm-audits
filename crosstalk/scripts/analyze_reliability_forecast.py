#!/usr/bin/env python3
"""Grade the label-free reliability forecast, counting correlated assays once.

ProteinGym contains several assays per protein (three for BLAT_ECOLX, two for
CP2C9, PTEN and RL401) and two influenza nucleoprotein strains that are 94%
identical to each other. Treating 41 assays as 41 independent observations would
make every p-value optimistic, because a signal only has to work on one protein
to appear to work several times.

Assays are therefore clustered: same protein prefix, or target sequences of equal
length that are at least 70% identical. Permutations shuffle whole clusters, and
the forecast is scored leave-one-CLUSTER-out, so a protein never appears in both
training and test.
"""
import csv, sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
from run_proxy_ladder import spearman
from run_readout_probe import ridge_fit, ridge_pred
from run_dms_transfer import GAUNTLET
from run_reliability_forecast import FEATURES


def clusters(ids, seqs):
    lab = {i: i.rsplit("_", 2)[0] for i in ids}
    keys = sorted(set(lab.values()))
    merged = {k: k for k in keys}
    for a in range(len(keys)):
        for b in range(a + 1, len(keys)):
            sa = [seqs[i] for i in ids if lab[i] == keys[a]][0]
            sb = [seqs[i] for i in ids if lab[i] == keys[b]][0]
            if len(sa) == len(sb) and sum(x == y for x, y in zip(sa, sb)) / len(sa) >= 0.70:
                merged[keys[b]] = merged[keys[a]]
    return {i: merged[lab[i]] for i in ids}


def loco(X, Y, g, lam=10.0):
    pred = np.zeros(len(Y))
    for c in set(g):
        te = np.array([x == c for x in g]); tr = ~te
        if tr.sum() < 4:
            pred[te] = Y[tr].mean() if tr.sum() else Y.mean(); continue
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-8
        w = ridge_fit((X[tr] - mu) / sd, Y[tr], lam)
        pred[te] = ridge_pred((X[te] - mu) / sd, w)
    return pred


def main():
    rows = list(csv.DictReader((ROOT / "results/reliability_forecast.csv").open()))
    ref = {r["DMS_id"]: r["target_seq"] for r in
           csv.DictReader((GAUNTLET / "reference.csv").open())}
    ids = [r["dms_id"] for r in rows]
    g = clusters(ids, {i: ref[i] for i in ids})
    gl = [g[i] for i in ids]
    Y = np.array([float(r["rho_esm"]) for r in rows])
    X = np.array([[float(r[f]) for f in FEATURES] for r in rows])
    uniq = sorted(set(gl))
    print(f"{len(rows)} assays in {len(uniq)} independent protein clusters")
    print(f"observed skill {Y.min():+.3f} to {Y.max():+.3f}, SD {Y.std(ddof=1):.3f}\n")

    rng = np.random.default_rng(0)

    def cluster_perm(Y):
        """Shuffle skill between clusters, keeping assays of one protein together."""
        vals = {c: Y[[i for i, x in enumerate(gl) if x == c]].mean() for c in uniq}
        order = rng.permutation(uniq)
        m = dict(zip(uniq, [vals[c] for c in order]))
        return np.array([m[c] for c in gl])

    print("--- univariate, cluster-permuted null ---")
    print(f"  {'signal':18s} {'rho':>7s} {'p':>8s}")
    for j, f in enumerate(FEATURES):
        rr = spearman(X[:, j], Y)
        null = np.array([spearman(X[:, j], cluster_perm(Y)) for _ in range(4000)])
        p = float((np.abs(null) >= abs(rr)).mean())
        star = "  <-" if p < 0.05 else ""
        print(f"  {f:18s} {rr:+7.3f} {p:8.4f}{star}")

    print("\n--- leave-one-protein-out forecast (all signals) ---")
    pred = loco(X, Y, gl)
    rr = spearman(pred, Y)
    null = np.array([spearman(loco(X, yp, gl), yp) for yp in
                     (cluster_perm(Y) for _ in range(400))])
    print(f"  forecast vs actual rho {rr:+.3f}   cluster-permuted p = "
          f"{float((np.abs(null) >= abs(rr)).mean()):.4f}")
    print(f"  MAE {np.mean(np.abs(pred - Y)):.3f} vs skill SD {Y.std(ddof=1):.3f}")

    print("\n--- the operational test: flag the assays where the model is useless ---")
    for thr in (0.10, 0.20, 0.30):
        bad = Y < thr
        if bad.sum() == 0:
            continue
        k = int(bad.sum())
        flagged = np.argsort(pred)[:k]
        hit = int(bad[flagged].sum())
        print(f"  skill < {thr:.2f}: {k} assays; forecast's {k} lowest catches {hit}"
              f"  (chance {k * k / len(Y):.1f})")

    print("\n--- single best label-free rule ---")
    j = int(np.argmax([abs(spearman(X[:, i], Y)) for i in range(len(FEATURES))]))
    print(f"  {FEATURES[j]}: rho {spearman(X[:, j], Y):+.3f}")
    lo = X[:, j] <= np.quantile(X[:, j], 0.25)
    print(f"  bottom-quartile {FEATURES[j]}: mean skill {Y[lo].mean():+.3f} "
          f"vs {Y[~lo].mean():+.3f} for the rest")


if __name__ == "__main__":
    main()
