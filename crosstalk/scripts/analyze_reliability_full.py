#!/usr/bin/env python3
"""Grade the label-free forecast on 194 assays, against the right baseline.

The pilot compared model-internal signals to nothing. The obvious competitor is
homolog depth: a protein with thousands of relatives is one an evolutionary model
should handle, and one with few is not. ProteinGym ships that as
MSA_Neff_L_category, computed from an alignment with no DMS labels involved, so it
is a legitimate decision-time predictor and the baseline any model-internal signal
has to beat.

The question is therefore not whether model internals predict skill. It is
whether they predict skill that homolog depth does not already explain.

Assays are clustered by protein prefix and by sequence identity, permutations
shuffle whole clusters, and forecasts are scored leave-one-cluster-out.
"""
import argparse, csv, sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
from run_proxy_ladder import spearman
from run_readout_probe import ridge_fit, ridge_pred

INTERNAL = ["score_dispersion", "wt_agreement", "mean_wt_logprob",
            "mean_entropy", "frac_confident"]
NEFF = {"Low": 0.0, "Medium": 1.0, "High": 2.0}


def clusters(ids, seqs):
    lab = {i: i.rsplit("_", 2)[0] for i in ids}
    keys = sorted(set(lab.values()))
    rep = {k: [seqs[i] for i in ids if lab[i] == k][0] for k in keys}
    parent = {k: k for k in keys}
    def find(a):
        while parent[a] != a: a = parent[a]
        return a
    for a in range(len(keys)):
        for b in range(a + 1, len(keys)):
            sa, sb = rep[keys[a]], rep[keys[b]]
            if len(sa) == len(sb) and sa and \
               sum(x == y for x, y in zip(sa, sb)) / len(sa) >= 0.70:
                parent[find(keys[b])] = find(keys[a])
    return {i: find(lab[i]) for i in ids}


def loco(X, Y, g, lam=10.0):
    pred = np.zeros(len(Y))
    for c in set(g):
        te = np.array([x == c for x in g]); tr = ~te
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-8
        w = ridge_fit((X[tr] - mu) / sd, Y[tr], lam)
        pred[te] = ridge_pred((X[te] - mu) / sd, w)
    return pred


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="results/reliability_forecast_full.csv")
    ap.add_argument("--perms", type=int, default=3000)
    args = ap.parse_args()
    rows = list(csv.DictReader((ROOT / args.input).open()))
    ref = {r["DMS_id"]: r for r in
           csv.DictReader((ROOT / "data/proteingym_reference_v1.csv").open())}
    rows = [r for r in rows if r["dms_id"] in ref]
    ids = [r["dms_id"] for r in rows]
    g = clusters(ids, {i: ref[i]["target_seq"] for i in ids})
    gl = [g[i] for i in ids]
    Y = np.array([float(r["rho_esm"]) for r in rows])
    uniq = sorted(set(gl))
    print(f"{len(rows)} assays in {len(uniq)} independent protein clusters")
    print(f"skill {Y.min():+.3f} to {Y.max():+.3f}, mean {Y.mean():+.3f}, "
          f"SD {Y.std(ddof=1):.3f}\n")

    taxon = np.array([ref[i]["taxon"] for i in ids])
    neffc = np.array([ref[i].get("MSA_Neff_L_category", "") for i in ids])

    print("--- skill by taxon ---")
    for t in ["Human", "Eukaryote", "Prokaryote", "Virus"]:
        m = taxon == t
        if m.sum():
            print(f"  {t:11s} n={int(m.sum()):3d}  mean skill {Y[m].mean():+.3f}  "
                  f"median {np.median(Y[m]):+.3f}  below 0.20: "
                  f"{int((Y[m] < 0.20).sum())}")

    print("\n--- skill by homolog depth (MSA Neff/L) ---")
    for t in ["Low", "Medium", "High"]:
        m = neffc == t
        if m.sum():
            print(f"  {t:11s} n={int(m.sum()):3d}  mean skill {Y[m].mean():+.3f}  "
                  f"below 0.20: {int((Y[m] < 0.20).sum())}")

    rng = np.random.default_rng(0)
    def cperm(Y):
        vals = {c: Y[[i for i, x in enumerate(gl) if x == c]].mean() for c in uniq}
        order = rng.permutation(uniq)
        m = dict(zip(uniq, [vals[c] for c in order]))
        return np.array([m[c] for c in gl])

    print("\n--- univariate, cluster-permuted null ---")
    feats = {f: np.array([float(r[f]) for r in rows]) for f in INTERNAL}
    feats["length"] = np.array([float(r["length"]) for r in rows])
    feats["n_variants"] = np.array([float(r["n_variants"]) for r in rows])
    feats["msa_neff_cat"] = np.array([NEFF.get(x, np.nan) for x in neffc])
    print(f"  {'signal':18s} {'rho':>7s} {'p':>8s}")
    for f, v in feats.items():
        ok = np.isfinite(v)
        rr = spearman(v[ok], Y[ok])
        null = np.array([spearman(v[ok], cperm(Y)[ok]) for _ in range(args.perms)])
        p = float((np.abs(null) >= abs(rr)).mean())
        print(f"  {f:18s} {rr:+7.3f} {p:8.4f}{'  <-' if p < 0.05 else ''}")

    print("\n--- nested forecasts, leave-one-protein-out ---")
    ok = np.isfinite(feats["msa_neff_cat"])
    sets = {
        "homolog depth only": ["msa_neff_cat"],
        "model internals only": INTERNAL,
        "both": INTERNAL + ["msa_neff_cat"],
        "internals + depth + size": INTERNAL + ["msa_neff_cat", "length", "n_variants"],
    }
    for name, fs in sets.items():
        X = np.column_stack([feats[f] for f in fs])[ok]
        yy, gg = Y[ok], [gl[i] for i in np.where(ok)[0]]
        pred = loco(X, yy, gg)
        rr = spearman(pred, yy)
        null = np.array([spearman(loco(X, yp, gg), yp)
                         for yp in (cperm(Y)[ok] for _ in range(max(args.perms//10, 50)))])
        p = float((np.abs(null) >= abs(rr)).mean())
        mae = float(np.mean(np.abs(pred - yy)))
        print(f"  {name:26s} rho {rr:+.3f}  p {p:.4f}  MAE {mae:.3f}"
              f"  (skill SD {yy.std(ddof=1):.3f})")
        if name == "internals + depth + size":
            best_pred, best_y = pred, yy

    print("\n--- operational: catching the assays where the model is useless ---")
    for thr in (0.10, 0.20, 0.30):
        bad = best_y < thr
        k = int(bad.sum())
        if k == 0: continue
        flagged = np.argsort(best_pred)[:k]
        hit = int(bad[flagged].sum())
        print(f"  skill < {thr:.2f}: {k:3d} assays; the {k} lowest forecasts catch "
              f"{hit:3d}  (chance {k*k/len(best_y):.1f}, precision {hit/k:.2f})")

    print("\n--- calibration of the full forecast ---")
    q = np.quantile(best_pred, [0, .25, .5, .75, 1.0])
    for i in range(4):
        m = (best_pred >= q[i]) & (best_pred <= q[i+1])
        print(f"  forecast quartile {i+1}: predicted {best_pred[m].mean():+.3f}  "
              f"actual {best_y[m].mean():+.3f}  n={int(m.sum())}")


if __name__ == "__main__":
    main()
