#!/usr/bin/env python3
"""Does the label-free reliability forecast hold at a larger model scale?

Section 22 fitted the forecast on ESM-2 150M because the machine was contended.
The 41-assay pilot in section 21 used 650M and found no significant forecast, so
a scale effect and a sample-size effect were confounded.

This compares the two sweeps on the assays scored by BOTH, paired within assay,
which removes the between-assay variance that dominates either sweep alone. The
comparison is valid on a partial 650M sweep, so it does not have to wait for the
last assay.
"""
import argparse, csv, sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
from run_proxy_ladder import spearman
from analyze_reliability_full import clusters, loco, INTERNAL, NEFF


def load(path):
    return {r["dms_id"]: r for r in csv.DictReader((ROOT / path).open())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--small", default="results/reliability_full_150M.csv")
    ap.add_argument("--large", default="results/reliability_forecast_full.csv")
    ap.add_argument("--perms", type=int, default=2000)
    args = ap.parse_args()

    A, B = load(args.small), load(args.large)
    ref = {r["DMS_id"]: r for r in
           csv.DictReader((ROOT / "data/proteingym_reference_v1.csv").open())}
    ids = sorted(set(A) & set(B) & set(ref))
    print(f"150M sweep {len(A)} assays, 650M sweep {len(B)}; "
          f"{len(ids)} scored by both\n")

    g = clusters(ids, {i: ref[i]["target_seq"] for i in ids})
    gl = [g[i] for i in ids]
    uniq = sorted(set(gl))
    ya = np.array([float(A[i]["rho_esm"]) for i in ids])
    yb = np.array([float(B[i]["rho_esm"]) for i in ids])
    print(f"{len(uniq)} independent protein clusters")

    print("\n--- is the larger model better, paired within assay? ---")
    d = yb - ya
    ci = 1.96 * d.std(ddof=1) / np.sqrt(len(d))
    print(f"  150M mean skill {ya.mean():+.4f}   650M mean skill {yb.mean():+.4f}")
    print(f"  paired difference {d.mean():+.4f}  95% CI [{d.mean()-ci:+.4f}, "
          f"{d.mean()+ci:+.4f}]   650M better on {int((d > 0).sum())}/{len(d)}")
    tax = np.array([ref[i]["taxon"] for i in ids])
    for t in ["Human", "Eukaryote", "Prokaryote", "Virus"]:
        m = tax == t
        if m.sum() >= 5:
            print(f"    {t:11s} n={int(m.sum()):3d}  150M {ya[m].mean():+.3f}  "
                  f"650M {yb[m].mean():+.3f}  diff {d[m].mean():+.3f}")

    rng = np.random.default_rng(0)
    def cperm(Y):
        vals = {c: Y[[i for i, x in enumerate(gl) if x == c]].mean() for c in uniq}
        order = rng.permutation(uniq)
        m = dict(zip(uniq, [vals[c] for c in order]))
        return np.array([m[c] for c in gl])

    print("\n--- does the forecast work at each scale, on the same assays? ---")
    for tag, src, Y in (("150M", A, ya), ("650M", B, yb)):
        F = {f: np.array([float(src[i][f]) for i in ids]) for f in INTERNAL}
        F["length"] = np.array([float(src[i]["length"]) for i in ids])
        F["n_variants"] = np.array([float(src[i]["n_variants"]) for i in ids])
        F["msa_neff_cat"] = np.array(
            [NEFF.get(ref[i].get("MSA_Neff_L_category", ""), np.nan) for i in ids])
        print(f"  [{tag}] strongest single signals:")
        for f in sorted(INTERNAL, key=lambda x: -abs(spearman(F[x], Y)))[:3]:
            print(f"      {f:20s} rho {spearman(F[f], Y):+.3f}")
        fs = INTERNAL + ["msa_neff_cat", "length", "n_variants"]
        X = np.column_stack([F[f] for f in fs])
        ok = np.isfinite(X).all(1)
        gg = [gl[i] for i in np.where(ok)[0]]
        pred = loco(X[ok], Y[ok], gg)
        rr = spearman(pred, Y[ok])
        null = np.array([spearman(loco(X[ok], yp, gg), yp)
                         for yp in (cperm(Y)[ok] for _ in range(max(args.perms // 10, 50)))])
        p = float((np.abs(null) >= abs(rr)).mean())
        print(f"      LOPO forecast rho {rr:+.3f}  p {p:.4f}  "
              f"MAE {np.mean(np.abs(pred - Y[ok])):.3f}  (skill SD {Y[ok].std(ddof=1):.3f})")
        bad = Y[ok] < 0.20
        k = int(bad.sum())
        if k:
            hit = int(bad[np.argsort(pred)[:k]].sum())
            print(f"      catches {hit}/{k} assays with skill < 0.20 "
                  f"(chance {k*k/len(pred):.1f})")


if __name__ == "__main__":
    main()
