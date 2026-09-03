#!/usr/bin/env python3
"""Selective prediction: what does abstaining on the forecast buy?

Sections 22 and 25 fit a label-free forecast of per-assay skill and grade it by
rank correlation. A rank correlation is not what a pipeline consumes. The
decision-time question is whether declining to use the model where the forecast
is lowest raises the quality of the assays it is still used on, and by how much
against declining at random.

This is the risk-coverage form of the same forecast. Leave-one-protein-out
predictions, clusters held out whole, and a random-abstention control at matched
coverage.
"""
import argparse, csv, sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
from analyze_reliability_full import clusters, loco, INTERNAL, NEFF

FS = INTERNAL + ["msa_neff_cat", "length", "n_variants"]


def forecast(path, ref):
    rows = [r for r in csv.DictReader((ROOT / path).open()) if r["dms_id"] in ref]
    ids = [r["dms_id"] for r in rows]
    g = clusters(ids, {i: ref[i]["target_seq"] for i in ids})
    gl = [g[i] for i in ids]
    Y = np.array([float(r["rho_esm"]) for r in rows])
    F = {f: np.array([float(r[f]) for r in rows]) for f in INTERNAL}
    F["length"] = np.array([float(r["length"]) for r in rows])
    F["n_variants"] = np.array([float(r["n_variants"]) for r in rows])
    F["msa_neff_cat"] = np.array(
        [NEFF.get(ref[i].get("MSA_Neff_L_category", ""), np.nan) for i in ids])
    X = np.column_stack([F[f] for f in FS])
    ok = np.isfinite(X).all(1)
    gg = [gl[i] for i in np.where(ok)[0]]
    return loco(X[ok], Y[ok], gg), Y[ok], len(set(gg))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--small", default="results/reliability_full_150M.csv")
    ap.add_argument("--large", default="results/reliability_forecast_full.csv")
    ap.add_argument("--draws", type=int, default=2000)
    ap.add_argument("--out", default="results/selective_forecast.csv")
    args = ap.parse_args()
    ref = {r["DMS_id"]: r for r in
           csv.DictReader((ROOT / "data/proteingym_reference_v1.csv").open())}
    rng = np.random.default_rng(0)
    out = []
    for tag, path in (("150M", args.small), ("650M", args.large)):
        pred, Y, ncl = forecast(path, ref)
        print(f"--- {tag}: {len(Y)} assays, {ncl} clusters ---")
        print(f"{'coverage':>9} {'mean skill':>11} {'random':>9} "
              f"{'frac skill<0.20':>16}")
        order = np.argsort(-pred)
        for cov in (1.0, 0.9, 0.8, 0.7, 0.6, 0.5):
            k = int(round(cov * len(Y)))
            sel = order[:k]
            rand = float(np.mean([Y[rng.permutation(len(Y))[:k]].mean()
                                  for _ in range(args.draws)]))
            print(f"{cov:9.0%} {Y[sel].mean():+11.4f} {rand:+9.4f} "
                  f"{(Y[sel] < 0.20).mean():16.3f}")
            out.append(dict(model=tag, coverage=cov, mean_skill=Y[sel].mean(),
                            random_control=rand,
                            frac_unreliable=(Y[sel] < 0.20).mean()))
        print()
    with (ROOT / args.out).open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0])); w.writeheader(); w.writerows(out)
    print(f"wrote {ROOT / args.out}")


if __name__ == "__main__":
    main()
