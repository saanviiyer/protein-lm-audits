#!/usr/bin/env python3
"""Is the reliability forecast a virus detector wearing a forecast's clothes?

Skill is far lower on viral proteins than anywhere else, so a forecast fitted on
all 194 assays could be reading taxon and nothing more. Two checks settle it.
Fit and evaluate the forecast inside one taxon at a time, where no cross-taxon
contrast exists. And fit a taxon one-hot on its own, then append it to the real
signals to see whether it adds anything.

Clusters are held out whole in every fit, as everywhere else.
"""
import argparse, csv, sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
from run_proxy_ladder import spearman
from analyze_reliability_full import clusters, loco, INTERNAL, NEFF

FS = INTERNAL + ["msa_neff_cat", "length", "n_variants"]
TAXA = ["Human", "Eukaryote", "Prokaryote", "Virus"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--small", default="results/reliability_full_150M.csv")
    ap.add_argument("--large", default="results/reliability_forecast_full.csv")
    args = ap.parse_args()
    ref = {r["DMS_id"]: r for r in
           csv.DictReader((ROOT / "data/proteingym_reference_v1.csv").open())}

    for tag, path in (("150M", args.small), ("650M", args.large)):
        rows = [r for r in csv.DictReader((ROOT / path).open()) if r["dms_id"] in ref]
        ids = [r["dms_id"] for r in rows]
        g = clusters(ids, {i: ref[i]["target_seq"] for i in ids})
        gl = [g[i] for i in ids]
        Y = np.array([float(r["rho_esm"]) for r in rows])
        tax = np.array([ref[i]["taxon"] for i in ids])
        F = {f: np.array([float(r[f]) for r in rows]) for f in INTERNAL}
        F["length"] = np.array([float(r["length"]) for r in rows])
        F["n_variants"] = np.array([float(r["n_variants"]) for r in rows])
        F["msa_neff_cat"] = np.array(
            [NEFF.get(ref[i].get("MSA_Neff_L_category", ""), np.nan) for i in ids])
        X = np.column_stack([F[f] for f in FS])
        ok = np.isfinite(X).all(1)

        print(f"--- ESM-2 {tag} ---")
        print(f"{'restriction':22s} {'n':>4s} {'clusters':>9s} {'LOPO rho':>9s}")
        subsets = [("all assays", np.ones(len(Y), bool)),
                   ("non-viral only", tax != "Virus")] + \
                  [(f"{t} only", tax == t) for t in ("Human", "Prokaryote", "Eukaryote")]
        for name, mask in subsets:
            m = mask & ok
            gg = [gl[i] for i in np.where(m)[0]]
            pred = loco(X[m], Y[m], gg)
            print(f"{name:22s} {int(m.sum()):4d} {len(set(gg)):9d} "
                  f"{spearman(pred, Y[m]):+9.3f}")
        T = np.column_stack([(tax == t).astype(float) for t in TAXA])
        print(f"{'taxon one-hot alone':22s} {len(Y):4d} {len(set(gl)):9d} "
              f"{spearman(loco(T, Y, gl), Y):+9.3f}")
        gg = [gl[i] for i in np.where(ok)[0]]
        XT = np.column_stack([X, T])[ok]
        print(f"{'taxon + all signals':22s} {int(ok.sum()):4d} {len(set(gg)):9d} "
              f"{spearman(loco(XT, Y[ok], gg), Y[ok]):+9.3f}\n")


if __name__ == "__main__":
    main()
