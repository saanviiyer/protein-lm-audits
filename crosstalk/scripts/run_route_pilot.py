#!/usr/bin/env python3
"""BEST-Route for protein language models: a pilot on data already collected.

Ding et al. [Ding2025best] route each query to a model chosen by predicted
difficulty, cutting cost up to 60% for under 1% quality loss. Routing needs a
difficulty predictor that runs before the expensive model does.

Sections 22 and 25 built exactly that for protein scoring: label-free signals,
computed from the small model and the sequence alone, that predict how well a
model will rank a mutational library. This asks whether those signals are good
enough to route.

Setup: 194 ProteinGym assays scored by ESM-2 150M and 650M. 650M is better on
average (+0.036 paired) but costs about 4.3x the parameters and roughly that in
compute. A router that sends only the assays that benefit would keep most of the
quality for a fraction of the cost. The router sees only 150M's own internals,
never 650M and never a label, which is the only setting in which routing is
possible at decision time.
"""
import csv, sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
from run_proxy_ladder import spearman
from run_readout_probe import ridge_fit, ridge_pred
from analyze_reliability_full import clusters, INTERNAL, NEFF

COST = 650.0 / 150.0     # parameter ratio, a stand-in for relative inference cost


def main():
    A = {r["dms_id"]: r for r in
         csv.DictReader((ROOT / "results/reliability_full_150M.csv").open())}
    B = {r["dms_id"]: r for r in
         csv.DictReader((ROOT / "results/reliability_forecast_full.csv").open())}
    ref = {r["DMS_id"]: r for r in
           csv.DictReader((ROOT / "data/proteingym_reference_v1.csv").open())}
    ids = sorted(set(A) & set(B) & set(ref))
    g = clusters(ids, {i: ref[i]["target_seq"] for i in ids})
    gl = [g[i] for i in ids]
    ya = np.array([float(A[i]["rho_esm"]) for i in ids])   # small-model skill
    yb = np.array([float(B[i]["rho_esm"]) for i in ids])   # large-model skill
    gain = yb - ya

    # every feature is computed from the SMALL model and the sequence alone
    F = {f: np.array([float(A[i][f]) for i in ids]) for f in INTERNAL}
    F["length"] = np.array([float(A[i]["length"]) for i in ids])
    F["n_variants"] = np.array([float(A[i]["n_variants"]) for i in ids])
    F["msa_neff"] = np.array([NEFF.get(ref[i].get("MSA_Neff_L_category", ""), np.nan)
                              for i in ids])
    fs = INTERNAL + ["length", "n_variants", "msa_neff"]
    X = np.column_stack([F[f] for f in fs])
    ok = np.isfinite(X).all(1)
    X, ya_, yb_, gain_ = X[ok], ya[ok], yb[ok], gain[ok]
    gg = [gl[i] for i in np.where(ok)[0]]
    print(f"{len(ya_)} assays, {len(set(gg))} protein clusters")
    print(f"150M mean skill {ya_.mean():+.4f}   650M {yb_.mean():+.4f}   "
          f"gain {gain_.mean():+.4f}\n")

    def loco_predict(target):
        pred = np.zeros(len(target))
        for c in set(gg):
            te = np.array([x == c for x in gg]); tr = ~te
            mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-8
            w = ridge_fit((X[tr] - mu) / sd, target[tr], 10.0)
            pred[te] = ridge_pred((X[te] - mu) / sd, w)
        return pred

    # two routing signals, both label-free and both leave-one-protein-out
    pred_small = loco_predict(ya_)      # "will the small model do badly here?"
    pred_gain = loco_predict(gain_)     # "will the large model help here?"
    print(f"router quality: rho(predicted small-model skill, actual) "
          f"{spearman(pred_small, ya_):+.3f}")
    print(f"                rho(predicted gain, actual gain)        "
          f"{spearman(pred_gain, gain_):+.3f}\n")

    def frontier(score, name):
        print(f"--- routing on {name} ---")
        print(f"{'sent to 650M':>13s} {'mean skill':>11s} {'of full-650M':>13s} "
              f"{'rel. cost':>10s}")
        order = np.argsort(-score)
        for frac in (0.0, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0):
            k = int(round(frac * len(order)))
            sel = np.zeros(len(order), bool); sel[order[:k]] = True
            skill = np.where(sel, yb_, ya_).mean()
            cost = (1.0 + frac * (COST - 1.0)) / COST     # relative to all-650M
            frac_of_max = (skill - ya_.mean()) / (yb_.mean() - ya_.mean())
            print(f"{frac:12.0%} {skill:+11.4f} {frac_of_max:12.1%} {cost:10.2f}")
        print()

    frontier(-pred_small, "predicted small-model weakness")
    frontier(pred_gain, "predicted gain from the larger model")
    rng = np.random.default_rng(0)
    frontier(rng.normal(size=len(ya_)), "random (control)")

    print("--- oracle ceiling, for reference only ---")
    order = np.argsort(-gain_)
    for frac in (0.1, 0.2, 0.3):
        k = int(round(frac * len(order)))
        sel = np.zeros(len(order), bool); sel[order[:k]] = True
        skill = np.where(sel, yb_, ya_).mean()
        print(f"{frac:12.0%} {skill:+11.4f} "
              f"{(skill - ya_.mean())/(yb_.mean()-ya_.mean()):12.1%}")


if __name__ == "__main__":
    main()
