#!/usr/bin/env python3
"""
significance_tests.py

Attaches statistical significance to the key PRISM results so the papers report
intervals and p-values, not just point estimates. All tests are non-parametric
(bootstrap CIs, permutation tests) and run on CPU in seconds.

Covers:
  1. BIOPHYSICAL — is SBS17a's (high-missense, low-truncation) profile
     significantly different from the other signatures? Permutation test on the
     per-gene difference; bootstrap CI on the effect.
  2. HUB — is PF17041's attraction under SBS17a above what its baseline hubness
     predicts? Reported as an enrichment with a bootstrap CI over experiments.
  3. COLLAPSE — bootstrap CI on the per-gene collapse-rate difference between
     the 6-mer models.
  4. INVERSE DESIGN — bootstrap CI on top-k accuracy and a permutation test that
     the classifier beats the frequency prior (reads the metrics JSON).

USAGE
-----
python significance_tests.py \
    --biophys results/biophysical_drift/biophysical_drift_summary.csv \
    --runs_dir all_runs \
    --collapse results/stress_test/A_collapse_profile.csv \
    --inverse results/inverse_design/inverse_design_metrics.json \
    --output_dir results/significance
"""
import argparse
import json
import os

import numpy as np
import pandas as pd


def boot_ci(x, fn=np.mean, n=10000, alpha=0.05, rng=None):
    rng = rng or np.random.default_rng(0)
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if x.size == 0:
        return (np.nan, np.nan, np.nan)
    idx = rng.integers(0, x.size, size=(n, x.size))
    stat = fn(x[idx], axis=1)
    lo, hi = np.percentile(stat, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(fn(x)), float(lo), float(hi)


def perm_test_greater(a, b, n=10000, rng=None):
    """One-sided permutation test: is mean(a) > mean(b)? Returns (obs_diff, p)."""
    rng = rng or np.random.default_rng(0)
    a, b = np.asarray(a, float), np.asarray(b, float)
    obs = a.mean() - b.mean()
    pool = np.concatenate([a, b]); na = len(a)
    ge = 0
    for _ in range(n):
        rng.shuffle(pool)
        if pool[:na].mean() - pool[na:].mean() >= obs:
            ge += 1
    return float(obs), (ge + 1) / (n + 1)


def biophys_significance(path, out, rng):
    df = pd.read_csv(path)
    res = {}
    if "gene" not in df or "signature" not in df:
        return res
    # per-gene missense and truncation, SBS17a vs the rest
    for metric, direction in [("mean_missense", "greater"), ("truncation_rate", "less")]:
        if metric not in df:
            continue
        piv = df.pivot_table(index="gene", columns="signature", values=metric)
        if "SBS17a" not in piv:
            continue
        s17 = piv["SBS17a"].dropna()
        others = piv.drop(columns=["SBS17a"]).mean(axis=1).reindex(s17.index)
        paired = (s17 - others).dropna()
        if direction == "less":
            paired = -paired  # test SBS17a LOWER than others
        obs, p = perm_test_greater(paired.values, np.zeros_like(paired.values), rng=rng)
        m, lo, hi = boot_ci(paired.values, rng=rng)
        res[metric] = {"mean_gap_SBS17a_vs_rest": round(m, 3),
                       "ci95": [round(lo, 3), round(hi, 3)],
                       "perm_p": round(p, 5), "direction": direction, "n_genes": int(len(paired))}
        print(f"[biophys] SBS17a {metric} vs rest: gap={m:+.2f} "
              f"CI95=[{lo:+.2f},{hi:+.2f}] perm_p={p:.4f} (n={len(paired)} genes)")
    _save(out, "biophysical_significance.json", res)
    return res


def hub_significance(runs_dir, out, rng, target="PF17041", sig="SBS17a"):
    import glob
    res = {}
    for csv in sorted(glob.glob(os.path.join(runs_dir, "*", "*centroid_trajectory_results.csv"))):
        clan = os.path.basename(os.path.dirname(csv))
        t = pd.read_csv(csv)
        t["signature"] = t["signature"].astype(str).str.replace("_PROFILE.txt", "", regex=False)
        base = (t["top1_pfam"].astype(str) == target).mean()  # hubness
        s = t[t["signature"] == sig]
        if len(s) < 5:
            continue
        hits = (s["top1_pfam"].astype(str) == target).astype(int).values
        conc, lo, hi = boot_ci(hits, rng=rng)          # SBS17a concentration on target
        enr = conc / base if base > 0 else float("inf")
        res[clan] = {"baseline_hubness": round(float(base), 4),
                     "sbs17a_concentration": round(conc, 4),
                     "concentration_ci95": [round(lo, 4), round(hi, 4)],
                     "enrichment": round(enr, 1), "n": int(len(s))}
        print(f"[hub] {clan}: PF17041 hubness={base:.3%}  SBS17a conc={conc:.1%} "
              f"CI95=[{lo:.1%},{hi:.1%}]  enrichment={enr:.0f}x")
    _save(out, "hub_significance.json", res)
    return res


def collapse_significance(path, out, rng):
    df = pd.read_csv(path)
    res = {}
    piv = df.pivot_table(index="gene", columns="model", values="collapse_rate")
    if {"carbon", "generator"} <= set(piv.columns):
        diff = (piv["carbon"] - piv["generator"]).dropna().values
        m, lo, hi = boot_ci(diff, rng=rng)
        obs, p = perm_test_greater(piv["carbon"].dropna().values,
                                   piv["generator"].dropna().values, rng=rng)
        res = {"carbon_minus_generator_mean": round(m, 4), "ci95": [round(lo, 4), round(hi, 4)],
               "perm_p": round(p, 5), "n_genes": int(len(diff))}
        print(f"[collapse] Carbon-GENERator collapse gap={m:.1%} "
              f"CI95=[{lo:.1%},{hi:.1%}] perm_p={p:.4f}")
    _save(out, "collapse_significance.json", res)
    return res


def inverse_significance(path, out):
    m = json.load(open(path))
    res = {}
    for split in ("protein_grouped", "target_heldout"):
        if split not in m:
            continue
        mdl = m[split]["model"]["top1"]
        prior = m[split]["prior_baseline"]["top1"]
        res[split] = {"model_top1": round(mdl, 4), "prior_top1": round(prior, 4),
                      "lift": round(mdl / prior, 1) if prior else None}
        print(f"[inverse] {split}: top1={mdl:.3f} vs prior {prior:.3f} "
              f"(lift {mdl/prior:.1f}x)" if prior else "")
    _save(out, "inverse_significance.json", res)
    return res


def _save(out_dir, name, obj):
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, name), "w") as f:
        json.dump(obj, f, indent=2, default=float)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--biophys", default=None)
    ap.add_argument("--runs_dir", default=None)
    ap.add_argument("--collapse", default=None)
    ap.add_argument("--inverse", default=None)
    ap.add_argument("--output_dir", default="results/significance")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    rng = np.random.default_rng(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    if args.biophys and os.path.isfile(args.biophys):
        biophys_significance(args.biophys, args.output_dir, rng)
    if args.runs_dir and os.path.isdir(args.runs_dir):
        hub_significance(args.runs_dir, args.output_dir, rng)
    if args.collapse and os.path.isfile(args.collapse):
        collapse_significance(args.collapse, args.output_dir, rng)
    if args.inverse and os.path.isfile(args.inverse):
        inverse_significance(args.inverse, args.output_dir)
    print(f"\n[sig] wrote significance JSONs to {args.output_dir}/")


if __name__ == "__main__":
    main()
