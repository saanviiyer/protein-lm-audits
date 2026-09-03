#!/usr/bin/env python3
"""How many PETML units would it take to place the corpus? And can it ever get them?

    python scripts/power_petml.py --out results

Phase 32 could not decide whether PETML sits on ProteinGym's line or SSMuLA's:
p = 0.82 and 0.41 on ESM-2, and the two scorers disagree on the sign of PETML's
mean residual. "Underpowered" is only useful if it comes with a number, and a
number is only useful if it comes with a ceiling -- there is no point costing a
literature-curation effort that cannot reach the required size.

WHY THIS IS NOT POWERED ON THE OBSERVED EFFECT. The conventional move -- take the
measured gap, compute n for 80% power -- would be powering on +0.072 (ESM-2) or
-0.034 (ESM-C), estimates that disagree in sign. Post-hoc power on a noisy point
estimate is a well-known way to produce a confident-looking number that means
nothing. Instead the effect size is swept, and the two anchors reported are
effects the project actually cares about:

  delta = 0.115   the SSMuLA-vs-ProteinGym gap at k=20% -- "is PETML as
                  different from ProteinGym as SSMuLA is?"
  delta = 0.05    half that -- a difference small enough to be uninteresting

Power is by simulation against the observed residual spreads, using the same
Mann-Whitney test Phase 32 used, so the answer applies to the test actually run.

THE NOISE IS ALSO DECOMPOSED. A PETML unit's residual is uncertain for two
reasons: genuine between-study heterogeneity, and estimation error because the
study is small. Only the second is fixed by finding bigger studies; only the
first is fixed by finding more of them. The within-unit component is estimated by
bootstrapping each study's own variants.
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

FRAC = 0.20
MIN_N = 25
ALPHA = 0.05
TARGET_POWER = 0.80
DELTAS = [0.05, 0.075, 0.115, 0.15, 0.20, 0.30]
NS = [10, 15, 20, 30, 40, 60, 80, 120, 200]


def topk_utility(proxy, y, frac=FRAC):
    k = max(1, int(round(frac * len(y))))
    rand, best = y.mean(), np.sort(y)[-k:].mean()
    got = y[np.argsort(-proxy)[:k]].mean()
    return np.nan if best == rand else float((got - rand) / (best - rand))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scorer", default="esm2", choices=["esm2", "esmc"])
    ap.add_argument("--sims", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results")
    args = ap.parse_args()
    tag = "" if args.scorer == "esm2" else "_esmc"
    rng = np.random.default_rng(args.seed)

    t = pd.read_csv(f"results/petml_placement{tag}.csv")
    other = t[t.corpus != "PETML"]
    fit = np.poly1d(np.polyfit(other.bulk, other.util, 1))
    t["resid"] = t.util - fit(t.bulk)
    pet = t[t.corpus == "PETML"]
    pgm = t[t.corpus == "ProteinGym"]

    sd_pet, sd_pgm = pet.resid.std(ddof=1), pgm.resid.std(ddof=1)
    print(f"scorer={args.scorer}")
    print(f"  PETML     n={len(pet)}  residual mean {pet.resid.mean():+.3f}  SD {sd_pet:.3f}")
    print(f"  ProteinGym n={len(pgm)} residual mean {pgm.resid.mean():+.3f}  SD {sd_pgm:.3f}")

    # ---- where does the PETML noise come from? ----------------------------
    d = pd.read_csv(f"results/scored_variants{tag}.csv")
    within = []
    for target in ["logActivity", "Tm"]:
        for study, g in d[d[target].notna()].groupby("study"):
            g = g[np.isfinite(g.esm2_wtm) & g[target].notna()]
            if len(g) < MIN_N:
                continue
            x, y = g.esm2_wtm.to_numpy(), g[target].to_numpy()
            boots = []
            for _ in range(400):
                i = rng.integers(0, len(x), len(x))
                if len(np.unique(y[i])) < 2:
                    continue
                u = topk_utility(x[i], y[i])
                if u == u:
                    boots.append(u)
            if len(boots) > 50:
                within.append(np.std(boots, ddof=1))
    sd_within = float(np.mean(within))
    sd_between = float(np.sqrt(max(sd_pet ** 2 - sd_within ** 2, 0.0)))
    print(f"\n  PETML residual SD {sd_pet:.3f} decomposes into:")
    print(f"    within-unit estimation noise (bootstrap)  {sd_within:.3f}"
          f"   -> fixed by BIGGER studies")
    print(f"    between-unit heterogeneity (remainder)    {sd_between:.3f}"
          f"   -> fixed by MORE studies")
    if sd_within > sd_between:
        print("    dominant term: estimation noise -- more small studies help little")
    else:
        print("    dominant term: real heterogeneity -- more studies is the lever")

    # ---- power by simulation ----------------------------------------------
    print("\n" + "=" * 74)
    print(f"POWER to detect a PETML-vs-ProteinGym gap (Mann-Whitney, alpha={ALPHA})")
    print(f"ProteinGym held at n={len(pgm)}; {args.sims} sims per cell")
    print("=" * 74)
    header = "  n_PETML " + "".join(f"{f'd={dd}':>9s}" for dd in DELTAS)
    print(header)
    grid = {}
    for n in NS:
        cells = []
        for dd in DELTAS:
            hits = 0
            for _ in range(args.sims):
                a = rng.normal(pgm.resid.mean() + dd, sd_pet, n)
                b = rng.normal(pgm.resid.mean(), sd_pgm, len(pgm))
                if stats.mannwhitneyu(a, b).pvalue < ALPHA:
                    hits += 1
            p = hits / args.sims
            grid[(n, dd)] = p
            cells.append(f"{p:9.2f}")
        print(f"  {n:8d} " + "".join(cells))

    print(f"\n  units needed for {int(TARGET_POWER*100)}% power:")
    for dd in DELTAS:
        need = next((n for n in NS if grid[(n, dd)] >= TARGET_POWER), None)
        print(f"    delta={dd:<6} " +
              (f"{need} units" if need else f">{NS[-1]} units"))

    # ---- the ceiling -------------------------------------------------------
    print("\n" + "=" * 74)
    print("CEILING -- how many units could PETML ever supply?")
    print("=" * 74)
    tot = 0
    for target in ["logActivity", "Tm"]:
        g = d[d[target].notna()].groupby("study").size()
        line = "  ".join(f"n>={k}: {(g >= k).sum():2d}" for k in [15, 20, 25, 30, 40])
        print(f"  {target:12s} {len(g)} studies total   {line}")
        tot += (g >= MIN_N).sum()
    print(f"\n  usable at n>={MIN_N} today: {tot} units (both targets)")
    print(f"  the ENTIRE public plastic-degradation corpus is ~514 activity")
    print(f"  measurements over ~501 variants, so the ceiling is not a curation")
    print(f"  problem -- it is the size of the field's measured record.")

    pd.DataFrame([{"n": n, "delta": dd, "power": grid[(n, dd)]}
                  for n in NS for dd in DELTAS]).to_csv(
        os.path.join(args.out, f"power_petml{tag}.csv"), index=False)
    print(f"\nwrote {args.out}/power_petml{tag}.csv")


if __name__ == "__main__":
    main()
