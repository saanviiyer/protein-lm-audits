#!/usr/bin/env python3
"""Place PETML in the bulk-versus-elite frame, at a k it can actually support.

    python scripts/test_petml_placement.py --scorer esm2 --out results

PETML is the corpus the whole project started from and the one that could not
contribute a point to the pooled figure: its largest study has 86 variants, so a
top-1% elite is one variant and a top-10% elite is two to eight. Phase 26
excluded it for that reason.

It can support a COARSER elite. This project's own Verify-Agents draft already
ranks PETML within study at top 20%, so that is the precedent used here, with a
floor of 25 variants per study so the elite is at least five. Ten units clear
that bar -- five activity studies and five melting-temperature studies.

THE COMPARABILITY POINT, which is the reason this is a separate script rather
than a flag. A top-20% utility is not comparable to a top-10% one: the coarser
the elite, the easier the selection and the higher the number, mechanically. So
every corpus is recomputed at k = 20% here. Placing PETML's top-20% points
against the other corpora's top-10% points would manufacture exactly the kind of
artefact this project documents.

ENTROPY IS DELIBERATELY NOT COMPUTED. It would need a GPU pass over the ~205
positions PETML mutates, and Phase 31 showed it is redundant with score skew
(correlation -0.882; it does not survive controlling for skew). Spending compute
on a quantity already shown to add nothing would be the wrong call.
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from gauntlet import proteingym as pg  # noqa: E402

FRAC = 0.20
MIN_N = 25


def topk_utility(proxy, y, frac=FRAC):
    k = max(1, int(round(frac * len(y))))
    rand, best = y.mean(), np.sort(y)[-k:].mean()
    got = y[np.argsort(-proxy)[:k]].mean()
    return np.nan if best == rand else float((got - rand) / (best - rand))


def unit(corpus, name, x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if len(x) < MIN_N or len(np.unique(x)) < 2 or len(np.unique(y)) < 2:
        return None
    return dict(corpus=corpus, unit=name, n=len(x),
                bulk=stats.spearmanr(x, y).statistic,
                util=topk_utility(x, y),
                score_skew=float(stats.skew(x)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pg_dir", default="data/proteingym")
    ap.add_argument("--scorer", default="esm2", choices=["esm2", "esmc"])
    ap.add_argument("--out", default="results")
    args = ap.parse_args()
    tag = "" if args.scorer == "esm2" else "_esmc"
    pg_cache = os.path.join(args.pg_dir,
                            "esm_cache" if args.scorer == "esm2" else "esmc_cache")

    rows = []

    d = pd.read_csv(f"results/scored_variants{tag}.csv")
    for target in ["logActivity", "Tm"]:
        for study, g in d[d[target].notna()].groupby("study"):
            u = unit("PETML", f"{study} [{target}]",
                     g.esm2_wtm.to_numpy(), g[target].to_numpy())
            if u:
                rows.append(u)
    print(f"PETML: {sum(r['corpus'] == 'PETML' for r in rows)} units at n>={MIN_N}")

    ref = pg.select_assays(os.path.join(args.pg_dir, "reference.csv"))
    cached = {f[:-4] for f in os.listdir(pg_cache)}
    for r in ref[ref.DMS_id.isin(cached)].itertuples():
        _, y, _ = pg.load_assay(r.DMS_id, r.target_seq,
                                os.path.join(args.pg_dir, "assays"))
        zs = np.load(os.path.join(pg_cache, f"{r.DMS_id}.npy"))
        if len(zs) == len(y):
            u = unit("ProteinGym", r.DMS_id, zs, y)
            if u:
                rows.append(u)

    s = pd.read_csv(f"results/ssmula_scored_variants{tag}.csv")
    for name, g in s.groupby("landscape"):
        u = unit("SSMuLA", name, g.esm2_wtm.to_numpy(), g.fitness.to_numpy())
        if u:
            rows.append(u)

    rb = pd.read_csv(f"results/rubisco_scored_variants{tag}.csv")
    rb = rb[(rb.vmax_err <= 0.5) & (rb.kc_err <= 0.5)]
    for tgt, label in [("vmax", "rubisco Vmax"), ("kc_affinity", "rubisco K_C"),
                       ("fitness", "rubisco fitness")]:
        y = (-rb.kc if tgt == "kc_affinity" else rb[tgt]).to_numpy()
        u = unit("rubisco", label, rb.esm2_wtm.to_numpy(), y)
        if u:
            rows.append(u)

    t = pd.DataFrame(rows)
    t.to_csv(os.path.join(args.out, f"petml_placement{tag}.csv"), index=False)

    print(f"\nall corpora recomputed at k = {int(FRAC*100)}%   ({len(t)} units)")
    print(t.groupby("corpus")[["bulk", "util", "score_skew"]].agg(["count", "mean"])
          .round(3).to_string())

    # Where does PETML sit relative to the fit through the other corpora?
    other = t[t.corpus != "PETML"]
    fit = np.poly1d(np.polyfit(other.bulk, other.util, 1))
    t["resid"] = t.util - fit(t.bulk)

    print("\n" + "=" * 72)
    print("PLACEMENT -- residual about the fit through the OTHER corpora")
    print("(positive = more elite utility than its bulk correlation predicts)")
    print("=" * 72)
    for c in ["ProteinGym", "SSMuLA", "rubisco", "PETML"]:
        g = t[t.corpus == c]
        print(f"  {c:11s} n={len(g):2d}   mean residual {g.resid.mean():+.3f}   "
              f"median {g.resid.median():+.3f}")

    pgm, ssm, pet = (t[t.corpus == c] for c in ["ProteinGym", "SSMuLA", "PETML"])
    print(f"\n  SSMuLA vs ProteinGym  gap {ssm.resid.mean() - pgm.resid.mean():+.3f}  "
          f"p={stats.mannwhitneyu(ssm.resid, pgm.resid).pvalue:.4f}")
    print(f"  PETML  vs ProteinGym  gap {pet.resid.mean() - pgm.resid.mean():+.3f}  "
          f"p={stats.mannwhitneyu(pet.resid, pgm.resid).pvalue:.4f}")
    print(f"  PETML  vs SSMuLA      gap {pet.resid.mean() - ssm.resid.mean():+.3f}  "
          f"p={stats.mannwhitneyu(pet.resid, ssm.resid).pvalue:.4f}")

    print("\n" + "=" * 72)
    print("Does score skew still track the residual with PETML included?")
    print("=" * 72)
    for lab, sub in [("all corpora", t), ("excluding PETML", other.assign(
            resid=other.util - fit(other.bulk))), ("PETML only", pet)]:
        if len(sub) < 5:
            continue
        r = stats.spearmanr(sub.score_skew, sub.resid)
        print(f"  {lab:16s} n={len(sub):2d}  rho {r.statistic:+.3f}  p={r.pvalue:.4f}")

    print(f"\nwrote {args.out}/petml_placement{tag}.csv")


if __name__ == "__main__":
    main()
