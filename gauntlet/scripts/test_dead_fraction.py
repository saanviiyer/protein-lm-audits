#!/usr/bin/env python3
"""Does dead-variant fraction explain the corpus offset in Phase 26?

    python scripts/test_dead_fraction.py --out results

Phase 26 found that at equal bulk correlation an SSMuLA landscape returns more
elite utility than a ProteinGym assay -- residuals +0.189 vs -0.076 about the
pooled fit, p = 0.0001. That is an observation about benchmark families. This
asks whether a measurable property of the fitness distribution accounts for it,
which would turn it into a mechanism.

THE CONFOUND THIS SCRIPT IS BUILT TO AVOID. ProteinGym ships its own
``DMS_score_bin`` and SSMuLA's TrpB landscapes ship an ``active`` flag, but the
two were drawn by different people under different conventions. Using each
corpus's native label to compute a per-corpus statistic, and then testing
whether that statistic explains a per-corpus offset, would risk measuring the
difference between two labelling conventions and calling it biology.

So: ONE definition, applied identically to all 60 units. Dead fraction is the
mass below **Otsu's threshold** -- the split that maximises between-class
variance, standard and parameter-free -- computed on each unit's own outcome
distribution. The native labels are then used only to CHECK that measure, never
to build it. If Otsu agrees with ``DMS_score_bin`` and with ``active``, the
uniform measure is reading the same thing both conventions were.

PRE-SPECIFIED COMPETITORS. Testing one candidate and stopping is how a fished
result gets published. Dead fraction is tested against four alternatives fixed
before looking at any of them: library size, fitness dispersion, distribution
skew, and whether the unit is multi-mutant. All five are reported whatever they
show, and the verdict is whether the SSMuLA-vs-ProteinGym residual gap closes.
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from gauntlet import proteingym as pg  # noqa: E402

# ``multi_mut`` was in this list on the first run and has been REMOVED as
# degenerate. Every SSMuLA landscape is multi-mutant and every ProteinGym and
# rubisco unit is single-mutant, so the variable is identical to the indicator
# "corpus == SSMuLA". It closed 96% of the corpus gap, which sounds like an
# explanation and is pure circularity: it is the grouping variable wearing a
# different name. The general rule this cost us -- a property that is CONSTANT
# WITHIN each corpus can never explain a between-corpus offset, it can only
# restate it. Only properties that vary within corpora are admissible here, and
# all four below do.
CANDIDATES = ["dead_frac", "log_n", "dispersion", "skew"]


def otsu_dead_fraction(y, bins=256):
    """Fraction of variants below the between-class-variance-maximising split.

    Returns NaN if the distribution has no usable split (all one value).
    """
    y = np.asarray(y, float)
    y = y[np.isfinite(y)]
    if len(y) < 20 or y.min() == y.max():
        return np.nan, np.nan
    hist, edges = np.histogram(y, bins=bins)
    w = hist / hist.sum()
    centers = (edges[:-1] + edges[1:]) / 2
    w0 = np.cumsum(w)
    m0 = np.cumsum(w * centers)
    mt = m0[-1]
    denom = w0 * (1 - w0)
    with np.errstate(divide="ignore", invalid="ignore"):
        between = (mt * w0 - m0) ** 2 / denom
    between[~np.isfinite(between)] = -1
    k = int(np.argmax(between))
    thr = float(centers[k])
    return float((y < thr).mean()), thr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pg_dir", default="data/proteingym")
    ap.add_argument("--ssmula", default="results/ssmula_scored_variants.csv")
    ap.add_argument("--ssmula_zip", default="data/ssmula/data.zip")
    ap.add_argument("--rubisco", default="results/rubisco_scored_variants.csv")
    ap.add_argument("--points", default="results/bulk_vs_elite.csv")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    rows, checks = [], []

    ref = pg.select_assays(os.path.join(args.pg_dir, "reference.csv"))
    cached = {f[:-4] for f in os.listdir(os.path.join(args.pg_dir, "esm_cache"))}
    for r in ref[ref.DMS_id.isin(cached)].itertuples():
        _, y, _ = pg.load_assay(r.DMS_id, r.target_seq,
                                os.path.join(args.pg_dir, "assays"))
        dead, thr = otsu_dead_fraction(y)
        rows.append(dict(unit=r.DMS_id, corpus="ProteinGym", dead_frac=dead,
                         log_n=np.log10(len(y)), dispersion=stats.iqr(y) / (np.ptp(y) or 1),
                         skew=stats.skew(y), multi_mut=0))
        raw = pd.read_csv(os.path.join(args.pg_dir, "assays", f"{r.DMS_id}.csv"))
        if "DMS_score_bin" in raw:
            checks.append(dict(unit=r.DMS_id, native="DMS_score_bin",
                               native_dead=1 - raw.DMS_score_bin.mean(), otsu=dead))

    s = pd.read_csv(args.ssmula)
    for name, g in s.groupby("landscape"):
        y = g.fitness.to_numpy()
        dead, thr = otsu_dead_fraction(y)
        rows.append(dict(unit=name, corpus="SSMuLA", dead_frac=dead,
                         log_n=np.log10(len(y)), dispersion=stats.iqr(y) / (np.ptp(y) or 1),
                         skew=stats.skew(y), multi_mut=1))

    # Native check for the TrpB landscapes, which ship an `active` flag.
    import io
    import zipfile
    z = zipfile.ZipFile(args.ssmula_zip)
    for n in z.namelist():
        if "/fitness_landscape/TrpB" in n and n.endswith(".csv"):
            name = os.path.basename(n)[:-4]
            d = pd.read_csv(io.BytesIO(z.read(n)))
            if "active" not in d:
                continue
            d = d[~d.AAs.astype(str).str.contains(r"\*")]
            o = [r for r in rows if r["unit"] == name]
            if o:
                checks.append(dict(unit=name, native="active",
                                   native_dead=1 - d.active.mean(),
                                   otsu=o[0]["dead_frac"]))

    d = pd.read_csv(args.rubisco)
    d = d[(d.vmax_err <= 0.5) & (d.kc_err <= 0.5)]
    for tgt, label in [("vmax", "rubisco Vmax"), ("kc_affinity", "rubisco K_C"),
                       ("fitness", "rubisco fitness")]:
        y = (-d.kc if tgt == "kc_affinity" else d[tgt]).to_numpy()
        dead, thr = otsu_dead_fraction(y)
        rows.append(dict(unit=label, corpus="rubisco", dead_frac=dead,
                         log_n=np.log10(len(y)), dispersion=stats.iqr(y) / (np.ptp(y) or 1),
                         skew=stats.skew(y), multi_mut=0))

    feat = pd.DataFrame(rows)
    pts = pd.read_csv(args.points)
    t = pts.merge(feat, on=["unit", "corpus"], how="inner")
    t.to_csv(os.path.join(args.out, "dead_fraction.csv"), index=False)
    print(f"{len(t)} units matched\n")

    chk = pd.DataFrame(checks)
    print("=" * 72)
    print("VALIDATION -- does the uniform Otsu measure agree with native labels?")
    print("=" * 72)
    for native, g in chk.groupby("native"):
        r = stats.spearmanr(g.native_dead, g.otsu)
        print(f"  vs {native:14s} n={len(g):3d}  rho={r.statistic:+.3f} "
              f"(p={r.pvalue:.1e})  mean |diff| = {(g.native_dead - g.otsu).abs().mean():.3f}")

    for frac in ["0.01", "0.1"]:
        col = f"util_{frac}"
        sub = t[t[col].notna()].copy()
        fit = np.poly1d(np.polyfit(sub.bulk, sub[col], 1))
        sub["resid"] = sub[col] - fit(sub.bulk)

        print("\n" + "=" * 72)
        print(f"TOP {'1' if frac == '0.01' else '10'}%  -- what predicts the residual?")
        print("=" * 72)
        print(f"  {'candidate':12s} {'rho vs resid':>13s} {'p':>9s}")
        for c in CANDIDATES:
            r = stats.spearmanr(sub[c], sub.resid)
            print(f"  {c:12s} {r.statistic:+13.3f} {r.pvalue:9.4f}")

        pgm = sub[sub.corpus == "ProteinGym"]
        ssm = sub[sub.corpus == "SSMuLA"]
        gap0 = ssm.resid.mean() - pgm.resid.mean()
        p0 = stats.mannwhitneyu(ssm.resid, pgm.resid).pvalue
        print(f"\n  corpus gap (SSMuLA - ProteinGym) on bulk alone: "
              f"{gap0:+.3f}  p={p0:.4f}")
        print(f"    mean dead fraction: SSMuLA {ssm.dead_frac.mean():.3f}   "
              f"ProteinGym {pgm.dead_frac.mean():.3f}   "
              f"rubisco {sub[sub.corpus=='rubisco'].dead_frac.mean():.3f}")

        # Add each candidate to the model and re-measure the gap. If a property
        # explains the offset, the gap collapses toward zero once it is in.
        print("\n  corpus gap after adding each candidate to the model:")
        for c in CANDIDATES:
            X = np.column_stack([sub.bulk, sub[c], np.ones(len(sub))])
            ok = np.isfinite(X).all(axis=1)
            beta, *_ = np.linalg.lstsq(X[ok], sub[col].to_numpy()[ok], rcond=None)
            res2 = sub[col].to_numpy()[ok] - X[ok] @ beta
            cor = sub.corpus.to_numpy()[ok]
            a, b = res2[cor == "SSMuLA"], res2[cor == "ProteinGym"]
            gap = a.mean() - b.mean()
            p = stats.mannwhitneyu(a, b).pvalue
            print(f"    + {c:12s} gap {gap:+.3f}  p={p:.4f}   "
                  f"({100*(1-abs(gap)/abs(gap0)):+.0f}% closed)")

    # The decisive test. A between-corpus difference in level can be an
    # ecological artefact; if dead fraction is a real mechanism it must also
    # act WITHIN a corpus, where the labelling convention is at least
    # self-consistent and where ProteinGym gives 41 units spanning a six-fold
    # range of dead fraction.
    print("\n" + "=" * 72)
    print("DECISIVE -- does dead fraction predict the residual WITHIN a corpus?")
    print("=" * 72)
    for frac in ["0.01", "0.1"]:
        col = f"util_{frac}"
        print(f"\n  top {'1' if frac == '0.01' else '10'}%")
        for c in ["ProteinGym", "SSMuLA"]:
            g = t[(t.corpus == c) & t[col].notna()].copy()
            if len(g) < 8:
                continue
            fit = np.poly1d(np.polyfit(g.bulk, g[col], 1))
            g["resid"] = g[col] - fit(g.bulk)
            r = stats.spearmanr(g.dead_frac, g.resid)
            print(f"    {c:11s} n={len(g):2d}  dead {g.dead_frac.min():.2f}-"
                  f"{g.dead_frac.max():.2f}   rho vs residual "
                  f"{r.statistic:+.3f}  p={r.pvalue:.3f}")

    print(f"\nwrote {args.out}/dead_fraction.csv")


if __name__ == "__main__":
    main()
