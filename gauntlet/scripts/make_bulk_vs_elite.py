#!/usr/bin/env python3
"""Bulk correlation versus elite selection utility, pooled across all corpora.

    python scripts/make_bulk_vs_elite.py --out results

One point per assay, landscape, or kinetic axis. The x-axis is the number the
field reports (Spearman over every variant); the y-axis is the number a design
campaign actually needs (what fraction of the achievable outcome you capture if
you select the proxy's top k).

WHAT THIS FIGURE CORRECTED. Phase 25 concluded from SSMuLA alone that bulk
correlation and elite utility are unrelated (rho = +0.135, p = 0.62 over 16
landscapes). Pooling refutes that as a general claim: within ProteinGym's 41
assays, over a WIDER bulk range than SSMuLA's, the two track at rho = +0.813,
and even within SSMuLA the top-10% relationship is +0.709. The Phase 25 null was
specific to SSMuLA at top-1%, not a property of proxies. Two things do survive,
and they are what this figure is now for:

  1. A corpus-level OFFSET. At equal bulk correlation an SSMuLA landscape
     delivers more elite utility than a ProteinGym assay -- residuals +0.189 vs
     -0.076 about the pooled fit at top-1%, Mann-Whitney p = 0.0001. Bulk is
     comparable within a corpus and not across corpora, so a leaderboard mixing
     benchmark families misranks them.
  2. DEGRADATION as the elite narrows. Pooled, bulk explains 50% of the rank
     variance in top-10% utility but only 16% at top-1%. The finer the selection
     you care about, the less the reported number tells you.

Both panels use the same axes and the same three corpora; only k changes. Two
panels rather than one because the claim should not depend on a single choice of
elite cutoff -- and here it does, which is the point.

CORPORA (60 points)
  ProteinGym  41 assays, ESM-2 650M scores from ``data/proteingym/esm_cache``
  SSMuLA      16 combinatorial landscapes (Phase 25)
  rubisco      3 kinetic axes -- Vmax, K_C affinity, growth fitness -- on the
               qbcov<=0.5 functional range, which is the honest subset (Phase 24)

PETML IS DEDUCED, NOT FORGOTTEN. Its largest study has 86 variants and only two
clear 50, so a within-study top-1% or top-10% elite would be one to eight
variants and the metric would be noise. Its bulk numbers are in Phase 1; it
cannot contribute a point here, and inventing one by pooling across studies is
exactly the cross-assay pooling the project argues against.

Colour: categorical slots 1/2/3 (blue, orange, violet), validated all-pairs in
light mode -- CVD deltaE 13.0, normal-vision 16.3, all three >= 3:1 on the
surface. Scatter needs the all-pairs gate rather than the adjacent one, which is
why this is three corpora and not more.
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from gauntlet import proteingym as pg  # noqa: E402

FRACS = [0.01, 0.10]
CORPORA = ["ProteinGym", "SSMuLA", "rubisco"]

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SERIES = {"ProteinGym": "#2a78d6", "SSMuLA": "#eb6834", "rubisco": "#4a3aa7"}


def topk_utility(proxy, y, frac):
    """Fraction of achievable mean outcome captured by the proxy's top k.

    1 = the oracle's own pick, 0 = a random draw. Normalising against both ends
    is what makes assays with different dynamic range comparable; a raw "mean
    fitness of the top k" is not.
    """
    k = max(1, int(round(frac * len(y))))
    rand, best = y.mean(), np.sort(y)[-k:].mean()
    got = y[np.argsort(-proxy)[:k]].mean()
    return np.nan if best == rand else float((got - rand) / (best - rand))


def collect(args):  # args.pg_cache resolved in main()
    rows = []

    ref = pg.select_assays(os.path.join(args.pg_dir, "reference.csv"))
    cached = {f[:-4] for f in os.listdir(args.pg_cache)}
    ref = ref[ref.DMS_id.isin(cached)]
    for r in ref.itertuples():
        _, y, _ = pg.load_assay(r.DMS_id, r.target_seq,
                                os.path.join(args.pg_dir, "assays"))
        zs = np.load(os.path.join(args.pg_cache, f"{r.DMS_id}.npy"))
        if len(zs) != len(y):
            print(f"  skip {r.DMS_id}: {len(zs)} scores vs {len(y)} labels")
            continue
        rows.append(dict(corpus="ProteinGym", unit=r.DMS_id, n=len(y),
                         bulk=stats.spearmanr(zs, y).statistic,
                         **{f"util_{f}": topk_utility(zs, y, f) for f in FRACS}))

    s = pd.read_csv(args.ssmula)
    for name, g in s.groupby("landscape"):
        x, y = g.esm2_wtm.to_numpy(), g.fitness.to_numpy()
        rows.append(dict(corpus="SSMuLA", unit=name, n=len(g),
                         bulk=stats.spearmanr(x, y).statistic,
                         **{f"util_{f}": topk_utility(x, y, f) for f in FRACS}))

    d = pd.read_csv(args.rubisco)
    d = d[(d.vmax_err <= 0.5) & (d.kc_err <= 0.5)]
    for tgt, label in [("vmax", "rubisco Vmax"), ("kc_affinity", "rubisco K_C"),
                       ("fitness", "rubisco fitness")]:
        y = (-d.kc if tgt == "kc_affinity" else d[tgt]).to_numpy()
        x = d.esm2_wtm.to_numpy()
        rows.append(dict(corpus="rubisco", unit=label, n=len(d),
                         bulk=stats.spearmanr(x, y).statistic,
                         **{f"util_{f}": topk_utility(x, y, f) for f in FRACS}))

    return pd.DataFrame(rows)


def render(t, out_png, out_pdf, scorer="ESM-2 650M"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family": ["DejaVu Sans"], "font.size": 8,
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
    })
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.7), sharex=True, sharey=True)

    # Direct-label only the points that carry the argument, not every point.
    highlight = {"TrpB3H": (9, -14), "GB1": (9, 6)}

    for ax, frac in zip(axes, FRACS):
        col = f"util_{frac}"
        sub = t[t[col].notna()]

        ax.axhline(0, color=AXIS, lw=1.0, zorder=1)
        ax.grid(True, color=GRID, lw=0.6, zorder=0)
        ax.set_axisbelow(True)

        # A thin per-corpus fit makes the offset legible: similar slopes,
        # different intercepts is the finding, and it is invisible in the
        # pooled cloud alone.
        for corpus in CORPORA:
            g = sub[sub.corpus == corpus]
            if len(g) >= 4:
                xs = np.linspace(g.bulk.min(), g.bulk.max(), 2)
                fit = np.poly1d(np.polyfit(g.bulk, g[col], 1))
                ax.plot(xs, fit(xs), color=SERIES[corpus], lw=2.0,
                        alpha=0.55, zorder=2, solid_capstyle="round")

        for corpus in CORPORA:
            g = sub[sub.corpus == corpus]
            r = stats.spearmanr(g.bulk, g[col]).statistic if len(g) >= 4 else np.nan
            lab = (f"{corpus} ({len(g)})" if np.isnan(r)
                   else f"{corpus} ({len(g)})   $\\rho$ {r:+.2f}")
            ax.scatter(g.bulk, g[col], s=68, c=SERIES[corpus],
                       edgecolors=SURFACE, linewidths=1.6, zorder=3,
                       label=lab if frac == FRACS[0] else None)

        rho, p = stats.spearmanr(sub.bulk, sub[col])
        pstr = "p < 0.001" if p < 0.001 else f"p = {p:.3f}"
        ax.set_title(f"top {int(frac*100)}%    pooled $\\rho$ = {rho:+.2f}, {pstr}"
                     f"    $\\rho^2$ = {rho**2:.2f}",
                     fontsize=8.5, color=INK, pad=8)

        for name, (dx, dy) in highlight.items():
            row = sub[sub.unit == name]
            if row.empty:
                continue
            r = row.iloc[0]
            ax.annotate(name, (r.bulk, r[col]), textcoords="offset points",
                        xytext=(dx, dy), fontsize=7, color=INK2,
                        ha="left" if dx > 0 else "right")

        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(AXIS)
            ax.spines[side].set_linewidth(0.8)
        ax.tick_params(colors=MUTED, labelsize=7.5, length=3, width=0.8)
        ax.set_xlabel("bulk Spearman  (what gets reported)", fontsize=8, color=INK2)

    # The y-axis label already says 0 = random, so the zero rule carries no
    # second annotation -- it would collide with the leftmost points.
    axes[0].set_ylabel("top-k utility\n(1 = oracle, 0 = random)", fontsize=8, color=INK2)

    leg = axes[0].legend(loc="upper left", frameon=False, fontsize=7.5,
                         handletextpad=0.4, borderpad=0.2, labelspacing=0.35)
    for txt in leg.get_texts():
        txt.set_color(INK2)

    # Every number in the subtitle is computed from the data being plotted.
    # Hardcoding them means a rerun on a different scorer silently inherits the
    # previous run's conclusions, which is precisely the failure this project
    # keeps documenting.
    def _gap(frac):
        col = f"util_{frac}"
        sub = t[t[col].notna()].copy()
        fit = np.poly1d(np.polyfit(sub.bulk, sub[col], 1))
        sub["resid"] = sub[col] - fit(sub.bulk)
        a = sub.resid[sub.corpus == "SSMuLA"]
        b = sub.resid[sub.corpus == "ProteinGym"]
        return (a.mean() - b.mean(), stats.mannwhitneyu(a, b).pvalue,
                stats.spearmanr(sub.bulk, sub[col]).statistic ** 2)

    g1, p1, r2_1 = _gap(FRACS[0])
    _, _, r2_10 = _gap(FRACS[1])
    direction = "more" if g1 > 0 else "less"
    pstr = f"p = {p1:.4f}" if p1 >= 1e-4 else f"p < 0.0001"

    fig.suptitle("Bulk correlation tracks elite utility within a corpus, "
                 "but sits at a different level in each",
                 fontsize=10, color=INK, x=0.5, y=0.995)
    fig.text(0.5, 0.925,
             f"{scorer}.  At equal bulk correlation an SSMuLA landscape returns "
             f"{direction} elite utility than a ProteinGym assay "
             f"(gap {g1:+.3f}, {pstr});\nand the pooled relationship weakens as "
             f"the elite narrows — $\\rho^2$ {r2_10:.2f} at top-10%, {r2_1:.2f} "
             f"at top-1%.",
             ha="center", va="top", fontsize=7.5, color=INK2, linespacing=1.45)
    fig.tight_layout(rect=[0, 0, 1, 0.855])
    fig.savefig(out_png, dpi=220)
    fig.savefig(out_pdf)
    print(f"wrote {out_png} and {out_pdf}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pg_dir", default="data/proteingym")
    ap.add_argument("--ssmula", default=None)
    ap.add_argument("--rubisco", default=None)
    ap.add_argument("--out", default="results")
    ap.add_argument("--scorer", default="esm2", choices=["esm2", "esmc"],
                    help="rebuild the whole figure on a second scorer family; "
                         "if the corpus offset persists it is a property of the "
                         "benchmarks, if it moves it was an ESM-2 artefact")
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    if args.tag is None:
        args.tag = "" if args.scorer == "esm2" else "_esmc"
    if args.ssmula is None:
        args.ssmula = f"results/ssmula_scored_variants{args.tag}.csv"
    if args.rubisco is None:
        args.rubisco = f"results/rubisco_scored_variants{args.tag}.csv"
    args.pg_cache = os.path.join(
        args.pg_dir, "esm_cache" if args.scorer == "esm2" else "esmc_cache")
    print(f"scorer: {args.scorer}   ProteinGym cache: {args.pg_cache}")

    t = collect(args)
    t.to_csv(os.path.join(args.out, f"bulk_vs_elite{args.tag}.csv"), index=False)

    print(f"\n{len(t)} points: " +
          ", ".join(f"{c} {int((t.corpus == c).sum())}" for c in CORPORA))
    for frac in FRACS:
        col = f"util_{frac}"
        sub = t[t[col].notna()]
        rho, p = stats.spearmanr(sub.bulk, sub[col])
        print(f"\ntop {int(frac*100)}%:  Spearman(bulk, utility) = {rho:+.3f}  "
              f"p = {p:.3f}  n = {len(sub)}")
        print(f"  utility exceeds bulk in {int((sub[col] > sub.bulk).sum())}/{len(sub)}")
        print(f"  bulk    range {sub.bulk.min():+.3f} .. {sub.bulk.max():+.3f}")
        print(f"  utility range {sub[col].min():+.3f} .. {sub[col].max():+.3f}")
        for c in CORPORA:
            g = sub[sub.corpus == c]
            print(f"    {c:11s} bulk {g.bulk.mean():+.3f}   utility {g[col].mean():+.3f}")

    render(t, os.path.join(args.out, f"bulk_vs_elite{args.tag}.png"),
           os.path.join(args.out, f"bulk_vs_elite{args.tag}.pdf"),
           scorer="ESM-2 650M" if args.scorer == "esm2" else "ESM-C 300M")


if __name__ == "__main__":
    main()
