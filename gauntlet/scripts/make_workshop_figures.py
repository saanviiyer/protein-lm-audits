#!/usr/bin/env python3
"""Figures for the TAE, GEM and EIML submissions.

    python scripts/make_workshop_figures.py

Everything is built from result tables already on disk. No scoring, no GPU, no
network. That matters: the three papers share one evidence base, so a figure
that silently recomputed a number would let the same measurement disagree with
itself across venues.

  TAE   fig_decay        bulk correlation against elite utility, k = 10% and 1%,
                         one point per assay/landscape/axis, per-corpus fits.
        fig_replicate    the corpus offset and the decay, ESM-2 beside ESM-C.
  GEM   fig_campaign     top-20% selection utility by corpus, the number a
                         campaign actually receives, PETML included.
        fig_power        units needed for 80% power against effect size, with
                         what the published record can supply.
  EIML  fig_simpson      score skew against elite utility within and between
                         benchmark families: the signs disagree.

Colour: categorical slots 1/2/3 (blue, orange, violet) from the project's
validated light-mode palette, all-pairs contrast checked. A fourth corpus
(PETML) appears only in panels where the other three are collapsed or where it
is the subject, so the three-slot cap holds everywhere.
"""

import os

import numpy as np
import pandas as pd
from scipy import stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

RES = "results"
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
BLUE, ORANGE, VIOLET = "#2a78d6", "#eb6834", "#4a3aa7"
SERIES = {"ProteinGym": BLUE, "SSMuLA": ORANGE, "rubisco": VIOLET, "PETML": INK}

plt.rcParams.update({
    "font.family": ["DejaVu Sans"], "font.size": 8,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
})


def tidy(ax):
    ax.grid(True, color=GRID, lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=MUTED, labelsize=7.5, length=3, width=0.8)


def resid(t, xcol, ycol, subset=None):
    """Residual about a straight-line fit, optionally fitted on a subset."""
    base = t if subset is None else subset
    fit = np.poly1d(np.polyfit(base[xcol], base[ycol], 1))
    return t[ycol] - fit(t[xcol])


def save(fig, name, dests):
    for d in dests:
        os.makedirs(d, exist_ok=True)
        fig.savefig(os.path.join(d, name + ".pdf"))
        fig.savefig(os.path.join(d, name + ".png"), dpi=220)
    plt.close(fig)
    print("wrote", name, "->", ", ".join(dests))


# --------------------------------------------------------------------- TAE

def fig_decay(dest):
    t = pd.read_csv(f"{RES}/bulk_vs_elite.csv")
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.5), sharex=True, sharey=True)

    for ax, col, k in [(axes[1], "util_0.01", 1), (axes[0], "util_0.1", 10)]:
        sub = t[t[col].notna()]
        ax.axhline(0, color=AXIS, lw=1.0, zorder=1)
        tidy(ax)
        for corpus, c in SERIES.items():
            g = sub[sub.corpus == corpus]
            if len(g) >= 4:
                xs = np.linspace(g.bulk.min(), g.bulk.max(), 2)
                f = np.poly1d(np.polyfit(g.bulk, g[col], 1))
                ax.plot(xs, f(xs), color=c, lw=2.0, alpha=0.55, zorder=2,
                        solid_capstyle="round")
        for corpus, c in SERIES.items():
            g = sub[sub.corpus == corpus]
            if not len(g):
                continue
            r = stats.spearmanr(g.bulk, g[col]).statistic if len(g) >= 4 else np.nan
            lab = (f"{corpus} ({len(g)})" if np.isnan(r)
                   else f"{corpus} ({len(g)})   $\\rho$ {r:+.2f}")
            ax.scatter(g.bulk, g[col], s=62, c=c, edgecolors=SURFACE,
                       linewidths=1.5, zorder=3, label=lab if k == 10 else None)
        rho = stats.spearmanr(sub.bulk, sub[col]).statistic
        ax.set_title(f"select the top {k}%      pooled $\\rho^2$ = {rho**2:.2f}",
                     fontsize=8.5, color=INK, pad=7)
        ax.set_xlabel("bulk Spearman  (the number that gets reported)",
                      fontsize=8, color=INK2)

    axes[0].set_ylabel("selection utility\n(1 = oracle, 0 = random)",
                       fontsize=8, color=INK2)
    leg = axes[0].legend(loc="upper left", frameon=False, fontsize=7.5,
                         handletextpad=0.4, borderpad=0.2, labelspacing=0.3)
    for txt in leg.get_texts():
        txt.set_color(INK2)

    fig.suptitle("One reported number, two different constructs",
                 fontsize=10, color=INK, y=0.995)
    fig.text(0.5, 0.915,
             "The same bulk correlation buys different amounts of selection on "
             "different benchmark families, and the\nagreement between the two "
             "decays as the selection sharpens. ESM-2 650M, 60 units.",
             ha="center", va="top", fontsize=7.5, color=INK2, linespacing=1.45)
    fig.tight_layout(rect=[0, 0, 1, 0.86])
    save(fig, "fig_decay", dest)


def fig_replicate(dest):
    rows = []
    for tag, lab in [("", "ESM-2 650M"), ("_esmc", "ESM-C 300M")]:
        t = pd.read_csv(f"{RES}/bulk_vs_elite{tag}.csv")
        for col, k in [("util_0.01", "top 1%"), ("util_0.1", "top 10%")]:
            s = t[t[col].notna()].copy()
            s["r"] = resid(s, "bulk", col)
            a = s.r[s.corpus == "SSMuLA"]
            b = s.r[s.corpus == "ProteinGym"]
            rho = stats.spearmanr(s.bulk, s[col]).statistic
            rows.append(dict(scorer=lab, k=k, gap=a.mean() - b.mean(),
                             p=stats.mannwhitneyu(a, b).pvalue, r2=rho ** 2))
    d = pd.DataFrame(rows)

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.9))
    ks = ["top 1%", "top 10%"]
    scorers = ["ESM-2 650M", "ESM-C 300M"]
    w = 0.34

    ax = axes[0]
    tidy(ax)
    for i, sc in enumerate(scorers):
        vals = [d[(d.scorer == sc) & (d.k == k)].gap.iloc[0] for k in ks]
        ax.bar(np.arange(2) + (i - 0.5) * w, vals, w,
               color=[BLUE, ORANGE][i], edgecolor=SURFACE, linewidth=1.2,
               label=sc, zorder=3)
        for j, v in enumerate(vals):
            ax.text(j + (i - 0.5) * w, v + 0.008, f"{v:+.3f}", ha="center",
                    fontsize=7, color=INK2)
    ax.axhline(0, color=AXIS, lw=1.0)
    ax.set_xticks(range(2))
    ax.set_xticklabels(ks, fontsize=8, color=INK2)
    ax.set_ylabel("offset between families\n(SSMuLA − ProteinGym)",
                  fontsize=8, color=INK2)
    ax.set_title("The offset is a property of the benchmarks",
                 fontsize=8.5, color=INK, pad=7)
    ax.set_ylim(0, 0.33)
    leg = ax.legend(frameon=False, fontsize=7.5, loc="upper right")
    for txt in leg.get_texts():
        txt.set_color(INK2)

    ax = axes[1]
    tidy(ax)
    for i, sc in enumerate(scorers):
        vals = [d[(d.scorer == sc) & (d.k == k)].r2.iloc[0] for k in ks]
        ax.bar(np.arange(2) + (i - 0.5) * w, vals, w,
               color=[BLUE, ORANGE][i], edgecolor=SURFACE, linewidth=1.2,
               zorder=3)
        for j, v in enumerate(vals):
            ax.text(j + (i - 0.5) * w, v + 0.012, f"{v:.2f}", ha="center",
                    fontsize=7, color=INK2)
    ax.set_xticks(range(2))
    ax.set_xticklabels(ks, fontsize=8, color=INK2)
    ax.set_ylabel("variance in selection utility\nexplained by bulk ($\\rho^2$)",
                  fontsize=8, color=INK2)
    ax.set_title("So is the decay", fontsize=8.5, color=INK, pad=7)
    ax.set_ylim(0, 0.62)

    fig.suptitle("Two scorer families, two laboratories, the same two effects",
                 fontsize=9.5, color=INK, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    save(fig, "fig_replicate", dest)


# --------------------------------------------------------------------- GEM

def fig_campaign(dest):
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2),
                             gridspec_kw={"width_ratios": [1.25, 1]})

    ax = axes[0]
    tidy(ax)
    order = ["PETML", "SSMuLA", "rubisco", "ProteinGym"]
    p2 = pd.read_csv(f"{RES}/petml_placement.csv")
    for i, c in enumerate(order):
        g = p2[p2.corpus == c]
        ax.plot([g.util.min(), g.util.max()], [i, i], color=SERIES[c],
                lw=2.6, alpha=0.28, zorder=2, solid_capstyle="round")
    for tag, lab, mk, off in [("", "ESM-2 650M", "o", -0.15),
                              ("_esmc", "ESM-C 300M", "s", +0.15)]:
        p = pd.read_csv(f"{RES}/petml_placement{tag}.csv")
        m = p.groupby("corpus").util.mean()
        ax.scatter([m[c] for c in order], np.arange(4) + off,
                   s=72, marker=mk, c=[SERIES[c] for c in order],
                   edgecolors=SURFACE, linewidths=1.4, zorder=3)
        ax.scatter([], [], s=72, marker=mk, c=MUTED, edgecolors=SURFACE,
                   linewidths=1.4, label=lab)
    ax.axvline(0, color=INK, lw=1.2, zorder=1)
    ax.set_ylim(-0.75, 3.75)
    ax.annotate("selecting at random", xy=(0, 3.55), xytext=(10, 0),
                textcoords="offset points", fontsize=7, color=INK2,
                va="center", ha="left")
    ax.set_yticks(range(4))
    ax.set_yticklabels([f"{c}  ({int((p2.corpus == c).sum())})" for c in order],
                       fontsize=8, color=INK2)
    ax.set_xlabel("utility of taking the proxy's top 20%\n(1 = oracle, 0 = random)",
                  fontsize=8, color=INK2)
    ax.set_title("What a campaign receives", fontsize=8.5, color=INK, pad=7)
    leg = ax.legend(frameon=False, fontsize=7.5, loc="center right",
                    bbox_to_anchor=(1.02, 0.17), handletextpad=0.3,
                    borderpad=0.2, labelspacing=0.35)
    for txt in leg.get_texts():
        txt.set_color(INK2)

    ax = axes[1]
    tidy(ax)
    for c in order:
        g = p2[p2.corpus == c]
        ax.scatter(g.bulk, g.util, s=52, c=SERIES[c], edgecolors=SURFACE,
                   linewidths=1.3, zorder=3)
    ax.axhline(0, color=AXIS, lw=1.0, zorder=1)
    ax.axvline(0, color=AXIS, lw=1.0, zorder=1)
    pm = p2[p2.corpus == "PETML"]
    ax.annotate("PETML", xy=(pm.bulk.median(), pm.util.median()),
                xytext=(6, -16), textcoords="offset points", fontsize=7.5,
                color=INK)
    ax.set_xlabel("bulk Spearman", fontsize=8, color=INK2)
    ax.set_ylabel("top-20% utility", fontsize=8, color=INK2)
    ax.set_title("Per unit, all four corpora, same colours",
                 fontsize=8.5, color=INK, pad=7)

    fig.suptitle("On the corpus that started this project, the top 20% is a "
                 "random draw", fontsize=9.5, color=INK, y=0.985)
    fig.tight_layout(rect=[0, 0, 1, 0.945])
    save(fig, "fig_campaign", dest)


def fig_power(dest):
    fig, ax = plt.subplots(figsize=(4.6, 3.1))
    tidy(ax)
    CEIL = 260  # anything not powered at the largest simulated n sits here
    for tag, lab, c in [("", "ESM-2 650M", BLUE), ("_esmc", "ESM-C 300M", ORANGE)]:
        w = pd.read_csv(f"{RES}/power_petml{tag}.csv")
        deltas, needed, capped = [], [], []
        for d in sorted(w.delta.unique()):
            g = w[w.delta == d].sort_values("n")
            ok = g[g.power >= 0.8]
            deltas.append(d)
            needed.append(int(ok.n.iloc[0]) if len(ok) else CEIL)
            capped.append(not len(ok))
        ax.plot(deltas, needed, "-", color=c, lw=1.8, zorder=3, label=lab)
        deltas, needed, capped = np.array(deltas), np.array(needed), np.array(capped)
        ax.plot(deltas[~capped], needed[~capped], "o", color=c, ms=5,
                markeredgecolor=SURFACE, markeredgewidth=1.2, zorder=4)
        # Open markers where the simulation never reached 80% power, so the
        # ceiling is not read as a measured value.
        ax.plot(deltas[capped], needed[capped], "o", mfc=SURFACE, mec=c,
                mew=1.6, ms=5, zorder=4)
    ax.axhline(10, color=INK, lw=1.4, ls=(0, (4, 3)), zorder=2)
    ax.text(0.303, 10.8, "what the entire published record supplies (10 units)",
            fontsize=7, color=INK, ha="right", va="bottom")
    ax.axvline(0.115, color=MUTED, lw=1.0, ls=(0, (2, 3)), zorder=2)
    ax.text(0.111, 62, "the effect we\nwant to detect", fontsize=7, color=MUTED,
            ha="right", va="center", linespacing=1.3)
    ax.set_yscale("log")
    ax.set_ylim(8.4, 330)
    ax.set_yticks([10, 20, 40, 120, CEIL])
    ax.set_yticklabels(["10", "20", "40", "120", "never\nreached"])
    ax.set_xlabel("true difference between benchmark families", fontsize=8, color=INK2)
    ax.set_ylabel("units needed for 80% power", fontsize=8, color=INK2)
    ax.set_title("The measurement the literature cannot support",
                 fontsize=9, color=INK, pad=8)
    leg = ax.legend(frameon=False, fontsize=7.5, loc="upper right")
    for txt in leg.get_texts():
        txt.set_color(INK2)
    fig.tight_layout()
    save(fig, "fig_power", dest)


# -------------------------------------------------------------------- EIML

def fig_simpson(dest):
    t = pd.read_csv(f"{RES}/proxy_geometry.csv")
    col = "util_0.1"
    s = t[t[col].notna()].copy()
    s["r"] = resid(s, "bulk", col)

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.3))

    ax = axes[0]
    tidy(ax)
    for corpus in ["ProteinGym", "SSMuLA"]:
        g = s[s.corpus == corpus]
        rho, p = stats.spearmanr(g.score_skew, g[col])
        ax.scatter(g.score_skew, g[col], s=60, c=SERIES[corpus],
                   edgecolors=SURFACE, linewidths=1.4, zorder=3,
                   label=f"{corpus} ({len(g)})   $\\rho$ {rho:+.2f}")
        xs = np.linspace(g.score_skew.min(), g.score_skew.max(), 2)
        f = np.poly1d(np.polyfit(g.score_skew, g[col], 1))
        ax.plot(xs, f(xs), color=SERIES[corpus], lw=2.0, alpha=0.55, zorder=2)
    for corpus, mk in [("ProteinGym", "P"), ("SSMuLA", "P")]:
        g = s[s.corpus == corpus]
        ax.scatter([g.score_skew.mean()], [g[col].mean()], s=190, marker=mk,
                   c=SERIES[corpus], edgecolors=INK, linewidths=1.1, zorder=4)
    a = s[s.corpus == "ProteinGym"]
    b = s[s.corpus == "SSMuLA"]
    ax.annotate("", xy=(b.score_skew.mean(), b[col].mean()),
                xytext=(a.score_skew.mean(), a[col].mean()),
                arrowprops=dict(arrowstyle="->", color=INK, lw=1.3,
                                linestyle=(0, (4, 3))), zorder=4)
    ax.set_xlabel("skew of the proxy's own score distribution", fontsize=8, color=INK2)
    ax.set_ylabel("top-10% selection utility", fontsize=8, color=INK2)
    ax.set_title("Within each family, up. Between them, down.",
                 fontsize=8.5, color=INK, pad=7)
    leg = ax.legend(frameon=False, fontsize=7.5, loc="upper left",
                    handletextpad=0.4, borderpad=0.2, labelspacing=0.3)
    for txt in leg.get_texts():
        txt.set_color(INK2)
    ax.set_ylim(-0.14, 0.95)

    ax = axes[1]
    tidy(ax)
    d = pd.read_csv(f"{RES}/skew_prospective.csv")
    d = d[(d.k == "top-10%") & (d.model.str.startswith("skew (label"))]
    labels, vals = [], []
    for split, nice in [("LOO", "held out\nwithin the pool"),
                        ("ProteinGym->SSMuLA", "trained on PGym\napplied to SSMuLA"),
                        ("SSMuLA->ProteinGym", "trained on SSMuLA\napplied to PGym")]:
        row = d[d.split == split]
        if len(row):
            labels.append(nice)
            vals.append(row.rho.iloc[0])
    cols = [VIOLET if v >= 0 else ORANGE for v in vals]
    ax.barh(range(len(vals)), vals, 0.6, color=cols, edgecolor=SURFACE,
            linewidth=1.2, zorder=3)
    for i, v in enumerate(vals):
        ax.text(v + (0.02 if v >= 0 else -0.02), i, f"{v:+.2f}",
                va="center", ha="left" if v >= 0 else "right",
                fontsize=7.5, color=INK2)
    ax.axvline(0, color=AXIS, lw=1.2, zorder=1)
    ax.set_yticks(range(len(vals)))
    ax.set_yticklabels(labels, fontsize=7.5, color=INK2)
    ax.set_xlim(-0.80, 0.52)
    ax.set_xlabel("out-of-sample rank correlation\nof the proposed diagnostic",
                  fontsize=8, color=INK2)
    ax.set_title("Carried to a new family, it inverts",
                 fontsize=8.5, color=INK, pad=7)

    fig.suptitle("The evidence for a diagnostic, and the test that withdrew it",
                 fontsize=9.5, color=INK, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    save(fig, "fig_simpson", dest)


def main():
    tae = ["papers/tae/figs"]
    gem = ["papers/gem_bio/figs"]
    eiml = ["papers/eiml/figs"]
    fig_decay(tae)
    fig_replicate(tae)
    fig_campaign(gem)
    fig_power(gem)
    fig_simpson(eiml)


if __name__ == "__main__":
    main()
