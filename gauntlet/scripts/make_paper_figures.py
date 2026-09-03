#!/usr/bin/env python3
"""Generate every figure used in the two submissions, from the result CSVs.

Nothing here recomputes science -- it reads what the analysis scripts wrote, so a
figure can never drift from the number in the text. PDF output for LaTeX.

    python scripts/make_paper_figures.py
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

plt.rcParams.update({
    "font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 200, "savefig.bbox": "tight", "pdf.fonttype": 42,
})
HONEST, GAMED, NEUTRAL = "#1f6f4a", "#c1440e", "#8a8f98"


def wmean(df, col, wcol="n"):
    d = df[[col, wcol]].dropna()
    return float(np.average(d[col], weights=d[wcol])) if len(d) else np.nan


# ---------------------------------------------------------------- Verify-Agents
def fig_attack(res, out):
    """The centrepiece: the gate rewards the gamed scorer and gets nothing."""
    a = pd.read_csv(os.path.join(res, "verifier_redteam.csv"))
    d = pd.read_csv(os.path.join(res, "verifier_defence.csv"))
    order = ["honest_zero_shot", "leaked_25pct", "leaked_50pct",
             "leaked_75pct", "leaked_full"]
    lbl = {"honest_zero_shot": "honest", "leaked_25pct": "leaked 25%",
           "leaked_50pct": "leaked 50%", "leaked_75pct": "leaked 75%",
           "leaked_full": "leaked 100%"}
    g = a.groupby("scorer").agg(gate=("gate_auroc", "mean"),
                                passes=("passes_gate", "mean"),
                                yld=("true_yield", "mean"),
                                rnd=("random_yield", "mean"))
    g["adv"] = g.yld - g.rnd
    g = g.loc[[o for o in order if o in g.index]]
    cols = [HONEST] + [GAMED] * (len(g) - 1)

    fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.25))
    x = np.arange(len(g))

    ax[0].bar(x, g.gate, color=cols)
    ax[0].axhline(0.55, ls="--", lw=1, color="k")
    ax[0].text(len(g) - 0.4, 0.565, "trust gate", ha="right", fontsize=6.5)
    ax[0].set_ylabel("AUROC on observed data")
    ax[0].set_title("What the verifier sees")

    ax[1].bar(x, g.adv, color=cols)
    ax[1].axhline(0, lw=1, color="k")
    ax[1].set_ylabel("good candidates vs random")
    ax[1].set_title("What the agent actually gets")

    dd = d.groupby("scorer").agg(
        naive=("naive_gate", lambda v: float((v >= 0.55).mean())),
        fwd=("forward_gate", lambda v: float((v >= 0.55).mean())))
    keep = [k for k in ["honest_zero_shot", "leaked_50pct", "leaked_full"] if k in dd.index]
    dd = dd.loc[keep]
    xs = np.arange(len(dd))
    ax[2].bar(xs - 0.19, dd.naive, 0.38, label="naive gate", color=NEUTRAL)
    ax[2].bar(xs + 0.19, dd.fwd, 0.38, label="forward validation", color=HONEST)
    ax[2].set_xticks(xs)
    ax[2].set_xticklabels([lbl.get(i, i) for i in dd.index], rotation=20, ha="right")
    ax[2].set_ylabel("pass rate")
    ax[2].set_title("The defence")
    ax[2].legend(frameon=False)

    for a_ in ax[:2]:
        a_.set_xticks(x)
        a_.set_xticklabels([lbl[i] for i in g.index], rotation=20, ha="right")
    fig.tight_layout()
    p = os.path.join(out, "fig_attack.pdf")
    fig.savefig(p)
    fig.savefig(p.replace(".pdf", ".png"))
    print("  wrote", p)


def fig_policies(res, out):
    """41 tasks: the adaptive agent has the best mean and the worst consistency."""
    t = pd.read_csv(os.path.join(res, "proteingym_backtest.csv"))
    pol = [c for c in ["random", "zero_shot_esm2", "supervised_greedy",
                       "sup_eps10", "planner"] if c in t.columns]
    lbl = {"random": "random", "zero_shot_esm2": "zero-shot\nscorer",
           "supervised_greedy": "fit on own\ndata", "sup_eps10": "fit + 10%\nexplore",
           "planner": "adaptive\nagent"}
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.5))
    rng = np.random.default_rng(0)
    for i, c in enumerate(pol):
        v = t[c].to_numpy()
        col = GAMED if c == "planner" else NEUTRAL
        ax[0].scatter(i + rng.uniform(-.14, .14, len(v)), v, s=7, alpha=.5,
                      color=col, edgecolor="none")
        ax[0].plot([i - .3, i + .3], [v.mean()] * 2, lw=2, color="k")
        ax[0].text(i, v.max() + .02, f"{v.mean():.3f}", ha="center", fontsize=6.5)
    ax[0].set_xticks(range(len(pol)))
    ax[0].set_xticklabels([lbl[c] for c in pol])
    ax[0].set_ylabel("recall of true top 1%")
    ax[0].set_title(f"Per-task outcome ({len(t)} tasks)")

    beats = [100 * (t[c] > t["random"]).mean() for c in pol]
    ax[1].bar(range(len(pol)), beats,
              color=[GAMED if c == "planner" else NEUTRAL for c in pol])
    ax[1].axhline(50, ls="--", lw=1, color="k")
    ax[1].set_xticks(range(len(pol)))
    ax[1].set_xticklabels([lbl[c] for c in pol])
    ax[1].set_ylabel("% of tasks beating random")
    ax[1].set_title("Consistency, not average")
    fig.tight_layout()
    p = os.path.join(out, "fig_policies.pdf")
    fig.savefig(p)
    fig.savefig(p.replace(".pdf", ".png"))
    print("  wrote", p)


# ------------------------------------------------------------------- AI4Science
def fig_confound(res, out):
    """Raw scores differ between models; corrected scores do not."""
    rows = []
    for tag, model in [("", "ESM-2 650M"), ("_esmc", "ESM-C 300M")]:
        for target, nice in [("logActivity", "activity"), ("Tm", "stability")]:
            f = os.path.join(res, f"partial_{target}{tag}.csv")
            if not os.path.exists(f):
                continue
            d = pd.read_csv(f)
            rows.append({"model": model, "target": nice,
                         "raw": wmean(d, "raw_rho"),
                         "corrected": wmean(d, "partial_rho")})
    t = pd.DataFrame(rows)
    fig, ax = plt.subplots(1, 2, figsize=(6.4, 2.5), sharey=True)
    for j, tgt in enumerate(["activity", "stability"]):
        s = t[t.target == tgt]
        x = np.arange(len(s))
        ax[j].bar(x - 0.19, s.raw, 0.38, label="raw", color=GAMED)
        ax[j].bar(x + 0.19, s.corrected, 0.38, label="mutation count removed",
                  color=HONEST)
        ax[j].axhline(0, lw=1, color="k")
        ax[j].set_xticks(x)
        ax[j].set_xticklabels(s.model)
        ax[j].set_title(tgt)
        for xi, (r, c) in enumerate(zip(s.raw, s.corrected)):
            ax[j].text(xi - 0.19, r + (0.012 if r >= 0 else -0.03),
                       f"{r:+.3f}", ha="center", fontsize=6.5)
            ax[j].text(xi + 0.19, c + 0.012, f"{c:+.3f}", ha="center", fontsize=6.5)
    ax[0].set_ylabel("Spearman vs measured outcome")
    ax[0].legend(frameon=False, loc="lower left")
    fig.suptitle("Raw scores differ between models; corrected scores agree",
                 fontsize=9)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    p = os.path.join(out, "fig_confound.pdf")
    fig.savefig(p)
    fig.savefig(p.replace(".pdf", ".png"))
    print("  wrote", p)


def fig_premise(res, out):
    """Per-study correlations behind Table 1: is the mean carried by a few?"""
    ci = pd.read_csv(os.path.join(res, "bootstrap_cis.csv"))
    rows = [("mutation count", "n_mut", "esm2", GAMED),
            ("hydropathy", "hydropathy", "esm2", NEUTRAL),
            ("BLOSUM62", "blosum62", "esm2", NEUTRAL),
            ("ESM-C 300M", "esm2_wtm", "esmc", HONEST),
            ("ESM-2 650M", "esm2_wtm", "esm2", HONEST)]
    key = {"zero-shot pLM": None, "BLOSUM62": "blosum62",
           "hydropathy": "hydropathy", "mutation count": "n_mut"}
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.6), sharey=True)
    rng = np.random.default_rng(0)

    for a, (tgt, fname, label) in zip(ax, [
            ("activity", "logActivity", "activity (24 studies)"),
            ("stability", "Tm", "stability (14 studies)")]):
        for i, (name, col, fam, colour) in enumerate(rows):
            suffix = "_esmc" if fam == "esmc" else ""
            d = pd.read_csv(os.path.join(res, f"per_study_{fname}{suffix}.csv"))
            v = d[[col, "n"]].dropna()
            y = i + rng.uniform(-0.13, 0.13, len(v))
            a.scatter(v[col], y, s=3 + 22 * (v.n / v.n.max()) ** 0.5,
                      color=colour, alpha=0.45, lw=0, zorder=2)
            # the weighted mean and its cluster-bootstrap interval, from Table 1
            model = "ESM-C 300M" if fam == "esmc" else "ESM-2 650M"
            scorer = "zero-shot pLM" if col == "esm2_wtm" else \
                     {v2: k for k, v2 in key.items() if v2}[col]
            r = ci[(ci.model == model) & (ci.target == tgt) & (ci.scorer == scorer)]
            if len(r):
                r = r.iloc[0]
                a.plot([r.lo, r.hi], [i, i], lw=1.6, color=colour, zorder=3)
                a.plot([r["mean"]], [i], marker="D", ms=4.5, color=colour,
                       zorder=4, markeredgecolor="white", markeredgewidth=0.6)
        a.axvline(0, lw=1, color="k", zorder=1)
        a.set_xlim(-1.08, 1.08)
        a.set_xlabel("within-study Spearman")
        a.set_title(label, fontsize=8.5)
    ax[0].set_yticks(range(len(rows)))
    ax[0].set_yticklabels([r[0] for r in rows])
    ax[0].set_ylim(-0.6, len(rows) - 0.4)
    fig.tight_layout()
    p = os.path.join(out, "fig_premise.pdf")
    fig.savefig(p)
    fig.savefig(p.replace(".pdf", ".png"))
    print("  wrote", p)


def fig_mechanism(res, out):
    """Flat within a fixed mutation count; collapses when counts are mixed --
    on the mean (left) and scan by scan (right)."""
    mx = pd.read_csv(os.path.join(res, "mechanism_mixed.csv"))
    fx = pd.read_csv(os.path.join(res, "mechanism_fixed.csv"))
    fig, ax = plt.subplots(1, 2, figsize=(6.6, 2.5))

    for cond, col, mark in [("random", NEUTRAL, "o"), ("beneficial", GAMED, "s")]:
        f = fx[fx.condition == cond].groupby("k").rho_sum.mean()
        m = mx[mx.condition == cond].groupby("max_k").rho_sum.mean()
        ax[0].plot(f.index, f.values, marker=mark, ms=3.5, lw=1.4, color=col,
                   label=f"fixed count ({cond})")
        ax[0].plot(m.index, m.values, marker=mark, ms=3.5, lw=1.4, ls="--",
                   color=col, alpha=0.75, label=f"mixed counts ({cond})")
    ax[0].axhline(0, lw=1, color="k")
    ax[0].set_xlabel("mutations per variant, $k$")
    ax[0].set_ylabel("Spearman vs truth")
    ax[0].set_title("Pooling across mutation counts\ndistorts score-fitness correlations",
                    fontsize=8.5)
    ax[0].legend(frameon=False, fontsize=6.2, loc="lower left")

    # right: every scan, single-mutant pool against the mixed-count pool
    for cond, col, mark in [("random", NEUTRAL, "o"), ("beneficial", GAMED, "s")]:
        d = mx[mx.condition == cond]
        x = d[d.max_k == 1].set_index("assay").rho_sum
        y = d[d.max_k == 6].set_index("assay").rho_sum
        k = sorted(set(x.dropna().index) & set(y.dropna().index))
        fell = int((y.reindex(k) < x.reindex(k)).sum())
        ax[1].scatter(x.reindex(k), y.reindex(k), s=13, color=col, marker=mark,
                      alpha=0.8, zorder=3,
                      label=f"{cond}: {fell}/{len(k)} fall")
    lim = (-0.85, 0.95)
    ax[1].plot(lim, lim, ls=":", lw=1, color="k", zorder=1)
    ax[1].axhline(0, lw=1, color="k", zorder=1)
    ax[1].set_xlim(lim); ax[1].set_ylim(lim)
    ax[1].set_xlabel("single-mutant pool ($k=1$)")
    ax[1].set_ylabel("mixed pool (counts 1 to 6)")
    ax[1].set_title("Per scan, not just on average", fontsize=8.5)
    ax[1].legend(frameon=False, fontsize=6.2, loc="upper left")

    fig.tight_layout()
    p = os.path.join(out, "fig_mechanism.pdf")
    fig.savefig(p)
    fig.savefig(p.replace(".pdf", ".png"))
    print("  wrote", p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--papers", default="papers")
    args = ap.parse_args()
    va = os.path.join(args.papers, "verify_agents", "figs")
    ai = os.path.join(args.papers, "ai4science_verification", "figs")
    os.makedirs(va, exist_ok=True)
    os.makedirs(ai, exist_ok=True)

    print("Verify-Agents:")
    fig_attack(args.results, va)
    fig_policies(args.results, va)
    print("AI4Science:")
    fig_confound(args.results, ai)
    fig_mechanism(args.results, ai)


if __name__ == "__main__":
    main()
