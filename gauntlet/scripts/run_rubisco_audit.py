#!/usr/bin/env python3
"""Audit ESM-2 zero-shot against the rubisco DMS, scoring Vmax and K_C separately.

    python scripts/run_rubisco_audit.py --out results

Companion to ``run_petase_audit.py``. Same scorer, same protocol, a corpus
chosen because it lacks the three defects the PETase corpus has: every variant
is a single mutant, everything was measured in one lab under one protocol, and
there are two orthogonal kinetic axes instead of one.

WHY THE TWO AXES ARE SCORED SEPARATELY. The paper reports a growth-selection
``Fitness`` alongside Vmax (turnover) and K_C (the CO2 half-saturation
constant). Collapsing to fitness is what a design pipeline would do, and it
hides the question worth asking: a proxy that tracks turnover but is blind to
substrate affinity is a different object from one that works, and only a
two-axis corpus can tell them apart.

SIGN CONVENTION, which is easy to get backwards. K_C is a Michaelis constant --
LOWER means tighter CO2 binding, so lower is better. Every target here is
therefore oriented so that HIGHER IS BETTER, with ``kc_affinity = -K_C``. A
positive rho always means the proxy is working, on every row of every table.

THE BASELINES. ``n_mut`` -- the proxy that beat both language models on the
PETase corpus -- is constant at 1 here by construction, so it cannot be
evaluated and is not a candidate. That is the point of this dataset rather than
a gap in it. In its place the informative trivial baseline is ``wt_logp``: how
much the model liked the WILD-TYPE residue at that position, which knows
nothing whatsoever about which substitution was made. Gordon et al. (2024)
showed up to 65% of a language model's apparent zero-shot skill is explained by
exactly this quantity. If a score that cannot see the mutation ranks the
outcomes as well as the full score does, the full score is largely measuring
positional conservation rather than the effect of the edit.

ERROR FILTERING. ``Km_qbcov`` and ``Vmax_qbcov`` are quantile-based
coefficients of variation and are heavy-tailed (K_C's reaches 24.9 against a
median of 0.29). Every correlation is therefore reported at three thresholds.
A single unfiltered number averages precisely-measured variants together with
ones carrying almost no information; a number from an aggressive filter is
computed on a subset selected partly for being well-behaved. Reporting the
sweep is the honest option, and if the answer moves across it, that movement is
the result.
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from gauntlet import proxies  # noqa: E402

# higher is better for all three, by construction
TARGETS = ["vmax", "kc_affinity", "fitness"]
PROXIES = ["esm2_wtm", "wt_logp", "blosum62", "hydropathy"]
QBCOV = [None, 1.0, 0.5]
TOPK_FRACS = [0.01, 0.05, 0.10]


def read_fasta(path):
    return "".join(l.strip() for l in open(path) if not l.startswith(">"))


def partial_rho(proxy, y, nuisance):
    """Spearman between proxy and outcome with a nuisance variable removed.

    Rank-transform all three, then correlate the residuals of a linear fit on
    the nuisance ranks -- the rank analogue of a partial correlation. Used here
    to strip positional conservation (``wt_logp``) out of the ESM-2 score, so
    what remains is the part attributable to WHICH substitution was made rather
    than to how constrained the position is.
    """
    if nuisance.nunique() < 2 or proxy.nunique() < 2 or y.nunique() < 2:
        return np.nan
    a, b, c = (stats.rankdata(v) for v in (proxy, y, nuisance))
    ra = a - np.poly1d(np.polyfit(c, a, 1))(c)
    rb = b - np.poly1d(np.polyfit(c, b, 1))(c)
    if np.std(ra) == 0 or np.std(rb) == 0:
        return np.nan
    return float(stats.pearsonr(ra, rb).statistic)


def topk_utility(proxy, y, frac):
    """Fraction of achievable mean outcome captured by the proxy's top-k.

    1.0 is the optimal selector, 0.0 is a random draw, negative is worse than
    random. Normalising against both the random and the oracle draw makes the
    number comparable across assays with different dynamic range -- a raw
    "mean fitness of the top k" is not.
    """
    k = max(1, int(round(frac * len(y))))
    rand = y.mean()
    best = np.sort(y)[-k:].mean()
    got = y[np.argsort(-proxy)[:k]].mean()
    return np.nan if best == rand else float((got - rand) / (best - rand))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--campaign", default="data/rubisco/rubisco_campaign.csv")
    ap.add_argument("--fasta", default="data/rubisco/rubisco_wt.fasta")
    ap.add_argument("--out", default="results")
    ap.add_argument("--model", default=None)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--cache", default=None)
    ap.add_argument("--scorer", default="esm2", choices=["esm2", "esmc"],
                    help="ESM-C is a different architecture from a different lab, "
                         "so agreement between them is evidence about the class "
                         "of scorer rather than about one checkpoint")
    ap.add_argument("--tag", default=None, help="suffix for output files")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    if args.model is None:
        args.model = ("facebook/esm2_t33_650M_UR50D" if args.scorer == "esm2"
                      else "esmc_300m")
    if args.cache is None:
        args.cache = (f"data/rubisco/{'esm2_650M' if args.scorer == 'esm2' else 'esmc_300m'}"
                      "_logprobs.npz")
    if args.tag is None:
        args.tag = "" if args.scorer == "esm2" else "_esmc"

    df = pd.read_csv(args.campaign)
    wt = read_fasta(args.fasta)
    df["kc_affinity"] = -df["kc"]
    df["muts"] = [[(r.wt_residue, int(r.position), r.mut_residue)] for r in df.itertuples()]
    print(f"corpus: {len(df)} variants, {df.position.nunique()} positions, "
          f"WT {len(wt)} aa, all single mutants")

    df["blosum62"] = df["muts"].apply(proxies.blosum_score)
    df["hydropathy"] = df["muts"].apply(proxies.hydropathy_shift)

    positions = sorted(df.position.unique())
    aa_index = {a: i for i, a in enumerate(proxies.ESM2Marginals.AA_ORDER)}

    if os.path.exists(args.cache):
        z = np.load(args.cache)
        table = {int(p): v for p, v in zip(z["positions"], z["logprobs"])}
        print(f"loaded cached log-probs for {len(table)} positions from {args.cache}")
    else:
        print(f"loading {args.model} ...")
        cls = proxies.ESM2Marginals if args.scorer == "esm2" else proxies.ESMCMarginals
        scorer = cls(args.model, batch_size=args.batch_size)
        print(f"  device={scorer.device}; masking {len(positions)} positions")
        # 0-based indices into the sequence; the campaign file is 1-based.
        cache = scorer.logprobs_at(wt, [p - 1 for p in positions])
        table = {p: cache[p - 1] for p in positions if (p - 1) in cache}
        np.savez_compressed(args.cache,
                            positions=np.array(sorted(table)),
                            logprobs=np.stack([table[p] for p in sorted(table)]))
        print(f"  cached to {args.cache}")

    def esm_row(r):
        v = table.get(r.position)
        if v is None:
            return np.nan
        return float(v[aa_index[r.mut_residue]] - v[aa_index[r.wt_residue]])

    df["esm2_wtm"] = [esm_row(r) for r in df.itertuples()]
    # Knows the position but NOT the substitution -- the Gordon confound.
    df["wt_logp"] = [float(table[r.position][aa_index[r.wt_residue]])
                     if r.position in table else np.nan for r in df.itertuples()]

    df.drop(columns=["muts"]).to_csv(
        os.path.join(args.out, f"rubisco_scored_variants{args.tag}.csv"), index=False)

    rows, util_rows = [], []
    for thr in QBCOV:
        sub = df if thr is None else df[(df.vmax_err <= thr) & (df.kc_err <= thr)]
        for target in TARGETS:
            s = sub[sub[target].notna()]
            for p in PROXIES:
                d = s[s[p].notna()]
                if len(d) < 5 or d[p].nunique() < 2:
                    continue
                r = stats.spearmanr(d[p], d[target])
                rows.append({"qbcov_max": "none" if thr is None else thr,
                             "n": len(d), "target": target, "proxy": p,
                             "rho": r.statistic, "p": r.pvalue,
                             "rho_partial_wt_logp": partial_rho(
                                 d[p], d[target], d["wt_logp"])})
                for f in TOPK_FRACS:
                    util_rows.append({
                        "qbcov_max": "none" if thr is None else thr,
                        "target": target, "proxy": p, "frac": f,
                        "utility": topk_utility(d[p].to_numpy(),
                                                d[target].to_numpy(), f)})

    res = pd.DataFrame(rows)
    util = pd.DataFrame(util_rows)
    res.to_csv(os.path.join(args.out, f"rubisco_audit{args.tag}.csv"), index=False)
    util.to_csv(os.path.join(args.out, f"rubisco_topk{args.tag}.csv"), index=False)

    print("\n" + "=" * 72)
    print("CONTROL 3 -- does the proxy rank the measured outcome?")
    print("higher rho is better for every target (kc_affinity = -K_C)")
    print("=" * 72)
    for thr in QBCOV:
        lab = "none" if thr is None else thr
        blk = res[res.qbcov_max == lab]
        if blk.empty:
            continue
        piv = blk.pivot_table(index="proxy", columns="target", values="rho").reindex(PROXIES)
        print(f"\nqbcov filter: {lab}   (n = {blk.n.iloc[0]})")
        print(piv.reindex(columns=TARGETS).round(3).to_string())

    print("\n" + "=" * 72)
    print("CONTROL 1 -- does ESM-2 beat scorers that know no biology?")
    print("=" * 72)
    full = res[res.qbcov_max == "none"]
    for target in TARGETS:
        blk = full[full.target == target].set_index("proxy").rho
        if "esm2_wtm" not in blk:
            continue
        base = blk.drop("esm2_wtm")
        best = base.idxmax()
        print(f"  {target:12s} ESM-2 {blk['esm2_wtm']:+.3f}  vs best baseline "
              f"{best} {base[best]:+.3f}   margin {blk['esm2_wtm'] - base[best]:+.3f}")

    print("\n" + "=" * 72)
    print("CONTROL 4 -- how much survives removing positional conservation?")
    print("wt_logp cannot see the substitution, only how constrained the site is")
    print("=" * 72)
    for thr in QBCOV:
        lab = "none" if thr is None else thr
        blk = res[(res.qbcov_max == lab) & (res.proxy == "esm2_wtm")]
        if blk.empty:
            continue
        print(f"\nqbcov filter: {lab}   (n = {blk.n.iloc[0]})")
        for r in blk.itertuples():
            print(f"  {r.target:12s} raw {r.rho:+.3f}  ->  "
                  f"partial {r.rho_partial_wt_logp:+.3f}")

    print("\n" + "=" * 72)
    print("ELITE REGIME -- normalised top-k utility (1 = optimal, 0 = random)")
    print("=" * 72)
    for thr in QBCOV:
        lab = "none" if thr is None else thr
        blk = util[util.qbcov_max == lab]
        if blk.empty:
            continue
        print(f"\nqbcov filter: {lab}")
        for target in TARGETS:
            piv = blk[blk.target == target].pivot_table(
                index="proxy", columns="frac", values="utility").reindex(PROXIES)
            print(f"  {target}")
            print(piv.round(3).to_string().replace("\n", "\n  "))

    print(f"\nwrote {args.out}/rubisco_audit{args.tag}.csv and {args.out}/rubisco_topk{args.tag}.csv")


if __name__ == "__main__":
    main()
