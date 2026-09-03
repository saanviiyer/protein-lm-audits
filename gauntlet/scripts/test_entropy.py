#!/usr/bin/env python3
"""Confidence concentration, recovered from cached scores instead of a rescore.

    python scripts/test_entropy.py --scorer esm2 --out results

Phase 29 listed one untested proxy-side hypothesis: whether the model's
CONFIDENCE -- how peaked its amino-acid distribution is at each position --
explains the corpus offset. It was left out because the per-position
log-probability tables exist for SSMuLA and rubisco but not for ProteinGym,
whose cache holds only per-variant scores, and measuring it on two corpora and
not the third would compare exactly the two groups whose difference is the
question. The stated cost was rescoring 41 assays on the GPU.

That rescore is unnecessary. A wild-type-marginal score IS

    s(a) = log p(a | position masked) - log p(wt | position masked)

so at any position, exponentiating the scores of every substitution -- plus the
wild type, whose score is 0 by definition -- recovers p(a) over the twenty amino
acids up to a normalising constant. Softmax the scores at a position and the
distribution comes back exactly. Entropy follows. The information was in the
cache the whole time; only the shape was inconvenient.

This works because ProteinGym assays are near-complete single-mutant scans, so
most positions carry all 19 substitutions. Positions with fewer are dropped
rather than renormalised over a partial alphabet, since a distribution over 12
of 20 amino acids is not comparable to one over 20 -- ``--min-subs`` sets the
bar and coverage is reported.

VALIDATION. rubisco and SSMuLA have BOTH the cached scores and the true
log-probability tables, so the reconstruction is checked against ground truth
there before any of it is believed on ProteinGym.
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from gauntlet import proteingym as pg  # noqa: E402
from gauntlet.proxies import ESM2Marginals  # noqa: E402

AA = ESM2Marginals.AA_ORDER


def entropy_from_scores(scores_by_pos, min_subs):
    """Mean/SD entropy over positions, reconstructed by softmaxing the scores.

    ``scores_by_pos`` maps position -> {mut_aa: score}. The wild type is added
    at score 0, which is what the score definition makes it.
    """
    ents = []
    for pos, d in scores_by_pos.items():
        if len(d) < min_subs:
            continue
        vals = np.array(list(d.values()) + [0.0], float)   # +wt at 0
        if not np.isfinite(vals).all():
            continue
        p = np.exp(vals - vals.max())
        p /= p.sum()
        ents.append(float(-(p * np.log(p + 1e-12)).sum()))
    if not ents:
        return None
    return dict(entropy_mean=float(np.mean(ents)),
                entropy_sd=float(np.std(ents)),
                n_positions=len(ents))


def entropy_from_table(table):
    """Ground-truth entropy from real log-probability vectors (20-d, unnormalised)."""
    ents = []
    for v in table.values():
        v = np.asarray(v, float)
        p = np.exp(v - v.max())
        p /= p.sum()
        ents.append(float(-(p * np.log(p + 1e-12)).sum()))
    return float(np.mean(ents)), ents


def bulk_residual(bulk, util):
    """Utility with the linear trend on bulk correlation removed."""
    bulk = np.asarray(bulk, float)
    util = np.asarray(util, float)
    return util - np.polyval(np.polyfit(bulk, util, 1), bulk)


def boot_ci(fn, n, n_boot, seed, alpha=0.05):
    """Percentile bootstrap CI. ``fn`` maps a resample index array to a statistic.

    Units are the resampling unit throughout: one benchmark assay/landscape is
    drawn or not drawn as a whole. Anything fitted on the sample (the bulk
    residual) is refitted inside ``fn``, so the CI carries that fit's own
    uncertainty rather than conditioning on the point-estimate fit.
    """
    rng = np.random.default_rng(seed)
    vals = np.empty(n_boot)
    for b in range(n_boot):
        vals[b] = fn(rng.integers(0, n, n))
    lo, hi = np.nanpercentile(vals, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def published_correlations(t, pts, n_boot, seed):
    """The three correlations quoted in the paper, each with n, CI, p and seed.

    They do NOT share an n, and the reason is structural rather than a slip:

      * entropy-vs-skew is computed on the merged entropy table, where an assay
        survives only if some position clears ``--min-subs``. One ProteinGym
        assay (MTH3_HAEAE_Rockah-Shmuel_2015) fails that bar, so n = 59.
      * the two skew-vs-residual correlations are computed on the geometry
        table, which needs no per-position coverage at all, so n = 60 (41 of
        them ProteinGym).

    Dropping the assay from BOTH would understate the skew result on data that
    supports it; keeping it in both would require an entropy estimate that does
    not exist. The n is therefore printed on every line.
    """
    print("\n" + "=" * 74)
    print("PUBLISHED CORRELATIONS  (Spearman, units are the resampling unit)")
    print(f"  percentile bootstrap, {n_boot:,} resamples over units, seed {seed};")
    print("  the bulk residual is refitted inside every resample")
    print("=" * 74)

    out = {}

    # (1) entropy vs skew -- merged entropy table, coverage bar applies
    ent = t.entropy_mean.to_numpy(float)
    skew_e = t.score_skew.to_numpy(float)
    r1 = stats.spearmanr(ent, skew_e)
    ci1 = boot_ci(lambda i: stats.spearmanr(ent[i], skew_e[i]).statistic,
                  len(t), n_boot, seed)
    print(f"  entropy vs skew            n = {len(t):2d}  rho = {r1.statistic:+.4f}  "
          f"95% CI [{ci1[0]:+.2f}, {ci1[1]:+.2f}]  p = {r1.pvalue:.1e}")
    out["entropy_vs_skew"] = dict(n=len(t), rho=float(r1.statistic),
                                  ci_lo=ci1[0], ci_hi=ci1[1], p=float(r1.pvalue))

    # (2) skew vs bulk residual at k = 1% -- geometry table, no coverage bar
    g = pts[pts["util_0.01"].notna()].reset_index(drop=True)
    bulk = g.bulk.to_numpy(float)
    util = g["util_0.01"].to_numpy(float)
    skew_g = g.score_skew.to_numpy(float)
    is_pg = (g.corpus == "ProteinGym").to_numpy()
    resid = bulk_residual(bulk, util)
    r2 = stats.spearmanr(skew_g, resid)

    def stat2(i):
        return stats.spearmanr(skew_g[i], bulk_residual(bulk[i], util[i])).statistic

    ci2 = boot_ci(stat2, len(g), n_boot, seed)
    print(f"  skew vs residual (k=1%)    n = {len(g):2d}  rho = {r2.statistic:+.4f}  "
          f"95% CI [{ci2[0]:+.2f}, {ci2[1]:+.2f}]  p = {r2.pvalue:.1e}")
    out["skew_vs_residual_k1"] = dict(n=len(g), rho=float(r2.statistic),
                                      ci_lo=ci2[0], ci_hi=ci2[1], p=float(r2.pvalue))

    # (3) the same, within ProteinGym alone -- the test that killed dead fraction.
    # The residual is fitted across ALL units and read off the ProteinGym ones,
    # both for the point estimate and inside each resample.
    r3 = stats.spearmanr(skew_g[is_pg], resid[is_pg])

    def stat3(i):
        sel = is_pg[i]
        return stats.spearmanr(skew_g[i][sel],
                               bulk_residual(bulk[i], util[i])[sel]).statistic

    ci3 = boot_ci(stat3, len(g), n_boot, seed)
    print(f"  ... within ProteinGym      n = {int(is_pg.sum()):2d}  rho = {r3.statistic:+.4f}  "
          f"95% CI [{ci3[0]:+.2f}, {ci3[1]:+.2f}]  p = {r3.pvalue:.4f}")
    out["skew_vs_residual_within_pgym"] = dict(n=int(is_pg.sum()),
                                               rho=float(r3.statistic),
                                               ci_lo=ci3[0], ci_hi=ci3[1],
                                               p=float(r3.pvalue))

    missing = sorted(set(pts.unit) - set(t.unit))
    print(f"\n  n = {len(t)} vs n = {len(g)}: {len(missing)} unit(s) carry a skew but no "
          f"entropy estimate,\n  dropped by the minimum-substitutions-per-position bar "
          f"-- {', '.join(missing) if missing else 'none'}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pg_dir", default="data/proteingym")
    ap.add_argument("--scorer", default="esm2", choices=["esm2", "esmc"])
    ap.add_argument("--min-subs", type=int, default=17,
                    help="positions needing at least this many substitutions")
    ap.add_argument("--n-boot", type=int, default=20000,
                    help="bootstrap resamples for the published confidence intervals")
    ap.add_argument("--boot-seed", type=int, default=0)
    ap.add_argument("--out", default="results")
    args = ap.parse_args()
    tag = "" if args.scorer == "esm2" else "_esmc"
    pg_cache = os.path.join(args.pg_dir,
                            "esm_cache" if args.scorer == "esm2" else "esmc_cache")

    # ---- validate the reconstruction where ground truth exists -------------
    print("=" * 74)
    print("VALIDATION -- reconstructed entropy vs the true log-prob tables")
    print("=" * 74)
    rb = pd.read_csv(f"results/rubisco_scored_variants{tag}.csv")
    rb = rb[(rb.vmax_err <= 0.5) & (rb.kc_err <= 0.5)]
    by_pos = {}
    for r in rb.itertuples():
        by_pos.setdefault(r.position, {})[r.mut_residue] = r.esm2_wtm
    rec = entropy_from_scores(by_pos, args.min_subs)

    npz = np.load(f"data/rubisco/{'esm2_650M' if args.scorer == 'esm2' else 'esmc_300m'}"
                  "_logprobs.npz")
    truth_tbl = {int(p): v for p, v in zip(npz["positions"], npz["logprobs"])}
    truth_mean, truth_list = entropy_from_table(truth_tbl)

    # per-position comparison on the shared positions
    shared, a, b = [], [], []
    for pos, d in by_pos.items():
        if len(d) < args.min_subs or pos not in truth_tbl:
            continue
        vals = np.array(list(d.values()) + [0.0], float)
        p = np.exp(vals - vals.max()); p /= p.sum()
        a.append(-(p * np.log(p + 1e-12)).sum())
        v = np.asarray(truth_tbl[pos], float)
        q = np.exp(v - v.max()); q /= q.sum()
        b.append(-(q * np.log(q + 1e-12)).sum())
        shared.append(pos)
    a, b = np.array(a), np.array(b)
    print(f"  rubisco: {len(a)} positions compared")
    print(f"    reconstructed mean entropy {a.mean():.4f}   "
          f"true mean entropy {b.mean():.4f}")
    print(f"    max |difference| {np.abs(a - b).max():.2e}   "
          f"Pearson r {stats.pearsonr(a, b).statistic:.6f}")
    ok = np.abs(a - b).max() < 1e-6
    print(f"    -> reconstruction is {'EXACT' if ok else 'APPROXIMATE'}")

    # ---- compute for every unit -------------------------------------------
    rows = []
    ref = pg.select_assays(os.path.join(args.pg_dir, "reference.csv"))
    cached = {f[:-4] for f in os.listdir(pg_cache)}
    dropped = 0
    for r in ref[ref.DMS_id.isin(cached)].itertuples():
        records, y, _ = pg.load_assay(r.DMS_id, r.target_seq,
                                      os.path.join(args.pg_dir, "assays"))
        zs = np.load(os.path.join(pg_cache, f"{r.DMS_id}.npy"))
        if len(zs) != len(y):
            continue
        d = {}
        for rec_i, s in zip(records, zs):
            wt, pos, mt = rec_i["muts"][0]
            d.setdefault(pos, {})[mt] = float(s)
        e = entropy_from_scores(d, args.min_subs)
        if e is None:
            dropped += 1
            continue
        rows.append(dict(unit=r.DMS_id, corpus="ProteinGym", **e))
    print(f"\n  ProteinGym: {len(rows)} assays usable, {dropped} dropped for coverage")

    cache = json.load(open(f"data/ssmula/"
                           f"{'esm2_650M' if args.scorer == 'esm2' else 'esmc_300m'}"
                           "_logprobs.json"))
    for name, tbl in cache.items():
        m, ents = entropy_from_table({int(k): v for k, v in tbl.items()})
        rows.append(dict(unit=name, corpus="SSMuLA", entropy_mean=m,
                         entropy_sd=float(np.std(ents)), n_positions=len(ents)))

    for label in ["rubisco Vmax", "rubisco K_C", "rubisco fitness"]:
        rows.append(dict(unit=label, corpus="rubisco", entropy_mean=truth_mean,
                         entropy_sd=float(np.std(truth_list)),
                         n_positions=len(truth_list)))

    feat = pd.DataFrame(rows)
    pts = pd.read_csv(f"results/proxy_geometry{tag}.csv")
    t = pts.merge(feat, on=["unit", "corpus"], how="inner")
    t.to_csv(os.path.join(args.out, f"entropy{tag}.csv"), index=False)

    print(f"\n  merged {len(t)} units   "
          f"({', '.join(f'{c} {n}' for c, n in t.corpus.value_counts().items())})")
    print("  corpus mean entropy: "
          + str(t.groupby('corpus').entropy_mean.mean().round(3).to_dict()))
    print("  within-corpus SD:    "
          + str(t.groupby('corpus').entropy_mean.std().round(3).to_dict()))

    # ---- the same three tests every candidate has had to pass --------------
    for frac in ["0.01", "0.1"]:
        col = f"util_{frac}"
        sub = t[t[col].notna()].copy()
        fit = np.poly1d(np.polyfit(sub.bulk, sub[col], 1))
        sub["resid"] = sub[col] - fit(sub.bulk)
        pgm, ssm = sub[sub.corpus == "ProteinGym"], sub[sub.corpus == "SSMuLA"]
        gap0 = ssm.resid.mean() - pgm.resid.mean()

        print("\n" + "=" * 74)
        print(f"TOP {'1' if frac == '0.01' else '10'}%   "
              f"(corpus gap on bulk alone {gap0:+.3f})")
        print("=" * 74)
        for c in ["entropy_mean", "entropy_sd"]:
            r = stats.spearmanr(sub[c], sub.resid)
            X = np.column_stack([sub.bulk, sub[c], np.ones(len(sub))])
            beta, *_ = np.linalg.lstsq(X, sub[col].to_numpy(float), rcond=None)
            res2 = sub[col].to_numpy(float) - X @ beta
            cor = sub.corpus.to_numpy()
            aa, bb = res2[cor == "SSMuLA"], res2[cor == "ProteinGym"]
            gap, gp = aa.mean() - bb.mean(), stats.mannwhitneyu(aa, bb).pvalue
            wr = stats.spearmanr(pgm[c], pgm.resid)
            print(f"  {c:14s} vs resid {r.statistic:+.3f} (p={r.pvalue:.4f}) | "
                  f"gap after {gap:+.3f} (p={gp:.4f}, {100*(1-abs(gap)/abs(gap0)):+.0f}% closed) | "
                  f"within-PGym {wr.statistic:+.3f} (p={wr.pvalue:.3f})")

    # ---- the correlations the paper quotes, with n, CI, p and seed ---------
    stats_out = published_correlations(t, pts, args.n_boot, args.boot_seed)
    stats_path = os.path.join(args.out, f"entropy_correlations{tag}.json")
    with open(stats_path, "w") as fh:
        json.dump(dict(scorer=args.scorer, min_subs=args.min_subs,
                       n_boot=args.n_boot, boot_seed=args.boot_seed,
                       correlations=stats_out), fh, indent=2)

    print(f"\nwrote {args.out}/entropy{tag}.csv")
    print(f"wrote {stats_path}")


if __name__ == "__main__":
    main()
