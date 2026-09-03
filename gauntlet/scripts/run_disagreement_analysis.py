#!/usr/bin/env python3
"""When is adapting worth it? Test the disagreement hypothesis.

Phase 10: the planner has the best mean top-1% recall of any policy (0.146 vs
0.128 and 0.126) but beats each fixed policy in only 6 of 13 assays, p>0.28. Its
advantage is real on a few assays and absent on the rest. Chasing significance on
a global average needs far more assays to maybe confirm a small effect.

The sharper question: adapting can only pay when the options would order
DIFFERENT variants. If the supervised model and the zero-shot score already agree
on what to buy, choosing between them is free -- pick either. So the claim to
test is not "adaptation beats a fixed default" but "adaptation beats a fixed
default WHERE THE SCORERS DISAGREE", which is both easier to establish and
directly actionable.

Disagreement is measured on the CANDIDATE POOL from what a developer already has
-- no oracle, no held-out fitness -- so it can gate the decision at plan time:

    disagreement = 1 - Jaccard(top-b by supervised, top-b by zero-shot)

At every decision point this records that disagreement, then the true yield of
the planner, of always-supervised, and of always-zero-shot. If the hypothesis
holds, the planner's advantage should rise with disagreement, and a gate on it
should beat always-supervised.

    python scripts/run_disagreement_analysis.py
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats as sps
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from gauntlet.campaign import (elite_auroc, factorized_features,  # noqa: E402
                               loo_ridge_predictions)
from gauntlet import proteingym as pg  # noqa: E402

ASSAYS = ["B3VI55_LIPST_Klesmith_2015", "AMIE_PSEAE_Wrenbeck_2017",
          "A4GRB6_PSEAI_Chen_2020", "BLAT_ECOLX_Firnberg_2014",
          "KKA2_KLEPN_Melnikov_2014", "TPMT_HUMAN_Matreyek_2018",
          "TPK1_HUMAN_Weile_2017", "DYR_ECOLI_Thompson_plusLon_2019",
          "ESTA_BACSU_Nutschel_2020", "UBC9_HUMAN_Weile_2017",
          "RASH_HUMAN_Bandaru_2017", "NUD15_HUMAN_Suiter_2020",
          "CALM1_HUMAN_Weile_2017"]

MIN_FOR_TRUST, FOLDS, ALPHA, TRUST, TIE = 12, 5, 1.0, 0.55, 0.01


def oof(X, y):
    """Leave-one-out predictions: the deployment training size."""
    if len(y) < MIN_FOR_TRUST or np.std(y) < 1e-12:
        return None
    return loo_ridge_predictions(X, y, ALPHA)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pg_dir", default="data/proteingym")
    ap.add_argument("--out", default="results")
    ap.add_argument("--budget", type=int, default=48)
    ap.add_argument("--rounds", type=int, default=4)
    ap.add_argument("--seeds", type=int, default=40)
    ap.add_argument("--all_cached", action="store_true",
                    help="use every cached scan instead of the diagnostic set")
    args = ap.parse_args()

    if args.all_cached:
        ref = pg.select_assays(os.path.join(args.pg_dir, "reference.csv"))
        cached = {f[:-4] for f in os.listdir(os.path.join(args.pg_dir, "esm_cache"))}
        ref = ref[ref.DMS_id.isin(cached)].reset_index(drop=True)
    else:
        ref = pg.select_assays(os.path.join(args.pg_dir, "reference.csv"), ids=ASSAYS)
    rows = []

    for r in ref.itertuples():
        records, y, _ = pg.load_assay(r.DMS_id, r.target_seq,
                                      os.path.join(args.pg_dir, "assays"))
        cache = os.path.join(args.pg_dir, "esm_cache", f"{r.DMS_id}.npy")
        if not os.path.exists(cache):
            continue
        X, _ = factorized_features(records)
        zs = np.load(cache)
        k = max(1, int(round(0.01 * len(y))))
        elite = np.zeros(len(y), bool)
        elite[np.argsort(-y)[:k]] = True

        for seed in range(args.seeds):
            rng = np.random.default_rng(seed)
            observed = np.zeros(len(y), bool)
            y_obs = np.full(len(y), np.nan)

            for rnd in range(args.rounds):
                pool = np.flatnonzero(~observed)
                seen = np.flatnonzero(observed)
                if len(pool) <= args.budget or len(seen) < MIN_FOR_TRUST \
                        or np.std(y_obs[seen]) < 1e-12:
                    # advance by exploring; nothing decidable yet
                    take = rng.choice(pool, min(args.budget, len(pool)), replace=False)
                    observed[take] = True
                    y_obs[take] = y[take]
                    continue

                model = Ridge(alpha=ALPHA).fit(X[seen], y_obs[seen])
                sup_pick = pool[np.argsort(-model.predict(X[pool]))[:args.budget]]
                zs_pick = pool[np.argsort(-zs[pool])[:args.budget]]

                # --- disagreement, computable without any held-out fitness ----
                inter = len(set(sup_pick) & set(zs_pick))
                union = len(set(sup_pick) | set(zs_pick))
                disagree = 1.0 - inter / union
                rho_pool = sps.spearmanr(model.predict(X[pool]), zs[pool]).statistic

                # --- the planner's call ---------------------------------------
                pred = oof(X[seen], y_obs[seen])
                sup_a = elite_auroc(pred, y_obs[seen]) if pred is not None else np.nan
                zs_a = elite_auroc(zs[seen], y_obs[seen])
                if np.isfinite(sup_a) and sup_a >= TRUST and (
                        not (np.isfinite(zs_a) and zs_a >= TRUST) or sup_a + TIE >= zs_a):
                    chosen, pick = "supervised", sup_pick
                elif np.isfinite(zs_a) and zs_a >= TRUST:
                    chosen, pick = "zero_shot", zs_pick
                else:
                    chosen, pick = "explore", rng.choice(pool, args.budget, replace=False)

                rows.append({
                    "assay": r.DMS_id, "seed": seed, "round": rnd,
                    "disagree": disagree, "rho_pool": rho_pool, "chosen": chosen,
                    "sup_auroc": sup_a, "zs_auroc": zs_a,
                    "planner": float(elite[pick].sum()),
                    "always_sup": float(elite[sup_pick].sum()),
                    "always_zs": float(elite[zs_pick].sum()),
                    "explore_ev": args.budget * elite[pool].sum() / len(pool),
                })
                observed[pick] = True
                y_obs[pick] = y[pick]
        print(f"  {r.DMS_id}", flush=True)

    t = pd.DataFrame(rows)
    os.makedirs(args.out, exist_ok=True)
    t.to_csv(os.path.join(args.out, "disagreement.csv"), index=False)
    t["adv_sup"] = t.planner - t.always_sup
    t["adv_zs"] = t.planner - t.always_zs
    t["adv_best_fixed"] = t.planner - t[["always_sup", "always_zs"]].max(axis=1)

    print(f"\n{'='*74}\nDISAGREEMENT vs PLANNER ADVANTAGE — {len(t)} decisions\n{'='*74}")
    print(f"disagreement (1 - Jaccard of the two top-{args.budget} picks): "
          f"median {t.disagree.median():.3f}, range {t.disagree.min():.2f}-{t.disagree.max():.2f}")

    print("\nCorrelation of disagreement with the planner's advantage:")
    for c, lab in [("adv_sup", "over always-supervised"), ("adv_zs", "over always-zero-shot")]:
        rho = sps.spearmanr(t.disagree, t[c])
        print(f"  {lab:26s} rho={rho.statistic:+.3f}  p={rho.pvalue:.2e}")

    print("\nBy disagreement quartile (mean elites captured per round):")
    # with many assays the disagreement distribution concentrates near 1.0, so
    # quartile edges collapse; drop the duplicates and label whatever survives.
    q = pd.qcut(t.disagree, 4, duplicates="drop")
    names = ["Q1 agree", "Q2", "Q3", "Q4 disagree"]
    k = len(q.cat.categories)
    t["q"] = q.cat.rename_categories(names[:k] if k == 4 else
                                     [f"bin {i + 1}" for i in range(k)])
    g = t.groupby("q", observed=True).agg(
        n=("planner", "size"), disagree=("disagree", "mean"),
        planner=("planner", "mean"), always_sup=("always_sup", "mean"),
        always_zs=("always_zs", "mean"), adv_sup=("adv_sup", "mean"))
    print(g.round(3).to_string())

    print("\nPaired test of planner vs always-supervised, within quartile:")
    for q, d in t.groupby("q", observed=True):
        nz = d.adv_sup[d.adv_sup != 0]
        p = sps.wilcoxon(nz).pvalue if len(nz) > 5 else float("nan")
        print(f"  {str(q):14s} diff {d.adv_sup.mean():+.3f}  p={p:.4f}  "
              f"n={len(d)} ({len(nz)} differ)")

    print(f"\nGATING RULE — adapt only when disagreement exceeds a threshold,\n"
          f"otherwise just use the supervised model:")
    print(f"  {'threshold':>10s} {'% adapted':>10s} {'mean elites':>12s} {'vs always-sup':>14s}")
    base = t.always_sup.mean()
    print(f"  {'(none)':>10s} {0.0:>9.1f}% {base:>12.3f} {0.0:>+14.3f}")
    for thr in [0.5, 0.6, 0.7, 0.8, 0.9, 0.95]:
        use = t.disagree >= thr
        val = np.where(use, t.planner, t.always_sup).mean()
        print(f"  {thr:>10.2f} {100*use.mean():>9.1f}% {val:>12.3f} {val - base:>+14.3f}")
    print(f"  {'always':>10s} {100.0:>9.1f}% {t.planner.mean():>12.3f} "
          f"{t.planner.mean() - base:>+14.3f}")
    print(f"\nwrote {args.out}/disagreement.csv")


if __name__ == "__main__":
    main()
