"""Arm A: re-score cached function proxies at screening operating points.

Design work reports bulk Spearman and top-k enrichment. A screening deployment runs
at a fixed, very low false-positive budget and needs sensitivity there. This computes
TPR at FPR in {1%, 0.1%, 0.01%} for every cached ProteinGym assay and asks whether the
bulk number predicts it.

Reads gauntlet's caches in place. No downloads, no GPU, no hazardous material.

Gates (PREREG.md): G1 bulk reproduction, G2 estimability, G3 permutation null.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

GAUNTLET = Path(__file__).resolve().parents[2] / "gauntlet"
FPR_TARGETS = (0.01, 0.001, 0.0001)


def tpr_at_fpr(scores, labels, fpr_target):
    """Highest TPR achievable without exceeding fpr_target.

    Returns (tpr, threshold, realised_fpr, n_neg_needed). Ties are handled by
    thresholding at score >= t, so the realised FPR can undershoot the target.
    """
    neg = np.sort(scores[labels == 0])[::-1]
    pos = scores[labels == 1]
    n_neg = neg.size
    if n_neg == 0 or pos.size == 0:
        return np.nan, np.nan, np.nan, n_neg
    # allow at most floor(fpr_target * n_neg) negatives above threshold
    k = int(np.floor(fpr_target * n_neg))
    if k == 0:
        # threshold strictly above the largest negative
        thresh = np.nextafter(neg[0], np.inf)
    else:
        thresh = neg[k - 1]
    realised = float((neg >= thresh).mean())
    tpr = float((pos >= thresh).mean())
    return tpr, float(thresh), realised, n_neg


def load_assay(dms_id, scorer_dir):
    cache = GAUNTLET / "data/proteingym" / scorer_dir / f"{dms_id}.npy"
    assay = GAUNTLET / "data/proteingym/assays" / f"{dms_id}.csv"
    if not (cache.exists() and assay.exists()):
        return None
    scores = np.load(cache, allow_pickle=True).astype(float)
    df = pd.read_csv(assay)
    if len(df) != len(scores):
        return None
    df["proxy"] = scores
    df = df.dropna(subset=["proxy", "DMS_score", "DMS_score_bin"])
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scorer", choices=["esm2", "esmc"], default="esm2")
    ap.add_argument("--permute", action="store_true",
                    help="G3: shuffle labels, expect TPR to land at the nominal FPR")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min-neg", type=int, default=20,
                    help="G2: assays with fewer negatives are dropped and reported")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    scorer_dir = "esm_cache" if args.scorer == "esm2" else "esmc_cache"
    ref = pd.read_csv(GAUNTLET / "data/proteingym/reference.csv")
    rng = np.random.default_rng(args.seed)

    rows, dropped = [], []
    for dms_id in ref["DMS_id"].dropna().unique():
        df = load_assay(dms_id, scorer_dir)
        if df is None or df.empty:
            dropped.append((dms_id, "no cache or length mismatch"))
            continue

        labels = df["DMS_score_bin"].to_numpy().astype(int)
        scores = df["proxy"].to_numpy()
        if args.permute:
            labels = rng.permutation(labels)

        n_neg = int((labels == 0).sum())
        n_pos = int((labels == 1).sum())
        # G2: FPR=1% is not estimable with fewer than 100 negatives; require min-neg
        # to even attempt, and record which targets are unreachable.
        if n_neg < args.min_neg or n_pos == 0:
            dropped.append((dms_id, f"G2: n_neg={n_neg}, n_pos={n_pos}"))
            continue

        rho = spearmanr(scores, df["DMS_score"]).statistic
        row = {"DMS_id": dms_id, "n": len(df), "n_pos": n_pos, "n_neg": n_neg,
               "bulk_spearman": rho}
        for target in FPR_TARGETS:
            tpr, thresh, realised, _ = tpr_at_fpr(scores, labels, target)
            tag = f"{target:g}"
            row[f"tpr_at_fpr_{tag}"] = tpr
            row[f"realised_fpr_{tag}"] = realised
            # a target finer than 1/n_neg cannot be distinguished from zero FPR
            row[f"estimable_{tag}"] = bool(target * n_neg >= 1)
        rows.append(row)

    res = pd.DataFrame(rows).sort_values("bulk_spearman", ascending=False)
    suffix = f"_{args.scorer}" + ("_permuted" if args.permute else "")
    out = Path(args.out) if args.out else (
        Path(__file__).resolve().parents[1] / "results" / f"screening_op{suffix}.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(out, index=False)

    summary = {"scorer": args.scorer, "permuted": args.permute,
               "n_assays": len(res), "n_dropped": len(dropped)}
    for target in FPR_TARGETS:
        tag = f"{target:g}"
        col, est = f"tpr_at_fpr_{tag}", f"estimable_{tag}"
        sub = res[res[est]]
        summary[f"median_tpr_at_fpr_{tag}"] = float(sub[col].median()) if len(sub) else None
        summary[f"n_estimable_{tag}"] = int(len(sub))
        if len(sub) > 2:
            r = spearmanr(sub["bulk_spearman"], sub[col])
            summary[f"bulk_predicts_tpr_{tag}_rho"] = float(r.statistic)
            summary[f"bulk_predicts_tpr_{tag}_p"] = float(r.pvalue)

    print(json.dumps(summary, indent=2))
    if dropped:
        print(f"\nDropped {len(dropped)} assays (G2 and cache misses), reported not hidden:")
        for dms_id, why in dropped:
            print(f"  {dms_id}: {why}")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
