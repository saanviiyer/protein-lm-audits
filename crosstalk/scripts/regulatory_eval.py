"""Shared evaluation harness for the regulatory arm.

One protocol for every feature set, trivial or neural: L2 logistic regression on
standardised features, regularisation chosen by grouped 3-fold CV on the training
rows only, groups being locus clusters so the inner selection cannot leak either.
The only feature set that skips the regression is the PWM, which is a count and
has nothing to tune.
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import matthews_corrcoef, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

C_GRID = (0.003, 0.03, 0.3, 3.0)


def fit_probe(Xtr, ytr, groups, seed=0, c_grid=C_GRID):
    sc = StandardScaler().fit(Xtr)
    Xs = sc.transform(Xtr)
    n_groups = len(np.unique(groups))
    best, best_c = -np.inf, c_grid[0]
    if len(c_grid) > 1 and n_groups >= 3:
        gkf = GroupKFold(n_splits=3)
        for C in c_grid:
            scores = []
            for tr, va in gkf.split(Xs, ytr, groups):
                if len(np.unique(ytr[tr])) < 2:
                    continue
                m = LogisticRegression(C=C, max_iter=3000, random_state=seed)
                m.fit(Xs[tr], ytr[tr])
                scores.append(_auc(ytr[va], _decision(m, Xs[va])))
            if scores and np.mean(scores) > best:
                best, best_c = float(np.mean(scores)), C
    m = LogisticRegression(C=best_c, max_iter=5000, random_state=seed).fit(Xs, ytr)
    return sc, m, best_c


def _decision(model, X):
    p = model.predict_proba(X)
    return p[:, 1] if p.shape[1] == 2 else p


def _auc(y, s):
    """Macro one-vs-rest AUC, computed per class column.

    sklearn's multi_class="ovr" path demands rows that sum to 1, which a PWM
    log-odds or a raw decision function does not satisfy, and normalising rows to
    satisfy it would change the cross-row ordering that AUC actually reads. Since
    macro OvR AUC is by definition the mean of per-class one-vs-rest AUCs, compute
    it that way and the constraint disappears.
    """
    y = np.asarray(y)
    if np.ndim(s) == 1:
        if len(np.unique(y)) < 2:
            return np.nan
        return float(roc_auc_score(y, s))
    classes = np.unique(y)
    if s.shape[1] != len(classes):
        return np.nan
    aucs = []
    for j, c in enumerate(classes):
        yb = (y == c).astype(int)
        if yb.sum() in (0, len(yb)):
            continue
        aucs.append(roc_auc_score(yb, s[:, j]))
    return float(np.mean(aucs)) if aucs else np.nan


def tpr_at_fpr(y_bin, score, fpr_budget):
    """TPR at a fixed false-positive budget: the coverage quantity of section 23,
    which is where an AUC of 0.804 turned into 7.7% of the viable set."""
    neg = np.sort(score[y_bin == 0])[::-1]
    k = int(np.floor(fpr_budget * len(neg)))
    thr = float(neg[0]) + 1e-12 if k <= 0 else float(neg[k - 1])
    flagged = score >= thr
    return float((flagged & (y_bin == 1)).sum() / max((y_bin == 1).sum(), 1)), thr, flagged


def cluster_coverage(y_bin, score, clusters, fpr_budget):
    """Fraction of positive-bearing loci with at least one flagged positive, and
    the longest run of consecutive missed positives in genomic order.

    Per-instance recall counts a locus twice if it has two windows. A screen that
    finds one window per locus has covered the locus; a screen whose misses line
    up along the chromosome has left a contiguous hole. Both are invisible to AUC.
    """
    tpr, thr, flagged = tpr_at_fpr(y_bin, score, fpr_budget)
    pos = y_bin == 1
    cl = clusters[pos]
    fl = flagged[pos]
    covered = {c for c, f in zip(cl, fl) if f}
    ncl = len(set(cl))
    run = best = 0
    for f in fl:
        run = 0 if f else run + 1
        best = max(best, run)
    return dict(tpr=tpr, cluster_coverage=len(covered) / max(ncl, 1),
                n_pos_clusters=ncl, longest_miss_run=best)


def evaluate(scores, y, y_bin, clusters):
    out = {"auc": _auc(y, scores)}
    if np.ndim(scores) == 1:
        pred = (scores >= np.median(scores)).astype(int)
    else:
        pred = scores.argmax(1)
    out["mcc"] = float(matthews_corrcoef(y, pred))
    s1 = scores if np.ndim(scores) == 1 else scores.max(1) * 0 + scores[np.arange(len(y)), 1]
    for b in (0.01, 0.10):
        cc = cluster_coverage(y_bin, s1, clusters, b)
        out[f"tpr@{int(b*100)}"] = cc["tpr"]
        out[f"clustercov@{int(b*100)}"] = cc["cluster_coverage"]
        out[f"missrun@{int(b*100)}"] = cc["longest_miss_run"]
    return out


def cluster_bootstrap_auc(y, scores, clusters, n=200, seed=0):
    """Resample locus clusters, not rows. ProteinGym taught this the hard way."""
    rng = np.random.default_rng(seed)
    uc = np.unique(clusters)
    idx_by = {c: np.where(clusters == c)[0] for c in uc}
    vals = []
    for _ in range(n):
        pick = rng.choice(uc, size=len(uc), replace=True)
        idx = np.concatenate([idx_by[c] for c in pick])
        try:
            vals.append(_auc(y[idx], scores[idx] if np.ndim(scores) == 1 else scores[idx]))
        except Exception:
            pass
    vals = np.asarray([v for v in vals if np.isfinite(v)])
    if len(vals) < 20:
        return (np.nan, np.nan)
    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)))
