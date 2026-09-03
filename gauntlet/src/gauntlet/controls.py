"""The audit battery.

Each control answers one question about a proxy that its headline correlation
cannot. They are deliberately adversarial: the null hypothesis is always that
the proxy is measuring something other than the biology it claims to measure.
"""

import numpy as np
import pandas as pd
from scipy import stats


def within_study_rank(df, proxy_cols, target_col, min_n=5):
    """Control 3 -- does the proxy rank measured outcomes?

    Correlations are computed inside each study because PET assay conditions
    (crystallinity, solids loading, temperature) vary enough between papers
    that pooled activity numbers are not comparable; Wei et al. Nat Commun
    16:4684 (2025) make this the field's standard caveat. Pooling inflates
    correlation by turning a between-assay offset into apparent signal.
    """
    rows = []
    for study, grp in df.groupby("study"):
        sub = grp[grp[target_col].notna()]
        if len(sub) < min_n:
            continue
        rec = {"study": study, "n": len(sub)}
        for col in proxy_cols:
            s = sub[sub[col].notna()]
            if len(s) < min_n or s[col].nunique() < 2 or s[target_col].nunique() < 2:
                rec[col] = np.nan
                continue
            rec[col] = stats.spearmanr(s[col], s[target_col]).statistic
        rows.append(rec)
    return pd.DataFrame(rows)


def pooled_summary(per_study, proxy_cols):
    """Sample-size-weighted mean rho, plus a sign test across studies.

    The sign test is the honest headline: a proxy that genuinely tracks the
    property should point the right way in most studies, not merely average
    above zero on the back of one large one.
    """
    out = []
    for col in proxy_cols:
        v = per_study[["n", col]].dropna()
        if v.empty:
            continue
        rho = v[col].to_numpy()
        n = v["n"].to_numpy()
        pos = int((rho > 0).sum())
        binom = stats.binomtest(pos, len(rho), 0.5)
        out.append(
            {
                "proxy": col,
                "studies": len(rho),
                "weighted_mean_rho": float(np.average(rho, weights=n)),
                "median_rho": float(np.median(rho)),
                "studies_positive": pos,
                "sign_test_p": float(binom.pvalue),
            }
        )
    return pd.DataFrame(out).sort_values("weighted_mean_rho", ascending=False)


def trivial_baseline_margin(summary, learned, baselines):
    """Control 1 -- does the learned proxy beat proxies that know no biology?

    If a substitution matrix or a bare mutation count matches the language
    model, the language model's apparent skill is not evidence that it
    understands the enzyme.
    """
    s = summary.set_index("proxy")["weighted_mean_rho"]
    if learned not in s.index:
        return None
    best = max(((b, s[b]) for b in baselines if b in s.index),
               key=lambda kv: abs(kv[1]), default=(None, 0.0))
    return {
        "learned": learned,
        "learned_rho": float(s[learned]),
        "best_baseline": best[0],
        "best_baseline_rho": float(best[1]),
        "margin": float(abs(s[learned]) - abs(best[1])),
    }


def partial_out(df, proxy_col, target_col, nuisance_col, min_n=5):
    """Control 4 -- how much survives once a nuisance variable is removed?

    Zero-shot scores are sums over mutated positions, so they are mechanically
    correlated with how many mutations a variant carries. Any proxy whose
    signal vanishes under this partial correlation was counting edits.
    """
    rows = []
    for study, grp in df.groupby("study"):
        sub = grp[[proxy_col, target_col, nuisance_col]].dropna()
        if len(sub) < min_n or sub[nuisance_col].nunique() < 2:
            continue
        r = {c: stats.rankdata(sub[c]) for c in sub.columns}
        rxy = np.corrcoef(r[proxy_col], r[target_col])[0, 1]
        rxz = np.corrcoef(r[proxy_col], r[nuisance_col])[0, 1]
        ryz = np.corrcoef(r[target_col], r[nuisance_col])[0, 1]
        denom = np.sqrt((1 - rxz**2) * (1 - ryz**2))
        rows.append(
            {
                "study": study,
                "n": len(sub),
                "raw_rho": rxy,
                "partial_rho": (rxy - rxz * ryz) / denom if denom > 1e-9 else np.nan,
                "proxy_vs_nuisance": rxz,
            }
        )
    return pd.DataFrame(rows)
