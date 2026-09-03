"""Deciding what to order next, and whether any model has earned that decision.

The measured result behind this module: across 13 DMS assays and 5 PET-hydrolase
campaigns, no single scoring strategy is safe. A zero-shot language model gave
both the best result observed (top-1% recall 0.409 against random 0.095) and the
worst (0.000, and 0.109 against random 0.851 on a multi-mutant campaign). A
ridge regression on the developer's own measurements was never the best and
never catastrophic, beating random in 12/13 assays and 5/5 campaigns.

So the useful thing a planner can do is not pick a favourite. It is to work out
which regime this campaign is in, say plainly whether anything has yet earned
trust, and fall back to exploration when nothing has.
"""

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold

from .campaign import (elite_auroc, elite_k, factorized_features,
                       loo_ridge_predictions, sequence_features,
                       top_decile_enrichment)
from .proxies import KD_HYDROPATHY, blosum_score, hydropathy_shift

MIN_FOR_TRUST = 12          # below this, cross-validation is not informative
#: The gate is AUROC, not enrichment. Both were compared on 2,080 decision
#: points across 13 ProteinGym assays with an oracle (FINDINGS.md, Phase 9):
#: AUROC at 0.55 won the 7 assays held out of the comparison (-0.117 regret,
#: p<1e-4) and was best pooled (-0.053, p<1e-4, ~9% less regret than the 1.5x
#: enrichment gate). Enrichment is still REPORTED because "2.5x better than
#: random" is what a developer can act on; it is just too quantised to decide on.
TRUST_AUROC = 0.55          # P(a top-decile member outranks a non-member)
TIE_AUROC = 0.01            # tie margin on the AUROC scale
TRUST_ENRICH = 1.5          # reported, no longer the gate
TRUST_RHO = 0.20            # secondary: out-of-fold Spearman, reported not enforced
TOP_FRAC = 0.10             # "good" means the top this fraction of the pool
MIN_ELITE = 5               # below this many elites the enrichment is noise
CONFOUND_RHO = 0.30         # |rho(proxy, n_mut)| above which we flag the confound
TIE_MARGIN = 0.05           # supervised wins near-ties; see module docstring
#: Never rank candidates by these. Mutation count predicts fitness well inside a
#: published campaign purely because later, more-mutated variants were selected
#: to be better -- it measures campaign progression, not merit, and ordering by
#: it would just pick whatever carries the most edits.
NOT_SELECTABLE = {"n_mut", "seq_len"}


def features(df, mode):
    """Design-space features, dispatched on campaign mode."""
    if mode == "sequence":
        return sequence_features(df.sequence.tolist())[0]
    return factorized_features([{"muts": m} for m in df.muts])[0]


def nuisance_col(mode):
    """The variable most likely to masquerade as signal in each regime.

    In a variant campaign it is mutation count; in a mining campaign over
    unrelated proteins there are no mutations, and the analogous nuisance is
    sequence length (with composition close behind).
    """
    return "seq_len" if mode == "sequence" else "n_mut"


def add_cheap_proxies(df, mode="variant"):
    df = df.copy()
    if mode == "sequence":
        df["seq_len"] = df.sequence.str.len()
        df["mean_hydropathy"] = [
            np.mean([KD_HYDROPATHY.get(c, 0.0) for c in str(s)]) if len(str(s)) else np.nan
            for s in df.sequence]
        return df
    df["blosum62"] = df.muts.apply(blosum_score)
    df["hydropathy"] = df.muts.apply(hydropathy_shift)
    # A campaign file may already carry precomputed model scores; derive the
    # per-mutation form so the confound correction is available either way.
    if "esm2_wtm" in df.columns and "esm2_per_mut" not in df.columns:
        df["esm2_per_mut"] = df.esm2_wtm / df.n_mut.clip(lower=1)
    return df


def add_esm(df, sequence, offset=0, model="facebook/esm2_t33_650M_UR50D",
            batch_size=4, progress=None, mode="variant"):
    """ESM-2 score for every row. Requires torch + transformers.

    Variant mode uses masked marginals against the scaffold. Sequence mode has no
    scaffold to mask against, so it uses the mean per-residue log-likelihood of
    each protein -- the natural zero-shot score for a mining campaign, and the
    one length-normalised by construction.
    """
    from .proxies import ESM2Marginals

    scorer = ESM2Marginals(model, batch_size=batch_size)
    if progress:
        progress(f"ESM-2 on {scorer.device}")

    if mode == "sequence":
        uniq = {s: None for s in df.sequence.astype(str)}
        for i, s in enumerate(uniq, 1):
            uniq[s] = scorer.sequence_loglik(s)
            if progress and i % 25 == 0:
                progress(f"scored {i}/{len(uniq)} sequences")
        df = df.copy()
        df["esm2_seq_loglik"] = [uniq[str(s)] for s in df.sequence]
        return df

    positions = sorted({p - 1 + offset for ms in df.muts for _, p, _ in ms})
    scorer.logprobs_at(sequence, positions)
    df = df.copy()
    df["esm2_wtm"] = [scorer.score(sequence, ms, offset) for ms in df.muts]
    df["esm2_per_mut"] = df.esm2_wtm / df.n_mut.clip(lower=1)
    return df


def _rho(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 5 or np.std(a[ok]) < 1e-12 or np.std(b[ok]) < 1e-12:
        return np.nan
    return float(stats.spearmanr(a[ok], b[ok]).statistic)


def cv_supervised(measured, folds=5, alpha=1.0, mode="variant"):
    """Out-of-fold performance of a ridge model fit on the developer's own data.

    Returns (auroc, fold_enrichment, spearman). AUROC is what the decision is
    made on; the other two are reported because they are more interpretable.
    """
    X = features(measured, mode)
    y = measured.fitness.to_numpy(float)
    if len(y) < MIN_FOR_TRUST or np.std(y) < 1e-12:
        return np.nan, np.nan, np.nan
    # Leave-one-out, not k-fold: a fold model trained on 4/5 of the data scores
    # worse than the model that actually gets deployed on all of it, and that
    # gap was losing the comparison against parameter-free proxies, which carry
    # no such handicap. See FINDINGS.md Phase 12.
    pred = loo_ridge_predictions(X, y, alpha)
    if pred is None:
        return np.nan, np.nan, np.nan
    return (elite_auroc(pred, y), top_decile_enrichment(pred, y), _rho(pred, y))


def measure_pooling(df, cond_key, mode="variant", alpha=1.0, reps=10,
                    min_per=20, folds=5, seed=0, margin=0.02):
    """Should measurements from different assay conditions be pooled? Measure it.

    This used to be assumed. The tool refused to pool, justified by a synthetic
    demonstration; on the only real multi-condition dataset available, pooling
    turned out to be BETTER (AUROC 0.716 vs 0.645) and a size-matched control
    showed condition purity was worth nothing (0.653, p=0.83). See FINDINGS.md
    Phase 13. Assuming either way is the same mistake this tool exists to catch,
    so it now measures on the developer's own campaign.

    Three models, each scored on the same held-out rows of one condition:
      within         trained on that condition only
      mixed_equal_n  trained on the same NUMBER of rows, drawn from OTHER
                     conditions -- isolates mixing from sample size
      pooled         trained on everything

    Returns per-condition rows plus a verdict.
    """
    measured = df[df.fitness.notna()]
    if len(measured) < 2 * min_per:
        return None
    key = cond_key.loc[measured.index]
    counts = key.value_counts()
    usable = [c for c in counts.index if counts[c] >= min_per]
    if len(usable) < 2:
        return None

    X = features(measured, mode)
    y = measured.fitness.to_numpy(float)
    pos = {ix: i for i, ix in enumerate(measured.index)}
    rng = np.random.default_rng(seed)

    # Pooled predictions once, by k-fold over everything: 5 fits, not one per row.
    pooled_pred = np.full(len(y), np.nan)
    for tr, te in KFold(min(folds, len(y)), shuffle=True, random_state=seed).split(X):
        if np.std(y[tr]) < 1e-12:
            continue
        pooled_pred[te] = Ridge(alpha=alpha).fit(X[tr], y[tr]).predict(X[te])

    rows = []
    for c in usable:
        idx = np.array([pos[i] for i in measured.index[key.to_numpy() == c]])
        if len(idx) < min_per or np.std(y[idx]) < 1e-12:
            continue
        other = np.setdiff1d(np.arange(len(y)), idx)
        if len(other) < len(idx):
            continue

        w = loo_ridge_predictions(X[idx], y[idx], alpha)
        within = elite_auroc(w, y[idx]) if w is not None else np.nan

        vals = []
        for _ in range(reps):
            tr = rng.choice(other, len(idx), replace=False)
            if np.std(y[tr]) < 1e-12:
                continue
            m = Ridge(alpha=alpha).fit(X[tr], y[tr])
            vals.append(elite_auroc(m.predict(X[idx]), y[idx]))
        mixed = float(np.nanmean(vals)) if vals else np.nan

        rows.append({"condition": c, "n": len(idx), "within": within,
                     "mixed_equal_n": mixed,
                     "pooled": elite_auroc(pooled_pred[idx], y[idx])})

    if not rows:
        return None
    t = pd.DataFrame(rows)
    w, m, p = t.within.mean(), t.mixed_equal_n.mean(), t.pooled.mean()

    if np.isfinite(p) and np.isfinite(w) and p > w + margin:
        verdict, why = "pool", (
            f"pooling scores higher on your data ({p:.3f} vs {w:.3f} AUROC)")
    elif np.isfinite(w) and np.isfinite(p) and w > p + margin:
        verdict, why = "stratify", (
            f"a model fit within one condition scores higher ({w:.3f} vs {p:.3f})")
    else:
        verdict, why = "pool", (
            f"no measurable difference ({w:.3f} within vs {p:.3f} pooled); "
            "pooling gives more data, so it is the safer default")

    purity = (np.isfinite(w) and np.isfinite(m) and abs(w - m) <= margin)
    return {"table": t, "within": w, "mixed_equal_n": m, "pooled": p,
            "verdict": verdict, "why": why, "purity_worthless": bool(purity)}


def diagnose(df, proxy_cols, mode="variant"):
    """Regime, confound and trust, computed only from what the developer has."""
    measured = df[df.fitness.notna()]
    candidates = df[df.fitness.isna()]
    pool = candidates if len(candidates) else measured
    nuis = nuisance_col(mode)

    d = {
        "mode": mode,
        "nuisance": nuis,
        "n_measured": len(measured),
        "n_candidates": len(candidates),
        "mixed_k": bool(len(pool) and pool[nuis].nunique() > 1),
        "k_range": ((int(pool[nuis].min()), int(pool[nuis].max()))
                    if len(pool) and pool[nuis].notna().any() else (0, 0)),
        "proxy_rho": {}, "proxy_enrich": {}, "proxy_auroc": {}, "confound": {},
    }
    (d["supervised_cv_auroc"], d["supervised_cv_enrich"],
     d["supervised_cv_rho"]) = cv_supervised(measured, mode=mode)
    for c in proxy_cols:
        if c in measured:
            d["proxy_rho"][c] = _rho(measured[c], measured.fitness)
            d["proxy_enrich"][c] = top_decile_enrichment(measured[c], measured.fitness)
            d["proxy_auroc"][c] = elite_auroc(measured[c], measured.fitness)
        if c in pool:
            d["confound"][c] = _rho(pool[c], pool[nuis])

    # Selection is by AUROC for the top decile. A batch only needs the good ones
    # near the top, so rank correlation is the wrong question -- but enrichment,
    # the obvious alternative, is a k-way set intersection that takes four
    # distinct values on 48 observations and cannot discriminate at that size.
    best = [(c, a) for c, a in d["proxy_auroc"].items()
            if np.isfinite(a) and c not in NOT_SELECTABLE]
    d["best_proxy"], d["best_proxy_auroc"] = max(
        best, key=lambda kv: kv[1], default=(None, np.nan))
    d["best_proxy_enrich"] = d["proxy_enrich"].get(d["best_proxy"], np.nan)
    d["best_proxy_rho"] = d["proxy_rho"].get(d["best_proxy"], np.nan)
    return d


def recommend(d):
    """Pick a policy and say why, in the developer's terms."""
    sup, prox = d["supervised_cv_auroc"], d.get("best_proxy_auroc", np.nan)
    sup_e, prox_e = d["supervised_cv_enrich"], d.get("best_proxy_enrich", np.nan)
    sup_rho, prox_rho = d["supervised_cv_rho"], d["best_proxy_rho"]
    reasons = []

    if d["n_measured"] < MIN_FOR_TRUST:
        return "diversity", [
            f"Only {d['n_measured']} measured variants — below the {MIN_FOR_TRUST} "
            "needed to cross-validate anything. No model has earned trust yet.",
            "Ordering a diverse batch now buys the data that makes round 2 decidable.",
        ]

    sup_ok = np.isfinite(sup) and sup >= TRUST_AUROC
    prox_ok = np.isfinite(prox) and prox >= TRUST_AUROC

    if sup_ok and (not prox_ok or sup + TIE_AUROC >= prox):
        reasons.append(
            f"A ridge model on your own {d['n_measured']} measurements reaches AUROC "
            f"{sup:.3f} for the top decile ({sup_e:.2f}x enrichment, Spearman "
            f"{sup_rho:+.2f}), clearing the {TRUST_AUROC:.2f} bar.")
        if prox_ok:
            if abs(sup - prox) < 1e-6:
                verb = "ties"
            elif sup > prox:
                verb = "beats"
            else:
                verb = "is within a nose of"
            reasons.append(
                f"It {verb} the best precomputed proxy ({d['best_proxy']}, "
                f"AUROC {prox:.3f}), and a model fit on your own data was never "
                "catastrophic across 13 DMS assays and 5 campaigns, whereas "
                "zero-shot scores were — so ties go to the model on your data.")
            if verb != "beats":
                reasons.append(
                    f"{d['best_proxy']} is worth ordering alongside: it reaches "
                    f"{prox_e:.2f}x enrichment on a rank correlation of only {prox_rho:+.2f}, so it "
                    "is finding the top decile by a different route than the model.")
        return "supervised", reasons

    if prox_ok:
        reasons.append(
            f"{d['best_proxy']} reaches AUROC {prox:.3f} for the top decile "
            f"({prox_e:.2f}x enrichment) and beats a model fit on your own data "
            f"(AUROC {sup:.3f}).")
        if np.isfinite(prox_rho) and abs(prox_rho) < 0.20:
            reasons.append(
                f"Note its rank correlation is only {prox_rho:+.2f} — it is a poor "
                "ranker overall but good at surfacing the top decile, which is what "
                "the batch needs.")
        c = d["confound"].get(d["best_proxy"], np.nan)
        if d["mixed_k"] and np.isfinite(c) and abs(c) >= CONFOUND_RHO:
            what = ("sequence lengths" if d.get("nuisance") == "seq_len"
                    else "mutation counts")
            reasons.append(
                f"WARNING: your candidates span {what} {d['k_range'][0]}–"
                f"{d['k_range'][1]} and {d['best_proxy']} correlates {c:+.2f} with "
                f"{d.get('nuisance')}. That much of its apparent signal is the "
                "nuisance variable, not function.")
        return "proxy", reasons

    reasons.append(
        f"Nothing has earned trust on your data: supervised AUROC {sup:.3f}, best "
        f"proxy {prox:.3f}, both below the {TRUST_AUROC:.2f} bar.")
    reasons.append("Exploring is the correct move; model-based ranking here would "
                   "be no better than chance and possibly worse.")
    return "diversity", reasons


def diversity_batch(candidates, budget, seed=0, mode="variant"):
    """Farthest-point selection: spread the batch across the design space."""
    X = features(candidates, mode)
    rng = np.random.default_rng(seed)
    picks = [int(rng.integers(len(X)))]
    dist = np.abs(X - X[picks[0]]).sum(1)
    while len(picks) < min(budget, len(X)):
        nxt = int(np.argmax(dist))
        picks.append(nxt)
        dist = np.minimum(dist, np.abs(X - X[nxt]).sum(1))
    return candidates.index[picks]


def select(df, policy, budget, proxy_col=None, beta=0.0, seed=0, mode="variant"):
    """Return the index of the batch to order, plus a per-pick score."""
    measured = df[df.fitness.notna()]
    candidates = df[df.fitness.isna()]
    if not len(candidates):
        raise ValueError("no unmeasured candidates to choose from")

    if policy == "diversity":
        idx = diversity_batch(candidates, budget, seed, mode=mode)
        return idx, {i: float("nan") for i in idx}

    if policy == "proxy":
        s = candidates[proxy_col].astype(float)
        idx = s.sort_values(ascending=False).index[:budget]
        return idx, s.loc[idx].to_dict()

    X = features(df, mode)
    Xm, Xc = X[df.fitness.notna().to_numpy()], X[df.fitness.isna().to_numpy()]
    model = Ridge(alpha=1.0).fit(Xm, measured.fitness.to_numpy(float))
    pred = model.predict(Xc)

    if beta > 0:
        novelty = np.abs(Xc[:, None, :] - Xm[None, :, :]).sum(-1).min(1)
        pred = pred + beta * np.std(measured.fitness) * novelty / (novelty.max() + 1e-9)

    order = np.argsort(-pred)[:budget]
    idx = candidates.index[order]
    return idx, dict(zip(idx, pred[order].astype(float)))
