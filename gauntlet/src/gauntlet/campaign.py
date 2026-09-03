"""Retrospective replay of directed-evolution campaigns.

A design campaign is a sequence of decisions under a budget: you can order and
assay B variants per round, and after each round you learn their fitness and
choose again. The question a developer actually faces is not "is my model
accurate" but "does my selection policy find a good enzyme in fewer rounds than
picking at random" -- and rounds cost weeks and thousands of dollars.

That question can be answered without a wet lab, by replaying campaigns whose
variants were already measured. The policy is only ever allowed to select
variants that exist in the measured set, so the replay is an honest simulation
of a chemist who could have ordered any of them in a different order.

VALIDITY. This is offline counterfactual evaluation and it is only sound where
the measured set is dense. On a complete single-mutant scan every action the
policy might take has a recorded outcome. On a sparse published campaign the
measured set is itself the product of somebody else's selection, so a replay
answers "could this policy have re-derived their result faster" -- a real but
weaker question. Densities are reported so the distinction is never hidden.
"""

import numpy as np
from sklearn.linear_model import Ridge

from .proxies import KD_HYDROPATHY, _blosum62

AA = list(KD_HYDROPATHY)
VOLUME = {  # Zamyatnin residue volumes, cubic angstroms
    "A": 88.6, "R": 173.4, "N": 114.1, "D": 111.1, "C": 108.5, "Q": 143.8,
    "E": 138.4, "G": 60.1, "H": 153.2, "I": 166.7, "L": 166.7, "K": 168.6,
    "M": 162.9, "F": 189.9, "P": 112.7, "S": 89.0, "T": 116.1, "W": 227.8,
    "Y": 193.6, "V": 140.0,
}
CHARGE = {"D": -1.0, "E": -1.0, "K": 1.0, "R": 1.0, "H": 0.5}


def mutation_features(records, vocab=None):
    """One-hot over (position, mutant residue) seen anywhere in the campaign.

    Appropriate only when variants share mutations, i.e. multi-mutant
    engineering campaigns. On a single-mutant scan every variant becomes its own
    feature and the model cannot generalise at all -- use
    ``factorized_features`` there instead.
    """
    if vocab is None:
        vocab = sorted({(p, m) for r in records for _, p, m in r["muts"]})
    index = {k: i for i, k in enumerate(vocab)}
    X = np.zeros((len(records), len(vocab)), dtype=np.float32)
    for i, r in enumerate(records):
        for _, p, m in r["muts"]:
            if (p, m) in index:
                X[i, index[(p, m)]] = 1.0
    return X, vocab


def sequence_features(seqs):
    """Amino-acid composition and length, for campaigns of unrelated proteins.

    A mining campaign screens homologs that share no common scaffold, so there
    are no mutations to featurise and the position x residue decomposition does
    not apply. Composition and length are the honest minimum: no pretraining, no
    alignment, no structure. They are also the nuisance variables most likely to
    be doing the work, which is the point -- a learned score that a composition
    model matches is not measuring function.
    """
    idx = {a: i for i, a in enumerate(AA)}
    X = np.zeros((len(seqs), len(AA) + 1), dtype=np.float32)
    for i, s in enumerate(seqs):
        s = str(s)
        for ch in s:
            if ch in idx:
                X[i, idx[ch]] += 1.0
        if len(s):
            X[i, :len(AA)] /= len(s)          # composition, not raw counts
        X[i, -1] = len(s)
    sd = X.std(0)
    X /= np.where(sd > 1e-9, sd, 1.0)
    return X, list(AA) + ["length"]


def factorized_features(records, positions=None):
    """Additive position x residue decomposition that generalises to unseen edits.

    Each variant is the sum over its mutations of: a position indicator, a
    wild-type residue indicator, a mutant residue indicator, and four scalar
    substitution descriptors. Because position and residue enter separately, a
    model fit on some (position, residue) pairs predicts ones it has never seen
    -- which is what a selection policy needs on the first few rounds.

    Still deliberately simple: no pretraining, no structure, no epistasis.
    Any margin over random is attributable to the decision loop.
    """
    if positions is None:
        positions = sorted({p for r in records for _, p, _ in r["muts"]})
    pidx = {p: i for i, p in enumerate(positions)}
    aidx = {a: i for i, a in enumerate(AA)}
    bl = _blosum62()
    P, A = len(positions), len(AA)
    X = np.zeros((len(records), P + 2 * A + 5), dtype=np.float32)
    for i, r in enumerate(records):
        for wt, p, mt in r["muts"]:
            if p in pidx:
                X[i, pidx[p]] += 1.0
            if wt in aidx:
                X[i, P + aidx[wt]] += 1.0
            if mt in aidx:
                X[i, P + A + aidx[mt]] += 1.0
            try:
                X[i, P + 2 * A] += float(bl[wt, mt])
            except KeyError:
                pass
            X[i, P + 2 * A + 1] += KD_HYDROPATHY.get(mt, 0) - KD_HYDROPATHY.get(wt, 0)
            X[i, P + 2 * A + 2] += VOLUME.get(mt, 0) - VOLUME.get(wt, 0)
            X[i, P + 2 * A + 3] += CHARGE.get(mt, 0) - CHARGE.get(wt, 0)
        X[i, P + 2 * A + 4] = len(r["muts"])
    # Put the scalar descriptors on a comparable scale to the indicators.
    tail = slice(P + 2 * A, P + 2 * A + 4)
    sd = X[:, tail].std(0)
    X[:, tail] /= np.where(sd > 1e-9, sd, 1.0)
    return X, positions


TOP_FRAC = 0.10             # "good" means the top this fraction of the pool
MIN_ELITE = 5               # below this many elites the enrichment is noise


def elite_k(n, top_frac=TOP_FRAC, min_elite=MIN_ELITE):
    """How many variants count as "the top" for a pool of this size."""
    return max(min_elite, int(round(top_frac * n)))


def top_decile_enrichment(scores, fitness, top_frac=TOP_FRAC, min_elite=MIN_ELITE):
    """Fold-enrichment of the true top decile among a scorer's own top decile.

    This is what a batch actually needs, and it is not the same thing as rank
    correlation. A scorer can be near-useless on overall rank and still be the
    best available way to find the top 10% -- measured on the NREL release, mean
    hydropathy correlates +0.07 with activity yet more than doubles the top-decile
    hit rate. Selecting proxies by Spearman passes over exactly that scorer.

    1.0 means no better than picking at random; 2.0 means twice the hit rate.
    """
    a = np.asarray(scores, float)
    b = np.asarray(fitness, float)
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    n = len(b)
    k = max(min_elite, int(round(top_frac * n)))
    if n < 2 * min_elite or k >= n or np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return np.nan
    elite = set(np.argsort(-b)[:k].tolist())
    picked = set(np.argsort(-a)[:k].tolist())
    expected = k * k / n                      # hits a random pick of k would get
    return float(len(elite & picked) / expected) if expected > 0 else np.nan


def loo_ridge_predictions(X, y, alpha=1.0):
    """Leave-one-out predictions for ridge, in closed form.

    Why this matters: 5-fold CV trains each fold model on 4/5 of the data, so its
    score understates the model that actually gets DEPLOYED on all of it. A
    parameter-free proxy has no such handicap -- its held-out and in-sample
    predictions are identical -- so comparing the two decides a coin flip against
    the model. Measured on 1,560 decisions, the proxy won 54% of comparisons and
    was right only 29% of the times the planner acted on it (FINDINGS.md,
    Phase 11).

    Leave-one-out trains on n-1, which is the deployment size. For ridge it costs
    one solve rather than n refits, via the standard hat-matrix identity
    yhat_i^(-i) = y_i - e_i / (1 - h_ii).
    """
    X = np.asarray(X, float)
    y = np.asarray(y, float)
    n = X.shape[0]
    if n < 3:
        return None
    # Augment with an unpenalised intercept column so the hat-matrix identity
    # matches Ridge(fit_intercept=True). Centring on the full sample instead
    # would be wrong: leaving a point out also moves the mean.
    Xa = np.hstack([np.ones((n, 1)), X])
    d = np.ones(Xa.shape[1])
    d[0] = 0.0                                   # never penalise the intercept
    A = Xa.T @ Xa + alpha * np.diag(d)
    try:
        W = np.linalg.solve(A, Xa.T)             # (p+1) x n
    except np.linalg.LinAlgError:
        W = np.linalg.pinv(A) @ Xa.T
    h = np.einsum("ij,ji->i", Xa, W)             # diag of the hat matrix
    fit = Xa @ (W @ y)
    denom = np.clip(1.0 - h, 1e-6, None)
    return y - (y - fit) / denom


def elite_auroc(scores, fitness, top_frac=TOP_FRAC, min_elite=MIN_ELITE):
    """Probability a random top-decile member outranks a random non-member.

    The continuous alternative to ``top_decile_enrichment``. Enrichment is a
    k-way set intersection, so on 48 observations it takes one of six values in
    steps of 1.92 and a 1.5x bar reduces to "did at least one of my top five
    land in the observed top five". AUROC uses every elite/non-elite pair --
    ~215 of them at the same sample size -- so it degrades gracefully instead of
    collapsing to a coin flip. 0.5 is random.
    """
    from sklearn.metrics import roc_auc_score

    a = np.asarray(scores, float)
    b = np.asarray(fitness, float)
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    n = len(b)
    k = max(min_elite, int(round(top_frac * n)))
    if n < 2 * min_elite or k >= n or np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return np.nan
    label = np.zeros(n, dtype=int)
    label[np.argsort(-b)[:k]] = 1
    if label.sum() in (0, n):
        return np.nan
    return float(roc_auc_score(label, a))


class Policy:
    """Chooses which unmeasured variants to order next."""

    def __init__(self, kind, alpha=1.0, beta=1.0):
        self.kind, self.alpha, self.beta = kind, alpha, beta

    def select(self, X, y_obs, observed, budget, prior, rng):
        pool = np.flatnonzero(~observed)
        if len(pool) <= budget:
            return pool

        if self.kind == "random":
            return rng.choice(pool, budget, replace=False)

        if self.kind == "zero_shot":
            return pool[np.argsort(-prior[pool])[:budget]]

        # Supervised policies fit on what has been measured so far.
        seen = np.flatnonzero(observed)
        if len(seen) < 3 or np.std(y_obs[seen]) < 1e-9:
            return rng.choice(pool, budget, replace=False)
        model = Ridge(alpha=self.alpha).fit(X[seen], y_obs[seen])
        pred = model.predict(X[pool])

        if self.kind == "greedy":
            return pool[np.argsort(-pred)[:budget]]

        if self.kind == "ucb":
            # Uncertainty proxy: Hamming distance to the nearest measured design.
            # Cheap, model-free, and it keeps the policy from re-ordering minor
            # variations on what it has already seen.
            novelty = np.abs(X[pool][:, None, :] - X[seen][None, :, :]).sum(-1).min(1)
            score = pred + self.beta * np.std(y_obs[seen]) * novelty / (novelty.max() + 1e-9)
            return pool[np.argsort(-score)[:budget]]

        raise ValueError(self.kind)


class PlannerPolicy:
    """Applies the tool's own decision rule at every round.

    This is the policy `gauntlet plan` actually implements, so backtesting it
    answers the question the fixed-policy comparison cannot: does the DECISION
    RULE make good calls, given only what the developer would have known at that
    point in the campaign? Every round it re-judges, on the measurements taken so
    far, whether a model fit on that data or one of the precomputed proxies
    enriches the top decile enough to be worth exploiting -- and explores when
    neither does.

    Nothing here sees the unmeasured fitnesses, so the comparison is fair.
    """

    def __init__(self, alpha=1.0, min_for_trust=12, trust=0.55, tie=0.01, folds=5):
        self.alpha, self.min_for_trust = alpha, min_for_trust
        self.trust, self.tie, self.folds = trust, tie, folds
        self.decisions = []          # what it chose each round, for reporting

    def _cv_enrichment(self, X, y):
        """Leave-one-out AUROC -- the deployment training size, not 4/5 of it."""
        if len(y) < self.min_for_trust or np.std(y) < 1e-12:
            return np.nan
        pred = loo_ridge_predictions(X, y, self.alpha)
        return np.nan if pred is None else elite_auroc(pred, y)

    def select(self, X, y_obs, observed, budget, priors, rng):
        pool = np.flatnonzero(~observed)
        if len(pool) <= budget:
            return pool
        seen = np.flatnonzero(observed)
        if len(seen) < self.min_for_trust or np.std(y_obs[seen]) < 1e-12:
            self.decisions.append("explore")
            return rng.choice(pool, budget, replace=False)

        sup_e = self._cv_enrichment(X[seen], y_obs[seen])
        best_name, best_e, best_arr = None, -np.inf, None
        for name, arr in (priors or {}).items():
            if name == "random":
                continue
            e = elite_auroc(arr[seen], y_obs[seen])
            if np.isfinite(e) and e > best_e:
                best_name, best_e, best_arr = name, e, arr

        sup_ok = np.isfinite(sup_e) and sup_e >= self.trust
        prox_ok = np.isfinite(best_e) and best_e >= self.trust

        if sup_ok and (not prox_ok or sup_e + self.tie >= best_e):
            self.decisions.append("supervised")
            model = Ridge(alpha=self.alpha).fit(X[seen], y_obs[seen])
            return pool[np.argsort(-model.predict(X[pool]))[:budget]]
        if prox_ok:
            self.decisions.append(f"proxy:{best_name}")
            return pool[np.argsort(-best_arr[pool])[:budget]]
        self.decisions.append("explore")
        return rng.choice(pool, budget, replace=False)


class EpsilonSupervisedPolicy:
    """Always fit a model; spend a fixed slice of every batch on exploration.

    The ablation for the whole adaptive layer. The planner beat fixed policies on
    cumulative recall (Phase 10) but chooses wrong 71% of the times it deviates
    (Phase 11), which suggests its edge comes from the exploration it does while
    switching rather than from the switching. If this matches the planner, the
    scorer-selection logic is dead weight and should be deleted.

    No proxies, no AUROC, no gate: fit ridge on everything measured so far, take
    the top (1-eps)*budget, fill the rest at random.
    """

    def __init__(self, eps=0.1, alpha=1.0, min_for_fit=3):
        self.eps, self.alpha, self.min_for_fit = eps, alpha, min_for_fit

    def select(self, X, y_obs, observed, budget, prior, rng):
        pool = np.flatnonzero(~observed)
        if len(pool) <= budget:
            return pool
        n_rand = int(round(self.eps * budget))
        n_greedy = budget - n_rand
        seen = np.flatnonzero(observed)
        if len(seen) < self.min_for_fit or np.std(y_obs[seen]) < 1e-9 or n_greedy <= 0:
            return rng.choice(pool, budget, replace=False)

        model = Ridge(alpha=self.alpha).fit(X[seen], y_obs[seen])
        order = np.argsort(-model.predict(X[pool]))
        greedy = pool[order[:n_greedy]]
        if n_rand:
            rest = np.setdiff1d(pool, greedy, assume_unique=False)
            if len(rest):
                extra = rng.choice(rest, min(n_rand, len(rest)), replace=False)
                return np.concatenate([greedy, extra])
        return greedy


def replay(X, y, prior, policy, budget, rounds, rng):
    """Run one campaign. Returns the boolean mask of what was ordered."""
    observed = np.zeros(len(y), dtype=bool)
    y_obs = np.full(len(y), np.nan, dtype=float)
    for _ in range(rounds):
        pick = policy.select(X, y_obs, observed, budget, prior, rng)
        observed[pick] = True
        y_obs[pick] = y[pick]
    return observed


def evaluate(X, y, prior, policy, budget, rounds, seeds=200, top_frac=0.01):
    """Average campaign outcome over random seeds.

    Two metrics, because they behave differently with pool size:

    - ``best_norm``: best fitness found, rescaled so 1.0 is the true best.
      Comparable across assays, but saturates once the budget is a large
      fraction of a small pool.
    - ``top_recall``: fraction of the assay's true top ``top_frac`` that the
      policy actually ordered. This is the metric that discriminates on large
      pools, and it is closer to what a developer wants -- not one lucky hit but
      a batch enriched for good variants.
    """
    lo, hi = float(np.min(y)), float(np.max(y))
    span = hi - lo if hi > lo else 1.0
    k = max(1, int(round(top_frac * len(y))))
    elite = set(np.argsort(-y)[:k].tolist())

    best, recall = [], []
    for s in range(seeds):
        obs = replay(X, y, prior, policy, budget, rounds, np.random.default_rng(s))
        best.append((float(np.max(y[obs])) - lo) / span)
        recall.append(len(elite & set(np.flatnonzero(obs).tolist())) / k)
    return {
        "best_norm": float(np.mean(best)),
        "top_recall": float(np.mean(recall)),
        "top_recall_sd": float(np.std(recall)),
    }
