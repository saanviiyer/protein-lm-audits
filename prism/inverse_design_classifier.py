#!/usr/bin/env python3
"""
inverse_design_classifier.py

Inverts PRISM: instead of asking "where does signature s push protein p?", it
asks the design question "which signature drives a desired Pfam->Pfam
transition?". This is the supervised classifier proposed in the PRISM paper's
Future Directions, trained on the released centroid-trajectory outputs.

TASK
----
Each Phase-2 experiment is a realized transition: a protein in origin domain
`pfam_orig`, mutated under `signature`, has its most-attracted domain `top1_pfam`
(attraction `top1_delta_dist` > 0). We treat each such experiment as a training
example
        X = [ origin_centroid , target_centroid , (target - origin) ]   (960-d)
        y = signature                                                   (78-class)
and learn P(signature | desired transition). Inverse design for a wanted
(origin, target) pair is then the top-k of the predicted signature distribution.
Because several signatures can drive the same transition (median ~2), we report
top-k accuracy and MRR, not just top-1.

FAILURE-MODE DISCIPLINE (this is a PRISM paper; we hold ourselves to it)
-----------------------------------------------------------------------
* PROTEIN-GROUPED split: train/test never share a protein (`acc`), so we measure
  generalization, not memorization of a protein's experiments.
* HELD-OUT-TARGET stress test: a second split holds out entire target domains,
  testing whether the model generalizes to transitions it never saw a driver for
  (the honest, harder regime; guards against the model just memorizing
  "target==PF17041 -> SBS17a").
* HONEST BASELINES: a global signature prior and a per-origin-domain most-frequent
  signature. The model must beat both to be worth anything.
* Flagship sanity check: for held-out transitions whose target is PF17041, is
  SBS17a recovered as the top predicted driver?

USAGE
-----
python inverse_design_classifier.py \
    --runs_dir all_runs \
    --centroid_chunks_dir pfam_centroid_approach/seed_alignments/centroid_chunks \
    --output_dir results/inverse_design
"""

import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# ── data assembly ────────────────────────────────────────────────────────
def load_transitions(runs_dir, min_delta, use_topk):
    """Load realized transitions from the trajectory CSVs. Returns a DataFrame
    with columns: acc, clan, origin, target, signature, delta."""
    frames = []
    for csv in sorted(glob.glob(os.path.join(runs_dir, "*", "*centroid_trajectory_results.csv"))):
        clan = os.path.basename(os.path.dirname(csv))
        t = pd.read_csv(csv)
        t["clan"] = clan
        frames.append(t)
    if not frames:
        raise SystemExit(f"No trajectory CSVs under {runs_dir}")
    T = pd.concat(frames, ignore_index=True)
    T["signature"] = T["signature"].astype(str).str.replace("_PROFILE.txt", "", regex=False)

    records = []
    for rank in range(1, use_topk + 1):
        pfam_col, delta_col = f"top{rank}_pfam", f"top{rank}_delta_dist"
        if pfam_col not in T or delta_col not in T:
            continue
        sub = T[["acc", "clan", "pfam_orig", "signature", pfam_col, delta_col]].copy()
        sub.columns = ["acc", "clan", "origin", "signature", "target", "delta"]
        sub = sub[(sub["delta"] > min_delta) & sub["target"].astype(str).str.startswith("PF")]
        sub["rank"] = rank
        records.append(sub)
    trans = pd.concat(records, ignore_index=True)
    trans = trans[trans["origin"].astype(str).str.startswith("PF")]
    return trans


def build_features(trans, centroids):
    """Attach centroid vectors; drop rows whose origin/target lack a centroid."""
    ids = {pf: i for i, pf in enumerate(centroids["ids"])}
    M = centroids["matrix"]
    keep, X, meta = [], [], []
    for row in trans.itertuples(index=False):
        oi, ti = ids.get(row.origin), ids.get(row.target)
        if oi is None or ti is None:
            continue
        o, t = M[oi], M[ti]
        X.append(np.concatenate([o, t, t - o]))
        clan = getattr(row, "clan", "")
        meta.append((row.acc, clan, row.origin, row.target, row.signature, row.delta))
        keep.append(True)
    X = np.asarray(X, dtype=np.float32)
    meta = pd.DataFrame(meta, columns=["acc", "clan", "origin", "target", "signature", "delta"])
    return X, meta


def load_centroids(chunks_dir):
    from prism_genmodel_analysis import load_centroid_chunks
    matrix, ids = load_centroid_chunks(chunks_dir)
    return {"matrix": matrix, "ids": list(ids)}


def load_signature_profiles(profile_dir):
    """Return {signature_name: 96-vector} in build_96_context_keys() order, so
    each signature has a mechanistic feature vector (its COSMIC profile)."""
    from prism_genmodel_analysis import build_96_context_keys
    keys = build_96_context_keys()

    def _parse(path):  # inlined prism_utils.load_mutation_profile (avoids the torch import)
        sigs = {}
        with open(path) as f:
            next(f, None)
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2 and len(parts[0]) >= 7 and parts[0][1] == "[" and parts[0][5] == "]":
                    try:
                        sigs[parts[0]] = float(parts[1])
                    except ValueError:
                        pass
        return sigs

    profiles = {}
    for root, _, files in os.walk(profile_dir):
        for fn in files:
            if fn.startswith(".") or "PROFILE" not in fn.upper():
                continue
            stem = fn.split("_")[0].split(".")[0]
            try:
                prof = _parse(os.path.join(root, fn))
            except Exception:
                continue
            if not prof:
                continue
            vec = np.array([prof.get(k, 0.0) for k in keys], dtype=np.float32)
            s = vec.sum()
            profiles[stem] = vec / s if s > 0 else vec
    return profiles


# ── evaluation ──────────────────────────────────────────────────────────
def topk_metrics(clf, X, y_true, classes, ks=(1, 3, 5)):
    proba = clf.predict_proba(X)
    order = np.argsort(-proba, axis=1)
    ranked = classes[order]
    out = {}
    for k in ks:
        hits = np.array([y_true[i] in ranked[i, :k] for i in range(len(y_true))])
        out[f"top{k}"] = float(hits.mean())
    # mean reciprocal rank of the true signature
    rr = []
    for i in range(len(y_true)):
        pos = np.where(ranked[i] == y_true[i])[0]
        rr.append(1.0 / (pos[0] + 1) if len(pos) else 0.0)
    out["mrr"] = float(np.mean(rr))
    return out, ranked


def prior_baseline(y_train, y_test, classes, ks=(1, 3, 5)):
    order = pd.Series(y_train).value_counts().index.tolist()
    out = {}
    for k in ks:
        topk = set(order[:k])
        out[f"top{k}"] = float(np.mean([y in topk for y in y_test]))
    return out


def per_origin_baseline(meta_tr, meta_te, ks=(1, 3, 5)):
    """Most-frequent signatures per origin domain, learned from train."""
    table = (meta_tr.groupby("origin")["signature"]
             .agg(lambda s: list(pd.Series(s).value_counts().index)))
    glob_order = list(pd.Series(meta_tr["signature"]).value_counts().index)
    out = {f"top{k}": [] for k in ks}
    for row in meta_te.itertuples(index=False):
        ranked = table.get(row.origin, glob_order)
        for k in ks:
            out[f"top{k}"].append(row.signature in set(ranked[:k]))
    return {k: float(np.mean(v)) for k, v in out.items()}


def flagship_check(ranked, meta_te, target="PF17041", signature="SBS17a"):
    idx = meta_te.index[meta_te["target"] == target].tolist()
    if not idx:
        return None
    pos = [np.where(ranked[i] == signature)[0] for i in range(len(meta_te))
           if meta_te.iloc[i]["target"] == target]
    top1 = np.mean([len(p) and p[0] == 0 for p in pos])
    top3 = np.mean([len(p) and p[0] < 3 for p in pos])
    return {"n": len(pos), f"{signature}_top1": float(top1), f"{signature}_top3": float(top3)}


# ── train + evaluate one split ──────────────────────────────────────────
def make_classifier(args):
    if args.model == "hgb":
        from sklearn.ensemble import HistGradientBoostingClassifier
        return HistGradientBoostingClassifier(
            max_iter=args.n_estimators, learning_rate=0.1, random_state=args.seed)
    from sklearn.ensemble import RandomForestClassifier
    return RandomForestClassifier(
        n_estimators=args.n_estimators, n_jobs=-1, random_state=args.seed,
        class_weight="balanced_subsample", min_samples_leaf=2)


def run_split(X, meta, groups, split_name, args):
    from sklearn.model_selection import GroupShuffleSplit

    gss = GroupShuffleSplit(n_splits=1, test_size=args.test_size, random_state=args.seed)
    tr, te = next(gss.split(X, meta["signature"], groups))
    Xtr, Xte = X[tr], X[te]
    ytr, yte = meta["signature"].values[tr], meta["signature"].values[te]
    meta_tr, meta_te = meta.iloc[tr].reset_index(drop=True), meta.iloc[te].reset_index(drop=True)

    clf = make_classifier(args)
    clf.fit(Xtr, ytr)
    classes = clf.classes_

    model_metrics, ranked = topk_metrics(clf, Xte, yte, classes)
    prior = prior_baseline(ytr, yte, classes)
    per_origin = per_origin_baseline(meta_tr, meta_te)
    flagship = flagship_check(ranked, meta_te)

    print(f"\n[{split_name}] train={len(tr)} test={len(te)}  "
          f"groups: {len(np.unique(groups[tr]))} train / {len(np.unique(groups[te]))} test")
    print(f"  {'metric':10}{'model':>10}{'per-origin':>12}{'prior':>10}")
    for k in ("top1", "top3", "top5"):
        print(f"  {k:10}{model_metrics[k]:>10.3f}{per_origin[k]:>12.3f}{prior[k]:>10.3f}")
    print(f"  {'mrr':10}{model_metrics['mrr']:>10.3f}")
    if flagship:
        print(f"  flagship (target=PF17041, n={flagship['n']}): "
              f"SBS17a top1={flagship['SBS17a_top1']:.3f}  top3={flagship['SBS17a_top3']:.3f}")

    return {"split": split_name, "n_train": len(tr), "n_test": len(te),
            "model": model_metrics, "per_origin_baseline": per_origin,
            "prior_baseline": prior, "flagship_PF17041_SBS17a": flagship,
            "classifier": clf, "classes": classes.tolist()}


def export_design_rules(clf, classes, centroids, meta, out_csv, topk=3):
    """The tangible inverse-design deliverable: for every distinct (origin ->
    target) transition observed, the model's top-k recommended signatures,
    next to the empirically most-frequent driver in the data."""
    ids = {pf: i for i, pf in enumerate(centroids["ids"])}
    M = centroids["matrix"]
    pairs = meta.groupby(["origin", "target"]).size().reset_index(name="n_observed")
    emp = (meta.groupby(["origin", "target"])["signature"]
           .agg(lambda s: pd.Series(s).value_counts().index[0]))
    rows = []
    feats, keep = [], []
    for r in pairs.itertuples(index=False):
        oi, ti = ids.get(r.origin), ids.get(r.target)
        if oi is None or ti is None:
            continue
        feats.append(np.concatenate([M[oi], M[ti], M[ti] - M[oi]]))
        keep.append(r)
    if not feats:
        return
    proba = clf.predict_proba(np.asarray(feats, dtype=np.float32))
    order = np.argsort(-proba, axis=1)
    for r, ordr in zip(keep, order):
        recs = classes[ordr[:topk]].tolist()
        rows.append([r.origin, r.target, r.n_observed, emp.get((r.origin, r.target), ""),
                     *recs])
    cols = ["origin", "target", "n_observed", "empirical_top_driver"] + \
           [f"recommended_{i+1}" for i in range(topk)]
    pd.DataFrame(rows, columns=cols).to_csv(out_csv, index=False)
    print(f"[inv] wrote design rules for {len(rows)} (origin->target) pairs -> {out_csv}")


def build_triples(meta, centroids, profiles, n_neg, rng):
    """Triple-scoring dataset: (transition geometry + signature profile) -> drives?
    Positives are observed drivers; negatives are sampled non-driving signatures
    for the same transition. Unlike the multiclass model, this can score
    signatures it never saw during training (it reads their COSMIC profile)."""
    ids = {pf: i for i, pf in enumerate(centroids["ids"])}
    M = centroids["matrix"]
    all_sigs = [s for s in profiles]
    # observed drivers per (origin,target) to avoid sampling false negatives
    observed = meta.groupby(["origin", "target"])["signature"].agg(set).to_dict()

    X, y, grp = [], [], []
    for r in meta.itertuples(index=False):
        oi, ti = ids.get(r.origin), ids.get(r.target)
        if oi is None or ti is None or r.signature not in profiles:
            continue
        geom = np.concatenate([M[ti], M[ti] - M[oi]])
        X.append(np.concatenate([geom, profiles[r.signature]])); y.append(1); grp.append(r.acc)
        pos_set = observed.get((r.origin, r.target), set())
        negs = [s for s in all_sigs if s not in pos_set]
        for s in rng.sample(negs, min(n_neg, len(negs))):
            X.append(np.concatenate([geom, profiles[s]])); y.append(0); grp.append(r.acc)
    return np.asarray(X, dtype=np.float32), np.asarray(y), np.asarray(grp)


def run_triple(meta, centroids, profiles, args):
    """Train the triple-scorer and evaluate it as a signature ranker with the
    same top-k protocol, using a protein-grouped split."""
    import random as _random
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.model_selection import GroupShuffleSplit

    rng = _random.Random(args.seed)
    X, y, grp = build_triples(meta, centroids, profiles, args.n_neg, rng)
    gss = GroupShuffleSplit(n_splits=1, test_size=args.test_size, random_state=args.seed)
    tr, te = next(gss.split(X, y, grp))
    clf = HistGradientBoostingClassifier(max_iter=args.n_estimators, random_state=args.seed)
    clf.fit(X[tr], y[tr])

    # rank all profiled signatures for each held-out positive transition
    ids = {pf: i for i, pf in enumerate(centroids["ids"])}
    M = centroids["matrix"]
    sig_names = list(profiles)
    sig_mat = np.stack([profiles[s] for s in sig_names])
    test_acc = set(grp[te])
    pos_meta = meta[meta["acc"].isin(test_acc)].reset_index(drop=True)

    ks = (1, 3, 5); hits = {k: [] for k in ks}; rr = []
    for r in pos_meta.itertuples(index=False):
        oi, ti = ids.get(r.origin), ids.get(r.target)
        if oi is None or ti is None or r.signature not in profiles:
            continue
        geom = np.concatenate([M[ti], M[ti] - M[oi]])
        feats = np.hstack([np.tile(geom, (len(sig_names), 1)), sig_mat]).astype(np.float32)
        score = clf.predict_proba(feats)[:, 1]
        ranked = [sig_names[i] for i in np.argsort(-score)]
        pos = ranked.index(r.signature) if r.signature in ranked else len(ranked)
        for k in ks:
            hits[k].append(pos < k)
        rr.append(1.0 / (pos + 1))
    metrics = {f"top{k}": float(np.mean(hits[k])) for k in ks}
    metrics["mrr"] = float(np.mean(rr))
    metrics["auc_triples"] = None
    try:
        from sklearn.metrics import roc_auc_score
        metrics["auc_triples"] = float(roc_auc_score(y[te], clf.predict_proba(X[te])[:, 1]))
    except Exception:
        pass
    print(f"\n[triple-scoring] protein-grouped, {len(pos_meta)} test transitions, "
          f"AUC(triples)={metrics['auc_triples']}")
    for k in ks:
        print(f"  top{k}: {metrics[f'top{k}']:.3f}")
    print(f"  mrr : {metrics['mrr']:.3f}")
    return {"clf": clf, "metrics": metrics, "sig_names": sig_names}


def run_loso(meta, centroids, profiles, args):
    """Leave-Signatures-Out: the strongest generalization test. Entire
    signatures are held out of training (as both positives AND negatives); at
    test we rank all 78 signatures for transitions driven by the HELD-OUT ones.
    A multiclass model scores 0 here by construction (it never saw the label);
    only the profile-based triple-scorer can succeed, because it reads the
    held-out signature's COSMIC profile. Flagship: hold out SBS17a and ask
    whether the model still recommends it for -> PF17041, i.e. re-derives the
    design rule from mechanism alone."""
    import random as _random
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.model_selection import KFold

    rng = _random.Random(args.seed)
    ids = {pf: i for i, pf in enumerate(centroids["ids"])}
    M = centroids["matrix"]
    sig_names = list(profiles)
    sig_mat = np.stack([profiles[s] for s in sig_names])

    m = meta[meta["signature"].isin(profiles)
             & meta["origin"].isin(ids) & meta["target"].isin(ids)].reset_index(drop=True)
    uniq_sigs = np.array(sorted(m["signature"].unique()))

    def geom_of(o, t):
        return np.concatenate([M[ids[t]], M[ids[t]] - M[ids[o]]])

    def rank_all(o, t):
        g = geom_of(o, t)
        feats = np.hstack([np.tile(g, (len(sig_names), 1)), sig_mat]).astype(np.float32)
        score = clf.predict_proba(feats)[:, 1]
        return [sig_names[i] for i in np.argsort(-score)]

    kf = KFold(n_splits=args.loso_folds, shuffle=True, random_state=args.seed)
    ks = (1, 3, 5)
    agg = {f"top{k}": [] for k in ks}
    agg["mrr"] = []
    flagship = None

    for fold, (tr_i, te_i) in enumerate(kf.split(uniq_sigs)):
        train_sigs = set(uniq_sigs[tr_i])
        test_sigs = set(uniq_sigs[te_i])
        train_rows = m[m["signature"].isin(train_sigs)]
        observed = train_rows.groupby(["origin", "target"])["signature"].agg(set).to_dict()
        train_sig_list = list(train_sigs)

        X, y = [], []
        for r in train_rows.itertuples(index=False):
            g = geom_of(r.origin, r.target)
            X.append(np.concatenate([g, profiles[r.signature]])); y.append(1)
            pos = observed.get((r.origin, r.target), set())
            negs = [s for s in train_sig_list if s not in pos]
            for s in rng.sample(negs, min(args.n_neg, len(negs))):
                X.append(np.concatenate([g, profiles[s]])); y.append(0)
        clf = HistGradientBoostingClassifier(max_iter=args.n_estimators, random_state=args.seed)
        clf.fit(np.asarray(X, dtype=np.float32), np.asarray(y))

        test_rows = m[m["signature"].isin(test_sigs)]
        # cap the aggregate eval for runtime, but keep it representative
        if len(test_rows) > args.loso_test_cap:
            test_rows = test_rows.sample(args.loso_test_cap, random_state=args.seed)
        for r in test_rows.itertuples(index=False):
            ranked = rank_all(r.origin, r.target)
            pos = ranked.index(r.signature) if r.signature in ranked else len(ranked)
            for k in ks:
                agg[f"top{k}"].append(pos < k)
            agg["mrr"].append(1.0 / (pos + 1))

        if "SBS17a" in test_sigs:  # the flagship fold: SBS17a was held out
            pf = m[(m["signature"] == "SBS17a") & (m["target"] == "PF17041")]
            h1, h3 = [], []
            for r in pf.itertuples(index=False):
                ranked = rank_all(r.origin, r.target)
                p = ranked.index("SBS17a") if "SBS17a" in ranked else 999
                h1.append(p < 1); h3.append(p < 3)
            flagship = {"n": len(h1),
                        "SBS17a_heldout_top1": float(np.mean(h1)) if h1 else None,
                        "SBS17a_heldout_top3": float(np.mean(h3)) if h3 else None}
        print(f"  [loso] fold {fold+1}/{args.loso_folds}: held out {len(test_sigs)} sigs, "
              f"{len(test_rows)} test transitions"
              + ("  <- SBS17a fold" if "SBS17a" in test_sigs else ""))

    metrics = {k: float(np.mean(v)) for k, v in agg.items()}
    print(f"\n[LOSO] generalization to UNSEEN signatures ({args.loso_folds}-fold over signatures):")
    for k in ks:
        print(f"  top{k}: {metrics[f'top{k}']:.3f}")
    print(f"  mrr : {metrics['mrr']:.3f}")
    if flagship:
        print(f"  FLAGSHIP (SBS17a fully held out, ->PF17041, n={flagship['n']}): "
              f"top1={flagship['SBS17a_heldout_top1']:.3f}  top3={flagship['SBS17a_heldout_top3']:.3f}")
    return {"loso_metrics": metrics, "flagship_SBS17a_heldout": flagship}


FEATURE_SLICES = {  # X = [origin(320) | target(320) | (target-origin)(320)]
    "origin": slice(0, 320),
    "target": slice(320, 640),
    "diff": slice(640, 960),
    "origin+target": slice(0, 640),
    "full": slice(0, 960),
}


def run_crossclan(X, meta, args):
    """Train on one Pfam clan, test on another -- a stronger generalization
    claim than protein-grouped (the folds share no clan, no protein, and the
    origin-domain distributions differ). CL0016 is a single protein and is
    excluded; we report both directions between the two substantive clans."""
    clans = [c for c in meta["clan"].value_counts().index if (meta["clan"] == c).sum() >= 200]
    pairs = [(a, b) for a in clans for b in clans if a != b]
    print(f"\n[cross-clan] clans with >=200 transitions: {clans}")
    out = {}
    for train_c, test_c in pairs:
        tr = np.where(meta["clan"].values == train_c)[0]
        te = np.where(meta["clan"].values == test_c)[0]
        clf = make_classifier(args)
        clf.fit(X[tr], meta["signature"].values[tr])
        m, ranked = topk_metrics(clf, X[te], meta["signature"].values[te], clf.classes_)
        fl = flagship_check(ranked, meta.iloc[te].reset_index(drop=True))
        print(f"  train {train_c} -> test {test_c} (n_test={len(te)}): "
              f"top1={m['top1']:.3f} top5={m['top5']:.3f} mrr={m['mrr']:.3f}"
              + (f" | PF17041->SBS17a top1={fl['SBS17a_top1']:.2f} (n={fl['n']})" if fl else ""))
        out[f"{train_c}__to__{test_c}"] = {"metrics": m, "flagship": fl, "n_test": len(te)}
    return out


def nn_baseline(Xtr, ytr, Xte, yte, k=15, ks=(1, 3, 5)):
    """Nearest-neighbour-in-embedding baseline: rank signatures by their vote
    among the k nearest training transitions. A strong non-parametric control
    the classifier must beat."""
    from sklearn.neighbors import NearestNeighbors
    nn = NearestNeighbors(n_neighbors=min(k, len(Xtr))).fit(Xtr)
    _, idx = nn.kneighbors(Xte)
    out = {f"top{kk}": [] for kk in ks}
    rr = []
    for i in range(len(Xte)):
        votes = pd.Series(ytr[idx[i]]).value_counts()
        ranked = list(votes.index)
        pos = ranked.index(yte[i]) if yte[i] in ranked else len(ranked)
        for kk in ks:
            out[f"top{kk}"].append(pos < kk)
        rr.append(1.0 / (pos + 1))
    res = {kk: float(np.mean(v)) for kk, v in out.items()}
    res["mrr"] = float(np.mean(rr))
    return res


def run_ablation(X, meta, args):
    """Feature ablation (which part of the transition geometry matters) and
    model ablation (RF/HGB/logistic), plus the NN baseline -- all under the
    same protein-grouped split for comparability."""
    from sklearn.model_selection import GroupShuffleSplit
    gss = GroupShuffleSplit(n_splits=1, test_size=args.test_size, random_state=args.seed)
    tr, te = next(gss.split(X, meta["signature"], meta["acc"].values))
    ytr, yte = meta["signature"].values[tr], meta["signature"].values[te]

    print("\n[ablation] feature sets (RandomForest, protein-grouped):")
    feat = {}
    for name, sl in FEATURE_SLICES.items():
        clf = make_classifier(args)
        clf.fit(X[tr, sl], ytr)
        m, _ = topk_metrics(clf, X[te, sl], yte, clf.classes_)
        feat[name] = m
        print(f"  {name:14} top1={m['top1']:.3f} top3={m['top3']:.3f} top5={m['top5']:.3f} mrr={m['mrr']:.3f}")

    print("[ablation] models (full features, protein-grouped):")
    models = {}
    for mdl in ["rf", "hgb", "logistic"]:
        args.model = mdl if mdl != "logistic" else "rf"  # placeholder; build below
        if mdl == "logistic":
            from sklearn.linear_model import LogisticRegression
            clf = LogisticRegression(max_iter=300, n_jobs=-1)
        else:
            args.model = mdl
            clf = make_classifier(args)
        clf.fit(X[tr], ytr)
        m, _ = topk_metrics(clf, X[te], yte, clf.classes_)
        models[mdl] = m
        print(f"  {mdl:14} top1={m['top1']:.3f} top3={m['top3']:.3f} top5={m['top5']:.3f} mrr={m['mrr']:.3f}")
    args.model = "rf"

    nn = nn_baseline(X[tr], ytr, X[te], yte)
    print(f"  {'nn-baseline':14} top1={nn['top1']:.3f} top3={nn['top3']:.3f} top5={nn['top5']:.3f} mrr={nn['mrr']:.3f}")
    return {"feature_ablation": feat, "model_ablation": models, "nn_baseline": nn}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs_dir", default="all_runs")
    ap.add_argument("--centroid_chunks_dir", required=True)
    ap.add_argument("--output_dir", default="results/inverse_design")
    ap.add_argument("--use_topk", type=int, default=1,
                    help="Use top-1..top-k attracted domains as positive transitions (default 1).")
    ap.add_argument("--min_delta", type=float, default=0.0,
                    help="Minimum attraction (delta_dist) to count a transition.")
    ap.add_argument("--n_estimators", type=int, default=300)
    ap.add_argument("--test_size", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--model", choices=["rf", "hgb"], default="rf",
                    help="Multiclass model: rf (RandomForest, default) or hgb.")
    ap.add_argument("--mode",
                    choices=["multiclass", "triple", "both", "loso", "crossclan", "ablation"],
                    default="both",
                    help="multiclass; triple (profile feature); loso (leave-signatures-out); "
                         "crossclan (train one Pfam clan, test another); ablation (feature/model/NN).")
    ap.add_argument("--loso_folds", type=int, default=5)
    ap.add_argument("--loso_test_cap", type=int, default=800,
                    help="Max test transitions ranked per LOSO fold (runtime cap).")
    ap.add_argument("--profile_dir", default="profiles",
                    help="COSMIC signature profiles dir (for triple-scoring).")
    ap.add_argument("--n_neg", type=int, default=3,
                    help="Negative signatures sampled per positive transition (triple mode).")
    ap.add_argument("--design_rules_topk", type=int, default=3)
    args = ap.parse_args(argv)
    os.makedirs(args.output_dir, exist_ok=True)

    print("[inv] loading transitions...")
    trans = load_transitions(args.runs_dir, args.min_delta, args.use_topk)
    print(f"[inv] {len(trans)} candidate transitions "
          f"({trans['origin'].nunique()} origins, {trans['target'].nunique()} targets, "
          f"{trans['signature'].nunique()} signatures, {trans['acc'].nunique()} proteins)")

    print("[inv] loading Pfam centroid library...")
    centroids = load_centroids(args.centroid_chunks_dir)
    X, meta = build_features(trans, centroids)
    print(f"[inv] {len(meta)} transitions with centroid coverage "
          f"({100*len(meta)/max(len(trans),1):.1f}% of candidates); feature dim {X.shape[1]}")

    results = {}
    import joblib

    if args.mode in ("multiclass", "both"):
        # 1) protein-grouped split (generalize to unseen proteins)
        results["protein_grouped"] = run_split(
            X, meta, meta["acc"].values, "protein-grouped", args)
        # 2) held-out-target split (generalize to unseen target domains -- harder)
        results["target_heldout"] = run_split(
            X, meta, meta["target"].values, "target-held-out", args)

        # design-rule table from the protein-grouped model (the usable deliverable)
        export_design_rules(
            results["protein_grouped"]["classifier"],
            np.asarray(results["protein_grouped"]["classes"]),
            centroids, meta,
            os.path.join(args.output_dir, "design_rules.csv"), topk=args.design_rules_topk)

        for name in ("protein_grouped", "target_heldout"):
            clf = results[name].pop("classifier")
            joblib.dump(clf, os.path.join(args.output_dir, f"model_{name}.joblib"))

    if args.mode in ("triple", "both"):
        print("[inv] loading COSMIC signature profiles for triple-scoring...")
        profiles = load_signature_profiles(args.profile_dir)
        print(f"[inv] {len(profiles)} signature profiles loaded")
        tri = run_triple(meta, centroids, profiles, args)
        joblib.dump(tri["clf"], os.path.join(args.output_dir, "model_triple.joblib"))
        results["triple_scoring"] = {"metrics": tri["metrics"], "n_signatures": len(profiles)}

    if args.mode == "loso":
        print("[inv] loading COSMIC signature profiles for leave-signatures-out...")
        profiles = load_signature_profiles(args.profile_dir)
        print(f"[inv] {len(profiles)} signature profiles loaded")
        results["loso"] = run_loso(meta, centroids, profiles, args)

    if args.mode == "crossclan":
        results["crossclan"] = run_crossclan(X, meta, args)

    if args.mode == "ablation":
        results["ablation"] = run_ablation(X, meta, args)

    with open(os.path.join(args.output_dir, "inverse_design_metrics.json"), "w") as f:
        json.dump(results, f, indent=2, default=float)
    print(f"\n[inv] wrote metrics + models + design_rules.csv to {args.output_dir}/")


if __name__ == "__main__":
    main()
