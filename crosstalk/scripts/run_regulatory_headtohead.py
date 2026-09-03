#!/usr/bin/env python3
"""Paired model-vs-trivial comparison on the regulatory tasks.

Separate AUCs with separate intervals cannot say whether a model beats a k-mer
count, because the two are scored on the same test rows and are therefore
correlated. This scores both on the same rows and bootstraps the DIFFERENCE,
resampling locus clusters rather than rows.

The trivial competitor is chosen to favour the baseline: for each task and split
it is whichever of the seven trivial feature sets already scored highest on that
split's test set. The model's config is chosen without any test number, by
grouped CV on training rows (`selected == "cv"` in results/regulatory_models_*).
So the baseline gets an oracle and the model does not, and a model win under that
handicap is a real win.

  HF_HOME=<AITHYRA cache> ./.venv-glm/bin/python scripts/run_regulatory_headtohead.py --model nt
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import regulatory_data as RD          # noqa: E402
import regulatory_eval as RE          # noqa: E402
import regulatory_embed as EMB        # noqa: E402
import regulatory_features as RF      # noqa: E402
from run_regulatory_models import SPLITS, BATCH   # noqa: E402


def trivial_scores(name, seqs, rc, tr, te, ytr, groups_tr):
    if name == "pwm":
        pwm = RF.PWM().fit([seqs[i] for i in tr], ytr)
        return pwm.score([seqs[i] for i in te]), pwm.score([rc[i] for i in te])
    fn = RF.FEATURE_SETS[name]
    Xf = fn(seqs)
    Xr = fn(rc)
    sc, m, _ = RE.fit_probe(Xf[tr], ytr, groups_tr, 0)
    return (RE._decision(m, sc.transform(Xf[te])),
            RE._decision(m, sc.transform(Xr[te])))


def paired_cluster_bootstrap(y, sa, sb, clusters, n=1000, seed=0):
    """Bootstrap AUC(a) - AUC(b) by resampling locus clusters."""
    rng = np.random.default_rng(seed)
    uc = np.unique(clusters)
    idx_by = {c: np.where(clusters == c)[0] for c in uc}
    diffs = []
    for _ in range(n):
        pick = rng.choice(uc, size=len(uc), replace=True)
        idx = np.concatenate([idx_by[c] for c in pick])
        try:
            da = RE._auc(y[idx], sa[idx]) - RE._auc(y[idx], sb[idx])
        except Exception:
            continue
        if np.isfinite(da):
            diffs.append(da)
    d = np.asarray(diffs)
    if len(d) < 50:
        return (np.nan,) * 4
    return (float(d.mean()), float(np.percentile(d, 2.5)),
            float(np.percentile(d, 97.5)), float((d <= 0).mean()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=list(EMB.EMBEDDERS), required=True)
    ap.add_argument("--tasks", nargs="*", default=list(RD.TASKS))
    ap.add_argument("--n-train", type=int, default=8000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--boot", type=int, default=1000)
    a = ap.parse_args()

    base = pd.read_csv(ROOT / "results/regulatory_baselines.csv")
    base = base[base.orient == "fwd"]
    mod = pd.read_csv(ROOT / f"results/regulatory_models_{a.model}.csv")
    mod = mod[(mod.orient == "fwd") & (mod.selected.astype(str).str.startswith("cv"))]

    emb = EMB.EMBEDDERS[a.model]()
    rows = []
    for task in a.tasks:
        d = RD.load_task(task)
        y = d["label"].values
        seqs = d["sequence"].tolist()
        rc = [RD.revcomp(s) for s in seqs]
        clusters = RD.locus_clusters(d)
        E = {}
        for tag, S in (("fwd", seqs), ("rc", rc)):
            E[tag] = EMB.embed_all(emb, S, BATCH.get(len(seqs[0]), 32), progress=100,
                                   tag=f"{task}/{tag}")
        for split_name in SPLITS:
            tr, te = SPLITS[split_name](d) if split_name == "shipped" else SPLITS[split_name](d, a.seed)
            tr = RD.subsample(tr, y, a.n_train, a.seed)
            ytr, yte = y[tr], y[te]
            cl_te = clusters[te]

            b = base[(base.task == task) & (base.split == split_name)]
            bname = b.loc[b.auc.idxmax(), "features"]
            m = mod[(mod.task == task) & (mod.split == split_name)]
            assert len(m) == 1, (task, split_name, len(m))
            cfg = m.iloc[0]["config"]

            bs_f, bs_r = trivial_scores(bname, seqs, rc, tr, te, ytr, clusters[tr])
            sc, pm, C = RE.fit_probe(E["fwd"][cfg][tr].astype(np.float32), ytr, clusters[tr], a.seed)
            ms_f = RE._decision(pm, sc.transform(E["fwd"][cfg][te].astype(np.float32)))
            ms_r = RE._decision(pm, sc.transform(E["rc"][cfg][te].astype(np.float32)))

            for tag, ms, bs in (("fwd", ms_f, bs_f), ("rc", ms_r, bs_r)):
                mean, lo, hi, p = paired_cluster_bootstrap(yte, ms, bs, cl_te, a.boot, a.seed)
                rows.append(dict(task=task, split=split_name, orient=tag,
                                 model=emb.name, model_config=cfg, trivial=bname,
                                 auc_model=RE._auc(yte, ms), auc_trivial=RE._auc(yte, bs),
                                 d_auc=mean, d_lo=lo, d_hi=hi, p_model_not_better=p,
                                 n_test=len(te), n_clusters=len(np.unique(cl_te))))
                print(f"  [{task}/{split_name}/{tag}] {emb.name}:{cfg} {rows[-1]['auc_model']:.3f} "
                      f"vs {bname} {rows[-1]['auc_trivial']:.3f}  d={mean:+.3f} "
                      f"[{lo:+.3f},{hi:+.3f}] p={p:.3f}", flush=True)
        pd.DataFrame(rows).to_csv(ROOT / f"results/regulatory_models_headtohead_{a.model}.csv", index=False)
    print(f"wrote results/regulatory_models_headtohead_{a.model}.csv ({len(rows)} rows)")


if __name__ == "__main__":
    main()
