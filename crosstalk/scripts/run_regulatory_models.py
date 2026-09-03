#!/usr/bin/env python3
"""Does a pretrained genomic LM beat a PWM, a k-mer count, or a positional one-hot?

Same tasks, same three splits, same seed, same 8000-row stratified training
subsample, same grouped-CV probe and same metrics as
scripts/run_regulatory_baselines.py. The only thing that changes is where the
features come from, so any difference is the model and not the protocol.

  HF_HOME=<AITHYRA cache> ./.venv-glm/bin/python scripts/run_regulatory_models.py \
      --model nt --n-train 8000

Every row carries the (layer, pooling) it came from. Two rows per task-split are
flagged: `selected=cv` is the config chosen by grouped 3-fold CV on the training
rows only, `selected=default` is the pre-registered final-layer mean-pooling
config. Neither is chosen with a test number.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import regulatory_data as RD          # noqa: E402
import regulatory_eval as RE          # noqa: E402
import regulatory_embed as EMB        # noqa: E402
from crosstalk import coverage as COV  # noqa: E402

SPLITS = {"shipped": RD.shipped_split, "cluster": RD.cluster_split, "random": RD.random_split}
BATCH = {300: 128, 400: 96, 600: 64, 1000: 32}


def cv_auc(X, y, groups, seed=0, c_grid=RE.C_GRID):
    """Best grouped-3-fold CV AUC over the C grid. Training rows only."""
    sc = StandardScaler().fit(X)
    Xs = sc.transform(X)
    if len(np.unique(groups)) < 3:
        return np.nan
    gkf = GroupKFold(n_splits=3)
    best = -np.inf
    for C in c_grid:
        vals = []
        for tr, va in gkf.split(Xs, y, groups):
            if len(np.unique(y[tr])) < 2:
                continue
            m = LogisticRegression(C=C, max_iter=2000, random_state=seed).fit(Xs[tr], y[tr])
            vals.append(RE._auc(y[va], RE._decision(m, Xs[va])))
        if vals and np.mean(vals) > best:
            best = float(np.mean(vals))
    return best


def check_coverage_agreement(y_bin, s):
    """crosstalk/coverage.py and regulatory_eval must give the same operating point."""
    thr_a = COV.operating_threshold(s, y_bin == 0, 0.01)
    cov_a = COV.coverage(s, y_bin == 1, y_bin == 0, 0.01)["coverage"]
    tpr_b, thr_b, _ = RE.tpr_at_fpr(y_bin, s, 0.01)
    assert abs(thr_a - thr_b) < 1e-9 and abs(cov_a - tpr_b) < 1e-12, \
        f"coverage mismatch {thr_a} {thr_b} {cov_a} {tpr_b}"


def run_task(task, emb, n_train, seed, splits, rows, sweep_rows, default=None):
    d = RD.load_task(task)
    y = d["label"].values
    seqs = d["sequence"].tolist()
    clusters = RD.locus_clusters(d)
    rc = [RD.revcomp(s) for s in seqs]
    L = len(seqs[0])
    bs = BATCH.get(L, 32)

    t0 = time.time()
    Ef = EMB.embed_all(emb, seqs, bs, progress=40, tag=f"{task}/fwd")
    print(f"  [{task}] forward embeddings {len(Ef)} configs "
          f"{next(iter(Ef.values())).shape} in {time.time()-t0:.0f}s", flush=True)
    t0 = time.time()
    Er = EMB.embed_all(emb, rc, bs, progress=40, tag=f"{task}/rc")
    print(f"  [{task}] revcomp embeddings in {time.time()-t0:.0f}s", flush=True)

    configs = list(Ef)
    default = default or f"L{emb.n_layers}_mean"
    assert default in configs, (default, configs)

    for split_name in splits:
        tr, te = SPLITS[split_name](d) if split_name == "shipped" else SPLITS[split_name](d, seed)
        tr = RD.subsample(tr, y, n_train, seed)
        ytr, yte = y[tr], y[te]
        y_bin = (yte == 1).astype(int)
        cl_te = clusters[te]

        # ---- config selection, training rows only -------------------------
        cvs = {}
        for cname in configs:
            Xtr = Ef[cname][tr].astype(np.float32)
            cvs[cname] = cv_auc(Xtr, ytr, clusters[tr], seed)
            sweep_rows.append(dict(model=emb.name, task=task, split=split_name,
                                   config=cname, train_cv_auc=cvs[cname]))
        best_cfg = max(configs, key=lambda c: (cvs[c] if np.isfinite(cvs[c]) else -np.inf))
        print(f"  [{task}/{split_name}] train-CV picks {best_cfg} "
              f"({cvs[best_cfg]:.3f}); default {default} ({cvs[default]:.3f})", flush=True)

        # ---- every config gets a test row; two of them are flagged --------
        for cname in configs:
            sc, model, C = RE.fit_probe(Ef[cname][tr].astype(np.float32), ytr,
                                        clusters[tr], seed)
            flag = ("cv" if cname == best_cfg else "") + \
                   ("+default" if cname == default else "")
            for tag, E in (("fwd", Ef), ("rc", Er)):
                s = RE._decision(model, sc.transform(E[cname][te].astype(np.float32)))
                m = RE.evaluate(s, yte, y_bin, cl_te)
                s1 = s if np.ndim(s) == 1 else s[:, 1]
                check_coverage_agreement(y_bin, s1)
                lo, hi = RE.cluster_bootstrap_auc(yte, s, cl_te) if tag == "fwd" else (np.nan, np.nan)
                rows.append(dict(task=task, split=split_name, features=f"{emb.name}:{cname}",
                                 family="model", model=emb.name, config=cname,
                                 layer=int(cname.split("_")[0][1:]), pooling=cname.split("_")[1],
                                 selected=flag.strip("+"), train_cv_auc=cvs[cname],
                                 n_train=len(tr), n_test=len(te), n_features=Ef[cname].shape[1],
                                 orient=tag, C=C, auc_lo=lo, auc_hi=hi, **m))
            if flag:
                print(f"    {cname:10s} [{flag:10s}] fwd auc={rows[-2]['auc']:.3f} "
                      f"rc auc={rows[-1]['auc']:.3f} (C={C})", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=list(EMB.EMBEDDERS), required=True)
    ap.add_argument("--tasks", nargs="*", default=list(RD.TASKS))
    ap.add_argument("--splits", nargs="*", default=list(SPLITS))
    ap.add_argument("--n-train", type=int, default=8000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--poolings", nargs="*", default=None,
                    help="override the pooling grid; used for the positional-readout arm")
    ap.add_argument("--default-config", default=None,
                    help="config name to flag as the pre-registered default")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    emb = EMB.EMBEDDERS[a.model]()
    if a.poolings:
        emb.poolings = tuple(a.poolings)
    print(f"{emb.name} on {emb.device}: {emb.n_layers} layers, dim {emb.dim}, "
          f"probe layers {emb.layers}, poolings {emb.poolings}", flush=True)

    out = Path(a.out or f"results/regulatory_models_{a.model}.csv")
    sweep = out.with_name(out.stem + "_layersweep.csv")
    rows, sweep_rows = [], []
    for t in a.tasks:
        print(f"=== {t} ({emb.name}) ===", flush=True)
        run_task(t, emb, a.n_train, a.seed, a.splits, rows, sweep_rows, a.default_config)
        pd.DataFrame(rows).to_csv(ROOT / out, index=False)
        pd.DataFrame(sweep_rows).to_csv(ROOT / sweep, index=False)
    print(f"wrote {out} ({len(rows)} rows) and {sweep}")


if __name__ == "__main__":
    main()
