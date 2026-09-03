"""The bar: what a position weight matrix and a k-mer count get on the
Nucleotide Transformer's own downstream benchmark.

Run this before any neural number exists, so that the neural number arrives into
a table that already has a floor in it.

  ./.venv-glm/bin/python scripts/run_regulatory_baselines.py --n-train 8000
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import regulatory_data as RD          # noqa: E402
import regulatory_eval as RE          # noqa: E402
import regulatory_features as RF      # noqa: E402

SPLITS = {"shipped": RD.shipped_split, "cluster": RD.cluster_split, "random": RD.random_split}


def run_task(task, n_train, seed, out_rows, splits):
    d = RD.load_task(task)
    y = d["label"].values
    seqs = d["sequence"].tolist()
    clusters = RD.locus_clusters(d)
    rc = [RD.revcomp(s) for s in seqs]

    cache = {}
    for name, fn in RF.FEATURE_SETS.items():
        t0 = time.time()
        cache[name] = (fn(seqs), fn(rc))
        print(f"  [{task}] features {name} {cache[name][0].shape} in {time.time()-t0:.1f}s", flush=True)

    for split_name in splits:
        tr, te = SPLITS[split_name](d) if split_name == "shipped" else SPLITS[split_name](d, seed)
        tr = RD.subsample(tr, y, n_train, seed)
        ytr, yte = y[tr], y[te]
        y_bin = (yte == 1).astype(int)
        cl_te = clusters[te]

        # PWM: counting only, no probe, no hyperparameter.
        pwm = RF.PWM().fit([seqs[i] for i in tr], ytr)
        for tag, source in (("fwd", seqs), ("rc", rc)):
            s = pwm.score([source[i] for i in te])
            m = RE.evaluate(s, yte, y_bin, cl_te)
            lo, hi = RE.cluster_bootstrap_auc(yte, s, cl_te) if tag == "fwd" else (np.nan, np.nan)
            out_rows.append(dict(task=task, split=split_name, features="pwm", family="trivial",
                                 n_train=len(tr), n_test=len(te), n_features=4 * len(seqs[0]) * len(np.unique(ytr)),
                                 orient=tag, C=np.nan, auc_lo=lo, auc_hi=hi, **m))
            print(f"  [{task}/{split_name}] pwm {tag}: auc={m['auc']:.3f}", flush=True)

        for name, (Xf, Xr) in cache.items():
            t0 = time.time()
            sc, model, C = RE.fit_probe(Xf[tr], ytr, clusters[tr], seed)
            for tag, X in (("fwd", Xf), ("rc", Xr)):
                s = RE._decision(model, sc.transform(X[te]))
                m = RE.evaluate(s, yte, y_bin, cl_te)
                lo, hi = RE.cluster_bootstrap_auc(yte, s, cl_te) if tag == "fwd" else (np.nan, np.nan)
                out_rows.append(dict(task=task, split=split_name, features=name, family="trivial",
                                     n_train=len(tr), n_test=len(te), n_features=Xf.shape[1],
                                     orient=tag, C=C, auc_lo=lo, auc_hi=hi, **m))
            print(f"  [{task}/{split_name}] {name} fwd auc={out_rows[-2]['auc']:.3f} "
                  f"rc auc={out_rows[-1]['auc']:.3f} (C={C}, {time.time()-t0:.0f}s)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", nargs="*", default=list(RD.TASKS))
    ap.add_argument("--splits", nargs="*", default=list(SPLITS))
    ap.add_argument("--n-train", type=int, default=8000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/regulatory_baselines.csv")
    a = ap.parse_args()

    rows = []
    for t in a.tasks:
        print(f"=== {t} ===", flush=True)
        run_task(t, a.n_train, a.seed, rows, a.splits)
        pd.DataFrame(rows).to_csv(a.out, index=False)
    print(f"wrote {a.out}  ({len(rows)} rows)")


if __name__ == "__main__":
    main()
