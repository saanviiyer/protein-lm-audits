#!/usr/bin/env python3
"""Joint table: every genomic-LM number next to the trivial number it must beat.

Reads results/regulatory_baselines.csv and results/regulatory_models_*.csv and
emits results/regulatory_models_joint.csv plus a printed table. No model AUC ever
appears without its baseline on the same row.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

def fmt(v, lo=None, hi=None):
    if not np.isfinite(v):
        return "  -  "
    s = f"{v:.3f}"
    if lo is not None and np.isfinite(lo):
        s += f" [{lo:.3f},{hi:.3f}]"
    return s


def main():
    base = pd.read_csv(ROOT / "results/regulatory_baselines.csv")
    mods = []
    for k in ("nt", "hyena"):
        p = ROOT / f"results/regulatory_models_{k}.csv"
        if p.exists():
            mods.append(pd.read_csv(p))
    mod = pd.concat(mods, ignore_index=True)

    # Positional readout: added AFTER the pre-registered mean/max/cls grid lost on
    # the two positional tasks. It is a diagnostic of the readout, not part of the
    # pre-registered grid, and is labelled as such wherever it appears.
    pos = []
    for k in ("nt", "hyena"):
        p = ROOT / f"results/regulatory_models_positional_{k}.csv"
        if p.exists():
            pos.append(pd.read_csv(p))
    pos = pd.concat(pos, ignore_index=True) if pos else None

    rows = []
    for task in base.task.unique():
        for split in ("shipped", "cluster", "random"):
            bf = base[(base.task == task) & (base.split == split) & (base.orient == "fwd")]
            br = base[(base.task == task) & (base.split == split) & (base.orient == "rc")]
            best = bf.loc[bf.auc.idxmax()]
            bestr = br[br.features == best.features].iloc[0]
            rows.append(dict(task=task, split=split, family="trivial",
                             entry=f"best trivial: {best.features}",
                             auc=best.auc, auc_lo=best.auc_lo, auc_hi=best.auc_hi,
                             auc_rc=bestr.auc, mcc=best.mcc,
                             tpr1=best["tpr@1"], clustercov1=best["clustercov@1"],
                             missrun1=best["missrun@1"], tpr10=best["tpr@10"],
                             clustercov10=best["clustercov@10"], n_test=best.n_test))
            for feat in ("onehot", "kmer5", "pwm"):
                r = bf[bf.features == feat]
                rr = br[br.features == feat]
                if len(r):
                    r = r.iloc[0]
                    rows.append(dict(task=task, split=split, family="trivial",
                                     entry=f"trivial: {feat}", auc=r.auc, auc_lo=r.auc_lo,
                                     auc_hi=r.auc_hi, auc_rc=rr.iloc[0].auc, mcc=r.mcc,
                                     tpr1=r["tpr@1"], clustercov1=r["clustercov@1"],
                                     missrun1=r["missrun@1"], tpr10=r["tpr@10"],
                                     clustercov10=r["clustercov@10"], n_test=r.n_test))
            mm = mod[(mod.task == task) & (mod.split == split)]
            for mname in mm.model.unique():
                sub = mm[mm.model == mname]
                for kind, label in (("cv", "train-CV config"), ("default", "final layer, mean")):
                    sel = sub[(sub.orient == "fwd") &
                              (sub.selected.astype(str).str.contains(kind))]
                    if not len(sel):
                        continue
                    s = sel.iloc[0]
                    sr = sub[(sub.orient == "rc") & (sub.config == s.config)].iloc[0]
                    rows.append(dict(task=task, split=split, family="model",
                                     entry=f"{mname} {label} ({s.config})",
                                     auc=s.auc, auc_lo=s.auc_lo, auc_hi=s.auc_hi,
                                     auc_rc=sr.auc, mcc=s.mcc, tpr1=s["tpr@1"],
                                     clustercov1=s["clustercov@1"], missrun1=s["missrun@1"],
                                     tpr10=s["tpr@10"], clustercov10=s["clustercov@10"],
                                     n_test=s.n_test))
                # best-on-test config, reported only to bound the sweep's headroom
                sel = sub[sub.orient == "fwd"]
                s = sel.loc[sel.auc.idxmax()]
                sr = sub[(sub.orient == "rc") & (sub.config == s.config)].iloc[0]
                rows.append(dict(task=task, split=split, family="model_oracle",
                                 entry=f"{mname} best-of-9 ON TEST ({s.config})",
                                 auc=s.auc, auc_lo=s.auc_lo, auc_hi=s.auc_hi,
                                 auc_rc=sr.auc, mcc=s.mcc, tpr1=s["tpr@1"],
                                 clustercov1=s["clustercov@1"], missrun1=s["missrun@1"],
                                 tpr10=s["tpr@10"], clustercov10=s["clustercov@10"],
                                 n_test=s.n_test))
            if pos is not None:
                pm = pos[(pos.task == task) & (pos.split == split)]
                for mname in pm.model.unique():
                    sub = pm[pm.model == mname]
                    sel = sub[(sub.orient == "fwd") &
                              (sub.selected.astype(str).str.contains("cv"))]
                    if not len(sel):
                        continue
                    s = sel.iloc[0]
                    sr = sub[(sub.orient == "rc") & (sub.config == s.config)].iloc[0]
                    rows.append(dict(task=task, split=split, family="model_positional",
                                     entry=f"{mname} POST-HOC windowed readout ({s.config})",
                                     auc=s.auc, auc_lo=s.auc_lo, auc_hi=s.auc_hi,
                                     auc_rc=sr.auc, mcc=s.mcc, tpr1=s["tpr@1"],
                                     clustercov1=s["clustercov@1"], missrun1=s["missrun@1"],
                                     tpr10=s["tpr@10"], clustercov10=s["clustercov@10"],
                                     n_test=s.n_test))
    out = pd.DataFrame(rows)
    out.to_csv(ROOT / "results/regulatory_models_joint.csv", index=False)

    pd.set_option("display.width", 250, "display.max_rows", 500)
    for task in out.task.unique():
        print(f"\n===== {task} =====")
        for split in ("shipped", "cluster", "random"):
            t = out[(out.task == task) & (out.split == split)]
            print(f"-- {split} (n_test={int(t.n_test.iloc[0])})")
            for _, r in t.iterrows():
                print(f"   {r.entry:46s} auc {fmt(r.auc, r.auc_lo, r.auc_hi):24s} "
                      f"rc {r.auc_rc:.3f}  mcc {r.mcc:+.3f}  "
                      f"tpr@1 {r.tpr1:.3f} cov@1 {r.clustercov1:.3f} miss {int(r.missrun1):3d} "
                      f"tpr@10 {r.tpr10:.3f}")
    print("\nwrote results/regulatory_models_joint.csv")


if __name__ == "__main__":
    main()
