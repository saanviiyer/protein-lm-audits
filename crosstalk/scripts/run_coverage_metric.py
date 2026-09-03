#!/usr/bin/env python3
"""What do current proxies score on a coverage-shaped metric?

Detectors are compared on the usual per-instance numbers (AUC, true positive rate
at a false-positive budget) and on coverage quantities that describe the missed
set as a set: how many viable variants slip through, whether they form one
mutually-reachable region or scattered points, how many uniform draws reach one,
and how often ordinary hill-climbing on measured function finishes on one.

The last is the operationally important one and involves no adversarial search at
all. It asks whether the detector is blind precisely where normal optimisation
converges.

The landscape is benign: a bacterial toxin-antitoxin pair used throughout this
project as a measurement testbed. Nothing here is specific to it beyond needing
an exhaustively measured space and a neighbour relation.
"""
import argparse, csv, sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))

from crosstalk.boltz import MUT_POSITIONS, PARD3
from crosstalk.landscape import load_pard3
from crosstalk import coverage as cov
from run_proxy_ladder import auc
from run_readout_probe import ALPHABET, onehot, residue_folds, ridge_fit, ridge_pred


def esm_scores(variants):
    from crosstalk.plm import ESMScorer
    sc = ESMScorer("facebook/esm2_t33_650M_UR50D")
    return sc.score_variants(variants)


def nt_scores(variants):
    from crosstalk import glm
    sc = glm.NTScorer()
    return sc.score_variants_masked_marginal(
        variants, glm.load_cds()["ParD3"]["cds"], cds_offset=0)


def supervised_scores(variants, y, lam=100.0):
    """One-hot ridge, out of fold under a residue-held-out split.

    Included as the best case a labelled model reaches, so the coverage numbers
    are not read as a limitation of zero-shot scoring alone.
    """
    X = onehot(variants)
    tot, cnt = np.zeros(len(y)), np.zeros(len(y))
    for tr, te in residue_folds(variants, 5, 0):
        if tr.sum() < 50 or te.sum() == 0:
            continue
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-8
        w = ridge_fit((X[tr] - mu) / sd, y[tr], lam)
        tot[te] += ridge_pred((X[te] - mu) / sd, w); cnt[te] += 1
    return np.where(cnt > 0, tot / np.maximum(cnt, 1), np.nan)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--functional-thr", type=float, default=0.8)
    ap.add_argument("--negative-thr", type=float, default=0.5)
    ap.add_argument("--fprs", type=float, nargs="+", default=[0.10, 0.01])
    ap.add_argument("--starts", type=int, default=2000)
    ap.add_argument("--out", default="results/coverage_metric.csv")
    args = ap.parse_args()

    L = load_pard3()
    variants = L.seqs
    w3 = L.F[:, 0]
    functional = w3 >= args.functional_thr
    negative = w3 < args.negative_thr
    nb = cov.build_neighbours(L)
    print(f"{len(variants)} variants: {int(functional.sum())} viable "
          f"(W>={args.functional_thr}), {int(negative.sum())} negative "
          f"(W<{args.negative_thr})\n")

    wt = "".join(PARD3[p - 1] for p in MUT_POSITIONS)
    nmut = np.array([sum(a != b for a, b in zip(v, wt)) for v in variants], float)

    print("scoring detectors ...", flush=True)
    detectors = {"mutation count (trivial)": -nmut}
    try:
        detectors["ESM-2 650M likelihood"] = esm_scores(variants)
    except Exception as e:
        print(f"  ESM unavailable: {e}")
    try:
        detectors["NT-v2 50M on the CDS"] = nt_scores(variants)
    except Exception as e:
        print(f"  NT unavailable: {e}")
    detectors["supervised one-hot (held-out)"] = supervised_scores(variants, w3)
    rng = np.random.default_rng(0)
    detectors["random (null)"] = rng.normal(size=len(variants))

    # how a hill-climb behaves is a property of the landscape, not the detector
    peaks_ref = None
    rows = []
    for name, s in detectors.items():
        s = np.asarray(s, float)
        ok = np.isfinite(s)
        a = auc(s[ok & (functional | negative)],
                functional[ok & (functional | negative)])
        print(f"\n=== {name} ===")
        print(f"  per-instance AUC (viable vs negative): {a:.3f}")
        for fpr in args.fprs:
            c = cov.coverage(np.where(ok, s, -np.inf), functional, negative, fpr)
            comp = cov.components(c["escape"], nb)
            draws = cov.expected_draws_to_escape(c["escape"])
            loc = cov.local_search_escape_rate(w3, c["escape"], nb, args.starts)
            topk = cov.top_k_coverage(np.where(ok, s, -np.inf), w3, functional,
                                      c["threshold"])
            print(f"  FPR {fpr:.0%}: coverage {c['coverage']:.3f}   "
                  f"missed {c['n_escape']:3d}   "
                  f"components {comp['n_components']:3d} "
                  f"(largest {comp['largest']:3d} = {comp['largest_frac']:.0%})")
            print(f"          uniform draws to a miss {draws:8.1f}   "
                  f"hill-climbs ending on a miss {loc['escape_rate']:.3f}")
            print(f"          coverage of the best viable variants: "
                  f"top10 {topk['coverage_top10']:.2f}  top50 {topk['coverage_top50']:.2f}"
                  f"  top100 {topk['coverage_top100']:.2f}")
            rows.append(dict(detector=name, auc=a, fpr=fpr,
                             coverage=c["coverage"], n_escape=c["n_escape"],
                             n_components=comp["n_components"],
                             largest_component=comp["largest"],
                             largest_frac=comp["largest_frac"],
                             singletons=comp["singletons"],
                             uniform_draws=draws,
                             local_escape_rate=loc["escape_rate"],
                             distinct_peaks=loc["distinct_peaks"],
                             escape_peaks=loc["escape_peaks"], **topk))
            peaks_ref = loc

    print(f"\nlandscape reference: hill-climbing on measured function from "
          f"{peaks_ref['n_starts']} starts reaches {peaks_ref['distinct_peaks']} "
          f"distinct local optima")

    out = ROOT / args.out
    keys = sorted({k for r in rows for k in r})
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
