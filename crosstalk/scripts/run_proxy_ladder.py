"""Audit protein-LM proxies against a two-sided specificity ground truth.

The task that matters is not "does this proxy correlate with binding" but
"can it tell a specific binder from a promiscuous one" -- two variants that both
bind ParE3 well, where only one avoids ParE2. Reported alongside trivial
baselines, per the Gauntlet pattern: a proxy that cannot beat counting mutations
is not measuring specificity.
"""
import argparse, csv, sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from crosstalk.boltz import MUT_POSITIONS, PARD3
from crosstalk.landscape import load_pard3
from crosstalk.plm import ESMScorer, complex_context

LADDER = ["facebook/esm2_t6_8M_UR50D", "facebook/esm2_t12_35M_UR50D",
          "facebook/esm2_t30_150M_UR50D", "facebook/esm2_t33_650M_UR50D"]
PARAMS = {"facebook/esm2_t6_8M_UR50D": 8, "facebook/esm2_t12_35M_UR50D": 35,
          "facebook/esm2_t30_150M_UR50D": 150, "facebook/esm2_t33_650M_UR50D": 650}


def _rank(x):
    x = np.asarray(x, float)
    order = np.argsort(x)
    r = np.empty(len(x)); r[order] = np.arange(len(x), dtype=float)
    _, inv, cnt = np.unique(x, return_inverse=True, return_counts=True)
    means = np.zeros(len(cnt)); np.add.at(means, inv, r); means /= cnt
    return means[inv]


def spearman(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    return float(np.corrcoef(_rank(a[ok]), _rank(b[ok]))[0, 1]) if ok.sum() > 2 else np.nan


def auc(scores, labels):
    scores, labels = np.asarray(scores, float), np.asarray(labels, bool)
    pos, neg = scores[labels], scores[~labels]
    if len(pos) == 0 or len(neg) == 0:
        return np.nan
    r = _rank(np.concatenate([pos, neg]))
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) - 1) / 2) / (len(pos) * len(neg)))


def boot_ci(scores, labels, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    scores, labels = np.asarray(scores, float), np.asarray(labels, bool)
    vals = []
    for _ in range(n):
        i = rng.integers(0, len(scores), len(scores))
        if labels[i].sum() in (0, len(i)):
            continue
        vals.append(auc(scores[i], labels[i]))
    return (np.percentile(vals, 2.5), np.percentile(vals, 97.5)) if vals else (np.nan, np.nan)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=LADDER)
    ap.add_argument("--out", default="results/proxy_ladder.csv")
    args = ap.parse_args()

    L = load_pard3()
    variants = L.seqs
    w3, w2 = L.F[:, 0], L.F[:, 1]

    specific = (w3 >= 0.8) & (w2 <= 0.2)
    promisc = (w3 >= 0.8) & (w2 >= 0.6)
    mask = specific | promisc
    lab = specific[mask]
    print(f"{len(variants)} variants; discrimination set = {int(mask.sum())} "
          f"({int(specific.sum())} specific vs {int(promisc.sum())} promiscuous)\n")

    wt = "".join(PARD3[p - 1] for p in MUT_POSITIONS)
    nmut = np.array([sum(a != b for a, b in zip(v, wt)) for v in variants], float)

    rows = []
    base_auc = auc(-nmut[mask], lab)
    lo, hi = boot_ci(-nmut[mask], lab)
    print(f"{'BASELINE mutation count':38s} AUC {base_auc:.3f} [{lo:.3f}, {hi:.3f}]\n")
    rows.append(dict(model="baseline:mutation_count", params_M=0, mode="trivial",
                     auc_specific_vs_promiscuous=base_auc, auc_lo=lo, auc_hi=hi,
                     rho_on_target=spearman(-nmut, w3), rho_margin=spearman(-nmut, w3 - w2)))

    for name in args.models:
        print(f"=== {name} ===", flush=True)
        sc = ESMScorer(name)

        blind = sc.score_variants(variants, context=None)
        a3 = sc.score_variants(variants, context=complex_context("ParE3"))
        a2 = sc.score_variants(variants, context=complex_context("ParE2"))
        aware_margin = a3 - a2

        for mode, s in (("partner_blind", blind),
                        ("partner_aware_ParE3", a3),
                        ("partner_aware_margin", aware_margin)):
            a = auc(s[mask], lab)
            lo, hi = boot_ci(s[mask], lab)
            rows.append(dict(model=name, params_M=PARAMS.get(name, np.nan), mode=mode,
                             auc_specific_vs_promiscuous=a, auc_lo=lo, auc_hi=hi,
                             rho_on_target=spearman(s, w3), rho_margin=spearman(s, w3 - w2)))
            flag = "  <- chance" if lo <= 0.5 <= hi else ""
            print(f"  {mode:22s} AUC {a:.3f} [{lo:.3f}, {hi:.3f}]  "
                  f"rho_on={spearman(s, w3):+.3f} rho_margin={spearman(s, w3-w2):+.3f}{flag}",
                  flush=True)

        # does partner-awareness change the score at all?
        d = float(np.abs(a3 - a2).mean())
        print(f"  mean |score(.|ParE3) - score(.|ParE2)| = {d:.4f}"
              f"   (0 would mean the partner is being ignored)\n", flush=True)
        del sc

    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
