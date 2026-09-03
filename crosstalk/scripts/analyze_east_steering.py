#!/usr/bin/env python3
"""Read results/east_steering_*.csv and decide whether the EAST direction exists.

Every comparison is PAIRED WITHIN ASSAY against the alpha=0 run of the same
assay, and the eval panel is one assay per UniProt protein, so the units are
independent proteins rather than correlated ProteinGym rows.
"""
import argparse, sys
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from run_proxy_ladder import spearman

METRICS = ["mean_entropy", "wt_agreement", "rho", "score_dispersion"]


def paired(df, base):
    """Attach per-assay deltas against that assay's alpha=0 run."""
    b = base.set_index("dms_id")
    out = df.copy()
    for m in METRICS:
        out["d_" + m] = out["mean_entropy"].index.map(lambda i: 0)  # placeholder
        out["d_" + m] = out[m].values - out["dms_id"].map(b[m]).values
    return out


def boot(x, n=10000, seed=0):
    x = np.asarray(x, float)
    r = np.random.default_rng(seed)
    m = r.choice(x, (n, len(x)), replace=True).mean(1)
    return float(x.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def wilcoxon(a, b):
    from scipy.stats import wilcoxon as w
    d = np.asarray(a, float) - np.asarray(b, float)
    if np.allclose(d, 0):
        return 1.0
    return float(w(d).pvalue)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", required=True)
    ap.add_argument("--mode", default="layers", choices=["layers", "alpha"])
    args = ap.parse_args()
    d = pd.read_csv(ROOT / args.file)
    base = d[d.alpha == 0].groupby("dms_id", as_index=False)[METRICS].mean()
    d = d[d.alpha != 0]
    d = paired(d, base)

    print(f"{len(base)} eval proteins; baseline mean entropy "
          f"{base.mean_entropy.mean():.3f}, wt_agree {base.wt_agreement.mean():.3f}, "
          f"rho {base.rho.mean():.3f}\n")

    if args.mode == "layers":
        for a in sorted(d.alpha.unique()):
            print(f"=== alpha = {a:+.2f} (units of mean activation norm) ===")
            print(f"{'layer':>5} " + " ".join(f"{k:>26s}" for k in
                  ["dEnt steer", "dEnt random", "dEnt orth", "dEnt mean", "dEnt steerPerp"]))
            for L in sorted(d.layer.unique()):
                cells = []
                for dr in ["steer", "random", "orth", "mean", "steerperp"]:
                    s = d[(d.layer == L) & (d.alpha == a) & (d.direction == dr)]
                    cells.append(f"{s.d_mean_entropy.mean():+7.3f}" if len(s) else "   n/a ")
                print(f"{L:>5} " + " ".join(f"{c:>26s}" for c in cells))
            print()
        print("=== skill (rho) change, same layout ===")
        for a in sorted(d.alpha.unique()):
            print(f"alpha {a:+.2f}")
            for L in sorted(d.layer.unique()):
                cells = []
                for dr in ["steer", "random", "orth", "mean", "steerperp"]:
                    s = d[(d.layer == L) & (d.alpha == a) & (d.direction == dr)]
                    cells.append(f"{dr}={s.d_rho.mean():+.3f}" if len(s) else f"{dr}=n/a")
                print(f"  L{L:<3} " + "  ".join(cells))
            print()
        # localisation summary: steer minus best null, per layer, at each |alpha|
        print("=== steer advantage over the strongest null, per layer ===")
        print(f"{'layer':>5} {'alpha':>6} {'dEnt steer':>11} {'best null':>10} "
              f"{'excess':>8} {'p(steer>null)':>14} {'dRho steer':>11} {'dRho null':>10}")
        for L in sorted(d.layer.unique()):
            for a in sorted(d.alpha.unique()):
                s = d[(d.layer == L) & (d.alpha == a) & (d.direction == "steer")] \
                    .set_index("dms_id").sort_index()
                if not len(s):
                    continue
                nulls = {}
                for dr in ["random", "orth", "mean", "steerperp"]:
                    n = d[(d.layer == L) & (d.alpha == a) & (d.direction == dr)] \
                        .set_index("dms_id").sort_index()
                    if len(n) == len(s):
                        nulls[dr] = n
                if not nulls:
                    continue
                bn = max(nulls, key=lambda k: abs(nulls[k].d_mean_entropy.mean()))
                n = nulls[bn]
                p = wilcoxon(s.d_mean_entropy.values, n.d_mean_entropy.values)
                print(f"{L:>5} {a:>+6.2f} {s.d_mean_entropy.mean():>+11.3f} "
                      f"{n.d_mean_entropy.mean():>+10.3f} "
                      f"{s.d_mean_entropy.mean()-n.d_mean_entropy.mean():>+8.3f} "
                      f"{p:>14.4f} {s.d_rho.mean():>+11.3f} {n.d_rho.mean():>+10.3f}"
                      f"   (null={bn})")
        return

    # ---- alpha mode: dose-response and selectivity
    L = int(d.layer.iloc[0])
    print(f"layer {L}\n")
    for dr in sorted(d.direction.unique()):
        s = d[d.direction == dr]
        print(f"--- {dr} ---")
        print(f"{'alpha':>7} {'entropy':>9} {'dEnt':>8} {'[95% CI]':>18} "
              f"{'wt_agree':>9} {'rho':>8} {'dRho':>8} {'ent/damage':>11}")
        for a in sorted(s.alpha.unique()):
            r = s[s.alpha == a]
            m, lo, hi = boot(r.d_mean_entropy.values)
            dmg = -r.d_wt_agreement.mean()
            print(f"{a:>+7.3f} {r.mean_entropy.mean():>9.3f} {m:>+8.3f} "
                  f"[{lo:+.3f},{hi:+.3f}] {r.wt_agreement.mean():>9.3f} "
                  f"{r.rho.mean():>8.3f} {r.d_rho.mean():>+8.3f} "
                  f"{(m/dmg if dmg > 1e-6 else float('nan')):>11.2f}")
        # monotonicity of the per-assay dose-response
        rhos = []
        for dms, g in s.groupby("dms_id"):
            g = g.sort_values("alpha")
            if g.alpha.nunique() >= 4:
                rhos.append(spearman(g.alpha.values, g.mean_entropy.values))
        print(f"  per-assay monotonicity of entropy in alpha: mean rho "
              f"{np.mean(rhos):+.3f}, {sum(r > 0 for r in rhos)}/{len(rhos)} positive\n")

    print("=== steer vs nulls at matched |alpha|, paired across assays ===")
    for a in sorted(d.alpha.unique()):
        s = d[(d.alpha == a) & (d.direction == "steer")].set_index("dms_id").sort_index()
        line = [f"alpha {a:+.3f}  steer dEnt {s.d_mean_entropy.mean():+.3f} "
                f"dRho {s.d_rho.mean():+.3f}"]
        for dr in ["random", "orth", "mean", "steerperp"]:
            n = d[(d.alpha == a) & (d.direction == dr)].set_index("dms_id").sort_index()
            if len(n) != len(s):
                continue
            line.append(f"| {dr} dEnt {n.d_mean_entropy.mean():+.3f} "
                        f"(p={wilcoxon(s.d_mean_entropy.values, n.d_mean_entropy.values):.4f}) "
                        f"dRho {n.d_rho.mean():+.3f} "
                        f"(p={wilcoxon(s.d_rho.values, n.d_rho.values):.4f})")
        print("  " + " ".join(line))

    print("\n=== does entropy still forecast skill after steering? ===")
    print(f"{'direction':>9} {'alpha':>7} {'rho(entropy, baseline skill)':>30} "
          f"{'rho(entropy, steered skill)':>30}")
    b = base.set_index("dms_id")
    for dr in sorted(d.direction.unique()):
        for a in sorted(d[d.direction == dr].alpha.unique()):
            r = d[(d.direction == dr) & (d.alpha == a)].sort_values("dms_id")
            print(f"{dr:>9} {a:>+7.3f} "
                  f"{spearman(r.mean_entropy.values, r.dms_id.map(b.rho).values):>30.3f} "
                  f"{spearman(r.mean_entropy.values, r.rho.values):>30.3f}")
    print(f"{'(alpha=0)':>9} {0.0:>+7.3f} "
          f"{spearman(base.mean_entropy.values, base.rho.values):>30.3f} "
          f"{spearman(base.mean_entropy.values, base.rho.values):>30.3f}")


if __name__ == "__main__":
    main()
