#!/usr/bin/env python3
"""
nnls_collinearity.py

Quantifies the NNLS signature-collinearity failure mode named in the PRISM paper:
the 78 COSMIC signatures are far from orthogonal, so decomposing an observed
substitution spectrum into per-signature exposures (via non-negative least
squares) is ill-conditioned -- collinear signatures absorb one another's
exposure, and a reported exposure vector is not a reliable attribution.

It reports (1) the conditioning of the signature matrix and its pairwise
collinearity, and (2) a non-identifiability demonstration: a spectrum generated
purely by one signature, resampled at realistic counts, is NNLS-decomposed many
times; we measure how often the true signature is NOT the top exposure and how
much of its exposure leaks to its most collinear partners.

USAGE
-----
python nnls_collinearity.py --profile_dir profiles --true_sig SBS17a \
    --counts 500 --trials 500 --output_dir results/collinearity
"""
import argparse
import glob
import os

import numpy as np
from scipy.optimize import nnls


def load_profile(path):
    d = {}
    for ln in open(path).readlines()[1:]:
        q = ln.split()
        if len(q) >= 2 and len(q[0]) >= 7 and q[0][1] == "[" and q[0][5] == "]":
            try:
                d[q[0]] = float(q[1])
            except ValueError:
                pass
    return d


def load_matrix(profile_dir):
    files = sorted(glob.glob(os.path.join(profile_dir, "**", "SBS*_PROFILE.txt"), recursive=True))
    keys, cols = None, {}
    for f in files:
        d = load_profile(f)
        name = os.path.basename(f).split("_")[0].split(".")[0]
        if keys is None:
            keys = sorted(d)
        v = np.array([d.get(k, 0.0) for k in keys])
        s = v.sum()
        cols[name] = v / s if s > 0 else v
    names = list(cols)
    M = np.array([cols[n] for n in names]).T   # 96 x N, columns normalized to sum 1
    return M, names, keys


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--profile_dir", default="profiles")
    ap.add_argument("--true_sig", default="SBS17a")
    ap.add_argument("--counts", type=int, default=500,
                    help="Total substitutions in the simulated spectrum.")
    ap.add_argument("--trials", type=int, default=500)
    ap.add_argument("--output_dir", default="results/collinearity")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    os.makedirs(args.output_dir, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    M, names, keys = load_matrix(args.profile_dir)
    Mn = M / (np.linalg.norm(M, axis=0, keepdims=True) + 1e-9)
    S = Mn.T @ Mn
    np.fill_diagonal(S, 0.0)
    iu = np.triu_indices_from(S, 1)
    cond = float(np.linalg.cond(M))
    print(f"[nnls] signature matrix {M.shape} (96 contexts x {M.shape[1]} signatures)")
    print(f"[nnls] condition number = {cond:.1f}   (orthogonal would be ~1)")
    print(f"[nnls] pairwise cosine: mean {S[iu].mean():.3f}, max {S[iu].max():.3f}, "
          f"pairs>0.8: {int((S[iu] > 0.8).sum())}/{len(iu[0])}")

    ti = names.index(args.true_sig)
    partners = [(names[j], float(S[ti, j])) for j in np.argsort(-S[ti])[:3]]
    print(f"[nnls] {args.true_sig} most-collinear partners: "
          + ", ".join(f"{n} ({c:.2f})" for n, c in partners))

    # Non-identifiability demo: spectrum generated purely by true_sig, resampled.
    p = M[:, ti]
    p = p / p.sum()
    top_true, leak = [], []
    exposures = []
    for _ in range(args.trials):
        obs = rng.multinomial(args.counts, p).astype(float)
        obs = obs / obs.sum()
        x, _ = nnls(M, obs)
        x = x / x.sum() if x.sum() > 0 else x
        exposures.append(x)
        top_true.append(int(np.argmax(x) == ti))
        leak.append(float(x[[names.index(n) for n, _ in partners[1:]]].sum()))
    exposures = np.array(exposures)
    frac_top = float(np.mean(top_true))
    true_exp = exposures[:, ti]
    print(f"\n[nnls] Non-identifiability of {args.true_sig} "
          f"(pure spectrum, {args.counts} counts, {args.trials} resamples):")
    print(f"  true signature recovered as TOP exposure: {frac_top:.1%} of trials")
    print(f"  mean exposure assigned to {args.true_sig}: {true_exp.mean():.2f} "
          f"(should be ~1.0 if identifiable)")
    print(f"  mean exposure leaked to collinear partners: {np.mean(leak):.2f}")

    _save(args.output_dir, cond, S, iu, names, args.true_sig, partners,
          frac_top, float(true_exp.mean()), float(np.mean(leak)), true_exp)
    _plot(true_exp, exposures, names, ti, partners, args.true_sig, args.output_dir)


def _save(out, cond, S, iu, names, true_sig, partners, frac_top, mean_true, mean_leak, true_exp):
    import json
    obj = {"condition_number": round(cond, 1),
           "pairwise_cosine_mean": round(float(S[iu].mean()), 3),
           "pairwise_cosine_max": round(float(S[iu].max()), 3),
           "pairs_over_0.8": int((S[iu] > 0.8).sum()),
           "true_sig": true_sig, "collinear_partners": partners,
           "recovered_as_top_frac": round(frac_top, 3),
           "mean_true_exposure": round(mean_true, 3),
           "mean_leak_to_partners": round(mean_leak, 3)}
    with open(os.path.join(out, "collinearity_summary.json"), "w") as f:
        json.dump(obj, f, indent=2)
    print(f"[nnls] wrote {out}/collinearity_summary.json")


def _plot(true_exp, exposures, names, ti, partners, true_sig, out):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    ax[0].hist(true_exp, bins=30, color="#c0392b", alpha=0.85)
    ax[0].axvline(1.0, ls="--", c="k", label="identifiable (=1.0)")
    ax[0].set_xlabel(f"NNLS exposure recovered for {true_sig}")
    ax[0].set_ylabel("resamples")
    ax[0].set_title(f"Exposure of the true signature\n(spectrum is 100\\% {true_sig})".replace("\\%", "%"))
    ax[0].legend(fontsize=8)
    # mean exposure of true + its collinear partners
    idxs = [ti] + [names.index(n) for n, _ in partners[1:]]
    labs = [true_sig] + [n for n, _ in partners[1:]]
    ax[1].bar(range(len(idxs)), exposures[:, idxs].mean(0),
              yerr=exposures[:, idxs].std(0), capsize=4, color="#2980b9")
    ax[1].set_xticks(range(len(idxs))); ax[1].set_xticklabels(labs, rotation=20)
    ax[1].set_ylabel("mean NNLS exposure")
    ax[1].set_title("Exposure leaks to collinear partners")
    for a in ax:
        a.spines[["top", "right"]].set_visible(False)
    fig.suptitle("NNLS signature attribution is non-identifiable under collinearity")
    fig.tight_layout(); fig.savefig(os.path.join(out, "collinearity.png"), dpi=200)
    print(f"[nnls] wrote {out}/collinearity.png")


if __name__ == "__main__":
    main()
