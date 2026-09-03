"""Score the Boltz structural proxy on the thing it is used for: specificity.

Reports, in the Gauntlet style, both the proxy's correlation with measured
fitness AND whether a trivial baseline matches it. A proxy that cannot beat
"count the mutations" is not measuring what it claims.
"""
import argparse, csv, sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from crosstalk.boltz import wild_type_code


def spearman(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    if len(a) < 3:
        return float("nan")
    ra, rb = _rank(a), _rank(b)
    return float(np.corrcoef(ra, rb)[0, 1])


def _rank(x):
    order = np.argsort(x)
    r = np.empty(len(x), float)
    r[order] = np.arange(len(x), dtype=float)
    # average ties
    _, inv, cnt = np.unique(x, return_inverse=True, return_counts=True)
    means = np.zeros(len(cnt))
    np.add.at(means, inv, r)
    means /= cnt
    return means[inv]


def auc(scores, labels):
    scores, labels = np.asarray(scores, float), np.asarray(labels, bool)
    ok = np.isfinite(scores)
    scores, labels = scores[ok], labels[ok]
    pos, neg = scores[labels], scores[~labels]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    r = _rank(np.concatenate([pos, neg]))
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) - 1) / 2) / (len(pos) * len(neg)))


def perm_p(scores, labels, n_max=200_000, seed=0):
    """One-sided permutation p for an AUC: exact if the label space is small."""
    import itertools, math
    scores = np.asarray(scores, float); labels = np.asarray(labels, bool)
    ok = np.isfinite(scores); scores, labels = scores[ok], labels[ok]
    n, k = len(scores), int(labels.sum())
    if n < 3 or k == 0 or k == n:
        return float("nan"), float("nan"), 0
    obs = auc(scores, labels)
    total = math.comb(n, k)
    if total <= n_max:
        nulls = np.empty(total)
        for j, c in enumerate(itertools.combinations(range(n), k)):
            l = np.zeros(n, bool); l[list(c)] = True
            nulls[j] = auc(scores, l)
        exact = True
    else:
        rng = np.random.default_rng(seed)
        nulls = np.array([auc(scores, rng.permutation(labels)) for _ in range(n_max)])
        total, exact = n_max, False
    return obs, float((nulls >= obs).mean()), (total if exact else -total)


def auc_ci(scores, labels, n_boot=10_000, seed=0):
    """Stratified bootstrap 95% CI for an AUC."""
    scores = np.asarray(scores, float); labels = np.asarray(labels, bool)
    ok = np.isfinite(scores); scores, labels = scores[ok], labels[ok]
    pos, neg = np.where(labels)[0], np.where(~labels)[0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    out = np.empty(n_boot)
    for b in range(n_boot):
        i = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
        out[b] = auc(scores[i], labels[i])
    out = out[np.isfinite(out)]
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def paired_auc_diff(a, b, labels, n_boot=10_000, seed=0):
    """Bootstrap CI for AUC(a) - AUC(b) on the SAME variants (paired)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    labels = np.asarray(labels, bool)
    pos, neg = np.where(labels)[0], np.where(~labels)[0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    out = np.empty(n_boot)
    for j in range(n_boot):
        i = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
        out[j] = auc(a[i], labels[i]) - auc(b[i], labels[i])
    out = out[np.isfinite(out)]
    return (float(auc(a, labels) - auc(b, labels)),
            float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5)))


def partial_spearman(x, y, z):
    """Spearman of x,y with rank-linear effect of z removed from both."""
    x, y, z = (_rank(np.asarray(v, float)) for v in (x, y, z))
    ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    x, y, z = x[ok], y[ok], z[ok]
    if len(x) < 4:
        return float("nan")
    Z = np.column_stack([np.ones(len(z)), z])
    rx = x - Z @ np.linalg.lstsq(Z, x, rcond=None)[0]
    ry = y - Z @ np.linalg.lstsq(Z, y, rcond=None)[0]
    if rx.std() == 0 or ry.std() == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inp", default="results/boltz_proxy.csv")
    args = ap.parse_args()
    p = Path(args.inp)
    if not p.exists():
        sys.exit(f"{p} not found. Run scripts/run_boltz_proxy.py on a GPU host first.")

    rows = list(csv.DictReader(open(p)))
    f = lambda k: np.array([float(r[k]) if r[k] not in ("", "None") else np.nan for r in rows])
    w3, w2 = f("W_ParE3"), f("W_ParE2")
    i3, i2 = f("iptm_ParE3"), f("iptm_ParE2")
    wt = wild_type_code()
    nmut = np.array([sum(a != b for a, b in zip(r["variant"], wt)) for r in rows], float)

    print(f"n = {len(rows)} variants\n")
    print("PER-PARTNER: does ipTM track measured binding?")
    print(f"  rho(ipTM_ParE3, W_ParE3) = {spearman(i3, w3):+.3f}")
    print(f"  rho(ipTM_ParE2, W_ParE2) = {spearman(i2, w2):+.3f}")

    print("\nSPECIFICITY: does the ipTM margin track the measured margin?")
    print(f"  rho(ipTM_E3 - ipTM_E2, W_E3 - W_E2) = {spearman(i3 - i2, w3 - w2):+.3f}")

    specific = (w3 >= 0.8) & (w2 <= 0.2)
    promisc = (w3 >= 0.8) & (w2 >= 0.6)
    mask = specific | promisc
    print("\nTHE TASK THAT MATTERS: separate specific from promiscuous binders")
    print(f"  (both bind ParE3 well; only one is selective)  n={int(mask.sum())}")
    print(f"  AUC, ipTM margin      = {auc((i3 - i2)[mask], specific[mask]):.3f}")
    print(f"  AUC, ipTM_ParE3 alone = {auc(i3[mask], specific[mask]):.3f}")
    print(f"  AUC, mutation count   = {auc(-nmut[mask], specific[mask]):.3f}   <-- trivial baseline")
    print("\n  0.5 = chance. A proxy that does not beat mutation count is not")
    print("  measuring specificity, whatever it correlates with in bulk.")

    # ---- uncertainty: bare AUCs at small n look far more decisive than they are
    lab = specific[mask]
    print(f"\nUNCERTAINTY (n={int(mask.sum())}: "
          f"{int(lab.sum())} specific vs {int((~lab).sum())} promiscuous)")
    cands = [("ipTM margin", (i3 - i2)[mask]),
             ("ipTM_ParE3 alone", i3[mask]),
             ("mutation count", -nmut[mask])]
    floor = None
    for name, sc in cands:
        obs, pv, tot = perm_p(sc, lab)
        lo, hi = auc_ci(sc, lab)
        kind = "exact" if tot > 0 else "sampled"
        print(f"  {name:20s} AUC={obs:.3f}  95% CI [{lo:.3f}, {hi:.3f}]  "
              f"p={pv:.3f} ({kind})")
        if floor is None:
            ones = np.where(lab, 1.0, 0.0)
            _, floor, _ = perm_p(ones, lab)
    print(f"  best p reachable at this n (a perfect AUC=1.0) = {floor:.3f}")
    if floor > 0.05:
        print("  ** UNDERPOWERED: no result here can reach p<0.05. Do not report a verdict. **")

    # ---- is the proxy adding anything over distance from wild type?
    d, lo, hi = paired_auc_diff((i3 - i2)[mask], -nmut[mask], lab)
    print("\nDOES THE PROXY BEAT THE TRIVIAL BASELINE? (paired, same variants)")
    print(f"  AUC(ipTM margin) - AUC(mutation count) = {d:+.3f}  95% CI [{lo:+.3f}, {hi:+.3f}]")
    print("  CI spanning 0 = the two are not distinguishable at this n.")

    # ---- the baseline's win may be a property of the sampling, not the proxy
    print("\nCONFOUND: is 'specific' just 'close to wild type'?")
    print(f"  wild type = {wt}")
    print(f"  mutations, specific    = {sorted(int(x) for x in nmut[specific])}")
    print(f"  mutations, promiscuous = {sorted(int(x) for x in nmut[promisc])}")
    print(f"  rho(ipTM margin, measured margin)             = "
          f"{spearman(i3 - i2, w3 - w2):+.3f}")
    print(f"  ... controlling for mutation count            = "
          f"{partial_spearman(i3 - i2, w3 - w2, nmut):+.3f}")
    print("  If the raw rho survives the control, the proxy carries signal beyond")
    print("  distance from wild type; if it collapses, it does not.")

    # ---- the one result that does not depend on n
    print("\nDYNAMIC RANGE (independent of sample size)")
    for nm, v, w in (("ipTM ParE3", i3, w3), ("ipTM ParE2", i2, w2)):
        v = v[np.isfinite(v)]
        print(f"  {nm}: {v.min():.3f}-{v.max():.3f}  span {np.ptp(v):.3f}"
              f"   vs measured fitness span {np.nanmax(w) - np.nanmin(w):.3f}")
    print("  A proxy whose span is a fraction of the measured span cannot rank the")
    print("  landscape however well it correlates.")


if __name__ == "__main__":
    main()
