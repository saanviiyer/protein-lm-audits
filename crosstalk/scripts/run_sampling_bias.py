"""A1: does mutation-sampling composition explain the per-system spread?

Section 16 found per-system correlations ranging -0.435 to +0.936 across 15
two-sided SKEMPI systems and attributed the spread to noise at n=10-31. Section
26 showed, on the one system where a dense DMS also exists, that the null there
is a property of *which mutations SKEMPI contains* rather than of the biology:
its BPTI entry is 48% alanine substitutions with 17 of 31 at a single position.

If sampling composition drives the result, it should be visible across systems:
the more a system's mutation set looks like an alanine scan concentrated on a
few hot-spot positions, the closer its measured correlation should sit to zero,
because that is the region where the effect is absent.

This needs no new data -- the composition is computable from SKEMPI itself.
"""
import argparse, collections, csv, math, sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from crosstalk import skempi as SK
from run_proxy_ladder import spearman


def composition(muts):
    """Sampling descriptors for one system's mutation set."""
    n = len(muts)
    if n == 0:
        return None
    to_ala = sum(1 for m in muts if m[-1] == "A") / n
    from_ala = sum(1 for m in muts if m[0] == "A") / n
    pos = [int("".join(c for c in m[1:-1] if c.isdigit())) for m in muts]
    counts = collections.Counter(pos)
    n_pos = len(counts)
    top_share = counts.most_common(1)[0][1] / n
    # Gini over positional counts: 0 = spread evenly, 1 = all at one position
    vals = sorted(counts.values())
    cum = np.cumsum(vals)
    gini = float((n_pos + 1 - 2 * np.sum(cum) / cum[-1]) / n_pos) if n_pos > 1 else 1.0
    subs_per_pos = n / n_pos
    distinct_mut_aa = len({m[-1] for m in muts})
    return dict(n=n, n_positions=n_pos, to_alanine=to_ala, from_alanine=from_ala,
                top_position_share=top_share, positional_gini=gini,
                subs_per_position=subs_per_pos, distinct_target_residues=distinct_mut_aa)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/sampling_bias.csv")
    args = ap.parse_args()

    prev = {}
    p = Path("results/skempi_transfer.csv")
    if p.exists():
        for r in csv.DictReader(open(p)):
            if r["protein"] and r["protein"] != "POOLED (within-system z)":
                prev[(r["protein"], r["partner_a"], r["partner_b"])] = r

    systems = SK.build(min_shared=10, verbose=False)
    print(f"{len(systems)} two-sided systems\n")

    rows = []
    for s in systems:
        comp = composition(s.mutations)
        if comp is None:
            continue
        key = (s.protein, s.partner_a, s.partner_b)
        rec = prev.get(key)
        rho = float(rec["rho_likelihood"]) if rec else float("nan")
        rows.append(dict(protein=s.protein, partner_a=s.partner_a, partner_b=s.partner_b,
                         rho_likelihood=rho, **comp))

    print(f"{'protein':26s} {'n':>3s} {'pos':>4s} {'%Ala':>6s} {'top':>6s} {'gini':>6s} {'rho':>7s}")
    for r in sorted(rows, key=lambda r: -r["to_alanine"]):
        print(f"{r['protein'][:26]:26s} {r['n']:3d} {r['n_positions']:4d} "
              f"{100*r['to_alanine']:5.0f}% {100*r['top_position_share']:5.0f}% "
              f"{r['positional_gini']:6.2f} {r['rho_likelihood']:+7.3f}")

    ok = [r for r in rows if r["rho_likelihood"] == r["rho_likelihood"]]
    print(f"\nDoes composition predict |rho|?  (n={len(ok)} systems)")
    absrho = [abs(r["rho_likelihood"]) for r in ok]
    for feat in ("to_alanine", "top_position_share", "positional_gini",
                 "subs_per_position", "n", "n_positions", "distinct_target_residues"):
        v = [r[feat] for r in ok]
        print(f"   rho(|rho_likelihood|, {feat:24s}) = {spearman(v, absrho):+.3f}")

    print("\nAnd against signed rho (is the effect suppressed toward zero, or flipped?):")
    signed = [r["rho_likelihood"] for r in ok]
    for feat in ("to_alanine", "subs_per_position"):
        v = [r[feat] for r in ok]
        print(f"   rho(rho_likelihood, {feat:24s}) = {spearman(v, signed):+.3f}")

    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
