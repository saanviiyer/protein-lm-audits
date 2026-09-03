#!/usr/bin/env python3
"""Is rotation-invariance a property of the scoring rule or of genomic models?

Section 26's retraction lists, as an open defect, that pseudo-likelihood is a sum
of LOCAL conditionals, so a one-token rotation preserves almost every conditioning
context and near-indifference may be a fact about the SCORER rather than about
NT-v2. The protein side settles it: rotating a protein by one residue has no
reading-frame semantics at all, it is simply meaning-destruction, so if ESM-2 under
the identical pseudo-likelihood rule is also near-indifferent then the invariance
travels with the scorer.

Two scale-free comparisons, because nats/6-mer-token and nats/residue are not
commensurable:
  dz      paired Cohen's dz = mean(delta)/sd(delta). Unitless, and the quantity
          that decides detectability at any n.
  frac    delta as a fraction of that model's OWN composition-preserving
          total-destruction anchor (codon-order shuffle for NT-v2, residue
          shuffle for ESM-2). Bootstrap CI over proteins/genes.

The DNA side is recomputed on UNIQUE coding sequences, matching the retraction's
dedup (29 rows -> 24 genes), with t-based intervals.
"""
import csv, json
from pathlib import Path
import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
RNG = np.random.default_rng(0)


def paired(real, alt, anchor=None, nboot=20000):
    d = real - alt
    n = len(d)
    tcrit = float(stats.t.ppf(0.975, n - 1))
    se = d.std(ddof=1) / np.sqrt(n)
    out = dict(n=n, mean_delta=float(d.mean()),
               lo=float(d.mean() - tcrit * se), hi=float(d.mean() + tcrit * se),
               p=float(stats.ttest_1samp(d, 0.0).pvalue),
               dz=float(d.mean() / d.std(ddof=1)),
               higher=int((d > 0).sum()))
    if anchor is not None:
        f = d.mean() / anchor.mean()
        bs = []
        for _ in range(nboot):
            i = RNG.integers(0, n, n)
            a = anchor[i].mean()
            if abs(a) > 1e-9:
                bs.append(d[i].mean() / a)
        out.update(frac=float(f), frac_lo=float(np.percentile(bs, 2.5)),
                   frac_hi=float(np.percentile(bs, 97.5)))
    return out


def load_dna():
    rows = list(csv.DictReader((ROOT / "results/granularity_per_gene.csv").open()))
    cds = json.loads((ROOT / "data/cds/dms_cds.json").read_text())
    seen, keep = set(), []
    for r in rows:                       # dedup on the coding sequence itself
        s = cds[r["gene"]]["cds"]
        if s in seen:
            continue
        seen.add(s); keep.append(r)
    cols = [c for c in rows[0] if c != "gene"]
    return {c: np.array([float(r[c]) for r in keep]) for c in cols}, len(keep)


def load_prot():
    rows = list(csv.DictReader(
        (ROOT / "results/protein_semantics_per_protein.csv").open()))
    cols = [c for c in rows[0] if c.startswith("esm:")]
    return ({c[4:]: np.array([float(r[c]) for r in rows]) for c in cols},
            len(rows), [r["protein"] for r in rows])


def main():
    dna, ndna = load_dna()
    prot, nprot, pnames = load_prot()
    print(f"NT-v2 50M  : {ndna} unique coding sequences (deduplicated on CDS)")
    print(f"ESM-2 650M : {nprot} unique proteins\n")

    OPS = [("rotate the string by 1", "frameshift +1", "rotate +1"),
           ("read backwards",         "reverse complement", "reverse"),
           ("total destruction",      "codon-order shuffle", "shuffle")]

    res, rows = {}, []
    for label, dc, pc in OPS:
        res[("dna", pc)] = paired(dna["real"], dna[dc], dna["real"] - dna["codon-order shuffle"])
        res[("prot", pc)] = paired(prot["real"], prot[pc], prot["real"] - prot["shuffle"])

    hdr = f"{'operation':22s} {'model':10s} {'delta':>9s} {'95% CI (t)':>20s} {'p':>9s} {'dz':>7s} {'frac of shuffle':>24s} {'real higher':>12s}"
    print(hdr); print("-" * len(hdr))
    for label, dc, pc in OPS:
        for tag, mname in (("dna", "NT-v2 50M"), ("prot", "ESM-2 650M")):
            r = res[(tag, pc)]
            print(f"{label if tag=='dna' else '':22s} {mname:10s} {r['mean_delta']:+9.4f} "
                  f"[{r['lo']:+.4f}, {r['hi']:+.4f}] {r['p']:9.2e} {r['dz']:+7.2f} "
                  f"{r['frac']:8.2f} [{r['frac_lo']:+.2f}, {r['frac_hi']:+.2f}] "
                  f"{r['higher']:8d}/{r['n']}")
            rows.append(dict(model=mname, operation=label, condition=dc if tag=="dna" else pc, **r))
        print()

    print("the decisive contrast")
    dr, pr = res[("dna", "rotate +1")], res[("prot", "rotate +1")]
    ps = res[("prot", "shuffle")]
    print(f"  NT-v2  rotation costs {dr['frac']*100:5.1f}% [{dr['frac_lo']*100:+.0f}%, {dr['frac_hi']*100:+.0f}%] of its own destruction anchor, dz {dr['dz']:+.2f}")
    print(f"  ESM-2  rotation costs {pr['frac']*100:5.1f}% [{pr['frac_lo']*100:+.0f}%, {pr['frac_hi']*100:+.0f}%] of its own destruction anchor, dz {pr['dz']:+.2f}")
    print(f"  ESM-2  rotation vs its own shuffle: dz ratio {pr['dz']/ps['dz']:.2f}")

    # is ESM's rotation effect detectable at the DNA study's sample size?
    n = dr["n"]
    d = prot["real"] - prot["rotate +1"]
    pw = float(stats.norm.sf(stats.norm.isf(0.025) - pr["dz"] * np.sqrt(n)))
    print(f"\n  power to detect ESM-2's rotation effect at the DNA study's n={n}: {pw:.2f}")
    print(f"  (the DNA non-detection had power {stats.norm.sf(stats.norm.isf(0.025) - dr['dz']*np.sqrt(n)):.2f} at its own effect size)")

    # equivalence: is ESM-2 rotation within +/- the DNA test's MDE, expressed in dz?
    mde_dz = float(stats.norm.isf(0.025) + stats.norm.isf(0.20)) / np.sqrt(n)
    print(f"  MDE at n={n}, 80% power: dz {mde_dz:.2f}. "
          f"ESM-2 rotation dz {pr['dz']:+.2f} is "
          f"{'ABOVE' if pr['dz'] > mde_dz else 'BELOW'} it; "
          f"NT-v2 frameshift dz {dr['dz']:+.2f} is "
          f"{'ABOVE' if dr['dz'] > mde_dz else 'BELOW'} it.")

    print("\nrotation vs meaning-destruction, ESM-2 only (all paired against real)")
    for c in ["rotate +1", "rotate mid", "reverse", "shuffle", "conservative 10%", "radical 10%"]:
        r = paired(prot["real"], prot[c], prot["real"] - prot["shuffle"])
        print(f"  {c:18s} delta {r['mean_delta']:+7.4f} [{r['lo']:+.4f}, {r['hi']:+.4f}] "
              f"dz {r['dz']:+6.2f}  p {r['p']:8.1e}  {r['frac']*100:5.1f}% of shuffle  "
              f"{r['higher']}/{r['n']}")
        rows.append(dict(model="ESM-2 650M", operation="esm ladder", condition=c, **r))

    # rotate +1 vs shuffle, directly paired: does rotation differ from destruction?
    print("\ndirect paired contrasts within ESM-2")
    for a, b in [("rotate +1", "shuffle"), ("rotate +1", "reverse"),
                 ("rotate +1", "rotate mid"), ("reverse", "shuffle")]:
        d2 = prot[b] - prot[a]     # >0 means a scores higher, i.e. a is milder
        t = stats.ttest_1samp(d2, 0.0)
        tc = float(stats.t.ppf(0.975, len(d2) - 1))
        se = d2.std(ddof=1) / np.sqrt(len(d2))
        print(f"  {a:12s} minus {b:12s} {d2.mean():+7.4f} "
              f"[{d2.mean()-tc*se:+.4f}, {d2.mean()+tc*se:+.4f}]  p {t.pvalue:8.1e}")
        rows.append(dict(model="ESM-2 650M", operation=f"{a} vs {b}", condition="contrast",
                         n=len(d2), mean_delta=float(d2.mean()),
                         lo=float(d2.mean()-tc*se), hi=float(d2.mean()+tc*se),
                         p=float(t.pvalue), dz=float(d2.mean()/d2.std(ddof=1)),
                         higher=int((d2 > 0).sum())))

    out = ROOT / "results/protein_semantics_rotation.csv"
    keys = sorted({k for r in rows for k in r})
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
