#!/usr/bin/env python3
"""Re-adjudicate every genomic battery on UNIQUE coding sequences, with t intervals.

Two defects, both found by the 2026-08-31 audit and both fixable without a GPU:

  PSEUDOREPLICATION. The 29 "sequences" in every biointerp run and in
  results/granularity_per_gene.csv are 24 distinct coding sequences. BLAT_ECOLX
  appears three times (Deng 2012, Firnberg 2014, Stiffler 2015) and CP2C9, PTEN
  and RL401 twice each, because ProteinGym indexes assays, not proteins, and
  several proteins were assayed more than once. The battery pairs within
  sequence, so a repeated sequence contributes a near-duplicate delta and
  shrinks the standard error without adding information. The ESM-2 positive
  control at n=10 has the same defect: RL401_YEAST appears twice, so it is 9.

  NORMAL INTERVALS AT SMALL n. Every interval was m +/- 1.96*SE. At n=24 the
  t interval is 5.3% wider and at n=9 it is 15.4% wider.

Neither needs rescoring: the battery already writes per-sequence deltas, and
`rebuild_from_csv` re-runs the adjudication from them. This script therefore
recomputes every affected number exactly, and prints the before/after table so
the correction is auditable rather than asserted.

The representative kept for each duplicated coding sequence is the
alphabetically first assay id, which is the same convention the FINDINGS
section-26 retraction used, so its four recomputed numbers reproduce here.

  ./.venv-glm/bin/python scripts/run_biointerp_dedup.py
"""
import argparse, collections, csv, json, sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from crosstalk import glm
from crosstalk.biointerp import rebuild_from_csv, render, write_csv, write_contrast_csv
from crosstalk.biointerp.battery import _paired_stats, _binom_p

# every biointerp per-sequence file, with the metadata its report header needs
RUNS = [
    dict(tag="nt50m_default", model="InstaDeepAI/nucleotide-transformer-v2-50m-multi-species",
         alphabet="dna", tokenization="non-overlapping 6-mer", objective="masked LM"),
    dict(tag="hyena_default", model="LongSafari/hyenadna-small-32k-seqlen-hf",
         alphabet="dna", tokenization="single nucleotide", objective="autoregressive"),
    dict(tag="markov1_default", model="markov-order-1",
         alphabet="dna", tokenization="single nucleotide", objective="order-1 Markov MLE"),
    dict(tag="esm35m_positive_control", model="facebook/esm2_t12_35M_UR50D",
         alphabet="protein", tokenization="amino acid", objective="masked LM"),
]


def cds_map() -> dict:
    """assay id -> coding sequence, for every sequence any battery has used."""
    m = {k: v["cds"] for k, v in
         json.loads((ROOT / "data/cds/dms_cds.json").read_text()).items()}
    m["ParD3_Lite_2020"] = glm.load_cds()["ParD3"]["cds"]
    return m


def unique_keys(keys, cds) -> tuple[list, dict]:
    """Alphabetically-first representative of each distinct coding sequence."""
    groups = collections.defaultdict(list)
    for k in sorted(keys):
        groups[cds[k]].append(k)
    return sorted(g[0] for g in groups.values()), {g[0]: g for g in groups.values()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-boot", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cds = cds_map()
    out_rows = []

    # ---------------------------------------------------------------- batteries
    for run in RUNS:
        f = ROOT / "results" / f"biointerp_{run['tag']}_per_sequence.csv"
        if not f.exists():
            print(f"skip {run['tag']}: no per-sequence file")
            continue
        keys = [r["sequence"] for r in csv.DictReader(f.open())]
        keep, groups = unique_keys(keys, cds)
        dupes = {k: v for k, v in groups.items() if len(v) > 1}
        print(f"\n{'='*96}\n{run['tag']}: {len(keys)} rows -> {len(keep)} unique "
              f"coding sequences")
        for k, v in sorted(dupes.items()):
            print(f"    collapsed {len(v)}: {', '.join(v)}  ->  {k}")

        meta = {k: v for k, v in run.items() if k != "tag"}
        before = rebuild_from_csv(f, seed=args.seed, n_boot=args.n_boot,
                                  ci_dist="normal", **meta)
        mid = rebuild_from_csv(f, seed=args.seed, n_boot=args.n_boot,
                               ci_dist="normal", keep=keep, **meta)
        after = rebuild_from_csv(f, seed=args.seed, n_boot=args.n_boot,
                                 ci_dist="t", keep=keep, **meta)

        for b, m, a in zip(before.results, mid.results, after.results):
            out_rows.append(dict(
                run=run["tag"], model=run["model"], kind="intervention",
                name=b.intervention,
                n_before=b.n, n_after=a.n,
                delta_before=b.mean_delta, lo_before=b.ci_lo, hi_before=b.ci_hi,
                verdict_before=b.verdict,
                delta_dedup_normal=m.mean_delta, lo_dedup_normal=m.ci_lo,
                hi_dedup_normal=m.ci_hi, verdict_dedup_normal=m.verdict,
                delta_after=a.mean_delta, lo_after=a.ci_lo, hi_after=a.ci_hi,
                verdict_after=a.verdict,
                sign_before=f"{b.real_higher}/{b.n - b.ties}",
                sign_after=f"{a.real_higher}/{a.n - a.ties}",
                p_before=b.binom_p, p_after=a.binom_p,
                ratio_to_ref_before=b.ratio_to_reference,
                ratio_to_ref_after=a.ratio_to_reference))
        for b, m, a in zip(before.contrasts, mid.contrasts, after.contrasts):
            out_rows.append(dict(
                run=run["tag"], model=run["model"], kind="contrast", name=b.contrast,
                n_before=b.n, n_after=a.n,
                delta_before=b.mean_diff, lo_before=b.ci_lo, hi_before=b.ci_hi,
                verdict_before=b.verdict,
                delta_dedup_normal=m.mean_diff, lo_dedup_normal=m.ci_lo,
                hi_dedup_normal=m.ci_hi, verdict_dedup_normal=m.verdict,
                delta_after=a.mean_diff, lo_after=a.ci_lo, hi_after=a.ci_hi,
                verdict_after=a.verdict,
                sign_before=f"{b.a_lower}/{b.n - b.ties}",
                sign_after=f"{a.a_lower}/{a.n - a.ties}",
                p_before=b.binom_p, p_after=a.binom_p,
                ratio_to_ref_before=b.ratio_to_reference,
                ratio_to_ref_after=a.ratio_to_reference))

        txt = render(after)
        (ROOT / "results" / f"biointerp_{run['tag']}_dedup.txt").write_text(txt + "\n")
        write_csv(after, ROOT / "results" / f"biointerp_{run['tag']}_dedup.csv")
        write_contrast_csv(after, ROOT / "results" /
                           f"biointerp_{run['tag']}_dedup_contrasts.csv")
        print(txt)

    # ------------------------------------------- section 26 granularity ladder
    g = ROOT / "results" / "granularity_per_gene.csv"
    if g.exists():
        rows = list(csv.DictReader(g.open()))
        cols = [c for c in rows[0] if c not in ("gene", "real")]
        keys = [r["gene"] for r in rows]
        keep, _ = unique_keys(keys, cds)
        by = {r["gene"]: r for r in rows}
        print(f"\n{'='*96}\ngranularity ladder (FINDINGS section 26): "
              f"{len(rows)} rows -> {len(keep)} unique")
        for c in cols:
            d_all = np.array([float(r["real"]) - float(r[c]) for r in rows])
            d_uni = np.array([float(by[k]["real"]) - float(by[k][c]) for k in keep])
            bm, _, blo, bhi, _, _ = _paired_stats(d_all, seed=args.seed,
                                                  n_boot=args.n_boot, ci_dist="normal")
            am, _, alo, ahi, _, _ = _paired_stats(d_uni, seed=args.seed,
                                                  n_boot=args.n_boot, ci_dist="t")
            mm, _, mlo, mhi, _, _ = _paired_stats(d_uni, seed=args.seed,
                                                  n_boot=args.n_boot, ci_dist="normal")
            out_rows.append(dict(
                run="granularity_ladder", model="nucleotide-transformer-v2-50m",
                kind="intervention", name=c, n_before=len(d_all), n_after=len(d_uni),
                delta_before=bm, lo_before=blo, hi_before=bhi, verdict_before="",
                delta_dedup_normal=mm, lo_dedup_normal=mlo, hi_dedup_normal=mhi,
                verdict_dedup_normal="",
                delta_after=am, lo_after=alo, hi_after=ahi, verdict_after="",
                sign_before=f"{int((d_all>0).sum())}/{len(d_all)}",
                sign_after=f"{int((d_uni>0).sum())}/{len(d_uni)}",
                p_before=_binom_p(int((d_all > 0).sum()), len(d_all)),
                p_after=_binom_p(int((d_uni > 0).sum()), len(d_uni)),
                ratio_to_ref_before=float("nan"), ratio_to_ref_after=float("nan")))

    # ------------------------------------------------------------------- output
    keys_out = list(out_rows[0])
    dest = ROOT / "results" / "biointerp_dedup_corrections.csv"
    with dest.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys_out)
        w.writeheader(); w.writerows(out_rows)

    lines = []
    A = lines.append
    A("=" * 132)
    A("BEFORE / AFTER: pseudoreplication and interval correction on every genomic result")
    A("  before = all rows, m +/- 1.96*SE      after = unique coding sequences, m +/- t(n-1)*SE")
    A("=" * 132)
    A(f"{'run':26s} {'intervention':22s} {'n':>7s} "
      f"{'delta before':>26s} {'delta after':>26s} {'signs':>13s} {'verdict':>22s}")
    A("-" * 132)
    last = None
    for r in out_rows:
        if r["run"] != last:
            A("-" * 132)
            last = r["run"]
        star = "  <-- verdict changed" if (r["verdict_before"] and
                                           r["verdict_before"] != r["verdict_after"]) else ""
        A(f"{r['run'][:26]:26s} {r['name'][:22]:22s} "
          f"{r['n_before']:3d}->{r['n_after']:<3d} "
          f"{r['delta_before']:+9.4f} [{r['lo_before']:+.4f},{r['hi_before']:+.4f}] "
          f"{r['delta_after']:+9.4f} [{r['lo_after']:+.4f},{r['hi_after']:+.4f}] "
          f"{r['sign_before']:>6s}->{r['sign_after']:<6s} "
          f"{r['verdict_before'][:10]:>10s}->{r['verdict_after'][:10]:<10s}{star}")
    A("=" * 132)
    text = "\n".join(lines)
    print("\n" + text)
    (ROOT / "results" / "biointerp_dedup_corrections.txt").write_text(text + "\n")
    print(f"\nwrote {dest} and its .txt")


if __name__ == "__main__":
    main()
