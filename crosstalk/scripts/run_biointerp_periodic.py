#!/usr/bin/env python3
"""The missing POSITIVE CONTROL for the reading-frame test, on a DNA-alphabet model.

The question this answers is not about any language model. It is about the
instrument. FINDINGS reports that Nucleotide Transformer v2 and HyenaDNA fail the
frameshift test and the matched in-frame/out-of-frame stop contrast. A failure is
only interpretable if the test can succeed, and the only model that has passed it
so far is ESM-2 -- which reads the translated protein, so its success shows the
contrast works on a PROTEIN model and says nothing about whether it works on a
model that eats nucleotides. The order-1 Markov chain already in the repository
cannot help either: with a single transition table it has no parameter that
depends on position modulo three, so it CANNOT represent the reading frame and is
a negative control by construction.

A 3-periodic Markov chain is the missing case. Separate transition tables for
codon positions 1, 2 and 3 -- the GeneMark device (Borodovsky & McIninch 1993) --
represent the frame explicitly, in the DNA alphabet, with no protein anywhere in
the model. If it passes, the contrast has power against a DNA model and the NT
and HyenaDNA nulls are informative. If it fails, the instrument is uninformative
and nothing in the genomic frame material can be concluded.

The comparison is made airtight by running a NON-periodic twin at the same order,
fitted on the same corpus by the same leave-one-out procedure and scored by the
same code. Periodicity is then the single difference between the two arms, so the
gap between them is attributable to frame representation and to nothing else.

Model order is chosen by held-out log-likelihood on the REAL coding sequences,
before any intervention is scored, so the choice cannot be tuned to the answer.

  ./.venv-glm/bin/python scripts/run_biointerp_periodic.py
"""
import argparse, collections, csv, json, sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from crosstalk import glm
from crosstalk.biointerp import (DEFAULT_BATTERY, run_battery, render, scorers,
                                 write_csv, write_per_sequence_csv, write_contrast_csv)
from crosstalk.biointerp.battery import _tcrit


def load_unique(max_nt=2400, min_nt=200) -> dict:
    """The DMS coding sequences plus ParD3, ONE ROW PER DISTINCT SEQUENCE.

    The 29-assay set is 24 proteins: BLAT_ECOLX was assayed three times and
    CP2C9, PTEN and RL401 twice each. Keeping all 29 is pseudoreplication, which
    is what the 2026-08-31 audit found in every earlier battery.
    """
    cds = {}
    for k, v in json.loads((ROOT / "data/cds/dms_cds.json").read_text()).items():
        s = v["cds"]
        if len(s) % 3 == 0 and min_nt <= len(s) <= max_nt and set(s) <= set("ACGT"):
            cds[k] = s
    cds["ParD3_Lite_2020"] = glm.load_cds()["ParD3"]["cds"]
    groups = collections.defaultdict(list)
    for k in sorted(cds):
        groups[cds[k]].append(k)
    return {g[0]: s for s, g in groups.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--orders", type=int, nargs="+", default=[0, 1, 2, 3, 4, 5])
    ap.add_argument("--order", type=int, default=None,
                    help="skip selection and force this order")
    ap.add_argument("--pseudocount", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-boot", type=int, default=20000)
    ap.add_argument("--all-29", action="store_true",
                    help="do NOT deduplicate, for comparison with the old runs")
    args = ap.parse_args()

    seqs = load_unique() if not args.all_29 else None
    if seqs is None:                                  # the pseudoreplicated set
        from run_biointerp_battery import load_sequences
        seqs = load_sequences(2400)
    keys = sorted(seqs)
    print(f"{len(keys)} coding sequences, "
          f"{min(map(len, seqs.values()))}-{max(map(len, seqs.values()))} nt, "
          f"{sum(map(len, seqs.values()))} nt total\n", flush=True)

    # ---------------------------------------------- order selection, before any battery
    print("held-out mean log-likelihood on the REAL sequences (leave-one-sequence-out).")
    print("order chosen here, before any intervention is scored.\n")
    print(f"{'order':>5s} {'params/phase':>13s} {'3-periodic':>12s} {'non-periodic':>13s} "
          f"{'gap':>9s}")
    sel = []
    for k in args.orders:
        p3 = scorers.PeriodicMarkovScorer(order=k, period=3, corpus=seqs,
                                          pseudocount=args.pseudocount).bind(keys)
        p1 = scorers.PeriodicMarkovScorer(order=k, period=1, corpus=seqs,
                                          pseudocount=args.pseudocount).bind(keys)
        raw = [seqs[x] for x in keys]
        a, b = p3.score(raw).mean(), p1.score(raw).mean()
        sel.append(dict(order=k, ll_periodic=float(a), ll_nonperiodic=float(b),
                        gap=float(a - b)))
        print(f"{k:5d} {4 ** k * 4:13d} {a:12.4f} {b:13.4f} {a - b:+9.4f}")
    best = args.order if args.order is not None else max(sel, key=lambda r: r["ll_periodic"])["order"]
    print(f"\nselected order {best}"
          f"{' (forced)' if args.order is not None else ' (best held-out likelihood)'}\n")

    with (ROOT / "results" / "biointerp_periodic_order_selection.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["order", "ll_periodic", "ll_nonperiodic", "gap"])
        w.writeheader(); w.writerows(sel)

    # ------------------------------------------------------------- the two batteries
    reports = {}
    for period, tag in ((3, "periodic3"), (1, "aperiodic")):
        sc = scorers.PeriodicMarkovScorer(order=best, period=period, corpus=seqs,
                                          pseudocount=args.pseudocount).bind(keys)
        rep = run_battery(sc, seqs, interventions=DEFAULT_BATTERY, seed=args.seed,
                          n_boot=args.n_boot, ci_dist="t", progress=False)
        reports[tag] = rep
        text = render(rep)
        print(text)
        name = f"markov{best}_{tag}"
        write_csv(rep, ROOT / "results" / f"biointerp_{name}.csv")
        write_per_sequence_csv(rep, ROOT / "results" / f"biointerp_{name}_per_sequence.csv")
        write_contrast_csv(rep, ROOT / "results" / f"biointerp_{name}_contrasts.csv")
        (ROOT / "results" / f"biointerp_{name}.txt").write_text(text + "\n")

    # ------------------------------------------------------------------ the verdict
    p3, p1 = reports["periodic3"], reports["aperiodic"]
    lines = []
    A = lines.append
    A("=" * 104)
    A(f"POSITIVE CONTROL FOR THE READING-FRAME TEST  |  order-{best} Markov, "
      f"n={p3.n_sequences} unique CDS")
    A("  3-periodic vs the identical model with one table. Same corpus, same order,")
    A("  same leave-one-out fit, same scoring code. Periodicity is the only difference.")
    A("=" * 104)
    A("")
    A(f"{'probe':24s} {'3-periodic (frame-aware)':>34s} {'aperiodic (frame-blind)':>34s}")
    A("-" * 104)
    for nm in ["frameshift +1", "frameshift +2", "reverse complement",
               "stop in frame", "stop out of frame", "mononucleotide shuffle"]:
        a, b = p3.by_name(nm), p1.by_name(nm)
        A(f"{nm:24s} {a.mean_delta:+10.4f} [{a.ci_lo:+.4f},{a.ci_hi:+.4f}] {a.real_higher:2d}/{a.n:<2d} "
          f"{b.mean_delta:+10.4f} [{b.ci_lo:+.4f},{b.ci_hi:+.4f}] {b.real_higher:2d}/{b.n:<2d}")
    ca = p3.contrasts[0] if p3.contrasts else None
    cb = p1.contrasts[0] if p1.contrasts else None
    if ca and cb:
        A(f"{'MATCHED CONTRAST':24s} {ca.mean_diff:+10.4f} [{ca.ci_lo:+.4f},{ca.ci_hi:+.4f}] "
          f"{ca.a_lower:2d}/{ca.n - ca.ties:<2d} "
          f"{cb.mean_diff:+10.4f} [{cb.ci_lo:+.4f},{cb.ci_hi:+.4f}] {cb.a_lower:2d}/{cb.n - cb.ties:<2d}")
        A("")
        A(f"  matched contrast, 3-periodic: {ca.mean_diff:+.4f} nats/token "
          f"[{ca.ci_lo:+.4f}, {ca.ci_hi:+.4f}], binomial p={ca.binom_p:.3g}, "
          f"{ca.ratio_to_reference * 100:.1f}% of its own reference effect  -> {ca.verdict}")
        A(f"  matched contrast, aperiodic:  {cb.mean_diff:+.4f} nats/token "
          f"[{cb.ci_lo:+.4f}, {cb.ci_hi:+.4f}], binomial p={cb.binom_p:.3g}, "
          f"{cb.ratio_to_reference * 100:.1f}% of its own reference effect  -> {cb.verdict}")
        A("")
        if ca.passed and not cb.passed:
            A("  VERDICT: THE CONTRAST HAS POWER AGAINST A DNA-ALPHABET MODEL.")
            A("  A model whose only frame machinery is three phase-indexed transition")
            A("  tables passes; the identical model without them fails. So a null on this")
            A("  contrast is evidence about the model, not about the instrument, and the")
            A("  NT-v2 and HyenaDNA nulls are interpretable.")
        elif ca.passed and cb.passed:
            A("  VERDICT: AMBIGUOUS. Both arms pass, so the contrast is picking up")
            A("  something the frame-blind model also has, and it is not specific to frame.")
        else:
            A("  VERDICT: THE INSTRUMENT IS UNINFORMATIVE. A model that represents the")
            A("  reading frame BY CONSTRUCTION does not pass its own test at this n, so")
            A("  no null it reports about a genomic LM can be read as absence of frame")
            A("  representation. The genomic frame material cannot support a conclusion.")
    A("")
    A("-" * 104)
    A("MINIMUM DETECTABLE EFFECT on the matched contrast (two-sided 0.05, 80% power),")
    A("in each model's own nats/token and as a fraction of its own reference effect")
    A("-" * 104)
    A(f"{'model':46s} {'n':>4s} {'paired SD':>10s} {'MDE':>10s} {'MDE/ref':>9s}")
    mde_rows = []
    for label, rep, ps in [("order-%d 3-periodic Markov" % best, p3, None),
                           ("order-%d aperiodic Markov" % best, p1, None)]:
        c = rep.contrasts[0]
        sd = (c.ci_hi - c.mean_diff) / _tcrit(c.n - 1) * np.sqrt(c.n)
        mde = 2.8 * sd / np.sqrt(c.n)
        A(f"{label:46s} {c.n:4d} {sd:10.4f} {mde:10.4f} {mde / rep.reference_delta:9.3f}")
        mde_rows.append(dict(model=label, n=c.n, paired_sd=float(sd), mde=float(mde),
                             mde_over_reference=float(mde / rep.reference_delta),
                             observed=c.mean_diff, ci_lo=c.ci_lo, ci_hi=c.ci_hi,
                             verdict=c.verdict))
    A("=" * 104)
    text = "\n".join(lines)
    print("\n" + text)
    (ROOT / "results" / "biointerp_periodic_verdict.txt").write_text(text + "\n")
    with (ROOT / "results" / "biointerp_periodic_mde.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(mde_rows[0])); w.writeheader(); w.writerows(mde_rows)
    print(f"\nwrote results/biointerp_markov{best}_{{periodic3,aperiodic}}.* "
          f"and results/biointerp_periodic_verdict.txt")


if __name__ == "__main__":
    main()
