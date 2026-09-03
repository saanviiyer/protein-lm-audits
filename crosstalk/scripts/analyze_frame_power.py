#!/usr/bin/env python3
"""How much frame signal could be hiding in the NT and HyenaDNA nulls?

The 3-periodic Markov control (scripts/run_biointerp_periodic.py) shows the
matched in-frame/out-of-frame stop contrast CAN detect frame representation in a
DNA-alphabet model. That validates the instrument but does not by itself make
every null on it a bounded null: an instrument can work and still be too coarse
for a particular model.

Nats per token are not comparable across models -- NT packs six nucleotides into
a token, HyenaDNA one, ESM-2 scores amino acids -- so every effect here is
expressed as a fraction of that model's OWN reference effect, the mononucleotide
shuffle, which is the battery's declared yardstick for what it can resolve. Two
numbers are then directly comparable across models: the contrast as a fraction of
reference, and the minimum effect the same experiment would have detected.

All batteries are re-adjudicated on the 24 unique coding sequences with t
intervals first, so nothing here inherits the pseudoreplication.

  ./.venv-glm/bin/python scripts/analyze_frame_power.py
"""
import argparse, csv, sys
from pathlib import Path



ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))

from crosstalk.biointerp import rebuild_from_csv, run_battery, scorers, DEFAULT_BATTERY
from crosstalk.biointerp.battery import _tcrit
from run_biointerp_dedup import cds_map, unique_keys
from run_biointerp_periodic import load_unique

# t_{0.80,df}: the power quantile. Same Hill inversion as _tcrit but at 0.80.
def _t80(df):
    x = 0.8416212336
    g1 = (x ** 3 + x) / 4
    g2 = (5 * x ** 5 + 16 * x ** 3 + 3 * x) / 96
    g3 = (3 * x ** 7 + 19 * x ** 5 + 17 * x ** 3 - 15 * x) / 384
    return x + g1 / df + g2 / df ** 2 + g3 / df ** 3


BATTERIES = [
    ("NT-v2 50M",   "nt50m_default",           "dna"),
    ("HyenaDNA 32k", "hyena_default",          "dna"),
    ("ESM-2 35M",   "esm35m_positive_control", "protein"),
]


def _row(label, alphabet, ref, n, m, lo, hi, sign, p, verdict, probe):
    se = (hi - m) / _tcrit(n - 1)
    mde = (_tcrit(n - 1) + _t80(n - 1)) * se
    return dict(model=label, probe=probe, alphabet=alphabet, n=n,
                reference_delta=ref, contrast=m, ci_lo=lo, ci_hi=hi, se=se, mde=mde,
                frac=m / ref, frac_lo=lo / ref, frac_hi=hi / ref,
                mde_frac=mde / ref, sign=sign, binom_p=p, verdict=verdict)


def contrast_row(label, rep, alphabet):
    c = rep.contrasts[0]
    return _row(label, alphabet, rep.reference_delta, c.n, c.mean_diff, c.ci_lo,
                c.ci_hi, f"{c.a_lower}/{c.n - c.ties}", c.binom_p, c.verdict,
                "matched stop contrast")


def frameshift_row(label, rep, alphabet):
    f = rep.by_name("frameshift +1")
    return _row(label, alphabet, rep.reference_delta, f.n, f.mean_delta, f.ci_lo,
                f.ci_hi, f"{f.real_higher}/{f.n - f.ties}", f.binom_p, f.verdict,
                "frameshift +1 (rotation)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--orders", type=int, nargs="+", default=[0, 1, 2, 3, 4, 5])
    ap.add_argument("--n-boot", type=int, default=20000)
    args = ap.parse_args()

    cds = cds_map()
    rows, frows = [], []

    # ------------------------------------------------- the Markov reference arms
    seqs = load_unique()
    keys = sorted(seqs)
    print("3-periodic Markov, matched contrast as a function of order")
    print(f"{'order':>5s} {'contrast':>10s} {'95% CI':>21s} {'ref':>9s} {'% of ref':>9s} "
          f"{'signs':>7s} {'verdict':>12s}")
    order_rows = []
    for k in args.orders:
        for period, tag in ((3, "3-periodic"), (1, "aperiodic")):
            sc = scorers.PeriodicMarkovScorer(order=k, period=period,
                                              corpus=seqs).bind(keys)
            rep = run_battery(sc, seqs, interventions=DEFAULT_BATTERY, seed=0,
                              n_boot=2000, ci_dist="t", progress=False)
            r = contrast_row(f"Markov order-{k} {tag}", rep, "dna")
            order_rows.append(dict(order=k, periodicity=period, **r))
            if period == 3:
                print(f"{k:5d} {r['contrast']:+10.4f} [{r['ci_lo']:+.4f},{r['ci_hi']:+.4f}] "
                      f"{r['reference_delta']:9.4f} {100*r['frac']:8.2f}% {r['sign']:>7s} "
                      f"{r['verdict']:>12s}")
    with (ROOT / "results" / "frame_power_markov_orders.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(order_rows[0])); w.writeheader(); w.writerows(order_rows)

    # the arm the main run selected, plus its twin, go into the cross-model table
    sel = 2
    for k, period, lab in [(sel, 3, f"Markov order-{sel} 3-PERIODIC (positive control)"),
                           (sel, 1, f"Markov order-{sel} aperiodic (negative control)")]:
        sc = scorers.PeriodicMarkovScorer(order=k, period=period, corpus=seqs).bind(keys)
        rep = run_battery(sc, seqs, interventions=DEFAULT_BATTERY, seed=0,
                          n_boot=args.n_boot, ci_dist="t", progress=False)
        rows.append(contrast_row(lab, rep, "dna"))
        frows.append(frameshift_row(lab, rep, "dna"))

    # ------------------------------------------------------------- the LM arms
    for label, tag, alph in BATTERIES:
        f = ROOT / "results" / f"biointerp_{tag}_per_sequence.csv"
        ks = [r["sequence"] for r in csv.DictReader(f.open())]
        keep, _ = unique_keys(ks, cds)
        rep = rebuild_from_csv(f, model=label, alphabet=alph, seed=0,
                               n_boot=args.n_boot, ci_dist="t", keep=keep)
        rows.append(contrast_row(label, rep, alph))
        frows.append(frameshift_row(label, rep, alph))

    # ------------------------------------------------------------------ report
    L = []
    A = L.append

    def block(title, note, table, positive_label):
        pos = [r for r in table if positive_label in r["model"]][0]
        A("=" * 118)
        A(title)
        A("  " + note)
        A("  every effect divided by that model's own reference effect (mononucleotide")
        A("  shuffle), because nats/token are not comparable across a 6-mer, a single")
        A("  nucleotide and an amino-acid tokenizer")
        A("=" * 118)
        A(f"{'model':44s} {'n':>3s} {'effect':>9s} {'as % of that model reference':>32s} "
          f"{'MDE %':>7s} {'signs':>7s} {'verdict':>11s}")
        A("-" * 118)
        for r in table:
            A(f"{r['model'][:44]:44s} {r['n']:3d} {r['contrast']:+9.4f} "
              f"{100*r['frac']:+8.2f}% [{100*r['frac_lo']:+7.2f}%,{100*r['frac_hi']:+7.2f}%] "
              f"{100*r['mde_frac']:6.2f}% {r['sign']:>7s} {r['verdict']:>11s}")
        A("-" * 118)
        A("")
        A(f"The positive control shows {100*pos['frac']:.2f}% of its own reference on this "
          f"probe.")
        A("Read every null against that number, not against zero:")
        A("")
        for r in table:
            if "control" in r["model"]:
                continue
            hi, p = 100 * r["frac_hi"], 100 * pos["frac"]
            if r["ci_lo"] > 0:
                A(f"  {r['model']:14s} DETECTS it, at {100*r['frac']:.2f}% of its reference.")
            elif hi < p:
                A(f"  {r['model']:14s} BOUNDED NULL. Its interval tops out at {hi:.2f}% of its")
                A(f"  {'':14s} own reference, below the {p:.2f}% a frame-representing model")
                A(f"  {'':14s} shows, so an effect that size is excluded ({p/max(hi,1e-9):.1f}x margin).")
            else:
                need = r["n"] * ((100 * r["mde_frac"]) / p) ** 2
                A(f"  {r['model']:14s} UNDERPOWERED. Its interval reaches {hi:.2f}% of its own")
                A(f"  {'':14s} reference, ABOVE the {p:.2f}% the positive control shows, so a")
                A(f"  {'':14s} frame effect that size would not have been detected. This null")
                A(f"  {'':14s} does not exclude frame representation. It would take about")
                A(f"  {'':14s} {need:.0f} unique coding sequences to bound it there.")
        A("")

    block("MATCHED IN-FRAME / OUT-OF-FRAME STOP CONTRAST, ALL MODELS, 24 UNIQUE CDS",
          "the confound-free probe: two 3-base edits, same trinucleotide, differing only in phase",
          rows, "3-PERIODIC")
    block("FRAMESHIFT +1 (CYCLIC ROTATION), ALL MODELS, 24 UNIQUE CDS",
          "the high-gain probe, but the one the section-26 retraction showed is confounded:",
          frows, "3-PERIODIC")
    A("=" * 118)
    text = "\n".join(L)
    print("\n" + text)
    (ROOT / "results" / "frame_power.txt").write_text(text + "\n")
    allrows = rows + frows
    with (ROOT / "results" / "frame_power.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(allrows[0])); w.writeheader(); w.writerows(allrows)
    print("\nwrote results/frame_power.{txt,csv} and results/frame_power_markov_orders.csv")


if __name__ == "__main__":
    main()
