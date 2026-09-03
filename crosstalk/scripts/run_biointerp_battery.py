#!/usr/bin/env python3
"""Run the biointerp typed-intervention battery on one model.

Generalises FINDINGS section 26 from a one-off script into a reusable audit. The
battery scores each real coding sequence and each of its typed perturbations,
pairs within sequence, and adjudicates each contrast against the expectation the
intervention declared before any number existed.

  ./.venv-glm/bin/python scripts/run_biointerp_battery.py --model nt50m
  ./.venv-glm/bin/python scripts/run_biointerp_battery.py --model hyena
  ./.venv-glm/bin/python scripts/run_biointerp_battery.py --model markov1   # audit

`--battery section26` restricts to the four interventions of section 26, for
reproduction. The default battery adds frameshift +2, a dinucleotide-preserving
shuffle and a mononucleotide shuffle; the last is the reference effect against
which every null is bounded.
"""
import argparse, json, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from crosstalk import glm
from crosstalk.biointerp import (DEFAULT_BATTERY, SECTION_26_BATTERY, run_battery,
                                 render, scorers, write_csv, write_per_sequence_csv,
                                 write_contrast_csv)

MODELS = {
    "nt50m":  lambda: scorers.NTScorer("InstaDeepAI/nucleotide-transformer-v2-50m-multi-species"),
    "nt100m": lambda: scorers.NTScorer("InstaDeepAI/nucleotide-transformer-v2-100m-multi-species"),
    "nt250m": lambda: scorers.NTScorer("InstaDeepAI/nucleotide-transformer-v2-250m-multi-species"),
    "nt500m": lambda: scorers.NTScorer("InstaDeepAI/nucleotide-transformer-v2-500m-multi-species"),
    "hyena":  lambda: scorers.HyenaScorer("LongSafari/hyenadna-small-32k-seqlen-hf"),
    "hyena-tiny": lambda: scorers.HyenaScorer("LongSafari/hyenadna-tiny-1k-seqlen-hf"),
    "esm35m": lambda: scorers.ESMScorer("facebook/esm2_t12_35M_UR50D"),
    "esm8m":  lambda: scorers.ESMScorer("facebook/esm2_t6_8M_UR50D"),
    "markov1": lambda: scorers.CountScorer(order=1),
    "markov2": lambda: scorers.CountScorer(order=2),
}

BATTERIES = {"default": DEFAULT_BATTERY, "section26": SECTION_26_BATTERY}


def load_sequences(max_nt: int, min_nt: int = 200, limit: int | None = None) -> dict:
    """The 29 verified DMS coding sequences plus ParD3, exactly as section 26 used."""
    cds = {}
    for k, v in json.loads((ROOT / "data/cds/dms_cds.json").read_text()).items():
        s = v["cds"]
        if len(s) % 3 == 0 and min_nt <= len(s) <= max_nt and set(s) <= set("ACGT"):
            cds[k] = s
    cds["ParD3_Lite_2020"] = glm.load_cds()["ParD3"]["cds"]
    if limit:
        cds = {k: cds[k] for k in sorted(cds, key=lambda k: len(cds[k]))[:limit]}
    return cds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="nt50m", choices=sorted(MODELS))
    ap.add_argument("--battery", default="default", choices=sorted(BATTERIES))
    ap.add_argument("--max-nt", type=int, default=2400)
    ap.add_argument("--limit", type=int, default=None,
                    help="use only the N shortest sequences (for expensive scorers)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()

    seqs = load_sequences(args.max_nt, limit=args.limit)
    print(f"{len(seqs)} coding sequences, "
          f"{min(map(len, seqs.values()))}-{max(map(len, seqs.values()))} nt", flush=True)

    t0 = time.time()
    model = MODELS[args.model]()
    rep = run_battery(model, seqs, interventions=BATTERIES[args.battery], seed=args.seed)
    print(f"\n[{time.time() - t0:.0f}s]\n", flush=True)

    text = render(rep)
    print(text)

    tag = args.tag or f"{args.model}_{args.battery}"
    out = ROOT / "results"
    write_csv(rep, out / f"biointerp_{tag}.csv")
    write_per_sequence_csv(rep, out / f"biointerp_{tag}_per_sequence.csv")
    write_contrast_csv(rep, out / f"biointerp_{tag}_contrasts.csv")
    (out / f"biointerp_{tag}.txt").write_text(text + "\n")
    print(f"\nwrote results/biointerp_{tag}.{{csv,txt}} and "
          f"results/biointerp_{tag}_per_sequence.csv")


if __name__ == "__main__":
    main()
