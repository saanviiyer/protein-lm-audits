#!/usr/bin/env python3
"""Cache zero-shot scores for a criteria-selected set of ProteinGym assays.

Scoring is the only GPU-bound step and it is reusable, so it is split out from
the backtest. Selection is by criteria rather than a hardcoded list so the set is
reproducible and can be widened without editing code -- 13 assays cannot resolve
differences under about 0.03 in top-1% recall, which is the width of most of the
policy effects measured so far.

Supports more than one scorer family. Everything to date is ESM-2 650M, which is
the most obvious objection to any claim about "protein language models".

    python scripts/score_proteingym.py --max_len 500 --min_singles 1200
    python scripts/score_proteingym.py --scorer esmc
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from gauntlet import proteingym as pg  # noqa: E402


def select(reference, max_len, min_singles):
    d = pd.read_csv(reference)
    d = d[(d.seq_len <= max_len) & (d.DMS_number_single_mutants >= min_singles)]
    return d.sort_values("DMS_number_single_mutants", ascending=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pg_dir", default="data/proteingym")
    ap.add_argument("--max_len", type=int, default=500)
    ap.add_argument("--min_singles", type=int, default=1200)
    ap.add_argument("--scorer", default="esm2", choices=["esm2", "esmc"])
    ap.add_argument("--model", default=None)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0, help="0 = no limit")
    args = ap.parse_args()

    ref = select(os.path.join(args.pg_dir, "reference.csv"),
                 args.max_len, args.min_singles)
    cache_dir = os.path.join(args.pg_dir,
                             "esm_cache" if args.scorer == "esm2" else f"{args.scorer}_cache")
    os.makedirs(cache_dir, exist_ok=True)

    todo = []
    for r in ref.itertuples():
        path = os.path.join(cache_dir, f"{r.DMS_id}.npy")
        if not os.path.exists(path):
            todo.append(r)
    if args.limit:
        todo = todo[:args.limit]

    print(f"{len(ref)} assays match (len<={args.max_len}, singles>={args.min_singles})")
    print(f"{len(ref) - len(todo)} already cached, {len(todo)} to score with {args.scorer}")
    print(f"{sum(r.seq_len for r in todo)} residue positions to mask\n")
    if not todo:
        return

    if args.scorer == "esm2":
        from gauntlet.proxies import ESM2Marginals
        scorer = ESM2Marginals(args.model or "facebook/esm2_t33_650M_UR50D",
                               batch_size=args.batch_size)
        print(f"ESM-2 on {scorer.device}\n", flush=True)
    else:
        from gauntlet.proxies import ESMCMarginals
        scorer = ESMCMarginals(args.model or "esmc_300m", batch_size=args.batch_size)
        print(f"ESM-C on {scorer.device}\n", flush=True)

    for i, r in enumerate(todo, 1):
        records, _, _ = pg.load_assay(r.DMS_id, r.target_seq,
                                      os.path.join(args.pg_dir, "assays"))
        if not records:
            print(f"  [{i}/{len(todo)}] {r.DMS_id}: no usable variants, skipped", flush=True)
            continue
        positions = sorted({p - 1 for rec in records for _, p, _ in rec["muts"]})
        scorer.logprobs_at(r.target_seq, positions)
        arr = np.array([scorer.score(r.target_seq, rec["muts"], offset=0)
                        for rec in records])
        np.save(os.path.join(cache_dir, f"{r.DMS_id}.npy"), arr)
        print(f"  [{i}/{len(todo)}] {r.DMS_id:38s} {len(records):5d} variants, "
              f"{len(positions):4d} positions", flush=True)

    print(f"\ncached to {cache_dir}/")


if __name__ == "__main__":
    main()
