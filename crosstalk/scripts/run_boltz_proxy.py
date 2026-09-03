"""Audit a structure-based proxy against the measured specificity landscape.

Question: does Boltz co-folding confidence (ipTM) recover binding SPECIFICITY,
or only bulk foldability? Every sampled variant has a measured fitness against
both ParE3 and ParE2, so the proxy can be scored on the thing it is used for.

Sampling is stratified over the specificity quadrants (specific binder,
promiscuous binder, non-binder, off-target-only) so the audit is not dominated
by the ~70% of the landscape that binds nothing.

--dry-run builds every input and exercises the whole pipeline except the model
call, so the run is verified before a GPU is rented.
"""
import argparse, csv, json, sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from crosstalk.boltz import PARTNERS, BoltzOracle, complex_yaml, variant_sequence
from crosstalk.landscape import load_pard3


def stratify(L, n_per_cell: int, rng) -> list[tuple[str, str]]:
    on, off = L.F[:, 0], L.F[:, 1]
    cells = {
        "specific_E3": (on >= 0.8) & (off <= 0.2),
        "promiscuous": (on >= 0.8) & (off >= 0.6),
        "specific_E2": (on <= 0.2) & (off >= 0.8),
        "dead":        (on <= 0.2) & (off <= 0.2),
    }
    picked = []
    for name, mask in cells.items():
        idx = np.where(mask)[0]
        if len(idx) == 0:
            print(f"  warning: quadrant {name} is empty")
            continue
        take = rng.choice(idx, size=min(n_per_cell, len(idx)), replace=False)
        picked += [(L.seqs[i], name) for i in take]
        print(f"  {name:12s}: {len(idx):5d} available, sampled {len(take)}")
    return picked


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-per-cell", type=int, default=15)
    ap.add_argument("--workdir", default="results/boltz")
    ap.add_argument("--out", default="results/boltz_proxy.csv")
    ap.add_argument("--dry-run", action="store_true",
                    help="build and validate all inputs without calling the model")
    ap.add_argument("--accelerator", default="gpu")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    L = load_pard3()
    rng = np.random.default_rng(args.seed)
    print("Stratified sample over specificity quadrants:")
    picked = stratify(L, args.n_per_cell, rng)
    n_folds = len(picked) * len(PARTNERS)
    print(f"\n{len(picked)} variants x {len(PARTNERS)} partners = {n_folds} co-folds")

    oracle = BoltzOracle(args.workdir, accelerator=args.accelerator)

    if args.dry_run:
        wd = Path(args.workdir); wd.mkdir(parents=True, exist_ok=True)
        for variant, cell in picked:
            for partner in PARTNERS:
                y = complex_yaml(variant, partner)
                assert variant_sequence(variant) in y
                assert len(y.splitlines()) == 8
                (wd / f"{variant}_{partner}.yaml").write_text(y)
        print(f"\nDRY RUN OK: wrote {n_folds} Boltz inputs to {wd}")
        print(f"boltz on PATH: {oracle.available}")
        print("Everything except the model call is verified. To run for real:")
        print("  pip install boltz   # needs a GPU; CPU inference is impractical here")
        print(f"  python3 {Path(__file__).name} --n-per-cell {args.n_per_cell}")
        return

    if not oracle.available:
        sys.exit("boltz not found. `pip install boltz` and run on a GPU host.")

    rows = []
    for k, (variant, cell) in enumerate(picked, 1):
        rec = {"variant": variant, "cell": cell,
               "W_ParE3": float(L.F[L.index[variant], 0]),
               "W_ParE2": float(L.F[L.index[variant], 1])}
        for partner in PARTNERS:
            r = oracle.predict(variant, partner)
            rec[f"iptm_{partner}"] = r.iptm
            rec[f"ptm_{partner}"] = r.ptm
            rec[f"plddt_{partner}"] = r.plddt
        rows.append(rec)
        print(f"[{k}/{len(picked)}] {variant} {cell} "
              f"iptm_E3={rec.get('iptm_ParE3')} iptm_E2={rec.get('iptm_ParE2')}", flush=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print(f"\nwrote {args.out}")
    print("Analyse with: python3 scripts/analyze_boltz_proxy.py")


if __name__ == "__main__":
    main()
