"""Rebuild results/boltz_proxy.csv from the folds on disk.

The parallel runner rewrites the CSV from whatever it holds in memory, so an
interrupted run truncates it to the folds that run had collected. The folds
themselves are durable, so the table is always recoverable by rescanning.
"""
import argparse, csv, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from crosstalk.boltz import PARTNERS, parse_confidence
from crosstalk.landscape import load_pard3


def cell_of(on, off):
    if on >= 0.8 and off <= 0.2: return "specific_E3"
    if on >= 0.8 and off >= 0.6: return "promiscuous"
    if on <= 0.2 and off >= 0.8: return "specific_E2"
    if on <= 0.2 and off <= 0.2: return "dead"
    return "other"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workdir", default="results/boltz")
    ap.add_argument("--out", default="results/boltz_proxy.csv")
    args = ap.parse_args()

    L = load_pard3()
    found = {}
    for case in sorted(Path(args.workdir).glob("*_*")):
        variant, _, partner = case.name.rpartition("_")
        if partner not in PARTNERS or variant not in L.index:
            continue
        try:
            found.setdefault(variant, {})[partner] = parse_confidence(case)
        except FileNotFoundError:
            pass

    rows = []
    for variant in sorted(found):
        on, off = (float(x) for x in L.F[L.index[variant]][:2])
        rec = {"variant": variant, "cell": cell_of(on, off),
               "W_ParE3": on, "W_ParE2": off}
        for partner in PARTNERS:
            c = found[variant].get(partner)
            rec[f"iptm_{partner}"] = c.get("iptm") if c else ""
            rec[f"ptm_{partner}"] = c.get("ptm") if c else ""
            rec[f"plddt_{partner}"] = c.get("complex_plddt") if c else ""
        rows.append(rec)

    if not rows:
        sys.exit(f"no completed folds under {args.workdir}")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    both = sum(1 for r in rows if all(r[f"iptm_{p}"] != "" for p in PARTNERS))
    print(f"wrote {args.out}: {len(rows)} variants ({both} with both partners)")


if __name__ == "__main__":
    main()
