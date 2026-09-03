#!/usr/bin/env python3
"""
audit_prism_results.py

Self-audit of PRISM's existing Phase-1 (pairwise) and Phase-2 (centroid
trajectory) outputs for the data-integrity failure modes that a homology-free
pipeline has "with no BLAST-style analogue" (the paper's Section 5 thesis).

It found, and this script quantifies + corrects, two contaminants in the raw
Phase-1 hit tables that inflate discovery counts:

  1. DUPLICATE (pair, profile) rows -- Phase-2 appends to results.csv across
     resumed SLURM array jobs; the append is not de-duplicated at the CSV level,
     so re-run pairs are counted multiple times.
  2. SELF-COMPARISONS (accA == accB) -- a protein's mutant cloud scored against
     *its own* wild-type embedding passes the significance battery essentially by
     construction, and should never enter a "most-sensitive pair" table.

The centroid-trajectory headline (SBS17a -> PF17041) is recomputed here too and
is UNAFFECTED by the above (it is per-protein, not per-pair), so the main
scientific claim survives the audit while the bookkeeping is corrected.

For each clan directory containing results.csv it writes results_clean.csv
(de-duplicated, self-pairs removed) and prints a corrected Table 1. For each
*_centroid_trajectory_results.csv it writes signature_attraction_summary.csv
and reproduces the Section 4.3 numbers.

USAGE
-----
python audit_prism_results.py --runs_dir all_runs \
    --target_domain PF17041 --focus_signature SBS17a
"""

import argparse
import glob
import os

import pandas as pd


TRUE_SET = {"true", "1", "1.0", "yes"}


def is_sig(series):
    return series.astype(str).str.strip().str.lower().isin(TRUE_SET)


def audit_phase1(results_csv: str):
    """Return (report dict, cleaned DataFrame)."""
    r = pd.read_csv(results_csv)
    n_raw = len(r)
    dup_mask = r.duplicated(subset=["pair", "profile"], keep="first")
    n_dup = int(dup_mask.sum())
    dedup = r[~dup_mask]
    self_mask = dedup["accA"] == dedup["accB"]
    n_self_rows = int(self_mask.sum())
    n_self_pairs = int(dedup[self_mask]["pair"].nunique())
    clean = dedup[~self_mask]

    def hit_stats(df):
        sig = df[is_sig(df["significant"])]
        n_exp = len(df)
        return len(sig), n_exp, (100.0 * len(sig) / n_exp if n_exp else 0.0)

    raw_hits, raw_exp, raw_rate = hit_stats(r)
    clean_hits, clean_exp, clean_rate = hit_stats(clean)

    # most-sensitive pair (by # significant profiles), self-pairs excluded
    sig_clean = clean[is_sig(clean["significant"])]
    top = (sig_clean.groupby("pair").size().sort_values(ascending=False)
           if len(sig_clean) else pd.Series(dtype=int))
    top_pair = top.index[0] if len(top) else None
    top_n = int(top.iloc[0]) if len(top) else 0

    # top signature by # significant hits (clean)
    top_sig = (sig_clean.groupby("profile").size().sort_values(ascending=False)
               if len(sig_clean) else pd.Series(dtype=int))
    top_sig_name = top_sig.index[0] if len(top_sig) else None

    report = {
        "n_rows_raw": n_raw, "n_duplicate_rows": n_dup,
        "n_self_pair_rows": n_self_rows, "n_self_pairs": n_self_pairs,
        "n_pairs_raw": int(r["pair"].nunique()),
        "n_pairs_clean": int(clean["pair"].nunique()),
        "n_proteins": int(pd.concat([clean["accA"], clean["accB"]]).nunique()),
        "hits_raw": raw_hits, "hit_rate_raw": round(raw_rate, 1),
        "hits_clean": clean_hits, "experiments_clean": clean_exp,
        "hit_rate_clean": round(clean_rate, 1),
        "top_signature_clean": top_sig_name,
        "top_pair_clean": top_pair, "top_pair_n_sigs": top_n,
    }
    return report, clean


def audit_phase2(traj_csv: str, target_domain: str, focus_signature: str):
    t = pd.read_csv(traj_csv)
    rows = []
    for sig, grp in t.groupby("signature"):
        top1 = grp["top1_pfam"].astype(str)
        frac_target = float(top1.str.contains(target_domain, na=False).mean())
        mode_domain = top1.value_counts(normalize=True)
        rows.append({
            "signature": sig, "n_experiments": len(grp),
            f"frac_top1_{target_domain}": round(frac_target, 4),
            "modal_top1_domain": mode_domain.index[0] if len(mode_domain) else None,
            "modal_top1_frac": round(float(mode_domain.iloc[0]), 4) if len(mode_domain) else None,
            "mean_delta_dist": round(float(grp["delta_dist_orig"].mean()), 4)
            if "delta_dist_orig" in grp else None,
        })
    summary = pd.DataFrame(rows).sort_values(f"frac_top1_{target_domain}", ascending=False)
    focus = summary[summary["signature"].astype(str).str.contains(focus_signature, na=False)]
    return summary, focus


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs_dir", default="all_runs",
                    help="Directory with per-clan subdirs (CL0016/, CL0023/, ...).")
    ap.add_argument("--target_domain", default="PF17041",
                    help="Domain to track as the attraction target (default: PF17041).")
    ap.add_argument("--focus_signature", default="SBS17a",
                    help="Signature to spotlight in the Phase-2 reproduction.")
    args = ap.parse_args(argv)

    print("=" * 74)
    print("PHASE-1 PAIRWISE AUDIT (corrected Table 1)")
    print("=" * 74)
    print(f"{'clan':8} {'pairs(raw/clean)':17} {'dupRows':8} {'selfPairs':10} "
          f"{'hits raw->clean':18} {'rate% raw->clean':17} {'top non-self pair'}")
    all_reports = []
    for results_csv in sorted(glob.glob(os.path.join(args.runs_dir, "*", "results.csv"))):
        clan = os.path.basename(os.path.dirname(results_csv))
        rep, clean = audit_phase1(results_csv)
        rep["clan"] = clan
        all_reports.append(rep)
        clean_path = os.path.join(os.path.dirname(results_csv), "results_clean.csv")
        clean.to_csv(clean_path, index=False)
        print(f"{clan:8} {str(rep['n_pairs_raw'])+'/'+str(rep['n_pairs_clean']):17} "
              f"{rep['n_duplicate_rows']:<8} {rep['n_self_pairs']:<10} "
              f"{str(rep['hits_raw'])+'->'+str(rep['hits_clean']):18} "
              f"{str(rep['hit_rate_raw'])+'->'+str(rep['hit_rate_clean']):17} "
              f"{rep['top_pair_clean']} ({rep['top_pair_n_sigs']})")

    if all_reports:
        pd.DataFrame(all_reports).to_csv(
            os.path.join(args.runs_dir, "phase1_audit_summary.csv"), index=False)
        print(f"\n[audit] wrote {os.path.join(args.runs_dir, 'phase1_audit_summary.csv')} "
              f"and per-clan results_clean.csv")

    print("\n" + "=" * 74)
    print(f"PHASE-2 TRAJECTORY REPRODUCTION ({args.focus_signature} -> {args.target_domain})")
    print("=" * 74)
    for traj_csv in sorted(glob.glob(os.path.join(args.runs_dir, "*", "*centroid_trajectory_results.csv"))):
        clan = os.path.basename(os.path.dirname(traj_csv))
        summary, focus = audit_phase2(traj_csv, args.target_domain, args.focus_signature)
        out = os.path.join(os.path.dirname(traj_csv), "signature_attraction_summary.csv")
        summary.to_csv(out, index=False)
        if len(focus):
            row = focus.iloc[0]
            print(f"{clan}: {args.focus_signature} n={int(row['n_experiments'])}  "
                  f"{args.target_domain} as top1 = {row[f'frac_top1_{args.target_domain}']:.1%}  "
                  f"(modal domain {row['modal_top1_domain']} @ {row['modal_top1_frac']:.1%})")
        # where does the focus signature rank among all signatures for this target?
        rank = summary.reset_index(drop=True)
        idx = rank.index[rank["signature"].astype(str).str.contains(args.focus_signature, na=False)]
        if len(idx):
            print(f"         {args.focus_signature} ranks #{int(idx[0])+1}/{len(rank)} "
                  f"signatures by {args.target_domain}-attraction")
    print("\n[audit] done.")


if __name__ == "__main__":
    main()
