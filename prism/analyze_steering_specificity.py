#!/usr/bin/env python3
"""
analyze_steering_specificity.py

Controls the PRISM headline (a signature "directs" sequences toward a domain)
against the embedding-hub confound the paper names but does not characterize:
maybe the target domain is simply a popular nearest neighbor in ESM-2 space, so
*any* signature would land there.

Operates on the Phase-2 trajectory outputs (<clan>/<n>_centroid_trajectory_results.csv,
columns include acc, signature, top1_pfam ...). For each clan it computes:

  * HUBNESS  b(d) = fraction of ALL (protein, signature) experiments whose most-
    attracted domain is d. Hub domains have large b(d) regardless of signature.

  * For each signature s, its dominant target t(s) = argmax_d P(top1=d | s), the
    RAW concentration c(s) = P(top1=t(s) | s), and the HUB-CORRECTED enrichment
        e(s) = c(s) / b(t(s)),
    i.e. how much more often s hits its target than that target's baseline rate.
    A signature that merely rides a hub has e(s) ~ 1; a genuinely directional
    signature toward a non-hub target has e(s) >> 1.

This separates genuine steering from hub attraction: it is the control that
turns "31.6% vs the next domain's 10.7%" into "126x over baseline," and it
exposes that some signatures' apparent directionality is a hub artifact (an
ICBINB-style failure mode of the naive directionality metric).

USAGE
-----
python analyze_steering_specificity.py --runs_dir all_runs \
    --focus_signature SBS17a --output_dir all_runs/steering_specificity
"""

import argparse
import glob
import os

import numpy as np
import pandas as pd


def analyze_clan(traj_csv, focus):
    t = pd.read_csv(traj_csv)
    t["sig"] = t["signature"].astype(str).str.replace("_PROFILE.txt", "", regex=False)
    top1 = t["top1_pfam"].astype(str)
    n = len(t)

    # baseline hubness b(d)
    hub = top1.value_counts(normalize=True)

    rows = []
    for sig, g in t.groupby("sig"):
        vc = g["top1_pfam"].astype(str).value_counts(normalize=True)
        target = vc.index[0]
        conc = float(vc.iloc[0])
        base = float(hub.get(target, 1.0 / n))
        rows.append({
            "signature": sig,
            "target_domain": target,
            "raw_concentration": round(conc, 4),
            "target_baseline_hubness": round(base, 4),
            "hub_corrected_enrichment": round(conc / base, 1) if base > 0 else np.inf,
            "n_experiments": len(g),
        })
    df = pd.DataFrame(rows).sort_values("hub_corrected_enrichment", ascending=False)
    df["rank_hub_corrected"] = range(1, len(df) + 1)
    # rank by raw concentration too, to show how the ranking changes
    df["rank_raw_concentration"] = (
        df["raw_concentration"].rank(ascending=False, method="min").astype(int))

    hub_top = hub.head(5)
    focus_row = df[df["signature"].str.contains(focus, na=False)]
    return df, hub_top, focus_row


def plot(all_df, out_dir):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    clans = sorted(all_df["clan"].unique())
    fig, axes = plt.subplots(1, len(clans), figsize=(7 * len(clans), 5.5), squeeze=False)
    for ax, clan in zip(axes[0], clans):
        d = all_df[all_df["clan"] == clan]
        x = d["target_baseline_hubness"] * 100
        y = d["raw_concentration"] * 100
        is_focus = d["signature"].str.contains("SBS17a", na=False)
        ax.scatter(x[~is_focus], y[~is_focus], c="#9db4c0", s=30, label="signatures")
        ax.scatter(x[is_focus], y[is_focus], c="#e32925", s=90, zorder=5, label="SBS17a")
        # diagonal y=x: on it means "no better than the target's baseline" (hub-riding)
        lim = max(x.max(), y.max()) * 1.05
        ax.plot([0, lim], [0, lim], "--", c="gray", lw=1, label="hub-riding (e=1)")
        for _, r in d[is_focus].iterrows():
            ax.annotate(f"SBS17a→{r['target_domain']}\n({r['hub_corrected_enrichment']:.0f}×)",
                        (r["target_baseline_hubness"] * 100, r["raw_concentration"] * 100),
                        textcoords="offset points", xytext=(8, -4), color="#e32925", fontsize=10)
        ax.set_xlabel("target domain baseline hubness  b(d)  [%]")
        ax.set_ylabel("signature→target concentration  c(s)  [%]")
        ax.set_title(f"{clan}: genuine steering vs hub-riding")
        ax.legend(fontsize=8)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Points far ABOVE the diagonal steer toward a NON-hub target (genuine); "
                 "points ON it merely ride a hub", fontsize=11)
    fig.tight_layout()
    p = os.path.join(out_dir, "steering_specificity.png")
    fig.savefig(p, dpi=200)
    print(f"[steer] wrote {p}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs_dir", default="all_runs")
    ap.add_argument("--focus_signature", default="SBS17a")
    ap.add_argument("--output_dir", default="all_runs/steering_specificity")
    args = ap.parse_args(argv)
    os.makedirs(args.output_dir, exist_ok=True)

    all_rows = []
    for traj in sorted(glob.glob(os.path.join(args.runs_dir, "*", "*centroid_trajectory_results.csv"))):
        clan = os.path.basename(os.path.dirname(traj))
        df, hub_top, focus = analyze_clan(traj, args.focus_signature)
        df.insert(0, "clan", clan)
        all_rows.append(df)
        df.to_csv(os.path.join(args.output_dir, f"{clan}_steering.csv"), index=False)

        print(f"\n===== {clan} =====")
        print("  hub domains (baseline top1 freq): "
              + ", ".join(f"{d} {v:.1%}" for d, v in hub_top.items()))
        if len(focus):
            r = focus.iloc[0]
            print(f"  {args.focus_signature}: target={r['target_domain']} "
                  f"conc={r['raw_concentration']:.1%} baseline={r['target_baseline_hubness']:.1%} "
                  f"=> HUB-CORRECTED ENRICHMENT {r['hub_corrected_enrichment']:.0f}x "
                  f"(rank #{int(r['rank_hub_corrected'])}/{len(df)} corrected; "
                  f"#{int(r['rank_raw_concentration'])} raw)")
        print("  top-5 by hub-corrected enrichment:")
        for _, r in df.head(5).iterrows():
            print(f"     {r['signature']:8}  {r['raw_concentration']:.1%} -> {r['target_domain']} "
                  f"(baseline {r['target_baseline_hubness']:.1%},  {r['hub_corrected_enrichment']:.0f}x)")

    if all_rows:
        combined = pd.concat(all_rows, ignore_index=True)
        combined.to_csv(os.path.join(args.output_dir, "steering_all_clans.csv"), index=False)
        plot(combined, args.output_dir)
    print(f"\n[steer] done -> {args.output_dir}/")


if __name__ == "__main__":
    main()
