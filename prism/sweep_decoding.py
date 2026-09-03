#!/usr/bin/env python3
"""
sweep_decoding.py  --  Experiment C: decoding-hyperparameter robustness.

A signal that only appears at one temperature / top_k is not a finding, it is a
decoding artifact. This script sweeps the sampling hyperparameters of the
genomic LLMs (Carbon, GENERator, Evo 2) and asks whether the quantities PRISM
cares about are STABLE across the sweep:

    * repetition-collapse rate            (the primary failure mode)
    * mean identity to true continuation  (generation quality)
    * target-signature exposure           (the actual scientific signal, e.g. UV/SBS7*)

Two subcommands:

  generate : run generate_batch_{carbon,generator,evo2}.py across a grid of
             (temperature, top_k), writing one output dir per (model, setting).
             Requires a GPU + the models (Colab). Dir names encode the setting,
             e.g.  sweep/evo2_T1.0_K4/ .

  analyze  : consume the per-setting dirs (no GPU needed) and emit
             sweep_results.csv + robustness curves (metric vs temperature,
             one line per top_k, faceted by model). Reuses the same collapse /
             identity / NNLS helpers as stress_test_generation.py so numbers are
             directly comparable.

USAGE
-----
# 1) generate the grid (Colab):
python sweep_decoding.py generate \
    --model evo2 --input_fasta bacterial_prompts.fasta \
    --temperatures 0.7,1.0,1.3 --top_ks 4,8 --num_samples 50 --prefix_frac 0.2 \
    --sweep_dir /content/drive/MyDrive/PRISM/sweep

# 2) analyze (anywhere):
python sweep_decoding.py analyze \
    --sweep_dir /content/drive/MyDrive/PRISM/sweep \
    --reference_fasta bacterial_prompts.fasta --prefix_frac 0.2 \
    --cosmic_dir /content/.../cosmic_signatures --target_signatures SBS7a,SBS7b,SBS7c,SBS7d \
    --output_dir /content/.../results/sweep_analysis
"""

import argparse
import csv
import os
import re
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

GEN_SCRIPT = {
    "carbon": "generate_batch_carbon.py",
    "generator": "generate_batch_generator.py",
    "evo2": "generate_batch_evo2.py",
}
DEFAULT_MODEL_NAME = {
    "carbon": "HuggingFaceBio/Carbon-500M",
    "generator": "GenerTeam/GENERator-v2-prokaryote-1.2b-base",
    "evo2": "evo2_7b",
}
SETTING_RE = re.compile(r"_T(?P<temp>[0-9.]+)_K(?P<topk>\d+)$")


def setting_dir_name(model, temp, topk):
    return f"{model}_T{temp}_K{topk}"


def parse_setting(dirname):
    """Extract (model, temperature, top_k) from a sweep sub-dir name."""
    m = SETTING_RE.search(dirname)
    if not m:
        return None
    model = dirname[: m.start()]
    return model, float(m.group("temp")), int(m.group("topk"))


# ── generate subcommand ─────────────────────────────────────────────────
def cmd_generate(args):
    here = os.path.dirname(os.path.abspath(__file__))
    script = os.path.join(here, GEN_SCRIPT[args.model])
    model_name = args.model_name or DEFAULT_MODEL_NAME[args.model]
    temps = [t.strip() for t in args.temperatures.split(",") if t.strip()]
    topks = [k.strip() for k in args.top_ks.split(",") if k.strip()]
    os.makedirs(args.sweep_dir, exist_ok=True)

    for temp in temps:
        for topk in topks:
            out = os.path.join(args.sweep_dir, setting_dir_name(args.model, temp, topk))
            cmd = [sys.executable, script,
                   "--input_fasta", args.input_fasta,
                   "--model_name", model_name,
                   "--num_samples", str(args.num_samples),
                   "--batch_size", str(args.batch_size),
                   "--max_new_tokens", str(args.max_new_tokens),
                   "--temperature", temp, "--top_k", topk,
                   "--prefix_frac", str(args.prefix_frac),
                   "--output_dir", out]
            if args.model == "carbon":
                cmd.append("--strip_dna_tag")
            print(f"\n[sweep] === {args.model} T={temp} K={topk} -> {out} ===")
            print("[sweep] " + " ".join(cmd))
            if args.dry_run:
                continue
            subprocess.run(cmd, check=True)
    print("\n[sweep] generation grid complete.")


# ── analyze subcommand ──────────────────────────────────────────────────
def cmd_analyze(args):
    from prism_genmodel_analysis import (
        is_degenerate, make_aligner, build_spectrum_reference,
        compute_identity_to_reference, load_expected_continuation,
    )
    from stress_test_generation import (
        list_generated_fastas, load_generated, load_reference_map,
        longest_homopolymer_frac, maybe_load_cosmic, target_exposure,
    )
    import random

    refs = load_reference_map(args.reference_fasta, args.prefix_frac)
    signatures = maybe_load_cosmic(args.cosmic_dir)
    target_sigs = [s.strip() for s in args.target_signatures.split(",") if s.strip()]
    aligner = make_aligner()
    rng = random.Random(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    settings = []
    for name in sorted(os.listdir(args.sweep_dir)):
        full = os.path.join(args.sweep_dir, name)
        if os.path.isdir(full) and parse_setting(name):
            settings.append((name, full))
    if not settings:
        raise SystemExit(f"[sweep] no setting dirs (model_T*_K*) found in {args.sweep_dir}")

    rows = []
    for name, sdir in settings:
        model, temp, topk = parse_setting(name)
        # aggregate across genes for this setting
        collapse_rates, identities, exposures = [], [], []
        for gene, path in list_generated_fastas(sdir):
            samples = load_generated(path)
            if not samples:
                continue
            collapse_rates.append(np.mean([is_degenerate(s) for s in samples]))
            survivors = [s for s in samples if not is_degenerate(s)]
            ref = refs.get(gene) or load_expected_continuation(sdir, gene)
            if ref and survivors:
                ids = compute_identity_to_reference(survivors, ref, aligner, 200, rng)
                if ids:
                    identities.append(float(np.mean(ids)))
                if signatures is not None and len(survivors) >= 2:
                    spec = build_spectrum_reference(survivors, ref, aligner, 200, rng)
                    exposures.append(target_exposure(spec, signatures, target_sigs))
        rows.append([
            model, temp, topk,
            round(float(np.mean(collapse_rates)), 4) if collapse_rates else "",
            round(float(np.mean(identities)), 4) if identities else "",
            round(float(np.mean(exposures)), 4) if exposures else "",
            len(collapse_rates),
        ])
        print(f"[sweep] {name}: collapse={_f(rows[-1][3])} identity={_f(rows[-1][4])} "
              f"target_sig_exposure={_f(rows[-1][5])} ({rows[-1][6]} genes)")

    out_csv = os.path.join(args.output_dir, "sweep_results.csv")
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "temperature", "top_k", "collapse_rate",
                    "mean_identity", "target_signature_exposure", "n_genes"])
        w.writerows(rows)
    print(f"[sweep] wrote {out_csv}")
    _plot_sweep(rows, args.output_dir, has_sig=signatures is not None)


def _f(x):
    return f"{x:.3f}" if isinstance(x, float) else str(x)


def _plot_sweep(rows, out_dir, has_sig):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[sweep] matplotlib unavailable; skipping plots")
        return
    metrics = [("collapse_rate", 3, "repetition-collapse rate"),
               ("mean_identity", 4, "mean identity to reference")]
    if has_sig:
        metrics.append(("target_signature_exposure", 5, "target-signature exposure"))
    models = sorted({r[0] for r in rows})
    fig, axes = plt.subplots(1, len(metrics), figsize=(6 * len(metrics), 5), squeeze=False)
    for ax, (mkey, col, mlabel) in zip(axes[0], metrics):
        for model in models:
            for topk in sorted({r[2] for r in rows if r[0] == model}):
                pts = sorted([(r[1], r[col]) for r in rows
                              if r[0] == model and r[2] == topk and isinstance(r[col], float)])
                if pts:
                    xs, ys = zip(*pts)
                    ax.plot(xs, ys, marker="o", label=f"{model} k={topk}")
        ax.set_xlabel("temperature")
        ax.set_ylabel(mlabel)
        ax.set_title(f"{mlabel} vs decoding")
        ax.legend(fontsize=8)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Decoding-hyperparameter robustness (Experiment C): "
                 "does the signal survive the sweep?")
    fig.tight_layout()
    path = os.path.join(out_dir, "sweep_robustness.png")
    fig.savefig(path, dpi=200)
    print(f"[sweep] wrote {path}")


# ── CLI ─────────────────────────────────────────────────────────────────
def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)

    g = sub.add_parser("generate", help="Run the generation grid (needs GPU + models).")
    g.add_argument("--model", required=True, choices=list(GEN_SCRIPT))
    g.add_argument("--model_name", default=None, help="Override the HF/native checkpoint name.")
    g.add_argument("--input_fasta", required=True)
    g.add_argument("--temperatures", default="0.7,1.0,1.3")
    g.add_argument("--top_ks", default="4,8")
    g.add_argument("--num_samples", type=int, default=50)
    g.add_argument("--batch_size", type=int, default=5)
    g.add_argument("--max_new_tokens", type=int, default=500)
    g.add_argument("--prefix_frac", type=float, default=0.2)
    g.add_argument("--sweep_dir", required=True)
    g.add_argument("--dry_run", action="store_true", help="Print commands without running.")

    a = sub.add_parser("analyze", help="Analyze the generation grid (no GPU needed).")
    a.add_argument("--sweep_dir", required=True)
    a.add_argument("--reference_fasta", required=True)
    a.add_argument("--prefix_frac", type=float, default=0.2)
    a.add_argument("--cosmic_dir", default=None)
    a.add_argument("--target_signatures", default="SBS7a,SBS7b,SBS7c,SBS7d")
    a.add_argument("--output_dir", required=True)
    a.add_argument("--seed", type=int, default=0)
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.command == "generate":
        cmd_generate(args)
    else:
        cmd_analyze(args)


if __name__ == "__main__":
    main()
