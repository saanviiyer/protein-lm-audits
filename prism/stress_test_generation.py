#!/usr/bin/env python3
"""
stress_test_generation.py

A cross-architecture ROBUSTNESS / FAILURE-MODE battery for the PRISM
generative-model-bias pipeline, built for the "I Can't Believe It's Not
Better: Failure Modes of AI in Biology" (NeurIPS 2026) framing.

It runs on outputs already produced by:
    generate_batch_carbon.py     (Carbon, 6-mer tokenizer)
    generate_batch_generator.py  (GENERator, 6-mer tokenizer)
    generate_batch_evo2.py       (Evo 2, single-nucleotide tokenizer)
each of which writes <gene>_generated.fasta [+ <gene>_expected_continuation.fasta]
into a per-model directory. No GPU is required to run this analysis; it
consumes the FASTAs only.

The battery reuses the *live* helpers in prism_genmodel_analysis.py so the
numbers are directly comparable to the paper's, and adds three experiments:

  A. Collapse & generation-quality profile (per model, per gene)
     - repetition-collapse rate (is_degenerate), and the full DISTRIBUTION of
       longest-homopolymer-run fraction (collapse severity), which exposes
       whether a single-nucleotide model (Evo 2) collapses differently than a
       6-mer model (Carbon/GENERator).
     - translation-success rate and identity-to-reference (bootstrap 95% CI).

  B. Tokenizer x alignment artifact isolation (needs reference continuations)
     - the paper traced a spurious UV/SBS7c signal to a
       collapse x mismatch-averse-alignment interaction. Here we rebuild the
       reference-based 96-channel spectrum under a 2x2 grid
       {degenerate kept / filtered} x {mismatch-averse / permissive aligner}
       for every model, and (if --cosmic_dir is given) NNLS-decompose each to
       report how much of the target signature family is ARTIFACT vs. signal.

  D. Sample-budget / bootstrap stability (needs reference [+ cosmic])
     - resamples accepted samples at increasing N and reports the CI width of
       the target-signature exposure vs. N, turning the paper's "low post-QC
       yield -> small effective N" observation into a power/robustness curve.

See the C-experiment (decoding-hyperparameter sweep) notes at the bottom:
it is an orchestration over multiple generation runs; this script analyses
its outputs when you pass them as extra --model_dir entries.

USAGE
-----
python stress_test_generation.py \
    --model_dir carbon:/content/drive/MyDrive/PRISM/genmodel_bias/bacterial_carbon_output \
    --model_dir generator:/content/drive/MyDrive/PRISM/genmodel_bias/bacterial_generator_output \
    --model_dir evo2:/content/drive/MyDrive/PRISM/genmodel_bias/bacterial_evo2_output \
    --reference_fasta /content/drive/MyDrive/PRISM/genmodel_bias/bacterial_prompts.fasta \
    --prefix_frac 0.2 \
    --cosmic_dir /content/drive/MyDrive/PRISM/cosmic_signatures \
    --target_signatures SBS7a,SBS7b,SBS7c,SBS7d \
    --output_dir /content/drive/MyDrive/PRISM/genmodel_bias/results/stress_test
"""

import argparse
import csv
import os
import random
import re
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prism_genmodel_analysis import (  # live helpers, shared with the paper's pipeline
    is_degenerate,
    make_aligner,
    build_96_context_keys,
    build_spectrum_reference,
    compute_identity_to_reference,
    translate_orf,
    load_expected_continuation,
    load_model_dirs,
)

try:
    from Bio import Align
    from Bio import SeqIO
except ImportError as e:  # pragma: no cover
    raise SystemExit(f"BioPython required: {e}")


# ── generic helpers ─────────────────────────────────────────────────────
TOKEN_RE = re.compile(r"<[^>]+>")


def sanitize_dna(seq: str) -> str:
    seq = TOKEN_RE.sub("", str(seq))
    return re.sub(r"[^ACGTNacgtn]", "", seq).upper()


def longest_homopolymer_frac(seq: str) -> float:
    """Fraction of the sequence occupied by its longest single-base run.
    This is the continuous quantity underlying the binary is_degenerate()."""
    if not seq:
        return 1.0
    longest = max(len(m.group(0)) for m in re.finditer(r"([ACGTN])\1*", seq))
    return longest / len(seq)


def list_generated_fastas(model_dir: str):
    """Yield (gene_id, path) for every <gene>_generated.fasta in a model dir."""
    for fname in sorted(os.listdir(model_dir)):
        if fname.endswith("_generated.fasta"):
            gene = fname[: -len("_generated.fasta")]
            yield gene, os.path.join(model_dir, fname)


def load_generated(path: str):
    return [sanitize_dna(rec.seq) for rec in SeqIO.parse(path, "fasta")]


def load_reference_map(reference_fasta: str, prefix_frac: float):
    """Return {gene_id: true_continuation_seq} using the SAME prefix split the
    generators used, so this works even if a model dir lacks the sidecar
    *_expected_continuation.fasta files."""
    refs = {}
    if not reference_fasta:
        return refs
    for rec in SeqIO.parse(reference_fasta, "fasta"):
        full = sanitize_dna(rec.seq)
        cut = int(len(full) * prefix_frac)
        refs[rec.id] = full[cut:]
    return refs


def bootstrap_ci(values, n_boot=2000, alpha=0.05, rng=None):
    """Percentile bootstrap CI of the mean. Returns (mean, lo, hi)."""
    vals = np.asarray([v for v in values if v is not None], dtype=float)
    if vals.size == 0:
        return (float("nan"), float("nan"), float("nan"))
    if vals.size == 1:
        return (float(vals[0]), float(vals[0]), float(vals[0]))
    rng = rng or np.random.default_rng(0)
    idx = rng.integers(0, vals.size, size=(n_boot, vals.size))
    means = vals[idx].mean(axis=1)
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return (float(vals.mean()), float(lo), float(hi))


def make_strict_aligner():
    """A deliberately mismatch-AVERSE aligner: heavy gap penalties push small
    length differences to be resolved as strings of substitutions instead of a
    single indel. This is the mis-configuration that inflated the paper's
    SBS7c signal; we keep it here on purpose to quantify the artifact."""
    aligner = Align.PairwiseAligner()
    aligner.mode = "global"
    aligner.match_score = 2
    aligner.mismatch_score = -1
    aligner.open_gap_score = -10.0
    aligner.extend_gap_score = -5.0
    return aligner


# ── COSMIC (optional) ────────────────────────────────────────────────────
def maybe_load_cosmic(cosmic_dir):
    if not cosmic_dir:
        return None
    from prism_genmodel_analysis import load_cosmic_signatures_dir
    sigs = load_cosmic_signatures_dir(cosmic_dir)
    print(f"[stress] Loaded {sigs.shape[1]} COSMIC signatures x {sigs.shape[0]} contexts")
    return sigs


def target_exposure(spectrum, signatures, target_sigs):
    """NNLS-decompose a 96-channel spectrum and return the summed exposure of
    the target signature family (e.g. the SBS7* UV family)."""
    from prism_genmodel_analysis import nnls_decompose
    exposures, _ = nnls_decompose(spectrum, signatures)
    present = [s for s in target_sigs if s in exposures.index]
    return float(exposures.loc[present].sum()) if present else 0.0


# ══════════════════════════════════════════════════════════════════════════
# Experiment A — collapse & generation-quality profile
# ══════════════════════════════════════════════════════════════════════════
def experiment_a(model_dirs, refs, aligner, out_dir, rng_np):
    print("\n[stress][A] Collapse & generation-quality profile")
    profile_rows = []
    severity_rows = []  # long-form: model, gene, sample_idx, homopolymer_frac, collapsed

    for model, mdir in model_dirs.items():
        for gene, path in list_generated_fastas(mdir):
            samples = load_generated(path)
            n_raw = len(samples)
            if n_raw == 0:
                continue

            fracs = [longest_homopolymer_frac(s) for s in samples]
            collapsed_flags = [is_degenerate(s) for s in samples]
            n_collapsed = sum(collapsed_flags)
            survivors = [s for s, c in zip(samples, collapsed_flags) if not c]

            for i, (f, c) in enumerate(zip(fracs, collapsed_flags)):
                severity_rows.append([model, gene, i, round(f, 4), int(c)])

            # translation success among survivors
            n_translate = sum(translate_orf(s, min_aa=10) is not None for s in survivors)
            trans_rate = n_translate / len(survivors) if survivors else float("nan")

            # identity to true reference continuation (survivors only)
            ref = refs.get(gene) or load_expected_continuation(mdir, gene)
            if ref:
                idents = compute_identity_to_reference(
                    survivors, ref, aligner, max_samples=200, rng=random.Random(0))
                id_mean, id_lo, id_hi = bootstrap_ci(idents, rng=rng_np)
            else:
                id_mean = id_lo = id_hi = float("nan")

            profile_rows.append([
                model, gene, n_raw, n_collapsed,
                round(n_collapsed / n_raw, 4),
                round(float(np.mean(fracs)), 4),
                round(float(np.median(fracs)), 4),
                round(trans_rate, 4) if trans_rate == trans_rate else "",
                round(id_mean, 4) if id_mean == id_mean else "",
                round(id_lo, 4) if id_lo == id_lo else "",
                round(id_hi, 4) if id_hi == id_hi else "",
            ])

    _write_csv(os.path.join(out_dir, "A_collapse_profile.csv"),
               ["model", "gene", "n_raw", "n_collapsed", "collapse_rate",
                "mean_homopolymer_frac", "median_homopolymer_frac",
                "translation_rate", "identity_mean", "identity_ci_lo", "identity_ci_hi"],
               profile_rows)
    _write_csv(os.path.join(out_dir, "A_collapse_severity_long.csv"),
               ["model", "gene", "sample_idx", "homopolymer_frac", "collapsed"],
               severity_rows)

    # per-model summary (matches the paper's headline numbers)
    print("  per-model mean collapse rate:")
    by_model = {}
    for r in profile_rows:
        by_model.setdefault(r[0], []).append(r[4])
    for model, rates in by_model.items():
        print(f"    {model:12s} collapse_rate = {np.mean(rates):.3%} "
              f"(range {min(rates):.1%}-{max(rates):.1%}, {len(rates)} genes)")

    _plot_collapse(profile_rows, severity_rows, out_dir)
    return profile_rows, severity_rows


# ══════════════════════════════════════════════════════════════════════════
# Experiment B — tokenizer x alignment artifact isolation
# ══════════════════════════════════════════════════════════════════════════
def experiment_b(model_dirs, refs, out_dir, signatures, target_sigs):
    print("\n[stress][B] Tokenizer x alignment artifact isolation")
    permissive = make_aligner()      # project default (open=-2): resolves length diffs as indels
    strict = make_strict_aligner()   # mismatch-averse (open=-10): the artifact-inducing setting
    keys = build_96_context_keys()
    rng = random.Random(0)

    rows = []
    conditions = [
        ("kept",     "strict",     False, strict),
        ("kept",     "permissive", False, permissive),
        ("filtered", "strict",     True,  strict),
        ("filtered", "permissive", True,  permissive),
    ]

    for model, mdir in model_dirs.items():
        for gene, path in list_generated_fastas(mdir):
            samples = load_generated(path)
            ref = refs.get(gene) or load_expected_continuation(mdir, gene)
            if not ref or not samples:
                continue

            for filt_label, aln_label, do_filter, aligner in conditions:
                seqs = [s for s in samples if not is_degenerate(s)] if do_filter else samples
                if len(seqs) < 2:
                    continue
                spectrum = build_spectrum_reference(
                    seqs, ref, aligner, max_samples=200, rng=rng)
                total_subs = float(spectrum.sum())
                row = [model, gene, filt_label, aln_label, len(seqs), total_subs]
                if signatures is not None:
                    row.append(round(target_exposure(spectrum, signatures, target_sigs), 4))
                rows.append(row)

    header = ["model", "gene", "degenerate", "aligner", "n_used", "total_substitutions"]
    if signatures is not None:
        header.append("target_signature_exposure")
    _write_csv(os.path.join(out_dir, "B_artifact_isolation.csv"), header, rows)

    # headline: substitution inflation from strict+kept vs permissive+filtered
    _summarize_artifact(rows, signatures is not None)
    return rows


# ══════════════════════════════════════════════════════════════════════════
# Experiment D — sample-budget / bootstrap stability
# ══════════════════════════════════════════════════════════════════════════
def experiment_d(model_dirs, refs, out_dir, signatures, target_sigs, budget_grid, rng_np):
    print("\n[stress][D] Sample-budget / bootstrap stability")
    if signatures is None:
        print("  [skip] needs --cosmic_dir to compute signature exposure stability")
        return []
    from prism_genmodel_analysis import extract_substitutions
    permissive = make_aligner()
    keys = build_96_context_keys()
    kidx = {k: i for i, k in enumerate(keys)}
    rows = []

    # Cap samples per gene so the one-time alignment pass stays bounded.
    max_precompute = 60

    for model, mdir in model_dirs.items():
        for gene, path in list_generated_fastas(mdir):
            samples = [s for s in load_generated(path) if not is_degenerate(s)]
            ref = refs.get(gene) or load_expected_continuation(mdir, gene)
            if not ref or len(samples) < 5:
                continue
            samples = samples[:max_precompute]
            # ── Precompute each sample's 96-channel spectrum ONCE (one alignment
            #    per sample). Bootstrap draws then just SUM these vectors, so no
            #    alignment happens inside the bootstrap loop (was the O(200*N)
            #    hot path that made this experiment take hours). ──
            per_sample = np.zeros((len(samples), 96), dtype=float)
            for si, s in enumerate(samples):
                for five, r, alt, three in extract_substitutions(ref, s, permissive):
                    k = f"{five}[{r}>{alt}]{three}"
                    if k in kidx:
                        per_sample[si, kidx[k]] += 1
            n_available = len(samples)
            for N in budget_grid:
                if N > n_available:
                    continue
                exposures = []
                for _ in range(200):  # bootstrap draws of size N (vectorized sum)
                    idx = rng_np.integers(0, n_available, size=N)
                    spec = pd.Series(per_sample[idx].sum(axis=0), index=keys)
                    exposures.append(target_exposure(spec, signatures, target_sigs))
                mean, lo, hi = bootstrap_ci(exposures, n_boot=1, rng=rng_np)  # already bootstrapped
                arr = np.asarray(exposures)
                rows.append([model, gene, N, n_available,
                             round(float(arr.mean()), 4),
                             round(float(np.percentile(arr, 2.5)), 4),
                             round(float(np.percentile(arr, 97.5)), 4),
                             round(float(np.percentile(arr, 97.5) - np.percentile(arr, 2.5)), 4)])

    _write_csv(os.path.join(out_dir, "D_budget_stability.csv"),
               ["model", "gene", "N", "n_available", "exposure_mean",
                "exposure_ci_lo", "exposure_ci_hi", "ci_width"], rows)
    _plot_stability(rows, out_dir)
    return rows


# ── output helpers ───────────────────────────────────────────────────────
def _write_csv(path, header, rows):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    print(f"  wrote {path}  ({len(rows)} rows)")


def _summarize_artifact(rows, has_sig):
    # index by (model, gene, degenerate, aligner)
    idx = {(r[0], r[1], r[2], r[3]): r for r in rows}
    models = sorted({r[0] for r in rows})
    print("  substitution-count inflation (strict+kept / permissive+filtered):")
    for model in models:
        ratios = []
        exp_kept, exp_filt = [], []
        for (m, g, deg, aln), r in idx.items():
            if m != model:
                continue
            worst = idx.get((m, g, "kept", "strict"))
            best = idx.get((m, g, "filtered", "permissive"))
            if worst and best and best[5] > 0:
                ratios.append(worst[5] / best[5])
            if has_sig and worst and best:
                exp_kept.append(worst[6]); exp_filt.append(best[6])
        if ratios:
            msg = f"    {model:12s} median inflation x{np.median(ratios):.2f} (n={len(ratios)} genes)"
            if has_sig and exp_kept:
                msg += (f" | target-sig exposure: worst={np.mean(exp_kept):.3f} "
                        f"best={np.mean(exp_filt):.3f}")
            print(msg)


def _plot_collapse(profile_rows, severity_rows, out_dir):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  [plot] matplotlib unavailable; skipping plots")
        return
    models = sorted({r[0] for r in profile_rows})
    # (1) collapse rate by model (bar of per-gene means +/- spread)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    rates = [[r[4] for r in profile_rows if r[0] == m] for m in models]
    axes[0].bar(range(len(models)), [np.mean(x) for x in rates],
                yerr=[np.std(x) for x in rates], capsize=4, color="#4c72b0")
    axes[0].set_xticks(range(len(models)))
    axes[0].set_xticklabels(models, rotation=20)
    axes[0].set_ylabel("repetition-collapse rate")
    axes[0].set_title("Collapse rate by architecture")
    # (2) collapse severity distribution (homopolymer frac) per model
    data = [[r[3] for r in severity_rows if r[0] == m] for m in models]
    parts = axes[1].violinplot(data, showmedians=True)
    axes[1].axhline(0.3, ls="--", c="red", lw=1, label="is_degenerate threshold (0.30)")
    axes[1].set_xticks(range(1, len(models) + 1))
    axes[1].set_xticklabels(models, rotation=20)
    axes[1].set_ylabel("longest-homopolymer fraction")
    axes[1].set_title("Collapse severity distribution")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "A_collapse.png"), dpi=200)
    plt.close(fig)
    print(f"  wrote {os.path.join(out_dir, 'A_collapse.png')}")


def _plot_stability(rows, out_dir):
    if not rows:
        return
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    models = sorted({r[0] for r in rows})
    fig, ax = plt.subplots(figsize=(8, 5))
    for m in models:
        by_N = {}
        for r in rows:
            if r[0] == m:
                by_N.setdefault(r[2], []).append(r[7])  # ci_width
        Ns = sorted(by_N)
        ax.plot(Ns, [np.mean(by_N[n]) for n in Ns], marker="o", label=m)
    ax.set_xlabel("sample budget N (post-QC)")
    ax.set_ylabel("mean 95% CI width of target-signature exposure")
    ax.set_title("Estimate stability vs. sample budget")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "D_stability.png"), dpi=200)
    plt.close(fig)
    print(f"  wrote {os.path.join(out_dir, 'D_stability.png')}")


# ── CLI ──────────────────────────────────────────────────────────────────
def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model_dir", action="append", required=True,
                   help="label:path, repeatable (carbon:/..., generator:/..., evo2:/...).")
    p.add_argument("--reference_fasta", type=str, default=None,
                   help="FASTA of full reference CDS per gene (for identity/spectrum). "
                        "Optional if each model dir has *_expected_continuation.fasta.")
    p.add_argument("--prefix_frac", type=float, default=0.2,
                   help="Prefix fraction used at generation time (to recover the true tail).")
    p.add_argument("--cosmic_dir", type=str, default=None,
                   help="COSMIC *_PROFILE.txt dir; enables NNLS signature-exposure metrics.")
    p.add_argument("--target_signatures", type=str, default="SBS7a,SBS7b,SBS7c,SBS7d",
                   help="Comma-separated signature family to track (default: UV/SBS7*).")
    p.add_argument("--budget_grid", type=str, default="5,10,15,20,30,50",
                   help="Sample-budget sizes for experiment D.")
    p.add_argument("--experiments", type=str, default="A,B,D",
                   help="Which experiments to run (subset of A,B,D).")
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    os.makedirs(args.output_dir, exist_ok=True)
    model_dirs = load_model_dirs(args.model_dir)
    refs = load_reference_map(args.reference_fasta, args.prefix_frac)
    signatures = maybe_load_cosmic(args.cosmic_dir)
    target_sigs = [s.strip() for s in args.target_signatures.split(",") if s.strip()]
    budget_grid = [int(x) for x in args.budget_grid.split(",") if x.strip()]
    todo = {e.strip().upper() for e in args.experiments.split(",")}
    aligner = make_aligner()
    rng_np = np.random.default_rng(args.seed)

    print(f"[stress] models: {list(model_dirs)}")
    print(f"[stress] reference genes: {len(refs)} | cosmic: {signatures is not None} "
          f"| target sigs: {target_sigs}")

    if "A" in todo:
        experiment_a(model_dirs, refs, aligner, args.output_dir, rng_np)
    if "B" in todo:
        experiment_b(model_dirs, refs, args.output_dir, signatures, target_sigs)
    if "D" in todo:
        experiment_d(model_dirs, refs, args.output_dir, signatures, target_sigs,
                     budget_grid, rng_np)

    print(f"\n[stress] Done. Results in {args.output_dir}/")


if __name__ == "__main__":
    main()
