#!/usr/bin/env python3
"""
analyze_bacterial_generation_matched.py

Reconciles our bacterial-gene generation results with Gabriela's
methodology, so the two are directly comparable:

  1. BLAST-based QC filter (>=80% query coverage of the true reference
     protein), matching her filter_generated() -- discards early-stop-
     codon and frameshifted junk that our homopolymer-only filter let
     through.
  2. BOTH MAFFT (real MSA) and Biopython pairwise alignment are run on
     the SAME accepted sequences, so we can directly answer her
     question: does the alignment method itself change the spectrum?
  3. Substitutions are only counted strictly AFTER the prompt boundary
     (matching her extract_generated_substitutions logic).
  4. Reports mean gap fraction for both alignment methods -- a direct,
     quantitative answer to "was the alignment very gappy?"
  5. Produces COSMIC-style spectrum plots for both alignment methods,
     per model, for side-by-side visual comparison.

REQUIRES (install first):
    !apt-get install -y ncbi-blast+ mafft

USAGE
-----
python analyze_bacterial_generation_matched.py \
    --bacterial_reference_fasta /content/drive/MyDrive/PRISM/genmodel_bias/bacterial_prompts.fasta \
    --carbon_dir /content/drive/MyDrive/PRISM/genmodel_bias/bacterial_carbon_output \
    --generator_dir /content/drive/MyDrive/PRISM/genmodel_bias/bacterial_generator_output \
    --output_dir /content/drive/MyDrive/PRISM/genmodel_bias/results/bacterial_matched \
    --prefix_frac 0.2
"""

import argparse
import csv
import os
import re
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from Bio import SeqIO
from Bio.Seq import Seq

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prism_genmodel_analysis import make_aligner, extract_substitutions

BASES = "ACGT"
TOKEN_RE = re.compile(r"<[^>]+>")
COMP = str.maketrans("ACGTNacgtn", "TGCANtgcan")


def sanitize_dna(seq: str) -> str:
    seq = TOKEN_RE.sub("", str(seq))
    return re.sub(r"[^ACGTNacgtn]", "", seq).upper()


def translate(seq: str, to_stop: bool = True) -> str:
    seq = sanitize_dna(seq)
    usable = seq[: len(seq) - (len(seq) % 3)]
    if not usable:
        return ""
    return str(Seq(usable).translate(to_stop=to_stop))


def trim_to_cds(seq: str, fallback_len: int) -> str:
    seq = sanitize_dna(seq)
    usable = seq[: len(seq) - (len(seq) % 3)]
    stops = {"TAA", "TAG", "TGA"}
    for idx in range(0, len(usable), 3):
        if usable[idx: idx + 3] in stops:
            return usable[: idx + 3]
    return usable[: min(len(usable), int(fallback_len * 1.25))]


def blastp_target_coverage(target_aa: str, candidate_aa: str) -> dict:
    """Same logic as Gabriela's blastp_target_coverage -- required for
    a fair, matched comparison."""
    if not target_aa or not candidate_aa:
        return {"pident": 0.0, "query_coverage": 0.0, "bitscore": 0.0}
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        query, subject, out = tmp / "query.faa", tmp / "subject.faa", tmp / "blast.tsv"
        query.write_text(f">target\n{target_aa}\n")
        subject.write_text(f">candidate\n{candidate_aa}\n")
        try:
            subprocess.run(
                ["blastp", "-query", str(query), "-subject", str(subject),
                 "-outfmt", "6 pident length qlen slen evalue bitscore",
                 "-max_target_seqs", "1", "-max_hsps", "1", "-out", str(out)],
                check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            print(f"    [debug] blastp failed: {e}")
            return {"pident": 0.0, "query_coverage": 0.0, "bitscore": 0.0}
        if not out.exists() or out.stat().st_size == 0:
            return {"pident": 0.0, "query_coverage": 0.0, "bitscore": 0.0}
        fields = out.read_text().strip().split("\t")
        pident, length, qlen, _slen, _evalue, bitscore = fields
        return {"pident": float(pident), "query_coverage": 100.0 * int(length) / int(qlen),
                "bitscore": float(bitscore)}


def write_fasta(records, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as h:
        for header, seq in records:
            h.write(f">{header}\n")
            for i in range(0, len(seq), 80):
                h.write(seq[i:i + 80] + "\n")


def revcomp(seq: str) -> str:
    return seq.translate(COMP)[::-1].upper()


def cosmic_category(ref_tri: str, alt_tri: str) -> str:
    ref, alt = ref_tri[1], alt_tri[1]
    if ref in "AG":
        ref_tri, alt_tri = revcomp(ref_tri), revcomp(alt_tri)
        ref, alt = ref_tri[1], alt_tri[1]
    return f"{ref_tri[0]}[{ref}>{alt}]{ref_tri[2]}"


def category_order():
    cats = []
    for ref, alts in [("C", "ATG"), ("T", "ACG")]:
        for alt in alts:
            for l in BASES:
                for r in BASES:
                    cats.append(f"{l}[{ref}>{alt}]{r}")
    return cats


def extract_from_mafft(aligned_path: Path, ref_id: str, prompt_len: int) -> Counter:
    """Substitutions from an MSA (reference + all accepted candidates),
    counted only strictly after the prompt boundary."""
    records = list(SeqIO.parse(aligned_path, "fasta")) if aligned_path.exists() else []
    if not records:
        return Counter(), 0.0
    ref_idx = next((i for i, r in enumerate(records) if r.id == ref_id), 0)
    ref_seq = str(records[ref_idx].seq).upper()

    ref_pos_by_col, ref_base_cols, pos = {}, [], 0
    for col, base in enumerate(ref_seq):
        if base != "-":
            pos += 1
            ref_pos_by_col[col] = pos
            if base in BASES:
                ref_base_cols.append(col)
    col_for_ref_pos = {ref_pos_by_col[c]: c for c in ref_base_cols}

    counts = Counter()
    gap_cols, total_cols = 0, 0
    for i, rec in enumerate(records):
        if i == ref_idx:
            continue
        seq = str(rec.seq).upper()
        for col, ref_base in enumerate(ref_seq):
            total_cols += 1
            alt = seq[col]
            if alt == "-" or ref_base == "-":
                gap_cols += 1
                continue
            if ref_base not in BASES:
                continue
            ref_pos = ref_pos_by_col[col]
            if ref_pos <= prompt_len:
                continue
            if alt == ref_base:
                continue
            left_col, right_col = col_for_ref_pos.get(ref_pos - 1), col_for_ref_pos.get(ref_pos + 1)
            if left_col is None or right_col is None:
                continue
            left, right = ref_seq[left_col], ref_seq[right_col]
            ref_tri, alt_tri = left + ref_base + right, left + alt + right
            if alt in BASES and set(ref_tri) <= set(BASES):
                counts[cosmic_category(ref_tri, alt_tri)] += 1
    gap_frac = gap_cols / total_cols if total_cols else 0.0
    return counts, gap_frac


def extract_from_pairwise(ref_seq: str, candidates: list, prompt_len: int, aligner) -> tuple:
    """Same substitution extraction, but via Biopython pairwise
    alignment (one alignment per candidate) instead of MAFFT MSA --
    run on the IDENTICAL accepted-sequence set for direct comparison."""
    counts = Counter()
    gap_fracs = []
    for candidate_seq in candidates:
        alignment = aligner.align(ref_seq, candidate_seq)[0]
        a_aligned, b_aligned = str(alignment[0]), str(alignment[1])
        gap_cols = sum(1 for x, y in zip(a_aligned, b_aligned) if x == "-" or y == "-")
        gap_fracs.append(gap_cols / len(a_aligned) if a_aligned else 0.0)

        ref_pos = 0
        for i in range(1, len(a_aligned) - 1):
            a0 = a_aligned[i]
            if a0 != "-":
                ref_pos += 1
            if a0 == "-" or ref_pos <= prompt_len:
                continue
            b0 = b_aligned[i]
            if b0 == "-" or b0 == a0 or a0 not in BASES:
                continue
            a5, a3 = a_aligned[i - 1], a_aligned[i + 1]
            if "-" in (a5, a3):
                continue
            ref_tri, alt_tri = a5 + a0 + a3, a5 + b0 + a3
            if b0 in BASES and set(ref_tri) <= set(BASES):
                counts[cosmic_category(ref_tri, alt_tri)] += 1
    mean_gap = sum(gap_fracs) / len(gap_fracs) if gap_fracs else 0.0
    return counts, mean_gap


def plot_spectrum(counts_by_model: dict, title: str, out_path: Path):
    order = category_order()
    palette = {"C>A": "#03bcee", "C>G": "#000000", "C>T": "#e32925",
               "T>A": "#cac9c9", "T>C": "#a1cf64", "T>G": "#edc6c4"}
    models = list(counts_by_model.keys())
    fig, axes = plt.subplots(len(models), 1, figsize=(18, 4 * len(models)), sharex=True)
    if len(models) == 1:
        axes = [axes]
    for ax, model in zip(axes, models):
        values = [counts_by_model[model].get(cat, 0) for cat in order]
        colors = [palette[cat[2:5]] for cat in order]
        ax.bar(range(len(order)), values, color=colors, width=0.85)
        ax.set_ylabel(f"{model}\ncount")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[-1].set_xticks(range(len(order)))
    axes[-1].set_xticklabels(order, rotation=90, fontsize=6)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bacterial_reference_fasta", required=True,
                    help="The bacterial_prompts.fasta with FULL reference CDS per gene "
                         "(from build_bacterial_prompts.py).")
    # Convenience per-model flags (backward compatible) ...
    p.add_argument("--carbon_dir", default=None)
    p.add_argument("--generator_dir", default=None)
    p.add_argument("--evo2_dir", default=None,
                    help="Evo 2 output dir (from generate_batch_evo2.py). Same "
                         "<gene>_generated.fasta convention as Carbon/GENERator.")
    # ... plus a general, repeatable label:path form for any set of models.
    p.add_argument("--model_dir", action="append", default=[],
                    help="label:path, repeatable, e.g. evo2:/path/evo2_output. "
                         "Use this for arbitrary models beyond carbon/generator/evo2.")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--prefix_frac", type=float, default=0.2)
    p.add_argument("--min_query_coverage", type=float, default=80.0)
    args = p.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    aligner = make_aligner()

    references = {r.id: str(r.seq).upper() for r in SeqIO.parse(args.bacterial_reference_fasta, "fasta")}

    # Assemble model_dirs from convenience flags + repeatable --model_dir entries.
    model_dirs = {}
    for label, path in (("carbon", args.carbon_dir),
                        ("generator", args.generator_dir),
                        ("evo2", args.evo2_dir)):
        if path:
            model_dirs[label] = Path(path)
    for entry in args.model_dir:
        if ":" not in entry:
            p.error(f"--model_dir expects 'label:path', got: {entry!r}")
        label, path = entry.split(":", 1)
        model_dirs[label] = Path(path)
    if not model_dirs:
        p.error("Provide at least one model directory "
                "(--carbon_dir/--generator_dir/--evo2_dir or --model_dir label:path).")

    mafft_counts, pairwise_counts = {}, {}
    qc_summary = []
    gap_report = []

    for model, model_dir in model_dirs.items():
        combined_mafft, combined_pairwise = Counter(), Counter()

        for gene, ref_seq in references.items():
            prompt_len = int(len(ref_seq) * args.prefix_frac)
            true_prompt = ref_seq[:prompt_len]
            target_aa = translate(ref_seq)

            fasta_path = model_dir / f"{gene}_generated.fasta"
            if not fasta_path.exists():
                print(f"  [warn] {model}/{gene}: no generated FASTA found at {fasta_path}, skipping")
                continue

            accepted = []
            n_raw = 0
            for rec in SeqIO.parse(fasta_path, "fasta"):
                n_raw += 1
                continuation = sanitize_dna(str(rec.seq))
                candidate_cds = trim_to_cds(true_prompt + continuation, len(ref_seq))
                candidate_aa = translate(candidate_cds)
                metrics = blastp_target_coverage(target_aa, candidate_aa)
                if metrics["query_coverage"] >= args.min_query_coverage:
                    accepted.append(candidate_cds)

            print(f"  {model}/{gene}: {len(accepted)}/{n_raw} passed >= {args.min_query_coverage}% "
                  f"BLAST query coverage")
            qc_summary.append([model, gene, n_raw, len(accepted)])

            if len(accepted) < 2:
                print(f"    [warn] fewer than 2 accepted sequences for {model}/{gene}, skipping alignment")
                continue

            # ── MAFFT (real MSA) ──
            gene_dir = output_dir / "msa" / model
            gene_dir.mkdir(parents=True, exist_ok=True)
            combined_fasta = gene_dir / f"{gene}_input.fasta"
            aligned_fasta = gene_dir / f"{gene}_aligned.fasta"
            write_fasta([(f"{gene}_REF", ref_seq)] + [(f"{gene}_c{i}", s) for i, s in enumerate(accepted)],
                        combined_fasta)
            try:
                with aligned_fasta.open("w") as out:
                    subprocess.run(["mafft", "--auto", str(combined_fasta)],
                                    check=True, stdout=out, stderr=subprocess.DEVNULL)
                mcounts, mgap = extract_from_mafft(aligned_fasta, f"{gene}_REF", prompt_len)
            except (subprocess.CalledProcessError, FileNotFoundError) as e:
                print(f"    [debug] mafft failed for {model}/{gene}: {e}")
                mcounts, mgap = Counter(), None

            # ── Pairwise (Biopython), SAME accepted sequences ──
            pcounts, pgap = extract_from_pairwise(ref_seq, accepted, prompt_len, aligner)

            combined_mafft.update(mcounts)
            combined_pairwise.update(pcounts)
            gap_report.append([model, gene, mgap, pgap])

        mafft_counts[model] = combined_mafft
        pairwise_counts[model] = combined_pairwise

    # ── Write QC + gap reports ──
    with (output_dir / "qc_summary.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "gene", "n_raw", "n_accepted"])
        w.writerows(qc_summary)

    with (output_dir / "alignment_gap_fraction.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "gene", "mafft_gap_fraction", "pairwise_mean_gap_fraction"])
        w.writerows(gap_report)

    print("\n=== Mean gap fraction by method (answers 'was the alignment very gappy?') ===")
    import statistics
    mafft_gaps = [r[2] for r in gap_report if r[2] is not None]
    pairwise_gaps = [r[3] for r in gap_report if r[3] is not None]
    if mafft_gaps:
        print(f"MAFFT:    mean gap fraction = {statistics.mean(mafft_gaps):.4f}")
    if pairwise_gaps:
        print(f"Pairwise: mean gap fraction = {statistics.mean(pairwise_gaps):.4f}")

    # ── Plots: both methods, side by side ──
    plot_spectrum(mafft_counts, "Bacterial gene spectrum (MAFFT alignment)",
                   output_dir / "spectrum_mafft.png")
    plot_spectrum(pairwise_counts, "Bacterial gene spectrum (pairwise alignment)",
                   output_dir / "spectrum_pairwise.png")
    print(f"\n[INFO] Wrote spectrum_mafft.png and spectrum_pairwise.png to {output_dir}")
    print("[INFO] Compare these two directly -- this is the answer to "
          "'does MAFFT vs. pairwise give different results'.")


if __name__ == "__main__":
    main()
