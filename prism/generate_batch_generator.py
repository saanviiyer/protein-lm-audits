#!/usr/bin/env python3
"""
Wrapper around GENERator inference that produces N samples (default 100)
per input sequence.  Uses BioPython for FASTA handling.

Each sample gets a deterministic seed (0 … num_samples-1) so that any
individual sample can be reproduced by re-running with the same seed
range and batch layout.

Each input prompt produces its own output FASTA file written to
--output_dir, named <record_id>_generated.fasta.

Usage
-----
python generate_samples.py \
    --input_fasta prompts.fasta \
    --num_samples 100 \
    --batch_size 10 \
    --max_new_tokens 1334 \
    --temperature 1.0 \
    --top_k 4 \
    --output_dir generated_output
"""

import argparse
import math
import os
import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio.SeqIO.FastaIO import FastaWriter  # FIX: use writer with wrap=0


# ── CLI ────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(
        description="Generate N samples per FASTA sequence with GENERator.",
    )
    p.add_argument(
        "--input_fasta", type=str, required=True,
        help="Path to a FASTA file containing prompt sequences.",
    )
    p.add_argument(
        "--model_name", type=str,
        default="GenerTeam/GENERator-v2-prokaryote-1.2b-base",
        help="HuggingFace model name or local path.",
    )
    p.add_argument(
        "--num_samples", type=int, default=100,
        help="Number of independent samples to generate per input sequence (default: 100).",
    )
    p.add_argument(
        "--batch_size", type=int, default=10,
        help="How many samples to generate in parallel per batch (default: 10). "
             "Lower this if you run out of VRAM.",
    )
    p.add_argument(
        "--max_new_tokens", type=int, default=1334,
        help="6-mer tokens to generate per sample (~8 kb at default 1334).",
    )
    p.add_argument(
        "--temperature", type=float, default=1.0,
        help="Sampling temperature (default: 1.0).",
    )
    p.add_argument(
        "--top_k", type=int, default=4,
        help="Top-k sampling (default: 4).",
    )
    p.add_argument(
        "--output_dir", type=str, default=None,
        help="Directory for per-prompt FASTA files.  Defaults to "
             "generated_<model>_<input>_<tokens>t_<n>s/",
    )
    p.add_argument(
        "--pad_or_truncate", type=str, choices=["pad", "truncate"],
        default="pad",
        help="Adjust sequences whose length is not a multiple of 6 "
             "by left-padding with 'A' or left-truncating (default: pad).",
    )
    p.add_argument(
        "--prefix_frac", type=float, default=0.2,
        help="Fraction of each gene used as the prompt (default: 0.2, "
             "i.e. the first 20%% of the sequence). The REMAINING "
             "(1 - prefix_frac) of the true sequence is saved as a "
             "reference/expected-continuation FASTA, so generated samples "
             "can be aligned against real biology instead of only each "
             "other. Set to 1.0 to reproduce the old whole-sequence-prompt "
             "behavior (no reference continuation will be available).",
    )
    # FIX: optionally keep raw, token-containing output for debugging
    p.add_argument(
        "--keep_raw", action="store_true",
        help="Also write an unfiltered <name>_raw.fasta file containing the "
             "model output before special-token stripping (debug only).",
    )
    return p.parse_args()


# ── helpers ────────────────────────────────────────────────────────────
def load_model(model_name: str):
    """Load tokenizer + model and return (tokenizer, model, device)."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Device : {device}")
    if device == "cuda":
        print(f"[INFO] GPU    : {torch.cuda.get_device_name(0)}")

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype)
    model.to(device).eval()
    return tokenizer, model, device


def set_seed(seed: int):
    """Set seed for Python, NumPy (if imported), and PyTorch for reproducibility."""
    import random
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sanitise_filename(name: str) -> str:
    """Replace characters that are unsafe in filenames."""
    return re.sub(r'[^\w\-.]', '_', name)


# FIX: strip GENERator's biological-feature vocab tokens
# (<plt>, <mit>, <arc>, <prt>, <inv>, <tRNA>, <sp1>, <eog>, <mask>, ...)
# and any stray non-DNA characters, BEFORE the sequence is written to a
# FASTA file. Two reasons:
#   1. These tokens are not nucleotides and corrupt every downstream tool
#      (translation, BLAST, ESMFold, CAI, ...).
#   2. When Biopython hard-wraps a FASTA line at 60 chars, a wrap that
#      lands between '<' and '>' of one of these tokens leaves the '>' at
#      column 0 of the next line, which FASTA parsers then interpret as a
#      brand-new record header. That is what produced the 147 spurious
#      seq_id rows in the retrieval CSV.
_TOKEN_RE = re.compile(r"<[^>]+>")


def clean_generation(seq: str) -> str:
    """Remove model vocabulary tokens and any non-DNA characters."""
    seq = _TOKEN_RE.sub("", seq)
    seq = re.sub(r"[^ACGTNacgtn]", "", seq)
    return seq.upper()


def adjust_sequence(seq_str: str, mode: str = "pad", multiple: int = 6) -> str:
    """Left-pad or left-truncate so len is a multiple of `multiple`."""
    remainder = len(seq_str) % multiple
    if remainder == 0:
        return seq_str
    if mode == "pad":
        return "A" * (multiple - remainder) + seq_str
    else:                         # truncate
        return seq_str[remainder:]


def read_prompts(fasta_path: str, mode: str = "pad", prefix_frac: float = 0.2):
    """
    Read a FASTA file with BioPython and return a list of
    (record_id, description, adjusted_prefix, true_suffix) tuples.

    Only the first `prefix_frac` of each sequence is used as the prompt
    (adjusted to a multiple of 6 bp). The remaining, UNMODIFIED tail of
    the true sequence is returned as `true_suffix` -- the real biological
    continuation, which the analysis script can align generated samples
    against instead of only comparing samples to each other.
    """
    prompts = []
    for record in SeqIO.parse(fasta_path, "fasta"):
        seq_str = str(record.seq).upper().replace(" ", "").replace("\t", "")
        cut = int(len(seq_str) * prefix_frac)
        prefix_raw, true_suffix = seq_str[:cut], seq_str[cut:]
        prefix_adjusted = adjust_sequence(prefix_raw, mode=mode)
        prompts.append((record.id, record.description, prefix_adjusted, true_suffix))
    return prompts


# FIX: write FASTA with wrap=0 so sequences live on a single line. Even
# after token stripping this is a safe default: it removes any future
# possibility of a stray '<' / '>' character being split across a line
# boundary and being mis-parsed as a record header.
def write_fasta_unwrapped(records, out_path):
    with open(out_path, "w") as fh:
        writer = FastaWriter(fh, wrap=0)
        writer.write_file(records)


# ── generation ─────────────────────────────────────────────────────────
@torch.inference_mode()
def generate_batch(
    sequences: list[str],
    tokenizer,
    model,
    device: str,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
) -> list[str]:
    """
    Tokenise a list of BOS-prepended DNA strings, run model.generate,
    and return decoded sequences containing ONLY the newly generated
    continuation (the prompt tokens are stripped).
    """
    tokenizer.padding_side = "left"
    inputs = tokenizer(
        sequences,
        add_special_tokens=False,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=model.config.max_position_embeddings,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        top_k=top_k,
    )

    # Strip the prompt: with left-padding, the prompt occupies the first
    # `input_length` columns of every row, so the continuation starts at
    # that offset for every sample in the batch.
    input_length = inputs["input_ids"].shape[1]
    continuation_ids = outputs[:, input_length:].cpu()
    # NOTE: skip_special_tokens=True only removes tokens flagged as
    # *special* in the tokenizer (BOS/EOS/PAD/UNK). The biological
    # feature tokens <plt>, <mit>, <arc>, <prt>, <inv>, <tRNA>, <sp1>,
    # <eog>, <mask> are regular vocab entries and therefore survive
    # decoding. They are removed downstream by clean_generation().
    return tokenizer.batch_decode(continuation_ids, skip_special_tokens=True)


# ── main loop ──────────────────────────────────────────────────────────
def main():
    args = parse_args()
    tokenizer, model, device = load_model(args.model_name)
    prompts = read_prompts(args.input_fasta, mode=args.pad_or_truncate, prefix_frac=args.prefix_frac)

    print(f"[INFO] Loaded {len(prompts)} prompt sequence(s)")
    print(f"[INFO] Using first {args.prefix_frac:.0%} of each gene as the prompt; "
          f"the remaining {1 - args.prefix_frac:.0%} is saved as the true "
          f"reference continuation for downstream alignment.")
    print(f"[INFO] Generating {args.num_samples} samples each "
          f"(batch_size={args.batch_size})")
    print(f"[INFO] Seeds will range from 0 to {args.num_samples - 1}")

    # Build default output directory name
    if args.output_dir is None:
        model_tag = args.model_name.rstrip("/").split("/")[-1]
        input_tag = args.input_fasta.rstrip("/").split("/")[-1].rsplit(".", 1)[0]
        args.output_dir = (
            f"generated_{model_tag}_{input_tag}"
            f"_{args.max_new_tokens}t_{args.num_samples}s"
        )

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"[INFO] Output directory: {args.output_dir}")

    num_batches = math.ceil(args.num_samples / args.batch_size)

    for prompt_idx, (rec_id, rec_desc, seq_str, true_suffix) in enumerate(prompts):
        print(f"\n[INFO] Prompt {prompt_idx + 1}/{len(prompts)}: {rec_id}  "
              f"(prompt len={len(seq_str)}, reference continuation len={len(true_suffix)})")

        prompt_records: list[SeqRecord] = []
        raw_records: list[SeqRecord] = []   # FIX: optional debug output
        sample_counter = 0
        dropped_chars_total = 0             # FIX: simple QC counter

        for batch_idx in range(num_batches):
            current_batch_size = min(
                args.batch_size,
                args.num_samples - sample_counter,
            )

            # ── Per-batch seed ────────────────────────────────────────
            # We set the seed to the index of the *first* sample in this
            # batch. Every sample within the batch is generated in one
            # model.generate call, so the RNG state advances naturally
            # for each element. Re-running with the same batch_size and
            # seed range reproduces identical output.
            batch_start_seed = sample_counter
            set_seed(batch_start_seed)

            # Replicate the prompt for the whole batch
            batch_seqs = [tokenizer.bos_token + seq_str] * current_batch_size

            decoded = generate_batch(
                batch_seqs,
                tokenizer,
                model,
                device,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
            )

            for i, gen_seq in enumerate(decoded):
                seed = sample_counter
                sample_counter += 1
                sample_id = f"{rec_id}|sample_{sample_counter:03d}|seed_{seed}"

                # FIX: strip model vocab tokens / non-DNA chars before
                # we ever hand the sequence to Biopython's FASTA writer.
                cleaned = clean_generation(gen_seq)
                dropped_chars_total += len(gen_seq) - len(cleaned)

                if not cleaned:
                    print(f"  [warn] {sample_id}: empty after cleaning, "
                          f"skipping")
                    continue

                record = SeqRecord(
                    Seq(cleaned),
                    id=sample_id,
                    description=f"generated from {rec_desc}",
                )
                prompt_records.append(record)

                if args.keep_raw:
                    raw_records.append(SeqRecord(
                        Seq(gen_seq),
                        id=sample_id,
                        description=f"RAW generated from {rec_desc}",
                    ))

            print(f"       batch {batch_idx + 1}/{num_batches} done  "
                  f"(seeds {batch_start_seed}-{sample_counter - 1}, "
                  f"{sample_counter}/{args.num_samples} samples)")

        # ── Write per-prompt FASTA ────────────────────────────────────
        safe_name = sanitise_filename(rec_id)
        out_path = os.path.join(args.output_dir, f"{safe_name}_generated.fasta")
        # FIX: write unwrapped FASTA
        write_fasta_unwrapped(prompt_records, out_path)
        print(f"[INFO] Wrote {len(prompt_records)} record(s) → {out_path}  "
              f"(stripped {dropped_chars_total} non-DNA char(s) total)")

        # ── Write the true reference continuation (if any) ────────────
        if true_suffix:
            ref_path = os.path.join(args.output_dir, f"{safe_name}_expected_continuation.fasta")
            ref_record = SeqRecord(
                Seq(true_suffix), id=rec_id,
                description=f"true continuation ({1 - args.prefix_frac:.0%} of {rec_desc})",
            )
            write_fasta_unwrapped([ref_record], ref_path)
            print(f"[INFO] Wrote reference continuation → {ref_path}")

        if args.keep_raw and raw_records:
            raw_path = os.path.join(args.output_dir, f"{safe_name}_raw.fasta")
            write_fasta_unwrapped(raw_records, raw_path)
            print(f"[INFO] Wrote raw debug FASTA → {raw_path}")

    print(f"\n[INFO] Done. {len(prompts)} file(s) in {args.output_dir}/")


if __name__ == "__main__":
    main()