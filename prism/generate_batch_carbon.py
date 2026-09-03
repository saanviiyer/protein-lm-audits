#!/usr/bin/env python3
"""
Wrapper around Carbon (HuggingFaceBio/Carbon-8B) inference that produces
N samples (default 100) per input sequence. Uses BioPython for FASTA
handling.

Each sample gets a deterministic seed (0 … num_samples-1) so any
individual sample can be reproduced by re-running with the same seed
range and batch layout.

Each input prompt produces its own output FASTA file written to
--output_dir, named <record_id>_generated.fasta.

Key Carbon-specific notes
-------------------------
* Carbon uses a hybrid tokenizer: BPE for English + non-overlapping
  6-mer for DNA, switched by a literal "<dna>" tag. We therefore prepend
  "<dna>" to each prompt rather than the BOS token.
* DNA prompts must be a multiple of 6 bp (we pad/truncate as before).
* Architecture is LlamaForCausalLM; trust_remote_code=True is used to
  match the official model card snippet (and is required for the `fns`
  branch).
* Native context = 32,768 tokens (≈ 196 kbp). max_new_tokens default is
  raised accordingly (1334 6-mers ≈ 8 kb is fine but Carbon can do far
  more — feel free to bump it).
* Saved FASTA records contain ONLY the generated continuation — the
  input prompt is stripped before decoding.

Usage
-----
python generate_batch_carbon.py \
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

# ── Carbon specifics ───────────────────────────────────────────────────
DNA_TAG = "<dna>"           # switches Carbon's tokenizer into 6-mer mode
DNA_KMER = 6                # Carbon uses non-overlapping 6-mers


# ── CLI ────────────────────────────────────────────────────────────────
def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Generate N samples per FASTA sequence with Carbon.",
    )
    p.add_argument(
        "--input_fasta", type=str, required=True,
        help="Path to a FASTA file containing prompt sequences.",
    )
    p.add_argument(
        "--model_name", type=str,
        default="HuggingFaceBio/Carbon-8B",
        help="HuggingFace model name or local path "
             "(default: HuggingFaceBio/Carbon-8B). "
             "You can also pass HuggingFaceBio/Carbon-3B or Carbon-500M.",
    )
    p.add_argument(
        "--revision", type=str, default=None,
        help="Optional model revision/branch. Use 'fns' for base-pair "
             "level generation with Factorized Nucleotide Supervision.",
    )
    p.add_argument(
        "--num_samples", type=int, default=100,
        help="Number of independent samples per input sequence (default: 100).",
    )
    p.add_argument(
        "--batch_size", type=int, default=10,
        help="Samples generated in parallel per batch (default: 10). "
             "Lower this if you run out of VRAM (8B is heavy).",
    )
    p.add_argument(
        "--max_new_tokens", type=int, default=1334,
        help="6-mer tokens to generate per sample (~8 kb at default 1334). "
             "Carbon-8B natively supports up to 32,768 tokens of context.",
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
        "--top_p", type=float, default=1.0,
        help="Top-p (nucleus) sampling (default: 1.0 = disabled).",
    )
    p.add_argument(
        "--output_dir", type=str, default=None,
        help="Directory for per-prompt FASTA files. Defaults to "
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
    p.add_argument(
        "--strip_dna_tag", action="store_true",
        help="Strip the <dna> tag from decoded outputs before writing FASTA.",
    )
    return p.parse_args(argv)


# ── helpers ────────────────────────────────────────────────────────────
def load_model(model_name: str, revision: str | None = None):
    """Load tokenizer + model and return (tokenizer, model, device)."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Device : {device}")
    if device == "cuda":
        print(f"[INFO] GPU    : {torch.cuda.get_device_name(0)}")

    kwargs = dict(trust_remote_code=True)
    if revision:
        kwargs["revision"] = revision
        print(f"[INFO] Revision: {revision}")

    tokenizer = AutoTokenizer.from_pretrained(model_name, **kwargs)

    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=dtype,
        device_map="auto",          # <- let accelerate place it
        low_cpu_mem_usage=True,          # note: Carbon card uses `dtype=` not `torch_dtype=`
        **kwargs,
    )
    model.eval()
    device = next(model.parameters()).device
    return tokenizer, model, device


def set_seed(seed: int):
    """Set seed for Python, NumPy, and PyTorch for reproducibility."""
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


def adjust_sequence(seq_str: str, mode: str = "pad", multiple: int = DNA_KMER) -> str:
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
    Read a FASTA file and return a list of
    (record_id, description, adjusted_prefix, true_suffix) tuples.

    Only the first `prefix_frac` of each (uppercased, whitespace-stripped)
    sequence is used as the prompt (adjusted to a multiple of 6 bp for
    Carbon's tokenizer). The remaining, UNMODIFIED tail of the true
    sequence is returned as `true_suffix` -- the real biological
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


def clean_decoded(text: str, strip_tag: bool) -> str:
    """Optionally strip the <dna> tag and any non-ACGTN characters."""
    if strip_tag:
        text = text.replace(DNA_TAG, "")
        # Carbon may also emit a closing </dna> in some checkpoints/branches
        text = text.replace("</dna>", "")
    return text


@torch.inference_mode()
def generate_batch(
    sequences: list[str],
    tokenizer,
    model,
    device: str,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    top_p: float,
) -> list[str]:
    """
    Tokenise a list of <dna>-prepended DNA strings, run model.generate,
    and return ONLY the newly generated continuations (prompt stripped).
    """
    tokenizer.padding_side = "left"
    # Carbon tokenizer may not have a pad token defined — fall back to eos.
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Use the model's max_position_embeddings as the truncation cap (32k for Carbon-8B).
    max_len = getattr(model.config, "max_position_embeddings", 32768)

    inputs = tokenizer(
        sequences,
        add_special_tokens=False,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_len,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        pad_token_id=tokenizer.pad_token_id,
    )

    # ── Strip the prompt: keep only newly generated tokens ─────────────
    # With left-padding, every row in `inputs["input_ids"]` has the same
    # length, so we can slice uniformly by that length.
    prompt_len = inputs["input_ids"].shape[1]
    gen_only = outputs[:, prompt_len:]

    return tokenizer.batch_decode(gen_only.cpu(), skip_special_tokens=True)


# ── main loop ──────────────────────────────────────────────────────────
def main(argv=None):
    args = parse_args(argv)
    tokenizer, model, device = load_model(args.model_name, revision=args.revision)
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
        if args.revision:
            model_tag += f"_{args.revision}"
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
              f"(prompt len={len(seq_str)} bp, reference continuation len={len(true_suffix)} bp)")

        prompt_records: list[SeqRecord] = []
        sample_counter = 0

        for batch_idx in range(num_batches):
            current_batch_size = min(
                args.batch_size,
                args.num_samples - sample_counter,
            )

            # ── Per-batch seed ────────────────────────────────────────
            batch_start_seed = sample_counter
            set_seed(batch_start_seed)

            # Replicate the <dna>-tagged prompt for the whole batch.
            # NOTE: Carbon uses the literal "<dna>" tag instead of a BOS token
            # to switch its hybrid tokenizer into 6-mer DNA mode.
            batch_seqs = [DNA_TAG + seq_str] * current_batch_size

            decoded = generate_batch(
                batch_seqs,
                tokenizer,
                model,
                device,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
            )

            for gen_seq in decoded:
                seed = sample_counter
                sample_counter += 1
                sample_id = f"{rec_id}|sample_{sample_counter:03d}|seed_{seed}"
                cleaned = clean_decoded(gen_seq, strip_tag=args.strip_dna_tag)
                record = SeqRecord(
                    Seq(cleaned),
                    id=sample_id,
                    description=f"generated from {rec_desc}",
                )
                prompt_records.append(record)

            print(f"       batch {batch_idx + 1}/{num_batches} done  "
                  f"(seeds {batch_start_seed}-{sample_counter - 1}, "
                  f"{sample_counter}/{args.num_samples} samples)")

        # ── Write per-prompt FASTA ────────────────────────────────────
        safe_name = sanitise_filename(rec_id)
        out_path = os.path.join(args.output_dir, f"{safe_name}_generated.fasta")
        SeqIO.write(prompt_records, out_path, "fasta")
        print(f"[INFO] Wrote {len(prompt_records)} record(s) → {out_path}")

        # ── Write the true reference continuation (if any) ────────────
        if true_suffix:
            ref_path = os.path.join(args.output_dir, f"{safe_name}_expected_continuation.fasta")
            ref_record = SeqRecord(
                Seq(true_suffix), id=rec_id,
                description=f"true continuation ({1 - args.prefix_frac:.0%} of {rec_desc})",
            )
            SeqIO.write([ref_record], ref_path, "fasta")
            print(f"[INFO] Wrote reference continuation → {ref_path}")


    print(f"\n[INFO] Done. {len(prompts)} file(s) in {args.output_dir}/")


if __name__ == "__main__":
    main()