#!/usr/bin/env python3
"""
generate_batch_evo2.py

Wrapper around Evo 2 (Arc Institute; Brixi et al.) inference that produces
N samples (default 100) per input sequence, written in the SAME on-disk
convention as generate_batch_carbon.py / generate_batch_generator.py so
that Evo 2 drops straight into the existing PRISM genmodel-bias pipeline
as a third model:

    python prism_genmodel_analysis.py \
        --model_dir carbon:/path/carbon_output \
        --model_dir generator:/path/generator_output \
        --model_dir evo2:/path/evo2_output \
        ...

    python analyze_bacterial_generation_matched.py \
        --carbon_dir ... --generator_dir ... [--evo2_dir /path/evo2_output]

Each input prompt produces two files in --output_dir:
    <record_id>_generated.fasta               (the N generated continuations)
    <record_id>_expected_continuation.fasta   (the true biological tail)

WHY EVO 2 IS THE INTERESTING THIRD MODEL (failure-mode / robustness angle)
--------------------------------------------------------------------------
Carbon and GENERator both tokenise DNA as non-overlapping 6-mers. Evo 2 is
a **single-nucleotide, byte-level** model (StripedHyena 2). That difference
is exactly what makes it the right control for the repetition-collapse and
alignment-miscounting failure modes reported in the PRISM paper: a 6-mer
tokenizer can only emit homopolymer runs in multiples of its k-mer stride,
whereas Evo 2 emits base-by-base. Comparing the collapse profile and the
resulting COSMIC substitution spectrum across all three architectures tests
whether those artifacts are a property of the *tokenizer/decoder* or of
genomic language models in general.

KEY EVO 2 SPECIFICS (differ from Carbon)
----------------------------------------
* NO "<dna>" tag and NO multiple-of-6 padding -- Evo 2 consumes the raw
  ACGT string directly (single-nucleotide vocabulary).
* Native interface is the `evo2` package: `from evo2 import Evo2`;
  `model.generate(prompt_seqs=[...], n_tokens=..., temperature=..., top_k=..., top_p=...)`.
  The returned object exposes `.sequences` (list of generated CONTINUATIONS,
  prompt not included) and per-sequence scores.
* Hardware: `evo2_7b` / `evo2_7b_base` / `evo2_7b_262k` run in **bfloat16 on
  any supported CUDA GPU** (no Transformer Engine needed). `evo2_1b_base`,
  `evo2_20b`, `evo2_40b*` require **FP8 via Transformer Engine on a Hopper
  (H100) GPU**. This script detects that and degrades gracefully (see below).

GRACEFUL DEGRADATION (this is the whole point of --model_name auto-fallback)
---------------------------------------------------------------------------
Load order, controlled by --load_strategy (default: auto):
  1. Native `evo2` package with the requested model.
     - If the requested model needs FP8 but no Transformer Engine / Hopper
       GPU is present, and --allow_downgrade is set (default), transparently
       switch to `evo2_7b` (bf16) and warn loudly.
  2. If the `evo2` package is not importable, and --allow_hf_fallback is set,
     try the HuggingFace path (arcinstitute/evo2_7b via AutoModelForCausalLM,
     trust_remote_code=True). This path is EXPERIMENTAL -- Arc does not
     officially document it -- and is clearly labelled in the output.
  3. If nothing works, raise a single actionable error with install
     instructions instead of a cryptic ImportError.

Usage
-----
python generate_batch_evo2.py \
    --input_fasta prompts.fasta \
    --model_name evo2_7b \
    --num_samples 50 \
    --batch_size 5 \
    --max_new_tokens 500 \
    --temperature 1.0 \
    --top_k 4 \
    --prefix_frac 0.2 \
    --output_dir /content/drive/MyDrive/PRISM/genmodel_bias/evo2_output
"""

import argparse
import math
import os
import re
import sys
import warnings

from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

# ── Evo 2 model catalogue ───────────────────────────────────────────────
# Which checkpoints can run in plain bfloat16 (no Transformer Engine / no
# Hopper requirement) vs. which need FP8. Kept here so the script can make a
# hardware-aware fallback decision *before* it tries to instantiate a model
# that will OOM or crash on a non-Hopper GPU.
BF16_OK_MODELS = {"evo2_7b", "evo2_7b_base", "evo2_7b_262k", "evo2_7b_microviridae"}
FP8_REQUIRED_MODELS = {"evo2_1b_base", "evo2_20b", "evo2_40b", "evo2_40b_base"}
DEFAULT_BF16_MODEL = "evo2_7b"


# ── CLI ────────────────────────────────────────────────────────────────
def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Generate N samples per FASTA sequence with Evo 2.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--input_fasta", type=str, required=True,
                   help="FASTA file of prompt sequences (DNA CDS).")
    p.add_argument("--model_name", type=str, default="evo2_7b",
                   help="Evo 2 checkpoint. evo2_7b/evo2_7b_base/evo2_7b_262k "
                        "run in bf16 on any GPU; evo2_1b_base/evo2_20b/"
                        "evo2_40b* need FP8 on an H100.")
    p.add_argument("--local_path", type=str, default=None,
                   help="Local path to model weights (passed to Evo2(local_path=...)); "
                        "skips the HuggingFace download.")
    p.add_argument("--num_samples", type=int, default=100,
                   help="Independent samples per input sequence.")
    p.add_argument("--batch_size", type=int, default=10,
                   help="Samples generated in parallel per batch. Lower if OOM.")
    p.add_argument("--max_new_tokens", type=int, default=500,
                   help="Nucleotides to generate per sample (Evo 2 is 1 token = 1 nt).")
    p.add_argument("--temperature", type=float, default=1.0,
                   help="Sampling temperature.")
    p.add_argument("--top_k", type=int, default=4,
                   help="Top-k sampling.")
    p.add_argument("--top_p", type=float, default=1.0,
                   help="Top-p (nucleus) sampling (1.0 = disabled).")
    p.add_argument("--prefix_frac", type=float, default=0.2,
                   help="Fraction of each gene used as the prompt. The remaining "
                        "(1 - prefix_frac) is saved as the true reference "
                        "continuation for downstream alignment. 1.0 = whole-CDS prompt "
                        "(no reference continuation written).")
    p.add_argument("--output_dir", type=str, default=None,
                   help="Directory for per-prompt FASTA files. "
                        "Defaults to generated_<model>_<input>_<tokens>t_<n>s/.")
    p.add_argument("--use_kernels", action="store_true",
                   help="Enable Evo 2's optional Triton conv kernels (needs vtx>=1.1.0).")
    # ── graceful-degradation controls ──
    p.add_argument("--load_strategy", choices=["auto", "native", "hf"], default="auto",
                   help="auto = native evo2 then HF fallback; native = evo2 only "
                        "(error if unavailable); hf = HuggingFace path only.")
    p.add_argument("--allow_downgrade", action="store_true", default=True,
                   help="If the requested model needs FP8 but no Hopper/TE is present, "
                        "fall back to evo2_7b (bf16) instead of failing.")
    p.add_argument("--no_downgrade", dest="allow_downgrade", action="store_false",
                   help="Disable the FP8->bf16 model downgrade; error instead.")
    p.add_argument("--allow_hf_fallback", action="store_true", default=True,
                   help="If the native evo2 package is unimportable, try "
                        "arcinstitute/evo2_7b via transformers (EXPERIMENTAL).")
    return p.parse_args(argv)


# ── hardware probing ────────────────────────────────────────────────────
def probe_hardware():
    """Return (has_cuda, device_name, compute_capability_tuple_or_None,
    has_transformer_engine, supports_fp8)."""
    try:
        import torch
    except ImportError:
        return False, "cpu", None, False, False

    has_cuda = torch.cuda.is_available()
    if not has_cuda:
        return False, "cpu", None, False, False

    name = torch.cuda.get_device_name(0)
    cc = torch.cuda.get_device_capability(0)  # e.g. (9, 0) Hopper, (8, 0) A100, (8, 9) Ada
    try:
        import transformer_engine  # noqa: F401
        has_te = True
    except Exception:
        has_te = False
    # FP8 needs compute capability >= 8.9 (Ada / Hopper) AND transformer_engine.
    supports_fp8 = has_te and (cc[0] > 8 or (cc[0] == 8 and cc[1] >= 9))
    return has_cuda, name, cc, has_te, supports_fp8


def resolve_model_name(requested: str, supports_fp8: bool, allow_downgrade: bool) -> str:
    """Pick a runnable checkpoint given the hardware, warning on any change."""
    if requested in BF16_OK_MODELS:
        return requested
    if requested in FP8_REQUIRED_MODELS and not supports_fp8:
        if allow_downgrade:
            warnings.warn(
                f"[evo2] Requested '{requested}' needs FP8 (Transformer Engine + "
                f"Hopper/Ada GPU), which is unavailable. Downgrading to "
                f"'{DEFAULT_BF16_MODEL}' (bf16). Pass --no_downgrade to error instead.",
                RuntimeWarning,
            )
            return DEFAULT_BF16_MODEL
        raise RuntimeError(
            f"[evo2] '{requested}' requires FP8 (Transformer Engine on an H100/Ada GPU) "
            f"but this machine does not support it. Either run on an H100, "
            f"`pip install transformer_engine`, choose a 7B model, or drop --no_downgrade "
            f"to auto-fall-back to {DEFAULT_BF16_MODEL}."
        )
    # Unknown name (custom/local checkpoint): trust the user.
    return requested


# ── model loading with graceful degradation ─────────────────────────────
class Evo2Runner:
    """Uniform generate() interface over either the native evo2 package or
    the experimental HuggingFace path, so the main loop doesn't branch."""

    def __init__(self, generate_fn, backend, model_name):
        self._generate_fn = generate_fn
        self.backend = backend            # "native" | "hf"
        self.model_name = model_name

    def generate(self, prompt_seqs, n_tokens, temperature, top_k, top_p):
        return self._generate_fn(prompt_seqs, n_tokens, temperature, top_k, top_p)


def _load_native(model_name, local_path, use_kernels, supports_fp8, allow_downgrade):
    """Load via `from evo2 import Evo2`. Returns an Evo2Runner or raises."""
    from evo2 import Evo2  # ImportError bubbles up to caller for fallback

    resolved = resolve_model_name(model_name, supports_fp8, allow_downgrade)
    print(f"[evo2] Loading native evo2 model: {resolved}"
          + (f" (local_path={local_path})" if local_path else ""))
    model = Evo2(resolved, local_path=local_path, use_kernels=use_kernels)

    def _gen(prompt_seqs, n_tokens, temperature, top_k, top_p):
        out = model.generate(
            prompt_seqs=list(prompt_seqs),
            n_tokens=n_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            verbose=0,
        )
        # Evo 2 versions differ: newer returns an object with `.sequences`,
        # older returns a (sequences, scores) tuple. Handle both.
        if hasattr(out, "sequences"):
            return list(out.sequences)
        if isinstance(out, tuple):
            return list(out[0])
        return list(out)

    return Evo2Runner(_gen, "native", resolved)


def _load_hf(model_name):
    """EXPERIMENTAL HuggingFace fallback (arcinstitute/evo2_7b). Returns an
    Evo2Runner or raises. Only the 7B checkpoint is attempted here because it
    is the one that runs without Transformer Engine."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    hf_id = "arcinstitute/evo2_7b"
    warnings.warn(
        f"[evo2] Native evo2 package unavailable; trying EXPERIMENTAL HuggingFace "
        f"path '{hf_id}'. This is not an Arc-documented interface -- verify outputs.",
        RuntimeWarning,
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    tok = AutoTokenizer.from_pretrained(hf_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        hf_id, trust_remote_code=True, torch_dtype=dtype, device_map="auto",
    )
    model.eval()

    @torch.inference_mode()
    def _gen(prompt_seqs, n_tokens, temperature, top_k, top_p):
        tok.padding_side = "left"
        if tok.pad_token_id is None and tok.eos_token is not None:
            tok.pad_token = tok.eos_token
        enc = tok(list(prompt_seqs), return_tensors="pt", padding=True,
                  add_special_tokens=False)
        enc = {k: v.to(model.device) for k, v in enc.items()}
        out = model.generate(
            **enc, max_new_tokens=n_tokens, do_sample=True,
            temperature=temperature, top_k=top_k, top_p=top_p,
            pad_token_id=tok.pad_token_id,
        )
        prompt_len = enc["input_ids"].shape[1]
        gen_only = out[:, prompt_len:]
        return tok.batch_decode(gen_only.cpu(), skip_special_tokens=True)

    return Evo2Runner(_gen, "hf", hf_id)


def load_runner(args) -> Evo2Runner:
    has_cuda, name, cc, has_te, supports_fp8 = probe_hardware()
    print(f"[evo2] Hardware: cuda={has_cuda} device={name} "
          f"compute_capability={cc} transformer_engine={has_te} fp8_ok={supports_fp8}")
    if not has_cuda:
        warnings.warn("[evo2] No CUDA GPU detected. Evo 2 is impractical on CPU; "
                      "this will be extremely slow or fail.", RuntimeWarning)

    strat = args.load_strategy
    errors = []

    if strat in ("auto", "native"):
        try:
            return _load_native(args.model_name, args.local_path, args.use_kernels,
                                 supports_fp8, args.allow_downgrade)
        except ImportError as e:
            errors.append(f"native evo2 import failed: {e}")
        except Exception as e:
            errors.append(f"native evo2 load failed: {e}")
            if strat == "native":
                raise

    if strat in ("auto", "hf") and (strat == "hf" or args.allow_hf_fallback):
        try:
            return _load_hf(args.model_name)
        except Exception as e:
            errors.append(f"HF fallback failed: {e}")

    raise RuntimeError(
        "[evo2] Could not load Evo 2 through any strategy.\n  - "
        + "\n  - ".join(errors)
        + "\n\nTo install the native package (recommended):\n"
        "    pip install flash-attn==2.8.0.post2 --no-build-isolation\n"
        "    pip install evo2\n"
        "  (7B models run in bf16 on any CUDA GPU. For 1B/40B you also need\n"
        "   Transformer Engine + an H100: conda install -c conda-forge "
        "transformer-engine-torch)."
    )


# ── FASTA helpers (mirrors generate_batch_carbon.py, minus 6-mer padding) ─
def sanitise_filename(name: str) -> str:
    return re.sub(r"[^\w\-.]", "_", name)


def clean_decoded(text: str) -> str:
    """Keep only ACGTN; strip any stray tokenizer tags/whitespace. Evo 2 has
    no <dna> tag, but this guards against the HF path emitting specials."""
    text = re.sub(r"<[^>]+>", "", text)
    return re.sub(r"[^ACGTNacgtn]", "", text).upper()


def read_prompts(fasta_path: str, prefix_frac: float):
    """Return (record_id, description, prompt_prefix, true_suffix) tuples.

    Unlike the Carbon reader, NO multiple-of-6 padding is applied: Evo 2's
    single-nucleotide tokenizer consumes the raw prefix directly."""
    prompts = []
    for record in SeqIO.parse(fasta_path, "fasta"):
        seq_str = str(record.seq).upper().replace(" ", "").replace("\t", "")
        cut = int(len(seq_str) * prefix_frac)
        prefix_raw, true_suffix = seq_str[:cut], seq_str[cut:]
        prompts.append((record.id, record.description, prefix_raw, true_suffix))
    return prompts


def strip_prompt_echo(gen_seq: str, prompt: str) -> str:
    """Defensive: if a backend returns prompt+continuation instead of the
    continuation alone, drop the echoed prefix so output matches Carbon's
    (continuation-only) convention."""
    if prompt and gen_seq.upper().startswith(prompt.upper()):
        return gen_seq[len(prompt):]
    return gen_seq


def set_seed(seed: int):
    import random
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


# ── main loop ──────────────────────────────────────────────────────────
def main(argv=None):
    args = parse_args(argv)
    runner = load_runner(args)
    prompts = read_prompts(args.input_fasta, prefix_frac=args.prefix_frac)

    print(f"[evo2] Backend: {runner.backend}  |  model: {runner.model_name}")
    print(f"[evo2] Loaded {len(prompts)} prompt sequence(s)")
    print(f"[evo2] Using first {args.prefix_frac:.0%} of each gene as the prompt; "
          f"remaining {1 - args.prefix_frac:.0%} saved as reference continuation.")
    print(f"[evo2] Generating {args.num_samples} samples each "
          f"(batch_size={args.batch_size}, n_tokens={args.max_new_tokens})")

    if args.output_dir is None:
        model_tag = runner.model_name.rstrip("/").split("/")[-1]
        input_tag = args.input_fasta.rstrip("/").split("/")[-1].rsplit(".", 1)[0]
        args.output_dir = (
            f"generated_{model_tag}_{input_tag}"
            f"_{args.max_new_tokens}t_{args.num_samples}s"
        )
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"[evo2] Output directory: {args.output_dir}")

    num_batches = math.ceil(args.num_samples / args.batch_size)

    for prompt_idx, (rec_id, rec_desc, prompt_seq, true_suffix) in enumerate(prompts):
        print(f"\n[evo2] Prompt {prompt_idx + 1}/{len(prompts)}: {rec_id}  "
              f"(prompt len={len(prompt_seq)} bp, "
              f"reference continuation len={len(true_suffix)} bp)")

        if not prompt_seq:
            print(f"       [warn] empty prompt for {rec_id} (prefix_frac too small?), skipping")
            continue

        prompt_records = []
        sample_counter = 0

        for batch_idx in range(num_batches):
            current_batch_size = min(args.batch_size, args.num_samples - sample_counter)
            batch_start_seed = sample_counter
            set_seed(batch_start_seed)

            batch_prompts = [prompt_seq] * current_batch_size
            decoded = runner.generate(
                batch_prompts,
                n_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
            )

            for gen_seq in decoded:
                seed = sample_counter
                sample_counter += 1
                sample_id = f"{rec_id}|sample_{sample_counter:03d}|seed_{seed}"
                cont = strip_prompt_echo(gen_seq, prompt_seq)
                cleaned = clean_decoded(cont)
                prompt_records.append(SeqRecord(
                    Seq(cleaned), id=sample_id,
                    description=f"generated by evo2:{runner.model_name} from {rec_desc}",
                ))

            print(f"       batch {batch_idx + 1}/{num_batches} done  "
                  f"(seeds {batch_start_seed}-{sample_counter - 1}, "
                  f"{sample_counter}/{args.num_samples} samples)")

        safe_name = sanitise_filename(rec_id)
        out_path = os.path.join(args.output_dir, f"{safe_name}_generated.fasta")
        SeqIO.write(prompt_records, out_path, "fasta")
        print(f"[evo2] Wrote {len(prompt_records)} record(s) -> {out_path}")

        if true_suffix:
            ref_path = os.path.join(args.output_dir, f"{safe_name}_expected_continuation.fasta")
            ref_record = SeqRecord(
                Seq(true_suffix), id=rec_id,
                description=f"true continuation ({1 - args.prefix_frac:.0%} of {rec_desc})",
            )
            SeqIO.write([ref_record], ref_path, "fasta")
            print(f"[evo2] Wrote reference continuation -> {ref_path}")

    print(f"\n[evo2] Done. {len(prompts)} prompt(s) processed in {args.output_dir}/")


if __name__ == "__main__":
    main()
