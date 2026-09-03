# # # #!/usr/bin/env python3
# # # """
# # # prism_genmodel_analysis.py

# # # Tests whether generative DNA models (Carbon, GENERator, ...) have
# # # implicitly learned COSMIC mutational-signature biases from training
# # # data, following the PRISM pipeline's conventions (ESM-2 8M embeddings,
# # # Pfam centroid library, per-protein analysis).

# # # This is the step that runs AFTER generate_batch_carbon.py /
# # # generate_batch_generator.py have produced <output_dir>/<record_id>_generated.fasta
# # # files for each model.

# # # WHY PAIRWISE-VS-PAIRWISE, NOT VS-WILD-TYPE
# # # -------------------------------------------
# # # generate_batch_carbon.py / generate_batch_generator.py feed the entire
# # # wild-type sequence as the prompt and return a *novel continuation* --
# # # there is no ground-truth continuation of the real gene to diff against.
# # # So instead of comparing each sample to the wild-type, this script
# # # aligns generated samples against EACH OTHER (all pairs, or a random
# # # subsample of pairs if N is large) and tallies where they diverge. That
# # # divergence pattern -- aggregated over many pairs -- is treated as the
# # # model's implicit mutational spectrum for that protein.

# # # PIPELINE (per protein, per model)
# # # ----------------------------------
# # # 1. Load the N generated samples from <model_dir>/<record_id>_generated.fasta
# # # 2. Pairwise-align a subsample of samples (Bio.Align, global alignment).
# # #    For every mismatched column that is NOT adjacent to a gap, extract
# # #    the trinucleotide context and orient it onto the pyrimidine
# # #    (C/T) reference strand per COSMIC convention.
# # # 3. Aggregate mismatches into a 96-channel substitution spectrum and
# # #    NNLS-decompose it against the COSMIC SBS signature matrix to get a
# # #    per-protein, per-model signature exposure vector.
# # # 4. Translate each sample (frame 0, truncate at first stop codon) and
# # #    embed with ESM-2 (facebook/esm2_t6_8M_UR50D, matching the PRISM
# # #    paper). Score:
# # #      - "creativity": number of distinct Pfam domain centroids the
# # #        sample cloud is nearest to (more spread = more "creative")
# # #      - mean pairwise cosine distance among sample embeddings
# # # 5. Compute mean pairwise Hamming distance (over aligned, ungapped
# # #    columns) as a sequence-level diversity metric independent of ESM-2.

# # # OUTPUTS (written to --output_dir)
# # # -----------------------------------
# # # signature_exposures.csv     one row per (model, protein), columns = SBS exposures
# # # creativity_diversity.csv    one row per (model, protein)
# # # substitution_spectra.csv    raw 96-channel counts, for debugging/inspection

# # # USAGE
# # # -----
# # # python prism_genmodel_analysis.py \
# # #     --model_dir carbon:/content/drive/MyDrive/PRISM/genmodel_bias/carbon_output \
# # #     --model_dir generator:/content/drive/MyDrive/PRISM/genmodel_bias/generator_output \
# # #     --cosmic_dir /content/drive/MyDrive/PRISM/cosmic_signatures \
# # #     --centroid_chunks_dir /content/drive/MyDrive/PRISM/centroids/centroid_chunks \
# # #     --output_dir /content/drive/MyDrive/PRISM/genmodel_bias/results
# # # """

# # # import argparse
# # # import itertools
# # # import os
# # # import random
# # # import re
# # # import sys

# # # import numpy as np
# # # import pandas as pd
# # # import torch
# # # from Bio import SeqIO, Align
# # # from Bio.Seq import Seq
# # # from scipy.optimize import nnls

# # # try:
# # #     from transformers import AutoTokenizer, AutoModel
# # # except ImportError:
# # #     AutoTokenizer = AutoModel = None


# # # COMPLEMENT = {"A": "T", "T": "A", "C": "G", "G": "C"}


# # # # ── CLI ──────────────────────────────────────────────────────────────
# # # def parse_args(argv=None):
# # #     p = argparse.ArgumentParser(
# # #         description="Test generative DNA models for implicit COSMIC signature bias.",
# # #     )
# # #     p.add_argument(
# # #         "--model_dir", action="append", required=True,
# # #         help="label:path pairs, e.g. carbon:/path/to/carbon_output. "
# # #              "Repeat this flag once per model (Carbon, GENERator, ...).",
# # #     )
# # #     p.add_argument(
# # #         "--cosmic_dir", type=str, required=True,
# # #         help="Root directory containing individual COSMIC signature "
# # #              "profile files (e.g. SBS2_PROFILE.txt, SBS17a_PROFILE.txt, ...). "
# # #              "Subfolders (e.g. Chemotherapy/, Clock-like/, Mismatch/) are "
# # #              "searched recursively.",
# # #     )
# # #     p.add_argument(
# # #         "--centroid_chunks_dir", type=str, required=True,
# # #         help="Directory containing the PRISM Pfam centroid library split "
# # #              "across multiple .npz chunk files. All chunks are loaded and "
# # #              "concatenated into one (M x 320) matrix.",
# # #     )
# # #     p.add_argument(
# # #         "--esm_model", type=str, default="facebook/esm2_t6_8M_UR50D",
# # #         help="ESM-2 checkpoint (default matches the PRISM paper's 8M model).",
# # #     )
# # #     p.add_argument(
# # #         "--max_pairs_substitution", type=int, default=200,
# # #         help="Max number of sample pairs to align per protein for the "
# # #              "substitution spectrum (default: 200). Alignment is O(N^2) "
# # #              "in the number of samples, so this caps runtime.",
# # #     )
# # #     p.add_argument(
# # #         "--max_pairs_hamming", type=int, default=300,
# # #         help="Max number of sample pairs to align per protein for the "
# # #              "Hamming diversity metric (default: 300).",
# # #     )
# # #     p.add_argument(
# # #         "--min_orf_aa", type=int, default=10,
# # #         help="Minimum translated ORF length (in amino acids) to keep a "
# # #              "sample for ESM-2 embedding (default: 10).",
# # #     )
# # #     p.add_argument(
# # #         "--batch_size", type=int, default=8,
# # #         help="ESM-2 embedding batch size (default: 8).",
# # #     )
# # #     p.add_argument(
# # #         "--seed", type=int, default=0,
# # #         help="Random seed for pair subsampling (default: 0).",
# # #     )
# # #     p.add_argument(
# # #         "--output_dir", type=str, required=True,
# # #         help="Directory to write result CSVs to.",
# # #     )
# # #     return p.parse_args(argv)


# # # # ── COSMIC signature matrix (individual per-signature profile files) ─
# # # _CONTEXT_KEY_RE = None  # set below, after build_96_context_keys is defined


# # # def parse_profile_file(path: str) -> pd.Series:
# # #     """
# # #     Parse a single COSMIC '<SIGNATURE>_PROFILE.txt' file into a Series
# # #     indexed by the 96 trinucleotide context keys ('A[C>A]A' format).

# # #     Handles the common COSMIC export schemas:
# # #       (a) two columns, first already in 'A[C>A]A' format
# # #       (b) separate 'Substitution Type' (e.g. 'C>A') and 'Trinucleotide'
# # #           (e.g. 'ACA') columns, plus one value column
# # #       (c) 'Type'/'MutationType' + value, same as (a) under a different header
# # #     Raises ValueError with a preview of the parsed columns if none of
# # #     these schemas match, so the mismatch is easy to diagnose.
# # #     """
# # #     df = None
# # #     for sep in ["\t", None, ",", r"\s+"]:
# # #         try:
# # #             candidate = pd.read_csv(path, sep=sep, engine="python")
# # #         except Exception:
# # #             continue
# # #         if candidate.shape[1] >= 2 and candidate.shape[0] >= 90:
# # #             df = candidate
# # #             break
# # #     if df is None:
# # #         raise ValueError(f"Could not parse {path} with any common separator.")

# # #     cols_lower = [c.strip().lower() for c in df.columns]
# # #     valid_keys = set(build_96_context_keys())

# # #     # Schema (a)/(c): a column already contains 'A[C>A]A'-style strings.
# # #     for col in df.columns:
# # #         sample_vals = df[col].astype(str).str.strip()
# # #         if sample_vals.isin(valid_keys).sum() >= 90:
# # #             value_col = [c for c in df.columns if c != col][0]
# # #             series = pd.Series(df[value_col].values, index=sample_vals.values)
# # #             return series.astype(float)

# # #     # Schema (b): separate substitution-type and trinucleotide columns.
# # #     sub_col = next((df.columns[i] for i, c in enumerate(cols_lower)
# # #                      if "substitution" in c or c == "type"), None)
# # #     tri_col = next((df.columns[i] for i, c in enumerate(cols_lower)
# # #                      if "trinucleotide" in c or "context" in c), None)
# # #     if sub_col is not None and tri_col is not None:
# # #         value_col = [c for c in df.columns if c not in (sub_col, tri_col)][0]
# # #         keys = [
# # #             f"{tri[0]}[{sub}]{tri[2]}"
# # #             for sub, tri in zip(df[sub_col].astype(str).str.strip(),
# # #                                  df[tri_col].astype(str).str.strip())
# # #         ]
# # #         if sum(k in valid_keys for k in keys) >= 90:
# # #             return pd.Series(df[value_col].values, index=keys).astype(float)

# # #     raise ValueError(
# # #         f"Could not identify context/value columns in {path}.\n"
# # #         f"Columns found: {list(df.columns)}\n"
# # #         f"First rows:\n{df.head(3)}\n"
# # #         f"Update parse_profile_file() to match this schema, or share a "
# # #         f"sample of this file's contents and I'll fix the parser."
# # #     )


# # # def load_cosmic_signatures_dir(root_dir: str) -> pd.DataFrame:
# # #     """
# # #     Recursively find all '<SIGNATURE>_PROFILE.txt' files under root_dir
# # #     (including category subfolders like Chemotherapy/, Clock-like/,
# # #     Mismatch/) and assemble them into one DataFrame: rows = 96 contexts,
# # #     columns = signature names (e.g. SBS2, SBS17a, ...).
# # #     """
# # #     profile_paths = []
# # #     for dirpath, _, filenames in os.walk(root_dir):
# # #         for fname in filenames:
# # #             if fname.upper().endswith("_PROFILE.TXT"):
# # #                 profile_paths.append(os.path.join(dirpath, fname))
# # #     if not profile_paths:
# # #         raise FileNotFoundError(
# # #             f"No '*_PROFILE.txt' files found under {root_dir} "
# # #             f"(searched recursively)."
# # #         )

# # #     columns = {}
# # #     for path in sorted(profile_paths):
# # #         sig_name = os.path.basename(path)
# # #         sig_name = sig_name[: -len("_PROFILE.txt")] if sig_name.upper().endswith("_PROFILE.TXT") \
# # #             else sig_name.rsplit(".", 1)[0]
# # #         columns[sig_name] = parse_profile_file(path)

# # #     df = pd.DataFrame(columns)
# # #     df = df.reindex(build_96_context_keys())  # enforce consistent row order
# # #     if df.isna().any().any():
# # #         missing = df.columns[df.isna().any()].tolist()
# # #         raise ValueError(
# # #             f"Some signatures are missing context rows after alignment: "
# # #             f"{missing}. Check that those profile files cover all 96 contexts."
# # #         )
# # #     # Normalize each signature to sum to 1.
# # #     df = df.div(df.sum(axis=0), axis=1)
# # #     return df


# # # def build_96_context_keys():
# # #     """All 96 trinucleotide substitution keys in 'X[R>A]Y' format."""
# # #     bases = ["A", "C", "G", "T"]
# # #     keys = []
# # #     for ref in ["C", "T"]:
# # #         for alt in bases:
# # #             if alt == ref:
# # #                 continue
# # #             for five in bases:
# # #                 for three in bases:
# # #                     keys.append(f"{five}[{ref}>{alt}]{three}")
# # #     return keys


# # # def canonicalize_substitution(five, ref, alt, three):
# # #     """
# # #     Fold a substitution onto the pyrimidine (C/T) reference strand per
# # #     COSMIC convention. If ref is a purine (A/G), take the reverse
# # #     complement of the trinucleotide and complement the alt base.
# # #     """
# # #     if ref in ("C", "T"):
# # #         return five, ref, alt, three
# # #     return COMPLEMENT[three], COMPLEMENT[ref], COMPLEMENT[alt], COMPLEMENT[five]


# # # # ── alignment-based substitution / Hamming extraction ───────────────
# # # def make_aligner():
# # #     aligner = Align.PairwiseAligner()
# # #     aligner.mode = "global"
# # #     aligner.match_score = 2
# # #     aligner.mismatch_score = -1
# # #     aligner.open_gap_score = -10
# # #     aligner.extend_gap_score = -0.5
# # #     return aligner


# # # def extract_substitutions(seq_a: str, seq_b: str, aligner) -> list:
# # #     """
# # #     Align two DNA sequences and return canonicalized (five, ref, alt, three)
# # #     tuples for every mismatch column whose immediate flanking columns (on
# # #     both sequences) are gap-free, so trinucleotide context is well-defined.
# # #     """
# # #     alignment = aligner.align(seq_a, seq_b)[0]
# # #     a_aligned, b_aligned = str(alignment[0]), str(alignment[1])
# # #     subs = []
# # #     for i in range(1, len(a_aligned) - 1):
# # #         a5, a0, a3 = a_aligned[i - 1], a_aligned[i], a_aligned[i + 1]
# # #         b5, b0, b3 = b_aligned[i - 1], b_aligned[i], b_aligned[i + 1]
# # #         if "-" in (a5, a0, a3, b5, b0, b3):
# # #             continue
# # #         if a0 == b0:
# # #             continue
# # #         five, ref, alt, three = canonicalize_substitution(a5, a0, b0, a3)
# # #         subs.append((five, ref, alt, three))
# # #     return subs


# # # def hamming_fraction(seq_a: str, seq_b: str, aligner) -> float:
# # #     """Mismatch fraction over ungapped aligned columns."""
# # #     alignment = aligner.align(seq_a, seq_b)[0]
# # #     a_aligned, b_aligned = str(alignment[0]), str(alignment[1])
# # #     mismatches = 0
# # #     length = 0
# # #     for x, y in zip(a_aligned, b_aligned):
# # #         if x == "-" or y == "-":
# # #             continue
# # #         length += 1
# # #         if x != y:
# # #             mismatches += 1
# # #     return mismatches / length if length > 0 else None


# # # def sample_pairs(n_items: int, max_pairs: int, rng: random.Random):
# # #     all_pairs = list(itertools.combinations(range(n_items), 2))
# # #     if len(all_pairs) > max_pairs:
# # #         all_pairs = rng.sample(all_pairs, max_pairs)
# # #     return all_pairs


# # # def build_spectrum(sequences, aligner, max_pairs, rng) -> pd.Series:
# # #     keys = build_96_context_keys()
# # #     counts = pd.Series(0, index=keys, dtype=float)
# # #     pairs = sample_pairs(len(sequences), max_pairs, rng)
# # #     for i, j in pairs:
# # #         for five, ref, alt, three in extract_substitutions(sequences[i], sequences[j], aligner):
# # #             key = f"{five}[{ref}>{alt}]{three}"
# # #             if key in counts.index:
# # #                 counts[key] += 1
# # #     return counts


# # # def nnls_decompose(spectrum: pd.Series, signatures: pd.DataFrame):
# # #     common = signatures.index.intersection(spectrum.index)
# # #     if len(common) == 0:
# # #         raise ValueError(
# # #             "No overlapping trinucleotide keys between computed spectrum "
# # #             "and COSMIC signature matrix -- check the context key format "
# # #             "in --cosmic_signatures (expects e.g. 'A[C>A]A')."
# # #         )
# # #     sig_matrix = signatures.loc[common].values
# # #     obs = spectrum.loc[common].values
# # #     total = obs.sum()
# # #     if total == 0:
# # #         return pd.Series(0.0, index=signatures.columns), None
# # #     obs_norm = obs / total
# # #     exposures, residual = nnls(sig_matrix, obs_norm)
# # #     exp_sum = exposures.sum()
# # #     if exp_sum > 0:
# # #         exposures = exposures / exp_sum
# # #     return pd.Series(exposures, index=signatures.columns), residual


# # # # ── ESM-2 embedding / centroid comparison ───────────────────────────
# # # _MATRIX_KEY_CANDIDATES = ["embeddings", "centroids", "X", "matrix", "vectors"]
# # # _ID_KEY_CANDIDATES = ["ids", "domain_ids", "labels", "names", "pfam_ids"]
# # # _PFAM_ID_RE = re.compile(r"^PF\d+$")


# # # def _extract_matrix_and_ids(data, source_desc: str):
# # #     keys = list(data.keys())

# # #     # Schema actually observed in this project's centroid chunks: each
# # #     # key IS a Pfam domain ID (e.g. 'PF00001') and its value is that
# # #     # domain's (320,) embedding vector directly -- there's no separate
# # #     # combined 'embeddings'/'ids' pair.
# # #     pfam_like_keys = [k for k in keys if _PFAM_ID_RE.match(k)]
# # #     if len(pfam_like_keys) >= max(1, 0.5 * len(keys)):
# # #         ids = pfam_like_keys
# # #         vectors = [np.asarray(data[k], dtype=np.float32).reshape(-1) for k in ids]
# # #         matrix = np.stack(vectors, axis=0)
# # #         return matrix, ids

# # #     # Fallback schema: one combined embeddings matrix + one id list.
# # #     matrix_key = next((k for k in _MATRIX_KEY_CANDIDATES if k in data), None)
# # #     id_key = next((k for k in _ID_KEY_CANDIDATES if k in data), None)
# # #     if matrix_key is None or id_key is None:
# # #         raise ValueError(
# # #             f"Could not find embedding matrix / id list in {source_desc}. "
# # #             f"Keys present: {list(data.keys())[:20]}"
# # #             f"{' ...' if len(data.keys()) > 20 else ''}. "
# # #             f"Update _extract_matrix_and_ids() to match, "
# # #             f"or share the key names and I'll fix this."
# # #         )
# # #     return np.asarray(data[matrix_key], dtype=np.float32), list(data[id_key])


# # # def load_centroid_chunks(chunks_dir: str):
# # #     """
# # #     Load and concatenate the PRISM Pfam centroid library from a directory
# # #     of chunked .npz files (e.g. centroid_chunk_001.npz, ...002.npz, ...).
# # #     Each chunk is expected to hold a (subset_M x 320) embedding matrix and
# # #     a parallel list of Pfam domain IDs.
# # #     """
# # #     chunk_paths = sorted(
# # #         os.path.join(chunks_dir, f) for f in os.listdir(chunks_dir) if f.endswith(".npz")
# # #     )
# # #     if not chunk_paths:
# # #         raise FileNotFoundError(f"No .npz files found in {chunks_dir}")

# # #     matrices, all_ids = [], []
# # #     for path in chunk_paths:
# # #         data = np.load(path, allow_pickle=True)
# # #         matrix, ids = _extract_matrix_and_ids(data, path)
# # #         matrices.append(matrix)
# # #         all_ids.extend(ids)

# # #     full_matrix = np.concatenate(matrices, axis=0)
# # #     full_matrix = full_matrix / np.linalg.norm(full_matrix, axis=1, keepdims=True)
# # #     print(f"[INFO] Loaded {len(chunk_paths)} centroid chunk(s) -> "
# # #           f"{full_matrix.shape[0]} total domains, {full_matrix.shape[1]}-d")
# # #     return full_matrix, all_ids


# # # def translate_orf(seq_str: str, min_aa: int):
# # #     seq_str = "".join(seq_str.split()).upper()
# # #     # Trim to a multiple of 3 so BioPython doesn't warn/fail on translate.
# # #     seq_str = seq_str[: len(seq_str) - (len(seq_str) % 3)]
# # #     if not seq_str:
# # #         return None
# # #     try:
# # #         protein = str(Seq(seq_str).translate(to_stop=True))
# # #     except Exception:
# # #         return None
# # #     if len(protein) < min_aa:
# # #         return None
# # #     return protein


# # # def embed_esm2(sequences, tokenizer, model, device, batch_size):
# # #     embeddings = []
# # #     for i in range(0, len(sequences), batch_size):
# # #         batch = sequences[i : i + batch_size]
# # #         inputs = tokenizer(
# # #             batch, return_tensors="pt", padding=True, truncation=True, max_length=1022,
# # #         )
# # #         inputs = {k: v.to(device) for k, v in inputs.items()}
# # #         with torch.no_grad():
# # #             out = model(**inputs)
# # #         hidden = out.last_hidden_state  # (B, L, D)
# # #         mask = inputs["attention_mask"].unsqueeze(-1)
# # #         pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)
# # #         embeddings.append(pooled.cpu().numpy())
# # #     if not embeddings:
# # #         return np.zeros((0, model.config.hidden_size), dtype=np.float32)
# # #     return np.concatenate(embeddings, axis=0)


# # # def creativity_and_embedding_diversity(embeddings, centroid_matrix, centroid_ids):
# # #     if embeddings.shape[0] == 0:
# # #         return 0, 0.0, None
# # #     norm_emb = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
# # #     sims_to_centroids = norm_emb @ centroid_matrix.T
# # #     nearest_idx = sims_to_centroids.argmax(axis=1)
# # #     nearest_domains = [centroid_ids[i] for i in nearest_idx]
# # #     n_unique_domains = len(set(nearest_domains))

# # #     if norm_emb.shape[0] > 1:
# # #         sims_pairwise = norm_emb @ norm_emb.T
# # #         iu = np.triu_indices_from(sims_pairwise, k=1)
# # #         mean_cos_dist = float(1.0 - sims_pairwise[iu].mean())
# # #     else:
# # #         mean_cos_dist = 0.0

# # #     top_domain = pd.Series(nearest_domains).value_counts().idxmax()
# # #     return n_unique_domains, mean_cos_dist, top_domain


# # # # ── main ─────────────────────────────────────────────────────────────
# # # def load_model_dirs(model_dir_args):
# # #     """Parse label:path CLI args into {label: path} and validate."""
# # #     model_dirs = {}
# # #     for entry in model_dir_args:
# # #         if ":" not in entry:
# # #             raise ValueError(
# # #                 f"--model_dir expects 'label:path', got: {entry!r}"
# # #             )
# # #         label, path = entry.split(":", 1)
# # #         if not os.path.isdir(path):
# # #             raise FileNotFoundError(f"Model output directory not found: {path}")
# # #         model_dirs[label] = path
# # #     return model_dirs


# # # def main(argv=None):
# # #     args = parse_args(argv)
# # #     os.makedirs(args.output_dir, exist_ok=True)
# # #     rng = random.Random(args.seed)
# # #     aligner = make_aligner()

# # #     print("[INFO] Loading COSMIC signature profile files...")
# # #     signatures = load_cosmic_signatures_dir(args.cosmic_dir)
# # #     print(f"[INFO] Loaded {signatures.shape[1]} signatures x {signatures.shape[0]} contexts")

# # #     print("[INFO] Loading Pfam centroid library chunks...")
# # #     centroid_matrix, centroid_ids = load_centroid_chunks(args.centroid_chunks_dir)

# # #     if AutoTokenizer is None:
# # #         sys.exit("transformers is required (pip install transformers)")

# # #     device = "cuda" if torch.cuda.is_available() else "cpu"
# # #     print(f"[INFO] Loading ESM-2 ({args.esm_model}) on {device}...")
# # #     tokenizer = AutoTokenizer.from_pretrained(args.esm_model)
# # #     model = AutoModel.from_pretrained(args.esm_model).to(device).eval()

# # #     model_dirs = load_model_dirs(args.model_dir)

# # #     exposure_rows = []
# # #     creativity_rows = []
# # #     spectrum_rows = []

# # #     for label, model_path in model_dirs.items():
# # #         fasta_files = sorted(f for f in os.listdir(model_path) if f.endswith("_generated.fasta"))
# # #         print(f"\n[INFO] Model '{label}': {len(fasta_files)} protein file(s) in {model_path}")

# # #         for fname in fasta_files:
# # #             protein_id = fname[: -len("_generated.fasta")]
# # #             fasta_path = os.path.join(model_path, fname)
# # #             records = list(SeqIO.parse(fasta_path, "fasta"))
# # #             sequences = [str(r.seq).upper() for r in records if len(str(r.seq)) > 0]

# # #             if len(sequences) < 2:
# # #                 print(f"  [warn] {label}/{protein_id}: fewer than 2 usable samples, skipping")
# # #                 continue

# # #             # ── substitution spectrum + NNLS ──────────────────────
# # #             spectrum = build_spectrum(sequences, aligner, args.max_pairs_substitution, rng)
# # #             exposures, residual = nnls_decompose(spectrum, signatures)

# # #             exposure_row = exposures.to_dict()
# # #             exposure_row.update({"model": label, "protein_id": protein_id,
# # #                                   "n_samples": len(sequences),
# # #                                   "total_substitutions": int(spectrum.sum()),
# # #                                   "nnls_residual": residual})
# # #             exposure_rows.append(exposure_row)

# # #             spectrum_row = spectrum.to_dict()
# # #             spectrum_row.update({"model": label, "protein_id": protein_id})
# # #             spectrum_rows.append(spectrum_row)

# # #             # ── ESM-2 creativity + embedding diversity ────────────
# # #             proteins = [translate_orf(s, args.min_orf_aa) for s in sequences]
# # #             proteins = [p for p in proteins if p is not None]
# # #             embeddings = embed_esm2(proteins, tokenizer, model, device, args.batch_size)
# # #             n_unique_domains, mean_cos_dist, top_domain = creativity_and_embedding_diversity(
# # #                 embeddings, centroid_matrix, centroid_ids,
# # #             )

# # #             # ── Hamming diversity (sequence-level, ESM-2-independent) ──
# # #             ham_pairs = sample_pairs(len(sequences), args.max_pairs_hamming, rng)
# # #             ham_dists = [
# # #                 d for d in (hamming_fraction(sequences[i], sequences[j], aligner) for i, j in ham_pairs)
# # #                 if d is not None
# # #             ]
# # #             mean_hamming = float(np.mean(ham_dists)) if ham_dists else None

# # #             creativity_rows.append({
# # #                 "model": label,
# # #                 "protein_id": protein_id,
# # #                 "n_samples": len(sequences),
# # #                 "n_translatable": len(proteins),
# # #                 "n_unique_domains_attracted": n_unique_domains,
# # #                 "top_attracted_domain": top_domain,
# # #                 "mean_esm2_cosine_distance": mean_cos_dist,
# # #                 "mean_hamming_distance": mean_hamming,
# # #             })

# # #             print(f"  {label}/{protein_id}: {len(sequences)} samples, "
# # #                   f"{int(spectrum.sum())} substitutions tallied, "
# # #                   f"{n_unique_domains} unique domain(s) attracted, "
# # #                   f"mean Hamming={mean_hamming}")

# # #     exposures_df = pd.DataFrame(exposure_rows)
# # #     creativity_df = pd.DataFrame(creativity_rows)
# # #     spectra_df = pd.DataFrame(spectrum_rows)

# # #     exposures_path = os.path.join(args.output_dir, "signature_exposures.csv")
# # #     creativity_path = os.path.join(args.output_dir, "creativity_diversity.csv")
# # #     spectra_path = os.path.join(args.output_dir, "substitution_spectra.csv")

# # #     exposures_df.to_csv(exposures_path, index=False)
# # #     creativity_df.to_csv(creativity_path, index=False)
# # #     spectra_df.to_csv(spectra_path, index=False)

# # #     print(f"\n[INFO] Wrote {exposures_path}")
# # #     print(f"[INFO] Wrote {creativity_path}")
# # #     print(f"[INFO] Wrote {spectra_path}")


# # # if __name__ == "__main__":
# # #     main()

# # #!/usr/bin/env python3
# # """
# # prism_genmodel_analysis.py

# # Tests whether generative DNA models (Carbon, GENERator, ...) have
# # implicitly learned COSMIC mutational-signature biases from training
# # data, following the PRISM pipeline's conventions (ESM-2 8M embeddings,
# # Pfam centroid library, per-protein analysis).

# # This is the step that runs AFTER generate_batch_carbon.py /
# # generate_batch_generator.py have produced <output_dir>/<record_id>_generated.fasta
# # files for each model.

# # WHY PAIRWISE-VS-PAIRWISE, NOT VS-WILD-TYPE
# # -------------------------------------------
# # generate_batch_carbon.py / generate_batch_generator.py feed the entire
# # wild-type sequence as the prompt and return a *novel continuation* --
# # there is no ground-truth continuation of the real gene to diff against.
# # So instead of comparing each sample to the wild-type, this script
# # aligns generated samples against EACH OTHER (all pairs, or a random
# # subsample of pairs if N is large) and tallies where they diverge. That
# # divergence pattern -- aggregated over many pairs -- is treated as the
# # model's implicit mutational spectrum for that protein.

# # PIPELINE (per protein, per model)
# # ----------------------------------
# # 1. Load the N generated samples from <model_dir>/<record_id>_generated.fasta
# # 2. Pairwise-align a subsample of samples (Bio.Align, global alignment).
# #    For every mismatched column that is NOT adjacent to a gap, extract
# #    the trinucleotide context and orient it onto the pyrimidine
# #    (C/T) reference strand per COSMIC convention.
# # 3. Aggregate mismatches into a 96-channel substitution spectrum and
# #    NNLS-decompose it against the COSMIC SBS signature matrix to get a
# #    per-protein, per-model signature exposure vector.
# # 4. Translate each sample (frame 0, truncate at first stop codon) and
# #    embed with ESM-2 (facebook/esm2_t6_8M_UR50D, matching the PRISM
# #    paper). Score:
# #      - "creativity": number of distinct Pfam domain centroids the
# #        sample cloud is nearest to (more spread = more "creative")
# #      - mean pairwise cosine distance among sample embeddings
# # 5. Compute mean pairwise Hamming distance (over aligned, ungapped
# #    columns) as a sequence-level diversity metric independent of ESM-2.

# # OUTPUTS (written to --output_dir)
# # -----------------------------------
# # signature_exposures.csv     one row per (model, protein), columns = SBS exposures
# # creativity_diversity.csv    one row per (model, protein)
# # substitution_spectra.csv    raw 96-channel counts, for debugging/inspection

# # USAGE
# # -----
# # python prism_genmodel_analysis.py \
# #     --model_dir carbon:/content/drive/MyDrive/PRISM/genmodel_bias/carbon_output \
# #     --model_dir generator:/content/drive/MyDrive/PRISM/genmodel_bias/generator_output \
# #     --cosmic_dir /content/drive/MyDrive/PRISM/cosmic_signatures \
# #     --centroid_chunks_dir /content/drive/MyDrive/PRISM/centroids/centroid_chunks \
# #     --output_dir /content/drive/MyDrive/PRISM/genmodel_bias/results
# # """

# # import argparse
# # import itertools
# # import os
# # import random
# # import re
# # import sys

# # import numpy as np
# # import pandas as pd
# # import torch
# # from Bio import SeqIO, Align
# # from Bio.Seq import Seq
# # from scipy.optimize import nnls

# # try:
# #     from transformers import AutoTokenizer, AutoModel
# # except ImportError:
# #     AutoTokenizer = AutoModel = None


# # COMPLEMENT = {"A": "T", "T": "A", "C": "G", "G": "C"}


# # # ── CLI ──────────────────────────────────────────────────────────────
# # def parse_args(argv=None):
# #     p = argparse.ArgumentParser(
# #         description="Test generative DNA models for implicit COSMIC signature bias.",
# #     )
# #     p.add_argument(
# #         "--model_dir", action="append", required=True,
# #         help="label:path pairs, e.g. carbon:/path/to/carbon_output. "
# #              "Repeat this flag once per model (Carbon, GENERator, ...).",
# #     )
# #     p.add_argument(
# #         "--cosmic_dir", type=str, required=True,
# #         help="Root directory containing individual COSMIC signature "
# #              "profile files (e.g. SBS2_PROFILE.txt, SBS17a_PROFILE.txt, ...). "
# #              "Subfolders (e.g. Chemotherapy/, Clock-like/, Mismatch/) are "
# #              "searched recursively.",
# #     )
# #     p.add_argument(
# #         "--centroid_chunks_dir", type=str, required=True,
# #         help="Directory containing the PRISM Pfam centroid library split "
# #              "across multiple .npz chunk files. All chunks are loaded and "
# #              "concatenated into one (M x 320) matrix.",
# #     )
# #     p.add_argument(
# #         "--esm_model", type=str, default="facebook/esm2_t6_8M_UR50D",
# #         help="ESM-2 checkpoint (default matches the PRISM paper's 8M model).",
# #     )
# #     p.add_argument(
# #         "--max_pairs_substitution", type=int, default=200,
# #         help="Max number of sample pairs to align per protein for the "
# #              "substitution spectrum (default: 200). Alignment is O(N^2) "
# #              "in the number of samples, so this caps runtime.",
# #     )
# #     p.add_argument(
# #         "--max_pairs_hamming", type=int, default=300,
# #         help="Max number of sample pairs to align per protein for the "
# #              "Hamming diversity metric (default: 300).",
# #     )
# #     p.add_argument(
# #         "--min_orf_aa", type=int, default=10,
# #         help="Minimum translated ORF length (in amino acids) to keep a "
# #              "sample for ESM-2 embedding (default: 10).",
# #     )
# #     p.add_argument(
# #         "--batch_size", type=int, default=8,
# #         help="ESM-2 embedding batch size (default: 8).",
# #     )
# #     p.add_argument(
# #         "--seed", type=int, default=0,
# #         help="Random seed for pair subsampling (default: 0).",
# #     )
# #     p.add_argument(
# #         "--output_dir", type=str, required=True,
# #         help="Directory to write result CSVs to.",
# #     )
# #     return p.parse_args(argv)


# # # ── COSMIC signature matrix (individual per-signature profile files) ─
# # _CONTEXT_KEY_RE = None  # set below, after build_96_context_keys is defined


# # def parse_profile_file(path: str) -> pd.Series:
# #     """
# #     Parse a single COSMIC '<SIGNATURE>_PROFILE.txt' file into a Series
# #     indexed by the 96 trinucleotide context keys ('A[C>A]A' format).

# #     Handles the common COSMIC export schemas:
# #       (a) two columns, first already in 'A[C>A]A' format
# #       (b) separate 'Substitution Type' (e.g. 'C>A') and 'Trinucleotide'
# #           (e.g. 'ACA') columns, plus one value column
# #       (c) 'Type'/'MutationType' + value, same as (a) under a different header
# #     Raises ValueError with a preview of the parsed columns if none of
# #     these schemas match, so the mismatch is easy to diagnose.
# #     """
# #     df = None
# #     for sep in ["\t", None, ",", r"\s+"]:
# #         try:
# #             candidate = pd.read_csv(path, sep=sep, engine="python")
# #         except Exception:
# #             continue
# #         if candidate.shape[1] >= 2 and candidate.shape[0] >= 90:
# #             df = candidate
# #             break
# #     if df is None:
# #         raise ValueError(f"Could not parse {path} with any common separator.")

# #     cols_lower = [c.strip().lower() for c in df.columns]
# #     valid_keys = set(build_96_context_keys())

# #     # Schema (a)/(c): a column already contains 'A[C>A]A'-style strings.
# #     for col in df.columns:
# #         sample_vals = df[col].astype(str).str.strip()
# #         if sample_vals.isin(valid_keys).sum() >= 90:
# #             value_col = [c for c in df.columns if c != col][0]
# #             series = pd.Series(df[value_col].values, index=sample_vals.values)
# #             return series.astype(float)

# #     # Schema (b): separate substitution-type and trinucleotide columns.
# #     sub_col = next((df.columns[i] for i, c in enumerate(cols_lower)
# #                      if "substitution" in c or c == "type"), None)
# #     tri_col = next((df.columns[i] for i, c in enumerate(cols_lower)
# #                      if "trinucleotide" in c or "context" in c), None)
# #     if sub_col is not None and tri_col is not None:
# #         value_col = [c for c in df.columns if c not in (sub_col, tri_col)][0]
# #         keys = [
# #             f"{tri[0]}[{sub}]{tri[2]}"
# #             for sub, tri in zip(df[sub_col].astype(str).str.strip(),
# #                                  df[tri_col].astype(str).str.strip())
# #         ]
# #         if sum(k in valid_keys for k in keys) >= 90:
# #             return pd.Series(df[value_col].values, index=keys).astype(float)

# #     raise ValueError(
# #         f"Could not identify context/value columns in {path}.\n"
# #         f"Columns found: {list(df.columns)}\n"
# #         f"First rows:\n{df.head(3)}\n"
# #         f"Update parse_profile_file() to match this schema, or share a "
# #         f"sample of this file's contents and I'll fix the parser."
# #     )


# # def load_cosmic_signatures_dir(root_dir: str) -> pd.DataFrame:
# #     """
# #     Recursively find all '<SIGNATURE>_PROFILE.txt' files under root_dir
# #     (including category subfolders like Chemotherapy/, Clock-like/,
# #     Mismatch/) and assemble them into one DataFrame: rows = 96 contexts,
# #     columns = signature names (e.g. SBS2, SBS17a, ...).
# #     """
# #     profile_paths = []
# #     for dirpath, _, filenames in os.walk(root_dir):
# #         for fname in filenames:
# #             if fname.upper().endswith("_PROFILE.TXT"):
# #                 profile_paths.append(os.path.join(dirpath, fname))
# #     if not profile_paths:
# #         raise FileNotFoundError(
# #             f"No '*_PROFILE.txt' files found under {root_dir} "
# #             f"(searched recursively)."
# #         )

# #     columns = {}
# #     for path in sorted(profile_paths):
# #         sig_name = os.path.basename(path)
# #         sig_name = sig_name[: -len("_PROFILE.txt")] if sig_name.upper().endswith("_PROFILE.TXT") \
# #             else sig_name.rsplit(".", 1)[0]
# #         columns[sig_name] = parse_profile_file(path)

# #     df = pd.DataFrame(columns)
# #     df = df.reindex(build_96_context_keys())  # enforce consistent row order
# #     if df.isna().any().any():
# #         missing = df.columns[df.isna().any()].tolist()
# #         raise ValueError(
# #             f"Some signatures are missing context rows after alignment: "
# #             f"{missing}. Check that those profile files cover all 96 contexts."
# #         )
# #     # Normalize each signature to sum to 1.
# #     df = df.div(df.sum(axis=0), axis=1)
# #     return df


# # def build_96_context_keys():
# #     """All 96 trinucleotide substitution keys in 'X[R>A]Y' format."""
# #     bases = ["A", "C", "G", "T"]
# #     keys = []
# #     for ref in ["C", "T"]:
# #         for alt in bases:
# #             if alt == ref:
# #                 continue
# #             for five in bases:
# #                 for three in bases:
# #                     keys.append(f"{five}[{ref}>{alt}]{three}")
# #     return keys


# # def canonicalize_substitution(five, ref, alt, three):
# #     """
# #     Fold a substitution onto the pyrimidine (C/T) reference strand per
# #     COSMIC convention. If ref is a purine (A/G), take the reverse
# #     complement of the trinucleotide and complement the alt base.
# #     """
# #     if ref in ("C", "T"):
# #         return five, ref, alt, three
# #     return COMPLEMENT[three], COMPLEMENT[ref], COMPLEMENT[alt], COMPLEMENT[five]


# # def is_degenerate(seq: str, max_homopolymer_frac: float = 0.3) -> bool:
# #     """
# #     Flag sequences dominated by a single-base repeat run (a common
# #     autoregressive-generation failure mode: "repetition collapse").
# #     Such sequences produce spurious, gene-agnostic substitution signal
# #     when pairwise-aligned against normal sequences, so they're excluded
# #     before the spectrum is built.
# #     """
# #     if not seq:
# #         return True
# #     longest_run = max(len(m.group(0)) for m in re.finditer(r"([ACGTN])\1*", seq))
# #     return (longest_run / len(seq)) > max_homopolymer_frac


# # def filter_degenerate(sequences: list) -> tuple:
# #     """Return (kept_sequences, n_dropped)."""
# #     kept = [s for s in sequences if not is_degenerate(s)]
# #     return kept, len(sequences) - len(kept)


# # # ── alignment-based substitution / Hamming extraction ───────────────
# # def make_aligner():
# #     aligner = Align.PairwiseAligner()
# #     aligner.mode = "global"
# #     aligner.match_score = 2
# #     aligner.mismatch_score = -1
# #     # Loosened relative to a stricter default: a mismatch-averse aligner
# #     # forces small length differences (e.g. from residual homopolymer
# #     # noise) to be represented as strings of substitutions instead of a
# #     # single gap, which inflates the substitution spectrum with artifacts.
# #     aligner.open_gap_score = -2
# #     aligner.extend_gap_score = -0.2
# #     return aligner


# # def extract_substitutions(seq_a: str, seq_b: str, aligner) -> list:
# #     """
# #     Align two DNA sequences and return canonicalized (five, ref, alt, three)
# #     tuples for every mismatch column whose immediate flanking columns (on
# #     both sequences) are gap-free, so trinucleotide context is well-defined.
# #     """
# #     alignment = aligner.align(seq_a, seq_b)[0]
# #     a_aligned, b_aligned = str(alignment[0]), str(alignment[1])
# #     subs = []
# #     for i in range(1, len(a_aligned) - 1):
# #         a5, a0, a3 = a_aligned[i - 1], a_aligned[i], a_aligned[i + 1]
# #         b5, b0, b3 = b_aligned[i - 1], b_aligned[i], b_aligned[i + 1]
# #         if "-" in (a5, a0, a3, b5, b0, b3):
# #             continue
# #         if a0 == b0:
# #             continue
# #         five, ref, alt, three = canonicalize_substitution(a5, a0, b0, a3)
# #         subs.append((five, ref, alt, three))
# #     return subs


# # def hamming_fraction(seq_a: str, seq_b: str, aligner) -> float:
# #     """Mismatch fraction over ungapped aligned columns."""
# #     alignment = aligner.align(seq_a, seq_b)[0]
# #     a_aligned, b_aligned = str(alignment[0]), str(alignment[1])
# #     mismatches = 0
# #     length = 0
# #     for x, y in zip(a_aligned, b_aligned):
# #         if x == "-" or y == "-":
# #             continue
# #         length += 1
# #         if x != y:
# #             mismatches += 1
# #     return mismatches / length if length > 0 else None


# # def sample_pairs(n_items: int, max_pairs: int, rng: random.Random):
# #     all_pairs = list(itertools.combinations(range(n_items), 2))
# #     if len(all_pairs) > max_pairs:
# #         all_pairs = rng.sample(all_pairs, max_pairs)
# #     return all_pairs


# # def build_spectrum(sequences, aligner, max_pairs, rng) -> pd.Series:
# #     keys = build_96_context_keys()
# #     counts = pd.Series(0, index=keys, dtype=float)
# #     pairs = sample_pairs(len(sequences), max_pairs, rng)
# #     for i, j in pairs:
# #         for five, ref, alt, three in extract_substitutions(sequences[i], sequences[j], aligner):
# #             key = f"{five}[{ref}>{alt}]{three}"
# #             if key in counts.index:
# #                 counts[key] += 1
# #     return counts


# # def nnls_decompose(spectrum: pd.Series, signatures: pd.DataFrame):
# #     common = signatures.index.intersection(spectrum.index)
# #     if len(common) == 0:
# #         raise ValueError(
# #             "No overlapping trinucleotide keys between computed spectrum "
# #             "and COSMIC signature matrix -- check the context key format "
# #             "in --cosmic_signatures (expects e.g. 'A[C>A]A')."
# #         )
# #     sig_matrix = signatures.loc[common].values
# #     obs = spectrum.loc[common].values
# #     total = obs.sum()
# #     if total == 0:
# #         return pd.Series(0.0, index=signatures.columns), None
# #     obs_norm = obs / total
# #     exposures, residual = nnls(sig_matrix, obs_norm)
# #     exp_sum = exposures.sum()
# #     if exp_sum > 0:
# #         exposures = exposures / exp_sum
# #     return pd.Series(exposures, index=signatures.columns), residual


# # # ── ESM-2 embedding / centroid comparison ───────────────────────────
# # _MATRIX_KEY_CANDIDATES = ["embeddings", "centroids", "X", "matrix", "vectors"]
# # _ID_KEY_CANDIDATES = ["ids", "domain_ids", "labels", "names", "pfam_ids"]
# # _PFAM_ID_RE = re.compile(r"^PF\d+$")


# # def _extract_matrix_and_ids(data, source_desc: str):
# #     keys = list(data.keys())

# #     # Schema actually observed in this project's centroid chunks: each
# #     # key IS a Pfam domain ID (e.g. 'PF00001') and its value is that
# #     # domain's (320,) embedding vector directly -- there's no separate
# #     # combined 'embeddings'/'ids' pair.
# #     pfam_like_keys = [k for k in keys if _PFAM_ID_RE.match(k)]
# #     if len(pfam_like_keys) >= max(1, 0.5 * len(keys)):
# #         ids = pfam_like_keys
# #         vectors = [np.asarray(data[k], dtype=np.float32).reshape(-1) for k in ids]
# #         matrix = np.stack(vectors, axis=0)
# #         return matrix, ids

# #     # Fallback schema: one combined embeddings matrix + one id list.
# #     matrix_key = next((k for k in _MATRIX_KEY_CANDIDATES if k in data), None)
# #     id_key = next((k for k in _ID_KEY_CANDIDATES if k in data), None)
# #     if matrix_key is None or id_key is None:
# #         raise ValueError(
# #             f"Could not find embedding matrix / id list in {source_desc}. "
# #             f"Keys present: {list(data.keys())[:20]}"
# #             f"{' ...' if len(data.keys()) > 20 else ''}. "
# #             f"Update _extract_matrix_and_ids() to match, "
# #             f"or share the key names and I'll fix this."
# #         )
# #     return np.asarray(data[matrix_key], dtype=np.float32), list(data[id_key])


# # def load_centroid_chunks(chunks_dir: str):
# #     """
# #     Load and concatenate the PRISM Pfam centroid library from a directory
# #     of chunked .npz files (e.g. centroid_chunk_001.npz, ...002.npz, ...).
# #     Each chunk is expected to hold a (subset_M x 320) embedding matrix and
# #     a parallel list of Pfam domain IDs.
# #     """
# #     chunk_paths = sorted(
# #         os.path.join(chunks_dir, f) for f in os.listdir(chunks_dir) if f.endswith(".npz")
# #     )
# #     if not chunk_paths:
# #         raise FileNotFoundError(f"No .npz files found in {chunks_dir}")

# #     matrices, all_ids = [], []
# #     for path in chunk_paths:
# #         data = np.load(path, allow_pickle=True)
# #         matrix, ids = _extract_matrix_and_ids(data, path)
# #         matrices.append(matrix)
# #         all_ids.extend(ids)

# #     full_matrix = np.concatenate(matrices, axis=0)
# #     full_matrix = full_matrix / np.linalg.norm(full_matrix, axis=1, keepdims=True)
# #     print(f"[INFO] Loaded {len(chunk_paths)} centroid chunk(s) -> "
# #           f"{full_matrix.shape[0]} total domains, {full_matrix.shape[1]}-d")
# #     return full_matrix, all_ids


# # def translate_orf(seq_str: str, min_aa: int):
# #     seq_str = "".join(seq_str.split()).upper()
# #     # Trim to a multiple of 3 so BioPython doesn't warn/fail on translate.
# #     seq_str = seq_str[: len(seq_str) - (len(seq_str) % 3)]
# #     if not seq_str:
# #         return None
# #     try:
# #         protein = str(Seq(seq_str).translate(to_stop=True))
# #     except Exception:
# #         return None
# #     if len(protein) < min_aa:
# #         return None
# #     return protein


# # def embed_esm2(sequences, tokenizer, model, device, batch_size):
# #     embeddings = []
# #     for i in range(0, len(sequences), batch_size):
# #         batch = sequences[i : i + batch_size]
# #         inputs = tokenizer(
# #             batch, return_tensors="pt", padding=True, truncation=True, max_length=1022,
# #         )
# #         inputs = {k: v.to(device) for k, v in inputs.items()}
# #         with torch.no_grad():
# #             out = model(**inputs)
# #         hidden = out.last_hidden_state  # (B, L, D)
# #         mask = inputs["attention_mask"].unsqueeze(-1)
# #         pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)
# #         embeddings.append(pooled.cpu().numpy())
# #     if not embeddings:
# #         return np.zeros((0, model.config.hidden_size), dtype=np.float32)
# #     return np.concatenate(embeddings, axis=0)


# # def creativity_and_embedding_diversity(embeddings, centroid_matrix, centroid_ids):
# #     if embeddings.shape[0] == 0:
# #         return 0, 0.0, None
# #     norm_emb = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
# #     sims_to_centroids = norm_emb @ centroid_matrix.T
# #     nearest_idx = sims_to_centroids.argmax(axis=1)
# #     nearest_domains = [centroid_ids[i] for i in nearest_idx]
# #     n_unique_domains = len(set(nearest_domains))

# #     if norm_emb.shape[0] > 1:
# #         sims_pairwise = norm_emb @ norm_emb.T
# #         iu = np.triu_indices_from(sims_pairwise, k=1)
# #         mean_cos_dist = float(1.0 - sims_pairwise[iu].mean())
# #     else:
# #         mean_cos_dist = 0.0

# #     top_domain = pd.Series(nearest_domains).value_counts().idxmax()
# #     return n_unique_domains, mean_cos_dist, top_domain


# # # ── main ─────────────────────────────────────────────────────────────
# # def load_model_dirs(model_dir_args):
# #     """Parse label:path CLI args into {label: path} and validate."""
# #     model_dirs = {}
# #     for entry in model_dir_args:
# #         if ":" not in entry:
# #             raise ValueError(
# #                 f"--model_dir expects 'label:path', got: {entry!r}"
# #             )
# #         label, path = entry.split(":", 1)
# #         if not os.path.isdir(path):
# #             raise FileNotFoundError(f"Model output directory not found: {path}")
# #         model_dirs[label] = path
# #     return model_dirs


# # def main(argv=None):
# #     args = parse_args(argv)
# #     os.makedirs(args.output_dir, exist_ok=True)
# #     rng = random.Random(args.seed)
# #     aligner = make_aligner()

# #     print("[INFO] Loading COSMIC signature profile files...")
# #     signatures = load_cosmic_signatures_dir(args.cosmic_dir)
# #     print(f"[INFO] Loaded {signatures.shape[1]} signatures x {signatures.shape[0]} contexts")

# #     print("[INFO] Loading Pfam centroid library chunks...")
# #     centroid_matrix, centroid_ids = load_centroid_chunks(args.centroid_chunks_dir)

# #     if AutoTokenizer is None:
# #         sys.exit("transformers is required (pip install transformers)")

# #     device = "cuda" if torch.cuda.is_available() else "cpu"
# #     print(f"[INFO] Loading ESM-2 ({args.esm_model}) on {device}...")
# #     tokenizer = AutoTokenizer.from_pretrained(args.esm_model)
# #     model = AutoModel.from_pretrained(args.esm_model).to(device).eval()

# #     model_dirs = load_model_dirs(args.model_dir)

# #     exposure_rows = []
# #     creativity_rows = []
# #     spectrum_rows = []

# #     for label, model_path in model_dirs.items():
# #         fasta_files = sorted(f for f in os.listdir(model_path) if f.endswith("_generated.fasta"))
# #         print(f"\n[INFO] Model '{label}': {len(fasta_files)} protein file(s) in {model_path}")

# #         for fname in fasta_files:
# #             protein_id = fname[: -len("_generated.fasta")]
# #             fasta_path = os.path.join(model_path, fname)
# #             records = list(SeqIO.parse(fasta_path, "fasta"))
# #             raw_sequences = [str(r.seq).upper() for r in records if len(str(r.seq)) > 0]
# #             sequences, n_dropped = filter_degenerate(raw_sequences)

# #             if n_dropped > 0:
# #                 print(f"  [warn] {label}/{protein_id}: dropped {n_dropped}/{len(raw_sequences)} "
# #                       f"degenerate (homopolymer-collapsed) sample(s)")

# #             if len(sequences) < 2:
# #                 print(f"  [warn] {label}/{protein_id}: fewer than 2 usable samples after "
# #                       f"filtering, skipping")
# #                 continue

# #             # ── substitution spectrum + NNLS ──────────────────────
# #             spectrum = build_spectrum(sequences, aligner, args.max_pairs_substitution, rng)
# #             exposures, residual = nnls_decompose(spectrum, signatures)

# #             exposure_row = exposures.to_dict()
# #             exposure_row.update({"model": label, "protein_id": protein_id,
# #                                   "n_samples": len(sequences),
# #                                   "n_raw_samples": len(raw_sequences),
# #                                   "n_dropped_degenerate": n_dropped,
# #                                   "total_substitutions": int(spectrum.sum()),
# #                                   "nnls_residual": residual})
# #             exposure_rows.append(exposure_row)

# #             spectrum_row = spectrum.to_dict()
# #             spectrum_row.update({"model": label, "protein_id": protein_id})
# #             spectrum_rows.append(spectrum_row)

# #             # ── ESM-2 creativity + embedding diversity ────────────
# #             proteins = [translate_orf(s, args.min_orf_aa) for s in sequences]
# #             proteins = [p for p in proteins if p is not None]
# #             embeddings = embed_esm2(proteins, tokenizer, model, device, args.batch_size)
# #             n_unique_domains, mean_cos_dist, top_domain = creativity_and_embedding_diversity(
# #                 embeddings, centroid_matrix, centroid_ids,
# #             )

# #             # ── Hamming diversity (sequence-level, ESM-2-independent) ──
# #             ham_pairs = sample_pairs(len(sequences), args.max_pairs_hamming, rng)
# #             ham_dists = [
# #                 d for d in (hamming_fraction(sequences[i], sequences[j], aligner) for i, j in ham_pairs)
# #                 if d is not None
# #             ]
# #             mean_hamming = float(np.mean(ham_dists)) if ham_dists else None

# #             creativity_rows.append({
# #                 "model": label,
# #                 "protein_id": protein_id,
# #                 "n_samples": len(sequences),
# #                 "n_translatable": len(proteins),
# #                 "n_unique_domains_attracted": n_unique_domains,
# #                 "top_attracted_domain": top_domain,
# #                 "mean_esm2_cosine_distance": mean_cos_dist,
# #                 "mean_hamming_distance": mean_hamming,
# #             })

# #             print(f"  {label}/{protein_id}: {len(sequences)} samples, "
# #                   f"{int(spectrum.sum())} substitutions tallied, "
# #                   f"{n_unique_domains} unique domain(s) attracted, "
# #                   f"mean Hamming={mean_hamming}")

# #     exposures_df = pd.DataFrame(exposure_rows)
# #     creativity_df = pd.DataFrame(creativity_rows)
# #     spectra_df = pd.DataFrame(spectrum_rows)

# #     exposures_path = os.path.join(args.output_dir, "signature_exposures.csv")
# #     creativity_path = os.path.join(args.output_dir, "creativity_diversity.csv")
# #     spectra_path = os.path.join(args.output_dir, "substitution_spectra.csv")

# #     exposures_df.to_csv(exposures_path, index=False)
# #     creativity_df.to_csv(creativity_path, index=False)
# #     spectra_df.to_csv(spectra_path, index=False)

# #     print(f"\n[INFO] Wrote {exposures_path}")
# #     print(f"[INFO] Wrote {creativity_path}")
# #     print(f"[INFO] Wrote {spectra_path}")


# # if __name__ == "__main__":
# #     main()

# #!/usr/bin/env python3
# """
# prism_genmodel_analysis.py

# Tests whether generative DNA models (Carbon, GENERator, ...) have
# implicitly learned COSMIC mutational-signature biases from training
# data, following the PRISM pipeline's conventions (ESM-2 8M embeddings,
# Pfam centroid library, per-protein analysis).

# This is the step that runs AFTER generate_batch_carbon.py /
# generate_batch_generator.py have produced <output_dir>/<record_id>_generated.fasta
# files for each model.

# WHY PAIRWISE-VS-PAIRWISE, NOT VS-WILD-TYPE
# -------------------------------------------
# generate_batch_carbon.py / generate_batch_generator.py feed the entire
# wild-type sequence as the prompt and return a *novel continuation* --
# there is no ground-truth continuation of the real gene to diff against.
# So instead of comparing each sample to the wild-type, this script
# aligns generated samples against EACH OTHER (all pairs, or a random
# subsample of pairs if N is large) and tallies where they diverge. That
# divergence pattern -- aggregated over many pairs -- is treated as the
# model's implicit mutational spectrum for that protein.

# PIPELINE (per protein, per model)
# ----------------------------------
# 1. Load the N generated samples from <model_dir>/<record_id>_generated.fasta
# 2. Pairwise-align a subsample of samples (Bio.Align, global alignment).
#    For every mismatched column that is NOT adjacent to a gap, extract
#    the trinucleotide context and orient it onto the pyrimidine
#    (C/T) reference strand per COSMIC convention.
# 3. Aggregate mismatches into a 96-channel substitution spectrum and
#    NNLS-decompose it against the COSMIC SBS signature matrix to get a
#    per-protein, per-model signature exposure vector.
# 4. Translate each sample (frame 0, truncate at first stop codon) and
#    embed with ESM-2 (facebook/esm2_t6_8M_UR50D, matching the PRISM
#    paper). Score:
#      - "creativity": number of distinct Pfam domain centroids the
#        sample cloud is nearest to (more spread = more "creative")
#      - mean pairwise cosine distance among sample embeddings
# 5. Compute mean pairwise Hamming distance (over aligned, ungapped
#    columns) as a sequence-level diversity metric independent of ESM-2.

# OUTPUTS (written to --output_dir)
# -----------------------------------
# signature_exposures.csv     one row per (model, protein), columns = SBS exposures
# creativity_diversity.csv    one row per (model, protein)
# substitution_spectra.csv    raw 96-channel counts, for debugging/inspection

# USAGE
# -----
# python prism_genmodel_analysis.py \
#     --model_dir carbon:/content/drive/MyDrive/PRISM/genmodel_bias/carbon_output \
#     --model_dir generator:/content/drive/MyDrive/PRISM/genmodel_bias/generator_output \
#     --cosmic_dir /content/drive/MyDrive/PRISM/cosmic_signatures \
#     --centroid_chunks_dir /content/drive/MyDrive/PRISM/centroids/centroid_chunks \
#     --output_dir /content/drive/MyDrive/PRISM/genmodel_bias/results
# """

# import argparse
# import itertools
# import os
# import random
# import re
# import sys

# import numpy as np
# import pandas as pd
# import torch
# from Bio import SeqIO, Align
# from Bio.Seq import Seq
# from scipy.optimize import nnls

# try:
#     from transformers import AutoTokenizer, AutoModel
# except ImportError:
#     AutoTokenizer = AutoModel = None


# COMPLEMENT = {"A": "T", "T": "A", "C": "G", "G": "C"}


# # ── CLI ──────────────────────────────────────────────────────────────
# def parse_args(argv=None):
#     p = argparse.ArgumentParser(
#         description="Test generative DNA models for implicit COSMIC signature bias.",
#     )
#     p.add_argument(
#         "--model_dir", action="append", required=True,
#         help="label:path pairs, e.g. carbon:/path/to/carbon_output. "
#              "Repeat this flag once per model (Carbon, GENERator, ...).",
#     )
#     p.add_argument(
#         "--cosmic_dir", type=str, required=True,
#         help="Root directory containing individual COSMIC signature "
#              "profile files (e.g. SBS2_PROFILE.txt, SBS17a_PROFILE.txt, ...). "
#              "Subfolders (e.g. Chemotherapy/, Clock-like/, Mismatch/) are "
#              "searched recursively.",
#     )
#     p.add_argument(
#         "--centroid_chunks_dir", type=str, required=True,
#         help="Directory containing the PRISM Pfam centroid library split "
#              "across multiple .npz chunk files. All chunks are loaded and "
#              "concatenated into one (M x 320) matrix.",
#     )
#     p.add_argument(
#         "--esm_model", type=str, default="facebook/esm2_t6_8M_UR50D",
#         help="ESM-2 checkpoint (default matches the PRISM paper's 8M model).",
#     )
#     p.add_argument(
#         "--max_pairs_substitution", type=int, default=200,
#         help="Max number of sample pairs to align per protein for the "
#              "substitution spectrum (default: 200). Alignment is O(N^2) "
#              "in the number of samples, so this caps runtime.",
#     )
#     p.add_argument(
#         "--max_pairs_hamming", type=int, default=300,
#         help="Max number of sample pairs to align per protein for the "
#              "Hamming diversity metric (default: 300). Only used in "
#              "legacy pairwise mode (no reference continuation available).",
#     )
#     p.add_argument(
#         "--max_samples_reference", type=int, default=200,
#         help="Max number of samples to align against the true reference "
#              "continuation, when available (default: 200). Reference "
#              "mode is O(N) so this can be set higher than the legacy "
#              "pairwise caps without much added cost.",
#     )
#     p.add_argument(
#         "--min_orf_aa", type=int, default=10,
#         help="Minimum translated ORF length (in amino acids) to keep a "
#              "sample for ESM-2 embedding (default: 10).",
#     )
#     p.add_argument(
#         "--batch_size", type=int, default=8,
#         help="ESM-2 embedding batch size (default: 8).",
#     )
#     p.add_argument(
#         "--seed", type=int, default=0,
#         help="Random seed for pair subsampling (default: 0).",
#     )
#     p.add_argument(
#         "--output_dir", type=str, required=True,
#         help="Directory to write result CSVs to.",
#     )
#     return p.parse_args(argv)


# # ── COSMIC signature matrix (individual per-signature profile files) ─
# _CONTEXT_KEY_RE = None  # set below, after build_96_context_keys is defined


# def parse_profile_file(path: str) -> pd.Series:
#     """
#     Parse a single COSMIC '<SIGNATURE>_PROFILE.txt' file into a Series
#     indexed by the 96 trinucleotide context keys ('A[C>A]A' format).

#     Handles the common COSMIC export schemas:
#       (a) two columns, first already in 'A[C>A]A' format
#       (b) separate 'Substitution Type' (e.g. 'C>A') and 'Trinucleotide'
#           (e.g. 'ACA') columns, plus one value column
#       (c) 'Type'/'MutationType' + value, same as (a) under a different header
#     Raises ValueError with a preview of the parsed columns if none of
#     these schemas match, so the mismatch is easy to diagnose.
#     """
#     df = None
#     for sep in ["\t", None, ",", r"\s+"]:
#         try:
#             candidate = pd.read_csv(path, sep=sep, engine="python")
#         except Exception:
#             continue
#         if candidate.shape[1] >= 2 and candidate.shape[0] >= 90:
#             df = candidate
#             break
#     if df is None:
#         raise ValueError(f"Could not parse {path} with any common separator.")

#     cols_lower = [c.strip().lower() for c in df.columns]
#     valid_keys = set(build_96_context_keys())

#     # Schema (a)/(c): a column already contains 'A[C>A]A'-style strings.
#     for col in df.columns:
#         sample_vals = df[col].astype(str).str.strip()
#         if sample_vals.isin(valid_keys).sum() >= 90:
#             value_col = [c for c in df.columns if c != col][0]
#             series = pd.Series(df[value_col].values, index=sample_vals.values)
#             return series.astype(float)

#     # Schema (b): separate substitution-type and trinucleotide columns.
#     sub_col = next((df.columns[i] for i, c in enumerate(cols_lower)
#                      if "substitution" in c or c == "type"), None)
#     tri_col = next((df.columns[i] for i, c in enumerate(cols_lower)
#                      if "trinucleotide" in c or "context" in c), None)
#     if sub_col is not None and tri_col is not None:
#         value_col = [c for c in df.columns if c not in (sub_col, tri_col)][0]
#         keys = [
#             f"{tri[0]}[{sub}]{tri[2]}"
#             for sub, tri in zip(df[sub_col].astype(str).str.strip(),
#                                  df[tri_col].astype(str).str.strip())
#         ]
#         if sum(k in valid_keys for k in keys) >= 90:
#             return pd.Series(df[value_col].values, index=keys).astype(float)

#     raise ValueError(
#         f"Could not identify context/value columns in {path}.\n"
#         f"Columns found: {list(df.columns)}\n"
#         f"First rows:\n{df.head(3)}\n"
#         f"Update parse_profile_file() to match this schema, or share a "
#         f"sample of this file's contents and I'll fix the parser."
#     )


# def load_cosmic_signatures_dir(root_dir: str) -> pd.DataFrame:
#     """
#     Recursively find all '<SIGNATURE>_PROFILE.txt' files under root_dir
#     (including category subfolders like Chemotherapy/, Clock-like/,
#     Mismatch/) and assemble them into one DataFrame: rows = 96 contexts,
#     columns = signature names (e.g. SBS2, SBS17a, ...).
#     """
#     profile_paths = []
#     for dirpath, _, filenames in os.walk(root_dir):
#         for fname in filenames:
#             if fname.upper().endswith("_PROFILE.TXT"):
#                 profile_paths.append(os.path.join(dirpath, fname))
#     if not profile_paths:
#         raise FileNotFoundError(
#             f"No '*_PROFILE.txt' files found under {root_dir} "
#             f"(searched recursively)."
#         )

#     columns = {}
#     for path in sorted(profile_paths):
#         sig_name = os.path.basename(path)
#         sig_name = sig_name[: -len("_PROFILE.txt")] if sig_name.upper().endswith("_PROFILE.TXT") \
#             else sig_name.rsplit(".", 1)[0]
#         columns[sig_name] = parse_profile_file(path)

#     df = pd.DataFrame(columns)
#     df = df.reindex(build_96_context_keys())  # enforce consistent row order
#     if df.isna().any().any():
#         missing = df.columns[df.isna().any()].tolist()
#         raise ValueError(
#             f"Some signatures are missing context rows after alignment: "
#             f"{missing}. Check that those profile files cover all 96 contexts."
#         )
#     # Normalize each signature to sum to 1.
#     df = df.div(df.sum(axis=0), axis=1)
#     return df


# def build_96_context_keys():
#     """All 96 trinucleotide substitution keys in 'X[R>A]Y' format."""
#     bases = ["A", "C", "G", "T"]
#     keys = []
#     for ref in ["C", "T"]:
#         for alt in bases:
#             if alt == ref:
#                 continue
#             for five in bases:
#                 for three in bases:
#                     keys.append(f"{five}[{ref}>{alt}]{three}")
#     return keys


# def canonicalize_substitution(five, ref, alt, three):
#     """
#     Fold a substitution onto the pyrimidine (C/T) reference strand per
#     COSMIC convention. If ref is a purine (A/G), take the reverse
#     complement of the trinucleotide and complement the alt base.
#     """
#     if ref in ("C", "T"):
#         return five, ref, alt, three
#     return COMPLEMENT[three], COMPLEMENT[ref], COMPLEMENT[alt], COMPLEMENT[five]


# def is_degenerate(seq: str, max_homopolymer_frac: float = 0.3) -> bool:
#     """
#     Flag sequences dominated by a single-base repeat run (a common
#     autoregressive-generation failure mode: "repetition collapse").
#     Such sequences produce spurious, gene-agnostic substitution signal
#     when pairwise-aligned against normal sequences, so they're excluded
#     before the spectrum is built.
#     """
#     if not seq:
#         return True
#     longest_run = max(len(m.group(0)) for m in re.finditer(r"([ACGTN])\1*", seq))
#     return (longest_run / len(seq)) > max_homopolymer_frac


# def filter_degenerate(sequences: list) -> tuple:
#     """Return (kept_sequences, n_dropped)."""
#     kept = [s for s in sequences if not is_degenerate(s)]
#     return kept, len(sequences) - len(kept)


# # ── alignment-based substitution / Hamming extraction ───────────────
# def make_aligner():
#     aligner = Align.PairwiseAligner()
#     aligner.mode = "global"
#     aligner.match_score = 2
#     aligner.mismatch_score = -1
#     # Loosened relative to a stricter default: a mismatch-averse aligner
#     # forces small length differences (e.g. from residual homopolymer
#     # noise) to be represented as strings of substitutions instead of a
#     # single gap, which inflates the substitution spectrum with artifacts.
#     aligner.open_gap_score = -2
#     aligner.extend_gap_score = -0.2
#     return aligner


# def extract_substitutions(seq_a: str, seq_b: str, aligner) -> list:
#     """
#     Align two DNA sequences and return canonicalized (five, ref, alt, three)
#     tuples for every mismatch column whose immediate flanking columns (on
#     both sequences) are gap-free, so trinucleotide context is well-defined.
#     """
#     alignment = aligner.align(seq_a, seq_b)[0]
#     a_aligned, b_aligned = str(alignment[0]), str(alignment[1])
#     subs = []
#     for i in range(1, len(a_aligned) - 1):
#         a5, a0, a3 = a_aligned[i - 1], a_aligned[i], a_aligned[i + 1]
#         b5, b0, b3 = b_aligned[i - 1], b_aligned[i], b_aligned[i + 1]
#         if "-" in (a5, a0, a3, b5, b0, b3):
#             continue
#         if a0 == b0:
#             continue
#         five, ref, alt, three = canonicalize_substitution(a5, a0, b0, a3)
#         subs.append((five, ref, alt, three))
#     return subs


# def hamming_fraction(seq_a: str, seq_b: str, aligner) -> float:
#     """Mismatch fraction over ungapped aligned columns."""
#     alignment = aligner.align(seq_a, seq_b)[0]
#     a_aligned, b_aligned = str(alignment[0]), str(alignment[1])
#     mismatches = 0
#     length = 0
#     for x, y in zip(a_aligned, b_aligned):
#         if x == "-" or y == "-":
#             continue
#         length += 1
#         if x != y:
#             mismatches += 1
#     return mismatches / length if length > 0 else None


# def sample_pairs(n_items: int, max_pairs: int, rng: random.Random):
#     all_pairs = list(itertools.combinations(range(n_items), 2))
#     if len(all_pairs) > max_pairs:
#         all_pairs = rng.sample(all_pairs, max_pairs)
#     return all_pairs


# def build_spectrum(sequences, aligner, max_pairs, rng) -> pd.Series:
#     """Legacy pairwise-self-comparison spectrum (used when no reference
#     continuation is available, e.g. old whole-CDS-prompt runs)."""
#     keys = build_96_context_keys()
#     counts = pd.Series(0, index=keys, dtype=float)
#     pairs = sample_pairs(len(sequences), max_pairs, rng)
#     for i, j in pairs:
#         for five, ref, alt, three in extract_substitutions(sequences[i], sequences[j], aligner):
#             key = f"{five}[{ref}>{alt}]{three}"
#             if key in counts.index:
#                 counts[key] += 1
#     return counts


# def build_spectrum_reference(sequences, reference_seq, aligner, max_samples, rng) -> pd.Series:
#     """
#     Reference-based spectrum: align each generated sample against the
#     TRUE continuation of the real gene (one alignment per sample, O(N)
#     instead of O(N^2)) and tally substitutions relative to that real
#     sequence. Only usable when a prefix-based prompt design (see
#     generate_batch_carbon.py / generate_batch_generator.py --prefix_frac)
#     was used to generate the samples, since that's what leaves a real
#     continuation to align against.
#     """
#     keys = build_96_context_keys()
#     counts = pd.Series(0, index=keys, dtype=float)
#     subset = sequences if len(sequences) <= max_samples else rng.sample(sequences, max_samples)
#     for sample_seq in subset:
#         for five, ref, alt, three in extract_substitutions(reference_seq, sample_seq, aligner):
#             key = f"{five}[{ref}>{alt}]{three}"
#             if key in counts.index:
#                 counts[key] += 1
#     return counts


# def compute_identity_to_reference(sequences, reference_seq, aligner, max_samples, rng):
#     """Per-sample identity fraction (matches / aligned ungapped length)
#     against the true reference continuation. Returns the list of
#     per-sample identities (not just the mean) so callers can report
#     both central tendency and spread."""
#     subset = sequences if len(sequences) <= max_samples else rng.sample(sequences, max_samples)
#     identities = []
#     for sample_seq in subset:
#         frac = hamming_fraction(reference_seq, sample_seq, aligner)
#         if frac is not None:
#             identities.append(1.0 - frac)
#     return identities


# def load_expected_continuation(model_dir: str, protein_id: str):
#     """
#     Load the true reference continuation FASTA for a gene, if the
#     generation run used --prefix_frac < 1.0. Returns None if not found
#     (e.g. legacy whole-CDS-prompt runs), in which case callers should
#     fall back to the pairwise self-comparison spectrum.
#     """
#     ref_path = os.path.join(model_dir, f"{protein_id}_expected_continuation.fasta")
#     if not os.path.isfile(ref_path):
#         return None
#     records = list(SeqIO.parse(ref_path, "fasta"))
#     if not records:
#         return None
#     return str(records[0].seq).upper()


# def nnls_decompose(spectrum: pd.Series, signatures: pd.DataFrame):
#     common = signatures.index.intersection(spectrum.index)
#     if len(common) == 0:
#         raise ValueError(
#             "No overlapping trinucleotide keys between computed spectrum "
#             "and COSMIC signature matrix -- check the context key format "
#             "in --cosmic_signatures (expects e.g. 'A[C>A]A')."
#         )
#     sig_matrix = signatures.loc[common].values
#     obs = spectrum.loc[common].values
#     total = obs.sum()
#     if total == 0:
#         return pd.Series(0.0, index=signatures.columns), None
#     obs_norm = obs / total
#     exposures, residual = nnls(sig_matrix, obs_norm)
#     exp_sum = exposures.sum()
#     if exp_sum > 0:
#         exposures = exposures / exp_sum
#     return pd.Series(exposures, index=signatures.columns), residual


# # ── ESM-2 embedding / centroid comparison ───────────────────────────
# _MATRIX_KEY_CANDIDATES = ["embeddings", "centroids", "X", "matrix", "vectors"]
# _ID_KEY_CANDIDATES = ["ids", "domain_ids", "labels", "names", "pfam_ids"]
# _PFAM_ID_RE = re.compile(r"^PF\d+$")


# def _extract_matrix_and_ids(data, source_desc: str):
#     keys = list(data.keys())

#     # Schema actually observed in this project's centroid chunks: each
#     # key IS a Pfam domain ID (e.g. 'PF00001') and its value is that
#     # domain's (320,) embedding vector directly -- there's no separate
#     # combined 'embeddings'/'ids' pair.
#     pfam_like_keys = [k for k in keys if _PFAM_ID_RE.match(k)]
#     if len(pfam_like_keys) >= max(1, 0.5 * len(keys)):
#         ids = pfam_like_keys
#         vectors = [np.asarray(data[k], dtype=np.float32).reshape(-1) for k in ids]
#         matrix = np.stack(vectors, axis=0)
#         return matrix, ids

#     # Fallback schema: one combined embeddings matrix + one id list.
#     matrix_key = next((k for k in _MATRIX_KEY_CANDIDATES if k in data), None)
#     id_key = next((k for k in _ID_KEY_CANDIDATES if k in data), None)
#     if matrix_key is None or id_key is None:
#         raise ValueError(
#             f"Could not find embedding matrix / id list in {source_desc}. "
#             f"Keys present: {list(data.keys())[:20]}"
#             f"{' ...' if len(data.keys()) > 20 else ''}. "
#             f"Update _extract_matrix_and_ids() to match, "
#             f"or share the key names and I'll fix this."
#         )
#     return np.asarray(data[matrix_key], dtype=np.float32), list(data[id_key])


# def load_centroid_chunks(chunks_dir: str):
#     """
#     Load and concatenate the PRISM Pfam centroid library from a directory
#     of chunked .npz files (e.g. centroid_chunk_001.npz, ...002.npz, ...).
#     Each chunk is expected to hold a (subset_M x 320) embedding matrix and
#     a parallel list of Pfam domain IDs.
#     """
#     chunk_paths = sorted(
#         os.path.join(chunks_dir, f) for f in os.listdir(chunks_dir) if f.endswith(".npz")
#     )
#     if not chunk_paths:
#         raise FileNotFoundError(f"No .npz files found in {chunks_dir}")

#     matrices, all_ids = [], []
#     for path in chunk_paths:
#         data = np.load(path, allow_pickle=True)
#         matrix, ids = _extract_matrix_and_ids(data, path)
#         matrices.append(matrix)
#         all_ids.extend(ids)

#     full_matrix = np.concatenate(matrices, axis=0)
#     full_matrix = full_matrix / np.linalg.norm(full_matrix, axis=1, keepdims=True)
#     print(f"[INFO] Loaded {len(chunk_paths)} centroid chunk(s) -> "
#           f"{full_matrix.shape[0]} total domains, {full_matrix.shape[1]}-d")
#     return full_matrix, all_ids


# def translate_orf(seq_str: str, min_aa: int):
#     seq_str = "".join(seq_str.split()).upper()
#     # Trim to a multiple of 3 so BioPython doesn't warn/fail on translate.
#     seq_str = seq_str[: len(seq_str) - (len(seq_str) % 3)]
#     if not seq_str:
#         return None
#     try:
#         protein = str(Seq(seq_str).translate(to_stop=True))
#     except Exception:
#         return None
#     if len(protein) < min_aa:
#         return None
#     return protein


# def embed_esm2(sequences, tokenizer, model, device, batch_size):
#     embeddings = []
#     for i in range(0, len(sequences), batch_size):
#         batch = sequences[i : i + batch_size]
#         inputs = tokenizer(
#             batch, return_tensors="pt", padding=True, truncation=True, max_length=1022,
#         )
#         inputs = {k: v.to(device) for k, v in inputs.items()}
#         with torch.no_grad():
#             out = model(**inputs)
#         hidden = out.last_hidden_state  # (B, L, D)
#         mask = inputs["attention_mask"].unsqueeze(-1)
#         pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)
#         embeddings.append(pooled.cpu().numpy())
#     if not embeddings:
#         return np.zeros((0, model.config.hidden_size), dtype=np.float32)
#     return np.concatenate(embeddings, axis=0)


# def creativity_and_embedding_diversity(embeddings, centroid_matrix, centroid_ids):
#     if embeddings.shape[0] == 0:
#         return 0, 0.0, None
#     norm_emb = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
#     sims_to_centroids = norm_emb @ centroid_matrix.T
#     nearest_idx = sims_to_centroids.argmax(axis=1)
#     nearest_domains = [centroid_ids[i] for i in nearest_idx]
#     n_unique_domains = len(set(nearest_domains))

#     if norm_emb.shape[0] > 1:
#         sims_pairwise = norm_emb @ norm_emb.T
#         iu = np.triu_indices_from(sims_pairwise, k=1)
#         mean_cos_dist = float(1.0 - sims_pairwise[iu].mean())
#     else:
#         mean_cos_dist = 0.0

#     top_domain = pd.Series(nearest_domains).value_counts().idxmax()
#     return n_unique_domains, mean_cos_dist, top_domain


# # ── main ─────────────────────────────────────────────────────────────
# def load_model_dirs(model_dir_args):
#     """Parse label:path CLI args into {label: path} and validate."""
#     model_dirs = {}
#     for entry in model_dir_args:
#         if ":" not in entry:
#             raise ValueError(
#                 f"--model_dir expects 'label:path', got: {entry!r}"
#             )
#         label, path = entry.split(":", 1)
#         if not os.path.isdir(path):
#             raise FileNotFoundError(f"Model output directory not found: {path}")
#         model_dirs[label] = path
#     return model_dirs


# def main(argv=None):
#     args = parse_args(argv)
#     os.makedirs(args.output_dir, exist_ok=True)
#     rng = random.Random(args.seed)
#     aligner = make_aligner()

#     print("[INFO] Loading COSMIC signature profile files...")
#     signatures = load_cosmic_signatures_dir(args.cosmic_dir)
#     print(f"[INFO] Loaded {signatures.shape[1]} signatures x {signatures.shape[0]} contexts")

#     print("[INFO] Loading Pfam centroid library chunks...")
#     centroid_matrix, centroid_ids = load_centroid_chunks(args.centroid_chunks_dir)

#     if AutoTokenizer is None:
#         sys.exit("transformers is required (pip install transformers)")

#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     print(f"[INFO] Loading ESM-2 ({args.esm_model}) on {device}...")
#     tokenizer = AutoTokenizer.from_pretrained(args.esm_model)
#     model = AutoModel.from_pretrained(args.esm_model).to(device).eval()

#     model_dirs = load_model_dirs(args.model_dir)

#     exposure_rows = []
#     creativity_rows = []
#     spectrum_rows = []

#     for label, model_path in model_dirs.items():
#         fasta_files = sorted(f for f in os.listdir(model_path) if f.endswith("_generated.fasta"))
#         print(f"\n[INFO] Model '{label}': {len(fasta_files)} protein file(s) in {model_path}")

#         for fname in fasta_files:
#             protein_id = fname[: -len("_generated.fasta")]
#             fasta_path = os.path.join(model_path, fname)
#             records = list(SeqIO.parse(fasta_path, "fasta"))
#             raw_sequences = [str(r.seq).upper() for r in records if len(str(r.seq)) > 0]
#             sequences, n_dropped = filter_degenerate(raw_sequences)

#             if n_dropped > 0:
#                 print(f"  [warn] {label}/{protein_id}: dropped {n_dropped}/{len(raw_sequences)} "
#                       f"degenerate (homopolymer-collapsed) sample(s)")

#             if len(sequences) < 2:
#                 print(f"  [warn] {label}/{protein_id}: fewer than 2 usable samples after "
#                       f"filtering, skipping")
#                 continue

#             # ── substitution spectrum + NNLS ──────────────────────
#             # Prefer reference-based analysis (align each sample against
#             # the TRUE gene continuation) when available -- this requires
#             # generation to have been run with --prefix_frac < 1.0. Falls
#             # back to legacy pairwise self-comparison otherwise (e.g. old
#             # whole-CDS-prompt runs, where no true continuation exists).
#             reference_seq = load_expected_continuation(model_path, protein_id)
#             used_reference_mode = reference_seq is not None and len(reference_seq) > 0

#             mean_identity = None
#             identity_std = None

#             if used_reference_mode:
#                 spectrum = build_spectrum_reference(
#                     sequences, reference_seq, aligner, args.max_samples_reference, rng,
#                 )
#                 identities = compute_identity_to_reference(
#                     sequences, reference_seq, aligner, args.max_samples_reference, rng,
#                 )
#                 mean_identity = float(np.mean(identities)) if identities else None
#                 identity_std = float(np.std(identities)) if identities else None
#                 mean_hamming = (1.0 - mean_identity) if mean_identity is not None else None
#             else:
#                 spectrum = build_spectrum(sequences, aligner, args.max_pairs_substitution, rng)
#                 ham_pairs = sample_pairs(len(sequences), args.max_pairs_hamming, rng)
#                 ham_dists = [
#                     d for d in (hamming_fraction(sequences[i], sequences[j], aligner) for i, j in ham_pairs)
#                     if d is not None
#                 ]
#                 mean_hamming = float(np.mean(ham_dists)) if ham_dists else None

#             exposures, residual = nnls_decompose(spectrum, signatures)

#             exposure_row = exposures.to_dict()
#             exposure_row.update({"model": label, "protein_id": protein_id,
#                                   "n_samples": len(sequences),
#                                   "n_raw_samples": len(raw_sequences),
#                                   "n_dropped_degenerate": n_dropped,
#                                   "total_substitutions": int(spectrum.sum()),
#                                   "nnls_residual": residual,
#                                   "analysis_mode": "reference" if used_reference_mode else "pairwise",
#                                   "mean_identity_to_reference": mean_identity,
#                                   "identity_std": identity_std})
#             exposure_rows.append(exposure_row)

#             spectrum_row = spectrum.to_dict()
#             spectrum_row.update({"model": label, "protein_id": protein_id})
#             spectrum_rows.append(spectrum_row)

#             # ── ESM-2 creativity + embedding diversity ────────────
#             proteins = [translate_orf(s, args.min_orf_aa) for s in sequences]
#             n_translatable = sum(1 for p in proteins if p is not None)
#             translation_rate = n_translatable / len(sequences) if sequences else 0.0
#             proteins = [p for p in proteins if p is not None]
#             embeddings = embed_esm2(proteins, tokenizer, model, device, args.batch_size)
#             n_unique_domains, mean_cos_dist, top_domain = creativity_and_embedding_diversity(
#                 embeddings, centroid_matrix, centroid_ids,
#             )

#             creativity_rows.append({
#                 "model": label,
#                 "protein_id": protein_id,
#                 "n_samples": len(sequences),
#                 "n_translatable": n_translatable,
#                 "translation_success_rate": translation_rate,
#                 "n_unique_domains_attracted": n_unique_domains,
#                 "top_attracted_domain": top_domain,
#                 "mean_esm2_cosine_distance": mean_cos_dist,
#                 "mean_hamming_distance": mean_hamming,
#                 "mean_identity_to_reference": mean_identity,
#                 "analysis_mode": "reference" if used_reference_mode else "pairwise",
#             })

#             print(f"  {label}/{protein_id}: {len(sequences)} samples "
#                   f"({'reference' if used_reference_mode else 'pairwise'} mode), "
#                   f"{int(spectrum.sum())} substitutions tallied, "
#                   f"translation_rate={translation_rate:.0%}, "
#                   f"{n_unique_domains} unique domain(s) attracted, "
#                   f"mean identity/Hamming={mean_hamming}")

#     exposures_df = pd.DataFrame(exposure_rows)
#     creativity_df = pd.DataFrame(creativity_rows)
#     spectra_df = pd.DataFrame(spectrum_rows)

#     exposures_path = os.path.join(args.output_dir, "signature_exposures.csv")
#     creativity_path = os.path.join(args.output_dir, "creativity_diversity.csv")
#     spectra_path = os.path.join(args.output_dir, "substitution_spectra.csv")

#     exposures_df.to_csv(exposures_path, index=False)
#     creativity_df.to_csv(creativity_path, index=False)
#     spectra_df.to_csv(spectra_path, index=False)

#     print(f"\n[INFO] Wrote {exposures_path}")
#     print(f"[INFO] Wrote {creativity_path}")
#     print(f"[INFO] Wrote {spectra_path}")


# if __name__ == "__main__":
#     main()

#!/usr/bin/env python3
"""
prism_genmodel_analysis.py

Tests whether generative DNA models (Carbon, GENERator, ...) have
implicitly learned COSMIC mutational-signature biases from training
data, following the PRISM pipeline's conventions (ESM-2 8M embeddings,
Pfam centroid library, per-protein analysis).

This is the step that runs AFTER generate_batch_carbon.py /
generate_batch_generator.py have produced <output_dir>/<record_id>_generated.fasta
files for each model.

WHY PAIRWISE-VS-PAIRWISE, NOT VS-WILD-TYPE
-------------------------------------------
generate_batch_carbon.py / generate_batch_generator.py feed the entire
wild-type sequence as the prompt and return a *novel continuation* --
there is no ground-truth continuation of the real gene to diff against.
So instead of comparing each sample to the wild-type, this script
aligns generated samples against EACH OTHER (all pairs, or a random
subsample of pairs if N is large) and tallies where they diverge. That
divergence pattern -- aggregated over many pairs -- is treated as the
model's implicit mutational spectrum for that protein.

PIPELINE (per protein, per model)
----------------------------------
1. Load the N generated samples from <model_dir>/<record_id>_generated.fasta
2. Pairwise-align a subsample of samples (Bio.Align, global alignment).
   For every mismatched column that is NOT adjacent to a gap, extract
   the trinucleotide context and orient it onto the pyrimidine
   (C/T) reference strand per COSMIC convention.
3. Aggregate mismatches into a 96-channel substitution spectrum and
   NNLS-decompose it against the COSMIC SBS signature matrix to get a
   per-protein, per-model signature exposure vector.
4. Translate each sample (frame 0, truncate at first stop codon) and
   embed with ESM-2 (facebook/esm2_t6_8M_UR50D, matching the PRISM
   paper). Score:
     - "creativity": number of distinct Pfam domain centroids the
       sample cloud is nearest to (more spread = more "creative")
     - mean pairwise cosine distance among sample embeddings
5. Compute mean pairwise Hamming distance (over aligned, ungapped
   columns) as a sequence-level diversity metric independent of ESM-2.

OUTPUTS (written to --output_dir)
-----------------------------------
signature_exposures.csv     one row per (model, protein), columns = SBS exposures
creativity_diversity.csv    one row per (model, protein)
substitution_spectra.csv    raw 96-channel counts, for debugging/inspection

USAGE
-----
python prism_genmodel_analysis.py \
    --model_dir carbon:/content/drive/MyDrive/PRISM/genmodel_bias/carbon_output \
    --model_dir generator:/content/drive/MyDrive/PRISM/genmodel_bias/generator_output \
    --cosmic_dir /content/drive/MyDrive/PRISM/cosmic_signatures \
    --centroid_chunks_dir /content/drive/MyDrive/PRISM/centroids/centroid_chunks \
    --output_dir /content/drive/MyDrive/PRISM/genmodel_bias/results
"""

import argparse
import itertools
import os
import random
import re
import sys

import numpy as np
import pandas as pd
import torch
from Bio import SeqIO, Align
from Bio.Seq import Seq
from scipy.optimize import nnls

try:
    from transformers import AutoTokenizer, AutoModel
except ImportError:
    AutoTokenizer = AutoModel = None


COMPLEMENT = {"A": "T", "T": "A", "C": "G", "G": "C"}


# ── CLI ──────────────────────────────────────────────────────────────
def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Test generative DNA models for implicit COSMIC signature bias.",
    )
    p.add_argument(
        "--model_dir", action="append", required=True,
        help="label:path pairs, e.g. carbon:/path/to/carbon_output. "
             "Repeat this flag once per model (Carbon, GENERator, ...).",
    )
    p.add_argument(
        "--cosmic_dir", type=str, required=True,
        help="Root directory containing individual COSMIC signature "
             "profile files (e.g. SBS2_PROFILE.txt, SBS17a_PROFILE.txt, ...). "
             "Subfolders (e.g. Chemotherapy/, Clock-like/, Mismatch/) are "
             "searched recursively.",
    )
    p.add_argument(
        "--centroid_chunks_dir", type=str, required=True,
        help="Directory containing the PRISM Pfam centroid library split "
             "across multiple .npz chunk files. All chunks are loaded and "
             "concatenated into one (M x 320) matrix.",
    )
    p.add_argument(
        "--esm_model", type=str, default="facebook/esm2_t6_8M_UR50D",
        help="ESM-2 checkpoint (default matches the PRISM paper's 8M model).",
    )
    p.add_argument(
        "--max_pairs_substitution", type=int, default=200,
        help="Max number of sample pairs to align per protein for the "
             "substitution spectrum (default: 200). Alignment is O(N^2) "
             "in the number of samples, so this caps runtime.",
    )
    p.add_argument(
        "--max_pairs_hamming", type=int, default=300,
        help="Max number of sample pairs to align per protein for the "
             "Hamming diversity metric (default: 300). Only used in "
             "legacy pairwise mode (no reference continuation available).",
    )
    p.add_argument(
        "--max_samples_reference", type=int, default=200,
        help="Max number of samples to align against the true reference "
             "continuation, when available (default: 200). Reference "
             "mode is O(N) so this can be set higher than the legacy "
             "pairwise caps without much added cost.",
    )
    p.add_argument(
        "--min_orf_aa", type=int, default=10,
        help="Minimum translated ORF length (in amino acids) to keep a "
             "sample for ESM-2 embedding (default: 10).",
    )
    p.add_argument(
        "--batch_size", type=int, default=8,
        help="ESM-2 embedding batch size (default: 8).",
    )
    p.add_argument(
        "--seed", type=int, default=0,
        help="Random seed for pair subsampling (default: 0).",
    )
    p.add_argument(
        "--output_dir", type=str, required=True,
        help="Directory to write result CSVs to.",
    )
    return p.parse_args(argv)


# ── COSMIC signature matrix (individual per-signature profile files) ─
_CONTEXT_KEY_RE = None  # set below, after build_96_context_keys is defined


def parse_profile_file(path: str) -> pd.Series:
    """
    Parse a single COSMIC '<SIGNATURE>_PROFILE.txt' file into a Series
    indexed by the 96 trinucleotide context keys ('A[C>A]A' format).

    Handles the common COSMIC export schemas:
      (a) two columns, first already in 'A[C>A]A' format
      (b) separate 'Substitution Type' (e.g. 'C>A') and 'Trinucleotide'
          (e.g. 'ACA') columns, plus one value column
      (c) 'Type'/'MutationType' + value, same as (a) under a different header
    Raises ValueError with a preview of the parsed columns if none of
    these schemas match, so the mismatch is easy to diagnose.
    """
    df = None
    for sep in ["\t", None, ",", r"\s+"]:
        try:
            candidate = pd.read_csv(path, sep=sep, engine="python")
        except Exception:
            continue
        if candidate.shape[1] >= 2 and candidate.shape[0] >= 90:
            df = candidate
            break
    if df is None:
        raise ValueError(f"Could not parse {path} with any common separator.")

    cols_lower = [c.strip().lower() for c in df.columns]
    valid_keys = set(build_96_context_keys())

    # Schema (a)/(c): a column already contains 'A[C>A]A'-style strings.
    for col in df.columns:
        sample_vals = df[col].astype(str).str.strip()
        if sample_vals.isin(valid_keys).sum() >= 90:
            value_col = [c for c in df.columns if c != col][0]
            series = pd.Series(df[value_col].values, index=sample_vals.values)
            return series.astype(float)

    # Schema (b): separate substitution-type and trinucleotide columns.
    sub_col = next((df.columns[i] for i, c in enumerate(cols_lower)
                     if "substitution" in c or c == "type"), None)
    tri_col = next((df.columns[i] for i, c in enumerate(cols_lower)
                     if "trinucleotide" in c or "context" in c), None)
    if sub_col is not None and tri_col is not None:
        value_col = [c for c in df.columns if c not in (sub_col, tri_col)][0]
        keys = [
            f"{tri[0]}[{sub}]{tri[2]}"
            for sub, tri in zip(df[sub_col].astype(str).str.strip(),
                                 df[tri_col].astype(str).str.strip())
        ]
        if sum(k in valid_keys for k in keys) >= 90:
            return pd.Series(df[value_col].values, index=keys).astype(float)

    raise ValueError(
        f"Could not identify context/value columns in {path}.\n"
        f"Columns found: {list(df.columns)}\n"
        f"First rows:\n{df.head(3)}\n"
        f"Update parse_profile_file() to match this schema, or share a "
        f"sample of this file's contents and I'll fix the parser."
    )


def load_cosmic_signatures_dir(root_dir: str) -> pd.DataFrame:
    """
    Recursively find all '<SIGNATURE>_PROFILE.txt' files under root_dir
    (including category subfolders like Chemotherapy/, Clock-like/,
    Mismatch/) and assemble them into one DataFrame: rows = 96 contexts,
    columns = signature names (e.g. SBS2, SBS17a, ...).
    """
    profile_paths = []
    for dirpath, _, filenames in os.walk(root_dir):
        for fname in filenames:
            if fname.upper().endswith("_PROFILE.TXT"):
                profile_paths.append(os.path.join(dirpath, fname))
    if not profile_paths:
        raise FileNotFoundError(
            f"No '*_PROFILE.txt' files found under {root_dir} "
            f"(searched recursively)."
        )

    columns = {}
    for path in sorted(profile_paths):
        sig_name = os.path.basename(path)
        sig_name = sig_name[: -len("_PROFILE.txt")] if sig_name.upper().endswith("_PROFILE.TXT") \
            else sig_name.rsplit(".", 1)[0]
        columns[sig_name] = parse_profile_file(path)

    df = pd.DataFrame(columns)
    df = df.reindex(build_96_context_keys())  # enforce consistent row order
    if df.isna().any().any():
        missing = df.columns[df.isna().any()].tolist()
        raise ValueError(
            f"Some signatures are missing context rows after alignment: "
            f"{missing}. Check that those profile files cover all 96 contexts."
        )
    # Normalize each signature to sum to 1.
    df = df.div(df.sum(axis=0), axis=1)
    return df


def build_96_context_keys():
    """All 96 trinucleotide substitution keys in 'X[R>A]Y' format."""
    bases = ["A", "C", "G", "T"]
    keys = []
    for ref in ["C", "T"]:
        for alt in bases:
            if alt == ref:
                continue
            for five in bases:
                for three in bases:
                    keys.append(f"{five}[{ref}>{alt}]{three}")
    return keys


def canonicalize_substitution(five, ref, alt, three):
    """
    Fold a substitution onto the pyrimidine (C/T) reference strand per
    COSMIC convention. If ref is a purine (A/G), take the reverse
    complement of the trinucleotide and complement the alt base.
    """
    if ref in ("C", "T"):
        return five, ref, alt, three
    return COMPLEMENT[three], COMPLEMENT[ref], COMPLEMENT[alt], COMPLEMENT[five]


def is_degenerate(seq: str, max_homopolymer_frac: float = 0.3) -> bool:
    """
    Flag sequences dominated by a single-base repeat run (a common
    autoregressive-generation failure mode: "repetition collapse").
    Such sequences produce spurious, gene-agnostic substitution signal
    when pairwise-aligned against normal sequences, so they're excluded
    before the spectrum is built.
    """
    if not seq:
        return True
    longest_run = max(len(m.group(0)) for m in re.finditer(r"([ACGTN])\1*", seq))
    return (longest_run / len(seq)) > max_homopolymer_frac


def filter_degenerate(sequences: list) -> tuple:
    """Return (kept_sequences, n_dropped)."""
    kept = [s for s in sequences if not is_degenerate(s)]
    return kept, len(sequences) - len(kept)


# ── alignment-based substitution / Hamming extraction ───────────────
def make_aligner():
    aligner = Align.PairwiseAligner()
    aligner.mode = "global"
    aligner.match_score = 2
    aligner.mismatch_score = -1
    # Loosened relative to a stricter default: a mismatch-averse aligner
    # forces small length differences (e.g. from residual homopolymer
    # noise) to be represented as strings of substitutions instead of a
    # single gap, which inflates the substitution spectrum with artifacts.
    aligner.open_gap_score = -2
    aligner.extend_gap_score = -0.2
    return aligner


def extract_substitutions(seq_a: str, seq_b: str, aligner) -> list:
    """
    Align two DNA sequences and return canonicalized (five, ref, alt, three)
    tuples for every mismatch column whose immediate flanking columns (on
    both sequences) are gap-free, so trinucleotide context is well-defined.
    """
    alignment = aligner.align(seq_a, seq_b)[0]
    a_aligned, b_aligned = str(alignment[0]), str(alignment[1])
    subs = []
    for i in range(1, len(a_aligned) - 1):
        a5, a0, a3 = a_aligned[i - 1], a_aligned[i], a_aligned[i + 1]
        b5, b0, b3 = b_aligned[i - 1], b_aligned[i], b_aligned[i + 1]
        # Skip columns with a gap OR an ambiguous base ('N') in the
        # 3-column window on either sequence -- 'N' has no defined
        # complement and isn't a real substitution to tally.
        if any(c not in "ACGT" for c in (a5, a0, a3, b5, b0, b3)):
            continue
        if a0 == b0:
            continue
        five, ref, alt, three = canonicalize_substitution(a5, a0, b0, a3)
        subs.append((five, ref, alt, three))
    return subs


def hamming_fraction(seq_a: str, seq_b: str, aligner) -> float:
    """Mismatch fraction over ungapped aligned columns."""
    alignment = aligner.align(seq_a, seq_b)[0]
    a_aligned, b_aligned = str(alignment[0]), str(alignment[1])
    mismatches = 0
    length = 0
    for x, y in zip(a_aligned, b_aligned):
        if x == "-" or y == "-":
            continue
        length += 1
        if x != y:
            mismatches += 1
    return mismatches / length if length > 0 else None


def sample_pairs(n_items: int, max_pairs: int, rng: random.Random):
    all_pairs = list(itertools.combinations(range(n_items), 2))
    if len(all_pairs) > max_pairs:
        all_pairs = rng.sample(all_pairs, max_pairs)
    return all_pairs


def build_spectrum(sequences, aligner, max_pairs, rng) -> pd.Series:
    """Legacy pairwise-self-comparison spectrum (used when no reference
    continuation is available, e.g. old whole-CDS-prompt runs)."""
    keys = build_96_context_keys()
    counts = pd.Series(0, index=keys, dtype=float)
    pairs = sample_pairs(len(sequences), max_pairs, rng)
    for i, j in pairs:
        for five, ref, alt, three in extract_substitutions(sequences[i], sequences[j], aligner):
            key = f"{five}[{ref}>{alt}]{three}"
            if key in counts.index:
                counts[key] += 1
    return counts


def build_spectrum_reference(sequences, reference_seq, aligner, max_samples, rng) -> pd.Series:
    """
    Reference-based spectrum: align each generated sample against the
    TRUE continuation of the real gene (one alignment per sample, O(N)
    instead of O(N^2)) and tally substitutions relative to that real
    sequence. Only usable when a prefix-based prompt design (see
    generate_batch_carbon.py / generate_batch_generator.py --prefix_frac)
    was used to generate the samples, since that's what leaves a real
    continuation to align against.
    """
    keys = build_96_context_keys()
    counts = pd.Series(0, index=keys, dtype=float)
    subset = sequences if len(sequences) <= max_samples else rng.sample(sequences, max_samples)
    for sample_seq in subset:
        for five, ref, alt, three in extract_substitutions(reference_seq, sample_seq, aligner):
            key = f"{five}[{ref}>{alt}]{three}"
            if key in counts.index:
                counts[key] += 1
    return counts


def compute_identity_to_reference(sequences, reference_seq, aligner, max_samples, rng):
    """Per-sample identity fraction (matches / aligned ungapped length)
    against the true reference continuation. Returns the list of
    per-sample identities (not just the mean) so callers can report
    both central tendency and spread."""
    subset = sequences if len(sequences) <= max_samples else rng.sample(sequences, max_samples)
    identities = []
    for sample_seq in subset:
        frac = hamming_fraction(reference_seq, sample_seq, aligner)
        if frac is not None:
            identities.append(1.0 - frac)
    return identities


def load_expected_continuation(model_dir: str, protein_id: str):
    """
    Load the true reference continuation FASTA for a gene, if the
    generation run used --prefix_frac < 1.0. Returns None if not found
    (e.g. legacy whole-CDS-prompt runs), in which case callers should
    fall back to the pairwise self-comparison spectrum.
    """
    ref_path = os.path.join(model_dir, f"{protein_id}_expected_continuation.fasta")
    if not os.path.isfile(ref_path):
        return None
    records = list(SeqIO.parse(ref_path, "fasta"))
    if not records:
        return None
    return str(records[0].seq).upper()


def nnls_decompose(spectrum: pd.Series, signatures: pd.DataFrame):
    common = signatures.index.intersection(spectrum.index)
    if len(common) == 0:
        raise ValueError(
            "No overlapping trinucleotide keys between computed spectrum "
            "and COSMIC signature matrix -- check the context key format "
            "in --cosmic_signatures (expects e.g. 'A[C>A]A')."
        )
    sig_matrix = signatures.loc[common].values
    obs = spectrum.loc[common].values
    total = obs.sum()
    if total == 0:
        return pd.Series(0.0, index=signatures.columns), None
    obs_norm = obs / total
    exposures, residual = nnls(sig_matrix, obs_norm)
    exp_sum = exposures.sum()
    if exp_sum > 0:
        exposures = exposures / exp_sum
    return pd.Series(exposures, index=signatures.columns), residual


# ── ESM-2 embedding / centroid comparison ───────────────────────────
_MATRIX_KEY_CANDIDATES = ["embeddings", "centroids", "X", "matrix", "vectors"]
_ID_KEY_CANDIDATES = ["ids", "domain_ids", "labels", "names", "pfam_ids"]
_PFAM_ID_RE = re.compile(r"^PF\d+$")


def _extract_matrix_and_ids(data, source_desc: str):
    keys = list(data.keys())

    # Schema actually observed in this project's centroid chunks: each
    # key IS a Pfam domain ID (e.g. 'PF00001') and its value is that
    # domain's (320,) embedding vector directly -- there's no separate
    # combined 'embeddings'/'ids' pair.
    pfam_like_keys = [k for k in keys if _PFAM_ID_RE.match(k)]
    if len(pfam_like_keys) >= max(1, 0.5 * len(keys)):
        ids = pfam_like_keys
        vectors = [np.asarray(data[k], dtype=np.float32).reshape(-1) for k in ids]
        matrix = np.stack(vectors, axis=0)
        return matrix, ids

    # Fallback schema: one combined embeddings matrix + one id list.
    matrix_key = next((k for k in _MATRIX_KEY_CANDIDATES if k in data), None)
    id_key = next((k for k in _ID_KEY_CANDIDATES if k in data), None)
    if matrix_key is None or id_key is None:
        raise ValueError(
            f"Could not find embedding matrix / id list in {source_desc}. "
            f"Keys present: {list(data.keys())[:20]}"
            f"{' ...' if len(data.keys()) > 20 else ''}. "
            f"Update _extract_matrix_and_ids() to match, "
            f"or share the key names and I'll fix this."
        )
    return np.asarray(data[matrix_key], dtype=np.float32), list(data[id_key])


def load_centroid_chunks(chunks_dir: str):
    """
    Load and concatenate the PRISM Pfam centroid library from a directory
    of chunked .npz files (e.g. centroid_chunk_001.npz, ...002.npz, ...).
    Each chunk is expected to hold a (subset_M x 320) embedding matrix and
    a parallel list of Pfam domain IDs.
    """
    chunk_paths = sorted(
        os.path.join(chunks_dir, f) for f in os.listdir(chunks_dir) if f.endswith(".npz")
    )
    if not chunk_paths:
        raise FileNotFoundError(f"No .npz files found in {chunks_dir}")

    matrices, all_ids = [], []
    for path in chunk_paths:
        data = np.load(path, allow_pickle=True)
        matrix, ids = _extract_matrix_and_ids(data, path)
        matrices.append(matrix)
        all_ids.extend(ids)

    full_matrix = np.concatenate(matrices, axis=0)
    full_matrix = full_matrix / np.linalg.norm(full_matrix, axis=1, keepdims=True)
    print(f"[INFO] Loaded {len(chunk_paths)} centroid chunk(s) -> "
          f"{full_matrix.shape[0]} total domains, {full_matrix.shape[1]}-d")
    return full_matrix, all_ids


def translate_orf(seq_str: str, min_aa: int):
    seq_str = "".join(seq_str.split()).upper()
    # Trim to a multiple of 3 so BioPython doesn't warn/fail on translate.
    seq_str = seq_str[: len(seq_str) - (len(seq_str) % 3)]
    if not seq_str:
        return None
    try:
        protein = str(Seq(seq_str).translate(to_stop=True))
    except Exception:
        return None
    if len(protein) < min_aa:
        return None
    return protein


def embed_esm2(sequences, tokenizer, model, device, batch_size):
    embeddings = []
    for i in range(0, len(sequences), batch_size):
        batch = sequences[i : i + batch_size]
        inputs = tokenizer(
            batch, return_tensors="pt", padding=True, truncation=True, max_length=1022,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            out = model(**inputs)
        hidden = out.last_hidden_state  # (B, L, D)
        mask = inputs["attention_mask"].unsqueeze(-1)
        pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)
        embeddings.append(pooled.cpu().numpy())
    if not embeddings:
        return np.zeros((0, model.config.hidden_size), dtype=np.float32)
    return np.concatenate(embeddings, axis=0)


def creativity_and_embedding_diversity(embeddings, centroid_matrix, centroid_ids):
    if embeddings.shape[0] == 0:
        return 0, 0.0, None
    norm_emb = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    sims_to_centroids = norm_emb @ centroid_matrix.T
    nearest_idx = sims_to_centroids.argmax(axis=1)
    nearest_domains = [centroid_ids[i] for i in nearest_idx]
    n_unique_domains = len(set(nearest_domains))

    if norm_emb.shape[0] > 1:
        sims_pairwise = norm_emb @ norm_emb.T
        iu = np.triu_indices_from(sims_pairwise, k=1)
        mean_cos_dist = float(1.0 - sims_pairwise[iu].mean())
    else:
        mean_cos_dist = 0.0

    top_domain = pd.Series(nearest_domains).value_counts().idxmax()
    return n_unique_domains, mean_cos_dist, top_domain


# ── main ─────────────────────────────────────────────────────────────
def load_model_dirs(model_dir_args):
    """Parse label:path CLI args into {label: path} and validate."""
    model_dirs = {}
    for entry in model_dir_args:
        if ":" not in entry:
            raise ValueError(
                f"--model_dir expects 'label:path', got: {entry!r}"
            )
        label, path = entry.split(":", 1)
        if not os.path.isdir(path):
            raise FileNotFoundError(f"Model output directory not found: {path}")
        model_dirs[label] = path
    return model_dirs


def main(argv=None):
    args = parse_args(argv)
    os.makedirs(args.output_dir, exist_ok=True)
    rng = random.Random(args.seed)
    aligner = make_aligner()

    print("[INFO] Loading COSMIC signature profile files...")
    signatures = load_cosmic_signatures_dir(args.cosmic_dir)
    print(f"[INFO] Loaded {signatures.shape[1]} signatures x {signatures.shape[0]} contexts")

    print("[INFO] Loading Pfam centroid library chunks...")
    centroid_matrix, centroid_ids = load_centroid_chunks(args.centroid_chunks_dir)

    if AutoTokenizer is None:
        sys.exit("transformers is required (pip install transformers)")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Loading ESM-2 ({args.esm_model}) on {device}...")
    tokenizer = AutoTokenizer.from_pretrained(args.esm_model)
    model = AutoModel.from_pretrained(args.esm_model).to(device).eval()

    model_dirs = load_model_dirs(args.model_dir)

    exposure_rows = []
    creativity_rows = []
    spectrum_rows = []

    for label, model_path in model_dirs.items():
        fasta_files = sorted(f for f in os.listdir(model_path) if f.endswith("_generated.fasta"))
        print(f"\n[INFO] Model '{label}': {len(fasta_files)} protein file(s) in {model_path}")

        for fname in fasta_files:
            protein_id = fname[: -len("_generated.fasta")]
            fasta_path = os.path.join(model_path, fname)
            records = list(SeqIO.parse(fasta_path, "fasta"))
            raw_sequences = [str(r.seq).upper() for r in records if len(str(r.seq)) > 0]
            sequences, n_dropped = filter_degenerate(raw_sequences)

            if n_dropped > 0:
                print(f"  [warn] {label}/{protein_id}: dropped {n_dropped}/{len(raw_sequences)} "
                      f"degenerate (homopolymer-collapsed) sample(s)")

            if len(sequences) < 2:
                print(f"  [warn] {label}/{protein_id}: fewer than 2 usable samples after "
                      f"filtering, skipping")
                continue

            # ── substitution spectrum + NNLS ──────────────────────
            # Prefer reference-based analysis (align each sample against
            # the TRUE gene continuation) when available -- this requires
            # generation to have been run with --prefix_frac < 1.0. Falls
            # back to legacy pairwise self-comparison otherwise (e.g. old
            # whole-CDS-prompt runs, where no true continuation exists).
            reference_seq = load_expected_continuation(model_path, protein_id)
            used_reference_mode = reference_seq is not None and len(reference_seq) > 0

            mean_identity = None
            identity_std = None

            if used_reference_mode:
                spectrum = build_spectrum_reference(
                    sequences, reference_seq, aligner, args.max_samples_reference, rng,
                )
                identities = compute_identity_to_reference(
                    sequences, reference_seq, aligner, args.max_samples_reference, rng,
                )
                mean_identity = float(np.mean(identities)) if identities else None
                identity_std = float(np.std(identities)) if identities else None
                mean_hamming = (1.0 - mean_identity) if mean_identity is not None else None
            else:
                spectrum = build_spectrum(sequences, aligner, args.max_pairs_substitution, rng)
                ham_pairs = sample_pairs(len(sequences), args.max_pairs_hamming, rng)
                ham_dists = [
                    d for d in (hamming_fraction(sequences[i], sequences[j], aligner) for i, j in ham_pairs)
                    if d is not None
                ]
                mean_hamming = float(np.mean(ham_dists)) if ham_dists else None

            exposures, residual = nnls_decompose(spectrum, signatures)

            exposure_row = exposures.to_dict()
            exposure_row.update({"model": label, "protein_id": protein_id,
                                  "n_samples": len(sequences),
                                  "n_raw_samples": len(raw_sequences),
                                  "n_dropped_degenerate": n_dropped,
                                  "total_substitutions": int(spectrum.sum()),
                                  "nnls_residual": residual,
                                  "analysis_mode": "reference" if used_reference_mode else "pairwise",
                                  "mean_identity_to_reference": mean_identity,
                                  "identity_std": identity_std})
            exposure_rows.append(exposure_row)

            spectrum_row = spectrum.to_dict()
            spectrum_row.update({"model": label, "protein_id": protein_id})
            spectrum_rows.append(spectrum_row)

            # ── ESM-2 creativity + embedding diversity ────────────
            proteins = [translate_orf(s, args.min_orf_aa) for s in sequences]
            n_translatable = sum(1 for p in proteins if p is not None)
            translation_rate = n_translatable / len(sequences) if sequences else 0.0
            proteins = [p for p in proteins if p is not None]
            embeddings = embed_esm2(proteins, tokenizer, model, device, args.batch_size)
            n_unique_domains, mean_cos_dist, top_domain = creativity_and_embedding_diversity(
                embeddings, centroid_matrix, centroid_ids,
            )

            creativity_rows.append({
                "model": label,
                "protein_id": protein_id,
                "n_samples": len(sequences),
                "n_translatable": n_translatable,
                "translation_success_rate": translation_rate,
                "n_unique_domains_attracted": n_unique_domains,
                "top_attracted_domain": top_domain,
                "mean_esm2_cosine_distance": mean_cos_dist,
                "mean_hamming_distance": mean_hamming,
                "mean_identity_to_reference": mean_identity,
                "analysis_mode": "reference" if used_reference_mode else "pairwise",
            })

            print(f"  {label}/{protein_id}: {len(sequences)} samples "
                  f"({'reference' if used_reference_mode else 'pairwise'} mode), "
                  f"{int(spectrum.sum())} substitutions tallied, "
                  f"translation_rate={translation_rate:.0%}, "
                  f"{n_unique_domains} unique domain(s) attracted, "
                  f"mean identity/Hamming={mean_hamming}")

    exposures_df = pd.DataFrame(exposure_rows)
    creativity_df = pd.DataFrame(creativity_rows)
    spectra_df = pd.DataFrame(spectrum_rows)

    exposures_path = os.path.join(args.output_dir, "signature_exposures.csv")
    creativity_path = os.path.join(args.output_dir, "creativity_diversity.csv")
    spectra_path = os.path.join(args.output_dir, "substitution_spectra.csv")

    exposures_df.to_csv(exposures_path, index=False)
    creativity_df.to_csv(creativity_path, index=False)
    spectra_df.to_csv(spectra_path, index=False)

    print(f"\n[INFO] Wrote {exposures_path}")
    print(f"[INFO] Wrote {creativity_path}")
    print(f"[INFO] Wrote {spectra_path}")


if __name__ == "__main__":
    main()