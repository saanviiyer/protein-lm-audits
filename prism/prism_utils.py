# ======================================================
#  PRISM PIPELINE — prism_utils.py
#  Shared utilities: sequence ops, embeddings, statistics
# ======================================================

import random
import numpy as np
from scipy import stats
from Bio.Seq import Seq
import torch
from transformers import AutoTokenizer, EsmModel

from prism_config import DEVICE, N_PERMUTATIONS, HF_MODEL_NAME


# ── Sequence utilities ──────────────────────────────────────────────────────

def translate_cds(nt_seq: str) -> str:
    """Translate a nucleotide CDS to amino acids, trimming to a codon-complete length."""
    trimmed = nt_seq[: len(nt_seq) - (len(nt_seq) % 3)]
    return str(Seq(trimmed).translate(to_stop=True))


def parse_fasta(fasta_path: str) -> dict:
    """Parse a FASTA file into {header: sequence} dict."""
    seqs = {}
    with open(fasta_path) as f:
        header, buf = None, []
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if header:
                    seqs[header] = "".join(buf)
                header, buf = line[1:], []
            else:
                buf.append(line.upper())
        if header:
            seqs[header] = "".join(buf)
    return seqs


def load_mutation_profile(path: str) -> dict:
    """
    Load a COSMIC SBS mutational signature file.
    Expected format: two-column (key, probability), whitespace-separated.
    Key format:  X[Y>Z]W  where position 2 is the original base and position 4 is the mutant base.
    """
    sigs = {}
    with open(path) as f:
        next(f, None)   # skip header
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                sig_key = parts[0]
                if len(sig_key) >= 7 and sig_key[1] == '[' and sig_key[5] == ']':
                    sigs[sig_key] = float(parts[1])
    return sigs


def mutate_cds(nt_seq: str, profile: dict) -> str:
    """
    Apply a trinucleotide mutational signature to a nucleotide sequence.
    Each position is tested independently; the first matching signature wins.
    """
    seq = list(nt_seq)
    for idx in range(len(seq) - 2):
        context = "".join(seq[idx: idx + 3])
        for sig, prob in profile.items():
            if len(sig) >= 7:
                orig = sig[0] + sig[2] + sig[6]
                if context == orig and random.random() < prob:
                    seq[idx + 1] = sig[4]
                    break
    return "".join(seq)


# ── ESM-2 embedding ─────────────────────────────────────────────────────────

def load_esm2():
    """Load ESM-2 tokenizer and model onto the configured device."""
    tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_NAME)
    model     = EsmModel.from_pretrained(HF_MODEL_NAME).to(DEVICE).eval()
    return tokenizer, model


def get_single_embedding(aa_seq: str, tokenizer, model) -> np.ndarray:
    inputs = tokenizer(aa_seq, return_tensors="pt", add_special_tokens=True).to(DEVICE)
    with torch.no_grad():
        out = model(**inputs)
    return out.last_hidden_state[0, 1:-1].mean(0).cpu().numpy()


def get_batch_embeddings(aa_seqs: list, tokenizer, model) -> np.ndarray:
    """
    Embed a batch of amino acid sequences.
    Masks padding and BOS/EOS tokens; returns shape (N, hidden_dim).
    """
    hidden_dim = model.config.hidden_size
    
    if not aa_seqs:
        return np.zeros((0, hidden_dim))
        
    inputs = tokenizer(
        aa_seqs, return_tensors="pt", padding=True, add_special_tokens=True
    ).to(DEVICE)
    
    attention_mask = inputs["attention_mask"]
    
    with torch.no_grad():
        out = model(**inputs)
        
    reps = []
    for i in range(len(aa_seqs)):
        valid_len = attention_mask[i].sum().item() - 2   # exclude BOS and EOS
        if valid_len <= 0:
            reps.append(np.zeros(hidden_dim))
        else:
            reps.append(out.last_hidden_state[i, 1:valid_len + 1].mean(0).cpu().numpy())
            
    return np.array(reps)


def vectorized_cosine_distance(batch_embs: np.ndarray, target_emb: np.ndarray) -> np.ndarray:
    """Cosine distance (1 - cosine similarity) between each row of batch_embs and target_emb."""
    dot_products = np.dot(batch_embs, target_emb)
    norms_batch  = np.linalg.norm(batch_embs, axis=1)
    norm_target  = np.linalg.norm(target_emb)
    valid        = (norms_batch > 0) & (norm_target > 0)
    distances    = np.zeros(len(batch_embs))
    distances[valid] = 1.0 - (dot_products[valid] / (norms_batch[valid] * norm_target))
    return distances


# ── Statistics ───────────────────────────────────────────────────────────────

def run_statistics(treatment: list, null: list) -> dict:
    """
    Three-test battery comparing treatment vs null cosine distance distributions.
      - Welch's t-test
      - Mann-Whitney U (one-sided: treatment < null)
      - Permutation test (N_PERMUTATIONS shuffles; p = fraction of perms where
        shuffled mean-diff <= observed mean-diff)
    """
    _, t_p = stats.ttest_ind(treatment, null, equal_var=False)
    _, u_p = stats.mannwhitneyu(treatment, null, alternative="less")

    combined = np.array(treatment + null)
    n_obs    = len(treatment)
    obs_stat = np.mean(treatment) - np.mean(null)
    
    count = sum(
        1 for _ in range(N_PERMUTATIONS)
        if (np.random.shuffle(combined) or True)   # shuffle in-place, always True
        and np.mean(combined[:n_obs]) - np.mean(combined[n_obs:]) <= obs_stat
    )

    return {
        "t_p":            t_p,
        "u_p":            u_p,
        "perm_p":         count / N_PERMUTATIONS,
        "mean_treatment": float(np.mean(treatment)),
        "mean_null":      float(np.mean(null)),
    }
