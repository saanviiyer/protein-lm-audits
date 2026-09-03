# ======================================================
#  PRISM PIPELINE — prism_config.py
#  Central configuration: paths, hyperparameters, device
# ======================================================

import os
import torch

# ── Data and Profile Paths ────────────────────────────────────────────────
# Output paths (embeddings, pairs, results) are handled dynamically via 
# argparse (--out-dir) in the main phase1 and phase2 execution scripts.

# Assumes the 'profiles' directory is in the same folder as the scripts.
PROFILE_DIR = "/content/saanvi/profiles"
CLAN_CACHE  = "clan_uniprot_cache.json"  # Speeds up re-runs

# ── Clan switch parameters ────────────────────────────────────────────────
CLAN            = "CL0036"   # Pfam clan to analyse
MIN_IDENTITY    = 0.60       # Minimum pairwise sequence identity to keep a pair
PER_FAMILY      = 25         # Max UniProt entries fetched per Pfam family
REVIEWED_ONLY   = True       # Restrict to Swiss-Prot reviewed entries
MAX_FAMILIES    = None       # Set an int to cap for quick testing

# ── ESM-2 ──────────────────────────────────────────────────────────────────
HF_MODEL_NAME = "facebook/esm2_t6_8M_UR50D"
DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Experimental parameters ────────────────────────────────────────────────
N_RUNS         = 10      # Mutated sequences generated per (pair, signature) condition
N_PERMUTATIONS = 10_000   # Permutation test iterations
ALPHA          = 0.05     # Significance threshold (all three tests must pass)

print(f"[config] Using device: {DEVICE}")
