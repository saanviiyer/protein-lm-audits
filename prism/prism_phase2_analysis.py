# prism_phase2_analysis.py
import os
import csv
import time
import argparse
import pickle
import torch
from statsmodels.stats.multitest import multipletests

import prism_config as cfg
from prism_utils import (
    load_mutation_profile, mutate_cds, translate_cds,
    load_esm2, get_batch_embeddings, vectorized_cosine_distance,
    run_statistics,
)

def load_pairs(tsv_path: str) -> list:
    pairs = []
    with open(tsv_path) as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        for row in reader:
            pairs.append((
                row["uniprotA"],
                row["uniprotB"],
                {
                    "clan":        row.get("clan", ""),
                    "pfamA":       row.get("pfamA", ""),
                    "pfamB":       row.get("pfamB", ""),
                    "ecA":         row.get("uniprotA_EC", ""),
                    "ecB":         row.get("uniprotB_EC", ""),
                    "ec_diff_pos": row.get("EC_diff_position", ""),
                    "seq_id":      row.get("seq_identity", ""),
                },
            ))
    return pairs


def load_profiles(profile_dir: str) -> list:
    all_profiles = []
    for root, _, files in os.walk(profile_dir):
        for fname in sorted(files):
            if fname.startswith("."):
                continue

            cat = os.path.basename(root)
            if cat in ("profiles", "DBS-MS", "SBS-MS"):
                cat = "General"

            try:
                sig = load_mutation_profile(os.path.join(root, fname))
                if sig:
                    all_profiles.append((cat, fname, sig))
            except Exception:
                pass
    return all_profiles


def build_null_signature(all_profiles: list) -> dict:
    null_sig = {}
    for _, _, sig in all_profiles:
        for k, v in sig.items():
            null_sig[k] = null_sig.get(k, 0.0) + v
    return {k: v / len(all_profiles) for k, v in null_sig.items()}


def load_done_keys(csv_path: str) -> set:
    done = set()
    if not os.path.exists(csv_path):
        return done
    with open(csv_path) as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            done.add((row["pair"], row["profile"]))
    return done


CSV_COLS = [
    "pair", "accA", "accB",
    "clan", "pfamA", "pfamB", "ecA", "ecB", "ec_diff_pos", "seq_id",
    "category", "profile",
    "mean_treat", "mean_null",
    "t_p", "u_p", "perm_p",
    "significant_raw",
]


def apply_multiple_testing_correction(input_csv: str, output_csv: str):
    """
    Reads the raw results, applies FDR (Benjamini-Hochberg) and Bonferroni 
    corrections to the p-values using statsmodels, and writes a final annotated CSV.
    """
    print("\n[phase2] Applying multiple testing corrections (FDR and Bonferroni)...")
    rows = []
    with open(input_csv, "r") as fh:
        reader = csv.DictReader(fh)
        fieldnames = list(reader.fieldnames)
        for row in reader:
            rows.append(row)

    if not rows:
        print("[phase2] No data to correct.")
        return

    # Extract all p-values as floats
    t_p_vals    = [float(r["t_p"]) for r in rows]
    u_p_vals    = [float(r["u_p"]) for r in rows]
    perm_p_vals = [float(r["perm_p"]) for r in rows]

    # Apply FDR (False Discovery Rate) correction
    _, t_p_fdr, _, _    = multipletests(t_p_vals, method='fdr_bh')
    _, u_p_fdr, _, _    = multipletests(u_p_vals, method='fdr_bh')
    _, perm_p_fdr, _, _ = multipletests(perm_p_vals, method='fdr_bh')

    # Apply Bonferroni correction
    _, t_p_bonf, _, _    = multipletests(t_p_vals, method='bonferroni')
    _, u_p_bonf, _, _    = multipletests(u_p_vals, method='bonferroni')
    _, perm_p_bonf, _, _ = multipletests(perm_p_vals, method='bonferroni')

    # Extend headers for the final CSV
    new_cols = [
        "t_p_fdr", "u_p_fdr", "perm_p_fdr",
        "t_p_bonf", "u_p_bonf", "perm_p_bonf",
        "significant_fdr", "significant_bonf"
    ]
    fieldnames.extend(new_cols)

    hits_fdr = 0
    hits_bonf = 0

    with open(output_csv, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        
        for i, row in enumerate(rows):
            # Save the corrected p-values
            row["t_p_fdr"]    = t_p_fdr[i]
            row["u_p_fdr"]    = u_p_fdr[i]
            row["perm_p_fdr"] = perm_p_fdr[i]
            
            row["t_p_bonf"]    = t_p_bonf[i]
            row["u_p_bonf"]    = u_p_bonf[i]
            row["perm_p_bonf"] = perm_p_bonf[i]

            mean_treat = float(row["mean_treat"])
            mean_null  = float(row["mean_null"])

            # Recalculate significance flags using the corrected p-values
            sig_fdr = (
                t_p_fdr[i] < cfg.ALPHA and
                u_p_fdr[i] < cfg.ALPHA and
                perm_p_fdr[i] < cfg.ALPHA and
                mean_treat < mean_null
            )
            sig_bonf = (
                t_p_bonf[i] < cfg.ALPHA and
                u_p_bonf[i] < cfg.ALPHA and
                perm_p_bonf[i] < cfg.ALPHA and
                mean_treat < mean_null
            )

            row["significant_fdr"]  = sig_fdr
            row["significant_bonf"] = sig_bonf

            if sig_fdr:  hits_fdr += 1
            if sig_bonf: hits_bonf += 1

            writer.writerow(row)

    print(f"[phase2] Correction complete.")
    print(f"  -> FDR Hits:        {hits_fdr}")
    print(f"  -> Bonferroni Hits: {hits_bonf}")
    print(f"[phase2] Final corrected results saved to {output_csv}")


def run_analysis(pairs: list, embeddings: dict, all_profiles: list,
                 null_sig: dict, tokenizer, model, csv_path: str,
                 max_pairs: int = None):

    valid_pairs = [
        (a, b, meta) for (a, b, meta) in pairs
        if a in embeddings and b in embeddings
    ]
    print(f"[phase2] {len(valid_pairs)}/{len(pairs)} pairs have embeddings.")

    if max_pairs:
        valid_pairs = valid_pairs[:max_pairs]
        print(f"[phase2] Dry-run: limiting to first {max_pairs} pairs.")

    done_keys = load_done_keys(csv_path)
    write_header = not os.path.exists(csv_path)

    total_experiments = len(valid_pairs) * len(all_profiles)
    print(f"[phase2] {len(valid_pairs)} pairs × {len(all_profiles)} profiles "
          f"= {total_experiments:,} experiments\n" + "=" * 60)

    t0 = time.time()
    exp_done = 0
    hits_raw = 0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Inference Loop
    with open(csv_path, "a", newline="") as csv_fh:
        writer = csv.writer(csv_fh)
        if write_header:
            writer.writerow(CSV_COLS)

        with torch.no_grad():
            for pair_idx, (acc_a, acc_b, meta) in enumerate(valid_pairs):
                entry_a = embeddings[acc_a]
                entry_b = embeddings[acc_b]
                nt_a    = entry_a["nt"]
                emb_b   = entry_b["emb"]

                pair_label = f"{acc_a}_vs_{acc_b}"

                null_aas = [aa for aa in
                            (translate_cds(mutate_cds(nt_a, null_sig)) for _ in range(cfg.N_RUNS))
                            if aa]
                
                if len(null_aas) < 5:
                    continue

                null_dists = vectorized_cosine_distance(
                    get_batch_embeddings(null_aas, tokenizer, model), emb_b
                ).tolist()

                for cat, prof_name, sig in all_profiles:
                    if (pair_label, prof_name) in done_keys:
                        exp_done += 1
                        continue

                    treat_aas = [aa for aa in
                                 (translate_cds(mutate_cds(nt_a, sig)) for _ in range(cfg.N_RUNS))
                                 if aa]
                    if len(treat_aas) < 5:
                        exp_done += 1
                        continue

                    treat_dists = vectorized_cosine_distance(
                        get_batch_embeddings(treat_aas, tokenizer, model), emb_b
                    ).tolist()

                    s = run_statistics(treat_dists, null_dists)

                    # Raw significance flag
                    sig_flag_raw = (
                        s["t_p"]            < cfg.ALPHA and
                        s["u_p"]            < cfg.ALPHA and
                        s["perm_p"]         < cfg.ALPHA and
                        s["mean_treatment"] < s["mean_null"]
                    )

                    writer.writerow([
                        pair_label, acc_a, acc_b,
                        meta["clan"], meta["pfamA"], meta["pfamB"],
                        meta["ecA"], meta["ecB"], meta["ec_diff_pos"], meta["seq_id"],
                        cat, prof_name,
                        s["mean_treatment"], s["mean_null"],
                        s["t_p"], s["u_p"], s["perm_p"],
                        sig_flag_raw,
                    ])
                    csv_fh.flush()
                    exp_done += 1

                    if sig_flag_raw:
                        hits_raw += 1
                        delta = s["mean_treatment"] - s["mean_null"]
                        print(f"*** RAW HIT [{cat}/{prof_name}] {pair_label} | Δ={delta:.4f} "
                              f"| p={s['perm_p']:.3e}")

                elapsed   = time.time() - t0
                rate      = exp_done / elapsed if elapsed > 0 else float("inf")
                remaining = (total_experiments - exp_done) / rate if rate > 0 else float("inf")
                print(f"  pair {pair_idx+1}/{len(valid_pairs)} done  "
                      f"| {exp_done:,}/{total_experiments:,} experiments  "
                      f"| {hits_raw} raw hits  "
                      f"| elapsed {elapsed/60:.1f} min  "
                      f"| ETA {remaining/60:.1f} min")

    print(f"\n[phase2] Inference complete. Found {hits_raw} raw significant hits.")
    
    # ── Post-processing: Multiple Testing Correction ──
    corrected_csv_path = os.path.join(os.path.dirname(csv_path), "results_corrected.csv")
    apply_multiple_testing_correction(csv_path, corrected_csv_path)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=str, required=True, help="Directory containing embeddings and pairs TSV")
    ap.add_argument("--dry-run", type=int, default=None, metavar="N", help="Limit to first N pairs")
    args = ap.parse_args()

    pairs_tsv = os.path.join(args.out_dir, "pairs.tsv")
    emb_path  = os.path.join(args.out_dir, "embeddings.pkl")
    csv_path  = os.path.join(args.out_dir, "results.csv")

    assert os.path.exists(pairs_tsv), f"Pairs TSV not found: {pairs_tsv}"
    assert os.path.exists(emb_path), f"Embeddings not found: {emb_path} — run phase1 first"
    assert os.path.exists(cfg.PROFILE_DIR), f"Profiles dir not found: {cfg.PROFILE_DIR}"

    pairs        = load_pairs(pairs_tsv)
    with open(emb_path, "rb") as f:
        embeddings = pickle.load(f)
        
    all_profiles = load_profiles(cfg.PROFILE_DIR)
    null_sig     = build_null_signature(all_profiles)

    tokenizer, model = load_esm2()
    run_analysis(pairs, embeddings, all_profiles, null_sig, tokenizer, model, csv_path, max_pairs=args.dry_run)