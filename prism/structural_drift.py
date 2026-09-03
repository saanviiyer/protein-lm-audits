#!/usr/bin/env python3
"""
structural_drift.py

PHYSICS-ALIGNED readout of mutational-signature-driven drift, for the
SIMBIOCHEM ("ML for Simulations in Biology & Chemistry") framing.

PRISM's core measures where a COSMIC mutational signature pushes a protein in
ESM-2 *representation* space. That is a learned-embedding readout. SIMBIOCHEM
wants a *physical* one. This module keeps PRISM's mechanistic mutational
simulator (the per-trinucleotide Bernoulli COSMIC process in prism_utils) but
replaces the readout with a STRUCTURAL / ENERGETIC one:

    wild-type CDS --COSMIC signature (Monte-Carlo)--> mutant CDS
        |                                                   |
        v                                                   v
     translate                                          translate
        |                                                   |
        v                                                   v
     ESMFold(WT)  ------ structural comparison ------  ESMFold(mutant)
        |                                                   |
     pLDDT_wt                                            pLDDT_mut
     coords_wt   --Kabsch CA-RMSD, ΔpLDDT, truncation--  coords_mut

Per (gene, signature) it reports:
  * mean ΔpLDDT (mutant - WT): predicted-confidence / foldability change
  * CA-RMSD (Kabsch-superposed, over the common ungapped length): backbone drift
  * nonsense/truncation rate: fraction of signature draws that introduce a stop
  * (optional) an inverse-folding stability proxy hook (ESM-IF / FAMPNN)

This is a genuine physical axis: it asks whether a named chemical mutational
process (e.g. SBS17a) systematically destabilises or restructures a fold, not
merely whether an embedding moved. It is deliberately the same simulator the
ICBINB failure-mode paper stress-tests, so the two submissions share an engine.

STRUCTURE PREDICTION — GRACEFUL DEGRADATION (like generate_batch_evo2.py)
------------------------------------------------------------------------
1. proto-tools ESMFold (`from proto_tools import predict_structures`) if Proto
   is installed -- keeps this on the same stack as the Proto design campaign.
2. HuggingFace `facebook/esmfold_v1` via transformers `EsmForProteinFolding`
   (`model.infer_pdb(seq)`), the portable Colab path.
3. If neither is importable, structural metrics are skipped with a clear
   message and only sequence-level metrics (truncation rate) are reported.

USAGE
-----
python structural_drift.py \
    --reference_fasta /content/drive/MyDrive/PRISM/genmodel_bias/bacterial_prompts.fasta \
    --profile_dir /content/saanvi/profiles \
    --signatures SBS17a,SBS1,SBS7a \
    --n_draws 20 \
    --output_dir /content/drive/MyDrive/PRISM/structural_drift
"""

import argparse
import csv
import os
import random
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# NOTE: prism_utils is imported lazily inside the functions that need it
# (mutate_cds/translate_cds/load_mutation_profile), because it imports the
# transformers/torch stack at module load. Keeping it lazy lets the pure,
# dependency-free structural utilities (parse_ca, kabsch_rmsd, mean_plddt)
# be imported and unit-tested without the full ML stack present.


# ══════════════════════════════════════════════════════════════════════════
# PDB parsing + Kabsch RMSD (pure, dependency-free, unit-testable)
# ══════════════════════════════════════════════════════════════════════════
def parse_ca(pdb_text: str):
    """Return (coords Nx3 float array, plddt list) for CA atoms in a PDB string.
    pLDDT is read from the B-factor column (cols 61-66), the ESMFold convention."""
    coords, plddt = [], []
    for line in pdb_text.splitlines():
        if line.startswith(("ATOM", "HETATM")) and line[12:16].strip() == "CA":
            try:
                x = float(line[30:38]); y = float(line[38:46]); z = float(line[46:54])
                b = float(line[60:66])
            except ValueError:
                continue
            coords.append((x, y, z))
            plddt.append(b)
    return np.asarray(coords, dtype=float), plddt


def kabsch_rmsd(P: np.ndarray, Q: np.ndarray) -> float:
    """Minimal-RMSD superposition of two Nx3 coordinate sets (Kabsch).
    P and Q must already be in correspondence (same N, residue i <-> residue i)."""
    if P.shape[0] < 3 or P.shape != Q.shape:
        return float("nan")
    Pc = P - P.mean(0)
    Qc = Q - Q.mean(0)
    H = Pc.T @ Qc
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1.0, 1.0, d])
    R = Vt.T @ D @ U.T
    P_rot = Pc @ R.T
    diff = P_rot - Qc
    return float(np.sqrt((diff * diff).sum() / P.shape[0]))


def ca_rmsd_common(pdb_a: str, pdb_b: str):
    """CA-RMSD over the common (shorter) length of two structures. Missense
    signatures preserve length; nonsense truncations shorten the mutant, so we
    compare over the common ungapped prefix. Returns (rmsd, n_common)."""
    ca_a, pl_a = parse_ca(pdb_a)
    ca_b, pl_b = parse_ca(pdb_b)
    n = min(len(ca_a), len(ca_b))
    if n < 3:
        return float("nan"), n
    return kabsch_rmsd(ca_a[:n], ca_b[:n]), n


def mean_plddt(pdb_text: str):
    _, pl = parse_ca(pdb_text)
    return float(np.mean(pl)) if pl else float("nan")


# ══════════════════════════════════════════════════════════════════════════
# Structure predictor with graceful degradation
# ══════════════════════════════════════════════════════════════════════════
class Folder:
    """Uniform fold(protein_seq)->pdb_string over proto-tools ESMFold or HF ESMFold."""

    def __init__(self, backend, fn):
        self.backend = backend
        self._fn = fn

    def fold(self, protein_seq: str):
        if not protein_seq:
            return None
        return self._fn(protein_seq)


def _load_proto_folder():
    from proto_tools import predict_structures  # noqa: F401
    # proto-tools returns Structure objects; we ask for PDB text. The exact
    # accessor can vary by version, so we probe a couple of common shapes.
    from proto_tools import predict_structures as _predict

    def _fold(seq):
        # proto-tools' predict_structures signature varies by version; probe a
        # few call shapes rather than assume a keyword like structure_tool=.
        out = None
        for call in (lambda: _predict([seq]), lambda: _predict((seq,)),
                     lambda: _predict(seq)):
            try:
                out = call(); break
            except TypeError:
                continue
        if out is None:
            raise TypeError("proto-tools predict_structures: no known call signature matched")
        struct = out[0] if isinstance(out, (list, tuple)) else out
        for attr in ("pdb", "pdb_string", "to_pdb"):
            val = getattr(struct, attr, None)
            if callable(val):
                return val()
            if isinstance(val, str):
                return val
        return str(struct)

    return Folder("proto-tools:esmfold", _fold)


def _load_hf_folder(device_pref="auto"):
    import torch
    from transformers import AutoTokenizer, EsmForProteinFolding

    hf_id = "facebook/esmfold_v1"
    print(f"[struct] loading {hf_id} (~8GB download on first use; be patient, don't interrupt)...")
    tok = AutoTokenizer.from_pretrained(hf_id)
    # use_safetensors=True avoids ALSO pulling the redundant pytorch_model.bin
    # (~8GB) -- halves the download. low_cpu_mem_usage speeds the load.
    try:
        model = EsmForProteinFolding.from_pretrained(
            hf_id, use_safetensors=True, low_cpu_mem_usage=True)
    except Exception:
        model = EsmForProteinFolding.from_pretrained(hf_id)  # fallback if safetensors absent
    device = "cuda" if (device_pref != "cpu" and torch.cuda.is_available()) else "cpu"
    model = model.to(device).eval()
    if device == "cuda":
        model.esm = model.esm.half()  # ESMFold's standard mixed-precision setup
    model.trunk.set_chunk_size(64)

    def _fold(seq):
        with torch.no_grad():
            return model.infer_pdb(seq)

    _ = tok  # infer_pdb tokenizes internally; keep ref to surface load errors early
    return Folder(f"hf:{hf_id}:{device}", _fold)


def load_folder(prefer="auto"):
    # In "auto" we try the HuggingFace ESMFold path FIRST: it is the
    # documented, reliable interface (model.infer_pdb). proto-tools ESMFold is
    # tried only if explicitly requested or as an auto fallback, because its
    # predict_structures signature varies across versions.
    order = {"auto": ["hf", "proto"], "hf": ["hf"], "proto": ["proto"]}[prefer]
    loaders = {"hf": _load_hf_folder, "proto": _load_proto_folder}
    errors = []
    for backend in order:
        try:
            f = loaders[backend]()
            # validate proto at load time with a tiny fold so a bad signature
            # fails HERE (and falls back) instead of mid-analysis.
            if backend == "proto":
                _ = f.fold("MKTAYIAKQR")
            print(f"[struct] Using folder backend: {f.backend}")
            return f
        except Exception as e:
            errors.append(f"{backend} ESMFold unavailable: {e}")
    print("[struct] No structure predictor available; structural metrics will be "
          "skipped (sequence-level metrics still reported).\n  - "
          + "\n  - ".join(errors))
    return None


# ══════════════════════════════════════════════════════════════════════════
# main analysis
# ══════════════════════════════════════════════════════════════════════════
def has_internal_stop(nt_seq: str) -> bool:
    """True if the signature introduced a premature stop (nonsense) before the
    natural end -- translate_cds stops at the first stop codon, so a shortened
    protein relative to the full-frame translation signals truncation."""
    from prism_utils import translate_cds
    full = str(translate_cds(nt_seq))
    frame_len = (len(nt_seq) - (len(nt_seq) % 3)) // 3
    return len(full) < frame_len - 1  # allow the natural terminal stop


def load_reference_cds(path: str):
    from Bio import SeqIO
    return {r.id: str(r.seq).upper().replace(" ", "") for r in SeqIO.parse(path, "fasta")}


def _safe_fold(folder, seq, tag):
    """Fold defensively: any failure (OOM, tokenizer, kernel) returns None with a
    warning instead of crashing the whole analysis. On a CUDA OOM we also clear
    the cache so the next fold can proceed."""
    if folder is None or not seq:
        return None
    try:
        return folder.fold(seq)
    except Exception as e:
        print(f"    [warn] fold failed ({tag}, len={len(seq)}): {type(e).__name__}: {e}")
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        return None


def analyze(reference, profiles, folder, n_draws, out_dir, seed, max_aa=None):
    from prism_utils import mutate_cds, translate_cds
    rng = random.Random(seed)
    os.makedirs(out_dir, exist_ok=True)
    per_draw_rows = []
    summary_rows = []
    summary_header = ["gene", "signature", "n_draws", "mean_delta_plddt", "mean_ca_rmsd",
                      "truncation_rate", "plddt_wt", "wt_len_aa", "n_folded"]

    def _flush():
        # Write after every gene so a crash/timeout never loses completed work.
        _write(os.path.join(out_dir, "structural_drift_per_draw.csv"),
               ["gene", "signature", "draw", "delta_plddt", "ca_rmsd", "truncated"], per_draw_rows)
        _write(os.path.join(out_dir, "structural_drift_summary.csv"), summary_header, summary_rows)

    for gene, cds in reference.items():
        wt_aa = str(translate_cds(cds))
        if len(wt_aa) < 10:
            print(f"[struct] {gene}: WT <10 aa, skipping")
            continue
        # Length cap: proteins beyond max_aa OOM ESMFold on small GPUs. We still
        # record truncation (sequence-level) but skip the fold for those.
        too_long = max_aa is not None and len(wt_aa) > max_aa
        wt_pdb = None if too_long else _safe_fold(folder, wt_aa, f"{gene}/WT")
        plddt_wt = mean_plddt(wt_pdb) if wt_pdb else float("nan")
        note = f"pLDDT_wt={plddt_wt:.1f}" if wt_pdb else (
            f"len>{max_aa} -> fold skipped" if too_long else "WT fold failed/none")
        print(f"[struct] {gene}: WT len={len(wt_aa)}aa  {note}")

        for sig_name, profile in profiles.items():
            dplddts, rmsds, truncs = [], [], []
            n_folded = 0
            for d in range(n_draws):
                random.seed(rng.randrange(1 << 30))
                mut_cds = mutate_cds(cds, profile)
                truncated = has_internal_stop(mut_cds)
                truncs.append(int(truncated))
                mut_aa = str(translate_cds(mut_cds))

                dplddt = rmsd = float("nan")
                if wt_pdb and len(mut_aa) >= 10:
                    mut_pdb = _safe_fold(folder, mut_aa, f"{gene}/{sig_name}/draw{d}")
                    if mut_pdb:
                        n_folded += 1
                        dplddt = mean_plddt(mut_pdb) - plddt_wt
                        rmsd, _ = ca_rmsd_common(wt_pdb, mut_pdb)
                dplddts.append(dplddt)
                rmsds.append(rmsd)
                per_draw_rows.append([gene, sig_name, d, _r(dplddt), _r(rmsd), int(truncated)])

            summary_rows.append([
                gene, sig_name, n_draws,
                _r(_nanmean(dplddts)), _r(_nanmean(rmsds)),
                _r(np.mean(truncs)), _r(plddt_wt), len(wt_aa), n_folded,
            ])
            print(f"    {sig_name:8s} ΔpLDDT={_nanmean(dplddts):+.2f}  "
                  f"CA-RMSD={_nanmean(rmsds):.2f}A  trunc={np.mean(truncs):.0%}  folded={n_folded}/{n_draws}")
        _flush()  # incremental save per gene

    _flush()
    _plot_structural(summary_rows, summary_header, out_dir)
    return summary_rows


def _plot_structural(rows, header, out_dir):
    """ΔpLDDT and CA-RMSD by signature (mean over folded genes). Skips cleanly if
    nothing folded (structural columns all empty)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    import collections
    di = {h: i for i, h in enumerate(header)}
    by_sig = collections.defaultdict(lambda: {"dp": [], "rm": []})
    for r in rows:
        dp, rm = r[di["mean_delta_plddt"]], r[di["mean_ca_rmsd"]]
        if dp != "":
            by_sig[r[di["signature"]]]["dp"].append(float(dp))
        if rm != "":
            by_sig[r[di["signature"]]]["rm"].append(float(rm))
    sigs = [s for s in by_sig if by_sig[s]["dp"]]
    if not sigs:
        print("[struct] no successful folds -> skipping structural plot "
              "(truncation metrics still in the CSVs)")
        return
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    ax[0].bar(range(len(sigs)), [np.mean(by_sig[s]["dp"]) for s in sigs], color="#4c72b0")
    ax[0].set_xticks(range(len(sigs))); ax[0].set_xticklabels(sigs, rotation=30)
    ax[0].set_ylabel("mean ΔpLDDT (mutant − WT)"); ax[0].axhline(0, c="k", lw=0.5)
    ax[0].set_title("Foldability change by signature")
    ax[1].bar(range(len(sigs)), [np.mean(by_sig[s]["rm"]) for s in sigs], color="#c44e52")
    ax[1].set_xticks(range(len(sigs))); ax[1].set_xticklabels(sigs, rotation=30)
    ax[1].set_ylabel("mean CA-RMSD (Å)"); ax[1].set_title("Backbone drift by signature")
    fig.suptitle("Structural drift under mutational signatures (ESMFold)")
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "structural_drift.png"), dpi=200)
    print(f"[struct] wrote {os.path.join(out_dir, 'structural_drift.png')}")


def _r(x, nd=3):
    return "" if x is None or (isinstance(x, float) and x != x) else round(float(x), nd)


def _nanmean(xs):
    arr = np.asarray([x for x in xs if x is not None and not (isinstance(x, float) and x != x)],
                     dtype=float)
    return float(arr.mean()) if arr.size else float("nan")


def _write(path, header, rows):
    with open(path, "w", newline="") as f:
        w = csv.writer(f); w.writerow(header); w.writerows(rows)
    print(f"[struct] wrote {path} ({len(rows)} rows)")


def load_profiles(profile_dir, wanted):
    """Load requested signatures from a directory of COSMIC profile files."""
    from prism_utils import load_mutation_profile
    wanted = {w.strip() for w in wanted if w.strip()}
    found = {}
    for root, _, files in os.walk(profile_dir):
        for fn in files:
            if fn.startswith("."):
                continue
            stem = fn.split("_")[0].split(".")[0]
            if not wanted or stem in wanted:
                try:
                    prof = load_mutation_profile(os.path.join(root, fn))
                    if prof:
                        found[stem] = prof
                except Exception:
                    pass
    missing = wanted - set(found)
    if missing:
        print(f"[struct] WARNING: requested signatures not found: {sorted(missing)}")
    return found


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--reference_fasta", required=True, help="FASTA of full reference CDS per gene.")
    p.add_argument("--profile_dir", required=True, help="Directory of COSMIC signature profile files.")
    p.add_argument("--signatures", default="SBS17a,SBS1,SBS7a",
                   help="Comma-separated signatures to test (default: SBS17a,SBS1,SBS7a).")
    p.add_argument("--n_draws", type=int, default=20, help="Monte-Carlo signature draws per gene.")
    p.add_argument("--max_genes", type=int, default=None,
                   help="Cap the number of genes folded (ESMFold is slow on a T4; "
                        "use e.g. 6 for a pilot).")
    p.add_argument("--max_aa", type=int, default=None,
                   help="Skip ESMFold for proteins longer than this many residues "
                        "(they OOM small GPUs); truncation metrics are still recorded. "
                        "Recommended ~400 on a T4.")
    p.add_argument("--folder", choices=["auto", "proto", "hf", "none"], default="auto",
                   help="Structure predictor backend preference.")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    os.makedirs(args.output_dir, exist_ok=True)
    reference = load_reference_cds(args.reference_fasta)
    if args.max_genes:
        reference = dict(list(reference.items())[:args.max_genes])
        print(f"[struct] capped to first {len(reference)} gene(s) (--max_genes)")
    profiles = load_profiles(args.profile_dir, args.signatures.split(","))
    if not profiles:
        raise SystemExit("[struct] No signatures loaded; check --profile_dir / --signatures.")
    folder = None if args.folder == "none" else load_folder(args.folder)
    print(f"[struct] genes={len(reference)} signatures={list(profiles)} "
          f"n_draws={args.n_draws} folder={folder.backend if folder else 'none'}")
    analyze(reference, profiles, folder, args.n_draws, args.output_dir, args.seed,
            max_aa=args.max_aa)
    print(f"[struct] Done. Results in {args.output_dir}/")


if __name__ == "__main__":
    main()
