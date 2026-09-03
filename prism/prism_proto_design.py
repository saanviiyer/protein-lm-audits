#!/usr/bin/env python3
"""
prism_proto_design.py

PRISM expressed as an AGENTIC, CLOSED-LOOP design campaign in Proto
(evo-design/proto-language), for the SIMBIOCHEM framing.

PRISM discovered that a *physical* mutational process (COSMIC SBS17a) directs
proteins toward a specific functional neighborhood (the PF17041 centroid) in
ESM-2 space. Proto reduces a design campaign to four primitives -- sequences,
generators, constraints, optimizers -- and drives a propose->score->refine
loop. This module turns PRISM's finding into exactly that loop:

    sequences : a wild-type coding sequence (the seed)
    generators: (a) the COSMIC-signature mutational process  [physics-grounded]
                (b) [optional] Evo 2 proposals via proto-tools
    constraints: (a) centroid-attraction -- cosine distance from the candidate's
                     ESM-2 embedding to a TARGET Pfam centroid (PRISM's core
                     metric), lower = closer = better         [this file's @constraint]
                 (b) [optional] ESMFold pLDDT -- keep designs foldable  [physics]
    optimizers : MCMC / rejection sampling toward the target neighborhood

That is a directed-evolution campaign whose *driving force is a named chemical
mutational process* and whose *acceptance test is a foundation-model readout* --
"physics-aligned generative design" in the workshop's language.

TWO EXECUTION PATHS (graceful degradation, cf. generate_batch_evo2.py)
----------------------------------------------------------------------
* --backend proto  : build a real Proto Program (proto_language + proto_tools).
* --backend standalone : a self-contained propose->score->refine loop using
  prism_utils (ESM-2 + the signature mutator) and, optionally, the ESMFold
  Folder from structural_drift.py. Runs anywhere the PRISM pipeline runs, with
  NO Proto dependency -- so the science is reproducible even if Proto is absent.
* --backend auto (default): try proto, fall back to standalone.

The centroid-attraction constraint below is registered with Proto's @constraint
decorator so it is reusable inside any Proto program (UI or Python), AND is
called directly by the standalone loop, so both paths score identically.

USAGE
-----
# Standalone (no Proto needed):
python prism_proto_design.py --backend standalone \
    --wt_fasta seed_cds.fasta \
    --target_centroid_npy PF17041_centroid.npy \
    --signature_profile /content/saanvi/profiles/Chemotherapy/SBS17a_PROFILE.txt \
    --steps 300 --output_dir results/prism_design_sbs17a

# Proto:
python prism_proto_design.py --backend proto ...   (same args)
"""

import argparse
import csv
import math
import os
import random
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# ══════════════════════════════════════════════════════════════════════════
# Shared scorer: ESM-2 embedding -> cosine distance to a target Pfam centroid
# (This is PRISM's core metric. Both the Proto constraint and the standalone
#  loop call it, so scores are identical across backends.)
# ══════════════════════════════════════════════════════════════════════════
class CentroidScorer:
    """Lazily loads ESM-2 once and scores protein strings by cosine distance to
    a fixed target centroid vector. Lower is better (0 = on the centroid)."""

    def __init__(self, target_centroid: np.ndarray):
        self.target = np.asarray(target_centroid, dtype=float).reshape(-1)
        self._tok = None
        self._model = None

    def _ensure_model(self):
        if self._model is None:
            from prism_utils import load_esm2
            self._tok, self._model = load_esm2()

    def embed(self, protein_seqs):
        """ESM-2 embedding matrix (N, D) for a list of protein strings; used for
        the embedding-trajectory visualization."""
        from prism_utils import get_batch_embeddings
        self._ensure_model()
        valid = [s for s in protein_seqs if s]
        if not valid:
            return np.zeros((0, self._model.config.hidden_size))
        return get_batch_embeddings(valid, self._tok, self._model)

    def distances(self, protein_seqs):
        """Cosine distance to the target centroid for each protein string."""
        from prism_utils import get_batch_embeddings, vectorized_cosine_distance
        self._ensure_model()
        valid = [s for s in protein_seqs if s]
        if not valid:
            return np.ones(len(protein_seqs))
        embs = get_batch_embeddings(valid, self._tok, self._model)
        d = vectorized_cosine_distance(embs, self.target)
        # map back, assigning worst distance (1.0) to any empty inputs
        out, k = [], 0
        for s in protein_seqs:
            if s:
                out.append(float(d[k])); k += 1
            else:
                out.append(1.0)
        return np.asarray(out)


def load_target_centroid(args):
    """Load the target Pfam centroid: from a single .npy vector, or by pulling
    --target_pfam out of the PRISM centroid chunk library."""
    if args.target_centroid_npy:
        vec = np.load(args.target_centroid_npy)
        return np.asarray(vec, dtype=float).reshape(-1)
    if args.target_pfam and args.centroid_chunks_dir:
        from prism_genmodel_analysis import load_centroid_chunks
        matrix, ids = load_centroid_chunks(args.centroid_chunks_dir)
        if args.target_pfam not in ids:
            raise SystemExit(f"[design] {args.target_pfam} not in centroid library.")
        return matrix[ids.index(args.target_pfam)]
    raise SystemExit("[design] Provide --target_centroid_npy OR "
                     "(--target_pfam AND --centroid_chunks_dir).")


# ══════════════════════════════════════════════════════════════════════════
# Proto constraint: centroid attraction (registered for reuse in any Program)
# ══════════════════════════════════════════════════════════════════════════
def register_proto_constraint(scorer: "CentroidScorer", scale: float):
    """Register PRISM's centroid-attraction as a Proto @constraint and return
    the decorated function + config. Import is inside the function so this file
    loads without proto_language present (standalone path)."""
    from proto_language.utils.base import BaseConfig, ConfigField
    from proto_language.constraint.constraint_registry import constraint
    from proto_language.core import ConstraintOutput, Sequence  # noqa: F401
    from proto_language.utils import MAX_ENERGY, MIN_ENERGY

    class CentroidAttractionConfig(BaseConfig):
        distance_scale: float = ConfigField(
            default=scale, title="Distance scale",
            description="Cosine distance mapped to energy via 1-exp(-d/scale).",
            gt=0.0,
        )

    @constraint(
        key="prism-centroid-attraction",
        label="PRISM centroid attraction (ESM-2)",
        config=CentroidAttractionConfig,
        description="Cosine distance from the candidate protein's ESM-2 embedding "
                    "to a target Pfam domain centroid; lower = closer to the "
                    "target functional neighborhood (PRISM's core metric).",
        uses_gpu=True,
        tools_called=[],
        category="protein_structure",
        supported_sequence_types=["protein"],
    )
    def prism_centroid_attraction(input_sequences, config):
        seqs = [seq.sequence for (seq,) in input_sequences]
        dists = scorer.distances(seqs)
        results = []
        for d in dists:
            # map cosine distance in [0,2] to energy in [0,1); 0 dist -> MIN_ENERGY
            energy = 1.0 - math.exp(-float(d) / config.distance_scale)
            energy = max(MIN_ENERGY, min(MAX_ENERGY, energy))
            results.append(ConstraintOutput(score=energy, metadata={"cosine_distance": float(d)}))
        return results

    return prism_centroid_attraction, CentroidAttractionConfig


# ══════════════════════════════════════════════════════════════════════════
# Standalone propose -> score -> refine loop (no Proto dependency)
# ══════════════════════════════════════════════════════════════════════════
def signature_proposal(wt_cds: str, profile: dict, rng: random.Random):
    """Physics-grounded generator: apply the COSMIC signature to the CDS."""
    from prism_utils import mutate_cds
    random.seed(rng.randrange(1 << 30))  # mutate_cds uses global random
    return mutate_cds(wt_cds, profile)


def random_aa_proposal_cds(cds: str, rng: random.Random, n_mut: int = 1):
    """Baseline generator: random single-nucleotide substitutions (for the
    ablation 'is the signature actually doing the steering?')."""
    bases = "ACGT"
    s = list(cds)
    for _ in range(n_mut):
        i = rng.randrange(len(s))
        s[i] = rng.choice(bases)
    return "".join(s)


def run_standalone(args, scorer, wt_cds, profile, folder=None):
    from prism_utils import translate_cds
    rng = random.Random(args.seed)

    current_cds = wt_cds
    current_aa = str(translate_cds(current_cds))
    current_d = float(scorer.distances([current_aa])[0])
    best_cds, best_aa, best_d = current_cds, current_aa, current_d

    T = args.temperature
    rows = [["step", "accepted", "cosine_distance", "best_distance", "plddt", "truncated", "aa_len"]]
    print(f"[design] standalone MCMC: start distance={current_d:.4f}  target -> minimize")

    # Optional embedding-trajectory logging: record the current (accepted) ESM-2
    # embedding at each step so the design's walk through function space can be
    # visualized against the WT and the target centroid.
    save_emb = getattr(args, "save_embeddings", False)
    emb_log, emb_steps, emb_dist = [], [], []
    wt_emb = scorer.embed([current_aa])[0] if save_emb else None
    cur_emb = wt_emb

    for step in range(args.steps):
        if args.generator == "signature" and profile is not None:
            cand_cds = signature_proposal(wt_cds if args.reset_each else current_cds, profile, rng)
        else:
            cand_cds = random_aa_proposal_cds(current_cds, rng, n_mut=args.n_mut)

        cand_aa = str(translate_cds(cand_cds))
        truncated = len(cand_aa) < (len(cand_cds) // 3) - 1
        if len(cand_aa) < 10:
            rows.append([step, 0, "", round(best_d, 4), "", int(truncated), len(cand_aa)])
            continue

        if save_emb:
            from prism_utils import vectorized_cosine_distance
            cand_emb = scorer.embed([cand_aa])[0]
            cand_d = float(vectorized_cosine_distance(cand_emb[None, :], scorer.target)[0])
        else:
            cand_emb = None
            cand_d = float(scorer.distances([cand_aa])[0])

        # optional foldability gate (ESMFold pLDDT)
        plddt = ""
        if folder is not None and args.plddt_min > 0:
            from structural_drift import mean_plddt
            pdb = folder.fold(cand_aa)
            plddt = mean_plddt(pdb) if pdb else float("nan")
            if plddt == plddt and plddt < args.plddt_min:
                rows.append([step, 0, round(cand_d, 4), round(best_d, 4),
                             round(plddt, 1), int(truncated), len(cand_aa)])
                continue

        # Metropolis acceptance on distance (lower is better)
        delta = cand_d - current_d
        accept = delta < 0 or rng.random() < math.exp(-delta / max(T, 1e-9))
        if accept:
            current_cds, current_aa, current_d = cand_cds, cand_aa, cand_d
            if save_emb:
                cur_emb = cand_emb
            if cand_d < best_d:
                best_cds, best_aa, best_d = cand_cds, cand_aa, cand_d

        if save_emb:
            emb_log.append(cur_emb); emb_steps.append(step); emb_dist.append(current_d)

        rows.append([step, int(accept), round(cand_d, 4), round(best_d, 4),
                     (round(plddt, 1) if plddt not in ("",) else ""), int(truncated), len(cand_aa)])
        if step % max(1, args.steps // 10) == 0:
            print(f"    step {step:4d}  cand={cand_d:.4f}  best={best_d:.4f}  acc={int(accept)}")

    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "trajectory.csv"), "w", newline="") as f:
        csv.writer(f).writerows(rows)
    with open(os.path.join(args.output_dir, "best_design.fasta"), "w") as f:
        f.write(f">best_design|cosine_distance={best_d:.4f}|generator={args.generator}\n{best_aa}\n")
        f.write(f">wild_type|cosine_distance={scorer.distances([str(translate_cds(wt_cds))])[0]:.4f}\n"
                f"{str(translate_cds(wt_cds))}\n")
    if save_emb and emb_log:
        embs = np.stack(emb_log)
        np.savez(os.path.join(args.output_dir, "embeddings.npz"),
                 embs=embs, steps=np.asarray(emb_steps), dists=np.asarray(emb_dist),
                 wt=np.asarray(wt_emb), target=np.asarray(scorer.target))
        plot_embedding_trajectory(
            embs, np.asarray(emb_dist), wt_emb, scorer.target, args.generator,
            os.path.join(args.output_dir, "embedding_trajectory.png"))
        print(f"[design] wrote embeddings.npz + embedding_trajectory.png")

    print(f"[design] done. start={current_d if False else best_d:.4f} best_distance={best_d:.4f}  "
          f"-> {args.output_dir}/  (trajectory.csv, best_design.fasta)")
    return best_aa, best_d


def plot_embedding_trajectory(embs, dists, wt_emb, target, generator, path):
    """Visualize the design's walk through ESM-2 function space on axes that
    respect the cosine objective: x = cosine similarity to the target centroid
    (higher = closer, the quantity being optimized), y = cosine similarity to the
    wild-type (drops as the design drifts away). A raw PCA of the 320-d space does
    not preserve cosine geometry, so this projection is used instead."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[design] matplotlib unavailable; skipping embedding plot")
        return

    def cos(a, B):
        B = np.atleast_2d(B)
        return (B @ a) / (np.linalg.norm(B, axis=1) * np.linalg.norm(a) + 1e-9)

    x = cos(target, embs)          # similarity to target -- the optimized axis
    y = cos(wt_emb, embs)          # similarity to wild-type
    wt_x, wt_y = float(cos(target, wt_emb)[0]), 1.0
    tg_x, tg_y = 1.0, float(cos(wt_emb, target)[0])

    fig, ax = plt.subplots(figsize=(7.5, 6))
    ax.plot(x, y, "-", color="0.6", lw=0.8, alpha=0.7, zorder=1)
    sc = ax.scatter(x, y, c=dists, cmap="viridis_r", s=18, zorder=2)
    ax.scatter(wt_x, wt_y, marker="*", s=320, color="#e74c3c", edgecolor="k",
               zorder=3, label="wild-type")
    ax.scatter(tg_x, tg_y, marker="X", s=260, color="#2c3e50", edgecolor="w",
               zorder=3, label="target centroid (PF17041)")
    ax.annotate("", xy=(tg_x, tg_y), xytext=(wt_x, wt_y),
                arrowprops=dict(arrowstyle="->", color="0.3", ls="--", lw=1))
    fig.colorbar(sc, ax=ax, label="cosine distance to target")
    ax.set_xlabel("cosine similarity to target  (higher = closer)")
    ax.set_ylabel("cosine similarity to wild-type")
    ax.set_title(f"Design walk through ESM-2 function space ({generator} generator)")
    ax.legend(loc="best"); ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(path, dpi=200)


# ══════════════════════════════════════════════════════════════════════════
# Proto backend (assembles a real Program; falls back on any import/API error)
# ══════════════════════════════════════════════════════════════════════════
def run_proto(args, scorer, wt_cds, profile):
    from prism_utils import translate_cds
    from proto_language.core import Segment, Construct, Constraint, Program
    from proto_language.generator import RandomProteinGenerator, RandomProteinGeneratorConfig
    from proto_language.core import MaskingStrategy
    from proto_language.optimizer import MCMCOptimizer, MCMCOptimizerConfig

    wt_aa = str(translate_cds(wt_cds))
    fn, cfg_cls = register_proto_constraint(scorer, args.distance_scale)

    segment = Segment(sequence=wt_aa, sequence_type="protein", label="design")
    construct = Construct([segment])

    gen = RandomProteinGenerator(
        RandomProteinGeneratorConfig(masking_strategy=MaskingStrategy(num_mutations=args.n_mut)))
    gen.assign(segment)

    constraints = [Constraint(inputs=[segment], function=fn, function_config={})]
    if args.plddt_min > 0:
        # built-in ESMFold pLDDT constraint keeps designs foldable
        from proto_language.constraint.protein_structure.structure_plddt_constraint import (
            structure_plddt_constraint,
        )
        constraints.append(Constraint(
            inputs=[segment], function=structure_plddt_constraint,
            function_config={"structure_tool": "esmfold", "min_plddt": args.plddt_min}))

    opt = MCMCOptimizer(
        constructs=[construct], generators=[gen], constraints=constraints,
        config=MCMCOptimizerConfig(num_results=args.num_results, num_steps=args.steps))
    program = Program(optimizers=[opt], num_results=args.num_results)
    program.run()
    os.makedirs(args.output_dir, exist_ok=True)
    program.export(path=args.output_dir, format="csv")
    print(f"[design] Proto Program complete -> {args.output_dir}/")


# ══════════════════════════════════════════════════════════════════════════
def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--backend", choices=["auto", "proto", "standalone"], default="auto")
    p.add_argument("--wt_fasta", required=True, help="FASTA with the seed wild-type CDS (first record).")
    p.add_argument("--target_centroid_npy", default=None, help="Target Pfam centroid vector (.npy).")
    p.add_argument("--target_pfam", default=None, help="e.g. PF17041 (needs --centroid_chunks_dir).")
    p.add_argument("--centroid_chunks_dir", default=None, help="PRISM centroid .npz library dir.")
    p.add_argument("--signature_profile", default=None,
                   help="COSMIC signature profile file for the physics-grounded generator.")
    p.add_argument("--generator", choices=["signature", "random"], default="signature")
    p.add_argument("--reset_each", action="store_true",
                   help="Signature generator: propose from the WT each step (cloud) "
                        "rather than from the current accepted sequence (walk).")
    p.add_argument("--n_mut", type=int, default=1, help="Mutations per proposal (random generator).")
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--temperature", type=float, default=0.02, help="MCMC temperature (standalone).")
    p.add_argument("--distance_scale", type=float, default=0.1,
                   help="Cosine-distance->energy scale for the Proto constraint.")
    p.add_argument("--plddt_min", type=float, default=0.0,
                   help=">0 enables an ESMFold foldability gate at this pLDDT.")
    p.add_argument("--num_results", type=int, default=5)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--save_embeddings", action="store_true",
                   help="Log the ESM-2 embedding at each step and write "
                        "embeddings.npz + embedding_trajectory.png (PCA walk toward the target).")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    from Bio import SeqIO
    wt_cds = str(next(SeqIO.parse(args.wt_fasta, "fasta")).seq).upper().replace(" ", "")
    target = load_target_centroid(args)
    scorer = CentroidScorer(target)

    profile = None
    if args.signature_profile:
        from prism_utils import load_mutation_profile
        profile = load_mutation_profile(args.signature_profile)

    folder = None
    if args.plddt_min > 0 and args.backend != "proto":
        from structural_drift import load_folder
        folder = load_folder("auto")

    if args.backend in ("auto", "proto"):
        try:
            run_proto(args, scorer, wt_cds, profile)
            return
        except Exception as e:
            if args.backend == "proto":
                raise
            print(f"[design] Proto backend unavailable ({e}); falling back to standalone.")

    run_standalone(args, scorer, wt_cds, profile, folder=folder)


if __name__ == "__main__":
    main()
