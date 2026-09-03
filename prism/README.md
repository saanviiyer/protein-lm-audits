# PRISM: Mutational Signatures as Directional Drivers in Protein Function Space

Homology-free tooling for measuring, auditing, and inverting the effect of
mutational processes on protein function. This repository reproduces the results
of three companion NeurIPS 2026 workshop papers:

| Paper | Venue | What it uses |
|---|---|---|
| *When There Is No BLAST to Blame* | ICBINB-BIO (Failure Modes of AI in Biology) | `audit_prism_results.py`, `stress_test_generation.py`, `sweep_decoding.py`, `analyze_steering_specificity.py`, `inverse_design_classifier.py` |
| *Inverse Design of Mutational Signatures* | MoML (Molecular ML) | `inverse_design_classifier.py` |
| *Physics-Aligned Generative Protein Design* | SIMBIOCHEM II | `biophysical_drift.py`, `prism_proto_design.py`, `inverse_design_classifier.py`, `structural_drift.py` |

## What PRISM does
1. **Simulate** a physical mutational process: apply each of the 78 COSMIC
   single-base-substitution (SBS) signatures to a coding sequence as a
   per-trinucleotide Monte-Carlo (`prism_utils.mutate_cds`).
2. **Read out** the consequence three ways: embedding drift toward Pfam domain
   centroids (ESM-2), biophysical disruptiveness (folding-free), and predicted
   structure (ESMFold).
3. **Probe** genomic language models (Carbon, GENERator, Evo 2) for the same
   biases, and **audit** the pipeline for the artifacts that arise without a
   homology anchor.
4. **Invert** the map: predict which signature drives a desired Pfam→Pfam
   transition, and drive a closed-loop design campaign toward a target domain.

## Layout
All scripts live at the repo root (they import each other as siblings, so keep
them together / on `PYTHONPATH`). By role:

- **core** — `prism_config.py`, `prism_utils.py`, `prism_genmodel_analysis.py`, `prism_phase2_analysis.py`
- **generate** (GPU) — `generate_batch_{carbon,generator,evo2}.py`
- **analyze** — `stress_test_generation.py`, `sweep_decoding.py`, `audit_prism_results.py`, `analyze_steering_specificity.py`, `analyze_bacterial_generation_matched.py`, `significance_tests.py`
- **design** — `inverse_design_classifier.py`, `prism_proto_design.py`
- **structure** — `structural_drift.py` (ESMFold), `biophysical_drift.py` (folding-free)
- **paper** — `fill_paper_numbers.py`, `fill_structural.py`
- **notebook** — `prism_extension_pipeline.ipynb` (one-run Colab driver)

## Reproducing each result (no GPU unless noted)
```bash
# ICBINB — self-audit + hub-confound (real data, CPU)
python audit_prism_results.py --runs_dir all_runs --target_domain PF17041 --focus_signature SBS17a
python analyze_steering_specificity.py --runs_dir all_runs --focus_signature SBS17a --output_dir all_runs/steering_specificity

# ICBINB — collapse + tokenizer×alignment artifact (needs Carbon/GENERator/Evo2 outputs; GPU to generate)
python stress_test_generation.py --model_dir carbon:carbon_output --model_dir generator:generator_output \
    --reference_fasta prompts.fasta --cosmic_dir cosmic_signatures --output_dir results/stress_test

# MoML — inverse design (real data, CPU)
python inverse_design_classifier.py --runs_dir all_runs \
    --centroid_chunks_dir centroids/centroid_chunks --profile_dir profiles \
    --mode both --output_dir results/inverse_design
python inverse_design_classifier.py ... --mode crossclan   # train one clan, test another
python inverse_design_classifier.py ... --mode ablation    # feature/model/NN ablations
python inverse_design_classifier.py ... --mode loso        # leave-signatures-out

# SIMBIOCHEM — biophysical drift (real data, CPU) + Proto design
python biophysical_drift.py --cds_fasta all_runs/CL0023/pairs_cds.fasta \
    --profile_dir profiles --signatures ALL --output_dir results/biophysical_drift
python prism_proto_design.py --backend auto --wt_fasta seed.fasta \
    --target_pfam PF17041 --centroid_chunks_dir centroids/centroid_chunks \
    --signature_profile profiles/SBS-MS/SBS17a_PROFILE.txt --output_dir results/prism_design

# Significance tests across all key results (real data, CPU)
python significance_tests.py --results_dir results --runs_dir all_runs --output_dir results/significance
```

GPU-only steps: generating continuations with the genomic LMs
(`generate_batch_*.py`), and the ESMFold tier of `structural_drift.py`
(Evo 2 needs an FP8 Hopper GPU; ESMFold needs >16 GB). `prism_extension_pipeline.ipynb`
runs the whole pipeline on Colab and degrades gracefully.

## Data not included here
`all_runs/` (PRISM Phase-1/2 outputs), `profiles/` (COSMIC SBS profiles),
`centroids/` (Pfam centroid library), and the generation outputs are large and
distributed separately.

## Citation
See the three papers above. If you use this tooling, please cite PRISM.

## License
MIT (see `LICENSE`).
