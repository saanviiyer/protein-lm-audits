"""biointerp -- interpretability for biological sequence models by typed intervention.

Biology supplies something NLP does not: input perturbations whose effect on
meaning is exactly known in advance. A synonymous recode provably preserves the
protein. A one-nucleotide rotation provably destroys the reading frame while
leaving every k-mer count identical. A reverse complement provably moves the gene
to the other strand. Each is a *typed* intervention, and a model's likelihood
response to it is directly interpretable -- no probe to fit, no dictionary to
learn, no correlational reading to defend.

The package generalises FINDINGS section 26, where four such interventions
falsified the "coarse competence" reconciliation that correlational analysis had
suggested: Nucleotide Transformer v2 turned out unable to separate a gene from
the same gene rotated one nucleotide, while penalising codon-order disruption
twelve times more strongly.

    from crosstalk.biointerp import scorers, run_battery, render
    rep = run_battery(scorers.NTScorer(), {"gene1": cds, ...})
    print(render(rep))

Three things the battery enforces, each of them a trap this project has already
paid for:
  * every null is bounded against a reference intervention, so "no effect" is a
    measurement rather than an absence of one;
  * every mean is reported with its paired sign count and exact binomial p;
  * every expectation is declared by the intervention before any number exists.
"""
from .interventions import (Intervention, Contrast, CONTRASTS, REGISTRY, DEFAULT_BATTERY,
                            SECTION_26_BATTERY, battery, get, register,
                            PROPERTIES, REAL_HIGHER, NO_PREFERENCE)
from .battery import (run_battery, rebuild_from_csv, BatteryReport,
                      InterventionResult, ContrastResult,
                      REPRESENTS, INVERTED, NULL, UNDERPOWERED, INVARIANT, SENSITIVE)
from .report import render, write_csv, write_per_sequence_csv, write_contrast_csv
from . import scorers

__all__ = [
    "Intervention", "Contrast", "CONTRASTS", "REGISTRY", "DEFAULT_BATTERY", "SECTION_26_BATTERY",
    "battery", "get", "register", "PROPERTIES", "REAL_HIGHER", "NO_PREFERENCE",
    "run_battery", "rebuild_from_csv", "BatteryReport", "InterventionResult", "ContrastResult", "REPRESENTS",
    "INVERTED", "NULL", "UNDERPOWERED", "INVARIANT", "SENSITIVE",
    "render", "write_csv", "write_per_sequence_csv", "write_contrast_csv", "scorers",
]
