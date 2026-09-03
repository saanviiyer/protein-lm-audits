# A genomic-language-model arm, scoped

## The gap this exists to close

Sections 14 to 31 all score **coding sequence** and ask about **protein fitness**.
Genomic models are pretrained mostly on **non-coding DNA**, and their intended
applications are regulatory. So every genomic result in this repository grades
these models on a task they were not built for, and the conclusions inherit that.

Section 30 sharpened the problem rather than resolving it. Evo 2 does carry
protein-fitness signal (+0.266 on 25 assays) but is statistically indistinguishable
from BLOSUM62 (+0.228), and section 31 showed the Nucleotide Transformer scale
trend does not reach it, so the difference is architecture and context rather than
size. None of that says anything about whether these models are good at regulatory
biology, which is what they are sold for.

## The two questions

**1. Do genomic models beat a position weight matrix on their own home task?**

This is the genomic restatement of section 18. There, adding BLOSUM62 at +0.228
against ESM-2's +0.466 reframed the entire claim: the bar a genomic model had to
clear was less than half what it looked like. The regulatory equivalent is a PWM,
k-mer logistic regression and GC content.

There is published reason to expect this bites. A 2026 workshop paper on genomic
model invariance reports that compositional effects dominate positional ones
46-fold and that a roughly 100-parameter position-aware PWM was perfect on their
benchmark. If a PWM matches a pretrained genomic model on regulatory tasks under
homology-aware splits, that is the same shape of result as section 23, where a
detector at AUC 0.804 covered 7.7% of the viable set.

**2. Is the context result about biology or about the window?**

Section 19 found more real genomic context made NT monotonically worse (trend
-0.77) and section 30 found the reverse for Evo 2 (+0.70). NT's window is roughly
12 kb and the sweep was capped at 5,400 nt of flank, so the negative result cannot
currently distinguish "context does not help" from "the model cannot use context
past its window". Recursive decomposition in the manner of [Zhang2025recursive]
separates those, because it lets a short-context model consume an arbitrarily long
locus.

The control that decides it is shuffled flanks of matched length and composition.
Recursion adds compute and changes the score's scale, so it can move numbers for
uninteresting reasons; a gain on the real locus but not the shuffled one is
contextual, a gain on both is an aggregation artifact.

## Discipline carried over, because this project has paid for each of these

- **Splits.** Genomic sequence is homologous and repetitive, so a random split
  leaks. Section 15 moved the same features from rho 0.155 to 0.997 on split choice
  alone. Report the shipped split and a stricter one, and state the gap.
- **A trivial baseline in every table**, never a model number alone.
- **Cluster correction.** 194 ProteinGym assays were 165 independent proteins.
- **A noise ceiling before any null.** DNA offers a natural label-preserving
  transformation in the reverse complement, since many regulatory elements are
  strand-symmetric.
- **A positive control before any negative claim.** Section 26 was retracted for
  want of one, and section 29 then showed the replacement probe was underpowered
  rather than null.

## What this is not

Not an attempt to build a better genomic model. The output is calibrated numbers
about what these models measure, with their limits attached, which is what the
rest of this repository is.
