# Why test language transfer across biological sequence models

## The question

Biology writes the same molecule in two alphabets. A coding sequence is a string
over four nucleotides. The protein it specifies is a string over twenty amino
acids. Large pretrained models exist for both alphabets, and they are trained on
different corpora with different objectives. Genomic language models read whole
genomes, including regions that never become protein. Protein language models
read curated protein databases.

The question this project asks is whether a model trained on one alphabet
acquires usable knowledge of the other. Stated for the application that motivates
it: can a genomic model, given the coding sequence, predict how a protein will
behave?

## Why biology permits a sharper test than other cross-modal settings

Cross-modal transfer is usually studied where the correspondence between
modalities is loose. An image and its caption describe the same scene, but no
function maps one to the other, so a claim that two representations "align" has
no exact referent.

Translation from coding sequence to protein has three properties that remove that
looseness, and each one supports a specific measurement.

The map is known and deterministic. The genetic code specifies which amino acid
each codon produces. Any claim about correspondence between a genomic and a
protein representation can be checked against the true map instead of an
estimated one.

The map is many to one. Sixty-one codons encode twenty amino acids, so most
amino acids have several synonymous codons. Different coding sequences therefore
translate to identical proteins. A protein-level assay measures the protein, so
synonymous encodings of one variant carry the same measured value by
construction. Any variation a genomic model assigns across those encodings cannot
correspond to anything the assay reports. Measuring that variation gives an exact
upper bound on how well a genomic score could correlate with protein function.
Section 18 of `FINDINGS.md` reports a mean bound of 0.900, which establishes that
an observed correlation near zero reflects the model and not the measurement.

The reading frame supports a causal intervention. Shifting a coding sequence by
one nucleotide preserves almost all of its nucleotide content and destroys every
codon downstream. Comparing a model's behaviour before and after a frameshift
separates two hypotheses that correlation alone cannot: whether a representation
encodes the protein, or whether it encodes nucleotide statistics that happen to
predict the protein.

## Why transfer is worth expecting

Genomic models see more sequence than protein models, and they see categories of
sequence that protein models never encounter. Coding sequences also carry
information the protein does not. Codon usage influences translation speed and
expression level, and synonymous changes can alter how much functional protein a
cell produces even though the amino acid sequence is unchanged.

The practical argument is stronger than the informational one. Protein
engineering pipelines order DNA. A proxy that scored the coding sequence would
price amino acid changes and codon choices with one model, and a single genomic
model that carried protein-level knowledge could replace a two-model stack.

## Why the result matters beyond the application

Zero-shot likelihood is the standard way to use a sequence model as a fitness
predictor. The score is the model's estimate of how probable a sequence is under
its training distribution, and the field treats that probability as a proxy for
function. Whether the proxy survives a change of alphabet distinguishes two
readings of what the likelihood measures. If likelihood tracks function, then a
model of the coding sequence should recover it, because the coding sequence
determines the protein. If likelihood tracks the statistics of a training corpus,
then changing the corpus and the alphabet should remove the signal.

Claims that a foundation model has learned the meaning of its domain are usually
hard to test, because the meaning is not independently defined. Here it is.
Binding and fitness are measured in an experiment that is indifferent to how the
model represents anything.

## What the controls caught

The design earned its cost. An initial reading of the evidence was wrong, and the
controls are what showed it.

A linear probe on frozen genomic-model embeddings reached a Spearman correlation
of +0.141 with measured specificity on held-out amino acids, while the same
model's likelihood sat at chance. That pattern suggested weak transfer at the
level of the representation. The frameshift control removed that reading.
Frameshifting the sequence destroys every codon and leaves the probe at +0.143,
and scoring the reverse complement, which is not a coding sequence, leaves it at
+0.129. A representation that survives the destruction of the reading frame is
not encoding the protein. Under any encoding rule the codon determines the amino
acid, so the nucleotides at a varying site already carry residue identity, and a
linear probe recovers it without representing the protein at all.

A variance decomposition agrees. Amino acid identity accounts for 0.672 of the
variance in the genomic representation and 0.643 of the variance in the
likelihood, against 1.000 for any protein model, which never sees DNA. About a
third of the genomic representation varies with codon choice, which cannot move a
measured label.

Without the frameshift and reverse-complement comparisons, this project would
have reported weak genomic-to-protein transfer. The claim would have been wrong.

## What the evidence now supports

Across 25 deep mutational scanning assays covering nine organisms, a protein
model reaches a mean Spearman correlation of +0.466 with measured fitness and is
positive on all 25 assays. Genomic models scoring the coding sequence that
encodes the same region reach -0.013 at 50M parameters, -0.013 at 100M, -0.007 at
250M, and +0.008 at 500M. Every confidence interval contains zero, and the
protein model has the larger absolute correlation on 24 of 25 assays at every
scale. Averaging the genomic score over synonymous encodings, which removes the
residue-identity channel by construction, gives -0.019.

On the ParD3 binding specificity task, four genomic model scales and an
autoregressive single-nucleotide model all sit at chance, against 0.664 for
counting mutations. Supplying real genomic context degrades the result further,
from 0.505 with no flanking sequence to 0.365 with 3,000 nucleotides on each
side, so withholding context does not explain the failure.

## What transfers to other work

Two measurements here apply to any claim that a model of one biological modality
has learned another. The synonymous floor converts an ambiguous null into a
quantitative one by measuring how much of a score cannot correspond to the label.
The frameshift comparison tests whether a representation depends on the property
it is claimed to encode. Both are cheap, and both changed a conclusion in this
project.

## Scope

The evidence covers coding sequences scored in isolation and in genomic context,
for one exhaustively measured specificity landscape and 25 deep mutational
scanning assays. Evo 2 has not been run and is the model most likely to behave
differently, since it is built for longer range and larger capacity than the
models tested here. `notebooks/Evo2_crosstalk_GPU.ipynb` runs every arm above
under the same protocol and needs a GPU.

The result does not show that no genomic model can represent protein function. It
shows that scoring a coding sequence with the genomic models available here does
not recover protein function, that marginalising over codons does not repair it,
that measurement noise does not explain it, and that the one apparent exception
was residue identity reaching the probe through the nucleotides.
