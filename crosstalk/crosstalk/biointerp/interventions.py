"""Typed causal interventions on biological sequence, with declared semantics.

The point of this module is the *metadata*, not the transforms. Biology admits
input interventions whose effect on meaning is exactly known, which natural
language does not: a synonymous recode provably preserves the protein, a
one-nucleotide rotation provably destroys the reading frame while preserving
every k-mer count. Each transform below therefore carries an explicit record of
what it PRESERVES and what it DESTROYS, and a statement of what the model's
response to it licenses you to conclude.

That record is what makes a likelihood delta interpretable without training a
probe or an SAE. A model that scores a real gene above its own one-nucleotide
rotation is discriminating something that only the reading frame can carry,
because the rotation is composition-identical at every order. A model that does
not is, on that evidence, not representing the frame -- and therefore not
representing which protein the sequence encodes.

Provenance: `synonymous_recode`, `codon_shuffle`, `frameshift` and `revcomp` are
lifted from scripts/run_granularity_ladder.py and scripts/run_transfer_analysis.py
(FINDINGS section 26 and section 17). `mono_shuffle` and `dinuc_shuffle` are new
and supply the sensitivity floor those sections did not measure.

The randomness contract differs from the original scripts on purpose. There, one
generator was threaded through every (sequence, condition) pair, so a value
depended on iteration order. Here each pair gets a generator seeded from
(seed, intervention name, sequence key), so a battery is reproducible under
reordering, subsetting or parallelism. Deterministic interventions -- frameshift,
reverse complement -- are bit-identical to the originals; the stochastic ones
agree only within Monte-Carlo noise.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from .. import glm

COMP = str.maketrans("ACGT", "TGCA")

# ---------------------------------------------------------------- vocabulary
#
# The properties an intervention can preserve or destroy. Kept small and
# explicit: an intervention that names a property outside this set is a bug,
# because the report renderer has nothing to say about it.

PROTEIN_IDENTITY = "protein_identity"          # the encoded amino-acid string
READING_FRAME = "reading_frame"                # the phase codons are read in
CODON_USAGE = "codon_usage"                    # the multiset of in-frame codons
CODON_ADJACENCY = "codon_adjacency"            # which codon follows which
NUC_COMPOSITION = "nucleotide_composition"     # mononucleotide counts
DINUC_COMPOSITION = "dinucleotide_composition" # dinucleotide counts
KMER_COMPOSITION = "kmer_composition"          # counts at every order
STRAND = "strand"                              # which strand it is read on
LENGTH = "length"

PROPERTIES = (PROTEIN_IDENTITY, READING_FRAME, CODON_USAGE, CODON_ADJACENCY,
              NUC_COMPOSITION, DINUC_COMPOSITION, KMER_COMPOSITION, STRAND, LENGTH)

REAL_HIGHER = "real_higher"      # a model with the destroyed property must prefer real
NO_PREFERENCE = "no_preference"  # a model reading only the preserved property must not care


@dataclass(frozen=True)
class Intervention:
    """A named transform on a nucleotide sequence with declared semantics.

    Fields
    ------
    fn          (sequence, rng) -> sequence. `rng` is ignored by deterministic ones.
    preserves   properties the transform provably leaves unchanged.
    destroys    properties the transform provably changes.
    probes      the single property this contrast isolates, given `preserves`.
                This is the thing the model's response is evidence about.
    expect      REAL_HIGHER  -- a model representing `probes` must score real above.
                NO_PREFERENCE -- a model representing only what is preserved must
                be indifferent; a large delta is evidence it reads surface form.
    claim_if_significant / claim_if_null
                English rendered into the report, so the conclusion is written
                down next to the transform rather than reconstructed later.
    requires_cds  the transform is only meaningful on an in-frame coding
                sequence whose length is a multiple of three.
    """

    name: str
    fn: Callable[[str, np.random.Generator], str]
    preserves: tuple[str, ...]
    destroys: tuple[str, ...]
    probes: str
    expect: str
    rationale: str
    claim_if_significant: str
    claim_if_null: str
    requires_cds: bool = True
    stochastic: bool = False
    tags: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self):
        bad = [p for p in self.preserves + self.destroys if p not in PROPERTIES]
        if bad:
            raise ValueError(f"{self.name}: unknown properties {bad}")
        both = set(self.preserves) & set(self.destroys)
        if both:
            raise ValueError(f"{self.name}: {sorted(both)} both preserved and destroyed")
        if self.expect not in (REAL_HIGHER, NO_PREFERENCE):
            raise ValueError(f"{self.name}: bad expect {self.expect!r}")

    def apply(self, seq: str, rng: np.random.Generator) -> str:
        return self.fn(seq, rng)

    def rng_for(self, key: str, seed: int) -> np.random.Generator:
        """A generator determined by (seed, intervention, sequence key) only."""
        h = hashlib.blake2b(f"{seed}|{self.name}|{key}".encode(), digest_size=8).digest()
        return np.random.default_rng(int.from_bytes(h, "big"))


# ------------------------------------------------------------------ transforms

def _synonymous_recode(cds: str, rng: np.random.Generator) -> str:
    """Same protein, every codon swapped for a synonymous alternative where one exists.

    Lifted from run_granularity_ladder.synonymous_recode. Methionine and
    tryptophan have no alternative and are left alone, so the protein is exactly
    preserved and roughly 94% of codons move.
    """
    out = []
    for i in range(0, len(cds) - 2, 3):
        c = cds[i:i + 3]
        alts = [x for x in glm.SYNONYMOUS.get(glm.CODON_TABLE.get(c), [c]) if x != c]
        out.append(alts[rng.integers(len(alts))] if alts else c)
    return "".join(out) + cds[len(cds) - len(cds) % 3:]


def _codon_shuffle(cds: str, rng: np.random.Generator) -> str:
    """Identical codon usage and identical frame, protein and codon order destroyed."""
    cods = [cds[i:i + 3] for i in range(0, len(cds) - 2, 3)]
    order = rng.permutation(len(cods))
    return "".join(cods[i] for i in order)


def _frameshift(k: int):
    def f(cds: str, rng: np.random.Generator) -> str:
        """Cyclic rotation by k nucleotides.

        Cyclic, not truncating, on purpose: a rotation of a sequence has exactly
        the same count of every k-mer as the original when read cyclically, and
        differs at only one junction linearly. So it changes the reading frame
        and essentially nothing else. This is the sharpest available frame probe.
        """
        return cds[k:] + cds[:k]
    return f


def _revcomp(cds: str, rng: np.random.Generator) -> str:
    return cds.translate(COMP)[::-1]


def _mono_shuffle(cds: str, rng: np.random.Generator) -> str:
    """Uniform permutation of nucleotides. Preserves base counts and nothing else."""
    idx = rng.permutation(len(cds))
    return "".join(cds[i] for i in idx)


def _dinuc_shuffle(cds: str, rng: np.random.Generator) -> str:
    """Altschul-Erikson shuffle: exact dinucleotide counts, everything above destroyed.

    Uniformly samples an Eulerian walk on the de Bruijn graph of order 1 by
    drawing a random last-edge tree rooted at the final vertex, checking it is a
    tree, then shuffling the remaining out-edges (Altschul & Erikson 1985). Every
    dinucleotide count is preserved exactly, which is what makes it the right
    control for a model whose sensitivity might be nothing but local base
    composition.
    """
    if len(cds) < 3:
        return cds
    verts = sorted(set(cds))
    if len(verts) == 1:
        return cds
    last_v = cds[-1]
    edges = {v: [] for v in verts}
    for a, b in zip(cds[:-1], cds[1:]):
        edges[a].append(b)

    last_edge = None
    for _ in range(200):
        cand = {}
        for v in verts:
            if v == last_v or not edges[v]:
                continue
            cand[v] = edges[v][int(rng.integers(len(edges[v])))]
        ok = True
        for v in verts:
            if v == last_v or not edges[v]:
                continue
            seen, u = set(), v
            while u != last_v:
                if u in seen or u not in cand:
                    ok = False
                    break
                seen.add(u)
                u = cand[u]
            if not ok:
                break
        if ok:
            last_edge = cand
            break
    if last_edge is None:                       # no connecting tree found; give up
        return cds

    new_edges = {}
    for v in verts:
        e = list(edges[v])
        if v in last_edge:
            e.remove(last_edge[v])
        e = [e[i] for i in rng.permutation(len(e))]
        if v in last_edge:
            e.append(last_edge[v])
        new_edges[v] = e

    out = [cds[0]]
    ptr = {v: 0 for v in verts}
    cur = cds[0]
    for _ in range(len(cds) - 1):
        nxt = new_edges[cur][ptr[cur]]
        ptr[cur] += 1
        out.append(nxt)
        cur = nxt
    return "".join(out)


# ------------------------------------------------------------------- registry

REGISTRY: dict[str, Intervention] = {}


def register(iv: Intervention) -> Intervention:
    REGISTRY[iv.name] = iv
    return iv


register(Intervention(
    name="synonymous recode",
    fn=_synonymous_recode,
    preserves=(PROTEIN_IDENTITY, READING_FRAME, LENGTH, STRAND),
    destroys=(CODON_USAGE, CODON_ADJACENCY, NUC_COMPOSITION, DINUC_COMPOSITION,
              KMER_COMPOSITION),
    probes="codon-level surface form",
    expect=NO_PREFERENCE,
    stochastic=True,
    rationale=("The protein is identical by construction, so any preference for "
               "the real sequence is a preference over encodings of one protein "
               "and cannot be protein-level."),
    claim_if_significant=("reads codon-level surface form: it prefers the native "
                          "encoding of a protein over an equivalent one"),
    claim_if_null="is invariant to how the protein is encoded",
))

register(Intervention(
    name="codon-order shuffle",
    fn=_codon_shuffle,
    preserves=(CODON_USAGE, READING_FRAME, NUC_COMPOSITION, LENGTH, STRAND),
    destroys=(PROTEIN_IDENTITY, CODON_ADJACENCY, DINUC_COMPOSITION, KMER_COMPOSITION),
    probes="codon context",
    expect=REAL_HIGHER,
    stochastic=True,
    rationale=("Codon usage is preserved exactly and the frame is untouched, so a "
               "penalty here is for disrupted codon-to-codon context, not for "
               "composition. It does not by itself show protein representation, "
               "because codon order is also a nucleotide-level statistic."),
    claim_if_significant="represents local codon context",
    claim_if_null="is indifferent to the order its codons appear in",
))

register(Intervention(
    name="frameshift +1",
    fn=_frameshift(1),
    preserves=(NUC_COMPOSITION, DINUC_COMPOSITION, KMER_COMPOSITION, LENGTH, STRAND),
    destroys=(READING_FRAME, PROTEIN_IDENTITY, CODON_USAGE, CODON_ADJACENCY),
    probes=READING_FRAME,
    expect=REAL_HIGHER,
    rationale=("A cyclic rotation has identical k-mer content at every order, so "
               "the only thing that changed is the phase codons are read in. "
               "Reading-frame discrimination is a NECESSARY condition for a "
               "likelihood to be a protein-level quantity, because the frame is "
               "exactly what determines which protein a sequence encodes."),
    claim_if_significant="represents the reading frame",
    claim_if_null=("cannot separate a gene from the same gene read out of frame, "
                   "so its likelihood is not a protein-level quantity"),
))

register(Intervention(
    name="frameshift +2",
    fn=_frameshift(2),
    preserves=(NUC_COMPOSITION, DINUC_COMPOSITION, KMER_COMPOSITION, LENGTH, STRAND),
    destroys=(READING_FRAME, PROTEIN_IDENTITY, CODON_USAGE, CODON_ADJACENCY),
    probes=READING_FRAME,
    expect=REAL_HIGHER,
    rationale="The other alternative phase; an independent replicate of the frame test.",
    claim_if_significant="represents the reading frame",
    claim_if_null="cannot separate a gene from its other out-of-frame reading",
))

register(Intervention(
    name="reverse complement",
    fn=_revcomp,
    preserves=(LENGTH,),
    destroys=(STRAND, PROTEIN_IDENTITY, READING_FRAME, CODON_USAGE, CODON_ADJACENCY),
    probes=STRAND,
    expect=REAL_HIGHER,
    rationale=("Base counts are complemented rather than preserved, and a "
               "double-stranded genome contains both strands, so this is a weaker "
               "test than the frameshift. It asks whether the model knows which "
               "strand carries the gene."),
    claim_if_significant="represents strand / coding orientation",
    claim_if_null="does not distinguish a gene from its reverse complement",
))

register(Intervention(
    name="dinucleotide shuffle",
    fn=_dinuc_shuffle,
    preserves=(NUC_COMPOSITION, DINUC_COMPOSITION, LENGTH),
    destroys=(PROTEIN_IDENTITY, READING_FRAME, CODON_USAGE, CODON_ADJACENCY,
              KMER_COMPOSITION),
    probes="structure above dinucleotide",
    expect=REAL_HIGHER,
    stochastic=True,
    rationale=("Exact dinucleotide counts, everything higher destroyed. A model "
               "that fails this has learned nothing beyond local base composition "
               "and no other result about it is interpretable."),
    claim_if_significant="represents sequence structure above dinucleotide composition",
    claim_if_null="has learned nothing beyond dinucleotide composition",
    requires_cds=False,
))

register(Intervention(
    name="mononucleotide shuffle",
    fn=_mono_shuffle,
    preserves=(NUC_COMPOSITION, LENGTH),
    destroys=(PROTEIN_IDENTITY, READING_FRAME, CODON_USAGE, CODON_ADJACENCY,
              DINUC_COMPOSITION, KMER_COMPOSITION),
    probes="structure above base composition",
    expect=REAL_HIGHER,
    stochastic=True,
    rationale=("The floor of the battery, and the reference effect the nulls are "
               "measured against. Every property this destroys is destroyed by "
               "every other intervention too, so its delta upper-bounds nothing "
               "but it lower-bounds what the scorer can resolve at this n."),
    claim_if_significant="distinguishes real sequence from base-composition noise",
    claim_if_null="cannot distinguish a real gene from a random sequence of the same bases",
    requires_cds=False,
    tags=("reference",),
))

DEFAULT_BATTERY = ["synonymous recode", "codon-order shuffle", "frameshift +1",
                   "frameshift +2", "reverse complement", "dinucleotide shuffle",
                   "mononucleotide shuffle"]

SECTION_26_BATTERY = ["synonymous recode", "codon-order shuffle", "frameshift +1",
                      "reverse complement"]


def get(name: str) -> Intervention:
    if name not in REGISTRY:
        raise KeyError(f"{name!r} not registered; have {sorted(REGISTRY)}")
    return REGISTRY[name]


def battery(names=None) -> list[Intervention]:
    return [get(n) for n in (names or DEFAULT_BATTERY)]


# ------------------------------------------------- matched-edit frame probes
#
# The cyclic frameshift has an objection attached to it. For a single-nucleotide
# model it is a *small* input edit -- a rotation leaves the local context of
# every base unchanged except at one junction -- so a null there might be
# architectural rather than informative. The pair below removes that objection by
# making the two conditions equally large edits that differ only in phase.
#
# Write the trinucleotide TAA over three bases a quarter of the way into the
# gene, either on a codon boundary or one nucleotide past it. On the boundary it
# is a premature stop and the protein is truncated. One nucleotide past it, it
# provably is not: the in-frame codons become s[o]+"TA" and "A"+s[o+4]+s[o+5],
# and neither can be TAA, TAG or TGA whatever s is, because every stop codon
# begins with T and has A or G second, and "A"+xx never starts with T. Same
# edit size, same inserted trinucleotide, same neighbourhood; only the phase
# differs, and only the in-frame one destroys the protein.

def _stop_at(offset_fraction: float, phase: int):
    def f(cds: str, rng: np.random.Generator) -> str:
        o = 3 * int(len(cds) * offset_fraction / 3) + phase
        o = min(max(o, 3), len(cds) - 9)
        return cds[:o] + "TAA" + cds[o + 3:]
    return f


register(Intervention(
    name="stop in frame",
    fn=_stop_at(0.25, 0),
    preserves=(READING_FRAME, LENGTH, STRAND),
    destroys=(PROTEIN_IDENTITY,),
    probes="premature termination",
    expect=REAL_HIGHER,
    rationale=("A stop codon written on a codon boundary a quarter of the way in "
               "truncates the protein. Only meaningful next to its out-of-phase "
               "twin; alone it is confounded with the three bases it overwrote."),
    claim_if_significant="penalises a premature stop codon",
    claim_if_null="does not penalise a premature stop codon",
    tags=("frame-pair",),
))

register(Intervention(
    name="stop out of frame",
    fn=_stop_at(0.25, 1),
    preserves=(READING_FRAME, LENGTH, STRAND, PROTEIN_IDENTITY),
    destroys=(CODON_USAGE,),
    probes="matched control for premature termination",
    expect=REAL_HIGHER,
    rationale=("The same three bases written one nucleotide later, where they "
               "provably cannot form a stop in the reading frame. Changes two "
               "codons and leaves the protein truncation-free."),
    claim_if_significant="penalises the matched non-terminating edit",
    claim_if_null="does not penalise the matched non-terminating edit",
    tags=("frame-pair",),
))


@dataclass(frozen=True)
class Contrast:
    """A difference of two paired deltas, where the two interventions are matched.

    The single-intervention deltas are each confounded with the size of the edit.
    Their difference is not, when the two edits are the same size and differ only
    in the property under test. `expect_a_lower` says a model with the property
    must score `a` below `b`.
    """

    name: str
    a: str
    b: str
    probes: str
    rationale: str
    claim_if_significant: str
    claim_if_null: str
    expect_a_lower: bool = True


CONTRASTS: dict[str, Contrast] = {
    "frame-anchored stop": Contrast(
        name="frame-anchored stop",
        a="stop in frame", b="stop out of frame",
        probes=READING_FRAME,
        rationale=("Two edits of identical size inserting the identical "
                   "trinucleotide three bases apart. One terminates the protein "
                   "and one cannot. Any model that reads the sequence as a gene "
                   "must separate them; a model reading only local nucleotide "
                   "statistics has no basis to."),
        claim_if_significant=("distinguishes a premature stop from the same bases "
                              "written out of phase, so it tracks the reading frame"),
        claim_if_null=("scores a premature stop the same as the identical bases "
                       "written one nucleotide off, so it is not reading the "
                       "sequence in any frame"),
    ),
}

DEFAULT_BATTERY = DEFAULT_BATTERY + ["stop in frame", "stop out of frame"]
