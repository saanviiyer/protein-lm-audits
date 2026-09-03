#!/usr/bin/env python3
"""Windows, annotations and matched-composition controls for the DNA memorisation test.

The membership facts this rests on, all verified from the corpus manifest that
ships with `InstaDeepAI/multi_species_genomes` (mirrored in
`data/genomes/nt_multispecies_urls.txt`, 850 lines, identical accession order to
`original_urls.csv`):

  * the loader splits by position -- train = urls[:-100], validation = urls[-100:-50],
    test = urls[-50:] -- so membership of a genome is decided by its line number.
  * line 330 (0-based 329) is Mesorhizobium albiziae GCF_900114255.1. It is in the
    TRAIN split. This is a VERIFIED member of NT-v2 multi-species pretraining.
  * `Mesorhizobium opportunistum` does not appear anywhere in the 850. The
    chromosome this project has on disk, CP002279, was therefore NEVER SEEN by
    NT-v2 multi-species. The premise that it is the training-plausible genome is
    false, and the arms below are built accordingly.
  * of the 100 held-out genomes, 99 are eukaryotes and exactly one is a bacterium
    (E. coli K-12, line 849). The corpus's own held-out split is therefore useless
    as a composition-matched bacterial control, which is why the non-member arm is
    built from genomes first released to ENA in 2025-2026, three years after the
    model.
"""
from __future__ import annotations

import gzip
import re
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
GEN = ROOT / "data" / "genomes"

# ------------------------------------------------------------------ genomes
# label -> (file, arm).  "member" = verified in the NT-v2 multi-species TRAIN
# split; "nonmember" = first public 2025-2026, three years after the corpus.
MEMBERS = {
    "M.albiziae":            "M_329_Mesorhizobium_albiziae.fna.gz",
    "Sphingomicrobium":      "M_332_Sphingomicrobium_aestuariivivum.fna.gz",
    "Tolypothrix":           "M_154_Tolypothrix_bouteillei.fna.gz",
    "Fulvimonas":            "M_405_Fulvimonas_soli.fna.gz",
    "Luteococcus":           "M_49_Luteococcus_japonicus.fna.gz",
    "Herminiimonas":         "M_74_Herminiimonas_arsenitoxidans.fna.gz",
    "Agitococcus":           "M_549_Agitococcus_lubricus.fna.gz",
    "Arboricoccus":          "M_375_Arboricoccus_pini.fna.gz",
    "Salibacter":            "M_597_Salibacter_halophilus.fna.gz",
    "Abiotrophia":           "M_59_Abiotrophia_defectiva.fna.gz",
    "Sulfobacillus":         "M_520_Sulfobacillus_thermosulfidooxidans.fna.gz",
}
NONMEMBERS = {
    "Mesorhizobium_U510":    "N_CP180532.fa",
    "Mesorhizobium_MC2":     "N_CP195205.fa",
    "Streptosporangiaceae":  "N_OZ507482.fa",
    "Rhodospirillaceae":     "N_OZ510882.fa",
    "Leptolyngbya":          "N_OZ503973.fa",
    "Xanthomonas_2026":      "N_OZ502491.fa",
    "Bacteroidota":          "N_OZ507335.fa",
    "Pseudomonas_2026":      "N_CP166784.fa",
    "Lysobacterales":        "N_OZ505920.fa",
    "Rubrobacteraceae":      "N_OZ509738.fa",
    "Chitinophagia":         "N_OZ503260.fa",
    "Acidobacteriota":       "N_OZ506907.fa",
    "Steroidobacteraceae":   "N_OZ508125.fa",
}
MESO_MEMBER = "M.albiziae"                       # train split, GC 0.622
MESO_NONMEMBER = ["Mesorhizobium_U510", "Mesorhizobium_MC2"]   # 2026, GC 0.630/0.636


def load_fasta(path: Path) -> dict[str, str]:
    op = gzip.open if str(path).endswith(".gz") else open
    out, name, cur = {}, None, []
    with op(path, "rt") as f:
        for line in f:
            if line.startswith(">"):
                if name is not None:
                    out[name] = "".join(cur)
                name, cur = line[1:].split()[0], []
            else:
                cur.append(line.strip().upper())
    if name is not None:
        out[name] = "".join(cur)
    return out


def load_opportunistum() -> dict[str, str]:
    """CP002279, the project's chromosome. NOT in the NT-v2 corpus."""
    return {"CP002279": (ROOT / "data/cds/CP002279.txt").read_text().strip().upper()}


# -------------------------------------------------------------- annotations
def cds_intervals_embl() -> list[tuple[int, int]]:
    """0-based half-open CDS intervals on CP002279 from the ENA feature table."""
    out = []
    for line in (GEN / "CP002279_cds_ft.txt").read_text().splitlines():
        for a, b in re.findall(r"(\d+)\.\.(\d+)", line):
            out.append((int(a) - 1, int(b)))
    return sorted(out)


def cds_intervals_gff(path: Path) -> dict[str, list[tuple[int, int]]]:
    """0-based half-open CDS intervals per contig from a RefSeq GFF."""
    out: dict[str, list[tuple[int, int]]] = {}
    with gzip.open(path, "rt") as f:
        for line in f:
            if line.startswith("#"):
                continue
            p = line.rstrip("\n").split("\t")
            if len(p) < 8 or p[2] != "CDS":
                continue
            out.setdefault(p[0], []).append((int(p[3]) - 1, int(p[4])))
    return {k: sorted(v) for k, v in out.items()}


def coding_mask(length: int, intervals: list[tuple[int, int]]) -> np.ndarray:
    m = np.zeros(length, bool)
    for a, b in intervals:
        m[max(0, a):min(length, b)] = True
    return m


# ------------------------------------------------------------ window sampling
def sample_windows(seqs: dict[str, str], n: int, W: int, rng,
                   masks: dict[str, np.ndarray] | None = None,
                   frame: int | None = 0) -> list[dict]:
    """n windows of exactly W nt, no ambiguity codes, one contig at a time.

    `frame` pins start % 6. The corpus loader cuts each contig into 6000-nt
    chunks from position 0 and NT tokenizes each chunk from its own start, and
    6000 % 6 == 0, so every 6-mer token the model ever saw began at a contig
    offset divisible by 6. Sampling windows at an arbitrary offset would present
    the same DNA under a token boundary the model never saw, which would hide
    verbatim recall for a reason that has nothing to do with memorisation.
    frame=0 reproduces the training tokenization; frame=3 deliberately breaks it.
    """
    names = [c for c, s in seqs.items() if len(s) >= W + 2]
    if not names:
        return []
    weights = np.array([len(seqs[c]) for c in names], float)
    weights /= weights.sum()
    out, tries = [], 0
    while len(out) < n and tries < n * 60:
        tries += 1
        c = names[rng.choice(len(names), p=weights)]
        s = seqs[c]
        st = int(rng.integers(0, len(s) - W))
        if frame is not None:
            st -= st % 6
            st += frame
            if st < 0 or st + W > len(s):
                continue
        w = s[st:st + W]
        if set(w) - set("ACGT"):
            continue
        rec = dict(contig=c, start=st, seq=w,
                   gc=(w.count("G") + w.count("C")) / W)
        if masks is not None and c in masks:
            rec["coding"] = masks[c][st:st + W]
        out.append(rec)
    return out


def gc_match(a: list[dict], b: list[dict], rng, binw: float = 0.01,
             cap: int | None = None) -> tuple[list[dict], list[dict]]:
    """Trim two window pools to identical GC histograms in `binw`-wide bins."""
    key = lambda r: int(round(r["gc"] / binw))
    A, B = {}, {}
    for r in a:
        A.setdefault(key(r), []).append(r)
    for r in b:
        B.setdefault(key(r), []).append(r)
    ka, kb = [], []
    for k in sorted(set(A) & set(B)):
        m = min(len(A[k]), len(B[k]))
        if cap:
            m = min(m, cap)
        ka += [A[k][i] for i in rng.permutation(len(A[k]))[:m]]
        kb += [B[k][i] for i in rng.permutation(len(B[k]))[:m]]
    return ka, kb


# ------------------------------------------------------- composition controls
def mono_shuffle(s: str, rng) -> str:
    """Exact mononucleotide composition, no higher-order structure."""
    return "".join(np.array(list(s))[rng.permutation(len(s))])


def dinuc_shuffle(s: str, rng) -> str:
    """Altschul-Erikson: exact dinucleotide (and so exact mononucleotide) counts."""
    if len(s) < 3:
        return s
    last = s[-1]
    edges: dict[str, list[str]] = {}
    for x, y in zip(s, s[1:]):
        edges.setdefault(x, []).append(y)
    verts = list(edges)
    for _ in range(500):
        lastedge = {v: edges[v][int(rng.integers(len(edges[v])))]
                    for v in verts if v != last}
        ok = True
        for v in lastedge:                       # every vertex must reach `last`
            cur, seen = v, set()
            while cur != last:
                if cur in seen or cur not in lastedge:
                    ok = False
                    break
                seen.add(cur)
                cur = lastedge[cur]
            if not ok:
                break
        if ok:
            break
    new = {}
    for v in verts:
        e = list(edges[v])
        if v in lastedge:
            e.remove(lastedge[v])
            e = [e[i] for i in rng.permutation(len(e))] + [lastedge[v]]
        else:
            e = [e[i] for i in rng.permutation(len(e))]
        new[v] = e
    out, idx, cur = [s[0]], {v: 0 for v in verts}, s[0]
    for _ in range(len(s) - 1):
        nxt = new[cur][idx[cur]]
        idx[cur] += 1
        out.append(nxt)
        cur = nxt
    return "".join(out)


class Markov5:
    """Order-5 chain fitted to a whole genome.

    Matches the genome's 6-mer frequencies, which is exactly the unigram
    distribution over NT-v2's 4096-token vocabulary, while emitting sequence
    that has never existed. This is the fairest "unseen but matched" arm: a
    shuffle destroys genome grammar, a distant real genome is probably also in
    the corpus, but this is novel by construction and token-matched.
    """

    def __init__(self, seqs: dict[str, str], order: int = 5, pseudo: float = 0.2):
        self.k = order
        idx = {b: i for i, b in enumerate(("A", "C", "G", "T"))}
        cnt = np.full((4 ** order, 4), pseudo)
        for s in seqs.values():
            a = np.frombuffer(s.encode(), dtype=np.uint8)
            code = np.full(len(a), -1, np.int64)
            for b, i in idx.items():
                code[a == ord(b)] = i
            good = code >= 0
            ctx = np.zeros(len(a), np.int64)
            valid = np.ones(len(a), bool)
            for j in range(order):
                ctx = ctx * 4 + np.roll(code, order - j)
                valid &= np.roll(good, order - j)
            valid &= good
            valid[:order] = False
            np.add.at(cnt, (ctx[valid], code[valid]), 1.0)
        self.p = cnt / cnt.sum(1, keepdims=True)
        # Seeds must be sampled from anywhere in the sequence, not from position 0.
        # Human chromosomes begin with megabases of N, so seeding from the start
        # gives an all-N context that no rejection loop can ever escape.
        rng0 = np.random.default_rng(0)
        self.start = []
        for s in seqs.values():
            if len(s) <= order + 1:
                continue
            for _ in range(4000):
                i = int(rng0.integers(0, len(s) - order))
                if not set(s[i:i + order]) - set("ACGT"):
                    self.start.append(s[i:i + order])
                if len(self.start) >= 256:
                    break
            if len(self.start) >= 256:
                break
        if not self.start:
            raise ValueError("no ACGT-only seed context found")

    def emit(self, n: int, rng) -> str:
        bases = "ACGT"
        idx = {b: i for i, b in enumerate(bases)}
        seed = self.start[int(rng.integers(len(self.start)))]
        out = list(seed)
        ctx = 0
        for c in seed:
            ctx = ctx * 4 + idx[c]
        mod = 4 ** self.k
        u = rng.random(n)
        for i in range(n - len(seed)):
            row = self.p[ctx]
            j = int(np.searchsorted(np.cumsum(row), u[i]))
            j = min(j, 3)
            out.append(bases[j])
            ctx = (ctx * 4 + j) % mod
        return "".join(out[:n])
