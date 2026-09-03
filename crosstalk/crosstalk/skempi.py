"""A measured, multi-system two-sided specificity set built from SKEMPI 2.0.

The ParD3 landscape is exhaustive but singular: one antitoxin, three positions,
one off-target. Any claim about proxies that rests on it alone is a claim about
one protein. SKEMPI 2.0 (Jankauskaite et al. 2019) curates measured binding
affinities for mutations in solved complexes, and buried inside it is the
structure this project needs: proteins whose *same* mutations were measured
against *two different partners*.

That intersection is the two-sided specificity set. For a mutation m of protein
P measured in complex with partner A and with partner B:

    ddG_A = RT ln(Kd_mut_A / Kd_wt_A)        (destabilising is positive)
    margin(m) = ddG_B - ddG_A

A mutation with a large positive margin hurts binding to B more than to A, which
is what a specificity-increasing design is trying to achieve. This is the same
quantity the ParD3 landscape measures directly as W_on - W_off, now across about
ten independent biological systems -- protease/inhibitor, hormone/receptor,
TCR/pMHC, enzyme/inhibitor -- rather than one.

Two disciplines carried over from the rest of the project:

  VERIFY, DO NOT ASSUME  Mutation positions are in PDB author numbering, which
                         does not index the sequence. Every mutation is checked
                         against the residue actually present at that position in
                         that chain, and discarded if it disagrees. A silent
                         off-by-one would read as a modelling result.

  KEEP THE TRIVIAL BASELINE  These systems differ in scale and assay, so anything
                         reported here is reported per-system as well as pooled.
"""
from __future__ import annotations

import collections
import csv
import gzip
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np

DATA = Path(__file__).resolve().parents[1] / "data" / "skempi"
RT = 1.9872041e-3 * 298.15          # kcal/mol at 25 C

THREE_TO_ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q",
    "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K",
    "MET": "M", "PHE": "F", "PRO": "P", "SER": "S", "THR": "T", "TRP": "W",
    "TYR": "Y", "VAL": "V", "MSE": "M", "SEC": "U", "PYL": "O"}


@dataclass
class TwoSided:
    """One protein whose mutations were measured against two partners."""
    protein: str
    partner_a: str
    partner_b: str
    pdb_a: str
    pdb_b: str
    chain: str
    sequence: str                    # the mutated chain, from the A-side structure
    mutations: list[str]             # "L38G" in *sequence* (1-based) coordinates
    ddg_a: np.ndarray
    ddg_b: np.ndarray

    @property
    def margin(self) -> np.ndarray:
        """Positive = the mutation costs more binding to B than to A."""
        return self.ddg_b - self.ddg_a

    def __len__(self) -> int:
        return len(self.mutations)


def _parse_affinity(x: str) -> float | None:
    try:
        v = float(x)
        return v if v > 0 else None
    except (TypeError, ValueError):
        return None


def fetch_pdb(pdb: str) -> str:
    """Cache a PDB entry locally; return its text."""
    DATA.mkdir(parents=True, exist_ok=True)
    dest = DATA / "pdb" / f"{pdb.lower()}.pdb"
    dest.parent.mkdir(exist_ok=True)
    if not dest.exists():
        url = f"https://files.rcsb.org/download/{pdb.upper()}.pdb.gz"
        with urllib.request.urlopen(url, timeout=120) as r:
            dest.write_bytes(gzip.decompress(r.read()))
    return dest.read_text()


def chain_residues(pdb_text: str, chain: str) -> tuple[str, dict[str, int]]:
    """Sequence of a chain from its CA atoms, plus {author_resnum: seq_index}.

    Author numbering is what SKEMPI quotes and it is not the sequence index --
    it skips, restarts, and carries insertion codes. Building the map from the
    structure itself is the only way to convert one to the other reliably.
    """
    seq, numbering, seen = [], {}, set()
    for line in pdb_text.splitlines():
        if not line.startswith("ATOM") or line[12:16].strip() != "CA":
            continue
        if line[21] != chain:
            continue
        resnum = line[22:27].strip()          # includes insertion code
        if resnum in seen:
            continue
        aa = THREE_TO_ONE.get(line[17:20].strip().upper())
        if aa is None:
            continue
        seen.add(resnum)
        numbering[resnum] = len(seq)
        seq.append(aa)
    return "".join(seq), numbering


def load_rows(path: Path | None = None) -> list[dict]:
    path = path or (DATA / "skempi_v2.csv")
    return list(csv.DictReader(path.open(), delimiter=";"))


def _mutation_records(rows):
    """Index rows by (mutated protein, partner) -> {cleaned key -> record}.

    SKEMPI carries two numberings and they disagree on 63% of rows.
    `Mutation(s)_cleaned` is renumbered to be comparable across entries, so it is
    the right key for asking "was this same mutation also measured against the
    other partner". `Mutation(s)_PDB` is author numbering in that row's own
    structure, so it is the only one that can be looked up in a PDB file.
    Using the cleaned numbering for the structural lookup silently loses most
    systems -- it was worth an hour to notice, and it is the kind of off-by-one
    that would otherwise have read as "these proxies do poorly here".
    """
    idx = collections.defaultdict(dict)
    for r in rows:
        try:
            pdb, c1, c2 = r["#Pdb"].split("_")
        except ValueError:
            continue
        clean = [m for m in r["Mutation(s)_cleaned"].split(",") if m]
        raw = [m for m in r["Mutation(s)_PDB"].split(",") if m]
        if len(clean) != 1 or len(raw) != 1:      # single mutants only
            continue
        mc, mp = clean[0], raw[0]
        ch = mp[1]
        in1, in2 = ch in c1, ch in c2
        if in1 == in2:
            continue
        mutprot = r["Protein 1"] if in1 else r["Protein 2"]
        partner = r["Protein 2"] if in1 else r["Protein 1"]
        kd_mut = _parse_affinity(r["Affinity_mut_parsed"])
        kd_wt = _parse_affinity(r["Affinity_wt_parsed"])
        if kd_mut is None or kd_wt is None:
            continue
        key = mc[0] + mc[2:]                      # comparable across entries
        idx[(mutprot, partner)][key] = dict(
            ddg=RT * np.log(kd_mut / kd_wt), pdb=pdb, chain=ch,
            resnum=mp[2:-1], wt=mp[0], mut=mp[-1])
    return idx


def build(min_shared: int = 10, verbose: bool = True) -> list[TwoSided]:
    """Every (protein, partner A, partner B) with enough verified shared mutations."""
    idx = _mutation_records(load_rows())
    by_protein = collections.defaultdict(list)
    for (prot, partner) in idx:
        by_protein[prot].append(partner)

    out = []
    for prot, partners in sorted(by_protein.items()):
        for i in range(len(partners)):
            for j in range(i + 1, len(partners)):
                a, b = partners[i], partners[j]
                A, B = idx[(prot, a)], idx[(prot, b)]
                shared = sorted(set(A) & set(B))
                if len(shared) < min_shared:
                    continue
                # reference structure = the PDB covering most of the shared set
                counts = collections.Counter((A[k]["pdb"], A[k]["chain"]) for k in shared)
                (pdb_a, chain_a), _ = counts.most_common(1)[0]
                try:
                    seq, numbering = chain_residues(fetch_pdb(pdb_a), chain_a)
                except Exception as e:
                    if verbose:
                        print(f"  skip {prot} ({pdb_a}): {e}")
                    continue

                keep, ddg_a, ddg_b, bad = [], [], [], 0
                for k in shared:
                    rec = A[k]
                    si = numbering.get(rec["resnum"])
                    if si is None or seq[si] != rec["wt"]:
                        bad += 1
                        continue
                    keep.append(f"{rec['wt']}{si + 1}{rec['mut']}")
                    ddg_a.append(rec["ddg"]); ddg_b.append(B[k]["ddg"])
                if len(keep) < min_shared:
                    if verbose:
                        print(f"  drop {prot[:30]:30s} | {a[:20]:20s} vs {b[:20]:20s} "
                              f"{len(keep):3d} verified of {len(shared)}")
                    continue
                out.append(TwoSided(prot, a, b, pdb_a, B[shared[0]]["pdb"], chain_a,
                                    seq, keep, np.array(ddg_a), np.array(ddg_b)))
                if verbose:
                    print(f"  keep {prot[:30]:30s} | {a[:20]:20s} vs {b[:20]:20s} "
                          f"n={len(keep):3d} ({bad} unmatched) {pdb_a}:{chain_a} "
                          f"len {len(seq)}")
    return out
