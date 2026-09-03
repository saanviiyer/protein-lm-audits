#!/usr/bin/env python3
"""Locate the par loci in the M. opportunistum genome and save the real operon context.

The protein rung had to invent a linker (25 glycines) to put two chains in one
context, because proteins are not naturally concatenated. DNA is: ParD3 and ParE3
are adjacent genes in one operon, so the genomic partner-aware context is the
sequence the organism actually has and the model was actually trained on. That is
a genuine advantage of the genomic rung, and it is only real if the coordinates
are read off the genome rather than assumed, which is what this does.
"""
import json, sys, urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GENOME_ACC = "CP002279"   # M. opportunistum WSM2075 chromosome
COMP = str.maketrans("ACGT", "TGCA")


def rc(s: str) -> str:
    return s.translate(COMP)[::-1]


def fetch_genome() -> str:
    cache = ROOT / "data" / "cds" / f"{GENOME_ACC}.txt"
    if cache.exists():
        return cache.read_text()
    url = f"https://www.ebi.ac.uk/ena/browser/api/fasta/{GENOME_ACC}?download=false&lineLimit=0"
    with urllib.request.urlopen(url, timeout=300) as r:
        txt = r.read().decode()
    seq = "".join(l.strip() for l in txt.splitlines() if not l.startswith(">")).upper()
    cache.write_text(seq)
    return seq


def locate(genome: str, cds: str):
    """Return (start, end, strand) of an exact CDS match, 0-based half-open."""
    i = genome.find(cds)
    if i >= 0:
        return i, i + len(cds), "+"
    j = genome.find(rc(cds))
    if j >= 0:
        return j, j + len(cds), "-"
    return None


def main():
    genome = fetch_genome()
    print(f"genome {GENOME_ACC}: {len(genome):,} nt")
    cds = json.loads((ROOT / "data" / "cds" / "cds.json").read_text())

    loc = {}
    for name, rec in cds.items():
        hit = locate(genome, rec["cds"])
        if hit is None:
            print(f"  {name}: NOT FOUND in genome (unexpected)")
            continue
        s, e, strand = hit
        loc[name] = dict(start=s, end=e, strand=strand)
        print(f"  {name:6s} {s:>9,}-{e:<9,} strand {strand}  len {e-s}")

    out = {}
    for anti, tox in (("ParD3", "ParE3"), ("ParD2", "ParE2")):
        if anti not in loc or tox not in loc:
            continue
        a, t = loc[anti], loc[tox]
        if a["strand"] != t["strand"]:
            print(f"  !! {anti}/{tox} on opposite strands; not one operon")
            continue
        lo, hi = min(a["start"], t["start"]), max(a["end"], t["end"])
        seg = genome[lo:hi]
        if a["strand"] == "-":
            seg = rc(seg)
        gap = max(a["start"], t["start"]) - min(a["end"], t["end"])
        first = anti if (a["start"] < t["start"]) == (a["strand"] == "+") else tox
        print(f"  operon {anti}/{tox}: {len(seg)} nt, intergenic gap {gap} nt, "
              f"strand {a['strand']}, {first} first")
        out[f"{anti}_{tox}"] = dict(segment=seg, strand=a["strand"], gap=gap,
                                    first=first, span=[lo, hi],
                                    offset_in_segment={
                                        n: (loc[n]["start"] - lo if a["strand"] == "+"
                                            else hi - loc[n]["end"])
                                        for n in (anti, tox)})

    # cross-operon context: ParD3 with the NON-cognate toxin ParE2, which is the
    # counterfactual the specificity question asks about and which does not exist
    # in any genome. Built explicitly so it is never confused with a real locus.
    if "ParD3_ParE3" in out and "ParD2_ParE2" in out:
        d3 = cds["ParD3"]["cds"]
        e2 = cds["ParE2"]["cds"]
        gap3 = out["ParD3_ParE3"]["gap"]
        spacer = out["ParD3_ParE3"]["segment"]
        # reuse the real ParD3/ParE3 intergenic spacer so only the toxin gene changes
        d3_off = out["ParD3_ParE3"]["offset_in_segment"]["ParD3"]
        e3_off = out["ParD3_ParE3"]["offset_in_segment"]["ParE3"]
        if e3_off > d3_off:
            inter = spacer[d3_off + len(d3):e3_off]
            out["ParD3_ParE2_synthetic"] = dict(
                segment=d3 + inter + e2, intergenic=inter, note=
                "synthetic: real ParD3 gene + real ParD3-ParE3 intergenic + real ParE2 gene")
            print(f"  synthetic ParD3/ParE2 context: {len(d3+inter+e2)} nt "
                  f"(intergenic {len(inter)} nt reused from the cognate operon)")

    dest = ROOT / "data" / "cds" / "genomic_context.json"
    dest.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {dest}")


if __name__ == "__main__":
    main()
