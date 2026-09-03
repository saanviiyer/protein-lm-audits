#!/usr/bin/env python3
"""Fetch the real coding DNA sequences for the ParD3/ParE landscape proteins.

The genomic rung of the proxy ladder scores DNA, so it needs the actual CDS the
organism uses, not a reverse translation invented here. Every CDS is verified by
translating it and requiring an exact match to the protein sequence already
pinned in crosstalk.boltz -- a silent frameshift or a wrong paralog would look
like a modelling result rather than a data error.

Route: UniProt JSON -> EMBL cross-reference ProteinId -> ENA CDS fasta.
"""
import json, sys, time, urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from crosstalk.boltz import PARD3, PARE3, PARD2, PARE2

ACCESSIONS = {"ParD3": ("F7YBW8", PARD3), "ParE3": ("F7YBW7", PARE3),
              "ParD2": ("F7Y4V9", PARD2), "ParE2": ("F7Y4W0", PARE2)}

CODON_TABLE = {}
for i, b1 in enumerate("TCAG"):
    for b2 in "TCAG":
        for b3 in "TCAG":
            CODON_TABLE[b1 + b2 + b3] = "FFLLSSSSYY**CC*WLLLLPPPPHHQQRRRRIIIMTTTTNNKKSSRRVVVVAAAADDEEGGGG"[
                len(CODON_TABLE)]


def translate(dna: str) -> str:
    dna = dna.upper().replace("U", "T")
    return "".join(CODON_TABLE.get(dna[i:i + 3], "X") for i in range(0, len(dna) - 2, 3))


def get(url: str, tries: int = 3) -> str:
    for k in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                return r.read().decode()
        except Exception as e:
            if k == tries - 1:
                raise
            time.sleep(2 * (k + 1))
    raise RuntimeError


def embl_protein_ids(acc: str) -> list[str]:
    j = json.loads(get(f"https://rest.uniprot.org/uniprotkb/{acc}.json"))
    ids = []
    for x in j.get("uniProtKBCrossReferences", []):
        if x.get("database") == "EMBL":
            for p in x.get("properties", []):
                if p.get("key") in ("ProteinId", "protein_sequence_id") and p.get("value") not in ("-", None):
                    ids.append(p["value"])
    return ids


def ena_cds(protein_id: str) -> str | None:
    """ENA serves the CDS nucleotide sequence under the EMBL-CDS namespace."""
    for url in (f"https://www.ebi.ac.uk/ena/browser/api/fasta/{protein_id}",
                f"https://www.ebi.ac.uk/ena/browser/api/fasta/{protein_id.split('.')[0]}"):
        try:
            txt = get(url)
        except Exception:
            continue
        seq = "".join(l.strip() for l in txt.splitlines() if l and not l.startswith(">"))
        if seq:
            return seq.upper()
    return None


def main():
    out, failures = {}, []
    for name, (acc, protein) in ACCESSIONS.items():
        print(f"=== {name} ({acc}, {len(protein)} aa) ===", flush=True)
        found = None
        for pid in embl_protein_ids(acc):
            cds = ena_cds(pid)
            if not cds:
                print(f"    {pid}: no sequence returned")
                continue
            aa = translate(cds).rstrip("*")
            if aa == protein:
                print(f"    {pid}: {len(cds)} nt, translation EXACT match")
                found = dict(accession=acc, protein_id=pid, cds=cds, protein=protein)
                break
            print(f"    {pid}: {len(cds)} nt, translation mismatch "
                  f"({sum(a != b for a, b in zip(aa, protein))} diffs, len {len(aa)} vs {len(protein)})")
        if found:
            out[name] = found
        else:
            failures.append(name)
            print(f"    !! no verified CDS for {name}")
    dest = Path(__file__).resolve().parents[1] / "data" / "cds" / "cds.json"
    dest.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {dest} with {len(out)}/{len(ACCESSIONS)} verified CDS")
    if failures:
        print(f"UNVERIFIED: {failures}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
