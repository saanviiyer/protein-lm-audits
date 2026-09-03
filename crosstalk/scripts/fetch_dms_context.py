#!/usr/bin/env python3
"""Put each DMS coding sequence back into its real genomic neighbourhood.

The strongest objection to section 18 is that genomic language models are built
for long-range context and were handed a 300-1,200 nt coding sequence in
isolation -- the wrong input, so of course they fail. This answers it directly by
scoring the identical variants inside the sequence the organism actually carries,
flanks and all.

Route: UniProt EMBL cross-references give both the parent nucleotide accession and
its MoleculeType, so genuinely genomic records can be preferred over mRNA. The
record is fetched, oriented so the CDS reads forward, and a window is cut around
the assayed region.

Two details that would otherwise corrupt the comparison:

  ORIENTATION  a CDS on the minus strand is located by reverse-complementing the
               whole record first, so flanks are the real upstream and downstream
               sequence rather than mirrored coordinates.
  ALIGNMENT    Nucleotide Transformer tokenises non-overlapping 6-mers, so a codon
               only stays inside one token if its offset is a multiple of 3. The
               window start is snapped so the assayed region begins at a multiple
               of 6, trimming at most five bases of flank. Without this, codons
               straddle token boundaries and the scores are not comparable to the
               unflanked run.
"""
import json, sys, urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from fetch_cds import get

COMP = str.maketrans("ACGT", "TGCA")
MAX_RECORD = 40_000_000
MAX_SEGMENT = 11_400          # under NT-v2's 2048-token context


def rc(s):
    return s.translate(COMP)[::-1]


def embl_refs(acc):
    """(nucleotide_id, protein_id, molecule_type), genomic records first."""
    j = json.loads(get(f"https://rest.uniprot.org/uniprotkb/{acc}.json"))
    refs = []
    for x in j.get("uniProtKBCrossReferences", []):
        if x.get("database") != "EMBL":
            continue
        pid = mol = None
        for p in x.get("properties", []):
            if p.get("key") in ("ProteinId", "protein_sequence_id"):
                pid = p.get("value")
            elif p.get("key") == "MoleculeType":
                mol = p.get("value")
        if x.get("id") and pid and pid != "-":
            refs.append((x["id"], pid, mol or "?"))
    refs.sort(key=lambda r: 0 if r[2] == "Genomic_DNA" else 1)
    return refs


def fetch_record(acc):
    cache = ROOT / "data" / "cds" / "records" / f"{acc}.txt"
    cache.parent.mkdir(parents=True, exist_ok=True)
    if cache.exists():
        return cache.read_text()
    url = f"https://www.ebi.ac.uk/ena/browser/api/fasta/{acc}?download=false&lineLimit=0"
    txt = get(url, tries=2)
    seq = "".join(l.strip() for l in txt.splitlines() if not l.startswith(">")).upper()
    if len(seq) > MAX_RECORD:
        raise ValueError(f"record too large ({len(seq)})")
    cache.write_text(seq)
    return seq


def main():
    flank = int(sys.argv[1]) if len(sys.argv) > 1 else 3000
    cdsmap = json.loads((ROOT / "data" / "cds" / "dms_cds.json").read_text())
    out, failed = {}, []
    for dms, rec in sorted(cdsmap.items()):
        cds, off, tlen = rec["cds"], rec["aa_offset"], rec["target_len"]
        got = None
        try:
            refs = embl_refs(rec["uniprot"])
        except Exception as e:
            failed.append((dms, f"uniprot: {e}")); continue
        for nuc, pid, mol in refs[:8]:
            try:
                r = fetch_record(nuc)
            except Exception:
                continue
            oriented, strand = (r, "+") if cds in r else (rc(r), "-")
            i = oriented.find(cds)
            if i < 0:
                continue
            ts, te = i + off * 3, i + off * 3 + tlen * 3
            room = max((MAX_SEGMENT - (te - ts)) // 2, 0)
            f = min(flank, room)
            lo, hi = max(ts - f, 0), min(te + f, len(oriented))
            lo += (ts - lo) % 6                      # keep the codon frame in-token
            seg = oriented[lo:hi]
            got = dict(dms_id=dms, record=nuc, molecule_type=mol, strand=strand,
                       record_len=len(r), segment=seg, cds_offset=ts - lo,
                       upstream=ts - lo, downstream=hi - te, target_len=tlen)
            break
        if got:
            out[dms] = got
            print(f"  {dms[:36]:36s} {got['record']:12s} {got['molecule_type']:12s} "
                  f"rec {got['record_len']:>9,}  up {got['upstream']:>5}  "
                  f"down {got['downstream']:>5}")
        else:
            failed.append((dms, "no record containing the CDS"))
            print(f"  {dms[:36]:36s} --")
    dest = ROOT / "data" / "cds" / "dms_context.json"
    dest.write_text(json.dumps(out, indent=2))
    n_gen = sum(1 for v in out.values() if v["molecule_type"] == "Genomic_DNA")
    print(f"\ncontext for {len(out)}/{len(cdsmap)} assays ({n_gen} genomic DNA) -> {dest}")
    for d, why in failed:
        print(f"  unresolved {d}: {why}")


if __name__ == "__main__":
    main()
