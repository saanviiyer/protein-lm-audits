#!/usr/bin/env python3
"""Find native coding sequences for the ProteinGym DMS assays.

The genomic rung so far is one protein. To ask whether genomic->protein transfer
generalises, each DMS assay needs the real CDS whose translation contains the
assayed protein -- not a reverse translation, which would bake in this script's
codon preferences and make the synonymous control circular.

An assay is kept only if some ENA CDS for its UniProt accession translates to a
protein CONTAINING the assay's target sequence exactly, and the offset of that
match is recorded so variant codons can be placed correctly. Anything that does
not verify is dropped and listed, because a silently mismatched CDS would look
like a genomic model failing rather than a join failing.
"""
import csv, json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from fetch_cds import embl_protein_ids, ena_cds, translate

GAUNTLET = Path("/Users/saanviiyer/Downloads/CALTECH/RESEARCH/gauntlet/data/proteingym")


def main():
    ref = list(csv.DictReader((GAUNTLET / "reference.csv").open()))
    have = {p.stem for p in (GAUNTLET / "assays").glob("*.csv")}
    todo = [r for r in ref if r["DMS_id"] in have]
    print(f"{len(ref)} assays in reference, {len(have)} downloaded, {len(todo)} to resolve\n")

    out, failed = {}, []
    for r in todo:
        dms, acc, tgt = r["DMS_id"], r["UniProt_ID"], r["target_seq"]
        try:
            pids = embl_protein_ids(acc)
        except Exception as e:
            print(f"  {dms:38s} uniprot lookup failed: {e}")
            failed.append((dms, acc, "uniprot")); continue
        hit = None
        for pid in pids[:12]:
            try:
                cds = ena_cds(pid)
            except Exception:
                continue
            if not cds or len(cds) % 3:
                continue
            prot = translate(cds).rstrip("*")
            off = prot.find(tgt)
            if off >= 0:
                hit = dict(dms_id=dms, uniprot=acc, protein_id=pid, cds=cds,
                           aa_offset=off, target_len=len(tgt),
                           full_protein_len=len(prot))
                break
        if hit:
            out[dms] = hit
            print(f"  {dms:38s} OK  {hit['protein_id']:14s} offset {hit['aa_offset']:4d} "
                  f"target {len(tgt):4d} aa of {hit['full_protein_len']}")
        else:
            failed.append((dms, acc, f"no CDS containing target ({len(pids)} ids)"))
            print(f"  {dms:38s} --  no CDS whose translation contains the target")

    dest = ROOT / "data" / "cds" / "dms_cds.json"
    dest.write_text(json.dumps(out, indent=2))
    print(f"\nverified {len(out)}/{len(todo)} assays -> {dest}")
    if failed:
        print(f"unresolved ({len(failed)}):")
        for d, a, why in failed:
            print(f"  {d:38s} {a:12s} {why}")


if __name__ == "__main__":
    main()
