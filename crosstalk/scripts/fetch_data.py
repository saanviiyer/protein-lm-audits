"""Download the ParD3 specificity landscape from GEO (GSE153897)."""
import sys
import urllib.request
from pathlib import Path

BASE = "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE153nnn/GSE153897/suppl/"
FILES = ["GSE153897_Variant_fitness.csv.gz", "GSE153897_Variant_frequency.csv.gz"]

def main():
    out = Path(__file__).resolve().parents[1] / "data/raw"
    out.mkdir(parents=True, exist_ok=True)
    for f in FILES:
        dest = out / f
        if dest.exists():
            print(f"have {f}")
            continue
        print(f"fetching {f} ...")
        urllib.request.urlretrieve(BASE + f, dest)
        print(f"  -> {dest} ({dest.stat().st_size/1024:.0f} KB)")
    print("done")

if __name__ == "__main__":
    sys.exit(main())
