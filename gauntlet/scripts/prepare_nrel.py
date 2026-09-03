#!/usr/bin/env python3
"""Convert the NREL condition-resolved PET-hydrolase release into a campaign file.

Source: Zenodo 10.5281/zenodo.15417757 (CC BY 4.0), "Activity across temperature
and pH of PET hydrolase candidates" -- the data behind Norton-Baker, Komp, Gado
et al., ACS Catalysis (2025), doi:10.1021/acscatal.5c03460.

The release is wide: one row per protein, one column per assay condition named
``activity_at_<pH>_<temperature>_<substrate>``. This unpivots it to one row per
(protein, condition), which is the shape a campaign file takes, and splits the
condition out into pH / temperature / substrate columns so the tool can stratify
on them.

Note this is a MINING campaign -- 213 distinct natural proteins, not variants of
a scaffold -- so the file carries a ``sequence`` column rather than ``variant``.

    python scripts/prepare_nrel.py
"""

import argparse
import os
import re

import pandas as pd

COND_RE = re.compile(r"^activity_at_([\d.]+)_(\d+)_(\w+)$")
SUBSTRATE = {"cryPow": "crystalline powder", "aFilm": "amorphous film"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/nrel/p740/label_data.csv")
    ap.add_argument("--out", default="data/nrel/nrel_campaign.csv")
    args = ap.parse_args()

    d = pd.read_csv(args.src).rename(columns={"Unnamed: 0": "id"})
    cond_cols = [c for c in d.columns if COND_RE.match(c)]

    rows = []
    for c in cond_cols:
        pH, temp, sub = COND_RE.match(c).groups()
        block = d[["id", "sequence", "round", c]].dropna(subset=[c])
        for r in block.itertuples():
            rows.append({
                "id": r.id,
                "sequence": r.sequence,
                "fitness": float(getattr(r, c) if hasattr(r, c) else r[-1]),
                "round": r.round,
                "pH": float(pH),
                "temperature_C": int(temp),
                "substrate": SUBSTRATE.get(sub, sub),
            })

    long = pd.DataFrame(rows)
    # Guard against the itertuples attribute-name mangling above.
    melted = d.melt(id_vars=["id", "sequence", "round"], value_vars=cond_cols,
                    var_name="cond", value_name="fitness").dropna(subset=["fitness"])
    parts = melted.cond.str.extract(COND_RE)
    melted["pH"] = parts[0].astype(float)
    melted["temperature_C"] = parts[1].astype(int)
    melted["substrate"] = parts[2].map(SUBSTRATE).fillna(parts[2])
    long = melted.drop(columns=["cond"])[
        ["id", "sequence", "fitness", "round", "pH", "temperature_C", "substrate"]]

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    long.to_csv(args.out, index=False)

    print(f"{len(d)} proteins x {len(cond_cols)} conditions -> {len(long)} measurements")
    print(f"unique proteins: {long.id.nunique()}   nonzero activity: "
          f"{int((long.fitness > 0).sum())} ({(long.fitness > 0).mean():.1%})")
    print("\nmeasurements per condition:")
    g = long.groupby(["pH", "temperature_C", "substrate"]).agg(
        n=("fitness", "size"), active=("fitness", lambda s: int((s > 0).sum())),
        max=("fitness", "max"))
    print(g.to_string())
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
