#!/usr/bin/env python3
"""How common are the readiness failures? A survey, not an anecdote.

Phases 1-11 established each failure mode on a handful of datasets. For a data
readiness claim that is not enough: the question is whether these are quirks of
the PETase corpus or the normal state of protein-engineering data. This measures
every failure mode across every dataset in the repo.

Per dataset it reports:
  mixed_k        does the pool mix mutation counts, so the confound can operate?
  rho_nmut       does mutation count alone predict fitness? (campaign progression)
  rho_pooled     a zero-shot proxy's apparent correlation, pooled
  rho_within     the same, computed within a fixed mutation count
  sign_flip      do those two disagree in sign?
  strata         how many distinct assay conditions are pooled together?

    python scripts/run_readiness_survey.py
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from gauntlet import proteingym as pg  # noqa: E402
from gauntlet.petase_data import load_corpus  # noqa: E402

PG_ASSAYS = ["B3VI55_LIPST_Klesmith_2015", "AMIE_PSEAE_Wrenbeck_2017",
             "A4GRB6_PSEAI_Chen_2020", "BLAT_ECOLX_Firnberg_2014",
             "KKA2_KLEPN_Melnikov_2014", "TPMT_HUMAN_Matreyek_2018",
             "TPK1_HUMAN_Weile_2017", "DYR_ECOLI_Thompson_plusLon_2019",
             "ESTA_BACSU_Nutschel_2020", "UBC9_HUMAN_Weile_2017",
             "RASH_HUMAN_Bandaru_2017", "NUD15_HUMAN_Suiter_2020",
             "CALM1_HUMAN_Weile_2017"]


def rho(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 5 or np.std(a[ok]) < 1e-12 or np.std(b[ok]) < 1e-12:
        return np.nan
    return float(stats.spearmanr(a[ok], b[ok]).statistic)


def within_k(scores, fitness, nmut, min_per=5):
    """Mean correlation inside strata of fixed mutation count."""
    vals = []
    for k in np.unique(nmut):
        m = nmut == k
        if m.sum() >= min_per:
            r = rho(scores[m], fitness[m])
            if np.isfinite(r):
                vals.append(r)
    return float(np.mean(vals)) if vals else np.nan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--petml_dir", default="data/petml")
    ap.add_argument("--pg_dir", default="data/proteingym")
    ap.add_argument("--nrel", default="data/nrel/nrel_campaign.csv")
    ap.add_argument("--scored", default="results/scored_variants.csv")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    rows = []

    # ---- PETML: 33 published PET-hydrolase campaigns ------------------------
    df = load_corpus(args.petml_dir)
    esm = pd.read_csv(args.scored)[["study", "protein", "esm2_wtm"]]
    df = df.merge(esm, on=["study", "protein"], how="left")
    for study, grp in df.groupby("study"):
        g = grp[grp.logActivity.notna()]
        if g.empty:
            continue
        g = g[[isinstance(m, list) and len(m) > 0 for m in g.muts]]
        if len(g) < 10:
            continue
        nm = np.array([len(m) for m in g.muts], float)
        y = g.logActivity.to_numpy(float)
        s = np.nan_to_num(g.esm2_wtm.to_numpy(float))
        rp, rw = rho(s, y), within_k(s, y, nm)
        rows.append({
            "source": "PETML", "dataset": study, "n": len(g),
            "mixed_k": bool(len(np.unique(nm)) > 1),
            "rho_nmut": rho(nm, y), "rho_pooled": rp, "rho_within": rw,
            "sign_flip": bool(np.isfinite(rp) and np.isfinite(rw)
                              and np.sign(rp) != np.sign(rw)),
            "strata": 1,
        })

    # ---- ProteinGym: single-mutant scans, the confound cannot operate -------
    ref = pg.select_assays(os.path.join(args.pg_dir, "reference.csv"), ids=PG_ASSAYS)
    for r in ref.itertuples():
        cache = os.path.join(args.pg_dir, "esm_cache", f"{r.DMS_id}.npy")
        if not os.path.exists(cache):
            continue
        records, y, _ = pg.load_assay(r.DMS_id, r.target_seq,
                                      os.path.join(args.pg_dir, "assays"))
        s = np.load(cache)
        nm = np.ones(len(y))
        rp = rho(s, y)
        rows.append({
            "source": "ProteinGym", "dataset": r.DMS_id, "n": len(y),
            "mixed_k": False, "rho_nmut": np.nan, "rho_pooled": rp,
            "rho_within": rp, "sign_flip": False, "strata": 1,
        })

    # ---- NREL: real assay conditions ---------------------------------------
    if os.path.exists(args.nrel):
        nd = pd.read_csv(args.nrel)
        key = nd[["pH", "temperature_C", "substrate"]].astype(str).agg("|".join, axis=1)
        rows.append({
            "source": "NREL", "dataset": "condition-resolved release",
            "n": len(nd), "mixed_k": False, "rho_nmut": np.nan,
            "rho_pooled": np.nan, "rho_within": np.nan, "sign_flip": False,
            "strata": int(key.nunique()),
        })

    t = pd.DataFrame(rows)
    os.makedirs(args.out, exist_ok=True)
    t.to_csv(os.path.join(args.out, "readiness_survey.csv"), index=False)

    print(f"\n{'='*74}\nDATA READINESS SURVEY — {len(t)} datasets\n{'='*74}")
    print(t.round(3).to_string(index=False))

    pet = t[t.source == "PETML"]
    print(f"\n--- Multi-mutant campaigns (PETML, n={len(pet)}) ---")
    mk = pet[pet.mixed_k]
    print(f"mix mutation counts, so the confound can operate: {len(mk)}/{len(pet)} "
          f"({100*len(mk)/len(pet):.0f}%)")
    strong = pet[pet.rho_nmut.abs() >= 0.5]
    print(f"mutation count alone predicts fitness at |rho|>=0.5: {len(strong)}/{len(pet)} "
          f"({100*len(strong)/len(pet):.0f}%)  median |rho| = "
          f"{pet.rho_nmut.abs().median():.2f}")
    flips = pet[pet.sign_flip]
    comparable = pet[pet.rho_pooled.notna() & pet.rho_within.notna()]
    if len(comparable):
        print(f"pooled and within-stratum correlations DISAGREE IN SIGN: "
              f"{len(flips)}/{len(comparable)} ({100*len(flips)/len(comparable):.0f}%)")
        print(f"mean pooled rho {comparable.rho_pooled.mean():+.3f}  ->  "
              f"within-stratum {comparable.rho_within.mean():+.3f}")

    pgs = t[t.source == "ProteinGym"]
    print(f"\n--- Single-mutant scans (ProteinGym, n={len(pgs)}) ---")
    print(f"mean proxy rho {pgs.rho_pooled.mean():+.3f}; the confound cannot operate "
          "because\nmutation count is constant by construction.")
    print(f"\nThe contrast is the point: the same proxy reads "
          f"{comparable.rho_pooled.mean():+.3f} on pooled campaign\ndata and "
          f"{pgs.rho_pooled.mean():+.3f} on scans where the data cannot confound it.")

    print(f"\nwrote {args.out}/readiness_survey.csv")


if __name__ == "__main__":
    main()
