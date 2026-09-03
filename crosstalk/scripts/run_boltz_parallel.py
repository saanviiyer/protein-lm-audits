"""Run the Boltz specificity audit as parallel co-folds.

One fold takes ~40 min on this hardware but averages only ~0.7 cores, so the
audit is throughput-bound rather than compute-bound and parallelises well.
Each fold is an independent process writing into its own directory; results are
collected as they land, so the run is resumable and a crash costs one fold.
"""
import argparse, csv, json, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from crosstalk.boltz import PARTNERS, complex_yaml, parse_confidence
from crosstalk.landscape import load_pard3

BOLTZ = Path(__file__).resolve().parents[1] / ".venv-boltz/bin/boltz"


def completed_variants(workdir):
    """Variants with at least one finished fold already on disk."""
    out = set()
    for case in Path(workdir).glob("*_*"):
        name = case.name
        if list((case / f"boltz_results_{name}" / "predictions" / name).glob("confidence_*.json")):
            out.add(name.rsplit("_", 1)[0])
    return out


def stratify(L, n_per_cell, rng, existing=()):
    """Sample n_per_cell per specificity quadrant, keeping any already-folded
    variant that belongs to the cell and topping the rest up at random.

    Drawing k uniformly and later topping up to n from the remainder yields the
    same distribution as drawing n at once, so resuming does not bias the sample.
    """
    on, off = L.F[:, 0], L.F[:, 1]
    cells = {"specific_E3": (on >= 0.8) & (off <= 0.2),
             "promiscuous": (on >= 0.8) & (off >= 0.6),
             "specific_E2": (on <= 0.2) & (off >= 0.8),
             "dead": (on <= 0.2) & (off <= 0.2)}
    existing = set(existing)
    picked = []
    for name, mask in cells.items():
        idx = np.where(mask)[0]
        seqs = [L.seqs[i] for i in idx]
        keep = [s for s in seqs if s in existing]
        pool = np.array([i for i, s in zip(idx, seqs) if s not in existing], dtype=int)
        need = min(n_per_cell, len(idx)) - len(keep)
        take = (rng.choice(pool, size=min(need, len(pool)), replace=False)
                if need > 0 and len(pool) else np.array([], dtype=int))
        picked += [(s, name) for s in keep] + [(L.seqs[i], name) for i in take]
        print(f"  {name:12s}: {len(keep)} kept + {len(take)} new "
              f"= {len(keep) + len(take)} of {len(idx)}")
    return picked


def one_fold(variant, partner, workdir, timeout, full_quality=False, accelerator="cpu"):
    name = f"{variant}_{partner}"
    case = Path(workdir) / name
    done = case / f"boltz_results_{name}" / "predictions" / name
    if list(done.glob("confidence_*.json")):
        return variant, partner, parse_confidence(case), "cached"
    case.mkdir(parents=True, exist_ok=True)
    y = case / f"{name}.yaml"
    y.write_text(complex_yaml(variant, partner))
    cmd = [str(BOLTZ), "predict", str(y), "--out_dir", str(case),
           "--accelerator", accelerator, "--devices", "1", "--use_msa_server",
           "--num_workers", "0", "--output_format", "pdb"]
    if not full_quality:
        # The reduced settings exist because the full defaults did not finish on
        # this project's laptop: one 196-residue complex took 39m45s on MPS even
        # at these settings, and would not complete at all without them
        # (FINDINGS section 5). They are a hardware concession, not a modelling
        # choice, and every number produced under them is a lower bound on what
        # Boltz can do. On a GPU, prefer --full-quality.
        cmd += ["--recycling_steps", "0", "--diffusion_samples", "1",
                "--sampling_steps", "25"]
    t = time.time()
    r = subprocess.run(cmd, capture_output=True, timeout=timeout)
    if not list(done.glob("confidence_*.json")):
        return variant, partner, None, f"failed rc={r.returncode}"
    return variant, partner, parse_confidence(case), f"{time.time()-t:.0f}s"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-per-cell", type=int, default=6)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--workdir", default="results/boltz")
    ap.add_argument("--out", default="results/boltz_proxy.csv")
    ap.add_argument("--timeout", type=int, default=7200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--full-quality", action="store_true",
                    help="drop the reduced recycling/diffusion/sampling settings and "
                         "use Boltz defaults. The stored results were NOT produced this "
                         "way, so a full-quality run is a different arm and must be "
                         "written to a different --out rather than merged.")
    ap.add_argument("--accelerator", default="cpu", choices=["cpu", "gpu"],
                    help="cpu is the laptop default; use gpu on a CUDA host")
    ap.add_argument("--resume", action="store_true",
                    help="keep variants already folded in --workdir and top up")
    args = ap.parse_args()

    L = load_pard3()
    rng = np.random.default_rng(args.seed)
    existing = completed_variants(args.workdir) if args.resume else set()
    picked = stratify(L, args.n_per_cell, rng, existing)
    jobs = [(v, p) for v, _ in picked for p in PARTNERS]
    quality = "FULL QUALITY (Boltz defaults)" if args.full_quality else \
              "reduced (recycling 0, diffusion 1, sampling 25)"
    print(f"\n{len(picked)} variants x {len(PARTNERS)} partners = {len(jobs)} folds, "
          f"{args.workers} at a time")
    print(f"accelerator={args.accelerator}  settings={quality}")
    if args.full_quality:
        print("NOTE: the stored results in results/boltz_proxy.csv were produced at the "
              "reduced settings.\n      Write this run to a different --out; do not merge "
              "the two.\n", flush=True)
    else:
        print("", flush=True)

    got = {}
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(one_fold, v, p, args.workdir, args.timeout,
                          args.full_quality, args.accelerator): (v, p)
                for v, p in jobs}
        for k, fut in enumerate(as_completed(futs), 1):
            v, p = futs[fut]
            try:
                v, p, conf, note = fut.result()
            except Exception as e:
                conf, note = None, f"error {type(e).__name__}"
            got[(v, p)] = conf
            ip = conf.get("iptm") if conf else None
            print(f"[{k}/{len(jobs)}] {v} {p}: iptm={ip if ip is None else round(ip,4)} ({note})",
                  flush=True)
            _write(L, picked, got, args.out)
    print(f"\nwrote {args.out}")


def _write(L, picked, got, out):
    """Merge this run's results into the CSV and write it atomically.

    Rows already on disk are preserved: an interrupted run must never truncate
    the table to the folds it happened to have collected. The folds themselves
    are durable, but the CSV is the artifact the analysis reads.
    """
    out = Path(out)
    prior = {}
    if out.exists():
        with open(out) as fh:
            prior = {r["variant"]: r for r in csv.DictReader(fh)}

    rows = dict(prior)
    for variant, cell in picked:
        rec = {"variant": variant, "cell": cell,
               "W_ParE3": float(L.F[L.index[variant], 0]),
               "W_ParE2": float(L.F[L.index[variant], 1])}
        for partner in PARTNERS:
            c = got.get((variant, partner))
            rec[f"iptm_{partner}"] = c.get("iptm") if c else ""
            rec[f"ptm_{partner}"] = c.get("ptm") if c else ""
            rec[f"plddt_{partner}"] = c.get("complex_plddt") if c else ""
        if not any(rec[f"iptm_{p}"] != "" for p in PARTNERS):
            continue  # nothing folded yet; keep whatever disk already had
        old = rows.get(variant)
        if old:  # never overwrite a finished fold with a blank
            for k, v in rec.items():
                if v == "" and old.get(k) not in (None, ""):
                    rec[k] = old[k]
        rows[variant] = rec

    if not rows:
        return
    ordered = [rows[v] for v in sorted(rows)]
    fields = ["variant", "cell", "W_ParE3", "W_ParE2"] + [
        f"{m}_{p}" for p in PARTNERS for m in ("iptm", "ptm", "plddt")]
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".csv.tmp")
    with open(tmp, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader(); w.writerows(ordered)
    tmp.replace(out)


if __name__ == "__main__":
    main()
