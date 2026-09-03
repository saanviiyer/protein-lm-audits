#!/usr/bin/env python3
"""The operational question: is the steered entropy a BETTER abstention signal?

Changing entropy is easy. The claim that would matter for a pipeline with no
wet-lab check is that steering makes the model's uncertainty more diagnostic of
where it is actually wrong. So:

  * dump per-position entropy under baseline and under each steered condition,
  * rank positions by each signal and keep the most confident fraction,
  * score ESM-2's (UNSTEERED) masked-marginal predictions on the kept variants.

Scoring the unsteered predictions is deliberate: the question is whether the
steered entropy is a better SELECTOR, not whether steering improves the model.
"""
import csv, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
from crosstalk.glm import _device
from run_proxy_ladder import spearman
from run_east_steering import build_panel, run_assay, make_directions

AA = "ACDEFGHIKLMNPQRSTVWY"
CONFIGS = [("base", None, 0.0), ("steer", "steer", 1.0), ("steer", "steer", -1.0),
           ("random", "random", 1.0), ("steerperp", "steerperp", 1.0),
           ("mean", "mean", 1.0)]
LAYER = 29


def main():
    panel = build_panel(400, 40, 0)
    rng = np.random.default_rng(7)
    order = rng.permutation(len(panel))
    ev = [panel[i] for i in order[16:36]]

    from transformers import AutoModelForMaskedLM, AutoTokenizer
    dev = _device()
    tok = AutoTokenizer.from_pretrained("facebook/esm2_t30_150M_UR50D")
    mdl = AutoModelForMaskedLM.from_pretrained(
        "facebook/esm2_t30_150M_UR50D").to(dev).eval()

    z = np.load(ROOT / "results/east_steering_vectors_150M.npz", allow_pickle=True)
    dirs = make_directions(z["steer"], z["mean"], 13)
    act = z["act_norm"]

    out = ROOT / "results/east_steering_perpos_150M.csv"
    keys = ["dms_id", "condition", "alpha", "position", "entropy", "agree",
            "wt_logprob", "pos_rho", "n_pos_var"]
    w = csv.DictWriter(out.open("w", newline=""), fieldnames=keys)
    w.writeheader()

    curves = {}
    for it in ev:
        by = {}
        for name, d, a in CONFIGS:
            vec = None if d is None else dirs[LAYER][d] * (a * act[LAYER])
            pp, _ = run_assay(mdl, tok, dev, it, LAYER, vec, 6000)
            by[(name, a)] = pp
        base = by[("base", 0.0)]
        # per-variant baseline scores, grouped by position
        byp = {}
        for p, wt, mu, val in it["muts"]:
            byp.setdefault(p, []).append(
                (base[p]["logp"][AA.index(mu)] - base[p]["logp"][AA.index(wt)], val))
        prho = {p: (spearman(np.array([x[0] for x in v]), np.array([x[1] for x in v]))
                    if len(v) >= 5 else np.nan) for p, v in byp.items()}
        for (name, a), pp in by.items():
            for p in it["positions"]:
                w.writerow(dict(dms_id=it["dms_id"], condition=name, alpha=a,
                                position=p, entropy=pp[p]["ent"],
                                agree=int(pp[p]["agree"]),
                                wt_logprob=pp[p]["wt_lp"],
                                pos_rho=prho.get(p, np.nan),
                                n_pos_var=len(byp.get(p, []))))
        curves[it["dms_id"]] = (by, byp)
        print(f"{it['dms_id'][:40]:40s} done", flush=True)

    # selective curves: rank positions by each signal, keep the most confident
    fracs = [0.25, 0.5, 0.75, 1.0]
    print("\n=== selective prediction: rank positions by a signal, keep the "
          "most confident, score the UNSTEERED model on what is kept ===")
    print(f"{'signal':>22} " + " ".join(f"keep{int(f*100):>3d}%" for f in fracs))
    sigs = [("baseline entropy", ("base", 0.0)),
            ("steered +1 entropy", ("steer", 1.0)),
            ("steered -1 entropy", ("steer", -1.0)),
            ("random +1 entropy", ("random", 1.0)),
            ("steerPerp +1 entropy", ("steerperp", 1.0)),
            ("mean +1 entropy", ("mean", 1.0)),
            ("delta entropy (steer+1 - base)", None)]
    for label, key in sigs:
        row = []
        for f in fracs:
            rr = []
            for dms, (by, byp) in curves.items():
                pos = sorted(byp)
                if key is None:
                    sig = np.array([by[("steer", 1.0)][p]["ent"] - by[("base", 0.0)][p]["ent"]
                                    for p in pos])
                else:
                    sig = np.array([by[key][p]["ent"] for p in pos])
                k = max(3, int(round(f * len(pos))))
                keep = set(np.array(pos)[np.argsort(sig)[:k]].tolist())
                s = np.array([x[0] for p in keep for x in byp[p]])
                y = np.array([x[1] for p in keep for x in byp[p]])
                rr.append(spearman(s, y))
            row.append(np.mean(rr))
        print(f"{label:>22} " + " ".join(f"{v:8.3f}" for v in row))


if __name__ == "__main__":
    main()
