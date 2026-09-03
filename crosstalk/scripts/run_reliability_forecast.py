#!/usr/bin/env python3
"""Can a model's unreliability on an input be forecast before any labels exist?

A rapid-response pipeline scores a sequence it has never seen and must act
without a wet-lab check. The operationally useful quantity there is not accuracy,
which cannot be computed without labels, but a warning: an advance estimate of
whether this model is about to be reliable on this input.

Shibboleth Arm A established that bulk correlation predicts tail performance
(rho +0.683 to TPR at a 1% false-positive budget). That predictor needs labels,
so it cannot run at decision time. Everything measured here is computable from
the model and the sequence alone, with the DMS scores held back and used only to
grade the forecast afterwards.

Candidate signals, all label-free:

  wt_agreement    fraction of positions where the model's most likely residue is
                  the one nature chose. A protein the model finds unsurprising is
                  plausibly one it models well.
  mean_wt_logprob average log-probability the model assigns the wild-type residue,
                  which is pseudo-perplexity without the exponential.
  mean_entropy    mean uncertainty of the masked distributions. High entropy means
                  the model has no strong opinion anywhere.
  frac_confident  fraction of positions where the top residue exceeds probability
                  0.5, separating a few sharp positions from uniform vagueness.
  score_dispersion spread of the variant scores themselves. A proxy that assigns
                  nearly the same score to every variant cannot rank them, and
                  this is visible without knowing the ranking.
  length, n_variants  properties of the assay, not the model, included so any
                  apparent skill signal can be checked against triviality.

The honest risk at this sample size is overfitting, so univariate correlations
are reported alongside a leave-one-assay-out forecast and a permutation null.
"""
import argparse, csv, json, sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))

from crosstalk.glm import _device
from run_proxy_ladder import spearman
from run_readout_probe import ridge_fit, ridge_pred
from run_dms_transfer import GAUNTLET, load_singles

FEATURES = ["wt_agreement", "mean_wt_logprob", "mean_entropy", "frac_confident",
            "score_dispersion", "length", "n_variants"]


@torch.no_grad()
def profile_assay(model, tok, dev, seq, positions):
    """Masked-marginal table plus every label-free summary of it."""
    lp, ent, top, wtlp, agree = {}, [], [], [], []
    for p in positions:
        masked = seq[:p - 1] + tok.mask_token + seq[p:]
        enc = tok(masked, return_tensors="pt").to(dev)
        logits = model(**enc).logits[0]
        at = (enc["input_ids"][0] == tok.mask_token_id).nonzero()[0, 0]
        v = torch.log_softmax(logits[at].float(), -1)
        lp[p] = v.cpu().numpy()
        # restrict to the 20 standard residues so special tokens cannot dominate
        ids = [tok.convert_tokens_to_ids(a) for a in "ACDEFGHIKLMNPQRSTVWY"]
        sub = torch.log_softmax(v[ids], -1)
        pr = sub.exp()
        ent.append(float(-(pr * sub).sum()))
        top.append(float(pr.max()))
        wtlp.append(float(sub["ACDEFGHIKLMNPQRSTVWY".index(seq[p - 1])]))
        agree.append("ACDEFGHIKLMNPQRSTVWY"[int(pr.argmax())] == seq[p - 1])
    return lp, dict(wt_agreement=float(np.mean(agree)),
                    mean_wt_logprob=float(np.mean(wtlp)),
                    mean_entropy=float(np.mean(ent)),
                    frac_confident=float(np.mean(np.array(top) > 0.5)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="facebook/esm2_t33_650M_UR50D")
    ap.add_argument("--max-len", type=int, default=700)
    ap.add_argument("--min-singles", type=int, default=200)
    ap.add_argument("--out", default="results/reliability_forecast.csv")
    args = ap.parse_args()

    ref = {r["DMS_id"]: r for r in csv.DictReader((GAUNTLET / "reference.csv").open())}
    have = sorted(p.stem for p in (GAUNTLET / "assays").glob("*.csv"))
    from transformers import AutoModelForMaskedLM, AutoTokenizer
    dev = _device()
    tok = AutoTokenizer.from_pretrained(args.model)
    mdl = AutoModelForMaskedLM.from_pretrained(args.model).to(dev).eval()
    print(f"{args.model} on {dev}; {len(have)} assays available\n", flush=True)

    rows = []
    print(f"{'assay':40s} {'n':>5s} {'rho':>7s} {'agree':>6s} {'wtlp':>7s} "
          f"{'ent':>6s} {'disp':>6s}")
    for dms in have:
        r = ref.get(dms)
        if r is None:
            continue
        tgt = r["target_seq"]
        if len(tgt) > args.max_len:
            continue
        muts = load_singles(dms, tgt)
        if len(muts) < args.min_singles:
            continue
        positions = sorted({p for p, _, _, _ in muts})
        lp, feats = profile_assay(mdl, tok, dev, tgt, positions)

        s = np.array([float(lp[p][tok.convert_tokens_to_ids(mu)]
                            - lp[p][tok.convert_tokens_to_ids(wt)])
                      for p, wt, mu, _ in muts])
        y = np.array([m[3] for m in muts])
        feats.update(score_dispersion=float(s.std(ddof=1)), length=float(len(tgt)),
                     n_variants=float(len(muts)))
        rho = spearman(s, y)          # the label-requiring outcome being forecast
        rows.append(dict(dms_id=dms, organism=r.get("source_organism", ""),
                         rho_esm=rho, **feats))
        print(f"{dms[:40]:40s} {len(muts):5d} {rho:+7.3f} {feats['wt_agreement']:6.3f} "
              f"{feats['mean_wt_logprob']:+7.3f} {feats['mean_entropy']:6.3f} "
              f"{feats['score_dispersion']:6.3f}", flush=True)

    if len(rows) < 8:
        print("too few assays to analyse"); return

    Y = np.array([r["rho_esm"] for r in rows])
    X = np.array([[r[f] for f in FEATURES] for r in rows])
    print(f"\n{len(rows)} assays; observed skill ranges {Y.min():+.3f} to {Y.max():+.3f}")

    print("\n--- univariate: each label-free signal against observed skill ---")
    for j, f in enumerate(FEATURES):
        rr = spearman(X[:, j], Y)
        rng = np.random.default_rng(0)
        null = np.array([spearman(X[:, j], rng.permutation(Y)) for _ in range(2000)])
        p = float((np.abs(null) >= abs(rr)).mean())
        print(f"  {f:18s} rho {rr:+.3f}   permutation p = {p:.4f}")

    print("\n--- leave-one-assay-out forecast from all label-free signals ---")
    pred = np.zeros(len(Y))
    for k in range(len(Y)):
        tr = np.arange(len(Y)) != k
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-8
        w = ridge_fit((X[tr] - mu) / sd, Y[tr], 10.0)
        pred[k] = ridge_pred(((X[k] - mu) / sd)[None, :], w)[0]
    rr = spearman(pred, Y)
    rng = np.random.default_rng(1)
    null = []
    for _ in range(1000):
        Yp = rng.permutation(Y); pp = np.zeros(len(Yp))
        for k in range(len(Yp)):
            tr = np.arange(len(Yp)) != k
            mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-8
            w = ridge_fit((X[tr] - mu) / sd, Yp[tr], 10.0)
            pp[k] = ridge_pred(((X[k] - mu) / sd)[None, :], w)[0]
        null.append(spearman(pp, Yp))
    null = np.array(null)
    print(f"  forecast vs actual: rho {rr:+.3f}   permutation p = "
          f"{float((np.abs(null) >= abs(rr)).mean()):.4f}")
    mae = float(np.mean(np.abs(pred - Y)))
    print(f"  mean absolute error {mae:.3f} against a spread of {Y.std(ddof=1):.3f}")

    print("\n--- the operational question: can it flag the worst assays? ---")
    q = np.quantile(Y, 0.25)
    flagged = pred <= np.quantile(pred, 0.25)
    worst = Y <= q
    tp = int((flagged & worst).sum())
    print(f"  of the {int(worst.sum())} least-reliable assays, the forecast's own "
          f"bottom quartile catches {tp}")
    print(f"  base rate would catch {worst.sum() * 0.25:.1f}")

    out = ROOT / args.out
    keys = ["dms_id", "organism", "rho_esm"] + FEATURES
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader()
        w.writerows([{k: r[k] for k in keys} for r in rows])
    np.save(ROOT / "results/reliability_forecast_pred.npy", pred)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
