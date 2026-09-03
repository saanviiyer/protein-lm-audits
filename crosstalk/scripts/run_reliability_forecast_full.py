#!/usr/bin/env python3
"""Section 21 at full scale: all 217 ProteinGym v1 assays.

The 41-assay pilot found one label-free signal that survived correction (score
dispersion, rho +0.458) but could not fit a calibrated multivariate forecast:
33 protein clusters is too few. This runs the same measurement on ProteinGym v1,
where 194 assays pass the length and sample-size filters and span 172 distinct
protein prefixes.

Two changes from the pilot, both about making the run finish and be trusted:

  BATCHED   the pilot masked one position per forward pass. 40,412 positions makes
            that wasteful, so masked copies are batched, with the batch size
            scaled down for long sequences to bound memory.
  RESUMABLE per-assay rows are appended to the output as they are computed and
            completed assays are skipped on restart, so a multi-hour run survives
            an interruption without repeating work.
"""
import argparse, csv, glob, sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
from crosstalk.glm import _device
from run_proxy_ladder import spearman
from run_reliability_forecast import FEATURES

AA = "ACDEFGHIKLMNPQRSTVWY"


def load_assays(min_singles, max_len):
    import json, pandas as pd
    cache = ROOT / "data" / "proteingym_v1" / f"parsed_{min_singles}_{max_len}.json"
    if cache.exists():
        return [(d, t, [tuple(m) for m in ms]) for d, t, ms in
                json.loads(cache.read_text())]
    df = pd.concat([pd.read_parquet(f) for f in
                    sorted(glob.glob(str(ROOT / "data/proteingym_v1/*.parquet")))])
    df = df[~df["mutant"].astype(str).str.contains(":")]
    out = []
    for dms, sub in df.groupby("DMS_id"):
        tgt = sub["target_seq"].iloc[0]
        if len(tgt) > max_len:
            continue
        muts = []
        for m, s in zip(sub["mutant"].astype(str), sub["DMS_score"]):
            if len(m) < 3 or not m[1:-1].isdigit():
                continue
            p = int(m[1:-1])
            # a stated wild type that disagrees with the target sequence is a
            # numbering mismatch, not a measurement, so drop it rather than coerce
            if 1 <= p <= len(tgt) and tgt[p - 1] == m[0] and m[-1] in AA:
                muts.append((p, m[0], m[-1], float(s)))
        if len(muts) >= min_singles:
            out.append((dms, tgt, muts))
    cache.write_text(json.dumps(out))
    return out


@torch.no_grad()
def profile(model, tok, dev, seq, positions, batch_tokens=6000):
    """Batched masked-marginal table plus the label-free summaries of it."""
    ids = [tok.convert_tokens_to_ids(a) for a in AA]
    B = max(1, min(24, batch_tokens // max(len(seq), 1)))
    lp, ent, top, wtlp, agree = {}, [], [], [], []
    for i in range(0, len(positions), B):
        chunk = positions[i:i + B]
        seqs = [seq[:p - 1] + tok.mask_token + seq[p:] for p in chunk]
        enc = tok(seqs, return_tensors="pt", padding=True).to(dev)
        logits = model(**enc).logits
        for r, p in enumerate(chunk):
            # The mask sits at token index p by construction: <cls> occupies 0 and
            # residues 1..p-1 precede it. Searching for it with nonzero() was
            # failing intermittently on MPS under memory pressure, which killed a
            # run 22 assays in; the analytic index cannot fail and is checked once.
            at = p
            assert int(enc["input_ids"][r, at]) == tok.mask_token_id, \
                f"mask not at index {p} for {len(seq)}-residue sequence"
            v = torch.log_softmax(logits[r, at].float(), -1)
            lp[p] = v.cpu().numpy()
            sub = torch.log_softmax(v[ids], -1)
            pr = sub.exp()
            ent.append(float(-(pr * sub).sum()))
            top.append(float(pr.max()))
            wtlp.append(float(sub[AA.index(seq[p - 1])]))
            agree.append(AA[int(pr.argmax())] == seq[p - 1])
    return lp, dict(wt_agreement=float(np.mean(agree)),
                    mean_wt_logprob=float(np.mean(wtlp)),
                    mean_entropy=float(np.mean(ent)),
                    frac_confident=float(np.mean(np.array(top) > 0.5)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="facebook/esm2_t33_650M_UR50D")
    ap.add_argument("--max-len", type=int, default=1022)
    ap.add_argument("--min-singles", type=int, default=200)
    ap.add_argument("--batch-tokens", type=int, default=6000,
                    help="tokens per forward pass; lower it when memory is tight, "
                         "since MPS degrades badly rather than failing cleanly")
    ap.add_argument("--out", default="results/reliability_forecast_full.csv")
    args = ap.parse_args()

    out = ROOT / args.out
    keys = ["dms_id", "target_seq_len", "rho_esm"] + FEATURES
    done = set()
    if out.exists():
        done = {r["dms_id"] for r in csv.DictReader(out.open())}
        print(f"resuming: {len(done)} assays already scored")
    else:
        with out.open("w", newline="") as f:
            csv.DictWriter(f, fieldnames=keys).writeheader()

    assays = load_assays(args.min_singles, args.max_len)
    todo = [a for a in assays if a[0] not in done]
    print(f"{len(assays)} assays pass filters; {len(todo)} to score "
          f"({sum(len({m[0] for m in a[2]}) for a in todo):,} masked positions)\n",
          flush=True)

    from transformers import AutoModelForMaskedLM, AutoTokenizer
    dev = _device()
    tok = AutoTokenizer.from_pretrained(args.model)
    mdl = AutoModelForMaskedLM.from_pretrained(args.model).to(dev).eval()

    for n, (dms, tgt, muts) in enumerate(todo, 1):
        positions = sorted({m[0] for m in muts})
        lp, feats = profile(mdl, tok, dev, tgt, positions, args.batch_tokens)
        s = np.array([float(lp[p][tok.convert_tokens_to_ids(mu)]
                            - lp[p][tok.convert_tokens_to_ids(wt)])
                      for p, wt, mu, _ in muts])
        y = np.array([m[3] for m in muts])
        feats.update(score_dispersion=float(s.std(ddof=1)),
                     length=float(len(tgt)), n_variants=float(len(muts)))
        row = dict(dms_id=dms, target_seq_len=len(tgt), rho_esm=spearman(s, y), **feats)
        with out.open("a", newline="") as f:
            csv.DictWriter(f, fieldnames=keys).writerow(row)
        print(f"[{n:3d}/{len(todo)}] {dms[:44]:44s} n={len(muts):6d} "
              f"rho {row['rho_esm']:+.3f} disp {feats['score_dispersion']:5.2f} "
              f"agree {feats['wt_agreement']:.3f}", flush=True)

    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
