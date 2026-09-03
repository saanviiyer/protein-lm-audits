#!/usr/bin/env python3
"""EAST (Rahn, D'Oro & Bellemare 2024, arXiv:2406.00244) transplanted to ESM-2.

Their recipe, verbatim in structure:
  1. collect the hidden state at a chosen layer at the decision point,
  2. build the steering vector as the ENTROPY-WEIGHTED mean of those states
     minus the UNWEIGHTED mean (so it encodes the entropy contrast, not the
     mean activation),
  3. add alpha * vector back with a forward hook at that layer, and look for a
     monotone dose-response.

The decision point for a masked protein LM is the masked token: its hidden
state at layer L is what the remaining layers turn into an amino-acid
distribution. Entropy is the entropy of that 20-way distribution.

Nulls, because this project has been burned by effects a null also produces:
  RANDOM  a gaussian direction at the same layer, matched norm.
  ORTH    a direction orthogonal to the steering vector, matched norm.
  MEAN    the unweighted mean activation direction, matched norm (tests whether
          any "in-distribution" direction does it).
"""
import argparse, csv, glob, json, sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
from crosstalk.glm import _device
from run_proxy_ladder import spearman
from run_reliability_forecast_full import load_assays

AA = "ACDEFGHIKLMNPQRSTVWY"


# ------------------------------------------------------------------ panel

def build_panel(max_len, n_pos, seed, min_singles=200):
    """One assay per UniProt protein, short enough to sweep, positions subsampled."""
    import pandas as pd
    ref = pd.read_csv(ROOT / "data/proteingym_reference_v1.csv")
    uni = dict(zip(ref["DMS_id"], ref["UniProt_ID"]))
    tax = dict(zip(ref["DMS_id"], ref["taxon"]))
    skill = {}
    for f, tag in [("results/reliability_full_150M.csv", "150M"),
                   ("results/reliability_forecast_full.csv", "650M")]:
        d = pd.read_csv(ROOT / f)
        skill[tag] = dict(zip(d["dms_id"], d["rho_esm"]))
        skill[tag + "_ent"] = dict(zip(d["dms_id"], d["mean_entropy"]))

    assays = [a for a in load_assays(min_singles, 1022) if len(a[1]) <= max_len]
    # one assay per protein: the one with most single mutants
    best = {}
    for dms, tgt, muts in assays:
        u = uni.get(dms, dms)
        if u not in best or len(muts) > len(best[u][2]):
            best[u] = (dms, tgt, muts)

    rng = np.random.default_rng(seed)
    panel = []
    for u, (dms, tgt, muts) in sorted(best.items()):
        pos = sorted({m[0] for m in muts})
        if len(pos) > n_pos:
            pos = sorted(rng.choice(pos, n_pos, replace=False).tolist())
        keep = [m for m in muts if m[0] in set(pos)]
        if len(keep) < 60:
            continue
        panel.append(dict(dms_id=dms, uniprot=u, taxon=tax.get(dms, "?"), seq=tgt,
                          positions=pos, muts=keep,
                          skill150=skill["150M"].get(dms, np.nan),
                          skill650=skill["650M"].get(dms, np.nan),
                          ent150=skill["150M_ent"].get(dms, np.nan)))
    return panel


# ------------------------------------------------------------------ hooking

class Injector:
    """Add `vec` to the residual stream at layer L, at the masked positions only.

    L = 0 hooks the embedding output; L >= 1 hooks the output of encoder block
    L-1, i.e. what `output_hidden_states` calls hidden_states[L].
    """

    def __init__(self, model, layer):
        self.mod = (model.esm.embeddings if layer == 0
                    else model.esm.encoder.layer[layer - 1])
        self.vec = None          # (d,) tensor
        self.mask = None         # (B, T) bool
        self.h = None

    def __enter__(self):
        def fn(_m, _i, out):
            if self.vec is None:
                return out
            tup = isinstance(out, tuple)
            hs = out[0] if tup else out
            hs = hs + self.mask.unsqueeze(-1).to(hs.dtype) * self.vec.to(hs.dtype)
            return (hs,) + out[1:] if tup else hs
        self.h = self.mod.register_forward_hook(fn)
        return self

    def __exit__(self, *a):
        self.h.remove()


@torch.no_grad()
def run_assay(model, tok, dev, item, layer, vec, batch_tokens, collect_states=False):
    """Masked-marginal pass over the assay's positions, optionally steered."""
    seq, positions = item["seq"], item["positions"]
    ids = [tok.convert_tokens_to_ids(a) for a in AA]
    B = max(1, min(24, batch_tokens // max(len(seq), 1)))
    per_pos, states = {}, {}
    inj = Injector(model, layer)
    with inj:
        if vec is not None:
            inj.vec = torch.as_tensor(vec, device=dev, dtype=torch.float32)
        for i in range(0, len(positions), B):
            chunk = positions[i:i + B]
            seqs = [seq[:p - 1] + tok.mask_token + seq[p:] for p in chunk]
            enc = tok(seqs, return_tensors="pt", padding=True).to(dev)
            m = torch.zeros_like(enc["input_ids"], dtype=torch.bool)
            for r, p in enumerate(chunk):
                assert int(enc["input_ids"][r, p]) == tok.mask_token_id
                m[r, p] = True
            inj.mask = m
            out = model(**enc, output_hidden_states=collect_states)
            for r, p in enumerate(chunk):
                v = torch.log_softmax(out.logits[r, p].float(), -1)[ids]
                v = torch.log_softmax(v, -1)
                pr = v.exp()
                per_pos[p] = dict(logp=v.cpu().numpy(),
                                  ent=float(-(pr * v).sum()),
                                  top=float(pr.max()),
                                  wt_lp=float(v[AA.index(seq[p - 1])]),
                                  agree=bool(AA[int(pr.argmax())] == seq[p - 1]))
                if collect_states:
                    states[p] = np.stack([h[r, p].float().cpu().numpy()
                                          for h in out.hidden_states])
    return per_pos, states


def summarise(item, per_pos):
    ent = np.array([per_pos[p]["ent"] for p in item["positions"]])
    agree = np.array([per_pos[p]["agree"] for p in item["positions"]], float)
    wtlp = np.array([per_pos[p]["wt_lp"] for p in item["positions"]])
    s, y = [], []
    for p, wt, mu, val in item["muts"]:
        lp = per_pos[p]["logp"]
        s.append(lp[AA.index(mu)] - lp[AA.index(wt)])
        y.append(val)
    s = np.array(s)
    return dict(mean_entropy=float(ent.mean()), wt_agreement=float(agree.mean()),
                mean_wt_logprob=float(wtlp.mean()),
                frac_confident=float(np.mean([per_pos[p]["top"] > 0.5
                                              for p in item["positions"]])),
                score_dispersion=float(s.std(ddof=1)),
                rho=spearman(s, np.array(y)), n_var=len(s))


# ------------------------------------------------------------------ vectors

def make_directions(V_steer, V_mean, seed):
    """steer / random / orth / mean, each unit-normalised, per layer."""
    rng = np.random.default_rng(seed)
    out = {}
    for L in range(V_steer.shape[0]):
        v = V_steer[L]
        vn = v / (np.linalg.norm(v) + 1e-12)
        g = rng.normal(size=v.shape)
        g /= np.linalg.norm(g)
        o = g - (g @ vn) * vn
        o /= np.linalg.norm(o)
        m = V_mean[L] / (np.linalg.norm(V_mean[L]) + 1e-12)
        # the entropy contrast is partly parallel to the mean activation
        # (cos = +0.56 at the last layer), so strip that component off: if the
        # effect survives, it is the entropy contrast and not "amplify the
        # average representation"
        sp = vn - (vn @ m) * m
        sp = sp / (np.linalg.norm(sp) + 1e-12)
        out[L] = dict(steer=vn, random=g, orth=o, mean=m, steerperp=sp)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="facebook/esm2_t30_150M_UR50D")
    ap.add_argument("--max-len", type=int, default=400)
    ap.add_argument("--n-pos", type=int, default=40)
    ap.add_argument("--n-train", type=int, default=16)
    ap.add_argument("--n-eval", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch-tokens", type=int, default=6000)
    ap.add_argument("--stage", default="extract",
                    choices=["extract", "alpha", "layers", "perpos"])
    ap.add_argument("--layer", type=int, default=None)
    ap.add_argument("--alphas", default="-1,-0.5,-0.25,0,0.25,0.5,1")
    ap.add_argument("--layer-list", default=None)
    ap.add_argument("--dirs", default="steer,random,orth,mean")
    ap.add_argument("--tag", default="150M")
    ap.add_argument("--out-tag", default=None)
    args = ap.parse_args()

    panel = build_panel(args.max_len, args.n_pos, args.seed)
    rng = np.random.default_rng(args.seed + 7)
    order = rng.permutation(len(panel))
    train = [panel[i] for i in order[:args.n_train]]
    ev = [panel[i] for i in order[args.n_train:args.n_train + args.n_eval]]
    print(f"panel {len(panel)} proteins -> train {len(train)}, eval {len(ev)}",
          flush=True)

    from transformers import AutoModelForMaskedLM, AutoTokenizer
    dev = _device()
    tok = AutoTokenizer.from_pretrained(args.model)
    mdl = AutoModelForMaskedLM.from_pretrained(args.model).to(dev).eval()

    vecfile = ROOT / f"results/east_steering_vectors_{args.tag}.npz"

    if args.stage == "extract":
        H, S = [], []
        for n, it in enumerate(train, 1):
            pp, st = run_assay(mdl, tok, dev, it, 0, None, args.batch_tokens,
                               collect_states=True)
            for p in it["positions"]:
                H.append(pp[p]["ent"]); S.append(st[p])
            print(f"[{n:2d}/{len(train)}] {it['dms_id'][:40]:40s} "
                  f"{len(it['positions'])} pos", flush=True)
        H = np.array(H); S = np.stack(S)              # (N,), (N, L+1, d)
        w = H / H.sum()
        V_steer = np.einsum("n,nld->ld", w, S) - S.mean(0)
        V_mean = S.mean(0)
        norms = np.linalg.norm(S, axis=2).mean(0)     # (L+1,) mean activation norm
        np.savez(vecfile, steer=V_steer, mean=V_mean, act_norm=norms,
                 entropies=H, train=[t["dms_id"] for t in train],
                 evalset=[t["dms_id"] for t in ev])
        print(f"\nsaved {vecfile}")
        for L in range(V_steer.shape[0]):
            cs = float(V_steer[L] @ V_mean[L] /
                       (np.linalg.norm(V_steer[L]) * np.linalg.norm(V_mean[L])))
            print(f"  layer {L:2d}  |v|={np.linalg.norm(V_steer[L]):8.3f}  "
                  f"|h|={norms[L]:8.3f}  ratio={np.linalg.norm(V_steer[L])/norms[L]:.4f} "
                  f" cos(v,mean)={cs:+.3f}")
        return

    z = np.load(vecfile, allow_pickle=True)
    dirs = make_directions(z["steer"], z["mean"], args.seed + 13)
    act = z["act_norm"]
    alphas = [float(x) for x in args.alphas.split(",")]
    want = args.dirs.split(",")
    layers = ([args.layer] if args.stage == "alpha"
              else [int(x) for x in args.layer_list.split(",")])

    out = ROOT / f"results/east_steering_{args.stage}_{args.out_tag or args.tag}.csv"
    keys = ["dms_id", "uniprot", "taxon", "layer", "direction", "alpha",
            "mean_entropy", "wt_agreement", "mean_wt_logprob", "frac_confident",
            "score_dispersion", "rho", "n_var", "skill150", "skill650"]
    done = set()
    if out.exists():
        done = {(r["dms_id"], r["layer"], r["direction"], r["alpha"])
                for r in csv.DictReader(out.open())}
        print(f"resuming: {len(done)} rows present")
    else:
        with out.open("w", newline="") as f:
            csv.DictWriter(f, fieldnames=keys).writeheader()

    jobs = []
    for L in layers:
        for d in want:
            for a in alphas:
                if a == 0 and d != "steer":
                    continue          # alpha=0 is the same run for every direction
                jobs.append((L, d, a))
    print(f"{len(jobs)} configs x {len(ev)} assays", flush=True)

    for j, (L, d, a) in enumerate(jobs, 1):
        vec = None if a == 0 else dirs[L][d] * (a * act[L])
        for it in ev:
            key = (it["dms_id"], str(L), d, str(a))
            if key in done:
                continue
            pp, _ = run_assay(mdl, tok, dev, it, L, vec, args.batch_tokens)
            row = dict(dms_id=it["dms_id"], uniprot=it["uniprot"],
                       taxon=it["taxon"], layer=L, direction=d, alpha=a,
                       skill150=it["skill150"], skill650=it["skill650"],
                       **summarise(it, pp))
            with out.open("a", newline="") as f:
                csv.DictWriter(f, fieldnames=keys).writerow(row)
        print(f"[{j:3d}/{len(jobs)}] L{L} {d:6s} a={a:+.3f} done", flush=True)


if __name__ == "__main__":
    main()
