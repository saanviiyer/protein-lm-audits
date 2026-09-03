#!/usr/bin/env python3
"""Is uncertainty a DIRECTION at all, independent of how EAST builds one?

Three questions the steering sweep cannot answer:
  1. SPLIT-HALF   does the entropy-weighted contrast vector reproduce when built
                  from disjoint halves of the extraction proteins?
  2. DECODABILITY is mask-position entropy linearly readable from the layer-L
                  hidden state, on HELD-OUT proteins? (ridge, grouped split)
  3. ALIGNMENT    does the EAST vector point along the ridge decoder, or is it a
                  different direction that happens to perturb the model?
"""
import argparse, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
from crosstalk.glm import _device
from run_east_steering import build_panel, run_assay


def east_vec(H, S):
    w = H / H.sum()
    return np.einsum("n,nld->ld", w, S) - S.mean(0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="facebook/esm2_t30_150M_UR50D")
    ap.add_argument("--max-len", type=int, default=400)
    ap.add_argument("--n-pos", type=int, default=40)
    ap.add_argument("--n-train", type=int, default=16)
    ap.add_argument("--n-eval", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="150M")
    args = ap.parse_args()

    panel = build_panel(args.max_len, args.n_pos, args.seed)
    rng = np.random.default_rng(args.seed + 7)
    order = rng.permutation(len(panel))
    train = [panel[i] for i in order[:args.n_train]]
    ev = [panel[i] for i in order[args.n_train:args.n_train + args.n_eval]]

    from transformers import AutoModelForMaskedLM, AutoTokenizer
    dev = _device()
    tok = AutoTokenizer.from_pretrained(args.model)
    mdl = AutoModelForMaskedLM.from_pretrained(args.model).to(dev).eval()

    def collect(items):
        H, S, G = [], [], []
        for n, it in enumerate(items, 1):
            pp, st = run_assay(mdl, tok, dev, it, 0, None, 6000, collect_states=True)
            for p in it["positions"]:
                H.append(pp[p]["ent"]); S.append(st[p]); G.append(it["dms_id"])
            print(f"  [{n:2d}/{len(items)}] {it['dms_id'][:38]:38s}", flush=True)
        return np.array(H), np.stack(S), np.array(G)

    print("extraction proteins:"); Ht, St, Gt = collect(train)
    print("held-out proteins:");   He, Se, Ge = collect(ev)
    np.savez(ROOT / f"results/east_probe_states_{args.tag}.npz",
             Ht=Ht, He=He, Gt=Gt, Ge=Ge)

    # 1. split-half by PROTEIN
    prots = sorted(set(Gt))
    r2 = np.random.default_rng(args.seed + 3)
    pi = r2.permutation(len(prots))
    A = {prots[i] for i in pi[:len(prots) // 2]}
    ma = np.array([g in A for g in Gt])
    Va, Vb = east_vec(Ht[ma], St[ma]), east_vec(Ht[~ma], St[~ma])
    Vf = east_vec(Ht, St)

    from numpy.linalg import norm
    print("\nlayer  splithalf_cos  ridge_r_heldout  cos(EAST, ridge)  "
          "r_of_EAST_projection")
    lam = 1e3
    for L in range(St.shape[1]):
        if norm(Va[L]) < 1e-9 or norm(Vb[L]) < 1e-9:
            print(f"{L:5d}  {'n/a (zero activation: ESM token dropout)':>40s}")
            continue
        sh = float(Va[L] @ Vb[L] / (norm(Va[L]) * norm(Vb[L])))
        X = St[:, L]; Xe = Se[:, L]
        mu, sd = X.mean(0), X.std(0) + 1e-8
        Xs, Xes = (X - mu) / sd, (Xe - mu) / sd
        yc = Ht - Ht.mean()
        w = np.linalg.solve(Xs.T @ Xs + lam * np.eye(Xs.shape[1]), Xs.T @ yc)
        pred = Xes @ w + Ht.mean()
        rr = float(np.corrcoef(pred, He)[0, 1])
        wv = w / sd                                  # back to raw activation space
        cos = float(Vf[L] @ wv / (norm(Vf[L]) * norm(wv)))
        proj = Xe @ (Vf[L] / norm(Vf[L]))
        rp = float(np.corrcoef(proj, He)[0, 1])
        print(f"{L:5d}  {sh:+13.3f}  {rr:+15.3f}  {cos:+16.3f}  {rp:+20.3f}")


if __name__ == "__main__":
    main()
