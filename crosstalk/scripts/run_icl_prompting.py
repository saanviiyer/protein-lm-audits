#!/usr/bin/env python3
"""Does multi-sequence prompting give a protein LM in-context learning of fitness?

Almonte et al. (2026) report that presenting several peptides in one context
enables in-context learning in protein language models. If that holds for
fitness, it addresses section 15 directly: the specificity information a probe
recovers from the representation might be reachable through the input instead,
with no training and no labelled probe.

The test is cheap on ParD3 because every variant differs at the same three
positions. One context yields one scoring function for the whole landscape from
three masked passes, so many contexts can be compared with error bars.

Four context conditions, each k sequences long, placed before the masked query:

  none        the section 9 baseline, no context at all
  high        k variants drawn from the top of measured fitness
  low         k variants drawn from the bottom
  random      k variants drawn uniformly

If the model learns from context, a context of functional variants should raise
the probability of functional residues at the masked sites and improve the
correlation with measured fitness.

THE CONTROL THAT DECIDES IT. A model can appear to learn in context by copying:
the context sequences carry residues at positions 61, 64 and 80, and simply
preferring residues seen in a high-fitness context would raise the correlation
without any learning. So each context is also scored by a bag-of-residues
predictor that counts which amino acids appear at those three positions among the
context examples and nothing else. Beating that is the minimum bar for calling
anything in-context learning rather than lookup.
"""
import argparse, csv, sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))

from crosstalk.boltz import MUT_POSITIONS, PARD3
from crosstalk.landscape import load_pard3
from crosstalk.plm import variant_full
from run_proxy_ladder import auc, spearman

AA = "ACDEFGHIKLMNPQRSTVWY"


@torch.no_grad()
def score_with_context(mdl, tok, dev, context_seqs, variants):
    """Masked-marginal scores for every variant, conditioned on a context.

    The query is the wild-type scaffold with the three sites masked, appended
    after the concatenated context, so all variants share three forward passes.
    """
    ctx = "".join(context_seqs)
    offset = len(ctx)
    lp = []
    for p in MUT_POSITIONS:
        full = ctx + PARD3
        masked = full[:offset + p - 1] + tok.mask_token + full[offset + p:]
        enc = tok(masked, return_tensors="pt").to(dev)
        logits = mdl(**enc).logits[0]
        at = (enc["input_ids"][0] == tok.mask_token_id).nonzero()[0, 0]
        lp.append(torch.log_softmax(logits[at].float(), -1).cpu().numpy())
    lp = np.stack(lp)
    wt = [PARD3[p - 1] for p in MUT_POSITIONS]
    tid = {a: tok.convert_tokens_to_ids(a) for a in AA}
    out = np.zeros(len(variants))
    for i, v in enumerate(variants):
        out[i] = sum(lp[k, tid[a]] - lp[k, tid[wt[k]]] for k, a in enumerate(v))
    return out


def bag_of_residues(context_variants, variants):
    """Count residues seen at each masked site in the context. The lookup baseline."""
    counts = np.zeros((3, 20))
    for v in context_variants:
        for k, a in enumerate(v):
            if a in AA:
                counts[k, AA.index(a)] += 1
    logp = np.log((counts + 0.5) / (counts.sum(1, keepdims=True) + 10))
    return np.array([sum(logp[k, AA.index(a)] for k, a in enumerate(v) if a in AA)
                     for v in variants])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="facebook/esm2_t33_650M_UR50D")
    ap.add_argument("--k", type=int, default=8, help="context sequences per prompt")
    ap.add_argument("--repeats", type=int, default=12)
    ap.add_argument("--out", default="results/icl_prompting.csv")
    args = ap.parse_args()

    L = load_pard3()
    variants = L.seqs
    w3, w2 = L.F[:, 0], L.F[:, 1]
    specific = (w3 >= 0.8) & (w2 <= 0.2)
    promisc = (w3 >= 0.8) & (w2 >= 0.6)
    mask = specific | promisc
    lab = specific[mask]

    from transformers import AutoModelForMaskedLM, AutoTokenizer
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(args.model)
    mdl = AutoModelForMaskedLM.from_pretrained(args.model).to(dev).eval()
    print(f"{args.model} on {dev}; k={args.k}, {args.repeats} repeats\n", flush=True)

    base = score_with_context(mdl, tok, dev, [], variants)
    print(f"{'condition':12s} {'rho(fitness)':>13s} {'AUC spec/promisc':>17s}")
    print(f"{'none':12s} {spearman(base, w3):+13.3f} {auc(base[mask], lab):17.3f}")

    order = np.argsort(w3)
    rows = [dict(condition="none", repeat=-1, rho_fitness=spearman(base, w3),
                 auc=auc(base[mask], lab), rho_bag=np.nan, auc_bag=np.nan)]
    pools = {"high": order[-800:], "low": order[:800],
             "random": np.arange(len(variants))}

    agg = {c: {"rho": [], "auc": [], "brho": [], "bauc": []} for c in pools}
    for cond, pool in pools.items():
        for r in range(args.repeats):
            rng = np.random.default_rng(1000 * r + hash(cond) % 997)
            pick = rng.choice(pool, size=args.k, replace=False)
            ctx_v = [variants[i] for i in pick]
            ctx = [variant_full(v) for v in ctx_v]
            s = score_with_context(mdl, tok, dev, ctx, variants)
            b = bag_of_residues(ctx_v, variants)
            agg[cond]["rho"].append(spearman(s, w3))
            agg[cond]["auc"].append(auc(s[mask], lab))
            agg[cond]["brho"].append(spearman(b, w3))
            agg[cond]["bauc"].append(auc(b[mask], lab))
            rows.append(dict(condition=cond, repeat=r, rho_fitness=agg[cond]["rho"][-1],
                             auc=agg[cond]["auc"][-1], rho_bag=agg[cond]["brho"][-1],
                             auc_bag=agg[cond]["bauc"][-1]))
        m = lambda x: float(np.mean(agg[cond][x]))
        ci = lambda x: 1.96 * float(np.std(agg[cond][x], ddof=1)) / np.sqrt(args.repeats)
        print(f"{cond:12s} {m('rho'):+13.3f} {m('auc'):17.3f}   "
              f"(+/-{ci('rho'):.3f}, +/-{ci('auc'):.3f})", flush=True)

    print(f"\n--- the control: does the model beat counting residues in its own context? ---")
    print(f"{'condition':12s} {'PLM rho':>9s} {'bag rho':>9s} {'PLM AUC':>9s} {'bag AUC':>9s}")
    for cond in pools:
        a = agg[cond]
        print(f"{cond:12s} {np.mean(a['rho']):+9.3f} {np.mean(a['brho']):+9.3f} "
              f"{np.mean(a['auc']):9.3f} {np.mean(a['bauc']):9.3f}")

    hi, lo = np.array(agg["high"]["rho"]), np.array(agg["low"]["rho"])
    d = hi.mean() - lo.mean()
    sd = np.sqrt(hi.var(ddof=1) / len(hi) + lo.var(ddof=1) / len(lo))
    print(f"\nhigh-fitness context minus low-fitness context: rho {d:+.3f} "
          f"+/- {1.96*sd:.3f}")
    print(f"context of any kind minus no context: rho "
          f"{np.mean([np.mean(agg[c]['rho']) for c in pools]) - spearman(base, w3):+.3f}")

    out = ROOT / args.out
    keys = sorted({k for r in rows for k in r})
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
