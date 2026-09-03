#!/usr/bin/env python3
"""Does forward validation survive an attacker who knows about it?

Phase 17 defended against an attacker that leaks the outcomes it already has.
Forward validation catches that: score the supplied column only on data acquired
after it arrived. But that attacker did not know the defence existed. Two
adaptive attackers do.

  A. ADAPTIVE LEAK. Fit a model to the outcomes already collected and supply its
     predictions. This is the honest-looking version of the attack, and forward
     validation should let it through exactly to the extent that it genuinely
     generalises -- which is the defence working, not failing.

  B. POOL-SIDE MANIPULATION. Leave the supplied score untouched on everything
     measured so far, so the gate sees a perfectly honest scorer, and corrupt
     only the UNMEASURED pool: promote the candidates the attacker predicts are
     worst to the top of the ranking. The gate never evaluates the pool, so
     neither the naive gate nor forward validation can see this at all.

Attack B is the one that matters. It needs no leakage, survives the defence by
construction, and is realisable using only data the attacker legitimately has.

    python scripts/run_adaptive_attacker.py
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from gauntlet.campaign import elite_auroc, factorized_features  # noqa: E402
from gauntlet import proteingym as pg  # noqa: E402

TRUST, ALPHA = 0.55, 1.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pg_dir", default="data/proteingym")
    ap.add_argument("--out", default="results")
    ap.add_argument("--budget", type=int, default=48)
    ap.add_argument("--seeds", type=int, default=40)
    ap.add_argument("--max_assays", type=int, default=12)
    ap.add_argument("--audit_frac", type=float, default=0.25,
                    help="fraction of each ordered batch spent on a random audit")
    args = ap.parse_args()

    ref = pg.select_assays(os.path.join(args.pg_dir, "reference.csv"),
                           max_len=500, min_singles=1200)
    cached = {f[:-4] for f in os.listdir(os.path.join(args.pg_dir, "esm_cache"))}
    ref = ref[ref.DMS_id.isin(cached)].head(args.max_assays)

    rows = []
    for r in ref.itertuples():
        records, y, _ = pg.load_assay(r.DMS_id, r.target_seq,
                                      os.path.join(args.pg_dir, "assays"))
        zs = np.load(os.path.join(args.pg_dir, "esm_cache", f"{r.DMS_id}.npy"))
        if len(zs) != len(y):
            continue
        X, _ = factorized_features(records)
        k = max(1, int(round(0.01 * len(y))))
        elite = np.zeros(len(y), bool)
        elite[np.argsort(-y)[:k]] = True
        B = args.budget

        for seed in range(args.seeds):
            rng = np.random.default_rng(seed)
            obs = rng.choice(len(y), B, replace=False)
            pool = np.setdiff1d(np.arange(len(y)), obs)
            early, late = obs[:B // 2], obs[B // 2:]

            # Everything the attacker legitimately has: the early observations.
            mdl = Ridge(alpha=ALPHA).fit(X[early], y[early])
            pred_pool = mdl.predict(X[pool])

            cand = {}
            cand["honest"] = zs.copy()

            # A. adaptive leak: supply the model's own predictions everywhere
            a = np.empty(len(y))
            a[early] = y[early]
            rest = np.setdiff1d(np.arange(len(y)), early)
            a[rest] = mdl.predict(X[rest])
            cand["adaptive_leak"] = a

            # B. pool-side manipulation: honest on everything measured, and the
            # predicted-worst candidates promoted to the top of the pool.
            b = zs.copy()
            worst = pool[np.argsort(pred_pool)[:B]]
            b[worst] = np.max(zs) + 1.0 + rng.uniform(0, 1e-6, len(worst))
            cand["pool_side"] = b

            for name, s in cand.items():
                naive = elite_auroc(s[obs], y[obs])
                fwd = elite_auroc(s[late], y[late])
                picks = pool[np.argsort(-s[pool])[:B]]
                got = float(elite[picks].sum())
                # audit defence: spend a slice of the batch at random instead
                n_audit = int(round(args.audit_frac * B))
                aud = pool[np.argsort(-s[pool])[:B - n_audit]]
                extra = rng.choice(np.setdiff1d(pool, aud), n_audit, replace=False)
                audited = float(elite[np.concatenate([aud, extra])].sum())
                rows.append({
                    "assay": r.DMS_id, "seed": seed, "scorer": name,
                    "naive_gate": naive, "forward_gate": fwd,
                    "naive_pass": bool(np.isfinite(naive) and naive >= TRUST),
                    "forward_pass": bool(np.isfinite(fwd) and fwd >= TRUST),
                    "yield": got, "yield_with_audit": audited,
                    "random_yield": B * elite[pool].sum() / len(pool)})
        print(f"  {r.DMS_id}", flush=True)

    t = pd.DataFrame(rows)
    os.makedirs(args.out, exist_ok=True)
    t.to_csv(os.path.join(args.out, "adaptive_attacker.csv"), index=False)

    g = t.groupby("scorer").agg(
        naive_gate=("naive_gate", "mean"), naive_pass=("naive_pass", "mean"),
        forward_gate=("forward_gate", "mean"), forward_pass=("forward_pass", "mean"),
        yld=("yield", "mean"), rnd=("random_yield", "mean"),
        audited=("yield_with_audit", "mean"))
    g["adv"] = g.yld - g.rnd
    g["adv_audit"] = g.audited - g.rnd

    print(f"\n{'=' * 78}\nADAPTIVE ATTACKERS vs FORWARD VALIDATION — "
          f"{len(t)} trials\n{'=' * 78}")
    print(g[["naive_gate", "naive_pass", "forward_gate", "forward_pass",
             "adv"]].round(3).to_string())

    hon, lk, ps = g.loc["honest"], g.loc["adaptive_leak"], g.loc["pool_side"]
    print(f"\nA. Adaptive leak: forward pass rate {lk.forward_pass:.1%} "
          f"(naive {lk.naive_pass:.1%}), delivers {lk.adv:+.3f} vs random.")
    print("   Forward validation admits it only in so far as the underlying model "
          "really\n   generalises. That is the defence behaving correctly.")
    print(f"\nB. Pool-side manipulation: naive gate {ps.naive_gate:.3f}, forward "
          f"gate {ps.forward_gate:.3f},\n   pass rates {ps.naive_pass:.1%} and "
          f"{ps.forward_pass:.1%} — INDISTINGUISHABLE from honest "
          f"({hon.naive_gate:.3f}/{hon.forward_gate:.3f}).")
    print(f"   And it delivers {ps.adv:+.3f} against random, where honest delivers "
          f"{hon.adv:+.3f}.")
    print("\n   Forward validation cannot see this attack. The gate only ever "
          "evaluates the\n   scorer on measured data, and the manipulation lives "
          "entirely in the pool.")

    print(f"\nA partial mitigation: spend {args.audit_frac:.0%} of each batch on a "
          "random audit sample.")
    print(f"  {'scorer':16s} {'no audit':>10s} {'with audit':>12s}")
    for nm in ["honest", "adaptive_leak", "pool_side"]:
        print(f"  {nm:16s} {g.loc[nm, 'adv']:>+10.3f} {g.loc[nm, 'adv_audit']:>+12.3f}")
    print("  The audit slice costs the honest scorer some yield and puts a floor "
          "under the\n  adversarial case. It bounds the damage; it does not detect "
          "the attack.")
    print(f"\nwrote {args.out}/adaptive_attacker.csv")


if __name__ == "__main__":
    main()
