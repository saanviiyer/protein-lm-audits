#!/usr/bin/env python3
"""Can the trust gate be gamed? Red-teaming our own verifier.

The agent decides whether to exploit a scoring function by measuring that
scorer's AUROC for the top decile ON THE MEASUREMENTS TAKEN SO FAR, and
exploiting it if it clears 0.55. That gate is the verifier. This attacks it.

THE VULNERABILITY. A supervised model is scored leave-one-out, so it cannot pass
by memorising. A *precomputed proxy column* is scored in-sample, because a
parameter-free scorer has identical held-out and in-sample predictions -- there
is nothing to hold out. But nothing checks that the supplied column really is
parameter-free. A developer who fits anything to their own measurements and hands
the result in as a "precomputed score" gets it certified.

This measures the cost of that. Three scorers are put to the gate:

  honest      a real zero-shot pLM score, computed without seeing any fitness
  leaked      the measured fitness itself on observed rows, noise on the pool
  partial     a noisy blend, sweeping how much leakage is needed to pass

For each: what the GATE sees (AUROC on observed) against what the agent actually
GETS (true elites captured from the pool). A verifier that cannot separate these
is certifying specification gaming.

    python scripts/run_verifier_redteam.py
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from gauntlet.campaign import elite_auroc, factorized_features  # noqa: E402
from gauntlet import proteingym as pg  # noqa: E402

TRUST = 0.55


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pg_dir", default="data/proteingym")
    ap.add_argument("--out", default="results")
    ap.add_argument("--budget", type=int, default=48)
    ap.add_argument("--seeds", type=int, default=40)
    ap.add_argument("--max_assays", type=int, default=12)
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
        k = max(1, int(round(0.01 * len(y))))
        elite = np.zeros(len(y), bool)
        elite[np.argsort(-y)[:k]] = True

        for seed in range(args.seeds):
            rng = np.random.default_rng(seed)
            obs = rng.choice(len(y), args.budget, replace=False)
            pool = np.setdiff1d(np.arange(len(y)), obs)
            yo = y[obs]

            # An attacker fits to the observed measurements and supplies the
            # result as a "precomputed score". On the pool it knows nothing.
            def leaked(frac):
                s = np.empty(len(y))
                s[obs] = frac * yo + (1 - frac) * rng.normal(size=len(obs)) * np.std(yo)
                s[pool] = rng.normal(size=len(pool))
                return s

            cands = {"honest_zero_shot": zs, "leaked_full": leaked(1.0)}
            for f in (0.25, 0.5, 0.75):
                cands[f"leaked_{int(f * 100)}pct"] = leaked(f)

            for name, s in cands.items():
                gate = elite_auroc(s[obs], yo)                 # what the verifier sees
                order = pool[np.argsort(-s[pool])[:args.budget]]
                got = float(elite[order].sum())                # what the agent gets
                rows.append({"assay": r.DMS_id, "seed": seed, "scorer": name,
                             "gate_auroc": gate, "passes_gate": bool(
                                 np.isfinite(gate) and gate >= TRUST),
                             "true_yield": got,
                             "random_yield": args.budget * elite[pool].sum() / len(pool)})
        print(f"  {r.DMS_id}", flush=True)

    t = pd.DataFrame(rows)
    os.makedirs(args.out, exist_ok=True)
    t.to_csv(os.path.join(args.out, "verifier_redteam.csv"), index=False)

    print(f"\n{'=' * 78}\nRED-TEAMING THE TRUST GATE — {len(t)} trials, "
          f"{t.assay.nunique()} assays\n{'=' * 78}")
    print("The gate sees AUROC on measurements taken so far. The agent gets what "
          "the scorer\nactually finds in the unmeasured pool. A sound verifier "
          "links the two.\n")
    g = t.groupby("scorer").agg(
        gate_auroc=("gate_auroc", "mean"),
        passes=("passes_gate", "mean"),
        true_yield=("true_yield", "mean"),
        random_yield=("random_yield", "mean"))
    g["yield_vs_random"] = g.true_yield - g.random_yield
    print(g.round(3).to_string())

    lk = g.loc["leaked_full"]
    hn = g.loc["honest_zero_shot"]
    print(f"\nA scorer that simply memorises the observed measurements clears the "
          f"{TRUST} gate\n{100 * lk.passes:.0f}% of the time at mean AUROC "
          f"{lk.gate_auroc:.3f} — higher than the honest scorer's "
          f"{hn.gate_auroc:.3f} —\nand then delivers "
          f"{lk.yield_vs_random:+.3f} elites against random, versus "
          f"{hn.yield_vs_random:+.3f} for the honest one.")
    print("\nThe gate ranks the gamed scorer ABOVE the useful one. It is measuring "
          "fit to\ndata already collected, which is exactly what an attacker "
          "optimises.")

    print("\nHow much leakage is needed to pass?")
    sub = g.loc[[i for i in g.index if i.startswith("leaked_")]].sort_values("gate_auroc")
    print(sub[["gate_auroc", "passes", "yield_vs_random"]].round(3).to_string())
    # ------------------------------------------------------------------
    # THE DEFENCE: forward validation.
    #
    # The attack works because the gate scores a supplied column on data the
    # attacker already had. A scorer handed over at round t can only have been
    # fitted to measurements from rounds <= t, so evaluate it ONLY on what was
    # acquired afterwards. Nothing about the scorer needs to be trusted or
    # inspected -- the ordering of acquisition does the work.
    # ------------------------------------------------------------------
    rows2 = []
    for r in ref.itertuples():
        records, y, _ = pg.load_assay(r.DMS_id, r.target_seq,
                                      os.path.join(args.pg_dir, "assays"))
        zs = np.load(os.path.join(args.pg_dir, "esm_cache", f"{r.DMS_id}.npy"))
        if len(zs) != len(y):
            continue
        half = args.budget // 2
        for seed in range(args.seeds):
            rng = np.random.default_rng(1000 + seed)
            obs = rng.choice(len(y), args.budget, replace=False)
            early, late = obs[:half], obs[half:]     # early = what the attacker had

            def attacker(frac):
                s_ = np.empty(len(y))
                s_[:] = rng.normal(size=len(y))
                s_[early] = frac * y[early] + (1 - frac) * rng.normal(
                    size=len(early)) * np.std(y[early])
                return s_

            for name, sc in [("honest_zero_shot", zs),
                             ("leaked_full", attacker(1.0)),
                             ("leaked_50pct", attacker(0.5))]:
                rows2.append({
                    "scorer": name,
                    "naive_gate": elite_auroc(sc[obs], y[obs]),
                    "forward_gate": elite_auroc(sc[late], y[late]),
                })

    d = pd.DataFrame(rows2)
    d.to_csv(os.path.join(args.out, "verifier_defence.csv"), index=False)
    gg = d.groupby("scorer").agg(
        naive_gate=("naive_gate", "mean"),
        naive_passes=("naive_gate", lambda v: float((v >= TRUST).mean())),
        forward_gate=("forward_gate", "mean"),
        forward_passes=("forward_gate", lambda v: float((v >= TRUST).mean())))
    print(f"\n{'=' * 78}\nTHE DEFENCE — score the supplied column only on data "
          f"acquired AFTER it arrived\n{'=' * 78}")
    print(gg.round(3).to_string())
    print("\nThe attacker is given the first half of the observations and the gate "
          "is then\nrecomputed on the second half only. No inspection of the "
          "scorer is required;\nthe acquisition order alone separates a leaked "
          "column from a real one.")
    print(f"\nwrote {args.out}/verifier_defence.csv")
    print(f"wrote {args.out}/verifier_redteam.csv")


if __name__ == "__main__":
    main()
