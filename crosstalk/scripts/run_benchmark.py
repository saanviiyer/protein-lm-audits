"""Headline experiment: reward specification under an equal assay budget.

An affinity-only campaign screens twice as many variants (one assay each) as a
counter-screening campaign (two assays each). Under a fixed assay budget, which
wins on ground-truth specificity? Every agent is scored on ground truth, never
on the reward it optimized.
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from crosstalk import agents as A
from crosstalk import objectives as O
from crosstalk.landscape import load_pard3
from crosstalk.metrics import evaluate

TASKS = {"cognate": (0, (1,)), "swap": (1, (0,))}
# reward name -> does the agent pay to counter-screen?
ARMS = [
    ("affinity", False),   # screen for binding only, 2x throughput
    ("affinity", True),    # counter-screens but ignores the result in its reward
    ("margin", True),
    ("gated", True),
    ("lagrangian", True),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=30)
    ap.add_argument("--budgets", type=int, nargs="+", default=[50, 100, 200, 400, 800])
    ap.add_argument("--out", default="results/benchmark.csv")
    args = ap.parse_args()

    L = load_pard3()
    rows = []
    for task, (tgt, off) in TASKS.items():
        for agent_name, agent_fn in A.AGENTS.items():
            for reward, counter in ARMS:
                obj_reward = O.make(reward, target=tgt, off=off)
                obj_eval = O.make("margin", target=tgt, off=off)
                for budget in args.budgets:
                    noms = []
                    for s in range(args.seeds):
                        rng = np.random.default_rng(10_000 * s + budget)
                        i = agent_fn(L, obj_reward, budget=budget,
                                     counter_screen=counter, rng=rng)
                        noms.append(L.seqs[i])
                    m = evaluate(L, noms, obj_eval)
                    rows.append(dict(
                        task=task, agent=agent_name, reward=reward,
                        counter_screen=int(counter), budget=budget, **m))
                    print(f"{task:8s} {agent_name:15s} {reward:11s} cs={int(counter)} "
                          f"B={budget:4d}  success={m['success_rate']:.2f} "
                          f"crosstalk={m['crosstalk_rate']:.2f} regret={m['specificity_regret']:.3f}",
                          flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {out} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
