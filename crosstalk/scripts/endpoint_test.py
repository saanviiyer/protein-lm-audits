"""High-power endpoint test: does training help or harm ground truth?

The per-checkpoint curves are noisy at 40 eval episodes. This re-evaluates the
untrained and trained policies at 200 episodes each and tests the difference.
"""
import csv, sys
from math import comb
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from crosstalk import objectives as O
from crosstalk.landscape import load_pard3
from crosstalk.metrics import evaluate
from crosstalk.policy import AcquisitionPolicy, rollout

TASKS = {"cognate": (0, (1,)), "swap": (1, (0,))}
N_EVAL = 200
BUDGET = 50


def fisher(a, b, c, d):
    N, r1, c1 = a + b + c + d, a + b, a + c
    def p(x): return comb(r1, x) * comb(N - r1, c1 - x) / comb(N, c1)
    p0 = p(a)
    lo, hi = max(0, c1 - (N - r1)), min(r1, c1)
    return sum(p(x) for x in range(lo, hi + 1) if p(x) <= p0 * (1 + 1e-9))


def run(pol, L, obj_r, obj_e, seed):
    rng = np.random.default_rng(seed)
    rw, noms = [], []
    for _ in range(N_EVAL):
        r, _, nom, _ = rollout(pol, L, obj_r, BUDGET, True, rng, collect_grad=False)
        rw.append(r); noms.append(L.seqs[nom])
    m = evaluate(L, noms, obj_e)
    m["reward_achieved"] = float(np.mean(rw))
    return m


def main():
    L = load_pard3()
    rows = []
    print(f"Endpoint test, {N_EVAL} eval episodes per cell, budget={BUDGET} assays\n")
    for task, (tgt, off) in TASKS.items():
        obj_e = O.make("margin", target=tgt, off=off)
        for reward in ("affinity", "margin"):
            obj_r = O.make(reward, target=tgt, off=off)
            torch.manual_seed(0)
            untrained = AcquisitionPolicy()
            trained = AcquisitionPolicy()
            trained.load_state_dict(torch.load(f"results/policy_{task}_{reward}.pt"))

            u = run(untrained, L, obj_r, obj_e, 4242)
            t = run(trained, L, obj_r, obj_e, 4242)
            ku, kt = round(u["success_rate"] * N_EVAL), round(t["success_rate"] * N_EVAL)
            p = fisher(ku, N_EVAL - ku, kt, N_EVAL - kt)
            rows.append(dict(task=task, reward=reward,
                             reward_untrained=u["reward_achieved"], reward_trained=t["reward_achieved"],
                             success_untrained=u["success_rate"], success_trained=t["success_rate"],
                             crosstalk_untrained=u["crosstalk_rate"], crosstalk_trained=t["crosstalk_rate"],
                             p_success=p))
            arrow = "UP" if t["reward_achieved"] > u["reward_achieved"] else "down"
            tarrow = "UP" if t["success_rate"] > u["success_rate"] else "DOWN"
            print(f"{task:8s} reward={reward:9s}")
            print(f"   its own reward : {u['reward_achieved']:.4f} -> {t['reward_achieved']:.4f}  ({arrow})")
            print(f"   truth success  : {u['success_rate']:.3f} -> {t['success_rate']:.3f}  ({tarrow})  p={p:.4g}")
            print(f"   crosstalk      : {u['crosstalk_rate']:.3f} -> {t['crosstalk_rate']:.3f}\n")

    with open("results/endpoint_test.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print("wrote results/endpoint_test.csv")


if __name__ == "__main__":
    main()
