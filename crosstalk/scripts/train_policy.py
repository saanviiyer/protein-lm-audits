"""Train acquisition policies and watch reward diverge from ground truth.

Both arms are counter-screened, so both PAY for the off-target measurement and
receive identical information. They differ only in whether the off-target
enters the reward. Ground-truth specificity is evaluated periodically during
training, so the question is direct: does optimizing the affinity reward harder
make the designs better, or only make the reward larger?
"""
import argparse, csv, json, sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from crosstalk import objectives as O
from crosstalk.landscape import load_pard3
from crosstalk.metrics import evaluate
from crosstalk.policy import AcquisitionPolicy, rollout, train

TASKS = {"cognate": (0, (1,)), "swap": (1, (0,))}


def eval_policy(pol, L, obj_reward, obj_eval, budget, counter_screen, n=40, seed=1234):
    rng = np.random.default_rng(seed)
    rewards, noms = [], []
    for _ in range(n):
        r, _, nom, _ = rollout(pol, L, obj_reward, budget, counter_screen, rng,
                               collect_grad=False)
        rewards.append(r)
        noms.append(L.seqs[nom])
    m = evaluate(L, noms, obj_eval)
    m["reward_achieved"] = float(np.mean(rewards))
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=int, default=50)
    ap.add_argument("--batches", type=int, default=150)
    ap.add_argument("--batch-size", type=int, default=10)
    ap.add_argument("--checkpoint-every", type=int, default=25)
    ap.add_argument("--eval-n", type=int, default=40)
    ap.add_argument("--out", default="results/policy_training.csv")
    args = ap.parse_args()

    L = load_pard3()
    rows = []
    for task, (tgt, off) in TASKS.items():
        obj_eval = O.make("margin", target=tgt, off=off)
        for reward in ("affinity", "margin"):
            obj_reward = O.make(reward, target=tgt, off=off)
            torch.manual_seed(0)
            pol = AcquisitionPolicy()
            print(f"\n=== {task} / reward={reward} / counter-screened ===", flush=True)

            done = 0
            while done <= args.batches:
                m = eval_policy(pol, L, obj_reward, obj_eval, args.budget, True, args.eval_n)
                rows.append(dict(task=task, reward=reward, batches=done, **m))
                print(f"  [{done:4d} batches] reward={m['reward_achieved']:.4f}  "
                      f"truth success={m['success_rate']:.2f} crosstalk={m['crosstalk_rate']:.2f}",
                      flush=True)
                if done == args.batches:
                    break
                step = min(args.checkpoint_every, args.batches - done)
                train(pol, L, obj_reward, budget=args.budget, counter_screen=True,
                      n_batches=step, batch_size=args.batch_size, seed=done,
                      log_every=10**9)
                done += step

            torch.save(pol.state_dict(), f"results/policy_{task}_{reward}.pt")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    print(f"\nwrote {out} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
