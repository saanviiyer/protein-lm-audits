"""One (task, reward, seed) training run. The unit of work the sweep fans out.

Training is continuous: a single optimizer for the whole run, with evaluation
via callback. Chunking training into repeated train() calls rebuilds Adam each
time and destabilises the run, which is what made an earlier single-seed sweep
uninterpretable.
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


def eval_policy(pol, L, obj_r, obj_e, budget, n, seed):
    rng = np.random.default_rng(seed)
    rw, noms = [], []
    for _ in range(n):
        r, _, nom, _ = rollout(pol, L, obj_r, budget, True, rng, collect_grad=False)
        rw.append(r); noms.append(L.seqs[nom])
    m = evaluate(L, noms, obj_e)
    m["reward_achieved"] = float(np.mean(rw))
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=list(TASKS))
    ap.add_argument("--reward", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--budget", type=int, default=50)
    ap.add_argument("--batches", type=int, default=150)
    ap.add_argument("--batch-size", type=int, default=10)
    ap.add_argument("--eval-every", type=int, default=25)
    ap.add_argument("--eval-n", type=int, default=60)
    ap.add_argument("--final-eval-n", type=int, default=200)
    ap.add_argument("--out", default="-")
    args = ap.parse_args()

    tgt, off = TASKS[args.task]
    L = load_pard3()
    obj_r = O.make(args.reward, target=tgt, off=off)
    obj_e = O.make("margin", target=tgt, off=off)

    torch.manual_seed(args.seed)
    pol = AcquisitionPolicy()
    eval_seed = 900_000 + args.seed
    rows = []

    def record(nb):
        m = eval_policy(pol, L, obj_r, obj_e, args.budget, args.eval_n, eval_seed)
        rows.append(dict(task=args.task, reward=args.reward, seed=args.seed,
                         batches=nb, final=0, **m))
        print(f"[{args.task}/{args.reward}/s{args.seed}] {nb:4d} batches "
              f"reward={m['reward_achieved']:.4f} success={m['success_rate']:.2f}", flush=True)

    before = eval_policy(pol, L, obj_r, obj_e, args.budget, args.final_eval_n, eval_seed)
    record(0)
    train(pol, L, obj_r, budget=args.budget, counter_screen=True,
          n_batches=args.batches, batch_size=args.batch_size, seed=args.seed,
          log_every=10**9, eval_every=args.eval_every, eval_fn=record)
    after = eval_policy(pol, L, obj_r, obj_e, args.budget, args.final_eval_n, eval_seed)

    for tag, m in (("untrained", before), ("trained", after)):
        rows.append(dict(task=args.task, reward=args.reward, seed=args.seed,
                         batches=(0 if tag == "untrained" else args.batches),
                         final=1, **m))

    payload = json.dumps(rows)
    if args.out == "-":
        print("RESULT_JSON " + payload)
    else:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
