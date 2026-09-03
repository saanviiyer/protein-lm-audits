"""Pool per-seed sweep cells and test reward-versus-truth divergence.

The question is paired and per-seed: within one training run, does optimizing
the reward harder move the reward and the ground-truth specificity in the same
direction? Pooling seeds without pairing would hide that.
"""
import argparse, csv, glob, json
from math import comb, sqrt
from pathlib import Path


def sign_test(deltas):
    """Two-sided exact sign test on paired per-seed deltas."""
    pos = sum(1 for d in deltas if d > 0)
    neg = sum(1 for d in deltas if d < 0)
    n = pos + neg
    if n == 0:
        return 1.0
    k = min(pos, neg)
    tail = sum(comb(n, i) for i in range(k + 1)) / (2 ** n)
    return min(1.0, 2 * tail)


def paired_t(deltas):
    """Two-sided paired t-test against zero. Higher power than the sign test at
    small seed counts, where the sign test cannot reach p<0.05 at all (n=5 has
    a minimum attainable p of 0.0625)."""
    n = len(deltas)
    if n < 2:
        return float("nan")
    m = sum(deltas) / n
    sd = sqrt(sum((d - m) ** 2 for d in deltas) / (n - 1))
    if sd == 0:
        return 0.0 if m != 0 else 1.0
    t = m / (sd / sqrt(n))
    # two-sided p from the t distribution via a continued-fraction incomplete beta
    df = n - 1
    x = df / (df + t * t)
    return _betainc(df / 2.0, 0.5, x)


def _betainc(a, b, x):
    """Regularised incomplete beta I_x(a,b), Lentz continued fraction."""
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    from math import lgamma, exp, log
    lbeta = lgamma(a) + lgamma(b) - lgamma(a + b)
    front = exp(a * log(x) + b * log(1 - x) - lbeta) / a
    f, c, d = 1.0, 1.0, 0.0
    for i in range(0, 300):
        m = i // 2
        if i == 0:
            num = 1.0
        elif i % 2 == 0:
            num = (m * (b - m) * x) / ((a + 2 * m - 1) * (a + 2 * m))
        else:
            num = -((a + m) * (a + b + m) * x) / ((a + 2 * m) * (a + 2 * m + 1))
        d = 1.0 + num * d
        if abs(d) < 1e-30: d = 1e-30
        d = 1.0 / d
        c = 1.0 + num / c
        if abs(c) < 1e-30: c = 1e-30
        f *= c * d
        if abs(1.0 - c * d) < 1e-10:
            break
    return front * (f - 1.0)


def mean_sd(xs):
    m = sum(xs) / len(xs)
    if len(xs) < 2:
        return m, 0.0
    return m, sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rows = []
    for f in sorted(glob.glob(str(Path(args.cells) / "*.csv"))):
        with open(f) as fh:
            rows.extend(list(csv.DictReader(fh)))

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

    finals = [r for r in rows if r["final"] == "1"]
    print(f"\n{'='*72}\nPAIRED PER-SEED ANALYSIS  (n seeds per cell shown)\n{'='*72}")
    summary = []
    for task in ("cognate", "swap"):
        for reward in ("affinity", "margin"):
            cell = [r for r in finals if r["task"] == task and r["reward"] == reward]
            seeds = sorted({int(r["seed"]) for r in cell})
            d_rew, d_suc, d_xt = [], [], []
            for s in seeds:
                pre = [r for r in cell if int(r["seed"]) == s and int(r["batches"]) == 0]
                post = [r for r in cell if int(r["seed"]) == s and int(r["batches"]) != 0]
                if not pre or not post:
                    continue
                d_rew.append(float(post[0]["reward_achieved"]) - float(pre[0]["reward_achieved"]))
                d_suc.append(float(post[0]["success_rate"]) - float(pre[0]["success_rate"]))
                d_xt.append(float(post[0]["crosstalk_rate"]) - float(pre[0]["crosstalk_rate"]))
            if not d_rew:
                continue
            mr, sr = mean_sd(d_rew); ms, ss = mean_sd(d_suc); mx, _ = mean_sd(d_xt)
            pr, ps = sign_test(d_rew), sign_test(d_suc)
            tr, ts = paired_t(d_rew), paired_t(d_suc)
            print(f"\n{task}/{reward}  (n={len(d_rew)} seeds)")
            print(f"  d(its own reward) = {mr:+.4f} +- {sr:.4f}   t p={tr:.4f} sign p={pr:.4f}"
                  f"   [{sum(1 for d in d_rew if d>0)}/{len(d_rew)} seeds up]")
            print(f"  d(truth success)  = {ms:+.4f} +- {ss:.4f}   t p={ts:.4f} sign p={ps:.4f}"
                  f"   [{sum(1 for d in d_suc if d>0)}/{len(d_suc)} seeds up]")
            print(f"  d(crosstalk)      = {mx:+.4f}")
            verdict = ("DIVERGES: reward up, truth not up" if mr > 0 and ms <= 0
                       else "aligned: both up" if mr > 0 and ms > 0
                       else "training failed to move its own reward")
            print(f"  -> {verdict}")
            summary.append(dict(task=task, reward=reward, n_seeds=len(d_rew),
                                d_reward=mr, d_reward_p_sign=pr, d_reward_p_t=tr,
                                d_success=ms, d_success_p_sign=ps, d_success_p_t=ts,
                                d_crosstalk=mx, verdict=verdict))

    with open(str(Path(args.out).with_name("sweep_summary.json")), "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"\nwrote {args.out} and sweep_summary.json")


if __name__ == "__main__":
    main()
