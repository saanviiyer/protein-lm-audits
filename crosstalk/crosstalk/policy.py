"""A learned acquisition policy for the budgeted specificity campaign.

The raw action space is one action per sequence (7,882 on ParD3), which is far
too large to parameterize directly and would not transfer between landscapes.
Instead the policy is permutation-equivariant: a shared MLP scores each
candidate from belief-derived features, and the next assay is sampled from a
softmax over those scores. That is a learned acquisition function, and it is
what makes the comparison meaningful -- the policy and the search baselines
differ only in how they choose the next measurement, not in how they nominate.

Trained with REINFORCE on the terminal reward, which is the agent's own
objective evaluated on the ground truth of its nominated design.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

# the nets here are tiny; extra threads only add contention when many
# runs share a host, and they make results depend on host load
torch.set_num_threads(1)

from .belief import AdditiveBelief
from .landscape import Landscape
from .objectives import Objective

N_FEATURES = 7


class AcquisitionPolicy(nn.Module):
    def __init__(self, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(N_FEATURES, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        """feats (n_cand, N_FEATURES) -> scores (n_cand,)"""
        return self.net(feats).squeeze(-1)


def _features(belief: AdditiveBelief, pool: np.ndarray, counts: np.ndarray,
              obj: Objective, frac_left: float) -> np.ndarray:
    mean, sd = belief.posterior(pool)
    on_m, on_s = mean[:, obj.target], sd[:, obj.target]
    off_cols = list(obj.off)
    off_m = mean[:, off_cols].max(axis=1)
    off_s = sd[:, off_cols].max(axis=1)
    obj_hat = obj(mean)
    return np.stack([
        on_m, on_s, off_m, off_s, obj_hat,
        np.log1p(counts[pool]),
        np.full(len(pool), frac_left),
    ], axis=1)


def _make_pool(belief: AdditiveBelief, n_seqs: int, obj: Objective,
               rng: np.random.Generator, n_random: int = 192, n_top: int = 32) -> np.ndarray:
    """Random candidates plus the current best-believed ones, so the policy can
    both explore the space and exploit its surrogate."""
    rand = rng.choice(n_seqs, size=min(n_random, n_seqs), replace=False)
    mean = belief.posterior_mean()          # mean only: no variance scan
    top = np.argpartition(-obj(mean), n_top)[:n_top]
    return np.unique(np.concatenate([rand, top]))


def rollout(policy: AcquisitionPolicy, L: Landscape, obj: Objective, budget: int,
            counter_screen: bool, rng: np.random.Generator,
            greedy: bool = False, collect_grad: bool = True):
    """Run one campaign.

    Returns (terminal_reward, log_probs, nominated_index, entropies).
    """
    partners = list(range(L.n_partners)) if counter_screen else [obj.target]
    cost = L.n_partners if counter_screen else 1
    n_steps = budget // cost

    belief = AdditiveBelief(L)
    counts = np.zeros(L.n_seqs)
    tested: list[int] = []
    log_probs, entropies = [], []

    for t in range(n_steps):
        pool = _make_pool(belief, L.n_seqs, obj, rng)
        feats = _features(belief, pool, counts, obj, 1.0 - t / max(n_steps, 1))
        ft = torch.as_tensor(feats, dtype=torch.float32)
        scores = policy(ft) if collect_grad else policy(ft).detach()
        dist = torch.distributions.Categorical(logits=scores)
        a = int(torch.argmax(scores)) if greedy else int(dist.sample())
        if collect_grad and not greedy:
            log_probs.append(dist.log_prob(torch.as_tensor(a)))
            entropies.append(dist.entropy())

        i = int(pool[a])
        y = L.measure(L.seqs[i], rng)
        belief.update(i, y, partners)
        counts[i] += 1
        tested.append(i)

    # nomination rule is identical to the search baselines: best believed
    # design among those actually measured
    uniq = np.unique(tested)
    mean = belief.posterior_mean(uniq)
    nominated = int(uniq[int(np.argmax(obj(mean)))])
    reward = float(obj(L.truth(L.seqs[nominated])))
    return reward, log_probs, nominated, entropies


def train(policy: AcquisitionPolicy, L: Landscape, obj: Objective, budget: int,
          counter_screen: bool, n_batches: int = 150, batch_size: int = 12,
          lr: float = 3e-3, entropy_coef: float = 0.01, seed: int = 0,
          log_every: int = 10, log_fn=print, eval_every: int = 0, eval_fn=None,
          opt: "torch.optim.Optimizer | None" = None):
    """REINFORCE with batch-normalized advantage.

    Pass `eval_every`/`eval_fn` to evaluate mid-training instead of chunking
    training into repeated calls: each call builds a fresh Adam, so chunking
    silently resets optimizer state and destabilises the run.
    """
    if opt is None:
        opt = torch.optim.Adam(policy.parameters(), lr=lr)
    rng = np.random.default_rng(seed)
    history = []

    for b in range(n_batches):
        rewards, batch_lp, batch_ent = [], [], []
        for _ in range(batch_size):
            r, lps, _, ents = rollout(policy, L, obj, budget, counter_screen, rng)
            rewards.append(r)
            batch_lp.append(torch.stack(lps).sum() if lps else torch.tensor(0.0))
            batch_ent.append(torch.stack(ents).mean() if ents else torch.tensor(0.0))

        R = torch.tensor(rewards, dtype=torch.float32)
        adv = (R - R.mean()) / (R.std() + 1e-6)
        logp = torch.stack(batch_lp)
        entropy = torch.stack(batch_ent).mean()
        # maximize entropy: a real per-decision entropy bonus, not the sum of
        # log-probs of taken actions (which is not entropy and swamps the
        # policy gradient at 25-100 decisions per episode)
        loss = -(adv * logp).mean() - entropy_coef * entropy

        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        opt.step()

        history.append(float(R.mean()))
        if (b + 1) % log_every == 0:
            recent = np.mean(history[-log_every:])
            log_fn(f"  batch {b+1:4d}/{n_batches}  mean reward {recent:.4f}")
        if eval_every and eval_fn is not None and (b + 1) % eval_every == 0:
            eval_fn(b + 1)
    return history
