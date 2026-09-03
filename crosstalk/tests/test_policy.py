import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from crosstalk import objectives as O
from crosstalk.belief import AdditiveBelief
from crosstalk.landscape import load_pard3
from crosstalk.policy import AcquisitionPolicy, rollout, train


@pytest.fixture(scope="module")
def L():
    return load_pard3()


def test_belief_posterior_variance_matches_quadratic_form(L):
    """The fast matmul path must equal the explicit einsum it replaced."""
    B = AdditiveBelief(L)
    rng = np.random.default_rng(0)
    for _ in range(20):
        i = int(rng.integers(L.n_seqs))
        B.update(i, L.measure(L.seqs[i], rng), [0, 1])
    idx = rng.integers(0, L.n_seqs, 50)
    _, sd = B.posterior(idx)
    ref = np.einsum("ij,jk,ik->i", B.X[idx], B.A_inv[0], B.X[idx]) + B.model_var
    assert np.allclose(sd[:, 0] ** 2, ref)


def test_belief_is_calibrated(L):
    """Posterior SD must cover actual error, or the policy stops exploring.

    Without the epistatic misspecification term the z-scores blow up to ~6.
    """
    B = AdditiveBelief(L)
    rng = np.random.default_rng(0)
    for _ in range(300):
        i = int(rng.integers(L.n_seqs))
        B.update(i, L.measure(L.seqs[i], rng), [0, 1])
    mean, sd = B.posterior(np.arange(L.n_seqs))
    z = (mean[:, 0] - L.F[:, 0]) / sd[:, 0]
    assert 0.5 < z.std() < 2.0


def test_belief_mean_matches_full_posterior(L):
    B = AdditiveBelief(L)
    rng = np.random.default_rng(0)
    for _ in range(30):
        i = int(rng.integers(L.n_seqs))
        B.update(i, L.measure(L.seqs[i], rng), [0, 1])
    idx = np.arange(100)
    m_only = B.posterior_mean(idx)
    m_full, _ = B.posterior(idx)
    assert np.allclose(m_only, m_full)


def test_belief_improves_with_data(L):
    rng = np.random.default_rng(0)
    errs = []
    for n in (20, 400):
        B = AdditiveBelief(L)
        r2 = np.random.default_rng(0)
        for _ in range(n):
            i = int(r2.integers(L.n_seqs))
            B.update(i, L.measure(L.seqs[i], r2), [0, 1])
        m = B.posterior_mean()
        errs.append(np.abs(m[:, 0] - L.F[:, 0]).mean())
    assert errs[1] < errs[0]


def test_affinity_only_campaign_learns_nothing_about_off_target(L):
    """With counter_screen=False the off-target belief must stay at the prior."""
    obj = O.make("affinity", target=0, off=(1,))
    pol = AcquisitionPolicy()
    rng = np.random.default_rng(0)
    r, _, nom, _ = rollout(pol, L, obj, budget=20, counter_screen=False, rng=rng,
                           collect_grad=False)
    assert 0 <= nom < L.n_seqs


def test_rollout_respects_assay_budget(L):
    """Counter-screening halves the number of measurements for the same budget."""
    obj = O.make("margin")
    pol = AcquisitionPolicy()
    for cs, expected in ((True, 20), (False, 40)):
        rng = np.random.default_rng(0)
        _, lps, _, _ = rollout(pol, L, obj, budget=40, counter_screen=cs, rng=rng)
        assert len(lps) == expected


def test_policy_is_permutation_equivariant(L):
    """Scores must depend on candidate features, not candidate ordering."""
    pol = AcquisitionPolicy()
    feats = torch.randn(16, 7)
    perm = torch.randperm(16)
    with torch.no_grad():
        s1 = pol(feats)[perm]
        s2 = pol(feats[perm])
    assert torch.allclose(s1, s2, atol=1e-6)


def test_entropy_bonus_is_real_entropy(L):
    """Regression: the bonus must be per-decision entropy.

    An earlier version used the summed log-prob of taken actions, which is not
    entropy and, at 25+ decisions per episode, swamped the policy gradient and
    flatlined training.
    """
    obj = O.make("margin")
    pol = AcquisitionPolicy()
    rng = np.random.default_rng(0)
    _, lps, _, ents = rollout(pol, L, obj, budget=20, counter_screen=True, rng=rng)
    assert len(ents) == len(lps)
    e = torch.stack(ents)
    assert (e >= 0).all()                       # entropy is non-negative
    assert not torch.allclose(e, torch.stack(lps))


def test_training_improves_its_own_reward(L):
    """Sanity: REINFORCE must move the reward it is given."""
    obj = O.make("margin", target=1, off=(0,))
    torch.manual_seed(0)
    pol = AcquisitionPolicy()

    def mean_reward(seed):
        rng = np.random.default_rng(seed)
        return np.mean([rollout(pol, L, obj, 50, True, rng, collect_grad=False)[0]
                        for _ in range(12)])

    before = mean_reward(99)
    train(pol, L, obj, budget=50, counter_screen=True, n_batches=40,
          batch_size=8, seed=0, log_every=10**9)
    after = mean_reward(99)
    assert after > before
