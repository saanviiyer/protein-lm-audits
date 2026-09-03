import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from crosstalk import agents as A
from crosstalk import objectives as O
from crosstalk.envs import BudgetEnv, WalkEnv
from crosstalk.landscape import load_pard3
from crosstalk.metrics import evaluate
from crosstalk.solve import best_design, optimal_walk_value, ruggedness


@pytest.fixture(scope="module")
def L():
    return load_pard3()


def test_landscape_loads(L):
    assert L.n_seqs == 7882
    assert L.partners == ["ParE3", "ParE2"]
    assert L.truth("DKE")[0] == pytest.approx(1.0)  # WT normalised to 1


def test_noise_is_calibrated_not_invented(L):
    # single-measurement SD should be a few percent of dynamic range, from the
    # two biological replicates -- not a round made-up number
    rng_med = np.median(L.noise_sd[:, 0])
    assert 0.02 < rng_med < 0.06
    assert (L.noise_sd > 0).all()


def test_objectives_agree_on_a_clean_case():
    f = np.array([1.0, 0.0])
    assert O.make("affinity")(f) == pytest.approx(1.0)
    assert O.make("margin")(f) == pytest.approx(1.0)
    assert O.make("gated")(f) == pytest.approx(1.0)
    assert O.make("lagrangian")(f) == pytest.approx(1.0)


def test_objectives_diverge_on_a_promiscuous_design():
    f = np.array([1.0, 0.9])  # binds target AND off-target
    assert O.make("affinity")(f) == pytest.approx(1.0)      # blind to crosstalk
    assert O.make("margin")(f) == pytest.approx(0.1)
    assert O.make("gated")(f) == pytest.approx(-1.0)        # rejected
    assert O.make("lagrangian")(f) < 0


def test_is_specific_requires_both_sides():
    assert O.is_specific(np.array([1.0, 0.0]))
    assert not O.is_specific(np.array([1.0, 0.9]))   # promiscuous
    assert not O.is_specific(np.array([0.2, 0.0]))   # does not bind target


def test_measure_is_unbiased(L):
    rng = np.random.default_rng(0)
    reads = np.array([L.measure("DKE", rng) for _ in range(4000)])
    assert reads[:, 0].mean() == pytest.approx(L.truth("DKE")[0], abs=0.01)
    assert reads[:, 0].std() == pytest.approx(L.noise_sd[L.index["DKE"], 0], rel=0.15)


def test_best_design_matches_enumeration(L):
    obj = O.make("margin")
    seq, score = best_design(L, obj)
    assert score == pytest.approx(float(obj(L.F).max()))
    assert obj(L.truth(seq)) == pytest.approx(score)


def test_optimal_walk_value_is_monotone_in_horizon(L):
    obj = O.make("margin")
    vals = [optimal_walk_value(L, obj, "DKE", h) for h in (0, 1, 2, 3)]
    assert vals == sorted(vals)
    assert vals[0] == pytest.approx(float(obj(L.truth("DKE"))))
    # sequence length is 3, so a radius-3 ball is the whole landscape
    assert vals[-1] == pytest.approx(float(obj(L.F).max()))


def test_specificity_objective_is_more_rugged_than_affinity(L):
    """The core claim: pricing off-target binding roughly doubles local optima."""
    aff = ruggedness(L, O.make("affinity"))
    mar = ruggedness(L, O.make("margin"))
    assert mar["n_local_optima"] > aff["n_local_optima"]
    assert mar["frac_ascents_reaching_global"] < aff["frac_ascents_reaching_global"]


def test_budget_env_respects_budget(L):
    env = BudgetEnv(L, O.make("margin"), budget=5, seed=0)
    env.reset(0)
    done = False
    steps = 0
    while not done:
        _, _, done, _, _ = env.step(0)
        steps += 1
    assert steps == 5
    assert env.spent == 5


def test_walk_env_reaches_horizon(L):
    env = WalkEnv(L, O.make("margin"), horizon=3, seed=0)
    s, _ = env.reset(0)
    assert s == "DKE"
    done = False
    while not done:
        s, r, done, _, _ = env.step(env.actions()[0])
    assert env.t == 3


def test_assay_accounting_gives_affinity_agents_double_throughput(L):
    """Budget is in assays: counter-screening halves the number of variants seen."""
    rng = np.random.default_rng(0)
    c1 = A.Campaign(L, budget=100, counter_screen=False, rng=rng)
    c2 = A.Campaign(L, budget=100, counter_screen=True, rng=rng)
    n1 = n2 = 0
    while c1.can_afford():
        c1.measure(n1 % L.n_seqs); n1 += 1
    while c2.can_afford():
        c2.measure(n2 % L.n_seqs); n2 += 1
    assert n1 == 2 * n2


def test_agents_return_valid_designs(L):
    obj = O.make("margin")
    for name, fn in A.AGENTS.items():
        rng = np.random.default_rng(1)
        i = fn(L, obj, budget=100, counter_screen=True, rng=rng)
        assert 0 <= i < L.n_seqs, name


def test_evaluate_scores_on_truth_not_reward(L):
    obj = O.make("margin")
    m = evaluate(L, ["DYE", "DYE"], obj)
    assert m["crosstalk_rate"] == 0.0
    assert m["success_rate"] == 1.0
    m2 = evaluate(L, ["DIE"], obj)  # on=1.002, off=0.729 -> promiscuous
    assert m2["crosstalk_rate"] == 1.0
    assert m2["success_rate"] == 0.0


def test_campaign_measures_its_own_target_channel(L):
    """Regression: an affinity-only campaign on the swap task must assay ParE2.

    A hardcoded channel 0 silently made swap-task agents measure the wrong
    assay, learn nothing, and nominate a constant sequence at every budget.
    """
    rng = np.random.default_rng(0)
    c = A.Campaign(L, budget=20, counter_screen=False, rng=rng, target=1)
    c.measure(L.index["DKE"])
    assert c.cnt[L.index["DKE"], 1] == 1      # target channel observed
    assert c.cnt[L.index["DKE"], 0] == 0      # off-target never observed
    assert len(c.tested()) == 1


def test_swap_agents_are_seed_sensitive(L):
    """A learning agent must not collapse to one sequence across seeds."""
    obj = O.make("affinity", target=1, off=(0,))
    noms = {L.seqs[A.additive_model(L, obj, budget=200, counter_screen=False,
                                    rng=np.random.default_rng(s))] for s in range(8)}
    assert len(noms) > 1


def test_bpti_sequence_matches_the_published_positions():
    """The paper names T11 G12 P13 K15 A16 R17 I18 V34 Y35 G36 G37 R39.

    Same identity check that picked ParD3 out of many paralogs: if the sequence
    or the numbering convention were wrong, these would not line up.
    """
    from crosstalk.bpti import BPTI, MUT_POSITIONS
    assert len(BPTI) == 58
    assert "".join(BPTI[p - 1] for p in MUT_POSITIONS) == "TGPKARIVYGGR"


def test_bpti_landscape_parses_to_the_published_counts():
    """228 single mutants is what the paper reports; anything else is a parse bug."""
    from crosstalk.bpti import available, load_bpti
    if not available():
        pytest.skip("BPTI supplementary xlsx not present")
    L = load_bpti()
    assert L.n_single == 228
    assert L.partners == ["trypsin", "chymotrypsin", "mesotrypsin"]
    assert L.n_seqs > 14000


def test_bpti_rejects_spreadsheet_aggregate_rows():
    """MAX / MIN / AVG rows sit at the end of both sheets and are not variants."""
    from crosstalk.bpti import is_variant
    for good in ("T11A", "T11A_G12S", "R39K"):
        assert is_variant(good), good
    for bad in ("MAX", "MIN", "AVG", "MAX DDG", "AVG DDG", ""):
        assert not is_variant(bad), bad
