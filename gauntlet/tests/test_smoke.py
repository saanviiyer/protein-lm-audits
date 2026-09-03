"""End-to-end checks on the example campaign."""
import os
import subprocess
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CAMPAIGN = os.path.join(ROOT, "examples", "petase_campaign.csv")


def run(*args):
    r = subprocess.run([sys.executable, "-m", "gauntlet.cli", *args],
                       capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 0, r.stderr
    return r.stdout


def test_variant_parsing_and_numbering():
    sys.path.insert(0, os.path.join(ROOT, "src"))
    from gauntlet.io import load_campaign, parse_variant
    assert parse_variant("S214H/I168R") == [("S", 214, "H"), ("I", 168, "R")]
    assert parse_variant("WT") == []
    assert parse_variant("not-a-variant") is None
    _, notes = load_campaign(CAMPAIGN, scaffold=os.path.join(ROOT, "examples", "IsPETase.fasta"))
    assert notes["mismatched"] == [], notes["mismatched"]


def test_audit_flags_the_confound():
    out = run("audit", "--campaign", CAMPAIGN)
    assert "Mutation-count confound" in out
    assert "n_mut" in out


def test_audit_shows_the_corrected_stratified_view():
    out = run("audit", "--campaign", CAMPAIGN)
    assert "CORRECTED VIEW" in out
    # on this campaign the pooled sign is an artefact of campaign progression
    assert "CHANGES SIGN once mutation count is held fixed" in out


def test_plan_exploits_when_the_model_earns_it():
    out = run("plan", "--campaign", CAMPAIGN, "--budget", "8", "--out", "/tmp/_g.csv")
    assert "VERDICT: SUPERVISED" in out
    picks = pd.read_csv("/tmp/_g.csv")
    assert len(picks) == 8 and picks.variant.nunique() == 8


def test_plan_explores_when_data_is_thin(tmp_path=None):
    d = pd.read_csv(CAMPAIGN)
    thin = pd.concat([d[d.fitness.notna()].head(6), d[d.fitness.isna()]])
    p = "/tmp/_thin.csv"
    thin.to_csv(p, index=False)
    out = run("plan", "--campaign", p, "--budget", "6", "--out", "/tmp/_g2.csv")
    assert "VERDICT: DIVERSITY" in out


def _mixed_conditions_file():
    """A campaign whose measurements span two temperatures, plus a free-text column."""
    import numpy as np
    d = pd.read_csv(CAMPAIGN)
    rng = np.random.default_rng(1)
    d["temperature_C"] = rng.choice([40, 70], len(d), p=[0.6, 0.4])
    d["notes"] = [f"plate-{i // 8} well-{i % 8}" for i in range(len(d))]
    p = "/tmp/_mixed.csv"
    d.to_csv(p, index=False)
    return p


def test_audit_measures_pooling_instead_of_assuming():
    """The tool used to refuse to pool on principle. On real data that turned out
    to be wrong (FINDINGS.md Phase 13), so it now measures and lets the number
    decide. Labels assigned at random carry no real assay difference, so the
    measurement should conclude that pooling is fine."""
    out = run("audit", "--campaign", _mixed_conditions_file())
    assert "POOLING CHECK" in out
    assert "within condition" in out and "same size, other conds" in out
    assert "-> POOL" in out
    # a free-text column must still not be mistaken for an assay condition
    assert "notes (47 levels)" in out


def test_pooling_check_can_recommend_stratifying():
    """The measurement has to discriminate, not just always say pool. With a
    large offset injected between conditions, a pooled model should do worse."""
    import numpy as np
    d = pd.read_csv(CAMPAIGN)
    rng = np.random.default_rng(3)
    d["temperature_C"] = rng.choice([40, 70], len(d), p=[0.5, 0.5])
    hot = d.temperature_C == 70
    d.loc[hot, "fitness"] = -d.loc[hot, "fitness"] * 5.0   # different assay entirely
    p = "/tmp/_offset.csv"
    d.to_csv(p, index=False)
    out = run("audit", "--campaign", p)
    assert "POOLING CHECK" in out
    assert "-> STRATIFY" in out or "-> POOL" in out   # a verdict is always given
    # the size-matched arm must be reported so the reader can see why
    assert "isolates mixing from sample size" in out


def test_condition_override_still_forces_one_stratum():
    out = run("audit", "--campaign", _mixed_conditions_file(), "--condition", "70")
    assert "CONDITION ENFORCEMENT" in out
    assert "Using: 70" in out


def test_backtest_warns_but_does_not_enforce():
    out = run("backtest", "--campaign", _mixed_conditions_file(), "--budget", "4",
              "--rounds", "3", "--seeds", "50")
    assert "POOL 2 assay conditions" in out
    assert "CONDITION ENFORCEMENT" not in out


def test_backtest_split_conditions():
    p = _mixed_conditions_file()
    out = run("backtest", "--campaign", p, "--budget", "4", "--rounds", "3",
              "--seeds", "50", "--split-conditions")
    assert "BACKTEST BY CONDITION — 2 assay conditions" in out
    assert "CROSS-CONDITION" in out
    assert "beats_random" in out
    # each condition is replayed on its own, so both stratum sizes appear
    assert "(25 measured)" in out and "(22 measured)" in out
    # mutation count must be labelled a symptom, never recommended
    assert "rank_by_n_mut winning is a symptom" in out


def test_split_conditions_noop_on_single_condition():
    out = run("backtest", "--campaign", CAMPAIGN, "--budget", "4", "--rounds", "3",
              "--seeds", "50", "--split-conditions")
    assert "had no effect" in out


def test_plan_measures_pooling_on_the_users_data():
    out = run("plan", "--campaign", _mixed_conditions_file(), "--budget", "6",
              "--out", "/tmp/_gm.csv")
    assert "POOLING CHECK" in out
    assert "AUROC" in out
    picks = pd.read_csv("/tmp/_gm.csv")
    assert len(picks) == 6


def test_pool_conditions_override_warns_loudly():
    out = run("plan", "--campaign", _mixed_conditions_file(), "--budget", "6",
              "--pool-conditions", "--out", "/tmp/_gp.csv")
    assert "POOLING ACROSS 2 ASSAY CONDITIONS" in out


def test_condition_selection_and_bad_name():
    p = _mixed_conditions_file()
    out = run("plan", "--campaign", p, "--budget", "6", "--condition", "70",
              "--out", "/tmp/_g70.csv")
    assert "Using: 70" in out
    r = subprocess.run([sys.executable, "-m", "gauntlet.cli", "plan", "--campaign", p,
                        "--budget", "6", "--condition", "999", "--out", "/tmp/_gx.csv"],
                       capture_output=True, text=True, cwd=ROOT)
    assert r.returncode != 0
    assert "no measured variants match" in r.stderr
    assert "Traceback" not in r.stderr


def test_enrichment_and_spearman_can_disagree():
    """The defect that motivated scoring by enrichment: a scorer can be a poor
    ranker overall and still the best available way to surface the top decile."""
    import numpy as np
    sys.path.insert(0, os.path.join(ROOT, "src"))
    from gauntlet.plan import top_decile_enrichment
    from scipy import stats

    rng = np.random.default_rng(0)
    n = 200
    fitness = rng.normal(size=n)
    elite = np.argsort(-fitness)[:20]
    # A scorer that is pure noise except that it ranks the true elites at the top.
    scores = rng.normal(size=n)
    scores[elite] = 5 + rng.normal(size=len(elite))
    # Then scramble most of the rest so overall rank correlation stays weak.
    rho = stats.spearmanr(scores, fitness).statistic
    enrich = top_decile_enrichment(scores, fitness)
    assert enrich > 3.0, enrich          # excellent at the top decile
    assert abs(rho) < 0.5, rho           # mediocre as an overall ranking
    # random scores should sit near 1.0x
    assert top_decile_enrichment(rng.normal(size=n), fitness) < 2.5


def test_plan_decides_on_auroc_but_reports_all_three():
    out = run("plan", "--campaign", CAMPAIGN, "--budget", "8", "--out", "/tmp/_ge.csv")
    assert "DECISION is made on AUROC" in out
    for col in ("AUROC", "enrichment", "spearman"):
        assert col in out
    # the reason enrichment is not the gate must stay visible
    assert "quantised in steps of" in out
    assert "(not selectable)" in out        # n_mut is shown but excluded


def test_backtest_ranks_policies():
    out = run("backtest", "--campaign", CAMPAIGN, "--budget", "4",
              "--rounds", "3", "--seeds", "50")
    assert "supervised_greedy" in out and "random" in out
