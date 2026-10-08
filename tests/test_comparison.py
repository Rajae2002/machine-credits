"""Tests de la comparaison statistique (Phase 8)."""
import numpy as np
import pytest

from algorithms.baselines import RandomPolicy, RiskBasedPolicy
from environment.credit_env import CreditEnv
from evaluation import comparison as comp
from evaluation import experiments as ex
from evaluation.exact import ExactMDP
from training.trainer import evaluate_by_simulation

KW = dict(eval_every=100, eval_episodes=10, final_eval_episodes=50)


@pytest.fixture(scope="module")
def mdp():
    return ExactMDP()


def test_compare_means_detects_clear_difference():
    a = [10.0, 10.2, 9.8, 10.1, 9.9, 10.0, 10.3, 9.7]
    b = [0.0, 0.2, -0.2, 0.1, -0.1, 0.0, 0.3, -0.3]
    d, (lo, hi), p = comp.compare_means(a, b)
    assert d == pytest.approx(10.0, abs=0.1)
    assert lo > 9 and hi < 11
    assert p < 0.01


def test_compare_means_identical_samples():
    x = [1.0, 2.0, 3.0, 4.0, 5.0]
    d, (lo, hi), p = comp.compare_means(x, x)
    assert d == 0.0 and p == 1.0 and lo < 0 < hi


def test_bootstrap_ci_brackets_the_mean_and_is_reproducible():
    v = [1.0, 2.0, 3.0, 4.0, 10.0]
    lo, hi = comp.bootstrap_ci(v)
    assert lo <= np.mean(v) <= hi
    assert (lo, hi) == comp.bootstrap_ci(v)


def test_episodes_to_threshold():
    ep = np.array([0, 1000, 2000])
    J = np.array([0.0, 5.0, 10.0])
    assert comp.episodes_to_threshold(ep, J, 5.0) == 1000
    assert np.isnan(comp.episodes_to_threshold(ep, J, 11.0))


def test_policy_diagnostics():
    pi_star = np.ones(54, dtype=int)
    policy = pi_star.copy()
    assert comp.policy_diagnostics(policy, pi_star) == (0, 0)
    policy[3], policy[4] = 0, 2          # un REFUSE à tort, un REDUCED au lieu de ACCEPT
    assert comp.policy_diagnostics(policy, pi_star) == (2, 1)


def test_simulate_policy_matches_trainer_evaluation():
    fn = RiskBasedPolicy().greedy_action
    sim = comp.simulate_policy(fn, 300)
    stats = evaluate_by_simulation(CreditEnv(seed=0), fn, 300, comp.GAMMA)
    assert sim["acceptance"] == pytest.approx(stats["acceptance_rate"])
    assert sim["mean_return"] == pytest.approx(stats["mean_return"])
    assert sim["default_contract"] == pytest.approx(stats["default_rate"])
    for risk, name in enumerate(("LOW", "MEDIUM", "HIGH")):
        assert sim[f"ret_{name}"] == pytest.approx(stats["by_risk"][risk])


def test_always_refuse_metrics():
    sim = comp.simulate_policy(lambda s: 0, 200)
    assert sim["acceptance"] == 0.0 and sim["mean_return"] == 0.0
    assert sim["default_contract"] == 0.0 and sim["default_period"] == 0.0


def test_default_per_period_identity():
    """défaut/période = (défaut/contrat) / (durée moyenne d'un contrat)."""
    sim = comp.simulate_policy(lambda s: 1, 500)
    assert sim["acceptance"] == 1.0
    assert sim["mean_periods"] > 1.0
    assert sim["default_period"] == pytest.approx(sim["default_contract"] / sim["mean_periods"])


def test_risk_rule_beats_random_in_simulation():
    risk = comp.simulate_policy(RiskBasedPolicy().greedy_action, 1000)
    rand = comp.simulate_policy(RandomPolicy(seed=0).greedy_action, 1000)
    assert risk["mean_return"] > rand["mean_return"]


def test_load_runs_filters_by_algo_and_config(tmp_path, mdp):
    d = str(tmp_path)
    ex.run_one("q_learning", 0.1, 0.99, 0, 200, mdp, d, **KW)
    ex.run_one("q_learning", 0.03, 0.99, 1, 200, mdp, d, **KW)
    assert len(comp.load_runs(d, "q_learning")) == 2
    only = comp.load_runs(d, "q_learning", alpha=0.03)
    assert len(only) == 1 and int(only[0]["seed"]) == 1
    assert comp.load_runs(d, "double_q_learning") == []
    assert comp.load_runs(str(tmp_path / "absent"), "q_learning") == []


def test_per_run_table_fields_and_ranges(tmp_path, mdp):
    ex.run_one("double_q_learning", 0.1, 0.99, 0, 300, mdp, str(tmp_path), **KW)
    runs = comp.load_runs(str(tmp_path), "double_q_learning")
    ref = ex.reference_values(mdp)
    pi_star = mdp.value_iteration(comp.GAMMA).argmax(axis=1)
    rows = comp.per_run_table(runs, "double_q_learning", ref["J_star"], pi_star, 50)
    assert len(rows) == 1
    r = rows[0]
    assert 0.0 <= r["acceptance"] <= 1.0
    assert r["J"] <= ref["J_star"] + 1e-9
    assert 0 <= r["wrong_refusals"] <= r["wrong_states"] <= 54
