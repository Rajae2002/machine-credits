"""Tests : solution exacte, baselines, évaluation, entraînement."""
import numpy as np
import pytest

from config import N_STATES, N_ACTIONS, HyperParams
from environment.credit_env import CreditEnv
from environment.state_space import decode_state
from algorithms.baselines import RandomPolicy, RiskBasedPolicy
from evaluation.exact import ExactMDP, initial_distribution
from training.trainer import train, make_agent, evaluate_by_simulation

GAMMA = 0.95


@pytest.fixture(scope="module")
def mdp():
    return ExactMDP()


def test_initial_distribution_sums_to_one():
    mu = initial_distribution()
    assert mu.shape == (N_STATES,)
    assert mu.sum() == pytest.approx(1.0)
    assert np.all(mu > 0)


def test_value_iteration_is_bellman_fixed_point(mdp):
    Q = mdp.value_iteration(GAMMA)
    Q_check = mdp.R + GAMMA * (mdp.T @ Q.max(axis=1))
    assert np.allclose(Q, Q_check, atol=1e-8)


def test_refuse_is_zero_and_policy_evaluation_matches_vstar(mdp):
    Q = mdp.value_iteration(GAMMA)
    assert np.allclose(Q[:, 0], 0.0)                       # REFUSE vaut toujours 0
    V_pi = mdp.policy_evaluation(Q.argmax(axis=1), GAMMA)
    assert np.allclose(V_pi, Q.max(axis=1), atol=1e-6)     # V^{pi*} = V*


def test_optimal_policy_beats_baselines(mdp):
    Q = mdp.value_iteration(GAMMA)
    J_star = mdp.policy_value(Q.argmax(axis=1), GAMMA)
    J_risk = mdp.policy_value(RiskBasedPolicy().policy(), GAMMA)
    J_rand = mdp.policy_value(RandomPolicy().policy_matrix(), GAMMA)
    J_refuse = mdp.policy_value(np.zeros(N_STATES, dtype=int), GAMMA)
    assert J_refuse == pytest.approx(0.0, abs=1e-9)
    assert J_star > 0
    assert J_star >= J_risk - 1e-9
    assert J_star >= J_rand - 1e-9


def test_policy_evaluation_matches_simulation(mdp):
    """Le calcul exact et la simulation décrivent le même MDP."""
    env, pol = CreditEnv(seed=0), RiskBasedPolicy()
    stats = evaluate_by_simulation(env, pol.greedy_action, 20_000, GAMMA)
    J = mdp.policy_value(pol.policy(), GAMMA)
    assert stats["mean_disc_return"] == pytest.approx(J, abs=1.0)


def test_risk_based_policy_mapping():
    pol = RiskBasedPolicy().policy()
    expected = {0: 1, 1: 2, 2: 0}                          # LOW->ACCEPT, MEDIUM->REDUCED, HIGH->REFUSE
    for s in range(N_STATES):
        assert pol[s] == expected[decode_state(s)[0]]


def test_random_policy_is_uniform():
    rp = RandomPolicy(seed=0)
    assert np.allclose(rp.policy_matrix(), 1.0 / 3.0)
    assert {rp.greedy_action(0) for _ in range(300)} == {0, 1, 2}


def test_evaluation_uses_same_clients_and_is_repeatable():
    env, pol = CreditEnv(seed=0), RiskBasedPolicy()
    s1 = evaluate_by_simulation(env, pol.greedy_action, 200, GAMMA)
    s2 = evaluate_by_simulation(env, pol.greedy_action, 200, GAMMA)
    assert s1 == s2


def test_train_history_shapes():
    hp = HyperParams(n_episodes=300, eval_every=100, eval_episodes=20, eps_decay=0.99, seed=0)
    h = train(make_agent("q_learning", hp), hp, mdp=ExactMDP())
    assert h["train_reward"].shape == (300,)
    assert h["eval_episode"] == [0, 100, 200, 300]
    assert len(h["eval_J"]) == 4 and np.all(np.isfinite(h["eval_J"]))


def test_train_is_reproducible():
    hp = HyperParams(n_episodes=300, eval_every=100, eval_episodes=20, eps_decay=0.99, seed=3)
    a1, a2 = make_agent("q_learning", hp), make_agent("q_learning", hp)
    h1, h2 = train(a1, hp), train(a2, hp)
    assert np.array_equal(a1.Q, a2.Q)
    assert np.array_equal(h1["train_reward"], h2["train_reward"])


def test_training_improves_over_untrained_and_never_exceeds_optimum(mdp):
    hp = HyperParams(n_episodes=5000, eval_every=5000, eval_episodes=50, eps_decay=0.999, seed=0)
    h = train(make_agent("q_learning", hp), hp, mdp=mdp)
    J_star = mdp.policy_value(mdp.value_iteration(GAMMA).argmax(axis=1), GAMMA)
    assert h["eval_J"][0] == pytest.approx(0.0)            # Q = 0 -> argmax = REFUSE partout
    assert h["eval_J"][-1] > 0.0
    assert h["eval_J"][-1] <= J_star + 1e-9


def test_double_q_learning_trains_with_same_trainer(mdp):
    hp = HyperParams(n_episodes=1000, eval_every=500, eval_episodes=20, eps_decay=0.995, seed=0)
    agent = make_agent("double_q_learning", hp)
    h = train(agent, hp, mdp=mdp)
    assert np.all(np.isfinite(h["eval_J"]))
    assert np.any(agent.Q1 != 0) and np.any(agent.Q2 != 0)


def test_make_agent_unknown_raises():
    with pytest.raises(ValueError):
        make_agent("sarsa", HyperParams())
