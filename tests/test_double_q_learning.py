"""Tests du Double Q-Learning tabulaire."""
import numpy as np
import pytest

from config import N_STATES, N_ACTIONS
from algorithms.double_q_learning import DoubleQLearningAgent
from environment.credit_env import CreditEnv


def test_tables_initialized_to_zero():
    agent = DoubleQLearningAgent()
    assert agent.Q1.shape == agent.Q2.shape == (N_STATES, N_ACTIONS) == (54, 3)
    assert np.all(agent.Q1 == 0) and np.all(agent.Q2 == 0)
    assert agent.Q.shape == (54, 3)


def test_update_uses_other_table_for_evaluation():
    """Le point clé : la table mise à jour CHOISIT a*, l'autre ÉVALUE a*."""
    seen = set()
    for seed in range(20):
        agent = DoubleQLearningAgent(alpha=0.1, gamma=0.95, seed=seed)
        agent.Q1[1] = [0.0, 2.0, 1.0]     # argmax Q1(s') = 1
        agent.Q2[1] = [0.0, 0.5, 9.0]     # argmax Q2(s') = 2, max Q2 = 9
        agent.update(0, 1, 10.0, 1, False)
        if agent.Q1[0, 1] != 0.0:          # Q1 mise à jour
            # a* = 1 (choisi par Q1), évalué par Q2 : 10 + 0.95 * 0.5 = 10.475
            assert agent.Q1[0, 1] == pytest.approx(0.1 * 10.475)
            assert agent.Q2[0, 1] == 0.0
            seen.add(1)
        else:                              # Q2 mise à jour
            # a* = 2 (choisi par Q2), évalué par Q1 : 10 + 0.95 * 1.0 = 10.95
            assert agent.Q2[0, 1] == pytest.approx(0.1 * 10.95)
            seen.add(2)
    assert seen == {1, 2}                  # les deux branches ont été observées


def test_update_modifies_exactly_one_entry():
    agent = DoubleQLearningAgent(seed=0)
    b1, b2 = agent.Q1.copy(), agent.Q2.copy()
    agent.update(5, 1, 3.0, 6, False)
    changed1 = np.argwhere(agent.Q1 != b1).tolist()
    changed2 = np.argwhere(agent.Q2 != b2).tolist()
    assert len(changed1) + len(changed2) == 1
    assert (changed1 or changed2) == [[5, 1]]


def test_update_terminal_ignores_next_state():
    agent = DoubleQLearningAgent(alpha=0.1, gamma=0.95, seed=0)
    agent.Q1[1] = [100.0, 100.0, 100.0]
    agent.Q2[1] = [100.0, 100.0, 100.0]
    agent.update(0, 1, 10.0, 1, True)
    values = sorted([agent.Q1[0, 1], agent.Q2[0, 1]])
    assert values[0] == 0.0
    assert values[1] == pytest.approx(1.0)  # 0.1 * 10


def test_update_returns_td_error():
    agent = DoubleQLearningAgent(alpha=0.5, gamma=0.9, seed=0)
    delta = agent.update(0, 2, 10.0, 0, True)   # tables à 0 : erreur TD = r
    assert delta == pytest.approx(10.0)


def test_both_tables_are_updated_about_half_the_time():
    agent = DoubleQLearningAgent(seed=0)
    n, n1 = 2000, 0
    for i in range(n):
        before = agent.Q1[0, 1]
        agent.update(0, 1, float(i + 1), 0, True)   # reward croissant : Q change toujours
        n1 += agent.Q1[0, 1] != before
    assert 0.45 < n1 / n < 0.55


def test_q_values_greedy_and_policy_use_both_tables():
    agent = DoubleQLearningAgent()
    agent.Q1[3] = [0.0, 4.0, 2.0]
    agent.Q2[3] = [0.0, 2.0, 0.0]
    assert agent.q_values(3).tolist() == [0.0, 3.0, 1.0]
    assert agent.greedy_action(3) == 1
    assert agent.policy()[3] == 1
    assert agent.Q[3].tolist() == [0.0, 3.0, 1.0]


def test_epsilon_zero_is_greedy():
    agent = DoubleQLearningAgent(seed=0)
    agent.Q1[3] = [0.0, 5.0, 1.0]
    agent.Q2[3] = [0.0, 3.0, 1.0]
    assert all(agent.select_action(3, 0.0) == 1 for _ in range(100))


def test_epsilon_one_explores_all_actions():
    agent = DoubleQLearningAgent(seed=0)
    agent.Q1[3] = [0.0, 5.0, 1.0]
    seen = {agent.select_action(3, 1.0) for _ in range(300)}
    assert seen == {0, 1, 2}


def test_ties_are_broken_randomly():
    agent = DoubleQLearningAgent(seed=0)
    seen = {agent.select_action(0, 0.0) for _ in range(300)}
    assert seen == {0, 1, 2}


def _train(seed, episodes=200):
    env, agent = CreditEnv(seed=seed), DoubleQLearningAgent(seed=seed)
    for _ in range(episodes):
        s, done = env.reset(), False
        while not done:
            a = agent.select_action(s, 0.3)
            s2, r, done, _ = env.step(a)
            agent.update(s, a, r, s2, done)
            s = s2
    return agent


def test_same_seed_same_training():
    a, b = _train(4), _train(4)
    assert np.array_equal(a.Q1, b.Q1) and np.array_equal(a.Q2, b.Q2)


def test_invalid_hyperparameters():
    with pytest.raises(ValueError):
        DoubleQLearningAgent(alpha=0.0)
    with pytest.raises(ValueError):
        DoubleQLearningAgent(alpha=1.5)
    with pytest.raises(ValueError):
        DoubleQLearningAgent(gamma=1.2)


def test_converges_on_toy_chain():
    """Même mini-MDP qu'en Phase 4 : Q*(1,1) = 1 et Q*(0,1) = gamma = 0.9."""
    agent = DoubleQLearningAgent(n_states=2, n_actions=2, alpha=0.5, gamma=0.9, seed=0)
    for _ in range(3000):
        s, done = 0, False
        while not done:
            a = agent.select_action(s, 0.3)
            if a == 0:
                r, s_next, done = 0.0, s, True
            elif s == 0:
                r, s_next, done = 0.0, 1, False
            else:
                r, s_next, done = 1.0, s, True
            agent.update(s, a, r, s_next, done)
            s = s_next
    assert agent.Q[1, 1] == pytest.approx(1.0, abs=1e-2)
    assert agent.Q[0, 1] == pytest.approx(0.9, abs=1e-2)
    assert agent.greedy_action(0) == 1 and agent.greedy_action(1) == 1


def test_runs_on_credit_env():
    agent = _train(0, episodes=500)
    assert np.all(np.isfinite(agent.Q1)) and np.all(np.isfinite(agent.Q2))
    assert np.any(agent.Q1 != 0) and np.any(agent.Q2 != 0)


def test_save_and_load(tmp_path):
    agent = DoubleQLearningAgent()
    agent.Q1[2, 1], agent.Q2[2, 1] = 3.5, 1.5
    path = tmp_path / "dq.npy"
    agent.save(path)
    other = DoubleQLearningAgent()
    other.load(path)
    assert other.Q1[2, 1] == 3.5 and other.Q2[2, 1] == 1.5
