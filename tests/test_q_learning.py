"""Tests du Q-Learning tabulaire."""
import numpy as np
import pytest

from config import N_STATES, N_ACTIONS
from algorithms.q_learning import QLearningAgent
from environment.credit_env import CreditEnv


def test_q_table_initialized_to_zero():
    agent = QLearningAgent()
    assert agent.Q.shape == (N_STATES, N_ACTIONS) == (54, 3)
    assert np.all(agent.Q == 0)


def test_update_formula_non_terminal():
    """Q(0,1) = 0 + 0.1 * (10 + 0.95 * 2 - 0) = 1.19"""
    agent = QLearningAgent(alpha=0.1, gamma=0.95)
    agent.Q[1] = [0.0, 2.0, 1.0]                 # max Q(s') = 2
    agent.update(0, 1, 10.0, 1, False)
    assert agent.Q[0, 1] == pytest.approx(1.19)


def test_update_terminal_ignores_next_state():
    """done=True : cible = r, même si Q(s') est grand."""
    agent = QLearningAgent(alpha=0.1, gamma=0.95)
    agent.Q[1] = [100.0, 100.0, 100.0]
    agent.update(0, 1, 10.0, 1, True)
    assert agent.Q[0, 1] == pytest.approx(1.0)   # 0.1 * 10


def test_update_returns_td_error():
    agent = QLearningAgent(alpha=0.5, gamma=0.9)
    agent.Q[0, 2] = 4.0
    delta = agent.update(0, 2, 10.0, 0, True)    # cible = 10
    assert delta == pytest.approx(6.0)
    assert agent.Q[0, 2] == pytest.approx(4.0 + 0.5 * 6.0)


def test_update_modifies_only_one_entry():
    agent = QLearningAgent()
    before = agent.Q.copy()
    agent.update(5, 1, 3.0, 6, False)
    changed = np.argwhere(agent.Q != before)
    assert changed.tolist() == [[5, 1]]


def test_greedy_action_and_policy():
    agent = QLearningAgent()
    agent.Q[3] = [0.0, 5.0, 1.0]
    assert agent.greedy_action(3) == 1
    assert agent.policy()[3] == 1
    assert agent.q_values(3).tolist() == [0.0, 5.0, 1.0]


def test_epsilon_zero_is_greedy():
    agent = QLearningAgent(seed=0)
    agent.Q[3] = [0.0, 5.0, 1.0]
    assert all(agent.select_action(3, 0.0) == 1 for _ in range(100))


def test_epsilon_one_explores_all_actions():
    agent = QLearningAgent(seed=0)
    agent.Q[3] = [0.0, 5.0, 1.0]
    seen = {agent.select_action(3, 1.0) for _ in range(300)}
    assert seen == {0, 1, 2}


def test_ties_are_broken_randomly():
    agent = QLearningAgent(seed=0)               # Q = 0 partout : égalité totale
    seen = {agent.select_action(0, 0.0) for _ in range(300)}
    assert seen == {0, 1, 2}


def test_same_seed_same_actions():
    a1, a2 = QLearningAgent(seed=7), QLearningAgent(seed=7)
    acts1 = [a1.select_action(0, 0.5) for _ in range(50)]
    acts2 = [a2.select_action(0, 0.5) for _ in range(50)]
    assert acts1 == acts2


def test_invalid_hyperparameters():
    with pytest.raises(ValueError):
        QLearningAgent(alpha=0.0)
    with pytest.raises(ValueError):
        QLearningAgent(alpha=1.5)
    with pytest.raises(ValueError):
        QLearningAgent(gamma=1.2)


def test_converges_on_toy_chain():
    """Mini-MDP déterministe : s0 --a1--> s1 --a1--> fin (reward 1). a0 : fin, reward 0.
    Valeurs optimales : Q*(1,1) = 1 et Q*(0,1) = gamma = 0.9."""
    agent = QLearningAgent(n_states=2, n_actions=2, alpha=0.5, gamma=0.9, seed=0)
    for _ in range(500):
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
    assert agent.Q[1, 1] == pytest.approx(1.0, abs=1e-3)
    assert agent.Q[0, 1] == pytest.approx(0.9, abs=1e-3)
    assert agent.greedy_action(0) == 1 and agent.greedy_action(1) == 1


def test_runs_on_credit_env():
    """Test d'intégration : l'agent s'entraîne sur CreditEnv sans erreur."""
    env, agent = CreditEnv(seed=0), QLearningAgent(seed=0)
    for _ in range(500):
        s, done = env.reset(), False
        while not done:
            a = agent.select_action(s, 0.2)
            s_next, r, done, _ = env.step(a)
            agent.update(s, a, r, s_next, done)
            s = s_next
    assert np.all(np.isfinite(agent.Q))
    assert np.any(agent.Q != 0)


def test_save_and_load(tmp_path):
    agent = QLearningAgent()
    agent.Q[2, 1] = 3.5
    path = tmp_path / "q.npy"
    agent.save(path)
    other = QLearningAgent()
    other.load(path)
    assert other.Q[2, 1] == 3.5
