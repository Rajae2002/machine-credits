"""Tests de l'environnement : espace d'états, probabilités, rewards, épisodes."""
import pytest

from config import N_STATES, N_ACTIONS, REWARDS, MAX_STEPS
from environment.state_space import encode_state, decode_state
from environment.credit_env import (
    CreditEnv, REFUSE, ACCEPT, ACCEPT_REDUCED, NORMAL, LATE, DEFAULT,
)


@pytest.fixture
def env():
    return CreditEnv(seed=0)


# ---------------- Espace d'états ----------------
def test_state_encoding_roundtrip_and_unique():
    seen = set()
    for idx in range(N_STATES):
        assert encode_state(*decode_state(idx)) == idx
        seen.add(decode_state(idx))
    assert len(seen) == N_STATES == 54


def test_invalid_state_raises():
    with pytest.raises(ValueError):
        decode_state(N_STATES)
    with pytest.raises(ValueError):
        encode_state(3, 0, 0, 0)


# ---------------- Probabilités ----------------
def test_outcome_probs_valid_for_all_states(env):
    for s in range(N_STATES):
        probs = env.outcome_probs(s)
        assert all(p >= 0 for p in probs)
        assert sum(probs) == pytest.approx(1.0)


def test_transition_model_sums_to_one(env):
    """Σ P(s'|s,a) = 1 pour les 162 couples (s, a)."""
    for s in range(N_STATES):
        for a in range(N_ACTIONS):
            branches = env.transition_model(s, a)
            assert sum(b[0] for b in branches) == pytest.approx(1.0)
            assert all(b[0] >= 0 for b in branches)


def test_simulation_matches_model(env):
    """Loi des grands nombres : fréquence simulée de défaut ~ p_default du modèle."""
    s = encode_state(2, 0, 2, 0)                    # profil très défavorable
    p_default = env.outcome_probs(s)[2]
    n = 20_000
    defaults = sum(env.transition(s, ACCEPT)[2] == DEFAULT for _ in range(n))
    assert defaults / n == pytest.approx(p_default, abs=0.02)


# ---------------- Structure des transitions ----------------
def test_default_and_refuse_are_terminal(env):
    for s in range(N_STATES):
        for p, nxt, r, outcome in env.transition_model(s, REFUSE):
            assert nxt is None
        for p, nxt, r, outcome in env.transition_model(s, ACCEPT):
            if outcome == DEFAULT:
                assert nxt is None


def test_state_dynamics_rules(env):
    for s in range(N_STATES):
        risk, income, debt, history = decode_state(s)
        for a in (ACCEPT, ACCEPT_REDUCED):
            for p, nxt, r, outcome in env.transition_model(s, a):
                if nxt is None:
                    continue
                n_risk, n_income, n_debt, n_hist = decode_state(nxt)
                assert n_income == income                       # revenu stable
                assert n_hist == (1 if outcome == NORMAL else 0)
                if outcome == NORMAL:
                    assert n_risk in (risk, max(risk - 1, 0))
                else:
                    assert n_risk in (risk, min(risk + 1, 2))
                if a == ACCEPT_REDUCED:
                    assert n_debt == debt                       # dette inchangée
                else:
                    assert n_debt in (debt, min(debt + 1, 2))


# ---------------- Rewards ----------------
def test_calculate_reward_matches_config(env):
    for a in range(N_ACTIONS):
        for o in (NORMAL, LATE, DEFAULT):
            assert env.calculate_reward(a, o) == REWARDS[a][o]


def test_one_period_expected_reward_medium_profile(env):
    """Profil MEDIUM/MEDIUM/MEDIUM/GOOD : sans ajustement de p_D (voir Phase 1)."""
    s = encode_state(1, 1, 1, 1)
    def expected(a):
        return sum(p * r for p, _, r, _ in env.transition_model(s, a))
    assert expected(REFUSE) == pytest.approx(0.0)
    assert expected(ACCEPT) == pytest.approx(1.75)
    assert expected(ACCEPT_REDUCED) == pytest.approx(2.075)


# ---------------- reset / step ----------------
def test_reset_returns_valid_state(env):
    for _ in range(100):
        s = env.reset()
        assert 0 <= s < N_STATES
        assert not env.is_terminal()


def test_reset_with_seed_is_reproducible(env):
    assert env.reset(seed=3) == env.reset(seed=3)


def test_refuse_ends_episode_with_zero_reward(env):
    env.reset(state=encode_state(2, 1, 1, 1))
    s, r, done, info = env.step(REFUSE)
    assert r == 0.0 and done and info["terminal_reason"] == "refuse"
    assert env.is_terminal()


def test_step_after_done_raises(env):
    env.reset()
    env.step(REFUSE)
    with pytest.raises(RuntimeError):
        env.step(ACCEPT)


def test_invalid_action_raises(env):
    env.reset()
    with pytest.raises(ValueError):
        env.step(99)


def test_random_episodes_always_terminate(env):
    import numpy as np
    rng = np.random.default_rng(1)
    for _ in range(300):
        env.reset()
        steps, done = 0, False
        while not done:
            _, _, done, _ = env.step(int(rng.integers(N_ACTIONS)))
            steps += 1
            assert steps <= MAX_STEPS


def test_reproducibility_same_seed_same_trajectory():
    import numpy as np
    actions = np.random.default_rng(123).integers(0, N_ACTIONS, size=50)

    def run(seed):
        e = CreditEnv(seed=seed)
        traj = [e.reset()]
        for a in actions:
            s, r, done, _ = e.step(int(a))
            traj.append((s, r, done))
            if done:
                break
        return traj

    assert run(5) == run(5)
