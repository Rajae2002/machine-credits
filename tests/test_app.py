"""Tests de la logique de l'application (sans interface Streamlit)."""
import numpy as np
import pytest

from app import logic
from environment.state_space import encode_state

CFG = {"q_learning": {"alpha": 0.01, "eps_decay": 0.9998},
       "double_q_learning": {"alpha": 0.03, "eps_decay": 0.9998}}


def test_expected_reward_matches_phase3_values():
    s = encode_state(1, 1, 1, 1)                    # MEDIUM / MEDIUM / MEDIUM / GOOD
    assert logic.expected_reward(s, 0) == pytest.approx(0.0)
    assert logic.expected_reward(s, 1) == pytest.approx(1.75)
    assert logic.expected_reward(s, 2) == pytest.approx(2.075)


def test_action_table_structure():
    s = encode_state(0, 1, 1, 1)
    rows = logic.action_table(np.array([0.0, 5.0, 4.0]), np.array([0.0, 6.0, 5.0]), s)
    assert [r["Action"] for r in rows] == ["REFUSE", "ACCEPT", "ACCEPT_REDUCED"]
    assert rows[1]["Q appris"] == 5.0 and rows[1]["Q* exact"] == 6.0


def test_load_selected_config_missing(tmp_path):
    assert logic.load_selected_config(str(tmp_path / "absent.json")) is None


def test_load_q_roundtrip(tmp_path):
    Q = np.arange(162, dtype=float).reshape(54, 3)
    path = logic.run_path("q_learning", CFG, 100, str(tmp_path))
    np.savez(path, Q=Q)
    assert logic.available_seeds("q_learning", CFG, str(tmp_path)) == [100]
    assert np.array_equal(logic.load_q("q_learning", CFG, 100, str(tmp_path)), Q)
    assert logic.available_seeds("double_q_learning", CFG, str(tmp_path)) == []


def test_justify_refuse_when_acceptance_not_better():
    lines = logic.justify(np.array([0.0, -3.0, -1.0]), np.array([0.0, -6.0, -2.0]))
    text = " ".join(lines)
    assert "REFUSE" in text and "coïncide" in text


def test_justify_reports_difference_from_optimal():
    lines = logic.justify(np.array([0.0, 8.0, 5.0]), np.array([0.0, 9.0, 10.0]))
    text = " ".join(lines)
    assert "diffère" in text and "perte de valeur exacte = 1.00" in text


def test_simulate_contract_total_and_termination():
    policy = np.ones(54, dtype=int)                 # toujours ACCEPT
    rows, total = logic.simulate_contract(policy, encode_state(0, 1, 1, 1), seed=0)
    assert len(rows) >= 1
    assert total == pytest.approx(sum(r["Reward"] for r in rows))


def test_simulate_contract_is_reproducible():
    policy = np.ones(54, dtype=int)
    s = encode_state(1, 1, 1, 1)
    assert logic.simulate_contract(policy, s, 5) == logic.simulate_contract(policy, s, 5)
