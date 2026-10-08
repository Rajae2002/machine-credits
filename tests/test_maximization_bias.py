"""Tests de l'expérience contrôlée du biais de maximisation."""
import numpy as np
import pytest

from algorithms.double_q_learning import DoubleQLearningAgent
from algorithms.q_learning import QLearningAgent
from evaluation.maximization_bias import run_experiment, run_once


def test_outputs_have_valid_shape_and_range():
    curves = run_experiment(n_runs=20, n_episodes=30)
    for name in ("q_learning", "double_q_learning"):
        assert curves[name].shape == (30,)
        assert np.all((curves[name] >= 0) & (curves[name] <= 1))


def test_first_choice_is_random_for_both_agents():
    """Q = 0 au départ : égalité totale en A, donc LEFT ~ 50 % au premier épisode."""
    curves = run_experiment(n_runs=2000, n_episodes=1)
    for name in ("q_learning", "double_q_learning"):
        assert curves[name][0] == pytest.approx(0.5, abs=0.05)


@pytest.mark.parametrize("cls", [QLearningAgent, DoubleQLearningAgent])
def test_run_once_is_reproducible(cls):
    assert np.array_equal(run_once(cls, 50, seed=3), run_once(cls, 50, seed=3))
