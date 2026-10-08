"""Tests du protocole expérimental (Phase 7)."""
import os

import numpy as np
import pytest

from evaluation import experiments as ex
from evaluation.exact import ExactMDP

KW = dict(eval_every=100, eval_episodes=10, final_eval_episodes=50)


@pytest.fixture(scope="module")
def mdp():
    return ExactMDP()


def _fake_run(dirpath, algo, alpha, decay, seed, eval_J):
    np.savez(os.path.join(dirpath, f"{algo}_a{alpha}_d{decay}_s{seed}.npz"),
             algo=algo, alpha=alpha, eps_decay=decay, seed=seed, eval_J=np.array(eval_J))


@pytest.mark.parametrize("algo", ["q_learning", "double_q_learning"])
def test_run_one_saves_and_resumes(tmp_path, mdp, algo):
    path, ran = ex.run_one(algo, 0.1, 0.99, 0, 300, mdp, str(tmp_path), **KW)
    assert ran and os.path.exists(path)
    with np.load(path) as d:
        assert d["policy"].shape == (54,)
        assert list(d["eval_episode"]) == [0, 100, 200, 300]
        assert float(d["final_J"]) == pytest.approx(float(d["eval_J"][-1]))
        assert d["final_by_risk"].shape == (3,)
    _, ran_again = ex.run_one(algo, 0.1, 0.99, 0, 300, mdp, str(tmp_path), **KW)
    assert not ran_again                      # reprise : déjà calculé


def test_run_one_recomputes_when_episodes_change(tmp_path, mdp):
    ex.run_one("q_learning", 0.1, 0.99, 0, 200, mdp, str(tmp_path), **KW)
    _, ran = ex.run_one("q_learning", 0.1, 0.99, 0, 300, mdp, str(tmp_path), **KW)
    assert ran                                # résultat périmé : recalculé


def test_run_one_is_reproducible(tmp_path, mdp):
    p1, _ = ex.run_one("double_q_learning", 0.1, 0.99, 5, 300, mdp, str(tmp_path / "a"), **KW)
    p2, _ = ex.run_one("double_q_learning", 0.1, 0.99, 5, 300, mdp, str(tmp_path / "b"), **KW)
    with np.load(p1) as a, np.load(p2) as b:
        assert np.array_equal(a["Q"], b["Q"])
        assert np.array_equal(a["eval_J"], b["eval_J"])


def test_final_J_never_exceeds_optimum(tmp_path, mdp):
    J_star = ex.reference_values(mdp)["J_star"]
    path, _ = ex.run_one("q_learning", 0.1, 0.99, 1, 500, mdp, str(tmp_path), **KW)
    with np.load(path) as d:
        assert float(d["final_J"]) <= J_star + 1e-9
        assert np.all(d["eval_J"] <= J_star + 1e-9)


def test_select_best_picks_best_config_per_algo(tmp_path):
    d = str(tmp_path)
    for seed in (0, 1):
        _fake_run(d, "q_learning", 0.1, 0.9995, seed, [0, 5, 5, 5, 5, 5])
        _fake_run(d, "q_learning", 0.03, 0.9995, seed, [0, 8, 8, 8, 8, 8])
        _fake_run(d, "double_q_learning", 0.1, 0.9995, seed, [0, 9, 9, 9, 9, 9])
        _fake_run(d, "double_q_learning", 0.03, 0.9995, seed, [0, 4, 4, 4, 4, 4])
    best, _ = ex.select_best(d, k=5)
    assert best["q_learning"]["alpha"] == 0.03
    assert best["double_q_learning"]["alpha"] == 0.1
    assert best["q_learning"]["n_seeds"] == 2


def test_select_best_uses_last_checkpoints_only(tmp_path):
    d = str(tmp_path)
    _fake_run(d, "q_learning", 0.1, 0.9995, 0, [10, 10, 10, 1, 1, 1])   # bon au début, mauvais à la fin
    _fake_run(d, "q_learning", 0.03, 0.9995, 0, [0, 0, 0, 5, 5, 5])
    best, _ = ex.select_best(d, k=3)
    assert best["q_learning"]["alpha"] == 0.03


def test_seed_sets_are_disjoint():
    assert set(ex.GRID_SEEDS).isdisjoint(ex.FINAL_SEEDS)
