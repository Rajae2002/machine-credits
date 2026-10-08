"""
baselines.py : politiques de référence (sans apprentissage).

Même interface que les agents (select_action, greedy_action, update, policy...),
pour qu'elles passent par exactement le même code d'évaluation.
"""
import numpy as np

from config import N_STATES, N_ACTIONS
from environment.state_space import decode_state

REFUSE, ACCEPT, ACCEPT_REDUCED = 0, 1, 2


class RandomPolicy:
    """Choisit une action uniformément au hasard (borne basse de comparaison)."""

    def __init__(self, n_states=N_STATES, n_actions=N_ACTIONS, seed=None):
        self.n_states = n_states
        self.n_actions = n_actions
        self.rng = np.random.default_rng(seed)

    def select_action(self, state: int, epsilon: float = 0.0) -> int:
        return int(self.rng.integers(self.n_actions))

    def greedy_action(self, state: int) -> int:
        return int(self.rng.integers(self.n_actions))

    def policy_matrix(self) -> np.ndarray:
        """pi(a|s) = 1/3 pour tout (s, a) : utilisé par le calcul exact."""
        return np.full((self.n_states, self.n_actions), 1.0 / self.n_actions)

    def update(self, *args, **kwargs) -> float:   # pas d'apprentissage
        return 0.0


class RiskBasedPolicy:
    """Règle simple selon le niveau de risque du client :
    LOW -> ACCEPT, MEDIUM -> ACCEPT_REDUCED, HIGH -> REFUSE.
    Baseline « métier » : elle ignore income, debt et history."""

    RULE = {0: ACCEPT, 1: ACCEPT_REDUCED, 2: REFUSE}

    def __init__(self, n_states=N_STATES):
        self.n_states = n_states

    def select_action(self, state: int, epsilon: float = 0.0) -> int:
        return self.greedy_action(state)

    def greedy_action(self, state: int) -> int:
        return self.RULE[decode_state(state)[0]]

    def policy(self) -> np.ndarray:
        return np.array([self.greedy_action(s) for s in range(self.n_states)])

    def update(self, *args, **kwargs) -> float:   # pas d'apprentissage
        return 0.0
