"""
q_learning.py : agent Q-Learning tabulaire (off-policy).

Règle de mise à jour :
    Q(s,a) <- Q(s,a) + alpha * [ r + gamma * max_a' Q(s',a') - Q(s,a) ]
Si l'épisode est terminé (done=True) : la cible vaut simplement r.

L'agent ne connaît PAS l'environnement : il manipule seulement des indices
d'états et d'actions.
"""
import numpy as np

from config import N_STATES, N_ACTIONS


class QLearningAgent:
    """Q-Learning tabulaire avec politique epsilon-greedy."""

    def __init__(self, n_states=N_STATES, n_actions=N_ACTIONS,
                 alpha=0.1, gamma=0.95, seed=None):
        if not 0.0 < alpha <= 1.0:
            raise ValueError("alpha doit être dans ]0, 1]")
        if not 0.0 <= gamma <= 1.0:
            raise ValueError("gamma doit être dans [0, 1]")
        self.n_states = n_states
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.Q = np.zeros((n_states, n_actions))     # table Q initialisée à 0
        self.rng = np.random.default_rng(seed)       # générateur propre à l'agent

    # ------------------------------------------------------------------
    # Choix d'action
    # ------------------------------------------------------------------
    def select_action(self, state: int, epsilon: float) -> int:
        """Politique epsilon-greedy (utilisée PENDANT l'entraînement).

        - probabilité epsilon     : exploration (action aléatoire)
        - probabilité 1 - epsilon : exploitation (meilleure action connue)
        Les égalités sont départagées au hasard (sinon, avec Q = 0 au départ,
        l'agent choisirait toujours l'action 0).
        """
        if self.rng.random() < epsilon:
            return int(self.rng.integers(self.n_actions))
        values = self.Q[state]
        best = np.flatnonzero(values == values.max())
        return int(self.rng.choice(best))

    def greedy_action(self, state: int) -> int:
        """Politique apprise, sans exploration (évaluation, interface).
        Déterministe : en cas d'égalité, la plus petite action est choisie."""
        return int(np.argmax(self.Q[state]))

    def q_values(self, state: int) -> np.ndarray:
        """Valeurs Q de toutes les actions pour un état (copie)."""
        return self.Q[state].copy()

    def policy(self) -> np.ndarray:
        """Politique gloutonne complète : une action par état."""
        return np.argmax(self.Q, axis=1)

    # ------------------------------------------------------------------
    # Apprentissage
    # ------------------------------------------------------------------
    def update(self, s: int, a: int, r: float, s_next: int, done: bool) -> float:
        """Une mise à jour Q-Learning. Retourne l'erreur TD.

        Si done=True, s_next est IGNORÉ (pas de futur après l'état terminal).
        """
        if done:
            target = r
        else:
            target = r + self.gamma * np.max(self.Q[s_next])
        td_error = target - self.Q[s, a]
        self.Q[s, a] += self.alpha * td_error
        return float(td_error)

    # ------------------------------------------------------------------
    # Sauvegarde / chargement de la Q-table
    # ------------------------------------------------------------------
    def save(self, path: str) -> None:
        np.save(path, self.Q)

    def load(self, path: str) -> None:
        Q = np.load(path)
        if Q.shape != self.Q.shape:
            raise ValueError(f"forme incompatible : {Q.shape} != {self.Q.shape}")
        self.Q = Q
