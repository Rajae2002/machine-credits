"""
double_q_learning.py : agent Double Q-Learning tabulaire (van Hasselt, 2010).

Deux tables Q1 et Q2. À chaque mise à jour, on tire au hasard (p = 1/2) la table
à mettre à jour. Exemple si c'est Q1 :
    a*  = argmax_a' Q1(s', a')                  <- Q1 CHOISIT l'action
    cible = r + gamma * Q2(s', a*)              <- Q2 ÉVALUE cette action
    Q1(s,a) <- Q1(s,a) + alpha * (cible - Q1(s,a))
Si done=True : cible = r.

Séparer le choix et l'évaluation réduit le biais de maximisation du Q-Learning.
L'agent expose la même interface que QLearningAgent.
"""
import numpy as np

from config import N_STATES, N_ACTIONS


class DoubleQLearningAgent:
    """Double Q-Learning tabulaire avec politique epsilon-greedy."""

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
        self.Q1 = np.zeros((n_states, n_actions))
        self.Q2 = np.zeros((n_states, n_actions))
        self.rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------
    @property
    def Q(self) -> np.ndarray:
        """Estimation combinée (Q1 + Q2) / 2, utilisée pour décider et afficher."""
        return (self.Q1 + self.Q2) / 2.0

    def _argmax_random(self, values: np.ndarray) -> int:
        """argmax avec départage aléatoire des égalités."""
        best = np.flatnonzero(values == values.max())
        return int(self.rng.choice(best))

    # ------------------------------------------------------------------
    # Choix d'action
    # ------------------------------------------------------------------
    def select_action(self, state: int, epsilon: float) -> int:
        """epsilon-greedy sur Q1 + Q2 (utilisée pendant l'entraînement)."""
        if self.rng.random() < epsilon:
            return int(self.rng.integers(self.n_actions))
        return self._argmax_random(self.Q1[state] + self.Q2[state])

    def greedy_action(self, state: int) -> int:
        """Politique apprise, sans exploration : argmax de Q1 + Q2 (déterministe)."""
        return int(np.argmax(self.Q1[state] + self.Q2[state]))

    def q_values(self, state: int) -> np.ndarray:
        """Valeurs Q combinées pour un état (copie)."""
        return (self.Q1[state] + self.Q2[state]) / 2.0

    def policy(self) -> np.ndarray:
        """Politique gloutonne complète : une action par état."""
        return np.argmax(self.Q1 + self.Q2, axis=1)

    # ------------------------------------------------------------------
    # Apprentissage
    # ------------------------------------------------------------------
    def update(self, s: int, a: int, r: float, s_next: int, done: bool) -> float:
        """Une mise à jour Double Q-Learning. Retourne l'erreur TD de la table mise à jour.

        Si done=True, s_next est IGNORÉ (pas de futur après l'état terminal).
        """
        # Pile ou face : quelle table est mise à jour ?
        if self.rng.random() < 0.5:
            q_update, q_eval = self.Q1, self.Q2
        else:
            q_update, q_eval = self.Q2, self.Q1

        if done:
            target = r
        else:
            a_star = self._argmax_random(q_update[s_next])      # la table mise à jour CHOISIT
            target = r + self.gamma * q_eval[s_next, a_star]    # l'autre table ÉVALUE

        td_error = target - q_update[s, a]
        q_update[s, a] += self.alpha * td_error
        return float(td_error)

    # ------------------------------------------------------------------
    # Sauvegarde / chargement (les deux tables, forme (2, n_states, n_actions))
    # ------------------------------------------------------------------
    def save(self, path: str) -> None:
        np.save(path, np.stack([self.Q1, self.Q2]))

    def load(self, path: str) -> None:
        tables = np.load(path)
        if tables.shape != (2, self.n_states, self.n_actions):
            raise ValueError(f"forme incompatible : {tables.shape}")
        self.Q1, self.Q2 = tables[0].copy(), tables[1].copy()
