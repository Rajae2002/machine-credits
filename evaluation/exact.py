"""
exact.py : solution EXACTE du MDP par programmation dynamique.

Utilise le modèle de CreditEnv.transition_model (source unique de vérité, D10) :
    R[s,a]    = Σ p · r                  (reward immédiat espéré)
    T[s,a,s'] = Σ p                      (masse terminale exclue)
Donne :
    - Q* et pi* (value iteration)
    - V^pi et J(pi) = Σ_s mu(s) V^pi(s) pour toute politique pi

ATTENTION : calcul sur le modèle SIMULÉ [SIMULÉ], pas sur des données réelles.
Le garde-fou MAX_STEPS est ignoré (probabilité ~3e-8, négligeable).
"""
import numpy as np

from config import N_STATES, N_ACTIONS, RISK_INIT_PROBS, HISTORY_GOOD_PROB
from environment.credit_env import CreditEnv
from environment.state_space import decode_state


def initial_distribution() -> np.ndarray:
    """mu(s) : distribution des clients à l'arrivée (identique à CreditEnv.reset)."""
    mu = np.zeros(N_STATES)
    for s in range(N_STATES):
        risk, income, debt, history = decode_state(s)
        p = RISK_INIT_PROBS[risk] * (1.0 / 3.0) * (1.0 / 3.0)
        p *= HISTORY_GOOD_PROB if history == 1 else 1.0 - HISTORY_GOOD_PROB
        mu[s] = p
    return mu


class ExactMDP:
    """Modèle exact (R, T, mu) extrait de l'environnement."""

    def __init__(self, env=None):
        env = env or CreditEnv(seed=0)
        self.R = np.zeros((N_STATES, N_ACTIONS))
        self.T = np.zeros((N_STATES, N_ACTIONS, N_STATES))
        for s in range(N_STATES):
            for a in range(N_ACTIONS):
                for p, nxt, r, _ in env.transition_model(s, a):
                    self.R[s, a] += p * r
                    if nxt is not None:           # nxt=None : état terminal (masse perdue)
                        self.T[s, a, nxt] += p
        self.mu = initial_distribution()

    def value_iteration(self, gamma: float, tol: float = 1e-10, max_iter: int = 100_000):
        """Q* : point fixe de Q = R + gamma * T @ max_a Q (équation de Bellman optimale)."""
        Q = np.zeros((N_STATES, N_ACTIONS))
        for _ in range(max_iter):
            V = Q.max(axis=1)
            Q_new = self.R + gamma * (self.T @ V)
            converged = np.max(np.abs(Q_new - Q)) < tol
            Q = Q_new
            if converged:
                break
        return Q

    def policy_evaluation(self, policy, gamma: float) -> np.ndarray:
        """V^pi exacte, par résolution du système linéaire (I - gamma T_pi) V = R_pi.
        `policy` : tableau d'actions (n_states,) ou matrice pi(a|s) (n_states, n_actions)."""
        policy = np.asarray(policy)
        if policy.ndim == 1:
            pi = np.zeros((N_STATES, N_ACTIONS))
            pi[np.arange(N_STATES), policy.astype(int)] = 1.0
        else:
            pi = policy
        R_pi = (pi * self.R).sum(axis=1)
        T_pi = np.einsum("sa,san->sn", pi, self.T)
        return np.linalg.solve(np.eye(N_STATES) - gamma * T_pi, R_pi)

    def policy_value(self, policy, gamma: float) -> float:
        """J(pi) = Σ_s mu(s) V^pi(s) : valeur espérée pour un client tiré au hasard."""
        return float(self.mu @ self.policy_evaluation(policy, gamma))


if __name__ == "__main__":
    from config import HyperParams, ACTIONS, RISK_LEVELS
    from algorithms.baselines import RiskBasedPolicy, RandomPolicy

    gamma = HyperParams().gamma
    mdp = ExactMDP()
    Q_star = mdp.value_iteration(gamma)
    pi_star = Q_star.argmax(axis=1)
    risk_pol = RiskBasedPolicy().policy()

    print(f"gamma = {gamma}   [SIMULÉ : modèle pédagogique]")
    print(f"J*  politique optimale     = {mdp.policy_value(pi_star, gamma):8.3f}")
    print(f"J   baseline par risque    = {mdp.policy_value(risk_pol, gamma):8.3f}")
    print(f"J   baseline aléatoire     = {mdp.policy_value(RandomPolicy().policy_matrix(), gamma):8.3f}")
    print(f"J   toujours refuser       = {mdp.policy_value(np.zeros(N_STATES, dtype=int), gamma):8.3f}")

    print("\nActions optimales par niveau de risque (18 états par niveau) :")
    for risk, name in enumerate(RISK_LEVELS):
        counts = {}
        for s in range(N_STATES):
            if decode_state(s)[0] == risk:
                counts[ACTIONS[pi_star[s]]] = counts.get(ACTIONS[pi_star[s]], 0) + 1
        print(f"  {name:<7}{counts}")

    diff = [s for s in range(N_STATES) if pi_star[s] != risk_pol[s]]
    regrets = [Q_star[s, pi_star[s]] - Q_star[s, risk_pol[s]] for s in diff]
    print(f"\nÉtats où pi* diffère de la baseline par risque : {len(diff)} / {N_STATES}")
    if diff:
        print(f"Perte maximale de Q en suivant la baseline sur ces états : {max(regrets):.3f}")
