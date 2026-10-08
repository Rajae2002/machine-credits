"""
credit_env.py : environnement stochastique « Machine des crédits ».

ATTENTION : toutes les probabilités et récompenses (voir config.py) sont
ARTIFICIELLES [SIMULÉ]. Environnement pédagogique, PAS des données bancaires.

Épisode = la vie d'une relation de crédit avec un client. À chaque période :
  1. la banque choisit une action (REFUSE / ACCEPT / ACCEPT_REDUCED) ;
  2. le client produit un résultat : NORMAL / LATE / DEFAULT ;
  3. l'état évolue (risk, history, debt) ;
  4. le contrat peut se terminer (maturité aléatoire, probabilité P_END).

Fins d'épisode : REFUSE, DEFAULT, maturité, ou MAX_STEPS (garde-fou technique).
"""
import numpy as np

from config import (
    N_ACTIONS, ACTIONS, OUTCOMES, BASE_OUTCOME_PROBS,
    DEBT_ADJ, INCOME_ADJ, HISTORY_ADJ, P_DEFAULT_MIN, P_DEFAULT_MAX,
    P_RISK_DOWN_ON_NORMAL, P_RISK_UP_ON_LATE, P_DEBT_UP_ON_ACCEPT,
    P_END, MAX_STEPS, REWARDS, RISK_INIT_PROBS, HISTORY_GOOD_PROB,
)
from environment.state_space import encode_state, decode_state, describe_state

# Constantes lisibles (mêmes indices que dans config.py)
REFUSE, ACCEPT, ACCEPT_REDUCED = 0, 1, 2
NORMAL, LATE, DEFAULT = 0, 1, 2


class CreditEnv:
    """MDP de décision de crédit. Interface proche de Gymnasium (sans dépendance)."""

    def __init__(self, seed=None):
        self.rng = np.random.default_rng(seed)
        self._cache = {}          # (state, action) -> (branches, cumul des probas)
        self._state_idx = None
        self._done = True         # tant que reset() n'est pas appelé
        self._steps = 0

    # ------------------------------------------------------------------
    # Modèle mathématique : P(s'|s,a) et R(s,a,s')
    # ------------------------------------------------------------------
    def outcome_probs(self, state_idx: int) -> tuple:
        """(p_normal, p_late, p_default) pour un état donné.

        p_D = clip(p_D_base(risk) + ajustements(debt, income, history))
        p_L = p_L_base(risk)
        p_N = 1 - p_D - p_L
        """
        risk, income, debt, history = decode_state(state_idx)
        _, p_late, p_def = BASE_OUTCOME_PROBS[risk]
        p_def = p_def + DEBT_ADJ[debt] + INCOME_ADJ[income] + HISTORY_ADJ[history]
        p_def = min(max(p_def, P_DEFAULT_MIN), P_DEFAULT_MAX)
        p_norm = 1.0 - p_def - p_late
        return p_norm, p_late, p_def

    def calculate_reward(self, action: int, outcome: int) -> float:
        """Reward de la période selon l'action et le résultat du client."""
        return REWARDS[action][outcome]

    def transition_model(self, state_idx: int, action: int) -> list:
        """Liste COMPLÈTE des issues possibles de (s, a).

        Retourne des tuples (proba, next_state_idx | None, reward, outcome) :
          next_state_idx = None  -> l'épisode se termine (état terminal)
          outcome = None         -> pas de résultat client (cas REFUSE)
        La somme des probabilités vaut 1.
        """
        if not 0 <= action < N_ACTIONS:
            raise ValueError(f"action invalide : {action}")
        risk, income, debt, history = decode_state(state_idx)

        if action == REFUSE:                      # fin immédiate, reward 0
            return [(1.0, None, self.calculate_reward(REFUSE, NORMAL), None)]

        branches = []
        probs = self.outcome_probs(state_idx)
        for outcome, p_o in enumerate(probs):
            reward = self.calculate_reward(action, outcome)

            if outcome == DEFAULT:                # défaut : état terminal
                branches.append((p_o, None, reward, outcome))
                continue

            # Maturité du contrat : fin avec probabilité P_END
            branches.append((p_o * P_END, None, reward, outcome))
            p_cont = p_o * (1.0 - P_END)

            # Évolution de history et de risk
            if outcome == NORMAL:
                new_history = 1                   # GOOD
                if risk > 0:
                    risk_dist = {risk: 1 - P_RISK_DOWN_ON_NORMAL,
                                 risk - 1: P_RISK_DOWN_ON_NORMAL}
                else:
                    risk_dist = {risk: 1.0}
            else:                                  # LATE
                new_history = 0                   # BAD
                if risk < 2:
                    risk_dist = {risk: 1 - P_RISK_UP_ON_LATE,
                                 risk + 1: P_RISK_UP_ON_LATE}
                else:
                    risk_dist = {risk: 1.0}

            # Évolution de debt : seul ACCEPT (crédit complet) peut l'alourdir
            if action == ACCEPT and debt < 2:
                debt_dist = {debt: 1 - P_DEBT_UP_ON_ACCEPT,
                             debt + 1: P_DEBT_UP_ON_ACCEPT}
            else:
                debt_dist = {debt: 1.0}

            for new_risk, p_r in risk_dist.items():
                for new_debt, p_d in debt_dist.items():
                    next_idx = encode_state(new_risk, income, new_debt, new_history)
                    branches.append((p_cont * p_r * p_d, next_idx, reward, outcome))
        return branches

    # ------------------------------------------------------------------
    # Simulation
    # ------------------------------------------------------------------
    def transition(self, state_idx: int, action: int) -> tuple:
        """Tire UNE issue de (s, a). Retourne (next_state_idx | None, reward, outcome).
        Ne modifie pas l'état interne de l'environnement."""
        key = (state_idx, action)
        if key not in self._cache:
            branches = self.transition_model(state_idx, action)
            probs = np.array([b[0] for b in branches], dtype=float)
            self._cache[key] = (branches, np.cumsum(probs / probs.sum()))
        branches, cumul = self._cache[key]
        k = int(np.searchsorted(cumul, self.rng.random(), side="right"))
        k = min(k, len(branches) - 1)
        _, next_idx, reward, outcome = branches[k]
        return next_idx, reward, outcome

    def reset(self, seed=None, state=None) -> int:
        """Démarre un épisode. `seed` re-seed le générateur ; `state` impose un
        état initial (utile pour les tests et l'interface). Retourne l'indice d'état."""
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        if state is None:                         # client aléatoire [SIMULÉ]
            risk = int(self.rng.choice(3, p=RISK_INIT_PROBS))
            income = int(self.rng.integers(3))
            debt = int(self.rng.integers(3))
            history = int(self.rng.random() < HISTORY_GOOD_PROB)
            state = encode_state(risk, income, debt, history)
        else:
            decode_state(state)                   # validation
        self._state_idx = int(state)
        self._done = False
        self._steps = 0
        return self._state_idx

    def step(self, action: int) -> tuple:
        """Applique une action. Retourne (next_state, reward, done, info).

        Convention : si done=True, next_state est le dernier état connu (il n'y a
        pas d'indice pour l'état terminal) ; les agents ignorent alors la
        valeur du futur dans leur mise à jour.
        """
        if self._done:
            raise RuntimeError("Épisode terminé ou non démarré : appelez reset().")
        if not 0 <= action < N_ACTIONS:
            raise ValueError(f"action invalide : {action}")

        next_idx, reward, outcome = self.transition(self._state_idx, action)
        self._steps += 1

        if next_idx is None:                      # fin naturelle
            done = True
            if action == REFUSE:
                reason = "refuse"
            elif outcome == DEFAULT:
                reason = "default"
            else:
                reason = "maturity"
        else:
            self._state_idx = next_idx
            done, reason = False, None
            if self._steps >= MAX_STEPS:          # garde-fou technique
                done, reason = True, "max_steps"

        self._done = done
        info = {
            "action": ACTIONS[action],
            "outcome": None if outcome is None else OUTCOMES[outcome],
            "terminal_reason": reason,
            "steps": self._steps,
        }
        return self._state_idx, reward, done, info

    # ------------------------------------------------------------------
    # Accesseurs
    # ------------------------------------------------------------------
    def get_state(self) -> tuple:
        """État courant sous forme (risk, income, debt, history)."""
        return decode_state(self._state_idx)

    def get_state_index(self) -> int:
        return self._state_idx

    def get_available_actions(self) -> list:
        return list(range(N_ACTIONS))

    def is_terminal(self) -> bool:
        return self._done


if __name__ == "__main__":
    # Démo : un épisode avec une politique aléatoire
    env = CreditEnv(seed=42)
    s = env.reset()
    print("Client initial :", describe_state(s))
    policy_rng = np.random.default_rng(0)
    total, done = 0.0, False
    while not done:
        a = int(policy_rng.integers(N_ACTIONS))
        s, r, done, info = env.step(a)
        total += r
        print(f"  {ACTIONS[a]:<15} résultat={str(info['outcome']):<8} "
              f"reward={r:+6.1f} -> {describe_state(s)}")
    print(f"Fin ({info['terminal_reason']}) | reward total = {total:+.1f}")
