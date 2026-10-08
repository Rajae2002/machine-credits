"""
config.py : paramètres du MDP « Machine des crédits » et des expériences.

ATTENTION : toutes les probabilités et récompenses ci-dessous sont
ARTIFICIELLES [SIMULÉ]. Elles servent un environnement pédagogique et ne
représentent PAS des données bancaires réelles.
"""
from dataclasses import dataclass

# ----------------------------------------------------------------------
# Espaces (les indices servent d'encodage : LOW=0, MEDIUM=1, HIGH=2)
# ----------------------------------------------------------------------
RISK_LEVELS = ("LOW", "MEDIUM", "HIGH")
INCOME_LEVELS = ("LOW", "MEDIUM", "HIGH")
DEBT_LEVELS = ("LOW", "MEDIUM", "HIGH")
HISTORY_LEVELS = ("BAD", "GOOD")            # BAD=0, GOOD=1

ACTIONS = ("REFUSE", "ACCEPT", "ACCEPT_REDUCED")   # 0, 1, 2
OUTCOMES = ("NORMAL", "LATE", "DEFAULT")           # 0, 1, 2

N_STATES = 3 * 3 * 3 * 2    # = 54 (hors état terminal)
N_ACTIONS = len(ACTIONS)    # = 3

# ----------------------------------------------------------------------
# Transitions [SIMULÉ]
# ----------------------------------------------------------------------
# (p_normal, p_late, p_default) selon le niveau de risque
BASE_OUTCOME_PROBS = {
    0: (0.80, 0.15, 0.05),   # LOW
    1: (0.60, 0.25, 0.15),   # MEDIUM
    2: (0.30, 0.30, 0.40),   # HIGH
}
# Ajustements de la probabilité de défaut
DEBT_ADJ = {0: -0.02, 1: 0.0, 2: +0.05}
INCOME_ADJ = {0: +0.05, 1: 0.0, 2: -0.02}
HISTORY_ADJ = {0: +0.05, 1: 0.0}            # BAD augmente le risque
P_DEFAULT_MIN, P_DEFAULT_MAX = 0.01, 0.60

# Évolution de l'état
P_RISK_DOWN_ON_NORMAL = 0.20   # risque baisse après un remboursement normal
P_RISK_UP_ON_LATE = 0.30       # risque monte après un retard
P_DEBT_UP_ON_ACCEPT = 0.15     # ACCEPT (crédit complet) peut alourdir la dette
P_END = 1.0 / 12.0             # fin de contrat (maturité aléatoire)
MAX_STEPS = 200                # garde-fou technique

# ----------------------------------------------------------------------
# Récompenses [SIMULÉ] : REWARDS[action] = (normal, late, default)
# ----------------------------------------------------------------------
REWARDS = {
    0: (0.0, 0.0, 0.0),        # REFUSE : fin immédiate, reward 0
    1: (10.0, -5.0, -20.0),    # ACCEPT
    2: (7.0, -2.5, -10.0),     # ACCEPT_REDUCED
}

# ----------------------------------------------------------------------
# Distribution initiale des clients [SIMULÉ]
# ----------------------------------------------------------------------
RISK_INIT_PROBS = (0.4, 0.4, 0.2)   # LOW / MEDIUM / HIGH
HISTORY_GOOD_PROB = 0.7              # income et debt : uniformes

# ----------------------------------------------------------------------
# Hyperparamètres (valeurs PROVISOIRES, à régler en Phase 6)
# ----------------------------------------------------------------------
@dataclass
class HyperParams:
    alpha: float = 0.1          # taux d'apprentissage
    gamma: float = 0.95         # facteur d'actualisation
    eps_start: float = 1.0      # exploration initiale
    eps_end: float = 0.05       # exploration minimale
    eps_decay: float = 0.9995   # décroissance multiplicative par épisode
    n_episodes: int = 50_000
    eval_every: int = 500       # fréquence d'évaluation greedy
    eval_episodes: int = 1_000
    seed: int = 0
