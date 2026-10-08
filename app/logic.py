"""
logic.py : logique de l'application (sans Streamlit, donc testable avec pytest).

Environnement SIMULÉ [SIMULÉ] : les décisions affichées par l'application ne
concernent PAS de vrais clients et ne constituent pas un système bancaire.
"""
import json
import os

import numpy as np

from config import ACTIONS
from environment.credit_env import CreditEnv, REFUSE
from environment.state_space import describe_state
from evaluation import experiments as ex

FINAL_DIR = os.path.join(ex.OUT_ROOT, "final")


# ----------------------------------------------------------------------
# Chargement des résultats d'entraînement
# ----------------------------------------------------------------------
def load_selected_config(path=None):
    """Configuration retenue par l'étape grid (None si absente)."""
    path = path or os.path.join(ex.OUT_ROOT, "selected_config.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def run_path(algo, cfg, seed, final_dir=FINAL_DIR):
    name = ex.run_name(algo, cfg[algo]["alpha"], cfg[algo]["eps_decay"], seed)
    return os.path.join(final_dir, name + ".npz")


def available_seeds(algo, cfg, final_dir=FINAL_DIR):
    """Seeds finales dont le résultat est disponible sur disque."""
    return [s for s in ex.FINAL_SEEDS if os.path.exists(run_path(algo, cfg, s, final_dir))]


def load_q(algo, cfg, seed, final_dir=FINAL_DIR):
    """Q-table apprise (54 x 3). Pour Double Q-Learning : (Q1 + Q2) / 2."""
    with np.load(run_path(algo, cfg, seed, final_dir)) as d:
        return d["Q"].copy()


# ----------------------------------------------------------------------
# Décision et explication
# ----------------------------------------------------------------------
def expected_reward(state, action, env=None):
    """Reward immédiat espéré d'UNE période, calculé exactement sur le modèle."""
    env = env or CreditEnv(seed=0)
    return float(sum(p * r for p, _, r, _ in env.transition_model(state, action)))


def action_table(q_learned, q_star, state, env=None):
    """Une ligne par action : Q appris, Q* exact, reward immédiat espéré."""
    env = env or CreditEnv(seed=0)
    return [{
        "Action": ACTIONS[a],
        "Q appris": round(float(q_learned[a]), 3),
        "Q* exact": round(float(q_star[a]), 3),
        "Reward espéré (1 période)": round(expected_reward(state, a, env), 3),
    } for a in range(len(ACTIONS))]


def justify(q_learned, q_star, tol=1.0):
    """Justification en langage clair de l'action recommandée (liste de phrases)."""
    best = int(np.argmax(q_learned))
    a_star = int(np.argmax(q_star))
    order = np.argsort(q_learned)[::-1]
    margin = float(q_learned[order[0]] - q_learned[order[1]])

    lines = [f"{ACTIONS[best]} a la valeur Q apprise la plus élevée ({q_learned[best]:+.2f})."]
    if best == REFUSE:
        lines.append("Les actions d'acceptation n'ont pas une valeur apprise supérieure à 0 : "
                     "l'agent estime qu'elles rapportent moins que refuser (REFUSE vaut toujours 0).")
    else:
        lines.append("La valeur espérée d'un crédit accepté (profit à long terme actualisé) "
                     "est estimée positive et supérieure à celle de REFUSE (0).")
    if margin < tol:
        lines.append(f"L'écart avec la deuxième meilleure action est faible ({margin:.2f}) : "
                     "quasi-égalité, la décision est peu sensible.")
    if best == a_star:
        lines.append("Cette décision coïncide avec la politique optimale exacte π* du modèle simulé.")
    else:
        loss = float(q_star[a_star] - q_star[best])
        lines.append(f"Elle diffère de la politique optimale exacte π* ({ACTIONS[a_star]}) : "
                     f"perte de valeur exacte = {loss:.2f}.")
    return lines


# ----------------------------------------------------------------------
# Simulation d'un contrat
# ----------------------------------------------------------------------
def simulate_contract(policy, state, seed):
    """Simule UN contrat à partir de `state` en suivant la politique `policy`
    (tableau d'actions, une par état). Retourne (lignes, reward total)."""
    env = CreditEnv(seed=seed)
    s = env.reset(state=state)
    rows, total, done, step = [], 0.0, False, 0
    while not done:
        a = int(policy[s])
        before = describe_state(s)
        s_next, r, done, info = env.step(a)
        step += 1
        total += r
        rows.append({
            "Période": step,
            "État": before,
            "Action": ACTIONS[a],
            "Résultat": info["outcome"] or "(pas de crédit)",
            "Reward": r,
        })
        s = s_next
    return rows, float(total)
