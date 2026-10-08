"""
maximization_bias.py : expérience CONTRÔLÉE du biais de maximisation.

MDP-jouet classique (Sutton & Barto, exemple 6.7), SANS lien avec le crédit :
c'est une illustration isolée du mécanisme.

          reward 0                   reward 0         reward ~ N(-0.1 ; 1)
    A --RIGHT--> fin         A --LEFT--> B --(n_b actions)--> fin

LEFT est sous-optimal (valeur espérée -0.1 contre 0 pour RIGHT), mais le max sur
les estimations bruitées de B tend à être positif : un agent qui surestime
choisit LEFT trop souvent. Mesure : % d'épisodes où l'agent choisit LEFT en A.
Avec epsilon = 0.1 et 2 actions en A, le minimum atteignable est epsilon / 2 = 5 %.

On réutilise nos classes QLearningAgent et DoubleQLearningAgent (mêmes alpha,
epsilon, seeds pour les deux). L'état A n'a que 2 actions valides : le choix en A
est donc fait ici (epsilon-greedy restreint aux 2 premières actions).
"""
import argparse
import os

import numpy as np

from algorithms.double_q_learning import DoubleQLearningAgent
from algorithms.q_learning import QLearningAgent

A, B = 0, 1
LEFT, RIGHT = 0, 1
OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "results", "experiments")


def _choose(agent, rng, state, n_valid, eps):
    """epsilon-greedy restreint aux n_valid premières actions (égalités au hasard)."""
    if rng.random() < eps:
        return int(rng.integers(n_valid))
    q = agent.q_values(state)[:n_valid]
    return int(rng.choice(np.flatnonzero(q == q.max())))


def run_once(agent_cls, n_episodes, seed, n_b=10, mean_b=-0.1, alpha=0.1, eps=0.1):
    """Un run. Retourne un tableau (n_episodes,) : 1 si LEFT choisi en A, sinon 0."""
    rng = np.random.default_rng(seed)                       # environnement + choix d'action
    agent = agent_cls(n_states=2, n_actions=n_b, alpha=alpha, gamma=1.0, seed=seed + 10_000)
    went_left = np.zeros(n_episodes)
    for ep in range(n_episodes):
        a = _choose(agent, rng, A, 2, eps)
        went_left[ep] = (a == LEFT)
        if a == RIGHT:
            agent.update(A, a, 0.0, A, True)                # terminal : s_next ignoré
        else:
            agent.update(A, a, 0.0, B, False)               # A -> B, reward 0
            b = _choose(agent, rng, B, n_b, eps)
            reward = rng.normal(mean_b, 1.0)
            agent.update(B, b, reward, B, True)             # B -> fin
    return went_left


def run_experiment(n_runs=1000, n_episodes=300, seed0=0, **kwargs):
    """Moyenne sur n_runs runs indépendants (mêmes seeds pour les deux algorithmes)."""
    curves = {}
    for name, cls in (("q_learning", QLearningAgent),
                      ("double_q_learning", DoubleQLearningAgent)):
        curves[name] = np.mean([run_once(cls, n_episodes, seed0 + i, **kwargs)
                                for i in range(n_runs)], axis=0)
    return curves


def main(argv=None):
    p = argparse.ArgumentParser(description="Expérience contrôlée du biais de maximisation")
    p.add_argument("--runs", type=int, default=1000)
    p.add_argument("--episodes", type=int, default=300)
    args = p.parse_args(argv)

    curves = run_experiment(args.runs, args.episodes)
    os.makedirs(OUT_DIR, exist_ok=True)
    np.savez(os.path.join(OUT_DIR, "maximization_bias.npz"), runs=args.runs, **curves)

    n = args.runs
    print(f"MDP-jouet (Sutton & Barto, ex. 6.7) | alpha=0.1, eps=0.1, gamma=1 | {n} runs")
    print("LEFT est sous-optimal ; minimum atteignable en exploration pure = eps/2 = 5 %\n")
    print(f"{'épisode':>8}{'Q-Learning':>24}{'Double Q-Learning':>24}")
    for e in [e for e in (1, 10, 50, 100, 200, 300) if e <= args.episodes]:
        cells = []
        for name in ("q_learning", "double_q_learning"):
            frac = curves[name][e - 1]
            se = np.sqrt(frac * (1 - frac) / n)             # erreur standard entre runs
            cells.append(f"{100 * frac:6.1f} % (± {100 * se:.1f})")
        print(f"{e:>8}{cells[0]:>24}{cells[1]:>24}")
    means = [f"{100 * curves[k].mean():6.1f} %" for k in ("q_learning", "double_q_learning")]
    print(f"{'moyenne':>8}{means[0]:>24}{means[1]:>24}")
    print("(± = erreur standard entre runs ; moyenne = sur tous les épisodes)")


if __name__ == "__main__":
    main()
