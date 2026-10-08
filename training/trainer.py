"""
trainer.py : boucle d'entraînement COMMUNE à Q-Learning et Double Q-Learning.

Pendant l'entraînement l'agent explore (epsilon-greedy). Tous les `eval_every`
épisodes, on évalue sa politique GREEDY :
  - exactement : J(pi), sans bruit (evaluation/exact.py)
  - par simulation : sur les MÊMES clients (seeds fixes) pour tous les agents

Environnement d'évaluation séparé de celui d'entraînement : l'évaluation ne
perturbe ni l'environnement ni l'agent d'entraînement.
"""
import argparse
import os

import numpy as np

from algorithms.double_q_learning import DoubleQLearningAgent
from algorithms.q_learning import QLearningAgent
from config import ACTIONS, HyperParams
from environment.credit_env import CreditEnv, REFUSE
from environment.state_space import decode_state
from evaluation.exact import ExactMDP

EVAL_SEED_BASE = 1_000_000
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def make_agent(name: str, hp: HyperParams):
    """Crée un agent. L'agent a son propre générateur (seed + 1), distinct de l'environnement."""
    if name == "q_learning":
        return QLearningAgent(alpha=hp.alpha, gamma=hp.gamma, seed=hp.seed + 1)
    if name == "double_q_learning":
        return DoubleQLearningAgent(alpha=hp.alpha, gamma=hp.gamma, seed=hp.seed + 1)
    raise ValueError(f"algorithme inconnu : {name}")


def evaluate_by_simulation(env, action_fn, n_episodes, gamma, seed_base=EVAL_SEED_BASE):
    """Évalue une politique (action_fn : état -> action) sur n_episodes.

    L'épisode i utilise toujours la seed seed_base + i : le client initial est
    donc le même pour toutes les politiques évaluées (comparaison équitable).
    """
    returns, disc_returns, accepted, defaulted = [], [], [], []
    by_risk = {0: [], 1: [], 2: []}
    for i in range(n_episodes):
        s = env.reset(seed=seed_base + i)
        risk0 = decode_state(s)[0]
        total, disc, k, done = 0.0, 0.0, 0, False
        first_action, is_default = None, False
        while not done:
            a = action_fn(s)
            if first_action is None:
                first_action = a
            s, r, done, info = env.step(a)
            total += r
            disc += (gamma ** k) * r
            k += 1
            is_default = is_default or info["terminal_reason"] == "default"
        returns.append(total)
        disc_returns.append(disc)
        accepted.append(first_action != REFUSE)
        defaulted.append(is_default)
        by_risk[risk0].append(total)

    n_acc = int(np.sum(accepted))
    return {
        "mean_return": float(np.mean(returns)),             # profit moyen par client
        "mean_disc_return": float(np.mean(disc_returns)),   # comparable à J(pi)
        "acceptance_rate": n_acc / n_episodes,
        "default_rate": (int(np.sum(defaulted)) / n_acc) if n_acc else 0.0,
        "by_risk": {r: (float(np.mean(v)) if v else float("nan")) for r, v in by_risk.items()},
    }


def train(agent, hp: HyperParams, mdp=None, verbose=False):
    """Entraîne `agent` pendant hp.n_episodes épisodes. Retourne l'historique (dict).

    mdp : ExactMDP optionnel ; s'il est fourni, J(pi greedy) exact est enregistré.
    """
    env = CreditEnv(seed=hp.seed)
    eval_env = CreditEnv(seed=0)
    n = hp.n_episodes
    h = {
        "train_reward": np.zeros(n),
        "train_default": np.zeros(n, dtype=bool),
        "train_accepted": np.zeros(n, dtype=bool),
        "epsilon": np.zeros(n),
        "eval_episode": [], "eval_J": [], "eval_return": [],
        "eval_disc_return": [], "eval_acceptance": [], "eval_default": [],
    }

    def run_eval(ep):
        stats = evaluate_by_simulation(eval_env, agent.greedy_action, hp.eval_episodes, hp.gamma)
        J = mdp.policy_value(agent.policy(), hp.gamma) if mdp is not None else float("nan")
        h["eval_episode"].append(ep)
        h["eval_J"].append(J)
        h["eval_return"].append(stats["mean_return"])
        h["eval_disc_return"].append(stats["mean_disc_return"])
        h["eval_acceptance"].append(stats["acceptance_rate"])
        h["eval_default"].append(stats["default_rate"])
        return stats, J

    run_eval(0)                                  # politique avant tout apprentissage
    for ep in range(n):
        eps = max(hp.eps_end, hp.eps_start * hp.eps_decay ** ep)
        s, done = env.reset(), False
        total, accepted, defaulted, first = 0.0, False, False, True
        while not done:
            a = agent.select_action(s, eps)
            if first:
                accepted, first = (a != REFUSE), False
            s_next, r, done, info = env.step(a)
            agent.update(s, a, r, s_next, done)
            s = s_next
            total += r
            defaulted = defaulted or info["terminal_reason"] == "default"
        h["train_reward"][ep] = total
        h["train_default"][ep] = defaulted
        h["train_accepted"][ep] = accepted
        h["epsilon"][ep] = eps

        if (ep + 1) % hp.eval_every == 0 or ep + 1 == n:
            stats, J = run_eval(ep + 1)
            if verbose and (len(h["eval_episode"]) % 10 == 0 or ep + 1 == n):
                print(f"ep {ep + 1:>7} | eps={eps:.3f} | J(pi)={J:8.3f} | "
                      f"retour={stats['mean_return']:7.3f} | accept={stats['acceptance_rate']:.2f} | "
                      f"défaut={stats['default_rate']:.3f}")
    return h


def save_results(agent, history, name, seed, out_dir=None):
    """Sauvegarde la/les Q-table(s) (.npy) et l'historique (.npz) dans results/."""
    out_dir = out_dir or os.path.join(ROOT, "results")
    os.makedirs(out_dir, exist_ok=True)
    agent.save(os.path.join(out_dir, f"{name}_seed{seed}_Q.npy"))
    np.savez(os.path.join(out_dir, f"{name}_seed{seed}_history.npz"),
             **{k: np.asarray(v) for k, v in history.items()})


def main(algo_name: str, argv=None):
    """Point d'entrée en ligne de commande, partagé par les deux scripts."""
    d = HyperParams()
    p = argparse.ArgumentParser(description=f"Entraînement : {algo_name}")
    p.add_argument("--episodes", type=int, default=d.n_episodes)
    p.add_argument("--seed", type=int, default=d.seed)
    p.add_argument("--alpha", type=float, default=d.alpha)
    p.add_argument("--gamma", type=float, default=d.gamma)
    p.add_argument("--eps-decay", type=float, default=d.eps_decay)
    args = p.parse_args(argv)

    hp = HyperParams(alpha=args.alpha, gamma=args.gamma, eps_decay=args.eps_decay,
                     n_episodes=args.episodes, seed=args.seed)
    agent = make_agent(algo_name, hp)
    mdp = ExactMDP()
    print(f"Entraînement {algo_name} | {hp}")
    history = train(agent, hp, mdp=mdp, verbose=True)
    save_results(agent, history, algo_name, hp.seed)

    Q_star = mdp.value_iteration(hp.gamma)
    J_star = mdp.policy_value(Q_star.argmax(axis=1), hp.gamma)
    J_final = history["eval_J"][-1]
    print(f"\nJ* (optimum exact) = {J_star:.3f} | J final de l'agent = {J_final:.3f} "
          f"| écart = {J_star - J_final:.3f}")
    print(f"Résultats sauvegardés dans results/{algo_name}_seed{hp.seed}_*")
