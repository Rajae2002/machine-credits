"""
experiments.py : protocole expérimental (Phase 7).

Deux étapes :
  1. grid  : réglage de (alpha, eps_decay) sur des seeds de RÉGLAGE (0..4)
  2. final : comparaison sur des seeds FRAÎCHES (100..109), avec la configuration
             retenue à l'étape 1 pour chaque algorithme (même procédure pour les deux)
Seeds de réglage et de comparaison disjointes ; clients d'évaluation (seeds >= 1 000 000)
disjoints des clients d'entraînement : la sélection ne « fuit » pas dans le résultat final.

gamma n'est PAS réglé : il fait partie de la définition du problème (J en dépend).
Environnement SIMULÉ [SIMULÉ] : aucun résultat ne concerne des données bancaires réelles.

Utilisation :
    python -m evaluation.experiments grid --quick     (test rapide)
    python -m evaluation.experiments final --quick
    python -m evaluation.experiments grid
    python -m evaluation.experiments final
Les calculs déjà faits sont conservés : on peut interrompre (Ctrl+C) puis relancer.
"""
import argparse
import json
import os
import time

import numpy as np

from algorithms.baselines import RandomPolicy, RiskBasedPolicy
from config import HyperParams
from environment.credit_env import CreditEnv
from evaluation.exact import ExactMDP
from training.trainer import ROOT, evaluate_by_simulation, make_agent, train

ALGOS = ("q_learning", "double_q_learning")
ALPHAS = (0.1, 0.03, 0.01)
EPS_DECAYS = (0.9995, 0.9998)
GRID_SEEDS = (0, 1, 2, 3, 4)
FINAL_SEEDS = tuple(range(100, 110))
GRID_EPISODES = 20_000
FINAL_EPISODES = 50_000
FINAL_EVAL_EPISODES = 5_000
OUT_ROOT = os.path.join(ROOT, "results", "experiments")

assert set(GRID_SEEDS).isdisjoint(FINAL_SEEDS)


# ----------------------------------------------------------------------
# Références exactes
# ----------------------------------------------------------------------
def reference_values(mdp, gamma=0.95):
    """J exact de la politique optimale, de la baseline par risque et de l'aléatoire."""
    Q = mdp.value_iteration(gamma)
    return {
        "gamma": gamma,
        "J_star": float(mdp.policy_value(Q.argmax(axis=1), gamma)),
        "J_risk": float(mdp.policy_value(RiskBasedPolicy().policy(), gamma)),
        "J_random": float(mdp.policy_value(RandomPolicy().policy_matrix(), gamma)),
    }


# ----------------------------------------------------------------------
# Un entraînement = un fichier .npz
# ----------------------------------------------------------------------
def run_name(algo, alpha, eps_decay, seed):
    return f"{algo}_a{alpha}_d{eps_decay}_s{seed}"


def run_one(algo, alpha, eps_decay, seed, episodes, mdp, out_dir,
            eval_every=1000, eval_episodes=100,
            final_eval_episodes=FINAL_EVAL_EPISODES, force=False):
    """Entraîne un agent et sauvegarde le résultat. Retourne (chemin, calcul_effectué).

    Si le fichier existe déjà avec le même nombre d'épisodes, il est réutilisé."""
    os.makedirs(out_dir, exist_ok=True)
    name = run_name(algo, alpha, eps_decay, seed)
    path = os.path.join(out_dir, name + ".npz")
    if os.path.exists(path) and not force:
        with np.load(path) as d:
            if int(d["episodes"]) == episodes:
                return path, False

    hp = HyperParams(alpha=alpha, eps_decay=eps_decay, n_episodes=episodes,
                     eval_every=eval_every, eval_episodes=eval_episodes, seed=seed)
    agent = make_agent(algo, hp)
    h = train(agent, hp, mdp=mdp)
    final = evaluate_by_simulation(CreditEnv(seed=0), agent.greedy_action,
                                   final_eval_episodes, hp.gamma)

    tmp = os.path.join(out_dir, "tmp_" + name + ".npz")      # écriture atomique
    np.savez_compressed(
        tmp, algo=algo, alpha=alpha, eps_decay=eps_decay, seed=seed, episodes=episodes,
        eval_episode=np.asarray(h["eval_episode"]), eval_J=np.asarray(h["eval_J"]),
        eval_return=np.asarray(h["eval_return"]),
        eval_acceptance=np.asarray(h["eval_acceptance"]),
        eval_default=np.asarray(h["eval_default"]),
        final_J=h["eval_J"][-1], final_mean_return=final["mean_return"],
        final_disc_return=final["mean_disc_return"],
        final_acceptance=final["acceptance_rate"], final_default_rate=final["default_rate"],
        final_by_risk=np.array([final["by_risk"][r] for r in (0, 1, 2)]),
        policy=agent.policy(), Q=agent.Q,
        train_reward=h["train_reward"].astype(np.float32),
        train_default=h["train_default"], train_accepted=h["train_accepted"],
    )
    os.replace(tmp, path)
    return path, True


def _run_files(stage_dir):
    return [os.path.join(stage_dir, f) for f in sorted(os.listdir(stage_dir))
            if f.endswith(".npz") and not f.startswith("tmp_")]


def _std(values):
    return float(np.std(values, ddof=1)) if len(values) > 1 else 0.0


def run_runs(configs, seeds, episodes, mdp, stage_dir, force=False):
    """Lance tous les entraînements (configs x seeds) en affichant la progression."""
    jobs = [(a, al, d, s) for (a, al, d) in configs for s in seeds]
    for i, (algo, alpha, decay, seed) in enumerate(jobs, 1):
        t0 = time.perf_counter()
        path, ran = run_one(algo, alpha, decay, seed, episodes, mdp, stage_dir, force=force)
        tag = f"[{i}/{len(jobs)}] {algo:<18} alpha={alpha:<5} decay={decay} seed={seed:<3}"
        if ran:
            with np.load(path) as d:
                J = float(d["final_J"])
            print(f"{tag} J={J:7.3f}  ({time.perf_counter() - t0:.1f} s)", flush=True)
        else:
            print(f"{tag} déjà calculé, ignoré", flush=True)


# ----------------------------------------------------------------------
# Sélection de configuration (étape grid)
# ----------------------------------------------------------------------
def select_best(stage_dir, k=5):
    """Pour chaque algorithme, la configuration au meilleur score moyen.
    Score d'un run = moyenne de J exact sur ses k derniers points d'évaluation."""
    scores = {}
    for path in _run_files(stage_dir):
        with np.load(path) as d:
            key = (str(d["algo"]), float(d["alpha"]), float(d["eps_decay"]))
            scores.setdefault(key, []).append(float(np.mean(d["eval_J"][-k:])))
    best = {}
    for (algo, alpha, decay), vals in sorted(scores.items()):
        m = float(np.mean(vals))
        if algo not in best or m > best[algo]["score"]:
            best[algo] = {"alpha": alpha, "eps_decay": decay, "score": m, "n_seeds": len(vals)}
    return best, scores


def print_grid_table(scores):
    print(f"\n{'algorithme':<19}{'alpha':>7}{'eps_decay':>11}{'score J':>10}{'écart-type':>12}{'n':>4}")
    for (algo, alpha, decay), vals in sorted(scores.items()):
        print(f"{algo:<19}{alpha:>7}{decay:>11}{np.mean(vals):>10.3f}{_std(vals):>12.3f}{len(vals):>4}")


# ----------------------------------------------------------------------
# Résumé provisoire de l'étape final (l'analyse statistique est en Phase 8)
# ----------------------------------------------------------------------
def summarize_final(stage_dir, ref):
    rows = {}
    for path in _run_files(stage_dir):
        with np.load(path) as d:
            rows.setdefault(str(d["algo"]), []).append(
                [float(d["final_J"]), float(d["final_acceptance"]),
                 float(d["final_default_rate"]), float(d["final_mean_return"])])
    print(f"\nRéférences exactes [SIMULÉ] : J* = {ref['J_star']:.3f} | "
          f"baseline par risque = {ref['J_risk']:.3f} | aléatoire = {ref['J_random']:.3f}")
    print(f"{'algorithme':<19}{'n':>3}{'J final':>17}{'acceptation':>17}"
          f"{'défaut/contrat':>17}{'retour moyen':>17}")
    for algo, vals in rows.items():
        arr = np.array(vals)
        cells = [f"{arr[:, j].mean():7.3f} ± {_std(arr[:, j]):5.3f}" for j in range(4)]
        print(f"{algo:<19}{len(arr):>3}" + "".join(f"{c:>17}" for c in cells))
    print("(moyenne ± écart-type entre seeds ; l'analyse comparative est en Phase 8)")


# ----------------------------------------------------------------------
def main(argv=None):
    p = argparse.ArgumentParser(description="Expériences Q-Learning vs Double Q-Learning")
    p.add_argument("stage", choices=["grid", "final"])
    p.add_argument("--quick", action="store_true", help="test rapide : 2 seeds, 2000 épisodes")
    p.add_argument("--episodes", type=int, default=None)
    p.add_argument("--force", action="store_true", help="recalculer même si déjà fait")
    args = p.parse_args(argv)

    suffix = "_quick" if args.quick else ""
    mdp = ExactMDP()
    ref = reference_values(mdp)
    os.makedirs(OUT_ROOT, exist_ok=True)
    with open(os.path.join(OUT_ROOT, "reference.json"), "w") as f:
        json.dump(ref, f, indent=2)
    print(f"Références exactes [SIMULÉ] : J* = {ref['J_star']:.3f} | "
          f"baseline par risque = {ref['J_risk']:.3f} | aléatoire = {ref['J_random']:.3f}\n")

    cfg_path = os.path.join(OUT_ROOT, f"selected_config{suffix}.json")
    stage_dir = os.path.join(OUT_ROOT, args.stage + suffix)

    if args.stage == "grid":
        seeds = (0, 1) if args.quick else GRID_SEEDS
        episodes = args.episodes or (2_000 if args.quick else GRID_EPISODES)
        configs = [(a, al, d) for a in ALGOS for al in ALPHAS for d in EPS_DECAYS]
        run_runs(configs, seeds, episodes, mdp, stage_dir, args.force)
        best, scores = select_best(stage_dir)
        print_grid_table(scores)
        with open(cfg_path, "w") as f:
            json.dump(best, f, indent=2)
        print("\nConfiguration retenue (même critère pour les deux algorithmes) :")
        for algo, cfg in best.items():
            print(f"  {algo:<19} alpha={cfg['alpha']}  eps_decay={cfg['eps_decay']}  "
                  f"score={cfg['score']:.3f}")
    else:
        if not os.path.exists(cfg_path):
            raise SystemExit("Lancez d'abord : python -m evaluation.experiments grid"
                             + (" --quick" if args.quick else ""))
        with open(cfg_path) as f:
            best = json.load(f)
        seeds = (100, 101) if args.quick else FINAL_SEEDS
        episodes = args.episodes or (2_000 if args.quick else FINAL_EPISODES)
        configs = [(a, best[a]["alpha"], best[a]["eps_decay"]) for a in ALGOS]
        run_runs(configs, seeds, episodes, mdp, stage_dir, args.force)
        summarize_final(stage_dir, ref)


if __name__ == "__main__":
    main()
