"""
plots.py : figures du projet (Phase 9). Lit UNIQUEMENT results/experiments/.
Tous les résultats viennent d'un environnement SIMULÉ [SIMULÉ] (pas de données réelles).

Utilisation : python -m visualization.plots   ->  results/figures/*.png
"""
import csv
import json
import os

import matplotlib
matplotlib.use("Agg")                       # écrit des PNG, sans écran
import matplotlib.pyplot as plt
import numpy as np

from algorithms.baselines import RandomPolicy, RiskBasedPolicy
from evaluation import comparison as comp
from evaluation import experiments as ex
from evaluation.exact import ExactMDP
from training.trainer import ROOT

FIG_DIR = os.path.join(ROOT, "results", "figures")
COLORS = {"q_learning": "tab:blue", "double_q_learning": "tab:orange"}
LABELS = comp.LABELS


def load_inputs():
    with open(os.path.join(ex.OUT_ROOT, "selected_config.json")) as f:
        cfg = json.load(f)
    with open(os.path.join(ex.OUT_ROOT, "reference.json")) as f:
        ref = json.load(f)
    runs = {a: comp.load_runs(comp.FINAL_DIR, a, cfg[a]["alpha"], cfg[a]["eps_decay"])
            for a in ex.ALGOS}
    for a, r in runs.items():
        if not r:
            raise SystemExit(f"Aucun run final pour {a} : lancez experiments final")
    csv_path = os.path.join(comp.OUT_DIR, "per_run_final.csv")
    if not os.path.exists(csv_path):
        raise SystemExit("per_run_final.csv absent : lancez  python -m evaluation.comparison report")
    with open(csv_path, newline="", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r["setting"] == "tuned"]
    return cfg, ref, runs, rows


def col(rows, algo, key):
    return np.array([float(r[key]) for r in rows if r["algo"] == algo])


def baseline_sims():
    mdp = ExactMDP()
    pi_star = mdp.value_iteration(comp.GAMMA).argmax(axis=1)
    fns = {
        "Aléatoire": RandomPolicy(seed=0).greedy_action,
        "Règle par risque": RiskBasedPolicy().greedy_action,
        "Optimale π*": lambda s: int(pi_star[s]),
    }
    return {name: comp.simulate_policy(fn) for name, fn in fns.items()}


def moving_average(v, w):
    c = np.cumsum(np.insert(np.asarray(v, dtype=float), 0, 0.0))
    return (c[w:] - c[:-w]) / w


def save(fig, name):
    os.makedirs(FIG_DIR, exist_ok=True)
    path = os.path.join(FIG_DIR, name)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("écrit :", path)


# ----------------------------------------------------------------------
def fig_convergence(runs, ref, cfg):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for algo, rs in runs.items():
        x = rs[0]["eval_episode"]
        J = np.array([r["eval_J"] for r in rs])
        m, s = J.mean(axis=0), J.std(axis=0, ddof=1)
        ax.plot(x, m, color=COLORS[algo], label=f"{LABELS[algo]} (α={cfg[algo]['alpha']})")
        ax.fill_between(x, m - s, m + s, color=COLORS[algo], alpha=0.2)
    ax.axhline(ref["J_star"], ls="--", color="black", label="Optimum exact J*")
    ax.axhline(ref["J_risk"], ls=":", color="gray", label="Règle par risque")
    ax.set_ylim(0, ref["J_star"] * 1.1)
    ax.set_xlabel("Épisodes d'entraînement")
    ax.set_ylabel("J(π) exact de la politique greedy")
    ax.set_title(f"Convergence ({len(runs['q_learning'])} seeds, moyenne ± écart-type) [SIMULÉ]")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    save(fig, "1_convergence.png")


def fig_training_reward(runs, window=1000):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for algo, rs in runs.items():
        curves = np.array([moving_average(r_["train_reward"], window) if "train_reward" in r_
                           else None for r_ in rs]) if False else None
    # train_reward n'est pas dans comp.KEYS : on relit les fichiers
    for algo, rs in runs.items():
        curves = []
        for r in rs:
            path = os.path.join(comp.FINAL_DIR, ex.run_name(
                algo, float(r["alpha"]), float(r["eps_decay"]), int(r["seed"])) + ".npz")
            with np.load(path) as d:
                curves.append(moving_average(d["train_reward"], window))
        curves = np.array(curves)
        x = np.arange(window, window + curves.shape[1])
        m, s = curves.mean(axis=0), curves.std(axis=0, ddof=1)
        ax.plot(x, m, color=COLORS[algo], label=LABELS[algo])
        ax.fill_between(x, m - s, m + s, color=COLORS[algo], alpha=0.2)
    ax.set_xlabel("Épisode")
    ax.set_ylabel(f"Reward par épisode (moyenne mobile, fenêtre {window})")
    ax.set_title("Reward PENDANT l'entraînement (avec exploration ε) [SIMULÉ]")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    save(fig, "2_training_reward.png")


def fig_max_bias():
    path = os.path.join(ex.OUT_ROOT, "maximization_bias.npz")
    if not os.path.exists(path):
        print("maximization_bias.npz absent : figure ignorée (python -m evaluation.maximization_bias)")
        return
    with np.load(path) as d:
        curves = {k: d[k] for k in ("q_learning", "double_q_learning")}
        n_runs = int(d["runs"])
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for algo, c in curves.items():
        ax.plot(np.arange(1, len(c) + 1), 100 * c, color=COLORS[algo], label=LABELS[algo])
    ax.axhline(5, ls="--", color="black", label="Minimum atteignable (ε/2 = 5 %)")
    ax.set_xlabel("Épisode")
    ax.set_ylabel("% d'épisodes où LEFT (sous-optimal) est choisi")
    ax.set_title(f"Biais de maximisation : MDP-jouet de Sutton & Barto ({n_runs} runs)")
    ax.legend()
    ax.grid(alpha=0.3)
    save(fig, "3_maximization_bias.png")


def fig_vs_baselines(runs, ref):
    names = ["Aléatoire", "Règle par risque", "Q-Learning", "Double Q-Learning", "Optimum J*"]
    means = [ref["J_random"], ref["J_risk"]]
    errs = [0.0, 0.0]
    for algo in ex.ALGOS:
        J = np.array([float(r["final_J"]) for r in runs[algo]])
        means.append(J.mean())
        errs.append(J.std(ddof=1))
    means.append(ref["J_star"])
    errs.append(0.0)
    colors = ["lightgray", "gray", COLORS["q_learning"], COLORS["double_q_learning"], "black"]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(names, means, yerr=errs, color=colors, capsize=4)
    for i, v in enumerate(means):
        ax.text(i, v + 0.2, f"{v:.2f}", ha="center")
    ax.set_ylabel("J(π) exact")
    ax.set_title("Valeur exacte des politiques (barres d'erreur : écart-type entre seeds) [SIMULÉ]")
    plt.setp(ax.get_xticklabels(), rotation=15)
    save(fig, "4_vs_baselines.png")


def fig_behavior(rows, base):
    metrics = (("acceptance", "Taux d'acceptation"),
               ("default_period", "Défaut par période"),
               ("mean_return", "Retour moyen par client"))
    names = list(base) + [LABELS[a] for a in ex.ALGOS]
    colors = ["lightgray"] * len(base) + [COLORS[a] for a in ex.ALGOS]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, (key, title) in zip(axes, metrics):
        vals = [base[n][key] for n in base] + [col(rows, a, key).mean() for a in ex.ALGOS]
        errs = [0.0] * len(base) + [col(rows, a, key).std(ddof=1) for a in ex.ALGOS]
        ax.bar(names, vals, yerr=errs, color=colors, capsize=3)
        ax.set_title(title)
        plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    fig.suptitle("Comportement des politiques (5 000 clients fixes) [SIMULÉ]")
    save(fig, "5_behavior.png")


def fig_by_risk(rows, base):
    keys = (("ret_LOW", "LOW"), ("ret_MEDIUM", "MEDIUM"), ("ret_HIGH", "HIGH"))
    series = [("Aléatoire", "lightgray", [base["Aléatoire"][k] for k, _ in keys], [0] * 3),
              ("Optimale π*", "black", [base["Optimale π*"][k] for k, _ in keys], [0] * 3)]
    for algo in ex.ALGOS:
        series.append((LABELS[algo], COLORS[algo],
                       [col(rows, algo, k).mean() for k, _ in keys],
                       [col(rows, algo, k).std(ddof=1) for k, _ in keys]))
    fig, ax = plt.subplots(figsize=(8, 4.5))
    w, x = 0.2, np.arange(3)
    for i, (name, color, vals, errs) in enumerate(series):
        ax.bar(x + (i - 1.5) * w, vals, w, yerr=errs, label=name, color=color, capsize=3)
    ax.set_xticks(x)
    ax.set_xticklabels([lab for _, lab in keys])
    ax.set_xlabel("Niveau de risque initial du client")
    ax.set_ylabel("Retour moyen")
    ax.set_title("Performance par niveau de risque [SIMULÉ]")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    save(fig, "6_by_risk.png")


def main():
    cfg, ref, runs, rows = load_inputs()
    base = baseline_sims()
    fig_convergence(runs, ref, cfg)
    fig_training_reward(runs)
    fig_max_bias()
    fig_vs_baselines(runs, ref)
    fig_behavior(rows, base)
    fig_by_risk(rows, base)
    print(f"\n6 figures dans {FIG_DIR}")


if __name__ == "__main__":
    main()
