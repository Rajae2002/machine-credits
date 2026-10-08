"""
comparison.py : comparaison statistique Q-Learning / Double Q-Learning (Phase 8).

Tout est calculé à partir des fichiers produits par evaluation/experiments.py :
aucun chiffre n'est écrit à la main. Environnement SIMULÉ [SIMULÉ] : les résultats
ne disent rien de données bancaires réelles.

Test principal (déclaré à l'avance) : différence de J(pi) final exact entre les deux
algorithmes, 10 seeds chacun. Les autres tableaux sont DESCRIPTIFS (pas de test),
pour éviter les comparaisons multiples.

Les deux algorithmes ont des seeds de même numéro mais des trajectoires d'entraînement
différentes : les échantillons sont INDÉPENDANTS (pas appariés). On utilise un test de
permutation et un IC bootstrap (numpy uniquement).

Utilisation :
    python -m evaluation.comparison robust   (contrôle à hyperparamètres identiques)
    python -m evaluation.comparison report   (tableaux + fichiers dans results/experiments/comparison/)
"""
import argparse
import csv
import json
import os

import numpy as np

from algorithms.baselines import RandomPolicy, RiskBasedPolicy
from config import HyperParams
from environment.credit_env import CreditEnv, REFUSE
from environment.state_space import decode_state
from evaluation import experiments as ex
from evaluation.exact import ExactMDP
from training.trainer import EVAL_SEED_BASE

GAMMA = HyperParams().gamma
OUT_DIR = os.path.join(ex.OUT_ROOT, "comparison")
FINAL_DIR = os.path.join(ex.OUT_ROOT, "final")
GRID_DIR = os.path.join(ex.OUT_ROOT, "grid")
ROBUST_DIR = os.path.join(ex.OUT_ROOT, "final_robust")
# Contrôle de robustesse POST-HOC : choisi APRÈS la lecture des résultats du grid
# (la sélection des alpha y était dans le bruit). Déclaré comme tel dans le rapport.
ROBUST_CONFIG = (0.03, 0.9998)

LABELS = {"q_learning": "Q-Learning", "double_q_learning": "Double Q-Learning"}
KEYS = ("eval_episode", "eval_J", "final_J", "policy", "seed", "alpha", "eps_decay")
METRICS = (("mean_return", "retour moyen"), ("acceptance", "acceptation"),
           ("default_contract", "défaut/contrat"), ("default_period", "défaut/période"),
           ("mean_periods", "durée (périodes)"))
RISK_METRICS = (("ret_LOW", "risque LOW"), ("ret_MEDIUM", "risque MEDIUM"),
                ("ret_HIGH", "risque HIGH"))


# ----------------------------------------------------------------------
# Chargement des runs
# ----------------------------------------------------------------------
def load_runs(stage_dir, algo, alpha=None, decay=None):
    """Runs sauvegardés d'un algorithme (filtrés par configuration si demandé), triés par seed."""
    runs = []
    if not os.path.isdir(stage_dir):
        return runs
    for path in ex._run_files(stage_dir):
        with np.load(path) as d:
            if str(d["algo"]) != algo:
                continue
            if alpha is not None and not np.isclose(float(d["alpha"]), alpha):
                continue
            if decay is not None and not np.isclose(float(d["eps_decay"]), decay):
                continue
            runs.append({k: d[k] for k in KEYS})
    return sorted(runs, key=lambda r: int(r["seed"]))


# ----------------------------------------------------------------------
# Statistiques (numpy uniquement)
# ----------------------------------------------------------------------
def bootstrap_ci(values, n_boot=10_000, seed=0, level=0.95):
    """IC percentile bootstrap de la moyenne."""
    rng = np.random.default_rng(seed)
    v = np.asarray(values, dtype=float)
    means = rng.choice(v, size=(n_boot, len(v))).mean(axis=1)
    a = (1.0 - level) / 2.0 * 100.0
    lo, hi = np.percentile(means, [a, 100.0 - a])
    return float(lo), float(hi)


def compare_means(a, b, n_boot=10_000, seed=0):
    """Différence de moyennes mean(a) - mean(b), IC95 % bootstrap et p-value de permutation
    (bilatérale) pour deux échantillons INDÉPENDANTS. Retourne (diff, (lo, hi), p)."""
    rng = np.random.default_rng(seed)
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    diff = float(a.mean() - b.mean())
    boots = (rng.choice(a, size=(n_boot, len(a))).mean(axis=1)
             - rng.choice(b, size=(n_boot, len(b))).mean(axis=1))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    pooled = np.concatenate([a, b])
    perm = pooled[np.argsort(rng.random((n_boot, len(pooled))), axis=1)]
    d_perm = perm[:, :len(a)].mean(axis=1) - perm[:, len(a):].mean(axis=1)
    p = (int(np.sum(np.abs(d_perm) >= abs(diff) - 1e-12)) + 1) / (n_boot + 1)
    return diff, (float(lo), float(hi)), float(p)


def ms(values):
    v = np.asarray(values, dtype=float)
    return f"{v.mean():.3f} ± {ex._std(v):.3f}"


# ----------------------------------------------------------------------
# Métriques d'un run
# ----------------------------------------------------------------------
def episodes_to_threshold(episodes, J, thr):
    """Premier point d'évaluation où J >= thr (NaN si jamais atteint)."""
    hit = np.flatnonzero(np.asarray(J) >= thr)
    return float(np.asarray(episodes)[hit[0]]) if len(hit) else float("nan")


def policy_diagnostics(policy, pi_star):
    """(nb d'états où la politique diffère de pi*, nb de REFUSE à tort)."""
    policy, pi_star = np.asarray(policy), np.asarray(pi_star)
    wrong = int(np.sum(policy != pi_star))
    wrong_refusals = int(np.sum((policy == REFUSE) & (pi_star != REFUSE)))
    return wrong, wrong_refusals


def simulate_policy(action_fn, n_episodes=ex.FINAL_EVAL_EPISODES):
    """Évalue une politique sur n_episodes clients FIXES (mêmes seeds que le trainer).

    - acceptation : part des clients dont la première décision n'est pas REFUSE
    - défaut/contrat : défauts / contrats acceptés
    - défaut/période : défauts / périodes où un crédit était en cours (action != REFUSE)
    """
    env = CreditEnv(seed=0)
    returns, by_risk = [], {0: [], 1: [], 2: []}
    accepted = defaults = periods = 0
    for i in range(n_episodes):
        s = env.reset(seed=EVAL_SEED_BASE + i)
        risk0 = decode_state(s)[0]
        total, done, first = 0.0, False, True
        while not done:
            a = action_fn(s)
            if first:
                accepted += int(a != REFUSE)
                first = False
            periods += int(a != REFUSE)
            s, r, done, info = env.step(a)
            total += r
            defaults += int(info["terminal_reason"] == "default")
        returns.append(total)
        by_risk[risk0].append(total)
    out = {
        "mean_return": float(np.mean(returns)),
        "acceptance": accepted / n_episodes,
        "default_contract": defaults / accepted if accepted else 0.0,
        "default_period": defaults / periods if periods else 0.0,
        "mean_periods": periods / accepted if accepted else 0.0,
    }
    for risk, name in enumerate(("LOW", "MEDIUM", "HIGH")):
        out[f"ret_{name}"] = float(np.mean(by_risk[risk])) if by_risk[risk] else float("nan")
    return out


def per_run_table(runs, algo, J_star, pi_star, n_eval, setting="tuned"):
    """Une ligne de métriques par run (J exact, simulation, diagnostics, convergence)."""
    rows = []
    for r in runs:
        policy = r["policy"]
        sim = simulate_policy(lambda s, p=policy: int(p[s]), n_eval)
        wrong, wrong_refusals = policy_diagnostics(policy, pi_star)
        rows.append({
            "setting": setting, "algo": algo, "seed": int(r["seed"]),
            "alpha": float(r["alpha"]), "eps_decay": float(r["eps_decay"]),
            "J": float(r["final_J"]), **sim,
            "wrong_states": wrong, "wrong_refusals": wrong_refusals,
            "ep_to_90": episodes_to_threshold(r["eval_episode"], r["eval_J"], 0.90 * J_star),
            "ep_to_95": episodes_to_threshold(r["eval_episode"], r["eval_J"], 0.95 * J_star),
            "auc": float(np.mean(r["eval_J"])) / J_star,
            "wobble": float(np.std(r["eval_J"][-10:], ddof=1)),
        })
    return rows


# ----------------------------------------------------------------------
# Sections du rapport
# ----------------------------------------------------------------------
def compare_pair(emit, title, ql_rows, dq_rows, ref):
    J_star, J_risk = ref["J_star"], ref["J_risk"]
    emit(title)
    for label, rows in (("Q-Learning", ql_rows), ("Double Q-Learning", dq_rows)):
        J = [r["J"] for r in rows]
        lo, hi = bootstrap_ci(J)
        emit(f"  {label:<18} n={len(J):<3} J = {np.mean(J):.3f} ± {ex._std(J):.3f} | "
             f"IC95% [{lo:.3f} ; {hi:.3f}] | {100 * np.mean(J) / J_star:.1f} % de J* | "
             f"écart à la règle par risque : {np.mean(J) - J_risk:+.3f}")
    d, (lo, hi), p = compare_means([r["J"] for r in ql_rows], [r["J"] for r in dq_rows])
    emit(f"  Différence (Q-Learning − Double Q-Learning) : {d:+.3f} | "
         f"IC95% bootstrap [{lo:+.3f} ; {hi:+.3f}] | p permutation = {p:.3f}")
    emit("  Lecture : un IC qui contient 0 signifie que les données ne distinguent pas les deux")
    emit("  algorithmes ; cela ne prouve pas qu'ils sont équivalents (10 seeds : puissance limitée).")
    emit("")


def section_behavior(emit, tables, base):
    emit("B. Comportement (5 000 clients fixes, identiques pour toutes les politiques) [SIMULÉ]")
    emit(f"  {'politique':<22}" + "".join(f"{lab:>18}" for _, lab in METRICS))
    for name, sim in base.items():
        emit(f"  {name:<22}" + "".join(f"{sim[k]:>18.3f}" for k, _ in METRICS))
    for algo, rows in tables.items():
        emit(f"  {LABELS[algo]:<22}" + "".join(f"{ms([r[k] for r in rows]):>18}" for k, _ in METRICS))
    emit("")
    emit("  Retour moyen selon le niveau de risque initial du client :")
    emit(f"  {'politique':<22}" + "".join(f"{lab:>18}" for _, lab in RISK_METRICS))
    for name, sim in base.items():
        emit(f"  {name:<22}" + "".join(f"{sim[k]:>18.3f}" for k, _ in RISK_METRICS))
    for algo, rows in tables.items():
        emit(f"  {LABELS[algo]:<22}" + "".join(f"{ms([r[k] for r in rows]):>18}" for k, _ in RISK_METRICS))
    emit("")
    emit("  Écarts à la politique optimale π* (sur 54 états ; REFUSE à tort = REFUSE là où π* accepte) :")
    for algo, rows in tables.items():
        emit(f"    {LABELS[algo]:<18} états ≠ π* : {ms([r['wrong_states'] for r in rows])} | "
             f"refus à tort : {ms([r['wrong_refusals'] for r in rows])}")
    emit("")


def section_convergence(emit, tables):
    emit("C. Vitesse et stabilité (J exact tous les 1 000 épisodes) : descriptif")
    for algo, rows in tables.items():
        parts = []
        for key, pct in (("ep_to_90", 90), ("ep_to_95", 95)):
            vals = np.array([r[key] for r in rows], dtype=float)
            ok = vals[~np.isnan(vals)]
            txt = f"{pct} % de J* : atteint par {len(ok)}/{len(vals)} runs"
            if len(ok):
                txt += f", épisode médian {np.median(ok):.0f}"
            parts.append(txt)
        emit(f"  {LABELS[algo]:<18} " + " | ".join(parts))
        emit(f"  {'':<18} aire sous la courbe (J/J* moyen) = {ms([r['auc'] for r in rows])} | "
             f"fluctuation (écart-type de J, 10 derniers points) = {ms([r['wobble'] for r in rows])}")
    emit("")


def section_grid(emit):
    emit("D. Sensibilité aux hyperparamètres (étape grid : 5 seeds, 20 000 épisodes, "
         "score = J moyen des 5 derniers points) : descriptif")
    if not os.path.isdir(GRID_DIR):
        emit("  (dossier grid absent)")
        emit("")
        return
    _, scores = ex.select_best(GRID_DIR)
    configs = sorted({(a, d) for (_, a, d) in scores}, reverse=True)
    emit(f"  {'alpha':>6}{'eps_decay':>11}{'Q-Learning':>18}{'Double Q-Learning':>20}{'DQL − QL':>11}")
    for alpha, decay in configs:
        q, dq = scores.get(("q_learning", alpha, decay)), scores.get(("double_q_learning", alpha, decay))
        if q and dq:
            emit(f"  {alpha:>6}{decay:>11}{ms(q):>18}{ms(dq):>20}{np.mean(dq) - np.mean(q):>+11.3f}")
    emit("")


# ----------------------------------------------------------------------
def write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def run_robust():
    """Entraîne ce qui manque pour comparer les deux algorithmes à hyperparamètres identiques."""
    mdp = ExactMDP()
    for algo in ex.ALGOS:
        if len(load_runs(FINAL_DIR, algo, *ROBUST_CONFIG)) >= len(ex.FINAL_SEEDS):
            print(f"{LABELS[algo]} : déjà disponible dans final/ pour {ROBUST_CONFIG}, rien à calculer")
            continue
        ex.run_runs([(algo, *ROBUST_CONFIG)], ex.FINAL_SEEDS, ex.FINAL_EPISODES, mdp, ROBUST_DIR)
    print("\nContrôle prêt : lancez  python -m evaluation.comparison report")


def build_report():
    mdp = ExactMDP()
    ref = ex.reference_values(mdp)
    J_star = ref["J_star"]
    pi_star = mdp.value_iteration(GAMMA).argmax(axis=1)
    cfg_path = os.path.join(ex.OUT_ROOT, "selected_config.json")
    if not os.path.exists(cfg_path):
        raise SystemExit("selected_config.json introuvable : lancez d'abord experiments grid puis final")
    with open(cfg_path) as f:
        cfg = json.load(f)

    os.makedirs(OUT_DIR, exist_ok=True)
    lines = []

    def emit(line=""):
        print(line)
        lines.append(line)

    n_eval = ex.FINAL_EVAL_EPISODES
    tables = {}
    for algo in ex.ALGOS:
        runs = load_runs(FINAL_DIR, algo, cfg[algo]["alpha"], cfg[algo]["eps_decay"])
        if not runs:
            raise SystemExit(f"Aucun run final pour {algo} : lancez experiments final")
        tables[algo] = per_run_table(runs, algo, J_star, pi_star, n_eval, "tuned")

    emit("COMPARAISON Q-LEARNING / DOUBLE Q-LEARNING  [SIMULÉ : modèle pédagogique, pas de données réelles]")
    emit(f"Références exactes : J* = {J_star:.3f} | règle par risque = {ref['J_risk']:.3f} | "
         f"aléatoire = {ref['J_random']:.3f}")
    for algo in ex.ALGOS:
        emit(f"Configuration retenue {LABELS[algo]:<18}: alpha={cfg[algo]['alpha']}, "
             f"eps_decay={cfg[algo]['eps_decay']}")
    emit("")
    compare_pair(emit, "A. TEST PRINCIPAL : J(π) final exact, configurations retenues par grid",
                 tables["q_learning"], tables["double_q_learning"], ref)

    # Contrôle de robustesse (post-hoc) : mêmes hyperparamètres pour les deux algorithmes
    robust, all_rows = {}, [r for t in tables.values() for r in t]
    for algo in ex.ALGOS:
        if (cfg[algo]["alpha"], cfg[algo]["eps_decay"]) == ROBUST_CONFIG:
            robust[algo] = tables[algo]
            continue
        runs = load_runs(FINAL_DIR, algo, *ROBUST_CONFIG) or load_runs(ROBUST_DIR, algo, *ROBUST_CONFIG)
        if runs:
            robust[algo] = per_run_table(runs, algo, J_star, pi_star, n_eval, "same_config")
            all_rows += robust[algo]
    if len(robust) == 2:
        compare_pair(emit, f"A bis. CONTRÔLE POST-HOC : mêmes hyperparamètres pour les deux "
                           f"(alpha={ROBUST_CONFIG[0]}, eps_decay={ROBUST_CONFIG[1]})",
                     robust["q_learning"], robust["double_q_learning"], ref)
    else:
        emit("A bis. Contrôle à hyperparamètres identiques non lancé : "
             "python -m evaluation.comparison robust")
        emit("")

    base_policies = {
        "Aléatoire": RandomPolicy(seed=0).greedy_action,
        "Règle par risque": RiskBasedPolicy().greedy_action,
        "Optimale π* (exacte)": lambda s: int(pi_star[s]),
    }
    base = {name: simulate_policy(fn, n_eval) for name, fn in base_policies.items()}
    section_behavior(emit, tables, base)
    section_convergence(emit, tables)
    section_grid(emit)

    with open(os.path.join(OUT_DIR, "report.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    write_csv(os.path.join(OUT_DIR, "per_run_final.csv"), all_rows)
    print(f"Fichiers écrits dans {OUT_DIR} : report.txt, per_run_final.csv")


def main(argv=None):
    p = argparse.ArgumentParser(description="Comparaison Q-Learning / Double Q-Learning")
    p.add_argument("stage", choices=["report", "robust"])
    args = p.parse_args(argv)
    run_robust() if args.stage == "robust" else build_report()


if __name__ == "__main__":
    main()
