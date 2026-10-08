"""
app.py : interface Streamlit « Machine des crédits » (démonstration universitaire).

Lancement (depuis la racine du projet) :  streamlit run app/app.py

ATTENTION : environnement SIMULÉ. Ce n'est PAS un système bancaire ; les
probabilités et récompenses sont artificielles (voir config.py).
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np
import pandas as pd
import streamlit as st

from app import logic
from config import (ACTIONS, DEBT_LEVELS, HISTORY_LEVELS, INCOME_LEVELS,
                    RISK_LEVELS, HyperParams)
from environment.state_space import describe_state, encode_state
from evaluation import comparison as comp
from evaluation import experiments as ex
from evaluation.exact import ExactMDP

LABELS = {"q_learning": "Q-Learning", "double_q_learning": "Double Q-Learning"}
FIG_DIR = os.path.join(ROOT, "results", "figures")
FIGURES = [
    ("1_convergence.png", "Convergence de J(π) vers l'optimum exact J*"),
    ("4_vs_baselines.png", "Valeur exacte des politiques"),
    ("3_maximization_bias.png", "Biais de maximisation (MDP-jouet de Sutton & Barto)"),
    ("5_behavior.png", "Acceptation, défaut par période, retour moyen"),
    ("6_by_risk.png", "Performance par niveau de risque"),
    ("2_training_reward.png", "Reward pendant l'entraînement (avec exploration)"),
]


# ----------------------------------------------------------------------
# Données (mises en cache)
# ----------------------------------------------------------------------
@st.cache_resource
def get_exact():
    mdp = ExactMDP()
    return mdp, mdp.value_iteration(HyperParams().gamma)


@st.cache_data
def get_q(algo, cfg_json, seed):
    return logic.load_q(algo, json.loads(cfg_json), seed)


@st.cache_data
def get_curves(cfg_json):
    cfg = json.loads(cfg_json)
    data, x = {}, None
    for algo in LABELS:
        runs = comp.load_runs(comp.FINAL_DIR, algo, cfg[algo]["alpha"], cfg[algo]["eps_decay"])
        if runs:
            data[LABELS[algo]] = np.mean([r["eval_J"] for r in runs], axis=0)
            x = runs[0]["eval_episode"]
    return pd.DataFrame(data, index=x) if data else pd.DataFrame()


def load_reference():
    path = os.path.join(ex.OUT_ROOT, "reference.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


# ----------------------------------------------------------------------
# Page 1 : décision client
# ----------------------------------------------------------------------
def page_decision(cfg):
    st.header("Décision pour un client")
    mdp, q_star_all = get_exact()

    st.sidebar.subheader("Agent")
    algo = st.sidebar.selectbox("Algorithme", list(LABELS), format_func=LABELS.get)
    seeds = logic.available_seeds(algo, cfg)
    if not seeds:
        st.error("Aucun résultat d'entraînement trouvé. Lancez : "
                 "`python -m evaluation.experiments final`")
        return
    seed = st.sidebar.selectbox("Seed d'entraînement", seeds,
                                help="Chaque seed est un entraînement indépendant.")
    c = cfg[algo]
    st.sidebar.caption(f"Configuration : α={c['alpha']}, ε-decay={c['eps_decay']}, "
                       f"{ex.FINAL_EPISODES:,} épisodes")
    Q = get_q(algo, json.dumps(cfg), seed)
    policy = Q.argmax(axis=1)

    st.subheader("Caractéristiques du client [SIMULÉ]")
    c1, c2, c3, c4 = st.columns(4)
    risk = c1.selectbox("Niveau de risque", RISK_LEVELS, index=1)
    income = c2.selectbox("Niveau de revenu", INCOME_LEVELS, index=1)
    debt = c3.selectbox("Niveau d'endettement", DEBT_LEVELS, index=1)
    history = c4.selectbox("Historique de paiement", HISTORY_LEVELS, index=1)
    state = encode_state(RISK_LEVELS.index(risk), INCOME_LEVELS.index(income),
                         DEBT_LEVELS.index(debt), HISTORY_LEVELS.index(history))
    st.info(f"**État du client** (indice {state} sur 54) : {describe_state(state)}")

    a = int(np.argmax(Q[state]))
    m1, m2, m3 = st.columns(3)
    m1.metric("Action recommandée", ACTIONS[a])
    m2.metric("Valeur Q de l'action", f"{Q[state, a]:+.2f}")
    m3.metric("Reward immédiat espéré (1 période)", f"{logic.expected_reward(state, a):+.2f}")

    st.subheader("Valeurs par action")
    st.dataframe(pd.DataFrame(logic.action_table(Q[state], q_star_all[state], state)),
                 hide_index=True, use_container_width=True)
    st.caption("Q appris : estimation de l'agent. Q* exact : valeur optimale calculée sur le modèle "
               "simulé. Reward espéré : gain moyen d'une seule période, sans le futur.")

    st.subheader("Justification de la décision")
    for line in logic.justify(Q[state], q_star_all[state]):
        st.write("• " + line)

    st.subheader("Simulation d'un contrat")
    st.write("Simule UN contrat pour ce client en suivant la politique de l'agent "
             "(le résultat est aléatoire : changez la seed pour en voir un autre).")
    sim_seed = st.number_input("Seed de la simulation", min_value=0, value=0, step=1)
    if st.button("Lancer la simulation"):
        rows, total = logic.simulate_contract(policy, state, int(sim_seed))
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
        st.success(f"Reward total du contrat : {total:+.1f}  (unités fictives)")
    st.caption("Le montant du crédit ne fait pas partie de l'état du MDP : il n'est donc pas demandé.")


# ----------------------------------------------------------------------
# Page 2 : tableau de bord
# ----------------------------------------------------------------------
def page_dashboard(cfg):
    st.header("Tableau de bord des expériences [SIMULÉ]")
    ref = load_reference()
    if ref:
        st.subheader("Références exactes (value iteration)")
        r1, r2, r3 = st.columns(3)
        r1.metric("Optimum J*", f"{ref['J_star']:.3f}")
        r2.metric("Règle par risque", f"{ref['J_risk']:.3f}")
        r3.metric("Aléatoire", f"{ref['J_random']:.3f}")

    st.subheader("Courbes d'apprentissage (moyenne sur les seeds finales)")
    curves = get_curves(json.dumps(cfg))
    if curves.empty:
        st.warning("Courbes indisponibles : lancez `python -m evaluation.experiments final`.")
    else:
        if ref:
            curves["Optimum J*"] = ref["J_star"]
        st.line_chart(curves)
        st.caption("J(π) exact de la politique greedy, évalué tous les 1 000 épisodes.")

    st.subheader("Comparaison Q-Learning / Double Q-Learning (10 seeds)")
    csv_path = os.path.join(comp.OUT_DIR, "per_run_final.csv")
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        df = df[df["setting"] == "tuned"]
        cols = {"J": "J exact", "mean_return": "Retour moyen", "acceptance": "Acceptation",
                "default_contract": "Défaut/contrat", "default_period": "Défaut/période"}
        rows = []
        for algo in LABELS:
            sub = df[df["algo"] == algo]
            if len(sub):
                rows.append({"Algorithme": LABELS[algo], "n seeds": len(sub), **{
                    lab: f"{sub[k].mean():.3f} ± {sub[k].std(ddof=1):.3f}" for k, lab in cols.items()}})
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
        st.caption("Moyenne ± écart-type entre seeds. L'écart entre les deux algorithmes est petit et "
                   "change de signe selon le réglage de α : voir le rapport statistique ci-dessous.")
    else:
        st.warning("Tableau indisponible : lancez `python -m evaluation.comparison report`.")

    st.subheader("Figures")
    shown = False
    for fname, caption in FIGURES:
        path = os.path.join(FIG_DIR, fname)
        if os.path.exists(path):
            st.image(path, caption=caption)
            shown = True
    if not shown:
        st.warning("Aucune figure : lancez `python -m visualization.plots`.")

    report = os.path.join(comp.OUT_DIR, "report.txt")
    if os.path.exists(report):
        with st.expander("Rapport statistique complet (report.txt)"):
            with open(report, encoding="utf-8") as f:
                st.code(f.read())


# ----------------------------------------------------------------------
# Page 3 : à propos et limites
# ----------------------------------------------------------------------
def page_about():
    st.header("À propos, hypothèses et limites")
    st.markdown("""
**Modèle.** MDP à 54 états (risque × revenu × endettement × historique), 3 actions
(REFUSE, ACCEPT, ACCEPT_REDUCED), γ = 0.95. Un épisode correspond à la vie d'un contrat
(durée aléatoire d'environ 12 périodes en moyenne).

**Algorithmes.** Q-Learning et Double Q-Learning tabulaires, comparés à une politique
aléatoire et à une règle simple basée sur le risque. La solution exacte du MDP est calculée
par value iteration et sert de référence.

**Limites et éthique.**
- Toutes les probabilités et récompenses sont **artificielles** (environnement pédagogique) :
  aucun résultat ne dit quoi que ce soit des vrais clients ou des vraies banques.
- Le taux de défaut par contrat (environ 50 %) découle des paramètres simulés ; il ne doit
  jamais être comparé à des taux réels.
- Dans ce modèle, une règle simple par risque est déjà presque optimale (écart d'environ 0.2 %).
- Le montant du crédit, le revenu exact, les biais et l'équité entre groupes ne sont pas modélisés.
- Un vrai système de crédit exigerait des données réelles validées, de l'explicabilité,
  une analyse d'équité, le respect de la réglementation et une **validation humaine**.
- Cette application est une démonstration universitaire : **ne pas l'utiliser pour de vraies décisions**.
""")


# ----------------------------------------------------------------------
def main():
    st.set_page_config(page_title="Machine des crédits", layout="wide")
    st.title("Machine des crédits")
    st.caption("MDP, Q-Learning et Double Q-Learning — projet de Processus stochastiques")
    st.warning("**Environnement SIMULÉ** : probabilités et récompenses artificielles, aucune donnée "
               "bancaire réelle. Ce n'est pas un système de décision bancaire.")

    page = st.sidebar.radio("Page", ["Décision client", "Tableau de bord", "À propos / limites"])
    cfg = logic.load_selected_config()
    if cfg is None and page != "À propos / limites":
        st.error("Résultats introuvables. Lancez d'abord :\n\n"
                 "`python -m evaluation.experiments grid` puis `python -m evaluation.experiments final`")
        st.stop()
    if page == "Décision client":
        page_decision(cfg)
    elif page == "Tableau de bord":
        page_dashboard(cfg)
    else:
        page_about()


main()
