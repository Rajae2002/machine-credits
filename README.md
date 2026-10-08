cat > README.md <<'EOF'
# Machine des crédits

Projet de Processus stochastiques (Master 2 IA & Informatique) :
MDP, Q-Learning et Double Q-Learning appliqués à une décision de crédit.

**Environnement SIMULÉ** : probabilités et récompenses artificielles, aucune donnée bancaire réelle.
Démonstration universitaire, à ne pas utiliser pour de vraies décisions.

## Installation
    python -m venv venv
    source venv/bin/activate        # Windows : venv\Scripts\activate
    pip install -r requirements.txt

## Commandes
    pytest                                          # tests
    python -m evaluation.exact                      # solution exacte du MDP
    python -m evaluation.experiments grid           # réglage des hyperparamètres
    python -m evaluation.experiments final          # comparaison finale
    python -m evaluation.comparison report          # statistiques
    python -m visualization.plots                   # figures
    streamlit run app/app.py                        # application

## Structure
    config.py        paramètres du MDP
    environment/     environnement de crédit
    algorithms/      Q-Learning, Double Q-Learning, baselines
    training/        boucle d'entraînement
    evaluation/      solution exacte, expériences, comparaison
    visualization/   figures
    app/             application Streamlit
    tests/           tests pytest
EOF