"""
state_space.py : encodage des états du MDP.

Un état est un tuple d'entiers (risk, income, debt, history) :
    risk, income, debt : 0=LOW, 1=MEDIUM, 2=HIGH
    history            : 0=BAD, 1=GOOD
Il est encodé par un indice unique dans [0, 53], utilisé comme ligne
de la Q-table. L'état terminal n'a PAS d'indice : il est signalé par done=True.
"""
from config import (
    RISK_LEVELS, INCOME_LEVELS, DEBT_LEVELS, HISTORY_LEVELS, N_STATES,
)


def encode_state(risk: int, income: int, debt: int, history: int) -> int:
    """(risk, income, debt, history) -> indice entier unique."""
    checks = (
        ("risk", risk, RISK_LEVELS),
        ("income", income, INCOME_LEVELS),
        ("debt", debt, DEBT_LEVELS),
        ("history", history, HISTORY_LEVELS),
    )
    for name, value, levels in checks:
        if not 0 <= value < len(levels):
            raise ValueError(f"{name}={value} invalide (attendu 0..{len(levels) - 1})")
    index = risk
    index = index * len(INCOME_LEVELS) + income
    index = index * len(DEBT_LEVELS) + debt
    index = index * len(HISTORY_LEVELS) + history
    return int(index)


def decode_state(index: int) -> tuple:
    """Indice entier -> (risk, income, debt, history)."""
    if not 0 <= index < N_STATES:
        raise ValueError(f"indice d'état invalide : {index} (attendu 0..{N_STATES - 1})")
    index = int(index)
    index, history = divmod(index, len(HISTORY_LEVELS))
    index, debt = divmod(index, len(DEBT_LEVELS))
    risk, income = divmod(index, len(INCOME_LEVELS))
    return risk, income, debt, history


def describe_state(index: int) -> str:
    """Description lisible d'un état (pour l'affichage et l'interface)."""
    risk, income, debt, history = decode_state(index)
    return (f"risk={RISK_LEVELS[risk]}, income={INCOME_LEVELS[income]}, "
            f"debt={DEBT_LEVELS[debt]}, history={HISTORY_LEVELS[history]}")


def all_states() -> list:
    """Liste des 54 états (sous forme d'indices)."""
    return list(range(N_STATES))
