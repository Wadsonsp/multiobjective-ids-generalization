# -*- coding: utf-8 -*-
"""Eu centralizo o classificador usado nas duas fases do experimento."""

from sklearn.tree import DecisionTreeClassifier


NOMES_VALIDOS = ("decision_tree",)


def criar_classificador(nome="decision_tree", seed=42, n_jobs=None):
    """Eu crio a árvore de decisão apresentada no problema original."""
    del n_jobs  # Eu mantenho o argumento apenas para uma interface estável.
    nome = nome.lower()
    if nome == "decision_tree":
        return DecisionTreeClassifier(
            max_depth=8,
            class_weight="balanced",
            random_state=seed,
        )
    raise ValueError(
        f"Classificador '{nome}' não reconhecido. Opção válida: decision_tree"
    )
