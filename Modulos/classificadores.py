# -*- coding: utf-8 -*-
"""Fábrica de classificadores - configurações exatas da Seção 5.2.

Centralizei aqui a criação dos 8 classificadores avaliados na fase
preliminar, com os mesmos hiperparâmetros que reportei no texto, para
garantir que qualquer experimento use exatamente a mesma configuração.

O classificador h da otimização (Algoritmo 1/2) é escolhido no
config.yaml (campo classificador.nome). O tratamento de desbalanceamento
segue a Seção 4.1: class_weight balanceado em Random Forest, XGBoost,
LightGBM e Logistic Regression. Como o XGBoost multiclasse não expõe
class_weight diretamente, sinalizo com USA_SAMPLE_WEIGHT e o peso por
amostra é calculado no momento do fit (Modulos/avaliacao.py).
"""

from lightgbm import LGBMClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier

# Classificadores em que o balanceamento entra via sample_weight no fit
# (calculado com compute_sample_weight('balanced'))
USA_SAMPLE_WEIGHT = {"xgboost"}

NOMES_VALIDOS = (
    "random_forest",
    "extra_trees",
    "xgboost",
    "lightgbm",
    "logistic_regression",
    "lda",
    "knn",
    "mlp",
)


def criar_classificador(nome, seed=42, n_jobs=-1):
    """Cria um classificador pelos nomes definidos no config.yaml."""
    nome = nome.lower()

    if nome == "random_forest":
        # 100 árvores, gini, profundidade não limitada, sqrt das variáveis
        return RandomForestClassifier(
            n_estimators=100,
            criterion="gini",
            max_depth=None,
            max_features="sqrt",
            class_weight="balanced",
            random_state=seed,
            n_jobs=n_jobs,
        )

    if nome == "extra_trees":
        # configuração análoga ao Random Forest
        return ExtraTreesClassifier(
            n_estimators=100,
            criterion="gini",
            max_depth=None,
            max_features="sqrt",
            class_weight="balanced",
            random_state=seed,
            n_jobs=n_jobs,
        )

    if nome == "xgboost":
        # booster gbtree, lr 0.3, profundidade 6, min_child_weight 1,
        # subsample 1, colsample_bytree 1 (defaults explícitos da Seção 5.2)
        return XGBClassifier(
            booster="gbtree",
            learning_rate=0.3,
            max_depth=6,
            min_child_weight=1,
            subsample=1.0,
            colsample_bytree=1.0,
            random_state=seed,
            n_jobs=n_jobs,
            # balanceamento via sample_weight no fit (ver USA_SAMPLE_WEIGHT)
        )

    if nome == "lightgbm":
        # gbdt, 100 estimadores, lr 0.1, 31 folhas, profundidade livre
        return LGBMClassifier(
            boosting_type="gbdt",
            n_estimators=100,
            learning_rate=0.1,
            num_leaves=31,
            max_depth=-1,
            class_weight="balanced",
            random_state=seed,
            n_jobs=n_jobs,
            verbose=-1,
        )

    if nome == "logistic_regression":
        # penalização L2, C=1, lbfgs, 100 iterações
        return LogisticRegression(
            penalty="l2",
            C=1.0,
            solver="lbfgs",
            max_iter=100,
            class_weight="balanced",
            n_jobs=n_jobs,
        )

    if nome == "lda":
        # solver svd (não suporta class_weight; minoritárias são
        # reportadas explicitamente via F1-macro e matriz de confusão)
        return LinearDiscriminantAnalysis(solver="svd")

    if nome == "knn":
        # 5 vizinhos, pesos uniformes, Minkowski p=2 (euclidiana)
        return KNeighborsClassifier(
            n_neighbors=5,
            weights="uniform",
            metric="minkowski",
            p=2,
            n_jobs=n_jobs,
        )

    if nome == "mlp":
        # 1 camada oculta de 100 neurônios, relu, adam, lr 0.001, 200 iterações
        return MLPClassifier(
            hidden_layer_sizes=(100,),
            activation="relu",
            solver="adam",
            learning_rate_init=0.001,
            max_iter=200,
            random_state=seed,
        )

    raise ValueError(
        f"Classificador '{nome}' não reconhecido. Opções válidas: {NOMES_VALIDOS}"
    )


def usa_sample_weight(nome):
    """Indica se o balanceamento do classificador é feito via sample_weight."""
    return nome.lower() in USA_SAMPLE_WEIGHT
