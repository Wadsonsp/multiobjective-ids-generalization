# -*- coding: utf-8 -*-
"""Algoritmo 2 - AvaliarFitness(x, D, h) (Figura 6 da dissertação).

Implementação fiel ao pseudocódigo, mantendo a numeração das linhas:

 1: Aplicar a máscara binária x aos atributos disponíveis
 2-4: Avaliar desempenho intra-dataset (CV estratificada em cada D_i;
      registrar F1-macro, precisão, recall e matriz de confusão)
 5-10: Avaliar desempenho cross-dataset (para cada par direcional
       (D_i, D_j), i != j: treinar h em D_i, testar em D_j e registrar
       o F1-macro POR DIREÇÃO de transferência, sem média única)
11-13: Avaliar custo computacional (tempo médio de inferência e k(x))
14-18: Retornar os critérios SEPARADOS

A saída separada é proposital: evita que uma única medida esconda
diferenças entre desempenho, generalização e custo (Seção 4.3).
"""

import time

import numpy as np
from sklearn.base import clone
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_sample_weight

from Modulos.classificadores import criar_classificador, usa_sample_weight
from Modulos.preprocessamento import ajustar_e_aplicar_minmax


def aplicar_mascara(X, mascara):
    """Linha 1: aplica a máscara binária x às colunas de atributos."""
    mascara = np.asarray(mascara).astype(bool)
    if mascara.shape[0] != X.shape[1]:
        raise ValueError(
            f"Máscara com {mascara.shape[0]} posições para {X.shape[1]} atributos."
        )
    if not mascara.any():
        raise ValueError("Máscara vazia: nenhuma característica selecionada.")
    return X.loc[:, mascara] if hasattr(X, "loc") else X[:, mascara]


def numero_de_atributos(mascara):
    """k(x) = soma das posições da máscara (Seção 4.2)."""
    return int(np.sum(np.asarray(mascara).astype(int)))


def _treinar(modelo, nome_clf, X_treino, y_treino):
    """Fit com sample_weight balanceado quando o classificador exige.

    O XGBoost multiclasse não tem class_weight, então o reequilíbrio da
    função de perda pela frequência inversa das classes (Seção 4.1) entra
    aqui via compute_sample_weight('balanced').
    """
    if usa_sample_weight(nome_clf):
        pesos = compute_sample_weight(class_weight="balanced", y=y_treino)
        modelo.fit(X_treino, y_treino, sample_weight=pesos)
    else:
        modelo.fit(X_treino, y_treino)
    return modelo


def _codificar_rotulos(y, classes):
    """Mapeia rótulos texto -> inteiros, com classes desconhecidas = -1.

    No cross-dataset as bases têm categorias diferentes; classes do teste
    ausentes no treino entram como -1 (nunca previstas corretamente),
    penalizando o F1-macro de forma honesta em vez de quebrar a execução.
    """
    indice = {c: i for i, c in enumerate(classes)}
    return np.array([indice.get(v, -1) for v in y])


def avaliar_intra_dataset(mascara, X, y, nome_clf, cv_folds=5, seed=42):
    """Linhas 2-4: CV estratificada em uma base D_i.

    A normalização Min-Max é ajustada dentro de cada fold de treino
    (nunca no fold de teste) para impedir vazamento estatístico.
    """
    X_sel = aplicar_mascara(X, mascara)
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)

    f1s, precisoes, recalls = [], [], []
    matriz_acumulada = None
    classes = np.unique(y)

    for idx_treino, idx_teste in skf.split(X_sel, y):
        X_tr, X_te = X_sel.iloc[idx_treino], X_sel.iloc[idx_teste]
        y_tr, y_te = y.iloc[idx_treino], y.iloc[idx_teste]

        # Min-Max ajustado SÓ no treino do fold (Seção 4.1)
        X_tr_n, X_te_n, _ = ajustar_e_aplicar_minmax(X_tr, X_te)

        modelo = criar_classificador(nome_clf, seed=seed)
        y_tr_cod = _codificar_rotulos(y_tr, classes)
        _treinar(modelo, nome_clf, X_tr_n, y_tr_cod)
        y_pred = modelo.predict(X_te_n)
        y_te_cod = _codificar_rotulos(y_te, classes)

        # Registro das métricas por fold (linha 4)
        f1s.append(f1_score(y_te_cod, y_pred, average="macro", zero_division=0))
        precisoes.append(
            precision_score(y_te_cod, y_pred, average="macro", zero_division=0)
        )
        recalls.append(recall_score(y_te_cod, y_pred, average="macro", zero_division=0))

        cm = confusion_matrix(y_te_cod, y_pred, labels=range(len(classes)))
        matriz_acumulada = cm if matriz_acumulada is None else matriz_acumulada + cm

    return {
        "f1_macro": float(np.mean(f1s)),
        "f1_macro_por_fold": [float(v) for v in f1s],
        "precisao_macro": float(np.mean(precisoes)),
        "recall_macro": float(np.mean(recalls)),
        "matriz_confusao": matriz_acumulada.tolist(),
        "classes": [str(c) for c in classes],
    }


def avaliar_cross_dataset(mascara, bases_Xy, nome_clf, seed=42):
    """Linhas 5-10: pares direcionais (D_i, D_j), i != j.

    Treino em D_i com os atributos selecionados por x, teste em D_j,
    F1-macro registrado por direção (sem reduzir a média única - a
    direção da transferência importa, Seção 4.3). Aproveito o predict
    do cross para medir o tempo médio de inferência (linhas 11-12).
    """
    resultados = {}
    tempos_inferencia = []

    nomes = list(bases_Xy.keys())
    for nome_i in nomes:
        for nome_j in nomes:
            if nome_i == nome_j:
                continue
            X_i, y_i = bases_Xy[nome_i]
            X_j, y_j = bases_Xy[nome_j]

            X_tr = aplicar_mascara(X_i, mascara)
            X_te = aplicar_mascara(X_j, mascara)

            # Min-Max ajustado na base de treino e aplicado na de teste
            X_tr_n, X_te_n, _ = ajustar_e_aplicar_minmax(X_tr, X_te)

            classes_treino = np.unique(y_i)
            modelo = criar_classificador(nome_clf, seed=seed)
            _treinar(modelo, nome_clf, X_tr_n, _codificar_rotulos(y_i, classes_treino))

            # Tempo médio de inferência POR AMOSTRA (linha 12)
            inicio = time.perf_counter()
            y_pred = modelo.predict(X_te_n)
            duracao = time.perf_counter() - inicio
            tempos_inferencia.append(duracao / max(len(X_te_n), 1))

            y_te_cod = _codificar_rotulos(y_j, classes_treino)
            f1 = f1_score(y_te_cod, y_pred, average="macro", zero_division=0)

            # Registro por direção de transferência (linha 9)
            resultados[f"{nome_i}->{nome_j}"] = float(f1)

    tempo_medio = float(np.mean(tempos_inferencia)) if tempos_inferencia else 0.0
    return resultados, tempo_medio


def avaliar_fitness(mascara, bases_Xy, nome_clf, cv_folds=5, seed=42):
    """Algoritmo 2 completo: retorna os critérios de avaliação de x.

    Parameters
    ----------
    mascara : array binário com d posições (1 = atributo selecionado)
    bases_Xy : dict {nome_base: (X, y)} já pré-processado e com colunas
        alinhadas entre as bases (ver preprocessamento.alinhar_colunas)
    nome_clf : classificador h (config.yaml)
    """
    # Linhas 2-4: intra-dataset em cada D_i
    intra = {
        nome: avaliar_intra_dataset(mascara, X, y, nome_clf, cv_folds, seed)
        for nome, (X, y) in bases_Xy.items()
    }

    # Linhas 5-12: cross-dataset por direção + tempo de inferência
    cross, tempo_medio = avaliar_cross_dataset(mascara, bases_Xy, nome_clf, seed)

    # Linhas 13-18: saída com os critérios separados
    return {
        "f1_macro_intra": {nome: r["f1_macro"] for nome, r in intra.items()},
        "detalhes_intra": intra,
        "f1_macro_cross_por_direcao": cross,
        "tempo_medio_inferencia": tempo_medio,
        "numero_atributos": numero_de_atributos(mascara),
    }
