# -*- coding: utf-8 -*-
"""Eu implemento a avaliação das máscaras nas duas fases do pipeline.

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
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold

from Modulos.classificadores import criar_classificador


def aplicar_mascara(X, mascara):
    """Eu aplico a máscara binária x às colunas de atributos."""
    mascara = np.asarray(mascara).astype(bool)
    if mascara.shape[0] != X.shape[1]:
        raise ValueError(
            f"Máscara com {mascara.shape[0]} posições para {X.shape[1]} atributos."
        )
    if not mascara.any():
        raise ValueError("Máscara vazia: nenhuma característica selecionada.")
    return X.loc[:, mascara] if hasattr(X, "loc") else X[:, mascara]


def numero_de_atributos(mascara):
    """Eu calculo k(x) somando as posições ligadas da máscara."""
    return int(np.sum(np.asarray(mascara).astype(int)))


def _treinar(modelo, X_treino, y_treino):
    """Eu ajusto o classificador com os rótulos codificados da origem."""
    modelo.fit(X_treino, np.asarray(y_treino))
    return modelo


def _prever(modelo, X):
    """Eu devolvo as previsões no mesmo espaço numérico do treinamento."""
    return modelo.predict(X)


def _codificar_rotulos(y, classes):
    """Eu mapeio rótulos texto para inteiros e desconhecidos para -1.

    No cross-dataset as bases têm categorias diferentes; classes do teste
    ausentes no treino entram como -1 (nunca previstas corretamente),
    penalizando o F1-macro de forma honesta em vez de quebrar a execução.
    """
    indice = {c: i for i, c in enumerate(classes)}
    return np.array([indice.get(v, -1) for v in y])


def avaliar_intra_dataset(mascara, X, y, nome_clf, cv_folds=5, seed=42):
    """Eu executo a validação cruzada estratificada em uma base D_i.

    Eu preservo os valores originais das features porque a árvore de decisão
    é invariante à escala e não exige ajuste de normalização.
    """
    X_sel = aplicar_mascara(X, mascara)
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)

    f1s, precisoes, recalls = [], [], []
    matriz_acumulada = None
    classes = np.unique(y)

    for idx_treino, idx_teste in skf.split(X_sel, y):
        X_tr, X_te = X_sel.iloc[idx_treino], X_sel.iloc[idx_teste]
        y_tr, y_te = y.iloc[idx_treino], y.iloc[idx_teste]

        # Eu uso os valores originais porque a árvore é invariante à escala.
        modelo = criar_classificador(nome_clf, seed=seed)
        y_tr_cod = _codificar_rotulos(y_tr, classes)
        _treinar(modelo, X_tr, y_tr_cod)
        y_pred = _prever(modelo, X_te)
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
    """Eu avalio todos os pares direcionais (D_i, D_j), com i diferente de j.

    Treino em D_i com os atributos selecionados por x, teste em D_j,
    F1-macro registrado por direção (sem reduzir a média única - a
    direção da transferência importa, Seção 4.3). Aproveito o predict
    do cross para medir o tempo médio de inferência (linhas 11-12).
    """
    resultados = {}
    tempos_inferencia = []
    diagnostico = {}

    nomes = list(bases_Xy.keys())
    for nome_i in nomes:
        for nome_j in nomes:
            if nome_i == nome_j:
                continue
            X_i, y_i = bases_Xy[nome_i]
            X_j, y_j = bases_Xy[nome_j]

            X_tr = aplicar_mascara(X_i, mascara)
            X_te = aplicar_mascara(X_j, mascara)

            # Eu uso exatamente as features selecionadas, sem reescalonar.
            classes_treino = np.unique(y_i)
            modelo = criar_classificador(nome_clf, seed=seed)
            _treinar(modelo, X_tr, _codificar_rotulos(y_i, classes_treino))

            # Tempo médio de inferência POR AMOSTRA (linha 12)
            inicio = time.perf_counter()
            y_pred = _prever(modelo, X_te)
            duracao = time.perf_counter() - inicio
            tempos_inferencia.append(duracao / max(len(X_te), 1))

            y_te_cod = _codificar_rotulos(y_j, classes_treino)
            f1 = f1_score(y_te_cod, y_pred, average="macro", zero_division=0)

            # Registro por direção de transferência (linha 9)
            resultados[f"{nome_i}->{nome_j}"] = float(f1)

            # --- Diagnóstico da queda cross-dataset ---------------------
            # Separo o teste em duas partições para distinguir as causas:
            # (a) classes do teste que EXISTEM no treino (código >= 0): o
            #     erro aqui é domain shift puro (mesma classe, distribuição
            #     diferente, o modelo poderia ter acertado);
            # (b) classes AUSENTES no treino (código -1): incompatibilidade
            #     de taxonomia - o modelo nunca viu essa classe e não tem
            #     como acertá-la. A queda dessas é estrutural, não de modelo.
            conhecidas = y_te_cod >= 0
            n_total = len(y_te_cod)
            n_desconhecidas = int(np.sum(~conhecidas))
            if conhecidas.any():
                acc_conhecidas = float(
                    np.mean(y_pred[conhecidas] == y_te_cod[conhecidas])
                )
                f1_conhecidas = float(f1_score(
                    y_te_cod[conhecidas], y_pred[conhecidas],
                    average="macro", zero_division=0,
                ))
            else:
                acc_conhecidas = 0.0
                f1_conhecidas = 0.0
            # Matriz de confusão da transferência em rótulos GLOBAIS do
            # teste (todas as classes da base de destino), para o gráfico.
            # A última linha/coluna extra representa "classe sem
            # correspondência" entre as taxonomias.
            classes_teste = np.unique(y_j)
            mapa_teste = {c: i for i, c in enumerate(classes_teste)}
            y_te_global = np.array([mapa_teste[v] for v in y_j])
            pred_texto = np.array(
                [classes_treino[c] if 0 <= c < len(classes_treino) else "__?__"
                 for c in y_pred]
            )
            y_pred_global = np.array([mapa_teste.get(v, -1) for v in pred_texto])
            cm = confusion_matrix(
                y_te_global,
                np.where(y_pred_global < 0, len(classes_teste), y_pred_global),
                labels=list(range(len(classes_teste) + 1)),
            )
            diagnostico[f"{nome_i}->{nome_j}"] = {
                "f1_macro": float(f1),
                "n_total": n_total,
                "n_classes_desconhecidas": n_desconhecidas,
                "prop_desconhecidas": n_desconhecidas / max(n_total, 1),
                "acuracia_classes_conhecidas": acc_conhecidas,
                "f1_macro_classes_conhecidas": f1_conhecidas,
                "classes_treino": [str(c) for c in classes_treino],
                "classes_teste": [str(c) for c in classes_teste],
                "matriz_confusao": cm.tolist(),
            }

    tempo_medio = float(np.mean(tempos_inferencia)) if tempos_inferencia else 0.0
    return resultados, tempo_medio, diagnostico


def avaliar_fitness(mascara, bases_Xy, nome_clf, cv_folds=5, seed=42):
    """Eu retorno todos os critérios detalhados da Fase 2 para a máscara x.

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

    # Linhas 5-12: cross-dataset por direção + tempo de inferência.
    # O diagnostico traz matriz de confusão e a decomposição das causas
    # da queda (domain shift vs. incompatibilidade de taxonomia).
    cross, tempo_medio, diagnostico = avaliar_cross_dataset(
        mascara, bases_Xy, nome_clf, seed
    )

    # Linhas 13-18: saída com os critérios separados
    return {
        "f1_macro_intra": {nome: r["f1_macro"] for nome, r in intra.items()},
        "detalhes_intra": intra,
        "f1_macro_cross_por_direcao": cross,
        "diagnostico_cross": diagnostico,
        "tempo_medio_inferencia": tempo_medio,
        "numero_atributos": numero_de_atributos(mascara),
    }


def avaliar_fase1_cross_dataset(mascara, bases_Xy, nome_clf, seed=42):
    """Eu calculo somente os critérios necessários ao pré-filtro NSGA-II.

    Na Fase 1, eu treino na primeira base e testo na segunda; depois faço o
    caminho inverso. Eu não executo a validação cruzada intra-dataset para
    cada indivíduo porque ela não faz parte dos dois objetivos definidos.
    A avaliação detalhada, incluindo CV intra-dataset, permanece na Fase 2.
    """
    cross, tempo_medio, diagnostico = avaliar_cross_dataset(
        mascara, bases_Xy, nome_clf, seed
    )
    return {
        "f1_macro_cross_por_direcao": cross,
        "diagnostico_cross": diagnostico,
        "tempo_medio_inferencia": tempo_medio,
        "numero_atributos": numero_de_atributos(mascara),
    }
