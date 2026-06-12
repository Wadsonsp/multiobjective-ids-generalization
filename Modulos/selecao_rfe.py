# -*- coding: utf-8 -*-
"""Baseline monobjetivo de seleção de características via RFE (Capítulo 5).

O RFE (Recursive Feature Elimination) entra na dissertação apenas como
baseline exploratório da fase preliminar, com o Random Forest como
estimador de referência (Seção 5.2), reduzindo o conjunto de 39 para 30
atributos. Ele NÃO faz parte do pré-processamento da Seção 4.1 nem da
proposta tri-objetivo do Capítulo 4.

A saída aqui é uma máscara binária no MESMO formato usado pelos
Algoritmos 1 e 2: assim a avaliação do baseline passa pelo mesmo
AvaliarFitness (Modulos/avaliacao.py), e a comparação baseline x Pareto
fica homogênea (mesmo protocolo, mesmas métricas).
"""

import numpy as np
from sklearn.feature_selection import RFE

from Modulos.classificadores import criar_classificador
from Modulos.preprocessamento import amostra_estratificada


def ajustar_mascara_rfe(X, y, n_atributos=30, seed=42, amostra=None, step=1):
    """Ajusta o RFE e devolve a máscara binária alinhada às colunas de X.

    Parameters
    ----------
    X : DataFrame de atributos já pré-processado (Seção 4.1, sem normalizar -
        o Random Forest é invariante a reescalonamento monotônico, então o
        ranking de importância não depende do Min-Max)
    y : rótulos multiclasse (coluna Attack)
    n_atributos : alvo de atributos selecionados (30 na análise preliminar)
    amostra : subamostra estratificada opcional só para o ajuste do RFE,
        já que eliminar atributos um a um treinando RF na base completa
        é caro; a AVALIAÇÃO da máscara depois usa os dados que eu quiser
    step : quantos atributos eliminar por iteração (1 = eliminação clássica)

    Returns
    -------
    mascara : np.ndarray binário com d posições (1 = atributo mantido)
    ranking : ranking do RFE por atributo (1 = selecionado)
    """
    if n_atributos >= X.shape[1]:
        # Nada a eliminar: máscara cheia (equivale ao baseline sem RFE)
        return np.ones(X.shape[1], dtype=int), np.ones(X.shape[1], dtype=int)

    if amostra is not None and amostra < len(X):
        # Subamostro de forma estratificada para o fit do RFE
        import pandas as pd
        df_xy = pd.concat([X, y.rename("__rotulo__")], axis=1)
        df_xy = amostra_estratificada(
            df_xy, amostra, coluna_rotulo="__rotulo__", seed=seed
        )
        y = df_xy["__rotulo__"]
        X = df_xy.drop(columns=["__rotulo__"])

    # Estimador de referência: Random Forest com a configuração exata da
    # Seção 5.2 (100 árvores, gini, profundidade livre, sqrt das variáveis)
    estimador = criar_classificador("random_forest", seed=seed)

    rfe = RFE(estimator=estimador, n_features_to_select=n_atributos, step=step)
    rfe.fit(X, y)

    mascara = rfe.support_.astype(int)
    return mascara, rfe.ranking_
