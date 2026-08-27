# -*- coding: utf-8 -*-
"""Pré-processamento dos datasets NF-v2 (Seção 4.1 da dissertação).

Etapas, na ordem em que descrevi na metodologia:
1. Remoção de fluxos com valores ausentes ou infinitos (sem imputação,
   para não introduzir distorções artificiais em atributos derivados).
2. Descarte de atributos associados a vazamento de informação
   (portas e MIN_TTL/MAX_TTL).
3. Alinhamento das features comuns, preservando a mesma ordem nas bases.

Eu trato o desbalanceamento no treinamento por meio do `class_weight`
balanceado da árvore de decisão.
"""

import numpy as np


def remover_ausentes_e_infinitos(df):
    """Remove fluxos com NaN ou inf em qualquer coluna numérica.

    Optei por remoção (e não imputação por média/mediana) porque valores
    infinitos aqui costumam vir de divisões por zero em métricas derivadas
    e de durações nulas - imputar mascararia o problema.
    """
    df = df.replace([np.inf, -np.inf], np.nan)
    return df.dropna(axis=0, how="any").reset_index(drop=True)


def remover_atributos_vazamento(df, atributos_vazamento):
    """Eu descarto portas e TTL usados como atalhos de classificação.

    Uso errors='ignore' porque nem toda base exporta exatamente as mesmas
    colunas identificadoras; o que existir da lista é removido.
    """
    return df.drop(columns=atributos_vazamento, errors="ignore")


def separar_atributos_e_rotulo(df, coluna_rotulo="Attack", coluna_binaria="Label"):
    """Separa a matriz de atributos X do vetor de rótulos y (multiclasse).

    A coluna binária Label sai da matriz de atributos: além de redundante
    com Attack, mantê-la seria vazamento direto do rótulo.
    """
    # Eu normalizo espaços e capitalização para não tratar "DoS" e "dos",
    # ou "Backdoor" e "backdoor", como taxonomias artificialmente distintas.
    y = df[coluna_rotulo].astype("string").str.strip().str.casefold()
    X = df.drop(columns=[c for c in (coluna_rotulo, coluna_binaria) if c in df.columns])
    # Garanto matriz exclusivamente numérica: qualquer coluna não numérica
    # remanescente (ex.: identificador esquecido) é descartada com aviso
    # implícito via lista de colunas resultante.
    X = X.select_dtypes(include=[np.number])
    return X, y


def preprocessar_base(df, config_pre):
    """Pipeline completo da Seção 4.1 para uma base: retorna (X, y).

    Eu não reescalono as features porque a árvore de decisão compara limiares
    e, portanto, é invariante a transformações monotônicas de escala.
    """
    df = remover_ausentes_e_infinitos(df)
    df = remover_atributos_vazamento(df, config_pre["atributos_vazamento"])
    X, y = separar_atributos_e_rotulo(
        df,
        coluna_rotulo=config_pre["coluna_rotulo"],
        coluna_binaria=config_pre["coluna_binaria"],
    )
    return X, y


def alinhar_colunas(bases_Xy):
    """Garante que todas as bases usem o MESMO conjunto/ordem de colunas.

    A avaliação cross-dataset exige schema idêntico entre as bases
    (vantagem da família NF-v2). Aqui faço a interseção das colunas e
    reordeno, para a máscara binária x indexar o mesmo atributo em
    qualquer base.
    """
    colunas_comuns = None
    for X, _ in bases_Xy.values():
        cols = set(X.columns)
        colunas_comuns = cols if colunas_comuns is None else (colunas_comuns & cols)

    # Ordem determinística (ordem de aparição na primeira base) para a
    # máscara ser reprodutível entre execuções
    primeira = next(iter(bases_Xy.values()))[0]
    ordem = [c for c in primeira.columns if c in colunas_comuns]

    return {nome: (X[ordem].copy(), y) for nome, (X, y) in bases_Xy.items()}, ordem
