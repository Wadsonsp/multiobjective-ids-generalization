# -*- coding: utf-8 -*-
"""Pré-processamento dos datasets NF-v2 (Seção 4.1 da dissertação).

Etapas, na ordem em que descrevi na metodologia:
1. Remoção de fluxos com valores ausentes ou infinitos (sem imputação,
   para não introduzir distorções artificiais em atributos derivados).
2. Descarte de atributos associados a vazamento de informação
   (IPs, portas e MIN_TTL/MAX_TTL).
3. Amostragem estratificada (quando configurada) para equiparar volume
   entre bases, preservando a proporção de todas as categorias.
4. Normalização Min-Max com parâmetros ajustados EXCLUSIVAMENTE no
   conjunto de treino e aplicados ao teste (sem vazamento estatístico).

O desbalanceamento de classes é tratado apenas no treino via
class_weight balanceado nos classificadores que suportam (ver
Modulos/classificadores.py), conforme decidi na Seção 4.1.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def remover_ausentes_e_infinitos(df):
    """Remove fluxos com NaN ou inf em qualquer coluna numérica.

    Optei por remoção (e não imputação por média/mediana) porque valores
    infinitos aqui costumam vir de divisões por zero em métricas derivadas
    e de durações nulas - imputar mascararia o problema.
    """
    df = df.replace([np.inf, -np.inf], np.nan)
    return df.dropna(axis=0, how="any").reset_index(drop=True)


def remover_atributos_vazamento(df, atributos_vazamento):
    """Descarta IPs, portas e TTL (atalhos/vícios de classificação).

    Uso errors='ignore' porque nem toda base exporta exatamente as mesmas
    colunas identificadoras; o que existir da lista é removido.
    """
    return df.drop(columns=atributos_vazamento, errors="ignore")


def separar_atributos_e_rotulo(df, coluna_rotulo="Attack", coluna_binaria="Label"):
    """Separa a matriz de atributos X do vetor de rótulos y (multiclasse).

    A coluna binária Label sai da matriz de atributos: além de redundante
    com Attack, mantê-la seria vazamento direto do rótulo.
    """
    y = df[coluna_rotulo].copy()
    X = df.drop(columns=[c for c in (coluna_rotulo, coluna_binaria) if c in df.columns])
    # Garanto matriz exclusivamente numérica: qualquer coluna não numérica
    # remanescente (ex.: identificador esquecido) é descartada com aviso
    # implícito via lista de colunas resultante.
    X = X.select_dtypes(include=[np.number])
    return X, y


def amostra_estratificada(df, n_amostras, coluna_rotulo="Attack", seed=42):
    """Amostragem estratificada preservando a proporção de TODAS as classes.

    Usei amostragem proporcional por grupo com piso de 1 amostra por
    classe, para que categorias minoritárias (ex.: Ransomware, MITM)
    não desapareçam da amostra de 2,0 milhões de fluxos.
    """
    if n_amostras is None or n_amostras >= len(df):
        return df.reset_index(drop=True)

    frac = n_amostras / len(df)
    rng = np.random.RandomState(seed)

    partes = []
    for _, grupo in df.groupby(coluna_rotulo, sort=False):
        # max(1, ...) garante o piso por classe
        n_grupo = max(1, int(round(len(grupo) * frac)))
        n_grupo = min(n_grupo, len(grupo))
        partes.append(grupo.sample(n=n_grupo, random_state=rng))

    amostra = pd.concat(partes, axis=0)
    # Embaralho ao final para não deixar os fluxos agrupados por classe
    return amostra.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def preprocessar_base(df, config_pre, n_amostras=None, seed=42):
    """Pipeline completo da Seção 4.1 para uma base: retorna (X, y).

    A normalização NÃO acontece aqui de propósito: o Min-Max precisa ser
    ajustado dentro de cada partição de treino (CV ou cross-dataset),
    então ela é aplicada no momento da avaliação (Algoritmo 2).
    """
    df = remover_ausentes_e_infinitos(df)
    df = remover_atributos_vazamento(df, config_pre["atributos_vazamento"])
    if n_amostras is not None:
        df = amostra_estratificada(
            df, n_amostras, coluna_rotulo=config_pre["coluna_rotulo"], seed=seed
        )
    X, y = separar_atributos_e_rotulo(
        df,
        coluna_rotulo=config_pre["coluna_rotulo"],
        coluna_binaria=config_pre["coluna_binaria"],
    )
    return X, y


def ajustar_e_aplicar_minmax(X_treino, X_teste):
    """Min-Max [0,1] ajustado só no treino e aplicado ao teste.

    Escolhi Min-Max (e não padronização z-score) porque os atributos de
    tráfego variam em escalas muito distintas e o Min-Max reescala sem
    pressupostos sobre a distribuição (Seção 4.1).
    """
    scaler = MinMaxScaler()
    X_treino_norm = scaler.fit_transform(X_treino)
    X_teste_norm = scaler.transform(X_teste)
    # clip evita que valores fora do range do treino (comuns no cenário
    # cross-dataset) explodam para fora de [0,1]
    return np.clip(X_treino_norm, 0.0, 1.0), np.clip(X_teste_norm, 0.0, 1.0), scaler


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
